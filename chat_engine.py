import json
import numpy as np
import faiss
import requests
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

load_dotenv()


class GroqChatbot:
    def __init__(self, json_file: str = "college.json"):
        self.json_file = json_file

        # Load JSON
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Storage
        self.chunks = []
        self.embeddings = None
        self.index = None

        # Groq Keys (with rotation)
        self.GROQ_KEYS = [
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2")
        ]
        self.GROQ_KEYS = [k for k in self.GROQ_KEYS if k]

        if not self.GROQ_KEYS:
            raise ValueError("❌ At least one GROQ_API_KEY must be set")

        self.current_key_index = 0
        self.model = "llama-3.3-70b-versatile"

        # Prepare RAG DB
        self._prepare_data()

    # -----------------------------
    # 1. Flatten JSON → chunks
    # -----------------------------
    def _flatten_json(self, obj, parent_key=""):
        chunks = []

        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key} {k}".strip()
                chunks.extend(self._flatten_json(v, new_key))

        elif isinstance(obj, list):
            for item in obj:
                chunks.extend(self._flatten_json(item, parent_key))

        else:
            chunks.append(f"{parent_key}: {obj}")

        return chunks

    # -----------------------------
    # 2. Build FAISS index
    # -----------------------------
    def _prepare_data(self):
        self.chunks = self._flatten_json(self.data)

        print(f"📦 Total chunks: {len(self.chunks)}")

        self.embeddings = self.embed_model.encode(self.chunks)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings))

        print("✅ Vector DB ready!")

    # -----------------------------
    # 3. Semantic search
    # -----------------------------
    def search(self, query: str, top_k: int = 5) -> List[str]:
        query_vec = self.embed_model.encode([query])
        distances, indices = self.index.search(np.array(query_vec), top_k)

        return [self.chunks[i] for i in indices[0]]

    # -----------------------------
    # 4. Groq API with rotation + debug
    # -----------------------------
    def call_groq(self, prompt: str) -> str:
        for i in range(len(self.GROQ_KEYS)):
            idx = (self.current_key_index + i) % len(self.GROQ_KEYS)
            key = self.GROQ_KEYS[idx]

            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1024
            }

            try:
                res = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                data = res.json()

                # 🔥 Handle error safely
                if "choices" not in data:
                    print("⚠️ API ERROR:", data)
                    continue

                self.current_key_index = idx
                return data["choices"][0]["message"]["content"]

            except Exception as e:
                print(f"⚠️ Key {idx+1} failed:", str(e))
                continue

        return "❌ All Groq API keys failed."

    # -----------------------------
    # 5. Main RAG pipeline
    # -----------------------------
    def process_query(self, user_input: str) -> Tuple[str, bool, List[str]]:
        relevant_chunks = self.search(user_input)

        context = "\n".join(relevant_chunks)

        prompt = f"""
You are a college assistant chatbot.

Use ONLY the context below to answer.

Context:
{context}

Question:
{user_input}

If answer is not in context, say "I don't know based on available data."
"""

        reply = self.call_groq(prompt)

        return reply, True, relevant_chunks