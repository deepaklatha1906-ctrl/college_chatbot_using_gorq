# chat_engine.py
import os
import re
import json
import requests
from dotenv import load_dotenv
from typing import List, Tuple, Optional

load_dotenv()

class GroqChatbot:
    def __init__(self, json_file: str = "college.json"):
        self.json_file = json_file
        self.JSON_DATA = self._load_json()
        self.STORED_LIST = self._extract_keys(self.JSON_DATA)
        
        # Load Groq API Keys
        self.GROQ_KEYS = [
            os.getenv("GROQ_API_KEY_1"), 
            os.getenv("GROQ_API_KEY_2")
        ]
        self.GROQ_KEYS = [k for k in self.GROQ_KEYS if k]
        
        if not self.GROQ_KEYS:
            raise ValueError("❌ At least one GROQ_API_KEY must be set in .env")
        
        self.current_key_index = 0
        self.model = "llama-3.3-70b-versatile"  # Update as needed
        
    def _load_json(self) -> dict:
        """Load JSON data file."""
        try:
            with open(self.json_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ {self.json_file} not found. Using empty mapping.")
            return {}
    
    def _extract_keys(self, obj, keys: List[str] = None) -> List[str]:
        """Recursively extract all keys from nested JSON."""
        if keys is None:
            keys = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                keys.append(key)
                self._extract_keys(value, keys)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_keys(item, keys)
        return list(set(keys))  # Remove duplicates
    
    def tokenize(self, text: str) -> List[str]:
        """Split input into lowercase word tokens."""
        return re.findall(r"\b\w+\b", text.lower())
    
    def check_keywords(self, tokens: List[str]) -> Tuple[List[str], List[str]]:
        """Check tokens against STORED_LIST and fetch values from JSON_DATA."""
        matched_tokens = [t for t in tokens if t in self.STORED_LIST]
        matched_values = []
        for t in matched_tokens:
            val = self.JSON_DATA.get(t)
            if val and isinstance(val, (str, int, float, bool)):
                matched_values.append(f"{t}: {val}")
            elif val:
                # Truncate complex objects for prompt safety
                truncated = json.dumps(val, ensure_ascii=False)[:150]
                matched_values.append(f"{t}: {truncated}...")
        return matched_tokens, matched_values
    
    def call_groq(self, prompt: str) -> str:
        """Call Groq API with automatic key rotation."""
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
                "max_tokens": 1024,
                "temperature": 0.7
            }
            
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers, json=payload, timeout=30
                )
                resp.raise_for_status()
                self.current_key_index = idx  # Remember working key
                return resp.json()["choices"][0]["message"]["content"]
                
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code
                if status in [401, 403, 429]:
                    print(f"⚠️ Groq Key {idx + 1} failed ({status}). Switching...")
                    continue
                else:
                    error_detail = e.response.text[:200] if e.response else "No response"
                    raise Exception(f"❌ Groq API Error {status}: {error_detail}")
                    
        return "❌ All Groq API keys failed. Please check your keys or try again later."
    
    def process_query(self, user_input: str) -> Tuple[str, bool, List[str]]:
        """
        Process user query and return response.
        Returns: (reply, keywords_matched, matched_context_list)
        """
        tokens = self.tokenize(user_input)
        matched_tokens, matched_values = self.check_keywords(tokens)
        
        if matched_values:
            prompt = (
                f"User Question: {user_input}\n"
                f"Relevant Context from Database: {'; '.join(matched_values)}\n"
                f"Please answer accurately using ONLY the provided context. If context is insufficient, say so."
            )
        else:
            prompt = user_input
            
        reply = self.call_groq(prompt)
        return reply, bool(matched_values), matched_values