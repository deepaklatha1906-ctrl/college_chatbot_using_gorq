# app.py
import streamlit as st
from chat_engine import GroqChatbot

# Page config - Professional look
st.set_page_config(
    page_title="🎓 College Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .stChatMessage {border-radius: 12px; padding: 10px 15px;}
    .stChatMessage[data-testid="stChatMessageUser"] {background-color: #e3f2fd;}
    .stChatMessage[data-testid="stChatMessageAssistant"] {background-color: #f5f5f5;}
    .context-badge {background-color: #4caf50; color: white; padding: 3px 10px; 
                   border-radius: 15px; font-size: 0.8em; font-weight: 500;}
    .header {text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
             border-radius: 10px; color: white; margin-bottom: 1rem;}
    .footer {text-align: center; padding: 1rem; color: #666; font-size: 0.9em; margin-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header"><h1>🎓 College Assistant</h1><p>Ask anything about admissions, fees, courses & more</p></div>', unsafe_allow_html=True)

# Initialize chatbot (cached to avoid reloading)
@st.cache_resource
def get_chatbot():
    return GroqChatbot(json_file="college.json")

try:
    bot = get_chatbot()
except Exception as e:
    st.error(f"❌ Failed to initialize chatbot: {e}")
    st.stop()

# Sidebar - Info panel
with st.sidebar:
    st.header("ℹ️ About")
    st.info("This assistant uses your college data (`college.json`) to provide accurate answers. Keywords like *admission*, *fees*, *scholarships* trigger context-aware responses.")
    
    st.divider()
    st.subheader("🔑 Status")
    st.success(f"✅ {len(bot.GROQ_KEYS)} Groq keys loaded")
    st.caption(f"Model: `{bot.model}`")
    
    st.divider()
    st.subheader("📚 Sample Questions")
    st.markdown("""
    - What is the admission deadline?
    - Tell me about tuition fees
    - What scholarships are available?
    - What are the requirements for application?
    """)

# Chat history management
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show context badge if keywords were matched
        if msg.get("context_used"):
            st.markdown('<span class="context-badge">📦 Context Used</span>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about college admissions, fees, courses..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show typing indicator
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                reply, used_context, context_list = bot.process_query(prompt)
                
                # Display response
                st.markdown(reply)
                
                # Show context badge if keywords matched
                if used_context:
                    st.markdown('<span class="context-badge">📦 Context Used</span>', unsafe_allow_html=True)
                    with st.expander("🔍 View matched context"):
                        for ctx in context_list:
                            st.code(ctx, language="text")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": reply,
                    "context_used": used_context
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown('<div class="footer">Powered by Groq AI • Your College Data • Built with Streamlit</div>', unsafe_allow_html=True)