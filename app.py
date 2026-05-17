# app.py
import streamlit as st
from backend.rag_engine import get_production_qa_chain
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="DocAgent - Clinical Assistant", 
    page_icon="🩺",
    layout="centered"
)

# Initialize persistent chat history structure in session state if not present
if "messages" not in st.session_state:
    st.session_state.messages = []
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

# ─── SIDEBAR AUTHENTICATION ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Authentication")
    custom_key = st.text_input(
        "Use your own OpenAI API Key (Optional):", 
        type="password",
        placeholder="Leave blank to use default key"
    )
    st.divider() 
    
    if custom_key.strip():
        st.info("⚡ **Status:** Personal API Key active.")
    else:
        st.success("🌐 **Status:** Shared App Key active.")
        
    # Clear conversation utility button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.langchain_history = []
        st.rerun()

# ─── MAIN USER INTERFACE ─────────────────────────────────────────────
st.markdown("# 🩺 DocAgent \n### *Clinical Assistant RAG Engine with Memory*")
st.markdown("---")

# Render previous messages from session history cache onto the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── CHAT CONTROLLER INPUT ───────────────────────────────────────────
if query := st.chat_input("Ask DocAgent a clinical question..."):
    
    # 1. Immediately render user question box
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # 2. Process query and stream the AI response block
    with st.chat_message("assistant"):
        with st.spinner("🧠 Analyzing case history..."):
            try:
                user_key_input = custom_key.strip() if custom_key.strip() else None
                chain = get_production_qa_chain(user_key=user_key_input)

                # Execute RAG workflow injecting history payload list
                result = chain.invoke({
                    "input": query,
                    "chat_history": st.session_state.langchain_history
                })
                
                st.markdown(result)
                
                # 3. Append conversational payloads into state trackers
                st.session_state.messages.append({"role": "assistant", "content": result})
                st.session_state.langchain_history.extend([
                    HumanMessage(content=query),
                    AIMessage(content=result)
                ])
                
            except Exception as e:
                st.error(f"❌ **System Error:** {str(e)}")
