# app.py

import streamlit as st
from backend.rag_engine import get_qa_chain

st.set_page_config(page_title="DocAgent", layout="centered")
st.title("🩺 DocAgent: Clinical Assistant")

# 1. Sidebar for optional user API key input
with st.sidebar:
    st.header("🔑 Authentication")
    custom_key = st.text_input(
        "Use your own OpenAI API Key (Optional):", 
        type="password",
        placeholder="Leave blank to use default key"
    )
    if custom_key.strip():
        st.caption("⚡ Running on your personal API key.")
    else:
        st.caption("🌐 Running on the app's default shared API key.")

query = st.text_area("Enter your clinical question below:")

if query:
    with st.spinner("Analyzing your question..."):
        try:
            # 2. Extract and strip the key to ensure it isn't empty spaces
            user_key_input = custom_key.strip() if custom_key.strip() else None
            
            # 3. Pass the user key (or None) down to the RAG engine
            chain = get_qa_chain(user_key=user_key_input)
            
            result = chain.run(query)
            st.success("✅ Result")
            st.markdown(result)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
