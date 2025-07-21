# app.py

import streamlit as st
from backend.rag_engine import get_qa_chain

st.set_page_config(page_title="DocAgent", layout="centered")
st.title("🩺 DocAgent: Clinical Assistant")

query = st.text_area("Enter your clinical question below:")

if query:
    with st.spinner("Analyzing your question..."):
        try:
            chain = get_qa_chain()
            result = chain.run(query)
            st.success("✅ Result")
            st.markdown(result)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
