# app.py

import streamlit as st
# Notice the updated function name here to match the modern production architecture
from backend.rag_engine import get_production_qa_chain

st.set_page_config(
    page_title="DocAgent - Clinical Assistant", 
    page_icon="🩺",
    layout="centered"
)

# ─── SIDEBAR AUTHENTICATION ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Authentication")
    st.write("Configure your language model credentials.")
    
    custom_key = st.text_input(
        "Use your own OpenAI API Key (Optional):", 
        type="password",
        placeholder="Leave blank to use default key"
    )
    st.divider() 
    
    if custom_key.strip():
        st.info("⚡ **Status:** Running on your personal OpenAI API key.")
    else:
        st.success("🌐 **Status:** Running on the app's default shared API key.")

# ─── MAIN USER INTERFACE ─────────────────────────────────────────────
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.markdown("## 🩺")
with col2:
    st.markdown("# DocAgent \n### *Clinical Assistant RAG Engine*")

st.markdown("---")

with st.form(key="clinical_query_form"):
    query = st.text_area(
        label="💬 Enter your clinical or medical question below:",
        placeholder="Type or paste your patient case notes, clinical queries, or research questions here...",
        height=180
    )
    submit_button = st.form_submit_button(label="🚀 Run Clinical Analysis", use_container_width=True)

# ─── BACKEND EXECUTION ───────────────────────────────────────────────
if submit_button and query.strip():
    with st.spinner("🧠 Analyzing corpus data and compiling response..."):
        try:
            user_key_input = custom_key.strip() if custom_key.strip() else None
            
            # 1. Initialize modern production chain
            chain = get_production_qa_chain(user_key=user_key_input)

            # 2. Invoke chain using the updated dict format
            response_payload = chain.invoke({"input": query})
            result = response_payload["answer"] 
            
            # 3. Render clean output block cleanly exactly once
            st.markdown("### 📋 Generated Clinical Insights")
            with st.container(border=True):
                st.markdown(result)
                
        except Exception as e:
            st.error(f"❌ **System Error:** {str(e)}")
            
elif submit_button and not query.strip():
    st.warning("⚠️ Please type a clinical question before clicking submit.")
