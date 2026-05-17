# app.py

import streamlit as st
from backend.rag_engine import get_qa_chain

# Configure page setup
st.set_page_config(
    page_title="DocAgent - Clinical Assistant", 
    page_icon="🩺",
    layout="centered"  # Keeps the content perfectly centered on large screens
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
    
    st.divider() # Clean visual line separating settings
    
    # Dynamic informational status boxes
    if custom_key.strip():
        st.info("⚡ **Status:** Running on your personal OpenAI API key.")
    else:
        st.success("🌐 **Status:** Running on the app's default shared API key.")

# ─── MAIN USER INTERFACE ─────────────────────────────────────────────
# Header section with clean layout alignment
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.markdown("## 🩺")
with col2:
    st.markdown("# DocAgent \n### *Clinical Assistant RAG Engine*")

st.markdown("---") # Visual horizontal rule

# Form layout wrapper prevents premature triggering while typing long blocks of text
with st.form(key="clinical_query_form"):
    query = st.text_area(
        label="💬 Enter your clinical or medical question below:",
        placeholder="Type or paste your patient case notes, clinical queries, or research questions here...",
        height=180
    )
    
    # Centered or clean aligned submit button inside the form boundary
    submit_button = st.form_submit_button(label="🚀 Run Clinical Analysis", use_container_width=True)

# ─── BACKEND EXECUTION ───────────────────────────────────────────────
if submit_button and query.strip():
    with st.spinner("🧠 Analyzing corpus data and compiling response..."):
        try:
            # Format and verify credentials string
            user_key_input = custom_key.strip() if custom_key.strip() else None
            
            # Execute backend RAG chain call
            chain = get_qa_chain(user_key=user_key_input)
            result = chain.run(query)
            
            # Display results inside a neat callout container block
            st.markdown("### 📋 Generated Clinical Insights")
            with st.container(border=True):
                st.markdown(result)
                
        except Exception as e:
            st.error(f"❌ **System Error:** {str(e)}")
            
elif submit_button and not query.strip():
    st.warning("⚠️ Please type a clinical question before clicking submit.")
