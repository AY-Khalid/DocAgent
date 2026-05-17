# backend/rag_engine.py
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 💡 FIX 1: Cache resources globally so they load ONCE, not on every query execution
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_cached_vectorstore():
    persist_path = "vectorstore"
    if not os.path.exists(persist_path):
        raise FileNotFoundError("Vectorstore directory missing. Please initialize embeddings first.")
    
    # Reuses the single cached embedding model instance safely
    embeddings = get_embeddings_model()
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)


def get_production_qa_chain(user_key: str = None):
    # Fast context lookup from memory cache
    vectorstore = load_cached_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # Strict document constraint

    final_api_key = user_key if user_key else os.getenv("MY_MASTER_API_KEY")
    if not final_api_key:
        raise ValueError("Missing Credentials: Secure authorization tokens not detected.")

    llm = ChatOpenAI(
        temperature=0.0, # Complete deterministic precision for clinical workflows
        model_name="gpt-3.5-turbo",
        openai_api_key=final_api_key,
        max_retries=3 # Resiliency against transient API dropped frames
    )

    # 💡 FIX 2: Modernized standard LCEL parsing schema
    system_prompt = (
        "You are an expert clinical assistant. Use the following context fragments to synthesize "
        "a precise, evidence-based response. If the answer cannot be found in the context, state "
        "transparently that you do not have sufficient information.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Standard chain composition
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Complete modern retrieval runtime structure
    return create_retrieval_chain(retriever, question_answer_chain)
