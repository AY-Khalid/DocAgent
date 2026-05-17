# backend/rag_engine.py
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
# Added to dynamically split documents if vectorstore isn't built yet
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_cached_vectorstore():
    persist_path = "vectorstore"
    embeddings = get_embeddings_model()
    
    # Check if the FAISS index already exists on disk
    if os.path.exists(persist_path):
        print("🔁 Loading existing cached vectorstore...")
        return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
    
    # 💡 PRODUCTION SAFEGUARD: Self-heal if the folder doesn't exist yet on Streamlit Cloud
    print("📚 Vectorstore missing. Building a fresh one on-the-fly...")
    corpus_path = "data/corpus/"
    os.makedirs(corpus_path, exist_ok=True)
    
    docs = []
    if os.path.exists(corpus_path):
        for filename in os.listdir(corpus_path):
            if filename.endswith(".txt"):
                loader = TextLoader(os.path.join(corpus_path, filename), encoding="utf-8")
                docs.extend(loader.load())
                
    if not docs:
        # If no documents exist yet, create a dummy vector store to keep the app working
        print("⚠️ Warning: No corpus documents found to index.")
        vectorstore = FAISS.from_texts(["Initial setup document. System active."], embeddings)
        vectorstore.save_local(persist_path)
        return vectorstore

    # Split and build the index
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(persist_path)
    return vectorstore


def get_production_qa_chain(user_key: str = None):
    # Fast context lookup from memory cache
    vectorstore = load_cached_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) 

    final_api_key = user_key if user_key else os.getenv("MY_MASTER_API_KEY")
    if not final_api_key:
        raise ValueError("Missing Credentials: Secure authorization tokens not detected.")

    llm = ChatOpenAI(
        temperature=0.0, 
        model_name="gpt-3.5-turbo",
        openai_api_key=final_api_key,
        max_retries=3 
    )

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

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)
