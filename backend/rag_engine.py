# backend/rag_engine.py

import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI

load_dotenv()  # Load environment variables from local .env (if present)

def create_or_load_vectorstore():
    persist_path = "vectorstore"

    if os.path.exists(persist_path):
        print("🔁 Loading existing vectorstore...")
        return FAISS.load_local(
            persist_path,
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )

    print("📚 Creating new vectorstore...")
    docs = []
    
    # Safety check if the directory doesn't exist yet
    os.makedirs("data/corpus/", exist_ok=True)
    
    for filename in os.listdir("data/corpus/"):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join("data/corpus", filename), encoding="utf-8")
            docs.extend(loader.load())

    if not docs:
        print("⚠️ Warning: No .txt files found in data/corpus/")
        # Return an empty or minimal structure if no docs exist, or let it handle gracefully
        
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(persist_path)
    return vectorstore


def get_qa_chain(user_key: str = None):
    vectorstore = create_or_load_vectorstore()
    retriever = vectorstore.as_retriever()

    # 1. Fallback logic: Use user's key if provided, else use your Streamlit Secret master key
    final_api_key = user_key if user_key else os.getenv("MY_MASTER_API_KEY")

    if not final_api_key:
        raise ValueError(
            "Missing Credentials: No API Key was entered in the sidebar, "
            "and no default app master key is configured."
        )

    # 2. Pass the finalized key to the ChatOpenAI client
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        openai_api_key=final_api_key
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    return chain
