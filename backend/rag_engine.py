# backend/rag_engine.py

import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

load_dotenv()  # Load environment variables from .env

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
    for filename in os.listdir("data/corpus/"):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join("data/corpus", filename), encoding="utf-8")
            docs.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(persist_path)
    return vectorstore


def get_qa_chain():
    vectorstore = create_or_load_vectorstore()
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo"  # You can use "gpt-4" if your API key supports it
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    return chain
