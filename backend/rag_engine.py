# backend/rag_engine.py
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
    
    if os.path.exists(persist_path):
        return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
    
    corpus_path = "data/corpus/"
    os.makedirs(corpus_path, exist_ok=True)
    
    docs = []
    if os.path.exists(corpus_path):
        for filename in os.listdir(corpus_path):
            if filename.endswith(".txt"):
                loader = TextLoader(os.path.join(corpus_path, filename), encoding="utf-8")
                docs.extend(loader.load())
                
    if not docs:
        vectorstore = FAISS.from_texts(["Initial setup document. System active."], embeddings)
        vectorstore.save_local(persist_path)
        return vectorstore

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(persist_path)
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_production_qa_chain(user_key: str = None):
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

    # 💡 1. Contextualize Question Chain: Re-writes follow-ups into standalone queries
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    # 💡 2. Main Response System Prompt
    system_prompt = (
        "You are an expert clinical assistant. Use the following context fragments to synthesize "
        "a precise, evidence-based response. If the answer cannot be found in the context, state "
        "transparently that you do not have sufficient information.\n\n"
        "Context:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # 💡 3. Unified Execution Pipeline
    def route_retrieval(inputs):
        # If there is chat history, use the contextualization subchain to rewrite the query
        if inputs.get("chat_history"):
            return contextualize_q_chain
        # Otherwise, pass the raw input straight through to the retriever
        return RunnablePassthrough() | (lambda x: x["input"])

    lcel_chain = (
        RunnablePassthrough.assign(
            # Dynamically resolves whether to use raw query or rewritten query for retrieval
            search_query=route_retrieval
        )

        | {
            "context": (lambda x: x["search_query"]) | retriever | format_docs,
            "chat_history": lambda x: x["chat_history"],
            "input": lambda x: x["input"]
        }

        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return lcel_chain
