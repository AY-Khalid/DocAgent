# backend/loader.py

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(path="data/corpus"):
    docs = []
    # Safety check for path creation
    os.makedirs(path, exist_ok=True)
    
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(path, filename), encoding='utf-8')
            docs.extend(loader.load())
    return docs

def split_documents(docs, chunk_size=500, chunk_overlap=50):
    # Fixed modern import endpoint used here
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
