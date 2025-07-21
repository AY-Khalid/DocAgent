# backend/loader.py

import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(path="data/corpus"):
    docs = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(path, filename), encoding='utf-8')
            docs.extend(loader.load())
    return docs

def split_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
