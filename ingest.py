"""
ingest.py
---------
Handles:
1. PDF loading
2. Text extraction
3. Chunking
4. Embeddings (local)
5. Storage in ChromaDB
"""

import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "chroma_db"


# -----------------------------
# STEP 1: Extract text from PDF
# -----------------------------
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    docs = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        if text.strip():
            docs.append({
                "text": text,
                "page": i + 1})
    return docs

# -----------------------------
# STEP 2: Chunk text
# -----------------------------
def split_text(text, source_name="unknown"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50)

    docs = splitter.create_documents([text])

    for d in docs:
        d.metadata["source"] = source_name
    return docs


# -----------------------------
# STEP 3: Load Vector DB
# -----------------------------
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )


# -----------------------------
# STEP 4: Ingest PDF
# -----------------------------
def ingest_pdf(pdf_path, source_name="unknown"):
    text = extract_text(pdf_path)
    chunks = split_text(text)

    vector_store = get_vector_store()

    for chunk in chunks:
        chunk.metadata["source"] = source_name

    vector_store.add_documents(chunks)
    vector_store.persist()

    return len(chunks)