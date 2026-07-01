"""
ingest.py
---------
Handles:
1. PDF loading
2. Text extraction
3. Chunking
4. Embeddings (local HuggingFace)
5. Storage in ChromaDB
"""

import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

DB_DIR = "./chroma_db"

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages_data = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages_data.append({
                "text": text,
                "page": i
            })
    return pages_data

def split_text(pages_data, source_name="unknown"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    all_chunks = []
    for page_entry in pages_data:
        chunks = splitter.split_text(page_entry["text"])
        for chunk in chunks:
            all_chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source_name,
                        "page": page_entry["page"]
                    }
                )
            )
    return all_chunks

def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="academic_docs"
    )

def ingest_pdfs(data_dir, chroma_dir=DB_DIR, openai_api_key=None, embedding_model=None):
    vector_store = get_vector_store()
    total_chunks = 0

    if not os.path.exists(data_dir):
        return 0

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(data_dir, filename)
            pages_data = extract_text(pdf_path)
            chunks = split_text(pages_data, source_name=pdf_path)
            if chunks:
                vector_store.add_documents(chunks)
                total_chunks += len(chunks)

    return total_chunks
