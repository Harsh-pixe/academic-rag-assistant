"""
app.py
------
Streamlit UI for Local RAG System
"""

import os
import streamlit as st
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from ingest import ingest_pdf, get_vector_store
from langchain_ollama import OllamaLLM

load_dotenv()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -----------------------------
# RETRIEVAL
# -----------------------------
def retrieve_chunks(question):
    vector_store = get_vector_store()
    return vector_store.similarity_search(question, k=2)


# -----------------------------
# PROMPT BUILDER
# -----------------------------
def build_prompt(question, chunks):
    context = "\n\n".join([c.page_content[:1000] for c in chunks])

    return f"""
You are an academic assistant.
Answer ONLY from context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


# -----------------------------
# LLM RESPONSE
# -----------------------------
def generate_answer(prompt):
    llm = OllamaLLM(model="phi3")
    return llm.invoke(prompt)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("📚 Local AI Academic Assistant (RAG)")

# Upload
st.header("Upload PDFs")

files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("Process Documents"):
    if files:
        for f in files:
            path = os.path.join(UPLOAD_DIR, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())

            chunks = ingest_pdf(path, f.name)
            st.success(f"{f.name} → {chunks} chunks stored")


st.divider()

# Chat
st.header("Ask Questions")

question = st.text_input("Enter your question")

if st.button("Ask"):
    if question:

        chunks = retrieve_chunks(question)

        if not chunks:
            st.warning("No data found. Upload PDFs first.")
        else:
            prompt = build_prompt(question, chunks)
            answer = generate_answer(prompt)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")
            for i, c in enumerate(chunks):
                source = c.metadata.get("source", "unknown")
                page = c.metadata.get("page", "N/A")

                with st.expander(f"Chunk {i+1} | {source} | Page {page}"):
                    st.write(c.page_content)

def export_pdf(text):
    file = "answer.pdf"
    c = canvas.Canvas(file)
    c.drawString(100, 800, text[:1000])
    c.save()
    return file
if st.button("Download Answer"):
    file = export_pdf(answer)
    st.success("PDF generated!")