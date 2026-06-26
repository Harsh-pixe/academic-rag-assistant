 # 📚 AI-Powered Academic Assistant (RAG)

A local Retrieval-Augmented Generation (RAG) system that allows students to upload PDFs and ask questions based only on document content.

---

## 🚀 Features

- Upload multiple PDF files
- Extract and chunk text automatically
- Store embeddings in ChromaDB
- Semantic search using vector similarity
- Local LLM (Ollama: phi3/mistral)
- Streamlit chat interface
- Displays retrieved context (sources)

---

## 🧠 Architecture

PDF Upload  
→ Text Extraction (PyPDF)  
→ Chunking  
→ Embeddings (SentenceTransformers)  
→ Vector DB (ChromaDB)  
→ Retrieval (Similarity Search)  
→ LLM (Ollama)  
→ Answer Output  

---

## ⚙️ Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- SentenceTransformers
- Ollama (Local LLM)

---

## 📦 Installation

```bash
git clone <your-repo>
cd academic-rag-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt