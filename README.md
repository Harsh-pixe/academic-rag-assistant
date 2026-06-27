# 📚 AI-Powered Academic Assistant using RAG

A **fully local** Retrieval-Augmented Generation (RAG) system that lets students upload PDF documents (notes, papers, textbooks) and ask questions about them. The assistant answers **only using the content of the uploaded PDFs** — no internet API calls, no cloud LLM, and no cost per query.

Built as a BSC level academic project to demonstrate the core RAG pipeline: **extract → chunk → embed → store → retrieve → generate**.

---

## 🔍 Project Overview

Traditional chatbots answer from their general training data, which means they can "hallucinate" facts that aren't in your study material. This project solves that problem using **Retrieval-Augmented Generation (RAG)**:

1. You upload your own PDFs (lecture notes, research papers, etc.)
2. The system breaks the text into small chunks and converts each chunk into a numerical representation (an *embedding*)
3. These embeddings are stored in a local vector database (**ChromaDB**)
4. When you ask a question, the system finds the most relevant chunks using semantic similarity search
5. Those chunks are passed to a **local LLM (Ollama)** along with your question, so the answer is grounded in your actual documents

Everything runs **on your own machine** — no OpenAI key, no internet connection required after setup, and no data leaves your computer.

---

## ✨ Features

- 📄 **Multi-PDF upload** — upload one or more PDF files at once
- ✂️ **Automatic text extraction & chunking** — splits long documents into manageable pieces
- 🧠 **Local embeddings** — uses `sentence-transformers` (`all-MiniLM-L6-v2`), no API key needed
- 🗄️ **Persistent vector storage** — powered by **ChromaDB**, so you don't need to re-process PDFs every time
- 🔎 **Semantic search** — finds the most relevant chunks, not just keyword matches
- 🤖 **Local LLM answering** — powered by **Ollama** (`phi3` model) running entirely offline
- 📑 **Source transparency** — every answer shows exactly which chunk(s) and page number(s) it came from
- 📥 **Export answers to PDF** — download your Q&A results for later reference
- 🖥️ **Simple Streamlit interface** — no coding required to use the app

---

## 🛠️ Technologies Used

| Category | Tool |
|---|---|
| Language | Python 3.11+ |
| Frontend / UI | [Streamlit](https://streamlit.io/) |
| RAG Orchestration | [LangChain](https://www.langchain.com/) |
| Vector Database | [ChromaDB](https://www.trychroma.com/) |
| Embeddings | [Sentence-Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| LLM (local) | [Ollama](https://ollama.com/) running `phi3` |
| PDF Text Extraction | [pypdf](https://pypdf.readthedocs.io/) |
| PDF Export | [ReportLab](https://www.reportlab.com/) |
| Config Management | [python-dotenv](https://pypi.org/project/python-dotenv/) |

> 💡 **Why local instead of OpenAI?** This setup runs entirely on your own hardware — no API key, no per-query cost, and your documents never leave your machine. Great for students who want a free, private, and offline-capable assistant.

---

## 📁 Folder Structure

```
academic-rag-assistant/
│
├── app.py                # Streamlit UI — upload, ask, view answers & sources
├── ingest.py             # Core RAG pipeline: extract → chunk → embed → store
├── requirements.txt      # Python dependencies
├── .env.example           # Example environment file
├── .gitignore             # Files/folders excluded from Git
├── README.md              # You are here
│
├── data/
│   └── uploads/           # Uploaded PDFs are saved here (auto-created, git-ignored)
│
└── chroma_db/              # ChromaDB persistent vector storage (auto-created, git-ignored)
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/academic-rag-assistant.git
cd academic-rag-assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Install & set up Ollama (local LLM)
This project uses [Ollama](https://ollama.com/) to run the language model **locally**, so download and install it first from the official site, then pull the model used by this app:
```bash
ollama pull phi3
```
Make sure the Ollama service is running in the background before starting the app.

### 5. Set up environment variables
```bash
cp .env.example .env
```
> ℹ️ No API key is required since the LLM runs locally — the `.env` file is kept for future flexibility (e.g. if you switch to a cloud LLM).

---

## ▶️ Usage

### 1. Run the app
```bash
streamlit run app.py
```

### 2. Upload your PDFs
- Go to the **Upload PDFs** section
- Select one or more PDF files
- Click **Process Documents** — wait for the success message confirming chunks were stored

### 3. Ask a question
- Type your question in the **Ask Questions** box
- Click **Ask**
- Read the generated answer, then expand the **Sources** section to see exactly which document chunks (and page numbers) the answer was based on

### 4. (Optional) Download your answer
- Click **Download Answer** to save the response as a PDF file for later reference

---

## 🧩 Architecture Workflow

```
        ┌──────────────┐
        │   Upload PDF  │
        └──────┬───────┘
               │
               ▼
   ┌────────────────────────┐
   │  Text Extraction (pypdf) │
   └───────────┬─────────────┘
               │
               ▼
   ┌────────────────────────┐
   │   Chunking (LangChain)   │
   └───────────┬─────────────┘
               │
               ▼
   ┌─────────────────────────────────┐
   │ Embeddings (Sentence-Transformers) │
   └───────────────┬─────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   ChromaDB (storage)  │
        └──────────┬───────────┘
                   │  (semantic search)
                   ▼
        ┌─────────────────────┐
        │ Relevant Chunks Found │
        └──────────┬───────────┘
                   │
                   ▼
   ┌────────────────────────────┐
   │ Prompt = Context + Question  │
   └─────────────┬──────────────┘
                   │
                   ▼
          ┌────────────────┐
          │ Local LLM (Ollama) │
          └────────┬─────────┘
                   │
                   ▼
          ┌────────────────┐
          │  Final Answer    │
          │ + Sources Shown  │
          └────────────────┘
```

**In short:** `PDF → Text → Chunks → Embeddings → ChromaDB → Retrieval → Prompt → LLM → Answer`

---

## 🚧 Future Enhancements

- [ ] Support for additional file types (Word, PowerPoint, plain text)
- [ ] Multi-turn conversational memory (follow-up questions)
- [ ] Highlight the exact answer span within the retrieved chunk
- [ ] Option to switch between local (Ollama) and cloud (OpenAI) models
- [ ] User authentication for multi-user/classroom use
- [ ] Deploy as a hosted web app for easier access
- [ ] Add evaluation metrics (answer relevance, retrieval accuracy)
- [ ] Support larger/better local models (e.g. Llama 3, Mistral)

---

## 📸 Screenshots

> _Screenshots will be added here once the UI is finalized._

| Upload & Process | Ask & Answer |
|---|---|
| *coming soon* | *coming soon* |

---

## 👤 Contributors

| Name | Role |
|---|---|
| *Your Name Here* | Developer / Author |

---

⭐ If you found this project useful, consider giving it a star on GitHub!