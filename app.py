"""
app.py — Main Streamlit Application for the AI-Powered Academic Assistant

This is the entry point of the project. It:
  1. Provides a chat interface for students to ask questions.
  2. Handles PDF uploads and triggers document ingestion.
  3. Retrieves relevant chunks from ChromaDB using LangChain.
  4. Sends retrieved chunks + question to GPT-4o-mini and shows the answer.
  5. Displays source references (document name + page number).
"""

import os
import streamlit as st
from dotenv import load_dotenv

# LangChain components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Local module for ingesting PDFs
from ingest import ingest_pdfs

# ── Load environment variables from .env file ──────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Constants ──────────────────────────────────────────────────────────────────
CHROMA_DIR   = "./chroma_db"      # Where ChromaDB stores its data
DATA_DIR     = "./data"           # Where uploaded PDFs are saved
MODEL_NAME   = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K        = 4                  # Number of document chunks to retrieve per query


# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Academic RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS for a clean, academic look ─────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .main { background-color: #f8f9fa; }

    /* Chat message bubbles */
    .user-bubble {
        background: #4A90D9;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
    }
    .assistant-bubble {
        background: #ffffff;
        color: #1a1a2e;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        word-wrap: break-word;
    }

    /* Source citation boxes */
    .source-box {
        background: #eef4ff;
        border-left: 3px solid #4A90D9;
        padding: 8px 12px;
        border-radius: 0 8px 8px 0;
        margin-top: 8px;
        font-size: 0.82em;
        color: #555;
    }

    /*
     * NEW: chunk-text shows the actual retrieved passage.
     * It sits beneath its source-box header, styled like a
     * blockquote so it's clearly "quoted from the document".
     */
    .chunk-text {
        background: #f5f8ff;
        border-left: 3px solid #c5d8f5;
        padding: 8px 12px;
        margin: 4px 0 12px 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.80em;
        color: #444;
        font-style: italic;
        line-height: 1.5;
        white-space: pre-wrap;   /* preserves line breaks from the PDF */
    }

    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.1em;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 4px;
    }

    /* Status badges */
    .badge-success {
        background: #d4edda;
        color: #155724;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.78em;
        font-weight: 600;
    }
    .badge-info {
        background: #d1ecf1;
        color: #0c5460;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.78em;
        font-weight: 600;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────────
# Streamlit reruns the script on every interaction, so we use st.session_state
# to persist data (chat history, the QA chain, etc.) across reruns.

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []        # List of (question, answer, sources) tuples

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None          # Will hold the LangChain QA chain

if "docs_ingested" not in st.session_state:
    st.session_state.docs_ingested = False    # Flag: have PDFs been processed?

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []      # List of ingested PDF filenames


# ── Helper: load or reload the RAG chain ──────────────────────────────────────
def load_qa_chain():
    """
    Builds the ConversationalRetrievalChain:
      - Connects to the ChromaDB vector store.
      - Uses OpenAI embeddings to encode queries.
      - Uses GPT-4o-mini as the answer-generating LLM.
      - Maintains conversation memory across turns.
    """
    # Step 1: Create embedding model (same one used during ingestion)
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

    # Step 2: Connect to the existing ChromaDB collection
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="academic_docs",
    )

    # Step 3: Create a retriever — this is what fetches the top-K relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",   # Cosine similarity search
        search_kwargs={"k": TOP_K}, # Return the 4 most relevant chunks
    )

    # Step 4: Initialise the LLM (GPT-4o-mini is fast and cost-effective)
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0.2,            # Low temperature = more factual, less creative
        openai_api_key=OPENAI_API_KEY,
    )

    # Step 5: Conversation memory — stores previous Q&A pairs so the chatbot
    #         understands follow-up questions like "Can you explain that further?"
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",        # Tell memory which output key to store
    )

    # Step 6: Build the full RAG chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  # We want to display the source chunks
        verbose=False,
    )

    return qa_chain


# ── Helper: render source citations for a list of retrieved chunks ─────────────
def render_sources(source_documents: list, expanded: bool = False):
    """
    Gives each retrieved chunk its own collapsible expander, showing:
      - PDF filename and page number in the expander header
      - The retrieved text passage inside the expander body
    """
    if not source_documents:
        return  # Nothing to show — exit early

    st.caption(f"📎 {len(source_documents)} source chunk(s) retrieved")

    for i, doc in enumerate(source_documents):

        # ── 1. Pull out filename and page number ──────────────────────────
        source_path = doc.metadata.get("source", "Unknown document")
        raw_page    = doc.metadata.get("page", 0)

        # PyPDF numbers pages from 0 internally — add 1 to match PDF viewer
        human_page = raw_page + 1

        # Strip folder path: "./data/notes.pdf"  →  "notes.pdf"
        file_label = os.path.basename(source_path)

        # ── 2. Build the expander label ───────────────────────────────────
        expander_label = f"📄 {file_label} — Page {human_page}"

        # ── 3. One expander per chunk ─────────────────────────────────────
        with st.expander(expander_label, expanded=expanded):
            st.caption(f"Chunk {i + 1} of {len(source_documents)}")
            st.markdown(doc.page_content.strip())
            # ── Render: header badge then quoted text ─────────────────────
            st.markdown(
                f'<div class="source-box">'
                f'📄 <strong>{file_label}</strong> &nbsp;—&nbsp; Page {human_page}'
                f'&nbsp; <span style="color:#999;font-size:0.85em;">'
                f'(chunk {i + 1} of {len(source_documents)})</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="chunk-text">{chunk_text}</div>',
                unsafe_allow_html=True,
            )


def check_existing_db():
    """Returns True if the ChromaDB directory has content from a previous session."""
    return (
        os.path.exists(CHROMA_DIR)
        and len(os.listdir(CHROMA_DIR)) > 0
    )


# ── Auto-load chain if DB already exists (e.g., from a previous session) ──────
if check_existing_db() and st.session_state.qa_chain is None:
    with st.spinner("🔄 Loading existing knowledge base..."):
        st.session_state.qa_chain = load_qa_chain()
        st.session_state.docs_ingested = True


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Academic RAG Assistant")
    st.markdown("*Upload your study materials and ask questions.*")
    st.divider()

    # ── PDF Upload Section ────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">📂 Upload Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        label="Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload lecture notes, assignments, or any academic PDFs.",
    )

    # Process button — only shown after files are selected
    if uploaded_files:
        if st.button("⚡ Process & Index PDFs", type="primary", use_container_width=True):
            os.makedirs(DATA_DIR, exist_ok=True)

            # Save uploaded files to the data/ directory
            saved_paths = []
            for uploaded_file in uploaded_files:
                save_path = os.path.join(DATA_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(save_path)

            # Run the ingestion pipeline (defined in ingest.py)
            with st.spinner("🔍 Extracting text, chunking, and embedding PDFs..."):
                num_chunks = ingest_pdfs(DATA_DIR, CHROMA_DIR, OPENAI_API_KEY, EMBEDDING_MODEL)

            # Load the QA chain after ingestion
            st.session_state.qa_chain = load_qa_chain()
            st.session_state.docs_ingested = True
            st.session_state.ingested_files = [f.name for f in uploaded_files]

            st.success(f"✅ Indexed {num_chunks} chunks from {len(uploaded_files)} PDF(s)!")

    st.divider()

    # ── Knowledge Base Status ────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">🗄️ Knowledge Base</div>', unsafe_allow_html=True)

    if st.session_state.docs_ingested:
        st.markdown('<span class="badge-success">● Active</span>', unsafe_allow_html=True)
        if st.session_state.ingested_files:
            st.markdown("**Loaded files:**")
            for fname in st.session_state.ingested_files:
                st.markdown(f"- 📄 {fname}")
    else:
        st.markdown('<span class="badge-info">○ No documents loaded</span>', unsafe_allow_html=True)
        st.info("Upload PDFs above to get started.")

    st.divider()

    # ── Clear Chat ────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        # Rebuild chain to reset memory
        if st.session_state.qa_chain:
            st.session_state.qa_chain = load_qa_chain()
        st.rerun()

    st.divider()

    # ── Settings Info ─────────────────────────────────────────────────────────
    with st.expander("⚙️ Settings"):
        st.markdown(f"**LLM:** `{MODEL_NAME}`")
        st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}`")
        st.markdown(f"**Chunks retrieved:** `{TOP_K}`")
        st.markdown(f"**Vector DB:** ChromaDB")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🎓 Academic Assistant")
st.markdown("Ask questions about your uploaded study materials. I answer *only* using the content in your documents.")
st.divider()

# ── Render existing chat history ──────────────────────────────────────────────
for question, answer, sources in st.session_state.chat_history:
    # User message
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)

    # Assistant message
    with st.chat_message("assistant", avatar="📚"):
        st.markdown(answer)

        # Show source references for this historical message.
        # expanded=False keeps old messages tidy — the student can
        # click open any one they want to re-read.
        render_sources(sources, expanded=False)


# ── Chat input box ─────────────────────────────────────────────────────────────
user_question = st.chat_input(
    placeholder="Ask a question about your documents...",
    disabled=not st.session_state.docs_ingested,
)

# ── If the user has typed a question ─────────────────────────────────────────
if user_question:
    if not st.session_state.qa_chain:
        st.error("⚠️ Please upload and process PDF documents first.")
        st.stop()

    # Show the user's message immediately
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_question)

    # Generate the answer using the RAG chain
    with st.chat_message("assistant", avatar="📚"):
        with st.spinner("🔍 Searching documents and generating answer..."):
            # The chain: retrieves chunks → formats prompt → calls GPT → returns answer
            result = st.session_state.qa_chain.invoke({"question": user_question})

            answer          = result.get("answer", "I could not find an answer in the uploaded documents.")
            source_documents = result.get("source_documents", [])

        st.markdown(answer)

        # Show source references for the new answer.
        # expanded=True pops it open immediately so the student sees
        # exactly which chunks the AI used to form this response.
        render_sources(source_documents, expanded=True)

    # Persist this exchange to session history
    st.session_state.chat_history.append((user_question, answer, source_documents))


# ── Empty state prompt ────────────────────────────────────────────────────────
if not st.session_state.chat_history and st.session_state.docs_ingested:
    st.markdown("""
    <div style='text-align:center; color:#888; padding:40px 0;'>
        <h3 style='color:#aaa;'>💬 Ask your first question!</h3>
        <p>Try: <em>"Summarise the key topics in this document."</em></p>
        <p>Or: <em>"What does the document say about neural networks?"</em></p>
    </div>
    """, unsafe_allow_html=True)

if not st.session_state.docs_ingested:
    st.markdown("""
    <div style='text-align:center; color:#888; padding:60px 0;'>
        <h2 style='color:#aaa;'>📂 Start by uploading your PDFs</h2>
        <p>Use the sidebar on the left to upload lecture notes, assignments, or any academic PDF.</p>
        <p>Once processed, you can ask questions and get answers grounded in your documents.</p>
    </div>
    """, unsafe_allow_html=True)
