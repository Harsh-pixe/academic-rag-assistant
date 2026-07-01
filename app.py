"""
app.py — Fully Local Streamlit Application for the AI-Powered Academic Assistant
"""

import os
import streamlit as st

# LangChain components
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

# Local module for ingesting PDFs
from ingest import ingest_pdfs
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

CHROMA_DIR   = "./chroma_db"
DATA_DIR     = "./data"
MODEL_NAME   = "llama3.2"
TOP_K        = 4

st.set_page_config(
    page_title="Academic RAG Assistant (Local)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "docs_ingested" not in st.session_state:
    st.session_state.docs_ingested = False
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="academic_docs",
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,
    streaming=False)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you don't know. Answer strictly based on the text.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def generate_pdf(question, answer):
    """
    Creates a PDF containing the user's question and the AI's answer.
    Returns the PDF as bytes.
    """
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    elements = [
        Paragraph("<b>AI-Powered Academic Assistant</b>", styles["Title"]),
        Paragraph("<br/>", styles["Normal"]),
        Paragraph(f"<b>Question:</b><br/>{question}", styles["BodyText"]),
        Paragraph("<br/>", styles["Normal"]),
        Paragraph(f"<b>Answer:</b><br/>{answer}", styles["BodyText"]),
    ]

    doc.build(elements)

    pdf = buffer.getvalue()
    buffer.close()

    return pdf

def render_sources(source_documents: list, expanded: bool = False):
    if not source_documents:
        return
    with st.expander("📎 Sources used", expanded=expanded):
        for i, doc in enumerate(source_documents):
            source_path = doc.metadata.get("source", "Unknown document")
            raw_page    = doc.metadata.get("page", 0)
            file_label = os.path.basename(source_path)
            chunk_text = doc.page_content.strip()
            if len(chunk_text) > 400:
                chunk_text = chunk_text[:400] + "…"
            st.markdown(f'📄 <strong>{file_label}</strong> — Page {raw_page + 1}', unsafe_allow_html=True)
            st.info(chunk_text)

if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0 and st.session_state.qa_chain is None:
    st.session_state.qa_chain = load_qa_chain()
    st.session_state.docs_ingested = True

with st.sidebar:
    st.markdown("## 📚 Local Academic RAG")

    uploaded_files = st.file_uploader(
        label="Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("⚡ Process & Index PDFs", type="primary", use_container_width=True):
            os.makedirs(DATA_DIR, exist_ok=True)

            for uploaded_file in uploaded_files:
                with open(os.path.join(DATA_DIR, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

            with st.spinner("🔍 Processing locally..."):
                num_chunks = ingest_pdfs(DATA_DIR)

            st.session_state.qa_chain = load_qa_chain()
            st.session_state.docs_ingested = True
            st.session_state.ingested_files = [f.name for f in uploaded_files]

            st.success(f"✅ Indexed {num_chunks} chunks!")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.markdown("# 🎓 Local Academic Assistant")
for question, answer, sources in st.session_state.chat_history:
    with st.chat_message("user"): st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(answer)
        render_sources(sources, expanded=False)

# Prepare chat history for download

chat_history_text = ""

for question, answer, _ in st.session_state.chat_history:
    chat_history_text += f"User: {question}\n"
    chat_history_text += f"Assistant: {answer}\n"
    chat_history_text += "-" * 60 + "\n"

# Download Chat History

st.download_button(
    label="📥 Download Chat History",
    data=chat_history_text,
    file_name="chat_history.txt",
    mime="text/plain",
    use_container_width=True,
)

user_question = st.chat_input(placeholder="Ask a question...", disabled=not st.session_state.docs_ingested)
if user_question:
    with st.chat_message("user"): st.markdown(user_question)
    with st.chat_message("assistant"):
        with st.spinner("🔍 Thinking..."):
            lcel_history = []
            for q, a, _ in st.session_state.chat_history:
                lcel_history.extend([HumanMessage(content=q), AIMessage(content=a)])
            result = st.session_state.qa_chain.invoke({"input": user_question, "chat_history": lcel_history})
            answer = result.get("answer", "No response.")
            sources = result.get("context", [])
        st.markdown(answer)
        render_sources(sources, expanded=True)
        pdf_file = generate_pdf(user_question, answer)

        st.download_button(
            label="📄 Download Answer as PDF",
            data=pdf_file,
            file_name="academic_answer.pdf",
            mime="application/pdf",)
    st.session_state.chat_history.append((user_question, answer, sources))

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:14px;">
        🎓 <b>AI-Powered Academic Assistant using Retrieval-Augmented Generation (RAG)</b><br>
        Developed by <b>Harsh</b> & <b>Karim</b>
    </div>
    """,
    unsafe_allow_html=True,
)
