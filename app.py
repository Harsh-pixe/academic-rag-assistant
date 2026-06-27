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
# CHANGE: added page_icon so the browser tab looks more polished
st.set_page_config(page_title="RAG Assistant", page_icon="📚", layout="wide")

# CHANGE: added a sidebar with project info (does not affect app logic at all,
# it's just a separate panel Streamlit renders next to the main page)
with st.sidebar:
    st.header("ℹ️ About this Project")
    st.markdown(
        """
        **AI-Powered Academic Assistant**

        A local Retrieval-Augmented Generation (RAG)
        system that answers questions using **only**
        the content of your uploaded PDFs.
        """
    )

    # CHANGE: small divider for visual separation in the sidebar
    st.divider()

    st.subheader("⚙️ Tech Stack")
    st.markdown(
        """
        - 🐍 Python + Streamlit
        - 🔗 LangChain
        - 🗄️ ChromaDB (vector storage)
        - 🧠 Sentence-Transformers (embeddings)
        - 🤖 Ollama — `phi3` (local LLM)
        """
    )

    st.divider()

    st.subheader("📝 How to Use")
    st.markdown(
        """
        1. Upload one or more PDFs
        2. Click **Process Documents**
        3. Type your question
        4. Click **Ask**
        """
    )

st.title("📚 Local AI Academic Assistant (RAG)")
# CHANGE: added a short caption under the title for extra context/spacing
st.caption("Upload your study material and ask questions — answers come only from your documents.")

# CHANGE: extra blank line equivalent for breathing room before the first section
st.write("")

# Upload
# CHANGE: added an icon and changed header level (st.subheader) so it's visually
# smaller than the main title, giving a cleaner heading hierarchy
st.subheader("📤 Step 1: Upload PDFs")

files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# CHANGE: wrapped button in a column so it doesn't stretch the full page width
# (purely cosmetic spacing change, does not change what the button does)
col1, _ = st.columns([1, 4])
with col1:
    process_clicked = st.button("⚙️ Process Documents", use_container_width=True)

if process_clicked:
    if files:
        # CHANGE: spinner gives feedback while files are being processed,
        # instead of the page looking "stuck" with no response
        with st.spinner("Processing documents, please wait..."):
            for f in files:
                path = os.path.join(UPLOAD_DIR, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())

                chunks = ingest_pdf(path, f.name)
                # CHANGE: added an icon to the success message for clearer visual feedback
                st.success(f"✅ **{f.name}** processed — {chunks} chunks stored")
    else:
        # CHANGE: added an explicit error message when no files are selected
        # (previously clicking the button with no files did nothing visible)
        st.error("⚠️ Please upload at least one PDF before processing.")


st.divider()

# Chat
# CHANGE: added icon + step numbering to match the "Step 1 / Step 2" flow .
st.subheader("💬 Step 2: Ask Questions")

question = st.text_input("Enter your question", placeholder="e.g. What is the main topic of chapter 2?")

# CHANGE:same column trick as above, just for consistent button sizing/spacing
col2, _ = st.columns([1, 4])
with col2:
    ask_clicked = st.button("🔍 Ask", use_container_width=True)

# CHANGE: track the last answer in session_state so the Download button
# (further down) can still access it after Streamlit reruns the script.
# This does not change the RAG logic — it only fixes the UI so the
# Download button reliably has an answer to export.
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if ask_clicked:
    if question:
        # CHANGE: spinner while retrieving + generating, so the user knows
        # the app is working instead of appearing frozeN
        with st.spinner("Searching documents and generating answer..."):
            chunks = retrieve_chunks(question)

            if not chunks:
                # CHANGE: added icon to warning message
                st.warning("⚠️ No data found. Please upload and process PDFs first.")
            else:
                prompt = build_prompt(question, chunks)
                answer = generate_answer(prompt)

                # CHANGE: save answer to session_state so it survives for the
                # Download Answer button below
                st.session_state.last_answer = answer

                st.write("")  # CHANGE: small spacing before the answer block
                st.subheader("✅ Answer")
                st.write(answer)

                st.write("")  # CHANGE: small spacing before the sources block
                st.subheader("📑 Sources")
                for i, c in enumerate(chunks):
                    source = c.metadata.get("source", "unknown")
                    page = c.metadata.get("page", "N/A")

                    with st.expander(f"📄 Chunk {i+1} | {source} | Page {page}"):
                        st.write(c.page_content)
    else:
        # CHANGE: added explicit error message when the question box is empty
        st.error("⚠️ Please type a question before clicking Ask.")


def export_pdf(text):
    file = "answer.pdf"
    c = canvas.Canvas(file)
    c.drawString(100, 800, text[:1000])
    c.save()
    return file


st.write("")  # CHANGE: spacing before the download section
st.divider()

# CHANGE: added a small heading so this section doesn't look orphaned
st.subheader("📥 Step 3 (Optional): Download Your Answer")

# CHANGE: column wrapper for consistent button sizing/spacing
col3, _ = st.columns([1, 4])
with col3:
    download_clicked = st.button("📥 Download Answer", use_container_width=True)

if download_clicked:
    # CHANGE: use the saved session_state answer instead of the old "answer"
    # variable, which only existed inside the "if ask_clicked" block above
    # and would crash this button with a NameError if clicked separately.
    if st.session_state.last_answer:
        file = export_pdf(st.session_state.last_answer)
        # CHANGE: added icon + clearer wording to the success message
        st.success("✅ PDF generated successfully! Check your project folder for 'answer.pdf'.")
    else:
        # CHANGE: added a friendly error instead of letting the app crash
        # when there is no answer yet to export
        st.error("⚠️ No answer available yet. Please ask a question first.")

# -----------------------------
# FOOTER
# -----------------------------
# CHANGE: added a footer with developer names, separated visually with a divider
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.85em;'>
        Developed by <b>Harsh and Karim</b> &nbsp;|&nbsp; AI-Powered Academic Assistant (RAG) Project
    </div>""",
    unsafe_allow_html=True,)
