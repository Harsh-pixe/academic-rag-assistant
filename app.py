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
