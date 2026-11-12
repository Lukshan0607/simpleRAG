"""Streamlit user interface for the NLP-powered PDF question-answering app."""

from __future__ import annotations

import streamlit as st
from langchain_core.language_models.chat_models import BaseChatModel

from embeddings import create_vector_store
from ingest import ingest_pdfs
from rag_pipeline import answer_question, create_llm


st.set_page_config(
    page_title="NLP-Powered AI PDF Assistant",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def get_vector_store():
    """Keep one persistent vector-store connection for this Streamlit process."""
    return create_vector_store()


@st.cache_resource
def get_llm() -> BaseChatModel:
    """Reuse the Gemini client across questions."""
    return create_llm()


def _render_sources(sources: list[dict]) -> None:
    """Display only the passages returned by the vector-store search."""
    if not sources:
        return

    st.markdown("**Sources**")
    for index, source in enumerate(sources, start=1):
        label = f"{source['filename']} · page {source['page']} · relevance {source['score']:.2f}"
        with st.expander(f"Source {index}: {label}"):
            st.write(source["text"])


st.title("NLP-Powered AI PDF Assistant")
st.caption("Ask questions about your PDFs and see the passages behind each answer.")

with st.sidebar:
    st.header("1. Add PDF documents")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text-based PDFs are supported. Scanned PDFs need OCR before upload.",
    )
    if uploaded_files:
        st.markdown("**Selected files**")
        for uploaded_file in uploaded_files:
            st.write(f"📄 {uploaded_file.name}")

    process_clicked = st.button(
        "Process documents",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )
    clear_clicked = st.button("Clear chat", use_container_width=True)
    if clear_clicked:
        st.session_state["messages"] = []

    with st.expander("How this assistant works"):
        st.write(
            "PDF text is split into overlapping passages and represented as "
            "embeddings: numerical vectors that place passages with similar "
            "meaning near one another. A question is embedded in the same way, "
            "then the three closest passages are sent to Google's Gemini API."
        )
        st.write(
            "NLTK tokenization, stop-word filtering, and lemmatization support "
            "text analysis. spaCy identifies sentence boundaries. These steps "
            "do not remove words from the passages used for search or answers."
        )

if "messages" not in st.session_state:
    st.session_state["messages"] = []

try:
    with st.spinner("Loading the local embedding model and document index..."):
        vector_store = get_vector_store()
except Exception as error:
    st.error(
        "The embedding model or ChromaDB could not be initialized. "
        f"Check the installation and try again. Details: {error}"
    )
    st.stop()

if process_clicked and uploaded_files:
    pdfs = [(uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in uploaded_files]
    with st.spinner("Extracting, analyzing, chunking, and indexing PDF text..."):
        result = ingest_pdfs(pdfs, vector_store)

    if result.processed_files:
        st.success(
            f"Processed {len(result.processed_files)} document(s) "
            f"and indexed {result.chunks_added} text chunk(s)."
        )
    for message in result.skipped_files:
        st.info(message)
    for message in result.errors:
        st.error(f"Could not process {message}")

try:
    indexed_documents = vector_store.get(include=["metadatas"]).get("ids", [])
except Exception as error:
    st.error(f"Could not read the local document index. Details: {error}")
    st.stop()

if not indexed_documents:
    st.info("Upload PDF documents in the sidebar and select **Process documents** to begin.")

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("sources", []))

question = st.chat_input(
    "Ask a question about your processed PDFs...",
    disabled=not indexed_documents,
)

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your PDFs and drafting an evidence-based answer..."):
            try:
                result = answer_question(question, vector_store, get_llm())
                source_details = [
                    {
                        "filename": str(source.document.metadata.get("source", "Unknown PDF")),
                        "page": source.document.metadata.get("page", "unknown"),
                        "score": source.relevance_score,
                        "text": source.document.page_content,
                    }
                    for source in result.sources
                ]
                st.markdown(result.answer)
                _render_sources(source_details)
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "sources": source_details,
                    }
                )
            except Exception as error:
                st.error(
                    "I couldn't complete the answer. Check your Gemini API key, "
                    "internet connection, and available API quota. "
                    f"Details: {error}"
                )
