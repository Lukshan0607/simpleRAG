"""Sentence-transformer embeddings and the persistent Chroma vector store."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_documents"
DEFAULT_VECTORSTORE_DIR = Path(__file__).resolve().parent / "vectorstore"


def create_embedding_model() -> HuggingFaceEmbeddings:
    """Create the local Hugging Face model used for both documents and questions."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(
    persist_directory: str | Path = DEFAULT_VECTORSTORE_DIR,
    embedding_model: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    """Open (or create) a persistent Chroma collection using cosine distance."""
    directory = Path(persist_directory)
    directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model or create_embedding_model(),
        persist_directory=str(directory),
        collection_metadata={"hnsw:space": "cosine"},
    )
