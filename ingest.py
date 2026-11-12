"""PDF text extraction, NLP analysis, sentence-aware chunking, and indexing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from io import BytesIO
from typing import Iterable

from langchain_core.documents import Document
from langchain_chroma import Chroma
from pypdf import PdfReader

from nlp_preprocessing import preprocess_text


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


@dataclass
class IngestionSummary:
    """Results and per-file problems returned to the user interface."""

    processed_files: list[str] = field(default_factory=list)
    chunks_added: int = 0
    skipped_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _split_long_sentence(sentence: str, max_characters: int) -> list[str]:
    """Break a sentence longer than a chunk at word boundaries."""
    parts: list[str] = []
    current = ""
    for word in sentence.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_characters:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _overlap_tail(text: str, overlap: int) -> str:
    """Return a trailing overlap, preferring to begin at a word boundary."""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    first_space = tail.find(" ")
    return tail[first_space + 1 :] if first_space >= 0 else tail


def chunk_sentences(
    sentences: list[str],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Build bounded chunks while keeping sentence boundaries where possible."""
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        for piece in _split_long_sentence(sentence.strip(), chunk_size):
            if not piece:
                continue
            candidate = f"{current} {piece}".strip()
            if not current or len(candidate) <= chunk_size:
                current = candidate
                continue

            chunks.append(current)
            tail = _overlap_tail(current, overlap)
            current = f"{tail} {piece}".strip()
            if len(current) > chunk_size:
                current = piece

    if current:
        chunks.append(current)
    return chunks


def _extract_page_texts(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Extract non-empty text from pages, keeping 1-based PDF page numbers."""
    reader = PdfReader(BytesIO(pdf_bytes), strict=False)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("the PDF is password-protected")

    page_texts: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            page_texts.append((page_number, text))
    return page_texts


def ingest_pdfs(
    pdf_files: Iterable[tuple[str, bytes]],
    vector_store: Chroma,
) -> IngestionSummary:
    """Extract and index uploaded PDFs, replacing older versions by filename.

    Only whitespace is normalized for chunking. Lowercasing, stop-word removal,
    and lemmatization are analysis outputs; the source wording is embedded.
    """
    summary = IngestionSummary()

    for filename, pdf_bytes in pdf_files:
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        try:
            page_texts = _extract_page_texts(pdf_bytes)
            new_documents: list[Document] = []
            new_ids: list[str] = []

            for page_number, page_text in page_texts:
                analysis = preprocess_text(page_text)
                chunks = chunk_sentences(analysis.sentences)
                for chunk_number, chunk in enumerate(chunks, start=1):
                    document_id = hashlib.sha256(
                        f"{filename}:{file_hash}:{page_number}:{chunk_number}".encode("utf-8")
                    ).hexdigest()
                    new_ids.append(document_id)
                    new_documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": filename,
                                "page": page_number,
                                "chunk": chunk_number,
                                "file_hash": file_hash,
                            },
                        )
                    )

            if not new_documents:
                previous = vector_store.get(
                    where={"source": filename},
                    include=["metadatas"],
                )
                previous_ids = previous.get("ids", [])
                if previous_ids:
                    vector_store.delete(ids=previous_ids)
                summary.skipped_files.append(
                    f"{filename}: no extractable text found; scanned PDFs need OCR first. "
                    "Any older indexed version with this filename was removed."
                )
                continue

            existing = vector_store.get(
                where={"source": filename},
                include=["metadatas"],
            )
            existing_ids = existing.get("ids", [])
            existing_metadata = existing.get("metadatas", []) or []
            if (
                existing_ids
                and len(existing_metadata) == len(existing_ids)
                and existing_metadata
                and all(metadata and metadata.get("file_hash") == file_hash for metadata in existing_metadata)
            ):
                summary.skipped_files.append(f"{filename}: already processed; skipped duplicate.")
                continue

            vector_store.add_documents(new_documents, ids=new_ids)
            obsolete_ids = [document_id for document_id in existing_ids if document_id not in new_ids]
            if obsolete_ids:
                vector_store.delete(ids=obsolete_ids)

            summary.processed_files.append(filename)
            summary.chunks_added += len(new_documents)
        except Exception as error:
            summary.errors.append(f"{filename}: {error}")

    return summary
