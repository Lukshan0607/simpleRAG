"""Semantic retrieval and evidence-grounded answer generation with Gemini."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from nlp_preprocessing import preprocess_text


TOP_K = 3
MIN_RELEVANCE_SCORE = 0.20
NO_ANSWER = "I couldn't find that information in the processed PDF documents."
GEMINI_MODEL = "gemini-3.5-flash-lite"
load_dotenv(Path(__file__).resolve().parent / ".env")


def create_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini API client using a key from the environment."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        raise RuntimeError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY before asking a question. "
            "Create a key in Google AI Studio."
        )
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        google_api_key=api_key,
    )


@dataclass(frozen=True)
class SourceEvidence:
    """A retrieved PDF passage and its similarity relevance score."""

    document: Document
    relevance_score: float


@dataclass(frozen=True)
class AnswerResult:
    """The answer plus the exact passages used to generate it."""

    answer: str
    sources: list[SourceEvidence]
    question_terms: list[str]


def answer_question(
    question: str,
    vector_store: Chroma,
    llm: BaseChatModel | None = None,
) -> AnswerResult:
    """Retrieve the three best passages and answer only from their contents."""
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Enter a question before submitting.")

    question_analysis = preprocess_text(cleaned_question)
    matches = vector_store.similarity_search_with_relevance_scores(
        cleaned_question,
        k=TOP_K,
    )
    sources = [
        SourceEvidence(document=document, relevance_score=float(score))
        for document, score in matches
        if score >= MIN_RELEVANCE_SCORE
    ]

    if not sources:
        return AnswerResult(
            answer=NO_ANSWER,
            sources=[],
            question_terms=question_analysis.filtered_tokens,
        )

    context = "\n\n".join(
        (
            f"[Source {index}: {source.document.metadata.get('source', 'Unknown PDF')}, "
            f"page {source.document.metadata.get('page', 'unknown')}]\n"
            f"{source.document.page_content}"
        )
        for index, source in enumerate(sources, start=1)
    )
    prompt = (
        "You answer questions using only the supplied PDF excerpts. "
        "Do not use outside knowledge. If the excerpts do not contain enough "
        "evidence to answer, reply exactly: "
        f'"{NO_ANSWER}" '
        "Do not invent facts, quotations, page numbers, or citations. "
        "Give a clear, concise answer. Do not output source labels; the application "
        "displays the exact retrieved sources separately.\n\n"
        "PDF excerpts:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    response = (llm or create_llm()).invoke(
        prompt.format(context=context, question=cleaned_question)
    )
    content = response.content
    if isinstance(content, str):
        answer = content.strip()
    elif isinstance(content, list):
        answer = "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ).strip()
    else:
        answer = ""
    if not answer:
        raise RuntimeError("Gemini returned no text content for the answer.")

    return AnswerResult(
        answer=answer,
        sources=sources,
        question_terms=question_analysis.filtered_tokens,
    )
