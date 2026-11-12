"""NLP helpers used for analysis and sentence-aware text chunking.

The original PDF text is deliberately kept separate from the normalized
tokens: embeddings and answer sources use the original wording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import nltk
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from spacy.language import Language
from spacy.lang.en.stop_words import STOP_WORDS


@dataclass(frozen=True)
class PreprocessingResult:
    """A sentence and token analysis that does not replace the source text."""

    sentences: list[str]
    tokens: list[str]
    filtered_tokens: list[str]
    lemmatized_tokens: list[str]


@lru_cache(maxsize=1)
def _load_nlp() -> tuple[Language, bool]:
    """Load spaCy's English model, or a lightweight sentence-splitting fallback."""
    try:
        nlp = spacy.load("en_core_web_sm")
        has_model_lemmas = True
    except OSError:
        nlp = spacy.blank("en")
        has_model_lemmas = False

    if not any(pipe in nlp.pipe_names for pipe in ("parser", "senter", "sentencizer")):
        nlp.add_pipe("sentencizer")
    return nlp, has_model_lemmas


@lru_cache(maxsize=1)
def _wordnet_available() -> bool:
    """Check whether the optional NLTK WordNet data is installed."""
    try:
        nltk.data.find("corpora/wordnet")
        return True
    except LookupError:
        return False


def segment_sentences(text: str) -> list[str]:
    """Split text into sentences with spaCy while retaining each sentence's words."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    nlp, _ = _load_nlp()
    return [sentence.text.strip() for sentence in nlp(normalized).sents if sentence.text.strip()]


def preprocess_text(text: str) -> PreprocessingResult:
    """Tokenize and normalize text for NLP analysis, not for answer generation."""
    sentences = segment_sentences(text)
    tokens = [
        token.lower()
        for sentence in sentences
        for token in word_tokenize(sentence, preserve_line=True)
        if re.search(r"[a-z0-9]", token, flags=re.IGNORECASE)
    ]
    filtered_tokens = [token for token in tokens if token not in STOP_WORDS]

    nlp, has_model_lemmas = _load_nlp()
    if has_model_lemmas:
        lemmatized_tokens = [
            token.lemma_.lower()
            for sentence in sentences
            for token in nlp(sentence)
            if token.text.lower() not in STOP_WORDS and token.is_alpha
        ]
    elif _wordnet_available():
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    else:
        lemmatized_tokens = filtered_tokens.copy()

    return PreprocessingResult(
        sentences=sentences,
        tokens=tokens,
        filtered_tokens=filtered_tokens,
        lemmatized_tokens=lemmatized_tokens,
    )
