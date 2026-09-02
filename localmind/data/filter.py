"""Stage 1-3 of the §7 pipeline: license filter -> language ID -> quality heuristics,
plus the PII scrub that runs after near-dedup.

Pure stdlib + no heavy deps, so this module is safe to import even when the `data`
extra (``datasets``, ``datasketch``) is not installed. Language ID and quality scoring
are small deterministic heuristics rather than a fastText/langid model -- the spec only
pins exact values for MinHash-LSH dedup and 13-gram decontamination (see `dedup.py`);
everything else here is a from-scratch, dependency-free stand-in that is good enough to
demonstrate (and unit-test) the pipeline stage offline.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace

__all__ = [
    "QualityThresholds",
    "RawDoc",
    "detect_language",
    "filter_language",
    "filter_license",
    "filter_quality",
    "quality_ok",
    "scrub_docs",
    "scrub_pii",
]


@dataclass(frozen=True, slots=True)
class RawDoc:
    """One document flowing through the pipeline, before tokenization."""

    text: str
    source: str
    license: str


# ---------------------------------------------------------------------------------
# License filter
# ---------------------------------------------------------------------------------
def filter_license(docs: Iterable[RawDoc], allowed_licenses: Iterable[str]) -> Iterator[RawDoc]:
    """Keep only documents whose recorded license is in `allowed_licenses`.

    Every source in the §7 mixture carries an explicit license string (see
    `configs/data/mixture.yaml`); this is the enforcement point.
    """
    allowed = set(allowed_licenses)
    for doc in docs:
        if doc.license in allowed:
            yield doc


# ---------------------------------------------------------------------------------
# Language ID (heuristic)
# ---------------------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z']+")
_EN_STOPWORDS = frozenset(
    [
        "the",
        "and",
        "is",
        "of",
        "to",
        "in",
        "a",
        "that",
        "it",
        "for",
        "on",
        "with",
        "as",
        "was",
        "are",
        "this",
        "be",
        "by",
        "an",
        "at",
        "from",
        "or",
        "have",
        "has",
        "had",
        "not",
        "you",
        "i",
        "we",
        "they",
        "he",
        "she",
        "his",
        "her",
        "their",
        "its",
        "but",
        "if",
        "then",
        "so",
    ]
)


def detect_language(text: str) -> str:
    """Cheap heuristic language ID: "en" if the text is ASCII-dominant and contains a
    plausible density of common English stopwords, else "unknown".

    Deterministic and dependency-free (no fastText/langid model), which matters here
    since tests must run fully offline against synthetic corpora.
    """
    words = _WORD_RE.findall(text.lower())
    if not words:
        return "unknown"
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "unknown"
    ascii_ratio = sum(1 for c in letters if ord(c) < 128) / len(letters)
    stop_ratio = sum(1 for w in words if w in _EN_STOPWORDS) / len(words)
    if ascii_ratio > 0.85 and stop_ratio > 0.03:
        return "en"
    return "unknown"


def filter_language(
    docs: Iterable[RawDoc], allowed_languages: Iterable[str] = ("en",)
) -> Iterator[RawDoc]:
    allowed = set(allowed_languages)
    for doc in docs:
        if detect_language(doc.text) in allowed:
            yield doc


# ---------------------------------------------------------------------------------
# Quality heuristics (Gopher/C4-style, simplified)
# ---------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class QualityThresholds:
    min_words: int = 20
    max_words: int = 200_000
    min_avg_word_len: float = 2.0
    max_avg_word_len: float = 12.0
    min_alpha_ratio: float = 0.6


def quality_ok(text: str, thresholds: QualityThresholds = QualityThresholds()) -> bool:
    """Reject too-short/too-long documents, documents with implausible average word
    length (boilerplate, code dumps mislabeled as prose, symbol soup), and documents
    that are mostly non-alphabetic characters.
    """
    words = text.split()
    n_words = len(words)
    if not (thresholds.min_words <= n_words <= thresholds.max_words):
        return False
    avg_word_len = sum(len(w) for w in words) / n_words
    if not (thresholds.min_avg_word_len <= avg_word_len <= thresholds.max_avg_word_len):
        return False
    non_space = sum(1 for c in text if not c.isspace())
    if non_space == 0:
        return False
    alpha_ratio = sum(1 for c in text if c.isalpha()) / non_space
    return alpha_ratio >= thresholds.min_alpha_ratio


def filter_quality(
    docs: Iterable[RawDoc], thresholds: QualityThresholds = QualityThresholds()
) -> Iterator[RawDoc]:
    for doc in docs:
        if quality_ok(doc.text, thresholds):
            yield doc


# ---------------------------------------------------------------------------------
# PII scrub (runs after near-dedup, per the §7 pipeline order)
# ---------------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")


def scrub_pii(text: str) -> str:
    """Replace emails, SSN-shaped numbers, IPv4 addresses, and US-style phone numbers
    with placeholder tokens. Order matters: SSNs/IPv4 are checked before the looser
    phone pattern so they aren't partially swallowed by it.
    """
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _SSN_RE.sub("[SSN]", text)
    text = _IPV4_RE.sub("[IP]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    return text


def scrub_docs(docs: Iterable[RawDoc]) -> Iterator[RawDoc]:
    for doc in docs:
        yield replace(doc, text=scrub_pii(doc.text))
