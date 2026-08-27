"""BM25 — sparse lexical retrieval.

Two implementations, on purpose (see `docs/decisions` note in the Phase 8 report):

- `BM25Index` — BM25 written from scratch, so the `k1` term-frequency-saturation knob and the
  `b` document-length-normalization knob are things we've actually implemented, not just cited.
  Verified against hand-computed scores on a tiny 2-document corpus in `tests/test_retrieval.py`.
- `BM25SIndex` — a thin wrapper around the `bm25s` library (lazy-imported; part of the `rag`
  extra), which is what a real deployment should use: it is a highly-optimized, Lucene-style
  BM25 implementation. `BM25Index` is for understanding; `BM25SIndex` is for production.

Both satisfy the `RetrievalArm` protocol from `localmind.retrieval`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from localmind.retrieval import Document, ScoredDoc

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Deterministic, dependency-free tokenizer: lowercase, keep alphanumeric runs.

    Shared by both BM25 implementations so their term vocabularies line up exactly, which
    matters for any apples-to-apples comparison between them.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Config(BaseModel):
    """Okapi BM25 hyperparameters. Defaults (k1=1.5, b=0.75) match Robertson & Zaragoza and
    are what both `rank_bm25` and `bm25s` ship as their default, so the two arms are
    comparable out of the box.
    """

    k1: float = Field(default=1.5, ge=0, description="Term-frequency saturation rate.")
    b: float = Field(default=0.75, ge=0, le=1, description="Document-length normalization.")
    epsilon: float = Field(
        default=0.25,
        ge=0,
        description=(
            "Floor for IDF of terms appearing in >50% of documents, as a fraction of the "
            "average IDF across the vocabulary. Without this floor, very common terms get "
            "negative IDF and *penalize* documents that contain them, which is not the "
            "intended BM25 behavior (Trotman et al. 2014 discuss this pitfall)."
        ),
    )


@dataclass
class BM25Index:
    """BM25 implemented from scratch.

    score(D, Q) = sum_{t in Q} IDF(t) * f(t,D) * (k1 + 1)
                  ---------------------------------------------
                  f(t,D) + k1 * (1 - b + b * |D| / avgdl)

    IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 ), floored at `epsilon * mean(IDF)` for
    terms whose raw IDF would be negative (i.e. terms present in more than half the corpus).

    - `k1` controls term-frequency saturation: how much a *second*, *third*, ... occurrence
      of a query term keeps adding score. k1=0 means term frequency doesn't matter at all
      beyond presence/absence; larger k1 lets repeated terms keep contributing.
    - `b` controls document-length normalization: b=0 disables it entirely (a term is worth
      the same regardless of document length); b=1 fully normalizes by document length
      relative to the corpus average.
    """

    config: BM25Config = field(default_factory=BM25Config)
    _doc_ids: list[str] = field(default_factory=list, init=False)
    _doc_freqs: list[Counter[str]] = field(default_factory=list, init=False)
    _doc_lens: list[int] = field(default_factory=list, init=False)
    _avgdl: float = field(default=0.0, init=False)
    _idf: dict[str, float] = field(default_factory=dict, init=False)
    _n_docs: int = field(default=0, init=False)

    def index(self, documents: list[Document]) -> None:
        self._doc_ids = [d.doc_id for d in documents]
        token_lists = [tokenize(d.text) for d in documents]
        self._doc_freqs = [Counter(toks) for toks in token_lists]
        self._doc_lens = [len(toks) for toks in token_lists]
        self._n_docs = len(documents)
        self._avgdl = (sum(self._doc_lens) / self._n_docs) if self._n_docs else 0.0
        self._idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        n = self._n_docs
        doc_freq: Counter[str] = Counter()
        for freqs in self._doc_freqs:
            doc_freq.update(freqs.keys())
        raw_idf = {
            term: math.log((n - df + 0.5) / (df + 0.5) + 1.0) for term, df in doc_freq.items()
        }
        if not raw_idf:
            return {}
        positive = [v for v in raw_idf.values() if v > 0]
        mean_idf = sum(positive) / len(positive) if positive else 0.0
        floor = self.config.epsilon * mean_idf
        return {term: (v if v > 0 else floor) for term, v in raw_idf.items()}

    def score(self, query: str, doc_index: int) -> float:
        """Score a single indexed document (by position) against `query`. Exposed separately
        from `search` so it can be hand-verified term by term in tests.
        """
        k1, b = self.config.k1, self.config.b
        freqs = self._doc_freqs[doc_index]
        dl = self._doc_lens[doc_index]
        denom_len_norm = k1 * (1 - b + b * dl / self._avgdl) if self._avgdl else 0.0
        total = 0.0
        for term in tokenize(query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            f = freqs.get(term, 0)
            if f == 0:
                continue
            total += idf * f * (k1 + 1) / (f + denom_len_norm)
        return total

    def search(self, query: str, top_k: int = 10) -> list[ScoredDoc]:
        scored = [
            ScoredDoc(doc_id=self._doc_ids[i], score=self.score(query, i))
            for i in range(self._n_docs)
        ]
        scored.sort(key=lambda sd: sd.score, reverse=True)
        return scored[:top_k]


class BM25SIndex:
    """Production BM25 arm: wraps the `bm25s` library (lazy-imported, part of the `rag`
    extra). Same `RetrievalArm` surface as `BM25Index` so callers (fusion, benchmarks) don't
    care which one they're holding.
    """

    def __init__(self, config: BM25Config | None = None) -> None:
        self.config = config or BM25Config()
        self._retriever = None
        self._doc_ids: list[str] = []

    def index(self, documents: list[Document]) -> None:
        try:
            import bm25s
        except ImportError as e:
            raise ImportError(
                "bm25s is required for BM25SIndex (production path). Install the `rag` "
                "extra: `uv pip install -e '.[rag]'`. For understanding/testing without it, "
                "use BM25Index instead."
            ) from e
        self._doc_ids = [d.doc_id for d in documents]
        corpus_tokens = [tokenize(d.text) for d in documents]
        retriever = bm25s.BM25(k1=self.config.k1, b=self.config.b)
        retriever.index(corpus_tokens)
        self._retriever = retriever

    def search(self, query: str, top_k: int = 10) -> list[ScoredDoc]:
        if self._retriever is None:
            raise RuntimeError("BM25SIndex.search() called before .index()")
        query_tokens = tokenize(query)
        k = min(top_k, len(self._doc_ids)) or 1
        doc_idx, scores = self._retriever.retrieve([query_tokens], k=k)
        results = []
        for idx, score in zip(doc_idx[0], scores[0], strict=True):
            results.append(ScoredDoc(doc_id=self._doc_ids[int(idx)], score=float(score)))
        return results
