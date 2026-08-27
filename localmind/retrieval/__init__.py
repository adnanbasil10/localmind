"""LocalMind retrieval — the four-arm hybrid search stack.

Arms: BM25 (`bm25.py`), SPLADE (`splade.py`), dense/Matryoshka (`dense.py`), ColBERT
late-interaction (`colbert.py`), and ColQwen2 visual retrieval (`colqwen.py`). Results are
combined in `fusion.py` (RRF baseline + tuned weighted fusion) and refined in `rerank.py`
(cross-encoder + listwise LLM reranker). ANN index engineering lives under `index/`.

This top-level module only defines shared, dependency-free types and a `RetrievalArm`
Protocol. Heavy third-party dependencies (bm25s, scipy, psycopg, pgvector, fastembed,
qdrant-client, torch) are imported lazily *inside* the functions that need them, never at
module import time — so `import localmind.retrieval` works even when the `rag` extra isn't
installed, per CONVENTIONS.md ("No network calls at import time").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

__all__ = [
    "Document",
    "RetrievalArm",
    "RetrievalConfig",
    "ScoredDoc",
    "ranks_from_scores",
    "reciprocal_rank",
]


@dataclass(frozen=True, slots=True)
class Document:
    """A unit of retrievable text. `doc_id` must be unique within a corpus."""

    doc_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredDoc:
    """A document with a retrieval score. Higher score always means more relevant —

    each arm is responsible for producing scores on that convention even when its native
    metric is a distance (e.g. dense/ColBERT arms report similarity, not distance).
    """

    doc_id: str
    score: float


@runtime_checkable
class RetrievalArm(Protocol):
    """One arm of the hybrid retrieval stack. BM25Index, SpladeIndex, DenseIndex, and
    ColBERTIndex all satisfy this structurally (duck typing, no inheritance required).
    """

    def index(self, documents: list[Document]) -> None:
        """Build the arm's index from scratch over `documents`."""
        ...

    def search(self, query: str, top_k: int = 10) -> list[ScoredDoc]:
        """Return up to `top_k` documents sorted best-first (descending score)."""
        ...


def ranks_from_scores(scored: list[ScoredDoc]) -> dict[str, int]:
    """1-indexed rank per doc_id, assuming `scored` is already sorted best-first.

    Used by RRF and other rank-based fusion, which deliberately ignore the score magnitude.
    """
    return {sd.doc_id: i + 1 for i, sd in enumerate(scored)}


def reciprocal_rank(scored: list[ScoredDoc], relevant_ids: set[str]) -> float:
    """1/rank of the first relevant doc in `scored`, or 0.0 if none is present. Used for MRR."""
    for i, sd in enumerate(scored):
        if sd.doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


class RetrievalConfig(BaseModel):
    """Top-level knobs for the retrieval stack, loaded from `configs/retrieval/*.yaml`.

    Deliberately small: each arm already carries its own hyperparameter model with its own
    defaults (`bm25.BM25Config`, `index.pgvector.HNSWConfig`, ...), which stay the source of
    truth. This model exists so a single YAML file can override the handful of values worth
    tuning from one place, mirroring `ModelConfig.from_yaml`'s pattern (CONVENTIONS.md).
    """

    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    dense_stage1_dim: int = 256
    fusion_rrf_k: int = 60
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40
    rerank_candidates: int = 50
    rerank_top_k: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path) -> RetrievalConfig:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
