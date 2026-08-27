"""ColBERT — late interaction, token-level max-sim.

Instead of collapsing a document to one vector (as `dense.py` does), ColBERT keeps one vector
per token and scores a query against a document with MaxSim:

    score(Q, D) = sum_{q in Q} max_{d in D} cos(q, d)

Every query token gets to independently pick its single best-matching document token, then
those per-token maxima are summed. This gives ColBERT an edge on long or multi-aspect queries,
where a single pooled vector would average away a minority aspect that a token-level match can
still catch.

No real ColBERT checkpoint is downloadable here, so the encoder is injected behind the
`MultiVectorEncoder` Protocol, same pattern as `dense.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from localmind.retrieval import Document, ScoredDoc
from localmind.retrieval.bm25 import tokenize
from localmind.retrieval.dense import l2_normalize, stable_str_hash


@runtime_checkable
class MultiVectorEncoder(Protocol):
    """Text -> one L2-normalized vector per token, shape (n_tokens, dim)."""

    @property
    def dim(self) -> int: ...

    def encode(self, text: str) -> np.ndarray: ...


def maxsim(query_vectors: np.ndarray, doc_vectors: np.ndarray) -> float:
    """ColBERT's MaxSim: sum over query tokens of the max cosine similarity to any doc token.

    Assumes both inputs are already L2-normalized so a dot product is a cosine similarity.
    """
    if query_vectors.shape[0] == 0 or doc_vectors.shape[0] == 0:
        return 0.0
    sims = query_vectors @ doc_vectors.T  # (n_query_tokens, n_doc_tokens)
    return float(sims.max(axis=1).sum())


@dataclass
class DeterministicFakeMultiVectorEncoder:
    """Test/demo double for `MultiVectorEncoder`. Deterministic, seeded, no model weights.

    Each token gets a fixed pseudo-random unit vector (same construction as
    `dense.DeterministicFakeEmbedder`'s per-token vectors, so the two arms agree on which
    tokens are "similar" in tests) — but unlike the dense embedder, tokens are kept
    *separate* rather than pooled, which is the entire point of late interaction.
    """

    dim_: int = 128
    seed: int = 0

    @property
    def dim(self) -> int:
        return self.dim_

    def _token_vector(self, token: str) -> np.ndarray:
        token_seed = (self.seed * 1_000_003 + stable_str_hash(token)) % (2**32)
        rng = np.random.default_rng(token_seed)
        return rng.standard_normal(self.dim_).astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        tokens = tokenize(text) or [""]
        vecs = np.stack([self._token_vector(t) for t in tokens], axis=0)
        return l2_normalize(vecs)


class ColBERTIndex:
    """Late-interaction arm. Brute-force MaxSim over all documents — fine at benchmark scale;
    a production deployment would use PLAID/centroid pruning, out of scope here.
    """

    def __init__(self, encoder: MultiVectorEncoder) -> None:
        self.encoder = encoder
        self._doc_ids: list[str] = []
        self._doc_vectors: list[np.ndarray] = []

    def index(self, documents: list[Document]) -> None:
        self._doc_ids = [d.doc_id for d in documents]
        self._doc_vectors = [self.encoder.encode(d.text) for d in documents]

    def search(self, query: str, top_k: int = 10) -> list[ScoredDoc]:
        query_vectors = self.encoder.encode(query)
        scored = [
            ScoredDoc(doc_id=doc_id, score=maxsim(query_vectors, doc_vecs))
            for doc_id, doc_vecs in zip(self._doc_ids, self._doc_vectors, strict=True)
        ]
        scored.sort(key=lambda sd: sd.score, reverse=True)
        return scored[:top_k]
