"""Dense retrieval — bge-m3 / gte-modernbert style embeddings with Matryoshka truncation.

Matryoshka Representation Learning (MRL) trains embeddings so that any length-`d` *prefix*
of the full vector is itself a valid, independently-usable embedding, just lower fidelity.
That lets us run a cheap two-stage search: stage 1 scores every document with a 256-dim
truncated prefix (small enough to keep an ANN index compressed and comparisons cheap), then
stage 2 re-scores only the stage-1 candidates with the full-dimension vector for the final
ranking. `TwoStageDenseIndex` implements exactly that.

No real embedding model is downloadable in this environment (no network budget), so the
embedder is injected behind the `Embedder` Protocol — production code plugs in
`fastembed_onnx_embedder` (bge-m3/gte-modernbert, ONNX int8, lazy-imported); tests plug in
`DeterministicFakeEmbedder`, a seeded, dependency-free stand-in with the same shape contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from localmind.retrieval import Document, ScoredDoc


@runtime_checkable
class Embedder(Protocol):
    """Text -> dense vector(s). Implementations must return L2-normalized rows so that a
    plain dot product equals cosine similarity everywhere downstream.
    """

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), self.dim) float32 array of L2-normalized embeddings."""
        ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def matryoshka_truncate(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Truncate MRL embeddings to their first `dim` dimensions and re-normalize.

    Re-normalizing after truncation is required: MRL is trained so the *direction* of the
    prefix is meaningful, but truncating drops norm, and cosine similarity on un-renormalized
    prefixes silently underweights every truncated vector relative to full ones.
    """
    if dim > vectors.shape[-1]:
        raise ValueError(f"Cannot truncate to {dim} dims; vectors only have {vectors.shape[-1]}")
    return l2_normalize(vectors[..., :dim])


@dataclass
class DeterministicFakeEmbedder:
    """Test double for `Embedder`. No model weights, no network, no torch.

    Deterministic given `seed`: each vocabulary token gets a fixed pseudo-random unit vector
    (seeded per-token via a stable hash), and a text's embedding is the normalized sum of its
    tokens' vectors — a bag-of-tokens embedding, which is enough to give semantically similar
    fake texts (sharing tokens) higher cosine similarity than dissimilar ones, so retrieval
    logic built on top of it is exercised meaningfully in tests.
    """

    dim_: int = 1024
    seed: int = 0

    @property
    def dim(self) -> int:
        return self.dim_

    def _token_vector(self, token: str) -> np.ndarray:
        # A stable per-token seed derived from `seed` and the token text (not Python's salted
        # `hash()`, which varies per-process and would break bit-exact reproducibility).
        token_seed = (self.seed * 1_000_003 + stable_str_hash(token)) % (2**32)
        rng = np.random.default_rng(token_seed)
        return rng.standard_normal(self.dim_).astype(np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        from localmind.retrieval.bm25 import tokenize

        out = np.zeros((len(texts), self.dim_), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = tokenize(text) or [""]
            vec = np.sum([self._token_vector(t) for t in tokens], axis=0)
            out[i] = vec
        return l2_normalize(out)


def stable_str_hash(s: str) -> int:
    """Deterministic string hash (FNV-1a), stable across processes and Python versions,
    unlike the builtin `hash()` which is salted per-process for security. Shared by every
    deterministic fake encoder in this package (dense, ColBERT) so that toy "semantic
    similarity" agrees across arms in tests.
    """
    h = 0x811C9DC5
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class TwoStageDenseIndex:
    """Dense arm with Matryoshka two-stage retrieval.

    Stage 1: cosine similarity on `stage1_dim`-truncated vectors over the *whole* corpus
    (cheap: small vectors, and this is the width you'd actually keep in a compressed ANN
    index). Stage 2: re-score only the stage-1 top `stage1_k` candidates using full-dimension
    vectors, then return the top_k of that re-ranking. `search_stage1_only` and `search` (the
    two-stage path) are both exposed so the benchmark can report "Dense (256d MRL)" and
    "Dense (full)" as separate rows.
    """

    def __init__(self, embedder: Embedder, stage1_dim: int = 256) -> None:
        self.embedder = embedder
        self.stage1_dim = stage1_dim
        self._doc_ids: list[str] = []
        self._full_vectors: np.ndarray = np.zeros((0, embedder.dim), dtype=np.float32)
        self._stage1_vectors: np.ndarray = np.zeros((0, stage1_dim), dtype=np.float32)

    def index(self, documents: list[Document]) -> None:
        self._doc_ids = [d.doc_id for d in documents]
        self._full_vectors = self.embedder.embed([d.text for d in documents])
        self._stage1_vectors = matryoshka_truncate(self._full_vectors, self.stage1_dim)

    def _rank(self, vectors: np.ndarray, query_vec: np.ndarray, top_k: int) -> list[ScoredDoc]:
        sims = vectors @ query_vec
        k = min(top_k, len(self._doc_ids))
        if k <= 0:
            return []
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [ScoredDoc(doc_id=self._doc_ids[i], score=float(sims[i])) for i in top_idx]

    def search_stage1_only(self, query: str, top_k: int = 10) -> list[ScoredDoc]:
        """ "Dense (256d MRL)" — stage 1 alone, the cheap-but-lossy path."""
        (query_full,) = self.embedder.embed([query])
        query_stage1 = matryoshka_truncate(query_full[None, :], self.stage1_dim)[0]
        return self._rank(self._stage1_vectors, query_stage1, top_k)

    def search_full_only(self, query: str, top_k: int = 10) -> list[ScoredDoc]:
        """ "Dense (full)" — brute-force full-dimension search, no truncation anywhere."""
        (query_full,) = self.embedder.embed([query])
        return self._rank(self._full_vectors, query_full, top_k)

    def search(self, query: str, top_k: int = 10, stage1_k: int = 100) -> list[ScoredDoc]:
        """Two-stage search: stage-1 truncated recall pass, stage-2 full-precision re-score."""
        (query_full,) = self.embedder.embed([query])
        query_stage1 = matryoshka_truncate(query_full[None, :], self.stage1_dim)[0]
        stage1_hits = self._rank(self._stage1_vectors, query_stage1, stage1_k)
        candidate_idx = [self._doc_ids.index(sd.doc_id) for sd in stage1_hits]
        if not candidate_idx:
            return []
        candidate_vectors = self._full_vectors[candidate_idx]
        sims = candidate_vectors @ query_full
        order = np.argsort(-sims)[:top_k]
        return [
            ScoredDoc(doc_id=self._doc_ids[candidate_idx[i]], score=float(sims[i])) for i in order
        ]


def fastembed_onnx_embedder(model_name: str = "BAAI/bge-m3", dim: int = 1024) -> Embedder:
    """Real production embedder: bge-m3 (or gte-modernbert) via `fastembed`'s ONNX int8
    runtime. Lazy-imported: `fastembed` is a heavy optional dependency, and constructing this
    downloads model weights over the network — unavailable in this environment. Exercised
    only by tests marked `@pytest.mark.net`.
    """
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise ImportError(
            "fastembed is required for fastembed_onnx_embedder. Install the `rag` extra "
            "(and fastembed) and ensure network access to download model weights."
        ) from e

    model = TextEmbedding(model_name=model_name)
    embedder_dim = dim

    class _FastEmbedEmbedder:
        def __init__(self) -> None:
            self.dim_ = embedder_dim

        @property
        def dim(self) -> int:
            return self.dim_

        def embed(self, texts: list[str]) -> np.ndarray:
            vectors = np.array(list(model.embed(texts)), dtype=np.float32)
            return l2_normalize(vectors)

    return _FastEmbedEmbedder()
