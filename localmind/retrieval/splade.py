"""SPLADE — learned sparse retrieval via term expansion into an inverted index.

Unlike BM25 (exact-term matching only), a SPLADE encoder maps text to a sparse weight vector
over the *model's vocabulary* — including terms that never appear verbatim in the text
("vocabulary mismatch" cases like "car" <-> "automobile"), while still living in a sparse
inverted index, so it keeps BM25-style index speed instead of paying for dense ANN search.

We don't have network access to download a real SPLADE checkpoint here, so the encoder is
injected behind the `SparseEncoder` Protocol. `SpladeIndex` only ever talks to that Protocol,
so swapping in a real ONNX SPLADE model (via fastembed, see `fastembed_splade_encoder`) is a
one-line change with no index-logic changes. Tests use `DeterministicExpansionEncoder`, a
small fixed-synonym-table stand-in that is fully deterministic and exercises the same
"query term not in doc, but a related term is" path that makes SPLADE useful.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from localmind.retrieval import Document, ScoredDoc
from localmind.retrieval.bm25 import tokenize


@runtime_checkable
class SparseEncoder(Protocol):
    """Maps text to a sparse term -> weight vector (non-negative weights)."""

    def encode(self, text: str) -> dict[str, float]: ...


@dataclass
class DeterministicExpansionEncoder:
    """Test/demo double for `SparseEncoder`.

    Weights = raw term counts of the text's own tokens, plus a fixed weight for each term in
    a small synonym/expansion table when one of its trigger terms is present. This is a toy,
    but it is deterministic and exercises exactly the mechanism SPLADE is for: a query can
    match a document via an expansion term that never appears in the document's surface text.
    """

    expansions: dict[str, list[str]] = field(default_factory=dict)
    expansion_weight: float = 0.5

    def encode(self, text: str) -> dict[str, float]:
        tokens = tokenize(text)
        weights: dict[str, float] = dict(Counter(tokens))
        weights = {k: float(v) for k, v in weights.items()}
        for token in tokens:
            for expanded_term in self.expansions.get(token, []):
                weights[expanded_term] = max(weights.get(expanded_term, 0.0), self.expansion_weight)
        return weights


class SpladeIndex:
    """Inverted-index SPLADE arm: term -> [(doc_id, weight), ...], scored by sparse dot
    product between the query's expansion vector and each candidate document's.
    """

    def __init__(self, encoder: SparseEncoder) -> None:
        self.encoder = encoder
        self._postings: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self._doc_ids: list[str] = []

    def index(self, documents: list[Document]) -> None:
        self._postings = defaultdict(list)
        self._doc_ids = [d.doc_id for d in documents]
        for doc in documents:
            for term, weight in self.encoder.encode(doc.text).items():
                if weight > 0:
                    self._postings[term].append((doc.doc_id, weight))

    def search(self, query: str, top_k: int = 10) -> list[ScoredDoc]:
        query_weights = self.encoder.encode(query)
        scores: dict[str, float] = defaultdict(float)
        for term, q_weight in query_weights.items():
            if q_weight <= 0:
                continue
            for doc_id, d_weight in self._postings.get(term, []):
                scores[doc_id] += q_weight * d_weight
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [ScoredDoc(doc_id=doc_id, score=score) for doc_id, score in ranked[:top_k]]


def fastembed_splade_encoder(model_name: str = "prithivida/Splade_PP_en_v1") -> SparseEncoder:
    """Real production encoder: ONNX SPLADE via `fastembed`. Lazy-imported because `fastembed`
    is a heavy optional dependency and, more importantly, constructing it downloads model
    weights over the network — unavailable in this environment. Exercised only by tests
    marked `@pytest.mark.net`.
    """
    try:
        from fastembed import SparseTextEmbedding
    except ImportError as e:
        raise ImportError(
            "fastembed is required for fastembed_splade_encoder. Install the `rag` extra "
            "(and fastembed) and ensure network access to download model weights."
        ) from e

    model = SparseTextEmbedding(model_name=model_name)

    class _FastEmbedSparseEncoder:
        def encode(self, text: str) -> dict[str, float]:
            (embedding,) = model.embed([text])
            # SPLADE's "terms" are sub-word token ids from the model's own vocabulary, not
            # our tokenizer's terms. Stringify so they satisfy `SparseEncoder`'s str keys and
            # so this arm's postings never collide with BM25's plain-word postings.
            return {
                f"tok{idx}": float(weight)
                for idx, weight in zip(
                    embedding.indices.tolist(), embedding.values.tolist(), strict=True
                )
            }

    return _FastEmbedSparseEncoder()
