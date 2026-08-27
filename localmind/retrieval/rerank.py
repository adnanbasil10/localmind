"""Reranking — refine a coarse candidate set (typically top-50 from fusion) down to a final
top-5 using a model that scores the (query, document) *pair* jointly, which is more accurate
than any single-vector or sparse arm but far too expensive to run over a whole corpus.

Two rerankers:
- `CrossEncoder` — pointwise: scores each (query, doc) pair independently (bge-reranker-v2-m3
  in production, or `-base` if CPU latency is too high; see `bge_reranker_v2_m3`).
- `ListwiseLLMReranker` — the local Qwen3-4B (via Ollama, see `qwen3_listwise_reranker`) is
  shown the whole candidate list at once and asked to reorder it. Listwise can fix relative
  ordering that pointwise scoring gets wrong (e.g. two documents that are individually
  plausible but one clearly answers the query better), at the cost of one much larger prompt.

Both are injected behind Protocols and both are latency-instrumented: `rerank_cross_encoder`
and `rerank_listwise` return a `RerankResult` carrying wall-clock latency alongside the
reordered documents, because reranking is where RAG systems most often blow their latency
budget silently, and that cost is exactly what we want the benchmark table to surface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from localmind.retrieval import ScoredDoc


@runtime_checkable
class CrossEncoder(Protocol):
    """Pointwise reranker: score a single (query, document text) pair. Higher = more relevant."""

    def score(self, query: str, doc_text: str) -> float: ...


@runtime_checkable
class ListwiseReranker(Protocol):
    """Listwise reranker: given a query and the full candidate list, return the indices of
    `doc_texts` in best-first order (a permutation of range(len(doc_texts))).
    """

    def rerank(self, query: str, doc_texts: list[str]) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class RerankResult:
    """Reordered top-k plus the latency it cost to get there — the two numbers the spec
    explicitly wants reported side by side (quality gain vs. added p95 latency).
    """

    ranked: list[ScoredDoc]
    latency_s: float


def rerank_cross_encoder(
    query: str,
    candidates: list[ScoredDoc],
    doc_lookup: dict[str, str],
    reranker: CrossEncoder,
    top_k: int = 5,
) -> RerankResult:
    """Pointwise rerank: score every candidate independently, keep the top_k by that score."""
    start = time.perf_counter()
    scored = [
        ScoredDoc(doc_id=sd.doc_id, score=reranker.score(query, doc_lookup[sd.doc_id]))
        for sd in candidates
        if sd.doc_id in doc_lookup
    ]
    scored.sort(key=lambda sd: sd.score, reverse=True)
    latency = time.perf_counter() - start
    return RerankResult(ranked=scored[:top_k], latency_s=latency)


def rerank_listwise(
    query: str,
    candidates: list[ScoredDoc],
    doc_lookup: dict[str, str],
    reranker: ListwiseReranker,
    top_k: int = 5,
) -> RerankResult:
    """Listwise rerank: hand the whole candidate list to the reranker at once and trust its
    returned order. Candidates missing from `doc_lookup` are dropped before the call.
    """
    present = [sd for sd in candidates if sd.doc_id in doc_lookup]
    doc_texts = [doc_lookup[sd.doc_id] for sd in present]
    start = time.perf_counter()
    order = reranker.rerank(query, doc_texts) if doc_texts else []
    latency = time.perf_counter() - start
    n = len(present)
    ranked = [
        ScoredDoc(doc_id=present[idx].doc_id, score=float(n - rank))
        for rank, idx in enumerate(order)
        if 0 <= idx < n
    ]
    return RerankResult(ranked=ranked[:top_k], latency_s=latency)


@dataclass
class DeterministicFakeCrossEncoder:
    """Test double: token-overlap score between query and doc (Jaccard-ish), deterministic and
    dependency-free. Good enough to meaningfully re-rank a candidate set in tests without
    pulling in a real cross-encoder checkpoint.
    """

    def score(self, query: str, doc_text: str) -> float:
        from localmind.retrieval.bm25 import tokenize

        q_tokens = set(tokenize(query))
        d_tokens = set(tokenize(doc_text))
        if not q_tokens or not d_tokens:
            return 0.0
        overlap = len(q_tokens & d_tokens)
        return overlap / len(q_tokens | d_tokens)


@dataclass
class DeterministicFakeListwiseReranker:
    """Test double: sorts by the same token-overlap heuristic as
    `DeterministicFakeCrossEncoder`, but exercises the listwise call shape (whole list in, a
    permutation out) rather than the pointwise one.
    """

    def rerank(self, query: str, doc_texts: list[str]) -> list[int]:
        from localmind.retrieval.bm25 import tokenize

        q_tokens = set(tokenize(query))
        scored = []
        for i, text in enumerate(doc_texts):
            d_tokens = set(tokenize(text))
            overlap = len(q_tokens & d_tokens) / len(q_tokens | d_tokens) if d_tokens else 0.0
            scored.append((overlap, i))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [i for _, i in scored]


def bge_reranker_v2_m3(model_name: str = "BAAI/bge-reranker-v2-m3") -> CrossEncoder:
    """Real production cross-encoder via `fastembed`'s ONNX `TextCrossEncoder`. Falls back to
    `-base` for CPU latency in the caller's config, not here (this function just loads whatever
    `model_name` is given). Lazy-imported: downloads weights over the network, unavailable
    here. Exercised only by tests marked `@pytest.mark.net`.
    """
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError as e:
        raise ImportError(
            "fastembed is required for bge_reranker_v2_m3. Install the `rag` extra "
            "(and fastembed) and ensure network access to download model weights."
        ) from e

    model = TextCrossEncoder(model_name=model_name)

    class _FastEmbedCrossEncoder:
        def score(self, query: str, doc_text: str) -> float:
            (score,) = list(model.rerank(query, [doc_text]))
            return float(score)

    return _FastEmbedCrossEncoder()


def qwen3_listwise_reranker(
    base_url: str = "http://localhost:11434", model: str = "qwen3:4b"
) -> ListwiseReranker:
    """Real production listwise reranker: prompts the local Qwen3-4B served by Ollama to
    return a reordering of the candidate indices. Lazy-imported (httpx) and requires a
    running Ollama server, unavailable here. Exercised only by tests marked `@pytest.mark.net`.
    """
    try:
        import httpx
    except ImportError as e:
        raise ImportError(
            "httpx is required for qwen3_listwise_reranker. Install the `rag` extra."
        ) from e

    class _OllamaListwiseReranker:
        def rerank(self, query: str, doc_texts: list[str]) -> list[int]:
            numbered = "\n".join(f"[{i}] {text}" for i, text in enumerate(doc_texts))
            prompt = (
                "Rank the following documents by relevance to the query, most relevant "
                "first. Respond with ONLY a comma-separated list of document indices.\n\n"
                f"Query: {query}\n\nDocuments:\n{numbered}\n\nRanking:"
            )
            resp = httpx.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=30.0,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")
            order = []
            for tok in text.replace("[", " ").replace("]", " ").split(","):
                tok = tok.strip()
                if tok.isdigit():
                    idx = int(tok)
                    if 0 <= idx < len(doc_texts) and idx not in order:
                        order.append(idx)
            for i in range(len(doc_texts)):
                if i not in order:
                    order.append(i)
            return order

    return _OllamaListwiseReranker()
