"""Chunking (implementation.md §11) -- the measurable core of Phase 7.

Five strategies, all producing the same `Chunk` type:

    fixed        -- baseline: fixed-size windows over whitespace-delimited words
    recursive    -- structure-aware: snap window boundaries to paragraph/sentence
                    breaks near the target size
    semantic     -- split where consecutive sentences' embeddings diverge
                    ("similarity drops")
    late         -- embed the whole document once, pool per chunk, so each
                    chunk vector carries document-level context
    contextual   -- prepend an LLM-generated 1-2 sentence situating context
                    (see `contextualize.py`) before the text that gets embedded

Chunk-size units are whitespace-delimited **words**, not tokenizer tokens --
this module deliberately does not depend on `localmind.tokenizer` (a separate
owned package) to stay dependency-free; swap `_word_spans` for a real
tokenizer's offsets to get exact token counts in production.

``chunk_id`` follows the ``"<doc_id>#<n>"`` convention `localmind.eval.datasets.
schema.CorpusChunk` / `doc_of_chunk` already enforce ("chunk ids are doc-scoped
by contract" -- that schema's words, not this module's invention): this is what
makes chunks produced here directly loadable into the eval harness's corpus
format without translation.

**The ablation harness** (`run_strategy_ablation`, `run_size_overlap_ablation`)
measures nDCG@10 on a **deterministic, code-generated synthetic corpus** --
there is no real embedder, no real corpus, and no network in this environment.
Ground truth relevance is defined by construction (a chunk is "relevant" iff it
contains enough of a known fact sentence's key terms) and similarity is scored
by `FakeHashEmbedder`, a hashed-bag-of-words embedder that approximates lexical
overlap, not semantics. Every number this harness produces is labelled
"SYNTHETIC HARNESS" and must never be read as a real retrieval-quality result --
see `HeatmapResult`/`run_strategy_ablation`'s docstrings.

Uses `localmind.eval.stats` (the shared, cross-phase statistics backbone --
"every phase's benchmark imports this module", per its own docstring) so every
ablation number carries a bootstrap/t-interval CI, per CONVENTIONS.md rule 5
("never report a bare number"). No torch, no `localmind.model`, no
`localmind.retrieval` -- `import localmind.ingestion.chunking` needs only numpy
+ pydantic + `localmind.eval.stats` (itself numpy-only).
"""

from __future__ import annotations

import bisect
import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from random import Random
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from localmind.eval.stats import Estimate, MetricRow, aggregate_seeds, rows_to_markdown
from localmind.ingestion.contextualize import Contextualizer, HeuristicContextualizer

__all__ = [
    "CHUNK_SIZES",
    "OVERLAP_FRACS",
    "STRATEGY_NAMES",
    "Chunk",
    "ChunkStrategyName",
    "Embedder",
    "FakeHashEmbedder",
    "HeatmapResult",
    "chunk_contextual",
    "chunk_document",
    "chunk_fixed",
    "chunk_late",
    "chunk_recursive",
    "chunk_semantic",
    "chunk_stats",
    "ndcg_at_k",
    "rows_to_markdown",
    "run_size_overlap_ablation",
    "run_strategy_ablation",
]

ChunkStrategyName = Literal["fixed", "recursive", "semantic", "late", "contextual"]
STRATEGY_NAMES: tuple[ChunkStrategyName, ...] = (
    "fixed",
    "recursive",
    "semantic",
    "late",
    "contextual",
)

CHUNK_SIZES: tuple[int, ...] = (256, 512, 768, 1024)
OVERLAP_FRACS: tuple[float, ...] = (0.0, 0.10, 0.20)


# --------------------------------------------------------------------------------------
# Chunk
# --------------------------------------------------------------------------------------


class Chunk(BaseModel):
    """One retrievable unit. `chunk_id` is `"<doc_id>#<index>"` -- the
    doc-scoped convention `localmind.eval.datasets.schema.CorpusChunk` expects,
    so chunks produced here plug straight into the eval harness's corpus
    format."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    index: int
    strategy: str
    size_tokens: int
    overlap_tokens: int = 0
    embedding: list[float] | None = None
    context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """What actually gets embedded: situating context (if any) + text."""
        return f"{self.context}\n\n{self.text}" if self.context else self.text


def chunk_stats(chunks: Sequence[Chunk]) -> dict[str, float]:
    """Real, measured statistics over a batch of chunks -- size distribution
    and overlap, in words. Used both by tests and by the ingestion report."""
    if not chunks:
        return {
            "n_chunks": 0,
            "mean_size": 0.0,
            "min_size": 0.0,
            "max_size": 0.0,
            "mean_overlap": 0.0,
        }
    sizes = [c.size_tokens for c in chunks]
    overlaps = [c.overlap_tokens for c in chunks]
    return {
        "n_chunks": float(len(chunks)),
        "mean_size": float(np.mean(sizes)),
        "min_size": float(np.min(sizes)),
        "max_size": float(np.max(sizes)),
        "mean_overlap": float(np.mean(overlaps)),
    }


# --------------------------------------------------------------------------------------
# Shared word/sentence span helpers
# --------------------------------------------------------------------------------------

_WORD_RE = re.compile(r"\S+")
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


def _word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_BOUNDARY_RE.finditer(text):
        end = m.start()
        if end > start and text[start:end].strip():
            spans.append((start, end))
        start = m.end()
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _structural_boundary_word_indices(text: str, spans: Sequence[tuple[int, int]]) -> set[int]:
    """Word indices `i` such that a paragraph/sentence boundary falls between
    word `i-1` and word `i` -- what `chunk_recursive` snaps window edges to."""
    starts = [s for s, _ in spans]
    boundaries = {0, len(spans)}
    for m in _SENT_BOUNDARY_RE.finditer(text):
        boundaries.add(bisect.bisect_left(starts, m.end()))
    return boundaries


# --------------------------------------------------------------------------------------
# Strategy 1: fixed-size windows (baseline)
# --------------------------------------------------------------------------------------


def chunk_fixed(text: str, doc_id: str, *, size: int = 512, overlap: int = 0) -> list[Chunk]:
    """Baseline: hard windows of `size` words, stepping by `size - overlap`."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size), got overlap={overlap} size={size}")
    spans = _word_spans(text)
    n = len(spans)
    if n == 0:
        return []
    stride = size - overlap
    chunks: list[Chunk] = []
    i = 0
    idx = 0
    while i < n:
        window = spans[i : i + size]
        if not window:
            break
        start, end = window[0][0], window[-1][1]
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}#{idx}",
                doc_id=doc_id,
                text=text[start:end],
                start=start,
                end=end,
                index=idx,
                strategy="fixed",
                size_tokens=len(window),
                overlap_tokens=min(overlap, i),
            )
        )
        idx += 1
        if i + size >= n:
            break
        i += stride
    return chunks


# --------------------------------------------------------------------------------------
# Strategy 2: recursive-by-separator (structure-aware)
# --------------------------------------------------------------------------------------


def chunk_recursive(
    text: str, doc_id: str, *, size: int = 512, overlap: int = 0, min_size_frac: float = 0.5
) -> list[Chunk]:
    """Structure-aware: like `chunk_fixed`, but a window's right edge snaps to
    the nearest paragraph/sentence boundary within
    `[i + size*min_size_frac, i + size]` when one exists, instead of always
    hard-cutting mid-sentence."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size), got overlap={overlap} size={size}")
    spans = _word_spans(text)
    n = len(spans)
    if n == 0:
        return []
    boundaries = sorted(_structural_boundary_word_indices(text, spans))
    min_end = max(1, int(size * min_size_frac))
    chunks: list[Chunk] = []
    i = 0
    idx = 0
    while i < n:
        hard_end = min(i + size, n)
        candidates = [b for b in boundaries if i + min_end <= b <= hard_end]
        end = max(candidates) if candidates else hard_end
        if end <= i:
            end = hard_end
        window = spans[i:end]
        start_c, end_c = window[0][0], window[-1][1]
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}#{idx}",
                doc_id=doc_id,
                text=text[start_c:end_c],
                start=start_c,
                end=end_c,
                index=idx,
                strategy="recursive",
                size_tokens=len(window),
                overlap_tokens=min(overlap, i),
            )
        )
        idx += 1
        if end >= n:
            break
        next_i = end - overlap
        i = next_i if next_i > i else i + 1  # guarantee progress
    return chunks


# --------------------------------------------------------------------------------------
# Embedder seam + deterministic offline fake
# --------------------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Seam onto a real embedding model. `embed` returns an `(n, dim)` array,
    one row per input text. Production implementations live outside this
    package (e.g. `localmind.retrieval`, not imported here); tests and the
    ablation harness use `FakeHashEmbedder`."""

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class FakeHashEmbedder:
    """Deterministic, model-free embedder: hashed bag-of-words (unigrams +
    bigrams) via `blake2b`, L2-normalized. **NOT a real semantic embedder** --
    cosine similarity under this embedder approximates *lexical* overlap. That
    is enough to make `chunk_semantic`/`chunk_late` and the ablation harness
    respond meaningfully to chunk boundaries and situating context, entirely
    offline, with no model weights, GPU, or network. Swap for a real embedder
    to get real retrieval-quality numbers."""

    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed  # reserved for future stochastic variants; hashing itself needs none

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", text.lower())
            grams = [*words, *(f"{a}_{b}" for a, b in pairwise(words))]
            for g in grams:
                idx = (
                    int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "big")
                    % self.dim
                )
                out[row, idx] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------------------
# Strategy 3: semantic (split on embedding-similarity drops)
# --------------------------------------------------------------------------------------


def chunk_semantic(
    text: str, doc_id: str, embedder: Embedder | None = None, *, max_size: int = 512, z: float = 1.0
) -> list[Chunk]:
    """Split sentences into groups wherever consecutive-sentence dissimilarity
    (`1 - cosine`) exceeds `mean + z * std` over the document -- a "similarity
    drop". Oversized groups are further split on sentence boundaries so no
    chunk exceeds `max_size` words. No `overlap` parameter: boundaries here
    are drop-determined, not stride-determined."""
    embedder = embedder or FakeHashEmbedder()
    spans = _sentence_spans(text)
    if len(spans) == 0:
        return []
    if len(spans) == 1:
        start, end = spans[0]
        return [
            Chunk(
                chunk_id=f"{doc_id}#0",
                doc_id=doc_id,
                text=text[start:end],
                start=start,
                end=end,
                index=0,
                strategy="semantic",
                size_tokens=len(text[start:end].split()),
            )
        ]
    sentences = [text[s:e] for s, e in spans]
    vecs = embedder.embed(sentences)
    sims = np.array([_cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)])
    dissim = 1.0 - sims
    threshold = float(dissim.mean() + z * dissim.std()) if dissim.size else 1.0
    breakpoints = {i + 1 for i, d in enumerate(dissim) if d > threshold}

    groups: list[list[int]] = []
    current = [0]
    for i in range(1, len(spans)):
        if i in breakpoints:
            groups.append(current)
            current = [i]
        else:
            current.append(i)
    groups.append(current)

    final_groups: list[list[int]] = []
    for g in groups:
        cur: list[int] = []
        cur_words = 0
        for i in g:
            w = len(sentences[i].split())
            if cur and cur_words + w > max_size:
                final_groups.append(cur)
                cur, cur_words = [], 0
            cur.append(i)
            cur_words += w
        if cur:
            final_groups.append(cur)

    chunks: list[Chunk] = []
    for idx, g in enumerate(final_groups):
        start, end = spans[g[0]][0], spans[g[-1]][1]
        t = text[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}#{idx}",
                doc_id=doc_id,
                text=t,
                start=start,
                end=end,
                index=idx,
                strategy="semantic",
                size_tokens=len(t.split()),
            )
        )
    return chunks


# --------------------------------------------------------------------------------------
# Strategy 4: late chunking (whole-doc embedding, pooled per chunk)
# --------------------------------------------------------------------------------------


def chunk_late(
    text: str,
    doc_id: str,
    embedder: Embedder | None = None,
    *,
    size: int = 512,
    overlap: int = 0,
    context_weight: float = 0.3,
) -> list[Chunk]:
    """Late chunking: embed the *whole document* once (`doc_vec`), and blend
    it into each chunk's own local embedding --
    `pooled = context_weight * doc_vec + (1 - context_weight) * local_vec`,
    renormalized. This is a deliberately simplified stand-in for "embed with
    full-document attention, then pool per chunk": with a real transformer
    embedder, document-level context leaks into every token's representation
    before pooling; here, with a linear bag-of-words embedder, we make that
    leakage explicit via the blend so the mechanism (chunk vectors carrying
    document context) is still exercised and testable. Reuses `chunk_fixed`
    for the underlying windows, so boundaries are identical to the `fixed`
    strategy -- only the resulting *embedding* differs."""
    embedder = embedder or FakeHashEmbedder()
    base = chunk_fixed(text, doc_id, size=size, overlap=overlap)
    if not base:
        return []
    doc_vec = embedder.embed([text])[0]
    local_vecs = embedder.embed([c.text for c in base])
    pooled = context_weight * doc_vec[None, :] + (1.0 - context_weight) * local_vecs
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pooled = pooled / norms
    return [
        c.model_copy(update={"strategy": "late_chunking", "embedding": pooled[i].tolist()})
        for i, c in enumerate(base)
    ]


# --------------------------------------------------------------------------------------
# Strategy 5: contextual retrieval
# --------------------------------------------------------------------------------------


def chunk_contextual(
    text: str,
    doc_id: str,
    contextualizer: Contextualizer | None = None,
    *,
    size: int = 512,
    overlap: int = 0,
) -> list[Chunk]:
    """Contextual retrieval: chunk with `chunk_recursive`, then prepend a
    situating context (see `contextualize.py`) to each chunk. `contextualizer`
    defaults to `HeuristicContextualizer` -- the always-on offline fallback."""
    ctx = contextualizer or HeuristicContextualizer()
    base = chunk_recursive(text, doc_id, size=size, overlap=overlap)
    out: list[Chunk] = []
    for c in base:
        situating = ctx.situate(text, c.text)
        out.append(c.model_copy(update={"strategy": "contextual_retrieval", "context": situating}))
    return out


# --------------------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------------------


def chunk_document(
    text: str,
    doc_id: str,
    strategy: ChunkStrategyName,
    *,
    size: int = 512,
    overlap: int = 0,
    embedder: Embedder | None = None,
    contextualizer: Contextualizer | None = None,
) -> list[Chunk]:
    """Single entry point over all five strategies -- what `pipeline.py` and
    the ablation harness call."""
    if strategy == "fixed":
        return chunk_fixed(text, doc_id, size=size, overlap=overlap)
    if strategy == "recursive":
        return chunk_recursive(text, doc_id, size=size, overlap=overlap)
    if strategy == "semantic":
        return chunk_semantic(text, doc_id, embedder, max_size=size)
    if strategy == "late":
        return chunk_late(text, doc_id, embedder, size=size, overlap=overlap)
    if strategy == "contextual":
        return chunk_contextual(text, doc_id, contextualizer, size=size, overlap=overlap)
    raise ValueError(f"unknown chunking strategy {strategy!r}; must be one of {STRATEGY_NAMES}")


# --------------------------------------------------------------------------------------
# nDCG
# --------------------------------------------------------------------------------------


def ndcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
    """Standard nDCG@k over a ranked relevance sequence. Returns 0.0 if the
    ideal DCG is 0 (no relevant items at all)."""
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float(np.sum(rel * discounts))
    idcg = float(np.sum(np.sort(rel)[::-1] * discounts))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------------------
# SYNTHETIC ablation harness
# --------------------------------------------------------------------------------------
#
# No real embedder, no real corpus, no network. Everything below is a
# deterministic, code-generated stand-in whose only purpose is to exercise the
# five chunking mechanisms and the size x overlap grid end to end. Numbers
# produced here are labelled "SYNTHETIC HARNESS" throughout and must not be
# read as real retrieval-quality results (see module docstring).

_FACTS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "refund_policy",
        "Refunds are processed within fourteen business days after the returned item "
        "passes warehouse inspection.",
        ("refund", "fourteen", "business", "days", "warehouse", "inspection"),
        "How many business days does a refund take after warehouse inspection?",
    ),
    (
        "vpn_setup",
        "The engineering VPN requires a hardware security key paired with the Okta "
        "authenticator app before the first login.",
        ("vpn", "hardware", "security", "key", "okta", "authenticator"),
        "What do I need besides a password to log into the engineering VPN for the first time?",
    ),
    (
        "battery_spec",
        "The X200 battery pack delivers eleven thousand milliamp hours at a nominal "
        "voltage of seven point four volts.",
        ("x200", "battery", "eleven", "thousand", "milliamp", "voltage"),
        "What is the capacity and voltage of the X200 battery pack?",
    ),
    (
        "onboarding",
        "New hires complete security awareness training within their first five working "
        "days and receive a laptop preloaded with the standard image.",
        ("hires", "security", "awareness", "training", "five", "laptop"),
        "How soon must new hires finish security training and what do they receive?",
    ),
    (
        "incident_sla",
        "Priority one incidents must receive an acknowledgement from on call engineering "
        "within fifteen minutes of being paged.",
        ("priority", "incidents", "acknowledgement", "call", "fifteen", "minutes", "paged"),
        "What is the acknowledgement time requirement for priority one incidents?",
    ),
    (
        "warranty",
        "The extended warranty covers accidental drops and liquid spills for twenty four "
        "months from the original purchase date.",
        ("extended", "warranty", "accidental", "drops", "liquid", "spills", "months"),
        "Does the extended warranty cover liquid damage, and for how long?",
    ),
    (
        "expense_report",
        "Travel expense reports must be submitted within thirty days of trip completion "
        "and require a manager digital signature above five hundred dollars.",
        ("travel", "expense", "reports", "thirty", "days", "manager", "signature"),
        "How long after a trip do I have to submit an expense report, and when do I need manager approval?",
    ),
    (
        "release_process",
        "Release candidates are frozen on Tuesdays and promoted to production only after "
        "passing the full regression suite on staging.",
        ("release", "candidates", "frozen", "tuesdays", "regression", "suite", "staging"),
        "On what day are release candidates frozen and what must they pass before production?",
    ),
)

_FILLER_SENTENCES: tuple[str, ...] = (
    "The cafeteria on the third floor serves lunch between eleven thirty and two "
    "o'clock on weekdays.",
    "Conference room bookings can be made through the shared calendar up to sixty days in advance.",
    "The design team reviews new mockups every other Thursday during the afternoon standup.",
    "Parking permits are renewed annually each spring for employees who commute to "
    "the main campus.",
    "The office plants are watered every Monday and Friday by the facilities contractor.",
    "Building badges deactivate automatically thirty minutes after the last "
    "recorded entry each night.",
    "The quarterly all-hands is streamed to every regional office with closed captions enabled.",
    "Recycling bins are collected from each floor twice a week by the cleaning crew.",
    "The library nook on the second floor has six desks that can be reserved for quiet work.",
    "Guest wifi credentials are printed on a card at the front desk and rotate every month.",
    "The company newsletter goes out on the first business day of every month.",
    "Bicycle racks near the east entrance were repainted last summer by a local vendor.",
    "The mentorship program pairs new employees with a volunteer from a different team.",
    "Printer paper and toner are restocked from the supply closet on the fourth floor.",
    "The rooftop garden hosts an informal gathering on the last Friday of each quarter.",
    "Desk plants are optional but several teams have adopted a shared succulent collection.",
    "The shuttle bus runs every twenty minutes between the two campus buildings.",
    "Employee surveys are sent twice a year and results are shared at the following town hall.",
    "The gym on the ground floor requires a signed waiver before first use.",
    "Lost and found items are held at the front desk for ninety days before donation.",
    "The book club meets monthly in the second floor lounge over coffee.",
    "Meeting rooms automatically release their booking if no one checks in within ten minutes.",
    "The volunteer day happens once a quarter and is coordinated by the community team.",
    "Desk assignments rotate between the two towers whenever a floor is renovated.",
)


@dataclass(frozen=True)
class _SyntheticDoc:
    doc_id: str
    text: str
    query: str
    key_terms: tuple[str, ...]


def _build_synthetic_corpus(seed: int, *, target_words: int = 1800) -> list[_SyntheticDoc]:
    """Deterministic synthetic corpus for one seed: one document per topic in
    `_FACTS`, built from shuffled filler paragraphs with the topic's fact
    sentence inserted at a seed-dependent position -- so the same eight
    queries land in different chunks depending on chunk boundaries. Ground
    truth relevance (see `_relevance`) is defined purely by key-term overlap
    with the fact sentence, i.e. by construction, not by any model judgment."""
    docs: list[_SyntheticDoc] = []
    for topic_idx, (topic, fact, key_terms, query) in enumerate(_FACTS):
        rng = Random((seed * 1000) + topic_idx)
        paragraphs = [f"# Policy note: {topic.replace('_', ' ').title()}"]
        insert_after = rng.randint(2, 5)
        word_count = 0
        para_idx = 0
        while word_count < target_words:
            sentences = [rng.choice(_FILLER_SENTENCES) for _ in range(rng.randint(3, 6))]
            para = " ".join(sentences)
            paragraphs.append(para)
            word_count += len(para.split())
            para_idx += 1
            if para_idx == insert_after:
                paragraphs.append(fact)
                word_count += len(fact.split())
        text = "\n\n".join(paragraphs)
        docs.append(
            _SyntheticDoc(doc_id=f"{topic}-s{seed}", text=text, query=query, key_terms=key_terms)
        )
    return docs


def _relevance(chunk: Chunk, doc: _SyntheticDoc, *, min_hits: int = 2) -> float:
    """A chunk is relevant to `doc.query` iff it belongs to `doc.doc_id` AND
    contains at least `min(min_hits, len(key_terms))` of the fact's key terms.
    Binary, deterministic, defined entirely by construction."""
    if chunk.doc_id != doc.doc_id:
        return 0.0
    low = chunk.text.lower()
    hits = sum(1 for term in doc.key_terms if term in low)
    return 1.0 if hits >= min(min_hits, len(doc.key_terms)) else 0.0


def _evaluate_corpus(
    docs: Sequence[_SyntheticDoc],
    strategy: ChunkStrategyName,
    size: int,
    overlap: int,
    embedder: Embedder,
    contextualizer: Contextualizer | None = None,
) -> list[float]:
    """Chunk every doc, embed every chunk, rank the whole pooled corpus per
    query, return the list of per-query nDCG@10 scores."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(
            chunk_document(
                doc.text,
                doc.doc_id,
                strategy,
                size=size,
                overlap=overlap,
                embedder=embedder,
                contextualizer=contextualizer,
            )
        )
    if not all_chunks:
        return [0.0 for _ in docs]

    need_embed = [c for c in all_chunks if c.embedding is None]
    vec_map: dict[str, np.ndarray] = {}
    if need_embed:
        vecs = embedder.embed([c.full_text for c in need_embed])
        for c, v in zip(need_embed, vecs, strict=True):
            vec_map[c.chunk_id] = v
    chunk_vecs = {
        c.chunk_id: (
            np.asarray(c.embedding, dtype=float) if c.embedding is not None else vec_map[c.chunk_id]
        )
        for c in all_chunks
    }

    scores: list[float] = []
    for doc in docs:
        q_vec = embedder.embed([doc.query])[0]
        ranked = sorted(all_chunks, key=lambda c: -_cosine(chunk_vecs[c.chunk_id], q_vec))
        rel = [_relevance(c, doc) for c in ranked]
        scores.append(ndcg_at_k(rel, k=10))
    return scores


def run_strategy_ablation(
    *,
    size: int = 512,
    overlap: int = 0,
    seeds: Sequence[int] = (1, 2, 3),
    embedder: Embedder | None = None,
) -> list[MetricRow]:
    """Compare all five chunking strategies at a fixed (size, overlap) on the
    SYNTHETIC harness, reporting nDCG@10 with a bootstrap/t-interval CI per
    strategy (>= 3 seeds, per CONVENTIONS.md rule 5). Render with
    `localmind.eval.stats.rows_to_markdown(rows, ["ndcg@10"])`."""
    embedder = embedder or FakeHashEmbedder()
    rows: list[MetricRow] = []
    for strategy in STRATEGY_NAMES:
        per_seed: dict[int, list[float]] = {}
        for seed in seeds:
            corpus = _build_synthetic_corpus(seed)
            per_seed[seed] = _evaluate_corpus(corpus, strategy, size, overlap, embedder)
        est = aggregate_seeds(per_seed, minimum_seeds=len(list(seeds)))
        rows.append(
            MetricRow(name=strategy, values={"ndcg@10": est}, extra={"harness": "synthetic"})
        )
    return rows


class HeatmapResult(BaseModel):
    """Chunk-size x overlap grid of nDCG@10 on the SYNTHETIC harness --
    deliberately a grid, not a single winning number, per the spec.
    `harness` is stamped onto every result so it can never be mistaken for a
    real corpus number downstream."""

    model_config = ConfigDict(frozen=True)

    metric: str = "nDCG@10"
    harness: str = "SYNTHETIC HARNESS (deterministic bag-of-words embedder + code-generated corpus)"
    strategy: str
    sizes: list[int]
    overlap_fracs: list[float]
    seeds: list[int]
    cells: dict[str, dict[str, Any]]  # "{size}x{frac:.2f}" -> Estimate.to_dict()

    def cell(self, size: int, overlap_frac: float) -> Estimate:
        return Estimate.from_dict(self.cells[f"{size}x{overlap_frac:.2f}"])

    def to_markdown(self) -> str:
        header = (
            "| size \\ overlap | "
            + " | ".join(f"{round(f * 100)}%" for f in self.overlap_fracs)
            + " |"
        )
        sep = "|" + "---|" * (len(self.overlap_fracs) + 1)
        lines = [
            f"**{self.metric} heatmap -- {self.harness}** "
            f"(strategy={self.strategy}, seeds={self.seeds})",
            "",
            header,
            sep,
        ]
        for size in self.sizes:
            row_cells = [self.cell(size, f).format(3) for f in self.overlap_fracs]
            lines.append(f"| {size} | " + " | ".join(row_cells) + " |")
        return "\n".join(lines)


def run_size_overlap_ablation(
    *,
    strategy: ChunkStrategyName = "recursive",
    sizes: Sequence[int] = CHUNK_SIZES,
    overlap_fracs: Sequence[float] = OVERLAP_FRACS,
    seeds: Sequence[int] = (1, 2, 3),
    embedder: Embedder | None = None,
) -> HeatmapResult:
    """The chunk-size x overlap ablation the spec asks for: a `sizes` x
    `overlap_fracs` grid of nDCG@10 (with CI) on the SYNTHETIC harness, for
    one chunking `strategy` (default `recursive`, the strategy for which
    "size" and "overlap" are both structurally meaningful)."""
    embedder = embedder or FakeHashEmbedder()
    seeds_l = list(seeds)
    cells: dict[str, dict[str, Any]] = {}
    for size in sizes:
        for frac in overlap_fracs:
            overlap = round(size * frac)
            per_seed: dict[int, list[float]] = {}
            for seed in seeds_l:
                corpus = _build_synthetic_corpus(seed)
                per_seed[seed] = _evaluate_corpus(corpus, strategy, size, overlap, embedder)
            est = aggregate_seeds(per_seed, minimum_seeds=len(seeds_l))
            cells[f"{size}x{frac:.2f}"] = est.to_dict()
    return HeatmapResult(
        strategy=strategy,
        sizes=list(sizes),
        overlap_fracs=list(overlap_fracs),
        seeds=seeds_l,
        cells=cells,
    )
