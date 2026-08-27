"""Contextual retrieval (implementation.md §11): prepend a 1-2 sentence
LLM-generated situating context to each chunk before embedding. Per the spec,
this is the highest-impact of the five chunking strategies, and the per-chunk
LLM call is exactly the job LocalMind-31M's `ControlPlane` is sized for -- a
short, cheap, deterministic-ish completion, run once per chunk at ingestion
time.

**The argument this module exists to make measurable**: on a paid frontier
API (illustrated here as "qwen3-4b (hosted API)"), contextual retrieval is
expensive -- every chunk of every document costs a metered call. Run on our
own model, it's a few CPU milliseconds and free. `Contextualizer` is the seam
that makes either backend injectable; `CostModel` carries the (illustrative,
clearly-labeled) per-call latency/cost so `benchmark_contextualizers` can
report the delta *without* needing torch, a GPU, or network access to prove
the shape of the argument.

`Contextualizer` is deliberately ``ControlPlane``-style (CONVENTIONS.md): a
narrow, duck-typed, `runtime_checkable` Protocol with an offline default
implementation, the same shape as `localmind.agent.state.ControlPlane`. It is
NOT that Protocol reused -- `ControlPlane`'s three methods (`route`/`grade`/
`rewrite`) are frozen by CONVENTIONS.md and have no situating-context method --
so this module defines its own seam in the same spirit, and imports neither
`torch` nor `localmind.model` nor (to keep this module decoupled from the
agent package's frozen contract) `localmind.agent`.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

__all__ = [
    "LOCAL_31M_COST_MODEL",
    "QWEN3_4B_API_COST_MODEL",
    "ChunkLike",
    "ContextualizeComparison",
    "ContextualizeRecord",
    "Contextualizer",
    "CostModel",
    "HeuristicContextualizer",
    "SimulatedContextualizer",
    "benchmark_contextualizers",
    "contextualize_chunks",
]


@runtime_checkable
class Contextualizer(Protocol):
    """Generates a short situating context for one chunk, given the whole
    document it came from. Production backends: `localmind.agent`'s 31M
    `ControlPlane` (CPU, ~ms, free) or a hosted Qwen3-4B call (network,
    metered). Neither is imported here -- callers inject whichever they have."""

    name: str

    def situate(self, document: str, chunk: str) -> str: ...


@runtime_checkable
class ChunkLike(Protocol):
    """The subset of `localmind.ingestion.chunking.Chunk` this module actually
    needs. Structural on purpose, so this module never imports `chunking.py`
    (which imports *this* module) -- there is no import cycle either way."""

    chunk_id: str
    text: str


class CostModel(BaseModel):
    """Illustrative per-call cost/latency profile for a `Contextualizer`
    backend. These numbers are placeholders that make the *shape* of the
    tradeoff measurable offline (no network, no live pricing lookup) --
    swap in measured wall-clock latency and a current price sheet before
    using this for a real budget."""

    model_config = ConfigDict(frozen=True)

    label: str
    simulated_latency_ms: float
    simulated_cost_usd_per_call: float
    note: str = "illustrative placeholder, not fetched from a live price sheet"


LOCAL_31M_COST_MODEL = CostModel(
    label="localmind-31m (own weights, CPU)",
    simulated_latency_ms=4.0,
    simulated_cost_usd_per_call=0.0,
    note="runs on hardware we already own; marginal cost is electricity, not a metered API bill",
)

QWEN3_4B_API_COST_MODEL = CostModel(
    label="qwen3-4b (hosted API)",
    simulated_latency_ms=450.0,
    simulated_cost_usd_per_call=0.0006,
    note=(
        "illustrative order-of-magnitude for a ~4B hosted-model call "
        "(typical short-prompt latency + per-token metering); not a live quote"
    ),
)


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _summarize_document(document: str, max_sentences: int = 2) -> str:
    """Deterministic extractive fallback: the document's own opening
    sentence(s) -- a cheap, model-free stand-in for "what is this document
    about"."""
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(document.strip()) if s.strip()]
    return " ".join(sentences[:max_sentences])


class HeuristicContextualizer:
    """Deterministic, model-free situating context: the document's own
    opening sentence(s), prefixed with a fixed template. Always-on offline
    default; also what `SimulatedContextualizer` delegates to under the hood
    (the two model backends differ in cost/latency profile, not -- in this
    offline environment -- in the text they produce)."""

    name = "heuristic"

    def situate(self, document: str, chunk: str) -> str:
        del chunk  # the heuristic only looks at document-level context
        summary = _summarize_document(document)
        return f'This chunk is from a document that begins: "{summary}"' if summary else ""


class SimulatedContextualizer:
    """Stands in for a real model backend (our own 31M `ControlPlane` or a
    hosted Qwen3-4B call) in an environment with no torch, no GPU, and no
    network. Text generation delegates to `HeuristicContextualizer` -- what
    differs between two instances of this class is the injected `CostModel`,
    which is what `benchmark_contextualizers` reports on.

    `simulate_wall_clock=True` actually sleeps for `cost_model.simulated_latency_ms`
    so a caller can observe the delta in real wall-clock time; it defaults to
    `False` so unit tests stay instant and latency is instead read off the
    `CostModel` analytically."""

    def __init__(self, cost_model: CostModel, *, simulate_wall_clock: bool = False) -> None:
        self.cost_model = cost_model
        self.name = cost_model.label
        self.simulate_wall_clock = simulate_wall_clock
        self._backend = HeuristicContextualizer()

    def situate(self, document: str, chunk: str) -> str:
        if self.simulate_wall_clock and self.cost_model.simulated_latency_ms > 0:
            time.sleep(self.cost_model.simulated_latency_ms / 1000.0)
        return self._backend.situate(document, chunk)


class ContextualizeRecord(BaseModel):
    """One chunk's situating context plus the cost of producing it."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    context: str
    latency_ms: float
    cost_usd: float
    backend: str


def contextualize_chunks(
    document: str,
    chunks: Sequence[ChunkLike],
    contextualizer: Contextualizer | None = None,
    *,
    cost_model: CostModel | None = None,
) -> list[ContextualizeRecord]:
    """Run `contextualizer.situate` over every chunk of `document`, measuring
    real wall-clock latency per call (`time.perf_counter`) and attaching
    `cost_model`'s per-call price (0.0 if none given -- the offline default)."""
    ctx = contextualizer or HeuristicContextualizer()
    backend_name = getattr(ctx, "name", type(ctx).__name__)
    cost = cost_model.simulated_cost_usd_per_call if cost_model is not None else 0.0
    out: list[ContextualizeRecord] = []
    for chunk in chunks:
        t0 = time.perf_counter()
        situating = ctx.situate(document, chunk.text)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        out.append(
            ContextualizeRecord(
                chunk_id=chunk.chunk_id,
                context=situating,
                latency_ms=latency_ms,
                cost_usd=cost,
                backend=backend_name,
            )
        )
    return out


class ContextualizeComparison(BaseModel):
    """Head-to-head: our own model vs. a hosted API, over the same chunks of
    the same document. `local`/`api` are `(records, cost_model)` pairs so the
    caller can read both the measured wall-clock latency and the (labelled
    illustrative) simulated cost."""

    model_config = ConfigDict(frozen=True)

    local_records: list[ContextualizeRecord]
    api_records: list[ContextualizeRecord]
    local_cost_model: CostModel
    api_cost_model: CostModel

    @property
    def local_total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.local_records)

    @property
    def api_total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.api_records)

    @property
    def cost_savings_usd(self) -> float:
        """>= 0: what running on our own model saves vs. the hosted API, over
        this batch, at the (illustrative) `CostModel` prices."""
        return self.api_total_cost_usd - self.local_total_cost_usd

    @property
    def simulated_latency_speedup(self) -> float:
        """`api / local` simulated per-call latency ratio (>1 means local is
        faster). Uses the `CostModel`s, not measured wall-clock, since the
        default `simulate_wall_clock=False` makes both run near-instantly in
        this offline environment."""
        if self.local_cost_model.simulated_latency_ms <= 0:
            return float("inf")
        return self.api_cost_model.simulated_latency_ms / self.local_cost_model.simulated_latency_ms

    def render(self) -> str:
        return (
            f"contextual retrieval, {len(self.local_records)} chunks -- "
            f"{self.local_cost_model.label}: ${self.local_total_cost_usd:.4f}, "
            f"~{self.local_cost_model.simulated_latency_ms:.0f} ms/call (SIMULATED) vs. "
            f"{self.api_cost_model.label}: ${self.api_total_cost_usd:.4f}, "
            f"~{self.api_cost_model.simulated_latency_ms:.0f} ms/call (SIMULATED) -- "
            f"local saves ${self.cost_savings_usd:.4f} and is "
            f"{self.simulated_latency_speedup:.1f}x faster (both SIMULATED, illustrative placeholders)"
        )


def benchmark_contextualizers(
    document: str,
    chunks: Sequence[ChunkLike],
    *,
    local: Contextualizer | None = None,
    api: Contextualizer | None = None,
    local_cost_model: CostModel = LOCAL_31M_COST_MODEL,
    api_cost_model: CostModel = QWEN3_4B_API_COST_MODEL,
) -> ContextualizeComparison:
    """Run the same chunks through both a "local 31M" and a "hosted Qwen3-4B"
    contextualizer and report the (simulated, clearly-labelled) cost/latency
    delta. Callers may inject real backends later; the defaults are
    deterministic offline simulators."""
    local_ctx = local or SimulatedContextualizer(local_cost_model)
    api_ctx = api or SimulatedContextualizer(api_cost_model)
    return ContextualizeComparison(
        local_records=contextualize_chunks(
            document, chunks, local_ctx, cost_model=local_cost_model
        ),
        api_records=contextualize_chunks(document, chunks, api_ctx, cost_model=api_cost_model),
        local_cost_model=local_cost_model,
        api_cost_model=api_cost_model,
    )
