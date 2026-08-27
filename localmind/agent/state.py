"""Typed state for the LocalMind agent, plus every seam to the outside world.

Design note: the agent is a plain Python state machine over a single pydantic
state object (see `docs/decisions/0009-state-machine-over-langgraph.md`). Every
external dependency -- the 31M control plane, the Ollama generator, the
retrieval stack, the web, the database, the image store, the clock -- is a
`Protocol` declared here. Nothing in this package imports torch, httpx,
duckduckgo_search, or `localmind.retrieval` at module import time, so the whole
agent runs offline against deterministic fakes.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------

Route = Literal["in_domain", "out_of_domain", "needs_web"]
"""The 3-way decision produced by LocalMind-31M's router head (spec section 9, 5a)."""

Trust = Literal["user", "agent", "untrusted"]
"""Provenance of a piece of text. `untrusted` text may never originate a tool call."""

ErrorCode = Literal[
    "timeout",
    "invalid_input",
    "unavailable",
    "not_found",
    "denied",
    "rate_limited",
    "internal",
]

Status = Literal["running", "answered", "refused", "error"]


class Node(StrEnum):
    """Nodes of the state machine. RESPOND and REFUSE are terminal."""

    ROUTE = "route"
    RETRIEVE = "retrieve"
    GRADE = "grade"
    REWRITE = "rewrite"
    WEB_SEARCH = "web_search"
    GENERATE = "generate"
    VERIFY = "verify"
    RESPOND = "respond"
    REFUSE = "refuse"


TERMINAL_NODES: frozenset[Node] = frozenset({Node.RESPOND, Node.REFUSE})


# --------------------------------------------------------------------------------------
# Seams (Protocols). Implementations live outside this package; tests inject fakes.
# --------------------------------------------------------------------------------------


@runtime_checkable
class ControlPlane(Protocol):
    """The three production jobs of the 31M model, run on CPU.

    Frozen in CONVENTIONS.md -- re-exported from `router.py`, `grader.py` and
    `rewriter.py` so callers can import it from any of them. Do not extend this
    Protocol; see `InjectionAwareControlPlane` for the optional fourth head.
    """

    def route(self, query: str) -> Route: ...
    def grade(self, query: str, chunk: str) -> tuple[bool, float]: ...
    def rewrite(self, query: str, history: list[str]) -> str: ...


@runtime_checkable
class InjectionAwareControlPlane(ControlPlane, Protocol):
    """Optional fourth head: a cheap binary prompt-injection classifier.

    Kept as a separate Protocol so the frozen `ControlPlane` contract is not
    modified. `guardrails.build_injection_classifier` duck-types on the presence
    of `classify_injection`, so a plain `ControlPlane` still works.
    """

    def classify_injection(self, text: str) -> tuple[bool, float]: ...


class GenerationResult(BaseModel):
    """What the Ollama-backed generator returns."""

    model_config = ConfigDict(frozen=True)

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "unknown"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class Generator(Protocol):
    """The big local model (Ollama). Only `generate` and `summarize_document` use it."""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult: ...


class RetrievedChunk(BaseModel):
    """A chunk coming back from the retrieval stack or the web.

    `text` is ALWAYS untrusted: it originates in a PDF or a web page we did not
    write. It is never concatenated into a prompt without `guardrails.wrap_untrusted`.
    """

    chunk_id: str
    doc_id: str = ""
    text: str
    score: float = 0.0
    source: Literal["documents", "web", "database", "image"] = "documents"
    uri: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    trust: Trust = "untrusted"


@runtime_checkable
class Retriever(Protocol):
    """Minimal contract onto `localmind.retrieval` (Phase 8, built concurrently).

    Deliberately narrow and duck-typed: an implementation may return any objects
    exposing `chunk_id`/`doc_id`/`text`/`score` attributes, or plain mappings with
    those keys. `tools.search_documents.coerce_chunk` normalises both. The agent
    never imports `localmind.retrieval`.
    """

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[Any]: ...


class WebResult(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""


@runtime_checkable
class WebSearchProvider(Protocol):
    """Free web search (DuckDuckGo in production, a fake in tests)."""

    def search(self, query: str, k: int = 5) -> Sequence[WebResult]: ...


@runtime_checkable
class Database(Protocol):
    """Read-only SQL seam for `query_database`."""

    def execute(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> Sequence[Mapping[str, Any]]: ...


class ImageRecord(BaseModel):
    image_id: str
    caption: str = ""
    uri: str = ""
    doc_id: str = ""
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ImageStore(Protocol):
    """Seam for `retrieve_image`. Returns metadata + captions, never raw bytes."""

    def get(self, image_id: str) -> ImageRecord | None: ...
    def search(self, query: str, k: int = 3) -> Sequence[ImageRecord]: ...


@runtime_checkable
class Clock(Protocol):
    """Injectable clock so retry/backoff and rate limits are testable instantly."""

    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class WallClock:
    """Real clock. `now()` is monotonic; never use it for wall-clock timestamps."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class ManualClock:
    """Deterministic clock for tests: time only moves when you move it."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = float(start)
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.slept.append(float(seconds))
            self.t += float(seconds)

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


# --------------------------------------------------------------------------------------
# Tool payloads
# --------------------------------------------------------------------------------------


class ToolError(BaseModel):
    """A structured failure returned TO the model, never raised at it."""

    model_config = ConfigDict(frozen=True)

    code: ErrorCode
    message: str
    tool: str
    retryable: bool = False

    def render(self) -> str:
        return f"[tool_error tool={self.tool} code={self.code}] {self.message}"


class ToolResult(BaseModel):
    """Uniform envelope for every tool call."""

    model_config = ConfigDict(frozen=True)

    tool: str
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: ToolError | None = None
    elapsed_ms: float = 0.0
    attempts: int = 1
    cached: bool = False
    idempotency_key: str = ""

    def render(self) -> str:
        """Model-facing rendering: successes as data, failures as structured errors."""
        if self.ok:
            return f"[tool_result tool={self.tool}] {self.data}"
        if self.error is None:  # pragma: no cover - defensive
            return f"[tool_error tool={self.tool} code=internal] unspecified failure"
        return self.error.render()


class ToolCall(BaseModel):
    """A tool invocation *request*, carrying its provenance.

    Guardrail (c): `provenance` may only ever be `user` or `agent`. Retrieved
    text cannot construct one of these, because the planner that builds them is
    never given access to retrieved text (see `graph.plan_tools`).
    """

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    provenance: Literal["user", "agent"] = "agent"
    result: ToolResult | None = None
    denied_reason: str = ""


# --------------------------------------------------------------------------------------
# Agent state
# --------------------------------------------------------------------------------------


class GradedChunk(BaseModel):
    """A retrieved chunk after the 31M grader and the injection classifier saw it."""

    chunk: RetrievedChunk
    relevant: bool = False
    score: float = 0.0
    injection_flagged: bool = False
    injection_score: float = 0.0
    quarantined: bool = False
    quarantine_reason: str = ""

    @property
    def usable(self) -> bool:
        return self.relevant and not self.quarantined


class Citation(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str = ""
    uri: str = ""


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    score: float = 0.0
    supporting_chunk_id: str = ""


class NodeTiming(BaseModel):
    """Per-node latency, so the README diagram can be annotated with real p50s."""

    node: str
    ms: float
    iteration: int = 0
    detail: str = ""


class Budget(BaseModel):
    """Hard caps. Every loop in `graph.py` is bounded by at least two of these."""

    model_config = ConfigDict(frozen=True)

    max_rewrites: int = 2
    max_web_fallbacks: int = 1
    max_regenerations: int = 1
    max_retrievals: int = 4
    max_steps: int = 32
    max_wall_clock_s: float = 30.0
    max_tokens: int = 8192
    min_relevant_chunks: int = 2
    min_relevant_after_fallback: int = 1
    top_k: int = 5


class AgentState(BaseModel):
    """The single mutable object threaded through the state machine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = "default"
    user_query: str
    search_query: str = ""
    route: Route | None = None

    graded: list[GradedChunk] = Field(default_factory=list)
    sources: list[RetrievedChunk] = Field(default_factory=list)
    quarantined: list[GradedChunk] = Field(default_factory=list)
    pending: list[RetrievedChunk] = Field(default_factory=list)
    planned: list[ToolCall] = Field(default_factory=list)

    rewrites: list[str] = Field(default_factory=list)
    n_rewrites: int = 0
    n_retrievals: int = 0
    n_web_fallbacks: int = 0
    n_regenerations: int = 0

    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    verdicts: list[ClaimVerdict] = Field(default_factory=list)

    tool_calls: list[ToolCall] = Field(default_factory=list)

    status: Status = "running"
    refusal_reason: str = ""
    node: Node = Node.ROUTE
    steps: int = 0
    tokens_used: int = 0
    started_at: float = 0.0
    elapsed_s: float = 0.0

    timings: list[NodeTiming] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    injection_detected: bool = False
    pii_detected: bool = False
    strict_mode: bool = False

    def log(self, message: str) -> None:
        self.trace.append(message)

    @property
    def relevant_chunks(self) -> list[GradedChunk]:
        return [g for g in self.graded if g.usable]

    @property
    def terminated(self) -> bool:
        return self.status != "running"


class AgentResult(BaseModel):
    """What `Agent.run` hands back."""

    answer: str
    status: Status
    refusal_reason: str = ""
    citations: list[Citation] = Field(default_factory=list)
    sources: list[RetrievedChunk] = Field(default_factory=list)
    state: AgentState
    steps: int = 0
    elapsed_s: float = 0.0
    tokens_used: int = 0

    @property
    def refused(self) -> bool:
        return self.status in ("refused", "error")

    def latency_by_node(self) -> dict[str, float]:
        """p50 latency per node, in ms -- the numbers that annotate the README diagram."""
        return percentile_by_node(self.state.timings, 50.0)


def percentile_by_node(timings: Sequence[NodeTiming], pct: float = 50.0) -> dict[str, float]:
    """Percentile of per-node latency, in milliseconds. Pure python, no numpy needed."""
    buckets: dict[str, list[float]] = {}
    for t in timings:
        buckets.setdefault(t.node, []).append(t.ms)
    out: dict[str, float] = {}
    for node, values in buckets.items():
        if not values:  # pragma: no cover - defensive
            continue
        values.sort()
        idx = (pct / 100.0) * (len(values) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(values) - 1)
        frac = idx - lo
        out[node] = values[lo] * (1 - frac) + values[hi] * frac
    return out


def aggregate_p50(results: Sequence[AgentResult], pct: float = 50.0) -> dict[str, float]:
    """Per-node percentile across many runs -- for the benchmark table."""
    timings: list[NodeTiming] = []
    for r in results:
        timings.extend(r.state.timings)
    return percentile_by_node(timings, pct)
