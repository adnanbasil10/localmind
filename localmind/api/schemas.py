"""Wire schemas for the RAG gateway — the request/response contract, and nothing else.

Pydantic only. FastAPI is never imported here, so the contract can be validated, serialised and
tested with the base dependencies alone (``fastapi``/``uvicorn`` live in the ``rag`` extra and are
very likely absent in a bare checkout). Same split as ``localmind/inference/server.py``: whatever
decides the bytes on the wire is plain Python, and the web framework only routes to it.

Boundary reminder (``docs/architecture.md``, "Boundaries that matter"): these are the *RAG
gateway's* schemas. The OpenAI-compatible model schemas (``/v1/chat/completions`` and friends)
belong to ``localmind.inference.server`` and are deliberately not mirrored here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from localmind.agent.guardrails import MAX_QUERY_CHARS

__all__ = [
    "MAX_QUERY_CHARS",
    "SNIPPET_CHARS",
    "ChatRequest",
    "ChatResponse",
    "CitationOut",
    "ErrorBody",
    "ErrorCode",
    "ErrorResponse",
    "HealthResponse",
    "SourceOut",
]

SNIPPET_CHARS = 400
"""How much source text is echoed back to the client per citation. Bounded so a response can
never become an exfiltration channel for a whole document."""

ErrorCode = Literal[
    "invalid_request",
    "unauthorized",
    "forbidden",
    "not_found",
    "rate_limited",
    "unavailable",
    "internal",
]
"""Client-facing error vocabulary. Deliberately coarse: an error code is a routing hint for the
caller, never a description of what went wrong inside the process."""


class ChatRequest(BaseModel):
    """A RAG question. ``filters`` are metadata equality constraints on the corpus."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    query: str = Field(
        validation_alias=AliasChoices("query", "question"),
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description="The user's question. `question` is accepted as an alias.",
    )
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        description="Conversation key for the agent's session memory.",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Metadata equality filters applied to retrieval, e.g. {'doc_type': 'pdf'}.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Chunks to ground the answer in. Server default if omitted.",
    )
    stream: bool = Field(
        default=False, description="Stream the answer as SSE instead of one JSON body."
    )
    use_cache: bool = Field(default=True, description="Consult (and populate) the response caches.")


class CitationOut(BaseModel):
    """One inline ``[n]`` marker in the answer, resolved to the chunk that supports it."""

    model_config = ConfigDict(frozen=True)

    marker: int
    chunk_id: str
    doc_id: str = ""
    uri: str = ""
    snippet: str = ""


class SourceOut(BaseModel):
    """A grounding chunk that survived relevance grading and injection quarantine."""

    model_config = ConfigDict(frozen=True)

    marker: int
    chunk_id: str
    doc_id: str = ""
    uri: str = ""
    score: float = 0.0
    origin: str = "documents"
    snippet: str = ""


class ChatResponse(BaseModel):
    """The answer, its citations, and enough provenance to debug a bad one."""

    answer: str
    status: Literal["answered", "refused", "error"] = "answered"
    refusal_reason: str = ""
    citations: list[CitationOut] = Field(default_factory=list)
    sources: list[SourceOut] = Field(default_factory=list)
    request_id: str = ""
    session_id: str = "default"
    route: str | None = None
    cached: bool = False
    cache_layer: str | None = None
    latency_ms: float = 0.0
    steps: int = 0
    tokens_used: int = 0
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """``GET /health``. Answers "is this process able to serve?", never "is Postgres up?".

    The Docker healthcheck in ``deploy/Dockerfile.api`` probes this endpoint, so it must return
    200 with zero backing services running; it therefore reports *configuration*, and performs no
    I/O of any kind.
    """

    status: Literal["ok"] = "ok"
    service: str = "localmind-api"
    version: str = "0.1.0"
    uptime_s: float = 0.0
    components: dict[str, str] = Field(default_factory=dict)


class ErrorBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ErrorCode
    message: str
    request_id: str = ""


class ErrorResponse(BaseModel):
    """Every non-2xx response has exactly this shape.

    ``message`` is always a fixed, human-written string chosen from a closed set — never an
    exception's ``str()``, never a traceback, never a database DSN. Internals go to the span and
    the log; the client gets a code and a request id to quote back.
    """

    model_config = ConfigDict(frozen=True)

    error: ErrorBody
