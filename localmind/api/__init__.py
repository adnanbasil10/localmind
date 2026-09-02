"""LocalMind API — the RAG gateway (implementation.md §16, Phase 12).

What this package is
--------------------
The front door of the *system*: authentication, rate limiting, request-ID propagation, the
OpenTelemetry **root span** every other span nests under, structured errors that never leak
internals, and the entrypoint that drives the agent state machine and answers with citations.

What it is **not**
------------------
It is not a model server. ``localmind.inference.server`` exposes the OpenAI-compatible endpoints
for *the model* (``/v1/chat/completions``, ``/v1/completions``, ``/v1/embeddings``); that is a
different process with a different concern, and nothing here imports it
(``docs/architecture.md``, "Boundaries that matter").

Composition, not reimplementation
---------------------------------
Every capability is borrowed from a package that already owns it:

=========================================  =========================================
``localmind.agent.graph.Agent``            the state machine ``POST /chat`` drives
``localmind.obs.tracing``                  the root span and its child stage spans
``localmind.obs.metrics``                  the Prometheus series behind ``/metrics``
``localmind.cache``                        response / semantic / embedding caches
``localmind.agent.guardrails.RateLimiter`` the token bucket behind rate limiting
=========================================  =========================================

The gateway's own code is the glue between them, plus the adapters in ``deps.py`` where two
packages' contracts do not line up — notably
:class:`~localmind.api.deps.ScoredDocRetriever`, which rejoins the text that
``localmind.retrieval``'s ``ScoredDoc`` (``doc_id``, ``score``) deliberately does not carry.

Import discipline
-----------------
``fastapi``/``uvicorn`` live in the ``rag`` extra and may not be installed. Every module here is
importable without them: FastAPI is imported lazily inside ``routes.create_router``,
``routes.install_error_handlers`` and ``main.create_app``, and ``create_app`` is re-exported
lazily below. ``import localmind.api`` must never fail because of a missing extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from localmind.api.deps import (
    ApiKeyAuth,
    Container,
    ExtractiveGenerator,
    HttpMetrics,
    OllamaGenerator,
    Principal,
    RateLimitPolicy,
    ScoredDocRetriever,
    Settings,
    TracedGenerator,
    TracingToolRegistry,
    build_container,
)
from localmind.api.middleware import (
    REQUEST_ID_HEADER,
    ApiError,
    RequestContext,
    RequestContextMiddleware,
    current_context,
    error_payload,
)
from localmind.api.routes import (
    SSE_DONE,
    build_chat_response,
    execute_chat,
    health_report,
    sse_frame,
    stream_events,
)
from localmind.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationOut,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    SourceOut,
)

__all__ = [
    "REQUEST_ID_HEADER",
    "SSE_DONE",
    "ApiError",
    "ApiKeyAuth",
    "ChatRequest",
    "ChatResponse",
    "CitationOut",
    "Container",
    "ErrorBody",
    "ErrorResponse",
    "ExtractiveGenerator",
    "HealthResponse",
    "HttpMetrics",
    "OllamaGenerator",
    "Principal",
    "RateLimitPolicy",
    "RequestContext",
    "RequestContextMiddleware",
    "ScoredDocRetriever",
    "Settings",
    "SourceOut",
    "TracedGenerator",
    "TracingToolRegistry",
    "build_chat_response",
    "build_container",
    "create_app",
    "current_context",
    "error_payload",
    "execute_chat",
    "health_report",
    "sse_frame",
    "stream_events",
]

_LAZY: dict[str, tuple[str, str]] = {
    "create_app": ("localmind.api.main", "create_app"),
}


def __getattr__(name: str) -> Any:
    """Lazy re-export, so a missing ``rag`` extra costs an error only at ``create_app`` time."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target[0]), target[1])


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from localmind.api.main import create_app
