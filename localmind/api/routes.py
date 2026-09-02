"""The gateway's endpoints, and the framework-free core they are thin wrappers around.

Layering, and why it is like this
---------------------------------
Everything that decides *what the client gets* — :func:`execute_chat`, :func:`build_chat_response`,
:func:`health_report`, the SSE framing — is plain Python with no web framework anywhere near it.
FastAPI is imported lazily inside :func:`create_router` and :func:`install_error_handlers`, so
``import localmind.api.routes`` works with only the base dependencies and the entire RAG path is
testable without the ``rag`` extra installed. This mirrors ``localmind/inference/server.py``.

The endpoints
-------------
``POST /chat`` (``POST /query`` is the same handler)
    The RAG entrypoint. Drives ``localmind.agent.graph.Agent`` — the state machine — and returns
    the answer **with citations**, or streams it as SSE. Authenticated and rate limited.
``GET /health``
    The Docker healthcheck target (``deploy/Dockerfile.api``). Performs no I/O, so it answers 200
    with no Postgres, no Redis, no Ollama and no OTel collector running.
``GET /metrics``
    Prometheus exposition, at the path ``deploy/prometheus.yml`` already scrapes.

This is the *RAG* gateway. Model serving — ``/v1/chat/completions``, ``/v1/completions``,
``/v1/embeddings`` — belongs to ``localmind.inference.server``, a different process with a
different concern (``docs/architecture.md``, "Boundaries that matter"). Nothing here imports it.

One deliberate deviation from the rest of the codebase: this module does **not** use
``from __future__ import annotations``. FastAPI resolves a handler's type hints with
``typing.get_type_hints`` against the *module* globals, and the handlers below are annotated with
``fastapi.Request`` — a name that only exists as a local inside :func:`create_router`, because
FastAPI is imported lazily. With PEP 563 on, every annotation becomes a string and OpenAPI
generation dies on the unresolvable forward reference. Python 3.11 evaluates ``X | None``
natively, so evaluating annotations eagerly here costs nothing.
"""

import json
import time
from collections.abc import Iterator, Mapping
from typing import Any

from localmind.agent.state import AgentResult, AgentState
from localmind.api.deps import Container
from localmind.api.middleware import (
    API_KEY_HEADER,
    GENERIC_MESSAGES,
    REQUEST_ID_HEADER,
    ApiError,
    RequestContext,
    context_from_scope,
    current_context,
    error_payload,
    new_request_id,
    use_context,
)
from localmind.api.schemas import (
    SNIPPET_CHARS,
    ChatRequest,
    ChatResponse,
    CitationOut,
    HealthResponse,
    SourceOut,
)
from localmind.obs.semconv import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_SYSTEM,
    LOCALMIND_CACHE_HIT,
    LOCALMIND_CACHE_LAYER,
    LOCALMIND_REFUSAL,
    LOCALMIND_RETRIEVAL_DOC_COUNT,
    LOCALMIND_RETRIEVAL_DOC_IDS,
    LOCALMIND_ROUTE_DECISION,
    LOCALMIND_VERIFY_PASSED,
)

__all__ = [
    "AGENT_NODE_DURATION_MS",
    "NODE_TO_STAGE",
    "SSE_DONE",
    "authorize",
    "build_chat_response",
    "chunk_text",
    "create_router",
    "execute_chat",
    "health_report",
    "install_error_handlers",
    "safe_refusal_reason",
    "safe_warnings",
    "sse_frame",
    "status_to_code",
    "stream_events",
]

SSE_DONE = "data: [DONE]\n\n"
"""Same terminator the OpenAI SSE protocol uses, so a client written against the inference server
can reuse its stream loop verbatim against this one."""

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

NODE_TO_STAGE: dict[str, str] = {
    "route": "route",
    "grade": "grade",
    "generate": "generate",
    "verify": "verify",
}
"""Agent node name -> ``obs`` pipeline stage name.

The two vocabularies were designed independently and only partly overlap, which is worth stating
plainly rather than papering over. ``obs.semconv.PIPELINE_STAGES`` is retrieval-shaped
(``route, embed, bm25, dense, colbert, fuse, rerank, grade, generate, verify``) while the agent's
nodes are control-flow-shaped (``route, retrieve, grade, rewrite, web_search, generate, verify,
respond, refuse``). Four names coincide; the agent's ``retrieve``/``rewrite``/``web_search`` have
no stage equivalent, and the stage list's ``embed``/``bm25``/``dense``/``colbert``/``fuse``/
``rerank`` are internal to the retrieval package, below the agent's ``Retriever`` seam. Nodes with
no stage are recorded as root-span attributes instead — see :data:`AGENT_NODE_DURATION_MS`."""

AGENT_NODE_DURATION_MS = "localmind.agent.node.{node}.duration_ms"
"""Root-span attribute template for agent nodes that have no ``PIPELINE_STAGES`` equivalent."""

_REFUSAL_VOCABULARY: tuple[tuple[str, str], ...] = (
    ("out of domain", "out_of_domain"),
    ("no relevant evidence", "insufficient_evidence"),
    ("no sources to ground", "insufficient_evidence"),
    ("no generator", "generator_unavailable"),
    ("generator unavailable", "generator_unavailable"),
    ("budget exhausted", "budget_exhausted"),
    ("cap reached", "budget_exhausted"),
    ("unsupported claim", "unverified_answer"),
    ("citation-required", "uncited_answer"),
    ("rate limit", "rate_limited"),
    ("prompt injection", "blocked_input"),
    ("internal error", "internal_error"),
)


# ======================================================================================
# Framework-free response construction
# ======================================================================================
def safe_refusal_reason(reason: str) -> str:
    """Collapse an agent refusal string onto a closed, client-safe vocabulary.

    ``AgentState.refusal_reason`` is written for an operator reading a trace, and can embed an
    exception's type and message (``"internal error in generate: ConnectError: ..."``) or a
    fragment of the offending claim. None of that may reach the client, so it is classified here
    and the raw string stays on the span.
    """
    low = reason.lower()
    for needle, code in _REFUSAL_VOCABULARY:
        if needle in low:
            return code
    return "refused" if reason else ""


def safe_warnings(state: AgentState) -> list[str]:
    """Derived, leak-proof flags. Never the raw ``state.warnings``, which quote tool errors."""
    flags: list[str] = []
    if state.pii_detected:
        flags.append("pii_redacted")
    if state.injection_detected:
        flags.append("injection_quarantined")
    if state.n_web_fallbacks:
        flags.append("web_fallback_used")
    if state.n_rewrites:
        flags.append("query_rewritten")
    if state.n_regenerations:
        flags.append("answer_regenerated")
    if any(c.result is not None and not c.result.ok for c in state.tool_calls):
        flags.append("tool_failure")
    return flags


def build_chat_response(
    result: AgentResult,
    *,
    request_id: str = "",
    session_id: str = "default",
    latency_ms: float = 0.0,
    cached: bool = False,
    cache_layer: str | None = None,
) -> ChatResponse:
    """Project an :class:`~localmind.agent.state.AgentResult` onto the wire schema.

    Source text is truncated to :data:`~localmind.api.schemas.SNIPPET_CHARS` so a citation is
    verifiable without the response becoming a way to pull whole documents out of the corpus.
    """
    sources = [
        SourceOut(
            marker=i + 1,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            uri=chunk.uri,
            score=chunk.score,
            origin=chunk.source,
            snippet=chunk.text[:SNIPPET_CHARS],
        )
        for i, chunk in enumerate(result.sources)
    ]
    snippets = {s.chunk_id: s.snippet for s in sources}
    citations = [
        CitationOut(
            marker=c.marker,
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            uri=c.uri,
            snippet=snippets.get(c.chunk_id, ""),
        )
        for c in result.citations
    ]
    status = result.status if result.status in ("answered", "refused", "error") else "error"
    return ChatResponse(
        answer=result.answer,
        status=status,
        refusal_reason=safe_refusal_reason(result.refusal_reason),
        citations=citations,
        sources=sources,
        request_id=request_id,
        session_id=session_id,
        route=result.state.route,
        cached=cached,
        cache_layer=cache_layer,
        latency_ms=latency_ms,
        steps=result.steps,
        tokens_used=result.tokens_used,
        warnings=safe_warnings(result.state),
    )


# ======================================================================================
# SSE framing (plain Python, asserted directly in tests)
# ======================================================================================
def sse_frame(event: Mapping[str, Any]) -> str:
    return "data: " + json.dumps(event, separators=(",", ":")) + "\n\n"


def chunk_text(text: str, size: int = 48) -> list[str]:
    step = max(1, int(size))
    return [text[i : i + step] for i in range(0, len(text), step)] if text else []


def stream_events(response: ChatResponse, chunk_size: int = 48) -> Iterator[str]:
    """SSE rendering of a completed :class:`ChatResponse`.

    Honest about what it is: the agent state machine returns a whole answer, so this streams a
    finished result rather than model tokens. TTFT over this endpoint is therefore end-to-end
    pipeline latency, not prefill latency — the token-level stream lives on the inference server's
    ``/v1/chat/completions``, which is a different process and a different concern. Citations
    arrive in their own frame before ``done``, so a UI can render the answer progressively and
    then attach markers.
    """
    yield sse_frame(
        {
            "type": "start",
            "request_id": response.request_id,
            "session_id": response.session_id,
            "status": response.status,
        }
    )
    for piece in chunk_text(response.answer, chunk_size):
        yield sse_frame({"type": "token", "text": piece})
    yield sse_frame(
        {
            "type": "citations",
            "citations": [c.model_dump() for c in response.citations],
            "sources": [s.model_dump() for s in response.sources],
        }
    )
    yield sse_frame(
        {
            "type": "done",
            "status": response.status,
            "refusal_reason": response.refusal_reason,
            "cached": response.cached,
            "latency_ms": response.latency_ms,
        }
    )
    yield SSE_DONE


# ======================================================================================
# Observability wiring
# ======================================================================================
def record_agent_observability(
    container: Container, result: AgentResult, context: RequestContext | None
) -> None:
    """Fan the agent's own measurements out to the root span and to Prometheus.

    Stages that were already traced *live* — retrieval and generation, via the wrappers in
    ``deps.py`` — are skipped here, detected through ``RequestTrace.stages_seen``, so no latency is
    counted twice. Everything else comes from ``AgentState.timings``, which the agent measures
    itself; this function never re-times anything.
    """
    metrics = container.metrics
    state = result.state
    covered: set[str] = set()
    if context is not None and context.trace is not None:
        covered = set(context.trace.stages_seen)

    for timing in state.timings:
        stage = NODE_TO_STAGE.get(timing.node)
        if stage is not None and stage not in covered:
            metrics.observe_stage_latency(stage, timing.ms / 1000.0)
        elif stage is None and context is not None:
            context.set_root_attribute(
                AGENT_NODE_DURATION_MS.format(node=timing.node), round(timing.ms, 3)
            )

    for call in state.tool_calls:
        if call.result is not None:
            metrics.record_tool_result(call.tool, call.result.ok)

    if result.status != "answered":
        metrics.record_refusal(state.route or "unknown")
    if result.status == "error":
        node_stage = NODE_TO_STAGE.get(state.node.value, "route")
        metrics.record_error(node_stage, "agent_error")

    if context is None:
        return
    doc_ids = [s.chunk_id for s in result.sources]
    context.set_root_attribute(GEN_AI_SYSTEM, "localmind")
    context.set_root_attribute(GEN_AI_OPERATION_NAME, "chat")
    context.set_root_attribute(LOCALMIND_ROUTE_DECISION, state.route or "unknown")
    context.set_root_attribute(LOCALMIND_RETRIEVAL_DOC_IDS, doc_ids)
    context.set_root_attribute(LOCALMIND_RETRIEVAL_DOC_COUNT, len(doc_ids))
    context.set_root_attribute(LOCALMIND_VERIFY_PASSED, result.status == "answered")
    if result.status != "answered":
        # The raw reason stays here, on the span, and never goes out on the wire.
        context.set_root_attribute(LOCALMIND_REFUSAL, state.refusal_reason or "refused")


# ======================================================================================
# The RAG entrypoint, framework-free
# ======================================================================================
def _semantic_cache_applicable(container: Container, request: ChatRequest) -> bool:
    """The semantic cache keys on query text alone.

    ``SemanticCache`` has no notion of filters or of a retrieval config, so consulting it for a
    filtered or non-default-``top_k`` request could serve an answer computed under different
    constraints — a false hit that its own tau sweep would never catch, because it is not a
    similarity failure. It is therefore only used for the plain case.
    """
    if container.semantic_cache is None:
        return False
    if request.filters:
        return False
    return request.top_k in (None, container.settings.top_k)


def _cache_lookup(
    container: Container, request: ChatRequest, config: Mapping[str, Any]
) -> tuple[ChatResponse, str] | None:
    filters = dict(request.filters or {})
    entry = container.response_cache.get(request.query, filters, config)
    container.metrics.record_cache("response", entry is not None)
    if entry is not None:
        return ChatResponse.model_validate_json(entry.response), "response"
    if _semantic_cache_applicable(container, request):
        assert container.semantic_cache is not None
        hit = container.semantic_cache.lookup(request.query)
        container.metrics.record_cache("semantic", hit.hit)
        if hit.hit and hit.answer:
            try:
                return ChatResponse.model_validate_json(hit.answer), "semantic"
            except ValueError:  # pragma: no cover - a legacy/foreign entry, treat as a miss
                return None
    return None


def execute_chat(
    container: Container,
    request: ChatRequest,
    *,
    request_id: str = "",
    context: RequestContext | None = None,
) -> ChatResponse:
    """Run one RAG request end to end. No framework, no I/O beyond the injected seams.

    Order: exact response cache (Layer 2) -> semantic cache (Layer 3) -> the agent state machine.
    A cache hit is reported as such in the body and on the span, never silently.
    """
    context = context if context is not None else current_context()
    rid = request_id or (context.request_id if context is not None else new_request_id())
    started = time.perf_counter()
    filters = dict(request.filters or {})
    config = container.cache_config(request.top_k)

    if request.use_cache:
        hit = _cache_lookup(container, request, config)
        if hit is not None:
            cached_response, layer = hit
            if context is not None:
                context.set_root_attribute(LOCALMIND_CACHE_HIT, True)
                context.set_root_attribute(LOCALMIND_CACHE_LAYER, layer)
            return cached_response.model_copy(
                update={
                    "request_id": rid,
                    "session_id": request.session_id,
                    "cached": True,
                    "cache_layer": layer,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
    if context is not None:
        context.set_root_attribute(LOCALMIND_CACHE_HIT, False)

    # Filters reach retrieval through the request context, not through `Agent.run`: the agent's
    # `retrieve` node calls `search_documents` with `{query, k}` only and `Agent.run` has no
    # filters parameter, so `deps.TracingToolRegistry` reads them from the ambient context and
    # merges them into the tool's arguments. That keeps the filter contract inside this package
    # rather than changing the agent's.
    if context is not None:
        context.filters = filters
        result = container.agent.run(
            request.query, request.session_id, budget=container.budget_for(request.top_k)
        )
    else:
        with use_context(RequestContext(request_id=rid, filters=filters)) as ctx:
            context = ctx
            result = container.agent.run(
                request.query, request.session_id, budget=container.budget_for(request.top_k)
            )

    latency_ms = (time.perf_counter() - started) * 1000.0
    record_agent_observability(container, result, context)
    response = build_chat_response(
        result,
        request_id=rid,
        session_id=request.session_id,
        latency_ms=latency_ms,
    )

    if request.use_cache and response.status == "answered":
        payload = response.model_copy(update={"request_id": "", "latency_ms": 0.0})
        raw = payload.model_dump_json()
        container.response_cache.put(request.query, raw, latency_ms, filters, config)
        if _semantic_cache_applicable(container, request):
            assert container.semantic_cache is not None
            container.semantic_cache.store(request.query, raw)
    return response


def health_report(container: Container) -> HealthResponse:
    """``GET /health``. Configuration only — no sockets, no queries, no imports of heavy things."""
    return HealthResponse(
        service=container.settings.service_name,
        version=container.settings.version,
        uptime_s=round(container.uptime_s, 3),
        components=container.health(),
    )


def authorize(container: Container, headers: Mapping[str, str], peer: str = "") -> str:
    """Authenticate, then rate limit. Returns the principal id used as the bucket key.

    Order matters: an unauthenticated caller must not be able to consume an authenticated
    caller's bucket, and rate limiting an anonymous caller by peer address is the only key
    available when auth is disabled.
    """
    principal = container.auth.identify(headers.get("authorization"), headers.get(API_KEY_HEADER))
    bucket = principal.key_id if principal.authenticated else f"anon:{peer or 'unknown'}"
    container.rate_limiter.check(bucket)
    return principal.key_id


def status_to_code(status: int) -> str:
    return {
        400: "invalid_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "invalid_request",
        422: "invalid_request",
        429: "rate_limited",
        503: "unavailable",
    }.get(status, "internal")


# ======================================================================================
# FastAPI surface (imported lazily; everything above works without it)
# ======================================================================================
def create_router(container: Container) -> Any:
    """Build the router. Raises a clear error if the ``rag`` extra is missing."""
    try:
        from fastapi import APIRouter, Request, Response
        from fastapi.responses import StreamingResponse
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "fastapi/uvicorn are in the 'rag' optional dependency group. "
            "Install with: uv pip install -e '.[rag]'"
        ) from exc

    router = APIRouter()

    def _headers(request: Request) -> dict[str, str]:
        return {k.lower(): v for k, v in request.headers.items()}

    def _run(payload: ChatRequest, request: Request) -> Any:
        context = context_from_scope(request.scope)
        rid = context.request_id if context is not None else new_request_id()
        peer = request.client.host if request.client else ""
        principal_id = authorize(container, _headers(request), peer)
        if context is not None:
            context.principal_id = principal_id

        response = execute_chat(container, payload, request_id=rid, context=context)
        if response.status == "error":
            # The agent caught an internal failure and returned a result rather than raising.
            # Surface it as a server-side fault, with the details left on the span.
            raise ApiError(503, "unavailable")
        if payload.stream:
            return StreamingResponse(
                stream_events(response, container.settings.stream_chunk_chars),
                media_type="text/event-stream",
                headers={REQUEST_ID_HEADER: rid, "cache-control": "no-cache"},
            )
        return response

    @router.post("/chat", response_model=None, summary="Ask a grounded question")
    def chat(payload: ChatRequest, request: Request) -> Any:
        """The RAG entrypoint: drives the agent state machine, answers with citations."""
        return _run(payload, request)

    @router.post("/query", response_model=None, include_in_schema=False)
    def query(payload: ChatRequest, request: Request) -> Any:
        """Alias for ``/chat``; both names are in the wild for a RAG endpoint."""
        return _run(payload, request)

    @router.get("/health", response_model=HealthResponse, summary="Liveness")
    def health() -> HealthResponse:
        return health_report(container)

    @router.get("/metrics", summary="Prometheus exposition")
    def metrics() -> Any:
        return Response(
            content=container.render_metrics(),
            media_type=_prometheus_content_type(),
        )

    return router


def _prometheus_content_type() -> str:
    try:
        from prometheus_client import CONTENT_TYPE_LATEST

        return str(CONTENT_TYPE_LATEST)
    except ImportError:
        return PROMETHEUS_CONTENT_TYPE


def install_error_handlers(app: Any) -> None:
    """Register the handlers that make :class:`~localmind.api.schemas.ErrorResponse` universal."""
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException

    def _request_id(request: Any) -> str:
        context = context_from_scope(request.scope)
        return context.request_id if context is not None else ""

    async def api_error_handler(request: Any, exc: Exception) -> Any:
        assert isinstance(exc, ApiError)
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(exc.code, exc.message, _request_id(request)),
            headers={**exc.headers, REQUEST_ID_HEADER: _request_id(request)},
        )

    async def http_error_handler(request: Any, exc: Exception) -> Any:
        status = getattr(exc, "status_code", 500)
        return JSONResponse(
            status_code=status,
            content=error_payload(
                status_to_code(status),
                GENERIC_MESSAGES.get(status, "request failed"),
                _request_id(request),
            ),
            headers={REQUEST_ID_HEADER: _request_id(request)},
        )

    async def validation_error_handler(request: Any, exc: Exception) -> Any:
        # Deliberately does not echo `exc.errors()`: those entries carry the offending input back
        # out, which is an easy way to reflect an attacker-controlled payload into a log sink.
        return JSONResponse(
            status_code=422,
            content=error_payload("invalid_request", GENERIC_MESSAGES[422], _request_id(request)),
            headers={REQUEST_ID_HEADER: _request_id(request)},
        )

    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
