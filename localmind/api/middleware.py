"""Cross-cutting request plumbing: request-ID propagation, the OpenTelemetry **root span**, and
the structured error envelope.

Written as raw ASGI, on purpose. Nothing in this module imports FastAPI or Starlette, so the
middleware can be exercised against a three-line fake ASGI app with only the base dependencies
installed — the same discipline ``localmind/inference/server.py`` applies to its wire format.

Where the root span comes from
------------------------------
:class:`RequestContextMiddleware` opens exactly one :class:`~localmind.obs.tracing.RequestTrace`
per HTTP request and holds it open for the whole call. Everything downstream — the route handler,
and through it the agent state machine and its injected seams — runs inside that span's context,
so any child span opened later (``RequestTrace.stage(...)``) nests under it. The trace is reachable
two ways, and both matter:

* ``scope[SCOPE_KEY]`` — for the route handler, which has the ASGI scope.
* :func:`current_context` (a :class:`~contextvars.ContextVar`) — for code with no access to the
  request at all, such as the injected retriever and generator wrappers in ``deps.py``. AnyIO
  copies the context into the worker thread it runs sync endpoints on, so this works from a
  ``def`` endpoint as well as an ``async def`` one.

``localmind.obs`` degrades to a no-op when ``opentelemetry`` is not installed, so none of this
requires an OTel collector — or the ``obs`` extra — to be present.

Error discipline
----------------
:func:`error_payload` is the only shape a non-2xx response ever has. The ``message`` is always a
fixed string from :data:`GENERIC_MESSAGES` or an :class:`ApiError` the gateway raised itself.
Exception text, tracebacks, DSNs, file paths and library names are recorded on the span and never
sent to the client.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from localmind.obs.semconv import LOCALMIND_REFUSAL, LOCALMIND_REQUEST_ID
from localmind.obs.tracing import RequestTrace

__all__ = [
    "API_KEY_HEADER",
    "GENERIC_MESSAGES",
    "HTTP_REQUEST_METHOD",
    "HTTP_RESPONSE_STATUS_CODE",
    "REQUEST_ID_HEADER",
    "SCOPE_KEY",
    "URL_PATH",
    "ApiError",
    "RequestContext",
    "RequestContextMiddleware",
    "context_from_scope",
    "current_context",
    "error_payload",
    "extract_request_id",
    "headers_from_scope",
    "new_request_id",
    "send_error",
    "use_context",
]

REQUEST_ID_HEADER = "x-request-id"
"""Propagated in *both* directions: honoured when the caller sends one, echoed on every response
(including error responses and SSE streams) so a client can correlate a complaint with a trace."""

API_KEY_HEADER = "x-api-key"

SCOPE_KEY = "localmind.request_context"
"""ASGI scope key. A plain dotted key rather than ``scope["state"]`` because the scope dict is
guaranteed to pass through every middleware unchanged, whereas ``state`` is framework-managed."""

# OpenTelemetry HTTP semantic conventions. `localmind/obs/semconv.py` is owned by the
# observability task and covers the GenAI + LocalMind *pipeline* attributes only; it has no HTTP
# names, and this task may not edit it. The three names the gateway needs therefore live here,
# spelled from the upstream spec, rather than as bare literals at their call sites.
HTTP_REQUEST_METHOD = "http.request.method"
URL_PATH = "url.path"
HTTP_RESPONSE_STATUS_CODE = "http.response.status_code"

GENERIC_MESSAGES: dict[int, str] = {
    400: "malformed request",
    401: "missing or invalid credentials",
    403: "not permitted",
    404: "no such endpoint",
    405: "method not allowed",
    422: "request failed validation",
    429: "rate limit exceeded",
    500: "internal server error",
    503: "service unavailable",
}
"""Fixed client-facing text. Deliberately uninformative: a message must not vary with internal
state, or it becomes an oracle."""

_SAFE_REQUEST_ID = re.compile(r"\A[A-Za-z0-9._:-]{1,128}\Z")

_CURRENT: contextvars.ContextVar[RequestContext | None] = contextvars.ContextVar(
    "localmind_request_context", default=None
)


class ApiError(Exception):
    """A deliberate, client-visible failure. Carries only what is safe to send back."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = int(status_code)
        self.code = code
        self.message = message or GENERIC_MESSAGES.get(self.status_code, "request failed")
        self.headers = dict(headers or {})
        super().__init__(self.message)


def new_request_id() -> str:
    return uuid.uuid4().hex


def extract_request_id(headers: Mapping[str, str], header: str = REQUEST_ID_HEADER) -> str | None:
    """Honour a caller-supplied request id, but only if it is safe to echo back.

    An id ends up in response headers, log lines and span attributes, so anything with a newline,
    a control character or unbounded length is rejected outright and a fresh id is minted instead.
    """
    raw = headers.get(header)
    if raw is None:
        return None
    candidate = raw.strip()
    return candidate if _SAFE_REQUEST_ID.match(candidate) else None


def headers_from_scope(scope: Mapping[str, Any]) -> dict[str, str]:
    """Lower-cased, last-wins view of the ASGI raw header list."""
    out: dict[str, str] = {}
    for key, value in scope.get("headers") or []:
        out[bytes(key).decode("latin-1").lower()] = bytes(value).decode("latin-1")
    return out


@dataclass
class RequestContext:
    """Everything the gateway learns about a request before the handler runs."""

    request_id: str
    method: str = "GET"
    path: str = "/"
    principal_id: str = "anonymous"
    started_at: float = field(default_factory=time.perf_counter)
    trace: RequestTrace | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    """Retrieval filters for this request. They live on the context rather than in the call chain
    because ``Agent.run`` takes no filters argument and its ``retrieve`` node calls
    ``search_documents`` with ``{query, k}`` only — so ``deps.TracingToolRegistry`` reads them
    from here and merges them into the tool's arguments."""

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0

    def set_root_attribute(self, key: str, value: Any) -> None:
        """Annotate the root span, if tracing produced one. Never raises."""
        if self.trace is not None and self.trace.root is not None:
            self.trace.root.set_attribute(key, value)


def current_context() -> RequestContext | None:
    """The context of the request being served on this task/thread, or ``None`` outside one."""
    return _CURRENT.get()


def context_from_scope(scope: Mapping[str, Any]) -> RequestContext | None:
    value = scope.get(SCOPE_KEY)
    return value if isinstance(value, RequestContext) else None


@contextlib.contextmanager
def use_context(context: RequestContext) -> Iterator[RequestContext]:
    """Bind ``context`` for the duration of the block.

    The middleware does this for real requests; this is the seam that lets the framework-free core
    (and its tests) run the same code path with no ASGI server anywhere.
    """
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


def error_payload(code: str, message: str, request_id: str = "") -> dict[str, Any]:
    """The one and only error body shape. Mirrors ``schemas.ErrorResponse``."""
    return {"error": {"code": code, "message": message, "request_id": request_id}}


async def send_error(
    send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    headers: Mapping[str, str] | None = None,
) -> None:
    """Emit an error envelope directly at the ASGI level (no framework involved)."""
    body = json.dumps(error_payload(code, message, request_id)).encode()
    raw = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        (REQUEST_ID_HEADER.encode(), request_id.encode()),
    ]
    for name, value in (headers or {}).items():
        raw.append((name.lower().encode("latin-1"), str(value).encode("latin-1")))
    await send({"type": "http.response.start", "status": status_code, "headers": raw})
    await send({"type": "http.response.body", "body": body})


class RequestContextMiddleware:
    """Raw-ASGI middleware: request id in and out, root span around everything, safe 500s.

    ``metrics`` is any object with the ``observe_request(method, path, status, seconds)`` and
    ``record_exception(method, path, exc_type)`` methods of ``deps.HttpMetrics``; it is optional so
    the middleware can be tested on its own.
    """

    def __init__(
        self,
        app: Any,
        *,
        metrics: Any = None,
        header: str = REQUEST_ID_HEADER,
    ) -> None:
        self.app = app
        self.metrics = metrics
        self.header = header

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = headers_from_scope(scope)
        request_id = extract_request_id(headers, self.header) or new_request_id()
        method = str(scope.get("method", "GET"))
        path = str(scope.get("path", "/"))
        started = time.perf_counter()
        status_holder = {"status": 500, "started": False}

        async def wrapped_send(message: MutableMapping[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                status_holder["started"] = True
                status_holder["status"] = int(message.get("status", 200))
                raw = [
                    (k, v)
                    for k, v in list(message.get("headers") or [])
                    if bytes(k).decode("latin-1").lower() != self.header
                ]
                raw.append((self.header.encode(), request_id.encode()))
                message = {**message, "headers": raw}
            await send(message)

        with RequestTrace(request_id=request_id) as trace:
            context = RequestContext(
                request_id=request_id,
                method=method,
                path=path,
                started_at=started,
                trace=trace,
            )
            scope[SCOPE_KEY] = context
            context.set_root_attribute(LOCALMIND_REQUEST_ID, request_id)
            context.set_root_attribute(HTTP_REQUEST_METHOD, method)
            context.set_root_attribute(URL_PATH, path)
            token = _CURRENT.set(context)
            try:
                try:
                    await self.app(scope, receive, wrapped_send)
                except ApiError as exc:
                    status_holder["status"] = exc.status_code
                    if status_holder["started"]:
                        raise
                    await send_error(
                        send, exc.status_code, exc.code, exc.message, request_id, exc.headers
                    )
                except Exception as exc:
                    # The only place an unexpected exception is allowed to stop. Details go to the
                    # span; the client is told nothing but "internal server error" + its id.
                    status_holder["status"] = 500
                    if trace.root is not None:
                        trace.root.record_exception(exc)
                    if self.metrics is not None:
                        self.metrics.record_exception(method, path, type(exc).__name__)
                    if status_holder["started"]:
                        raise
                    await send_error(send, 500, "internal", GENERIC_MESSAGES[500], request_id)
            finally:
                _CURRENT.reset(token)
                elapsed = time.perf_counter() - started
                context.set_root_attribute(HTTP_RESPONSE_STATUS_CODE, status_holder["status"])
                if status_holder["status"] >= 400:
                    context.set_root_attribute(LOCALMIND_REFUSAL, True)
                if self.metrics is not None:
                    self.metrics.observe_request(
                        method, path, int(status_holder["status"]), elapsed
                    )
