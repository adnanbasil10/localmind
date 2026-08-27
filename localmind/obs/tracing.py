"""OpenTelemetry tracing, GenAI-semconv style, exported to Phoenix over OTLP.

implementation.md SS15: one trace per request, with child spans exactly
``route -> embed -> bm25 -> dense -> colbert -> fuse -> rerank -> grade -> generate -> verify``.
Each span carries duration (free from the span's own start/end), token counts, model name,
cache hit/miss, and retrieved doc IDs.

Import discipline: ``opentelemetry-*`` lives in the ``obs`` extra and is very likely NOT
installed in a bare checkout (CONVENTIONS.md / task brief). Every OTel name is therefore
imported lazily inside functions, never at module scope, and every public entry point degrades
to a documented no-op instead of raising when the extra is absent — instrumentation must never
crash an uninstrumented run.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from localmind.obs.semconv import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    LOCALMIND_CACHE_HIT,
    LOCALMIND_CACHE_LAYER,
    LOCALMIND_PIPELINE_STAGE,
    LOCALMIND_REQUEST_ID,
    LOCALMIND_RETRIEVAL_DOC_COUNT,
    LOCALMIND_RETRIEVAL_DOC_IDS,
    LOCALMIND_STAGE_DURATION_MS,
    PIPELINE_STAGES,
)

__all__ = [
    "LocalMindTracer",
    "RequestTrace",
    "StageSpan",
    "get_tracer",
    "init_tracing",
    "otel_available",
    "shutdown_tracing",
]

_INITIALIZED = False


def otel_available() -> bool:
    """Whether the ``opentelemetry`` package is importable in this environment."""
    try:
        import opentelemetry  # noqa: F401
    except ImportError:
        return False
    return True


def _is_initialized() -> bool:
    return _INITIALIZED


def init_tracing(
    service_name: str = "localmind-api",
    otlp_endpoint: str | None = None,
) -> bool:
    """Set up a global TracerProvider exporting to Phoenix over OTLP/gRPC.

    Returns ``True`` if tracing was actually initialized, ``False`` if
    ``opentelemetry-sdk`` / ``opentelemetry-exporter-otlp`` are not installed — in which case
    every tracer handed out by :func:`get_tracer` is a no-op and nothing is ever exported.
    Never raises for a missing optional dependency.

    ``otlp_endpoint`` matches the ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var the ``api`` service in
    docker-compose.yml sets for the ``obs``/``full`` profiles (``http://phoenix:4317``).
    """
    global _INITIALIZED
    if not otel_available():
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return False

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        except ImportError:
            OTLPSpanExporter = None
        if OTLPSpanExporter is not None:
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _INITIALIZED = True
    return True


def shutdown_tracing() -> None:
    """Flush and reset. Safe to call even if tracing was never initialized."""
    global _INITIALIZED
    if not _INITIALIZED or not otel_available():
        _INITIALIZED = False
        return
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        force_flush = getattr(provider, "force_flush", None)
        if callable(force_flush):
            force_flush()
    except ImportError:
        pass
    finally:
        _INITIALIZED = False


class _SpanHandle:
    """Uniform interface over a real OTel span or the no-op fallback.

    Every attribute set here is mirrored into a local dict regardless of backend, so tests (and
    any caller) can inspect ``.attributes`` without needing a live OTel SDK or a Phoenix
    container.
    """

    __slots__ = ("_local_attrs", "_otel_span", "name")

    def __init__(self, name: str, otel_span: Any = None) -> None:
        self.name = name
        self._otel_span = otel_span
        self._local_attrs: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self._local_attrs[key] = value
        if self._otel_span is not None:
            self._otel_span.set_attribute(key, value)

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def record_exception(self, exc: BaseException) -> None:
        self._local_attrs["exception.type"] = type(exc).__name__
        self._local_attrs["exception.message"] = str(exc)
        if self._otel_span is not None:
            self._otel_span.record_exception(exc)

    @property
    def attributes(self) -> Mapping[str, Any]:
        return dict(self._local_attrs)


class LocalMindTracer:
    """A tracer that is a real OTel tracer when available and a no-op otherwise.

    Callers never need to branch on ``otel_available()`` themselves: :meth:`start_span` always
    returns a :class:`_SpanHandle` with the same interface.
    """

    def __init__(self, name: str = "localmind") -> None:
        self._name = name
        self._otel_tracer: Any = None
        if otel_available() and _is_initialized():
            from opentelemetry import trace

            self._otel_tracer = trace.get_tracer(name)

    @property
    def enabled(self) -> bool:
        """Whether this tracer is backed by a real, initialized OTel SDK."""
        return self._otel_tracer is not None

    @contextlib.contextmanager
    def start_span(
        self, name: str, attributes: Mapping[str, Any] | None = None
    ) -> Iterator[_SpanHandle]:
        if self._otel_tracer is not None:
            with self._otel_tracer.start_as_current_span(name) as otel_span:
                handle = _SpanHandle(name, otel_span)
                if attributes:
                    handle.set_attributes(attributes)
                try:
                    yield handle
                except Exception as exc:
                    handle.record_exception(exc)
                    raise
        else:
            handle = _SpanHandle(name, None)
            if attributes:
                handle.set_attributes(attributes)
            try:
                yield handle
            except Exception as exc:
                handle.record_exception(exc)
                raise


def get_tracer(name: str = "localmind") -> LocalMindTracer:
    """Factory for a :class:`LocalMindTracer`. Cheap; call per-module or per-request as needed."""
    return LocalMindTracer(name)


class StageSpan:
    """Convenience setters over a pipeline-stage span's :class:`_SpanHandle`.

    Wraps the raw handle with GenAI-semconv-named setters so callers never write a bare
    attribute-name string literal; the names live in ``semconv.py``.
    """

    __slots__ = ("handle",)

    def __init__(self, handle: _SpanHandle) -> None:
        self.handle = handle

    @property
    def attributes(self) -> Mapping[str, Any]:
        return self.handle.attributes

    def set_model(self, name: str) -> StageSpan:
        self.handle.set_attribute(GEN_AI_REQUEST_MODEL, name)
        return self

    def set_tokens(
        self, input_tokens: int | None = None, output_tokens: int | None = None
    ) -> StageSpan:
        if input_tokens is not None:
            self.handle.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, int(input_tokens))
        if output_tokens is not None:
            self.handle.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, int(output_tokens))
        return self

    def set_cache(self, hit: bool, layer: str | None = None) -> StageSpan:
        self.handle.set_attribute(LOCALMIND_CACHE_HIT, bool(hit))
        if layer is not None:
            self.handle.set_attribute(LOCALMIND_CACHE_LAYER, layer)
        return self

    def set_doc_ids(self, doc_ids: Sequence[str]) -> StageSpan:
        ids = list(doc_ids)
        self.handle.set_attribute(LOCALMIND_RETRIEVAL_DOC_IDS, ids)
        self.handle.set_attribute(LOCALMIND_RETRIEVAL_DOC_COUNT, len(ids))
        return self

    def set_attribute(self, key: str, value: Any) -> StageSpan:
        self.handle.set_attribute(key, value)
        return self


class RequestTrace:
    """One trace per request: a root span plus exactly the ten pipeline child spans.

    ``.stage(name)`` enforces the exact, ordered pipeline shape CONVENTIONS.md / implementation.md
    SS15 mandate — ``route -> embed -> bm25 -> dense -> colbert -> fuse -> rerank -> grade ->
    generate -> verify`` — by raising on any stage name outside :data:`PIPELINE_STAGES`. A given
    request is free to skip stages that don't apply to it (e.g. a cached exact-response hit skips
    everything past ``route``), so repeats/omissions are allowed; unknown names are not.
    """

    def __init__(
        self, tracer: LocalMindTracer | None = None, request_id: str | None = None
    ) -> None:
        self.tracer = tracer if tracer is not None else get_tracer()
        self.request_id = request_id or uuid.uuid4().hex
        self._root_cm: contextlib.AbstractContextManager[_SpanHandle] | None = None
        self.root: _SpanHandle | None = None
        self._stages_seen: list[str] = []

    def __enter__(self) -> RequestTrace:
        self._root_cm = self.tracer.start_span(
            "localmind.request", {LOCALMIND_REQUEST_ID: self.request_id}
        )
        self.root = self._root_cm.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        assert self._root_cm is not None, "RequestTrace used outside a `with` block"
        return bool(self._root_cm.__exit__(exc_type, exc, tb))

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[StageSpan]:
        if name not in PIPELINE_STAGES:
            raise ValueError(f"unknown pipeline stage {name!r}; must be one of {PIPELINE_STAGES}")
        self._stages_seen.append(name)
        t0 = time.perf_counter()
        with self.tracer.start_span(name, {LOCALMIND_PIPELINE_STAGE: name}) as handle:
            stage_span = StageSpan(handle)
            try:
                yield stage_span
            finally:
                handle.set_attribute(
                    LOCALMIND_STAGE_DURATION_MS, (time.perf_counter() - t0) * 1000.0
                )

    @property
    def stages_seen(self) -> tuple[str, ...]:
        return tuple(self._stages_seen)
