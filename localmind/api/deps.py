"""Dependency injection for the RAG gateway — every external thing behind a seam, with a fake.

Nothing here imports FastAPI. The container, the settings, the auth check, the rate limiter, the
adapters and the caches are all plain Python, so the whole service can be assembled and driven in
a test with only the base dependencies installed. ``routes.py`` is the only module that knows a
web framework exists.

What this module owns
---------------------
* :class:`Settings` — configuration, from the environment ``docker-compose.yml`` already sets.
* :class:`ApiKeyAuth` / :class:`Principal` — bearer or ``X-API-Key`` credentials.
* :class:`RateLimitPolicy` — a thin policy over ``agent.guardrails.RateLimiter`` (the token bucket
  already written for the agent; not a second implementation of one).
* :class:`ScoredDocRetriever` — **the retrieval adapter**, see its docstring; the one place the
  ``(doc_id, score)``/``text`` mismatch between ``localmind.retrieval`` and ``localmind.agent``
  is bridged.
* :class:`TracingToolRegistry` / :class:`TracedGenerator` — the two seams that turn agent work
  into *live* child spans of the request root span, and feed Prometheus. The registry, not a
  retriever wrapper, is where per-request state has to be injected; its docstring says why.
* :class:`OllamaGenerator` — the production generator (``httpx`` imported lazily, no import-time
  I/O), and :class:`ExtractiveGenerator`, an offline stand-in that is explicitly not a model.
* :class:`Container` — the assembled service, built by :func:`build_container`.

Offline guarantee: constructing a :class:`Container` performs no network I/O and requires no
Postgres, Redis, Ollama or OTel collector. Every one of those degrades to a documented fallback,
which is what lets ``GET /health`` answer 200 on a cold, serviceless machine.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict

from localmind.agent.graph import Agent
from localmind.agent.guardrails import RateLimiter
from localmind.agent.state import (
    Budget,
    Clock,
    ControlPlane,
    GenerationResult,
    RetrievedChunk,
    ToolResult,
    WallClock,
)
from localmind.agent.tools import ToolRegistry
from localmind.agent.tools.search_documents import coerce_chunk
from localmind.api.middleware import ApiError, current_context
from localmind.cache import (
    CacheBackend,
    EmbeddingCache,
    HashingEmbedder,
    InMemoryCacheBackend,
    ResponseCache,
    SemanticCache,
)
from localmind.obs import PIPELINE_STAGES, Metrics, otel_available, prometheus_available

__all__ = [
    "ApiKeyAuth",
    "CachedEmbedder",
    "Container",
    "ExtractiveGenerator",
    "HttpMetrics",
    "OllamaGenerator",
    "Principal",
    "RateLimitPolicy",
    "RetrievalArmLike",
    "ScoredDocRetriever",
    "Settings",
    "TracedGenerator",
    "TracingToolRegistry",
    "build_container",
    "build_generator",
    "key_id",
]

_ENV_PREFIX = "LOCALMIND_"


# ======================================================================================
# Settings
# ======================================================================================
def _env(env: Mapping[str, str], name: str, default: str | None = None) -> str | None:
    value = env.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _env(env, name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _env(env, name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _env(env, name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_tuple(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    raw = _env(env, name)
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


class Settings(BaseModel):
    """Gateway configuration. Every field is env-overridable so the container needs no argparse.

    The env var names match what ``docker-compose.yml`` already puts on the ``api`` service
    (``LOCALMIND_PG_DSN``, ``LOCALMIND_REDIS_URL``, ``OTEL_EXPORTER_OTLP_ENDPOINT``); the rest use
    the ``LOCALMIND_`` prefix.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    service_name: str = "localmind-api"
    version: str = "0.1.0"

    api_keys: tuple[str, ...] = ()
    """Accepted credentials. **Empty means authentication is disabled** — that is what makes the
    Hugging Face Space demo and ``just up`` work out of the box. Set ``LOCALMIND_API_KEYS`` to
    turn it on; there is no half-open state."""

    rate_limit_per_s: float = 5.0
    rate_limit_burst: float = 30.0

    otlp_endpoint: str | None = None
    redis_url: str | None = None
    pg_dsn: str | None = None

    generator: str = "ollama"
    """``ollama`` | ``extractive`` | ``none``."""
    ollama_endpoint: str = "http://localhost:11434/api/generate"
    ollama_model: str = "qwen3:4b"
    ollama_timeout_s: float = 120.0

    response_cache_ttl_s: int = 3600
    semantic_cache_enabled: bool = True
    semantic_cache_tau: float = 0.92

    top_k: int = 5
    max_wall_clock_s: float = 30.0
    stream_chunk_chars: int = 48

    cors_origins: tuple[str, ...] = ()
    docs_enabled: bool = True

    @property
    def auth_required(self) -> bool:
        return bool(self.api_keys)

    @property
    def rate_limit_enabled(self) -> bool:
        return self.rate_limit_per_s > 0 and self.rate_limit_burst > 0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = dict(os.environ if env is None else env)
        p = _ENV_PREFIX
        return cls(
            service_name=_env(e, f"{p}SERVICE_NAME", "localmind-api") or "localmind-api",
            version=_env(e, f"{p}VERSION", "0.1.0") or "0.1.0",
            api_keys=_env_tuple(e, f"{p}API_KEYS"),
            rate_limit_per_s=_env_float(e, f"{p}RATE_LIMIT_PER_S", 5.0),
            rate_limit_burst=_env_float(e, f"{p}RATE_LIMIT_BURST", 30.0),
            otlp_endpoint=_env(e, "OTEL_EXPORTER_OTLP_ENDPOINT"),
            redis_url=_env(e, f"{p}REDIS_URL"),
            pg_dsn=_env(e, f"{p}PG_DSN"),
            generator=(_env(e, f"{p}GENERATOR", "ollama") or "ollama").lower(),
            ollama_endpoint=_env(e, f"{p}OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
            or "http://localhost:11434/api/generate",
            ollama_model=_env(e, f"{p}OLLAMA_MODEL", "qwen3:4b") or "qwen3:4b",
            ollama_timeout_s=_env_float(e, f"{p}OLLAMA_TIMEOUT_S", 120.0),
            response_cache_ttl_s=_env_int(e, f"{p}RESPONSE_CACHE_TTL_S", 3600),
            semantic_cache_enabled=_env_bool(e, f"{p}SEMANTIC_CACHE", True),
            semantic_cache_tau=_env_float(e, f"{p}SEMANTIC_CACHE_TAU", 0.92),
            top_k=_env_int(e, f"{p}TOP_K", 5),
            max_wall_clock_s=_env_float(e, f"{p}MAX_WALL_CLOCK_S", 30.0),
            stream_chunk_chars=_env_int(e, f"{p}STREAM_CHUNK_CHARS", 48),
            cors_origins=_env_tuple(e, f"{p}CORS_ORIGINS"),
            docs_enabled=_env_bool(e, f"{p}DOCS", True),
        )


# ======================================================================================
# Auth
# ======================================================================================
@dataclass(frozen=True)
class Principal:
    """Who is calling. ``key_id`` is a digest prefix — the raw key is never stored or logged."""

    key_id: str = "anonymous"
    authenticated: bool = False


def key_id(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()[:12]


class ApiKeyAuth:
    """API key or bearer token, compared in constant time.

    Both header forms are accepted because both are in the wild: ``Authorization: Bearer <key>``
    for anything OpenAI-shaped, ``X-API-Key: <key>`` for everything else. With no keys configured
    the gateway is open and every caller is the same anonymous principal — see
    :attr:`Settings.api_keys`.
    """

    def __init__(self, keys: Sequence[str] = ()) -> None:
        self._keys = tuple(k for k in keys if k)

    @property
    def required(self) -> bool:
        return bool(self._keys)

    def identify(self, authorization: str | None, api_key: str | None) -> Principal:
        """Return the caller's principal, or raise :class:`ApiError` 401."""
        presented = self._presented(authorization, api_key)
        if not self.required:
            return Principal(key_id=key_id(presented) if presented else "anonymous")
        if presented is None:
            raise ApiError(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})
        # Compare against every key, without short-circuiting, so timing leaks nothing about
        # which key matched or how many are configured.
        matched = False
        for candidate in self._keys:
            matched |= hmac.compare_digest(candidate, presented)
        if not matched:
            raise ApiError(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})
        return Principal(key_id=key_id(presented), authenticated=True)

    @staticmethod
    def _presented(authorization: str | None, api_key: str | None) -> str | None:
        if authorization:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() == "bearer" and token.strip():
                return token.strip()
        if api_key and api_key.strip():
            return api_key.strip()
        return None


class RateLimitPolicy:
    """Per-principal token bucket. Wraps ``agent.guardrails.RateLimiter`` rather than repeating it.

    Keyed on the principal (an API key digest, or the peer address for anonymous callers) so one
    noisy client cannot starve the rest, and so the limit is not trivially reset by reconnecting.
    """

    def __init__(
        self,
        per_second: float = 5.0,
        burst: float = 30.0,
        *,
        clock: Clock | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled and per_second > 0 and burst > 0
        self.per_second = per_second
        self.burst = burst
        self._limiter = RateLimiter(capacity=burst, refill_per_s=per_second, clock=clock)

    def check(self, key: str) -> None:
        """Raise :class:`ApiError` 429 with a ``Retry-After`` when the bucket is empty."""
        if not self.enabled:
            return
        decision = self._limiter.check(key)
        if decision.allowed:
            return
        retry = max(1, int(decision.retry_after_s + 0.999)) if decision.retry_after_s else 1
        raise ApiError(429, "rate_limited", headers={"Retry-After": str(retry)})

    def describe(self) -> str:
        if not self.enabled:
            return "disabled"
        return f"{self.per_second:g}/s burst {self.burst:g}"


# ======================================================================================
# Gateway-level Prometheus metrics
# ======================================================================================
class HttpMetrics:
    """RED metrics for the HTTP surface: request counts by status, and request duration.

    ``localmind.obs.metrics.Metrics`` is deliberately *pipeline*-scoped — its latency histogram
    validates the ``stage`` label against :data:`~localmind.obs.semconv.PIPELINE_STAGES`, so an
    HTTP request has no legal home in it. Rather than edit a package this task does not own, the
    two HTTP series are defined here and rendered alongside it by
    :meth:`Container.render_metrics`. Same lazy-import discipline as ``obs``: without
    ``prometheus_client`` every call still records into an inspectable in-memory snapshot.

    The ``path`` label is restricted to :data:`KNOWN_PATHS` so a scanner hitting random URLs
    cannot blow up label cardinality.
    """

    KNOWN_PATHS: frozenset[str] = frozenset({"/chat", "/query", "/health", "/metrics"})

    def __init__(self, registry: Any = None) -> None:
        self.requests_total: dict[tuple[str, str, str], int] = {}
        self.duration_seconds: dict[tuple[str, str], list[float]] = {}
        self.exceptions_total: dict[tuple[str, str, str], int] = {}
        self._registry: Any = None
        self._counter: Any = None
        self._hist: Any = None
        self._exceptions: Any = None
        if prometheus_available():
            import prometheus_client as pc

            self._registry = registry if registry is not None else pc.CollectorRegistry()
            self._counter = pc.Counter(
                "localmind_http_requests_total",
                "Gateway HTTP requests",
                labelnames=["method", "path", "status"],
                registry=self._registry,
            )
            self._hist = pc.Histogram(
                "localmind_http_request_duration_seconds",
                "Gateway HTTP request duration",
                labelnames=["method", "path"],
                registry=self._registry,
            )
            self._exceptions = pc.Counter(
                "localmind_http_exceptions_total",
                "Unhandled exceptions escaping a route handler",
                labelnames=["method", "path", "exception_type"],
                registry=self._registry,
            )

    @property
    def enabled(self) -> bool:
        return self._registry is not None

    def _path(self, path: str) -> str:
        return path if path in self.KNOWN_PATHS else "other"

    def observe_request(self, method: str, path: str, status: int, seconds: float) -> None:
        label_path = self._path(path)
        key = (method, label_path, str(status))
        self.requests_total[key] = self.requests_total.get(key, 0) + 1
        self.duration_seconds.setdefault((method, label_path), []).append(float(seconds))
        if self._counter is not None and self._hist is not None:
            self._counter.labels(method=method, path=label_path, status=str(status)).inc()
            self._hist.labels(method=method, path=label_path).observe(float(seconds))

    def record_exception(self, method: str, path: str, exception_type: str) -> None:
        label_path = self._path(path)
        key = (method, label_path, exception_type)
        self.exceptions_total[key] = self.exceptions_total.get(key, 0) + 1
        if self._exceptions is not None:
            self._exceptions.labels(
                method=method, path=label_path, exception_type=exception_type
            ).inc()

    def render(self) -> bytes:
        if self._registry is None:
            return b""
        import prometheus_client as pc

        return pc.generate_latest(self._registry)

    def snapshot(self) -> dict[str, Any]:
        return {
            "requests_total": {"|".join(k): v for k, v in self.requests_total.items()},
            "duration_seconds": {"|".join(k): list(v) for k, v in self.duration_seconds.items()},
            "exceptions_total": {"|".join(k): v for k, v in self.exceptions_total.items()},
        }


# ======================================================================================
# The retrieval adapter — the one real impedance mismatch in this integration
# ======================================================================================
@runtime_checkable
class RetrievalArmLike(Protocol):
    """Structural view of ``localmind.retrieval.RetrievalArm`` — declared, not imported.

    Keeping it structural means the gateway works with any of BM25 / SPLADE / dense / ColBERT, or
    with a fused-and-reranked pipeline object, without the API package taking a hard dependency on
    the retrieval package's class hierarchy.
    """

    def search(self, query: str, top_k: int = 10) -> Sequence[Any]: ...


def _field_of(value: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off a mapping or an object, whichever it is."""
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _entry_fields(value: Any) -> dict[str, Any]:
    """Normalise one corpus entry (a ``Document``, a mapping, or a bare string)."""
    if isinstance(value, str):
        return {"text": value, "metadata": {}, "uri": ""}
    metadata = _field_of(value, "metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "text": str(_field_of(value, "text", "") or ""),
        "metadata": dict(metadata),
        "uri": str(_field_of(value, "uri", "") or metadata.get("uri", "") or ""),
    }


class ScoredDocRetriever:
    """Adapter: a ``localmind.retrieval`` arm -> the agent's ``Retriever`` Protocol.

    **This is the integration seam the two packages deliberately leave open**, and it lives here
    because neither of them may be bent to suit the other:

    1. *Payload.* ``localmind.retrieval.ScoredDoc`` is ``(doc_id, score)`` and nothing else — by
       design, since fusion consumes ranks, not text (``docs/architecture.md``, "Retrieval arms are
       independent"). The agent's ``Retriever`` contract needs ``text`` on every hit, because the
       grader, the injection classifier and the prompt builder all read it. The text is re-joined
       here, from the same corpus that was indexed, and then normalised through the agent
       package's own ``coerce_chunk`` — so the shape of a ``RetrievedChunk`` still has exactly one
       definition, in the agent package.
    2. *Signature.* ``RetrievalArm.search(query, top_k)`` versus
       ``Retriever.search(query, k, filters)``. The arm is called positionally so the parameter
       name never has to match, and ``filters`` — which retrieval arms do not support at all — is
       applied here as a post-filter over corpus metadata, with ``oversample`` extra candidates
       requested so filtering does not silently shrink the result set.

    A ``doc_id`` the arm returns that is not in ``corpus`` is dropped rather than emitted with
    empty text: an uncitable chunk is worse than a missing one.
    """

    def __init__(
        self,
        arm: RetrievalArmLike,
        corpus: Mapping[str, Any] | Sequence[Any],
        *,
        oversample: int = 4,
    ) -> None:
        self.arm = arm
        self.oversample = max(1, int(oversample))
        self._corpus: dict[str, dict[str, Any]] = {}
        if isinstance(corpus, Mapping):
            for doc_id, value in corpus.items():
                self._corpus[str(doc_id)] = _entry_fields(value)
        else:
            for value in corpus:
                doc_id = getattr(value, "doc_id", None)
                if doc_id is None and isinstance(value, Mapping):
                    doc_id = value.get("doc_id")
                if doc_id is None:
                    continue
                self._corpus[str(doc_id)] = _entry_fields(value)

    def __len__(self) -> int:
        return len(self._corpus)

    @staticmethod
    def _matches(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
        for key, wanted in filters.items():
            actual = metadata.get(key)
            if isinstance(wanted, (list, tuple, set, frozenset)):
                if not any(str(actual) == str(w) for w in wanted):
                    return False
            elif str(actual) != str(wanted):
                return False
        return True

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        want = max(1, int(k))
        ask = want * self.oversample if filters else want
        hits = self.arm.search(query, ask) or []
        out: list[RetrievedChunk] = []
        for index, hit in enumerate(hits):
            doc_id = hit.get("doc_id") if isinstance(hit, Mapping) else getattr(hit, "doc_id", None)
            if doc_id is None:
                continue
            entry = self._corpus.get(str(doc_id))
            if entry is None:
                continue
            if filters and not self._matches(entry["metadata"], filters):
                continue
            score = hit.get("score") if isinstance(hit, Mapping) else getattr(hit, "score", 0.0)
            out.append(
                coerce_chunk(
                    {
                        # One indexed document is one retrievable chunk in this pipeline, so the
                        # chunk id is the doc id. Chunk-level indexing simply produces chunk-level
                        # doc_ids upstream; nothing here has to change for that.
                        "chunk_id": str(doc_id),
                        "doc_id": str(doc_id),
                        "text": entry["text"],
                        "score": float(score or 0.0),
                        "uri": entry["uri"],
                        "metadata": entry["metadata"],
                    },
                    index,
                )
            )
            if len(out) >= want:
                break
        return out


class TracingToolRegistry(ToolRegistry):
    """Where per-request state meets the agent's tool execution — and it has to be *here*.

    ``agent.tools.base.call_with_timeout`` runs every tool on a fresh daemon thread, and a new
    thread starts with an empty :mod:`contextvars` context. Anything the gateway binds per request
    — the trace, the filters — is therefore invisible from inside a tool.
    ``ToolRegistry.call`` is the last frame that still runs on the request's own thread, so this
    subclass is the correct seam, and a wrapper around the retriever would silently be a no-op.

    Two things happen here:

    * the request's metadata filters are merged into ``search_documents``' arguments.
      ``Agent.run`` has no filters parameter and its ``retrieve`` node calls the tool with
      ``{query, k}`` only — but ``SearchDocumentsArgs`` already declares ``filters``, so this
      needs no new contract on either side, just the value delivered to it.
    * the retrieval stage span is opened around the call, making it a genuine child of the request
      root span with a *measured* duration and the retrieved doc ids attached.

    ``stage`` names the pipeline stage this retriever represents: ``"bm25"``/``"dense"``/
    ``"colbert"`` for a single arm, the default ``"fuse"`` for an already-fused hybrid.
    """

    SEARCH_TOOL = "search_documents"

    def __init__(
        self,
        tools: Sequence[Any] = (),
        *,
        stage: str = "fuse",
        metrics: Metrics | None = None,
    ) -> None:
        if stage not in PIPELINE_STAGES:
            raise ValueError(f"unknown pipeline stage {stage!r}; must be one of {PIPELINE_STAGES}")
        super().__init__(tools)
        self.stage = stage
        self.metrics = metrics

    @classmethod
    def wrapping(
        cls,
        registry: ToolRegistry,
        *,
        stage: str = "fuse",
        metrics: Metrics | None = None,
    ) -> TracingToolRegistry:
        """Re-home an existing registry's tools, through its public accessors only."""
        tools = [registry.get(name) for name in registry.names()]
        return cls([t for t in tools if t is not None], stage=stage, metrics=metrics)

    @staticmethod
    def _doc_ids(result: ToolResult) -> list[str]:
        if not result.ok:
            return []
        chunks = result.data.get("chunks") or []
        return [str(c.get("doc_id", "")) for c in chunks if isinstance(c, Mapping)]

    def call(
        self,
        name: str,
        args: Mapping[str, Any] | None = None,
        *,
        allowlist: Collection[str] | None = None,
    ) -> ToolResult:
        context = current_context()
        merged = dict(args or {})
        if name != self.SEARCH_TOOL:
            return super().call(name, merged, allowlist=allowlist)
        if context is not None and context.filters and not merged.get("filters"):
            merged["filters"] = dict(context.filters)

        t0 = time.perf_counter()
        if context is None or context.trace is None:
            result = super().call(name, merged, allowlist=allowlist)
        else:
            with context.trace.stage(self.stage) as span:
                result = super().call(name, merged, allowlist=allowlist)
                span.set_doc_ids([d for d in self._doc_ids(result) if d])
        if self.metrics is not None:
            self.metrics.observe_stage_latency(self.stage, time.perf_counter() - t0)
        return result


# ======================================================================================
# Generators
# ======================================================================================
@dataclass
class OllamaGenerator:
    """The production ``Generator``: the local Ollama server (§ architecture, "Answer generation").

    ``httpx`` is imported lazily and the endpoint is only touched inside :meth:`generate`, so
    importing or constructing this class performs no I/O — CONVENTIONS.md forbids network calls at
    import time, and ``GET /health`` must answer without Ollama running. When Ollama is absent the
    exception propagates into ``Agent._generate``, which turns it into a refusal rather than a
    crash.
    """

    model: str = "qwen3:4b"
    endpoint: str = "http://localhost:11434/api/generate"
    timeout_s: float = 120.0

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult:
        import httpx

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if stop:
            payload["options"]["stop"] = list(stop)
        response = httpx.post(self.endpoint, json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        body = response.json()
        return GenerationResult(
            text=str(body.get("response", "")),
            prompt_tokens=int(body.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(body.get("eval_count", 0) or 0),
            model=str(body.get("model", self.model)),
        )


class ExtractiveGenerator:
    """An offline stand-in for Ollama. **It is not a language model.**

    It quotes the best-scoring sentence of the top source and cites it, which is enough to drive
    the whole gateway — grading, citation extraction, claim verification, streaming — end to end
    with no model server anywhere. Its ``model`` field says ``extractive-stub`` so no benchmark
    can ever mistake its output for generation quality (CONVENTIONS.md rule 5).
    """

    name = "extractive-stub"

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult:
        del max_tokens, temperature, stop
        source = self._first_source(prompt)
        if not source:
            return GenerationResult(text="INSUFFICIENT_EVIDENCE", model=self.name)
        return GenerationResult(
            text=f"{source} [1]",
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(source) // 4),
            model=self.name,
        )

    @staticmethod
    def _first_source(prompt: str) -> str:
        """Pull the first source block out of the rendered context, without parsing the prompt."""
        marker = "SOURCES\n"
        start = prompt.find(marker)
        if start < 0:
            return ""
        body = prompt[start + len(marker) :]
        for line in body.splitlines():
            text = line.strip()
            if len(text) >= 24 and not text.startswith(("[", "<", "USER", "ANSWER")):
                return text[:600]
        return ""


class TracedGenerator:
    """Wraps a ``Generator`` so ``generate`` is a live child span carrying GenAI semconv tokens."""

    def __init__(self, inner: Any, *, metrics: Metrics | None = None) -> None:
        self.inner = inner
        self.metrics = metrics

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult:
        context = current_context()
        t0 = time.perf_counter()
        if context is None or context.trace is None:
            result = self.inner.generate(
                prompt, max_tokens=max_tokens, temperature=temperature, stop=stop
            )
        else:
            with context.trace.stage("generate") as span:
                result = self.inner.generate(
                    prompt, max_tokens=max_tokens, temperature=temperature, stop=stop
                )
                span.set_model(result.model).set_tokens(
                    input_tokens=result.prompt_tokens, output_tokens=result.completion_tokens
                )
        if self.metrics is not None:
            self.metrics.observe_stage_latency("generate", time.perf_counter() - t0)
            self.metrics.record_tokens(
                "generate",
                result.model,
                input_tokens=result.prompt_tokens,
                output_tokens=result.completion_tokens,
            )
        return result


# ======================================================================================
# Caches
# ======================================================================================
class CachedEmbedder:
    """Layer 1 in front of the semantic cache's embedder.

    The semantic cache embeds every incoming query; the embedding cache makes the repeat cost of
    that zero. Wiring them together here is the only place the two layers meet, and it keeps
    ``EmbeddingCache``'s "embedding calls avoided" number honest — every avoided call is a real
    one this wrapper did not make.
    """

    def __init__(self, inner: Any, cache: EmbeddingCache) -> None:
        self.inner = inner
        self.cache = cache

    def embed(self, text: str) -> np.ndarray:
        vector, _hit = self.cache.get_or_compute(text, self.inner.embed)
        return vector


def _make_backend(redis_url: str | None) -> tuple[CacheBackend, str]:
    """Redis when configured *and* importable, in-memory otherwise. Never raises, never connects.

    ``redis.Redis.from_url`` is lazy, so this does no I/O; a Redis that is configured but down
    surfaces as a cache miss at call time, not as a failed startup — ``/health`` must not depend
    on it.
    """
    if not redis_url:
        return InMemoryCacheBackend(), "in-memory"
    try:
        from localmind.cache import RedisCacheBackend

        return RedisCacheBackend(redis_url), "redis"
    except Exception:
        return InMemoryCacheBackend(), "in-memory (redis unavailable)"


# ======================================================================================
# Container
# ======================================================================================
@dataclass
class Container:
    """The assembled service. One per process; built once by ``main.create_app``."""

    settings: Settings
    agent: Agent
    metrics: Metrics
    http_metrics: HttpMetrics
    auth: ApiKeyAuth
    rate_limiter: RateLimitPolicy
    response_cache: ResponseCache
    embedding_cache: EmbeddingCache
    semantic_cache: SemanticCache | None = None
    started_at: float = field(default_factory=time.monotonic)
    tracing_enabled: bool = False
    generator_name: str = "none"
    retriever_name: str = "none"
    cache_backend_name: str = "in-memory"

    @property
    def uptime_s(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def budget_for(self, top_k: int | None = None) -> Budget:
        return Budget(
            top_k=int(top_k or self.settings.top_k),
            max_wall_clock_s=self.settings.max_wall_clock_s,
        )

    def cache_config(self, top_k: int | None = None) -> dict[str, Any]:
        """The ``config`` half of the Layer-2 cache key: anything that changes the answer."""
        return {
            "top_k": int(top_k or self.settings.top_k),
            "generator": self.generator_name,
            "retriever": self.retriever_name,
            "version": self.settings.version,
        }

    def render_metrics(self) -> bytes:
        """Prometheus exposition for ``GET /metrics``: the pipeline series plus the HTTP series."""
        return self.metrics.render() + self.http_metrics.render()

    def health(self) -> dict[str, str]:
        """Configuration-only health. Performs no I/O, so it is true with nothing else running."""
        return {
            "agent": "ready",
            "generator": self.generator_name,
            "retriever": self.retriever_name,
            "auth": "api-key" if self.auth.required else "disabled",
            "rate_limit": self.rate_limiter.describe(),
            "tracing": (
                "otlp"
                if self.tracing_enabled
                else ("no-op (not initialised)" if otel_available() else "no-op (otel absent)")
            ),
            "metrics": "prometheus" if self.metrics.enabled else "in-memory (prometheus absent)",
            "cache_backend": self.cache_backend_name,
            "semantic_cache": (
                f"tau={self.settings.semantic_cache_tau:g}"
                if self.semantic_cache is not None
                else "disabled"
            ),
            "postgres": "configured" if self.settings.pg_dsn else "not configured",
            "redis": "configured" if self.settings.redis_url else "not configured",
        }


def build_generator(settings: Settings) -> tuple[Any, str]:
    if settings.generator == "ollama":
        return (
            OllamaGenerator(
                model=settings.ollama_model,
                endpoint=settings.ollama_endpoint,
                timeout_s=settings.ollama_timeout_s,
            ),
            f"ollama:{settings.ollama_model}",
        )
    if settings.generator == "extractive":
        return ExtractiveGenerator(), ExtractiveGenerator.name
    return None, "none"


def build_container(
    settings: Settings | None = None,
    *,
    agent: Agent | None = None,
    retriever: Any = None,
    retriever_stage: str = "fuse",
    generator: Any = None,
    control_plane: ControlPlane | None = None,
    metrics: Metrics | None = None,
    http_metrics: HttpMetrics | None = None,
    cache_backend: CacheBackend | None = None,
    clock: Clock | None = None,
    tracing_enabled: bool = False,
) -> Container:
    """Assemble the service. Every seam is overridable, which is how the tests inject fakes.

    Pass ``agent=`` to supply a fully built :class:`~localmind.agent.graph.Agent`; otherwise one is
    constructed from ``retriever``/``generator``/``control_plane``, each of which may be ``None``
    (the agent falls back to its heuristic router/grader/rewriter and refuses cleanly if it has no
    generator). Nothing here connects to anything.
    """
    settings = settings or Settings.from_env()
    metrics = metrics if metrics is not None else Metrics()
    http_metrics = http_metrics if http_metrics is not None else HttpMetrics()
    clock = clock or WallClock()

    generator_name = "none"
    retriever_name = "none"
    if agent is None:
        if generator is None:
            generator, generator_name = build_generator(settings)
        else:
            generator_name = getattr(generator, "name", type(generator).__name__)
        if generator is not None:
            generator = TracedGenerator(generator, metrics=metrics)
        if retriever is not None:
            retriever_name = type(retriever).__name__
        agent = Agent(
            control_plane=control_plane,
            generator=generator,
            retriever=retriever,
            budget=Budget(top_k=settings.top_k, max_wall_clock_s=settings.max_wall_clock_s),
            clock=clock,
        )
    else:
        generator_name = (
            "none" if agent.generator is None else getattr(agent.generator, "name", "attached")
        )
        retriever_name = "attached"

    if not isinstance(agent.registry, TracingToolRegistry):
        # Re-home the tools into the tracing registry. Done after construction because `Agent`
        # builds the default registry itself, wiring in the injection classifier it derived from
        # the control plane -- which is not something this layer should have to reproduce.
        agent.registry = TracingToolRegistry.wrapping(
            agent.registry, stage=retriever_stage, metrics=metrics
        )

    if cache_backend is not None:
        backend, backend_name = cache_backend, "injected"
    else:
        backend, backend_name = _make_backend(settings.redis_url)

    embedding_cache = EmbeddingCache(
        backend, model="query-hash", ttl_seconds=settings.response_cache_ttl_s
    )
    semantic: SemanticCache | None = None
    if settings.semantic_cache_enabled:
        semantic = SemanticCache(
            backend,
            CachedEmbedder(HashingEmbedder(), embedding_cache),
            tau=settings.semantic_cache_tau,
            model="query-hash",
        )

    return Container(
        settings=settings,
        agent=agent,
        metrics=metrics,
        http_metrics=http_metrics,
        auth=ApiKeyAuth(settings.api_keys),
        rate_limiter=RateLimitPolicy(
            settings.rate_limit_per_s,
            settings.rate_limit_burst,
            clock=clock,
            enabled=settings.rate_limit_enabled,
        ),
        response_cache=ResponseCache(backend, ttl_seconds=settings.response_cache_ttl_s),
        embedding_cache=embedding_cache,
        semantic_cache=semantic,
        tracing_enabled=tracing_enabled,
        generator_name=generator_name,
        retriever_name=retriever_name,
        cache_backend_name=backend_name,
    )
