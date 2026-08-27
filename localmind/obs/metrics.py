"""Prometheus metrics for the LocalMind pipeline, scraped by the ``prometheus`` service
(``deploy/prometheus.yml``, job ``localmind-api``) and visualized in ``deploy/grafana/``.

Import discipline: ``prometheus-client`` lives in the ``obs`` extra and is very likely NOT
installed in a bare checkout. It is imported lazily inside :class:`Metrics`, which falls back to
an in-memory-only recorder when the package is absent — every ``record_*``/``observe_*`` call
still works and is inspectable via :meth:`Metrics.snapshot`, so instrumentation call sites never
need to branch on whether Prometheus is installed, and this module never crashes an
uninstrumented run.

Metric names (stable — dashboards in ``deploy/grafana/`` reference these by name):

- ``localmind_stage_latency_seconds{stage}``      Histogram — per-stage latency percentiles.
- ``localmind_tokens_total{stage,model,direction}`` Counter  — tokens/day (direction=input|output).
- ``localmind_cache_requests_total{layer,result}`` Counter  — cache hit rate per layer
  (layer=embedding|response|semantic|prefix; result=hit|miss).
- ``localmind_errors_total{stage,error_type}``     Counter  — error rate.
- ``localmind_refusals_total{route}``              Counter  — refusal rate.
- ``localmind_tool_results_total{tool,result}``    Counter  — tool success rate (result=success|failure).
- ``localmind_eval_metric{metric_name}``           Gauge    — retrieval-quality trend. Written by
  the nightly eval job (``localmind.eval``, a different task); this module only defines the
  series name so the eval harness and the dashboards agree on it.

The fourth cache layer, KV prefix caching, lives in ``localmind.inference.prefix_cache`` (a
different task). Its hit rate should be reported through
``record_cache(layer="prefix", hit=...)`` from that module so it lands in the same
``localmind_cache_requests_total`` series as the three layers owned here.
"""

from __future__ import annotations

from typing import Any

from localmind.obs.semconv import PIPELINE_STAGES

__all__ = ["Metrics", "get_metrics", "reset_metrics"]

_HISTOGRAM_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)


def prometheus_available() -> bool:
    """Whether the ``prometheus_client`` package is importable in this environment."""
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        return False
    return True


class _Snapshot:
    """Backend-independent counters, always maintained so tests and callers can inspect state
    without a live Prometheus registry."""

    def __init__(self) -> None:
        self.stage_latency_seconds: dict[str, list[float]] = {}
        self.tokens_total: dict[tuple[str, str, str], int] = {}
        self.cache_requests_total: dict[tuple[str, str], int] = {}
        self.errors_total: dict[tuple[str, str], int] = {}
        self.refusals_total: dict[str, int] = {}
        self.tool_results_total: dict[tuple[str, str], int] = {}
        self.eval_metric: dict[str, float] = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_latency_seconds": {
                stage: list(values) for stage, values in self.stage_latency_seconds.items()
            },
            "tokens_total": {"|".join(k): v for k, v in self.tokens_total.items()},
            "cache_requests_total": {"|".join(k): v for k, v in self.cache_requests_total.items()},
            "errors_total": {"|".join(k): v for k, v in self.errors_total.items()},
            "refusals_total": dict(self.refusals_total),
            "tool_results_total": {"|".join(k): v for k, v in self.tool_results_total.items()},
            "eval_metric": dict(self.eval_metric),
        }


class Metrics:
    """Records the LocalMind metric surface, backed by Prometheus when available.

    Uses a fresh ``CollectorRegistry`` per instance by default (pass one explicitly to share the
    process-global ``REGISTRY``) so constructing more than one ``Metrics()`` — e.g. once per test
    — never raises Prometheus's "duplicated timeseries" error.
    """

    def __init__(self, registry: Any = None) -> None:
        self._snapshot = _Snapshot()
        self._registry: Any = None
        self._hist: Any = None
        self._tokens: Any = None
        self._cache: Any = None
        self._errors: Any = None
        self._refusals: Any = None
        self._tool: Any = None
        self._eval_gauge: Any = None

        if prometheus_available():
            import prometheus_client as pc

            self._registry = registry if registry is not None else pc.CollectorRegistry()
            self._hist = pc.Histogram(
                "localmind_stage_latency_seconds",
                "Per-stage pipeline latency",
                labelnames=["stage"],
                buckets=_HISTOGRAM_BUCKETS,
                registry=self._registry,
            )
            self._tokens = pc.Counter(
                "localmind_tokens_total",
                "Tokens processed",
                labelnames=["stage", "model", "direction"],
                registry=self._registry,
            )
            self._cache = pc.Counter(
                "localmind_cache_requests_total",
                "Cache lookups per layer",
                labelnames=["layer", "result"],
                registry=self._registry,
            )
            self._errors = pc.Counter(
                "localmind_errors_total",
                "Errors per stage",
                labelnames=["stage", "error_type"],
                registry=self._registry,
            )
            self._refusals = pc.Counter(
                "localmind_refusals_total",
                "Refusals issued at routing time",
                labelnames=["route"],
                registry=self._registry,
            )
            self._tool = pc.Counter(
                "localmind_tool_results_total",
                "Agent tool call outcomes",
                labelnames=["tool", "result"],
                registry=self._registry,
            )
            self._eval_gauge = pc.Gauge(
                "localmind_eval_metric",
                "Nightly retrieval-quality eval metrics",
                labelnames=["metric_name"],
                registry=self._registry,
            )

    @property
    def enabled(self) -> bool:
        """Whether this instance is backed by a real Prometheus registry."""
        return self._registry is not None

    def observe_stage_latency(self, stage: str, seconds: float) -> None:
        if stage not in PIPELINE_STAGES:
            raise ValueError(f"unknown pipeline stage {stage!r}; must be one of {PIPELINE_STAGES}")
        self._snapshot.stage_latency_seconds.setdefault(stage, []).append(float(seconds))
        if self._hist is not None:
            self._hist.labels(stage=stage).observe(float(seconds))

    def record_tokens(
        self, stage: str, model: str, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        for direction, count in (("input", input_tokens), ("output", output_tokens)):
            if count <= 0:
                continue
            key = (stage, model, direction)
            self._snapshot.tokens_total[key] = self._snapshot.tokens_total.get(key, 0) + int(count)
            if self._tokens is not None:
                self._tokens.labels(stage=stage, model=model, direction=direction).inc(count)

    def record_cache(self, layer: str, hit: bool) -> None:
        result = "hit" if hit else "miss"
        key = (layer, result)
        self._snapshot.cache_requests_total[key] = (
            self._snapshot.cache_requests_total.get(key, 0) + 1
        )
        if self._cache is not None:
            self._cache.labels(layer=layer, result=result).inc()

    def record_error(self, stage: str, error_type: str) -> None:
        key = (stage, error_type)
        self._snapshot.errors_total[key] = self._snapshot.errors_total.get(key, 0) + 1
        if self._errors is not None:
            self._errors.labels(stage=stage, error_type=error_type).inc()

    def record_refusal(self, route: str = "in_domain") -> None:
        self._snapshot.refusals_total[route] = self._snapshot.refusals_total.get(route, 0) + 1
        if self._refusals is not None:
            self._refusals.labels(route=route).inc()

    def record_tool_result(self, tool: str, success: bool) -> None:
        result = "success" if success else "failure"
        key = (tool, result)
        self._snapshot.tool_results_total[key] = self._snapshot.tool_results_total.get(key, 0) + 1
        if self._tool is not None:
            self._tool.labels(tool=tool, result=result).inc()

    def record_eval_metric(self, metric_name: str, value: float) -> None:
        """Set a point of the nightly retrieval-quality trend. Called by ``localmind.eval``."""
        self._snapshot.eval_metric[metric_name] = float(value)
        if self._eval_gauge is not None:
            self._eval_gauge.labels(metric_name=metric_name).set(float(value))

    def cache_hit_rate(self, layer: str) -> float | None:
        hits = self._snapshot.cache_requests_total.get((layer, "hit"), 0)
        misses = self._snapshot.cache_requests_total.get((layer, "miss"), 0)
        total = hits + misses
        return (hits / total) if total else None

    def snapshot(self) -> dict[str, Any]:
        """Backend-independent view of everything recorded so far. Always available, regardless
        of whether ``prometheus_client`` is installed — this is what tests assert against."""
        return self._snapshot.to_dict()

    def render(self) -> bytes:
        """Prometheus text exposition format. Falls back to an explanatory comment (still valid
        Prometheus exposition-format text, just empty of series) when Prometheus isn't installed,
        so a caller wiring this into an HTTP ``/metrics`` handler never crashes."""
        if self._registry is not None:
            import prometheus_client as pc

            return pc.generate_latest(self._registry)
        return b"# prometheus_client is not installed; metrics unavailable in exposition format\n"


_default_metrics: Metrics | None = None


def get_metrics() -> Metrics:
    """Process-wide default :class:`Metrics` singleton."""
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = Metrics()
    return _default_metrics


def reset_metrics() -> None:
    """Reset the process-wide singleton. Mainly for test isolation."""
    global _default_metrics
    _default_metrics = None
