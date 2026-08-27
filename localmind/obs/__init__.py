"""Observability: OpenTelemetry tracing (GenAI semantic conventions, exported to Phoenix over
OTLP) and Prometheus metrics (scraped for Grafana, ``deploy/grafana/``).

``import localmind.obs`` must succeed with only numpy/pydantic on the path — see the module
docstrings in ``tracing.py`` and ``metrics.py`` for the lazy-import discipline that guarantees it.
"""

from __future__ import annotations

from localmind.obs.metrics import Metrics, get_metrics, prometheus_available, reset_metrics
from localmind.obs.semconv import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    LOCALMIND_CACHE_HIT,
    LOCALMIND_CACHE_LAYER,
    LOCALMIND_GRADE_RELEVANT,
    LOCALMIND_GRADE_SCORE,
    LOCALMIND_PIPELINE_STAGE,
    LOCALMIND_REFUSAL,
    LOCALMIND_REQUEST_ID,
    LOCALMIND_RETRIEVAL_DOC_COUNT,
    LOCALMIND_RETRIEVAL_DOC_IDS,
    LOCALMIND_ROUTE_DECISION,
    LOCALMIND_STAGE_DURATION_MS,
    LOCALMIND_TOOL_NAME,
    LOCALMIND_TOOL_SUCCESS,
    LOCALMIND_VERIFY_PASSED,
    PIPELINE_STAGES,
)
from localmind.obs.tracing import (
    LocalMindTracer,
    RequestTrace,
    StageSpan,
    get_tracer,
    init_tracing,
    otel_available,
    shutdown_tracing,
)

__all__ = [
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_REQUEST_MAX_TOKENS",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_REQUEST_TEMPERATURE",
    "GEN_AI_REQUEST_TOP_P",
    "GEN_AI_RESPONSE_FINISH_REASONS",
    "GEN_AI_RESPONSE_MODEL",
    "GEN_AI_SYSTEM",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    "LOCALMIND_CACHE_HIT",
    "LOCALMIND_CACHE_LAYER",
    "LOCALMIND_GRADE_RELEVANT",
    "LOCALMIND_GRADE_SCORE",
    "LOCALMIND_PIPELINE_STAGE",
    "LOCALMIND_REFUSAL",
    "LOCALMIND_REQUEST_ID",
    "LOCALMIND_RETRIEVAL_DOC_COUNT",
    "LOCALMIND_RETRIEVAL_DOC_IDS",
    "LOCALMIND_ROUTE_DECISION",
    "LOCALMIND_STAGE_DURATION_MS",
    "LOCALMIND_TOOL_NAME",
    "LOCALMIND_TOOL_SUCCESS",
    "LOCALMIND_VERIFY_PASSED",
    "PIPELINE_STAGES",
    "LocalMindTracer",
    "Metrics",
    "RequestTrace",
    "StageSpan",
    "get_metrics",
    "get_tracer",
    "init_tracing",
    "otel_available",
    "prometheus_available",
    "reset_metrics",
    "shutdown_tracing",
]
