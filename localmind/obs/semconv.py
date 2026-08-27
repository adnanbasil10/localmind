"""Span/metric attribute name constants — GenAI semantic conventions + LocalMind extensions.

implementation.md SS15 is explicit that vendor-neutral OpenTelemetry instrumentation using the
**GenAI semantic conventions** (``gen_ai.*``) is the correct 2026 choice, and that no span
attribute should be a bare string literal scattered across the codebase. This module is the one
place those names are spelled out; ``tracing.py`` and ``metrics.py`` import from here instead of
hardcoding strings.

Names under ``gen_ai.*`` follow the OpenTelemetry GenAI semantic conventions
(https://opentelemetry.io/docs/specs/semconv/gen-ai/). LocalMind's retrieval pipeline (BM25,
dense, ColBERT fusion, reranking, grading, verification, cache layers) has no first-class GenAI
attributes upstream, so those are namespaced ``localmind.*`` and documented individually. This
module has zero third-party imports so it is always safe to import.
"""

from __future__ import annotations

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
]

# --------------------------------------------------------------------------- #
# OpenTelemetry GenAI semantic conventions (gen_ai.*)
# --------------------------------------------------------------------------- #
GEN_AI_SYSTEM = "gen_ai.system"
"""The GenAI product/provider, e.g. ``"localmind"``."""

GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
"""The kind of GenAI operation the span represents, e.g. ``"chat"``, ``"embeddings"``."""

GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
"""Name of the model requested for this span (LocalMind-31M, the reranker, the embedder, ...)."""

GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
"""Name of the model that actually produced the response, if it can differ from the request."""

GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# --------------------------------------------------------------------------- #
# LocalMind extensions — the RAG pipeline shape has no GenAI semconv equivalent
# --------------------------------------------------------------------------- #
LOCALMIND_REQUEST_ID = "localmind.request_id"
LOCALMIND_PIPELINE_STAGE = "localmind.pipeline.stage"
LOCALMIND_STAGE_DURATION_MS = "localmind.stage.duration_ms"

LOCALMIND_CACHE_HIT = "localmind.cache.hit"
LOCALMIND_CACHE_LAYER = "localmind.cache.layer"
"""One of ``"embedding"``, ``"response"``, ``"semantic"``, ``"prefix"`` (the fourth layer, owned
by ``localmind.inference.prefix_cache`` — see that module for where its hit rate is surfaced)."""

LOCALMIND_RETRIEVAL_DOC_IDS = "localmind.retrieval.doc_ids"
LOCALMIND_RETRIEVAL_DOC_COUNT = "localmind.retrieval.doc_count"

LOCALMIND_ROUTE_DECISION = "localmind.route.decision"
LOCALMIND_GRADE_RELEVANT = "localmind.grade.relevant"
LOCALMIND_GRADE_SCORE = "localmind.grade.score"
LOCALMIND_VERIFY_PASSED = "localmind.verify.passed"
LOCALMIND_REFUSAL = "localmind.refusal"

LOCALMIND_TOOL_NAME = "localmind.tool.name"
LOCALMIND_TOOL_SUCCESS = "localmind.tool.success"

# --------------------------------------------------------------------------- #
# Pipeline shape (implementation.md SS15, CONVENTIONS.md): frozen, exact, ordered.
# --------------------------------------------------------------------------- #
PIPELINE_STAGES: tuple[str, ...] = (
    "route",
    "embed",
    "bm25",
    "dense",
    "colbert",
    "fuse",
    "rerank",
    "grade",
    "generate",
    "verify",
)
"""One trace per request, one child span per stage, in exactly this order."""
