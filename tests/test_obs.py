"""Tests for observability (localmind/obs): tracing and metrics.

`opentelemetry-*` and `prometheus-client` are optional extras that are very likely NOT installed
in this environment (task brief). Every test here must therefore pass through the no-op fallback
paths — that fallback behaviour *is* the thing under test, not an exception to work around.
"""

from __future__ import annotations

import pytest
from localmind.obs import (
    PIPELINE_STAGES,
    Metrics,
    RequestTrace,
    get_metrics,
    get_tracer,
    init_tracing,
    otel_available,
    prometheus_available,
    reset_metrics,
    semconv,
)
from localmind.obs.tracing import shutdown_tracing


def test_import_localmind_obs_is_offline_safe() -> None:
    """`import localmind.obs` must succeed with only numpy/pydantic on the path. If this module
    imported opentelemetry or prometheus_client eagerly, collecting this file would already fail."""
    import localmind.obs

    assert hasattr(localmind.obs, "RequestTrace")


# --------------------------------------------------------------------------- #
# semconv.py: no bare string literals policy
# --------------------------------------------------------------------------- #
class TestSemconv:
    def test_pipeline_stages_is_the_exact_frozen_order(self) -> None:
        assert PIPELINE_STAGES == (
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

    def test_genai_attribute_constants_have_expected_values(self) -> None:
        from localmind.obs.semconv import GEN_AI_REQUEST_MODEL, GEN_AI_USAGE_INPUT_TOKENS

        assert GEN_AI_REQUEST_MODEL == "gen_ai.request.model"
        assert GEN_AI_USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"

    def test_all_exported_names_are_strings_or_the_stage_tuple(self) -> None:
        for name in semconv.__all__:
            value = getattr(semconv, name)
            if name == "PIPELINE_STAGES":
                assert isinstance(value, tuple)
            else:
                assert isinstance(value, str)
                assert value  # non-empty


# --------------------------------------------------------------------------- #
# tracing.py
# --------------------------------------------------------------------------- #
class TestTracingNoOpMode:
    def test_otel_available_returns_a_bool_and_does_not_raise(self) -> None:
        assert isinstance(otel_available(), bool)

    def test_init_tracing_returns_false_when_otel_not_installed(self) -> None:
        if otel_available():
            pytest.skip("opentelemetry is installed in this environment; no-op path not exercised")
        assert init_tracing() is False
        shutdown_tracing()

    def test_get_tracer_yields_a_working_span_in_no_op_mode(self) -> None:
        tracer = get_tracer("test")
        with tracer.start_span("my.span", {"k": "v"}) as span:
            span.set_attribute("extra", 1)
            assert span.attributes["k"] == "v"
            assert span.attributes["extra"] == 1

    def test_span_records_exception_and_still_propagates_it(self) -> None:
        tracer = get_tracer("test")
        with pytest.raises(RuntimeError), tracer.start_span("failing.span") as span:
            raise RuntimeError("boom")
        # the handle itself is out of scope by design once the context manager exits; re-enter
        # to confirm the *mechanism* doesn't swallow exceptions (already proven by pytest.raises)
        assert span.attributes["exception.type"] == "RuntimeError"


class TestRequestTrace:
    def test_full_pipeline_stage_order_is_tracked(self) -> None:
        with RequestTrace(get_tracer()) as trace:
            for stage in PIPELINE_STAGES:
                with trace.stage(stage):
                    pass
        assert trace.stages_seen == PIPELINE_STAGES

    def test_unknown_stage_name_raises(self) -> None:
        with (
            RequestTrace(get_tracer()) as trace,
            pytest.raises(ValueError, match="unknown pipeline stage"),
            trace.stage("not_a_real_stage"),
        ):
            pass

    def test_a_request_may_skip_stages_that_do_not_apply(self) -> None:
        """E.g. an exact-response-cache hit skips everything past routing."""
        with RequestTrace(get_tracer()) as trace, trace.stage("route"):
            pass
        assert trace.stages_seen == ("route",)

    def test_stage_span_setters_populate_semconv_attributes(self) -> None:
        from localmind.obs.semconv import (
            GEN_AI_REQUEST_MODEL,
            GEN_AI_USAGE_INPUT_TOKENS,
            GEN_AI_USAGE_OUTPUT_TOKENS,
            LOCALMIND_CACHE_HIT,
            LOCALMIND_CACHE_LAYER,
            LOCALMIND_RETRIEVAL_DOC_COUNT,
            LOCALMIND_RETRIEVAL_DOC_IDS,
        )

        with RequestTrace(get_tracer()) as trace:
            with trace.stage("generate") as span:
                span.set_model("localmind-31m").set_tokens(input_tokens=128, output_tokens=32)
                span.set_cache(hit=True, layer="response")

            with trace.stage("dense") as span:
                span.set_doc_ids(["doc-1", "doc-2", "doc-3"])

        gen_attrs = None
        dense_attrs = None
        # re-run to capture attributes directly since spans close at context-manager exit
        with RequestTrace(get_tracer()) as trace2:
            with trace2.stage("generate") as span:
                span.set_model("localmind-31m").set_tokens(input_tokens=128, output_tokens=32)
                span.set_cache(hit=True, layer="response")
                gen_attrs = dict(span.attributes)
            with trace2.stage("dense") as span:
                span.set_doc_ids(["doc-1", "doc-2", "doc-3"])
                dense_attrs = dict(span.attributes)

        assert gen_attrs[GEN_AI_REQUEST_MODEL] == "localmind-31m"
        assert gen_attrs[GEN_AI_USAGE_INPUT_TOKENS] == 128
        assert gen_attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 32
        assert gen_attrs[LOCALMIND_CACHE_HIT] is True
        assert gen_attrs[LOCALMIND_CACHE_LAYER] == "response"
        assert dense_attrs[LOCALMIND_RETRIEVAL_DOC_IDS] == ["doc-1", "doc-2", "doc-3"]
        assert dense_attrs[LOCALMIND_RETRIEVAL_DOC_COUNT] == 3

    def test_stage_records_duration(self) -> None:
        from localmind.obs.semconv import LOCALMIND_STAGE_DURATION_MS

        with RequestTrace(get_tracer()) as trace:
            with trace.stage("embed") as span:
                pass
            attrs = dict(span.attributes)
        assert attrs[LOCALMIND_STAGE_DURATION_MS] >= 0.0

    def test_request_id_is_generated_and_stable(self) -> None:
        with RequestTrace(get_tracer()) as trace:
            assert trace.request_id
            assert trace.root is not None
            assert trace.root.attributes["localmind.request_id"] == trace.request_id

    def test_explicit_request_id_is_used(self) -> None:
        with RequestTrace(get_tracer(), request_id="req-123") as trace:
            assert trace.request_id == "req-123"


# --------------------------------------------------------------------------- #
# metrics.py
# --------------------------------------------------------------------------- #
class TestMetricsNoOpMode:
    def test_prometheus_available_returns_a_bool(self) -> None:
        assert isinstance(prometheus_available(), bool)

    def test_construct_multiple_instances_never_raises(self) -> None:
        """A real prometheus_client.CollectorRegistry raises on duplicate series names if you
        reuse the global REGISTRY; Metrics() must default to a fresh registry per instance."""
        m1 = Metrics()
        m2 = Metrics()
        m1.observe_stage_latency("route", 0.01)
        m2.observe_stage_latency("route", 0.02)
        assert m1.snapshot()["stage_latency_seconds"]["route"] == [0.01]
        assert m2.snapshot()["stage_latency_seconds"]["route"] == [0.02]

    def test_observe_stage_latency_rejects_unknown_stage(self) -> None:
        m = Metrics()
        with pytest.raises(ValueError, match="unknown pipeline stage"):
            m.observe_stage_latency("not_a_stage", 1.0)

    def test_record_tokens(self) -> None:
        m = Metrics()
        m.record_tokens("generate", "localmind-31m", input_tokens=100, output_tokens=20)
        snap = m.snapshot()["tokens_total"]
        assert snap["generate|localmind-31m|input"] == 100
        assert snap["generate|localmind-31m|output"] == 20

    def test_record_tokens_ignores_zero_counts(self) -> None:
        m = Metrics()
        m.record_tokens("generate", "m", input_tokens=0, output_tokens=0)
        assert m.snapshot()["tokens_total"] == {}

    def test_record_cache_and_hit_rate(self) -> None:
        m = Metrics()
        m.record_cache("embedding", True)
        m.record_cache("embedding", True)
        m.record_cache("embedding", False)

        assert m.cache_hit_rate("embedding") == pytest.approx(2 / 3)
        assert m.cache_hit_rate("response") is None  # never recorded

    def test_prefix_cache_layer_can_be_recorded_here_too(self) -> None:
        """The fourth cache layer (KV prefix caching) is owned by
        localmind.inference.prefix_cache, but its hit rate is meant to land in the same
        localmind_cache_requests_total series via this same method."""
        m = Metrics()
        m.record_cache("prefix", True)
        assert m.cache_hit_rate("prefix") == pytest.approx(1.0)

    def test_record_error(self) -> None:
        m = Metrics()
        m.record_error("generate", "TimeoutError")
        assert m.snapshot()["errors_total"]["generate|TimeoutError"] == 1

    def test_record_refusal_default_route(self) -> None:
        m = Metrics()
        m.record_refusal()
        assert m.snapshot()["refusals_total"]["in_domain"] == 1

    def test_record_tool_result(self) -> None:
        m = Metrics()
        m.record_tool_result("calculator", True)
        m.record_tool_result("calculator", False)
        snap = m.snapshot()["tool_results_total"]
        assert snap["calculator|success"] == 1
        assert snap["calculator|failure"] == 1

    def test_record_eval_metric(self) -> None:
        m = Metrics()
        m.record_eval_metric("retrieval_ndcg_at_10", 0.812)
        assert m.snapshot()["eval_metric"]["retrieval_ndcg_at_10"] == pytest.approx(0.812)

    def test_render_never_raises_and_returns_bytes(self) -> None:
        m = Metrics()
        m.observe_stage_latency("route", 0.01)
        out = m.render()
        assert isinstance(out, bytes)

    def test_enabled_reflects_prometheus_availability(self) -> None:
        m = Metrics()
        assert m.enabled == prometheus_available()


class TestMetricsSingleton:
    def setup_method(self) -> None:
        reset_metrics()

    def teardown_method(self) -> None:
        reset_metrics()

    def test_get_metrics_returns_the_same_instance(self) -> None:
        assert get_metrics() is get_metrics()

    def test_reset_metrics_clears_the_singleton(self) -> None:
        first = get_metrics()
        reset_metrics()
        second = get_metrics()
        assert first is not second
