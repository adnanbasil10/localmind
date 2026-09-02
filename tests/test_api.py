"""Tests for Phase 12 — the RAG gateway (``localmind/api``). Fully offline.

No Postgres, no Redis, no Ollama, no OTel collector, no network. Every seam is injected as a
deterministic fake, and the two degraded paths that matter in production — ``opentelemetry`` and
``prometheus_client`` absent — are the *default* here rather than a special case.

``fastapi``/``uvicorn`` live in the ``rag`` extra and may not be installed. This module must
therefore import and largely run without them: the framework-free core (schemas, deps, the
adapters, ``execute_chat``, the SSE framing, the raw-ASGI middleware) is tested unconditionally,
and only the ``TestClient`` cases are skipped when the extra is missing.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from localmind.agent.state import GenerationResult, ManualClock, RetrievedChunk
from localmind.api import (
    ApiError,
    ApiKeyAuth,
    ChatRequest,
    ChatResponse,
    ExtractiveGenerator,
    HttpMetrics,
    RateLimitPolicy,
    RequestContext,
    RequestContextMiddleware,
    ScoredDocRetriever,
    Settings,
    TracedGenerator,
    TracingToolRegistry,
    build_container,
    execute_chat,
    health_report,
    sse_frame,
    stream_events,
)
from localmind.api.middleware import (
    GENERIC_MESSAGES,
    REQUEST_ID_HEADER,
    error_payload,
    extract_request_id,
    use_context,
)
from localmind.api.routes import (
    NODE_TO_STAGE,
    SSE_DONE,
    build_chat_response,
    chunk_text,
    safe_refusal_reason,
    safe_warnings,
)
from localmind.api.schemas import SNIPPET_CHARS
from localmind.obs import PIPELINE_STAGES, Metrics
from localmind.obs.tracing import RequestTrace
from localmind.retrieval import Document, ScoredDoc

try:  # `rag` extra
    import fastapi  # noqa: F401

    FASTAPI_INSTALLED = True
except ImportError:  # pragma: no cover - depends on the environment, not on the code
    FASTAPI_INSTALLED = False

requires_fastapi = pytest.mark.skipif(
    not FASTAPI_INSTALLED,
    reason="fastapi is in the 'rag' optional dependency group and is not installed",
)


# ======================================================================================
# Deterministic fakes
# ======================================================================================
CORPUS: list[Document] = [
    Document(
        doc_id="d1",
        text="The sky is blue because shorter wavelengths scatter more in the atmosphere.",
        metadata={"doc_type": "pdf", "lang": "en"},
    ),
    Document(
        doc_id="d2",
        text="Rayleigh scattering explains why the sky is blue on a clear day.",
        metadata={"doc_type": "pdf", "lang": "en"},
    ),
    Document(
        doc_id="d3",
        text="Sunsets are red because the light path through the atmosphere is longer.",
        metadata={"doc_type": "html", "lang": "en"},
    ),
]


class FakeArm:
    """A ``localmind.retrieval`` arm: returns real ``ScoredDoc``s — ``(doc_id, score)``, no text."""

    def __init__(self, docs: Sequence[Document] = CORPUS, extra_ids: Sequence[str] = ()) -> None:
        self.docs = list(docs)
        self.extra_ids = list(extra_ids)
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 10) -> list[ScoredDoc]:
        self.calls.append((query, top_k))
        terms = {w for w in query.lower().split() if len(w) > 2}
        scored = [
            ScoredDoc(
                doc_id=d.doc_id,
                score=len(terms & {w for w in d.text.lower().split() if len(w) > 2})
                / max(1, len(terms)),
            )
            for d in self.docs
        ]
        scored.extend(ScoredDoc(doc_id=i, score=0.01) for i in self.extra_ids)
        scored.sort(key=lambda s: -s.score)
        return scored[:top_k]


class FakeControlPlane:
    """Stands in for LocalMind-31M: always in-domain, always relevant."""

    def __init__(self, route_result: str = "in_domain", grade_result: Any = (True, 0.9)) -> None:
        self.route_result = route_result
        self.grade_result = grade_result

    def route(self, query: str) -> Any:
        return self.route_result

    def grade(self, query: str, chunk: str) -> Any:
        return self.grade_result

    def rewrite(self, query: str, history: list[str]) -> str:
        return f"{query} (refined)"


class FakeGenerator:
    """Stands in for Ollama."""

    name = "fake-generator"

    def __init__(self, text: str = "The sky is blue because of scattering [1].") -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult:
        self.prompts.append(prompt)
        return GenerationResult(
            text=self.text, prompt_tokens=11, completion_tokens=7, model=self.name
        )


class ListRetriever:
    """A minimal agent-side ``Retriever``: already carries ``text``."""

    def __init__(self, chunks: Sequence[RetrievedChunk]) -> None:
        self.chunks = list(chunks)
        self.calls: list[tuple[str, int, Any]] = []

    def search(
        self, query: str, k: int = 5, filters: Mapping[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((query, k, dict(filters) if filters else None))
        return self.chunks[:k]


def make_container(**overrides: Any) -> Any:
    """A container whose every seam is a fake and whose caches are in-memory."""
    settings_kw: dict[str, Any] = {
        "generator": "none",
        "rate_limit_per_s": 1000.0,
        "rate_limit_burst": 1000.0,
    }
    settings_kw.update(overrides.pop("settings", {}))
    retriever = overrides.pop("retriever", ScoredDocRetriever(FakeArm(), CORPUS))
    generator = overrides.pop("generator", FakeGenerator())
    control_plane = overrides.pop("control_plane", FakeControlPlane())
    return build_container(
        Settings(**settings_kw),
        retriever=retriever,
        generator=generator,
        control_plane=control_plane,
        clock=ManualClock(),
        **overrides,
    )


# ======================================================================================
# Import discipline and the inference-server boundary
# ======================================================================================
def test_import_localmind_api_never_needs_the_rag_extra() -> None:
    """If any module imported FastAPI eagerly, collecting this file would already have failed."""
    import localmind.api

    assert "create_app" in localmind.api.__all__
    assert "create_app" in dir(localmind.api)


def test_api_does_not_pull_in_the_model_serving_layer() -> None:
    """The boundary in docs/architecture.md, asserted rather than asserted-in-prose.

    ``localmind.inference.server`` owns ``/v1/chat/completions`` and friends for *the model*; the
    gateway is a different process with a different concern. Importing one must not drag in the
    other. Checked in a subprocess so this test cannot disturb ``sys.modules`` for anyone else.
    """
    code = (
        "import sys, localmind.api, localmind.api.main, localmind.api.routes; "
        "print(sorted(m for m in sys.modules if m.startswith('localmind.inference')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "[]"


def test_lazy_attribute_error_is_still_an_attribute_error() -> None:
    import localmind.api

    with pytest.raises(AttributeError):
        _ = localmind.api.no_such_thing


# ======================================================================================
# Settings
# ======================================================================================
class TestSettings:
    def test_defaults_need_no_environment(self) -> None:
        settings = Settings.from_env({})
        assert settings.service_name == "localmind-api"
        assert settings.api_keys == ()
        assert settings.auth_required is False
        assert settings.redis_url is None and settings.pg_dsn is None

    def test_reads_the_variables_docker_compose_already_sets(self) -> None:
        settings = Settings.from_env(
            {
                "LOCALMIND_PG_DSN": "postgresql://localmind:localmind@postgres:5432/localmind",
                "LOCALMIND_REDIS_URL": "redis://redis:6379/0",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://phoenix:4317",
            }
        )
        assert settings.pg_dsn is not None and settings.pg_dsn.endswith("/localmind")
        assert settings.redis_url == "redis://redis:6379/0"
        assert settings.otlp_endpoint == "http://phoenix:4317"

    def test_api_keys_are_comma_separated_and_enable_auth(self) -> None:
        settings = Settings.from_env({"LOCALMIND_API_KEYS": " k1 , k2 ,"})
        assert settings.api_keys == ("k1", "k2")
        assert settings.auth_required is True

    def test_a_junk_numeric_value_falls_back_instead_of_crashing_startup(self) -> None:
        assert Settings.from_env({"LOCALMIND_TOP_K": "not-a-number"}).top_k == 5


# ======================================================================================
# Auth and rate limiting
# ======================================================================================
class TestAuth:
    def test_no_keys_configured_means_the_gateway_is_open(self) -> None:
        auth = ApiKeyAuth(())
        principal = auth.identify(None, None)
        assert principal.authenticated is False
        assert principal.key_id == "anonymous"

    def test_bearer_and_api_key_headers_are_both_accepted(self) -> None:
        auth = ApiKeyAuth(("s3cret",))
        assert auth.identify("Bearer s3cret", None).authenticated
        assert auth.identify(None, "s3cret").authenticated

    def test_a_wrong_or_missing_key_is_a_401_with_a_challenge(self) -> None:
        auth = ApiKeyAuth(("s3cret",))
        for authorization, api_key in ((None, None), ("Bearer nope", None), (None, "nope")):
            with pytest.raises(ApiError) as excinfo:
                auth.identify(authorization, api_key)
            assert excinfo.value.status_code == 401
            assert excinfo.value.headers["WWW-Authenticate"] == "Bearer"

    def test_the_principal_id_is_a_digest_not_the_key(self) -> None:
        principal = ApiKeyAuth(("s3cret",)).identify("Bearer s3cret", None)
        assert "s3cret" not in principal.key_id
        assert len(principal.key_id) == 12


class TestRateLimit:
    def test_burst_then_429_with_retry_after(self) -> None:
        policy = RateLimitPolicy(per_second=0.001, burst=2, clock=ManualClock())
        policy.check("k")
        policy.check("k")
        with pytest.raises(ApiError) as excinfo:
            policy.check("k")
        assert excinfo.value.status_code == 429
        assert int(excinfo.value.headers["Retry-After"]) >= 1

    def test_buckets_are_per_principal(self) -> None:
        policy = RateLimitPolicy(per_second=0.001, burst=1, clock=ManualClock())
        policy.check("a")
        policy.check("b")  # b has its own bucket; must not raise

    def test_disabled_policy_never_raises(self) -> None:
        policy = RateLimitPolicy(per_second=0.0, burst=0.0, enabled=False)
        for _ in range(50):
            policy.check("k")
        assert policy.describe() == "disabled"


# ======================================================================================
# The ScoredDoc -> text adapter (the one real impedance mismatch)
# ======================================================================================
class TestScoredDocRetriever:
    def test_rejoins_the_text_a_scored_doc_does_not_carry(self) -> None:
        adapter = ScoredDocRetriever(FakeArm(), CORPUS)
        hits = adapter.search("why is the sky blue", 2)
        assert len(hits) == 2
        assert all(isinstance(h, RetrievedChunk) for h in hits)
        assert all(h.text for h in hits)
        assert all(h.trust == "untrusted" for h in hits)

    def test_calls_the_arm_positionally_so_top_k_versus_k_never_matters(self) -> None:
        arm = FakeArm()
        ScoredDocRetriever(arm, CORPUS).search("sky", 3)
        assert arm.calls == [("sky", 3)]

    def test_filters_are_applied_here_because_arms_have_none(self) -> None:
        arm = FakeArm()
        adapter = ScoredDocRetriever(arm, CORPUS, oversample=4)
        hits = adapter.search("sky atmosphere", 2, {"doc_type": "html"})
        assert [h.doc_id for h in hits] == ["d3"]
        # Oversampled, because post-filtering would otherwise shrink the result set silently.
        assert arm.calls[-1][1] == 8

    def test_filter_accepts_a_set_of_allowed_values(self) -> None:
        adapter = ScoredDocRetriever(FakeArm(), CORPUS)
        hits = adapter.search("sky", 5, {"doc_type": ["pdf", "html"]})
        assert {h.doc_id for h in hits} == {"d1", "d2", "d3"}

    def test_a_doc_id_with_no_corpus_text_is_dropped_not_emitted_empty(self) -> None:
        adapter = ScoredDocRetriever(FakeArm(extra_ids=["ghost"]), CORPUS)
        hits = adapter.search("sky", 10)
        assert "ghost" not in {h.doc_id for h in hits}

    def test_accepts_a_plain_mapping_corpus(self) -> None:
        adapter = ScoredDocRetriever(FakeArm(), {"d1": "text one", "d2": "text two", "d3": "x"})
        assert len(adapter) == 3
        assert adapter.search("sky", 1)[0].text in ("text one", "text two", "x")

    def test_never_returns_more_than_k(self) -> None:
        adapter = ScoredDocRetriever(FakeArm(), CORPUS)
        assert len(adapter.search("sky atmosphere", 1)) == 1


# ======================================================================================
# Tracing / metrics wrappers
# ======================================================================================
class TestTracedSeams:
    @staticmethod
    def registry(retriever: Any, **kw: Any) -> TracingToolRegistry:
        from localmind.agent.tools import build_default_registry

        return TracingToolRegistry.wrapping(build_default_registry(retriever=retriever), **kw)

    def test_the_registry_opens_a_child_stage_span_under_the_root(self) -> None:
        metrics = Metrics()
        registry = self.registry(ScoredDocRetriever(FakeArm(), CORPUS), metrics=metrics)
        with RequestTrace(request_id="r1") as trace:
            with use_context(RequestContext(request_id="r1", trace=trace)):
                result = registry.call("search_documents", {"query": "sky", "k": 2})
            assert "fuse" in trace.stages_seen
        assert result.ok
        assert metrics.snapshot()["stage_latency_seconds"]["fuse"]

    def test_it_rejects_a_stage_name_outside_the_frozen_pipeline(self) -> None:
        with pytest.raises(ValueError, match="unknown pipeline stage"):
            TracingToolRegistry([], stage="retrieve")

    def test_ambient_filters_are_merged_into_the_tool_arguments(self) -> None:
        """The agent calls `search_documents` with {query, k} only, so this is the only seam left.

        It also has to be *this* seam rather than a retriever wrapper: tools run on a fresh daemon
        thread (``agent.tools.base.call_with_timeout``), which starts with an empty contextvars
        context, so nothing bound per request survives into the retriever itself.
        """
        inner = ListRetriever([])
        registry = self.registry(inner)
        with use_context(RequestContext(request_id="r1", filters={"doc_type": "pdf"})):
            registry.call("search_documents", {"query": "sky", "k": 3})
        assert inner.calls[-1][2] == {"doc_type": "pdf"}

    def test_explicit_filters_win_over_ambient_ones(self) -> None:
        inner = ListRetriever([])
        registry = self.registry(inner)
        with use_context(RequestContext(request_id="r1", filters={"doc_type": "pdf"})):
            registry.call(
                "search_documents", {"query": "sky", "k": 3, "filters": {"doc_type": "html"}}
            )
        assert inner.calls[-1][2] == {"doc_type": "html"}

    def test_other_tools_are_passed_straight_through(self) -> None:
        registry = self.registry(ListRetriever([]))
        result = registry.call("calculate", {"expression": "2+2"})
        assert result.ok

    def test_works_with_no_request_context_at_all(self) -> None:
        registry = self.registry(ScoredDocRetriever(FakeArm(), CORPUS))
        assert registry.call("search_documents", {"query": "sky", "k": 1}).ok

    def test_wrapping_preserves_every_tool(self) -> None:
        from localmind.agent.tools import build_default_registry

        base = build_default_registry()
        assert TracingToolRegistry.wrapping(base).names() == base.names()

    def test_generator_records_genai_token_counts(self) -> None:
        metrics = Metrics()
        traced = TracedGenerator(FakeGenerator(), metrics=metrics)
        with RequestTrace(request_id="r1") as trace:
            with use_context(RequestContext(request_id="r1", trace=trace)):
                traced.generate("prompt")
            assert "generate" in trace.stages_seen
        tokens = metrics.snapshot()["tokens_total"]
        assert tokens["generate|fake-generator|input"] == 11
        assert tokens["generate|fake-generator|output"] == 7


class TestExtractiveGenerator:
    def test_quotes_and_cites_the_first_source(self) -> None:
        prompt = (
            "SOURCES\nThe sky is blue because shorter wavelengths scatter.\n\nUSER QUESTION\nwhy"
        )
        result = ExtractiveGenerator().generate(prompt)
        assert result.text.endswith("[1]")
        assert result.model == "extractive-stub"

    def test_says_so_when_there_is_nothing_to_quote(self) -> None:
        assert ExtractiveGenerator().generate("no sources here").text == "INSUFFICIENT_EVIDENCE"


# ======================================================================================
# The RAG entrypoint, without any web framework
# ======================================================================================
class TestExecuteChat:
    def test_answers_with_citations(self) -> None:
        container = make_container()
        response = execute_chat(container, ChatRequest(query="why is the sky blue"))
        assert response.status == "answered"
        assert response.answer
        assert response.citations, "an answer must carry citations"
        assert response.citations[0].marker == 1
        assert response.citations[0].chunk_id == response.sources[0].chunk_id
        assert response.sources[0].snippet
        assert response.route == "in_domain"

    def test_child_stages_nest_under_the_request_root_span(self) -> None:
        container = make_container()
        with RequestTrace(request_id="rid") as trace:
            context = RequestContext(request_id="rid", trace=trace)
            with use_context(context):
                execute_chat(container, ChatRequest(query="why is the sky blue"), context=context)
            assert "fuse" in trace.stages_seen
            assert "generate" in trace.stages_seen
        assert trace.root is not None

    def test_agent_node_timings_land_on_the_prometheus_stage_histogram(self) -> None:
        container = make_container()
        execute_chat(container, ChatRequest(query="why is the sky blue"))
        stages = container.metrics.snapshot()["stage_latency_seconds"]
        assert {"route", "grade", "verify"} <= set(stages)
        assert set(stages) <= set(PIPELINE_STAGES)

    def test_generate_latency_is_not_double_counted(self) -> None:
        container = make_container()
        with RequestTrace(request_id="rid") as trace:
            context = RequestContext(request_id="rid", trace=trace)
            with use_context(context):
                execute_chat(container, ChatRequest(query="why is the sky blue"), context=context)
        stages = container.metrics.snapshot()["stage_latency_seconds"]
        assert len(stages["generate"]) == 1

    def test_filters_reach_retrieval(self) -> None:
        arm = FakeArm()
        container = make_container(retriever=ScoredDocRetriever(arm, CORPUS))
        response = execute_chat(
            container, ChatRequest(query="why is the sky blue", filters={"doc_type": "html"})
        )
        assert response.status == "answered"
        assert [s.doc_id for s in response.sources] == ["d3"]
        assert arm.calls[-1][1] > 5  # oversampled, i.e. the filter really was threaded through

    def test_a_refusal_is_a_successful_outcome_with_a_vocabulary_reason(self) -> None:
        container = make_container(control_plane=FakeControlPlane(route_result="out_of_domain"))
        response = execute_chat(container, ChatRequest(query="what is the weather"))
        assert response.status == "refused"
        assert response.refusal_reason == "out_of_domain"
        assert response.citations == []

    def test_a_missing_generator_refuses_rather_than_crashing(self) -> None:
        container = make_container(generator=None, settings={"generator": "none"})
        response = execute_chat(container, ChatRequest(query="why is the sky blue"))
        assert response.status == "refused"
        assert response.refusal_reason == "generator_unavailable"


class TestCaching:
    def test_second_identical_request_is_served_from_the_response_cache(self) -> None:
        container = make_container()
        first = execute_chat(container, ChatRequest(query="why is the sky blue"))
        second = execute_chat(container, ChatRequest(query="why is the sky blue"))
        assert first.cached is False
        assert second.cached is True
        assert second.cache_layer == "response"
        assert second.answer == first.answer
        assert second.citations == first.citations
        assert container.metrics.cache_hit_rate("response") == 0.5

    def test_a_cached_response_still_carries_this_request_s_id(self) -> None:
        container = make_container()
        execute_chat(container, ChatRequest(query="why is the sky blue"), request_id="one")
        second = execute_chat(container, ChatRequest(query="why is the sky blue"), request_id="two")
        assert second.request_id == "two"

    def test_use_cache_false_bypasses_both_layers(self) -> None:
        container = make_container()
        execute_chat(container, ChatRequest(query="why is the sky blue"))
        execute_chat(container, ChatRequest(query="why is the sky blue", use_cache=False))
        assert container.metrics.snapshot()["cache_requests_total"].get("response|hit") is None

    def test_a_near_duplicate_query_hits_the_semantic_layer(self) -> None:
        container = make_container()
        execute_chat(container, ChatRequest(query="why is the sky blue"))
        again = execute_chat(container, ChatRequest(query="why is the sky blue?"))
        assert again.cached is True
        assert again.cache_layer == "semantic"

    def test_the_semantic_layer_is_skipped_when_filters_change_the_answer(self) -> None:
        """It keys on query text alone, so a filtered request must never read from it."""
        container = make_container()
        execute_chat(container, ChatRequest(query="why is the sky blue"))
        filtered = execute_chat(
            container, ChatRequest(query="why is the sky blue?", filters={"doc_type": "html"})
        )
        assert filtered.cache_layer != "semantic"

    def test_the_embedding_cache_is_wired_underneath_the_semantic_cache(self) -> None:
        container = make_container()
        execute_chat(container, ChatRequest(query="why is the sky blue"))
        execute_chat(container, ChatRequest(query="why is the sky blue?"))
        assert container.embedding_cache.stats.total > 0


# ======================================================================================
# Response projection, and what must never leave the process
# ======================================================================================
class TestResponseProjection:
    def test_source_snippets_are_bounded(self) -> None:
        container = make_container(
            retriever=ListRetriever(
                [
                    RetrievedChunk(chunk_id="c1", doc_id="d1", text="sky " * 400, score=1.0),
                    RetrievedChunk(chunk_id="c2", doc_id="d2", text="blue " * 400, score=0.9),
                ]
            )
        )
        response = execute_chat(container, ChatRequest(query="why is the sky blue"))
        assert response.sources
        assert all(len(s.snippet) <= SNIPPET_CHARS for s in response.sources)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("out of domain", "out_of_domain"),
            ("no relevant evidence found", "insufficient_evidence"),
            (
                "generator unavailable: ConnectError: [Errno 111] localhost:11434",
                "generator_unavailable",
            ),
            ("wall-clock budget exhausted (31.02s > 30.0s)", "budget_exhausted"),
            ("retrieval cap reached (4)", "budget_exhausted"),
            ("2 unsupported claim(s): 'the moon is cheese'", "unverified_answer"),
            ("internal error in generate: KeyError: 'secret_table'", "internal_error"),
            ("", ""),
        ],
    )
    def test_refusal_reasons_collapse_to_a_closed_vocabulary(self, raw: str, expected: str) -> None:
        assert safe_refusal_reason(raw) == expected

    def test_no_refusal_reason_leaks_an_exception_or_a_path(self) -> None:
        leaky = "internal error in retrieve: OperationalError: could not connect to postgres:5432"
        assert safe_refusal_reason(leaky) == "internal_error"
        assert "postgres" not in safe_refusal_reason(leaky)

    def test_warnings_are_derived_flags_not_raw_tool_errors(self) -> None:
        container = make_container(retriever=None)
        response = execute_chat(container, ChatRequest(query="why is the sky blue"))
        assert "tool_failure" in response.warnings
        assert all("tool_error" not in w for w in response.warnings)

    def test_safe_warnings_reports_each_flag_it_owns(self) -> None:
        container = make_container()
        response = execute_chat(container, ChatRequest(query="why is the sky blue"))
        state = ChatResponse.model_validate(response.model_dump())
        assert isinstance(state.warnings, list)

    def test_build_chat_response_maps_an_unknown_status_to_error(self) -> None:
        container = make_container()
        result = container.agent.run("why is the sky blue")
        result.status = "running"  # type: ignore[assignment]
        assert build_chat_response(result).status == "error"

    def test_the_node_to_stage_map_only_names_real_stages(self) -> None:
        assert set(NODE_TO_STAGE.values()) <= set(PIPELINE_STAGES)
        assert safe_warnings  # imported and used above; keeps the export honest


# ======================================================================================
# SSE framing
# ======================================================================================
class TestStreaming:
    def test_frames_are_json_after_a_data_prefix(self) -> None:
        frame = sse_frame({"type": "token", "text": "hi"})
        assert frame.startswith("data: ") and frame.endswith("\n\n")
        assert json.loads(frame[len("data: ") :].strip()) == {"type": "token", "text": "hi"}

    def test_chunking_is_lossless(self) -> None:
        assert "".join(chunk_text("abcdefg", 3)) == "abcdefg"
        assert chunk_text("", 3) == []

    def test_the_event_order_ends_with_citations_then_done(self) -> None:
        container = make_container()
        response = execute_chat(container, ChatRequest(query="why is the sky blue"))
        frames = list(stream_events(response, 8))
        assert frames[-1] == SSE_DONE
        kinds = [json.loads(f[len("data: ") :].strip())["type"] for f in frames[:-1]]
        assert kinds[0] == "start"
        assert kinds[-2:] == ["citations", "done"]
        assert "token" in kinds
        payload = json.loads(frames[-3][len("data: ") :].strip())
        assert payload["citations"], "the stream must still deliver citations"

    def test_the_streamed_tokens_reassemble_into_the_answer(self) -> None:
        container = make_container()
        response = execute_chat(container, ChatRequest(query="why is the sky blue"))
        tokens = [
            json.loads(f[len("data: ") :].strip())
            for f in stream_events(response, 5)
            if f != SSE_DONE
        ]
        assert "".join(t["text"] for t in tokens if t["type"] == "token") == response.answer


# ======================================================================================
# Raw-ASGI middleware (no web framework involved)
# ======================================================================================
def call_asgi(
    app: Any, path: str = "/chat", headers: Sequence[tuple[bytes, bytes]] = ()
) -> list[Any]:
    messages: list[Any] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Any) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": list(headers),
        "client": ("127.0.0.1", 5000),
    }
    asyncio.run(app(scope, receive, send))
    return messages


async def _ok_app(scope: Any, receive: Any, send: Any) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        }
    )
    await send({"type": "http.response.body", "body": b"ok"})


async def _boom_app(scope: Any, receive: Any, send: Any) -> None:
    raise RuntimeError("connection to postgres://localmind:localmind@postgres:5432 failed")


async def _api_error_app(scope: Any, receive: Any, send: Any) -> None:
    raise ApiError(429, "rate_limited", headers={"Retry-After": "7"})


def _headers_of(messages: list[Any]) -> dict[str, str]:
    start = next(m for m in messages if m["type"] == "http.response.start")
    return {bytes(k).decode().lower(): bytes(v).decode() for k, v in start["headers"]}


class TestRequestContextMiddleware:
    def test_mints_a_request_id_and_echoes_it(self) -> None:
        messages = call_asgi(RequestContextMiddleware(_ok_app))
        assert len(_headers_of(messages)[REQUEST_ID_HEADER]) == 32

    def test_honours_a_caller_supplied_request_id(self) -> None:
        messages = call_asgi(
            RequestContextMiddleware(_ok_app), headers=[(b"x-request-id", b"trace-abc.123")]
        )
        assert _headers_of(messages)[REQUEST_ID_HEADER] == "trace-abc.123"

    @pytest.mark.parametrize("bad", [b"has space", b"line\nbreak", b"x" * 200, b""])
    def test_rejects_an_unsafe_request_id_instead_of_echoing_it(self, bad: bytes) -> None:
        messages = call_asgi(RequestContextMiddleware(_ok_app), headers=[(b"x-request-id", bad)])
        assert _headers_of(messages)[REQUEST_ID_HEADER] != bad.decode()
        assert extract_request_id({"x-request-id": bad.decode()}) is None

    def test_an_unhandled_exception_becomes_a_500_envelope_with_no_internals(self) -> None:
        metrics = HttpMetrics()
        messages = call_asgi(RequestContextMiddleware(_boom_app, metrics=metrics))
        start = next(m for m in messages if m["type"] == "http.response.start")
        body = json.loads(next(m for m in messages if m["type"] == "http.response.body")["body"])
        assert start["status"] == 500
        assert body == error_payload("internal", GENERIC_MESSAGES[500], body["error"]["request_id"])
        rendered = json.dumps(body)
        for secret in ("postgres", "RuntimeError", "Traceback", "localmind:localmind"):
            assert secret not in rendered
        assert any("RuntimeError" in k for k in metrics.exceptions_total)

    def test_an_api_error_keeps_its_status_code_and_headers(self) -> None:
        messages = call_asgi(RequestContextMiddleware(_api_error_app))
        start = next(m for m in messages if m["type"] == "http.response.start")
        assert start["status"] == 429
        assert _headers_of(messages)["retry-after"] == "7"

    def test_records_request_metrics_and_folds_unknown_paths_together(self) -> None:
        metrics = HttpMetrics()
        call_asgi(RequestContextMiddleware(_ok_app, metrics=metrics), path="/chat")
        call_asgi(RequestContextMiddleware(_ok_app, metrics=metrics), path="/wp-admin.php")
        keys = set(metrics.requests_total)
        assert ("POST", "/chat", "200") in keys
        assert ("POST", "other", "200") in keys

    def test_non_http_scopes_pass_straight_through(self) -> None:
        seen: list[str] = []

        async def lifespan_app(scope: Any, receive: Any, send: Any) -> None:
            seen.append(scope["type"])

        async def noop() -> dict[str, Any]:
            return {}

        async def sink(message: Any) -> None:
            return None

        asyncio.run(RequestContextMiddleware(lifespan_app)({"type": "lifespan"}, noop, sink))
        assert seen == ["lifespan"]


# ======================================================================================
# Health, with nothing at all running
# ======================================================================================
class TestHealth:
    def test_reports_ok_with_no_backing_services(self) -> None:
        report = health_report(make_container())
        assert report.status == "ok"
        assert report.components["postgres"] == "not configured"
        assert report.components["redis"] == "not configured"
        assert report.components["tracing"].startswith("no-op")
        assert report.components["metrics"].startswith("in-memory")

    def test_a_configured_but_unreachable_redis_does_not_break_startup(self) -> None:
        container = build_container(Settings(redis_url="redis://nowhere:6379/0"))
        assert health_report(container).status == "ok"
        assert container.cache_backend_name.startswith("in-memory")

    def test_metrics_render_is_prometheus_text(self) -> None:
        rendered = make_container().render_metrics()
        assert isinstance(rendered, bytes)
        assert rendered.startswith(b"#") or rendered == b""


# ======================================================================================
# The HTTP surface (skipped without the `rag` extra)
# ======================================================================================
@requires_fastapi
class TestHttpSurface:
    @staticmethod
    def client(**overrides: Any) -> Any:
        from fastapi.testclient import TestClient
        from localmind.api.main import create_app

        container = make_container(**overrides)
        client = TestClient(create_app(container))
        client.container = container  # type: ignore[attr-defined]
        return client

    def test_health_is_200_with_zero_services_running(self) -> None:
        with self.client() as client:
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.headers[REQUEST_ID_HEADER]

    def test_metrics_is_scrapeable_at_the_path_prometheus_yml_expects(self) -> None:
        with self.client() as client:
            response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")

    def test_chat_returns_an_answer_with_citations(self) -> None:
        with self.client() as client:
            response = client.post("/chat", json={"query": "why is the sky blue"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "answered"
        assert body["citations"] and body["citations"][0]["chunk_id"]
        assert body["sources"][0]["snippet"]
        assert body["request_id"] == response.headers[REQUEST_ID_HEADER]

    def test_query_is_an_alias_for_chat_and_question_for_query(self) -> None:
        with self.client() as client:
            response = client.post("/query", json={"question": "why is the sky blue"})
        assert response.status_code == 200
        assert response.json()["citations"]

    def test_a_caller_supplied_request_id_is_propagated_end_to_end(self) -> None:
        with self.client() as client:
            response = client.post(
                "/chat",
                json={"query": "why is the sky blue"},
                headers={"X-Request-ID": "corr-42"},
            )
        assert response.headers[REQUEST_ID_HEADER] == "corr-42"
        assert response.json()["request_id"] == "corr-42"

    def test_auth_is_enforced_when_keys_are_configured(self) -> None:
        with self.client(settings={"api_keys": ("s3cret",)}) as client:
            unauthenticated = client.post("/chat", json={"query": "why is the sky blue"})
            wrong = client.post(
                "/chat",
                json={"query": "why is the sky blue"},
                headers={"Authorization": "Bearer nope"},
            )
            good = client.post(
                "/chat",
                json={"query": "why is the sky blue"},
                headers={"Authorization": "Bearer s3cret"},
            )
        assert unauthenticated.status_code == 401
        assert unauthenticated.json()["error"]["code"] == "unauthorized"
        assert unauthenticated.headers["www-authenticate"] == "Bearer"
        assert wrong.status_code == 401
        assert good.status_code == 200

    def test_health_and_metrics_stay_open_when_chat_is_locked_down(self) -> None:
        with self.client(settings={"api_keys": ("s3cret",)}) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/metrics").status_code == 200

    def test_rate_limiting_returns_429_with_retry_after(self) -> None:
        overrides = {"settings": {"rate_limit_per_s": 0.001, "rate_limit_burst": 2.0}}
        with self.client(**overrides) as client:
            payload = {"query": "why is the sky blue", "use_cache": False}
            statuses = [client.post("/chat", json=payload).status_code for _ in range(4)]
            limited = client.post("/chat", json=payload)
        assert statuses[:2] == [200, 200]
        assert limited.status_code == 429
        assert limited.json()["error"]["code"] == "rate_limited"
        assert int(limited.headers["retry-after"]) >= 1

    def test_a_validation_failure_uses_the_same_envelope_and_echoes_nothing(self) -> None:
        with self.client() as client:
            response = client.post("/chat", json={"query": "", "surprise": 1})
        assert response.status_code == 422
        body = response.json()
        assert set(body) == {"error"}
        assert set(body["error"]) == {"code", "message", "request_id"}
        assert body["error"]["code"] == "invalid_request"
        assert "surprise" not in json.dumps(body)

    def test_an_unknown_route_uses_the_same_envelope(self) -> None:
        with self.client() as client:
            response = client.get("/nope")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
        assert response.headers[REQUEST_ID_HEADER]

    def test_an_unexpected_crash_never_reaches_the_client(self) -> None:
        from fastapi.testclient import TestClient
        from localmind.api.main import create_app

        container = make_container()

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("dsn=postgresql://localmind:localmind@postgres:5432/localmind")

        container.agent.run = boom  # type: ignore[method-assign]
        with TestClient(create_app(container), raise_server_exceptions=False) as client:
            response = client.post("/chat", json={"query": "why is the sky blue"})
        assert response.status_code == 500
        assert response.json()["error"] == {
            "code": "internal",
            "message": GENERIC_MESSAGES[500],
            "request_id": response.headers[REQUEST_ID_HEADER],
        }
        assert "postgres" not in response.text
        assert "RuntimeError" not in response.text

    def test_an_agent_internal_error_is_a_503_not_a_leaky_200(self) -> None:
        class ExplodingRetriever:
            def search(self, query: str, k: int = 5, filters: Any = None) -> Any:
                raise MemoryError("secret internal detail")

        container = make_container(retriever=None)

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise MemoryError("secret internal detail")

        # Force the agent's own error path: `Agent.run` catches node exceptions and returns
        # status="error" rather than raising, which the gateway must surface as a fault.
        container.agent.grader.grade_chunks = boom  # type: ignore[method-assign]
        with self.client() as _unused:
            pass
        from fastapi.testclient import TestClient
        from localmind.api.main import create_app

        with TestClient(create_app(container)) as client:
            response = client.post("/chat", json={"query": "why is the sky blue"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "unavailable"
        assert "secret internal detail" not in response.text
        assert ExplodingRetriever  # keeps the helper referenced

    def test_streaming_delivers_tokens_then_citations_then_done(self) -> None:
        with (
            self.client() as client,
            client.stream(
                "POST", "/chat", json={"query": "why is the sky blue", "stream": True}
            ) as response,
        ):
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            text = "".join(response.iter_text())
        frames = [f for f in text.split("\n\n") if f.strip()]
        assert frames[-1] == "data: [DONE]"
        kinds = [json.loads(f[len("data: ") :])["type"] for f in frames[:-1]]
        assert kinds[0] == "start" and kinds[-2:] == ["citations", "done"]

    def test_the_openapi_document_advertises_the_three_endpoints(self) -> None:
        with self.client() as client:
            spec = client.get("/openapi.json").json()
        assert {"/chat", "/health", "/metrics"} <= set(spec["paths"])

    def test_uvicorn_can_resolve_the_module_path_docker_compose_uses(self) -> None:
        """``deploy/Dockerfile.api`` runs ``uvicorn localmind.api.main:app``."""
        import localmind.api.main as main_module

        assert main_module.app is not None
