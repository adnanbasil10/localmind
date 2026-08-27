"""Tests for the three Redis-backed cache layers (localmind/cache).

Everything here runs fully offline against InMemoryCacheBackend, per the task brief: Redis is not
running in this environment. Tests that require a real Redis are marked ``docker`` and use
``pytest.importorskip("redis")`` so they skip cleanly (not fail) when the optional dependency
isn't installed, which is the default state here.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from localmind.cache import (
    EmbeddingCache,
    HashingEmbedder,
    InMemoryCacheBackend,
    ResponseCache,
    SemanticCache,
    build_synthetic_queryset,
    choose_operating_point,
    run_and_write_benchmark,
    sweep_tau,
)
from localmind.cache.semantic_cache import _SweepPoint  # type: ignore[attr-defined]


def test_import_localmind_cache_is_offline_safe() -> None:
    """The task brief requires `import localmind.cache` to work with only numpy/pydantic on the
    path. If this module imported redis eagerly, collecting this test file would already fail."""
    import localmind.cache

    assert hasattr(localmind.cache, "SemanticCache")


# --------------------------------------------------------------------------- #
# InMemoryCacheBackend
# --------------------------------------------------------------------------- #
class TestInMemoryCacheBackend:
    def test_set_get_roundtrip(self) -> None:
        backend = InMemoryCacheBackend()
        backend.set("k", b"v")
        assert backend.get("k") == b"v"
        assert backend.exists("k") is True

    def test_missing_key_is_none(self) -> None:
        backend = InMemoryCacheBackend()
        assert backend.get("nope") is None
        assert backend.exists("nope") is False

    def test_delete(self) -> None:
        backend = InMemoryCacheBackend()
        backend.set("k", b"v")
        backend.delete("k")
        assert backend.get("k") is None

    def test_ttl_expiry_with_injected_clock(self) -> None:
        clock = {"t": 0.0}
        backend = InMemoryCacheBackend(clock=lambda: clock["t"])
        backend.set("k", b"v", ttl_seconds=10)
        assert backend.get("k") == b"v"
        clock["t"] = 9.9
        assert backend.get("k") == b"v"
        clock["t"] = 10.1
        assert backend.get("k") is None
        assert backend.exists("k") is False

    def test_no_ttl_never_expires(self) -> None:
        clock = {"t": 0.0}
        backend = InMemoryCacheBackend(clock=lambda: clock["t"])
        backend.set("k", b"v")
        clock["t"] = 1e9
        assert backend.get("k") == b"v"


# --------------------------------------------------------------------------- #
# Layer 1: EmbeddingCache
# --------------------------------------------------------------------------- #
class TestEmbeddingCache:
    def test_miss_then_hit(self) -> None:
        cache = EmbeddingCache(InMemoryCacheBackend(), model="fake-embed-v1")
        calls = []

        def embed_fn(text: str) -> np.ndarray:
            calls.append(text)
            return np.array([1.0, 2.0, 3.0], dtype=np.float32)

        vec1, hit1 = cache.get_or_compute("hello world", embed_fn)
        vec2, hit2 = cache.get_or_compute("hello world", embed_fn)

        assert hit1 is False
        assert hit2 is True
        assert calls == ["hello world"]  # embed_fn only ran once
        assert np.allclose(vec1, vec2)

    def test_stats_hit_rate_and_calls_avoided(self) -> None:
        cache = EmbeddingCache(InMemoryCacheBackend(), model="m")

        def embed_fn(text: str) -> np.ndarray:
            return np.zeros(4, dtype=np.float32)

        cache.get_or_compute("a", embed_fn)  # miss
        cache.get_or_compute("a", embed_fn)  # hit
        cache.get_or_compute("b", embed_fn)  # miss
        cache.get_or_compute("a", embed_fn)  # hit

        stats = cache.stats
        assert stats.hits == 2
        assert stats.misses == 2
        assert stats.total == 4
        assert stats.hit_rate == pytest.approx(0.5)
        assert stats.embedding_calls_avoided == 2

    def test_different_models_do_not_collide(self) -> None:
        backend = InMemoryCacheBackend()
        cache_a = EmbeddingCache(backend, model="model-a")
        cache_b = EmbeddingCache(backend, model="model-b")

        cache_a.put("text", np.array([1.0], dtype=np.float32))
        assert cache_b.get("text") is None  # different model -> different key -> miss

    def test_vector_roundtrips_with_correct_values(self) -> None:
        cache = EmbeddingCache(InMemoryCacheBackend(), model="m")
        original = np.array([0.5, -1.25, 3.0, 7.75], dtype=np.float32)
        cache.put("text", original)
        restored = cache.get("text")
        assert restored is not None
        assert np.allclose(restored, original)


# --------------------------------------------------------------------------- #
# Layer 2: ResponseCache
# --------------------------------------------------------------------------- #
class TestResponseCache:
    def test_miss_then_hit_calls_compute_fn_once(self) -> None:
        cache = ResponseCache(InMemoryCacheBackend())
        calls = {"n": 0}

        def compute() -> str:
            calls["n"] += 1
            time.sleep(0.001)
            return "the answer"

        r1, hit1 = cache.get_or_compute("what is x?", compute)
        r2, hit2 = cache.get_or_compute("what is x?", compute)

        assert hit1 is False
        assert hit2 is True
        assert r1 == r2 == "the answer"
        assert calls["n"] == 1

    def test_filters_and_config_are_part_of_the_key(self) -> None:
        cache = ResponseCache(InMemoryCacheBackend())
        calls = {"n": 0}

        def compute() -> str:
            calls["n"] += 1
            return f"answer #{calls['n']}"

        r1, hit1 = cache.get_or_compute("q", compute, filters={"lang": "en"})
        r2, hit2 = cache.get_or_compute("q", compute, filters={"lang": "fr"})

        assert hit1 is False
        assert hit2 is False  # different filters -> different key -> both misses
        assert r1 != r2
        assert calls["n"] == 2

    def test_stats_report_latency_saved(self) -> None:
        cache = ResponseCache(InMemoryCacheBackend())
        cache.put("q", "resp", compute_latency_ms=42.0)
        cache.get("q")
        cache.get("q")

        stats = cache.stats
        assert stats.hits == 2
        assert stats.misses == 0
        assert stats.latency_saved_ms_total == pytest.approx(84.0)
        assert stats.latency_saved_ms_mean == pytest.approx(42.0)

    def test_key_for_is_order_independent_for_filter_values_but_sensitive_to_content(self) -> None:
        k1 = ResponseCache.key_for("q", filters={"a": 1, "b": 2}, config={"temp": 0.0})
        k2 = ResponseCache.key_for("q", filters={"b": 2, "a": 1}, config={"temp": 0.0})
        k3 = ResponseCache.key_for("q", filters={"a": 1, "b": 3}, config={"temp": 0.0})
        assert k1 == k2  # dict key order must not matter
        assert k1 != k3  # but content must


# --------------------------------------------------------------------------- #
# Layer 3: SemanticCache
# --------------------------------------------------------------------------- #
class _DictEmbedder:
    """Test double giving exact, hand-picked cosine similarities -- used where a test needs to
    pin down tau-boundary behaviour precisely rather than rely on HashingEmbedder's output."""

    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self.mapping = mapping

    def embed(self, text: str) -> np.ndarray:
        return self.mapping[text]


class TestHashingEmbedder:
    def test_deterministic_for_same_seed(self) -> None:
        e1 = HashingEmbedder(seed=7)
        e2 = HashingEmbedder(seed=7)
        assert np.array_equal(e1.embed("hello world"), e2.embed("hello world"))

    def test_same_text_same_vector_repeated_calls(self) -> None:
        e = HashingEmbedder(seed=1)
        assert np.array_equal(e.embed("a query"), e.embed("a query"))

    def test_is_l2_normalized(self) -> None:
        e = HashingEmbedder(seed=1)
        v = e.embed("some reasonably long piece of text here")
        assert np.linalg.norm(v) == pytest.approx(1.0)

    def test_empty_text_is_zero_vector_and_does_not_crash(self) -> None:
        e = HashingEmbedder(seed=1)
        v = e.embed("")
        assert v.shape == (256,)
        assert np.linalg.norm(v) == pytest.approx(0.0)


class TestSemanticCache:
    def test_rejects_invalid_tau(self) -> None:
        with pytest.raises(ValueError):
            SemanticCache(InMemoryCacheBackend(), HashingEmbedder(seed=1), tau=0.0)
        with pytest.raises(ValueError):
            SemanticCache(InMemoryCacheBackend(), HashingEmbedder(seed=1), tau=1.5)

    def test_exact_repeat_query_hits(self) -> None:
        cache = SemanticCache(InMemoryCacheBackend(), HashingEmbedder(seed=1), tau=0.9)
        cache.store("What is the refund policy for the Basic plan?", "Basic: 7 days")

        result = cache.lookup("What is the refund policy for the Basic plan?")

        assert result.hit is True
        assert result.answer == "Basic: 7 days"
        assert result.similarity == pytest.approx(1.0)

    def test_unrelated_query_misses(self) -> None:
        cache = SemanticCache(InMemoryCacheBackend(), HashingEmbedder(seed=1), tau=0.8)
        cache.store("What is the refund policy for the Basic plan?", "Basic: 7 days")

        result = cache.lookup("How do I make pancakes at home?")

        assert result.hit is False
        assert result.answer is None

    def test_empty_cache_misses_cleanly(self) -> None:
        cache = SemanticCache(InMemoryCacheBackend(), HashingEmbedder(seed=1), tau=0.9)
        result = cache.lookup("anything")
        assert result.hit is False
        assert result.similarity == 0.0
        assert result.matched_query is None

    def test_tau_boundary_with_exact_similarity(self) -> None:
        """Pin a known 0.6-cosine pair via a dict embedder and check the tau threshold is
        applied as `similarity >= tau`, not `>`."""
        v1 = np.array([1.0, 0.0], dtype=np.float64)
        v2 = np.array([0.6, 0.8], dtype=np.float64)  # cosine(v1, v2) == 0.6 exactly
        embedder = _DictEmbedder({"canonical": v1, "probe": v2})

        cache_at_boundary = SemanticCache(InMemoryCacheBackend(), embedder, tau=0.6)
        cache_at_boundary.store("canonical", "the answer")
        assert cache_at_boundary.lookup("probe").hit is True  # 0.6 >= 0.6

        cache_above_boundary = SemanticCache(InMemoryCacheBackend(), embedder, tau=0.61)
        cache_above_boundary.store("canonical", "the answer")
        assert cache_above_boundary.lookup("probe").hit is False  # 0.6 < 0.61

    def test_record_verdict_and_false_hit_rate(self) -> None:
        cache = SemanticCache(InMemoryCacheBackend(), HashingEmbedder(seed=1), tau=0.5)
        assert cache.stats.false_hit_rate is None  # nothing recorded yet

        cache.record_verdict(was_correct=True)
        cache.record_verdict(was_correct=False)
        cache.record_verdict(was_correct=False)

        stats = cache.stats
        assert stats.verdicts_recorded == 3
        assert stats.false_hits_recorded == 2
        assert stats.false_hit_rate == pytest.approx(2 / 3)

    def test_lookups_and_hits_are_counted(self) -> None:
        cache = SemanticCache(InMemoryCacheBackend(), HashingEmbedder(seed=1), tau=0.9)
        cache.store("What is the refund policy for the Basic plan?", "Basic: 7 days")

        cache.lookup("What is the refund policy for the Basic plan?")  # hit
        cache.lookup("How do I make pancakes at home?")  # miss

        stats = cache.stats
        assert stats.lookups == 2
        assert stats.hits == 1
        assert stats.hit_rate == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# The tau sweep: this is where the honest false-hit measurement lives
# --------------------------------------------------------------------------- #
class TestTauSweep:
    def test_build_synthetic_queryset_is_well_formed(self) -> None:
        population, evaluation = build_synthetic_queryset()
        assert len(population) >= 10
        assert len(evaluation) >= 20
        population_intents = {item.intent_id for item in population}
        for item in evaluation:
            # every evaluation item's answer must correspond to *some* real intent's answer --
            # otherwise "false hit" would be checking against a typo, not a real ground truth.
            assert item.intent_id in population_intents or item.intent_id.startswith("refund_")

    def test_sweep_tau_requires_at_least_min_seeds(self) -> None:
        with pytest.raises(ValueError):
            sweep_tau(seeds=(1, 2))  # CONVENTIONS.md rule 5: >=3 seeds

    def test_sweep_tau_endpoints_hit_rate_is_monotone_non_increasing(self) -> None:
        result = sweep_tau(taus=[0.80, 0.99], seeds=(1, 2, 3))
        loose = result.points_by_seed[1][0]
        strict = result.points_by_seed[1][1]
        assert loose.tau == 0.80
        assert strict.tau == 0.99
        assert loose.hit_rate >= strict.hit_rate

    def test_false_hit_definition_is_answer_mismatch_not_similarity(self) -> None:
        """The task brief is explicit: a false hit must be defined by comparing the SERVED answer
        to the new query's ground-truth answer, not merely "similarity below some other
        threshold". Exercise `_SweepPoint` bookkeeping directly against a hand-built scenario."""
        population, evaluation = build_synthetic_queryset()
        embedder = HashingEmbedder(seed=1)
        cache = SemanticCache(InMemoryCacheBackend(), embedder, tau=1.0, model="test")
        for item in population:
            cache.store(item.query, item.answer)

        # Find an evaluation item known to collide with a *different* intent's cached answer at
        # a high similarity: the context-ambiguous duplicate is guaranteed to do this.
        ambiguous = [item for item in evaluation if item.query == "What's the refund window?"]
        assert len(ambiguous) == 3
        wrong_context_items = [
            item for item in ambiguous if item.intent_id != "refund_policy_basic"
        ]
        assert wrong_context_items  # at least one of the three is NOT the cached (basic) context

        for item in wrong_context_items:
            match = cache.nearest(item.query)
            assert match is not None
            assert match.similarity == pytest.approx(1.0)  # identical text
            assert match.answer != item.answer  # but it's the WRONG answer for this context

    def test_sweep_tau_rows_carry_bootstrap_ci(self) -> None:
        from localmind.eval.stats import Estimate

        result = sweep_tau(taus=[0.85, 0.90], seeds=(1, 2, 3))
        from localmind.cache.semantic_cache import _rows_from_sweep

        rows = _rows_from_sweep(result)
        assert len(rows) == 2
        for row in rows:
            assert isinstance(row.values["hit_rate"], Estimate)
            assert isinstance(row.values["false_hit_rate_of_hits"], Estimate)
            assert isinstance(row.values["false_hit_rate_of_total"], Estimate)
            assert 0.0 <= row.values["hit_rate"].mean <= 1.0
            assert 0.0 <= row.values["false_hit_rate_of_hits"].mean <= 1.0

    def test_choose_operating_point_picks_a_swept_tau(self) -> None:
        result = sweep_tau(seeds=(1, 2, 3))
        from localmind.cache.semantic_cache import _rows_from_sweep

        rows = _rows_from_sweep(result)
        op = choose_operating_point(rows)
        assert op["tau"] in result.taus
        assert 0.0 <= op["hit_rate"] <= 1.0
        assert 0.0 <= op["false_hit_rate_of_total"] <= 1.0
        assert isinstance(op["meets_threshold"], bool)

    def test_choose_operating_point_prefers_higher_hit_rate_among_ties(self) -> None:
        """Two rows tied at the minimum false_hit_rate_of_total: the tie must break toward the
        higher hit_rate (loosest tau that is exactly as safe), not an arbitrary one."""
        from localmind.eval.stats import MetricRow, bootstrap_ci

        rows = [
            MetricRow(
                name="tau=0.90",
                values={
                    "hit_rate": bootstrap_ci([0.30, 0.30, 0.30]),
                    "false_hit_rate_of_hits": bootstrap_ci([0.10, 0.10, 0.10]),
                    "false_hit_rate_of_total": bootstrap_ci([0.03, 0.03, 0.03]),
                },
                extra={"tau": 0.90},
            ),
            MetricRow(
                name="tau=0.95",
                values={
                    "hit_rate": bootstrap_ci([0.10, 0.10, 0.10]),
                    "false_hit_rate_of_hits": bootstrap_ci([0.30, 0.30, 0.30]),
                    "false_hit_rate_of_total": bootstrap_ci([0.03, 0.03, 0.03]),
                },
                extra={"tau": 0.95},
            ),
        ]
        op = choose_operating_point(rows)
        assert op["tau"] == 0.90  # tied on false_hit_rate_of_total -> higher hit_rate wins

    def test_run_and_write_benchmark_matches_conventions_schema(self, tmp_path) -> None:
        output = tmp_path / "semantic_cache.json"
        payload = run_and_write_benchmark(output, taus=[0.85, 0.90, 0.95], seeds=(1, 2, 3))

        assert output.exists()
        import json

        on_disk = json.loads(output.read_text(encoding="utf-8"))
        assert on_disk == payload

        for key in ("name", "hardware", "seeds", "rows", "ci"):
            assert key in payload
        assert payload["ci"] == "bootstrap95"
        assert len(payload["seeds"]) >= 3
        assert len(payload["rows"]) == 3
        assert "operating_point" in payload
        assert "tau" in payload["operating_point"]

    def test_sweep_point_dataclass_fields_are_consistent(self) -> None:
        point = _SweepPoint(
            tau=0.9,
            hit_rate=0.5,
            false_hit_rate_of_hits=0.1,
            false_hit_rate_of_total=0.05,
            hits=10,
            false_hits=1,
            n=20,
        )
        assert point.hits / point.n == pytest.approx(point.hit_rate)
        assert point.false_hits / point.hits == pytest.approx(point.false_hit_rate_of_hits)


# --------------------------------------------------------------------------- #
# Real Redis (marked docker; skips cleanly when `redis` isn't installed, which is the default
# state in this environment per the task brief)
# --------------------------------------------------------------------------- #
@pytest.mark.docker
def test_redis_cache_backend_roundtrip_against_a_live_redis() -> None:
    pytest.importorskip("redis")
    from localmind.cache import RedisCacheBackend

    backend = RedisCacheBackend("redis://localhost:6379/0")
    try:
        backend.set("localmind:test:key", b"value", ttl_seconds=5)
        assert backend.get("localmind:test:key") == b"value"
    finally:
        backend.delete("localmind:test:key")


def test_redis_cache_backend_raises_a_clear_error_without_the_redis_package(monkeypatch) -> None:
    """Not docker-marked: this tests the *absence* path, which is the default state of this
    environment (task brief: redis is an optional dependency, likely not installed) -- it must
    run, and pass, with no Docker and no redis package at all. Forcing `import redis` to fail via
    sys.modules makes this deterministic even if `redis` happens to be installed."""
    import sys

    monkeypatch.setitem(sys.modules, "redis", None)
    from localmind.cache import RedisCacheBackend

    with pytest.raises(RuntimeError, match="redis"):
        RedisCacheBackend("redis://localhost:6379/0")
