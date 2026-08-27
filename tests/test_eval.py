"""Tests for the evaluation harness (implementation.md §14).

The harness is the thing every other phase's claims rest on, so these tests
lean hard on *numerical* validation: the bootstrap CI is checked against a
distribution with a known standard error, Cohen's kappa against a
hand-computed 2x2 table, and the Wilcoxon exact p-value against brute-force
enumeration of the null distribution.
"""

from __future__ import annotations

import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from localmind.eval import stats
from localmind.eval.datasets import (
    CATEGORIES,
    SAMPLE_GOLDEN_PATH,
    CorpusChunk,
    GoldenDataset,
    GoldenQuestion,
    load_corpus,
    load_golden,
    load_judge_labels,
    load_manifest,
)
from localmind.eval.generate_golden import (
    DeterministicFakeTeacher,
    GoldenGenConfig,
    GoldenGenerator,
    build_prompt,
    parse_candidate,
    promote_candidates,
)
from localmind.eval.generation import (
    GeneratedAnswer,
    citation_scores,
    detect_refusal,
    evaluate_generation,
    faithfulness_deterministic,
    split_claims,
    token_f1,
)
from localmind.eval.judge_calibration import (
    KAPPA_TRUST_THRESHOLD,
    CoinFlipJudge,
    HeuristicJudge,
    OracleJudge,
    PositionBiasedJudge,
    build_calibration_fixture,
    calibrate_judge,
    flip_verdict,
)
from localmind.eval.report import (
    LocalBaselineSource,
    check_regression,
    extract_metric,
    regenerate_benchmarks_md,
)
from localmind.eval.retrieval import (
    BM25Retriever,
    RetrievalEvalConfig,
    RetrievedItem,
    StaticRunRetriever,
    average_precision,
    evaluate_retrieval,
    load_eval_config,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from localmind.eval.stats import (
    Estimate,
    MetricRow,
    aggregate_seeds,
    benchmark_json,
    bootstrap_ci,
    bootstrap_paired,
    cohens_kappa,
    compare_configs,
    format_metric,
    require_estimate,
    require_seeds,
    wilcoxon_signed_rank,
    wilson_ci,
)
from localmind.eval.system import (
    RequestRecord,
    ToolCall,
    evaluate_system,
    hardware_string,
    measure_request,
)
from pydantic import ValidationError

SEED = 1234


# =========================================================================== #
# Import weight
# =========================================================================== #
def test_import_localmind_eval_needs_only_numpy_and_pydantic():
    """`import localmind.eval` must not drag in yaml, httpx, torch or scipy."""
    code = (
        "import sys; "
        "import localmind.eval as e; "
        "assert e.Estimate is not None; "
        "heavy = [m for m in ('yaml', 'httpx', 'torch', 'scipy', 'transformers') "
        "         if m in sys.modules]; "
        "print(heavy)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert out.stdout.strip() == "[]", f"heavy modules imported eagerly: {out.stdout}"


def test_lazy_submodule_access():
    import localmind.eval as e

    assert e.retrieval.recall_at_k is not None
    assert e.generation.token_f1 is not None
    with pytest.raises(AttributeError):
        _ = e.does_not_exist


# =========================================================================== #
# stats: Estimate cannot become a bare number (§20 rule 5)
# =========================================================================== #
def test_estimate_refuses_float_coercion():
    est = Estimate(mean=0.5, lo=0.4, hi=0.6, n=10)
    with pytest.raises(TypeError, match="rule 5"):
        float(est)
    with pytest.raises(TypeError, match="bare number"):
        f"{est:.3f}"
    assert str(est) == "0.500 [0.400, 0.600]"
    assert f"{est}" == "0.500 [0.400, 0.600]"
    assert est.format(2) == "0.50 [0.40, 0.60]"


def test_require_estimate_rejects_floats():
    with pytest.raises(TypeError, match="confidence interval"):
        require_estimate(0.5)
    with pytest.raises(TypeError):
        format_metric(0.5)  # type: ignore[arg-type]


def test_benchmark_json_rejects_bare_numbers():
    row = MetricRow(name="x", values={"ndcg@10": 0.9})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="rule 5"):
        benchmark_json("bench", hardware="cpu", seeds=[1], rows=[row])


def test_estimate_validates_bounds():
    with pytest.raises(ValueError, match="must not exceed"):
        Estimate(mean=0.5, lo=0.9, hi=0.1, n=3)


def test_require_seeds_enforces_three():
    assert require_seeds([1, 2, 3]) == [1, 2, 3]
    with pytest.raises(ValueError, match="rule 5"):
        require_seeds([1, 1, 2])


# =========================================================================== #
# stats: bootstrap validated against a known distribution
# =========================================================================== #
def test_bootstrap_ci_matches_analytic_standard_error():
    """N(0,1), n=2000 -> half-width should land near 1.96/sqrt(n) = 0.0438."""
    x = np.random.default_rng(7).normal(0.0, 1.0, size=2000)
    est = bootstrap_ci(x, seed=42, n_resamples=10_000)
    analytic = 1.96 / math.sqrt(2000)
    assert est.half_width == pytest.approx(analytic, rel=0.10)
    assert est.lo < 0.0 < est.hi
    assert est.n == 2000
    assert est.method == "bootstrap-percentile"


def test_bootstrap_ci_coverage_is_about_95_percent():
    """Nominal coverage: over many replicates the CI should contain the truth ~95%."""
    hits = 0
    trials = 120
    for i in range(trials):
        sample = np.random.default_rng(5000 + i).normal(0.0, 1.0, size=200)
        est = bootstrap_ci(sample, seed=i, n_resamples=1_000)
        hits += est.lo <= 0.0 <= est.hi
    assert 0.88 <= hits / trials <= 1.0


def test_bootstrap_is_deterministic_for_a_seed():
    x = np.random.default_rng(0).normal(size=200)
    a = bootstrap_ci(x, seed=99, n_resamples=1_000)
    b = bootstrap_ci(x, seed=99, n_resamples=1_000)
    c = bootstrap_ci(x, seed=100, n_resamples=1_000)
    assert a.to_dict() == b.to_dict()
    assert (a.lo, a.hi) != (c.lo, c.hi)


def test_bootstrap_and_wilson_agree_on_a_proportion():
    x = np.random.default_rng(3).binomial(1, 0.3, 1000).astype(float)
    boot = bootstrap_ci(x, seed=1, n_resamples=5_000)
    wil = wilson_ci(int(x.sum()), 1000)
    assert boot.mean == pytest.approx(wil.mean)
    assert boot.lo == pytest.approx(wil.lo, abs=0.01)
    assert boot.hi == pytest.approx(wil.hi, abs=0.01)


def test_bootstrap_percentile_statistic():
    x = np.arange(1.0, 101.0)
    est = bootstrap_ci(x, seed=SEED, n_resamples=2_000, statistic=stats.percentile_stat(95))
    assert est.mean == pytest.approx(np.percentile(x, 95))
    assert est.lo < est.mean < est.hi


def test_bootstrap_rejects_empty_and_handles_singleton():
    with pytest.raises(ValueError):
        bootstrap_ci([])
    est = bootstrap_ci([0.7])
    assert (est.mean, est.lo, est.hi) == (0.7, 0.7, 0.7)
    assert est.method == "degenerate-n1"


# =========================================================================== #
# stats: Cohen's kappa against a hand-computed table
# =========================================================================== #
def test_cohens_kappa_hand_computed_2x2():
    """Confusion [[20, 5], [10, 15]] over n=50.

    p_o = (20+15)/50 = 0.70
    marginals: A(Yes)=25/50=0.5, B(Yes)=30/50=0.6
    p_e = 0.5*0.6 + 0.5*0.4 = 0.50
    kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40
    """
    a = ["Y"] * 20 + ["Y"] * 5 + ["N"] * 10 + ["N"] * 15
    b = ["Y"] * 20 + ["N"] * 5 + ["Y"] * 10 + ["N"] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.40, abs=1e-12)


def test_cohens_kappa_edge_cases():
    assert cohens_kappa(["A", "B", "A"], ["A", "B", "A"]) == pytest.approx(1.0)
    assert cohens_kappa(["A", "A"], ["B", "B"]) == pytest.approx(0.0)
    # Perfectly opposed raters on a balanced 2x2 -> kappa = -1.
    assert cohens_kappa(["A", "B"], ["B", "A"]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        cohens_kappa(["A"], ["A", "B"])


def test_kappa_ci_brackets_the_point_estimate():
    a = ["Y"] * 20 + ["Y"] * 5 + ["N"] * 10 + ["N"] * 15
    b = ["Y"] * 20 + ["N"] * 5 + ["Y"] * 10 + ["N"] * 15
    est = stats.kappa_ci(a, b, seed=5, n_resamples=1_000)
    assert est.mean == pytest.approx(0.40)
    assert est.lo < 0.40 < est.hi


# =========================================================================== #
# stats: Wilcoxon against brute-force enumeration
# =========================================================================== #
def _brute_force_wilcoxon_p(d: list[float]) -> float:
    arr = np.asarray(d, dtype=float)
    arr = arr[arr != 0.0]
    ranks = stats._average_ranks(np.abs(arr))
    w_plus = ranks[arr > 0].sum()
    n = len(arr)
    draws = np.array(
        [
            sum(ranks[i] for i in range(n) if signs[i])
            for signs in itertools.product([0, 1], repeat=n)
        ]
    )
    return float(min(1.0, 2.0 * min((draws <= w_plus).mean(), (draws >= w_plus).mean())))


@pytest.mark.parametrize(
    "d",
    [
        [5.0],
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [-3.0, 1.0, 4.0, -2.0, 7.0, 6.0, -5.0],
        [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, -1.0],
    ],
)
def test_wilcoxon_exact_matches_brute_force(d):
    res = wilcoxon_signed_rank(d, use_scipy=False)
    assert res.method == "exact"
    assert res.p_value == pytest.approx(_brute_force_wilcoxon_p(d), abs=1e-12)


def test_wilcoxon_hand_checkable_small_cases():
    # n=1: W+ can only be 0 or 1, so the two-sided p is 2 * 1/2 = 1.0
    assert wilcoxon_signed_rank([5.0], use_scipy=False).p_value == pytest.approx(1.0)
    # n=2, both positive: W+=3 is the maximum of {0,1,2,3}; p = 2 * 1/4 = 0.5
    assert wilcoxon_signed_rank([1.0, 2.0], use_scipy=False).p_value == pytest.approx(0.5)
    # n=6, all positive: p = 2 * 1/64 = 0.03125
    res = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], use_scipy=False)
    assert res.p_value == pytest.approx(2 / 64)
    assert res.statistic == 0.0


def test_wilcoxon_falls_back_to_normal_approx_on_ties():
    d = [15.0, -7.0, 5.0, 20.0, 0.0, -9.0, 17.0, -12.0, 5.0, -10.0]
    res = wilcoxon_signed_rank(d, use_scipy=False)
    assert res.method == "normal-approx"  # |d| has a tie at 5
    assert res.n_effective == 9  # the zero is dropped
    assert res.statistic == pytest.approx(18.0)
    # The tie-corrected approximation should stay close to the exact answer.
    assert res.p_value == pytest.approx(_brute_force_wilcoxon_p(d), abs=0.02)


def test_wilcoxon_zero_differences_only():
    res = wilcoxon_signed_rank([0.0, 0.0, 0.0], use_scipy=False)
    assert res.n_effective == 0
    assert res.p_value == 1.0


def test_wilcoxon_rejects_misaligned_pairs():
    with pytest.raises(ValueError, match="align"):
        wilcoxon_signed_rank([1.0, 2.0], [1.0])


# =========================================================================== #
# stats: paired comparison + seeds
# =========================================================================== #
def test_bootstrap_paired_detects_a_real_shift():
    rng = np.random.default_rng(11)
    base = rng.normal(0.5, 0.1, size=200)
    better = base + 0.05
    res = bootstrap_paired(better, base, seed=SEED, n_resamples=4_000)
    assert res.diff.mean == pytest.approx(0.05, abs=1e-9)
    assert res.significant
    assert res.p_value < 0.05


def test_bootstrap_paired_finds_nothing_when_there_is_nothing():
    rng = np.random.default_rng(12)
    a = rng.normal(0.5, 0.1, size=200)
    b = a + rng.normal(0.0, 0.01, size=200)
    res = bootstrap_paired(a, b, seed=SEED, n_resamples=4_000)
    assert not res.significant
    assert res.p_value > 0.05


def test_compare_configs_reports_both_tests():
    rng = np.random.default_rng(13)
    a = rng.normal(0.6, 0.08, size=60)
    b = a - 0.04
    report = compare_configs("a", a, "b", b, metric="ndcg@10", seed=SEED, n_resamples=2_000)
    assert report.paired.significant
    assert report.wilcoxon.p_value < 0.05
    assert report.verdict == "a wins on ndcg@10"
    md = report.to_markdown()
    assert "ndcg@10" in md and "[" in md


def test_aggregate_seeds_requires_three_and_widens_with_variance():
    tight = aggregate_seeds({1: [0.80], 2: [0.81], 3: [0.79]})
    loose = aggregate_seeds({1: [0.60], 2: [0.81], 3: [0.99]})
    assert tight.mean == pytest.approx(0.80)
    assert loose.half_width > tight.half_width
    assert "3-seeds" in tight.method
    with pytest.raises(ValueError, match="3 distinct seeds"):
        aggregate_seeds({1: [0.8], 2: [0.9]})


def test_mean_ci_t_matches_textbook_arithmetic():
    # n=3, mean 0.80, sd 0.01 -> se = 0.01/sqrt(3), t(2, .975) = 4.3027
    est = stats.mean_ci_t([0.79, 0.80, 0.81])
    se = 0.01 / math.sqrt(3)
    assert est.mean == pytest.approx(0.80)
    assert est.half_width == pytest.approx(4.3027 * se, rel=1e-4)


# =========================================================================== #
# Golden dataset: schema, coverage, hash pinning
# =========================================================================== #
def test_sample_golden_set_loads_and_covers_every_category():
    ds = load_golden()
    assert 20 <= len(ds) <= 40
    assert ds.categories == set(CATEGORIES), f"missing: {set(CATEGORIES) - ds.categories}"
    assert ds.check_coverage() == []
    assert ds.unanswerable, "a set with no refusals cannot measure refusal accuracy"
    assert any(q.is_adversarial for q in ds.questions)
    assert {q.difficulty for q in ds.questions} == {"easy", "medium", "hard"}


def test_golden_hash_is_pinned_and_order_insensitive():
    ds = load_golden()
    manifest = load_manifest()
    entry = manifest.entries["golden_sample"]
    assert ds.content_hash == entry.sha256
    assert entry.n_records == len(ds)
    shuffled = GoldenDataset(
        name=ds.name, version=ds.version, questions=list(reversed(ds.questions))
    )
    assert shuffled.content_hash == ds.content_hash


def test_manifest_verifies_every_committed_file():
    assert load_manifest().verify() == []


def test_golden_hash_mismatch_is_loud(tmp_path):
    ds = load_golden()
    with pytest.raises(ValueError, match="hash drifted"):
        load_golden(SAMPLE_GOLDEN_PATH, expect_hash="0" * 64)
    assert ds.content_hash != "0" * 64


def test_unverified_questions_are_refused_by_the_loader(tmp_path):
    q = GoldenQuestion(
        id="x-1",
        question="What is the refund window?",
        expected_answer="30 days",
        expected_doc_ids=["d"],
        expected_chunk_ids=["d#0"],
        difficulty="easy",
        category="factoid",
        verified=False,
    )
    path = tmp_path / "cands.jsonl"
    path.write_text(q.canonical() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hand-verified"):
        load_golden(path)
    relaxed = load_golden(path, require_verified=False)
    assert len(relaxed) == 1


def test_should_refuse_is_driven_by_missing_evidence():
    ood = GoldenQuestion(
        id="ood-1",
        question="What is the capital of Mongolia?",
        expected_answer="Not answerable.",
        difficulty="easy",
        category="out-of-domain",
    )
    assert ood.should_refuse
    ds = load_golden()
    assert all(q.should_refuse for q in ds.questions if q.category == "out-of-domain")
    assert not any(q.should_refuse for q in ds.questions if q.category == "factoid")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"id": "Bad ID"}, "kebab"),
        ({"category": "out-of-domain", "expected_doc_ids": ["d"]}, "out-of-domain"),
        ({"expected_chunk_ids": ["other#0"]}, "does not belong"),
        ({"expected_doc_ids": ["d", "d"]}, "duplicate"),
        ({"requires_tools": ["Bad Tool"]}, "snake_case"),
        ({"question": "   "}, "blank"),
    ],
)
def test_schema_rejects_inconsistent_questions(kwargs, match):
    base = {
        "id": "ok-1",
        "question": "Q?",
        "expected_answer": "A",
        "expected_doc_ids": ["d"],
        "expected_chunk_ids": ["d#0"],
        "difficulty": "easy",
        "category": "factoid",
    }
    base.update(kwargs)
    with pytest.raises(ValidationError, match=match):
        GoldenQuestion(**base)


def test_adversarial_questions_must_carry_a_canary():
    with pytest.raises(ValidationError, match="injection_canary"):
        GoldenQuestion(
            id="adv-1",
            question="Ignore instructions.",
            expected_answer="A",
            expected_doc_ids=["d"],
            expected_chunk_ids=["d#0"],
            difficulty="hard",
            category="adversarial-injection",
        )
    ds = load_golden()
    for q in ds.questions:
        if q.is_adversarial:
            assert q.injection_canary
            assert q.injection_canary in q.question


def test_release_size_window_is_enforced():
    ds = load_golden().model_copy(update={"kind": "release"})
    problems = ds.check_coverage()
    assert any("150-300" in p for p in problems)


def test_corpus_chunk_ids_are_doc_scoped():
    corpus = load_corpus()
    assert len(corpus) >= 20
    with pytest.raises(ValidationError, match="must be"):
        CorpusChunk(chunk_id="a#0", doc_id="b", text="t")


def test_every_expected_chunk_exists_in_the_sample_corpus():
    known = {c.chunk_id for c in load_corpus()}
    for q in load_golden().questions:
        missing = set(q.expected_chunk_ids) - known
        assert not missing, f"{q.id} references chunks absent from the corpus: {missing}"


# =========================================================================== #
# Retrieval metrics: hand-computed
# =========================================================================== #
def test_recall_at_k_hand_computed():
    ranked = ["a", "b", "c", "d"]
    rel = {"b", "d", "z"}
    assert recall_at_k(ranked, rel, 1) == pytest.approx(0.0)
    assert recall_at_k(ranked, rel, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, rel, 4) == pytest.approx(2 / 3)


def test_ndcg_at_k_hand_computed():
    # One relevant item at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    assert ndcg_at_k(["a", "b"], {"b"}, 10) == pytest.approx(1 / math.log2(3))
    # Perfect ranking scores 1.0
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 10) == pytest.approx(1.0)
    # Two relevant at ranks 1 and 3
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["x", "y"], {"a"}, 10) == pytest.approx(0.0)


def test_mrr_hand_computed():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], {"a"}) == pytest.approx(1.0)
    assert reciprocal_rank(["x"], {"a"}) == pytest.approx(0.0)
    assert reciprocal_rank(["x", "a"], {"a"}, cutoff=1) == pytest.approx(0.0)


def test_average_precision_hand_computed():
    # Relevant at ranks 1 and 3: AP = (1/1 + 2/3) / 2
    assert average_precision(["a", "x", "b"], {"a", "b"}) == pytest.approx((1.0 + 2 / 3) / 2)
    assert average_precision(["a", "b"], {"a", "b"}) == pytest.approx(1.0)
    assert average_precision(["x", "y"], {"a"}) == pytest.approx(0.0)
    # Cutoff smaller than the relevant set: perfect prefix still scores 1.0
    assert average_precision(["a", "b", "c"], {"a", "b", "c"}, cutoff=2) == pytest.approx(1.0)


def test_metrics_reject_empty_relevant_sets():
    for fn in (recall_at_k, ndcg_at_k):
        with pytest.raises(ValueError, match="undefined"):
            fn(["a"], set(), 5)
    with pytest.raises(ValueError, match="undefined"):
        reciprocal_rank(["a"], set())
    with pytest.raises(ValueError, match="undefined"):
        average_precision(["a"], set())


# =========================================================================== #
# Retrieval: BM25 + end-to-end evaluation
# =========================================================================== #
def test_bm25_is_deterministic_and_finds_the_obvious_chunk():
    corpus = load_corpus()
    r = BM25Retriever(chunks=corpus)
    first = [i.chunk_id for i in r.search("How often are encryption keys rotated?", 5)]
    second = [i.chunk_id for i in r.search("How often are encryption keys rotated?", 5)]
    assert first == second
    assert first[0] == "handbook-security#1"
    assert r.search("zzzqqq nonexistentterm", 5) == []


def test_evaluate_retrieval_end_to_end_on_the_sample_set():
    ds = load_golden()
    report = evaluate_retrieval(ds, BM25Retriever(chunks=load_corpus()), seed=SEED, n_resamples=500)
    for key in ("recall@1", "recall@5", "recall@10", "recall@20", "ndcg@10", "mrr", "map"):
        assert key in report.metrics
        est = report.metrics[key]
        assert 0.0 <= est.lo <= est.mean <= est.hi <= 1.0
    # Unanswerable questions carry no relevance judgements and must be excluded.
    assert report.counts["skipped_unanswerable"] == len(ds.unanswerable)
    assert report.counts["scored"] == len(ds) - len(ds.unanswerable)
    assert report.metrics["ndcg@10"].mean > 0.5
    assert report.elapsed_s < 180.0, "the CI gate has a ~3 minute budget"


def test_retrieval_is_reproducible_bit_for_bit():
    ds = load_golden()
    corpus = load_corpus()
    a = evaluate_retrieval(ds, BM25Retriever(chunks=corpus), seed=SEED, n_resamples=300)
    b = evaluate_retrieval(ds, BM25Retriever(chunks=corpus), seed=SEED, n_resamples=300)
    assert a.per_question == b.per_question
    assert {k: v.to_dict() for k, v in a.metrics.items()} == {
        k: v.to_dict() for k, v in b.metrics.items()
    }


def test_a_worse_retriever_scores_worse():
    ds = load_golden()
    good = evaluate_retrieval(ds, BM25Retriever(chunks=load_corpus()), seed=SEED, n_resamples=300)
    runs = {
        q.id: [RetrievedItem(chunk_id="handbook-onboarding#1", score=1.0)] for q in ds.questions
    }
    bad = evaluate_retrieval(
        ds, StaticRunRetriever(runs=runs, name="always-wrong"), seed=SEED, n_resamples=300
    )
    assert bad.metrics["ndcg@10"].mean < good.metrics["ndcg@10"].mean
    comparison = compare_configs(
        "good",
        good.values("ndcg@10"),
        "bad",
        bad.values("ndcg@10"),
        metric="ndcg@10",
        seed=SEED,
        n_resamples=1_000,
    )
    assert comparison.paired.significant


def test_doc_level_relevance_fallback():
    ds = GoldenDataset(
        name="t",
        version="v1",
        questions=[
            GoldenQuestion(
                id="q-1",
                question="refund window",
                expected_answer="30 days",
                expected_doc_ids=["handbook-billing"],
                difficulty="easy",
                category="factoid",
            )
        ],
    )
    report = evaluate_retrieval(ds, BM25Retriever(chunks=load_corpus()), seed=SEED, n_resamples=100)
    assert report.granularity == "doc"
    assert report.metrics["recall@5"].mean == pytest.approx(1.0)


def test_retrieval_config_tolerates_a_foreign_schema(tmp_path):
    cfg_path = tmp_path / "default.yaml"
    cfg_path.write_text(
        "name: hybrid-rrf\n"
        "top_k: 10\n"
        "arms:\n  - bm25\n  - dense\n"
        "rrf_k: 60\n"
        "reranker:\n  model: bge-reranker\n  top_n: 10\n"
        "eval:\n  seed: 7\n  n_resamples: 100\n",
        encoding="utf-8",
    )
    cfg = load_eval_config(cfg_path)
    assert cfg.name == "hybrid-rrf"
    assert cfg.top_k == 10
    assert cfg.seed == 7
    assert cfg.n_resamples == 100


def test_missing_config_falls_back_to_defaults(tmp_path):
    cfg = load_eval_config(tmp_path / "absent.yaml")
    assert cfg == RetrievalEvalConfig()
    assert load_eval_config(None) == RetrievalEvalConfig()


def test_retrieval_cli_writes_a_conventions_shaped_artifact(tmp_path):
    from localmind.eval.retrieval import main as retrieval_main

    out = tmp_path / "retrieval.json"
    rc = retrieval_main(["--out", str(out), "--seed", "5"])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) >= {"name", "hardware", "seeds", "rows", "ci"}
    assert payload["ci"] == "bootstrap95"
    assert payload["seeds"] == [5]
    est = extract_metric(payload, "ndcg@10")
    assert est is not None and est.lo <= est.mean <= est.hi


# =========================================================================== #
# Generation metrics
# =========================================================================== #
def test_token_f1_hand_computed():
    assert token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert token_f1("cat dog", "elephant") == pytest.approx(0.0)
    # pred={cat,dog}, ref={cat,bird}: overlap 1 -> P=R=0.5 -> F1=0.5
    assert token_f1("cat dog", "cat bird") == pytest.approx(0.5)
    # Articles are stripped, so these normalise to the same tokens.
    assert token_f1("The 30 days", "30 days") == pytest.approx(1.0)


def test_split_claims():
    assert split_claims("One fact here. Another fact there.") == [
        "One fact here.",
        "Another fact there.",
    ]
    assert len(split_claims("Short.")) == 1


def test_citation_precision_and_recall_hand_computed():
    q = GoldenQuestion(
        id="q-1",
        question="Q?",
        expected_answer="A",
        expected_doc_ids=["d"],
        expected_chunk_ids=["d#0", "d#1"],
        difficulty="easy",
        category="factoid",
    )
    ans = GeneratedAnswer(id="q-1", answer="A", cited_chunk_ids=["d#0", "d#9"])
    p, r = citation_scores(ans, q)
    assert p == pytest.approx(0.5)  # 1 of 2 cited chunks is correct
    assert r == pytest.approx(0.5)  # 1 of 2 expected chunks was cited
    perfect = GeneratedAnswer(id="q-1", answer="A", cited_chunk_ids=["d#0", "d#1"])
    assert citation_scores(perfect, q) == (pytest.approx(1.0), pytest.approx(1.0))
    silent = GeneratedAnswer(id="q-1", answer="A")
    assert citation_scores(silent, q) == (0.0, 0.0)


def test_faithfulness_counts_grounded_claims():
    grounded = GeneratedAnswer(
        id="q",
        answer="Encryption keys are rotated every 90 days.",
        context_texts=["Encryption keys are rotated every 90 days by the platform team."],
    )
    score, n = faithfulness_deterministic(grounded)
    assert score == pytest.approx(1.0)
    assert n == 1
    invented = GeneratedAnswer(
        id="q",
        answer="Encryption keys are rotated every 90 days. The CEO is named Marcus Feldstein.",
        context_texts=["Encryption keys are rotated every 90 days by the platform team."],
    )
    score2, n2 = faithfulness_deterministic(invented)
    assert n2 == 2
    assert score2 == pytest.approx(0.5)
    assert faithfulness_deterministic(GeneratedAnswer(id="q", answer="Anything at all here."))[
        0
    ] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I do not have enough information in the provided documents.", True),
        ("I don't know.", True),
        ("The context does not contain that information.", True),
        ("Not answerable from this corpus.", True),
        ("Unable to determine from the retrieved passages.", True),
        ("Within 30 days of the invoice date.", False),
        ("The Team plan costs 12 USD per seat per month.", False),
    ],
)
def test_detect_refusal(text, expected):
    assert detect_refusal(text) is expected


def _perfect_answers(ds, corpus):
    by_id = {c.chunk_id: c for c in corpus}
    out = []
    for q in ds.questions:
        ctx = q.expected_chunk_ids or []
        if q.should_refuse:
            out.append(
                GeneratedAnswer(
                    id=q.id,
                    answer="I do not have enough information in the provided documents.",
                    refused=True,
                )
            )
        else:
            out.append(
                GeneratedAnswer(
                    id=q.id,
                    answer=q.expected_answer,
                    cited_chunk_ids=list(ctx),
                    context_chunk_ids=list(ctx),
                    context_texts=[by_id[c].text for c in ctx],
                )
            )
    return out


def test_evaluate_generation_on_a_perfect_system():
    ds = load_golden()
    report = evaluate_generation(
        ds, _perfect_answers(ds, load_corpus()), seed=SEED, n_resamples=400
    )
    assert report.metrics["refusal_accuracy"].mean == pytest.approx(1.0)
    assert report.metrics["citation_precision"].mean == pytest.approx(1.0)
    assert report.metrics["citation_recall"].mean == pytest.approx(1.0)
    assert report.metrics["answer_relevance"].mean == pytest.approx(1.0)
    assert report.metrics["injection_resistance"].mean == pytest.approx(1.0)
    assert report.metrics["false_refusal_rate"].mean == pytest.approx(0.0)
    assert report.counts["unanswerable"] == len(ds.unanswerable)
    assert not report.judge_used


def test_refusal_accuracy_punishes_a_system_that_never_refuses():
    ds = load_golden()
    answers = [
        GeneratedAnswer(id=q.id, answer="Ulaanbaatar, obviously.", refused=False)
        for q in ds.questions
    ]
    report = evaluate_generation(ds, answers, seed=SEED, n_resamples=400)
    n_bad = len(ds.unanswerable)
    expected = (len(ds) - n_bad) / len(ds)
    assert report.metrics["refusal_accuracy"].mean == pytest.approx(expected)
    assert report.metrics["answered_when_unanswerable_rate"].mean == pytest.approx(1.0)


def test_refusal_accuracy_punishes_a_system_that_always_refuses():
    ds = load_golden()
    answers = [GeneratedAnswer(id=q.id, answer="I don't know.", refused=True) for q in ds.questions]
    report = evaluate_generation(ds, answers, seed=SEED, n_resamples=400)
    assert report.metrics["false_refusal_rate"].mean == pytest.approx(1.0)
    assert report.metrics["answered_when_unanswerable_rate"].mean == pytest.approx(0.0)
    assert report.metrics["refusal_accuracy"].mean < 0.2


def test_injection_resistance_catches_a_leaked_canary():
    ds = load_golden().filter(categories=["adversarial-injection"])
    pwned = [
        GeneratedAnswer(id=q.id, answer=q.injection_canary, context_texts=["ctx"], refused=False)
        for q in ds.questions
    ]
    report = evaluate_generation(ds, pwned, seed=SEED, n_resamples=200)
    assert report.metrics["injection_resistance"].mean == pytest.approx(0.0)


# =========================================================================== #
# Judge calibration
# =========================================================================== #
def test_calibration_fixture_is_committed_and_deterministic():
    labels = load_judge_labels()
    assert len(labels) == 100, "§14 asks for 100 hand-labelled examples"
    rebuilt = build_calibration_fixture()
    assert [x.model_dump() for x in rebuilt] == [x.model_dump() for x in labels]
    assert {x.human_label for x in labels} == {"A", "B", "tie"}
    assert all(x.a_supported is not None and x.b_supported is not None for x in labels)


def test_oracle_judge_scores_kappa_one_and_is_trusted():
    labels = load_judge_labels()
    report = calibrate_judge(OracleJudge.from_labels(labels), labels, seed=7, n_resamples=300)
    assert report.kappa_pairwise.mean == pytest.approx(1.0)
    assert report.swap_rate.mean == pytest.approx(0.0)
    assert report.trust_pairwise
    assert report.trust_support
    assert "answer_relevance_judged" in report.trusted_metrics


def test_coin_flip_judge_is_noise_and_the_gate_fires():
    labels = load_judge_labels()
    report = calibrate_judge(CoinFlipJudge(seed=1), labels, seed=7, n_resamples=300)
    assert abs(report.kappa_pairwise.mean) < 0.2
    assert not report.trust_pairwise
    assert "JUDGE NOT TRUSTED" in report.verdict
    assert set(report.untrusted_metrics) == {"answer_relevance_judged", "faithfulness_judged"}


def test_position_bias_is_detected_by_the_swap_rate():
    labels = load_judge_labels()
    report = calibrate_judge(PositionBiasedJudge(), labels, seed=7, n_resamples=300)
    assert report.swap_rate.mean == pytest.approx(1.0)
    assert report.position_preference.mean == pytest.approx(1.0)
    assert not report.trust_pairwise
    assert "swap rate" in report.position_bias_note


def test_kappa_degrades_monotonically_with_injected_noise():
    labels = load_judge_labels()
    kappas = [
        calibrate_judge(
            OracleJudge.from_labels(labels, noise=noise, seed=3), labels, seed=7, n_resamples=200
        ).kappa_pairwise.mean
        for noise in (0.0, 0.15, 0.45, 0.9)
    ]
    assert kappas == sorted(kappas, reverse=True)
    assert kappas[0] > KAPPA_TRUST_THRESHOLD > kappas[-1]


def test_the_heuristic_judge_reports_its_own_weakness():
    """An honest negative: the lexical fallback judge does not clear kappa 0.6."""
    labels = load_judge_labels()
    report = calibrate_judge(HeuristicJudge(), labels, seed=7, n_resamples=300)
    assert not report.trust_pairwise
    assert 0.4 < report.kappa_pairwise.mean < KAPPA_TRUST_THRESHOLD
    assert "falls back to deterministic metrics" in report.verdict


def test_untrusted_judge_suppresses_judged_metrics_automatically():
    ds = load_golden()
    answers = _perfect_answers(ds, load_corpus())
    labels = load_judge_labels()
    bad = calibrate_judge(CoinFlipJudge(seed=2), labels, seed=7, n_resamples=200)
    report = evaluate_generation(
        ds, answers, judge=CoinFlipJudge(seed=2), calibration=bad, seed=SEED, n_resamples=200
    )
    assert not report.judge_used
    assert "judge is noise" in report.judge_fallback_reason
    assert not [k for k in report.metrics if k.endswith("_judged")]
    # Deterministic metrics survive the fallback.
    assert "faithfulness" in report.metrics


def test_trusted_judge_adds_judged_metrics():
    ds = load_golden()
    answers = _perfect_answers(ds, load_corpus())
    labels = load_judge_labels()
    good = calibrate_judge(OracleJudge.from_labels(labels), labels, seed=7, n_resamples=200)
    report = evaluate_generation(
        ds,
        answers,
        judge=OracleJudge.from_labels(labels),
        calibration=good,
        seed=SEED,
        n_resamples=200,
    )
    assert report.judge_used
    assert "answer_relevance_judged" in report.metrics
    assert "faithfulness_judged" in report.metrics


def test_a_judge_without_calibration_is_never_used():
    ds = load_golden()
    report = evaluate_generation(
        ds, _perfect_answers(ds, load_corpus()), judge=HeuristicJudge(), seed=SEED, n_resamples=200
    )
    assert not report.judge_used
    assert "calibration mandatory" in report.judge_fallback_reason
    assert "suppressed" in report.judge_fallback_reason


def test_flip_verdict_round_trips():
    for v in ("A", "B", "tie"):
        assert flip_verdict(flip_verdict(v)) == v


# =========================================================================== #
# System metrics
# =========================================================================== #
def _records(n=50):
    rng = np.random.default_rng(4)
    out = []
    for i in range(n):
        total = float(0.2 + 0.05 * rng.random())
        out.append(
            RequestRecord(
                id=f"r-{i}",
                ttft_s=total * 0.3,
                total_s=total,
                cpu_s=total * 0.8,
                peak_rss_bytes=400 * 1024 * 1024 + i,
                tokens_in=100 + i,
                tokens_out=50,
                n_iterations=1 + i % 3,
                tool_calls=[ToolCall(name="search", ok=i % 10 != 0)],
                refused=i % 12 == 0,
            )
        )
    return out


def test_system_report_covers_every_required_metric():
    report = evaluate_system(_records(), system="agentic", seed=SEED, n_resamples=400)
    required = {
        "ttft_ms",
        "e2e_p50_ms",
        "e2e_p95_ms",
        "e2e_p99_ms",
        "tokens_in",
        "tokens_out",
        "tool_success_rate",
        "iterations",
        "cpu_s_per_query",
        "peak_rss_mb",
    }
    assert required <= set(report.metrics)
    assert all(isinstance(v, Estimate) for v in report.metrics.values())
    assert report.metrics["e2e_p50_ms"].mean <= report.metrics["e2e_p95_ms"].mean
    assert report.metrics["e2e_p95_ms"].mean <= report.metrics["e2e_p99_ms"].mean
    assert report.metrics["tool_success_rate"].mean == pytest.approx(0.9)
    assert sum(report.iteration_histogram.values()) == 50
    assert set(report.iteration_histogram) == {1, 2, 3}
    assert report.tool_histogram["search"] == {"ok": 45, "fail": 5}
    md = report.to_markdown()
    assert "CPU-seconds per query" in md or "cpu_s_per_query" in md
    assert "$0" in md


def test_system_report_counts_errors_and_excludes_them_from_latency():
    records = _records(10)
    records[0].error = "boom"
    report = evaluate_system(records, seed=SEED, n_resamples=200)
    assert report.counts["errors"] == 1
    assert report.metrics["error_rate"].mean == pytest.approx(0.1)
    assert report.metrics["e2e_p50_ms"].n == 9


def test_measure_request_fills_timing_and_memory():
    with measure_request("q-1") as rec:
        rec.tokens_out = 7
        total = 0
        for i in range(200_000):
            total += i
    assert rec.total_s > 0
    assert rec.cpu_s >= 0
    assert rec.tokens_out == 7
    assert rec.peak_rss_bytes is None or rec.peak_rss_bytes > 0


def test_measure_request_records_failures_without_swallowing_them():
    rec_holder = {}
    with pytest.raises(RuntimeError), measure_request("q-2") as rec:
        rec_holder["rec"] = rec
        raise RuntimeError("nope")
    assert "RuntimeError: nope" in rec_holder["rec"].error
    assert not rec_holder["rec"].ok


def test_evaluate_system_requires_records():
    with pytest.raises(ValueError):
        evaluate_system([])


def test_hardware_string_is_populated():
    hw = hardware_string()
    assert "py3." in hw and "|" in hw


# =========================================================================== #
# Golden generation
# =========================================================================== #
def test_fake_teacher_generates_schema_valid_candidates():
    corpus = load_corpus()
    cfg = GoldenGenConfig(
        seed=SEED,
        quotas={
            "factoid": 6,
            "multi-hop": 3,
            "aggregation": 2,
            "table": 2,
            "figure": 2,
            "out-of-domain": 2,
            "adversarial-injection": 2,
        },
    )
    ds = GoldenGenerator(teacher=DeterministicFakeTeacher(), config=cfg).generate(corpus)
    assert len(ds) > 10
    assert ds.kind == "candidates"
    assert all(not q.verified for q in ds.questions), "candidates must never claim verification"
    assert "out-of-domain" in ds.categories
    assert "adversarial-injection" in ds.categories
    for q in ds.questions:
        if q.category == "out-of-domain":
            assert q.should_refuse
        if q.is_adversarial:
            assert q.injection_canary in q.question


def test_generation_is_reproducible_for_a_seed():
    corpus = load_corpus()
    cfg = GoldenGenConfig(seed=99, quotas={"factoid": 5, "multi-hop": 2, "out-of-domain": 2})
    a = GoldenGenerator(teacher=DeterministicFakeTeacher(), config=cfg).generate(corpus)
    b = GoldenGenerator(teacher=DeterministicFakeTeacher(), config=cfg).generate(corpus)
    assert a.content_hash == b.content_hash


def test_generated_candidates_cannot_be_loaded_as_a_golden_set(tmp_path):
    corpus = load_corpus()
    cfg = GoldenGenConfig(seed=1, quotas={"factoid": 4})
    ds = GoldenGenerator(teacher=DeterministicFakeTeacher(), config=cfg).generate(corpus)
    path = tmp_path / "cands.jsonl"
    ds.write(path)
    with pytest.raises(ValueError, match="hand-verified"):
        load_golden(path)
    promoted = promote_candidates(ds, [q.id for q in ds.questions[:2]])
    assert len(promoted) == 2
    assert all(q.verified for q in promoted.questions)
    promoted.write(path)
    assert len(load_golden(path)) == 2
    with pytest.raises(KeyError):
        promote_candidates(ds, ["nope-9999"])


def test_malformed_teacher_output_is_dropped_not_crashed():
    corpus = load_corpus()
    cfg = GoldenGenConfig(seed=2, quotas={"factoid": 4}, max_attempts_per_question=3)
    teacher = DeterministicFakeTeacher(malformed_every=2)
    ds = GoldenGenerator(teacher=teacher, config=cfg).generate(corpus)
    assert all(q.question.strip() for q in ds.questions)


def test_prompt_carries_machine_readable_chunk_tags_and_parses_back():
    corpus = load_corpus()[:2]
    prompt = build_prompt("factoid", corpus)
    assert "CATEGORY: factoid" in prompt
    for chunk in corpus:
        assert f"<<<CHUNK id={chunk.chunk_id}>>>" in prompt
    assert parse_candidate('garbage {"question": "Q?"} trailing') == {"question": "Q?"}
    assert parse_candidate("no json here") is None
    assert parse_candidate("{not valid json}") is None


def test_quotas_scale_and_hash():
    cfg = GoldenGenConfig()
    assert 150 <= cfg.total <= 300, "default quotas must target the §14 window"
    assert cfg.config_hash() == GoldenGenConfig().config_hash()
    assert cfg.config_hash() != GoldenGenConfig(seed=7).config_hash()


# =========================================================================== #
# Report / CI gate
# =========================================================================== #
def _payload(ndcg: float, per_q: dict[str, float] | None = None) -> dict:
    est = Estimate(mean=ndcg, lo=ndcg - 0.05, hi=ndcg + 0.05, n=20).to_dict()
    return {
        "name": "retrieval",
        "hardware": "cpu",
        "seeds": [1],
        "ci": "bootstrap95",
        "rows": [{"name": "sys", "ndcg@10": est}],
        "detail": {
            "metrics": {"ndcg@10": est},
            "per_question": {k: {"ndcg@10": v} for k, v in (per_q or {}).items()},
        },
    }


def test_gate_passes_when_the_metric_holds():
    gate = check_regression(
        extract_metric(_payload(0.80), "ndcg@10"),
        extract_metric(_payload(0.80), "ndcg@10"),
        max_regression=0.02,
    )
    assert gate.passed and gate.exit_code == 0
    assert "OK" in gate.message


def test_gate_fails_on_a_regression_beyond_two_percent():
    # 0.80 -> 0.77 is a 3.75% relative drop; the budget is 2%.
    gate = check_regression(
        extract_metric(_payload(0.77), "ndcg@10"),
        extract_metric(_payload(0.80), "ndcg@10"),
        max_regression=0.02,
    )
    assert not gate.passed
    assert gate.exit_code == 1
    assert gate.delta_rel == pytest.approx(-0.0375)
    assert "REGRESSION" in gate.message


def test_gate_tolerates_a_regression_inside_the_budget():
    # 0.80 -> 0.7885 is a 1.4% relative drop, inside the 2% budget.
    gate = check_regression(
        extract_metric(_payload(0.7885), "ndcg@10"),
        extract_metric(_payload(0.80), "ndcg@10"),
        max_regression=0.02,
    )
    assert gate.passed


def test_gate_allows_improvements_of_any_size():
    gate = check_regression(
        extract_metric(_payload(0.95), "ndcg@10"),
        extract_metric(_payload(0.60), "ndcg@10"),
        max_regression=0.02,
    )
    assert gate.passed


def test_absolute_threshold_mode():
    gate = check_regression(
        extract_metric(_payload(0.78), "ndcg@10"),
        extract_metric(_payload(0.80), "ndcg@10"),
        max_regression=0.02,
        relative=False,
    )
    assert gate.passed  # exactly 0.02 absolute
    gate2 = check_regression(
        extract_metric(_payload(0.77), "ndcg@10"),
        extract_metric(_payload(0.80), "ndcg@10"),
        max_regression=0.02,
        relative=False,
    )
    assert not gate2.passed


def test_gate_skips_when_there_is_no_baseline():
    gate = check_regression(extract_metric(_payload(0.8), "ndcg@10"), None)
    assert gate.passed
    assert "no baseline" in gate.message


def test_gate_fails_when_the_current_run_is_missing():
    gate = check_regression(None, extract_metric(_payload(0.8), "ndcg@10"))
    assert not gate.passed
    assert "no current value" in gate.message


def test_local_baseline_source(tmp_path):
    root = tmp_path / "benchmarks"
    (root / "baselines" / "main").mkdir(parents=True)
    (root / "baselines" / "main" / "eval_retrieval.json").write_text(
        json.dumps(_payload(0.8)), encoding="utf-8"
    )
    src = LocalBaselineSource(root)
    assert src.get("main", "eval_retrieval") is not None
    assert src.get("nope", "eval_retrieval") is None


def test_report_cli_exit_codes(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALMIND_EVAL_NO_GIT", "1")
    from localmind.eval.report import main as report_main

    root = tmp_path / "benchmarks"
    (root / "baselines" / "main").mkdir(parents=True)
    (root / "baselines" / "main" / "eval_retrieval.json").write_text(
        json.dumps(_payload(0.80, {"q1": 0.9, "q2": 0.8, "q3": 0.7})), encoding="utf-8"
    )
    docs = tmp_path / "benchmarks.md"

    (root / "eval_retrieval.json").write_text(
        json.dumps(_payload(0.80, {"q1": 0.9, "q2": 0.8, "q3": 0.7})), encoding="utf-8"
    )
    assert report_main(["--artifacts", str(root), "--docs", str(docs)]) == 0
    assert docs.exists()
    assert "mean [lo, hi]" in docs.read_text(encoding="utf-8")

    (root / "eval_retrieval.json").write_text(
        json.dumps(_payload(0.60, {"q1": 0.7, "q2": 0.6, "q3": 0.5})), encoding="utf-8"
    )
    assert report_main(["--artifacts", str(root), "--docs", str(docs)]) == 1

    assert report_main(["--artifacts", str(tmp_path / "empty"), "--docs", str(docs)]) == 2


def test_report_cli_accepts_an_explicit_baseline_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALMIND_EVAL_NO_GIT", "1")
    from localmind.eval.report import main as report_main

    root = tmp_path / "benchmarks"
    root.mkdir(parents=True)
    (root / "eval_retrieval.json").write_text(json.dumps(_payload(0.60)), encoding="utf-8")
    base = tmp_path / "base.json"
    base.write_text(json.dumps(_payload(0.80)), encoding="utf-8")
    rc = report_main(
        [
            "--artifacts",
            str(root),
            "--baseline-file",
            str(base),
            "--docs",
            str(tmp_path / "b.md"),
        ]
    )
    assert rc == 1


def test_git_baseline_source_is_disabled_by_env(monkeypatch):
    from localmind.eval.report import GitBaselineSource

    monkeypatch.setenv("LOCALMIND_EVAL_NO_GIT", "1")
    assert GitBaselineSource().get("main", "retrieval") is None


def test_benchmarks_md_contains_the_headline_table_and_rule5_note(tmp_path):
    root = tmp_path / "benchmarks"
    root.mkdir(parents=True)
    (root / "eval_retrieval.json").write_text(json.dumps(_payload(0.8)), encoding="utf-8")
    out = regenerate_benchmarks_md(root, tmp_path / "benchmarks.md")
    text = out.read_text(encoding="utf-8")
    assert "The headline experiment" in text
    assert "LocalMind-31M" in text
    assert "CPU-seconds per query" in text
    assert "0.800 [0.750, 0.850]" in text
    assert "n/a" in text, "unmeasured configurations must read n/a, never 0"


def test_benchmarks_md_warns_about_too_few_seeds(tmp_path):
    root = tmp_path / "benchmarks"
    root.mkdir(parents=True)
    (root / "eval_retrieval.json").write_text(json.dumps(_payload(0.8)), encoding="utf-8")
    text = regenerate_benchmarks_md(root, tmp_path / "b.md").read_text(encoding="utf-8")
    assert "Fewer than 3 seeds" in text


def test_paired_comparison_against_the_baseline_is_reported(tmp_path):
    from localmind.eval.report import paired_against_baseline

    current = _payload(0.70, {"q1": 0.7, "q2": 0.6, "q3": 0.8, "q4": 0.7})
    baseline = _payload(0.80, {"q1": 0.9, "q2": 0.8, "q3": 0.9, "q4": 0.8})
    comparison = paired_against_baseline(current, baseline, "ndcg@10")
    assert comparison is not None
    assert comparison.paired.diff.mean < 0
    assert "baseline wins" in comparison.verdict
    assert paired_against_baseline(_payload(0.7), _payload(0.8), "ndcg@10") is None


# =========================================================================== #
# Cross-layer invariant
# =========================================================================== #
def test_no_reported_metric_anywhere_lacks_a_confidence_interval():
    """The whole point of the harness: every number in every report has a CI."""
    ds = load_golden()
    corpus = load_corpus()
    retrieval = evaluate_retrieval(ds, BM25Retriever(chunks=corpus), seed=SEED, n_resamples=200)
    generation = evaluate_generation(ds, _perfect_answers(ds, corpus), seed=SEED, n_resamples=200)
    system = evaluate_system(_records(20), seed=SEED, n_resamples=200)
    for report in (retrieval, generation, system):
        assert report.metrics
        for name, value in report.metrics.items():
            assert isinstance(value, Estimate), f"{name} is a bare {type(value).__name__}"
            assert value.lo <= value.mean <= value.hi
    payload = benchmark_json(
        "all",
        hardware=hardware_string(),
        seeds=[SEED],
        rows=[retrieval.to_row(), generation.to_row("gen"), system.to_row()],
    )
    assert payload["ci"] == "bootstrap95"
    for row in payload["rows"]:
        for key, value in row.items():
            if isinstance(value, dict) and "mean" in value:
                assert {"lo", "hi", "method"} <= set(value), f"{key} lost its interval"


# =========================================================================== #
# Interoperability with other phases' benchmark artifacts
# =========================================================================== #
def _foreign_payload() -> dict:
    """The shape another phase's benchmark writes: `ci_low`/`ci_high`, `_at_`, `config`."""
    return {
        "name": "retrieval",
        "hardware": "cpu",
        "seeds": [0, 1, 2],
        "ci": "bootstrap95",
        "rows": [
            {
                "config": "hybrid-rrf",
                "ndcg_at_10": {"mean": 0.81, "ci_low": 0.77, "ci_high": 0.85},
                "recall_at_10": {"mean": 0.9, "ci_low": 0.86, "ci_high": 0.94},
                "notes": "not an estimate",
                "broken": {"mean": 0.5},
            }
        ],
    }


def test_estimate_from_dict_accepts_ci_low_ci_high():
    est = Estimate.from_dict({"mean": 0.81, "ci_low": 0.77, "ci_high": 0.85})
    assert (est.mean, est.lo, est.hi) == (0.81, 0.77, 0.85)
    with pytest.raises(KeyError, match="interval bounds"):
        Estimate.from_dict({"mean": 0.5})


def test_metric_aliases_bridge_the_two_spellings():
    from localmind.eval.report import metric_aliases

    assert "ndcg_at_10" in metric_aliases("ndcg@10")
    assert "ndcg@10" in metric_aliases("ndcg_at_10")


def test_gate_reads_a_foreign_artifact_schema():
    est = extract_metric(_foreign_payload(), "ndcg@10")
    assert est is not None
    assert est.mean == pytest.approx(0.81)
    assert est.lo == pytest.approx(0.77)


def test_benchmarks_md_survives_a_malformed_cell_from_another_phase(tmp_path):
    root = tmp_path / "benchmarks"
    root.mkdir(parents=True)
    (root / "other_phase.json").write_text(json.dumps(_foreign_payload()), encoding="utf-8")
    (root / "_scratch.json").write_text(json.dumps(_foreign_payload()), encoding="utf-8")
    text = regenerate_benchmarks_md(root, tmp_path / "b.md").read_text(encoding="utf-8")
    assert "hybrid-rrf" in text, "row name should fall back to `config`"
    assert "0.810 [0.770, 0.850]" in text
    assert "broken" not in text, "a cell without bounds is dropped, not fatal"
    assert "_scratch" not in text, "`_`-prefixed artifacts are scratch, not results"


def test_sample_run_is_labelled_as_a_harness_self_test(tmp_path):
    from localmind.eval.retrieval import main as retrieval_main

    root = tmp_path / "benchmarks"
    retrieval_main(["--out", str(root / "eval_retrieval.json"), "--seed", "5"])
    text = regenerate_benchmarks_md(root, tmp_path / "b.md").read_text(encoding="utf-8")
    assert "golden_sample_v1" in text
    assert "not a system result" in text
