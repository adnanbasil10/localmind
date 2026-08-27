"""Fusion — combining ranked lists from multiple retrieval arms into one ranking.

Baseline: Reciprocal Rank Fusion (RRF, Cormack et al. 2009).

    score(d) = sum_arms 1 / (k + rank_arm(d))          (k=60)

RRF is rank-based, not score-based, so it needs no cross-arm score normalization — BM25
scores, cosine similarities, SPLADE dot products, and ColBERT MaxSim totals live on entirely
different, incomparable scales, and RRF sidesteps that by only ever looking at *position*.
See `docs/decisions/0006-rrf-over-tuned-fusion.md` for why this is the default.

Alternative: min-max-normalized weighted fusion, tuned on a dev set (`tune_weights`).
`compare_fusion_strategies` runs both and reports which one actually wins — per the spec,
a tuned scheme that loses to the parameter-free baseline is a real, reportable finding, not
a bug, so this module is built so that a negative result is a first-class return value, not
something that has to be dug out of logs.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from localmind.retrieval import ScoredDoc, ranks_from_scores

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    arm_results: dict[str, list[ScoredDoc]], k: int = DEFAULT_RRF_K
) -> list[ScoredDoc]:
    """score(d) = sum_arms 1/(k + rank_arm(d)), rank is 1-indexed. A doc missing from an arm's
    results simply contributes 0 from that arm.
    """
    fused: dict[str, float] = {}
    for results in arm_results.values():
        ranks = ranks_from_scores(results)
        for doc_id, rank in ranks.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [ScoredDoc(doc_id=doc_id, score=score) for doc_id, score in ranked]


def minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max scale a set of scores to [0, 1]. A constant score set maps everywhere to 1.0
    (every candidate is equally likely, so it shouldn't be zeroed out and dropped from a sum).
    """
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return dict.fromkeys(scores, 1.0)
    return {doc_id: (s - lo) / (hi - lo) for doc_id, s in scores.items()}


def weighted_fusion(
    arm_results: dict[str, list[ScoredDoc]], weights: dict[str, float]
) -> list[ScoredDoc]:
    """Min-max normalize each arm's scores independently, then combine with per-arm weights.

    Unlike RRF, this uses score *magnitude*, not just rank — which is exactly why it needs
    normalization first, and exactly why it's more fragile: min-max is sensitive to outliers
    and to how the candidate set was truncated per arm.
    """
    fused: dict[str, float] = {}
    for arm, results in arm_results.items():
        weight = weights.get(arm, 0.0)
        if weight == 0.0:
            continue
        normalized = minmax_normalize({sd.doc_id: sd.score for sd in results})
        for doc_id, score in normalized.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * score
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [ScoredDoc(doc_id=doc_id, score=score) for doc_id, score in ranked]


def _weight_grid(arms: Sequence[str], steps: int = 5) -> list[dict[str, float]]:
    """All weight combinations over {0, 1/steps, 2/steps, ..., 1} per arm (not required to sum
    to 1 — weighted_fusion normalizes scores, not weights). Small `steps` keeps this a dev-set
    grid search, not an optimizer; that's deliberate; see `tune_weights` docstring.
    """
    levels = [i / steps for i in range(steps + 1)]
    combos = itertools.product(levels, repeat=len(arms))
    return [dict(zip(arms, combo, strict=True)) for combo in combos if any(combo)]


@dataclass(frozen=True, slots=True)
class FusionComparisonReport:
    """Result of comparing RRF against tuned weighted fusion on a dev set. `tuned_beat_rrf` is
    the headline field: the spec explicitly asks that a negative result here be reportable,
    not buried, since tuned fusion often does *not* beat RRF.
    """

    rrf_metric: float
    tuned_metric: float
    best_weights: dict[str, float]
    tuned_beat_rrf: bool
    delta: float  # tuned_metric - rrf_metric; negative means RRF won


def tune_weights(
    dev_arm_results: dict[str, dict[str, list[ScoredDoc]]],
    qrels: dict[str, set[str]],
    metric_fn: Callable[[list[ScoredDoc], set[str]], float],
    steps: int = 5,
) -> tuple[dict[str, float], float]:
    """Grid-search per-arm fusion weights on a dev set of queries.

    `dev_arm_results`: {query_id: {arm_name: ranked results}}.
    `qrels`: {query_id: set of relevant doc_ids}.
    `metric_fn`: e.g. nDCG@10, applied per query and averaged.

    Returns (best_weights, best_mean_metric). A small grid (default 6^n_arms combinations) is
    intentional: this is meant to demonstrate whether *any* reasonable weighting beats RRF,
    not to squeeze out a marginal win via an expensive search — an expensive search would
    itself risk overfitting the (small, synthetic) dev set.
    """
    arms = sorted({arm for results in dev_arm_results.values() for arm in results})
    best_weights: dict[str, float] = dict.fromkeys(arms, 1.0)
    best_metric = float("-inf")
    for weights in _weight_grid(arms, steps=steps):
        metrics = []
        for query_id, arm_results in dev_arm_results.items():
            fused = weighted_fusion(arm_results, weights)
            metrics.append(metric_fn(fused, qrels.get(query_id, set())))
        mean_metric = sum(metrics) / len(metrics) if metrics else 0.0
        if mean_metric > best_metric:
            best_metric = mean_metric
            best_weights = weights
    return best_weights, best_metric


def compare_fusion_strategies(
    dev_arm_results: dict[str, dict[str, list[ScoredDoc]]],
    qrels: dict[str, set[str]],
    metric_fn: Callable[[list[ScoredDoc], set[str]], float],
    rrf_k: int = DEFAULT_RRF_K,
    steps: int = 5,
) -> FusionComparisonReport:
    """Run RRF and tuned weighted fusion on the same dev set and report which wins."""
    rrf_metrics = [
        metric_fn(reciprocal_rank_fusion(arm_results, k=rrf_k), qrels.get(query_id, set()))
        for query_id, arm_results in dev_arm_results.items()
    ]
    rrf_metric = sum(rrf_metrics) / len(rrf_metrics) if rrf_metrics else 0.0
    best_weights, tuned_metric = tune_weights(dev_arm_results, qrels, metric_fn, steps=steps)
    delta = tuned_metric - rrf_metric
    return FusionComparisonReport(
        rrf_metric=rrf_metric,
        tuned_metric=tuned_metric,
        best_weights=best_weights,
        tuned_beat_rrf=delta > 0,
        delta=delta,
    )
