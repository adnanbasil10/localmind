"""LocalMind evaluation harness -- "the most valuable artifact in the repo" (§14).

Four layers, one statistics backbone:

* ``datasets`` -- the versioned, hash-pinned golden set and its schema.
* ``retrieval`` -- Recall@k, nDCG@10, MRR, MAP. Deterministic, no LLM, CI gate.
* ``generation`` -- faithfulness, answer relevance, citation P/R, refusal.
* ``system``    -- TTFT, p50/p95/p99, tokens, tools, iterations, and (since
  cost per query is $0) CPU-seconds per query and peak RSS.
* ``stats``     -- bootstrap CIs, paired bootstrap and Wilcoxon, Cohen's kappa.
  Rule 5 lives here: an :class:`~localmind.eval.stats.Estimate` cannot be
  reported as a bare number.

``import localmind.eval`` pulls in nothing heavier than numpy and pydantic;
every other dependency (yaml, httpx, scipy, a model server) is imported lazily
inside the function that needs it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from localmind.eval.stats import (
    DEFAULT_SEED,
    MIN_SEEDS,
    ComparisonReport,
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
)

if TYPE_CHECKING:  # pragma: no cover
    from localmind.eval import (
        generate_golden,
        generation,
        judge_calibration,
        report,
        retrieval,
        system,
    )
    from localmind.eval.datasets import schema

__all__ = [
    "DEFAULT_SEED",
    "MIN_SEEDS",
    "ComparisonReport",
    "Estimate",
    "MetricRow",
    "aggregate_seeds",
    "benchmark_json",
    "bootstrap_ci",
    "bootstrap_paired",
    "cohens_kappa",
    "compare_configs",
    "format_metric",
    "generate_golden",
    "generation",
    "judge_calibration",
    "report",
    "require_estimate",
    "require_seeds",
    "retrieval",
    "schema",
    "system",
    "wilcoxon_signed_rank",
]

_LAZY = {
    "generate_golden": "localmind.eval.generate_golden",
    "generation": "localmind.eval.generation",
    "judge_calibration": "localmind.eval.judge_calibration",
    "report": "localmind.eval.report",
    "retrieval": "localmind.eval.retrieval",
    "system": "localmind.eval.system",
    "schema": "localmind.eval.datasets.schema",
}


def __getattr__(name: str) -> Any:
    """Import the heavier submodules only when they are actually touched."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(__all__)
