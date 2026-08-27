"""The CI gate and the deliverable: `docs/benchmarks.md`.

Two jobs, both driven by the justfile::

    just eval-retrieval   ->  python -m localmind.eval.retrieval --config ...
    just eval-compare     ->  python -m localmind.eval.report --baseline main \\
                                     --max-regression 0.02

The gate fails the build (exit 1) when nDCG@10 drops more than the allowed
fraction against the baseline.  Where the baseline and the current run share
question ids, it also runs the paired tests from ``stats.py`` so the verdict
carries a CI and a p-value rather than a bare delta.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from localmind.eval.stats import (
    DEFAULT_SEED,
    ComparisonReport,
    Estimate,
    compare_configs,
)

__all__ = [
    "BaselineSource",
    "ChainedBaselineSource",
    "GateResult",
    "GitBaselineSource",
    "LocalBaselineSource",
    "check_regression",
    "extract_metric",
    "extract_per_question",
    "main",
    "regenerate_benchmarks_md",
]

DEFAULT_ARTIFACTS = Path("artifacts/benchmarks")
DEFAULT_DOCS = Path("docs/benchmarks.md")
GATE_METRIC = "ndcg@10"
GATE_ARTIFACT = "eval_retrieval"
"""The gate reads the harness's own artifact, not whatever else lands in the dir.

Other phases write their own benchmark JSON into `artifacts/benchmarks/`; the
name is namespaced so a phase benchmark can never be mistaken for the gate's
measurement (and vice versa).
"""
NO_GIT_ENV = "LOCALMIND_EVAL_NO_GIT"


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
@runtime_checkable
class BaselineSource(Protocol):
    """Where a baseline benchmark artifact comes from."""

    def get(self, ref: str, artifact: str) -> dict[str, Any] | None: ...


@dataclass
class LocalBaselineSource:
    """``artifacts/benchmarks/baselines/<ref>/<artifact>.json`` -- offline, no git."""

    root: Path = DEFAULT_ARTIFACTS

    def get(self, ref: str, artifact: str) -> dict[str, Any] | None:
        path = self.root / "baselines" / ref / f"{artifact}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class GitBaselineSource:
    """``git show <ref>:artifacts/benchmarks/<artifact>.json``.

    Best-effort and entirely optional: any failure returns ``None`` and set
    ``LOCALMIND_EVAL_NO_GIT=1`` to disable it outright (tests do).
    """

    root: Path = DEFAULT_ARTIFACTS
    timeout_s: float = 20.0

    def get(self, ref: str, artifact: str) -> dict[str, Any] | None:
        if os.environ.get(NO_GIT_ENV) == "1":
            return None
        target = f"{ref}:{(self.root / f'{artifact}.json').as_posix()}"
        try:
            out = subprocess.run(
                ["git", "show", target],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0 or not out.stdout.strip():
            return None
        try:
            return json.loads(out.stdout)
        except json.JSONDecodeError:
            return None


@dataclass
class ChainedBaselineSource:
    sources: Sequence[BaselineSource]

    def get(self, ref: str, artifact: str) -> dict[str, Any] | None:
        for source in self.sources:
            found = source.get(ref, artifact)
            if found is not None:
                return found
        return None


def default_baseline_source(root: Path = DEFAULT_ARTIFACTS) -> BaselineSource:
    return ChainedBaselineSource([LocalBaselineSource(root), GitBaselineSource(root)])


# --------------------------------------------------------------------------- #
# Artifact readers
# --------------------------------------------------------------------------- #
def metric_aliases(metric: str) -> list[str]:
    """`ndcg@10` and `ndcg_at_10` name the same thing; accept both spellings."""
    names = [metric]
    if "@" in metric:
        names.append(metric.replace("@", "_at_"))
    if "_at_" in metric:
        names.append(metric.replace("_at_", "@"))
    return names


def extract_metric(payload: Mapping[str, Any], metric: str) -> Estimate | None:
    """Pull one metric out of a benchmark artifact, wherever it lives."""
    names = metric_aliases(metric)
    for row in payload.get("rows", []) or []:
        for name in names:
            value = row.get(name)
            if isinstance(value, Mapping) and "mean" in value:
                with contextlib.suppress(KeyError, TypeError, ValueError):
                    return Estimate.from_dict(value)
    detail = payload.get("detail")
    if isinstance(detail, Mapping):
        metrics = detail.get("metrics") or {}
        for name in names:
            value = metrics.get(name)
            if isinstance(value, Mapping) and "mean" in value:
                with contextlib.suppress(KeyError, TypeError, ValueError):
                    return Estimate.from_dict(value)
    return None


def extract_per_question(payload: Mapping[str, Any], metric: str) -> dict[str, float]:
    """Per-question values, so the gate can run a *paired* test."""
    detail = payload.get("detail")
    if not isinstance(detail, Mapping):
        return {}
    per_q = detail.get("per_question") or {}
    if not isinstance(per_q, Mapping):
        return {}
    return {
        qid: float(vals[metric])
        for qid, vals in per_q.items()
        if isinstance(vals, Mapping) and metric in vals
    }


def paired_against_baseline(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    metric: str,
    *,
    seed: int = DEFAULT_SEED,
) -> ComparisonReport | None:
    """Paired bootstrap + Wilcoxon on the questions both runs answered."""
    cur = extract_per_question(current, metric)
    base = extract_per_question(baseline, metric)
    shared = sorted(set(cur) & set(base))
    if len(shared) < 2:
        return None
    return compare_configs(
        "current",
        [cur[q] for q in shared],
        "baseline",
        [base[q] for q in shared],
        metric=metric,
        seed=seed,
        n_resamples=2_000,
    )


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
@dataclass
class GateResult:
    passed: bool
    metric: str
    current: Estimate | None
    baseline: Estimate | None
    max_regression: float
    relative: bool = True
    message: str = ""
    comparison: ComparisonReport | None = None

    @property
    def delta_abs(self) -> float | None:
        if self.current is None or self.baseline is None:
            return None
        return self.current.mean - self.baseline.mean

    @property
    def delta_rel(self) -> float | None:
        if self.current is None or self.baseline is None or self.baseline.mean == 0:
            return None
        return (self.current.mean - self.baseline.mean) / abs(self.baseline.mean)

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "metric": self.metric,
            "current": self.current.to_dict() if self.current else None,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "delta_abs": self.delta_abs,
            "delta_rel": self.delta_rel,
            "max_regression": self.max_regression,
            "relative": self.relative,
            "message": self.message,
            "comparison": self.comparison.to_dict() if self.comparison else None,
        }


def check_regression(
    current: Estimate | None,
    baseline: Estimate | None,
    *,
    metric: str = GATE_METRIC,
    max_regression: float = 0.02,
    relative: bool = True,
    comparison: ComparisonReport | None = None,
) -> GateResult:
    """Fail when ``metric`` drops by more than ``max_regression``.

    Relative by default -- "drops more than 2%" reads as 2% *of the baseline*.
    Pass ``relative=False`` for an absolute threshold in metric units.
    """
    if current is None:
        return GateResult(
            passed=False,
            metric=metric,
            current=None,
            baseline=baseline,
            max_regression=max_regression,
            relative=relative,
            message=f"no current value for {metric}: run the retrieval eval first",
        )
    if baseline is None:
        return GateResult(
            passed=True,
            metric=metric,
            current=current,
            baseline=None,
            max_regression=max_regression,
            relative=relative,
            message=(
                f"no baseline for {metric}; gate skipped and current value recorded as the "
                f"baseline candidate ({current.format()})"
            ),
        )

    drop = baseline.mean - current.mean
    allowed = max_regression * abs(baseline.mean) if relative else max_regression
    passed = drop <= allowed + 1e-12
    pct = (drop / abs(baseline.mean) * 100.0) if baseline.mean else float("nan")
    verdict = "OK" if passed else "REGRESSION"
    result = GateResult(
        passed=passed,
        metric=metric,
        current=current,
        baseline=baseline,
        max_regression=max_regression,
        relative=relative,
        comparison=comparison,
        message=(
            f"{verdict}: {metric} {current.format()} vs baseline {baseline.format()} "
            f"(delta {current.mean - baseline.mean:+.4f}, {-pct:+.2f}%; "
            f"budget {max_regression:.2%} {'relative' if relative else 'absolute'})"
        ),
    )
    return result


# --------------------------------------------------------------------------- #
# docs/benchmarks.md
# --------------------------------------------------------------------------- #
HEADLINE_SYSTEMS: tuple[str, ...] = (
    "1. Naive RAG (fixed chunks, dense only)",
    "2. Hybrid RAG (4-arm RRF)",
    "3. Hybrid + cross-encoder rerank",
    "4. + contextual chunking",
    "5. Agentic (grade + rewrite + web fallback)",
    "6. Agentic with LocalMind-31M as router/grader/rewriter",
    "7. Agentic, all-4B control",
)

_HEADLINE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Recall@10", "recall@10"),
    ("nDCG@10", "ndcg@10"),
    ("Faithfulness", "faithfulness"),
    ("Citation P", "citation_precision"),
    ("Citation R", "citation_recall"),
    ("Refusal acc", "refusal_accuracy"),
    ("p95 ms", "e2e_p95_ms"),
    ("CPU-s/query", "cpu_s_per_query"),
)

_PREAMBLE = """# Benchmarks

> Every number on this page is `mean [lo, hi]` -- a point estimate with a
> bootstrap 95% CI. That is rule 5 of `implementation.md` §20, and it is
> enforced in code: `localmind.eval.stats.Estimate` refuses to be coerced to a
> bare float, and `benchmark_json()` rejects any row value that is not an
> `Estimate`. A table of bare point estimates here would be a bug.
>
> Cost per query is $0 -- everything runs locally -- so the resource columns
> are **CPU-seconds per query** and **peak RSS**, which are the numbers that
> actually constrain this system.

_This file is generated by `python -m localmind.eval.report`. Edit the harness,
not the table._
"""


def _fmt(value: Estimate | None, digits: int = 3) -> str:
    return value.format(digits) if value is not None else "n/a"


def _iter_artifacts(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    if not root.exists():
        return out
    for path in sorted(root.glob("*.json")):
        if path.stem.startswith("_"):
            continue  # convention: `_`-prefixed artifacts are scratch, not results
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "rows" in payload:
            out.append((path, payload))
    return out


def _row_values(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Estimate]]]:
    """Read rows from any phase's artifact, skipping cells we cannot parse.

    `docs/benchmarks.md` aggregates every benchmark in the repo, so this has to
    survive schemas it does not own. A malformed cell is dropped, never fatal:
    one phase writing an odd row must not take the whole report down.
    """
    rows: list[tuple[str, dict[str, Estimate]]] = []
    for row in payload.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        name = str(
            row.get("name")
            or row.get("config")
            or row.get("system")
            or row.get("variant")
            or row.get("label")
            or row.get("arm")
            or row.get("id")
            or "?"
        )
        values: dict[str, Estimate] = {}
        for key, value in row.items():
            if not isinstance(value, Mapping) or "mean" not in value:
                continue
            try:
                values[key] = Estimate.from_dict(value)
            except (KeyError, TypeError, ValueError):
                continue
        rows.append((name, values))
    return rows


def regenerate_benchmarks_md(
    artifacts_root: Path = DEFAULT_ARTIFACTS,
    out_path: Path = DEFAULT_DOCS,
    *,
    gate: GateResult | None = None,
) -> Path:
    """Rebuild `docs/benchmarks.md` from everything in `artifacts/benchmarks/`."""
    artifacts = _iter_artifacts(artifacts_root)
    merged: dict[str, dict[str, Estimate]] = {}
    hardware: set[str] = set()
    for _, payload in artifacts:
        hardware.add(str(payload.get("hardware", "")))
        for name, values in _row_values(payload):
            merged.setdefault(name, {}).update(values)

    lines: list[str] = [_PREAMBLE, "", "## The headline experiment (§14)", ""]
    header = "| System | " + " | ".join(c for c, _ in _HEADLINE_COLUMNS) + " |"
    lines += [header, "|" + "---|" * (len(_HEADLINE_COLUMNS) + 1)]
    for system in HEADLINE_SYSTEMS:
        values = _match_system(merged, system)
        cells = [
            _fmt(values.get(key), 2 if key.endswith("_ms") else 3) for _, key in _HEADLINE_COLUMNS
        ]
        lines.append("| " + " | ".join([system, *cells]) + " |")
    lines += [
        "",
        "Rows 6 vs 7 are the payoff: comparable quality at a fraction of the "
        "latency and CPU cost. `n/a` means that configuration has not been "
        "measured yet -- a missing measurement is reported as missing, never "
        "as zero.",
        "",
    ]

    if gate is not None:
        lines += [
            "## CI gate",
            "",
            f"- metric: `{gate.metric}`",
            f"- budget: {gate.max_regression:.2%} ({'relative' if gate.relative else 'absolute'})",
            f"- current: {_fmt(gate.current)}",
            f"- baseline: {_fmt(gate.baseline)}",
            f"- status: **{'PASS' if gate.passed else 'FAIL'}** -- {gate.message}",
            "",
        ]
        if gate.comparison is not None:
            lines += [
                "Paired comparison against the baseline:",
                "",
                gate.comparison.to_markdown(),
                "",
            ]

    for path, payload in artifacts:
        # Two phases can legitimately name a benchmark the same thing; the file
        # stem disambiguates so headings stay unique.
        bench_name = str(payload.get("name", path.stem))
        heading = bench_name if bench_name == path.stem else f"{bench_name} (`{path.stem}`)"
        lines += [f"## {heading}", ""]
        seeds = payload.get("seeds") or []
        lines.append(
            f"_source: `{path.as_posix()}`; hardware: {payload.get('hardware', 'unknown')}; "
            f"seeds: {seeds}; CI: {payload.get('ci', 'bootstrap95')}_"
        )
        lines.append("")
        dataset_note = _dataset_note(payload)
        if dataset_note:
            lines += [dataset_note, ""]
        for name, values in _row_values(payload):
            if not values:
                continue
            lines += [f"**{name}**", "", "| metric | value (95% CI) |", "|---|---|"]
            for key in sorted(values):
                lines.append(f"| {key} | {values[key].format()} |")
            lines.append("")
        detail = payload.get("detail")
        if isinstance(detail, Mapping):
            cal = detail.get("calibration")
            if isinstance(cal, Mapping):
                lines += [_calibration_block(cal), ""]

    if len(seeds := _all_seeds(artifacts)) < 3:
        lines += [
            "> **Caveat.** Fewer than 3 seeds are represented in these artifacts "
            f"(seen: {seeds}). §20 rule 5 requires 3+ seeds for anything "
            "stochastic; treat single-seed rows as provisional.",
            "",
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path


def _all_seeds(artifacts: Sequence[tuple[Path, dict[str, Any]]]) -> list[int]:
    seeds: set[int] = set()
    for _, payload in artifacts:
        for s in payload.get("seeds") or []:
            seeds.add(int(s))
    return sorted(seeds)


def _match_system(merged: Mapping[str, dict[str, Estimate]], system: str) -> dict[str, Estimate]:
    """Match a headline row to measured rows by name, then by leading number."""
    if system in merged:
        return merged[system]
    prefix = system.split(".", 1)[0].strip()
    for name, values in merged.items():
        if name.strip().startswith(f"{prefix}.") or name.strip() == prefix:
            return values
    return {}


def _dataset_note(payload: Mapping[str, Any]) -> str:
    """Name the golden set behind a row, so a sample can never pass for a release."""
    detail = payload.get("detail")
    if not isinstance(detail, Mapping):
        return ""
    dataset = detail.get("dataset")
    if not isinstance(dataset, Mapping):
        return ""
    name = str(dataset.get("name", "unknown"))
    sha = str(dataset.get("sha256", ""))[:12]
    note = (
        f"Golden set: `{name}` (sha256 `{sha}`), "
        f"{dataset.get('granularity', 'chunk')}-level relevance."
    )
    if "sample" in name:
        note += (
            " **This is the committed sample set on a 28-chunk toy corpus** -- a harness "
            "self-test that keeps the CI gate meaningful before the real corpus exists, "
            "not a system result. Numbers here are not comparable to a release run."
        )
    return f"_{note}_"


def _calibration_block(cal: Mapping[str, Any]) -> str:
    kappa = cal.get("kappa_pairwise")
    kappa_str = Estimate.from_dict(kappa).format() if isinstance(kappa, Mapping) else "n/a"
    swap = cal.get("swap_rate")
    swap_str = Estimate.from_dict(swap).format() if isinstance(swap, Mapping) else "n/a"
    trusted = cal.get("trust_pairwise")
    verdict = cal.get("verdict", "")
    return (
        f"_Judge calibration: `{cal.get('judge')}` on n={cal.get('n')} human labels, "
        f"Cohen's kappa {kappa_str}, swap rate {swap_str}, "
        f"trusted: {'yes' if trusted else 'NO'}._\n\n{verdict}"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m localmind.eval.report",
        description="CI gate on retrieval quality, and regeneration of docs/benchmarks.md.",
    )
    parser.add_argument("--baseline", default="main", help="git ref or baselines/<name> directory")
    parser.add_argument(
        "--baseline-file",
        default=None,
        help=(
            "explicit path to the baseline artifact; overrides --baseline. Preferred in CI, "
            "where artifacts/ is gitignored: check out the base ref, run the eval, keep the "
            "JSON, then compare against it."
        ),
    )
    parser.add_argument("--max-regression", type=float, default=0.02)
    parser.add_argument("--metric", default=GATE_METRIC)
    parser.add_argument("--artifact", default=GATE_ARTIFACT, help="artifact stem to gate on")
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument("--docs", default=str(DEFAULT_DOCS))
    parser.add_argument("--no-docs", action="store_true", help="skip regenerating benchmarks.md")
    parser.add_argument("--absolute", action="store_true", help="threshold in metric units")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json-out", default=None, help="write the gate result as JSON")
    args = parser.parse_args(argv)

    root = Path(args.artifacts)
    current_path = root / f"{args.artifact}.json"
    if not current_path.exists():
        print(
            f"[eval.report] {current_path} not found -- run `just eval-retrieval` first",
            file=sys.stderr,
        )
        return 2
    current_payload = json.loads(current_path.read_text(encoding="utf-8"))
    if args.baseline_file:
        baseline_path = Path(args.baseline_file)
        if not baseline_path.exists():
            print(f"[eval.report] baseline file {baseline_path} not found", file=sys.stderr)
            return 2
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline_payload = default_baseline_source(root).get(args.baseline, args.artifact)

    comparison = (
        paired_against_baseline(current_payload, baseline_payload, args.metric, seed=args.seed)
        if baseline_payload
        else None
    )
    gate = check_regression(
        extract_metric(current_payload, args.metric),
        extract_metric(baseline_payload, args.metric) if baseline_payload else None,
        metric=args.metric,
        max_regression=args.max_regression,
        relative=not args.absolute,
        comparison=comparison,
    )

    print(gate.message)
    if comparison is not None:
        print()
        print(comparison.to_markdown())
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(gate.to_dict(), indent=2), encoding="utf-8")

    if not args.no_docs:
        docs = regenerate_benchmarks_md(root, Path(args.docs), gate=gate)
        print(f"regenerated {docs}")

    return gate.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
