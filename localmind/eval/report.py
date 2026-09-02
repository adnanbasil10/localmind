"""The CI gate and the deliverable: `docs/benchmarks.md`.

Two jobs, both driven by the justfile::

    just eval-retrieval   ->  python -m localmind.eval.retrieval --config ...
    just eval-compare     ->  python -m localmind.eval.report --baseline main \\
                                     --max-regression 0.02

The gate fails the build (exit 1) when nDCG@10 drops more than the allowed
fraction against the baseline.  Where the baseline and the current run share
question ids, it also runs the paired tests from ``stats.py`` so the verdict
carries a CI and a p-value rather than a bare delta.

**Pass ``--require-baseline`` in CI.**  Without it, a baseline that cannot be
resolved is a *skip* that exits 0 -- correct on a laptop and on a genuine first
run, and catastrophic in CI, where it makes "I could not find anything to
compare against" indistinguishable from "I compared and it was fine".  That is
not hypothetical: this gate ran green on every PR without ever having compared
anything, and ``docs/benchmarks.md`` recorded it as ``status: **PASS**``.  With
the flag, an unresolvable baseline exits 3 and the message names the ref, every
path tried, and git's own reason for each failure.
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
from typing import Any, Literal, Protocol, runtime_checkable

from localmind.eval.stats import (
    DEFAULT_SEED,
    ComparisonReport,
    Estimate,
    compare_configs,
)

__all__ = [
    "BaselineLookup",
    "BaselineSource",
    "ChainedBaselineSource",
    "GateResult",
    "GitBaselineSource",
    "LocalBaselineSource",
    "check_regression",
    "extract_metric",
    "extract_per_question",
    "main",
    "read_contributed_sections",
    "regenerate_benchmarks_md",
    "write_contributed_section",
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

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_NO_CURRENT = 2
EXIT_NO_BASELINE = 3
"""CLI exit codes, distinct on purpose.

`1` means the gate ran and the metric regressed. `2` and `3` mean the gate could
not run at all -- no current measurement, or no baseline under
``--require-baseline``. Collapsing "could not run" into "passed" is the bug this
module used to have; collapsing it into "regressed" would be almost as
misleading to whoever reads the CI log.
"""

NO_GIT_ENV = "LOCALMIND_EVAL_NO_GIT"


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
BaselineStatus = Literal["found", "absent", "error"]


@dataclass(frozen=True)
class BaselineLookup:
    """The outcome of looking for a baseline -- including *why* there isn't one.

    ``payload is None`` used to be the entire answer, and that collapsed three
    unrelated situations into one green build: a genuine first run, a typo'd ref,
    and git not being installed all looked identical. They are not the same
    thing, so they no longer share a status:

    * ``found``  -- resolved; compare against it.
    * ``absent`` -- the lookup worked and there is genuinely nothing there yet.
      Legitimate on a first run, and the only case where skipping is honest.
    * ``error``  -- the lookup itself broke. "No baseline" here is a statement
      about our plumbing, not about the repo, and must never read as a pass.

    ``detail`` always names what was tried and what went wrong, because "no
    baseline" with nothing further is exactly what made the old gate impossible
    to debug from a CI log.
    """

    status: BaselineStatus
    detail: str
    payload: dict[str, Any] | None = None

    @property
    def resolved(self) -> bool:
        return self.status == "found" and self.payload is not None

    @staticmethod
    def none_supplied(metric: str = "the metric") -> BaselineLookup:
        return BaselineLookup("absent", f"no baseline was supplied for {metric}")


@runtime_checkable
class BaselineSource(Protocol):
    """Where a baseline benchmark artifact comes from."""

    def get(self, ref: str, artifact: str) -> dict[str, Any] | None: ...

    def lookup(self, ref: str, artifact: str) -> BaselineLookup: ...


@dataclass
class LocalBaselineSource:
    """``artifacts/benchmarks/baselines/<ref>/<artifact>.json`` -- offline, no git."""

    root: Path = DEFAULT_ARTIFACTS

    def lookup(self, ref: str, artifact: str) -> BaselineLookup:
        path = self.root / "baselines" / ref / f"{artifact}.json"
        where = f"local baseline `{path.as_posix()}`"
        if not path.exists():
            return BaselineLookup("absent", f"{where}: no such file")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # The file is there and unreadable. That is a broken baseline, not a
            # missing one, and must not be reported as "first run".
            return BaselineLookup("error", f"{where}: {type(exc).__name__}: {exc}")
        return BaselineLookup("found", where, payload)

    def get(self, ref: str, artifact: str) -> dict[str, Any] | None:
        return self.lookup(ref, artifact).payload


@dataclass
class GitBaselineSource:
    """``git show <ref>:artifacts/benchmarks/<artifact>.json``.

    Optional -- set ``LOCALMIND_EVAL_NO_GIT=1`` to disable it outright (tests do)
    -- but no longer silent. `lookup` separates "that ref has no artifact yet"
    from "that ref does not exist / git is broken", which is the difference
    between a first run and a typo in the workflow.
    """

    root: Path = DEFAULT_ARTIFACTS
    timeout_s: float = 20.0

    def _run(self, args: Sequence[str]) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None

    def _ref_exists(self, ref: str) -> bool | None:
        """True/False if we could ask git, None if we could not."""
        out = self._run(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"])
        if out is None:
            return None
        return out.returncode == 0

    def lookup(self, ref: str, artifact: str) -> BaselineLookup:
        rel = (self.root / f"{artifact}.json").as_posix()
        where = f"`git show {ref}:{rel}`"
        if os.environ.get(NO_GIT_ENV) == "1":
            return BaselineLookup("absent", f"{where}: skipped ({NO_GIT_ENV}=1)")
        out = self._run(["show", f"{ref}:{rel}"])
        if out is None:
            return BaselineLookup("error", f"{where}: git could not be run (missing, or timed out)")
        if out.returncode != 0 or not out.stdout.strip():
            # Ask git whether the *ref* resolves rather than parsing its English
            # error text, so this classification survives locales and versions.
            exists = self._ref_exists(ref)
            stderr = " ".join(out.stderr.split())[:200]
            if exists is False:
                return BaselineLookup("error", f"{where}: ref {ref!r} does not resolve -- {stderr}")
            if exists is None:
                return BaselineLookup("error", f"{where}: could not verify ref {ref!r} -- {stderr}")
            return BaselineLookup(
                "absent", f"{where}: ref {ref!r} resolves but has no {rel} yet -- {stderr}"
            )
        try:
            payload = json.loads(out.stdout)
        except json.JSONDecodeError as exc:
            return BaselineLookup("error", f"{where}: baseline JSON is unparseable: {exc}")
        return BaselineLookup("found", where, payload)

    def get(self, ref: str, artifact: str) -> dict[str, Any] | None:
        return self.lookup(ref, artifact).payload


@dataclass
class ChainedBaselineSource:
    sources: Sequence[BaselineSource]

    def lookup(self, ref: str, artifact: str) -> BaselineLookup:
        """First hit wins; otherwise report every place we looked.

        An ``error`` anywhere in the chain is preserved even when a later source
        merely says ``absent`` -- a broken lookup must not be laundered into a
        clean "first run" by the source that happened to run after it.
        """
        attempts: list[BaselineLookup] = []
        for source in self.sources:
            found = source.lookup(ref, artifact)
            if found.resolved:
                return found
            attempts.append(found)
        status: BaselineStatus = "error" if any(a.status == "error" for a in attempts) else "absent"
        detail = "; ".join(a.detail for a in attempts) or f"no sources configured for {ref!r}"
        return BaselineLookup(status, detail)

    def get(self, ref: str, artifact: str) -> dict[str, Any] | None:
        return self.lookup(ref, artifact).payload


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
    baseline_status: BaselineStatus = "found"
    baseline_detail: str = ""
    ran: bool = True
    """False when the gate never got to compare anything.

    `passed` alone is ambiguous -- it is also True when the gate skipped -- so
    every consumer that reports a verdict reads `ran` too. `docs/benchmarks.md`
    prints SKIPPED rather than PASS when this is False.
    """

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
        if self.passed:
            return EXIT_OK
        if self.current is None:
            return EXIT_NO_CURRENT
        return EXIT_REGRESSION if self.ran else EXIT_NO_BASELINE

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "ran": self.ran,
            "metric": self.metric,
            "current": self.current.to_dict() if self.current else None,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "baseline_status": self.baseline_status,
            "baseline_detail": self.baseline_detail,
            "delta_abs": self.delta_abs,
            "delta_rel": self.delta_rel,
            "max_regression": self.max_regression,
            "relative": self.relative,
            "message": self.message,
            "exit_code": self.exit_code,
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
    lookup: BaselineLookup | None = None,
    require_baseline: bool = False,
) -> GateResult:
    """Fail when ``metric`` drops by more than ``max_regression``.

    Relative by default -- "drops more than 2%" reads as 2% *of the baseline*.
    Pass ``relative=False`` for an absolute threshold in metric units.

    ``require_baseline`` is the CI setting and the reason this function is not
    simply ``drop <= allowed``. Without it, a missing baseline skips the gate and
    passes, which is right on a developer's laptop and on the very first run.
    With it, the caller is asserting that a baseline *must* exist, so failing to
    resolve one is a build failure: a gate that returns 0 having compared
    nothing is indistinguishable from a gate that compared and approved.

    ``lookup`` carries *why* there is no baseline (see `BaselineLookup`). It only
    affects the message and the recorded status -- ``--require-baseline`` fails on
    ``absent`` and ``error`` alike, because the caller said one was required
    either way -- but that message is the difference between a five-second fix
    and an afternoon.
    """
    if current is None:
        return GateResult(
            passed=False,
            ran=False,
            metric=metric,
            current=None,
            baseline=baseline,
            max_regression=max_regression,
            relative=relative,
            baseline_status=lookup.status if lookup else "found",
            baseline_detail=lookup.detail if lookup else "",
            message=f"no current value for {metric}: run the retrieval eval first",
        )
    if baseline is None:
        found = lookup or BaselineLookup.none_supplied(metric)
        because = (
            "no baseline exists yet (legitimate on a first run)"
            if found.status == "absent"
            else "the baseline lookup FAILED -- this is a defect, not a first run"
        )
        if require_baseline:
            return GateResult(
                passed=False,
                ran=False,
                metric=metric,
                current=current,
                baseline=None,
                max_regression=max_regression,
                relative=relative,
                baseline_status=found.status,
                baseline_detail=found.detail,
                message=(
                    f"GATE COULD NOT RUN: --require-baseline was given but no baseline for "
                    f"{metric} could be resolved -- {because}. Tried: {found.detail}. "
                    f"Current value is {current.format()}; refusing to report a pass for a "
                    f"comparison that never happened."
                ),
            )
        return GateResult(
            passed=True,
            ran=False,
            metric=metric,
            current=current,
            baseline=None,
            max_regression=max_regression,
            relative=relative,
            baseline_status=found.status,
            baseline_detail=found.detail,
            message=(
                f"SKIPPED (not a pass): no baseline for {metric} -- {because}. "
                f"Tried: {found.detail}. Current value recorded as the baseline candidate "
                f"({current.format()}). Pass --require-baseline to make this a failure."
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
        baseline_status=lookup.status if lookup else "found",
        baseline_detail=lookup.detail if lookup else "",
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

> **Rule 5, and exactly how far it reaches.** Every cell in a *generated* table
> below is `mean [lo, hi]` -- a point estimate with a bootstrap 95% CI -- and
> that is enforced in code: `localmind.eval.stats.Estimate` refuses to be
> coerced to a bare float, and `benchmark_json()` rejects any row value that is
> not an `Estimate`. A bare point estimate in a generated table is a bug.
>
> The **contributed sections** (marked as such, below the generated ones) are
> rendered by each phase's own harness from the same committed artifacts. They
> carry CIs on every stochastic measurement, but they also carry exact,
> non-stochastic quantities -- byte counts, tensor counts, block-occupancy
> ratios -- as bare numbers, because an interval on an exact count would be
> noise dressed as rigour. The enforcement above does not cover those, so do not
> read a missing interval there as an evaded one.
>
> Cost per query is $0 -- everything runs locally -- so the resource columns
> are **CPU-seconds per query** and **peak RSS**, which are the numbers that
> actually constrain this system.

_This file is generated by `python -m localmind.eval.report`, which composes the
generated tables with every contributed section in
`artifacts/benchmarks/sections/`. Edit the harness, not the table._
"""

SECTIONS_DIRNAME = "sections"
SUPERSEDED_KEY = "superseded"
SUPERSEDED_BY_KEY = "superseded_by"


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


_PRIMARY_LABEL_KEYS = ("name", "config", "system", "label", "id", "tokenizer", "bench")
_QUALIFIER_LABEL_KEYS = ("variant", "arm")


def _row_label(row: Mapping[str, Any]) -> str:
    """Name a row from whichever identifying key its phase happens to use.

    `bench` was missing from this list, which is why every inference and model
    row rendered as a literal `**?**` -- 72 unlabelled tables in the deliverable.
    A qualifier (`variant`/`arm`) is appended when present, because `kv_cache`
    alone appears eleven times in `inference.json` and the variant is the only
    thing that tells those rows apart.
    """
    primary = next((str(row[k]) for k in _PRIMARY_LABEL_KEYS if row.get(k)), "")
    qualifier = next((str(row[k]) for k in _QUALIFIER_LABEL_KEYS if row.get(k)), "")
    parts = [primary] + ([qualifier] if qualifier and qualifier != primary else [])
    return " / ".join(p for p in parts if p) or "?"


def _superseded_reason(path: Path, payload: Mapping[str, Any]) -> str | None:
    """Why this artifact has been retracted, or None if it still counts.

    Read from the artifact's own `superseded` field, with the `*_superseded_*`
    filename convention as a backstop: a retracted run must not get republished
    as a peer result just because someone forgot to set the field. It used to
    be republished with no marker at all, 55 rows of withdrawn numbers sitting
    beside the good ones and distinguishable only by a filename stem.
    """
    marker = payload.get(SUPERSEDED_KEY)
    if isinstance(marker, str) and marker.strip():
        return marker.strip()
    if marker is True:
        return "marked superseded in the artifact, with no reason recorded"
    if SUPERSEDED_KEY in path.stem:
        return (
            f"the filename marks this run superseded, but `{path.name}` carries no "
            f"`{SUPERSEDED_KEY}` field explaining why"
        )
    return None


def _retraction_block(path: Path, payload: Mapping[str, Any], reason: str) -> list[str]:
    """A retraction notice instead of the data. The file stays; the numbers do not.

    Rendering the rows under a banner was the other option and it is weaker: the
    banner scrolls off after twenty lines and the remaining 300 lines of tables
    look exactly like a result. The audit trail is the JSON on disk, which is
    still committed and still cited here by path.
    """
    superseded_by = str(payload.get(SUPERSEDED_BY_KEY, "")).strip()
    lines = [
        f"## RETRACTED -- {payload.get('name', path.stem)} (`{path.stem}`)",
        "",
        "> **These numbers were withdrawn. Do not cite anything from this run.**",
        f"> Reason: {reason}",
    ]
    if superseded_by:
        lines.append(f"> Superseded by: `{superseded_by}`")
    lines += [
        f"> The artifact is kept at `{path.as_posix()}` so the correction is auditable;"
        " its rows are deliberately not reproduced here.",
        "",
    ]
    return lines


def read_contributed_sections(root: Path = DEFAULT_ARTIFACTS) -> list[tuple[Path, str]]:
    """Markdown contributed by phases whose tables this module cannot generate.

    `docs/benchmarks.md` has three other producers -- `inference.bench`,
    `model.transformer` and `post` -- whose tables encode phase-specific
    structure (speedup-vs-naive columns, GGUF export status, a not-run matrix)
    that a generic `metric | value` renderer cannot express. They used to `open(
    ..., "a")` on the deliverable directly, so the documented regeneration
    command silently deleted all three.

    Composition is now explicit and one-directional: each producer writes the
    section it owns to `artifacts/benchmarks/sections/<order>-<phase>.md`, and
    this module concatenates them in filename order. Nobody appends to the
    deliverable; regeneration is idempotent and no producer can destroy another's
    work by running second.
    """
    directory = root / SECTIONS_DIRNAME
    if not directory.is_dir():
        return []
    return [(p, p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.md"))]


def write_contributed_section(
    name: str, markdown: str, root: Path | str = DEFAULT_ARTIFACTS
) -> Path:
    """Record one producer's section. Overwrites its own file, touches nothing else."""
    path = Path(root) / SECTIONS_DIRNAME / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    return path


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
        name = _row_label(row)
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
    """Rebuild `docs/benchmarks.md` from everything in `artifacts/benchmarks/`.

    Composition, in order: the generated tables (from `*.json` here), then every
    contributed section (from `sections/*.md` here). Retracted artifacts
    contribute a retraction notice and nothing else, and never reach the headline
    table. The output is a pure function of the committed inputs, so running this
    twice is a no-op and running it never destroys another producer's section.
    """
    artifacts = _iter_artifacts(artifacts_root)
    retracted = {
        path: reason for path, payload in artifacts if (reason := _superseded_reason(path, payload))
    }
    merged: dict[str, dict[str, Estimate]] = {}
    hardware: set[str] = set()
    for path, payload in artifacts:
        if path in retracted:
            continue  # withdrawn numbers must never feed the headline table
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
        # SKIPPED is its own word. A gate that compared nothing has not passed,
        # and printing PASS there is how this document came to advertise
        # protection that had never once been exercised.
        status = "PASS" if gate.passed and gate.ran else ("SKIPPED" if gate.passed else "FAIL")
        lines += [
            "## CI gate",
            "",
            f"- metric: `{gate.metric}`",
            f"- budget: {gate.max_regression:.2%} ({'relative' if gate.relative else 'absolute'})",
            f"- current: {_fmt(gate.current)}",
            f"- baseline: {_fmt(gate.baseline)} (lookup: {gate.baseline_status})",
            f"- status: **{status}** -- {gate.message}",
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
        if path in retracted:
            lines += _retraction_block(path, payload, retracted[path])
            continue
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

    contributed = read_contributed_sections(artifacts_root)
    if contributed:
        lines += [
            "---",
            "",
            "# Contributed sections",
            "",
            "Rendered by each phase's own harness from the artifacts above, and "
            "composed here rather than appended to this file. Source: "
            f"`{(artifacts_root / SECTIONS_DIRNAME).as_posix()}/`.",
            "",
        ]
        for path, text in contributed:
            lines += [f"<!-- contributed by {path.as_posix()} -->", text.rstrip(), ""]

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
    parser.add_argument(
        "--require-baseline",
        action="store_true",
        help=(
            "treat an unresolvable baseline as a FAILURE (exit 3) instead of a skip. Set this "
            "in CI: without it the gate exits 0 when it could not find anything to compare "
            "against, which is indistinguishable from a genuine pass. Leave it off locally, "
            "where a missing baseline is normal."
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
        return EXIT_NO_CURRENT
    current_payload = json.loads(current_path.read_text(encoding="utf-8"))
    if args.baseline_file:
        # An explicitly named file that is missing or corrupt is operator error,
        # so it fails closed regardless of --require-baseline. Only *discovery*
        # of a baseline is allowed to come up empty.
        baseline_path = Path(args.baseline_file)
        if not baseline_path.exists():
            print(f"[eval.report] baseline file {baseline_path} not found", file=sys.stderr)
            return EXIT_NO_CURRENT
        try:
            payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"[eval.report] baseline file {baseline_path} is unreadable: {exc}",
                file=sys.stderr,
            )
            return EXIT_NO_CURRENT
        lookup = BaselineLookup("found", f"baseline file `{baseline_path.as_posix()}`", payload)
    else:
        lookup = default_baseline_source(root).lookup(args.baseline, args.artifact)
    baseline_payload = lookup.payload

    comparison = (
        paired_against_baseline(current_payload, baseline_payload, args.metric, seed=args.seed)
        if baseline_payload
        else None
    )
    baseline_metric = extract_metric(baseline_payload, args.metric) if baseline_payload else None
    if lookup.resolved and baseline_metric is None:
        # We found the artifact and it does not carry the metric we gate on.
        # That is a broken baseline, not a missing one -- keep them apart.
        lookup = BaselineLookup(
            "error", f"{lookup.detail}: resolved, but contains no {args.metric!r} metric"
        )
    gate = check_regression(
        extract_metric(current_payload, args.metric),
        baseline_metric,
        metric=args.metric,
        max_regression=args.max_regression,
        relative=not args.absolute,
        comparison=comparison,
        lookup=lookup,
        require_baseline=args.require_baseline,
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
