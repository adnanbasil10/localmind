"""LocalMind post-training (implementation.md SS9, Phase 5) -- "where 31M becomes useful".

Four stages, each with an eval, plus the deliverable that makes them mean something:

``sft``    SS9 5a  supervised fine-tuning on the three `ControlPlane` jobs, prompt-masked
``kd``     SS9 5b  three distillation arms, benchmarked against each other (ADR 0005)
``dpo``    SS9 5c  preference optimisation on the rewriter, ``beta = 0.1``
``grpo``   SS9 5d  verifiable-reward RL, group size 8, no value network
``lora``   SS9 5e  r=16 adapters, for the ``Qwen3-4B + LoRA`` row of the matrix

**The comparison matrix (SS9 5e) lives here**, in the package root, because it is the one
artifact that spans every stage: it compares prompting against retrieval against
fine-tuning against distillation, so it belongs to none of the individual modules. See
:func:`build_comparison_matrix`.

The matrix ships with **every cell empty and explicitly marked ``not-run``**. Filling the
Qwen3-4B rows needs a 4B model and a GPU; filling the LocalMind rows needs trained
checkpoints. Neither exists on the machine this package was written on, and a plausible
number in a results table is worse than a blank one -- it is the only kind of error that
survives review. :func:`render_matrix_markdown` therefore prints ``not-yet-run`` rather
than a dash, so an unfilled cell cannot be misread as a zero or as "not applicable".

Import cost: this module pulls in nothing heavier than ``pydantic``. The submodules need
torch and are resolved lazily through ``__getattr__``, so ``import localmind.post`` is
safe in a CLI, a test collector, or a laptop with no torch installed at all.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from localmind.post import dpo, grpo, kd, lora, sft

__all__ = [
    "MATRIX_COLUMNS",
    "MATRIX_ROWS",
    "NOT_RUN",
    "ComparisonMatrix",
    "MatrixCell",
    "MatrixRow",
    "build_comparison_matrix",
    "dod_verdict",
    "dpo",
    "grpo",
    "kd",
    "lora",
    "render_matrix_markdown",
    "sft",
    "write_matrix_artifact",
]

#: Printed in any cell that has not been measured. Deliberately not ``"-"``, ``"n/a"`` or
#: ``""``: those all read as legitimate results at a glance, and this must not.
NOT_RUN = "not-yet-run"

#: The metric columns of the SS9 5e table, in the spec's order.
MATRIX_COLUMNS: tuple[str, ...] = (
    "router_acc",
    "grader_f1",
    "rewrite_win_rate",
    "p50_latency_ms",
)

#: The six systems SS9 5e compares, verbatim, with the two columns the spec fixes for
#: each row (``params updated`` and ``hardware``) filled in from the spec itself. Those
#: two are properties of the *method*, not measurements, so they are known in advance;
#: everything else has to be run.
MATRIX_ROWS: tuple[tuple[str, str, str], ...] = (
    ("Qwen3-4B, zero-shot prompt", "0", "laptop GPU/CPU"),
    ("Qwen3-4B, few-shot prompt", "0", "laptop"),
    ("Qwen3-4B + RAG context", "0", "laptop"),
    ("Qwen3-4B + LoRA (r=16)", "~0.1%", "T4, ~2 GPU-h"),
    ("LocalMind-31M, SFT only", "100%", "laptop CPU"),
    ("LocalMind-31M, SFT + KD + GRPO", "100%", "laptop CPU"),
)

_LAZY = {
    "sft": "localmind.post.sft",
    "kd": "localmind.post.kd",
    "dpo": "localmind.post.dpo",
    "grpo": "localmind.post.grpo",
    "lora": "localmind.post.lora",
}


def __getattr__(name: str) -> Any:
    """Resolve the torch-dependent submodules on first touch, not at import."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(__all__)


# ------------------------------------------------------------------------------------ #
# SS9 5e -- the comparison matrix
# ------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class MatrixCell:
    """One measurement, or an explicit statement that it has not been made.

    ``mean``/``lo``/``hi`` rather than a bare float because CONVENTIONS.md rule 5 forbids
    reporting a bare number: >=3 seeds and a bootstrap 95% CI, or nothing. A cell with a
    ``mean`` but no interval is rejected at construction, so the rule cannot be dodged by
    populating the matrix from a single run.
    """

    mean: float | None = None
    lo: float | None = None
    hi: float | None = None
    n_seeds: int = 0
    note: str = ""

    def __post_init__(self) -> None:
        if self.mean is None:
            return
        if self.lo is None or self.hi is None:
            raise ValueError(
                "a measured cell needs a confidence interval (CONVENTIONS.md rule 5: "
                "never report a bare number)"
            )
        if self.lo > self.hi:
            raise ValueError(f"lo ({self.lo}) must not exceed hi ({self.hi})")
        if self.n_seeds < 3:
            raise ValueError(
                f"a measured cell needs >=3 seeds (CONVENTIONS.md rule 5), got {self.n_seeds}"
            )

    @property
    def measured(self) -> bool:
        return self.mean is not None

    def render(self, digits: int = 3) -> str:
        if not self.measured:
            return NOT_RUN
        assert self.mean is not None and self.lo is not None and self.hi is not None
        return f"{self.mean:.{digits}f} [{self.lo:.{digits}f}, {self.hi:.{digits}f}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "lo": self.lo,
            "hi": self.hi,
            "n_seeds": self.n_seeds,
            "status": "measured" if self.measured else "not-run",
            "note": self.note,
        }


@dataclass
class MatrixRow:
    """One system in the SS9 5e table."""

    system: str
    params_updated: str
    hardware: str
    cells: dict[str, MatrixCell] = field(default_factory=dict)

    def cell(self, column: str) -> MatrixCell:
        return self.cells.get(column, MatrixCell())

    @property
    def complete(self) -> bool:
        return all(self.cell(c).measured for c in MATRIX_COLUMNS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.system,
            "params_updated": self.params_updated,
            "hardware": self.hardware,
            "complete": self.complete,
            **{c: self.cell(c).to_dict() for c in MATRIX_COLUMNS},
        }


@dataclass
class ComparisonMatrix:
    """The SS9 5e deliverable: six rows, six columns, and an honest completion state."""

    rows: list[MatrixRow]
    hardware_note: str = "no GPU available in this environment"
    seeds: tuple[int, ...] = ()

    def row(self, system: str) -> MatrixRow:
        for r in self.rows:
            if r.system == system:
                return r
        raise KeyError(f"no such row: {system!r}")

    @property
    def n_measured(self) -> int:
        return sum(1 for r in self.rows for c in MATRIX_COLUMNS if r.cell(c).measured)

    @property
    def n_cells(self) -> int:
        return len(self.rows) * len(MATRIX_COLUMNS)

    @property
    def complete(self) -> bool:
        return self.n_measured == self.n_cells

    def to_dict(self) -> dict[str, Any]:
        """The CONVENTIONS.md artifact envelope: ``{name, hardware, seeds, rows, ci}``."""
        return {
            "name": "phase5_comparison_matrix",
            "hardware": self.hardware_note,
            "seeds": list(self.seeds),
            "rows": [r.to_dict() for r in self.rows],
            "ci": "bootstrap95",
            "columns": list(MATRIX_COLUMNS),
            "cells_measured": self.n_measured,
            "cells_total": self.n_cells,
            "complete": self.complete,
            "dod": dod_verdict(self),
        }


def build_comparison_matrix(
    measurements: Mapping[str, Mapping[str, MatrixCell]] | None = None,
    *,
    hardware_note: str = "no GPU available in this environment",
    seeds: Sequence[int] = (),
) -> ComparisonMatrix:
    """Build the SS9 5e table, filling only cells that were actually measured.

    Args:
        measurements: ``{system_name: {column: MatrixCell}}``. Anything absent stays
            :data:`NOT_RUN`. There is no code path that populates a cell with a default,
            an estimate, or a value carried over from another row.
        hardware_note: the machine the measured cells were produced on. CONVENTIONS.md
            rule 5: state the hardware.
        seeds: the seeds behind the measured cells.

    Raises:
        KeyError: if ``measurements`` names a system that is not one of the six rows, or
            a column that is not one of the four metrics. A typo'd row name would
            otherwise be silently dropped and read as "not run".
    """
    rows = [MatrixRow(system=s, params_updated=p, hardware=h) for s, p, h in MATRIX_ROWS]
    index = {r.system: r for r in rows}

    for system, cells in (measurements or {}).items():
        if system not in index:
            raise KeyError(
                f"unknown system {system!r}; the SS9 5e rows are {[s for s, _, _ in MATRIX_ROWS]}"
            )
        for column, cell in cells.items():
            if column not in MATRIX_COLUMNS:
                raise KeyError(f"unknown column {column!r}; expected one of {MATRIX_COLUMNS}")
            index[system].cells[column] = cell

    return ComparisonMatrix(rows=rows, hardware_note=hardware_note, seeds=tuple(seeds))


_HEADERS = {
    "router_acc": "Router acc",
    "grader_f1": "Grader F1",
    "rewrite_win_rate": "Rewrite win-rate",
    "p50_latency_ms": "p50 latency (ms)",
}


def render_matrix_markdown(matrix: ComparisonMatrix, digits: int = 3) -> str:
    """The spec's six-row table, with unmeasured cells reading ``not-yet-run``."""
    header = ["Method", "Params updated", *(_HEADERS[c] for c in MATRIX_COLUMNS), "Hardware"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in matrix.rows:
        cells = [row.cell(c).render(digits) for c in MATRIX_COLUMNS]
        lines.append(
            "| " + " | ".join([row.system, row.params_updated, *cells, row.hardware]) + " |"
        )
    lines.append("")
    lines.append(
        f"{matrix.n_measured}/{matrix.n_cells} cells measured on: {matrix.hardware_note}. "
        f"`{NOT_RUN}` means exactly that -- no GPU and no trained checkpoint were available, "
        "so no number was produced. None of these cells is estimated."
    )
    return "\n".join(lines)


def dod_verdict(matrix: ComparisonMatrix) -> dict[str, Any]:
    """Evaluate SS9's Definition of Done, or say why it cannot be evaluated yet.

    The DoD: *the 31M model beats the 4B on latency by >20x at >=95% of its accuracy on at
    least one of the three tasks, running on CPU with no GPU at all.* If it does not, that
    is a publishable negative result and must be reported as one -- but "we could not
    measure it" is a third state, and collapsing it into "failed" would be just as
    dishonest as collapsing it into "passed".
    """
    small = "LocalMind-31M, SFT + KD + GRPO"
    big = "Qwen3-4B, zero-shot prompt"
    quality_cols = ("router_acc", "grader_f1", "rewrite_win_rate")

    try:
        small_row = matrix.row(small)
        big_row = matrix.row(big)
    except KeyError as exc:  # pragma: no cover - rows are fixed
        return {"status": "not-evaluable", "reason": str(exc)}

    small_lat = small_row.cell("p50_latency_ms")
    big_lat = big_row.cell("p50_latency_ms")
    missing: list[str] = []
    if not small_lat.measured:
        missing.append(f"{small}/p50_latency_ms")
    if not big_lat.measured:
        missing.append(f"{big}/p50_latency_ms")
    for col in quality_cols:
        if not small_row.cell(col).measured:
            missing.append(f"{small}/{col}")
        if not big_row.cell(col).measured:
            missing.append(f"{big}/{col}")

    if missing:
        return {
            "status": "not-evaluable",
            "reason": (
                "the DoD compares measured latency and accuracy; these cells have not "
                "been run (no GPU, no trained checkpoint)"
            ),
            "missing_cells": missing,
        }

    assert small_lat.mean is not None and big_lat.mean is not None
    speedup = big_lat.mean / small_lat.mean if small_lat.mean else float("inf")
    ratios: dict[str, float] = {}
    for col in quality_cols:
        b = big_row.cell(col).mean
        s = small_row.cell(col).mean
        assert b is not None and s is not None
        ratios[col] = s / b if b else float("inf")

    tasks_passing = [c for c, ratio in ratios.items() if ratio >= 0.95]
    passed = speedup > 20.0 and bool(tasks_passing)
    return {
        "status": "passed" if passed else "failed",
        "latency_speedup": speedup,
        "accuracy_ratios": ratios,
        "tasks_at_95pct": tasks_passing,
        "note": (
            ""
            if passed
            else "negative result: report it honestly and state what would close the gap "
            "(CONVENTIONS.md rule 2)"
        ),
    }


def write_matrix_artifact(
    matrix: ComparisonMatrix,
    json_path: str | Path = "artifacts/benchmarks/phase5_comparison_matrix.json",
    markdown_path: str | Path | None = None,
) -> Path:
    """Write the CONVENTIONS.md JSON envelope, and optionally append the markdown table."""
    jp = Path(json_path)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(matrix.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    if markdown_path is not None:
        mp = Path(markdown_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with mp.open("a", encoding="utf-8") as fh:
            fh.write("\n\n## Phase 5 (SS9 5e) — prompt vs retrieve vs finetune vs distill\n\n")
            fh.write(render_matrix_markdown(matrix))
            fh.write("\n")
    return jp
