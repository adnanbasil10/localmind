"""Learning-rate schedules: WSD, and the branch decay that is the whole reason for it.

Warmup-Stable-Decay (ADR 0002, §8). Three phases over ``total_steps``:

    |<- 2% warmup ->|<------------ 80% stable ------------>|<- 18% decay ->|
    0 ------------> peak_lr ------------------------------> linear -> min_lr

**Branch decay is the point.** With cosine you must commit to a total step count up
front and only the final checkpoint is deployable; a run killed at 90% yields nothing.
WSD lets you take *any* checkpoint from the stable phase and spend a short decay to get
a deployable model. Under a 12-hour hard session cap and a 30 GPU-h/week quota that is
the difference between having a model and having a loss curve.

`branch_decay` is therefore a real, tested function, not a comment: it returns a
schedule that reproduces the parent schedule exactly up to the branch point and then
decays to ``min_lr`` over a chosen number of steps. Branching at 250M / 500M / 1B /
1.5B tokens is the §8 scaling-law study (`scaling_law_branches`).

This module is deliberately torch-free: it is pure arithmetic, so it is trivially
testable and cheap to import.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

__all__ = [
    "SCALING_LAW_TOKEN_BUDGETS",
    "BranchDecaySchedule",
    "LRSchedule",
    "WSDSchedule",
    "branch_decay",
    "constant_schedule",
    "lr_trace",
    "rescale",
    "scaling_law_branches",
    "steps_for_tokens",
    "wsd_schedule",
]

#: §8: "decay at 250M, 500M, 1B, 1.5B tokens and plot loss vs compute".
SCALING_LAW_TOKEN_BUDGETS: tuple[int, ...] = (
    250_000_000,
    500_000_000,
    1_000_000_000,
    1_500_000_000,
)


@runtime_checkable
class LRSchedule(Protocol):
    """A callable ``step -> lr`` that knows how long it runs.

    Structural, not a base class: `WSDSchedule` is a frozen dataclass with a
    ``total_steps`` *field* and `BranchDecaySchedule` derives its ``total_steps`` from
    the branch point, and inheritance cannot reconcile those two.
    """

    @property
    def total_steps(self) -> int: ...

    def __call__(self, step: int) -> float: ...

    def phase(self, step: int) -> str: ...


def lr_trace(schedule: LRSchedule) -> list[float]:
    """The full LR trace, for plotting and for tests."""
    return [schedule(s) for s in range(schedule.total_steps)]


@dataclass(frozen=True)
class WSDSchedule:
    """Warmup -> Stable -> linear Decay (§8, ADR 0002).

    Args:
        peak_lr: the stable-phase learning rate.
        total_steps: number of optimizer steps in the *full* run.
        warmup_frac / stable_frac / decay_frac: must sum to 1.0 (within 1e-6).
        min_lr_ratio: ``min_lr = peak_lr * min_lr_ratio``. 0.1 per `pretrain.yaml`.

    Step indices are 0-based: ``step`` is the index of the optimizer step *about to be
    taken*, so ``schedule(0)`` is the LR of the very first update.
    """

    peak_lr: float
    total_steps: int
    warmup_frac: float = 0.02
    stable_frac: float = 0.80
    decay_frac: float = 0.18
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if self.peak_lr <= 0.0:
            raise ValueError(f"peak_lr must be positive, got {self.peak_lr}")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        for name in ("warmup_frac", "stable_frac", "decay_frac"):
            frac = getattr(self, name)
            if not 0.0 <= frac <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {frac}")
        total = self.warmup_frac + self.stable_frac + self.decay_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"warmup_frac + stable_frac + decay_frac must be 1.0, got {total!r} "
                f"({self.warmup_frac} + {self.stable_frac} + {self.decay_frac})"
            )

    # --- phase boundaries ------------------------------------------------------------
    @property
    def min_lr(self) -> float:
        return self.peak_lr * self.min_lr_ratio

    @property
    def warmup_steps(self) -> int:
        """At least 1 whenever ``warmup_frac > 0`` -- a 0-step warmup on a tiny run is
        a silent way to reintroduce the LR spike warmup exists to prevent."""
        n = round(self.warmup_frac * self.total_steps)
        return max(1, n) if self.warmup_frac > 0.0 else 0

    @property
    def decay_steps(self) -> int:
        n = round(self.decay_frac * self.total_steps)
        return max(1, n) if self.decay_frac > 0.0 else 0

    @property
    def decay_start(self) -> int:
        """First step of the decay phase. Stable phase is ``[warmup_steps, decay_start)``."""
        return max(self.warmup_steps, self.total_steps - self.decay_steps)

    @property
    def stable_steps(self) -> int:
        return max(0, self.decay_start - self.warmup_steps)

    def phase(self, step: int) -> str:
        """``"warmup"`` | ``"stable"`` | ``"decay"`` | ``"post"`` -- for logging."""
        if step < self.warmup_steps:
            return "warmup"
        if step < self.decay_start:
            return "stable"
        if step < self.total_steps:
            return "decay"
        return "post"

    def in_stable_phase(self, step: int) -> bool:
        """Can a branch decay legally start here? (`branch_decay` enforces this.)"""
        return self.warmup_steps <= step < self.decay_start

    # --- the schedule itself ---------------------------------------------------------
    def __call__(self, step: int) -> float:
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        if step < self.warmup_steps:
            # (step + 1) / warmup: the first update gets a non-zero LR. A 0.0 first
            # step is a wasted step and hides ordering bugs.
            return self.peak_lr * (step + 1) / self.warmup_steps
        if step < self.decay_start:
            return self.peak_lr
        if step >= self.total_steps:
            return self.min_lr
        progress = (step - self.decay_start) / self.decay_steps
        return self.peak_lr + (self.min_lr - self.peak_lr) * progress


@dataclass(frozen=True)
class BranchDecaySchedule:
    """A short decay branched off a parent WSD run's stable phase.

    Reproduces ``parent`` exactly for ``step < branch_step``; from ``branch_step`` it
    decays linearly ``peak_lr -> min_lr`` over ``decay_steps`` steps. Continuous at the
    branch point by construction: the first branch step still has LR ``peak_lr``, which
    is what the parent's stable phase would have returned.

    The parent run is untouched -- it keeps going. That is the point: one long stable
    run yields N deployable models for the cost of N short decays.
    """

    parent: WSDSchedule
    branch_step: int
    decay_steps: int

    def __post_init__(self) -> None:
        if self.decay_steps <= 0:
            raise ValueError(f"decay_steps must be positive, got {self.decay_steps}")
        if not self.parent.in_stable_phase(self.branch_step):
            raise ValueError(
                f"branch_step={self.branch_step} is in the parent's "
                f"{self.parent.phase(self.branch_step)!r} phase; a branch decay must start "
                f"in the stable phase [{self.parent.warmup_steps}, {self.parent.decay_start})"
            )

    @property
    def total_steps(self) -> int:
        return self.branch_step + self.decay_steps

    @property
    def peak_lr(self) -> float:
        return self.parent.peak_lr

    @property
    def min_lr(self) -> float:
        return self.parent.min_lr

    def phase(self, step: int) -> str:
        if step < self.branch_step:
            return self.parent.phase(step)
        return "decay" if step < self.total_steps else "post"

    def __call__(self, step: int) -> float:
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        if step < self.branch_step:
            return self.parent(step)
        if step >= self.total_steps:
            return self.min_lr
        progress = (step - self.branch_step) / self.decay_steps
        return self.peak_lr + (self.min_lr - self.peak_lr) * progress


def branch_decay(
    parent: WSDSchedule,
    branch_step: int,
    decay_steps: int | None = None,
    decay_frac: float | None = None,
) -> BranchDecaySchedule:
    """Branch a short decay off ``parent`` at ``branch_step``.

    Exactly one of ``decay_steps`` / ``decay_frac`` may be given. ``decay_frac`` is a
    fraction of the tokens seen *so far* (i.e. of ``branch_step``), which is the form
    the WSD literature uses -- "decay over the last 10% of the branch's own budget".
    Defaults to the parent's own ``decay_frac`` applied to ``branch_step``.

    Raises:
        ValueError: if ``branch_step`` is not inside the parent's stable phase. Decaying
            from warmup would be meaningless and decaying from the parent's own decay
            phase is just a shorter version of the parent run.
    """
    if decay_steps is not None and decay_frac is not None:
        raise ValueError("pass decay_steps or decay_frac, not both")
    if decay_steps is None:
        frac = parent.decay_frac if decay_frac is None else decay_frac
        if frac <= 0.0:
            raise ValueError(f"decay_frac must be positive, got {frac}")
        decay_steps = max(1, round(frac * branch_step))
    return BranchDecaySchedule(parent=parent, branch_step=branch_step, decay_steps=decay_steps)


def steps_for_tokens(tokens: int, tokens_per_step: int) -> int:
    """Optimizer steps needed to consume ``tokens`` at ``tokens_per_step``."""
    if tokens_per_step <= 0:
        raise ValueError(f"tokens_per_step must be positive, got {tokens_per_step}")
    return max(1, tokens // tokens_per_step)


def scaling_law_branches(
    parent: WSDSchedule,
    tokens_per_step: int,
    token_budgets: Sequence[int] = SCALING_LAW_TOKEN_BUDGETS,
    decay_frac: float | None = None,
) -> list[BranchDecaySchedule]:
    """The §8 scaling-law study: one branch decay per token budget.

    Budgets whose branch step falls outside the parent's stable phase are dropped --
    e.g. the 1.5B budget on a 1.5B-token parent run *is* the parent's own decay, so
    there is nothing to branch. The caller gets a shorter list, not an exception,
    because "decay at 250M/500M/1B/1.5B" is a wish list against whatever run exists.
    """
    out: list[BranchDecaySchedule] = []
    for budget in token_budgets:
        step = steps_for_tokens(budget, tokens_per_step)
        if not parent.in_stable_phase(step):
            continue
        out.append(branch_decay(parent, step, decay_frac=decay_frac))
    return out


def wsd_schedule(
    peak_lr: float,
    total_steps: int,
    warmup_frac: float = 0.02,
    stable_frac: float = 0.80,
    decay_frac: float = 0.18,
    min_lr_ratio: float = 0.1,
) -> WSDSchedule:
    """Functional constructor, matching the §8 pseudo-code's ``wsd_schedule(step)``."""
    return WSDSchedule(
        peak_lr=peak_lr,
        total_steps=total_steps,
        warmup_frac=warmup_frac,
        stable_frac=stable_frac,
        decay_frac=decay_frac,
        min_lr_ratio=min_lr_ratio,
    )


def constant_schedule(peak_lr: float, total_steps: int, warmup_frac: float = 0.02) -> WSDSchedule:
    """Warmup then flat forever -- the parent of an open-ended stable run.

    Useful when you genuinely do not know the token budget: run this, then
    `branch_decay` whenever the quota runs out.
    """
    return WSDSchedule(
        peak_lr=peak_lr,
        total_steps=total_steps,
        warmup_frac=warmup_frac,
        stable_frac=1.0 - warmup_frac,
        decay_frac=0.0,
        min_lr_ratio=1.0,
    )


def rescale(schedule: WSDSchedule, peak_lr: float) -> WSDSchedule:
    """Same shape, different peak -- the "AdamW at 2x LR" control arm (§8)."""
    return replace(schedule, peak_lr=peak_lr)
