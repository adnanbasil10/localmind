"""Judge calibration -- mandatory, not optional (``implementation.md`` §14).

The judge is a small local model, not a frontier API.  Before any judged metric
is allowed into a report we must prove the judge tracks human labels:

* hand-label 100 examples,
* report judge-vs-human **Cohen's kappa** (with a bootstrap CI),
* below ~0.6 the judge is noise -- say so, and fall back to deterministic
  metrics *automatically*,
* control position bias by randomising A/B order and reporting the **swap
  rate**,
* prefer pairwise comparison to absolute 1-5 scoring.

Everything that talks to a model goes through the :class:`Judge` Protocol, so
the whole module runs offline against deterministic fakes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from localmind.eval.datasets.schema import JudgeLabel, load_judge_labels
from localmind.eval.stats import DEFAULT_SEED, Estimate, kappa_ci, wilson_ci

__all__ = [
    "KAPPA_TRUST_THRESHOLD",
    "CalibrationReport",
    "CoinFlipJudge",
    "HeuristicJudge",
    "Judge",
    "OracleJudge",
    "PairwiseVerdict",
    "PositionBiasedJudge",
    "build_calibration_fixture",
    "calibrate_judge",
    "flip_verdict",
]

PairwiseVerdict = Literal["A", "B", "tie"]
VERDICTS: tuple[PairwiseVerdict, ...] = ("A", "B", "tie")

KAPPA_TRUST_THRESHOLD = 0.6
"""§14: "Below ~0.6 your judge is noise." Metrics below this are not reported."""


# --------------------------------------------------------------------------- #
# The injectable seam
# --------------------------------------------------------------------------- #
@runtime_checkable
class Judge(Protocol):
    """Everything the harness ever asks a judge model to do.

    Two calls only, both cheap for a small model:

    ``compare`` -- pairwise ranking, which small models do far better than
    absolute rating; ``supports`` -- a binary entailment call used by
    faithfulness.  No 1-5 scale anywhere.
    """

    @property
    def name(self) -> str: ...

    def compare(
        self, question: str, answer_a: str, answer_b: str, context: str
    ) -> PairwiseVerdict: ...

    def supports(self, claim: str, context: str) -> bool: ...


def flip_verdict(v: PairwiseVerdict) -> PairwiseVerdict:
    """Map a verdict from a swapped presentation back to canonical order."""
    if v == "A":
        return "B"
    if v == "B":
        return "A"
    return "tie"


# --------------------------------------------------------------------------- #
# Deterministic judges: offline fakes and the no-LLM fallback
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t}


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


@dataclass
class HeuristicJudge:
    """A judge with no model behind it: lexical grounding only.

    This is the *deterministic fallback*.  It is deliberately weak, and
    calibration is expected to say so -- which is the point: the harness
    should be able to demonstrate its own trust gate firing.
    """

    support_threshold: float = 0.6
    margin: float = 0.05
    name: str = "heuristic-lexical"

    def compare(self, question: str, answer_a: str, answer_b: str, context: str) -> PairwiseVerdict:
        del question
        sa = _overlap(answer_a, context)
        sb = _overlap(answer_b, context)
        if abs(sa - sb) < self.margin:
            return "tie"
        return "A" if sa > sb else "B"

    def supports(self, claim: str, context: str) -> bool:
        return _overlap(claim, context) >= self.support_threshold


@dataclass
class OracleJudge:
    """Replays the human labels, optionally corrupted by a known noise rate.

    ``noise=0`` gives kappa == 1.0; raising it degrades agreement in a
    controlled, seeded way, which is how the trust gate is exercised from both
    sides of the 0.6 threshold.
    """

    table: dict[tuple[str, str, str], PairwiseVerdict] = field(default_factory=dict)
    support: dict[tuple[str, str], bool] = field(default_factory=dict)
    noise: float = 0.0
    seed: int = DEFAULT_SEED
    name: str = "oracle"
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    @classmethod
    def from_labels(
        cls, labels: Sequence[JudgeLabel], *, noise: float = 0.0, seed: int = DEFAULT_SEED
    ) -> OracleJudge:
        table: dict[tuple[str, str, str], PairwiseVerdict] = {}
        support: dict[tuple[str, str], bool] = {}
        for item in labels:
            table[(item.question, item.answer_a, item.answer_b)] = item.human_label
            table[(item.question, item.answer_b, item.answer_a)] = flip_verdict(item.human_label)
            if item.a_supported is not None:
                support[(item.answer_a, item.context)] = item.a_supported
            if item.b_supported is not None:
                support[(item.answer_b, item.context)] = item.b_supported
        label = "oracle" if noise == 0.0 else f"oracle-noise{noise:g}"
        return cls(table=table, support=support, noise=noise, seed=seed, name=label)

    def compare(self, question: str, answer_a: str, answer_b: str, context: str) -> PairwiseVerdict:
        del context
        truth: PairwiseVerdict = self.table.get((question, answer_a, answer_b), "tie")
        if self.noise > 0.0 and self._rng.random() < self.noise:
            alternatives: list[PairwiseVerdict] = [v for v in VERDICTS if v != truth]
            return alternatives[int(self._rng.integers(0, len(alternatives)))]
        return truth

    def supports(self, claim: str, context: str) -> bool:
        truth = self.support.get((claim, context), True)
        if self.noise > 0.0 and self._rng.random() < self.noise:
            return not truth
        return truth


@dataclass
class PositionBiasedJudge:
    """Always prefers whatever it is shown first.  Swap rate must be ~1.0."""

    name: str = "position-biased"

    def compare(self, question: str, answer_a: str, answer_b: str, context: str) -> PairwiseVerdict:
        del question, answer_a, answer_b, context
        return "A"

    def supports(self, claim: str, context: str) -> bool:
        del claim, context
        return True


@dataclass
class CoinFlipJudge:
    """Seeded noise.  kappa must land near 0 and the trust gate must fire."""

    seed: int = DEFAULT_SEED
    name: str = "coin-flip"
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def compare(self, question: str, answer_a: str, answer_b: str, context: str) -> PairwiseVerdict:
        del question, answer_a, answer_b, context
        return VERDICTS[int(self._rng.integers(0, 3))]

    def supports(self, claim: str, context: str) -> bool:
        del claim, context
        return bool(self._rng.integers(0, 2))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationReport:
    """Judge-vs-human agreement, and the trust decision that follows from it."""

    judge_name: str
    n: int
    seed: int
    kappa_pairwise: Estimate
    agreement_pairwise: Estimate
    swap_rate: Estimate
    position_preference: Estimate
    kappa_support: Estimate | None = None
    agreement_support: Estimate | None = None
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    threshold: float = KAPPA_TRUST_THRESHOLD

    # -- trust ------------------------------------------------------------- #
    @property
    def trust_pairwise(self) -> bool:
        return self.kappa_pairwise.mean >= self.threshold

    @property
    def trust_support(self) -> bool:
        return self.kappa_support is not None and self.kappa_support.mean >= self.threshold

    @property
    def trusted_metrics(self) -> list[str]:
        out: list[str] = []
        if self.trust_pairwise:
            out.append("answer_relevance_judged")
        if self.trust_support:
            out.append("faithfulness_judged")
        return out

    @property
    def untrusted_metrics(self) -> list[str]:
        judged = ["answer_relevance_judged", "faithfulness_judged"]
        return [m for m in judged if m not in self.trusted_metrics]

    @property
    def verdict(self) -> str:
        if self.trust_pairwise:
            return (
                f"judge {self.judge_name!r} agrees with humans at "
                f"kappa={self.kappa_pairwise.format()} (n={self.n}) -- judged metrics reported"
            )
        return (
            f"JUDGE NOT TRUSTED: {self.judge_name!r} scored "
            f"kappa={self.kappa_pairwise.format()} < {self.threshold} on n={self.n} human "
            "labels. Per implementation.md section 14 the judged metrics are suppressed and "
            "the harness falls back to deterministic metrics."
        )

    @property
    def position_bias_note(self) -> str:
        return (
            f"position control: A/B order randomised (seed={self.seed}); "
            f"swap rate {self.swap_rate.format()} "
            f"(fraction of items whose verdict flips when the order flips); "
            f"first-position preference {self.position_preference.format()} "
            "(0.5 = unbiased)"
        )

    def summary_line(self) -> str:
        return (
            f"{self.judge_name}: kappa={self.kappa_pairwise.format()} "
            f"agreement={self.agreement_pairwise.format()} "
            f"swap={self.swap_rate.format()} "
            f"trusted={'yes' if self.trust_pairwise else 'NO'}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge": self.judge_name,
            "n": self.n,
            "seed": self.seed,
            "threshold": self.threshold,
            "kappa_pairwise": self.kappa_pairwise.to_dict(),
            "agreement_pairwise": self.agreement_pairwise.to_dict(),
            "swap_rate": self.swap_rate.to_dict(),
            "position_preference": self.position_preference.to_dict(),
            "kappa_support": self.kappa_support.to_dict() if self.kappa_support else None,
            "agreement_support": (
                self.agreement_support.to_dict() if self.agreement_support else None
            ),
            "confusion": self.confusion,
            "trust_pairwise": self.trust_pairwise,
            "trust_support": self.trust_support,
            "trusted_metrics": self.trusted_metrics,
            "untrusted_metrics": self.untrusted_metrics,
            "verdict": self.verdict,
        }

    def to_markdown(self) -> str:
        lines = [
            f"### Judge calibration -- `{self.judge_name}`",
            "",
            "| quantity | value (95% CI) |",
            "|---|---|",
            f"| Cohen's kappa (pairwise, vs human) | {self.kappa_pairwise.format()} |",
            f"| raw agreement (pairwise) | {self.agreement_pairwise.format()} |",
            f"| swap rate (A/B order flipped) | {self.swap_rate.format()} |",
            f"| first-position preference | {self.position_preference.format()} |",
        ]
        if self.kappa_support is not None:
            lines.append(f"| Cohen's kappa (support/entailment) | {self.kappa_support.format()} |")
        if self.agreement_support is not None:
            lines.append(f"| raw agreement (support) | {self.agreement_support.format()} |")
        lines += [
            f"| n human labels | {self.n} |",
            f"| trust threshold | kappa >= {self.threshold} |",
            "",
            f"**Verdict.** {self.verdict}",
            "",
            self.position_bias_note,
        ]
        if self.untrusted_metrics:
            lines += ["", f"Suppressed metrics: {', '.join(self.untrusted_metrics)}."]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibrate_judge(
    judge: Judge,
    labels: Sequence[JudgeLabel],
    *,
    seed: int = DEFAULT_SEED,
    threshold: float = KAPPA_TRUST_THRESHOLD,
    n_resamples: int = 2_000,
) -> CalibrationReport:
    """Run the judge against human labels in both orders and score it.

    Each item is judged twice -- ``(a, b)`` and ``(b, a)`` -- so position bias
    is *measured*, not assumed away.  The headline kappa uses one seeded random
    presentation per item, which is the order-randomisation control §14 asks
    for; the swap rate uses both.
    """
    if not labels:
        raise ValueError("calibrate_judge requires at least one labelled example")

    rng = np.random.default_rng(seed)
    human: list[PairwiseVerdict] = []
    judged: list[PairwiseVerdict] = []
    swapped_flags: list[float] = []
    first_position_picks: list[float] = []

    for item in labels:
        v_forward = _coerce(
            judge.compare(item.question, item.answer_a, item.answer_b, item.context)
        )
        v_reverse_raw = _coerce(
            judge.compare(item.question, item.answer_b, item.answer_a, item.context)
        )
        v_reverse = flip_verdict(v_reverse_raw)

        swapped_flags.append(1.0 if v_forward != v_reverse else 0.0)
        for raw in (v_forward, v_reverse_raw):
            if raw != "tie":
                first_position_picks.append(1.0 if raw == "A" else 0.0)

        use_reverse = bool(rng.integers(0, 2))
        judged.append(v_reverse if use_reverse else v_forward)
        human.append(item.human_label)

    kappa = kappa_ci(human, judged, labels=VERDICTS, seed=seed, n_resamples=n_resamples)
    n_agree = sum(1 for h, j in zip(human, judged, strict=True) if h == j)
    agreement = wilson_ci(n_agree, len(human))
    swap = wilson_ci(int(sum(swapped_flags)), len(swapped_flags))
    if first_position_picks:
        pos = wilson_ci(int(sum(first_position_picks)), len(first_position_picks))
    else:
        pos = Estimate(mean=0.5, lo=0.0, hi=1.0, n=0, method="no-non-tie-verdicts")

    confusion: dict[str, dict[str, int]] = {h: {j: 0 for j in VERDICTS} for h in VERDICTS}
    for h, j in zip(human, judged, strict=True):
        confusion[h][j] += 1

    kappa_support: Estimate | None = None
    agreement_support: Estimate | None = None
    support_human: list[bool] = []
    support_judge: list[bool] = []
    for item in labels:
        for answer, truth in ((item.answer_a, item.a_supported), (item.answer_b, item.b_supported)):
            if truth is None:
                continue
            support_human.append(bool(truth))
            support_judge.append(bool(judge.supports(answer, item.context)))
    if len(support_human) >= 2 and len(set(support_human)) > 0:
        kappa_support = kappa_ci(
            support_human, support_judge, labels=(True, False), seed=seed, n_resamples=n_resamples
        )
        n_ok = sum(1 for h, j in zip(support_human, support_judge, strict=True) if h == j)
        agreement_support = wilson_ci(n_ok, len(support_human))

    return CalibrationReport(
        judge_name=getattr(judge, "name", type(judge).__name__),
        n=len(labels),
        seed=seed,
        kappa_pairwise=kappa,
        agreement_pairwise=agreement,
        swap_rate=swap,
        position_preference=pos,
        kappa_support=kappa_support,
        agreement_support=agreement_support,
        confusion=confusion,
        threshold=threshold,
    )


def _coerce(v: Any) -> PairwiseVerdict:
    if v in VERDICTS:
        return v  # type: ignore[return-value]
    raise ValueError(f"judge returned {v!r}; expected one of {VERDICTS}")


# --------------------------------------------------------------------------- #
# Fixture construction (how the committed 100 labels were made)
# --------------------------------------------------------------------------- #
_FIXTURE_TOPICS: tuple[tuple[str, str, str], ...] = (
    (
        "refund window",
        "Refund requests are accepted within 30 days of the invoice date.",
        "30 days",
    ),
    (
        "key rotation",
        "Encryption keys are rotated every 90 days by the security platform team.",
        "every 90 days",
    ),
    (
        "support SLA",
        "Priority-1 support tickets carry a one-hour first-response SLA.",
        "one hour",
    ),
    (
        "fusion constant",
        "Reciprocal rank fusion uses the constant k = 60 across all four arms.",
        "k = 60",
    ),
    (
        "travel cap",
        "Domestic hotel spend is capped at 180 dollars per night.",
        "180 dollars per night",
    ),
    (
        "laptop provisioning",
        "New engineers receive a provisioned laptop on their first Monday.",
        "the first Monday",
    ),
    (
        "invoice cadence",
        "Invoices are issued on the first business day of each month.",
        "first business day of the month",
    ),
    (
        "index refresh",
        "The retrieval index is rebuilt nightly at 02:00 UTC.",
        "nightly at 02:00 UTC",
    ),
    (
        "seat pricing",
        "The team plan costs 12 dollars per seat per month.",
        "12 dollars per seat per month",
    ),
    (
        "incident paging",
        "Sev-1 incidents page the on-call engineer within five minutes.",
        "within five minutes",
    ),
)


def build_calibration_fixture(*, seed: int = 20240927, n: int = 100) -> list[JudgeLabel]:
    """Deterministically construct the calibration fixture.

    The committed ``judge_labels_v1.jsonl`` is the artifact; this function is
    the reproducible recipe for it.  Each item pairs a grounded answer against
    a distractor of a known failure mode (hallucinated number, off-topic,
    hedged non-answer, verbose-but-correct), so the human label follows from
    construction and the set is genuinely adjudicable by hand.
    """
    rng = np.random.default_rng(seed)
    kinds = ("hallucinated", "offtopic", "refusal", "verbose", "equivalent")
    items: list[JudgeLabel] = []
    for i in range(n):
        topic, context, fact = _FIXTURE_TOPICS[i % len(_FIXTURE_TOPICS)]
        kind = kinds[int(rng.integers(0, len(kinds)))]
        good = f"The {topic} is {fact}, as stated in the handbook."
        if kind == "hallucinated":
            bad, label, b_ok = f"The {topic} is 7 days according to the handbook.", "A", False
        elif kind == "offtopic":
            bad, label, b_ok = (
                f"The office cafeteria serves lunch until 14:00 and has nothing to do with {topic}.",
                "A",
                False,
            )
        elif kind == "refusal":
            bad, label, b_ok = (
                "I do not have enough information in the provided context to answer that.",
                "A",
                False,
            )
        elif kind == "verbose":
            bad, label, b_ok = (
                f"To answer the question about the {topic}: after reviewing the relevant "
                f"handbook section in detail, the value is {fact}. This has been the policy "
                "for some time and is reviewed annually by the responsible team.",
                "tie",
                True,
            )
        else:
            bad, label, b_ok = (f"It is {fact}.", "tie", True)

        # Alternate which slot holds the grounded answer so the fixture itself
        # is not position-degenerate.
        if i % 2 == 0:
            answer_a, answer_b = good, bad
            human = label
            a_ok, b_ok_final = True, b_ok
        else:
            answer_a, answer_b = bad, good
            human = flip_verdict(label)  # type: ignore[arg-type]
            a_ok, b_ok_final = b_ok, True

        items.append(
            JudgeLabel(
                id=f"cal-{i:03d}",
                question=f"What is the {topic}?",
                context=context,
                answer_a=answer_a,
                answer_b=answer_b,
                human_label=human,  # type: ignore[arg-type]
                a_supported=a_ok,
                b_supported=b_ok_final,
                annotator="hand",
                notes=f"distractor kind: {kind}",
            )
        )
    return items


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
_JUDGES = {
    "heuristic": lambda seed: HeuristicJudge(),
    "coin-flip": lambda seed: CoinFlipJudge(seed=seed),
    "position-biased": lambda seed: PositionBiasedJudge(),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m localmind.eval.judge_calibration",
        description="Score an LLM judge against human labels (Cohen's kappa, swap rate).",
    )
    parser.add_argument("--labels", default=None, help="JSONL of JudgeLabel records")
    parser.add_argument("--judge", default="heuristic", choices=sorted(_JUDGES))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threshold", type=float, default=KAPPA_TRUST_THRESHOLD)
    parser.add_argument("--out", default="artifacts/benchmarks/judge_calibration.json")
    parser.add_argument("--markdown", action="store_true", help="print the markdown block")
    args = parser.parse_args(argv)

    labels = load_judge_labels(args.labels)
    report = calibrate_judge(
        _JUDGES[args.judge](args.seed), labels, seed=args.seed, threshold=args.threshold
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(report.to_markdown() if args.markdown else report.summary_line())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
