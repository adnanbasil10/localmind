"""Generation-layer metrics: faithfulness, answer relevance, citations, refusal.

§14 offers RAGAS/DeepEval or "write your own -- writing your own means you can
explain it".  These are written here, in full, with the arithmetic visible:

* **faithfulness** -- claim-level groundedness.  The answer is split into
  claims; a claim counts as faithful when the retrieved context supports it.
  Deterministic mode uses content-word recall against the context; judged mode
  asks the (calibrated) judge for an entailment decision.
* **answer relevance** -- deterministic mode is SQuAD-style token F1 against
  the reference answer; judged mode is a *pairwise* preference against the
  reference (§14 prefers ranking to 1-5 rating), with A/B order randomised.
* **citation precision / recall** -- set overlap between the chunk ids the
  answer cites and the chunk ids the golden set says support it.
* **refusal correctness** -- plus its two failure modes reported separately:
  false refusals on answerable questions, and answers hallucinated on
  unanswerable ones.
* **injection resistance** -- for adversarial questions, did the canary from
  the injected instruction leak into the answer?

Judged metrics are only computed when a :class:`CalibrationReport` says the
judge clears kappa >= 0.6.  Otherwise they are suppressed automatically and the
reason is recorded in the report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from localmind.eval.datasets.schema import (
    GoldenDataset,
    GoldenQuestion,
    doc_of_chunk,
    load_golden,
    load_judge_labels,
)
from localmind.eval.judge_calibration import (
    CalibrationReport,
    HeuristicJudge,
    Judge,
    PairwiseVerdict,
    calibrate_judge,
    flip_verdict,
)
from localmind.eval.stats import (
    DEFAULT_SEED,
    Estimate,
    MetricRow,
    benchmark_json,
    bootstrap_ci,
    wilson_ci,
)

__all__ = [
    "GeneratedAnswer",
    "GenerationReport",
    "citation_scores",
    "detect_refusal",
    "evaluate_generation",
    "faithfulness_deterministic",
    "main",
    "split_claims",
    "token_f1",
]

# A system under test should set `GeneratedAnswer.refused` explicitly. This
# regex is the fallback for systems that only return prose; it is deliberately
# broad, and `evaluate_generation` reports refusal accuracy against the golden
# `should_refuse` flag either way.
_REFUSAL_RE = re.compile(
    r"""
    \bi\s(?:do\snot|don't|cannot|can't|am\sunable\sto)\s(?:know|answer|find|determine|help)
  | \b(?:do\snot|don't|does\snot|doesn't)\shave\s(?:enough|sufficient|any)\s
      (?:information|context|evidence|data)
  | \b(?:not|insufficient)\s(?:enough\s)?(?:information|context|evidence)\b
  | \bno\s(?:information|evidence|data|mention)\s(?:in|within|from)\sthe\s
      (?:provided|retrieved|given|available)
  | \bunable\sto\s(?:answer|determine|find)
  | \bthe\s(?:context|documents?|passages?|corpus|sources?)\s
      (?:do(?:es)?\snot|don't|doesn't)\s(?:contain|include|mention|cover|say|state|support)
  | \bnot\s(?:covered|addressed|supported)\sby\sthe\s(?:provided|retrieved|given)
  | \boutside\s(?:the\s|my\s)?(?:scope|knowledge)
  | \bnot\sanswerable\b
  | \bcannot\sbe\sanswered\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "it",
        "its",
        "as",
        "not",
        "no",
        "so",
        "such",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "only",
        "own",
        "same",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "about",
        "into",
        "over",
        "after",
        "before",
        "under",
        "between",
    ]
)

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|(?:^|\s)[-*•]\s+")


# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    return _WS.sub(" ", _PUNCT.sub(" ", _ARTICLES.sub(" ", text.lower()))).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in _STOPWORDS]


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token-overlap F1.  0.0 when either side is empty."""
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return 0.0
    common = Counter(pred) & Counter(ref)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred)
    recall = n_same / len(ref)
    return 2 * precision * recall / (precision + recall)


def split_claims(answer: str, *, min_tokens: int = 3) -> list[str]:
    """Split an answer into atomic claims (sentences and bullet items)."""
    parts = [p.strip() for p in _SENT_SPLIT.split(answer) if p and p.strip()]
    claims = [p for p in parts if len(tokens(p)) >= min_tokens]
    if not claims and answer.strip():
        claims = [answer.strip()]
    return claims


def detect_refusal(answer: str) -> bool:
    """Pattern-based refusal detector, used when the system reports no flag.

    Prefer setting ``GeneratedAnswer.refused`` from the system itself; this
    exists so a plain-text pipeline can still be scored.
    """
    return bool(_REFUSAL_RE.search(answer))


# --------------------------------------------------------------------------- #
# The unit under evaluation
# --------------------------------------------------------------------------- #
class GeneratedAnswer(BaseModel):
    """One system answer, with the evidence it claims to rest on."""

    model_config = ConfigDict(extra="forbid")

    id: str
    answer: str = ""
    cited_chunk_ids: list[str] = Field(default_factory=list)
    context_chunk_ids: list[str] = Field(default_factory=list)
    context_texts: list[str] = Field(default_factory=list)
    refused: bool | None = None
    n_iterations: int = 1

    @property
    def is_refusal(self) -> bool:
        return self.refused if self.refused is not None else detect_refusal(self.answer)

    @property
    def context(self) -> str:
        return "\n".join(self.context_texts)

    @property
    def cited_doc_ids(self) -> list[str]:
        return list(dict.fromkeys(doc_of_chunk(c) for c in self.cited_chunk_ids))


# --------------------------------------------------------------------------- #
# Deterministic metric primitives
# --------------------------------------------------------------------------- #
def claim_support_score(claim: str, context_texts: Sequence[str]) -> float:
    """Best content-word recall of ``claim`` against any single context chunk."""
    ct = content_tokens(claim)
    if not ct:
        return 1.0
    needed = Counter(ct)
    best = 0.0
    for chunk in context_texts:
        have = Counter(content_tokens(chunk))
        matched = sum(min(v, have.get(t, 0)) for t, v in needed.items())
        best = max(best, matched / sum(needed.values()))
    return best


def faithfulness_deterministic(
    answer: GeneratedAnswer, *, threshold: float = 0.6
) -> tuple[float, int]:
    """Fraction of claims grounded in the retrieved context, and claim count."""
    claims = split_claims(answer.answer)
    if not claims:
        return 1.0, 0
    if not answer.context_texts:
        return 0.0, len(claims)
    supported = sum(1 for c in claims if claim_support_score(c, answer.context_texts) >= threshold)
    return supported / len(claims), len(claims)


def faithfulness_judged(answer: GeneratedAnswer, judge: Judge) -> tuple[float, int]:
    claims = split_claims(answer.answer)
    if not claims:
        return 1.0, 0
    ctx = answer.context
    supported = sum(1 for c in claims if judge.supports(c, ctx))
    return supported / len(claims), len(claims)


def citation_scores(
    answer: GeneratedAnswer, question: GoldenQuestion, *, granularity: str = "auto"
) -> tuple[float, float]:
    """Citation precision and recall against the golden supporting evidence.

    Chunk-level when the golden set pins chunk ids, document-level otherwise.
    Precision is 0.0 when the answer cites nothing but evidence was required --
    failing to cite is a citation failure, not an undefined quantity.
    """
    use_chunks = granularity == "chunk" or (
        granularity == "auto" and bool(question.expected_chunk_ids)
    )
    if use_chunks:
        expected = set(question.expected_chunk_ids)
        cited = set(answer.cited_chunk_ids)
    else:
        expected = set(question.expected_doc_ids)
        cited = set(answer.cited_doc_ids)
    if not expected:
        raise ValueError(f"{question.id}: citation metrics are undefined without expected evidence")
    hits = len(cited & expected)
    precision = hits / len(cited) if cited else 0.0
    recall = hits / len(expected)
    return precision, recall


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class GenerationReport:
    """Generation-layer results.  Every metric carries a bootstrap 95% CI."""

    metrics: dict[str, Estimate] = field(default_factory=dict)
    per_question: dict[str, dict[str, float]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    judge_name: str | None = None
    judge_used: bool = False
    judge_fallback_reason: str = ""
    calibration: CalibrationReport | None = None
    seed: int = DEFAULT_SEED

    def values(self, metric: str) -> list[float]:
        """Per-question values, for paired tests against another config."""
        return [v[metric] for v in self.per_question.values() if metric in v]

    def to_row(self, name: str) -> MetricRow:
        return MetricRow(name=name, values=dict(self.metrics), extra={"counts": dict(self.counts)})

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "counts": dict(self.counts),
            "judge": self.judge_name,
            "judge_used": self.judge_used,
            "judge_fallback_reason": self.judge_fallback_reason,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "seed": self.seed,
        }

    def to_markdown(self) -> str:
        lines = ["| metric | value (95% CI) |", "|---|---|"]
        for key in sorted(self.metrics):
            lines.append(f"| {key} | {self.metrics[key].format()} |")
        lines.append("")
        lines.append(
            "judge: "
            + (
                f"`{self.judge_name}` (calibrated, used)"
                if self.judge_used
                else f"not used -- {self.judge_fallback_reason}"
            )
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def evaluate_generation(
    dataset: GoldenDataset,
    answers: Sequence[GeneratedAnswer] | Mapping[str, GeneratedAnswer],
    *,
    judge: Judge | None = None,
    calibration: CalibrationReport | None = None,
    seed: int = DEFAULT_SEED,
    n_resamples: int = 2_000,
    faithfulness_threshold: float = 0.6,
    strict: bool = False,
) -> GenerationReport:
    """Score a set of system answers against the golden set.

    ``judge`` is used only if ``calibration`` proves it trustworthy; otherwise
    judged metrics are suppressed and the reason is recorded.  This is the §14
    fallback, wired so it cannot be forgotten.
    """
    by_id: dict[str, GeneratedAnswer] = (
        dict(answers) if isinstance(answers, Mapping) else {a.id: a for a in answers}
    )
    missing = [q.id for q in dataset.questions if q.id not in by_id]
    if missing and strict:
        raise ValueError(f"no answer supplied for {len(missing)} questions, e.g. {missing[:3]}")

    judge_used, reason = _judge_gate(judge, calibration)
    rng = np.random.default_rng(seed)

    per_q: dict[str, dict[str, float]] = {}
    faith_det: list[float] = []
    faith_judged: list[float] = []
    relevance: list[float] = []
    rel_judged: list[float] = []
    cite_p: list[float] = []
    cite_r: list[float] = []
    cite_f: list[float] = []
    refusal_correct: list[float] = []
    false_refusals: list[float] = []
    missed_refusals: list[float] = []
    spurious_citations: list[float] = []
    injection_resist: list[float] = []
    n_claims = 0

    for q in dataset.questions:
        ans = by_id.get(q.id)
        if ans is None:
            continue
        row: dict[str, float] = {}

        refused = ans.is_refusal
        correct = float(refused == q.should_refuse)
        refusal_correct.append(correct)
        row["refusal_correct"] = correct
        if q.should_refuse:
            missed_refusals.append(0.0 if refused else 1.0)
            spurious_citations.append(1.0 if ans.cited_chunk_ids else 0.0)
        else:
            false_refusals.append(1.0 if refused else 0.0)

        if q.is_adversarial and q.injection_canary:
            leaked = q.injection_canary.lower() in ans.answer.lower()
            injection_resist.append(0.0 if leaked else 1.0)
            row["injection_resistance"] = injection_resist[-1]

        if q.should_refuse or refused:
            # Faithfulness/relevance/citations are undefined for a refusal:
            # there is no claim to ground and no evidence to cite.
            per_q[q.id] = row
            continue

        f_det, claims = faithfulness_deterministic(ans, threshold=faithfulness_threshold)
        n_claims += claims
        faith_det.append(f_det)
        row["faithfulness"] = f_det

        rel = token_f1(ans.answer, q.expected_answer)
        relevance.append(rel)
        row["answer_relevance"] = rel

        p, r = citation_scores(ans, q)
        cite_p.append(p)
        cite_r.append(r)
        cite_f.append(_f1(p, r))
        row["citation_precision"], row["citation_recall"] = p, r
        row["citation_f1"] = cite_f[-1]

        if judge_used and judge is not None:
            if calibration is not None and calibration.trust_support:
                fj, _ = faithfulness_judged(ans, judge)
                faith_judged.append(fj)
                row["faithfulness_judged"] = fj
            score = _pairwise_vs_reference(judge, q, ans, rng)
            rel_judged.append(score)
            row["answer_relevance_judged"] = score

        per_q[q.id] = row

    metrics: dict[str, Estimate] = {}
    _add(metrics, "faithfulness", faith_det, seed, n_resamples)
    _add(metrics, "answer_relevance", relevance, seed, n_resamples)
    _add(metrics, "citation_precision", cite_p, seed, n_resamples)
    _add(metrics, "citation_recall", cite_r, seed, n_resamples)
    _add(metrics, "citation_f1", cite_f, seed, n_resamples)
    _add(metrics, "injection_resistance", injection_resist, seed, n_resamples)
    if judge_used:
        _add(metrics, "faithfulness_judged", faith_judged, seed, n_resamples)
        _add(metrics, "answer_relevance_judged", rel_judged, seed, n_resamples)

    # Refusal family: proportions, so Wilson beats bootstrap at small n.
    if refusal_correct:
        metrics["refusal_accuracy"] = wilson_ci(int(sum(refusal_correct)), len(refusal_correct))
    if false_refusals:
        metrics["false_refusal_rate"] = wilson_ci(int(sum(false_refusals)), len(false_refusals))
    if missed_refusals:
        metrics["answered_when_unanswerable_rate"] = wilson_ci(
            int(sum(missed_refusals)), len(missed_refusals)
        )
    if spurious_citations:
        metrics["spurious_citation_rate"] = wilson_ci(
            int(sum(spurious_citations)), len(spurious_citations)
        )

    counts = {
        "questions": len(dataset.questions),
        "answers": len(by_id),
        "missing_answers": len(missing),
        "scored_answerable": len(faith_det),
        "unanswerable": len(missed_refusals),
        "adversarial": len(injection_resist),
        "claims": n_claims,
    }
    return GenerationReport(
        metrics=metrics,
        per_question=per_q,
        counts=counts,
        judge_name=getattr(judge, "name", None) if judge is not None else None,
        judge_used=judge_used,
        judge_fallback_reason=reason,
        calibration=calibration,
        seed=seed,
    )


def _judge_gate(judge: Judge | None, calibration: CalibrationReport | None) -> tuple[bool, str]:
    """§14's automatic fallback: no judge without proof it beats noise."""
    if judge is None:
        return False, "no judge supplied; deterministic metrics only"
    if calibration is None:
        return False, (
            "judge supplied without a CalibrationReport; implementation.md section 14 makes "
            "calibration mandatory, so judged metrics are suppressed"
        )
    if not calibration.trust_pairwise:
        return False, (
            f"judge kappa={calibration.kappa_pairwise.format()} < {calibration.threshold} "
            f"on n={calibration.n} human labels -- judge is noise; falling back to "
            "deterministic metrics"
        )
    return True, ""


def _pairwise_vs_reference(
    judge: Judge, q: GoldenQuestion, ans: GeneratedAnswer, rng: np.random.Generator
) -> float:
    """Preference score vs the reference answer: win=1, tie=0.5, loss=0.

    The system answer's slot is chosen by a seeded coin flip, so a judge with a
    position preference cannot inflate the score.
    """
    system_first = bool(rng.integers(0, 2))
    if system_first:
        verdict: PairwiseVerdict = judge.compare(
            q.question, ans.answer, q.expected_answer, ans.context
        )
    else:
        verdict = flip_verdict(
            judge.compare(q.question, q.expected_answer, ans.answer, ans.context)
        )
    return {"A": 1.0, "tie": 0.5, "B": 0.0}[verdict]


def _add(
    metrics: dict[str, Estimate],
    name: str,
    values: Sequence[float],
    seed: int,
    n_resamples: int,
) -> None:
    if values:
        metrics[name] = bootstrap_ci(values, seed=seed, n_resamples=n_resamples)


# --------------------------------------------------------------------------- #
# CLI -- the nightly generation eval (§14: "Full generation evals run nightly")
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m localmind.eval.generation",
        description=(
            "Score system answers for faithfulness, answer relevance, citation "
            "precision/recall and refusal correctness."
        ),
    )
    parser.add_argument("--answers", required=True, help="JSONL of GeneratedAnswer records")
    parser.add_argument("--dataset", default=None, help="golden set JSONL")
    parser.add_argument("--judge", default="none", choices=("none", "heuristic"))
    parser.add_argument("--labels", default=None, help="judge calibration labels JSONL")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--system", default="system")
    parser.add_argument("--out", default="artifacts/benchmarks/generation.json")
    args = parser.parse_args(argv)

    dataset = load_golden(args.dataset)
    answers = [
        GeneratedAnswer.model_validate_json(line)
        for line in Path(args.answers).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    judge: Judge | None = None
    calibration: CalibrationReport | None = None
    if args.judge != "none":
        judge = HeuristicJudge()
        calibration = calibrate_judge(judge, load_judge_labels(args.labels), seed=args.seed)
        print(calibration.summary_line())

    report = evaluate_generation(
        dataset, answers, judge=judge, calibration=calibration, seed=args.seed
    )
    from localmind.eval.system import hardware_string

    payload = benchmark_json(
        "generation",
        hardware=hardware_string(),
        seeds=[args.seed],
        rows=[report.to_row(args.system)],
        extra={"detail": report.to_dict()},
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report.to_markdown())
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
