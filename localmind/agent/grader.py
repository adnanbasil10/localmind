"""Grade node: per-chunk relevance from LocalMind-31M, plus injection quarantine.

Order matters. A chunk is screened for prompt injection *before* the relevance
grader sees it, and a flagged chunk is quarantined without ever being graded --
so hostile text never reaches a model prompt at all, not even the small one.

The same `ControlPlane.grade` head doubles as a claim-support checker in the
`verify` node: "is this claim entailed by this source?" is the same short
binary judgement as "is this chunk relevant to this query?", which is exactly
the 31M model's weight class.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from localmind.agent.guardrails import InjectionClassifier, build_injection_classifier
from localmind.agent.state import ClaimVerdict, ControlPlane, GradedChunk, RetrievedChunk

__all__ = ["ControlPlane", "GradeReport", "Grader"]

MAX_CHUNK_CHARS_FOR_GRADER = 4000

#: Fraction of query terms a chunk must contain for the model-free fallback to
#: call it relevant. One term in three -- deliberately generous, because the
#: fallback only runs when the 31M control plane is unavailable or broken.
LEXICAL_FALLBACK_THRESHOLD = 0.3


class GradeReport(BaseModel):
    """Result of grading one batch of chunks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graded: list[GradedChunk]
    latency_ms: float = 0.0
    n_relevant: int = 0
    n_quarantined: int = 0
    fallbacks: int = 0


class Grader:
    """Wraps `ControlPlane.grade`; never raises at the caller."""

    def __init__(
        self,
        control_plane: ControlPlane | None = None,
        classifier: InjectionClassifier | None = None,
        *,
        threshold: float = 0.5,
    ) -> None:
        self.control_plane = control_plane
        self.classifier = classifier or build_injection_classifier(control_plane)
        self.threshold = threshold

    # -- internals ---------------------------------------------------------------------

    def _grade_one(self, query: str, text: str) -> tuple[bool, float, bool]:
        """Returns (relevant, score, used_fallback)."""
        if self.control_plane is None:
            return (*self._lexical_fallback(query, text), True)
        try:
            raw = self.control_plane.grade(query, text[:MAX_CHUNK_CHARS_FOR_GRADER])
        except Exception:
            return (*self._lexical_fallback(query, text), True)
        try:
            relevant, score = raw
            return bool(relevant), float(score), False
        except (TypeError, ValueError):
            return (*self._lexical_fallback(query, text), True)

    @staticmethod
    def _lexical_fallback(query: str, text: str) -> tuple[bool, float]:
        """Jaccard overlap on lowercased word sets. Deterministic, no model needed."""
        q = {w for w in query.lower().split() if len(w) > 2}
        c = {w for w in text.lower().split() if len(w) > 2}
        if not q or not c:
            return False, 0.0
        score = len(q & c) / len(q)
        return score >= LEXICAL_FALLBACK_THRESHOLD, score

    # -- public ------------------------------------------------------------------------

    def grade_chunks(self, query: str, chunks: Sequence[RetrievedChunk]) -> GradeReport:
        t0 = time.perf_counter()
        out: list[GradedChunk] = []
        fallbacks = 0
        for chunk in chunks:
            verdict = self.classifier.classify(chunk.text)
            if verdict.flagged:
                out.append(
                    GradedChunk(
                        chunk=chunk,
                        relevant=False,
                        score=0.0,
                        injection_flagged=True,
                        injection_score=verdict.score,
                        quarantined=True,
                        quarantine_reason=(
                            f"prompt injection detected ({verdict.detector}; "
                            f"families={','.join(verdict.families) or 'unknown'})"
                        ),
                    )
                )
                continue
            relevant, score, used_fallback = self._grade_one(query, chunk.text)
            fallbacks += int(used_fallback)
            out.append(
                GradedChunk(
                    chunk=chunk,
                    relevant=relevant and score >= 0.0,
                    score=score,
                    injection_flagged=False,
                    injection_score=verdict.score,
                )
            )
        return GradeReport(
            graded=out,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            n_relevant=sum(1 for g in out if g.usable),
            n_quarantined=sum(1 for g in out if g.quarantined),
            fallbacks=fallbacks,
        )

    def support(self, claim: str, sources: Sequence[RetrievedChunk]) -> ClaimVerdict:
        """Is `claim` supported by any source? Used by the `verify` node."""
        best_score = 0.0
        best_id = ""
        supported = False
        for src in sources:
            relevant, score, _ = self._grade_one(claim, src.text)
            if score > best_score:
                best_score, best_id = score, src.chunk_id
            if relevant and score >= self.threshold:
                supported = True
                best_score, best_id = max(best_score, score), src.chunk_id
                break
        return ClaimVerdict(
            claim=claim, supported=supported, score=best_score, supporting_chunk_id=best_id
        )
