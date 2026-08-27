"""Rewrite node: query rewriting by LocalMind-31M, with a progress guarantee.

Termination detail: a rewriter that returns the same string forever would spin
the `(rewrite -> retrieve)*` loop until the iteration cap. We do not rely on the
cap alone -- `Rewriter.rewrite` reports `novel=False` when the candidate
normalises to a query we have already tried, and `graph` treats a non-novel
rewrite as immediate loop exit. So the loop terminates on *either* the cap or
the absence of progress, whichever comes first.
"""

from __future__ import annotations

import re
import time

from pydantic import BaseModel, ConfigDict

from localmind.agent.state import ControlPlane

__all__ = ["ControlPlane", "RewriteResult", "Rewriter", "heuristic_rewrite", "normalize_query"]

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "on",
        "is",
        "are",
        "was",
        "were",
        "what",
        "which",
        "who",
        "how",
        "does",
        "do",
        "did",
        "please",
        "tell",
        "me",
        "about",
        "can",
        "you",
        "explain",
        "and",
        "it",
        "that",
        "this",
    }
)

MAX_REWRITE_CHARS = 300


def normalize_query(query: str) -> str:
    """Canonical form used for the loop-progress check."""
    return re.sub(r"\s+", " ", query.strip().lower())


def heuristic_rewrite(query: str, history: list[str], attempt: int = 0) -> str:
    """Deterministic fallback rewrite: keyword distillation, then broadening.

    Attempt 0 strips stopwords and question framing (a tighter keyword query);
    attempt 1+ drops the rarest trailing terms (a broader query). Both are pure
    string transforms -- no model, no network.
    """
    words = re.findall(r"[\w'-]+", query.lower())
    kept = [w for w in words if w not in _STOPWORDS]
    if not kept:
        kept = words
    if attempt >= 1 and len(kept) > 2:
        kept = kept[: max(2, len(kept) - 1)]
    context = ""
    if history and attempt >= 1:
        prior = re.findall(r"[\w'-]+", history[-1].lower())
        extra = [w for w in prior if w not in _STOPWORDS and w not in kept][:2]
        context = " " + " ".join(extra) if extra else ""
    return (" ".join(kept) + context).strip()[:MAX_REWRITE_CHARS]


class RewriteResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    novel: bool
    latency_ms: float = 0.0
    source: str = "control_plane"
    detail: str = ""


class Rewriter:
    """Wraps `ControlPlane.rewrite` with validation, fallback and a novelty check."""

    def __init__(self, control_plane: ControlPlane | None = None) -> None:
        self.control_plane = control_plane

    def rewrite(self, query: str, history: list[str], *, attempt: int = 0) -> RewriteResult:
        seen = {normalize_query(h) for h in history} | {normalize_query(query)}
        source = "control_plane"
        detail = ""
        latency_ms = 0.0
        candidate = ""

        if self.control_plane is not None:
            t0 = time.perf_counter()
            try:
                raw = self.control_plane.rewrite(query, list(history))
                latency_ms = (time.perf_counter() - t0) * 1000.0
                if isinstance(raw, str) and raw.strip():
                    candidate = raw.strip()[:MAX_REWRITE_CHARS]
                else:
                    detail = f"control plane returned {type(raw).__name__}"
            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                detail = f"control plane raised {type(exc).__name__}"
        else:
            detail = "no control plane"

        if not candidate:
            candidate = heuristic_rewrite(query, history, attempt)
            source = "heuristic"

        if normalize_query(candidate) in seen:
            fallback = heuristic_rewrite(query, history, attempt + 1)
            if normalize_query(fallback) not in seen:
                return RewriteResult(
                    query=fallback,
                    novel=True,
                    latency_ms=latency_ms,
                    source="heuristic",
                    detail=(detail + "; control-plane rewrite repeated a prior query").strip("; "),
                )
            return RewriteResult(
                query=candidate,
                novel=False,
                latency_ms=latency_ms,
                source=source,
                detail=(detail + "; no novel rewrite available").strip("; "),
            )
        return RewriteResult(
            query=candidate, novel=True, latency_ms=latency_ms, source=source, detail=detail
        )
