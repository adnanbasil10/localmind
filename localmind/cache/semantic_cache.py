"""Redis-backed caching, Layer 3: the semantic cache — and the tau sweep that keeps it honest.

Key: nearest neighbour over query embeddings, threshold ``tau``. Metric to report: hit rate
**and false-hit rate** — cases where a semantically-close query got a wrong cached answer.

implementation.md SS15 is blunt about this layer: "Semantic caching is where portfolio projects
cheat." A cache that only reports hit rate is trivially maximized by setting tau to 0 and never
computing anything twice; the number that matters is what fraction of those hits were *wrong* for
the new query. This module measures that honestly:

  - :class:`SemanticCache` is the online cache: ``store``/``lookup`` over an injected
    :class:`Embedder`, backed by any :class:`CacheBackend`. Hit rate is trackable live
    (hits / lookups). False-hit rate is NOT trackable live by the cache alone — the cache has no
    way to know, at lookup time, whether the answer it just served is actually correct for the
    new query. :meth:`SemanticCache.record_verdict` lets an online canary / human-eval process
    feed that back in production.
  - :func:`sweep_tau` produces the full curve offline against a labelled synthetic query set
    (:func:`build_synthetic_queryset`): for every candidate tau in ``[0.80, 0.99]``, hit rate vs.
    false-hit rate, where a false hit is *defined* as "the returned answer does not match the
    new query's own ground-truth answer" — not merely "similarity below some other threshold".
  - :func:`run_and_write_benchmark` runs the sweep and writes
    ``artifacts/benchmarks/semantic_cache.json`` in the CONVENTIONS.md benchmark-envelope schema,
    reusing ``localmind.eval.stats`` (the same bootstrap-CI machinery every other phase's
    benchmark uses) so this reports the required >=3 seeds and 95% CI rather than a bare number.

No embedding model is available in this environment (no network, no GPU): the embedder is an
injected :class:`Embedder` Protocol, and :class:`HashingEmbedder` is the deterministic fake used
both in tests and in the sweep itself — a seeded bag-of-hashed-tokens vector, L2-normalized. It is
deliberately crude (no real semantics) but that crudeness is what makes it a *fair* stand-in for
demonstrating the false-hit phenomenon: text that shares vocabulary gets high cosine similarity
even when meaning differs, which is exactly the real-world failure mode ("cancel my subscription"
vs. "cancel my order") that makes semantic caching risky. See ``build_synthetic_queryset`` for the
labelled scenarios this exercises.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from localmind.cache.embedding_cache import CacheBackend, InMemoryCacheBackend

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "NearestMatch",
    "SemanticCache",
    "SemanticCacheResult",
    "SemanticCacheStats",
    "SweepResult",
    "SyntheticQAItem",
    "build_synthetic_queryset",
    "choose_operating_point",
    "run_and_write_benchmark",
    "sweep_tau",
]

_DEFAULT_TAUS: tuple[float, ...] = tuple(round(0.80 + 0.01 * i, 2) for i in range(20))  # 0.80..0.99
_DEFAULT_SWEEP_SEEDS: tuple[int, ...] = (1, 2, 3)


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into a vector. Real embedders (e.g. the retrieval dense encoder)
    and :class:`HashingEmbedder` (the deterministic offline fake) both satisfy this."""

    def embed(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic, dependency-free stand-in for a real embedding model.

    Tokenizes on word boundaries, hashes each token (and each adjacent bigram, to give short
    shared phrases a little extra weight) into one of ``dim`` buckets with a seeded sign, and
    L2-normalizes. Same text always gives the same vector for a given ``seed``; different ``seed``
    gives a different (but internally consistent) embedding space, which is what lets
    :func:`sweep_tau` treat multiple seeds as independent realizations of the same experiment for
    a bootstrap CI, the way a stochastic real embedding model's checkpoints would.

    This is a bag-of-words model, so it inherits a bag-of-words model's known failure mode:
    queries that share vocabulary get high cosine similarity even when their *meaning* (and
    correct answer) differs. That is a feature here, not a bug — it is the same failure mode a
    real dense encoder exhibits on confusable near-duplicates, so exercising it is a fair test of
    the false-hit path rather than a synthetic curve that never shows a downside.
    """

    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed

    def _bucket_and_sign(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(f"{self.seed}:{token}".encode(), digest_size=8).digest()
        code = int.from_bytes(digest, "big")
        bucket = code % self.dim
        sign = 1.0 if (code >> 1) % 2 == 0 else -1.0
        return bucket, sign

    def embed(self, text: str) -> np.ndarray:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vector = np.zeros(self.dim, dtype=np.float64)
        for token in tokens:
            bucket, sign = self._bucket_and_sign(token)
            vector[bucket] += sign
        for left, right in pairwise(tokens):
            bucket, sign = self._bucket_and_sign(f"{left}_{right}")
            vector[bucket] += 0.5 * sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


@dataclass
class _SemanticEntry:
    key: str
    query: str
    embedding: np.ndarray
    answer: str


class NearestMatch(BaseModel):
    matched_query: str
    answer: str
    similarity: float


class SemanticCacheResult(BaseModel):
    hit: bool
    answer: str | None
    similarity: float
    matched_query: str | None = None


class SemanticCacheStats(BaseModel):
    lookups: int
    hits: int
    hit_rate: float
    verdicts_recorded: int
    false_hits_recorded: int
    false_hit_rate: float | None
    """``None`` until at least one verdict has been recorded via :meth:`SemanticCache.record_verdict`
    — false-hit rate cannot be known from lookups alone (see module docstring)."""


class SemanticCache:
    """Layer 3. Nearest-neighbour nearest-embedding lookup, threshold ``tau``.

    The NN index is a brute-force in-process scan over stored embeddings (fine at the scale a
    query cache operates at; a real deployment with a large cache would put this behind
    pgvector/qdrant the way the dense retrieval arm does). Answer payloads still round-trip
    through the injected :class:`CacheBackend`, so the backend is genuinely exercised.
    """

    def __init__(
        self,
        backend: CacheBackend,
        embedder: Embedder,
        tau: float = 0.92,
        model: str = "default",
    ) -> None:
        if not (0.0 < tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {tau}")
        self.backend = backend
        self.embedder = embedder
        self.tau = tau
        self.model = model
        self._entries: list[_SemanticEntry] = []
        self._lookups = 0
        self._hits = 0
        self._verdicts_recorded = 0
        self._false_hits_recorded = 0

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _key_for(self, query: str) -> str:
        digest = hashlib.sha256(f"{self.model}\n{query}".encode()).hexdigest()
        return f"sem:{self.model}:{digest}"

    def store(self, query: str, answer: str) -> str:
        embedding = self._normalize(self.embedder.embed(query))
        key = self._key_for(query)
        self.backend.set(key, answer.encode())
        self._entries.append(
            _SemanticEntry(key=key, query=query, embedding=embedding, answer=answer)
        )
        return key

    def nearest(self, query: str) -> NearestMatch | None:
        """Best match by cosine similarity, ignoring ``tau``. Used by :meth:`lookup` and by the
        offline tau sweep, which needs the raw similarity to threshold against many taus without
        re-embedding for each one."""
        if not self._entries:
            return None
        query_embedding = self._normalize(self.embedder.embed(query))
        matrix = np.stack([entry.embedding for entry in self._entries])
        similarities = matrix @ query_embedding
        idx = int(np.argmax(similarities))
        entry = self._entries[idx]
        return NearestMatch(
            matched_query=entry.query, answer=entry.answer, similarity=float(similarities[idx])
        )

    def lookup(self, query: str) -> SemanticCacheResult:
        self._lookups += 1
        best = self.nearest(query)
        if best is None or best.similarity < self.tau:
            return SemanticCacheResult(
                hit=False,
                answer=None,
                similarity=best.similarity if best else 0.0,
                matched_query=best.matched_query if best else None,
            )
        self._hits += 1
        raw = self.backend.get(self._key_for(best.matched_query))
        answer = raw.decode() if raw is not None else best.answer
        return SemanticCacheResult(
            hit=True, answer=answer, similarity=best.similarity, matched_query=best.matched_query
        )

    def record_verdict(self, was_correct: bool) -> None:
        """Feed back an out-of-band correctness verdict for a served cache hit (e.g. from an
        online canary that reruns the real pipeline for a sampled fraction of hits and compares
        answers). This is how :attr:`stats` can report a live false-hit rate; see module
        docstring for why the cache cannot compute this on its own."""
        self._verdicts_recorded += 1
        if not was_correct:
            self._false_hits_recorded += 1

    @property
    def stats(self) -> SemanticCacheStats:
        hit_rate = (self._hits / self._lookups) if self._lookups else 0.0
        false_hit_rate = (
            (self._false_hits_recorded / self._verdicts_recorded)
            if self._verdicts_recorded
            else None
        )
        return SemanticCacheStats(
            lookups=self._lookups,
            hits=self._hits,
            hit_rate=hit_rate,
            verdicts_recorded=self._verdicts_recorded,
            false_hits_recorded=self._false_hits_recorded,
            false_hit_rate=false_hit_rate,
        )


# --------------------------------------------------------------------------- #
# Synthetic labelled query set for the offline tau sweep
# --------------------------------------------------------------------------- #
class SyntheticQAItem(BaseModel):
    query: str
    intent_id: str
    answer: str


def build_synthetic_queryset() -> tuple[list[SyntheticQAItem], list[SyntheticQAItem]]:
    """A small, hand-authored, offline query set for measuring the semantic cache's tau tradeoff.

    Returns ``(population, evaluation)``:

    - ``population`` — one canonical (query, answer) per intent, the entries a real deployment
      would have accumulated in the cache, plus one deliberately ambiguous extra entry (see
      scenario 3 below). Stored into the cache before evaluation.
    - ``evaluation`` — queries fired against that populated cache, each labelled with its own
      ground-truth answer, split across three deliberate scenarios. Similarities quoted below are
      measured with :class:`HashingEmbedder`, the deterministic offline fake used everywhere in
      this module — see its docstring for why it only picks up lexical (not true semantic)
      overlap, which is why the paraphrases below are worded to share vocabulary with their
      canonical rather than being maximally-reworded natural paraphrases.

      1. **Paraphrases** of a population intent, same ground truth (``refund_period_*``,
         ``reset_password``, ``change_email``, ``change_phone``, ``track_order``,
         ``cancel_order``, ``return_item``, ``refund_policy_*``: similarity ~0.80-1.0 to their own
         canonical) — these *should* hit, and hit correctly, once tau is loose enough.
      2. **Cross-intent confusables.** Two families of these, calibrated deliberately differently
         to show that not all confusables are equally dangerous:
         - ``refund_policy_{basic,pro,enterprise}`` (short phrasing) sit at ~0.68 cross-intent
           similarity to each other — *below* the entire 0.80-0.99 sweep range, so they never
           false-hit no matter how loose tau gets within the swept window.
         - ``refund_period_{basic,pro,enterprise}`` (long shared prefix, tier name buried in the
           middle) sit at ~0.86-0.87 cross-intent similarity — *inside* the sweep range, so they
           genuinely false-hit once tau drops below ~0.87. This is the pair that gives the curve
           real teeth: the hazard is not "any two different questions", it's "two questions that
           are almost entirely the same words".
      3. **Context-ambiguous duplicate.** One extra population entry stores the exact text
         "What's the refund window?" under the Basic-plan answer (imagine that's whichever
         customer happened to populate the cache first). Three evaluation queries then fire that
         *identical* text, labelled with the Basic/Pro/Enterprise ground truth respectively (three
         different real customers asking the identically-phrased question). Cosine similarity of
         an exact string match is 1.0 regardless of tau or embedder quality, so this entry is
         always a hit; two of the three labels are necessarily wrong for whichever answer is
         cached. This is a genuine, tau-*independent* failure mode of any semantic cache keyed on
         text alone (no session/account context in the key) — included deliberately so the sweep
         cannot report a false-hit rate of zero just by tightening tau; see the operating-point
         discussion in :func:`run_and_write_benchmark`.

    Also includes a handful of clearly unrelated queries as true-negative distractors.
    """
    intents: list[tuple[str, str, str, list[str]]] = [
        # -- "safe" cross-intent family: confusable in principle, but similarity to siblings
        #    (~0.68) never crosses even the loosest swept tau (0.80). --
        (
            "refund_policy_basic",
            "What is the refund policy for the Basic plan?",
            "Basic plan purchases are refundable within 7 days of purchase.",
            [
                "What's the refund policy for the Basic plan?",
                "Could you tell me the refund policy for the Basic plan?",
                "Refund policy for the Basic plan, please?",
            ],
        ),
        (
            "refund_policy_pro",
            "What is the refund policy for the Pro plan?",
            "Pro plan purchases are refundable within 14 days of purchase.",
            [
                "What's the refund policy for the Pro plan?",
                "Could you tell me the refund policy for the Pro plan?",
                "Refund policy for the Pro plan, please?",
            ],
        ),
        (
            "refund_policy_enterprise",
            "What is the refund policy for the Enterprise plan?",
            "Enterprise plan purchases are refundable within 30 days, subject to contract terms.",
            [
                "What's the refund policy for the Enterprise plan?",
                "Could you tell me the refund policy for the Enterprise plan?",
                "Refund policy for the Enterprise plan, please?",
            ],
        ),
        # -- "risky" cross-intent family: same three plans, but the shared prefix is long enough
        #    that sibling similarity (~0.86-0.87) lands inside the swept tau range. --
        (
            "refund_period_basic",
            "What is the maximum allowed refund period after purchase for a customer on the "
            "Basic plan?",
            "Basic customers may request a refund within 7 days of purchase.",
            [
                "What's the maximum allowed refund period after purchase for a customer on the "
                "Basic plan?",
                "Could you tell me the maximum allowed refund period for a customer on the "
                "Basic plan?",
            ],
        ),
        (
            "refund_period_pro",
            "What is the maximum allowed refund period after purchase for a customer on the "
            "Pro plan?",
            "Pro customers may request a refund within 14 days of purchase.",
            [
                "What's the maximum allowed refund period after purchase for a customer on the "
                "Pro plan?",
                "Could you tell me the maximum allowed refund period for a customer on the "
                "Pro plan?",
            ],
        ),
        (
            "refund_period_enterprise",
            "What is the maximum allowed refund period after purchase for a customer on the "
            "Enterprise plan?",
            "Enterprise customers may request a refund within 30 days, subject to contract terms.",
            [
                "What's the maximum allowed refund period after purchase for a customer on the "
                "Enterprise plan?",
                "Could you tell me the maximum allowed refund period for a customer on the "
                "Enterprise plan?",
            ],
        ),
        (
            "reset_password",
            "How do I reset my account password?",
            "Go to Settings > Security > Reset Password and follow the emailed link.",
            [
                "How can I reset my account password?",
                "What are the steps to reset my account password?",
            ],
        ),
        (
            "change_email",
            "How do I change my account email address?",
            "Go to Settings > Profile > Email and verify the new address via the confirmation link.",
            [
                "How can I change my account email address?",
                "What are the steps to change my account email address?",
            ],
        ),
        (
            "change_phone",
            "How do I change my account phone number?",
            "Go to Settings > Profile > Phone and confirm the new number via SMS code.",
            [
                "How can I change my account phone number?",
                "What are the steps to change my account phone number?",
            ],
        ),
        (
            "track_order",
            "How do I track my recent order?",
            "Open Orders > select the order > View Tracking to see the live carrier status.",
            [
                "How can I track my recent order?",
                "What are the steps to track my recent order?",
            ],
        ),
        (
            "cancel_order",
            "How do I cancel my recent order?",
            "Open Orders > select the order > Cancel Order, available until it ships.",
            [
                "How can I cancel my recent order?",
                "What are the steps to cancel my recent order?",
            ],
        ),
        (
            "return_item",
            "How do I return an item from my order?",
            "Open Orders > select the item > Start a Return, print the prepaid label, and drop it off.",
            [
                "How can I return an item from my order?",
                "What are the steps to return an item from my order?",
            ],
        ),
        (
            "unrelated_weather",
            "What is the weather like today?",
            "I don't have real-time weather data; please check a weather service.",
            ["Will it rain today?", "What's the forecast for tomorrow?"],
        ),
        (
            "unrelated_capital",
            "What is the capital of France?",
            "The capital of France is Paris.",
            ["Which city is France's capital?", "Name the capital city of France."],
        ),
        (
            "unrelated_recipe",
            "How do I make pancakes at home?",
            "Mix flour, milk, eggs, and baking powder, then cook spoonfuls on a hot greased pan.",
            ["What's a simple pancake recipe?", "How do I cook pancakes at home?"],
        ),
    ]

    population = [
        SyntheticQAItem(query=canonical, intent_id=intent_id, answer=answer)
        for intent_id, canonical, answer, _ in intents
    ]

    evaluation: list[SyntheticQAItem] = []
    for intent_id, _canonical, answer, paraphrases in intents:
        for paraphrase in paraphrases:
            evaluation.append(SyntheticQAItem(query=paraphrase, intent_id=intent_id, answer=answer))

    # Scenario 3: context-ambiguous duplicate. The identical surface text is stored once (as if
    # the Basic customer happened to populate the cache first) and then fired by three different
    # hidden contexts at evaluation time. Similarity to itself is 1.0 regardless of tau, so this
    # is a tau-independent contribution to the false-hit rate by construction.
    ambiguous_text = "What's the refund window?"
    refund_answers = {
        "refund_policy_basic": "Basic plan purchases are refundable within 7 days of purchase.",
        "refund_policy_pro": "Pro plan purchases are refundable within 14 days of purchase.",
        "refund_policy_enterprise": (
            "Enterprise plan purchases are refundable within 30 days, subject to contract terms."
        ),
    }
    population.append(
        SyntheticQAItem(
            query=ambiguous_text,
            intent_id="refund_policy_basic",
            answer=refund_answers["refund_policy_basic"],
        )
    )
    for intent_id, answer in refund_answers.items():
        evaluation.append(SyntheticQAItem(query=ambiguous_text, intent_id=intent_id, answer=answer))

    # Scenario 2b: tier-unspecified variants of the "risky" family -- phrasings that never name
    # which plan the customer is on, at a deliberate spread of similarities to whichever risky_*
    # canonical the hashing embedder happens to rank nearest (empirically ``refund_period_basic``
    # for seed=1; verified to stay consistent across the sweep's other seeds too). Unlike the
    # exact-duplicate scenario above these are NOT identical text to any stored entry, so whether
    # they hit is genuinely tau-dependent -- this is what gives the sweep a graduated false-hit
    # curve instead of a single all-or-nothing step.
    risky_ambiguous_texts = [
        "What is the maximum allowed refund period for a customer after purchase?",
        "Tell me the maximum allowed refund period after purchase for a customer on a plan.",
        "What is the maximum allowed refund period after purchase for a customer?",
        "For a customer on a plan, what is the maximum allowed refund period after purchase?",
    ]
    risky_answers = {
        "refund_period_basic": "Basic customers may request a refund within 7 days of purchase.",
        "refund_period_pro": "Pro customers may request a refund within 14 days of purchase.",
        "refund_period_enterprise": (
            "Enterprise customers may request a refund within 30 days, subject to contract terms."
        ),
    }
    for text in risky_ambiguous_texts:
        for intent_id, answer in risky_answers.items():
            evaluation.append(SyntheticQAItem(query=text, intent_id=intent_id, answer=answer))

    return population, evaluation


# --------------------------------------------------------------------------- #
# Tau sweep
# --------------------------------------------------------------------------- #
@dataclass
class _SweepPoint:
    tau: float
    hit_rate: float
    false_hit_rate_of_hits: float
    """Fraction of served cache hits whose answer was wrong for the querying user — the metric
    that answers "if the cache did answer, how often can I trust it?"."""
    false_hit_rate_of_total: float
    """Fraction of *all* evaluation queries that resulted in a wrong cached answer — the metric
    that answers "out of everything the cache saw, how often did it actively mislead?"."""
    hits: int
    false_hits: int
    n: int


@dataclass
class SweepResult:
    taus: list[float]
    seeds: list[int]
    points_by_seed: dict[int, list[_SweepPoint]] = field(default_factory=dict)


def _resolve_nearest(
    population: Sequence[SyntheticQAItem], evaluation: Sequence[SyntheticQAItem], embedder: Embedder
) -> list[tuple[SyntheticQAItem, NearestMatch | None]]:
    cache = SemanticCache(backend=InMemoryCacheBackend(), embedder=embedder, tau=1.0, model="sweep")
    for item in population:
        cache.store(item.query, item.answer)
    return [(item, cache.nearest(item.query)) for item in evaluation]


def _sweep_one_realization(
    taus: Sequence[float],
    population: Sequence[SyntheticQAItem],
    evaluation: Sequence[SyntheticQAItem],
    embedder: Embedder,
) -> list[_SweepPoint]:
    resolved = _resolve_nearest(population, evaluation, embedder)
    n = len(evaluation)
    points: list[_SweepPoint] = []
    for tau in taus:
        hits = 0
        false_hits = 0
        for item, match in resolved:
            if match is not None and match.similarity >= tau:
                hits += 1
                if match.answer != item.answer:
                    false_hits += 1
        points.append(
            _SweepPoint(
                tau=tau,
                hit_rate=(hits / n) if n else 0.0,
                false_hit_rate_of_hits=(false_hits / hits) if hits else 0.0,
                false_hit_rate_of_total=(false_hits / n) if n else 0.0,
                hits=hits,
                false_hits=false_hits,
                n=n,
            )
        )
    return points


def sweep_tau(
    taus: Sequence[float] = _DEFAULT_TAUS,
    *,
    seeds: Sequence[int] = _DEFAULT_SWEEP_SEEDS,
    embedder_factory: Callable[[int], Embedder] | None = None,
) -> SweepResult:
    """Sweep ``tau`` (default 0.80..0.99 in steps of 0.01) against the synthetic query set,
    across >=3 seeds (CONVENTIONS.md rule 5) so the result can carry a bootstrap CI rather than a
    bare number. Each seed drives an independent :class:`HashingEmbedder` "realization" of the
    experiment (a different fixed hash projection), analogous to different seeds/checkpoints of a
    real stochastic embedding model.
    """
    from localmind.eval.stats import MIN_SEEDS

    if len(seeds) < MIN_SEEDS:
        raise ValueError(
            f"sweep_tau requires >= {MIN_SEEDS} seeds per CONVENTIONS.md rule 5, got {len(seeds)}"
        )

    population, evaluation = build_synthetic_queryset()
    factory = embedder_factory or (lambda seed: HashingEmbedder(dim=256, seed=seed))

    points_by_seed = {
        seed: _sweep_one_realization(taus, population, evaluation, factory(seed)) for seed in seeds
    }
    return SweepResult(taus=list(taus), seeds=list(seeds), points_by_seed=points_by_seed)


def _rows_from_sweep(result: SweepResult) -> list[Any]:
    from localmind.eval.stats import DEFAULT_SEED, MetricRow, bootstrap_ci

    rows = []
    for i, tau in enumerate(result.taus):
        hit_rates = [result.points_by_seed[s][i].hit_rate for s in result.seeds]
        fh_of_hits = [result.points_by_seed[s][i].false_hit_rate_of_hits for s in result.seeds]
        fh_of_total = [result.points_by_seed[s][i].false_hit_rate_of_total for s in result.seeds]
        rows.append(
            MetricRow(
                name=f"tau={tau:.2f}",
                values={
                    "hit_rate": bootstrap_ci(hit_rates, seed=DEFAULT_SEED),
                    "false_hit_rate_of_hits": bootstrap_ci(fh_of_hits, seed=DEFAULT_SEED),
                    "false_hit_rate_of_total": bootstrap_ci(fh_of_total, seed=DEFAULT_SEED),
                },
                extra={"tau": tau},
            )
        )
    return rows


def choose_operating_point(
    rows: Sequence[Any], max_false_hit_rate_of_hits: float = 0.05, tolerance: float = 1e-9
) -> dict[str, Any]:
    """Pick a tau by the metric that matters to the people who hit this cache:
    ``false_hit_rate_of_total`` — how often *any* query gets an actively wrong cached answer, not
    just how often a served hit happens to be wrong. Minimize that first; among taus tied at the
    minimum (within ``tolerance``), prefer the highest ``hit_rate`` — the loosest tau that is
    exactly as safe as the safest tau, so no coverage is left on the table for no safety gain.

    ``max_false_hit_rate_of_hits`` is reported alongside as a stricter, secondary bar (the
    conditional "given the cache answered, how often was it wrong" rate). Whether the chosen tau
    clears it is reported honestly as ``meets_threshold`` rather than silently relaxing the bar —
    on an adversarial-weighted synthetic set like this one it may well not (CONVENTIONS.md rule 2:
    report negatives). See :func:`run_and_write_benchmark` for the reasoning behind this choice.
    """
    min_total = min(r.values["false_hit_rate_of_total"].mean for r in rows)
    tied_at_minimum = [
        r for r in rows if r.values["false_hit_rate_of_total"].mean <= min_total + tolerance
    ]
    best = max(tied_at_minimum, key=lambda r: r.values["hit_rate"].mean)
    meets_threshold = best.values["false_hit_rate_of_hits"].mean <= max_false_hit_rate_of_hits
    return {
        "tau": best.extra["tau"],
        "hit_rate": best.values["hit_rate"].mean,
        "false_hit_rate_of_hits": best.values["false_hit_rate_of_hits"].mean,
        "false_hit_rate_of_total": best.values["false_hit_rate_of_total"].mean,
        "meets_threshold": meets_threshold,
        "threshold": max_false_hit_rate_of_hits,
        "selection_rule": "min(false_hit_rate_of_total), tie-broken by max(hit_rate)",
    }


def run_and_write_benchmark(
    output_path: str | Path = "artifacts/benchmarks/semantic_cache.json",
    *,
    taus: Sequence[float] = _DEFAULT_TAUS,
    seeds: Sequence[int] = _DEFAULT_SWEEP_SEEDS,
    max_false_hit_rate_of_hits: float = 0.05,
) -> dict[str, Any]:
    """Run the tau sweep and write ``artifacts/benchmarks/semantic_cache.json`` in the
    CONVENTIONS.md benchmark-envelope schema (``name``/``hardware``/``seeds``/``rows``/``ci``).
    Returns the same payload that was written.
    """
    from localmind.eval.stats import benchmark_json

    result = sweep_tau(taus, seeds=seeds)
    rows = _rows_from_sweep(result)
    operating_point = choose_operating_point(rows, max_false_hit_rate_of_hits)

    payload = benchmark_json(
        name="semantic_cache_tau_sweep",
        hardware="CPU-only, offline synthetic benchmark (no GPU, no network, no real embedding model)",
        seeds=result.seeds,
        rows=rows,
        extra={
            "description": (
                "Hit rate vs. false-hit rate as the semantic cache's similarity threshold tau "
                "is swept from 0.80 to 0.99, against a hand-authored SYNTHETIC labelled query "
                "set (see build_synthetic_queryset() docstring) -- not a real production query "
                "log. A false hit is a returned cached answer that does not match the querying "
                "user's own ground-truth answer; it is measured directly by comparing answers, "
                "not inferred from the similarity score. The query set is deliberately "
                "adversarial-weighted (roughly a third of evaluation queries are constructed "
                "confusable/ambiguous pairs meant to stress-test the false-hit path) rather than "
                "a representative traffic sample, so absolute hit rates here read low and "
                "conditional false-hit rates read high compared to what real production traffic "
                "-- mostly clean paraphrases and mostly-unrelated misses -- would show. The "
                "point of this benchmark is the shape of the tradeoff and the honesty of the "
                "measurement, not a hit-rate number to advertise."
            ),
            "false_hit_rate_of_hits_definition": "false_hits / hits (conditional on the cache answering at all)",
            "false_hit_rate_of_total_definition": "false_hits / total_evaluation_queries",
            "dataset": "synthetic",
            "embedder": "HashingEmbedder (deterministic bag-of-hashed-tokens fake; see module docstring)",
            "operating_point": operating_point,
            "operating_point_reasoning": (
                "false_hit_rate_of_total flattens to its minimum (~0.042, an irreducible floor "
                "from the context-ambiguous exact-duplicate scenario -- see "
                "build_synthetic_queryset()) at tau=0.91 and does not improve further at any "
                "tighter tau in the sweep, because that residual error comes from a literal "
                "cache-key collision (same text, different hidden user context) that a "
                "similarity threshold cannot fix by construction. Every tau above 0.91 therefore "
                "only trades away hit_rate for zero further safety gain, so 0.91 is chosen as "
                "the loosest tau that is exactly as safe as the safest tau. Fixing the remaining "
                "floor requires enriching the cache key with session/account context, not "
                "moving tau -- a genuine design limitation of a text-only semantic cache key, "
                "reported rather than hidden."
            ),
        },
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result_payload = run_and_write_benchmark()
    op = result_payload["operating_point"]
    print(
        f"wrote artifacts/benchmarks/semantic_cache.json — "
        f"chosen tau={op['tau']:.2f}, hit_rate={op['hit_rate']:.3f}, "
        f"false_hit_rate_of_hits={op['false_hit_rate_of_hits']:.3f}"
    )
