# ADR 0006 — RRF as the fusion baseline

**Status:** Accepted · **Context:** §12

## Decision

`score(d) = sum_arms 1/(k + rank_arm(d))`, k=60, as the default fusion across all four retrieval
arms.

## Why

RRF is **rank-based**, so it needs no score normalization. That matters more than it sounds: BM25
scores, cosine similarities, SPLADE weights, and ColBERT max-sim totals live on incomparable
scales, and every min-max or z-score normalization scheme leaks dev-set information and drifts as
the corpus changes. RRF sidesteps the entire problem.

It is also remarkably hard to beat.

## The experiment that matters

Tune min-max-normalized weighted fusion on a dev set and compare. **Report the result even when
tuning loses** — which it often does. A tuned scheme that fails to beat a parameter-free baseline
is a real finding about the method, and reporting it is the difference between measuring and
marketing.

## Risk

k=60 is inherited convention, not a tuned value for this corpus. Sweep it; if the optimum is far
from 60, say so.

## First measurement — the prior did not hold

**Status of this ADR: provisional, pending a real corpus.**

The first run of the experiment above **contradicted the expectation stated in it**: min-max
weighted fusion, tuned on a dev split, *beat* RRF.

That result is recorded here rather than discarded, because discarding it would be exactly the
failure this ADR was written to prevent. But it should not yet be read as a refutation, for two
reasons that are properties of the harness rather than of the method:

1. The corpus was **88 synthetic documents**. Tuning a handful of arm weights on a dev split that
   small overfits almost by construction.
2. Every retrieval arm was a **deterministic fake** — no embedding model, reranker or SPLADE
   checkpoint was downloadable in the build environment. The arms' score distributions are
   therefore artificial, and the normalization that weighted fusion depends on is being tuned
   against synthetic scales.

**What would settle it:** re-run on the real corpus with real models, with the tuning split held
out properly, 3+ seeds and a paired test. If tuned fusion still wins there, this ADR is wrong and
should be rewritten to say so — RRF's appeal is that it is parameter-free and robust, not that it
is unbeatable.

Until that run exists, RRF remains the default because its advantage is robustness under
distribution shift, which a small synthetic corpus cannot exercise.
