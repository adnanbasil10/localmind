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
