# ADR 0002 — WSD schedule, not cosine

**Status:** Accepted · **Context:** §8

## Decision

Warmup-Stable-Decay: 2% warmup, 80% stable, 18% linear-to-zero decay.

## Why

With cosine you must fix the total step count in advance, and **only the final checkpoint is
usable** — a run killed at 90% yields nothing deployable. With WSD you can branch a short decay
from *any* point in the stable phase and get a deployable model.

Under a 12-hour hard session cap and a 30 GPU-h/week quota, that property is the difference
between "I have a model" and "the run died at 90% and I have nothing." This is a scheduling
decision forced by the compute constraint, not a chase after a better loss curve.

## Bonus

Branch-decaying at 250M / 500M / 1B / 1.5B tokens yields a scaling-law study for ~2 extra
GPU-hours — a one-page result that looks like real research and is nearly free.

## What we give up

Cosine is marginally better-tuned for a known fixed budget. We do not have a known fixed budget;
we have a quota that can evaporate.
