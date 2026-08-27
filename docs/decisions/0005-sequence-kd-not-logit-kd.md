# ADR 0005 — Sequence-level KD, not logit-level KD

**Status:** Accepted · **Context:** §3.3, §9

## The problem

True logit-level KD requires teacher and student to **share a vocabulary**. Qwen's vocab is ~151k;
ours is 16,384. The token boundaries do not align, so there is no correspondence between the two
logit vectors. You cannot take a KL divergence between distributions over different supports.

This is the single most-missed subtlety in small-model distillation, and being able to explain it
is worth more than the technique itself.

## Decision — three arms, in order of practicality

1. **Sequence-level KD (primary).** Train on the teacher's sampled outputs as hard labels.
   Tokenizer-agnostic, and what most real-world distillation actually does.
2. **On-policy correction.** Sample from the *student*, have the teacher score or rewrite, train on
   the correction. Fixes exposure bias — plausibly the real reason Gemma 3 1B and Llama 3.2 1B work
   as well as they do.
3. **Same-tokenizer logit KD (optional arm).** Pretrain LocalMind-100M with *our* tokenizer, then
   do true top-K logit KL 100M to 31M: `L = a*KL(student || teacher_topK) + (1-a)*CE`, K=64, a=0.7.
   Clean and principled; costs one extra pretrain run (~10 GPU-h).

## Storage note

Top-64 logits over 20k sequences x 256 tokens is about 1.2 GB. Dump to HF Hub; do not try to hold
it in a Kaggle session.

## Deliverable

Benchmark the arms against each other and report which won — including if arm 3's extra 10 GPU-h
bought nothing.
