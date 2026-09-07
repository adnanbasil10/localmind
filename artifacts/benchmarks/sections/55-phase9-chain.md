## Phase 9 - post-training chain, first end-to-end run (§9)

Hardware: 2x Tesla T4 (Kaggle free tier). Base model: v1 (`pytorch_model_step5722.pt`, held-out
val ce 2.6891). Teacher: **`--fake-teacher`, a deterministic synthetic generator — NOT
Qwen2.5-3B-Instruct.** One run, no seeds, no CI. **measured**, but see the verdict.

### What this run establishes, and what it does not

**Establishes:** the four stages train, chain correctly off one another, and load checkpoints
cleanly. That was genuinely unknown beforehand — all three CLIs were stubs whose error message
told the reader to run a notebook that invoked those same stubs, so §9 had never executed.

**Does not establish:** anything about router accuracy, grader F1, or rewrite quality. The teacher
is synthetic. The §9 5e comparison matrix stays `not run`.

### Results

| Stage | Steps | Measured | Verdict |
|---|---|---|---|
| SFT data | — | 4,000 examples: grader 667/666, rewriter 1,333, router 445/444/445 | balanced, fine |
| KD (sequence) | 308 | loss **5.2013 → 0.0012** | **too good — see below** |
| DPO | 625 | loss **0.6931 → 0.0000**, reward margin **+12.1608**, acc **1.00**, KL k1 −4.5926 / k3 **4,005,490** | **degenerate, correctly flagged** |
| GRPO | — | T=1.25, group 8 | see the run log |

### The failure chain, which is the actual finding

1. The fake teacher emits **deterministic templated** outputs.
2. KD therefore drives training loss to **0.0012** — it memorised the templates. A distillation run
   that reaches near-zero loss has overfit its teacher, whichever teacher that is.
3. That leaves a policy with **near-zero entropy**: it assigns almost all probability mass to exact
   token sequences.
4. DPO forms log-probability **ratios** against that policy as a frozen reference. Ratios against a
   near-certain distribution explode: **KL = 4,005,490 nats/token** against a 0.5 warning bar.
5. DPO's `loss → 0.0000` and `acc 1.00` then look like triumphs. They are the opposite: perfect
   separation of every preference pair means the model learned the *template*, not the preference.

The two KL estimators disagreeing in sign (k1 = −4.59, k3 = +4.0e6) is the numerical fingerprint of
log-ratios blowing up in both directions — a healthy run has both small and positive.

### Why this counts as the guardrails working

§9 5c says plainly: *"runaway KL means your β is wrong."* The run **said so itself**, loudly, rather
than writing out a broken policy and reporting success. The same is true of GRPO's greedy-sampling
check added the same day: greedy collapses every group, making advantages exactly zero and
`grad_norm` 0.0, so it now hard-errors instead of "succeeding" while learning nothing.

Three silent no-ops were found and closed getting this far:
- greedy sampling made GRPO a no-op (now a hard error),
- `_pad_rollouts` right-truncated, masking **every** label when the prompt exceeded `seq_len`
  (the grader prompt is ~400 tokens, so every small-`seq_len` run trained on nothing),
- `StageResult` field-name mismatch reported `nan -> nan` while training correctly.

### What is needed for numbers worth quoting

1. **A real teacher.** Qwen2.5-3B-Instruct via vLLM, ~1–2 GPU-h for 50k short outputs.
2. **Stop KD short of convergence.** Early-stop on a held-out slice of the teacher data; a KD loss
   near zero means the student has memorised the teacher rather than absorbed it.
3. **Then re-tune DPO's β** against the resulting reference, using the KL warning as the signal it
   was built to be.

Until those exist, no §9 quality claim appears anywhere in this repository.
