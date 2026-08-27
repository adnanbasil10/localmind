# ADR 0001 — fp16 + GradScaler, not bf16

**Status:** Accepted · **Context:** §3.2

## Decision

Train in fp16 autocast with `torch.amp.GradScaler`, fp32 master weights, and fp32 loss.
Never bf16, anywhere, in any config.

## Why

The free GPU is a Kaggle T4 — Turing, SM 7.5. **Turing has no bfloat16 tensor cores.** bf16 on
this hardware is either unsupported or emulated and slow. This is not a preference; it is the
hardware.

## Consequence: things that become load-bearing

fp16 carries ~5 exponent bits against bf16's 8, so overflow is a live failure mode rather than a
theoretical one. Four things stop being optional decorations and become structural:

1. **QK-norm** — RMSNorm on Q and K per-head before the dot product. Without it, attention-logit
   blowup kills a run at hour 9 of a 12-hour session, after the quota is already spent.
2. **z-loss** (1e-4) — keeps the softmax normalizer from drifting.
3. **fp32 loss computation** — cross-entropy in fp16 loses precision exactly where it matters.
4. **fp32 norm reductions** — the `.float()` cast inside RMSNorm is not stylistic.

## Monitoring

Log the GradScaler scale factor **every step**. Repeated scale halving means overflow, and it
means something upstream is wrong — not that the scaler is doing its job.

## What we give up

The bf16 convenience of never thinking about dynamic range. We accept that cost because the
alternative is renting an Ampere GPU, which violates the $0 constraint that defines the project.

## If this is wrong

On Ampere+ hardware, switch to bf16 and delete the GradScaler. QK-norm and z-loss should stay —
they are cheap and they are good practice regardless.
