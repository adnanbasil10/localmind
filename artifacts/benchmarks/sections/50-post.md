## Phase 5 (SS9 5e) — prompt vs retrieve vs finetune vs distill

| Method | Params updated | Router acc | Grader F1 | Rewrite win-rate | p50 latency (ms) | Hardware |
|---|---|---|---|---|---|---|
| Qwen3-4B, zero-shot prompt | 0 | not-yet-run | not-yet-run | not-yet-run | not-yet-run | laptop GPU/CPU |
| Qwen3-4B, few-shot prompt | 0 | not-yet-run | not-yet-run | not-yet-run | not-yet-run | laptop |
| Qwen3-4B + RAG context | 0 | not-yet-run | not-yet-run | not-yet-run | not-yet-run | laptop |
| Qwen3-4B + LoRA (r=16) | ~0.1% | not-yet-run | not-yet-run | not-yet-run | not-yet-run | T4, ~2 GPU-h |
| LocalMind-31M, SFT only | 100% | not-yet-run | not-yet-run | not-yet-run | not-yet-run | laptop CPU |
| LocalMind-31M, SFT + KD + GRPO | 100% | not-yet-run | not-yet-run | not-yet-run | not-yet-run | laptop CPU |

0/24 cells measured on: no GPU and no trained checkpoint available in this environment. `not-yet-run` means exactly that -- no GPU and no trained checkpoint were available, so no number was produced. None of these cells is estimated.
