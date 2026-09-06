## Phase 4 - pretraining (§8)

Hardware: 2x NVIDIA Tesla T4 (SM 7.5, Turing) on Kaggle free tier | DDP, world_size=2 |
torch 2.10.0+cu128 | fp16 autocast + GradScaler (no bf16 — the hardware has no bf16 tensor cores).
Single run, one seed. **measured**, not projected.

### The run

| | |
|---|---|
| Config hash | `77e970cd68e7156d` |
| Parameters | **30,932,992** (matches §6's assert exactly) |
| Steps | 5,722 |
| Tokens | **1,499,987,968** |
| Wall clock | **14,133 s (3 h 55 m)** |
| Optimizer | AdamW, β=(0.9, 0.95), wd 0.1 |
| Schedule | WSD — 2% warmup / 80% stable / 18% decay |
| Peak LR | 3e-3 |
| Global batch | 262,144 tokens (grad_accum 8 × 2 ranks) |
| Cash cost | **$0.00** |

### Loss

| Step | ce_loss | LR | Phase |
|---|---|---|---|
| 20 | 7.671 | 5.26e-4 | warmup |
| 520 | 3.876 | 3.0e-3 | stable |
| 1,020 | 3.497 | 3.0e-3 | stable |
| 1,510 | 3.348 | 3.0e-3 | stable |
| 2,010 | 3.177 | 3.0e-3 | stable |
| 2,510 | 3.137 | 3.0e-3 | stable |
| 3,010 | 3.071 | 3.0e-3 | stable |
| 3,500 | 2.997 | 3.0e-3 | stable |
| 4,000 | 2.976 | 3.0e-3 | stable |
| 4,500 | 2.956 | 3.0e-3 | stable |
| 5,040 | 2.778 | 2.09e-3 | **decay** |
| 5,540 | 2.714 | 7.80e-4 | decay |
| **5,722** | **2.4509** | 3.03e-4 | decay (end) |

**ce_loss 9.7821 → 2.4509.** The §8 target band was 3.0–3.5; the run finished well below it.

**The WSD decay is where the last gain came from**, which is the argument ADR 0002 makes. Loss moved
only 2.956 → 2.976 over 500 steps late in the stable phase, then **2.956 → 2.454 across the 18%
decay** once the LR started falling. A cosine schedule fixed in advance could not have branched a
deployable checkpoint from mid-run; WSD's whole point under a 12-hour session cap is that it can.

### Throughput and MFU

| Metric | Value |
|---|---|
| Tokens/sec | 107,100 sustained (peak 121,800) |
| **MFU** | **19.4% sustained** |
| grad_norm | 0.92 → 0.15, monotone, never clipped hard |

§8 says: *"Expect 10–15% MFU on your first T4 run. Treat getting to 25%+ as an explicit
sub-project."* This first run reached **19.4%** with no optimisation work — above the predicted
band, below the stretch target. The 25% sub-project (larger micro-batch, fused cross-entropy,
removing dataloader stalls) has not been attempted.

### fp16 stability — the load-bearing result

`scaler_scale` across the whole run: 65,536 → 131,072 (step ~2,000) → 262,144 (step ~4,500).

**Zero halvings in 5,722 steps.** §3.2 warns that repeated GradScaler halving means fp16 overflow.
It never halved; it *doubled twice*, meaning the scaler kept finding headroom and reclaimed more of
fp16's range. That is direct evidence that QK-norm, z-loss and fp32 loss computation are doing the
job they were included for — the things §3.2 calls "load-bearing, not optional decorations" on
hardware with no bf16.

### Honest limitations

- **~30 epochs, not 1.5B fresh tokens.** The corpus was 49,566,720 unique tokens, so 1.5B tokens is
  roughly 30 passes over it. Expect memorisation effects. §8's plan assumes 1.5B *unique* tokens;
  this run does not meet that, and no claim here should be read as if it did.
- **No code data.** `bigcode/the-stack-v2` (20% of the §7 mixture) was gated, then streamed
  pathologically slowly, then returned HTTP 503 from the Hub. It was excluded and the remaining
  weights renormalised to FineWeb-Edu 71.4% / Cosmopedia-v2 21.4% / TinyStories 7.1%. **This model
  will be weak at code.**
- **One seed.** Every other table in this document carries 3 seeds and a bootstrap CI. This run does
  not: a second 4-hour GPU run to produce an error bar was not a good use of a 30 h/week quota. The
  numbers above are a single observation and are labelled as such.
- **No val split.** `val/` was absent, so no held-out loss or bits-per-byte was computed. Training
  loss alone cannot distinguish learning from memorisation — with ~30 epochs, that distinction is
  exactly the one that matters, and it is unmeasured.
- **Not yet distilled.** This is the pretrained base only. The §9 router / grader / rewriter heads,
  which are what make the model useful to the RAG system, have not been trained.

### Artifacts

- Checkpoint, tokenizer and full 5,722-row `metrics.jsonl`: <https://huggingface.co/Adnanbasil/localmind-31m>
- Reproduce: `python -m localmind.data.prepare --n-docs 50000 --exclude stack_v2 --out shards`
  then `torchrun --nproc_per_node=2 -m localmind.train.loop --config configs/train/pretrain.yaml`
