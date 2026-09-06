# Compute log

Every GPU-hour spent, and on what. The line worth having at the end is: **"the whole thing cost
~60 free GPU-hours and $0."** That claim is only credible if this file is kept honestly, including
the runs that were wasted.

**Cash cost to date: $0.00.** No cloud bills, no paid APIs, no rented GPUs.

## Budget

| Bucket | Estimate (§8) | Spent | Remaining |
|---|---|---|---|
| Pretrain + ablations | 35 h | **4.2 h** | 30.8 h |
| Post-training | 10 h | 0 h | 10 h |
| Teacher generation | 4 h | 0 h | 4 h |
| ColQwen2 indexing | 2 h | 0 h | 2 h |
| Eval sweeps | 8 h | 0 h | 8 h |
| **Total** | **~60 h** | **~8.4 h** | **~51.6 h** |

Kaggle allowance is 30 GPU-h/week, so ~60 h is two weeks of quota spread across 16 calendar weeks.
Comfortable — provided nothing is wasted on runs that were not checkpointed.

## Ledger

Append one row per session. Never delete a row: a session that died at hour 11 with no Hub push is
the most useful entry in the file.

| Date | Platform | GPU | Hours | Run | Outcome | Notes |
|---|---|---|---|---|---|---|
| 2026-09-02 | Kaggle | 2x T4 | ~0.1 | smoke test | ok | ce_loss 9.7772 -> 4.5088 in 18.3 s, 23.4% MFU |
| 2026-09-02 | Kaggle | 2x T4 | ~0.1 | pretrain (aborted) | **loss** | Stopped deliberately: only 19.3M unique tokens available |
| 2026-09-02/03 | Kaggle | 2x T4 | **~4.0** | data prep | **WASTED** | Data prep is CPU-only work; a GPU sat idle attached to it. Session then reclaimed mid-build and the shards were lost. |
| 2026-09-06 | Kaggle | 2x T4 | ~0.3 | data prep (rebuild) | ok | 49,566,720 tokens; done in same session as training this time |
| 2026-09-06 | Kaggle | 2x T4 | **3.93** | **main pretrain** | **ok** | 5,722 steps, 1.5B tokens, ce_loss 9.7821 -> 2.4509, 19.4% MFU |

## Non-GPU compute

All of this ran on a laptop CPU at zero cost and is not counted against the GPU budget:
tokenizer training and benchmarks, data prep and dedup, the entire inference-engine chapter,
retrieval, the agent, and the eval harness.

- Development machine: Windows 11, 12 logical cores, CPU-only PyTorch 2.13.0+cpu.

## Rules

1. Log the session **before** you start it, so a session that dies still leaves a row.
2. Record hours actually consumed from the quota, not wall-clock of useful work.
3. A run with no Hub push is a **loss** — record it as such. That is the whole point of the file.
4. Ablations run at the 12M proxy first (~1 h each); only the winner is confirmed at 31M.
5. **Do not attach a GPU to CPU-only work.** ~4 h of this log was burned running data preparation
   inside a GPU session. Kaggle meters the session, not the device utilisation. Prepare data and
   train in the *same* session (`/kaggle/working` is wiped on every restart), but never leave a GPU
   attached to a job that does not use it.
