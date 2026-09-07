## Phase 4b - corpus size at a fixed token budget (§8, negative result)

Hardware: 2x Tesla T4 (Kaggle free tier), fp16 + GradScaler, DDP world_size=2. Both runs use the
**same** config (`77e970cd68e7156d`), the same 1,499,987,968-token budget, the same 5,722 steps, and
the **same tokenizer** (sha256 `faf7b8b576f31408` — verified identical, so token ids and therefore
losses are directly comparable). One seed each. **measured**.

### The question

The first run trained 1.5B tokens over a 49.6M-token corpus — about 30 epochs. That is far more
repetition than §8 assumes, so the obvious next move was more data. This tests whether it helps.

### The result — it did not

| Model | Corpus | Epochs | Train ce | **Held-out val ce** | val bpb | gap |
|---|---|---|---|---|---|---|
| **v1** | 49,566,720 tok | ~30.3 | 2.4509 | **2.6891** | **0.9699** | ~0.24 |
| v2 | 196,147,200 tok | ~7.6 | 2.9673 | 3.0437 | 1.0978 | 0.0764 |

Both scored on the same 2,000-document held-out split, 13-gram decontaminated against training,
50 iterations at batch size 8.

**v1 — trained on 4x LESS data — generalises better by 0.35 nats, about 13% in bits-per-byte.**

### Why, most likely

v2's generalisation gap is **0.0764**: its validation loss almost exactly tracks its training loss.
That is not overfitting, it is **underfitting** — at 7.6 epochs the model had not finished extracting
what its corpus had to offer when the token budget ran out. Its *training* loss is worse too
(2.9673 vs 2.4509), which is the tell: a model that is overfitting has a low train loss and a high
val loss, and v2 has neither.

v1, with ~30 passes over a smaller corpus, had enough repetition to consolidate. At 31M parameters
and a fixed 1.5B-token budget, more epochs over less data beat fewer epochs over more.

### What this does not say

- **Not** that data size does not matter. It says that at a *fixed token budget*, spending it on
  breadth rather than depth lost. A larger corpus with a proportionally larger budget was not tested
  and is the obvious follow-up.
- **Not** a robust result. One seed per arm, no confidence interval. The 0.35-nat difference is
  large enough to be unlikely to be noise, but it has not been shown to survive reseeding.
- **Not** a scaling law. Two points do not describe a curve. §8's WSD branch-decay study, which
  would give several points for ~2 extra GPU-hours, remains unrun.

### Honest note on the prediction

Before running this, the stated expectation was that 4x the corpus would produce a better model, and
GPU time was committed on that basis. The measurement contradicted it. The recommendation was wrong,
and it cost ~4 GPU-hours to find out — which is what the compute log is for.

### Artifacts

- v1 (better): `pytorch_model_step5722.pt` · v2: `v2_196M_step5722.pt` — both at
  <https://huggingface.co/Adnanbasil/localmind-31m>
- Reproduce: `python -m localmind.eval.lm --checkpoint <ckpt> --shards <dir-with-val>`
