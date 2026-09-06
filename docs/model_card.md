# Model card — LocalMind-31M

> **Status: pretrained base trained; post-training not started.** The §8 pretraining run is
> complete and its numbers are below. The §9 router / grader / rewriter heads — the thing that makes
> this model useful to the RAG system — have **not** been trained. Fields still marked _pending_
> require that work.
>
> Weights: <https://huggingface.co/Adnanbasil/localmind-31m>

## Model details

| | |
|---|---|
| **Name** | LocalMind-31M |
| **Architecture** | Decoder-only transformer: RMSNorm (pre-norm), RoPE, GQA 4:1, SwiGLU, QK-norm, tied embeddings, no biases |
| **Parameters** | 30,932,992 excluding norms; 30,942,720 including them. 22.5M non-embedding. |
| **Vocab** | 16,384, byte-level BPE (no UNK token is representable) |
| **Context** | 1024, extended to 2048 in a short annealing phase |
| **Precision** | Trained fp16 + GradScaler (T4 is SM 7.5, no bf16); served int8 |
| **License** | Apache-2.0 |

## Intended use

**In scope.** LocalMind-31M is the **control plane** of an agentic RAG system, not an answer
generator. It does exactly three jobs, on a laptop CPU:

1. **Domain routing** — `in_domain` / `out_of_domain` / `needs_web`
2. **Query rewriting** — conversation-aware rewrite to a standalone search query
3. **Relevance grading** — binary judgement on a (query, chunk) pair
4. (Guardrail) **Prompt-injection classification** over retrieved chunks

**Out of scope — stated plainly.** A 31M model **cannot** synthesize a grounded answer from 4k
tokens of retrieved context. Do not use it for that. Answer generation is Qwen3-4B-Instruct via
Ollama. Any use as a general-purpose chat model will produce bad output, and that is expected
rather than a defect.

## Training data

| Source | Share | License | Purpose |
|---|---|---|---|
| FineWeb-Edu (`sample-10BT`) | 50% | ODC-By | general English quality |
| Cosmopedia v2 | 15% | Apache-2.0 | synthetic textbook density |
| The Stack v2 / StarCoder2 (Python, Rust, SQL, YAML) | 20% | permissive-only filter | code + structure |
| Domain corpus (framework docs, RFCs, permissive papers) | 10% | per-document, recorded at ingest | domain alignment |
| TinyStories | 5% | CDLA | early curriculum |

Target 1.5B tokens (~48 tokens/param); 3B is the stretch. **The table above is the PLAN. What
was actually used differs — see the notes directly below.**

**Nothing was scraped.** Every corpus is permissively licensed and its license is recorded.

Processing: license filter, language ID, quality heuristics, MinHash-LSH near-dedup (5-grams,
threshold 0.8), PII scrub, 13-gram decontamination against the golden eval set.

- Corpus actually used: **49,566,720 tokens** from 49,976 documents (50,000 drawn, 18 removed by
  filters, 6 by MinHash near-dedup).
- **The Stack v2 was NOT included.** It is gated on the Hub, then streamed pathologically slowly,
  then returned HTTP 503. Weights were renormalised to FineWeb-Edu 71.4% / Cosmopedia-v2 21.4% /
  TinyStories 7.1%. **This model is weak at code**, and that is a data decision, not a model one.
- The run trained 1.5B tokens over a 49.6M-token corpus, i.e. **~30 epochs**. §8 assumes 1.5B
  *unique* tokens. Memorisation effects should be assumed until a held-out evaluation says otherwise.

## Training

| | |
|---|---|
| Hardware | 2x NVIDIA T4 16GB (Kaggle free tier) |
| Schedule | WSD — 2% warmup, 80% stable, 18% linear decay |
| Peak LR | 3e-3 (min = peak/10) |
| Optimizer | AdamW, betas (0.9, 0.95), wd 0.1 excluding norms/bias/embeddings |
| Global batch | ~262,144 tokens/step |
| Grad clip | 1.0 |
| z-loss | 1e-4 |
| Total compute | **3 h 55 m** on 2x T4 (14,133 s), 5,722 steps, 1,499,987,968 tokens |
| Final train loss | **2.4509** (from 9.7821) |
| Sustained MFU | **19.4%** |
| GradScaler halvings | **0** across the run (no fp16 overflow) |

## Evaluation

_All pending._ No result is published until it carries 3+ seeds and a bootstrap 95% CI.

| Task | Metric | LocalMind-31M | Qwen3-4B baseline |
|---|---|---|---|
| Domain router | accuracy | _pending_ | _pending_ |
| Relevance grader | F1 | _pending_ | _pending_ |
| Query rewriter | win-rate | _pending_ | _pending_ |
| Latency (CPU) | p50 ms | _pending_ | _pending_ |

**Target (§9 DoD):** beat the 4B on latency by >20x at >=95% of its accuracy on at least one task,
running on CPU with no GPU. If this target is missed, the negative result is reported here rather
than removed.

## Limitations and risks

- **Not a generator.** See "Out of scope" above.
- **Trained ~30 epochs over 49.6M tokens**, not 1.5B unique ones. Memorisation is likely and
  unmeasured: no held-out split was built, so training loss cannot distinguish learning from
  recall. Treat 2.4509 as a training-loss figure only.
- **Weak at code.** The Stack v2 portion of the mixture was unavailable; there is no code in
  this model's training data.
- **English-centric.** The mixture is overwhelmingly English; other languages will be poor.
- **Small-model brittleness.** Expect sensitivity to prompt format. The chat template is rendered
  by the tokenizer specifically so format drift cannot silently creep in.
- **Not distilled yet.** The §9 SFT/KD/DPO/GRPO stages have not run, so this checkpoint has no
  task heads and no teacher-inherited bias — it is a raw pretrained base. Once distillation
  happens, Qwen2.5-3B-Instruct's biases will propagate and this note must be updated.
- **Web-derived corpus.** FineWeb-Edu is filtered but not curated by hand; toxic and factually
  wrong text is present at some rate.
- **Adversarial input.** The injection classifier is a mitigation, not a guarantee. Measured block
  rate is reported in the eval harness; it will not be 100%.

## Reproduction

```bash
git clone <repo> && cd localmind
uv venv && uv pip install -e ".[torch,tok,data,dev]"
just test
# GPU runs go through the Kaggle launchers in notebooks/kaggle/
```

Every run is a config file under `configs/`, hash-logged. Checkpoints resume bit-exactly across a
session boundary — model, optimizer, GradScaler, dataloader shard position, and RNG state.
