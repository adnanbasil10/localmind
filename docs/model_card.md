# Model card — LocalMind-31M

> **Status: not yet trained.** This card is scaffolded with everything determined by the
> architecture and the data plan. Every field marked _pending_ requires a GPU run that has not
> happened. No number appears here until it has been measured.

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

Target 1.5B tokens (~48 tokens/param); 3B is the stretch.

**Nothing was scraped.** Every corpus is permissively licensed and its license is recorded.

Processing: license filter, language ID, quality heuristics, MinHash-LSH near-dedup (5-grams,
threshold 0.8), PII scrub, 13-gram decontamination against the golden eval set.

- Dedup ratio: _pending_
- Tokens removed by decontamination: _pending_

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
| Total compute | _pending_ (budgeted ~35 GPU-h incl. ablations) |

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
- **English-centric.** The mixture is overwhelmingly English; other languages will be poor.
- **Small-model brittleness.** Expect sensitivity to prompt format. The chat template is rendered
  by the tokenizer specifically so format drift cannot silently creep in.
- **Inherited teacher bias.** Distilled from Qwen2.5-3B-Instruct; its biases propagate.
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
