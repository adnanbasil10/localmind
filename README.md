# LocalMind

**A 31M-parameter decoder-only LM, built from scratch, that earns its place inside a production
agentic RAG system — designed to train and deploy entirely on free-tier compute, a laptop, and
Docker.**

Hard constraint: **total cash cost = $0.** Every model is open-weights, every service is
self-hosted, every GPU-hour comes from a free allowance. Spend to date: **$0.00, 0 GPU-hours.**

---

## ⚠️ Read this before the tables

**The model is not trained yet.** Every number below is a measurement of the *systems* — the
tokenizer, the attention backends, the inference engine, the retrieval fusion, the caches — running
on **CPU with untrained weights**, or on **synthetic corpora with deterministic stand-in models**.

That distinction is load-bearing, and it is labelled on every table:

| Label | Meaning |
|---|---|
| **measured** | Real timing/memory on this machine. Reproducible via `just`. |
| **synthetic** | Real code, real measurement, but a synthetic corpus and/or fake embedder — the *harness* is validated, not retrieval quality on a real corpus. |
| **not run** | Requires a GPU, network, or a trained checkpoint. **No value is invented.** |

Nothing here reports model *quality*, because no model has been trained. The pretraining,
distillation and quality tables are scaffolded with every cell marked `not run`.

---

## Results

### Inference engine — CPU, 12M proxy, 3 seeds, bootstrap 95% CI · *measured*

| Stage | Result |
|---|---|
| Naive → contiguous KV cache | **7.9×** @ 512 new tokens *(spec expected 10–20×; reporting what was measured)* |
| Paged vs contiguous — reserved-KV waste | **66–96% → 1.1–16.7%** |
| Max concurrent sequences @ 64 MB KV budget | **21 → 341 (16.2×)** |
| External fragmentation, paged | **identically zero**, by construction |
| Prefix cache hit rate | **87.5%** request / **73.7%** token · TTFT −11.4% |
| Chunked prefill | **49%** better p99 ITL |
| Constrained decoding — invalid JSON | **100% → 0%** |
| n-gram speculative decoding | **1.68×** |
| 31M export, `q8_0` GGUF | **33,217,664 B** = 31.68 MiB = 33.22 MB |
| vLLM baseline | **not run** — no GPU. No number invented. |

### Retrieval — 88-doc corpus, deterministic stand-in models · *synthetic*

| Config | nDCG@10 |
|---|---|
| BM25 (own implementation, hand-verified) | 0.713 |
| 4-arm RRF + cross-encoder | **0.857** |

| Index engineering | Result |
|---|---|
| Binary quantization + rescoring | **94.7% recall retained at 32× compression** |
| Post-filter recall under a 1%-selective filter | **1.0 → 0.33** |
| Pre-filter recall, same filter | **1.0** (holds) |

That last pair is the failure §12 says almost no candidate can explain, reproduced as a runnable
test rather than an assertion.

### Tokenizer · *measured*

All values below are read directly from `artifacts/benchmarks/tokenizer.json`.

| | Yours | tiktoken cl100k | GPT-2 |
|---|---|---|---|
| Bytes/token ↑ | **5.5958** | 5.2527 | 4.8776 |
| Fertility ↓ | **1.1926** | 1.2705 | 1.3682 |
| Encode MB/s ↑ | 4.4838 | 12.5794 | 12.7910 |
| Ratio to tiktoken | **2.81×** (DoD: ≤5×) | 1.0× | — |

Merge loop, naive → incremental: **22.08×** (45.0119s → 2.0382s @ vocab 2048).

### Model · *measured*

| | |
|---|---|
| Parameters | **30,932,992** excl. norms · 30,942,720 incl. · 22.5M non-embedding |
| KV cache | MHA 16 KB/token → **GQA 4:1 = 4 KB/token** |
| Attention backends | naive / sdpa_math / sdpa_efficient agree to **1e-3** in fp32 |
| Overfit correctness test | **CE 0.00614** on 100 random sequences, CPU, 140 steps |
| Doc-boundary masking | packed doc B vs alone, max-abs-diff **5.96e-8** with mask (1.19e-7 on `sdpa_math`) vs **0.182** without — printed on every run by `tests/test_model.py::test_packed_document_cannot_attend_across_a_boundary`; the assertions are the looser bounds that guard it |

### Guardrails · *measured*

| | |
|---|---|
| Injection block rate, **held-out** paraphrases | **3/8 = 37.5%**, Wilson 95% CI **[13.7%, 69.4%]** ← the honest number |
| Injection block rate, in-sample corpus | 41/41 — *in-sample; 4 patterns widened after seeing misses* |
| `calculate` sandbox | **69/69** hostile inputs rejected (`__import__`, `__mro__`, `9**9**9`, `round(2, -10**7)`, fullwidth-dunder, …) |
| Agent termination | 300 seeded fuzzer runs, **all terminate**, 0 hit the step cap |

**n = 8.** That interval is far too wide to support any comparison, and it is quoted here rather
than hidden because the project's own rule 5 forbids a bare point estimate. Treat 37.5% as
"this defence does not currently generalise", not as a measurement.

**On the sandbox:** the 69/69 refusals are pre-execution. CPython's watchdog *cannot* interrupt
GIL-holding C code — `round(2, -10**7)` returned `ok=True` after 11.05 s under a 1 s cap before the
guard was added — so containment comes from refusing the expression, never from the timeout.

### Semantic cache — τ sweep · *synthetic*

Hit rate falls 0.562 → 0.062 across τ 0.80 → 0.99, but the false-hit rate **floors at τ = 0.91** —
an irreducible cache-key collision that no threshold can fix. Operating point τ = 0.91, reported
`meets_threshold=False` against the 5% bar rather than moving the bar.

### Eval harness · *validated against hand-computed values*

Cohen's κ reproduces **0.4000000000** exactly on a hand-computed 2×2 · bootstrap half-width
0.04287 vs analytic 0.04383 · Wilcoxon matches brute-force enumeration to **1e-12**. The lexical
fallback judge scores **κ = 0.591**, below the 0.6 trust bar, so the harness **refuses it** and
auto-suppresses judged metrics.

### Not run — needs a GPU, and says so

**Blocked on a GPU:** pretraining (1.5B tokens), the WSD scaling-law study, the Muon-vs-AdamW
comparison, all four post-training stages, the §9 5e comparison matrix (**0 of 24 cells measured**),
ColQwen2 indexing, and the vLLM baseline. The harnesses exist and are tested; they run the moment a
checkpoint does. The 5e matrix evaluates to `not-evaluable`, never `failed`, so an unrun experiment
can never be mistaken for a negative result.

**Not blocked — simply not run yet.** These need no GPU and are honestly outstanding:
- §5 DoD says round-trip fuzz over **1M** random Unicode strings; the suite runs **200** hypothesis
  examples. The property holds on everything tried, but 200 is not 1M.
- The vocab ablation {4k, 8k, 16k, 32k} has **not** been run. `tokenizer.json` covers one vocabulary.
  The bytes/token half of that sweep is CPU-only; only the proxy-BPB half needs a GPU.
  ADR 0003 is marked "Accepted (pending ablation confirmation)" for exactly this reason.
- The SentencePiece-unigram comparison row (§5's table) is absent, though `sentencepiece` is
  already declared in the `tok` extra.

---

## Quickstart

```bash
uv venv && uv pip install -e ".[torch,tok,dev]"
just test-fast            # offline-safe: no network, GPU, or docker needed
just up core              # postgres+pgvector, redis, api  (~1.5 GB)
just bench-inference      # reproduces the KV-cache and paged-attention tables
```

GPU work runs through the thin Kaggle launchers in `notebooks/kaggle/`.

## Architecture

`docs/architecture.md`. The short version: the 31M model is the **control plane** — routing, query
rewriting, relevance grading, injection classification — on a laptop CPU at int8. Answer generation
is Qwen3-4B via Ollama, because a 31M model cannot synthesize a grounded answer and the model card
says so plainly.

## Documentation

| | |
|---|---|
| `docs/benchmarks.md` | **the deliverable** — every ablation, every CI, every negative result |
| `docs/decisions/` | 7 ADRs: fp16-not-bf16, WSD-over-cosine, vocab 16k, GQA 4:1, sequence-KD, RRF, Prefect-over-Airflow |
| `docs/model_card.md` | data licenses, limitations, and what this model **cannot** do |
| `docs/compute_log.md` | every GPU-hour spent, including the wasted ones |
| `docs/runbook.md` | what to do when an alert fires |
| `CONVENTIONS.md` | the contracts every contributor is bound by |

## Honest negatives

Kept deliberately, per §20 rule 2:

- Naive→KV speedup is **7.9×**, under the 10–20× the plan expected.
- Injection defense generalizes to **37.5%** on held-out paraphrases despite 100% in-sample.
- Tuned fusion **beat** RRF on the synthetic corpus, contradicting ADR 0006's prior — recorded in
  the ADR as provisional, with what would settle it.
- Continuous batching wins TTFT but **not** throughput; length-bucketed decode is a workaround for
  the model's dense `past_kv`, and real ragged batching needs a Phase 2 change.
- The GGUF export is verified against our own reader, **never llama.cpp**, and is lossy — the
  `llama` architecture has no QK-norm tensors, so export refuses without `allow_lossy=True`.
- BPE exhausted mergeable pairs at 8,062 of a requested 16,384 vocab on the small corpus.
- The GGUF files themselves are **not committed** (~90 MB); regenerate with
  `uv run python -m localmind.inference.quantize --export-gguf`. Only the ~200 KB of benchmark JSON
  is versioned. An earlier draft of this README quoted "31.68 MB" where the true figure is 31.68
  **MiB** — corrected above to exact bytes, since "it fits on free CPU tier" is the claim resting
  on it.
- An earlier inference benchmark was **discarded** for aliasing machine drift onto variant identity;
  the discredited run is kept as `artifacts/benchmarks/inference_superseded_blocked_ordering.json`
  so the correction is auditable. Do not read numbers from that file — it is retained as a record of
  a retraction, not as a result.

## License

Apache-2.0
