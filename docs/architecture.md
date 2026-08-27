# Architecture

The design principle, stated once and applied everywhere: **every component has (a) a naive
baseline, (b) an upgraded version, (c) a benchmark proving the delta.** If a component's
contribution cannot be measured, it does not belong in the repo.

## The one-sentence version

A 31M-parameter decoder-only LM, tokenizer, trainer and inference engine built from scratch on
free-tier GPUs, distilled from a 3B teacher, then deployed as the CPU-resident low-latency control
plane of a multimodal agentic RAG system — with a CI-gated evaluation harness proving each
component's contribution.

## System diagram

```
                          USER
                            |
                    +-------v--------+
                    |  Next.js UI    |
                    +-------+--------+
                    +-------v--------+
                    | FastAPI Gateway|  auth, rate limit, OTel root span
                    +-------+--------+
                +-----------v------------+
                |   AGENT (state machine)|
                |  +------------------+  |
                |  | LocalMind-31M    |<-+-- OUR MODEL LIVES HERE
                |  | - domain router  |  |   laptop CPU, int8
                |  | - query rewrite  |  |   p50 < 20 ms, no GPU
                |  | - relevance grade|  |   3 distilled task heads
                |  +------------------+  |
                +-----------+------------+
                            |  tool calls over MCP
        +-------------------+-------------------+
        v                   v                   v
   RAG SEARCH           SQL TOOL            WEB TOOL
        |                   |                   |
   +----+------+--------+---+----+         (sandboxed,
   v           v        v        v          injection-filtered)
  BM25      SPLADE   Dense   ColBERT
   +--------+-+--------+--------+
            v  RRF fusion (k=60)
       Cross-encoder reranker  (bge-reranker, CPU/ONNX)
            v
     Top-K context + citations
            |
   +--------v------------------+
   | GENERATOR                 |  Qwen3-4B-Instruct Q4_K_M via Ollama
   | (swappable; LocalMind-31M |  <- swap in our own model for the
   |  is an ablation arm)      |     quality-vs-latency plot
   +--------+------------------+
            v
   Answer + inline citations
            |
   +--------v------------------------------+
   | OpenTelemetry (GenAI semconv)         |  all self-hosted, all free
   | -> Phoenix + Prometheus/Grafana       |
   | -> nightly eval job -> regressions    |
   +---------------------------------------+
```

## Where each thing runs, and why

| Component | Runs on | Why there |
|---|---|---|
| Tokenizer training, data prep | Laptop CPU | Embarrassingly parallel, no GPU benefit |
| Pretraining, distillation | Kaggle 2x T4 | The only free GPU allowance; 30 h/week |
| ColQwen2 page indexing | Kaggle T4 | ~2 GPU-h batch job; vectors then queried on laptop |
| **Router / rewriter / grader** | **Laptop CPU** | **31 MB at int8. This is the thesis.** |
| Answer generation | Laptop (Ollama) | 4B Q4_K_M; needs the capacity a 31M model lacks |
| Retrieval, agent, eval | Laptop + Docker | CPU-bound; latency costs are honestly visible |

The split is the argument. The control plane runs on CPU at single-digit milliseconds because it
was distilled for exactly three narrow jobs; the generator is a bigger model because synthesis
genuinely needs one. Rows 6 vs 7 of the headline experiment are where that claim gets tested.

## Boundaries that matter

**`localmind/inference/server.py` vs `localmind/api/`.** The inference server exposes
OpenAI-compatible endpoints for *the model* (`/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`). The API package is the *RAG gateway* — auth, rate limiting, the OTel root span,
and the agent entrypoint. They are separate processes with separate concerns; conflating them is
how a serving layer becomes untestable.

**The `ControlPlane` seam.** The agent never imports torch. It depends on a Protocol with three
methods (`route`, `grade`, `rewrite`). That is what makes the agent testable offline, and what
makes "swap the 31M model for the 4B control" a one-line ablation instead of a refactor.

**Retrieval arms are independent.** Each of BM25 / SPLADE / Dense / ColBERT produces a ranked list
and nothing else. Fusion consumes ranks, not scores, which is why RRF needs no normalization
(ADR 0006) and why adding a fifth arm costs nothing structurally.

## Constraints that shaped the design

Three external limits did more to determine this architecture than any preference:

1. **T4 is SM 7.5** — no bf16, no FlashAttention-2. Forces fp16 + GradScaler, and promotes QK-norm
   and z-loss from nice-to-have to load-bearing (ADR 0001).
2. **12-hour hard session cap** — makes bit-exact resumability architectural, not a feature. Drives
   WSD over cosine (ADR 0002) and the memory-mapped resumable loader.
3. **16 GB laptop** — drives compose profiles, GQA over MHA (ADR 0004), binary quantization with
   rescoring in the vector index, and the decision not to run Airflow (ADR 0007).

Each of these is a case where the free-tier constraint produced a *better* engineering answer than
an unlimited budget would have, which is the argument §20 rule 4 makes.
