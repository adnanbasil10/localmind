# LocalMind — Implementation Plan (Zero-Cost Edition)

**A small language model, built from scratch, that earns its place inside a production agentic RAG system — trained and deployed entirely on free-tier compute, a laptop, and Docker.**

Version 3.0. Hard constraint: **total cash cost = $0.** No cloud bills, no paid APIs, no rented GPUs. Every model is open-weights, every service is self-hosted, every GPU-hour comes from a free allowance. Read §0 and §3 first — §3 is where the constraint actually changes engineering decisions.

---

## 0. What changed from the original plan

The original 17-part outline is structurally sound: build bottom-up, understand every layer, benchmark everything. Eleven upgrades make it defensible in a 2026 interview instead of a 2022 one.

| # | Original | Upgrade | Why |
|---|---|---|---|
| 1 | Tiny LLM generates the final RAG answers | **Tiny LLM is the router / grader / rewriter; a 4B local model generates answers** | A 31M model cannot synthesize a grounded answer from 4k tokens of context. It *can* do binary relevance grading and query rewriting at 50× lower latency, **on CPU**. This turns the toy into a real system component with a measurable win. |
| 2 | GPT-2 architecture (LayerNorm, learned pos-emb, MHA, GELU MLP) | **Llama-3/Qwen3-class: RMSNorm, RoPE, GQA, SwiGLU, QK-norm, tied embeddings, no biases** | Every frontier open model uses these. Building GPT-2 in 2026 signals you read a 2019 tutorial. |
| 3 | AdamW + cosine schedule | **AdamW baseline → benchmark against Muon; WSD schedule instead of cosine** | Muon (Kimi K2) is the biggest optimizer result in two years. WSD (MiniCPM/DeepSeek) lets you branch checkpoints — which matters enormously when your GPU sessions are capped at 12 hours. |
| 4 | "Fine-tune with LoRA" | **SFT → sequence-level KD from a 3B teacher → DPO → GRPO; optional same-tokenizer logit KD from your own 100M sibling** | Distillation is how Gemma 3 1B and Llama 3.2 1B were actually made. It's the only way a 31M model gets genuinely useful. |
| 5 | Custom `/generate` endpoint | **OpenAI-compatible `/v1/chat/completions` + paged KV cache + continuous batching + prefix caching + speculative decoding** | Custom APIs are unusable by real clients. Paged/continuous batching is the vLLM contribution; a simplified version you wrote yourself is top-tier ML-systems interview material. |
| 6 | Fixed-size chunking | **Contextual retrieval (Anthropic) + late chunking + layout-aware splits** | Cuts retrieval failures ~35–50%. Costs one cheap LLM call per chunk at ingest — a perfect job for your own model, which is free to run. |
| 7 | BM25 + dense + rerank | **Add SPLADE (learned sparse) and ColBERT late interaction as arms 3 and 4; RRF fusion baseline** | Two-arm hybrid is table stakes. Four arms with a proper fusion ablation is a research-grade retrieval study. |
| 8 | OCR → text → embed for PDFs | **ColQwen2 visual document retrieval as arm B, benchmarked against the OCR pipeline** | Skipping OCR entirely by embedding page images with late interaction is current SOTA for document RAG. Index on a free T4, query on your laptop. |
| 9 | Tools defined in the agent code | **Expose every tool as an MCP server; agent is a typed state machine; constrained decoding for tool-call JSON** | MCP is the interop standard now. Constrained decoding is how you make tool calls reliable instead of hoping. |
| 10 | Eval as the last step | **Eval harness built at Phase 4, wired into GitHub Actions as a merge gate from then on** | Free for public repos. "I ran evals at the end" is a demo. "Every PR runs 240 eval questions and blocks on nDCG@10 regression >2%" is engineering. |
| 11 | AWS deployment | **Docker Compose + a free Hugging Face Space + Cloudflare Tunnel** | "Deployed on AWS" is a weaker credential than "one command, works on any machine, here's a live public URL." And it costs nothing. |

Two things the original understates completely:

- **Statistical rigor.** Every benchmark gets ≥3 seeds, bootstrap 95% CIs, paired tests. A single number with no variance is not a result.
- **Prompt injection.** You ingest untrusted PDFs and hit the open web, then feed both to a tool-calling agent. Indirect prompt injection is the real security story of this architecture and almost nobody building portfolio RAG addresses it.

**Scoping note:** Phases 6–12 (ingestion, retrieval, RAG, agent, observability, caching, infra) are standard production-RAG work. If that ground is already familiar, compress those phases hard — reuse libraries, don't reimplement — and spend the reclaimed weeks on Phases 1–5 and the eval harness. Phases 1–5 are what almost no candidate has.

---

## 1. North star

**One sentence:** A 31M-parameter decoder-only LM, tokenizer, trainer and inference engine built from scratch on free-tier GPUs, distilled from a 3B teacher, then deployed as the CPU-resident low-latency control plane of a multimodal agentic RAG system — with a CI-gated evaluation harness proving each component's contribution.

```
                          USER
                            │
                    ┌───────▼────────┐
                    │  Next.js UI    │
                    └───────┬────────┘
                    ┌───────▼────────┐
                    │ FastAPI Gateway│  auth, rate limit, OTel root span
                    └───────┬────────┘
                ┌───────────▼────────────┐
                │   AGENT (state machine)│
                │  ┌──────────────────┐  │
                │  │ LocalMind-31M    │◄─┼── YOUR MODEL LIVES HERE
                │  │ • domain router  │  │   runs on laptop CPU, int8
                │  │ • query rewrite  │  │   p50 < 20 ms, no GPU needed
                │  │ • relevance grade│  │   3 distilled task heads
                │  └──────────────────┘  │
                └───────────┬────────────┘
                            │  tool calls over MCP
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   RAG SEARCH           SQL TOOL            WEB TOOL
        │                   │                   │
   ┌────┴──────┬────────┬───┴────┐         (sandboxed,
   ▼           ▼        ▼        ▼          injection-filtered)
  BM25      SPLADE   Dense   ColBERT
   └────────┬─┴────────┴────────┘
            ▼  RRF fusion
       Cross-encoder reranker  (bge-reranker, CPU/ONNX)
            ▼
     Top-K context + citations
            │
   ┌────────▼──────────────────┐
   │ GENERATOR                  │  Qwen3-4B-Instruct Q4_K_M via Ollama
   │ (swappable; LocalMind-31M  │  ← swap in your own model for the
   │  is an ablation arm)       │    published quality-vs-latency plot
   └────────┬──────────────────┘
            ▼
   Answer + inline citations
            │
   ┌────────▼──────────────────────────────┐
   │ OpenTelemetry (GenAI semconv)          │  all self-hosted, all free
   │ → Phoenix / Langfuse + Prometheus       │
   │ → nightly eval job → regression alerts  │
   └────────────────────────────────────────┘
```

**Design principle:** every component has (a) a naive baseline, (b) an upgraded version, (c) a benchmark proving the delta. If you can't measure a component's contribution, don't build it.

---

## 2. Repository layout

```
localmind/
├── README.md                  # results-first: benchmark tables above the fold
├── pyproject.toml             # uv-managed
├── justfile
├── docker-compose.yml         # profiles: core | obs | full
├── .github/workflows/
│   ├── ci.yml                 # ruff + pyright + pytest
│   └── eval-gate.yml          # CPU-only, blocks merge on regression
│
├── configs/                   # every run is a config file, hash-logged
│   ├── model/{12m_proxy,31m,100m_teacher}.yaml
│   ├── train/{pretrain,sft,kd,dpo,grpo}.yaml
│   └── retrieval/*.yaml
│
├── notebooks/kaggle/          # thin launchers: clone repo, pip install, run just
│   ├── 01_pretrain.ipynb
│   ├── 02_distill.ipynb
│   └── 03_index_colqwen.ipynb
│
├── localmind/
│   ├── tokenizer/  bpe.py regex_split.py tokenizer.py bench.py
│   ├── model/      config.py rmsnorm.py rope.py attention.py swiglu.py
│   │               block.py transformer.py init.py
│   ├── data/       prepare.py dedup.py filter.py packing.py loader.py
│   ├── train/      loop.py optim/{adamw,muon}.py schedule.py mfu.py
│   │               checkpoint.py  # HF Hub push/resume — session-cap safe
│   ├── post/       sft.py kd.py dpo.py grpo.py lora.py
│   ├── inference/  kv_cache.py scheduler.py prefix_cache.py sampling.py
│   │               speculative.py constrained.py quantize.py server.py
│   ├── ingestion/  parse/ ocr.py tables.py vlm_caption.py chunking.py
│   │               contextualize.py pipeline.py
│   ├── retrieval/  bm25.py splade.py dense.py colbert.py colqwen.py
│   │               fusion.py rerank.py index/{pgvector,qdrant}.py
│   ├── agent/      state.py graph.py tools/ grader.py rewriter.py router.py
│   │               guardrails.py memory.py mcp_server.py
│   ├── eval/       datasets/ generate_golden.py retrieval.py generation.py
│   │               judge_calibration.py system.py stats.py report.py
│   ├── api/  cache/  obs/
│
├── frontend/
├── deploy/         # compose, HF Space Dockerfile, cloudflared config
├── tests/
└── docs/
    ├── architecture.md
    ├── model_card.md
    ├── compute_log.md         # every GPU-hour spent, and on what
    ├── decisions/             # ADRs
    └── benchmarks.md          # THE deliverable
```

---

## 3. The free-compute stack — and how it changes the engineering

### 3.1 What you actually get

| Resource | Free allowance | Used for |
|---|---|---|
| **Kaggle Notebooks** | 2× **T4 16GB** (or 1× P100), **30 GPU-hours/week**, 12h max session | the workhorse: all pretraining, distillation, teacher generation, ColQwen indexing, GPU eval sweeps |
| **Google Colab (free)** | 1× T4, variable quota, aggressive disconnects | debugging, overflow, backup |
| **Your laptop** | CPU + Apple MPS / integrated GPU | tokenizer, data prep, all of Phases 6–12, Ollama generator, CPU inference demos |
| **Hugging Face Hub** | unlimited public repos | checkpoints, tokenized shards, datasets, model card |
| **Hugging Face Spaces** | 2 vCPU / 16 GB RAM, Docker SDK, free CPU tier | the public live demo URL |
| **GitHub Actions** | unlimited minutes on **public** repos | CI + nightly eval + eval gate |
| **Docker Compose** | your machine | Postgres/pgvector, Redis, Phoenix, Prometheus, Grafana |
| **Cash cost** | **$0** | |

Make the repo **public from day one** — that's what makes Actions free and unlimited, and it's the portfolio anyway.

### 3.2 T4 gotchas that change your code (this is the important part)

The free GPU is Turing (SM 7.5), not Ampere. Four consequences you must design around:

1. **No bf16.** Turing has no bfloat16 tensor cores. You must use **fp16 autocast + `torch.amp.GradScaler`**, or fp32. fp16 has ~5 exponent bits vs bf16's 8, so overflow is real. This makes **QK-norm, z-loss, fp32 loss computation, and fp32 norm reductions load-bearing, not optional decorations.** Log the GradScaler scale factor every step — repeated scale halving means you're overflowing and something upstream is wrong.
2. **FlashAttention-2 requires SM 8.0+.** It will not build or run. `F.scaled_dot_product_attention` falls back to the **memory-efficient (xformers-style) backend**, which still gives you the linear-memory win over naive attention — which is the point you're benchmarking anyway. Assert which backend fired:
   ```python
   from torch.nn.attention import sdpa_kernel, SDPBackend
   with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
       out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
   ```
   Say this explicitly in the README. "I benchmarked memory-efficient attention because FlashAttention-2 needs Ampere and my budget was a free T4" is a *better* answer than a FA2 number you didn't produce.
3. **12-hour session cap, hard kill.** Checkpointing is not a nice-to-have, it's the architecture. Save every ~15 minutes to `/kaggle/working`, push to HF Hub every hour, and make resume **bit-exact** — model + optimizer + GradScaler + dataloader shard position + RNG state. You built the resumable loader in Phase 3 precisely for this.
4. **30 GPU-h/week quota.** Compute is rationed, not billed. That forces good discipline: **run every ablation at a 12M proxy config first** (`d_model=320, n_layers=6, 300M tokens, ~1 hour`), and only confirm the winner at 31M. If you implement muP, this transfer is principled rather than hopeful.

Also: 2× T4 means `torchrun --nproc_per_node=2` DDP works. Kaggle gives ~20 GB working disk — tokenized shards for 1.5B tokens at `uint16` are ~3 GB, fine. `torch.compile` with `max-autotune` can burn 5–10 minutes of warmup per session; use `mode="default"` on short runs and cache the inductor artifacts to HF Hub for long ones.

### 3.3 Every model you'll use, and where it runs

| Role | Model (open weights) | Runs on |
|---|---|---|
| SFT / KD data teacher | **Qwen2.5-3B-Instruct** or Qwen3-4B-Instruct, fp16 | Kaggle T4 + vLLM batch generation |
| Same-tokenizer KD teacher *(optional arm)* | **LocalMind-100M** — your own bigger sibling | Kaggle T4 |
| RAG answer generator | **Qwen3-4B-Instruct Q4_K_M** via Ollama / llama.cpp | laptop (4–30 tok/s depending on machine) |
| LLM judge | Qwen3-4B / Qwen2.5-7B-Instruct Q4 | laptop for spot checks, Kaggle for full sweeps |
| Embedder | bge-m3, gte-modernbert-base, or Qwen3-Embedding-0.6B — **ONNX int8 via fastembed** | laptop CPU |
| Reranker | bge-reranker-v2-m3 (or `-base` for CPU speed) | laptop CPU |
| Visual doc retrieval | **ColQwen2-v1.0** (~2B) | Kaggle T4 to index; vectors stored, queried on laptop |
| Figure/chart captioning | Qwen2.5-VL-3B or SmolVLM-500M | Kaggle batch job |
| Sparse expansion | SPLADE-v3 / naver efficient-splade | laptop CPU |

**The cross-tokenizer distillation problem — flag this, it's a great interview point.** True logit-level KD needs the teacher and student to share a vocabulary. Qwen's vocab is ~151k; yours is 16k. They don't align, so you cannot naively KL-divergence their logits. Your options, in order of practicality:

- **Sequence-level KD (primary):** train on the teacher's sampled outputs as hard labels. Tokenizer-agnostic, works fine, and is what most distillation in the wild actually does.
- **On-policy correction:** sample from the *student*, have the teacher score/rewrite, train on the corrected version. Fixes exposure bias without needing aligned logits.
- **Same-tokenizer logit KD (optional arm):** train `LocalMind-100M` with your own tokenizer, then do full top-K logit KD 100M → 31M. Clean, principled, costs one extra pretrain run (~10 GPU-h).

Benchmark the arms against each other and report which won. Explaining *why* you couldn't just do logit KD from Qwen is a better answer than most candidates can give about distillation at all.

### 3.4 Docker RAM budget (your laptop is the cluster)

Use **compose profiles** so you're never running everything at once:

| Profile | Services | ~RAM |
|---|---|---|
| `core` | postgres+pgvector, redis, api | 1.5 GB |
| `obs` | + Phoenix (single container, OTel-native), prometheus, grafana | +1.2 GB |
| `full` | + Qdrant, OpenSearch | +2.5 GB |

Notes: **Phoenix over Langfuse v3** as the default tracer — Langfuse v3 self-hosted needs ClickHouse + MinIO + Redis + Postgres, which is ~4 GB before you've traced anything. Phoenix is one container and speaks OTel natively. Keep Langfuse v2 as an optional profile if you want its prompt-management views. **Skip OpenSearch entirely** unless you specifically want the comparison — Postgres FTS (`tsvector` + GIN) or the `bm25s` Python library covers the sparse arm at a fraction of the footprint.

Experiment tracking: **MLflow in Docker** or `wandb offline` + sync later. Both free; MLflow keeps you fully local. TensorBoard is fine for loss curves alone.

### 3.5 Toolchain

| Concern | Use | Not |
|---|---|---|
| Packaging | **uv** | pip, poetry, conda |
| Lint/format | **ruff** | black + isort + flake8 |
| Types | **pyright** (strict on `localmind/`) | untyped |
| Config | **pydantic-settings** or Hydra | argparse spaghetti |
| Testing | **pytest + hypothesis** | manual scripts |
| Tracking | **MLflow (Docker)** or wandb-offline | printing loss |
| Artifacts | **HF Hub** | files on a laptop that will die |
| Serving | FastAPI + pydantic v2, async | Flask |
| Local LLM | **Ollama** (dev UX) / **llama.cpp** (control + benchmarks) | paid APIs |
| Tracing | **OpenTelemetry GenAI semconv → Phoenix** | vendor SDK only |
| Vector | **pgvector 0.8 (+pgvectorscale)** | five databases |
| Sparse | Postgres FTS or `bm25s` | rank_bm25 (too slow), OpenSearch (too heavy) |
| CI | GitHub Actions, public repo | none |

**Optional high-signal track:** write the BPE inner loop in **Rust** with PyO3 bindings and benchmark against Python and `tiktoken`. HuggingFace's `tokenizers` is Rust for exactly this reason; a 20–50× speedup you measured is a concrete systems credential and costs nothing but time.

---

## 4. Phase 0 — Foundation (3 days)

**Build:** public repo, `uv` env, ruff/pyright/pytest green, Actions running on push, `docker-compose.yml` with profiles, MLflow up, a `justfile` (`just up`, `just test`, `just train`, `just eval`), and a Kaggle launcher notebook that clones the repo and runs `just train-pretrain --config configs/train/pretrain.yaml`.

**Definition of done:** `git clone && just up && just test` works on a clean machine, and the Kaggle notebook trains a 12M model on TinyStories for 5 minutes without editing anything. Keep both true forever.

**Trap:** don't spend a week on infra aesthetics. Three days, hard stop.

---

## 5. Phase 1 — Tokenizer (1 week) — laptop only, no GPU

### Build

1. **Byte-level BPE trainer.** Operate on UTF-8 bytes so there is no UNK token, ever. Base vocab = 256 bytes.
2. **Regex pre-tokenization** with a GPT-4/`cl100k`-style pattern: contractions, letter runs, number runs capped at 3 digits, punctuation runs, whitespace handling. Splitting digits into ≤3-digit groups measurably improves arithmetic.
3. **Merge loop:** count adjacent pairs, merge the most frequent, repeat. Naive is fine at your corpus size — then implement an incremental counter and report the speedup.
4. **Special tokens as reserved IDs:** `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|user|>`, `<|assistant|>`, `<|tool_call|>`, `<|tool_result|>`, plus 16 unused reserved slots so you can add capabilities later without resizing embeddings.
5. **Chat template** rendered by the tokenizer, not by string concatenation in the training script.
6. **Round-trip property test with hypothesis:** `decode(encode(s)) == s` for arbitrary Unicode, emoji, CJK, control chars, malformed sequences.

### Vocab size — do the math, don't guess

At `d_model=512`, a 16,384 vocab costs 8.4M embedding params. A 50,257 vocab costs 25.7M — meaning **83% of a 31M model is a lookup table.** Weight-tie embedding and LM head, and pick vocab from the ablation.

**Ablation (all on the 12M proxy, ~4 GPU-h total):** tokenizers at vocab ∈ {4k, 8k, 16k, 32k}. For each, measure bytes/token on held-out text, then train the proxy for a fixed *token* budget and a fixed *FLOP* budget. Plot final val loss and **bits-per-byte** — BPB is the only tokenizer-invariant loss metric, so use it whenever comparing across tokenizers.

### Benchmarks to produce

| Tokenizer | Vocab | Bytes/token ↑ | Fertility ↓ | Encode MB/s ↑ | Proxy val BPB ↓ |
|---|---|---|---|---|---|
| Yours (Python) | 16k | | | | |
| Yours (Rust, optional) | 16k | | | | |
| tiktoken cl100k | 100k | | | | |
| GPT-2 | 50k | | | | |
| SentencePiece unigram | 16k | | | | |

**DoD:** round-trip fuzz passes on 1M random Unicode strings; vocab ablation plotted; your encoder within 5× of tiktoken throughput (or the Rust version beating it).

**Traps:** BPE merges are order-dependent — store the *ranked* merge list, not a set; training the tokenizer on data overlapping your eval set; not escaping special-token strings appearing in user text.

---

## 6. Phase 2 — The model (1.5 weeks)

### Reference config: `LocalMind-31M`

```yaml
vocab_size:      16384
d_model:         512
n_layers:        8
n_heads:         8
n_kv_heads:      2        # GQA, 4:1
head_dim:        64
ffn_hidden:      1408     # ~8/3 * d_model, rounded to multiple of 128
max_seq_len:     1024     # extend to 2048 in a short annealing phase
rope_theta:      10000.0
norm:            rmsnorm  # pre-norm
qk_norm:         true     # load-bearing on fp16 hardware
activation:      swiglu
bias:            false
tie_embeddings:  true
z_loss:          1.0e-4
attn_dropout:    0.0      # you are data-limited, not overfitting
```

Companion configs: `12m_proxy` (d=320, L=6, heads=5, kv=1) for ablations; `100m_teacher` (d=768, L=12, heads=12, kv=4) as the optional same-tokenizer KD teacher.

**Parameter accounting (do by hand, then assert in a test):**

| Component | Params |
|---|---|
| Embedding (tied with LM head) | 8,388,608 |
| Attention × 8 (Wq 262k, Wk 66k, Wv 66k, Wo 262k) | 5,242,880 |
| SwiGLU × 8 (gate + up + down) | 17,301,504 |
| Norms | ~33,000 |
| **Total** | **~30.9M** (22.5M non-embedding) |

### The pieces, in build order

**RMSNorm** — no mean subtraction, no bias, one learnable scale.

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dt) * self.w
```
The fp32 cast is not optional on fp16 hardware.

**RoPE** — precompute `cos`/`sin` once, apply to Q and K only (never V). Rotate-half form. Test: attention score between positions `i` and `j` depends only on `i-j`.

**Attention with GQA** — three interchangeable backends behind one interface:
1. `naive` — explicit `softmax(QKᵀ/√d + mask)V`, for teaching and as numerical reference.
2. `sdpa_math` — SDPA math backend.
3. `sdpa_efficient` — SDPA memory-efficient backend (**this is your "flash" on a T4**).

Assert all three agree to 1e-3 in fp32. Benchmark at seq ∈ {512, 1024, 2048, 4096}: latency and peak memory. The naive-vs-efficient memory curve (quadratic vs linear) is the single best plot in the repo.

**QK-norm** — RMSNorm on Q and K per-head before the dot product. Gemma 2/3 and Olmo 2 use it; on fp16 it is what stops attention-logit blowup from killing your run at hour 9 of a 12-hour session.

**SwiGLU** — `down(silu(gate(x)) * up(x))`. Three matrices, so hidden is 8/3·d rather than 4·d to keep params matched.

**Init** — normal(0, 0.02), with attention and FFN output projections scaled by `1/√(2·n_layers)`. Optional stretch: **muP**, so LR tuned on the 12M proxy transfers to 31M and 100M without re-tuning — which under a 30 GPU-h/week quota is not an academic nicety, it's how you afford the ablations.

**Stretch (pick one, not all):** MoE (8 fine-grained experts, top-2, 1 shared, aux-loss-free balancing) benchmarked active-params-matched against dense; or **MLA** (DeepSeek's latent KV compression) benchmarked against GQA on cache bytes/token; or sliding-window attention on alternating layers.

### Benchmarks to produce

- **KV cache bytes/token:** MHA 16 KB vs GQA(8q/2kv) **4 KB** vs MLA. At 2048 context that's 32 MB → 8 MB per sequence — 4× more concurrent users on the same VRAM, and the difference between your model fitting in a Docker container's memory limit or not. Quote this number in interviews.
- Attention backend latency/memory vs sequence length, 3 seeds, error bars.
- Forward/backward TFLOP/s and MFU at target batch size.

**DoD:** `assert count_params(cfg) == 30_932_992`; three backends numerically agree; a 5M-param version overfits 100 sequences to near-zero loss in <2 minutes on CPU (the single best correctness test for a from-scratch transformer, and it needs no GPU at all).

**Traps:** applying RoPE to V; missing causal mask in the naive path (loss looks suspiciously good); `expand` instead of `repeat_interleave` for KV heads where the kernel needs contiguity; sharing a RoPE cache across different `max_seq_len`.

---

## 7. Phase 3 — Data (1 week) — laptop, streaming

Do not scrape. Use permissively licensed corpora and record every license in `docs/model_card.md`.

### Mixture (target 1.5B tokens; 3B is the stretch)

| Source | Share | License | Purpose |
|---|---|---|---|
| **FineWeb-Edu** (`sample-10BT`) | 50% | ODC-By | general English quality |
| **Cosmopedia v2** | 15% | Apache-2.0 | synthetic textbook density |
| **The Stack v2 / StarCoder2** (Python, Rust, SQL, YAML) | 20% | permissive-only filter | code + structure |
| **Your domain corpus** (the docs you'll RAG over: framework docs, RFCs, permissively-licensed papers) | 10% | check each | domain alignment |
| **TinyStories** | 5% | CDLA | early curriculum |

Start with **TinyStories alone** for the first two days of debugging — a 12M model produces coherent English within minutes, which proves your loop before you spend quota.

**Stream, don't download.** `load_dataset(..., streaming=True)` then tokenize to shards. Full FineWeb-Edu 10BT is ~28 GB on disk; you only need 750M tokens of it. Stream → tokenize → write `uint16` shards → push shards to HF Hub. Then Kaggle sessions pull only the shards they need. Never store the raw corpus on a laptop.

### Pipeline

```
stream → license filter → language ID → quality heuristics → MinHash-LSH near-dedup
       → PII scrub → eval-set decontamination (13-gram) → tokenize
       → pack to seq_len with doc-boundary attention masking → uint16 memmap shards → HF Hub
```

- **Deduplication beats filtering.** Document-level near-dedup gives the biggest val-loss improvement per unit of effort. MinHash-LSH, 5-grams, threshold 0.8. Report the dedup ratio.
- **Decontamination is not optional** if you want your numbers believed. 13-gram overlap against the golden eval set; log how much you removed.
- **Sequence packing:** concatenate documents to fill `seq_len` with zero padding waste, but pass document boundaries so attention doesn't cross them (block-diagonal mask). Ablate it — naive packing that bleeds across documents costs real quality.
- **Memory-mapped `uint16` shards** (vocab < 65536). One `.bin` + one `.idx` per shard, ~256 MB each. Dataloader mmaps and slices: no Python-object overhead, no shuffle buffer, fully resumable.

**DoD:** `prepare.py` is deterministic given a seed and emits a manifest with a content hash per shard; a test proves that resuming from step N yields the identical next batch. **This test is what makes 12-hour session caps survivable.**

---

## 8. Phase 4 — Pretraining (2 weeks, ~25 GPU-hours)

### The loop, explicitly

Write it yourself. No `Trainer`. ~150 lines, and you must know every one.

```python
scaler = torch.amp.GradScaler("cuda")            # fp16 on T4, NOT bf16
for step in range(start_step, max_steps):
    lr = wsd_schedule(step); set_lr(opt, lr)
    opt.zero_grad(set_to_none=True)
    for _ in range(grad_accum):
        x, y, doc_mask = next(loader)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x, doc_mask)
        loss = fused_cross_entropy(logits.float(), y) / grad_accum   # CE in fp32
        scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    if step % ckpt_every == 0: save_and_maybe_push(step)             # session-cap insurance
```

### Hyperparameters (starting point)

```
tokens/step (global batch):  ~262,144   # via grad accumulation across 2×T4
peak_lr:                     3e-3       # small models take large LRs
min_lr:                      peak/10
schedule:                    WSD — 2% warmup, 80% stable, 18% linear-to-zero decay
optimizer:                   AdamW β=(0.9, 0.95), wd=0.1 (exclude norms/bias/embeddings)
grad_clip:                   1.0
precision:                   fp16 autocast + GradScaler, fp32 master weights, fp32 loss
z_loss:                      1e-4
torch.compile:               mode="default"     # max-autotune wastes session time
checkpoint:                  every 15 min local, every 60 min → HF Hub
```

**Why WSD, and why it matters more here than anywhere:** with cosine you must fix the total step count in advance and only the final checkpoint is usable. With Warmup-Stable-Decay you can branch a short decay from *any* point in the stable phase and get a deployable model. When your sessions are capped at 12 hours and your quota is 30 h/week, that property is the difference between "I have a model" and "the run died at 90% and I have nothing." It also gives you a free scaling-law study: decay at 250M, 500M, 1B, 1.5B tokens and plot loss vs compute. **Run this — it's a one-page result that looks like real research and costs ~2 extra GPU-hours.**

### Optimizer study (high-value, run on the 12M proxy: ~3 GPU-h)

1. AdamW (baseline)
2. **Muon** on all 2D hidden matrices + AdamW on embeddings/LM-head/norms (the Kimi K2 recipe)
3. AdamW at 2× LR (control for "Muon just likes bigger steps")

Report tokens-to-target-loss *and* wall-clock-to-target-loss. They can disagree, and saying so shows you understand the difference between sample efficiency and throughput.

### Instrumentation (the systems-engineering half)

Log every step: loss, val loss, **bits-per-byte**, grad norm, per-layer weight norm, per-layer activation RMS, LR, **MFU**, tokens/sec, GPU memory, **GradScaler scale**, data shard index.

```python
def mfu(params, n_layers, d_model, ctx, tokens_per_sec, peak_flops):
    flops_per_token = 6 * params + 12 * n_layers * d_model * ctx
    return flops_per_token * tokens_per_sec / peak_flops
# T4 fp16 tensor-core peak ≈ 65e12
```

**Expect 10–15% MFU on your first T4 run. Treat getting to 25%+ as an explicit sub-project.** Levers: larger micro-batch, `torch.compile`, **fused cross-entropy** (Liger / cut-cross-entropy, which avoids materializing the `[B, T, V]` logit tensor — a large VRAM win on 16 GB), avoiding host-device syncs in the logging path (don't call `.item()` every step), and eliminating dataloader stalls. The before/after MFU number with profiler traces justifying each fix is one of the strongest things you can bring to an ML-systems interview — and it costs zero dollars, only attention.

### Compute budget — the real numbers

`FLOPs ≈ 6ND = 6 × 3.1e7 × 1.5e9 ≈ 2.8e17`

| Run | Tokens | 2×T4 wall clock | Notes |
|---|---|---|---|
| TinyStories smoke test | 100M | ~30 min | proves the loop |
| 12M proxy ablations (×8) | 300M each | ~1 h each | vocab, packing, optimizer, LR |
| **Main pretrain, 31M** | **1.5B** | **~7 h** | one session, ~48 tokens/param |
| Context extension anneal | 75M @ 2048 | ~1 h | |
| Stretch: 3B tokens | 3B | ~14 h | two sessions, resume required |

**Total project GPU estimate:** pretrain + ablations ~35 h, post-training ~10 h, teacher generation ~4 h, ColQwen indexing ~2 h, eval sweeps ~8 h ≈ **~60 GPU-hours ≈ 2 weeks of Kaggle quota**, spread across 16 weeks. Comfortable. Track it in `docs/compute_log.md` — "the whole thing cost 60 free GPU-hours" is a line worth having.

Chinchilla-optimal for 31M is ~620M tokens (20 tok/param). You deliberately over-train to ~48–97 tok/param because the model will be *served*, and inference cost dominates training cost for anything deployed. Say exactly this in the README — it shows you know why Llama 3 trained 8B on 15T tokens.

### Context extension

After the stable phase, anneal the last ~5% of tokens at `seq_len=2048` with `rope_theta` scaled (NTK-aware or YaRN). Report perplexity vs position to prove the model *uses* the longer context rather than merely tolerating it.

**DoD:** loss curve with no unrecovered spikes; val BPB reported; scaling-law plot from WSD branch decays; MFU >25%; a checkpoint that resumes bit-exactly across a session boundary; model card written.

**Traps:** LR too low (small models want 1e-3 to 5e-3, not 3e-4); weight decay on embeddings and norms; validating on leaked data; `loss.item()` every step (forces a sync, tanks throughput); forgetting to shuffle shard order across epochs; letting a session die without a Hub push.

---

## 9. Phase 5 — Post-training (2 weeks, ~10 GPU-hours)

This is where 31M becomes useful. Four stages, each with an eval.

### 5a. SFT

Build ~30–50k instruction pairs targeting the **three jobs the model actually does in production**:

| Task | Input | Output | Why 31M can do it |
|---|---|---|---|
| **Domain router** | user query | `in_domain` / `out_of_domain` / `needs_web` | 3-way classification |
| **Query rewriter** | query + conversation | standalone search query | short templated transformation |
| **Relevance grader** | query + chunk | `relevant` / `irrelevant` + score | binary judgement on short text |

Generate with **Qwen2.5-3B-Instruct on a Kaggle T4 via vLLM** (batch generation: 50k short outputs in ~1–2 hours, free). Then **hand-verify a stratified sample** and report agreement with the teacher. Never ship synthetic data you haven't spot-checked.

Implementation: apply the chat template; **mask loss on prompt tokens** (train only on assistant spans); pack with boundary masking; cosine decay from 10% of pretrain LR.

### 5b. Distillation (the biggest quality lever)

Three arms, benchmarked against each other:

1. **Sequence-level KD** *(primary)* — train on teacher outputs as hard labels. Tokenizer-agnostic, so it works across Qwen's 151k vocab and your 16k.
2. **On-policy correction** — sample from the *student*, have the teacher rewrite/score, train on the correction. Fixes exposure bias, which is the actual reason Gemma 3 1B and Llama 3.2 1B work as well as they do.
3. **Same-tokenizer logit KD** *(optional, ~10 extra GPU-h)* — pretrain `LocalMind-100M` with your tokenizer, then distill 100M → 31M with true top-K logit KL:
   `L = α·KL(student ‖ teacher_topK) + (1-α)·CE(student, hard_labels)`, K=64, α=0.7.

Storage note: top-64 logits over 20k sequences × 256 tokens ≈ 1.2 GB — dump to HF Hub, don't try to keep it in a Kaggle session.

### 5c. DPO

~5k preference pairs on the rewriter (good rewrite vs bad). β=0.1. Track reward margin and KL from the reference policy; runaway KL means your β is wrong.

### 5d. GRPO with verifiable rewards

Pick a task with a programmatic reward — the grader must emit strict JSON *and* be correct against ground truth. Reward = format ✓ + correctness ✓. Group size 8, no value network. This is the post-R1 recipe, and it works at small scale precisely *because* the reward is verifiable rather than learned. It's also nearly free: no reward model to train, no judge calls.

### 5e. The comparison matrix (the deliverable)

| Method | Params updated | Router acc | Grader F1 | Rewrite win-rate | p50 latency | Hardware |
|---|---|---|---|---|---|---|
| Qwen3-4B, zero-shot prompt | 0 | | | | | laptop GPU/CPU |
| Qwen3-4B, few-shot prompt | 0 | | | | | laptop |
| Qwen3-4B + RAG context | 0 | | | | | laptop |
| Qwen3-4B + LoRA (r=16) | ~0.1% | | | | | T4, ~2 GPU-h |
| LocalMind-31M, SFT only | 100% | | | | | **laptop CPU** |
| **LocalMind-31M, SFT + KD + GRPO** | 100% | | | | | **laptop CPU** |

Then answer, with your own numbers: *when should I prompt, when should I retrieve, when should I fine-tune, and when should I distill a small specialist?* That answer, backed by a table you produced, is worth more than every buzzword on a resume combined.

**DoD:** the 31M model beats the 4B on latency by >20× at ≥95% of its accuracy on at least one of the three tasks, **running on CPU with no GPU at all**. If it doesn't, that's a publishable negative result — report it honestly and say what would close it.

---

## 10. Phase 6 — Inference engine (2 weeks) — mostly laptop

Your model is 31M params. At int8 that's ~31 MB. **The entire inference chapter runs on your laptop CPU**, which is not a compromise — it's the thesis. Only the vLLM comparison baseline needs the free T4.

### Build in order, benchmarking each step

1. **Naive generation** — recompute everything every token. Measure tok/s. This is your denominator.
2. **Contiguous KV cache** — preallocate `[B, n_kv_heads, max_len, head_dim]`. Expect ~10–20× at 512 tokens. Explain *why*: prefill is compute-bound, decode is memory-bandwidth-bound.
3. **Paged KV cache** — 16-token blocks + per-sequence block table, vLLM-style. Report **memory fragmentation before vs after** and max concurrent sequences at fixed memory. The single most impressive thing in this section.
4. **Continuous batching** — iteration-level scheduler admitting new requests and evicting finished ones every step, instead of static batching. Measure throughput under Poisson-arrival load.
5. **Chunked prefill** — cap prefill tokens per iteration so long prompts don't stall other users' decoding. Report p99 TPOT improvement.
6. **Prefix caching** — radix tree over the KV cache. With RAG, system prompts and retrieved chunks repeat constantly; report hit rate and TTFT reduction.
7. **Speculative decoding** — start with n-gram / prompt-lookup (zero extra model, great for RAG where output copies from context), then a real draft model. Report **acceptance rate** and net speedup; below ~0.6 acceptance there's usually no win.
8. **Constrained decoding** — FSM/grammar-masked logits so tool-call JSON is valid by construction. Report invalid-JSON rate before vs after (should hit exactly 0) and the tokens/s cost.
9. **Quantization** — int8 and int4 weight-only. Report size, tok/s, and val BPB delta, then plot the quality/latency frontier. Also export to **GGUF** so your model runs in llama.cpp — a 31 MB GGUF someone can download and run is a shockingly good demo artifact.

### API surface

Serve **OpenAI-compatible** endpoints — `/v1/chat/completions` (SSE streaming), `/v1/completions`, `/v1/embeddings`, plus `/health` and `/metrics`. Then any client, eval harness, or the `openai` SDK works against your model with a base-URL change. Worth far more than a bespoke `/generate`.

### Metrics that matter (learn this vocabulary)

**TTFT** (prefill-dominated) · **TPOT/ITL** (decode-dominated) · **throughput** (aggregate tok/s) · **goodput** (requests/sec meeting an SLO, e.g. TTFT<500 ms and TPOT<50 ms — the metric serving teams actually optimize) · p50/p95/p99 on all of them, under load from a generator (locust), never single-request timings.

### Honest baseline

Serve the same weights through **vLLM on a free T4** and put its numbers beside yours. You will lose. "My engine reaches 62% of vLLM's throughput; the gap is kernel fusion and CUDA graph capture, here's the profiler trace" is a far stronger signal than an unbenchmarked claim of having built an inference engine.

**DoD:** `just bench-inference` reproduces the KV-cache and paged-attention benchmarks on CPU; the server passes a conformance test driven by the real `openai` Python client; a GGUF export is published on HF Hub.

---

## 11. Phase 7 — Multimodal ingestion (1 week)

**Parsers (all free, all local):** Docling (structure-preserving, reading order + table structure), PyMuPDF4LLM (fast text), Marker (accurate, slower), Surya or PaddleOCR for scans. Route by document type; log which parser handled what.

**Tables:** extract to markdown *and* to a normalized Postgres table so the SQL tool can query them. Store provenance (doc, page, bbox).

**Figures/charts:** caption with SmolVLM-500M on CPU, or Qwen2.5-VL-3B as a Kaggle batch job. Index the caption text; keep the crop for citation display.

**The differentiator — visual retrieval:** build a second arm with **ColQwen2-v1.0**: render each page to an image, embed into multi-vector patch embeddings, retrieve by late interaction. No OCR, no chunking, no layout parsing. Run the *indexing* as a Kaggle batch job (~2 GPU-h for a few thousand pages), store the vectors, then query on your laptop. Benchmark head-to-head against the OCR pipeline. On dense-layout documents — financial reports, scanned forms, slide decks — the visual arm usually wins, and it's a result most engineers haven't seen.

Storage warning: multi-vector page embeddings are large (~1030 patches × 128 dims per page). Use binary or int8 quantization on the patch vectors and report the recall you retained — that constraint is itself a good engineering story.

**Chunking ablation** (run all five, measure with the Phase 10 harness):

| Strategy | Description |
|---|---|
| Fixed 512 | baseline |
| Recursive by separator | structure-aware splits |
| Semantic | split on embedding-similarity drops |
| **Late chunking** | embed the whole document, pool per chunk — chunk vectors carry document context |
| **Contextual retrieval** | prepend an LLM-generated 1–2 sentence situating context before embedding |

Contextual retrieval is the highest-impact one, and the per-chunk LLM call is a perfect job for **your own model** — measure the cost and latency difference against using Qwen3-4B for the same job. On a paid API this technique is expensive; running it on a model you trained yourself, it's free. Say that.

Also ablate chunk size {256, 512, 768, 1024} × overlap {0, 10%, 20%}. Report a heatmap of nDCG@10, not a single winning number.

---

## 12. Phase 8 — Retrieval (1.5 weeks) — laptop + Docker

### Four arms

| Arm | Implementation | Strength |
|---|---|---|
| **BM25** | your own impl for understanding + `bm25s` or Postgres FTS for production | exact terms, rare tokens, IDs, code symbols |
| **SPLADE** | learned sparse; term expansion into the inverted index | vocabulary mismatch, keeps sparse-index speed |
| **Dense** | bge-m3 / gte-modernbert, **Matryoshka-truncated** to 256 dims for stage 1, full for stage 2; ONNX int8 via fastembed | semantics, paraphrase |
| **ColBERT** | late interaction, token-level max-sim | precision on long/multi-aspect queries |

Implement BM25 yourself once — the `k1` tf-saturation and `b` length-normalization terms are worth understanding — then use the fast library for real workloads and say so in the ADR.

### Fusion

Baseline **RRF**: `score(d) = Σ_arms 1/(k + rank_arm(d))`, k=60. Rank-based, so no score normalization needed, and remarkably hard to beat. Then try min-max-normalized weighted fusion tuned on a dev set. Report whether the tuning actually beat RRF — often it doesn't, and saying so shows judgement.

### Reranking

Cross-encoder (bge-reranker-v2-m3, or `-base` if CPU latency hurts) over top-50 → top-5. Then a **listwise LLM reranker** with your local Qwen3-4B for comparison. Report the quality gain *and* the added p95 latency — reranking is where most RAG systems silently blow their latency budget, and on CPU that cost is brutally visible, which makes your measurement more honest than a GPU-hosted one.

### Index engineering

- pgvector **HNSW** (`m=16, ef_construction=64`); tune `ef_search` and plot the **recall-vs-latency curve.** Every vector-DB interview question is really about this curve.
- **Binary quantization + rescoring:** store 1-bit vectors (32× smaller), retrieve top-200 by Hamming distance, rescore with full precision. Report memory saved and recall retained. This is how large-scale vector search is actually run — and on a 16 GB laptop it's not a flex, it's a necessity.
- **Filtered ANN:** demonstrate why post-filtering breaks recall when filters are selective, and what pre-filtering costs. The most common production vector-DB failure, and almost no candidate can explain it.
- Add pgvectorscale (DiskANN) and Qdrant as comparison arms — only after pgvector works. One store first.

### Benchmark table

| Config | Recall@5 | Recall@20 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| BM25 | | | | | | |
| Dense (256d MRL) | | | | | | |
| Dense (full) | | | | | | |
| SPLADE | | | | | | |
| ColBERT | | | | | | |
| BM25+Dense (RRF) | | | | | | |
| 4-arm RRF | | | | | | |
| 4-arm RRF + cross-encoder | | | | | | |
| + contextual chunks | | | | | | |

Every cell: mean over 3 seeds with a bootstrap 95% CI. State the hardware — "all latencies measured on 8-core laptop CPU" is a fine and honest footnote.

---

## 13. Phase 9 — Agent (1.5 weeks)

### Architecture

An explicit typed state machine: `route → retrieve → grade → (rewrite → retrieve)* → generate → verify → respond`. LangGraph is fine, but a plain Python state machine over a pydantic state object is ~200 lines, fully debuggable, and easier to test. Choose deliberately and write an ADR.

**Where your model runs:** `route`, `rewrite` and `grade` call LocalMind-31M **on CPU**. `generate` calls Ollama. Log per-node latency so the README architecture diagram can be annotated with real p50s.

### Tools

`search_documents`, `search_web`, `query_database`, `calculate`, `retrieve_image`, `summarize_document`.

Expose all of them as an **MCP server** as well as calling them in-process. Then Claude Desktop, Cursor, or any MCP client can drive your retrieval stack — a live demo that takes 30 seconds and is far more memorable than a screenshot.

Every tool: typed pydantic schema, timeout, retry with exponential backoff, structured error returned *to the model* (not raised), idempotency where possible. `calculate` runs in a sandbox — never `eval()`. `search_web` uses a free provider (DuckDuckGo/SearXNG self-hosted in Docker) with caching, since you have no API budget.

### Self-correction

```
retrieve → grade each chunk
    ├── ≥2 relevant  → generate
    ├── 0 relevant   → rewrite query → retrieve (max 2 attempts)
    └── still 0      → web search → grade → generate or refuse
generate → verify claims against sources
    └── unsupported claim → regenerate with stricter prompt, or refuse
```

Hard caps on iterations, wall-clock, and token budget. Every loop must terminate.

### Guardrails — including the one everyone skips

- **Indirect prompt injection.** You ingest untrusted PDFs and untrusted web pages, then feed them to a tool-calling agent. That is a live attack path. Defenses: delimit retrieved content and frame it explicitly as data, not instructions; run an injection classifier over retrieved chunks (**another job for your 31M model** — a cheap binary classifier is exactly its weight class); **never let retrieved text trigger a tool call directly** (tool selection reads only the user turn and the agent's own reasoning); allowlist tools per route. Build ~30 attack cases into the eval set and report your catch rate. **Almost no portfolio RAG project addresses this.**
- Input: PII detection, rate limits, out-of-domain refusal.
- Output: citation-required mode (refuse rather than answer uncited), leak checks.

**DoD:** ≥30 adversarial injection cases with a reported block rate; every tool has a timeout test; the agent provably terminates under a fuzzer.

---

## 14. Phase 10 — Evaluation harness (build at Phase 4, harden here)

**This is the most valuable artifact in the repo.** Build it early and keep it green.

### Golden dataset

150–300 questions, versioned and hash-pinned:

```
id, question, expected_answer, expected_doc_ids, expected_chunk_ids,
difficulty {easy|medium|hard}, category {factoid|multi-hop|aggregation|
  table|figure|out-of-domain|adversarial-injection}, requires_tools[]
```

Generate candidates with the local teacher over your corpus, then **hand-verify every one.** Include deliberate out-of-domain and unanswerable questions — a system that never refuses is broken, and refusal accuracy is a metric.

### Three layers

**Retrieval:** Recall@{1,5,10,20}, nDCG@10, MRR, MAP. Cheap, deterministic, no LLM in the loop. Runs on every PR in GitHub Actions on CPU in ~3 minutes.

**Generation:** faithfulness, answer relevance, **citation precision and recall**, refusal correctness. Use RAGAS/DeepEval pointed at your local Ollama endpoint, or write your own — writing your own means you can explain it.

**System:** TTFT, end-to-end p50/p95/p99, tokens in/out, tool success rate, iteration-count distribution. (Cost per query is $0 — instead report **CPU-seconds per query** and **peak RSS**, which are the metrics that actually constrain you and are more interesting anyway.)

### Judge calibration — mandatory here, not optional

Your judge is a 4B local model, not a frontier API. It is weaker, so you *must* prove it works:
- Hand-label 100 examples.
- Report judge-vs-human **Cohen's κ**. Below ~0.6 your judge is noise — say so in the README and fall back to deterministic metrics.
- Control position bias: randomize A/B order in pairwise comparisons, report the swap rate.
- Prefer pairwise comparison to absolute 1–5 scoring; small judges are far more reliable at ranking than rating.

"I measured my judge's agreement with human labels at κ=0.71 and reported which metrics I therefore trust" is a stronger answer than anything a frontier-API judge would have bought you.

### Statistics

3+ seeds for anything stochastic. Bootstrap 95% CIs on every reported metric. Paired tests (bootstrap or Wilcoxon) when comparing configs on the same questions. **Never report a single number without a CI.** A table of bare point estimates is the most common way portfolio projects reveal they aren't measuring anything real.

### CI gate (free on public repos)

```yaml
# .github/workflows/eval-gate.yml
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest        # CPU only, no GPU needed
    steps:
      - run: just eval-retrieval  # deterministic, ~3 min, no LLM calls
      - run: just eval-compare --baseline main --max-regression 0.02
      # fails the build if nDCG@10 drops more than 2%
```

Full generation evals run nightly on a schedule against a cached corpus, post results to the PR, and regenerate `docs/benchmarks.md`.

### The headline experiment

| System | Recall@10 | nDCG@10 | Faithfulness | Citation P/R | Refusal acc | p95 ms | CPU-s/query |
|---|---|---|---|---|---|---|---|
| 1. Naive RAG (fixed chunks, dense only) | | | | | | | |
| 2. Hybrid RAG (4-arm RRF) | | | | | | | |
| 3. Hybrid + cross-encoder rerank | | | | | | | |
| 4. + contextual chunking | | | | | | | |
| 5. Agentic (grade + rewrite + web fallback) | | | | | | | |
| 6. Agentic with **LocalMind-31M** as router/grader/rewriter | | | | | | | |
| 7. Agentic, all-4B control | | | | | | | |

Rows 6 vs 7 are the payoff: comparable quality, dramatically lower latency and CPU cost, because you built and distilled the small model yourself. That row is the whole project.

---

## 15. Phase 11 — Observability and caching (1 week)

### Observability

Instrument with **OpenTelemetry using the GenAI semantic conventions**, export to **Phoenix** (single container, OTel-native, free, open source) and Prometheus/Grafana. Vendor-neutral instrumentation is the correct 2026 choice; a vendor SDK locks you in and signals less. Langfuse v2 is a fine optional profile; skip Langfuse v3 self-hosted — ClickHouse + MinIO + Redis + Postgres will eat your laptop.

One trace per request with child spans: `route → embed → bm25 → dense → colbert → fuse → rerank → grade → generate → verify`. Each span carries duration, token counts, model name, cache hit/miss, retrieved doc IDs.

Dashboards: per-stage latency percentiles, tokens/day, retrieval-quality trend from the nightly eval, error and refusal rates, tool success rate. Alert on p95 latency and eval-metric regression.

### Caching

Three Redis layers, each with an honest metric:

| Layer | Key | Metric to report |
|---|---|---|
| Embedding cache | hash(text)+model | hit rate, embedding calls avoided |
| Exact response cache | hash(query+filters+config) | hit rate, latency saved |
| **Semantic cache** | nearest neighbour over query embeddings, threshold τ | hit rate **and false-hit rate** |

Semantic caching is where portfolio projects cheat. Sweep τ from 0.80 to 0.99 and plot hit rate against **false-hit rate** — cases where a semantically-close query got a wrong cached answer. Report the operating point you chose and why. That curve, and the willingness to show its downside, is the mark of someone who has run a cache in production. On CPU-bound inference the cache win is enormous, so this section has real teeth here.

KV prefix caching in your inference server (Phase 6) is a fourth layer; report its hit rate under real RAG traffic where system prompts repeat.

---

## 16. Phase 12 — Production, for free (1 week)

- **Docker**: multi-stage, non-root, pinned base digests, `.dockerignore`, <500 MB API image.
- **Compose with profiles** (`core` / `obs` / `full`) so the stack fits in 16 GB. `just up` and it works.
- **CI/CD**: lint → typecheck → test → build → **eval gate** → push image to **GHCR** (free for public repos). Tag with git SHA.
- **Live public demo**: a **Hugging Face Space** (Docker SDK, free CPU tier, 2 vCPU / 16 GB). Ship the frontend + API + LocalMind-31M int8 + a pre-built pgvector-free SQLite/FAISS index. Your model is 31 MB — it fits trivially, and the whole point is that it runs without a GPU. This gives you a public URL to put on a resume.
- **GPU-backed demo when needed**: run the full stack locally and expose it with a **Cloudflare Tunnel** (free) for the duration of an interview.
- **Ingestion orchestration**: Airflow is heavy for one pipeline and will not fit alongside everything else. Use **Prefect** or a plain scheduled job — and say "I chose the simpler tool because the DAG has 6 nodes." That's better engineering than adopting Airflow to have it on a resume. Write the ADR.
- **Load test**: locust, Poisson arrivals, report goodput under SLO on your actual laptop.
- **Runbook**: `docs/runbook.md` — how to roll back, how to reindex, what each alert means.

**Say this in the README, deliberately:** the entire system reproduces with one command on any machine, needs no cloud account, and has a live public demo. That's a stronger claim than "deployed to AWS," and a reviewer can verify it in five minutes instead of taking your word for it.

---

## 17. Timeline

16 weeks part-time (~15 h/week), ~60 free GPU-hours total. Gates are hard: if a phase runs 50% over, cut scope, don't extend.

| Week | Phase | Ships | GPU-h |
|---|---|---|---|
| 1 | 0 + 1 | Repo, CI, tokenizer + vocab ablation | 4 |
| 2–3 | 2 | Model, three attention backends, param test, CPU overfit test | 2 |
| 4 | 3 | Data pipeline, dedup, packing, shards on HF Hub | 1 |
| 5–6 | 4 | Pretrain 1.5B tokens, WSD scaling plot, Muon study, MFU work | 25 |
| 7–8 | 5 | SFT + KD + DPO + GRPO, method comparison matrix | 10 |
| 9–10 | 6 | Inference engine through paged KV + continuous batching + spec decoding + GGUF | 2 |
| 11 | 7 | Ingestion + ColQwen2 arm + chunking ablation | 3 |
| 12 | 8 | Four-arm retrieval + fusion + rerank + ANN curves | 2 |
| 13 | 9 + 10 | Agent, MCP tools, guardrails, full eval harness in CI | 4 |
| 14 | 11 | OTel + Phoenix, three-layer cache with false-hit curve | 1 |
| 15 | 12 | Docker, HF Space, load test, runbook | 0 |
| 16 | — | `docs/benchmarks.md`, README rewrite, demo video, blog post | 2 |

**Minimum defensible version (week 8 stop-line):** tokenizer + model + trained checkpoint + post-training comparison matrix + eval harness. Even with zero RAG, that beats 95% of "I built an agentic RAG chatbot" repos. Everything after week 8 is upside.

**Cut list, in order:** MoE/MLA variants → same-tokenizer 100M KD arm → SPLADE arm → Qdrant/OpenSearch comparisons → Prefect → frontend polish → speculative decoding → 3B-token stretch run.

**Never cut:** the eval harness, the benchmark tables, the statistical rigor, the README.

---

## 18. What makes this get you interviews

The repo is not the deliverable. **`docs/benchmarks.md` is the deliverable**, and the README's first screen should be results, not architecture.

Five artifacts to write:

1. **README with results above the fold** — three tables and two plots before any prose. Then the architecture diagram annotated with measured p50 latencies. Then the one-command setup.
2. **`docs/benchmarks.md`** — every ablation, every CI, every negative result.
3. **`docs/decisions/`** — ADRs for the non-obvious calls: why GQA 4:1, why WSD over cosine, why fp16+GradScaler instead of bf16, why sequence-level KD instead of logit KD, why RRF over tuned fusion, why Prefect over Airflow, why vocab 16k. Interviewers read these.
4. **`docs/model_card.md`** and **`docs/compute_log.md`** — data sources with licenses, training compute, evals, limitations, and every GPU-hour spent. Nobody writes these. They read as professional maturity.
5. **A 3-minute demo video** — MCP tools driving your retrieval stack from a real client, live, on a laptop with no GPU.

### Resume lines (pick two)

> Built a 31M-parameter decoder-only LM end-to-end in PyTorch — byte-level BPE tokenizer, RoPE/GQA/SwiGLU/RMSNorm architecture, custom fp16 training loop with WSD schedule and a Muon-vs-AdamW study at 27% MFU — trained entirely on free-tier T4s within a 30 GPU-hour/week budget, then distilled it into the routing, query-rewriting and relevance-grading layer of an agentic RAG system, cutting control-plane latency 24× at 97% of teacher accuracy **on CPU**.

> Built a from-scratch inference engine (paged KV cache, continuous batching, chunked prefill, prefix caching, speculative decoding, grammar-constrained tool calls) reaching 62% of vLLM throughput, plus a CI-gated evaluation harness over 240 human-verified questions that blocks merges on >2% nDCG@10 regression — full stack reproducible with one Docker command, zero cloud spend.

### Questions you'll be able to answer that most candidates can't

- What exactly happens between the user's string and the first logit?
- Why GQA over MHA, and what does it cost in quality? *(You have the KV-bytes-per-token table and the loss delta.)*
- Why is decoding memory-bandwidth-bound but prefill compute-bound?
- What's your MFU and why isn't it higher? *(You have profiler traces.)*
- Why couldn't you do logit-level distillation from Qwen? *(Vocabulary misalignment — and here's what you did instead.)*
- Why does fp16 need a GradScaler and bf16 doesn't?
- When is fine-tuning the wrong answer? *(You have the six-row matrix.)*
- Why does post-filtering break ANN recall? *(You have the curve.)*
- How do you stop a malicious PDF from making your agent call a tool? *(You have 30 adversarial cases and a block rate.)*

---

## 19. Reading list (in build order)

- **Tokenizer** — Karpathy, *Let's build the GPT Tokenizer*; the tiktoken repo; SentencePiece paper.
- **Model** — Llama 2/3 papers (GQA, RoPE); RoFormer; *GLU Variants Improve Transformer*; *Root Mean Square Layer Normalization*; Gemma 2/3 tech reports (QK-norm, sliding window); DeepSeek-V2 (MLA); Qwen3 tech report.
- **Training** — Chinchilla; MiniCPM (WSD); the Muon writeups and the Kimi K2 report; nanoGPT and modded-nanogpt; Tensor Programs V (muP); the FineWeb technical report — the best public writeup on data curation, and free.
- **Post-training** — DPO; DeepSeekMath (GRPO); DeepSeek-R1; *Distilling Step-by-Step*; the Gemma 3 distillation section; MiniLLM (small-model KD).
- **Inference** — vLLM/PagedAttention; Orca (continuous batching); SGLang/RadixAttention; Medusa and EAGLE; FlashAttention 2/3; XGrammar; llama.cpp's quantization docs.
- **Retrieval** — Robertson & Zaragoza (BM25); ColBERTv2; SPLADE; RRF (Cormack et al.); Matryoshka Representation Learning; ColPali; Anthropic's *Contextual Retrieval*; Jina's *Late Chunking*; HNSW; DiskANN.
- **Agents/eval** — Self-RAG; CRAG; RAGAS; *Judging LLM-as-a-Judge* (MT-Bench); the MCP specification; OWASP LLM Top 10.

Every one of these is free to read.

---

## 20. The four rules

1. **Nothing goes in the repo without a benchmark.** If you can't measure it, you can't defend it, and if you can't defend it, it's a liability in the interview.
2. **Report the negatives.** "Tuned fusion did not beat RRF." "Speculative decoding lost at batch>8." "My engine reaches 62% of vLLM." "My judge's κ was 0.58 so I don't trust that metric." Honest negatives are the strongest credibility signal you have.
3. **The small model must earn its keep.** It's not a demo of a transformer — it's the low-latency, CPU-resident control plane of a real system, with a table proving it's the right tool for that job.
4. **The constraint is a feature.** Trained on free T4s in 60 GPU-hours, runs on a laptop, reproduces in one Docker command, zero cloud spend. Don't apologize for it — lead with it. It proves you can engineer under real limits, which is the actual job.
