"""Phase 6 benchmark harness.

    uv run python -m localmind.inference.bench            # or: just bench-inference

Writes ``artifacts/benchmarks/inference.json`` in the CONVENTIONS.md schema
(``{"name","hardware","seeds","rows":[...],"ci":"bootstrap95"}``) and this phase's section
of ``docs/benchmarks.md`` to ``artifacts/benchmarks/sections/60-inference.md``, which
``localmind.eval.report`` composes into the deliverable. It used to append straight to the
deliverable, so the regeneration command that document names as its own generator deleted
this section on every run. Three seeds and a percentile bootstrap 95% CI on every stochastic
quantity; hardware is recorded in the artifact, because a tokens/s figure without a
machine attached to it is not a measurement.

Every section is one rung of the implementation.md section 10 ladder, and every rung is
reported as a **delta against the rung below it**. Negative results are kept.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from localmind.eval.system import os_release
from localmind.inference.constrained import (
    ConstrainedDecoder,
    generate_json,
    make_synthetic_vocab,
)
from localmind.inference.engine import CachedEngine, NaiveEngine, PagedEngine
from localmind.inference.kv_cache import (
    DEFAULT_BLOCK_SIZE,
    PagedKVCache,
    fragmentation_report,
    max_concurrent_sequences,
)
from localmind.inference.prefix_cache import RadixPrefixCache
from localmind.inference.quantize import (
    QuantLinear,
    bits_per_byte,
    logit_kl,
    quantize_dynamic_int8,
    quantize_model,
    write_gguf,
)
from localmind.inference.sampling import SamplingParams
from localmind.inference.scheduler import (
    ContinuousBatchingScheduler,
    SchedulerConfig,
    StaticBatchScheduler,
    make_requests,
    poisson_arrivals,
    run_load,
    summarise,
)
from localmind.inference.speculative import (
    DraftModelProposer,
    NgramProposer,
    speculative_generate,
)
from localmind.model import LocalMindTransformer, ModelConfig
from localmind.model.config import kv_cache_bytes_per_token

__all__ = ["bootstrap_ci", "main", "run_benchmarks", "summarise_samples"]

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
DEFAULT_CONFIG = "configs/model/12m_proxy.yaml"
INFERENCE_SECTION = "artifacts/benchmarks/sections/60-inference.md"
"""This phase's contributed section of `docs/benchmarks.md`. The `60-` prefix
orders it among the other producers; `localmind.eval.report` composes them in
filename order."""


# ---------------------------------------------------------------------------------
# Statistics (CONVENTIONS.md rule 5: never a bare number)
# ---------------------------------------------------------------------------------
def bootstrap_ci(
    samples: Sequence[float], n_boot: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    vals = [float(v) for v in samples if v == v]
    if not vals:
        return (float("nan"), float("nan"))
    if len(vals) == 1:
        return (vals[0], vals[0])
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def summarise_samples(samples: Sequence[float]) -> dict[str, float]:
    vals = [float(v) for v in samples if v == v]
    if not vals:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    lo, hi = bootstrap_ci(vals)
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "ci_low": lo,
        "ci_high": hi,
        "n": len(vals),
    }


def hardware_string() -> str:
    parts = [
        "CPU-only",
        platform.processor() or platform.machine(),
        f"{torch.get_num_threads()} torch thread(s)",
        f"{platform.system()} {os_release()}",
        f"python {sys.version.split()[0]}",
        f"torch {torch.__version__}",
    ]
    if torch.cuda.is_available():  # pragma: no cover - no GPU here
        parts.insert(0, f"CUDA {torch.cuda.get_device_name(0)}")
    return " | ".join(p for p in parts if p)


def _model(cfg: ModelConfig, seed: int) -> LocalMindTransformer:
    torch.manual_seed(seed)
    return LocalMindTransformer(cfg).eval()


def _prompt(cfg: ModelConfig, n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(cfg.vocab_size) for _ in range(n)]


# ---------------------------------------------------------------------------------
# 0. Thread scaling -- a real CPU-serving finding, and it sets the rest of the run
# ---------------------------------------------------------------------------------
_THREAD_PROBE = """
import json, sys, time, torch
torch.set_num_threads(int(sys.argv[1]))
from localmind.model import LocalMindTransformer, ModelConfig
cfg = ModelConfig.from_yaml(sys.argv[2])
seeds = [int(s) for s in sys.argv[4].split(",")]
iters = int(sys.argv[3])
out = []
for seed in seeds:
    torch.manual_seed(seed)
    m = LocalMindTransformer(cfg).eval()
    with torch.no_grad():
        past = m(torch.arange(64).unsqueeze(0) % cfg.vocab_size, use_cache=True).kv_caches
        one = torch.tensor([[7]])
        for _ in range(3):
            m(one, past_kvs=past, use_cache=True)
        t0 = time.perf_counter()
        for _ in range(iters):
            m(one, past_kvs=past, use_cache=True)
        out.append((time.perf_counter() - t0) / iters * 1e3)
print(json.dumps({"threads": torch.get_num_threads(), "ms": out}))
"""


def bench_thread_scaling(
    cfg: ModelConfig, seeds: Sequence[int], iters: int = 15, config_path: str = DEFAULT_CONFIG
) -> list[dict[str, Any]]:
    """Decode-step latency vs ``torch.set_num_threads``.

    Included because it is counter-intuitive and it matters for CPU serving: a single-token
    decode is a stack of tiny GEMVs, and the thread count that is right for prefill can be
    wrong for decode.

    Measured in a **fresh subprocess per thread count**. ``torch.set_num_threads`` is only
    reliably honoured before the intra-op thread pool is created; calling it repeatedly
    inside one process yields numbers that look like a result and are not one. Getting this
    wrong is easy and silent, which is exactly why it is worth doing properly.
    """
    import subprocess

    rows: list[dict[str, Any]] = []
    seed_arg = ",".join(str(s) for s in seeds)
    for n_threads in (1, 2, 4, 8):
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _THREAD_PROBE,
                    str(n_threads),
                    str(config_path),
                    str(iters),
                    seed_arg,
                ],
                capture_output=True,
                text=True,
                timeout=600,
                check=True,
                cwd=str(Path.cwd()),
            )
            data = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as exc:
            rows.append(
                {
                    "bench": "thread_scaling",
                    "threads": n_threads,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        samples = [float(v) for v in data["ms"]]
        rows.append(
            {
                "bench": "thread_scaling",
                "threads_requested": n_threads,
                "threads_effective": data["threads"],
                "decode_step_ms": summarise_samples(samples),
                "decode_tok_s": summarise_samples([1e3 / s for s in samples]),
                "method": "fresh subprocess per thread count",
            }
        )
    return rows


# ---------------------------------------------------------------------------------
# 1-3. Naive -> contiguous KV cache -> paged KV cache
# ---------------------------------------------------------------------------------
def bench_naive_vs_cache(
    cfg: ModelConfig,
    seeds: Sequence[int],
    prompt_len: int,
    new_tokens: Sequence[int],
    include_naive: bool = True,
    repeats: int = 2,
) -> list[dict[str, Any]]:
    """The headline table: tokens/s, TTFT and TPOT for each storage strategy.

    Two deliberate choices about *how* this is measured, both learned the hard way on this
    machine:

    1. **Variants are interleaved in the innermost loop.** The first version of this
       function measured all of naive, then all of contiguous, then all of dynamic. This
       laptop's throughput drifts by tens of percent over minutes, so any drift mapped
       directly onto variant identity and produced a confidently wrong ordering. Measuring
       the variants back-to-back within one (seed, size) cell makes drift common-mode.
    2. **Median over `repeats` within a seed, then bootstrap over seeds.** The median
       rejects a single transient stall; the bootstrap over seeds is what CONVENTIONS.md
       rule 5 asks for. A mean-over-repeats would let one scheduling hiccup move the row.

    The naive engine is swept over the same points as the cached ones and the speedup is
    reported *per point* rather than as one number -- the speedup is a function of sequence
    length, and collapsing it to a scalar hides the mechanism.
    """
    variants: list[tuple[str, Any]] = []
    if include_naive:
        variants.append(("naive_no_cache", NaiveEngine))
    variants += [
        ("contiguous_kv", lambda m: CachedEngine(m, cache="contiguous")),
        ("dynamic_kv", lambda m: CachedEngine(m, cache="dynamic")),
        ("paged_kv", lambda m: PagedEngine(m, num_blocks=256, block_size=DEFAULT_BLOCK_SIZE)),
    ]

    keys = ("tok_s", "ttft", "tpot", "fwd")
    acc: dict[tuple[str, int], dict[str, list[float]]] = {
        (name, n): {k: [] for k in keys} for name, _ in variants for n in new_tokens
    }

    for seed in seeds:
        model = _model(cfg, seed)
        prompt = _prompt(cfg, prompt_len, seed)
        for n_new in new_tokens:
            per_rep: dict[tuple[str, int], dict[str, list[float]]] = {
                (name, n_new): {k: [] for k in keys} for name, _ in variants
            }
            for _ in range(repeats):
                for name, factory in variants:
                    engine = factory(model)
                    engine.generate(prompt, 2)  # warmup this variant's code path
                    res = engine.generate_detailed(prompt, n_new)
                    cell = per_rep[(name, n_new)]
                    cell["tok_s"].append(res.n_generated / res.total_s)
                    cell["ttft"].append(res.ttft_s)
                    cell["tpot"].append(res.tpot_s)
                    cell["fwd"].append(float(res.tokens_forwarded))
                    del engine
            for name, _ in variants:
                for k in keys:
                    acc[(name, n_new)][k].append(statistics.median(per_rep[(name, n_new)][k]))
        del model

    rows: list[dict[str, Any]] = []
    baseline: dict[int, float] = {}
    for name, _ in variants:
        for n_new in new_tokens:
            cell = acc[(name, n_new)]
            per_token = kv_cache_bytes_per_token(cfg, dtype_bytes=4)
            final = prompt_len + n_new
            # Exact KV-storage accounting, not a timing: 'reserved' is the ceiling a
            # request holds, 'churn' is the total bytes the allocator has to hand out over
            # the request, which is where the dynamic strategy is quadratic.
            reserved = {
                "naive_no_cache": 0,
                "dynamic_kv": final * per_token,
                "contiguous_kv": cfg.max_seq_len * per_token,
                "paged_kv": -(-final // DEFAULT_BLOCK_SIZE) * DEFAULT_BLOCK_SIZE * per_token,
            }[name]
            churn = sum(range(prompt_len, final)) * per_token if name == "dynamic_kv" else reserved
            row: dict[str, Any] = {
                "bench": "kv_cache_generation",
                "variant": name,
                "prompt_len": prompt_len,
                "new_tokens": n_new,
                "total_len": final,
                "repeats_per_seed": repeats,
                "aggregation": "median over repeats within a seed, bootstrap over seeds",
                "ordering": "variants interleaved within each (seed, size) cell",
                "tokens_per_s": summarise_samples(cell["tok_s"]),
                "ttft_ms": summarise_samples([t * 1e3 for t in cell["ttft"]]),
                "tpot_ms": summarise_samples([t * 1e3 for t in cell["tpot"]]),
                "token_positions_forwarded": summarise_samples(cell["fwd"]),
                "kv_bytes_reserved": reserved,
                "kv_bytes_allocator_churn": churn,
            }
            if name == "naive_no_cache":
                baseline[n_new] = statistics.fmean(cell["tok_s"])
            elif n_new in baseline and baseline[n_new] > 0:
                row["speedup_vs_naive"] = statistics.fmean(cell["tok_s"]) / baseline[n_new]
            rows.append(row)
    return rows


def bench_paged_fragmentation(
    cfg: ModelConfig,
    seeds: Sequence[int],
    budget_mb: int = 64,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> list[dict[str, Any]]:
    """Memory fragmentation, contiguous vs paged -- "the single most impressive thing".

    Three sequence-length mixes, because the whole point is that the win grows with
    *variance*: a workload where every request is exactly ``max_seq_len`` long has nothing
    to gain, and a realistic RAG mix (short questions, long retrieved contexts) has a great
    deal to gain.
    """
    budget = budget_mb * 1024 * 1024
    rows: list[dict[str, Any]] = []
    mixes: dict[str, list[int]] = {}
    for seed in seeds:
        rng = random.Random(seed)
        mixes[f"uniform_short(seed={seed})"] = [rng.randint(16, 64) for _ in range(32)]
        mixes[f"rag_mixed(seed={seed})"] = [rng.choice([24, 40, 96, 320, 512]) for _ in range(32)]
        mixes[f"long_tail(seed={seed})"] = [rng.choice([8, 12, 16, 900]) for _ in range(32)]
    for label, lens in mixes.items():
        rep = fragmentation_report(cfg, lens, budget, block_size=block_size, dtype_bytes=4)
        rep["bench"] = "fragmentation"
        rep["mix"] = label
        rep["budget_mb"] = budget_mb
        rep["mean_seq_len"] = statistics.fmean(lens)
        rows.append(rep)

    # Max concurrent sequences at a fixed budget, both schemes, several lengths.
    for seq_len in (64, 128, 256, 512):
        cont = max_concurrent_sequences(cfg, budget, seq_len, paged=False, block_size=block_size)
        paged = max_concurrent_sequences(cfg, budget, seq_len, paged=True, block_size=block_size)
        rows.append(
            {
                "bench": "max_concurrent",
                "budget_mb": budget_mb,
                "seq_len": seq_len,
                "max_seq_len": cfg.max_seq_len,
                "kv_bytes_per_token": kv_cache_bytes_per_token(cfg, dtype_bytes=4),
                "contiguous_max_sequences": cont,
                "paged_max_sequences": paged,
                "paged_gain": paged / cont if cont else float("inf"),
            }
        )

    # Measured (not modelled) allocator behaviour on a live pool.
    for seed in seeds:
        rng = random.Random(seed)
        pool = PagedKVCache(cfg, num_blocks=512, block_size=block_size)
        live: list[str] = []
        for i in range(40):
            sid = f"s{i}"
            pool.add_sequence(sid)
            n = rng.randint(8, 100)
            try:
                pool.reserve(sid, n)
            except Exception:
                pool.free_sequence(sid)
                break
            pool.tables[sid].length = n
            live.append(sid)
            if len(live) > 12:
                pool.free_sequence(live.pop(0))
        rows.append(
            {
                "bench": "paged_allocator_measured",
                "seed": seed,
                "block_size": block_size,
                "num_blocks": pool.num_blocks,
                "blocks_in_use": pool.allocator.num_used,
                "blocks_free": pool.allocator.num_free,
                "internal_fragmentation": pool.internal_fragmentation(),
                "external_fragmentation": 0.0,
                "external_fragmentation_note": (
                    "identically zero by construction: every free block is interchangeable, "
                    "so a free block can always satisfy the next allocation"
                ),
                "total_allocations": pool.allocator.total_allocated,
                "total_frees": pool.allocator.total_freed,
            }
        )
    return rows


# ---------------------------------------------------------------------------------
# 4-5. Continuous batching and chunked prefill
# ---------------------------------------------------------------------------------
def bench_continuous_batching(
    cfg: ModelConfig,
    seeds: Sequence[int],
    n_requests: int = 12,
    rate_per_s: float = 4.0,
    scenarios: Sequence[tuple[str, float]] = (("poisson", 4.0), ("burst", 0.0)),
    prompt_lens: Sequence[int] = (64,),
    output_lens: Sequence[int] = (4, 8, 48),
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    """Static vs continuous batching under Poisson arrivals.

    Workload design, stated up front so the result can be judged:

    * **Output lengths are unequal** (4 / 8 / 48). With equal lengths static batching is
      perfectly fine and there is nothing to show. The gap here comes from head-of-line
      blocking -- a slot held by a finished request that a waiting request may not take --
      which is exactly what iteration-level scheduling removes.
    * **Prompt lengths are uniform.** Not to flatter the scheduler, but because this
      engine batches decode by exact cache-length bucket (see ``scheduler`` docstring);
      with ragged prompt lengths every bucket has size 1 and *neither* scheduler batches,
      which measures the missing kernel rather than the scheduling policy.
    * **Both schedulers use the same executor and the same concurrency**, so the delta is
      admission policy and nothing else.
    """
    del rate_per_s
    rows: list[dict[str, Any]] = []
    for scenario, rate in scenarios:
        for mode in ("static", "continuous"):
            throughput: list[float] = []
            goodput: list[float] = []
            ttft_p99: list[float] = []
            tpot_p99: list[float] = []
            latency_p99: list[float] = []
            batch: list[float] = []
            for seed in seeds:
                model = _model(cfg, seed)
                reqs = make_requests(
                    n_requests, prompt_lens, output_lens, cfg.vocab_size, seed=seed
                )
                arrivals = poisson_arrivals(rate, n_requests, seed=seed)
                if mode == "static":
                    t0 = time.perf_counter()
                    for r, a in zip(reqs, arrivals, strict=True):
                        r.arrival_time = t0 + a
                    sched = StaticBatchScheduler(model, batch_size=concurrency, num_blocks=1024)
                    sched.run(reqs, available=arrivals)
                    report = summarise(reqs, time.perf_counter() - t0, sched)
                else:
                    sched_c = ContinuousBatchingScheduler(
                        model,
                        SchedulerConfig(
                            max_num_seqs=concurrency,
                            max_num_batched_tokens=256,
                            enable_chunked_prefill=True,
                            num_blocks=1024,
                        ),
                    )
                    report = run_load(sched_c, reqs, arrivals)
                throughput.append(report.output_throughput_tok_s)
                goodput.append(report.goodput_req_s)
                ttft_p99.append(report.ttft["p99"])
                tpot_p99.append(report.tpot["p99"])
                latency_p99.append(report.latency["p99"])
                batch.append(report.achieved_batch_size)
                del model
            rows.append(
                {
                    "bench": "batching",
                    "mode": mode,
                    "scenario": scenario,
                    "n_requests": n_requests,
                    "arrival_rate_req_s": rate,
                    "concurrency": concurrency,
                    "prompt_lens": list(prompt_lens),
                    "output_lens": list(output_lens),
                    "output_throughput_tok_s": summarise_samples(throughput),
                    "goodput_req_s": summarise_samples(goodput),
                    "ttft_p99_ms": summarise_samples([v * 1e3 for v in ttft_p99]),
                    "tpot_p99_ms": summarise_samples([v * 1e3 for v in tpot_p99]),
                    "e2e_latency_p99_ms": summarise_samples([v * 1e3 for v in latency_p99]),
                    "achieved_batch_size": summarise_samples(batch),
                }
            )
    return rows


def bench_chunked_prefill(
    cfg: ModelConfig,
    seeds: Sequence[int],
    n_requests: int = 8,
    long_prompt: int = 512,
    short_prompt: int = 16,
    chunk: int = 64,
) -> list[dict[str, Any]]:
    """One long prompt against several short decoders, with and without chunking.

    p99 TPOT is the metric: an unchunked long prefill lands as a single large stall on
    every other in-flight request, and a single stall is invisible in a mean.
    """
    rows: list[dict[str, Any]] = []
    for enabled in (False, True):
        tpot_p99: list[float] = []
        tpot_p50: list[float] = []
        itl_p50: list[float] = []
        itl_p99: list[float] = []
        itl_max: list[float] = []
        ttft_p99: list[float] = []
        throughput: list[float] = []
        for seed in seeds:
            model = _model(cfg, seed)
            reqs = make_requests(n_requests - 1, [short_prompt], [16], cfg.vocab_size, seed=seed)
            long_reqs = make_requests(1, [long_prompt], [8], cfg.vocab_size, seed=seed + 100)
            long_reqs[0].request_id = "long-0"
            all_reqs = reqs + long_reqs
            arrivals = [0.0] * (n_requests - 1) + [0.02]
            sched = ContinuousBatchingScheduler(
                model,
                SchedulerConfig(
                    max_num_seqs=8,
                    max_num_batched_tokens=256,
                    enable_chunked_prefill=enabled,
                    prefill_chunk_size=chunk,
                    num_blocks=1024,
                ),
            )
            report = run_load(sched, all_reqs, arrivals)
            tpot_p99.append(report.tpot["p99"])
            tpot_p50.append(report.tpot["p50"])
            itl_p50.append(report.itl["p50"])
            itl_p99.append(report.itl["p99"])
            itl_max.append(max((g for r in all_reqs for g in r.itls), default=float("nan")))
            ttft_p99.append(report.ttft["p99"])
            throughput.append(report.output_throughput_tok_s)
            del model
        rows.append(
            {
                "bench": "chunked_prefill",
                "enabled": enabled,
                "chunk_size": chunk if enabled else None,
                "long_prompt_len": long_prompt,
                "n_requests": n_requests,
                "tpot_p50_ms": summarise_samples([v * 1e3 for v in tpot_p50]),
                "tpot_p99_ms": summarise_samples([v * 1e3 for v in tpot_p99]),
                "itl_p50_ms": summarise_samples([v * 1e3 for v in itl_p50]),
                "itl_p99_ms": summarise_samples([v * 1e3 for v in itl_p99]),
                "itl_max_ms": summarise_samples([v * 1e3 for v in itl_max]),
                "ttft_p99_ms": summarise_samples([v * 1e3 for v in ttft_p99]),
                "output_throughput_tok_s": summarise_samples(throughput),
            }
        )
    if len(rows) == 2:
        for key, label in (
            ("tpot_p99_ms", "p99_tpot_improvement_pct"),
            ("itl_p99_ms", "p99_itl_improvement_pct"),
            ("itl_max_ms", "max_itl_improvement_pct"),
        ):
            off = rows[0][key]["mean"]
            on = rows[1][key]["mean"]
            rows[1][label] = (off - on) / off * 100 if off else float("nan")
    return rows


# ---------------------------------------------------------------------------------
# 6. Prefix caching
# ---------------------------------------------------------------------------------
def bench_prefix_cache(
    cfg: ModelConfig,
    seeds: Sequence[int],
    n_requests: int = 8,
    shared_prefix_len: int = 128,
    unique_len: int = 24,
) -> list[dict[str, Any]]:
    """RAG-shaped load: one long shared system prompt, short unique questions.

    ``shared/(shared+unique)`` is the ceiling on the token hit rate, and TTFT should fall
    towards that ratio once the tree is warm. The first request is always a miss and is
    included in the average, because excluding it would be flattering rather than true.
    """
    rows: list[dict[str, Any]] = []
    for enabled in (False, True):
        ttft_mean: list[float] = []
        ttft_p99: list[float] = []
        hit_rate: list[float] = []
        token_hit: list[float] = []
        wall: list[float] = []
        for seed in seeds:
            model = _model(cfg, seed)
            shared = _prompt(cfg, shared_prefix_len, seed=999)
            reqs = make_requests(
                n_requests,
                [shared_prefix_len + unique_len],
                [8],
                cfg.vocab_size,
                seed=seed,
                shared_prefix=shared,
            )
            cache = RadixPrefixCache(cfg, min_prefix_len=16) if enabled else None
            sched = ContinuousBatchingScheduler(
                model,
                SchedulerConfig(
                    max_num_seqs=1,  # serialise so the cache is warm for later requests
                    max_num_batched_tokens=512,
                    enable_chunked_prefill=False,
                    enable_prefix_cache=enabled,
                    num_blocks=1024,
                ),
                prefix_cache=cache,
            )
            report = run_load(sched, reqs, [0.0] * n_requests)
            ttft_mean.append(report.ttft["mean"])
            ttft_p99.append(report.ttft["p99"])
            wall.append(report.wall_s)
            stats = report.extra.get("prefix_cache", {})
            hit_rate.append(float(stats.get("hit_rate", 0.0)))
            token_hit.append(float(stats.get("token_hit_rate", 0.0)))
            del model
        rows.append(
            {
                "bench": "prefix_cache",
                "enabled": enabled,
                "n_requests": n_requests,
                "shared_prefix_len": shared_prefix_len,
                "unique_len": unique_len,
                "max_possible_token_hit_rate": shared_prefix_len / (shared_prefix_len + unique_len),
                "ttft_mean_ms": summarise_samples([v * 1e3 for v in ttft_mean]),
                "ttft_p99_ms": summarise_samples([v * 1e3 for v in ttft_p99]),
                "wall_s": summarise_samples(wall),
                "request_hit_rate": summarise_samples(hit_rate),
                "token_hit_rate": summarise_samples(token_hit),
            }
        )
    if len(rows) == 2:
        off = rows[0]["ttft_mean_ms"]["mean"]
        on = rows[1]["ttft_mean_ms"]["mean"]
        rows[1]["ttft_reduction_pct"] = (off - on) / off * 100 if off else float("nan")
    return rows


# ---------------------------------------------------------------------------------
# 7. Speculative decoding
# ---------------------------------------------------------------------------------
def bench_speculative(
    cfg: ModelConfig,
    seeds: Sequence[int],
    prompt_len: int = 96,
    new_tokens: int = 48,
    k: int = 4,
    draft_depth_ratio: int = 3,
) -> list[dict[str, Any]]:
    """Acceptance rate and net speedup, for greedy and for temperature sampling.

    Read the caveat in the artifact before quoting these: the weights are **untrained**.
    Greedy decoding of an untrained model is degenerate (it emits one token forever), so
    the n-gram proposer trivially hits ~100% acceptance -- an upper bound, not a
    prediction. Temperature-1 sampling from an untrained model is near-uniform over 16k
    tokens, so acceptance is near zero -- a lower bound. Real acceptance on trained weights
    with RAG-shaped prompts sits between them, and this repository cannot measure it until
    Phase 4 produces a checkpoint. Reporting the bracket is the honest move; reporting the
    greedy number alone would not be.
    """
    rows: list[dict[str, Any]] = []
    settings = [("greedy", 0.0), ("temp1.0", 1.0)]
    for label, temp in settings:
        base_tok_s: list[float] = []
        for seed in seeds:
            model = _model(cfg, seed)
            prompt = _prompt(cfg, prompt_len, seed)
            engine = CachedEngine(model)
            engine.generate(prompt, 2)
            res = engine.generate_detailed(
                prompt, new_tokens, params=SamplingParams(temperature=temp, seed=seed)
            )
            base_tok_s.append(res.n_generated / res.total_s)
            del model
        rows.append(
            {
                "bench": "speculative",
                "proposer": "none_baseline",
                "sampling": label,
                "prompt_len": prompt_len,
                "new_tokens": new_tokens,
                "tokens_per_s": summarise_samples(base_tok_s),
                "acceptance_rate": None,
            }
        )
        baseline = statistics.fmean(base_tok_s)

        proposers: list[tuple[str, Any]] = [
            ("ngram", lambda _m: NgramProposer(max_ngram=4, min_ngram=2)),
        ]
        # The draft must be genuinely CHEAPER than the target or speculation cannot win no
        # matter how high acceptance goes -- every proposed token costs a draft forward.
        # Same width (shape constraints: d_model == n_heads * head_dim), 1/draft_depth_ratio
        # the depth. Derived from the target config rather than hardcoded.
        draft_cfg = cfg.model_copy(
            update={
                "name": f"{cfg.name}-draft-d{max(1, cfg.n_layers // draft_depth_ratio)}",
                "n_layers": max(1, cfg.n_layers // draft_depth_ratio),
                "expected_params": None,
            }
        )
        proposers.append(
            (
                f"draft_model(L={draft_cfg.n_layers}/{cfg.n_layers})",
                lambda _m, dc=draft_cfg, tp=temp: DraftModelProposer(
                    _model(dc, 12345), params=SamplingParams(temperature=tp, seed=7)
                ),
            )
        )
        for pname, pfactory in proposers:
            tok_s: list[float] = []
            accept: list[float] = []
            per_iter: list[float] = []
            for seed in seeds:
                model = _model(cfg, seed)
                prompt = _prompt(cfg, prompt_len, seed)
                proposer = pfactory(model)
                res = speculative_generate(
                    model,
                    proposer,
                    prompt,
                    SamplingParams(max_new_tokens=new_tokens, temperature=temp, seed=seed),
                    num_speculative=k,
                )
                tok_s.append(res.tokens_per_s)
                accept.append(res.acceptance_rate)
                per_iter.append(res.tokens_per_iteration)
                del model, proposer
            mean_accept = statistics.fmean(accept)
            rows.append(
                {
                    "bench": "speculative",
                    "proposer": pname,
                    "sampling": label,
                    "prompt_len": prompt_len,
                    "new_tokens": new_tokens,
                    "num_speculative": k,
                    "tokens_per_s": summarise_samples(tok_s),
                    "acceptance_rate": summarise_samples(accept),
                    "tokens_per_iteration": summarise_samples(per_iter),
                    "speedup_vs_baseline": (
                        statistics.fmean(tok_s) / baseline if baseline else float("nan")
                    ),
                    "above_spec_threshold_0_6": mean_accept >= 0.6,
                    "draft_layers": draft_cfg.n_layers if pname.startswith("draft") else None,
                    "target_layers": cfg.n_layers,
                    "weights": "UNTRAINED - see docstring; treat as a bracket, not a prediction",
                }
            )
    return rows


# ---------------------------------------------------------------------------------
# 8. Constrained decoding
# ---------------------------------------------------------------------------------
def bench_constrained(
    cfg: ModelConfig,
    seeds: Sequence[int],
    n_samples: int = 12,
    max_new_tokens: int = 40,
) -> list[dict[str, Any]]:
    """Invalid-JSON rate and tokens/s, with and without the grammar mask."""
    vocab = make_synthetic_vocab(cfg.vocab_size, seed=0)
    eos = ord("\x00")
    rows: list[dict[str, Any]] = []
    for constrained in (False, True):
        invalid: list[float] = []
        forced_rate: list[float] = []
        tok_s: list[float] = []
        mask_stats: dict[str, Any] = {}
        for seed in seeds:
            model = _model(cfg, seed)
            decoder = (
                ConstrainedDecoder(vocab, eos_token_id=eos, root="object") if constrained else None
            )
            bad = 0
            forced = 0
            speeds: list[float] = []
            for i in range(n_samples):
                prompt = _prompt(cfg, 16, seed * 1000 + i)
                out = generate_json(
                    model,
                    prompt,
                    SamplingParams(
                        max_new_tokens=max_new_tokens, temperature=1.0, seed=seed * 100 + i
                    ),
                    decoder=decoder,
                    vocab=vocab,
                )
                bad += 0 if out["valid_json"] else 1
                forced += 1 if out.get("forced_close") else 0
                speeds.append(out["tokens_per_s"])
            invalid.append(bad / n_samples)
            forced_rate.append(forced / n_samples)
            tok_s.append(statistics.fmean(speeds))
            if decoder is not None:
                mask_stats = decoder.stats()
            del model
        rows.append(
            {
                "bench": "constrained_decoding",
                "constrained": constrained,
                "grammar": "json_object" if constrained else None,
                "n_samples_per_seed": n_samples,
                "invalid_json_rate": summarise_samples(invalid),
                "forced_close_rate": summarise_samples(forced_rate),
                "forced_close_note": (
                    "fraction of generations that hit max_new_tokens mid-document and were "
                    "completed deterministically from the FSM state (constrained.closing_suffix). "
                    "Masking makes an invalid transition unreachable; it cannot invent token "
                    "budget, so truncation is closed explicitly and counted here rather than "
                    "reported as a valid-JSON win."
                ),
                "tokens_per_s": summarise_samples(tok_s),
                "mask_cache": mask_stats,
            }
        )
    if len(rows) == 2:
        off = rows[0]["tokens_per_s"]["mean"]
        on = rows[1]["tokens_per_s"]["mean"]
        rows[1]["tokens_per_s_cost_pct"] = (off - on) / off * 100 if off else float("nan")
    return rows


# ---------------------------------------------------------------------------------
# 9. Quantization + GGUF
# ---------------------------------------------------------------------------------
def bench_quantization(
    cfg: ModelConfig,
    seeds: Sequence[int],
    prompt_len: int = 64,
    new_tokens: int = 24,
    eval_tokens: int = 512,
    group_size: int = 64,
    gguf_dir: str | Path = "artifacts/gguf",
) -> list[dict[str, Any]]:
    """Size / tokens-per-second / quality frontier for fp32, int8, int4 and torch-dynamic.

    ``val BPB`` is computed as the spec asks. It is also, on **untrained** weights, close
    to ``log2(vocab)/bytes_per_token`` for every precision -- so the BPB *delta* here is
    noise, and the metric that actually resolves quantization error is the KL divergence
    against the fp32 logits. Both are reported; neither is dressed up.
    """
    rows: list[dict[str, Any]] = []
    vocab = make_synthetic_vocab(cfg.vocab_size, seed=0)
    bytes_per_token = statistics.fmean(len(t.encode("utf-8")) for t in vocab)

    variants: list[tuple[str, Any]] = [
        ("fp32", None),
        ("int8_weight_only_g64", ("weight_only", 8)),
        ("int4_weight_only_g64", ("weight_only", 4)),
        ("torch_dynamic_int8", ("dynamic", 8)),
    ]
    for name, spec in variants:
        size: list[float] = []
        tok_s: list[float] = []
        bpb: list[float] = []
        kl: list[float] = []
        error: str | None = None
        for seed in seeds:
            ref = _model(cfg, seed)
            model = _model(cfg, seed)
            if spec is None:
                pass
            elif spec[0] == "weight_only":
                quantize_model(model, bits=spec[1], group_size=group_size)
            else:
                try:
                    model = quantize_dynamic_int8(model)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    break
            size.append(float(_weight_bytes(model)))
            prompt = _prompt(cfg, prompt_len, seed)
            # quantize_dynamic_int8 is annotated -> nn.Module (it wraps torch's own
            # quantize_dynamic), but with inplace=False it deep-copies and returns an
            # instance of the same top-level class it was given, so `model` is always
            # still a LocalMindTransformer here.
            assert isinstance(model, LocalMindTransformer)
            engine = CachedEngine(model)
            engine.generate(prompt, 2)
            res = engine.generate_detailed(prompt, new_tokens)
            tok_s.append(res.n_generated / res.total_s)
            ids = torch.tensor([_prompt(cfg, eval_tokens, seed + 500)], dtype=torch.long)
            bpb.append(bits_per_byte(model, ids, bytes_per_token))
            kl.append(logit_kl(ref, model, ids[:, :128]))
            del model, ref
        if error is not None:
            rows.append(
                {
                    "bench": "quantization",
                    "variant": name,
                    "available": False,
                    "error": error,
                    "note": "no quantized CPU backend registered on this machine",
                }
            )
            continue
        rows.append(
            {
                "bench": "quantization",
                "variant": name,
                "available": True,
                "weight_bytes": summarise_samples(size),
                "weight_mb": summarise_samples([s / 1024**2 for s in size]),
                "tokens_per_s": summarise_samples(tok_s),
                "val_bpb": summarise_samples(bpb),
                "logit_kl_vs_fp32": summarise_samples(kl),
                "bytes_per_token_assumed": bytes_per_token,
            }
        )
    if rows and rows[0].get("available"):
        base_size = rows[0]["weight_bytes"]["mean"]
        base_tok = rows[0]["tokens_per_s"]["mean"]
        base_bpb = rows[0]["val_bpb"]["mean"]
        for r in rows[1:]:
            if not r.get("available"):
                continue
            r["compression_vs_fp32"] = base_size / r["weight_bytes"]["mean"]
            r["tok_s_ratio_vs_fp32"] = r["tokens_per_s"]["mean"] / base_tok
            r["bpb_delta_vs_fp32"] = r["val_bpb"]["mean"] - base_bpb

    # GGUF export -- a real file, with its verification status attached.
    model = _model(cfg, seeds[0])
    for quant in ("f32", "q8_0"):
        path = Path(gguf_dir) / f"{cfg.name.lower()}-{quant}.gguf"
        try:
            info = write_gguf(
                path,
                model,
                quant=quant,
                tokens=vocab,
                allow_lossy=True,
                extra_metadata={
                    "localmind.note": (
                        "written by localmind.inference.quantize.write_gguf; round-trip "
                        "verified by localmind's own reader, NOT verified against llama.cpp"
                    )
                },
            )
        except Exception as exc:
            info = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
        info["bench"] = "gguf_export"
        info["quant"] = quant
        rows.append(info)
    return rows


def _weight_bytes(model: torch.nn.Module) -> int:
    """Distinct weight storage, counting :class:`QuantLinear` buffers at their real width."""
    total = 0
    seen: set[int] = set()
    for module in model.modules():
        if isinstance(module, QuantLinear):
            total += module.weight_bytes()
            seen.update(id(b) for b in module.buffers())
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total


# ---------------------------------------------------------------------------------
# The honest baseline we cannot run
# ---------------------------------------------------------------------------------
def vllm_comparison_row(cfg: ModelConfig) -> dict[str, Any]:
    """The vLLM-on-a-T4 column, deliberately left empty.

    implementation.md is explicit that the same weights should be served through vLLM on a
    free T4 and the gap reported -- "you will lose", and saying so beats an unbenchmarked
    claim. There is no GPU and no vLLM in this environment, so the harness exists and the
    column is marked not-run. Inventing a number here would be worse than having none.
    """
    return {
        "bench": "vllm_baseline",
        "status": "NOT RUN",
        "reason": "no CUDA device and no vllm package in this environment (CPU-only laptop)",
        "harness": "localmind/inference/bench.py::run_vllm_comparison (requires vllm + a T4)",
        "how_to_run": [
            "provision a T4 (free Kaggle/Colab tier)",
            "pip install vllm",
            "export the LocalMind checkpoint to HF format",
            "python -m localmind.inference.bench --vllm-baseline --model <hf-dir>",
            "paste the resulting rows into artifacts/benchmarks/inference.json",
        ],
        "expected_outcome": (
            "LocalMind loses. The gap is kernel fusion, CUDA-graph capture, a real "
            "paged-attention kernel (no gather copy), and continuous batching with ragged "
            "attention instead of length bucketing."
        ),
        "config": cfg.name,
        "vllm_output_throughput_tok_s": None,
        "localmind_fraction_of_vllm": None,
    }


def run_vllm_comparison(
    model_dir: str, prompts: Sequence[str]
) -> dict[str, Any]:  # pragma: no cover
    """Run the same workload through vLLM. Requires CUDA + ``vllm``; never runs here."""
    from vllm import LLM  # type: ignore[import-not-found]
    from vllm import SamplingParams as VLLMSamplingParams

    llm = LLM(model=model_dir, dtype="float16")  # fp16, never bf16 (T4 is SM 7.5)
    sp = VLLMSamplingParams(max_tokens=64, temperature=0.0)
    t0 = time.perf_counter()
    outs = llm.generate(list(prompts), sp)
    wall = time.perf_counter() - t0
    gen = sum(len(o.outputs[0].token_ids) for o in outs)
    return {
        "bench": "vllm_baseline",
        "status": "RUN",
        "output_throughput_tok_s": gen / wall,
        "n_requests": len(prompts),
        "wall_s": wall,
    }


# ---------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------
SECTIONS = (
    "threads",
    "kv_cache",
    "paged",
    "batching",
    "chunked_prefill",
    "prefix_cache",
    "speculative",
    "constrained",
    "quantization",
)


def run_benchmarks(
    config_path: str | Path = DEFAULT_CONFIG,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    sections: Sequence[str] = SECTIONS,
    quick: bool = False,
    full: bool = False,
    out_path: str | Path = "artifacts/benchmarks/inference.json",
    section_path: str | Path | None = INFERENCE_SECTION,
    threads: int = 2,
) -> dict[str, Any]:
    """Run the selected sections and write the CONVENTIONS.md artifact."""
    torch.set_num_threads(threads)
    cfg = ModelConfig.from_yaml(config_path)
    rows: list[dict[str, Any]] = []
    timings: dict[str, float] = {}

    def section(name: str, fn: Any) -> None:
        if name not in sections:
            return
        t0 = time.perf_counter()
        rows.extend(fn())
        timings[name] = time.perf_counter() - t0
        print(f"  [{name}] {timings[name]:.1f}s", flush=True)

    print(f"benchmarking {cfg.name} on {hardware_string()}", flush=True)
    section("threads", lambda: bench_thread_scaling(cfg, seeds, config_path=str(config_path)))
    section(
        "kv_cache",
        lambda: bench_naive_vs_cache(
            cfg,
            seeds,
            prompt_len=32,
            new_tokens=(16, 32) if quick else ((64, 256, 512) if full else (32, 128, 256)),
        ),
    )
    section("paged", lambda: bench_paged_fragmentation(cfg, seeds))
    section(
        "batching",
        lambda: bench_continuous_batching(cfg, seeds, n_requests=6 if quick else 12),
    )
    section(
        "chunked_prefill",
        lambda: bench_chunked_prefill(
            cfg, seeds, n_requests=4 if quick else 8, long_prompt=256 if quick else 512
        ),
    )
    section(
        "prefix_cache",
        lambda: bench_prefix_cache(
            cfg, seeds, n_requests=4 if quick else 8, shared_prefix_len=64 if quick else 128
        ),
    )
    section(
        "speculative",
        lambda: bench_speculative(
            cfg, seeds, prompt_len=64 if quick else 96, new_tokens=16 if quick else 32
        ),
    )
    section(
        "constrained",
        lambda: bench_constrained(
            cfg, seeds, n_samples=4 if quick else 10, max_new_tokens=24 if quick else 40
        ),
    )
    section(
        "quantization",
        lambda: bench_quantization(
            cfg, seeds, new_tokens=8 if quick else 24, eval_tokens=256 if quick else 512
        ),
    )
    rows.append(vllm_comparison_row(cfg))

    payload: dict[str, Any] = {
        "name": "inference",
        "hardware": hardware_string(),
        "seeds": list(seeds),
        "rows": rows,
        "ci": "bootstrap95",
        "config": {
            "path": str(config_path),
            "name": cfg.name,
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "n_kv_heads": cfg.n_kv_heads,
            "max_seq_len": cfg.max_seq_len,
            "kv_bytes_per_token_fp32": kv_cache_bytes_per_token(cfg, dtype_bytes=4),
        },
        "section_seconds": timings,
        "quick": quick,
        "full": full,
        "notes": [
            "CPU only. No GPU was available; every number here was measured on the "
            "machine named in 'hardware'. The vLLM-on-T4 baseline is NOT RUN and is "
            "marked as such rather than estimated.",
            "Weights are randomly initialised: no pretrained checkpoint exists yet. "
            "Systems metrics (tokens/s, TTFT, TPOT, throughput, goodput, memory) are "
            "unaffected by weight quality and are real. Quality-dependent metrics "
            "(speculative acceptance rate, val BPB) are bracketed and flagged, not "
            "presented as predictions.",
            "torch.set_num_threads is pinned; see the thread_scaling rows for why more "
            "threads makes single-token decoding slower.",
            "Paged attention here gathers blocks into a dense tensor before each forward "
            "because localmind/model/attention.py (Phase 2-owned) takes a dense past_kv. "
            "Memory results are exact; latency carries a gather overhead a fused "
            "paged-attention kernel would not pay.",
            "No bf16 anywhere. No FlashAttention-2 anywhere.",
        ],
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if section_path is not None:
        write_section(Path(section_path), payload)
    return payload


def _fmt(est: Any, digits: int = 1) -> str:
    if not isinstance(est, dict) or "mean" not in est:
        return str(est)
    return f"{est['mean']:.{digits}f} [{est['ci_low']:.{digits}f}, {est['ci_high']:.{digits}f}]"


def write_section(path: Path, payload: dict[str, Any]) -> None:
    """Render this phase's contributed section. Owns one file; appends to nothing."""
    rows = payload["rows"]
    lines: list[str] = [
        "",
        "## Phase 6 - inference engine",
        "",
        f"Hardware: {payload['hardware']}. Seeds: {payload['seeds']}. CI: bootstrap 95%.",
        f"Config: `{payload['config']['path']}` ({payload['config']['name']}).",
        "",
        "### 0. CPU thread scaling of one decode step",
        "",
        "| threads | decode step | decode tok/s |",
        "|---|---|---|",
    ]
    for r in rows:
        if r.get("bench") == "thread_scaling" and "decode_step_ms" in r:
            lines.append(
                f"| threads={r['threads_effective']} | decode step "
                f"{_fmt(r['decode_step_ms'], 2)} ms | {_fmt(r['decode_tok_s'], 2)} tok/s |"
            )
    lines += [
        "",
        "### 1-3. KV cache: naive -> contiguous -> paged",
        "",
        "| variant | prompt | new | tokens/s (95% CI) | TTFT ms | TPOT ms | speedup vs naive |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("bench") == "kv_cache_generation":
            sp = r.get("speedup_vs_naive")
            lines.append(
                f"| {r['variant']} | {r['prompt_len']} | {r['new_tokens']} | "
                f"{_fmt(r['tokens_per_s'], 2)} | {_fmt(r['ttft_ms'])} | "
                f"{_fmt(r['tpot_ms'], 2)} | {'-' if sp is None else f'{sp:.2f}x'} |"
            )
    lines += [
        "",
        "### 3. Fragmentation: contiguous vs paged",
        "",
        "| mix | mean len | contiguous internal frag | paged internal frag | memory amplification |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("bench") == "fragmentation":
            lines.append(
                f"| {r['mix']} | {r['mean_seq_len']:.0f} | "
                f"{r['contiguous']['internal_fragmentation'] * 100:.1f}% | "
                f"{r['paged']['internal_fragmentation'] * 100:.1f}% | "
                f"{r['memory_amplification_contiguous_over_paged']:.2f}x |"
            )
    lines += [
        "",
        "| budget MB | seq len | contiguous max seqs | paged max seqs | gain |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("bench") == "max_concurrent":
            lines.append(
                f"| {r['budget_mb']} | {r['seq_len']} | {r['contiguous_max_sequences']} | "
                f"{r['paged_max_sequences']} | {r['paged_gain']:.1f}x |"
            )
    lines += [
        "",
        "### 4-5. Batching and chunked prefill",
        "",
        "| scenario | mode | throughput tok/s | goodput req/s | p99 TTFT ms | p99 TPOT ms | p99 e2e ms | batch |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("bench") == "batching":
            lines.append(
                f"| {r['scenario']} | {r['mode']} | {_fmt(r['output_throughput_tok_s'], 2)} | "
                f"{_fmt(r['goodput_req_s'], 3)} | {_fmt(r['ttft_p99_ms'])} | "
                f"{_fmt(r['tpot_p99_ms'])} | {_fmt(r['e2e_latency_p99_ms'])} | "
                f"{_fmt(r['achieved_batch_size'], 2)} |"
            )
    lines += [
        "",
        "| chunked prefill | p50 ITL ms | p99 ITL ms | max ITL ms | p99 TPOT ms | p99 ITL improvement |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("bench") == "chunked_prefill":
            imp = r.get("p99_itl_improvement_pct")
            lines.append(
                f"| {r['enabled']} | {_fmt(r['itl_p50_ms'])} | {_fmt(r['itl_p99_ms'])} | "
                f"{_fmt(r['itl_max_ms'])} | {_fmt(r['tpot_p99_ms'])} | "
                f"{'-' if imp is None else f'{imp:.1f}%'} |"
            )
    lines += [
        "",
        "### 6. Prefix caching (RAG-shaped: shared system prompt)",
        "",
        "| enabled | mean TTFT ms | request hit rate | token hit rate | TTFT reduction |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("bench") == "prefix_cache":
            red = r.get("ttft_reduction_pct")
            lines.append(
                f"| {r['enabled']} | {_fmt(r['ttft_mean_ms'])} | "
                f"{_fmt(r['request_hit_rate'], 3)} | {_fmt(r['token_hit_rate'], 3)} | "
                f"{'-' if red is None else f'{red:.1f}%'} |"
            )
    lines += [
        "",
        "### 7. Speculative decoding (UNTRAINED weights - a bracket, not a prediction)",
        "",
        "| proposer | sampling | tokens/s | acceptance | tokens/iter | speedup |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("bench") == "speculative":
            acc = r.get("acceptance_rate")
            sp = r.get("speedup_vs_baseline")
            lines.append(
                f"| {r['proposer']} | {r['sampling']} | {_fmt(r['tokens_per_s'], 2)} | "
                f"{'-' if acc is None else _fmt(acc, 3)} | "
                f"{_fmt(r.get('tokens_per_iteration'), 2)} | "
                f"{'-' if sp is None else f'{sp:.2f}x'} |"
            )
    lines += [
        "",
        "### 8. Constrained decoding",
        "",
        "| constrained | invalid JSON rate | forced-close rate | tokens/s | tok/s cost |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("bench") == "constrained_decoding":
            cost = r.get("tokens_per_s_cost_pct")
            lines.append(
                f"| {r['constrained']} | {_fmt(r['invalid_json_rate'], 3)} | "
                f"{_fmt(r['forced_close_rate'], 3)} | "
                f"{_fmt(r['tokens_per_s'], 2)} | {'-' if cost is None else f'{cost:.1f}%'} |"
            )
    lines += [
        "",
        "### 9. Quantization",
        "",
        "| variant | weight MB | tokens/s | val BPB | KL vs fp32 | compression |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("bench") == "quantization":
            if not r.get("available"):
                lines.append(f"| {r['variant']} | not available | - | - | - | {r['error']} |")
                continue
            comp = r.get("compression_vs_fp32")
            lines.append(
                f"| {r['variant']} | {_fmt(r['weight_mb'], 2)} | {_fmt(r['tokens_per_s'], 2)} | "
                f"{_fmt(r['val_bpb'], 3)} | {_fmt(r['logit_kl_vs_fp32'], 5)} | "
                f"{'1.00x' if comp is None else f'{comp:.2f}x'} |"
            )
    lines += [
        "",
        "| GGUF export | quant | MB | tensors | verified against llama.cpp |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("bench") == "gguf_export":
            if "error" in r:
                lines.append(f"| {r['path']} | {r['quant']} | - | - | ERROR: {r['error']} |")
            else:
                lines.append(
                    f"| {r['path']} | {r['quant']} | {r['mb']:.2f} | {r['n_tensors']} | **NO** |"
                )
    for r in rows:
        if r.get("bench") == "vllm_baseline":
            lines += [
                "",
                "### Honest baseline",
                "",
                f"vLLM on a T4: **{r['status']}** - {r['reason']}.",
                f"Expected outcome when run: {r['expected_outcome']}",
            ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LocalMind Phase 6 inference benchmarks")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--sections", nargs="+", default=list(SECTIONS), choices=list(SECTIONS))
    parser.add_argument("--quick", action="store_true", help="smaller sizes; for CI")
    parser.add_argument(
        "--full",
        action="store_true",
        help="sweep the KV-cache benchmark out to 512 new tokens (slow: the naive engine "
        "is quadratic, so this costs several minutes)",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", default="artifacts/benchmarks/inference.json")
    parser.add_argument(
        "--section",
        default=INFERENCE_SECTION,
        help="contributed-section file this harness owns; composed into docs/benchmarks.md "
        "by `python -m localmind.eval.report`",
    )
    parser.add_argument("--no-docs", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = run_benchmarks(
        config_path=args.config,
        seeds=tuple(args.seeds),
        sections=tuple(args.sections),
        quick=args.quick,
        full=args.full,
        out_path=args.out,
        section_path=None if args.no_docs else args.section,
        threads=args.threads,
    )
    print(f"wrote {len(payload['rows'])} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
