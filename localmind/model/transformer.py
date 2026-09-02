"""The LocalMind decoder-only transformer, plus the Phase 2 benchmark harness.

Run the benchmarks with::

    uv run python -m localmind.model.transformer

which writes `artifacts/benchmarks/model.json` (CONVENTIONS.md schema) and the Phase 2
section of `docs/benchmarks.md` to `artifacts/benchmarks/sections/20-model.md`.

That indirection is the point: `docs/benchmarks.md` has four producers, and this one
used to append to it directly, so `python -m localmind.eval.report` -- the command the
document itself names as its generator -- deleted this section every time it ran. Each
producer now owns exactly one file under `sections/` and the report composes them.
"""

from __future__ import annotations

import json
import platform
import random
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from localmind.model.attention import (
    ATTN_BACKENDS,
    AttnBackend,
    KVCache,
    benchmark_attention_backends,
)
from localmind.model.block import TransformerBlock
from localmind.model.config import (
    ModelConfig,
    count_params,
    flops_per_forward,
    flops_per_step,
    kv_cache_bytes_per_token,
)
from localmind.model.init import init_weights
from localmind.model.rmsnorm import RMSNorm
from localmind.model.rope import RotaryEmbedding

MODEL_SECTION = "artifacts/benchmarks/sections/20-model.md"
"""This phase's contributed section of `docs/benchmarks.md`. The `20-` prefix
orders it among the other producers; `localmind.eval.report` composes them in
filename order."""

__all__ = [
    "LocalMindTransformer",
    "ModelOutput",
    "bootstrap_ci",
    "count_module_params",
    "measure_device_peak_flops",
    "run_benchmarks",
]


class ModelOutput(NamedTuple):
    """What `LocalMindTransformer.forward` returns.

    `loss` is `ce_loss + cfg.z_loss * z_loss` -- the thing to call `.backward()` on.
    `ce_loss` is the number to report as "loss" in any curve, because z-loss is a
    regulariser, not a measure of prediction quality.
    """

    logits: Tensor
    loss: Tensor | None = None
    ce_loss: Tensor | None = None
    z_loss: Tensor | None = None
    kv_caches: list[KVCache] | None = None


class LocalMindTransformer(nn.Module):
    """Pre-norm decoder-only transformer: RMSNorm + RoPE + GQA + QK-norm + SwiGLU.

    One `RotaryEmbedding` is built for `cfg.max_seq_len` and shared by every layer.
    It is an attribute of *this* model, so it can never be reused by a model with a
    different `max_seq_len` (§6 trap list).
    """

    def __init__(self, cfg: ModelConfig, backend: AttnBackend = "sdpa_efficient") -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, backend) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # Weight tying: one matrix, counted once. 8.4M of the 31M budget.
            self.lm_head.weight = self.tok_emb.weight
        init_weights(self, cfg)

    @classmethod
    def from_yaml(
        cls, path: str | Path, backend: AttnBackend = "sdpa_efficient"
    ) -> LocalMindTransformer:
        return cls(ModelConfig.from_yaml(path), backend=backend)

    def set_backend(self, backend: AttnBackend) -> None:
        """Swap the attention backend on every layer (benchmarks and equivalence tests)."""
        if backend not in ATTN_BACKENDS:
            raise ValueError(f"backend must be one of {ATTN_BACKENDS}, got {backend!r}")
        for block in self.blocks:
            # `self.blocks` is an nn.ModuleList; iterating it statically yields `Module`,
            # which lacks `set_backend`. Every element really is a TransformerBlock (built
            # in __init__ above), so narrow it explicitly.
            assert isinstance(block, TransformerBlock)
            block.set_backend(backend)

    def num_params(self, include_norms: bool = False, include_embedding: bool = True) -> int:
        """Actual parameter count of the instantiated module (tied weights counted once).

        Cross-checked against the analytic `config.count_params` in the tests.
        """
        return count_module_params(
            self, include_norms=include_norms, include_embedding=include_embedding
        )

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        past_kvs: list[KVCache] | None = None,
        use_cache: bool = False,
        doc_ids: Tensor | None = None,
    ) -> ModelOutput:
        """
        Args:
            input_ids: `(B, T)` int64 token ids.
            targets: `(B, T)` int64 labels, **already shifted** by the data pipeline --
                `targets[b, t]` is the token that follows `input_ids[b, t]`. Use
                `-100` to mask a position out of the loss.
            past_kvs: per-layer `(k, v)` caches from a previous call.
            use_cache: return fresh per-layer caches in `ModelOutput.kv_caches`.
            doc_ids: optional `(B, T)` per-position document segment ids for packed
                sequences, as produced by
                `localmind.data.packing.doc_ids_from_boundaries`. When supplied,
                attention is block-diagonal: a token can only attend to its causal
                past *within its own document*, which is the §7 requirement that
                packing "pass document boundaries so attention doesn't cross them".
                Passing `None` is the naive-packing ablation arm and is byte-identical
                to the plain causal model.

                Loss masking is NOT a substitute for this: masking the loss stops
                boundary positions contributing gradients, but tokens from document A
                would still *attend* to document B.

                With `past_kvs`, `doc_ids` must cover the whole context
                (`past_len + T`), not just the new tokens; a width mismatch raises.
        """
        b, t = input_ids.shape
        past_len = 0 if past_kvs is None else past_kvs[0][0].shape[-2]
        if past_len + t > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence of {past_len + t} exceeds max_seq_len={self.cfg.max_seq_len}"
            )

        if doc_ids is not None:
            expected = past_len + t
            if doc_ids.shape != (b, expected):
                raise ValueError(
                    f"doc_ids must be (B, past_len + T) = {(b, expected)}, got "
                    f"{tuple(doc_ids.shape)}"
                )

        x = self.tok_emb(input_ids)
        caches: list[KVCache] | None = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = None if past_kvs is None else past_kvs[i]
            x, present = block(
                x, self.rope, past_kv=layer_past, use_cache=use_cache, doc_ids=doc_ids
            )
            if caches is not None and present is not None:
                caches.append(present)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return ModelOutput(logits=logits, kv_caches=caches)

        # Loss in fp32 -- mandatory under fp16 autocast on a T4 (§3.2).
        flat = logits.float().view(b * t, -1)
        ce = F.cross_entropy(flat, targets.reshape(-1), ignore_index=-100)
        # z-loss pins log Z near 0, which keeps the logits themselves small enough that
        # fp16 softmax cannot overflow. Cheap insurance for a 12-hour unattended run.
        z = torch.logsumexp(flat, dim=-1).pow(2).mean()
        loss = ce + self.cfg.z_loss * z
        return ModelOutput(logits=logits, loss=loss, ce_loss=ce, z_loss=z, kv_caches=caches)


def count_module_params(
    model: nn.Module, include_norms: bool = False, include_embedding: bool = True
) -> int:
    """Count parameters of a live module, de-duplicating tied weights.

    `include_norms=False` drops every `RMSNorm` scale, so this is directly comparable
    to `config.count_params(cfg, include_norms=False)` -- see the note there about the
    spec's "~33,000" norm figure being wrong.
    """
    norm_ids: set[int] = {
        id(p) for m in model.modules() if isinstance(m, RMSNorm) for p in m.parameters()
    }
    emb_ids: set[int] = {
        id(p) for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()
    }
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if not include_norms and id(p) in norm_ids:
            continue
        if not include_embedding and id(p) in emb_ids:
            continue
        total += p.numel()
    return total


# -----------------------------------------------------------------------------------
# Benchmark harness (§6 "Benchmarks to produce")
# -----------------------------------------------------------------------------------
def bootstrap_ci(
    samples: Sequence[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. CONVENTIONS.md rule 5: no bare numbers."""
    vals = [float(v) for v in samples]
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


def _summarise(samples: Sequence[float]) -> dict[str, float]:
    if not samples:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    lo, hi = bootstrap_ci(samples)
    return {
        "mean": statistics.fmean(samples),
        "std": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "ci_low": lo,
        "ci_high": hi,
        "n": len(samples),
    }


def measure_device_peak_flops(
    device: str = "cpu", dtype: torch.dtype = torch.float32, size: int = 1024, iters: int = 5
) -> float:
    """Empirical peak dense-matmul FLOP/s of this device.

    MFU needs a denominator. On a T4 you would use the datasheet 65.13 TFLOP/s fp16
    tensor-core number; on CPU there is no meaningful datasheet figure, so we measure
    the machine's own large-matmul roofline and report MFU against that. This is
    stated explicitly in the JSON so the number is never mistaken for a GPU MFU.
    """
    dev = torch.device(device)
    a = torch.randn(size, size, device=dev, dtype=dtype)
    b = torch.randn(size, size, device=dev, dtype=dtype)
    torch.matmul(a, b)  # warmup
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (2.0 * size**3 * iters) / elapsed


def benchmark_kv_cache(cfg: ModelConfig, context_len: int = 2048) -> list[dict[str, Any]]:
    """MHA vs GQA KV-cache cost, computed from the config -- never hardcoded.

    For LocalMind-31M this reproduces the §6 headline: 16 KB/token for MHA vs
    4 KB/token for GQA(8q/2kv); at 2048 context, 32 MB vs 8 MB per sequence, i.e. 4x
    the concurrent users on the same VRAM.
    """
    rows: list[dict[str, Any]] = []
    variants: list[tuple[str, int]] = [
        ("MHA", cfg.n_heads),
        (f"GQA({cfg.n_heads}q/{cfg.n_kv_heads}kv)", cfg.n_kv_heads),
        ("MQA", 1),
    ]
    for label, kv_heads in variants:
        per_token = kv_cache_bytes_per_token(cfg, n_kv_heads=kv_heads, dtype_bytes=2)
        rows.append(
            {
                "bench": "kv_cache",
                "variant": label,
                "n_kv_heads": kv_heads,
                "dtype": "fp16",
                "bytes_per_token": per_token,
                "kb_per_token": per_token / 1024,
                "context_len": context_len,
                "mb_per_sequence": per_token * context_len / 1024**2,
                "vs_mha_ratio": kv_cache_bytes_per_token(cfg, cfg.n_heads, 2) / per_token,
            }
        )
    return rows


def benchmark_throughput(
    cfg: ModelConfig,
    batch_size: int,
    seq_len: int,
    seeds: Sequence[int] = (0, 1, 2),
    device: str = "cpu",
    iters: int = 2,
) -> list[dict[str, Any]]:
    """Forward and forward+backward TFLOP/s and MFU at a target batch size."""
    dev = torch.device(device)
    peak = measure_device_peak_flops(device)
    rows: list[dict[str, Any]] = []

    fwd_flops = flops_per_forward(cfg, batch_size, seq_len)
    step_flops = flops_per_step(cfg, batch_size, seq_len)

    for mode, total_flops in (("forward", fwd_flops), ("forward_backward", step_flops)):
        tflops_samples: list[float] = []
        tok_per_s: list[float] = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = LocalMindTransformer(cfg).to(dev)
            model.train(mode == "forward_backward")
            ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
            tgt = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)

            def once(
                model: LocalMindTransformer = model,
                ids: Tensor = ids,
                tgt: Tensor = tgt,
                mode: str = mode,
            ) -> None:
                if mode == "forward":
                    with torch.no_grad():
                        model(ids)
                else:
                    model.zero_grad(set_to_none=True)
                    out = model(ids, tgt)
                    assert out.loss is not None
                    out.loss.backward()

            once()  # warmup
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                once()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / iters
            tflops_samples.append(total_flops / elapsed / 1e12)
            tok_per_s.append(batch_size * seq_len / elapsed)
            del model

        rows.append(
            {
                "bench": "throughput",
                "mode": mode,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "flops_per_iter": total_flops,
                "tflops": _summarise(tflops_samples),
                "tokens_per_s": _summarise(tok_per_s),
                "device_peak_tflops_measured": peak / 1e12,
                "mfu_vs_measured_device_peak": statistics.fmean(tflops_samples) * 1e12 / peak,
                "mfu_denominator": (
                    "empirically measured fp32 dense-matmul roofline of THIS CPU, not a "
                    "GPU datasheet number; a T4 MFU requires CUDA and is gpu-marked"
                ),
            }
        )
    return rows


def _hardware_string() -> str:
    parts = [
        "CPU",
        platform.processor() or platform.machine(),
        f"{torch.get_num_threads()} torch threads",
        platform.system(),
        f"torch {torch.__version__}",
    ]
    if torch.cuda.is_available():  # pragma: no cover - no GPU in this environment
        parts.insert(0, torch.cuda.get_device_name(0))
    return " | ".join(p for p in parts if p)


def run_benchmarks(
    config_path: str | Path = "configs/model/31m.yaml",
    seq_lens: Sequence[int] = (512, 1024, 2048, 4096),
    seeds: Sequence[int] = (0, 1, 2),
    throughput_batch: int = 2,
    throughput_seq: int = 512,
    out_path: str | Path = "artifacts/benchmarks/model.json",
    section_path: str | Path | None = MODEL_SECTION,
) -> dict[str, Any]:
    """Produce every §6 benchmark and write the CONVENTIONS.md JSON artifact."""
    cfg = ModelConfig.from_yaml(config_path)
    rows: list[dict[str, Any]] = []
    rows.extend(benchmark_kv_cache(cfg))

    # One attention layer, batch 1, so the seq-length sweep is the only variable.
    attn_rows = benchmark_attention_backends(
        cfg, seq_lens=seq_lens, seeds=seeds, batch_size=1, device="cpu"
    )
    for row in attn_rows:
        row["latency_ms"] = _summarise(row.pop("latency_ms_samples"))
        row["peak_bytes"] = _summarise([float(v) for v in row.pop("peak_bytes_samples")])
        row["total_alloc_bytes"] = _summarise(
            [float(v) for v in row.pop("total_alloc_bytes_samples")]
        )
    rows.extend(attn_rows)

    rows.extend(benchmark_throughput(cfg, throughput_batch, throughput_seq, seeds=seeds))

    payload: dict[str, Any] = {
        "name": "model",
        "hardware": _hardware_string(),
        "seeds": list(seeds),
        "rows": rows,
        "ci": "bootstrap95",
        "config": {
            "path": str(config_path),
            "name": cfg.name,
            "params_excl_norms": count_params(cfg, include_norms=False),
            "params_incl_norms": count_params(cfg, include_norms=True),
        },
        "notes": [
            "No GPU was available: every number here is CPU. The code paths are the "
            "same ones that run on a T4 (SM 7.5).",
            "'sdpa_efficient' pins EFFICIENT_ATTENTION on CUDA. On CPU that kernel is "
            "not registered, so it pins ATen's tiled CPU SDPA kernel "
            "(_scaled_dot_product_flash_attention_for_cpu) instead. That is an ATen "
            "kernel, NOT the FlashAttention-2 library, which needs SM 8.0+ and is "
            "banned repo-wide.",
            "CPU peak memory is integrated from profiler per-op allocation deltas; "
            "total_alloc_bytes is exact, peak_bytes is a running-sum estimate.",
            "bf16 appears nowhere. fp16 + GradScaler or fp32 only.",
        ],
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if section_path is not None:
        _write_section(Path(section_path), payload)
    return payload


def _write_section(path: Path, payload: dict[str, Any]) -> None:
    """Write this phase's contributed section. Owns one file; appends to nothing.

    This used to `open("docs/benchmarks.md", "a")`, which meant the documented
    regeneration command (`python -m localmind.eval.report`) deleted it, and
    re-running this harness duplicated it. `localmind.eval.report` now composes
    `artifacts/benchmarks/sections/*.md` into the deliverable instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "",
        "## Phase 2 - model",
        "",
        f"Hardware: {payload['hardware']}. Seeds: {payload['seeds']}. CI: bootstrap 95%.",
        "",
        "### KV cache cost (fp16)",
        "",
        "| variant | n_kv_heads | KB/token | MB @ 2048 ctx | vs MHA |",
        "|---|---|---|---|---|",
    ]
    for r in payload["rows"]:
        if r["bench"] == "kv_cache":
            lines.append(
                f"| {r['variant']} | {r['n_kv_heads']} | {r['kb_per_token']:.2f} | "
                f"{r['mb_per_sequence']:.2f} | {r['vs_mha_ratio']:.2f}x |"
            )
    lines += [
        "",
        "### Attention backends (one layer, batch 1)",
        "",
        "| backend | kernel | seq | latency ms (95% CI) | peak MB | total alloc MB |",
        "|---|---|---|---|---|---|",
    ]
    for r in payload["rows"]:
        if r["bench"] == "attn_backend":
            lat = r["latency_ms"]
            lines.append(
                f"| {r['backend']} | {r['observed_kernel']} | {r['seq_len']} | "
                f"{lat['mean']:.2f} [{lat['ci_low']:.2f}, {lat['ci_high']:.2f}] | "
                f"{r['peak_bytes']['mean'] / 1024**2:.1f} | "
                f"{r['total_alloc_bytes']['mean'] / 1024**2:.1f} |"
            )
    lines += [
        "",
        "### Throughput",
        "",
        "| mode | batch x seq | TFLOP/s (95% CI) | tokens/s | MFU vs measured CPU peak |",
        "|---|---|---|---|---|",
    ]
    for r in payload["rows"]:
        if r["bench"] == "throughput":
            tf = r["tflops"]
            lines.append(
                f"| {r['mode']} | {r['batch_size']}x{r['seq_len']} | "
                f"{tf['mean']:.4f} [{tf['ci_low']:.4f}, {tf['ci_high']:.4f}] | "
                f"{r['tokens_per_s']['mean']:.1f} | "
                f"{r['mfu_vs_measured_device_peak'] * 100:.1f}% |"
            )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover - benchmark entrypoint
    payload = run_benchmarks()
    print(json.dumps({"name": payload["name"], "hardware": payload["hardware"]}, indent=2))
    print(f"rows: {len(payload['rows'])} -> artifacts/benchmarks/model.json")


if __name__ == "__main__":  # pragma: no cover
    main()
