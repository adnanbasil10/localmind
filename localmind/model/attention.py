"""Causal self-attention with GQA, QK-norm, and three interchangeable backends.

The three backends (implementation.md §6) sit behind **one** interface so they are
swappable at runtime and comparable in a benchmark:

1. ``naive``          -- explicit ``softmax(QK^T / sqrt(d) + mask) V``. The numerical
                         reference. Materialises the full ``T x T`` score matrix, so
                         its memory grows quadratically. This is the curve we plot.
2. ``sdpa_math``      -- ``F.scaled_dot_product_attention`` pinned to the MATH backend.
3. ``sdpa_efficient`` -- SDPA pinned to the memory-efficient backend. **This is "flash"
                         on a T4.** FlashAttention-2 needs SM 8.0+; the T4 is SM 7.5,
                         so it is neither imported nor used anywhere in this repo.

Backend selection goes through ``torch.nn.attention.sdpa_kernel([SDPBackend...])``, and
``observed_sdpa_backend`` asserts which kernel *actually* fired by reading the ATen op
names off the profiler -- pinning a backend and hoping is not the same as verifying it.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal, get_args

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

from localmind.model.config import ModelConfig
from localmind.model.rmsnorm import RMSNorm
from localmind.model.rope import RotaryEmbedding, apply_rope

__all__ = [
    "ATTN_BACKENDS",
    "AttnBackend",
    "CausalSelfAttention",
    "KVCache",
    "benchmark_attention_backends",
    "measure_memory",
    "observed_sdpa_backend",
    "repeat_kv",
]

AttnBackend = Literal["naive", "sdpa_math", "sdpa_efficient"]
ATTN_BACKENDS: tuple[AttnBackend, ...] = get_args(AttnBackend)

#: One layer's cache: ``(k, v)`` each ``(B, n_kv_heads, T_past, head_dim)``.
KVCache = tuple[Tensor, Tensor]

# ATen op name -> the backend label it proves fired.
_KERNEL_MARKERS: dict[str, str] = {
    "aten::_scaled_dot_product_efficient_attention": "efficient",
    "aten::_efficient_attention_forward": "efficient",
    "aten::_scaled_dot_product_cudnn_attention": "cudnn",
    "aten::_scaled_dot_product_flash_attention_for_cpu": "flash_cpu",
    "aten::_scaled_dot_product_flash_attention": "flash",
    "aten::_flash_attention_forward": "flash",
    "aten::_scaled_dot_product_attention_math": "math",
}
#: Backends that stream tiles instead of materialising the T x T score matrix.
FUSED_KERNELS = frozenset({"efficient", "cudnn", "flash_cpu", "flash"})


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand ``n_kv_heads`` up to ``n_heads`` for GQA.

    ``x`` is ``(B, n_kv_heads, T, head_dim)``; the result is
    ``(B, n_kv_heads * n_rep, T, head_dim)`` with query head ``h`` reading KV head
    ``h // n_rep``.

    Uses ``repeat_interleave``, **not** ``expand`` (implementation.md §6 trap list).
    ``expand`` produces a zero-stride view; the SDPA fused kernels want a contiguous,
    materialised tensor and either fall back to math or misread strides. The memory
    cost is real but bounded, and it keeps every backend seeing identical inputs.
    """
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


def observed_sdpa_backend(fn: Callable[[], Any]) -> str:
    """Run ``fn`` under the profiler and report which SDPA kernel actually fired.

    Returns one of ``efficient`` / ``cudnn`` / ``flash_cpu`` / ``flash`` / ``math``,
    or ``unknown`` if no recognised ATen SDPA op was recorded. Profiling is expensive;
    call this in tests and benchmarks, never in a training loop.
    """
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        fn()
    events = prof.events()
    assert events is not None, "profiler must have run before events() is read"
    names = {evt.name for evt in events}
    # Ordered by specificity: a fused kernel marker beats the generic math marker.
    for op, label in _KERNEL_MARKERS.items():
        if op in names:
            return label
    return "unknown"


def measure_memory(fn: Callable[[], Any], device_type: str = "cpu") -> dict[str, int]:
    """Peak / total tensor bytes allocated while running ``fn``.

    On CUDA both come from the allocator's own high-water-mark stats and are exact.

    On CPU there is no allocator high-water-mark API, so we integrate the per-op
    allocation deltas the profiler records:

    * ``total_alloc_bytes`` -- sum of positive deltas. **This is the headline memory
      metric on CPU**: it is exact and it reproduces the quadratic-vs-linear curve.
    * ``peak_bytes`` -- running-sum maximum. A *lower bound*: allocations made and
      freed inside a single composite ATen op (notably the SDPA math kernel's T x T
      score buffer) net to zero in the self-memory accounting and are invisible here.
      Verified empirically: at T=2048 the math backend reports 20 MB peak but 384 MB
      total allocated, and the 384 MB is the honest figure.
    """
    if device_type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        return {
            "peak_bytes": int(torch.cuda.max_memory_allocated()),
            "total_alloc_bytes": int(
                torch.cuda.memory_stats().get("allocated_bytes.all.allocated", 0)
            ),
        }

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        fn()
    running = 0
    peak = 0
    total = 0
    events = prof.events()
    assert events is not None, "profiler must have run before events() is read"
    for evt in events:
        delta = int(getattr(evt, "self_cpu_memory_usage", 0) or 0)
        running += delta
        peak = max(peak, running)
        if delta > 0:
            total += delta
    return {"peak_bytes": peak, "total_alloc_bytes": total}


def _sdpa_backend_request(backend: AttnBackend, device_type: str) -> list[SDPBackend]:
    """Map our backend name onto the ``SDPBackend`` list to pin."""
    if backend == "sdpa_math":
        return [SDPBackend.MATH]
    if backend == "sdpa_efficient":
        if device_type == "cuda":
            return [SDPBackend.EFFICIENT_ATTENTION]
        # On CPU, ATen registers its tiled linear-memory SDPA kernel
        # (`aten::_scaled_dot_product_flash_attention_for_cpu`) under the
        # FLASH_ATTENTION flag. That is an ATen CPU kernel, NOT the FlashAttention-2
        # library -- which is banned here because it requires SM 8.0+ (§3.2). It is
        # the CPU stand-in for the memory-efficient CUDA kernel we would get on a T4.
        return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    raise ValueError(f"{backend!r} is not an SDPA backend")


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with grouped-query KV heads and QK-norm.

    Order of operations per forward: project -> split heads -> **QK-norm** ->
    **RoPE on Q and K only** -> repeat KV for GQA -> attention -> output projection.

    QK-norm (RMSNorm over ``head_dim``, per head, before the dot product) is applied
    *before* RoPE, matching Gemma 2/3 and Olmo 2. RoPE is a rotation, so it preserves
    the norm QK-norm just imposed. Under fp16 autocast on a T4 this is what stops
    attention logits from blowing past the fp16 range at hour 9 of a 12-hour run.
    """

    def __init__(self, cfg: ModelConfig, backend: AttnBackend = "sdpa_efficient") -> None:
        super().__init__()
        if backend not in ATTN_BACKENDS:
            raise ValueError(f"backend must be one of {ATTN_BACKENDS}, got {backend!r}")
        self.cfg = cfg
        self.backend: AttnBackend = backend
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_rep
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim**-0.5
        self.attn_dropout = cfg.attn_dropout

        self.wq = nn.Linear(cfg.d_model, cfg.q_dim, bias=cfg.bias)
        self.wk = nn.Linear(cfg.d_model, cfg.kv_dim, bias=cfg.bias)
        self.wv = nn.Linear(cfg.d_model, cfg.kv_dim, bias=cfg.bias)
        self.wo = nn.Linear(cfg.q_dim, cfg.d_model, bias=cfg.bias)

        if cfg.qk_norm:
            self.q_norm: nn.Module | None = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
            self.k_norm: nn.Module | None = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.dropout = nn.Dropout(cfg.attn_dropout)

    def set_backend(self, backend: AttnBackend) -> None:
        if backend not in ATTN_BACKENDS:
            raise ValueError(f"backend must be one of {ATTN_BACKENDS}, got {backend!r}")
        self.backend = backend

    # -- backends -------------------------------------------------------------------
    def _naive(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None) -> Tensor:
        """Explicit ``softmax(QK^T/sqrt(d) + mask)V``. The numerical reference.

        The causal mask is applied here and is NOT optional: without it the model sees
        the future, and the training loss looks *suspiciously good* rather than
        obviously broken (implementation.md §6 trap list).
        """
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            att = att.masked_fill(~mask, torch.finfo(att.dtype).min)
        # Softmax in fp32 even under fp16 autocast: exp() of an fp16 logit overflows
        # long before an fp32 one does.
        att = torch.softmax(att.float(), dim=-1).to(q.dtype)
        att = self.dropout(att)
        return torch.matmul(att, v)

    def _sdpa(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None,
        is_causal: bool,
        backend: AttnBackend,
    ) -> Tensor:
        dropout_p = self.attn_dropout if self.training else 0.0
        request = _sdpa_backend_request(backend, q.device.type)
        with sdpa_kernel(request):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal
            )

    # -- forward --------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        rope: RotaryEmbedding,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
        backend: AttnBackend | None = None,
        doc_ids: Tensor | None = None,
    ) -> tuple[Tensor, KVCache | None]:
        """
        Args:
            x: ``(B, T, d_model)``.
            rope: the rotary table for this model. Passed in rather than owned so all
                layers share one table (it is a function of head_dim/max_seq_len only)
                while remaining impossible to share across different ``max_seq_len``.
            past_kv: cached ``(k, v)`` from previous steps, for incremental decoding.
            use_cache: return the concatenated ``(k, v)`` for the next step.
            backend: one-shot override of ``self.backend``.
            doc_ids: optional per-position document segment ids for packed sequences
                (§7/§8). When given, attention is restricted to the same document
                *and* the causal past -- a block-diagonal causal mask. When ``None``
                the behaviour is exactly the plain-causal path. With a KV cache,
                ``doc_ids`` must cover the whole context (past + current), i.e. width
                ``past_len + T``; see `_doc_causal_mask`.

        Returns:
            ``(output (B, T, d_model), present_kv or None)``.
        """
        b, t, _ = x.shape
        chosen: AttnBackend = self.backend if backend is None else backend

        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-norm, per head, over head_dim -- before the dot product. Never on V.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        past_len = 0 if past_kv is None else past_kv[0].shape[-2]
        cos, sin = rope(t, offset=past_len)
        # RoPE touches Q and K only. V carries content, not position.
        q, k = apply_rope(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=-2)
            v = torch.cat((past_kv[1], v), dim=-2)
        present: KVCache | None = (k, v) if use_cache else None

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        kv_len = k.shape[-2]

        if doc_ids is not None:
            # Packed-sequence path: one explicit mask, identical for every backend.
            # Causality is folded INTO the mask, so `is_causal` must be False -- SDPA
            # rejects combining an explicit mask with is_causal=True, and it would
            # double-apply the triangle anyway.
            doc_mask = self._doc_causal_mask(doc_ids, t, kv_len, x.device)
            if chosen == "naive":
                out = self._naive(q, k, v, doc_mask)
            else:
                out = self._sdpa(q, k, v, doc_mask, False, chosen)
        elif chosen == "naive":
            # The naive reference always gets an explicit mask. A missing causal mask
            # here is the §6 trap where the loss looks suspiciously *good*.
            naive_mask = None if t == 1 else self._causal_mask(t, kv_len, x.device)
            out = self._naive(q, k, v, naive_mask)
        else:
            sdpa_mask, is_causal = self._sdpa_mask(t, kv_len, x.device)
            out = self._sdpa(q, k, v, sdpa_mask, is_causal, chosen)

        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.wo(out), present

    def _doc_causal_mask(
        self, doc_ids: Tensor, q_len: int, kv_len: int, device: torch.device
    ) -> Tensor:
        """Block-diagonal causal mask: ``(B, 1, q_len, kv_len)`` bool, True = attend.

        ``mask[b, 0, i, j]`` is True iff position ``j`` is in the causal past of ``i``
        **and** both positions belong to the same document. This is what stops
        attention bleeding across document boundaries in a packed row (§7); the naive
        packing ablation is simply not passing ``doc_ids`` at all.

        Rectangular by construction, so it is correct with a KV cache: the queries are
        the last ``q_len`` positions of the context, the keys are all ``kv_len`` of
        them. That requires ``doc_ids`` to describe the WHOLE context, which is why a
        width mismatch is a hard error rather than a broadcast.
        """
        if doc_ids.dim() != 2:
            raise ValueError(f"doc_ids must be (B, T), got shape {tuple(doc_ids.shape)}")
        if doc_ids.shape[-1] != kv_len:
            raise ValueError(
                f"doc_ids has width {doc_ids.shape[-1]} but the attention context is "
                f"{kv_len} positions ({kv_len - q_len} cached + {q_len} new). With a KV "
                f"cache, doc_ids must cover the whole context (past + current), not just "
                f"the new tokens."
            )
        doc_ids = doc_ids.to(device)
        # Queries are the trailing q_len positions of the context.
        doc_q = doc_ids[:, kv_len - q_len :]
        same_doc = (doc_q[:, :, None] == doc_ids[:, None, :])[:, None, :, :]
        return self._causal_mask(q_len, kv_len, device) & same_doc

    def _sdpa_mask(
        self, q_len: int, kv_len: int, device: torch.device
    ) -> tuple[Tensor | None, bool]:
        """Return ``(attn_mask, is_causal)`` for the SDPA call.

        * No cache (``q_len == kv_len``): ``is_causal=True``; the fused kernels take a
          fast path and never materialise the mask.
        * Single-token decode (``q_len == 1``): every cached position is visible, so no
          mask is needed at all. ``is_causal=True`` would be *wrong* here -- SDPA aligns
          the causal triangle top-left, which for one query row means "see only
          position 0".
        * Chunked prefill against a cache: build the offset-aware boolean mask.
        """
        if q_len == kv_len:
            return None, True
        if q_len == 1:
            return None, False
        return self._causal_mask(q_len, kv_len, device), False

    @staticmethod
    def _causal_mask(q_len: int, kv_len: int, device: torch.device) -> Tensor:
        """Boolean ``(1, 1, q_len, kv_len)`` mask, True = attend, offset-aware."""
        offset = kv_len - q_len
        q_idx = torch.arange(q_len, device=device).unsqueeze(-1) + offset
        k_idx = torch.arange(kv_len, device=device).unsqueeze(0)
        return (q_idx >= k_idx)[None, None, :, :]


# -----------------------------------------------------------------------------------
# Benchmark: latency + memory vs sequence length, per backend (§6)
# -----------------------------------------------------------------------------------
def benchmark_attention_backends(
    cfg: ModelConfig,
    seq_lens: Sequence[int],
    seeds: Sequence[int] = (0, 1, 2),
    batch_size: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    iters: int = 3,
    backends: Sequence[AttnBackend] = ATTN_BACKENDS,
) -> list[dict[str, Any]]:
    """Latency and memory for one attention layer at each sequence length.

    One row per ``(backend, seq_len)``, carrying the per-seed samples so the caller can
    bootstrap a CI (CONVENTIONS.md rule 5: never report a bare number).
    """
    dev = torch.device(device)
    rows: list[dict[str, Any]] = []

    for backend in backends:
        for seq_len in seq_lens:
            # RoPE tables are per-max_seq_len and never shared across lengths.
            rope = RotaryEmbedding(cfg.head_dim, seq_len, cfg.rope_theta, device=dev)
            latencies_ms: list[float] = []
            peaks: list[int] = []
            totals: list[int] = []
            observed = "unknown"
            oom = False

            for seed in seeds:
                torch.manual_seed(seed)
                attn = CausalSelfAttention(cfg, backend=backend).to(device=dev, dtype=dtype).eval()
                x = torch.randn(batch_size, seq_len, cfg.d_model, device=dev, dtype=dtype)

                def run(
                    attn: CausalSelfAttention = attn,
                    x: Tensor = x,
                    rope: RotaryEmbedding = rope,
                ) -> None:
                    with torch.no_grad():
                        attn(x, rope)

                try:
                    run()  # warmup
                    ts: list[float] = []
                    for _ in range(iters):
                        if dev.type == "cuda":
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        run()
                        if dev.type == "cuda":
                            torch.cuda.synchronize()
                        ts.append((time.perf_counter() - t0) * 1e3)
                    latencies_ms.append(statistics.median(ts))
                    mem = measure_memory(run, dev.type)
                    peaks.append(mem["peak_bytes"])
                    totals.append(mem["total_alloc_bytes"])
                    if backend != "naive" and observed == "unknown":
                        observed = observed_sdpa_backend(run)
                    elif backend == "naive":
                        observed = "naive_reference"
                except RuntimeError:  # OOM (a RuntimeError subclass) or no-kernel
                    oom = True
                    break

            # The whole point of the plot: naive and math materialise a B x H x T x T
            # score matrix, the fused kernel never does. Reported analytically next to
            # the measured allocation so the quadratic term is unambiguous.
            materialises_scores = backend in ("naive", "sdpa_math")
            score_bytes = (
                batch_size * cfg.n_heads * seq_len * seq_len * torch.finfo(dtype).bits // 8
                if materialises_scores
                else 0
            )

            rows.append(
                {
                    "bench": "attn_backend",
                    "backend": backend,
                    "observed_kernel": observed,
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "dtype": str(dtype).replace("torch.", ""),
                    "score_matrix_bytes_analytic": score_bytes,
                    "oom": oom,
                    "latency_ms_samples": latencies_ms,
                    "peak_bytes_samples": peaks,
                    "total_alloc_bytes_samples": totals,
                }
            )
    return rows
