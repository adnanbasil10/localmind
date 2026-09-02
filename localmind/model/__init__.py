"""LocalMind model package (implementation.md Phase 2).

A from-scratch decoder-only transformer: RMSNorm, RoPE, GQA + QK-norm attention with
three interchangeable backends, SwiGLU, and depth-scaled init.

`config` is importable without torch; everything else needs it.

Public surface
--------------
`ModelConfig`                frozen cross-task config (CONVENTIONS.md)
`count_params`               analytic parameter count, `include_norms` opt-in
`kv_cache_bytes_per_token`   the 16 KB (MHA) vs 4 KB (GQA) number, computed
`LocalMindTransformer`       the model; `forward` returns a `ModelOutput`
`CausalSelfAttention`        `backend in {"naive", "sdpa_math", "sdpa_efficient"}`
`init_weights`               normal(0, init_std), residual-out scaled 1/sqrt(2L)

Not implemented, on purpose (noted as future work): muP (`init.py`), and the §6
stretch goals MoE / MLA / sliding-window attention.
"""

from localmind.model.attention import (
    ATTN_BACKENDS,
    AttnBackend,
    CausalSelfAttention,
    KVCache,
    benchmark_attention_backends,
    observed_sdpa_backend,
    repeat_kv,
)
from localmind.model.block import TransformerBlock
from localmind.model.config import (
    ModelConfig,
    count_norm_params,
    count_params,
    flops_per_forward,
    flops_per_step,
    kv_cache_bytes_per_token,
    non_embedding_params,
    swiglu_hidden_dim,
)
from localmind.model.init import init_weights, residual_scale
from localmind.model.rmsnorm import RMSNorm
from localmind.model.rope import RotaryEmbedding, apply_rope, rotate_half
from localmind.model.swiglu import SwiGLU
from localmind.model.transformer import (
    LocalMindTransformer,
    ModelOutput,
    count_module_params,
    run_benchmarks,
)

__all__ = [
    "ATTN_BACKENDS",
    "AttnBackend",
    "CausalSelfAttention",
    "KVCache",
    "LocalMindTransformer",
    "ModelConfig",
    "ModelOutput",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "TransformerBlock",
    "apply_rope",
    "benchmark_attention_backends",
    "count_module_params",
    "count_norm_params",
    "count_params",
    "flops_per_forward",
    "flops_per_step",
    "init_weights",
    "kv_cache_bytes_per_token",
    "non_embedding_params",
    "observed_sdpa_backend",
    "repeat_kv",
    "residual_scale",
    "rotate_half",
    "run_benchmarks",
    "swiglu_hidden_dim",
]
