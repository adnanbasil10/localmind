"""Model configuration and torch-free analytic accounting helpers.

This module deliberately does **not** import torch: every other phase (data, agent,
retrieval) needs to read a `ModelConfig` and reason about parameter counts / KV-cache
budgets without paying for a torch import.

`ModelConfig` is a frozen cross-task interface (see `CONVENTIONS.md` -> "Shared
interfaces"). Fields may be *added* here, never removed or renamed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ModelConfig",
    "count_norm_params",
    "count_params",
    "flops_per_forward",
    "flops_per_step",
    "kv_cache_bytes_per_token",
    "non_embedding_params",
    "swiglu_hidden_dim",
]


class ModelConfig(BaseModel):
    """Architecture hyperparameters for a LocalMind transformer.

    Loaded from `configs/model/*.yaml`. No architecture magic numbers live in code.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    vocab_size: int = Field(gt=0)
    d_model: int = Field(gt=0)
    n_layers: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    n_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    ffn_hidden: int = Field(gt=0)
    max_seq_len: int = Field(gt=0)
    rope_theta: float = Field(gt=0.0)
    qk_norm: bool
    bias: bool
    tie_embeddings: bool
    z_loss: float = Field(ge=0.0)
    attn_dropout: float = Field(ge=0.0, lt=1.0)
    init_std: float = Field(gt=0.0)

    # --- fields beyond the frozen surface, present in configs/model/*.yaml ----------
    # Only one value of each is implemented in Phase 2; they exist so the YAML is
    # self-documenting and so future ablations can switch on them.
    norm: Literal["rmsnorm"] = "rmsnorm"
    activation: Literal["swiglu"] = "swiglu"
    rms_norm_eps: float = Field(default=1e-6, gt=0.0)
    #: `count_params(cfg, include_norms=False)` must equal this. See the note on the
    #: spec's norm-count error in `count_params`.
    expected_params: int | None = None

    @model_validator(mode="after")
    def _check_shapes(self) -> ModelConfig:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError(
                f"n_heads * head_dim ({self.n_heads} * {self.head_dim} = "
                f"{self.n_heads * self.head_dim}) must equal d_model ({self.d_model})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for rotate-half RoPE")
        return self

    # --- derived -------------------------------------------------------------------
    @property
    def n_rep(self) -> int:
        """How many query heads share one KV head (GQA group size). 1 == MHA."""
        return self.n_heads // self.n_kv_heads

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def is_mha(self) -> bool:
        return self.n_kv_heads == self.n_heads

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        """Load a config from YAML. Unknown keys are a hard error (`extra="forbid"`)."""
        with Path(path).open("r", encoding="utf-8") as fh:
            raw: Any = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
        return cls.model_validate(raw)


# -------------------------------------------------------------------------------------
# Parameter accounting
# -------------------------------------------------------------------------------------
def count_params(cfg: ModelConfig, include_norms: bool = False) -> int:
    """Analytic parameter count (no torch, no instantiation).

    Args:
        cfg: model config.
        include_norms: if False (default) count only embedding + attention + SwiGLU,
            which is what the spec's DoD assert equals. If True, additionally count
            every RMSNorm scale vector (the honest total).

    NOTE ON A SPEC ERROR (implementation.md §6, "Parameter accounting" table).
    The table lists "Norms ~33,000" as a component of the ~30.9M total, but the DoD
    assert is `count_params(cfg) == 30_932_992`, and

        30_932_992 = 8_388_608 (embedding) + 5_242_880 (attention) + 17_301_504 (SwiGLU)

    exactly -- i.e. the DoD number *excludes* norms. The "~33,000" figure is also simply
    wrong: LocalMind-31M has 8 * (2 block norms * 512 + q/k QK-norms * 64) + 512 final
    = 9,728 norm parameters, not ~33,000. Ruling (controller): `include_norms=False` is
    the default and reproduces the DoD number 30,932,992; `include_norms=True` returns
    the honest 30,942,720. Both are asserted in `tests/test_model.py`.
    """
    embedding = cfg.vocab_size * cfg.d_model
    if not cfg.tie_embeddings:
        embedding += cfg.vocab_size * cfg.d_model

    # Wq + Wo are d_model x (n_heads * head_dim); Wk + Wv are d_model x (n_kv_heads * head_dim).
    attn_per_layer = 2 * cfg.d_model * cfg.q_dim + 2 * cfg.d_model * cfg.kv_dim
    # SwiGLU: gate + up (d -> hidden) and down (hidden -> d).
    ffn_per_layer = 3 * cfg.d_model * cfg.ffn_hidden

    total = embedding + cfg.n_layers * (attn_per_layer + ffn_per_layer)

    if cfg.bias:
        total += cfg.n_layers * (2 * cfg.q_dim + 2 * cfg.kv_dim + 2 * cfg.ffn_hidden + cfg.d_model)

    if include_norms:
        total += count_norm_params(cfg)
    return total


def count_norm_params(cfg: ModelConfig) -> int:
    """RMSNorm scales: 2 per block, +2 per block for QK-norm, +1 final norm."""
    per_layer = 2 * cfg.d_model
    if cfg.qk_norm:
        per_layer += 2 * cfg.head_dim
    return cfg.n_layers * per_layer + cfg.d_model


def non_embedding_params(cfg: ModelConfig, include_norms: bool = False) -> int:
    """Params excluding the (tied) token embedding -- the number that drives FLOPs."""
    return count_params(cfg, include_norms=include_norms) - cfg.vocab_size * cfg.d_model


# -------------------------------------------------------------------------------------
# KV cache accounting  (§6 benchmarks: MHA 16 KB/token vs GQA 4 KB/token)
# -------------------------------------------------------------------------------------
def kv_cache_bytes_per_token(
    cfg: ModelConfig,
    n_kv_heads: int | None = None,
    dtype_bytes: int = 2,
) -> int:
    """Bytes of KV cache consumed per token of context, for the whole model.

    K and V are each `n_kv_heads * head_dim` wide, per layer, per token.

    Args:
        cfg: model config (supplies head_dim / n_layers, and the default n_kv_heads).
        n_kv_heads: override, so the same config can be priced as MHA (`cfg.n_heads`)
            or GQA (`cfg.n_kv_heads`) without editing the YAML. Defaults to
            `cfg.n_kv_heads`.
        dtype_bytes: 2 for fp16 (the T4 target; bf16 is forbidden by CONVENTIONS.md).
    """
    kv_heads = cfg.n_kv_heads if n_kv_heads is None else n_kv_heads
    if kv_heads <= 0 or cfg.n_heads % kv_heads != 0:
        raise ValueError(f"n_kv_heads={kv_heads} must divide n_heads={cfg.n_heads}")
    return 2 * kv_heads * cfg.head_dim * cfg.n_layers * dtype_bytes


# -------------------------------------------------------------------------------------
# Shape / FLOP helpers
# -------------------------------------------------------------------------------------
def swiglu_hidden_dim(d_model: int, multiple_of: int = 128, expansion: float = 8 / 3) -> int:
    """`8/3 * d_model` rounded **up** to a multiple of `multiple_of`.

    SwiGLU uses three matrices instead of two, so 8/3 (not 4) keeps FFN params matched
    to a vanilla 4x MLP. Reproduces every `ffn_hidden` in `configs/model/*.yaml`.
    """
    target = expansion * d_model
    return int(math.ceil(target / multiple_of) * multiple_of)


def flops_per_forward(cfg: ModelConfig, batch_size: int, seq_len: int) -> int:
    """FLOPs for one forward pass (one multiply-add = 2 FLOPs).

    Dense matmuls contribute `2 * params * tokens`; the two attention batched matmuls
    (QK^T and AV) contribute `4 * L * T^2 * d_model` per sequence and are counted
    separately because they scale quadratically in T, not linearly.
    """
    tokens = batch_size * seq_len
    dense = 2 * non_embedding_params(cfg) * tokens
    lm_head = 2 * cfg.vocab_size * cfg.d_model * tokens
    attn = 4 * cfg.n_layers * seq_len * seq_len * cfg.d_model * batch_size
    return dense + lm_head + attn


def flops_per_step(cfg: ModelConfig, batch_size: int, seq_len: int) -> int:
    """Forward + backward. Backward is ~2x forward (grad wrt inputs and wrt weights)."""
    return 3 * flops_per_forward(cfg, batch_size, seq_len)
