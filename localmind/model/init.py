"""Weight initialisation.

`normal(0, init_std)` everywhere, except the two projections that write into the
residual stream -- `attn.wo` and `ffn.down` -- whose std is scaled by
``1 / sqrt(2 * n_layers)``.

Why: with `n_layers` blocks each adding two residual branches, the variance of the
residual stream grows like ``2 * n_layers`` if every branch is initialised the same.
Down-scaling the output projections keeps the stream's variance ~O(1) at depth, which
is what stops the first few hundred steps from being a loss spike (GPT-2 / Megatron
convention).

FUTURE WORK (deliberately not implemented in Phase 2):
  * **muP** -- would let a learning rate tuned on `12m_proxy.yaml` transfer to 31M and
    100M without re-tuning. Under a 30 GPU-h/week quota (§3.2) that is a real budget
    win, not an academic nicety. It needs per-tensor init/LR/output multipliers and a
    coordinate-check test, so it is its own task.
  * MoE / MLA / sliding-window attention (§6 "Stretch") -- also out of scope here.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from localmind.model.config import ModelConfig
from localmind.model.rmsnorm import RMSNorm

__all__ = ["RESIDUAL_OUT_MODULES", "init_weights", "residual_scale"]

#: Module-name suffixes for projections that write into the residual stream.
RESIDUAL_OUT_MODULES: tuple[str, ...] = ("attn.wo", "ffn.down")


def residual_scale(n_layers: int) -> float:
    """``1 / sqrt(2 * n_layers)`` -- two residual branches per block."""
    return 1.0 / math.sqrt(2.0 * n_layers)


def _normal_(t: Tensor, std: float, gen: torch.Generator | None) -> None:
    """Fill ``t`` with ``N(0, std)``.

    When a generator is supplied the samples are drawn on CPU and copied, so a seeded
    init is bit-exact regardless of the device the model happens to live on
    (CONVENTIONS.md: seeded runs must reproduce bit-exactly).
    """
    with torch.no_grad():
        if gen is None:
            t.normal_(0.0, std)
        else:
            sample = torch.empty(t.shape, dtype=torch.float32).normal_(0.0, std, generator=gen)
            t.copy_(sample)


def init_weights(model: nn.Module, cfg: ModelConfig, seed: int | None = None) -> None:
    """Initialise every parameter of ``model`` in place.

    * `nn.Linear` / `nn.Embedding` weights: ``N(0, cfg.init_std)``.
    * `attn.wo` and `ffn.down` weights: ``N(0, cfg.init_std / sqrt(2 * n_layers))``.
    * biases: zero. RMSNorm scales: one.

    Args:
        model: usually a `LocalMindTransformer`, but any module tree works.
        cfg: supplies ``init_std`` and ``n_layers``.
        seed: if given, the init is deterministic and bit-exact across runs/devices.
    """
    gen: torch.Generator | None = None
    if seed is not None:
        seeded = torch.Generator(device="cpu")
        seeded.manual_seed(seed)
        gen = seeded

    std = cfg.init_std
    out_std = std * residual_scale(cfg.n_layers)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            is_residual_out = any(name.endswith(suffix) for suffix in RESIDUAL_OUT_MODULES)
            _normal_(module.weight, out_std if is_residual_out else std, gen)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            _normal_(module.weight, std, gen)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.w)
