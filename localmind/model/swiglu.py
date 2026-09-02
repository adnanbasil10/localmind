"""SwiGLU feed-forward network: ``down(silu(gate(x)) * up(x))``.

Three matrices instead of two, so the hidden width is ``8/3 * d_model`` (rounded up to
a multiple of 128 for tensor-core-friendly shapes) rather than ``4 * d_model``. That
keeps the parameter count matched to a vanilla 4x MLP while buying the gating.
See `localmind.model.config.swiglu_hidden_dim`, which reproduces every `ffn_hidden`
in `configs/model/*.yaml`.
"""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from localmind.model.config import ModelConfig

__all__ = ["SwiGLU"]


class SwiGLU(nn.Module):
    """Gated FFN with a SiLU-activated gate branch.

    ``down`` is the residual-stream output projection, so `localmind.model.init`
    down-scales its init std by ``1/sqrt(2 * n_layers)``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=cfg.bias)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=cfg.bias)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=cfg.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
