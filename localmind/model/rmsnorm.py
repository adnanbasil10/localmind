"""RMSNorm -- no mean subtraction, no bias, one learnable scale.

The fp32 cast in `forward` is **load-bearing, not a decoration** (implementation.md
§3.2): the target GPU is a T4 (SM 7.5) with no bf16, so training runs under fp16
autocast. `x.pow(2).mean(-1)` in fp16 overflows at |x| ~ 256 and silently produces
inf -> NaN. Reducing in fp32 and casting back is what keeps a 12-hour run alive.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["RMSNorm"]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019).

    `y = x / sqrt(mean(x^2) + eps) * w`

    No mean subtraction and no bias: RMSNorm keeps only the re-scaling invariance of
    LayerNorm, which is the part that matters, at ~2x lower cost.

    Args:
        d: size of the last dimension being normalised. For block norms this is
            `d_model`; for QK-norm it is `head_dim` (per-head normalisation).
        eps: added inside the sqrt for numerical stability.
    """

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dt) * self.w

    def extra_repr(self) -> str:
        return f"d={self.w.numel()}, eps={self.eps}"
