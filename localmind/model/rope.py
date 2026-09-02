"""Rotary position embeddings (RoPE), rotate-half form.

Key properties, both tested in `tests/test_model.py`:

* RoPE is applied to **Q and K only, never to V**. Rotating V would destroy the
  content it carries -- one of the four traps called out in implementation.md §6.
* The attention logit between positions `i` and `j` depends only on `i - j`. That is
  the entire point of RoPE and it is asserted directly.

The cos/sin tables are precomputed once per `(head_dim, max_seq_len, theta)` and are
owned by the module instance. They are **not** shared across different `max_seq_len`
(another §6 trap): asking for positions past `max_seq_len` raises rather than silently
wrapping or reading stale values.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["RotaryEmbedding", "apply_rope", "rotate_half"]


def rotate_half(x: Tensor) -> Tensor:
    """`[x1, x2] -> [-x2, x1]` over the last dim (the "rotate-half" convention)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate Q and K in place of their position. **V is never passed here.**

    Args:
        q: `(B, n_heads, T, head_dim)`
        k: `(B, n_kv_heads, T, head_dim)`
        cos, sin: `(1, 1, T, head_dim)`, broadcast over batch and heads.
    """
    cos_q = cos.to(q.dtype)
    sin_q = sin.to(q.dtype)
    cos_k = cos.to(k.dtype)
    sin_k = sin.to(k.dtype)
    q_out = q * cos_q + rotate_half(q) * sin_q
    k_out = k * cos_k + rotate_half(k) * sin_k
    return q_out, k_out


class RotaryEmbedding(nn.Module):
    """Precomputed rotary tables for one `(head_dim, max_seq_len, theta)` triple.

    Buffers are non-persistent: they are derived, not learned, so they stay out of
    checkpoints and a checkpoint trained at `max_seq_len=1024` can be loaded into a
    model built for 2048 (the §6 context-extension anneal) without a shape clash.
    """

    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for rotate-half RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Tables are always built and stored in fp32; they are cast to the activation
        # dtype at apply time. Building them in fp16 loses ~3 decimal digits of angle
        # precision at large positions.
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
        inv_freq = 1.0 / (theta**exponent)  # (head_dim/2,)
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[Tensor, Tensor]:
        """Return `(cos, sin)` of shape `(1, 1, seq_len, head_dim)` for positions
        `[offset, offset + seq_len)`.

        `offset` is what makes incremental (KV-cached) decoding position-correct.
        """
        end = offset + seq_len
        if offset < 0 or seq_len < 0:
            raise ValueError(f"offset and seq_len must be non-negative, got {offset}, {seq_len}")
        if end > self.max_seq_len:
            raise ValueError(
                f"RoPE cache was built for max_seq_len={self.max_seq_len} but positions up to "
                f"{end} were requested. Build a new RotaryEmbedding -- never reuse a cache "
                f"across different max_seq_len."
            )
        return self.cos_cached[:, :, offset:end, :], self.sin_cached[:, :, offset:end, :]

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, theta={self.theta}"
