"""One pre-norm transformer block: attention + SwiGLU, both residual.

Pre-norm (``x + f(norm(x))``) rather than post-norm: it leaves a clean identity path
through the whole stack, which is what lets a from-scratch model train without a
warmup-and-pray learning-rate schedule, and it is far better behaved under fp16.
"""

from __future__ import annotations

from torch import Tensor, nn

from localmind.model.attention import AttnBackend, CausalSelfAttention, KVCache
from localmind.model.config import ModelConfig
from localmind.model.rmsnorm import RMSNorm
from localmind.model.rope import RotaryEmbedding
from localmind.model.swiglu import SwiGLU

__all__ = ["TransformerBlock"]


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: AttnBackend = "sdpa_efficient") -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg, backend=backend)
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg)

    def set_backend(self, backend: AttnBackend) -> None:
        self.attn.set_backend(backend)

    def forward(
        self,
        x: Tensor,
        rope: RotaryEmbedding,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        attn_out, present = self.attn(self.attn_norm(x), rope, past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, present
