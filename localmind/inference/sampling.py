"""Token sampling: logits -> one token id, deterministically given a seed.

Everything stochastic in this package funnels through :class:`SamplingParams` and
:func:`sample_token`, so a seeded run reproduces bit-exactly (CONVENTIONS.md
"Determinism").  The warpers are applied in the canonical order used by every
production server -- repetition penalty, then temperature, then top-k, then top-p,
then min-p -- because the order changes the distribution and an undocumented order is
an unreproducible one.

``temperature == 0.0`` is greedy argmax and never touches the RNG.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

__all__ = [
    "SamplingParams",
    "apply_min_p",
    "apply_repetition_penalty",
    "apply_top_k",
    "apply_top_p",
    "make_generator",
    "prepare_logits",
    "sample_batch",
    "sample_token",
    "sampling_params_from_openai",
]


class SamplingParams(BaseModel):
    """Per-request decoding configuration.

    Mirrors the OpenAI request surface where the names overlap so ``server.py`` can map
    a request onto this without a translation table full of magic.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_new_tokens: int = Field(default=64, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    min_p: float = Field(default=0.0, ge=0.0, lt=1.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    seed: int = 0
    stop_token_ids: tuple[int, ...] = ()
    logit_bias: dict[int, float] = Field(default_factory=dict)
    ignore_eos: bool = False

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """A private RNG stream. Never ``torch.manual_seed`` -- that is global mutable state."""
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def apply_repetition_penalty(logits: Tensor, prev_ids: Sequence[int], penalty: float) -> Tensor:
    """CTRL-style penalty (Keskar et al. 2019): divide positive logits, multiply negatives.

    Naively dividing every logit by ``penalty`` *rewards* tokens whose logit is negative,
    which is the classic sign bug. Branching on the sign is the fix.
    """
    if penalty == 1.0 or len(prev_ids) == 0:
        return logits
    unique = sorted({int(i) for i in prev_ids})
    idx = torch.tensor(unique, dtype=torch.long, device=logits.device)
    seen = logits.index_select(-1, idx)
    updated = torch.where(seen > 0, seen / penalty, seen * penalty)
    return logits.index_copy(-1, idx, updated)


def apply_top_k(logits: Tensor, k: int) -> Tensor:
    """Keep the k highest logits, -inf the rest. ``k <= 0`` disables."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def apply_top_p(logits: Tensor, p: float) -> Tensor:
    """Nucleus sampling (Holtzman et al. 2020). ``p >= 1.0`` disables.

    The token that *crosses* the threshold is kept, so the retained mass is always
    >= p and the set is never empty even when one token holds more than p.
    """
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    # Shift right: a token is removed only if the mass *before* it already reached p.
    remove = (cumulative - probs) > p
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)


def apply_min_p(logits: Tensor, min_p: float) -> Tensor:
    """Min-p (Nguyen et al. 2024): keep tokens with ``p >= min_p * p_max``.

    Adapts the cut to how peaked the distribution is, which is why it degrades more
    gracefully than top-p at high temperature.
    """
    if min_p <= 0.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    threshold = min_p * probs.amax(dim=-1, keepdim=True)
    return logits.masked_fill(probs < threshold, float("-inf"))


def prepare_logits(
    logits: Tensor,
    params: SamplingParams,
    prev_ids: Sequence[int] = (),
    extra_mask: Tensor | None = None,
) -> Tensor:
    """Everything between the model's raw logits and a sampleable distribution.

    Args:
        logits: ``(..., vocab)`` raw scores. Cast to fp32 here -- sampling in fp16 is a
            silent quality bug that only shows up in the tail.
        params: sampling configuration.
        prev_ids: tokens already in the sequence, for the repetition penalty.
        extra_mask: additive ``(..., vocab)`` mask (0 keep / -inf ban). This is the hook
            ``constrained.py`` uses to make invalid JSON unreachable.
    """
    out = logits.float()
    if params.logit_bias:
        idx = torch.tensor(list(params.logit_bias), dtype=torch.long, device=out.device)
        bias = torch.tensor(list(params.logit_bias.values()), dtype=out.dtype, device=out.device)
        out = out.index_add(-1, idx, bias)
    if extra_mask is not None:
        out = out + extra_mask
    out = apply_repetition_penalty(out, prev_ids, params.repetition_penalty)
    if params.greedy:
        return out
    out = out / params.temperature
    out = apply_top_k(out, params.top_k)
    out = apply_top_p(out, params.top_p)
    out = apply_min_p(out, params.min_p)
    return out


def sample_token(
    logits: Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
    prev_ids: Sequence[int] = (),
    extra_mask: Tensor | None = None,
) -> int:
    """Sample one token id from a ``(vocab,)`` or ``(1, vocab)`` logit row."""
    row = logits.reshape(-1)
    processed = prepare_logits(row, params, prev_ids, extra_mask)
    if params.greedy:
        return int(torch.argmax(processed).item())
    probs = torch.softmax(processed, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def sample_batch(
    logits: Tensor,
    params_list: Sequence[SamplingParams],
    generators: Sequence[torch.Generator | None],
    prev_ids_list: Sequence[Sequence[int]],
) -> list[int]:
    """Per-row sampling for a batched decode step.

    Rows are sampled independently rather than vectorised: each request carries its own
    ``SamplingParams`` and its own RNG stream, and folding those into one ``multinomial``
    call would break per-request determinism the moment batch composition changes.
    """
    return [
        sample_token(logits[i], params, generators[i], prev_ids_list[i])
        for i, params in enumerate(params_list)
    ]


def sampling_params_from_openai(payload: dict[str, Any]) -> SamplingParams:
    """Map an OpenAI-shaped request body onto :class:`SamplingParams`.

    OpenAI's ``temperature=0`` means greedy, which is exactly our convention. ``n > 1``
    is handled by the caller (one ``SamplingParams`` per choice, different seeds).
    """
    stop_ids = payload.get("stop_token_ids") or ()
    bias_raw = payload.get("logit_bias") or {}
    return SamplingParams(
        max_new_tokens=int(payload.get("max_tokens") or payload.get("max_new_tokens") or 64),
        temperature=float(payload.get("temperature", 0.0) or 0.0),
        top_p=float(payload.get("top_p", 1.0) or 1.0),
        top_k=int(payload.get("top_k", 0) or 0),
        min_p=float(payload.get("min_p", 0.0) or 0.0),
        repetition_penalty=float(payload.get("repetition_penalty", 1.0) or 1.0),
        seed=int(payload.get("seed") or 0),
        stop_token_ids=tuple(int(s) for s in stop_ids),
        logit_bias={int(k): float(v) for k, v in bias_raw.items()},
    )
