"""Model FLOPs Utilisation, tokens/sec, and bits-per-byte.

The §8 formula, verbatim::

    flops_per_token = 6 * params + 12 * n_layers * d_model * ctx
    # T4 fp16 tensor-core peak ~= 65e12

RECONCILIATION with `localmind.model.config.flops_per_step` (which this module reuses
rather than duplicates). That function counts

    fwd = 2 * non_embedding_params * tokens        # dense matmuls
        + 2 * vocab_size * d_model * tokens        # the LM head
        + 4 * n_layers * T^2 * d_model * B         # QK^T and AV
    step = 3 * fwd                                 # backward is ~2x forward

Divide by ``tokens = B * T`` and you get, per token::

    6 * non_embedding_params + 6 * vocab_size * d_model + 12 * n_layers * d_model * T

For a **weight-tied** model ``count_params(cfg) == non_embedding_params + vocab*d_model``
exactly, so that expression is *identically* the spec's
``6 * params + 12 * n_layers * d_model * ctx``. The two agree to the last FLOP; see
`tests/test_train.py::test_spec_flops_formula_matches_model_config`.

For an **untied** model they diverge: `count_params` counts the embedding table twice
(input embedding + LM head) while the forward pass only pays matmul FLOPs for the LM
head -- an embedding *lookup* is a gather, not a matmul, and costs ~0 FLOPs. So the
spec's literal formula over-counts an untied model by ``6 * vocab_size * d_model`` per
token. `flops_per_token_for_config` uses the honest count; `flops_per_token` is the
spec's literal formula and is what `mfu()` documents itself against. All three model
configs in `configs/model/` set ``tie_embeddings: true``, so in practice they coincide.

Expect 10-15% MFU on a first T4 run (§8). 25%+ is an explicit sub-project, not a given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from localmind.model.config import (
    ModelConfig,
    count_params,
    flops_per_step,
    non_embedding_params,
)

__all__ = [
    "A100_BF16_PEAK_FLOPS",
    "P100_FP16_PEAK_FLOPS",
    "T4_FP16_PEAK_FLOPS",
    "T4_FP32_PEAK_FLOPS",
    "ThroughputReport",
    "bits_per_byte",
    "device_peak_flops",
    "flops_per_token",
    "flops_per_token_for_config",
    "flops_per_token_from_step_counter",
    "mfu",
    "perplexity",
    "throughput_report",
]

#: §8 / §3.2: the free Kaggle GPU is a Turing T4. fp16 tensor-core peak, dense.
T4_FP16_PEAK_FLOPS: float = 65e12
#: Non-tensor-core fp32 on the same card, for the `precision: fp32` config path.
T4_FP32_PEAK_FLOPS: float = 8.1e12
#: The other card Kaggle sometimes hands you (1x P100, no tensor cores at all).
P100_FP16_PEAK_FLOPS: float = 19.05e12
#: Present only so the "we are not on Ampere" comparison in the README has a number.
#: bf16 is forbidden on this project (ADR 0001); this constant is documentation.
A100_BF16_PEAK_FLOPS: float = 312e12

_PEAK_BY_DEVICE: tuple[tuple[str, float], ...] = (
    ("t4", T4_FP16_PEAK_FLOPS),
    ("p100", P100_FP16_PEAK_FLOPS),
    ("v100", 125e12),
    ("a100", A100_BF16_PEAK_FLOPS),
    ("l4", 121e12),
)


def flops_per_token(params: int, n_layers: int, d_model: int, ctx: int) -> float:
    """The §8 formula, verbatim. Forward + backward, per token.

    ``6 * params`` is 2 FLOPs per multiply-add x 3 (one forward + two backward passes)
    over every weight; ``12 * n_layers * d_model * ctx`` is the quadratic attention term
    (QK^T and AV), which the parameter count cannot capture because those matmuls have
    no weights.
    """
    return 6.0 * params + 12.0 * n_layers * d_model * ctx


def flops_per_token_for_config(cfg: ModelConfig, ctx: int | None = None) -> float:
    """`flops_per_token` for a `ModelConfig`, honest about untied embeddings.

    Equals `flops_per_token(count_params(cfg), ...)` exactly when
    ``cfg.tie_embeddings`` (which every shipped config sets). See the module docstring.

    When untied, `count_params` counts the embedding table twice and
    `non_embedding_params` drops exactly one copy -- leaving the LM head (a real matmul)
    plus the hidden layers, which is the honest FLOP-bearing parameter count. The input
    embedding is a gather, not a matmul, and costs ~0 FLOPs.
    """
    ctx = cfg.max_seq_len if ctx is None else ctx
    params = count_params(cfg) if cfg.tie_embeddings else non_embedding_params(cfg)
    return flops_per_token(params, cfg.n_layers, cfg.d_model, ctx)


def mfu(
    params: int,
    n_layers: int,
    d_model: int,
    ctx: int,
    tokens_per_sec: float,
    peak_flops: float = T4_FP16_PEAK_FLOPS,
) -> float:
    """Model FLOPs Utilisation as a fraction in ``[0, 1]``. §8, verbatim.

    Args:
        peak_flops: **aggregate across all GPUs in the job.** On 2xT4 that is
            ``2 * 65e12``; pass `device_peak_flops(..., world_size=2)`.
        tokens_per_sec: likewise aggregate (sum over ranks).
    """
    if peak_flops <= 0.0:
        raise ValueError(f"peak_flops must be positive, got {peak_flops}")
    return flops_per_token(params, n_layers, d_model, ctx) * tokens_per_sec / peak_flops


def device_peak_flops(
    device_name: str | None = None,
    precision: str = "fp16",
    world_size: int = 1,
) -> float:
    """Aggregate peak FLOP/s for the job, matched by GPU name substring.

    Unknown devices (and CPU, where "peak FLOPs" is not a number anyone can defend)
    fall back to the T4 number so MFU stays comparable across environments -- a CPU MFU
    figure is meaningless either way and is logged only so the code path is exercised.
    """
    name = (device_name or "").lower()
    peak = T4_FP16_PEAK_FLOPS
    for key, value in _PEAK_BY_DEVICE:
        if key in name:
            peak = value
            break
    if precision == "fp32" and "t4" in name:
        peak = T4_FP32_PEAK_FLOPS
    return peak * max(1, world_size)


def bits_per_byte(ce_loss_nats: float, bytes_per_token: float) -> float:
    """Bits-per-byte from a mean per-token cross-entropy in nats.

    ``BPB = (nats / ln 2) / bytes_per_token``. §5: BPB is the only tokenizer-invariant
    loss metric, so it is the number to quote whenever two runs used different
    tokenizers -- which the vocab ablation guarantees they will.

    ``bytes_per_token`` comes from the data pipeline (UTF-8 bytes of the held-out text
    divided by the tokens it encodes to), not from a constant in this file.
    """
    if bytes_per_token <= 0.0:
        raise ValueError(f"bytes_per_token must be positive, got {bytes_per_token}")
    return ce_loss_nats / math.log(2.0) / bytes_per_token


def perplexity(ce_loss_nats: float) -> float:
    """``exp(loss)``. Clamped so an early-training loss of 12 does not overflow a log line."""
    return math.exp(min(ce_loss_nats, 20.0))


@dataclass(frozen=True)
class ThroughputReport:
    """One step's worth of systems numbers. Everything here is host-side arithmetic."""

    tokens_per_sec: float
    step_time_s: float
    mfu: float
    flops_per_token: float
    achieved_flops: float
    peak_flops: float

    def as_dict(self) -> dict[str, float]:
        return {
            "tokens_per_sec": self.tokens_per_sec,
            "step_time_s": self.step_time_s,
            "mfu": self.mfu,
            "flops_per_token": self.flops_per_token,
            "achieved_flops": self.achieved_flops,
            "peak_flops": self.peak_flops,
        }


def throughput_report(
    cfg: ModelConfig,
    tokens_this_step: int,
    step_time_s: float,
    ctx: int | None = None,
    peak_flops: float = T4_FP16_PEAK_FLOPS,
) -> ThroughputReport:
    """Bundle tokens/sec + MFU for one optimizer step.

    ``tokens_this_step`` is the **global** token count for the step (all ranks, all
    grad-accumulation micro-batches), so the result is job-level, matching
    `device_peak_flops(..., world_size=N)`.
    """
    if step_time_s <= 0.0:
        raise ValueError(f"step_time_s must be positive, got {step_time_s}")
    fpt = flops_per_token_for_config(cfg, ctx)
    tps = tokens_this_step / step_time_s
    achieved = fpt * tps
    return ThroughputReport(
        tokens_per_sec=tps,
        step_time_s=step_time_s,
        mfu=achieved / peak_flops,
        flops_per_token=fpt,
        achieved_flops=achieved,
        peak_flops=peak_flops,
    )


def flops_per_token_from_step_counter(cfg: ModelConfig, batch_size: int, seq_len: int) -> float:
    """`model.config.flops_per_step` reduced to a per-token number.

    Exists so the equivalence asserted in the module docstring is executable rather
    than a claim: for a tied model this returns exactly
    `flops_per_token(count_params(cfg), cfg.n_layers, cfg.d_model, seq_len)`.
    """
    return flops_per_step(cfg, batch_size, seq_len) / (batch_size * seq_len)
