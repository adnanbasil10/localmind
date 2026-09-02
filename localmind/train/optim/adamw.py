"""AdamW arm of the §8 optimizer study, and the parameter grouping every arm shares.

Two things live here.

**Parameter grouping.** §8 says ``wd=0.1 (exclude norms/bias/embeddings)``, and §8's
trap list names "weight decay on embeddings and norms" explicitly. Decaying an RMSNorm
scale pulls it toward 0, which scales the whole residual branch toward 0; decaying a
token embedding penalises rare tokens hardest, because they receive gradient least
often and decay every step regardless. `split_param_groups` implements the split by
*module type*, not by a name heuristic -- `p.ndim >= 2` would wrongly decay the tied
embedding matrix, which is 8.4M of a 31M model.

**The optimizer itself is `torch.optim.AdamW`, deliberately.** Hand-rolling Adam here
would be strictly worse: torch's has a `fused=True` CUDA path (one kernel for the whole
parameter list, which matters when the step is a rounding error next to the forward on
a small model) and its correctness is not in question. The content of this file is the
grouping and the LR-scale plumbing, which is the part that is actually project-specific.

Every group carries an ``lr_scale``. `set_lr` multiplies the schedule's LR by it, which
is how the study's three arms stay one code path:

    1. ``adamw``      -- lr_scale 1.0 everywhere (baseline)
    2. ``muon``       -- see `muon.py`; hidden matrices get their own group
    3. ``adamw_2x``   -- lr_scale 2.0 everywhere (the control for "Muon just likes
                         bigger steps"). Scaling the group and not the schedule keeps
                         the *shape* of the WSD curve identical between arms, so the
                         only difference is magnitude.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from localmind.model.rmsnorm import RMSNorm

__all__ = [
    "ParamGroupSpec",
    "build_adamw",
    "fused_adamw_available",
    "iter_trainable_params",
    "set_lr",
    "split_param_groups",
]

ParamGroupSpec = dict[str, Any]


def _no_decay_param_ids(model: nn.Module) -> set[int]:
    """Ids of every parameter that must NOT receive weight decay.

    Norm scales, biases, and embeddings. With ``tie_embeddings: true`` the LM head *is*
    the embedding (one tensor, two names), so it lands here once and is not decayed --
    which is the intended reading of "exclude ... embeddings".
    """
    ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, RMSNorm | nn.LayerNorm | nn.GroupNorm | nn.Embedding):
            ids.update(id(p) for p in module.parameters(recurse=False))
    for p in model.parameters():
        if p.ndim < 2:  # every bias, and any stray 1-D scale
            ids.add(id(p))
    return ids


def iter_trainable_params(model: nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """``(name, param)`` for trainable params, tied weights yielded **once**.

    A tied parameter appearing twice in an optimizer's param list would have its
    gradient applied twice per step -- a silent 2x LR on 27% of the model.
    """
    seen: set[int] = set()
    out: list[tuple[str, torch.nn.Parameter]] = []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        out.append((name, p))
    return out


def split_param_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
    lr_scale: float = 1.0,
) -> list[ParamGroupSpec]:
    """Two groups: ``decay`` (hidden matrices) and ``no_decay`` (norms/bias/embeddings).

    Args:
        weight_decay: applied to the ``decay`` group only.
        lr_scale: stamped on both groups; `set_lr` multiplies the schedule by it.

    Empty groups are dropped -- torch accepts them but they make `state_dict` diffs
    across arms confusing.
    """
    skip = _no_decay_param_ids(model)
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for _name, p in iter_trainable_params(model):
        (no_decay if id(p) in skip else decay).append(p)

    groups: list[ParamGroupSpec] = []
    if decay:
        groups.append(
            {"name": "decay", "params": decay, "weight_decay": weight_decay, "lr_scale": lr_scale}
        )
    if no_decay:
        groups.append(
            {"name": "no_decay", "params": no_decay, "weight_decay": 0.0, "lr_scale": lr_scale}
        )
    return groups


def fused_adamw_available(params: Sequence[torch.nn.Parameter]) -> bool:
    """`fused=True` needs a CUDA build, float params, and every param on CUDA.

    On the CPU-only dev box this is always False, which is why it is a function and not
    an `if torch.cuda.is_available()` scattered through the call site.
    """
    if not torch.cuda.is_available():
        return False
    if "fused" not in inspect.signature(torch.optim.AdamW).parameters:
        return False
    return all(
        p.is_cuda and p.dtype in (torch.float32, torch.float64, torch.float16) for p in params
    )


def build_adamw(
    model: nn.Module,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    eps: float = 1e-8,
    lr_scale: float = 1.0,
    fused: bool | None = None,
) -> Optimizer:
    """AdamW over `split_param_groups(model)`.

    Args:
        lr: the *peak* LR. `set_lr` overwrites it every step from the schedule; this
            value only matters for the zeroth step before the loop calls `set_lr`.
        betas: ``(0.9, 0.95)`` per §8 -- beta2 0.95 rather than 0.999 because a 5.7k-step
            run never accumulates enough history for 0.999 to be anything but stale.
        lr_scale: 2.0 gives the study's "AdamW at 2x LR" control arm.
        fused: ``None`` auto-detects (see `fused_adamw_available`).
    """
    groups = split_param_groups(model, weight_decay=weight_decay, lr_scale=lr_scale)
    flat = [p for g in groups for p in g["params"]]
    use_fused = fused_adamw_available(flat) if fused is None else fused
    kwargs: dict[str, Any] = {"lr": lr, "betas": betas, "eps": eps}
    if use_fused:
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)


def set_lr(optimizers: Optimizer | Iterable[Optimizer], lr: float) -> None:
    """Write ``lr * group["lr_scale"]`` into every param group of every optimizer.

    Accepts one optimizer or several, because the Muon arm is two optimizers (Muon on
    the hidden matrices, AdamW on embeddings/head/norms) that must move in lockstep on
    one shared WSD schedule.
    """
    opts = [optimizers] if isinstance(optimizers, Optimizer) else list(optimizers)
    for opt in opts:
        for group in opt.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
