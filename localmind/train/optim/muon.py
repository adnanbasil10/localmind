"""Muon: momentum orthogonalised by Newton-Schulz, plus the Kimi K2 hybrid split.

§8 optimizer study, arm 2: *"Muon on all 2D hidden matrices + AdamW on
embeddings/LM-head/norms (the Kimi K2 recipe)"*.

## What Muon actually does

SGD-with-momentum proposes an update ``M`` for a weight matrix ``W``. Muon replaces
``M`` with the nearest **semi-orthogonal** matrix -- ``U V^T`` where ``M = U S V^T`` is
the SVD -- and steps with that instead.

The motivation is that a momentum matrix for a hidden layer is typically dominated by a
few large singular directions, so plain SGD spends almost the whole step budget moving
along them and starves every other direction. Orthogonalising equalises the singular
values, so every direction gets a step of comparable size. Empirically this buys a
large constant-factor reduction in tokens-to-target-loss on transformer hidden layers.

Computing an SVD every step would be absurd, so Muon uses a **quintic Newton-Schulz
iteration** instead: five matmuls-per-step of fixed cost, no decomposition. The
coefficients ``(a, b, c) = (3.4445, -4.7750, 2.0315)`` are Jordan's, tuned to converge
in ~5 steps at the cost of leaving singular values in a band around 1 rather than
exactly 1. That is fine -- what matters is the *relative* equalisation, not landing on
the exact Stiefel manifold. `newton_schulz_error` measures how far off it is, and
`tests/test_train.py` asserts the band.

## Why the hybrid split (and not Muon on everything)

Muon's argument only holds for matrices whose two dimensions are both "hidden". It
breaks for:

* **Embeddings / LM head** -- one axis is the vocabulary. Rows are per-token and are
  updated at wildly different frequencies; orthogonalising across the vocab axis mixes
  a common token's update into a rare token's row. Kimi K2 and Moonlight both keep
  AdamW here, and with ``tie_embeddings: true`` this single matrix is 8.4M of a 31M
  model, so getting it wrong is not a rounding error.
* **Norm scales and biases** -- 1-D. There is nothing to orthogonalise.

So the run carries two optimizers on one shared WSD schedule. `set_lr` drives both.

## T4-specific deviation from the reference implementation (ADR 0001)

Every published Muon implementation runs the Newton-Schulz iteration in **bfloat16** --
it is a bandwidth-bound sequence of small matmuls and bf16 halves the traffic for free.
**We cannot.** Turing (SM 7.5) has no bf16 tensor cores; ADR 0001 makes "bf16 in a
commit is a bug" a hard rule. `NS_DTYPES` therefore admits float32 (the default) and
float16 only, and `newton_schulz` raises on bfloat16 rather than silently emulating it.

fp32 is the right default anyway: the iteration opens with ``X / ||X||_F``, and a
gradient matrix late in a run can have a Frobenius norm small enough that fp16 flushes
the quotient to subnormals. The matrices here are at most 1408x512, so five fp32
matmuls per parameter per step is noise next to the forward pass.

## Distributed

Under DDP the gradient is all-reduced *before* `step`, so every rank orthogonalises an
identical matrix and produces an identical update. No extra communication is required
and the ranks stay bit-identical. The cost is that the Newton-Schulz work is done
redundantly on every rank; sharding it across ranks (as Keller Jordan's distributed
implementation does) is a throughput optimisation left undone, and is noted as such in
the task report rather than half-implemented.

## Not implemented: MuonClip

Kimi K2 pairs Muon with *QK-clip*, rescaling ``W_q``/``W_k`` whenever the max attention
logit exceeds a threshold, because Muon at scale drives attention logits up. LocalMind
already applies **QK-norm** (ADR 0001 makes it load-bearing on fp16 hardware), which
bounds the same quantity by construction and unconditionally. Stacking both would make
it impossible to attribute stability to either.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from localmind.model.rmsnorm import RMSNorm
from localmind.train.optim.adamw import ParamGroupSpec, build_adamw, iter_trainable_params

__all__ = [
    "NS_COEFFS",
    "NS_DTYPES",
    "Muon",
    "MuonParamSplit",
    "UpdateScale",
    "build_muon_hybrid",
    "build_optimizer_arm",
    "muon_update_scale",
    "newton_schulz",
    "newton_schulz_error",
    "split_muon_params",
]

#: Jordan's quintic coefficients: ``X <- a X + (b A + c A^2) X`` with ``A = X X^T``.
#: Chosen to maximise the slope at 0 so that tiny singular values are lifted fast, which
#: is what lets 5 iterations do the job. The price is overshoot: converged singular
#: values sit in a band around 1 instead of on it.
NS_COEFFS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)

#: ADR 0001: bfloat16 is absent from this tuple on purpose, and `newton_schulz`
#: rejects it explicitly rather than letting it through as "some other dtype".
NS_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float64, torch.float16)

UpdateScale = Literal["moonlight", "jordan", "none"]


def newton_schulz(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Approximate ``U V^T`` for ``G = U S V^T``, without an SVD.

    Args:
        G: a 2-D matrix (a momentum buffer, not a weight).
        steps: Newton-Schulz iterations. 5 is the reference value; more tightens the
            singular-value band at 3 matmuls each.
        eps: guards the opening ``G / ||G||_F`` against an all-zero gradient.
        dtype: working precision. **float32 by default and bfloat16 forbidden** -- see
            the module docstring and ADR 0001.

    Returns:
        A matrix the same shape and dtype as ``G`` whose singular values are all
        approximately 1 (see `newton_schulz_error` for how approximately).
    """
    if G.ndim != 2:
        raise ValueError(f"newton_schulz expects a 2-D matrix, got shape {tuple(G.shape)}")
    if dtype is torch.bfloat16:
        raise ValueError(
            "bfloat16 is forbidden project-wide (ADR 0001): the target GPU is a Turing T4 "
            "(SM 7.5) with no bf16 tensor cores. Use torch.float32 (default) or torch.float16."
        )
    if dtype not in NS_DTYPES:
        raise ValueError(f"dtype must be one of {NS_DTYPES}, got {dtype}")
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    a, b, c = NS_COEFFS
    X = G.to(dtype)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        # Iterate on the short side: A = X X^T is then (min_dim, min_dim), which for a
        # 1408x512 FFN matrix is a 512x512 matmul instead of a 1408x1408 one.
        X = X.mT
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


def newton_schulz_error(G: Tensor, steps: int = 5) -> tuple[float, float]:
    """``(min, max)`` singular value of ``newton_schulz(G)`` -- the convergence diagnostic.

    A perfectly orthogonalised matrix returns ``(1.0, 1.0)``. Jordan's coefficients
    trade exactness for speed, so 5 steps on a well-conditioned matrix land in roughly
    ``(0.7, 1.3)``. This is asserted in the tests: it is the single check that would
    catch a transposed matmul or a sign error in the coefficients, both of which are
    otherwise invisible -- a broken Muon still trains, just worse than AdamW.
    """
    out = newton_schulz(G.float(), steps=steps)
    sv = torch.linalg.svdvals(out)
    return float(sv.min()), float(sv.max())


def muon_update_scale(shape: Sequence[int], mode: UpdateScale = "moonlight") -> float:
    """How far to step along the orthogonalised direction.

    ``newton_schulz`` returns a matrix of unit-ish singular values, so its scale carries
    no information about the parameter's size -- it must be reintroduced, and the two
    published choices differ:

    * ``"moonlight"`` -- ``0.2 * sqrt(max(rows, cols))``. Moonlight/Kimi K2. Chosen so
      the update's RMS matches what AdamW would have produced for the same tensor,
      which is what makes a **single shared LR** meaningful across the two optimizers.
      That property is exactly what the §8 study needs: if the arms did not share an
      LR, "Muon beat AdamW" would be confounded with "Muon was tuned and AdamW wasn't".
    * ``"jordan"`` -- ``sqrt(max(1, rows/cols))``. Keller Jordan's original. Corrects
      only for aspect ratio and expects its own separately-tuned LR (~0.02).
    * ``"none"`` -- 1.0. For unit tests of the raw direction.
    """
    if mode == "none":
        return 1.0
    rows, cols = int(shape[0]), int(math.prod(shape[1:]))
    if mode == "moonlight":
        return 0.2 * math.sqrt(max(rows, cols))
    if mode == "jordan":
        return math.sqrt(max(1.0, rows / cols))
    raise ValueError(f"unknown update_scale {mode!r}")


class Muon(Optimizer):
    """Momentum-SGD whose update is orthogonalised by Newton-Schulz before it is applied.

    Only give this optimizer 2-D hidden matrices; use `build_muon_hybrid` and let it do
    the splitting. Parameters with ``ndim > 2`` are reshaped to ``(shape[0], -1)`` for
    the orthogonalisation (there are none in LocalMind, but the reshape means a future
    conv or fused QKV tensor does not silently do something wrong). ``ndim < 2`` is a
    hard error, because a 1-D parameter in this optimizer means the split upstream is
    broken and it should fail loudly rather than train slightly wrong for 7 hours.

    Args:
        lr: peak LR. Overwritten each step by `adamw.set_lr` from the WSD schedule.
        momentum: ``beta`` in ``buf <- beta*buf + (1-beta)*g``. 0.95 is the reference.
        nesterov: step with ``(1-beta)*g + beta*buf`` instead of ``buf``. On by default,
            as in every reference implementation.
        ns_steps: Newton-Schulz iterations per parameter per step.
        weight_decay: decoupled (AdamW-style), applied as ``p *= 1 - lr*wd``. Moonlight
            found this necessary for Muon at scale; the reference nanoGPT speedrun runs
            it at 0.
        update_scale: see `muon_update_scale`.
        ns_dtype: working precision of the iteration. bfloat16 raises (ADR 0001).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[ParamGroupSpec],
        lr: float = 3e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        update_scale: UpdateScale = "moonlight",
        ns_dtype: torch.dtype = torch.float32,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr must be non-negative, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if ns_dtype is torch.bfloat16:
            raise ValueError("bfloat16 is forbidden project-wide (ADR 0001); see newton_schulz")
        defaults: dict[str, Any] = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
            "update_scale": update_scale,
            "ns_dtype": ns_dtype,
            "lr_scale": 1.0,
        }
        super().__init__(list(params), defaults)  # type: ignore[arg-type]

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta: float = group["momentum"]
            nesterov: bool = group["nesterov"]
            ns_steps: int = group["ns_steps"]
            wd: float = group["weight_decay"]
            scale_mode: UpdateScale = group["update_scale"]
            ns_dtype: torch.dtype = group["ns_dtype"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if p.ndim < 2:
                    raise ValueError(
                        f"Muon received a {p.ndim}-D parameter of shape {tuple(p.shape)}. "
                        "Norms, biases and embeddings belong in the AdamW group -- use "
                        "build_muon_hybrid()."
                    )
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                buf: Tensor = state["momentum_buffer"]
                buf.lerp_(grad, 1.0 - beta)
                # Out-of-place: the reference implementation writes into `grad` here, but
                # `p.grad` is still live for the logging path (grad-norm buffering) and
                # for any second optimizer sharing the step.
                update = grad.lerp(buf, beta) if nesterov else buf.clone()

                original_shape = update.shape
                if update.ndim > 2:
                    update = update.reshape(original_shape[0], -1)
                update = newton_schulz(update, steps=ns_steps, dtype=ns_dtype)
                update = update * muon_update_scale(original_shape, scale_mode)
                if update.shape != original_shape:
                    update = update.reshape(original_shape)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)  # decoupled, exactly as AdamW does it
                p.add_(update, alpha=-lr)

        return loss


@dataclass(frozen=True)
class MuonParamSplit:
    """The Kimi K2 split: which tensors go to Muon and which stay on AdamW."""

    muon_params: list[torch.nn.Parameter]
    adamw_groups: list[ParamGroupSpec]
    muon_names: list[str]
    adamw_names: list[str]

    @property
    def muon_numel(self) -> int:
        return sum(p.numel() for p in self.muon_params)

    @property
    def adamw_numel(self) -> int:
        return sum(p.numel() for g in self.adamw_groups for p in g["params"])

    def summary(self) -> dict[str, Any]:
        """For the run log -- "Muon covers 22.5M of 30.9M params" is worth recording."""
        total = self.muon_numel + self.adamw_numel
        return {
            "muon_tensors": len(self.muon_params),
            "muon_params": self.muon_numel,
            "adamw_tensors": len(self.adamw_names),
            "adamw_params": self.adamw_numel,
            "muon_fraction": self.muon_numel / total if total else 0.0,
        }


def _embedding_param_ids(model: nn.Module) -> set[int]:
    """Ids of embedding weights -- and therefore of any tied LM head, by identity.

    Name matching alone would miss a tied head that is only ever reached through
    ``lm_head.weight``; identity cannot.
    """
    return {
        id(p)
        for m in model.modules()
        if isinstance(m, nn.Embedding)
        for p in m.parameters(recurse=False)
    }


def split_muon_params(
    model: nn.Module,
    weight_decay: float = 0.1,
    lr_scale: float = 1.0,
    exclude_name_parts: Sequence[str] = ("lm_head", "tok_emb", "embed", "output"),
) -> MuonParamSplit:
    """Partition ``model``'s parameters into the Muon set and the AdamW groups.

    Muon gets every trainable parameter with ``ndim >= 2`` that is not an embedding
    (by identity), not a norm scale, and whose name does not contain one of
    ``exclude_name_parts``. For LocalMind that is exactly ``attn.{wq,wk,wv,wo}`` and
    ``ffn.{gate,up,down}`` in every block.

    AdamW gets the rest, split into decay / no-decay groups by
    `adamw.split_param_groups`'s rules -- which for the leftovers means everything is
    no-decay, since what is left is precisely norms, biases and embeddings.
    """
    emb_ids = _embedding_param_ids(model)
    norm_ids = {
        id(p)
        for m in model.modules()
        if isinstance(m, RMSNorm | nn.LayerNorm | nn.GroupNorm)
        for p in m.parameters(recurse=False)
    }

    muon_params: list[torch.nn.Parameter] = []
    muon_names: list[str] = []
    rest: list[torch.nn.Parameter] = []
    rest_names: list[str] = []

    for name, p in iter_trainable_params(model):
        excluded = (
            p.ndim < 2
            or id(p) in emb_ids
            or id(p) in norm_ids
            or any(part in name for part in exclude_name_parts)
        )
        if excluded:
            rest.append(p)
            rest_names.append(name)
        else:
            muon_params.append(p)
            muon_names.append(name)

    # Same decay rule as `adamw.split_param_groups`, applied to the leftovers: decay
    # only 2-D non-embedding non-norm tensors. In practice nothing here qualifies for a
    # tied model, so the AdamW arm is all no-decay -- which is the §8 instruction
    # ("exclude norms/bias/embeddings") reached from the other direction.
    adamw_groups: list[ParamGroupSpec] = []
    decay = [p for p in rest if p.ndim >= 2 and id(p) not in emb_ids and id(p) not in norm_ids]
    decay_ids = {id(p) for p in decay}
    no_decay = [p for p in rest if id(p) not in decay_ids]
    if decay:
        adamw_groups.append(
            {
                "name": "adamw_decay",
                "params": decay,
                "weight_decay": weight_decay,
                "lr_scale": lr_scale,
            }
        )
    if no_decay:
        adamw_groups.append(
            {
                "name": "adamw_no_decay",
                "params": no_decay,
                "weight_decay": 0.0,
                "lr_scale": lr_scale,
            }
        )

    return MuonParamSplit(
        muon_params=muon_params,
        adamw_groups=adamw_groups,
        muon_names=muon_names,
        adamw_names=rest_names,
    )


def build_muon_hybrid(
    model: nn.Module,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    momentum: float = 0.95,
    ns_steps: int = 5,
    update_scale: UpdateScale = "moonlight",
    muon_lr_scale: float = 1.0,
    muon_weight_decay: float | None = None,
    ns_dtype: torch.dtype = torch.float32,
) -> tuple[list[Optimizer], MuonParamSplit]:
    """The §8 arm-2 optimizer: ``[Muon(hidden matrices), AdamW(everything else)]``.

    Returns a **list** of optimizers, which is the shape the loop wants anyway: the AMP
    pattern ``for o in opts: scaler.unscale_(o)`` / ``clip_grad_norm_`` /
    ``for o in opts: scaler.step(o)`` / ``scaler.update()`` is exactly what
    `torch.amp.GradScaler` is built to support (it keys its state by optimizer id).

    Args:
        muon_lr_scale: 1.0 with ``update_scale="moonlight"``, because that scaling
            exists precisely so Muon and AdamW share one LR. Raise it only if you also
            switch to ``"jordan"``.
        muon_weight_decay: defaults to ``weight_decay``; Moonlight applies decoupled wd
            to the Muon group too. Set 0.0 to reproduce the nanoGPT speedrun recipe.
    """
    split = split_muon_params(model, weight_decay=weight_decay, lr_scale=1.0)
    if not split.muon_params:
        raise ValueError(
            "no 2-D hidden matrices found for Muon; the model is all embeddings and norms?"
        )
    muon = Muon(
        [
            {
                "name": "muon",
                "params": split.muon_params,
                "lr_scale": muon_lr_scale,
                "weight_decay": weight_decay if muon_weight_decay is None else muon_weight_decay,
            }
        ],
        lr=lr,
        momentum=momentum,
        ns_steps=ns_steps,
        update_scale=update_scale,
        ns_dtype=ns_dtype,
    )
    aux = torch.optim.AdamW(split.adamw_groups, lr=lr, betas=betas, eps=1e-8)
    return [muon, aux], split


def build_optimizer_arm(
    model: nn.Module,
    arm: str,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
) -> list[Optimizer]:
    """The three §8 study arms behind one name.

    ``"adamw"`` (baseline) | ``"muon"`` | ``"adamw_2x"`` (the control for "Muon just
    likes bigger steps" -- same schedule shape, every group's LR doubled).
    """
    if arm == "adamw":
        return [build_adamw(model, lr=lr, betas=betas, weight_decay=weight_decay)]
    if arm == "adamw_2x":
        return [build_adamw(model, lr=lr, betas=betas, weight_decay=weight_decay, lr_scale=2.0)]
    if arm == "muon":
        opts, _split = build_muon_hybrid(model, lr=lr, betas=betas, weight_decay=weight_decay)
        return opts
    raise ValueError(f"unknown optimizer arm {arm!r}; expected adamw | muon | adamw_2x")
