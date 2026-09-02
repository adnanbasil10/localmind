"""LoRA (r=16) -- the ``Qwen3-4B + LoRA`` row of the SS9 5e comparison matrix.

The matrix asks what "~0.1% of params updated" actually buys, and that row cannot be
filled without an implementation to point at. LoRA freezes ``W`` and learns a rank-``r``
correction::

    h = W x + (alpha / r) * B A x        A: (r, in),  B: (out, r),  B initialised to 0

``B = 0`` at init makes the adapted model **exactly** the base model on step zero, which
is the property that lets you attach adapters to a trained checkpoint without a warmup
phase to undo the damage of attaching them. It is also trivially testable, so it is
tested rather than assumed.

Two things are worth stating because they are where LoRA implementations usually go
wrong:

* **The base weight must be frozen, and the freeze must be verified.** A LoRA run that
  quietly leaves ``requires_grad=True`` on ``W`` is just full fine-tuning with extra
  steps, and it will look *better* on the metrics while invalidating the entire "0.1% of
  params" claim the row exists to make. :func:`apply_lora` returns a
  :class:`LoRAReport` carrying the measured trainable fraction, and
  :func:`assert_lora_frozen` turns the claim into an assertion.
* **Merging must be numerically exact.** :func:`merge_lora` folds ``(alpha/r) B A`` into
  ``W`` and hands back plain ``nn.Linear`` modules, so a merged model costs zero extra
  latency at inference -- which matters here because the entire thesis of this project is
  the latency column.

Scope: this module is deliberately generic ``nn.Module`` surgery. It targets the
LocalMind attention projections by default (``wq``/``wk``/``wv``/``wo``), and the same
code adapts a HuggingFace Qwen3-4B for the comparison row, where the target names are
supplied by config rather than hard-coded.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = [
    "DEFAULT_LORA_RANK",
    "DEFAULT_TARGET_MODULES",
    "LoRAConfig",
    "LoRALinear",
    "LoRAReport",
    "apply_lora",
    "assert_lora_frozen",
    "iter_lora_modules",
    "load_lora_state_dict",
    "lora_state_dict",
    "mark_only_lora_trainable",
    "merge_lora",
    "trainable_parameter_summary",
]

#: SS9 5e names ``r=16`` explicitly. Restated in config; never hard-coded at a call site.
DEFAULT_LORA_RANK = 16

#: LocalMind's attention projections. Attention-only targeting is the LoRA paper's own
#: finding: adapting the attention matrices captures nearly all of the benefit at a
#: fraction of the adapter parameters, and adding the MLP mostly buys parameter count.
DEFAULT_TARGET_MODULES: tuple[str, ...] = ("wq", "wk", "wv", "wo")


class LoRAConfig(BaseModel):
    """LoRA hyperparameters. Lives in config, per CONVENTIONS.md."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    r: int = Field(default=DEFAULT_LORA_RANK, gt=0)
    #: ``alpha/r`` is the update scale. Keeping ``alpha = 2r`` (so the scale is 2) means
    #: changing ``r`` does not implicitly change the effective learning rate, which is
    #: the usual reason a rank sweep produces an uninterpretable curve.
    alpha: float = Field(default=2.0 * DEFAULT_LORA_RANK, gt=0.0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    #: Excluded by default: with ``tie_embeddings: true`` the head shares the embedding
    #: matrix, so adapting it silently adapts the input embedding too.
    include_lm_head: bool = False
    seed: int = 1337

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    @classmethod
    def from_yaml(cls, path: str | Path) -> LoRAConfig:
        import yaml

        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping")
        return cls.model_validate(raw)


class LoRALinear(nn.Module):
    """``nn.Linear`` with a frozen base weight and a trainable rank-``r`` correction.

    The base module is held as a submodule rather than copied, so the original weight
    tensor (and any quantisation or tying applied to it) is preserved exactly.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = DEFAULT_LORA_RANK,
        alpha: float | None = None,
        dropout: float = 0.0,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        if r > min(base.in_features, base.out_features):
            raise ValueError(
                f"rank {r} exceeds min(in={base.in_features}, out={base.out_features}); "
                "a full-rank 'low-rank' adapter has more parameters than the weight it adapts"
            )
        self.base = base
        self.r = r
        self.alpha = float(alpha) if alpha is not None else 2.0 * r
        self.scaling = self.alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype))
        self.reset_lora_parameters(generator)

        for p in self.base.parameters():
            p.requires_grad_(False)

    def reset_lora_parameters(self, generator: torch.Generator | None = None) -> None:
        """Kaiming-uniform ``A``, **zero** ``B``.

        Zero ``B`` is not a stylistic choice: it makes ``B A = 0`` so the adapted module
        is bit-identical to the base module before the first optimizer step. Initialising
        both factors randomly injects a random rank-``r`` perturbation into a trained
        checkpoint, and the first hundred steps are then spent undoing it.
        """
        with torch.no_grad():
            bound = math.sqrt(5.0)
            std = 1.0 / math.sqrt(self.lora_A.shape[1])
            limit = bound * std
            if generator is not None:
                self.lora_A.uniform_(-limit, limit, generator=generator)
            else:
                self.lora_A.uniform_(-limit, limit)
            self.lora_B.zero_()

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + lora_out * self.scaling

    def delta_weight(self) -> Tensor:
        """``(alpha/r) * B @ A``, shaped like ``base.weight``."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def merged_linear(self) -> nn.Linear:
        """A plain ``nn.Linear`` with the adapter folded in -- zero inference overhead."""
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            dtype=self.base.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.delta_weight())
            if self.base.bias is not None and merged.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}"


@dataclass
class LoRAReport:
    """What :func:`apply_lora` actually did -- the evidence for the "params updated" cell."""

    n_replaced: int
    replaced: list[str] = field(default_factory=list)
    total_params: int = 0
    trainable_params: int = 0
    base_params: int = 0
    rank: int = DEFAULT_LORA_RANK

    @property
    def trainable_fraction(self) -> float:
        """Trainable / total. This is the number the SS9 5e "~0.1%" claim refers to."""
        return self.trainable_params / self.total_params if self.total_params else 0.0

    @property
    def trainable_percent_str(self) -> str:
        return f"{100.0 * self.trainable_fraction:.3f}%"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_replaced": self.n_replaced,
            "replaced": self.replaced,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "base_params": self.base_params,
            "rank": self.rank,
            "trainable_fraction": self.trainable_fraction,
            "params_updated": self.trainable_percent_str,
        }


def _split_parent(model: nn.Module, qualified: str) -> tuple[nn.Module, str]:
    parts = qualified.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_lora(
    model: nn.Module,
    cfg: LoRAConfig | None = None,
    *,
    target_modules: Sequence[str] | None = None,
    r: int | None = None,
) -> LoRAReport:
    """Replace every targeted ``nn.Linear`` with a :class:`LoRALinear`, **in place**.

    Targets match on the module's *leaf* name (``wq``) or on its full qualified name
    (``blocks.0.attn.wq``), so a config can adapt one layer or all of them without a
    different code path.

    Raises when nothing matched. A silent no-op here produces a LoRA "run" that trains
    zero parameters and reports a plausible-looking loss curve from the frozen base
    model, which is among the harder results to debug after the fact.
    """
    config = cfg or LoRAConfig(r=r or DEFAULT_LORA_RANK)
    if r is not None and r != config.r:
        config = config.model_copy(update={"r": r})
    targets = tuple(target_modules) if target_modules is not None else config.target_modules

    generator = torch.Generator().manual_seed(config.seed)
    to_replace: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "lm_head" and not config.include_lm_head:
            continue
        if leaf in targets or name in targets:
            to_replace.append((name, module))

    if not to_replace:
        raise ValueError(
            f"no nn.Linear matched target_modules={targets!r}. Available leaf names: "
            f"{sorted({n.rsplit('.', 1)[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)})}"
        )

    for name, module in to_replace:
        parent, leaf = _split_parent(model, name)
        setattr(
            parent,
            leaf,
            LoRALinear(
                module,
                r=config.r,
                alpha=config.alpha,
                dropout=config.dropout,
                generator=generator,
            ),
        )

    mark_only_lora_trainable(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return LoRAReport(
        n_replaced=len(to_replace),
        replaced=[n for n, _ in to_replace],
        total_params=total,
        trainable_params=trainable,
        base_params=total - trainable,
        rank=config.r,
    )


def mark_only_lora_trainable(model: nn.Module) -> int:
    """Freeze everything except ``lora_A`` / ``lora_B``. Returns the trainable count."""
    for name, param in model.named_parameters():
        param.requires_grad_(name.endswith("lora_A") or name.endswith("lora_B"))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def iter_lora_modules(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def assert_lora_frozen(model: nn.Module) -> None:
    """Raise unless *only* LoRA factors are trainable.

    Called at the top of any LoRA training run. The failure this catches -- a base weight
    left unfrozen -- does not crash, does not warn, and makes the results better, so it
    has to be checked rather than noticed.
    """
    if not any(True for _ in iter_lora_modules(model)):
        raise AssertionError("no LoRALinear modules found; apply_lora was never called")
    leaked = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and not (name.endswith("lora_A") or name.endswith("lora_B"))
    ]
    if leaked:
        raise AssertionError(
            f"{len(leaked)} non-LoRA parameters are trainable, so this is not a LoRA run: "
            f"{leaked[:5]}{' ...' if len(leaked) > 5 else ''}"
        )


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Only the adapter tensors -- a few MB, versus ~120 MB for the full 31M checkpoint."""
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name.endswith("lora_A") or name.endswith("lora_B")
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, Tensor]) -> None:
    """Load adapter tensors back onto a model that already has adapters attached."""
    params = dict(model.named_parameters())
    missing = [k for k in state if k not in params]
    if missing:
        raise KeyError(f"adapter keys absent from the model: {missing[:5]}")
    with torch.no_grad():
        for name, tensor in state.items():
            params[name].copy_(tensor)


def merge_lora(model: nn.Module) -> int:
    """Fold every adapter into its base weight and restore plain ``nn.Linear``, in place.

    Returns the number of modules merged. After this the model has no LoRA parameters and
    no LoRA forward-pass overhead, which is the only honest way to measure the latency
    column of the SS9 5e matrix for the LoRA row.
    """
    targets = list(iter_lora_modules(model))
    for name, module in targets:
        parent, leaf = _split_parent(model, name)
        setattr(parent, leaf, module.merged_linear())
    return len(targets)


def trainable_parameter_summary(model: nn.Module) -> dict[str, Any]:
    """``params updated`` for any model, LoRA or not -- one column of the matrix.

    Written to work on a plain full-finetune model too, so the "100%" cells of the matrix
    and the "~0.1%" cell are produced by the same function rather than by a number typed
    into a table.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frac = trainable / total if total else 0.0
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_fraction": frac,
        "params_updated": f"{100.0 * frac:.3f}%",
        "n_lora_modules": sum(1 for _ in iter_lora_modules(model)),
    }
