"""The pretraining loop (§8). Written by hand; no `Trainer`, no framework.

Run it::

    uv run python -m localmind.train.loop --config configs/train/pretrain.yaml --resume auto
    uv run python -m localmind.train.loop --config configs/train/smoke.yaml --synthetic

Two ranks on Kaggle::

    torchrun --nproc_per_node=2 -m localmind.train.loop --config configs/train/pretrain.yaml

The step, exactly as §8 specifies it:

    lr = wsd(step); set_lr()                     -- WSD, ADR 0002
    zero_grad(set_to_none=True)
    for _ in range(grad_accum):
        with autocast(float16):  out = model(x, targets=y)
        scaler.scale(out.loss / grad_accum).backward()
    scaler.unscale_(opt)                         -- BEFORE clipping, always
    gnorm = clip_grad_norm_(params, 1.0)         -- on true-scale gradients
    scaler.step(opt); scaler.update()
    maybe checkpoint                             -- session-cap insurance

Three deviations from the §8 pseudo-code, all deliberate:

1. **Loss is computed inside the model, not by `fused_cross_entropy` here.**
   `LocalMindTransformer.forward(input_ids, targets=...)` already casts logits to fp32
   and returns ``loss = ce + z_loss_coeff * z``. Recomputing CE out here would either
   duplicate that or bypass the z-loss that ADR 0001 makes load-bearing. `out.loss` is
   what gets `.backward()`; `out.ce_loss` is what gets reported as "loss", because
   z-loss is a regulariser and not a measure of prediction quality.
2. **`doc_mask` is accepted from the loader and logged, but not passed to the model.**
   The Phase 2 attention module exposes causal masking only; there is no block-diagonal
   document mask in its signature. The loader still yields the mask, this loop still
   carries it, and wiring it through is a Phase 2 change, not a Phase 4 one. Until then
   packed sequences attend across document boundaries -- §7 flags that as costing real
   quality and asks for the ablation, so it must not be silently dropped.
3. **`optimizers` is a list.** The Muon arm is Muon(hidden matrices) + AdamW(everything
   else); `GradScaler` keys its state by optimizer id and supports exactly this.

## Instrumentation, and the sync trap

§8's trap list names ``loss.item()`` every step as a throughput killer: it forces a
host-device sync that serialises the CPU against the GPU and can cost double-digit
percentages of MFU. So every per-step scalar -- loss, ce, z, grad norm, scaler scale,
per-layer weight norm, per-layer activation RMS -- is written into a **preallocated
on-device buffer** and transferred once every `log_every` steps in a single `.cpu()`.

One sync per step remains and is unavoidable: `GradScaler.step` reads ``found_inf`` to
decide whether to skip the update, and that decision is host-side control flow. This is
inherent to fp16 + GradScaler; it is the cost of not having bf16 (ADR 0001), and it is
the only sync in the step path.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor, nn
from torch.optim import Optimizer

from localmind.model import LocalMindTransformer, ModelConfig, ModelOutput
from localmind.model.config import count_params
from localmind.train.checkpoint import (
    CheckpointManager,
    RngSnapshot,
    TrainerState,
    load_checkpoint,
)
from localmind.train.mfu import bits_per_byte, device_peak_flops, throughput_report
from localmind.train.optim.adamw import set_lr
from localmind.train.optim.muon import build_optimizer_arm
from localmind.train.schedule import LRSchedule, WSDSchedule, branch_decay

__all__ = [
    "Batch",
    "BatchLoader",
    "MetricBuffer",
    "SyntheticLoader",
    "TrainConfig",
    "Trainer",
    "evaluate",
    "main",
    "run_optimizer_study",
    "seed_everything",
    "tokens_to_target_loss",
    "train_from_config",
    "wall_clock_to_target_loss",
]

# =====================================================================================
# Data interface
# =====================================================================================
#: ``(input_ids, targets, doc_mask)``.
#:
#: * ``input_ids``: ``(B, T)`` int64.
#: * ``targets``:   ``(B, T)`` int64, **already shifted by the data pipeline** --
#:   ``targets[b, t]`` is the token following ``input_ids[b, t]``. ``-100`` masks a
#:   position out of the loss. This loop never shifts anything.
#: * ``doc_mask``:  ``(B, T)`` int32/int64 document ids for block-diagonal attention, or
#:   ``None`` for naive packing. Carried and logged; see deviation 2 in the module
#:   docstring for why it is not yet passed to the model.
Batch = tuple[Tensor, Tensor, Tensor | None]


@runtime_checkable
class BatchLoader(Protocol):
    """What the training loop needs from `localmind.data.loader`, and nothing more.

    Declared **here**, structurally, on purpose: the data package is being built
    concurrently, and importing it at module scope would (a) couple this module's import
    graph to a moving target and (b) break `just test-fast` the moment that package
    touches the network. The loop is tested against `SyntheticLoader`, which satisfies
    this Protocol and nothing else.

    The loader is its own iterator (``__next__`` on the object, not on a fresh
    ``iter()``), because the object that yields batches must be the same object whose
    position is checkpointed.
    """

    def __next__(self) -> Batch: ...

    def __iter__(self) -> Iterator[Batch]: ...

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to resume mid-epoch: shard index, offset within the shard,
        epoch counter, and the permutation seed if shard order is shuffled."""
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a position such that the *next* batch is the one that would have come
        next in the uninterrupted run. §7's DoD requires a test proving exactly this."""
        ...

    @property
    def shard_index(self) -> int:
        """Which shard is currently being read. §8 requires it in the step log -- a
        stalled or looping shard index is how a broken resume announces itself."""
        ...


class SyntheticLoader:
    """A deterministic, *learnable* fake corpus. Satisfies `BatchLoader`.

    Not a mock: it emits a real sequence-modelling task, so a training loop run against
    it must show the loss actually fall. Each row is a deterministic first-order chain,
    ``x[t] = (a * x[t-1] + c) mod effective_vocab``, seeded from a random start token.
    The rule is learnable from the immediately preceding token alone, so a healthy loop
    drives cross-entropy from ``ln(vocab_size)`` toward ``ln(1)`` within tens of steps
    -- and a loop with the sign of the update flipped, or the LR schedule wired
    backwards, visibly does not.

    ``effective_vocab`` is smaller than the model's vocab so the entropy floor is
    reachable in seconds on a CPU.

    It draws its start tokens from the **global** torch RNG on purpose: that makes
    global-RNG restoration load-bearing for resume, so `test_resume_is_bit_exact` fails
    if `RngSnapshot` forgets a source rather than passing by luck.
    """

    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        seq_len: int,
        effective_vocab: int = 512,
        device: torch.device | str = "cpu",
        a: int = 7,
        c: int = 13,
        shards: int = 4,
    ) -> None:
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.effective_vocab = min(effective_vocab, vocab_size)
        self.device = torch.device(device)
        self.a = a
        self.c = c
        self.shards = max(1, shards)
        self.position = 0

    @property
    def shard_index(self) -> int:
        return self.position % self.shards

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        start = torch.randint(
            0, self.effective_vocab, (self.batch_size, 1), device=self.device, dtype=torch.long
        )
        cols = [start]
        cur = start
        for _ in range(self.seq_len):
            cur = (cur * self.a + self.c) % self.effective_vocab
            cols.append(cur)
        chain = torch.cat(cols, dim=1)
        x = chain[:, : self.seq_len].contiguous()
        y = chain[:, 1 : self.seq_len + 1].contiguous()
        self.position += 1
        return x, y, None

    def state_dict(self) -> dict[str, Any]:
        return {"position": self.position, "shards": self.shards}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.position = int(state["position"])
        self.shards = int(state.get("shards", self.shards))


# =====================================================================================
# Config
# =====================================================================================
class TrainConfig(BaseModel):
    """`configs/train/*.yaml`, validated. CONVENTIONS.md: no hyperparameter in code.

    The YAML key is ``model_config``, which pydantic v2 reserves for its own class-level
    `ConfigDict`. The field is therefore named ``model_config_path`` and carries
    ``alias="model_config"``, with ``protected_namespaces=()`` to silence the ``model_``
    prefix warning. Dump with ``by_alias=True`` to round-trip back to the YAML spelling.

    Fields with defaults do not appear in the shipped YAMLs (which are controller-owned
    and must not be edited); they are run plumbing, not model hyperparameters.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, protected_namespaces=()
    )

    model_config_path: str = Field(alias="model_config")
    tokens_per_step: int = Field(gt=0)
    micro_batch_size: int = Field(gt=0)
    seq_len: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    peak_lr: float = Field(gt=0.0)
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    schedule: Literal["wsd"] = "wsd"
    warmup_frac: float = Field(default=0.02, ge=0.0, le=1.0)
    stable_frac: float = Field(default=0.80, ge=0.0, le=1.0)
    decay_frac: float = Field(default=0.18, ge=0.0, le=1.0)
    optimizer: Literal["adamw", "muon", "adamw_2x"] = "adamw"
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(default=0.1, ge=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    #: ADR 0001. ``bf16`` is not a member of this Literal, so a bf16 config is a
    #: validation error at load time rather than a crash nine hours into a session.
    precision: Literal["fp16", "fp32"] = "fp16"
    z_loss: float = Field(default=1e-4, ge=0.0)
    torch_compile: Literal["off", "default", "max-autotune"] = "default"
    ckpt_every_min: float = Field(default=15.0, gt=0.0)
    hub_push_every_min: float | None = Field(default=60.0)
    seed: int = 1337

    # --- run plumbing (defaulted; absent from the shipped YAMLs) ---------------------
    out_dir: str = "artifacts/runs"
    run_name: str | None = None
    hub_repo_id: str | None = None
    keep_last_checkpoints: int = Field(default=3, ge=1)
    log_every: int = Field(default=10, ge=1)
    val_every: int = Field(default=250, ge=1)
    val_iters: int = Field(default=20, ge=1)
    #: Cadence for per-layer weight-norm / activation-RMS capture. §8 says "every step";
    #: 1 honours that, and the knob exists because the hooks are the one instrument here
    #: with a measurable cost.
    stats_every: int = Field(default=1, ge=1)
    #: UTF-8 bytes per token on held-out text, from the tokenizer benchmark (§5). Only
    #: used to turn cross-entropy into bits-per-byte; a loader exposing
    #: ``bytes_per_token`` overrides it.
    bytes_per_token: float = Field(default=4.0, gt=0.0)
    attn_backend: Literal["naive", "sdpa_math", "sdpa_efficient"] = "sdpa_efficient"

    @field_validator("torch_compile", mode="before")
    @classmethod
    def _coerce_torch_compile(cls, value: Any) -> Any:
        """`smoke.yaml` says ``torch_compile: off``, and YAML 1.1 parses bare ``off`` as
        the boolean ``False`` (same for ``on``/``yes``/``no``). The configs are
        controller-owned and must not be edited, so the coercion lives here."""
        if isinstance(value, bool):
            return "default" if value else "off"
        return value

    @model_validator(mode="after")
    def _check(self) -> TrainConfig:
        total = self.warmup_frac + self.stable_frac + self.decay_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"warmup+stable+decay must be 1.0, got {total}")
        tokens_per_micro = self.micro_batch_size * self.seq_len
        if self.tokens_per_step % tokens_per_micro != 0:
            raise ValueError(
                f"tokens_per_step ({self.tokens_per_step}) must be divisible by "
                f"micro_batch_size * seq_len ({tokens_per_micro})"
            )
        return self

    # --- derived ---------------------------------------------------------------------
    @property
    def tokens_per_micro_batch(self) -> int:
        return self.micro_batch_size * self.seq_len

    def grad_accum(self, world_size: int = 1) -> int:
        """Micro-batches per optimizer step **per rank**.

        ``tokens_per_step`` is the global batch, so 2xT4 halves the accumulation each
        rank does rather than doubling the global batch -- which is the whole point of
        expressing the batch in tokens instead of sequences.
        """
        per_step = self.tokens_per_micro_batch * world_size
        if self.tokens_per_step % per_step != 0:
            raise ValueError(
                f"tokens_per_step ({self.tokens_per_step}) is not divisible by "
                f"micro_batch_size * seq_len * world_size ({per_step}); adjust the config "
                f"or the number of ranks"
            )
        return self.tokens_per_step // per_step

    @property
    def max_steps(self) -> int:
        return max(1, self.max_tokens // self.tokens_per_step)

    @property
    def config_hash(self) -> str:
        """sha256 of the canonical config. CONVENTIONS.md: every run is hash-logged."""
        blob = json.dumps(self.model_dump(mode="json", by_alias=True), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def schedule_for(self, total_steps: int | None = None) -> WSDSchedule:
        return WSDSchedule(
            peak_lr=self.peak_lr,
            total_steps=self.max_steps if total_steps is None else total_steps,
            warmup_frac=self.warmup_frac,
            stable_frac=self.stable_frac,
            decay_frac=self.decay_frac,
            min_lr_ratio=self.min_lr_ratio,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        import yaml

        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping")
        return cls.model_validate(raw)


# =====================================================================================
# Instrumentation
# =====================================================================================
class MetricBuffer:
    """On-device scalar ring, flushed to the host once per `log_every` steps.

    `record` performs only device-side writes (`Tensor.copy_` into a preallocated slot),
    so it never synchronises. `flush` does exactly one device-to-host transfer for the
    whole window. See the module docstring for why this matters.
    """

    def __init__(
        self,
        fields: Sequence[str],
        capacity: int,
        device: torch.device,
        n_layers: int = 0,
    ) -> None:
        self.fields = list(fields)
        self.capacity = max(1, capacity)
        self.device = device
        self.n_layers = n_layers
        self._buf = torch.zeros(self.capacity, len(self.fields), dtype=torch.float32, device=device)
        self._wnorm = torch.zeros(self.capacity, n_layers, dtype=torch.float32, device=device)
        self._arms = torch.zeros(self.capacity, n_layers, dtype=torch.float32, device=device)
        self._index = {name: i for i, name in enumerate(self.fields)}
        self._host: list[dict[str, Any]] = []
        self.n = 0

    def __len__(self) -> int:
        return self.n

    @property
    def full(self) -> bool:
        return self.n >= self.capacity

    def record(
        self,
        device_values: Mapping[str, Tensor | float],
        host_values: Mapping[str, Any],
        weight_norms: Tensor | None = None,
        act_rms: Tensor | None = None,
    ) -> None:
        if self.full:
            raise RuntimeError("MetricBuffer is full; call flush() before recording again")
        row = self._buf[self.n]
        for name, value in device_values.items():
            j = self._index.get(name)
            if j is None:
                continue
            if isinstance(value, Tensor):
                row[j].copy_(value.detach().reshape(()), non_blocking=True)
            else:
                row[j].fill_(float(value))
        if weight_norms is not None and self.n_layers:
            self._wnorm[self.n].copy_(weight_norms.detach(), non_blocking=True)
        if act_rms is not None and self.n_layers:
            self._arms[self.n].copy_(act_rms.detach(), non_blocking=True)
        self._host.append(dict(host_values))
        self.n += 1

    def flush(self) -> list[dict[str, Any]]:
        """Drain the buffer. **The only host-device sync in the logging path.**"""
        if self.n == 0:
            return []
        values = self._buf[: self.n].to("cpu").tolist()
        wnorms = self._wnorm[: self.n].to("cpu").tolist() if self.n_layers else None
        arms = self._arms[: self.n].to("cpu").tolist() if self.n_layers else None
        rows: list[dict[str, Any]] = []
        for i, host in enumerate(self._host):
            row: dict[str, Any] = dict(host)
            row.update(dict(zip(self.fields, values[i], strict=True)))
            if wnorms is not None:
                row["weight_norm"] = wnorms[i]
            if arms is not None:
                row["act_rms"] = arms[i]
            rows.append(row)
        self.n = 0
        self._host.clear()
        return rows


class LayerProbe:
    """Forward hooks that write each block's output RMS into a device tensor slot.

    §8 asks for per-layer activation RMS every step. Doing it with `.item()` per layer
    would be 8 syncs per step on the 31M model; doing it with hooks into a preallocated
    ``(n_layers,)`` tensor is zero. ``enabled`` is toggled by the loop so the cost is
    paid on the final micro-batch of an accumulation window only, not on all of them.
    """

    def __init__(self, blocks: Sequence[nn.Module], device: torch.device) -> None:
        self.values = torch.zeros(len(blocks), dtype=torch.float32, device=device)
        self.enabled = False
        self._handles = [
            block.register_forward_hook(self._make_hook(i)) for i, block in enumerate(blocks)
        ]

    def _make_hook(self, index: int) -> Any:
        def hook(_module: nn.Module, _args: Any, output: Any) -> None:
            if not self.enabled:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, Tensor):
                self.values[index].copy_(
                    tensor.detach().float().pow(2).mean().sqrt(), non_blocking=True
                )

        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def layer_weight_norms(blocks: Sequence[nn.Module], out: Tensor) -> Tensor:
    """L2 norm of every parameter in each block, written into ``out`` on-device.

    A per-layer weight norm that climbs monotonically while the loss is flat is the
    signature of a run being held together by weight decay alone; a norm that collapses
    in one layer is a dead layer. Neither is visible in the scalar loss.
    """
    for i, block in enumerate(blocks):
        total = None
        for p in block.parameters():
            sq = p.detach().float().pow(2).sum()
            total = sq if total is None else total + sq
        if total is not None:
            out[i].copy_(total.sqrt(), non_blocking=True)
    return out


def scaler_scale_tensor(scaler: torch.amp.GradScaler) -> Tensor | float:
    """The GradScaler's current scale **without** forcing a sync.

    `GradScaler.get_scale()` calls `.item()` on the internal scale tensor. ADR 0001
    requires logging this value every step, so read the device tensor directly and let
    `MetricBuffer` batch the transfer. Falls back to the public accessor if the private
    attribute ever disappears.
    """
    private = getattr(scaler, "_scale", None)
    if isinstance(private, Tensor):
        return private
    return float(scaler.get_scale()) if scaler.is_enabled() else 1.0


class JsonlSink:
    """Metrics to `metrics.jsonl` plus a one-line console summary. Rank 0 only."""

    CONSOLE_KEYS = ("step", "loss", "lr", "grad_norm", "mfu", "tokens_per_sec", "scaler_scale")

    def __init__(self, path: Path | None, echo: bool = True) -> None:
        self.path = path
        self.echo = echo
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(_jsonable(row)) + "\n")
        if self.echo:
            row = rows[-1]
            parts = []
            for key in self.CONSOLE_KEYS:
                if key not in row:
                    continue
                value = row[key]
                parts.append(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}")
            print("  ".join(parts))


def _jsonable(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            out[key] = None
        elif isinstance(value, list):
            out[key] = [None if isinstance(v, float) and not math.isfinite(v) else v for v in value]
        else:
            out[key] = value
    return out


# =====================================================================================
# Setup helpers
# =====================================================================================
def seed_everything(seed: int) -> None:
    """Seed python / numpy / torch / CUDA. CONVENTIONS.md: seeded runs reproduce."""
    random.seed(seed)
    np.random.seed(seed % (2**32))  # noqa: NPY002 - seeds the global stream RngSnapshot captures
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple[int, int, int]:
    """``(rank, local_rank, world_size)`` from torchrun's environment. 2xT4 = 2 ranks."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 0, 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextlib.contextmanager
def _maybe_no_sync(model: nn.Module, skip_sync: bool) -> Iterator[None]:
    """Suppress DDP's gradient all-reduce on every micro-batch but the last.

    Without this, a 16-way accumulation all-reduces 16 times per optimizer step instead
    of once -- on a 2xT4 box over PCIe that is the difference between a communication
    cost you can ignore and one that dominates.
    """
    if skip_sync and hasattr(model, "no_sync"):
        with model.no_sync():  # type: ignore[operator]
            yield
    else:
        yield


def build_model(
    train_cfg: TrainConfig, device: torch.device, seed: int | None = None
) -> tuple[LocalMindTransformer, ModelConfig]:
    """Load the model config, reconcile ``z_loss``, instantiate, move to device."""
    model_cfg = ModelConfig.from_yaml(train_cfg.model_config_path)
    if abs(model_cfg.z_loss - train_cfg.z_loss) > 1e-12:
        # The run config is the run-level authority; the model YAML is the architecture
        # default. Ablating z_loss must not require editing a controller-owned file.
        model_cfg = model_cfg.model_copy(update={"z_loss": train_cfg.z_loss})
    if train_cfg.seq_len > model_cfg.max_seq_len:
        raise ValueError(
            f"seq_len={train_cfg.seq_len} exceeds the model's max_seq_len="
            f"{model_cfg.max_seq_len}; extend rope/max_seq_len first (§8 context extension)"
        )
    if seed is not None:
        seed_everything(seed)
    model = LocalMindTransformer(model_cfg, backend=train_cfg.attn_backend)
    return model.to(device), model_cfg


def maybe_compile(model: nn.Module, mode: str, device: torch.device) -> nn.Module:
    """`torch.compile` when it can help and cannot hurt.

    Skipped on CPU: inductor needs a working C++ toolchain, the dev box is Windows, and
    a compile failure inside a Kaggle session is an hour of quota. §8 also pins
    ``mode="default"`` -- ``max-autotune`` burns 5-10 minutes of warmup per session,
    which on a 30 h/week budget is a real cost for a short run.
    """
    if mode == "off" or device.type != "cuda":
        return model
    try:
        return torch.compile(model, mode=None if mode == "default" else mode)  # type: ignore[return-value]
    except Exception as exc:
        print(f"[train] torch.compile({mode}) failed ({type(exc).__name__}: {exc}); running eager")
        return model


# =====================================================================================
# Evaluation
# =====================================================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: BatchLoader,
    iters: int,
    device: torch.device,
    autocast_enabled: bool = False,
) -> Tensor:
    """Mean cross-entropy over ``iters`` batches, as a **device tensor**.

    Returns a tensor rather than a float so the caller decides when to sync. Reports
    `ce_loss`, not `loss`: z-loss is a regulariser and including it in a validation
    number would make the metric depend on a hyperparameter rather than on the model.
    """
    was_training = model.training
    model.eval()
    total = torch.zeros((), dtype=torch.float32, device=device)
    count = 0
    ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled)
    for _ in range(iters):
        x, y, _mask = next(loader)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with ctx:
            out: ModelOutput = model(x, targets=y)
        if out.ce_loss is None:
            raise RuntimeError("model returned no ce_loss; targets were not passed")
        total = total + out.ce_loss.detach().float()
        count += 1
    if was_training:
        model.train()
    return total / max(1, count)


# =====================================================================================
# Trainer
# =====================================================================================
@dataclass
class TrainResult:
    """What a run hands back. `history` is what the §8 optimizer study consumes."""

    state: TrainerState
    first_loss: float = float("nan")
    last_loss: float = float("nan")
    steps_run: int = 0
    wall_clock_s: float = 0.0
    metrics: list[dict[str, Any]] = field(default_factory=list)
    checkpoint: Path | None = None


class Trainer:
    """Owns the model, optimizers, scaler, schedule, checkpoints and instrumentation."""

    LOSS_FIELDS = ("loss", "ce_loss", "z_loss", "grad_norm", "scaler_scale")

    def __init__(
        self,
        cfg: TrainConfig,
        device: torch.device | str | None = None,
        rank: int = 0,
        world_size: int = 1,
        out_dir: str | Path | None = None,
        schedule: LRSchedule | None = None,
        sink: JsonlSink | None = None,
        compile_model: bool | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(device if isinstance(device, str) else None)
        if isinstance(device, torch.device):
            self.device = device
        self.rank = rank
        self.world_size = world_size
        self.grad_accum = cfg.grad_accum(world_size)

        seed_everything(cfg.seed)
        self.model, self.model_cfg = build_model(cfg, self.device)
        self.raw_model = self.model
        self.blocks: list[nn.Module] = list(self.raw_model.blocks)

        if world_size > 1:
            device_ids = [self.device.index] if self.device.type == "cuda" else None
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=device_ids)
        should_compile = (cfg.torch_compile != "off") if compile_model is None else compile_model
        if should_compile:
            self.model = maybe_compile(self.model, cfg.torch_compile, self.device)

        self.params = [p for p in self.raw_model.parameters() if p.requires_grad]
        self.optimizers: list[Optimizer] = build_optimizer_arm(
            self.raw_model,
            cfg.optimizer,
            lr=cfg.peak_lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        # ADR 0001. Two separate switches, and they are not the same switch:
        #  * autocast runs the *math* in fp16 -- CUDA only; CPU fp16 autocast is slow
        #    and is not what the T4 path exercises.
        #  * the GradScaler runs the *control flow* -- scale, unscale before clipping,
        #    skip-on-inf, halve-on-overflow. That is enabled on CPU too, so the entire
        #    fp16 loss-scaling protocol is exercised and tested without a GPU. Scaling
        #    by a power of two is exact in fp32, so this changes no result.
        self.autocast_enabled = cfg.precision == "fp16" and self.device.type == "cuda"
        self.scaler_enabled = cfg.precision == "fp16"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.scaler_enabled)

        self.schedule: LRSchedule = schedule or cfg.schedule_for()
        self.state = TrainerState()
        #: Optimizer steps skipped because the global grad norm was non-finite. ADR 0001
        #: asks for the scaler's scale every step; this is the same signal, cumulative.
        self.skipped_steps = 0

        run_dir = Path(out_dir) if out_dir is not None else self._default_run_dir()
        self.run_dir = run_dir
        self.ckpt = CheckpointManager(
            run_dir,
            every_min=cfg.ckpt_every_min,
            hub_every_min=cfg.hub_push_every_min,
            hub_repo_id=cfg.hub_repo_id,
            keep_last=cfg.keep_last_checkpoints,
            rank=rank,
        )
        self.sink = (
            sink
            if sink is not None
            else JsonlSink(run_dir / "metrics.jsonl" if rank == 0 else None, echo=rank == 0)
        )
        self.buffer = MetricBuffer(
            self.LOSS_FIELDS,
            capacity=cfg.log_every,
            device=self.device,
            n_layers=len(self.blocks),
        )
        self.probe = LayerProbe(self.blocks, self.device)
        self._wnorm_scratch = torch.zeros(len(self.blocks), dtype=torch.float32, device=self.device)
        self.peak_flops = device_peak_flops(
            torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
            precision=cfg.precision,
            world_size=world_size,
        )
        self.n_params = count_params(self.model_cfg)

    def _default_run_dir(self) -> Path:
        name = self.cfg.run_name or f"{Path(self.cfg.model_config_path).stem}-{self.cfg.optimizer}"
        return Path(self.cfg.out_dir) / f"{name}-{self.cfg.config_hash}"

    # -- persistence ------------------------------------------------------------------
    def save(self, loader: BatchLoader | None, force: bool = False) -> Path | None:
        return self.ckpt.maybe_save(
            self.state.step,
            force=force,
            model=self.raw_model,
            optimizers=self.optimizers,
            scaler=self.scaler,
            state=self.state,
            loader_state=loader.state_dict() if loader is not None else None,
            train_config=self.cfg.model_dump(mode="json", by_alias=True),
            model_config=self.model_cfg.model_dump(mode="json"),
            config_hash=self.cfg.config_hash,
            rng=RngSnapshot.capture(),
        )

    def resume(self, path: str | Path, loader: BatchLoader | None = None) -> TrainerState:
        """Restore weights, optimizers, scaler, RNG and the loader's shard position."""
        state, loader_state, _payload = load_checkpoint(
            path,
            self.raw_model,
            optimizers=self.optimizers,
            scaler=self.scaler,
            map_location=self.device,
        )
        self.state = state
        if loader is not None and loader_state is not None:
            loader.load_state_dict(loader_state)
        return state

    # -- the step ---------------------------------------------------------------------
    def _forward_backward(self, loader: BatchLoader, collect_stats: bool) -> Tensor:
        """One accumulation window. Returns ``[loss, ce, z]`` accumulated on-device."""
        totals = torch.zeros(3, dtype=torch.float32, device=self.device)
        inv = 1.0 / self.grad_accum
        for micro in range(self.grad_accum):
            last = micro == self.grad_accum - 1
            self.probe.enabled = collect_stats and last
            x, y, _doc_mask = next(loader)
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with _maybe_no_sync(self.model, skip_sync=not last):
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.autocast_enabled,
                ):
                    out: ModelOutput = self.model(x, targets=y)
                if out.loss is None or out.ce_loss is None or out.z_loss is None:
                    raise RuntimeError("model returned no loss; targets were not passed")
                # Divide by grad_accum so the accumulated gradient equals the gradient of
                # the mean over the full global batch, not its sum.
                self.scaler.scale(out.loss * inv).backward()
            totals += (
                torch.stack([out.loss.detach(), out.ce_loss.detach(), out.z_loss.detach()]).float()
                * inv
            )
        self.probe.enabled = False
        return totals

    def step(self, loader: BatchLoader) -> dict[str, Any]:
        """One optimizer step: §8's inner block, verbatim in structure."""
        t0 = time.perf_counter()
        step_index = self.state.step
        lr = self.schedule(step_index)
        set_lr(self.optimizers, lr)
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

        collect_stats = (step_index % self.cfg.stats_every) == 0
        totals = self._forward_backward(loader, collect_stats)

        # unscale BEFORE clipping: clipping scaled gradients would clip to
        # `grad_clip / scale`, i.e. to effectively nothing, and the run would crawl.
        for opt in self.optimizers:
            self.scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
        scale_before = scaler_scale_tensor(self.scaler)
        if isinstance(scale_before, Tensor):
            scale_before = scale_before.detach().clone()

        # The global grad norm is the single authority on whether this step happens.
        #
        # `GradScaler.step` skips *per optimizer*, based on the inf check recorded when
        # that optimizer was unscaled. With the two-optimizer Muon arm that is wrong and
        # actively dangerous: an fp16 overflow in a hidden matrix sets found_inf for
        # Muon only, `clip_grad_norm_` then multiplies *every* gradient by a NaN clip
        # coefficient, and AdamW -- whose own inf check passed a moment earlier -- steps
        # the embeddings with NaN. One overflow silently kills the whole model.
        #
        # `isfinite(total_norm)` is exactly the union of the per-optimizer inf checks
        # (the norm is non-finite iff some gradient is), so gating on it reproduces
        # GradScaler's semantics across every optimizer at once. Skipping `step` does
        # not cost the scaler its backoff: `update()` combines found_inf from every
        # optimizer that was unscaled, whether or not it was stepped.
        skipped = not bool(torch.isfinite(gnorm))
        if not skipped:
            for opt in self.optimizers:
                self.scaler.step(opt)
        else:
            self.skipped_steps += 1
        self.scaler.update()

        if collect_stats:
            layer_weight_norms(self.blocks, self._wnorm_scratch)

        self.state.step += 1
        self.state.tokens_seen += self.cfg.tokens_per_step
        step_time = time.perf_counter() - t0
        self.state.wall_clock_s += step_time

        report = throughput_report(
            self.model_cfg,
            tokens_this_step=self.cfg.tokens_per_step,
            step_time_s=step_time,
            ctx=self.cfg.seq_len,
            peak_flops=self.peak_flops,
        )
        host: dict[str, Any] = {
            "step": self.state.step,
            "tokens": self.state.tokens_seen,
            "lr": lr,
            "phase": self.schedule.phase(step_index),
            "wall_clock_s": self.state.wall_clock_s,
            "shard_index": getattr(loader, "shard_index", -1),
            "grad_accum": self.grad_accum,
            # A run whose skip count climbs is overflowing, not converging (ADR 0001).
            "step_skipped": int(skipped),
            "skipped_steps": self.skipped_steps,
            **report.as_dict(),
        }
        if self.device.type == "cuda":
            host["gpu_mem_alloc_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
            host["gpu_mem_reserved_gb"] = torch.cuda.max_memory_reserved(self.device) / 1e9
        self.buffer.record(
            device_values={
                "loss": totals[0],
                "ce_loss": totals[1],
                "z_loss": totals[2],
                "grad_norm": gnorm,
                "scaler_scale": scale_before,
            },
            host_values=host,
            weight_norms=self._wnorm_scratch if collect_stats else None,
            act_rms=self.probe.values if collect_stats else None,
        )
        return host

    # -- the run ----------------------------------------------------------------------
    def run(
        self,
        loader: BatchLoader,
        val_loader: BatchLoader | None = None,
        max_steps: int | None = None,
        save_at_end: bool = True,
    ) -> TrainResult:
        """Train until ``max_steps`` (default: the schedule's ``total_steps``)."""
        target = self.schedule.total_steps if max_steps is None else max_steps
        self.model.train()
        collected: list[dict[str, Any]] = []
        first_loss = float("nan")
        last_loss = float("nan")
        start_step = self.state.step
        t_start = time.perf_counter()

        while self.state.step < target:
            self.step(loader)

            if val_loader is not None and self.state.step % self.cfg.val_every == 0:
                val = float(
                    evaluate(
                        self.model,
                        val_loader,
                        self.cfg.val_iters,
                        self.device,
                        self.autocast_enabled,
                    ).item()
                )
                self.state.best_val_loss = min(self.state.best_val_loss, val)
                self.state.history.append(
                    (self.state.step, self.state.tokens_seen, self.state.wall_clock_s, val)
                )
                if self.rank == 0:
                    self.sink.write(
                        [
                            {
                                "step": self.state.step,
                                "val_ce_loss": val,
                                "val_bpb": bits_per_byte(val, self._bytes_per_token(loader)),
                                "val_ppl": math.exp(min(val, 20.0)),
                            }
                        ]
                    )

            if self.buffer.full:
                rows = self._flush(loader)
                collected.extend(rows)
                if rows:
                    if math.isnan(first_loss):
                        first_loss = float(rows[0]["ce_loss"])
                    last_loss = float(rows[-1]["ce_loss"])

            self.save(loader)

        rows = self._flush(loader)
        collected.extend(rows)
        if rows:
            if math.isnan(first_loss):
                first_loss = float(rows[0]["ce_loss"])
            last_loss = float(rows[-1]["ce_loss"])

        ckpt_path = self.save(loader, force=True) if save_at_end else None
        return TrainResult(
            state=self.state,
            first_loss=first_loss,
            last_loss=last_loss,
            steps_run=self.state.step - start_step,
            wall_clock_s=time.perf_counter() - t_start,
            metrics=collected,
            checkpoint=ckpt_path,
        )

    def _bytes_per_token(self, loader: BatchLoader) -> float:
        override = getattr(loader, "bytes_per_token", None)
        return float(override) if override else self.cfg.bytes_per_token

    def _flush(self, loader: BatchLoader) -> list[dict[str, Any]]:
        rows = self.buffer.flush()
        bpt = self._bytes_per_token(loader)
        for row in rows:
            row["bits_per_byte"] = bits_per_byte(float(row["ce_loss"]), bpt)
        if self.rank == 0:
            self.sink.write(rows)
        return rows

    def close(self) -> None:
        self.probe.close()


# =====================================================================================
# Study helpers (§8: report tokens-to-target AND wall-clock-to-target)
# =====================================================================================
def _interpolate_to_target(
    history: Sequence[tuple[int, int, float, float]], target: float, x_index: int
) -> float | None:
    """First crossing of ``target``, linearly interpolated on the chosen x axis."""
    prev: tuple[int, int, float, float] | None = None
    for row in history:
        if row[3] <= target:
            if prev is None:
                return float(row[x_index])
            span = prev[3] - row[3]
            frac = 0.0 if span == 0 else (prev[3] - target) / span
            return float(prev[x_index]) + frac * (float(row[x_index]) - float(prev[x_index]))
        prev = row
    return None


def tokens_to_target_loss(
    history: Sequence[tuple[int, int, float, float]], target: float
) -> float | None:
    """Tokens consumed before val loss first reached ``target``. **Sample efficiency.**"""
    return _interpolate_to_target(history, target, x_index=1)


def wall_clock_to_target_loss(
    history: Sequence[tuple[int, int, float, float]], target: float
) -> float | None:
    """Seconds spent before val loss first reached ``target``. **Throughput.**

    §8: these two can disagree, and saying so is the point of reporting both. Muon does
    5 extra matmuls per 2-D parameter per step, so it can win on tokens and lose on
    wall clock -- which is the difference between sample efficiency and throughput, and
    which of the two you care about depends on whether your budget is data or GPU-hours.
    On a 30 GPU-h/week quota it is GPU-hours.
    """
    return _interpolate_to_target(history, target, x_index=2)


def run_optimizer_study(
    cfg: TrainConfig,
    loader_factory: Any,
    arms: Sequence[str] = ("adamw", "muon", "adamw_2x"),
    seeds: Sequence[int] = (0, 1, 2),
    max_steps: int = 200,
    target_loss: float = 4.0,
    out_dir: str | Path | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """The §8 optimizer study: three arms, N seeds, both target metrics.

    Args:
        loader_factory: ``(cfg, seed, split) -> BatchLoader``. Injected so the study can
            run against real shards on Kaggle and against `SyntheticLoader` in tests
            without this function importing the data package.
        target_loss: the val loss whose time-to-reach is the comparison metric. Pick it
            *after* a pilot run, in the region all arms actually reach -- a target below
            the worst arm's floor turns its number into ``None`` and the comparison into
            a footnote.

    Returns a CONVENTIONS.md-shaped payload (``name`` / ``hardware`` / ``seeds`` /
    ``rows`` / ``ci``) with a bootstrap 95% CI per arm.
    """
    from localmind.model.transformer import bootstrap_ci

    dev = resolve_device(device if isinstance(device, str) else None)
    if isinstance(device, torch.device):
        dev = device
    base_out = Path(out_dir) if out_dir else Path(cfg.out_dir) / "optimizer_study"
    rows: list[dict[str, Any]] = []

    for arm in arms:
        tokens: list[float] = []
        wall: list[float] = []
        finals: list[float] = []
        for seed in seeds:
            arm_cfg = cfg.model_copy(update={"optimizer": arm, "seed": cfg.seed + seed})
            trainer = Trainer(
                arm_cfg,
                device=dev,
                out_dir=base_out / f"{arm}-seed{seed}",
                schedule=arm_cfg.schedule_for(max_steps),
                sink=JsonlSink(None, echo=False),
                compile_model=False,
            )
            try:
                result = trainer.run(
                    loader_factory(arm_cfg, seed, "train"),
                    val_loader=loader_factory(arm_cfg, seed, "val"),
                    max_steps=max_steps,
                    save_at_end=False,
                )
                history = result.state.history
                t = tokens_to_target_loss(history, target_loss)
                w = wall_clock_to_target_loss(history, target_loss)
                if t is not None:
                    tokens.append(t)
                if w is not None:
                    wall.append(w)
                if history:
                    finals.append(history[-1][3])
            finally:
                trainer.close()

        rows.append(
            {
                "arm": arm,
                "target_loss": target_loss,
                "reached_target": len(tokens),
                "tokens_to_target_mean": float(np.mean(tokens)) if tokens else None,
                "tokens_to_target_ci": list(bootstrap_ci(tokens)) if tokens else None,
                "wall_clock_to_target_mean": float(np.mean(wall)) if wall else None,
                "wall_clock_to_target_ci": list(bootstrap_ci(wall)) if wall else None,
                "final_val_loss_mean": float(np.mean(finals)) if finals else None,
                "final_val_loss_ci": list(bootstrap_ci(finals)) if finals else None,
            }
        )

    payload = {
        "name": "optimizer_study",
        "hardware": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "seeds": list(seeds),
        "rows": rows,
        "ci": "bootstrap95",
        "config_hash": cfg.config_hash,
        "max_steps": max_steps,
    }
    return payload


def branch_decay_run(
    cfg: TrainConfig,
    checkpoint: str | Path,
    branch_step: int,
    loader: BatchLoader,
    decay_steps: int | None = None,
    out_dir: str | Path | None = None,
    device: torch.device | str | None = None,
) -> TrainResult:
    """Take a stable-phase checkpoint and spend a short decay on it (§8, ADR 0002).

    This is what turns "the session died at 90%" into a deployable model, and what
    produces the scaling-law points at 250M / 500M / 1B / 1.5B tokens. The parent run's
    checkpoint is not modified; the branch writes to its own directory.
    """
    parent = cfg.schedule_for()
    schedule = branch_decay(parent, branch_step, decay_steps=decay_steps)
    trainer = Trainer(
        cfg,
        device=device,
        out_dir=out_dir or (Path(cfg.out_dir) / f"branch-{branch_step}"),
        schedule=schedule,
    )
    try:
        trainer.resume(checkpoint, loader)
        return trainer.run(loader, max_steps=schedule.total_steps)
    finally:
        trainer.close()


# =====================================================================================
# Entry point
# =====================================================================================
def _synthetic_loaders(
    cfg: TrainConfig, model_cfg: ModelConfig, device: torch.device
) -> tuple[SyntheticLoader, SyntheticLoader]:
    return (
        SyntheticLoader(model_cfg.vocab_size, cfg.micro_batch_size, cfg.seq_len, device=device),
        SyntheticLoader(model_cfg.vocab_size, cfg.micro_batch_size, cfg.seq_len, device=device),
    )


def _real_loaders(cfg: TrainConfig, rank: int, world_size: int) -> tuple[Any, Any]:
    """Build the Phase 3 loaders, imported lazily.

    Deliberately not a module-level import: `localmind.data.loader` is built by another
    task, and this module must import (and its tests must run) without it.
    """
    try:
        from localmind.data.loader import (
            build_loaders,  # type: ignore[import-not-found]
        )
    except ImportError as exc:
        raise SystemExit(
            "localmind.data.loader is not available. Either build the Phase 3 shards, or "
            "run with --synthetic to exercise the loop against a deterministic fake corpus."
        ) from exc
    return build_loaders(
        seq_len=cfg.seq_len,
        micro_batch_size=cfg.micro_batch_size,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
    )


def train_from_config(
    config_path: str | Path,
    resume: str | None = "auto",
    out_dir: str | None = None,
    max_steps: int | None = None,
    device: str | None = None,
    synthetic: bool = False,
    seed: int | None = None,
) -> TrainResult:
    """The body of `main`, callable from a notebook or a test."""
    cfg = TrainConfig.from_yaml(config_path)
    if seed is not None:
        cfg = cfg.model_copy(update={"seed": seed})
    rank, _local_rank, world_size = init_distributed()
    dev = resolve_device(device)

    trainer = Trainer(cfg, device=dev, rank=rank, world_size=world_size, out_dir=out_dir)
    try:
        if synthetic:
            loader, val_loader = _synthetic_loaders(cfg, trainer.model_cfg, dev)
        else:
            loader, val_loader = _real_loaders(cfg, rank, world_size)

        resume_path = trainer.ckpt.resolve_resume(resume)
        if resume_path is not None:
            trainer.resume(resume_path, loader)
            if rank == 0:
                print(f"[train] resumed {resume_path} at step {trainer.state.step}")
        elif rank == 0:
            print(f"[train] fresh run in {trainer.run_dir}")

        if rank == 0:
            print(
                f"[train] config_hash={cfg.config_hash} params={trainer.n_params:,} "
                f"steps={cfg.max_steps} grad_accum={trainer.grad_accum} "
                f"world_size={world_size} device={dev} precision={cfg.precision} "
                f"autocast={trainer.autocast_enabled} scaler={trainer.scaler_enabled} "
                f"optimizer={cfg.optimizer}"
            )
            (trainer.run_dir / "run.json").parent.mkdir(parents=True, exist_ok=True)
            (trainer.run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "train_config": cfg.model_dump(mode="json", by_alias=True),
                        "config_hash": cfg.config_hash,
                        "model_config": trainer.model_cfg.model_dump(mode="json"),
                        "params": trainer.n_params,
                        "world_size": world_size,
                        "device": str(dev),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return trainer.run(loader, val_loader=val_loader, max_steps=max_steps)
    finally:
        trainer.close()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m localmind.train.loop",
        description="LocalMind pretraining loop (implementation.md Phase 4).",
    )
    parser.add_argument("--config", required=True, help="path to configs/train/*.yaml")
    parser.add_argument(
        "--resume",
        default="auto",
        help="'auto' (latest in the run dir), 'none', or a checkpoint path. "
        "Defaults to auto: on a 12h-capped session the common case is relaunching "
        "the same command after a kill.",
    )
    parser.add_argument("--out-dir", default=None, help="override the run directory")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="debug override; the config's max_tokens is the real budget",
    )
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="train against a deterministic in-memory corpus instead of Phase 3 shards",
    )
    args = parser.parse_args(argv)

    result = train_from_config(
        args.config,
        resume=args.resume,
        out_dir=args.out_dir,
        max_steps=args.max_steps,
        device=args.device,
        synthetic=args.synthetic,
        seed=args.seed,
    )
    print(
        f"[train] done: {result.steps_run} steps, "
        f"ce_loss {result.first_loss:.4f} -> {result.last_loss:.4f}, "
        f"{result.state.tokens_seen:,} tokens, {result.wall_clock_s:.1f}s, "
        f"checkpoint={result.checkpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
