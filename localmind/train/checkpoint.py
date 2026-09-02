"""Bit-exact checkpoint/resume. §3.2 item 3: this is the architecture, not a feature.

The free GPU comes with a **12-hour hard kill**. A 1.5B-token run at ~7 h fits, a 3B
stretch run does not, and either way a session can die at any moment with no warning
and no way to appeal. So the unit of progress is not "the run" -- it is "the last
checkpoint", and a checkpoint that resumes to a *slightly different* trajectory is a
checkpoint that silently invalidates every loss curve stitched across a session
boundary. Hence bit-exact, and hence a test that proves it.

Five things must be saved. Miss any one and resume is merely approximate:

============================  ========================================================
model weights                 obvious
optimizer state               Adam's ``exp_avg``/``exp_avg_sq``, Muon's momentum buffer
GradScaler state              the loss scale and its growth tracker (ADR 0001). Resuming
                              at the default 65536 after the run had settled at 512
                              guarantees a burst of skipped steps.
dataloader shard position     otherwise resume re-reads data the model has already seen
RNG state                     python + numpy + torch CPU + **every CUDA device**
============================  ========================================================

`save` writes atomically (temp file then `os.replace`) so a kill *during* a save leaves
the previous checkpoint intact rather than a truncated file that fails to load. Local
saves run on a wall-clock timer (~15 min); Hub pushes on a slower one (~60 min), because
the local disk survives a Python crash but not the session teardown.

DDP: rank 0 writes the full checkpoint. Every rank additionally writes its own
`loader_rank{K}.pt`, because each rank reads a different slice of the shards and a
single loader position would resume 2 ranks onto rank 0's data.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointManager",
    "RngSnapshot",
    "TrainerState",
    "find_latest_checkpoint",
    "load_checkpoint",
    "push_to_hub",
    "save_checkpoint",
    "unwrap_model",
    "unwrapped_state_dict",
]

CHECKPOINT_FORMAT_VERSION = 1

#: Prefixes that wrappers graft onto every key of `state_dict`. Stripping them means a
#: checkpoint written by a `torch.compile`d DDP run loads into a bare CPU model for
#: evaluation, which is exactly what the laptop half of this project needs.
_WRAPPER_PREFIXES: tuple[str, ...] = ("module.", "_orig_mod.")


# ---------------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------------
@dataclass
class RngSnapshot:
    """Every random-number source that can perturb a training step.

    ``numpy`` is included because the data pipeline shuffles shard order with it, and
    ``cuda`` is a list because a 2-GPU job has two independent generators -- capturing
    only ``torch.cuda.get_rng_state()`` (device 0) would leave rank 1 unrestored.

    The numpy calls are the **legacy** global-state API (``np.random.get_state`` /
    ``set_state``) rather than a `np.random.Generator`, and the ``NPY002`` suppressions
    below are deliberate. A `Generator` is a private object: capturing one would restore
    only the streams whose owner handed it to us, and would silently miss every
    ``np.random.*`` call anywhere else in the process. What has to be restored here is
    the *global* stream, and the legacy API is the only thing that addresses it.
    """

    python: Any
    numpy: Any
    torch_cpu: torch.Tensor
    cuda: list[torch.Tensor] | None

    @classmethod
    def capture(cls) -> RngSnapshot:
        cuda_states: list[torch.Tensor] | None = None
        if torch.cuda.is_available():
            cuda_states = list(torch.cuda.get_rng_state_all())
        return cls(
            python=random.getstate(),
            numpy=np.random.get_state(),  # noqa: NPY002 - global stream, see class docstring
            torch_cpu=torch.get_rng_state(),
            cuda=cuda_states,
        )

    def restore(self) -> None:
        random.setstate(self.python)
        np.random.set_state(self.numpy)  # noqa: NPY002 - global stream, see class docstring
        torch.set_rng_state(self.torch_cpu.to(dtype=torch.uint8, device="cpu"))
        if self.cuda is not None and torch.cuda.is_available():
            available = torch.cuda.device_count()
            if len(self.cuda) == available:
                torch.cuda.set_rng_state_all(self.cuda)
            else:
                # Resuming a 2-GPU run on 1 GPU. Restoring what we can beats crashing,
                # but the run is no longer bit-exact and must say so.
                for i in range(min(available, len(self.cuda))):
                    torch.cuda.set_rng_state(self.cuda[i], device=i)

    def to_payload(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "numpy": self.numpy,
            "torch_cpu": self.torch_cpu,
            "cuda": self.cuda,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RngSnapshot:
        return cls(
            python=payload["python"],
            numpy=payload["numpy"],
            torch_cpu=payload["torch_cpu"],
            cuda=payload.get("cuda"),
        )


# ---------------------------------------------------------------------------------
# Trainer state
# ---------------------------------------------------------------------------------
@dataclass
class TrainerState:
    """Scalar bookkeeping that is not owned by any module's `state_dict`."""

    step: int = 0
    tokens_seen: int = 0
    wall_clock_s: float = 0.0
    best_val_loss: float = float("inf")
    #: Appended once per validation: ``(step, tokens, wall_clock_s, val_ce_loss)``.
    #: This is what `tokens_to_target_loss` / `wall_clock_to_target_loss` read, so the
    #: §8 optimizer study survives a session boundary along with the weights.
    history: list[tuple[int, int, float, float]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "wall_clock_s": self.wall_clock_s,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TrainerState:
        return cls(
            step=int(payload.get("step", 0)),
            tokens_seen=int(payload.get("tokens_seen", 0)),
            wall_clock_s=float(payload.get("wall_clock_s", 0.0)),
            best_val_loss=float(payload.get("best_val_loss", float("inf"))),
            history=[tuple(row) for row in payload.get("history", [])],  # type: ignore[misc]
        )


# ---------------------------------------------------------------------------------
# Model unwrapping
# ---------------------------------------------------------------------------------
def unwrap_model(model: nn.Module) -> nn.Module:
    """Peel `DistributedDataParallel` and `torch.compile` wrappers off ``model``."""
    inner = model
    for _ in range(8):  # bounded: DDP(compile(model)) is 2 deep, 8 is paranoia
        next_inner = getattr(inner, "module", None) or getattr(inner, "_orig_mod", None)
        if next_inner is None or next_inner is inner or not isinstance(next_inner, nn.Module):
            break
        inner = next_inner
    return inner


def unwrapped_state_dict(model: nn.Module) -> dict[str, Any]:
    """`state_dict` with wrapper prefixes stripped from every key.

    Belt and braces: `unwrap_model` normally makes this a no-op, but a custom wrapper
    that does not expose ``.module`` would otherwise poison the checkpoint permanently.
    """
    sd = unwrap_model(model).state_dict()
    out: dict[str, Any] = {}
    for key, value in sd.items():
        clean = key
        changed = True
        while changed:
            changed = False
            for prefix in _WRAPPER_PREFIXES:
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :]
                    changed = True
        out[clean] = value
    return out


# ---------------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------------
def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    """Write via a sibling temp file and `os.replace`.

    `os.replace` is atomic on POSIX and on Windows (``MoveFileEx`` with
    ``REPLACE_EXISTING``), so a hard kill mid-write leaves the *previous* checkpoint
    readable instead of a half-written file that costs the whole session.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizers: list[Optimizer],
    scaler: torch.amp.GradScaler | None,
    state: TrainerState,
    loader_state: dict[str, Any] | None = None,
    train_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    config_hash: str | None = None,
    rng: RngSnapshot | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write one checkpoint containing everything needed for a bit-exact resume."""
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": unwrapped_state_dict(model),
        "optimizers": [opt.state_dict() for opt in optimizers],
        "scaler": scaler.state_dict() if scaler is not None else None,
        "state": state.to_payload(),
        "loader": loader_state,
        "rng": (rng or RngSnapshot.capture()).to_payload(),
        "train_config": train_config,
        "model_config": model_config,
        "config_hash": config_hash,
        "torch_version": torch.__version__,
        "saved_at": time.time(),
        "extra": extra or {},
    }
    out = Path(path)
    _atomic_save(payload, out)
    return out


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizers: list[Optimizer] | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> tuple[TrainerState, dict[str, Any] | None, dict[str, Any]]:
    """Restore a checkpoint in place.

    Returns:
        ``(state, loader_state, payload)``. ``loader_state`` is handed back rather than
        applied, because the loader is owned by the data package and this module refuses
        to import it (it is built concurrently, and importing it here would couple the
        checkpoint format to a moving target).

    Raises:
        ValueError: on a format-version mismatch, which is a real risk once a run has
            been resumed across three sessions and a week of commits.
    """
    payload: dict[str, Any] = torch.load(Path(path), map_location=map_location, weights_only=False)
    version = int(payload.get("format_version", 0))
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint {path} has format_version {version}, this build writes "
            f"{CHECKPOINT_FORMAT_VERSION}. Refusing to guess."
        )

    unwrap_model(model).load_state_dict(payload["model"], strict=strict)

    if optimizers is not None:
        saved = payload.get("optimizers") or []
        if len(saved) != len(optimizers):
            raise ValueError(
                f"checkpoint holds {len(saved)} optimizer state(s) but {len(optimizers)} were "
                "passed. The Muon arm uses two optimizers and the AdamW arms use one -- you "
                "are probably resuming one arm's checkpoint into another."
            )
        for opt, sd in zip(optimizers, saved, strict=True):
            opt.load_state_dict(sd)

    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])

    if restore_rng and payload.get("rng"):
        RngSnapshot.from_payload(payload["rng"]).restore()

    return TrainerState.from_payload(payload["state"]), payload.get("loader"), payload


# ---------------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------------
def _step_of(path: Path) -> int:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def find_latest_checkpoint(directory: str | Path, pattern: str = "step_*.pt") -> Path | None:
    """Highest-numbered `step_*.pt` in ``directory``, or None.

    Deliberately *not* a symlink to ``latest.pt``: symlink creation on Windows needs
    developer mode or admin, and this repo is developed on Windows and run on Linux.
    Sorting by the step number embedded in the filename works on both, and survives a
    partially-copied directory.
    """
    d = Path(directory)
    if not d.is_dir():
        return None
    candidates = [p for p in d.glob(pattern) if p.is_file() and _step_of(p) >= 0]
    if not candidates:
        return None
    return max(candidates, key=_step_of)


# ---------------------------------------------------------------------------------
# HF Hub
# ---------------------------------------------------------------------------------
def push_to_hub(
    path: str | Path,
    repo_id: str,
    token: str | None = None,
    path_in_repo: str | None = None,
    private: bool = False,
) -> str:
    """Upload one checkpoint file to a HF Hub model repo. §3.2: session-cap insurance.

    `huggingface_hub` is imported **inside the function**: it is not a hard dependency
    (it arrives via the ``data`` extra), and CONVENTIONS.md forbids network calls at
    import time. A missing token is a `RuntimeError`, not a silent no-op -- discovering
    that hourly pushes were quietly disabled *after* the session is killed is the exact
    failure this function exists to prevent.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise RuntimeError(
            "huggingface_hub is not installed; `uv pip install -e '.[data]'` or disable "
            "hub_push_every_min in the train config."
        ) from exc

    resolved = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not resolved:
        raise RuntimeError("no HF token: set HF_TOKEN, or pass token=..., or disable hub pushes")

    src = Path(path)
    api = HfApi(token=resolved)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=path_in_repo or src.name,
        repo_id=repo_id,
        repo_type="model",
    )
    return f"https://huggingface.co/{repo_id}/blob/main/{path_in_repo or src.name}"


# ---------------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------------
class CheckpointManager:
    """Wall-clock-triggered saving, retention, and Hub pushes.

    Two independent timers, per §8's ``ckpt_every_min: 15`` / ``hub_push_every_min: 60``.
    They are *time* triggers rather than *step* triggers on purpose: step time varies by
    an order of magnitude between the 12M proxy and the 31M run at seq_len 2048, so a
    step interval tuned for one wastes disk or loses an hour on the other. What is being
    bounded is "work lost to a kill", and that is measured in minutes.
    """

    def __init__(
        self,
        out_dir: str | Path,
        every_min: float = 15.0,
        hub_every_min: float | None = 60.0,
        hub_repo_id: str | None = None,
        keep_last: int = 3,
        rank: int = 0,
        clock: Any = time.monotonic,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.every_s = every_min * 60.0
        self.hub_every_s = None if hub_every_min is None else hub_every_min * 60.0
        self.hub_repo_id = hub_repo_id
        self.keep_last = max(1, keep_last)
        self.rank = rank
        self._clock = clock
        now = self._clock()
        self._last_save = now
        self._last_push = now
        self.saved_paths: list[Path] = []
        self.pushed: list[str] = []

    # -- triggers --------------------------------------------------------------------
    def due(self, now: float | None = None) -> bool:
        now = self._clock() if now is None else now
        return (now - self._last_save) >= self.every_s

    def hub_due(self, now: float | None = None) -> bool:
        if self.hub_every_s is None or not self.hub_repo_id:
            return False
        now = self._clock() if now is None else now
        return (now - self._last_push) >= self.hub_every_s

    # -- writing ---------------------------------------------------------------------
    def path_for(self, step: int) -> Path:
        return self.out_dir / f"step_{step:08d}.pt"

    def save(self, step: int, **kwargs: Any) -> Path | None:
        """Write a checkpoint. Non-zero ranks write only their loader position."""
        if self.rank != 0:
            self._save_rank_loader(step, kwargs.get("loader_state"))
            self._last_save = self._clock()
            return None
        path = save_checkpoint(self.path_for(step), **kwargs)
        self._last_save = self._clock()
        self.saved_paths.append(path)
        self._write_manifest(step, path)
        self._prune()
        return path

    def maybe_save(self, step: int, force: bool = False, **kwargs: Any) -> Path | None:
        """Save if the local timer has expired (or ``force``), then maybe push."""
        path: Path | None = None
        if force or self.due():
            path = self.save(step, **kwargs)
        if path is not None and self.hub_due():
            self.maybe_push(path)
        return path

    def maybe_push(self, path: Path) -> str | None:
        """Push to the Hub, swallowing failures.

        A dead network must not kill a run that is otherwise fine -- the local
        checkpoint already exists and the next push is an hour away. The failure is
        printed rather than raised, and recorded so the run summary can say how many
        pushes actually landed.
        """
        if not self.hub_repo_id:
            return None
        try:
            url = push_to_hub(path, self.hub_repo_id)
        except Exception as exc:
            print(f"[checkpoint] hub push failed ({type(exc).__name__}: {exc}); continuing")
            return None
        self._last_push = self._clock()
        self.pushed.append(url)
        return url

    def _save_rank_loader(self, step: int, loader_state: dict[str, Any] | None) -> None:
        if loader_state is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_save(
            {"step": step, "loader": loader_state},
            self.out_dir / f"loader_rank{self.rank:02d}.pt",
        )

    def _write_manifest(self, step: int, path: Path) -> None:
        manifest = {"latest": path.name, "step": step, "saved_at": time.time()}
        tmp = self.out_dir / "latest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, self.out_dir / "latest.json")

    def _prune(self) -> None:
        """Keep the most recent ``keep_last``. Kaggle gives ~20 GB of working disk and a
        31M fp32 checkpoint with two Adam moments is ~370 MB, so unbounded retention
        fills the disk in about four hours -- which then kills the run it was protecting.
        """
        existing = sorted(self.out_dir.glob("step_*.pt"), key=_step_of)
        for stale in existing[: -self.keep_last]:
            stale.unlink(missing_ok=True)

    # -- resuming --------------------------------------------------------------------
    def resolve_resume(self, resume: str | None) -> Path | None:
        """``"auto"`` -> latest in ``out_dir``; ``None``/``"none"`` -> fresh; else a path."""
        if resume in (None, "", "none", "scratch"):
            return None
        if resume == "auto":
            return find_latest_checkpoint(self.out_dir)
        path = Path(resume)
        if not path.exists():
            raise FileNotFoundError(f"--resume {resume}: no such checkpoint")
        return path

    def rank_loader_state(self) -> dict[str, Any] | None:
        """The per-rank loader position written by `_save_rank_loader`."""
        path = self.out_dir / f"loader_rank{self.rank:02d}.pt"
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload.get("loader")

    def export_final(self, src: Path, name: str = "final.pt") -> Path:
        """Copy a checkpoint to a stable filename for downstream phases."""
        dst = self.out_dir / name
        shutil.copy2(src, dst)
        return dst
