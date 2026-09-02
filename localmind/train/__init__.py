"""LocalMind training package (implementation.md Phase 4).

The hand-written pretraining loop, the WSD schedule and its branch decays, the optimizer
study, MFU accounting, and bit-exact checkpoint/resume.

Run it::

    uv run python -m localmind.train.loop --config configs/train/pretrain.yaml --resume auto
    torchrun --nproc_per_node=2 -m localmind.train.loop --config configs/train/pretrain.yaml

Public surface
--------------
`TrainConfig`        `configs/train/*.yaml`, validated; ``precision`` admits fp16|fp32 only
`Trainer` / `train_from_config`   the loop
`BatchLoader`        the Protocol the loop needs from `localmind.data.loader`
`SyntheticLoader`    a deterministic, learnable fake corpus -- the loop's test fixture
`WSDSchedule` / `branch_decay`    2%/80%/18%, and decay branched from any stable step
`build_optimizer_arm`             adamw | muon | adamw_2x
`CheckpointManager`  model + optimizer + GradScaler + loader position + RNG
`mfu` / `bits_per_byte`           §8's formulas, reconciled with `model.config`

Two project-wide rules this package enforces rather than assumes (ADR 0001): ``bf16``
is not a valid ``precision`` value and `newton_schulz` rejects `torch.bfloat16`, because
the target GPU is a Turing T4 with no bf16 tensor cores.

`schedule` and `mfu` import cheaply; `loop`, `checkpoint` and `optim` need torch.
"""

from localmind.train.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManager,
    RngSnapshot,
    TrainerState,
    find_latest_checkpoint,
    load_checkpoint,
    push_to_hub,
    save_checkpoint,
)
from localmind.train.loop import (
    Batch,
    BatchLoader,
    JsonlSink,
    LayerProbe,
    MetricBuffer,
    SyntheticLoader,
    TrainConfig,
    Trainer,
    TrainResult,
    branch_decay_run,
    evaluate,
    run_optimizer_study,
    seed_everything,
    tokens_to_target_loss,
    train_from_config,
    wall_clock_to_target_loss,
)
from localmind.train.mfu import (
    T4_FP16_PEAK_FLOPS,
    ThroughputReport,
    bits_per_byte,
    device_peak_flops,
    flops_per_token,
    flops_per_token_for_config,
    mfu,
    throughput_report,
)
from localmind.train.optim import (
    Muon,
    build_adamw,
    build_muon_hybrid,
    build_optimizer_arm,
    newton_schulz,
    set_lr,
    split_muon_params,
    split_param_groups,
)
from localmind.train.schedule import (
    SCALING_LAW_TOKEN_BUDGETS,
    BranchDecaySchedule,
    LRSchedule,
    WSDSchedule,
    branch_decay,
    constant_schedule,
    lr_trace,
    scaling_law_branches,
    steps_for_tokens,
    wsd_schedule,
)

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "SCALING_LAW_TOKEN_BUDGETS",
    "T4_FP16_PEAK_FLOPS",
    "Batch",
    "BatchLoader",
    "BranchDecaySchedule",
    "CheckpointManager",
    "JsonlSink",
    "LRSchedule",
    "LayerProbe",
    "MetricBuffer",
    "Muon",
    "RngSnapshot",
    "SyntheticLoader",
    "ThroughputReport",
    "TrainConfig",
    "TrainResult",
    "Trainer",
    "TrainerState",
    "WSDSchedule",
    "bits_per_byte",
    "branch_decay",
    "branch_decay_run",
    "build_adamw",
    "build_muon_hybrid",
    "build_optimizer_arm",
    "constant_schedule",
    "device_peak_flops",
    "evaluate",
    "find_latest_checkpoint",
    "flops_per_token",
    "flops_per_token_for_config",
    "load_checkpoint",
    "lr_trace",
    "mfu",
    "newton_schulz",
    "push_to_hub",
    "run_optimizer_study",
    "save_checkpoint",
    "scaling_law_branches",
    "seed_everything",
    "set_lr",
    "split_muon_params",
    "split_param_groups",
    "steps_for_tokens",
    "throughput_report",
    "tokens_to_target_loss",
    "train_from_config",
    "wall_clock_to_target_loss",
    "wsd_schedule",
]
