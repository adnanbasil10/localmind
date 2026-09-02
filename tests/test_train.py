"""Phase 4 tests: WSD + branch decay, Muon, MFU, instrumentation, bit-exact resume.

Everything here runs on CPU in seconds. Three deliberate choices make that possible
without the tests becoming vacuous:

1. **A ~90k-param model** (`tiny_model_yaml`) rather than the 12M proxy. The proxy's
   16,384-wide LM head costs ~2 s per micro-batch on a CPU, so a 20-step run is a
   minute. The loop's correctness does not depend on width, and the real 12M run is
   reported separately in the task report.
2. **The GradScaler is enabled on CPU.** `torch.amp.GradScaler("cpu", enabled=True)`
   is fully functional, so scale / unscale-before-clip / skip-on-inf / halve-on-overflow
   -- the whole ADR 0001 protocol -- is exercised here rather than deferred to hardware
   nobody in CI has. Only fp16 *autocast* is GPU-gated.
3. **`SyntheticLoader` emits a learnable task**, so "the loss went down" is a real
   assertion rather than a smoke test that a tensor was produced.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from localmind.model import LocalMindTransformer, ModelConfig
from localmind.model.config import count_params, flops_per_step
from localmind.train.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManager,
    RngSnapshot,
    TrainerState,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    unwrap_model,
    unwrapped_state_dict,
)
from localmind.train.loop import (
    BatchLoader,
    MetricBuffer,
    SyntheticLoader,
    TrainConfig,
    Trainer,
    evaluate,
    tokens_to_target_loss,
    train_from_config,
    wall_clock_to_target_loss,
)
from localmind.train.mfu import (
    T4_FP16_PEAK_FLOPS,
    bits_per_byte,
    device_peak_flops,
    flops_per_token,
    flops_per_token_for_config,
    flops_per_token_from_step_counter,
    mfu,
)
from localmind.train.optim.adamw import build_adamw, set_lr, split_param_groups
from localmind.train.optim.muon import (
    Muon,
    build_muon_hybrid,
    build_optimizer_arm,
    muon_update_scale,
    newton_schulz,
    newton_schulz_error,
    split_muon_params,
)
from localmind.train.schedule import (
    SCALING_LAW_TOKEN_BUDGETS,
    BranchDecaySchedule,
    WSDSchedule,
    branch_decay,
    constant_schedule,
    lr_trace,
    scaling_law_branches,
    steps_for_tokens,
)
from torch import nn

REPO = Path(__file__).resolve().parents[1]
PRETRAIN_YAML = REPO / "configs" / "train" / "pretrain.yaml"
SMOKE_YAML = REPO / "configs" / "train" / "smoke.yaml"
PROXY_MODEL_YAML = REPO / "configs" / "model" / "12m_proxy.yaml"
M31_MODEL_YAML = REPO / "configs" / "model" / "31m.yaml"

TINY_MODEL: dict[str, Any] = {
    "name": "LocalMind-tiny-test",
    "vocab_size": 256,
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 2,
    "n_kv_heads": 1,
    "head_dim": 32,
    "ffn_hidden": 128,
    "max_seq_len": 64,
    "rope_theta": 10000.0,
    "qk_norm": True,
    "bias": False,
    "tie_embeddings": True,
    "z_loss": 1.0e-4,
    "attn_dropout": 0.0,
    "init_std": 0.02,
}


# =====================================================================================
# Fixtures
# =====================================================================================
@pytest.fixture
def tiny_model_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(TINY_MODEL), encoding="utf-8")
    return path


@pytest.fixture
def tiny_train_cfg(tiny_model_yaml: Path, tmp_path: Path) -> TrainConfig:
    """A complete, valid `TrainConfig` whose steps take milliseconds.

    ``ckpt_every_min`` is enormous so the wall-clock autosave never fires mid-test and
    perturbs a determinism assertion; tests that want a checkpoint force one.
    """
    return TrainConfig.model_validate(
        {
            "model_config": str(tiny_model_yaml),
            "tokens_per_step": 128,
            "micro_batch_size": 2,
            "seq_len": 32,
            "max_tokens": 128 * 12,
            "peak_lr": 3.0e-3,
            "min_lr_ratio": 0.1,
            "precision": "fp16",
            "torch_compile": "off",
            "log_every": 1,
            "val_every": 4,
            "val_iters": 2,
            "stats_every": 1,
            "ckpt_every_min": 10_000.0,
            "hub_push_every_min": None,
            "out_dir": str(tmp_path / "runs"),
            "seed": 1337,
        }
    )


def make_tiny_model(seed: int = 0) -> LocalMindTransformer:
    torch.manual_seed(seed)
    return LocalMindTransformer(ModelConfig.model_validate(TINY_MODEL), backend="sdpa_math")


def make_loader(cfg: TrainConfig, vocab: int = 256) -> SyntheticLoader:
    return SyntheticLoader(vocab, cfg.micro_batch_size, cfg.seq_len, effective_vocab=64)


def build_trainer(cfg: TrainConfig, out_dir: Path, **kw: Any) -> Trainer:
    return Trainer(cfg, device="cpu", out_dir=out_dir, compile_model=False, **kw)


# =====================================================================================
# 1. Config
# =====================================================================================
def test_shipped_configs_load_and_derive_the_spec_numbers() -> None:
    """`configs/train/*.yaml` are controller-owned; this asserts we read them correctly."""
    pre = TrainConfig.from_yaml(PRETRAIN_YAML)
    assert pre.tokens_per_step == 262_144
    assert pre.peak_lr == 3.0e-3
    assert pre.precision == "fp16"  # ADR 0001
    assert (pre.warmup_frac, pre.stable_frac, pre.decay_frac) == (0.02, 0.80, 0.18)
    # 1.5B tokens at 262,144 tokens/step
    assert pre.max_steps == 5722
    # 16 micro-batches of 16x1024 on one GPU; 8 each on 2xT4.
    assert pre.grad_accum(1) == 16
    assert pre.grad_accum(2) == 8

    smoke = TrainConfig.from_yaml(SMOKE_YAML)
    assert smoke.precision == "fp32"
    # YAML 1.1 parses a bare `off` as boolean False; the config says torch_compile: off.
    assert smoke.torch_compile == "off"
    assert smoke.grad_accum(1) == 8


def test_bf16_is_not_a_valid_precision() -> None:
    """ADR 0001 is enforced by the type, not by a comment: a bf16 config cannot load."""
    raw = yaml.safe_load(PRETRAIN_YAML.read_text(encoding="utf-8"))
    raw["precision"] = "bf16"
    with pytest.raises(Exception, match="precision"):
        TrainConfig.model_validate(raw)


def test_config_hash_is_stable_and_sensitive() -> None:
    a = TrainConfig.from_yaml(PRETRAIN_YAML)
    b = TrainConfig.from_yaml(PRETRAIN_YAML)
    assert a.config_hash == b.config_hash
    assert a.model_copy(update={"peak_lr": 3.1e-3}).config_hash != a.config_hash


def test_grad_accum_rejects_an_indivisible_world_size() -> None:
    cfg = TrainConfig.from_yaml(SMOKE_YAML)
    with pytest.raises(ValueError, match="not divisible"):
        cfg.grad_accum(3)


def test_config_round_trips_through_the_model_config_alias() -> None:
    cfg = TrainConfig.from_yaml(PRETRAIN_YAML)
    dumped = cfg.model_dump(mode="json", by_alias=True)
    assert dumped["model_config"] == "configs/model/31m.yaml"
    assert TrainConfig.model_validate(dumped).config_hash == cfg.config_hash


# =====================================================================================
# 2. WSD schedule (ADR 0002)
# =====================================================================================
def test_wsd_matches_the_spec_phase_split() -> None:
    cfg = TrainConfig.from_yaml(PRETRAIN_YAML)
    s = cfg.schedule_for()
    assert s.total_steps == 5722
    assert s.warmup_steps == round(0.02 * 5722) == 114
    assert s.decay_steps == round(0.18 * 5722) == 1030
    assert s.stable_steps == 5722 - 114 - 1030
    assert s.min_lr == pytest.approx(cfg.peak_lr / 10)  # min_lr_ratio 0.1

    assert s.phase(0) == "warmup"
    assert s.phase(113) == "warmup"
    assert s.phase(114) == "stable"
    assert s.phase(s.decay_start - 1) == "stable"
    assert s.phase(s.decay_start) == "decay"
    assert s.phase(5722) == "post"


def test_wsd_warmup_rises_to_peak_and_never_starts_at_zero() -> None:
    s = WSDSchedule(peak_lr=3e-3, total_steps=1000)
    warm = [s(i) for i in range(s.warmup_steps)]
    assert warm[0] > 0.0, "a zero first step is a wasted step"
    assert warm == sorted(warm)
    assert warm[-1] == pytest.approx(s.peak_lr)
    assert s(s.warmup_steps) == pytest.approx(s.peak_lr)


def test_wsd_decay_is_linear_to_min_lr() -> None:
    s = WSDSchedule(peak_lr=3e-3, total_steps=1000)
    tail = [s(i) for i in range(s.decay_start, s.total_steps)]
    assert tail[0] == pytest.approx(s.peak_lr)
    assert tail == sorted(tail, reverse=True)
    # Linear: equal first differences.
    diffs = [b - a for a, b in itertools.pairwise(tail)]
    assert max(diffs) == pytest.approx(min(diffs), rel=1e-9)
    assert s(s.total_steps) == pytest.approx(s.min_lr)
    assert s(s.total_steps + 5000) == pytest.approx(s.min_lr)


def test_wsd_rejects_fractions_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match=r"must be 1\.0"):
        WSDSchedule(peak_lr=1e-3, total_steps=100, warmup_frac=0.1, stable_frac=0.1, decay_frac=0.1)


def test_lr_trace_length() -> None:
    s = WSDSchedule(peak_lr=1e-3, total_steps=57)
    assert len(lr_trace(s)) == 57


# =====================================================================================
# 3. Branch decay -- the whole reason WSD was chosen (ADR 0002)
# =====================================================================================
def test_branch_decay_reproduces_the_parent_then_decays() -> None:
    parent = WSDSchedule(peak_lr=3e-3, total_steps=10_000)
    branch = branch_decay(parent, branch_step=4000, decay_steps=500)

    # Identical to the parent everywhere before the branch: a branch decay is a *fork*,
    # so the checkpoint it starts from is the parent's, unmodified.
    assert all(branch(i) == parent(i) for i in range(0, 4000, 37))
    # Continuous at the branch: the first decayed step is still at peak.
    assert branch(4000) == pytest.approx(parent.peak_lr)
    assert branch.total_steps == 4500
    tail = [branch(i) for i in range(4000, 4500)]
    assert tail == sorted(tail, reverse=True)
    assert branch(4500) == pytest.approx(parent.min_lr)
    assert branch.phase(4499) == "decay"
    assert branch.phase(4500) == "post"


def test_branch_decay_defaults_to_the_parent_decay_fraction() -> None:
    parent = WSDSchedule(peak_lr=3e-3, total_steps=10_000)
    branch = branch_decay(parent, branch_step=2000)
    assert branch.decay_steps == round(0.18 * 2000) == 360


def test_branch_decay_must_start_in_the_stable_phase() -> None:
    parent = WSDSchedule(peak_lr=3e-3, total_steps=10_000)
    with pytest.raises(ValueError, match="stable phase"):
        branch_decay(parent, branch_step=10, decay_steps=100)  # still warming up
    with pytest.raises(ValueError, match="stable phase"):
        branch_decay(parent, branch_step=9_999, decay_steps=100)  # already decaying


def test_scaling_law_branches_give_the_spec_budgets() -> None:
    """§8: "decay at 250M, 500M, 1B, 1.5B tokens and plot loss vs compute"."""
    cfg = TrainConfig.from_yaml(PRETRAIN_YAML)
    parent = cfg.schedule_for()
    branches = scaling_law_branches(parent, cfg.tokens_per_step)

    assert all(isinstance(b, BranchDecaySchedule) for b in branches)
    got = [b.branch_step * cfg.tokens_per_step for b in branches]
    # 250M/500M/1B land in the stable phase; 1.5B is the parent's own budget, so its
    # branch step is inside the parent decay and is correctly dropped.
    assert len(branches) == 3
    for budget, tokens in zip(SCALING_LAW_TOKEN_BUDGETS[:3], got, strict=True):
        assert abs(tokens - budget) < cfg.tokens_per_step
    assert all(b(b.total_steps) == pytest.approx(parent.min_lr) for b in branches)


def test_constant_schedule_supports_an_open_ended_run() -> None:
    """The 30 GPU-h/week story: run flat, branch whenever the quota evaporates."""
    parent = constant_schedule(peak_lr=3e-3, total_steps=100_000)
    assert parent(50_000) == pytest.approx(3e-3)
    branch = branch_decay(parent, branch_step=50_000, decay_steps=1000)
    assert branch(51_000) == pytest.approx(parent.min_lr)


def test_steps_for_tokens() -> None:
    assert steps_for_tokens(1_500_000_000, 262_144) == 5722
    assert steps_for_tokens(1, 262_144) == 1
    with pytest.raises(ValueError):
        steps_for_tokens(10, 0)


# =====================================================================================
# 4. MFU (§8 formula, reconciled with localmind/model/config.py)
# =====================================================================================
def test_spec_flops_formula_matches_model_config_exactly() -> None:
    """The §8 formula and `model.config.flops_per_step` are the same number.

    ``flops_per_step / tokens == 6*params + 12*L*d*ctx`` for a weight-tied model. If
    this ever fails, one of the two is wrong and the MFU number in the README is a lie.
    """
    for path in (M31_MODEL_YAML, PROXY_MODEL_YAML):
        cfg = ModelConfig.from_yaml(path)
        assert cfg.tie_embeddings
        for batch, seq in ((16, 1024), (4, 256)):
            if seq > cfg.max_seq_len:
                continue
            spec = flops_per_token(count_params(cfg), cfg.n_layers, cfg.d_model, seq)
            from_model = flops_per_token_from_step_counter(cfg, batch, seq)
            assert spec == pytest.approx(from_model, rel=1e-12)
            assert flops_per_token_for_config(cfg, seq) == pytest.approx(spec, rel=1e-12)
            # And the step counter is just 3x the forward, over the same tokens.
            assert flops_per_step(cfg, batch, seq) == pytest.approx(
                from_model * batch * seq, rel=1e-12
            )


def test_untied_embeddings_are_where_the_two_formulas_diverge() -> None:
    """Documented divergence: the spec formula double-counts an untied embedding table."""
    cfg = ModelConfig.model_validate({**TINY_MODEL, "tie_embeddings": False})
    literal = flops_per_token(count_params(cfg), cfg.n_layers, cfg.d_model, cfg.max_seq_len)
    honest = flops_per_token_for_config(cfg, cfg.max_seq_len)
    assert literal > honest
    assert literal - honest == pytest.approx(6 * cfg.vocab_size * cfg.d_model)


def test_mfu_is_a_fraction_of_peak() -> None:
    cfg = ModelConfig.from_yaml(M31_MODEL_YAML)
    params = count_params(cfg)
    fpt = flops_per_token(params, cfg.n_layers, cfg.d_model, 1024)
    # By construction: running at exactly peak gives MFU 1.0.
    at_peak = T4_FP16_PEAK_FLOPS / fpt
    assert mfu(params, cfg.n_layers, cfg.d_model, 1024, at_peak) == pytest.approx(1.0)
    # §8 expects 10-15% on a first 2xT4 run. Sanity-check the order of magnitude.
    two_t4 = device_peak_flops("Tesla T4", world_size=2)
    assert two_t4 == pytest.approx(2 * T4_FP16_PEAK_FLOPS)
    realistic = mfu(params, cfg.n_layers, cfg.d_model, 1024, 200_000, peak_flops=two_t4)
    assert 0.0 < realistic < 1.0


def test_bits_per_byte() -> None:
    # ln(2) nats per token over 1 byte/token is exactly 1 bit per byte.
    assert bits_per_byte(math.log(2), 1.0) == pytest.approx(1.0)
    assert bits_per_byte(math.log(2), 4.0) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        bits_per_byte(1.0, 0.0)


def test_device_peak_flops_falls_back_but_never_returns_zero() -> None:
    assert device_peak_flops("cpu") == T4_FP16_PEAK_FLOPS
    assert device_peak_flops("Tesla P100-PCIE-16GB") < T4_FP16_PEAK_FLOPS
    assert device_peak_flops(None, world_size=4) == 4 * T4_FP16_PEAK_FLOPS


# =====================================================================================
# 5. Parameter grouping (§8: wd=0.1, exclude norms/bias/embeddings)
# =====================================================================================
def test_weight_decay_excludes_norms_bias_and_the_tied_embedding() -> None:
    model = make_tiny_model()
    groups = split_param_groups(model, weight_decay=0.1)
    by_name = {g["name"]: g for g in groups}
    assert by_name["decay"]["weight_decay"] == 0.1
    assert by_name["no_decay"]["weight_decay"] == 0.0

    decay_ids = {id(p) for p in by_name["decay"]["params"]}
    # The tied embedding (== the LM head) must not be decayed: §8's trap list.
    assert id(model.tok_emb.weight) not in decay_ids
    assert id(model.lm_head.weight) not in decay_ids
    # Every norm scale is 1-D and excluded.
    assert all(p.ndim >= 2 for p in by_name["decay"]["params"])
    assert all(p.ndim == 1 or p.ndim >= 2 for p in by_name["no_decay"]["params"])
    # No parameter is in both groups, and every trainable parameter is in exactly one.
    no_decay_ids = {id(p) for p in by_name["no_decay"]["params"]}
    assert not (decay_ids & no_decay_ids)
    assert decay_ids | no_decay_ids == {id(p) for p in model.parameters() if p.requires_grad}


def test_tied_weights_appear_once_in_the_optimizer() -> None:
    """A tied tensor listed twice gets its update applied twice -- a silent 2x LR."""
    model = make_tiny_model()
    opt = build_adamw(model, lr=1e-3)
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids))


def test_set_lr_applies_the_group_scale_across_several_optimizers() -> None:
    model = make_tiny_model()
    base = build_optimizer_arm(model, "adamw", lr=3e-3)
    doubled = build_optimizer_arm(model, "adamw_2x", lr=3e-3)
    hybrid = build_optimizer_arm(model, "muon", lr=3e-3)

    set_lr(base, 1e-3)
    set_lr(doubled, 1e-3)
    set_lr(hybrid, 1e-3)
    assert all(g["lr"] == pytest.approx(1e-3) for g in base[0].param_groups)
    assert all(g["lr"] == pytest.approx(2e-3) for g in doubled[0].param_groups)
    assert len(hybrid) == 2
    assert all(g["lr"] == pytest.approx(1e-3) for o in hybrid for g in o.param_groups)


def test_unknown_optimizer_arm_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown optimizer arm"):
        build_optimizer_arm(make_tiny_model(), "lion", lr=1e-3)


# =====================================================================================
# 6. Muon
# =====================================================================================
@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (128, 128), (256, 64)])
def test_newton_schulz_equalises_singular_values(shape: tuple[int, int]) -> None:
    """The one check that catches a transposed matmul or a wrong coefficient sign.

    A badly implemented Muon still trains -- just worse than AdamW -- so "the loss went
    down" cannot be the test. The orthogonalisation itself has to be verified.
    """
    torch.manual_seed(0)
    # A deliberately ill-conditioned matrix: this is what Muon exists to fix.
    U, _, Vh = torch.linalg.svd(torch.randn(*shape), full_matrices=False)
    k = min(shape)
    spectrum = torch.logspace(0, -2, k)
    G = U @ torch.diag(spectrum) @ Vh
    assert (spectrum.max() / spectrum.min()) == pytest.approx(100.0)  # condition number before

    lo, hi = newton_schulz_error(G, steps=5)
    assert 0.5 < lo <= hi < 1.5, f"singular values {lo}..{hi} are not near 1"
    assert (hi / lo) < 2.0, "5 quintic steps should compress cond(G) from 100 to <2"

    # Jordan's coefficients do not converge to exactly 1 -- they are tuned for slope at
    # 0, and the iteration settles into a fixed band of [~0.682, ~1.134]. Pinning that
    # band is the sharpest available regression test: a transposed matmul or a flipped
    # coefficient sign lands somewhere else entirely.
    lo_c, hi_c = newton_schulz_error(G, steps=12)
    assert lo_c == pytest.approx(0.6818, abs=2e-3)
    assert hi_c == pytest.approx(1.1344, abs=2e-3)


def test_more_newton_schulz_steps_tighten_the_band() -> None:
    torch.manual_seed(1)
    U, _, Vh = torch.linalg.svd(torch.randn(96, 48), full_matrices=False)
    G = U @ torch.diag(torch.logspace(0, -3, 48)) @ Vh
    lo2, hi2 = newton_schulz_error(G, steps=2)
    lo8, hi8 = newton_schulz_error(G, steps=8)
    assert (hi8 / lo8) < (hi2 / lo2)


def test_newton_schulz_is_transpose_equivariant() -> None:
    """``ns(G^T) == ns(G)^T``. The implementation transposes tall matrices internally to
    keep the iteration on the short side; this proves that optimisation is invisible."""
    torch.manual_seed(2)
    G = torch.randn(80, 40)
    a = newton_schulz(G, steps=5)
    b = newton_schulz(G.T.contiguous(), steps=5).T
    assert torch.allclose(a, b, atol=1e-5)


def test_newton_schulz_rejects_bfloat16() -> None:
    """ADR 0001. Every reference Muon runs this iteration in bf16; the T4 cannot."""
    with pytest.raises(ValueError, match="bfloat16 is forbidden"):
        newton_schulz(torch.randn(8, 8), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="bfloat16 is forbidden"):
        Muon([nn.Parameter(torch.randn(4, 4))], ns_dtype=torch.bfloat16)


def test_newton_schulz_survives_a_zero_gradient() -> None:
    out = newton_schulz(torch.zeros(8, 16), steps=5)
    assert torch.isfinite(out).all()


def test_newton_schulz_requires_a_matrix() -> None:
    with pytest.raises(ValueError, match="2-D"):
        newton_schulz(torch.randn(8))


def test_muon_update_scale_modes() -> None:
    # Moonlight/Kimi: 0.2 * sqrt(max(dim)) -- matches AdamW's update RMS so one LR serves both.
    assert muon_update_scale((1408, 512), "moonlight") == pytest.approx(0.2 * math.sqrt(1408))
    # Jordan: aspect-ratio correction only.
    assert muon_update_scale((128, 64), "jordan") == pytest.approx(math.sqrt(2.0))
    assert muon_update_scale((64, 128), "jordan") == pytest.approx(1.0)
    assert muon_update_scale((10, 10), "none") == 1.0
    with pytest.raises(ValueError):
        muon_update_scale((4, 4), "nope")  # type: ignore[arg-type]


def test_muon_split_is_the_kimi_k2_recipe() -> None:
    """Muon on the 2-D hidden matrices; AdamW on embeddings / LM head / norms."""
    cfg = ModelConfig.from_yaml(M31_MODEL_YAML)
    model = LocalMindTransformer(cfg, backend="sdpa_math")
    split = split_muon_params(model)

    assert split.muon_names, "no hidden matrices found"
    # Exactly the attention and FFN projections, in every layer.
    expected = {
        f"blocks.{i}.{leaf}.weight"
        for i in range(cfg.n_layers)
        for leaf in ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "ffn.gate", "ffn.up", "ffn.down")
    }
    assert set(split.muon_names) == expected
    assert all(p.ndim == 2 for p in split.muon_params)

    adamw_ids = {id(p) for g in split.adamw_groups for p in g["params"]}
    assert id(model.tok_emb.weight) in adamw_ids
    assert id(model.lm_head.weight) in adamw_ids  # tied: same tensor, still not Muon's
    muon_ids = {id(p) for p in split.muon_params}
    assert not (muon_ids & adamw_ids)
    assert muon_ids | adamw_ids == {id(p) for p in model.parameters() if p.requires_grad}

    summary = split.summary()
    assert summary["muon_params"] + summary["adamw_params"] == sum(
        p.numel() for p in model.parameters()
    )
    # The 8.4M tied embedding is 27% of 31M, so Muon covers well under 80% of the model.
    assert 0.5 < summary["muon_fraction"] < 0.8


def test_muon_rejects_a_one_dimensional_parameter() -> None:
    """A norm scale reaching Muon means the split upstream is broken. Fail loudly."""
    p = nn.Parameter(torch.randn(8))
    p.grad = torch.randn(8)
    opt = Muon([p], lr=1e-3)
    with pytest.raises(ValueError, match="1-D parameter"):
        opt.step()


def test_muon_minimises_an_ill_conditioned_quadratic_faster_than_sgd() -> None:
    """Muon's claim, in miniature: equalised singular values mean no starved direction.

    ``f(W) = ||W A - B||^2`` with a badly conditioned ``A`` is exactly the situation
    Muon is built for. Plain SGD-momentum at the same LR crawls along the small
    singular directions; the orthogonalised update does not.
    """
    torch.manual_seed(0)
    A = torch.diag(torch.logspace(0, -2, 32)) @ torch.randn(32, 32)
    target = torch.randn(32, 32)

    def run(make_opt: Any) -> float:
        torch.manual_seed(7)
        W = nn.Parameter(torch.zeros(32, 32))
        opt = make_opt([W])
        for _ in range(60):
            opt.zero_grad(set_to_none=True)
            ((W @ A) - target).pow(2).mean().backward()
            opt.step()
        return float(((W @ A) - target).pow(2).mean().detach())

    muon_loss = run(lambda ps: Muon(ps, lr=0.02, update_scale="jordan"))
    sgd_loss = run(lambda ps: torch.optim.SGD(ps, lr=0.02, momentum=0.95))
    start = float(target.pow(2).mean())
    assert muon_loss < start
    assert muon_loss < sgd_loss


def test_muon_applies_decoupled_weight_decay() -> None:
    p = nn.Parameter(torch.ones(4, 4))
    p.grad = torch.zeros(4, 4)  # only weight decay can move it
    opt = Muon([p], lr=0.1, weight_decay=0.5)
    opt.step()
    assert torch.allclose(p.detach(), torch.full((4, 4), 1.0 - 0.1 * 0.5))


def test_muon_does_not_clobber_the_gradient_tensor() -> None:
    """The reference implementation writes the Nesterov blend into `grad` in place; the
    logging path still needs `p.grad` afterwards for the grad-norm buffer."""
    p = nn.Parameter(torch.randn(8, 8))
    p.grad = torch.randn(8, 8)
    before = p.grad.clone()
    Muon([p], lr=1e-3).step()
    assert torch.equal(p.grad, before)


def test_muon_state_round_trips() -> None:
    model = make_tiny_model()
    opts, _ = build_muon_hybrid(model, lr=1e-3)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    for o in opts:
        o.step()
    payload = [o.state_dict() for o in opts]

    fresh_model = make_tiny_model()
    fresh, _ = build_muon_hybrid(fresh_model, lr=1e-3)
    for o, sd in zip(fresh, payload, strict=True):
        o.load_state_dict(sd)
    buf_a = next(iter(opts[0].state.values()))["momentum_buffer"]
    buf_b = next(iter(fresh[0].state.values()))["momentum_buffer"]
    assert torch.equal(buf_a, buf_b)


# =====================================================================================
# 7. Instrumentation
# =====================================================================================
def test_metric_buffer_batches_the_host_transfer() -> None:
    device = torch.device("cpu")
    buf = MetricBuffer(("loss", "grad_norm"), capacity=3, device=device, n_layers=2)
    for i in range(3):
        assert not buf.full
        buf.record(
            device_values={"loss": torch.tensor(float(i)), "grad_norm": float(i) * 2},
            host_values={"step": i, "lr": 1e-3},
            weight_norms=torch.tensor([float(i), float(i) + 0.5]),
            act_rms=torch.tensor([1.0, 2.0]),
        )
    assert buf.full
    with pytest.raises(RuntimeError, match="full"):
        buf.record({}, {})

    rows = buf.flush()
    assert len(rows) == 3
    assert [r["loss"] for r in rows] == [0.0, 1.0, 2.0]
    assert [r["grad_norm"] for r in rows] == [0.0, 2.0, 4.0]
    assert rows[2]["weight_norm"] == pytest.approx([2.0, 2.5])
    assert rows[1]["act_rms"] == pytest.approx([1.0, 2.0])
    assert rows[0]["step"] == 0 and rows[0]["lr"] == 1e-3
    assert buf.flush() == []  # drained


def test_step_logs_every_metric_the_spec_requires(
    tiny_train_cfg: TrainConfig, tmp_path: Path
) -> None:
    """§8's instrumentation list, checked field by field."""
    trainer = build_trainer(tiny_train_cfg, tmp_path / "logs")
    try:
        loader = make_loader(tiny_train_cfg)
        trainer.step(loader)
        rows = trainer._flush(loader)
    finally:
        trainer.close()

    row = rows[0]
    required = {
        "loss",
        "ce_loss",
        "z_loss",
        "bits_per_byte",
        "grad_norm",
        "weight_norm",
        "act_rms",
        "lr",
        "mfu",
        "tokens_per_sec",
        "scaler_scale",
        "shard_index",
        "step",
        "tokens",
        "phase",
    }
    assert required <= set(row)
    assert len(row["weight_norm"]) == TINY_MODEL["n_layers"]
    assert len(row["act_rms"]) == TINY_MODEL["n_layers"]
    assert all(v > 0 for v in row["act_rms"]), "activation RMS hooks never fired"
    assert row["scaler_scale"] == pytest.approx(65536.0)  # fp16 => scaler live
    assert row["lr"] > 0 and row["mfu"] > 0 and row["tokens_per_sec"] > 0
    assert row["bits_per_byte"] == pytest.approx(
        row["ce_loss"] / math.log(2) / tiny_train_cfg.bytes_per_token
    )
    # GPU memory is a CUDA-only field, by construction.
    assert ("gpu_mem_alloc_gb" in row) == torch.cuda.is_available()


def test_metrics_are_written_as_jsonl(tiny_train_cfg: TrainConfig, tmp_path: Path) -> None:
    run_dir = tmp_path / "jsonl"
    trainer = build_trainer(tiny_train_cfg, run_dir)
    try:
        loader = make_loader(tiny_train_cfg)
        trainer.run(loader, max_steps=2, save_at_end=False)
    finally:
        trainer.close()
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 1


# =====================================================================================
# 8. fp16 + GradScaler protocol (ADR 0001) -- exercised on CPU
# =====================================================================================
def test_per_optimizer_inf_checks_alone_would_poison_the_model() -> None:
    """Characterisation test for the hazard `Trainer.step` guards against.

    `GradScaler.step` skips *per optimizer*, using the inf check taken when that
    optimizer was unscaled. With the two-optimizer Muon arm, a NaN in a hidden matrix
    marks Muon only. `clip_grad_norm_` then computes ``clip_coef = 1 / (NaN + 1e-6)``
    and multiplies *every* gradient by it -- so AdamW, whose own check passed a moment
    earlier, steps the embeddings with NaN and the model is gone. NaN gradients are not
    exotic under fp16: ``inf - inf`` and ``0 * inf`` in a softmax both produce them.

    This test pins the raw torch behaviour, so the guard in `Trainer.step` has an
    executable reason to exist rather than a comment.
    """
    model = make_tiny_model()
    opts = build_optimizer_arm(model, "muon", lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    assert scaler.is_enabled(), "CPU GradScaler must be functional for this test to mean anything"

    x = torch.randint(0, TINY_MODEL["vocab_size"], (2, 16))
    out = model(x, targets=x)
    assert out.loss is not None
    scaler.scale(out.loss).backward()
    model.blocks[0].ffn.gate.weight.grad[0, 0] = float("nan")  # a Muon-group tensor

    for opt in opts:
        scaler.unscale_(opt)
    gnorm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
    assert not torch.isfinite(gnorm), "the global norm is the signal the guard uses"
    for opt in opts:
        scaler.step(opt)  # unguarded: this is the dangerous call
    scaler.update()

    # The embedding is in the AdamW group, not Muon's. It is now NaN anyway.
    assert torch.isnan(model.tok_emb.weight).any()


def test_trainer_skips_the_whole_step_and_halves_the_scale_on_overflow(
    tiny_train_cfg: TrainConfig, tmp_path: Path
) -> None:
    """The guarded path: one overflow anywhere skips *every* optimizer, intact.

    ADR 0001: a single halving after an inf is the scaler working; repeated halving is
    the signal that something upstream is wrong, which is why the scale and the skip
    count are both logged every step.
    """
    cfg = tiny_train_cfg.model_copy(update={"optimizer": "muon"})
    trainer = build_trainer(cfg, tmp_path / "overflow")
    try:
        # Force an fp16-style overflow in a Muon-group tensor, on every backward.
        hooked = trainer.raw_model.blocks[0].ffn.gate.weight
        hooked.register_hook(lambda g: torch.full_like(g, float("inf")))

        before = {k: v.clone() for k, v in unwrapped_state_dict(trainer.raw_model).items()}
        scale_before = trainer.scaler.get_scale()
        trainer.step(make_loader(cfg))
        row = trainer._flush(make_loader(cfg))[0]
        after = unwrapped_state_dict(trainer.raw_model)
    finally:
        trainer.close()

    assert row["step_skipped"] == 1
    assert row["skipped_steps"] == 1
    assert trainer.scaler.get_scale() == pytest.approx(scale_before * 0.5)
    for key, expected in before.items():
        assert torch.equal(after[key], expected), f"{key} moved during a skipped step"


def test_gradients_are_unscaled_before_clipping(
    tiny_train_cfg: TrainConfig, tmp_path: Path
) -> None:
    """Clipping scaled gradients would clip to ``grad_clip / 65536`` and stall the run.

    fp16 (scaler live, scale 65536) and fp32 (scaler off) must produce *identical* grad
    norms, because scaling by a power of two and dividing it back out is exact in fp32.
    A missing or mis-ordered `unscale_` shows up here as a 65,536x discrepancy.
    """

    def grad_norm_for(precision: str) -> float:
        cfg = tiny_train_cfg.model_copy(update={"precision": precision})
        trainer = build_trainer(cfg, tmp_path / f"clip-{precision}")
        try:
            torch.manual_seed(4242)
            loader = make_loader(cfg)
            trainer.step(loader)
            return float(trainer._flush(loader)[0]["grad_norm"])
        finally:
            trainer.close()

    fp16_norm = grad_norm_for("fp16")
    fp32_norm = grad_norm_for("fp32")
    assert fp16_norm == pytest.approx(fp32_norm, rel=1e-5)
    assert fp16_norm < 1e3, "grad norm looks like it is still multiplied by the loss scale"


def test_grad_accumulation_equals_one_large_batch(
    tiny_train_cfg: TrainConfig, tmp_path: Path
) -> None:
    """Dividing by ``grad_accum`` must reproduce the gradient of the *mean* over the
    global batch, not its sum. Getting this wrong scales the effective LR by 16."""
    cfg = tiny_train_cfg.model_copy(update={"precision": "fp32"})
    trainer = build_trainer(cfg, tmp_path / "accum")
    try:
        assert trainer.grad_accum == 2
        torch.manual_seed(11)
        loader = make_loader(cfg)
        batches = [next(loader) for _ in range(2)]

        class Fixed:
            def __init__(self) -> None:
                self.i = 0

            def __iter__(self) -> Any:
                return self

            def __next__(self) -> Any:
                b = batches[self.i % 2]
                self.i += 1
                return b

            def state_dict(self) -> dict[str, Any]:
                return {"i": self.i}

            def load_state_dict(self, s: Any) -> None:
                self.i = s["i"]

            @property
            def shard_index(self) -> int:
                return 0

        for opt in trainer.optimizers:
            opt.zero_grad(set_to_none=True)
        trainer._forward_backward(Fixed(), collect_stats=False)
        accumulated = [p.grad.detach().clone() for p in trainer.params if p.grad is not None]

        for opt in trainer.optimizers:
            opt.zero_grad(set_to_none=True)
        big_x = torch.cat([b[0] for b in batches], dim=0)
        big_y = torch.cat([b[1] for b in batches], dim=0)
        out = trainer.raw_model(big_x, targets=big_y)
        assert out.loss is not None
        out.loss.backward()
        one_shot = [p.grad.detach().clone() for p in trainer.params if p.grad is not None]
    finally:
        trainer.close()

    assert len(accumulated) == len(one_shot)
    for a, b in zip(accumulated, one_shot, strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-4)


# =====================================================================================
# 9. Checkpointing -- §3.2 item 3
# =====================================================================================
def test_rng_snapshot_round_trips_every_source() -> None:
    random.seed(0)
    np.random.seed(0)  # noqa: NPY002 - the global stream is exactly what is under test
    torch.manual_seed(0)
    snap = RngSnapshot.capture()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))  # noqa: NPY002 - the global stream is exactly what is under test

    random.seed(999)
    np.random.seed(999)  # noqa: NPY002 - the global stream is exactly what is under test
    torch.manual_seed(999)
    random.random()
    np.random.rand()  # noqa: NPY002 - the global stream is exactly what is under test
    torch.rand(1)

    snap.restore()
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected  # noqa: NPY002 - the global stream is exactly what is under test


def test_synthetic_loader_is_deterministic_and_resumable() -> None:
    """The loop's fixture must itself be resumable, or the resume test proves nothing."""
    torch.manual_seed(5)
    a = SyntheticLoader(256, 2, 16)
    for _ in range(3):  # advance the loader before snapshotting it
        next(a)
    state = a.state_dict()
    rng = RngSnapshot.capture()
    rest = [next(a)[0].clone() for _ in range(3)]

    torch.manual_seed(1234)
    b = SyntheticLoader(256, 2, 16)
    b.load_state_dict(state)
    rng.restore()
    assert all(torch.equal(x, y) for x, y in zip(rest, [next(b)[0] for _ in range(3)], strict=True))
    assert a.state_dict()["position"] == 6
    assert isinstance(a.shard_index, int)
    # And the emitted task is the learnable chain it claims to be.
    x, y, mask = next(a)
    assert mask is None
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_synthetic_loader_satisfies_the_batch_loader_protocol() -> None:
    assert isinstance(SyntheticLoader(256, 2, 16), BatchLoader)


def test_checkpoint_carries_everything_needed(tiny_train_cfg: TrainConfig, tmp_path: Path) -> None:
    trainer = build_trainer(tiny_train_cfg, tmp_path / "full")
    try:
        loader = make_loader(tiny_train_cfg)
        trainer.run(loader, max_steps=2, save_at_end=True)
        path = find_latest_checkpoint(trainer.run_dir)
        assert path is not None
    finally:
        trainer.close()

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert set(payload) >= {"model", "optimizers", "scaler", "state", "loader", "rng"}
    assert payload["scaler"]["scale"] > 0  # GradScaler state, per ADR 0001
    assert payload["loader"]["position"] == 4  # 2 steps x grad_accum 2
    assert payload["rng"]["numpy"] is not None and payload["rng"]["python"] is not None
    assert payload["config_hash"] == tiny_train_cfg.config_hash
    assert payload["train_config"]["model_config"].endswith("tiny.yaml")
    # No wrapper prefixes leaked into the weights.
    assert not any(k.startswith(("module.", "_orig_mod.")) for k in payload["model"])


def test_resume_is_bit_exact(tiny_train_cfg: TrainConfig, tmp_path: Path) -> None:
    """**The §8 DoD test.** A run split across a session boundary must be
    indistinguishable from one that was never interrupted.

    Six steps straight through, versus three + checkpoint + a *fresh* Trainer + three
    more, with the global RNG deliberately trashed in between so that a forgotten RNG
    source fails the test instead of passing by luck. Both the per-step losses and the
    final weights must match exactly, not approximately.
    """
    cfg = tiny_train_cfg

    # --- reference: six uninterrupted steps ---------------------------------------
    ref = build_trainer(cfg, tmp_path / "ref")
    try:
        ref_result = ref.run(make_loader(cfg), max_steps=6, save_at_end=False)
        ref_losses = [r["ce_loss"] for r in ref_result.metrics]
        ref_weights = {k: v.clone() for k, v in unwrapped_state_dict(ref.raw_model).items()}
        ref_scale = ref.scaler.get_scale()
    finally:
        ref.close()

    # --- first half, then a checkpoint --------------------------------------------
    part = build_trainer(cfg, tmp_path / "split")
    try:
        loader = make_loader(cfg)
        first = part.run(loader, max_steps=3, save_at_end=True)
        ckpt = find_latest_checkpoint(part.run_dir)
        assert ckpt is not None
    finally:
        part.close()

    # The session dies. Something else runs. Every RNG stream is now wrong.
    random.seed(4242)
    np.random.seed(4242)  # noqa: NPY002 - the global stream is exactly what is under test
    torch.manual_seed(4242)
    torch.randn(1000)
    np.random.rand(1000)  # noqa: NPY002 - the global stream is exactly what is under test

    # --- second half, from the checkpoint ------------------------------------------
    resumed = build_trainer(cfg, tmp_path / "split")
    try:
        loader2 = make_loader(cfg)
        state = resumed.resume(ckpt, loader2)
        assert state.step == 3
        assert loader2.state_dict()["position"] == 6  # 3 steps x grad_accum 2
        second = resumed.run(loader2, max_steps=6, save_at_end=False)
        got_losses = [r["ce_loss"] for r in first.metrics + second.metrics]
        got_weights = unwrapped_state_dict(resumed.raw_model)
        got_scale = resumed.scaler.get_scale()
    finally:
        resumed.close()

    assert len(got_losses) == len(ref_losses) == 6
    assert got_losses == ref_losses, "resumed losses diverge from the uninterrupted run"
    assert got_scale == ref_scale, "GradScaler scale was not restored"
    for key, expected in ref_weights.items():
        assert torch.equal(got_weights[key], expected), f"weight {key} diverged after resume"


def test_resume_without_rng_restoration_actually_diverges(
    tiny_train_cfg: TrainConfig, tmp_path: Path
) -> None:
    """A control for the test above: prove the assertion has teeth.

    If restoring RNG made no difference, `test_resume_is_bit_exact` would pass whether
    or not `RngSnapshot` worked. Skipping the restore must break the run.
    """
    cfg = tiny_train_cfg
    ref = build_trainer(cfg, tmp_path / "ctl-ref")
    try:
        ref_losses = [
            r["ce_loss"] for r in ref.run(make_loader(cfg), max_steps=4, save_at_end=False).metrics
        ]
    finally:
        ref.close()

    part = build_trainer(cfg, tmp_path / "ctl")
    try:
        loader = make_loader(cfg)
        head = part.run(loader, max_steps=2, save_at_end=True)
        ckpt = find_latest_checkpoint(part.run_dir)
        assert ckpt is not None
    finally:
        part.close()

    resumed = build_trainer(cfg, tmp_path / "ctl")
    try:
        loader2 = make_loader(cfg)
        state, loader_state, _ = load_checkpoint(
            ckpt,
            resumed.raw_model,
            optimizers=resumed.optimizers,
            scaler=resumed.scaler,
            restore_rng=False,  # <-- the bug being simulated
        )
        resumed.state = state
        assert loader_state is not None
        loader2.load_state_dict(loader_state)
        torch.manual_seed(31337)
        tail = [r["ce_loss"] for r in resumed.run(loader2, max_steps=4, save_at_end=False).metrics]
    finally:
        resumed.close()

    assert [r["ce_loss"] for r in head.metrics] + tail != ref_losses


def test_checkpoint_manager_timer_prunes_and_finds_the_latest(tmp_path: Path) -> None:
    now = [0.0]
    mgr = CheckpointManager(
        tmp_path / "ck", every_min=15.0, hub_every_min=None, keep_last=2, clock=lambda: now[0]
    )
    model = make_tiny_model()
    opts = build_optimizer_arm(model, "adamw", lr=1e-3)
    kwargs: dict[str, Any] = {
        "model": model,
        "optimizers": opts,
        "scaler": None,
        "state": TrainerState(),
    }

    assert not mgr.due()
    assert mgr.maybe_save(1, **kwargs) is None  # timer has not expired
    now[0] = 15 * 60 + 1
    assert mgr.due()
    assert mgr.maybe_save(1, **kwargs) is not None
    assert not mgr.due()  # timer resets on save

    for step in (2, 3, 4):
        mgr.save(step, **kwargs)
    kept = sorted(p.name for p in (tmp_path / "ck").glob("step_*.pt"))
    assert kept == ["step_00000003.pt", "step_00000004.pt"]  # keep_last=2
    latest = find_latest_checkpoint(tmp_path / "ck")
    assert latest is not None and latest.name == "step_00000004.pt"
    assert json.loads((tmp_path / "ck" / "latest.json").read_text())["step"] == 4
    assert not list((tmp_path / "ck").glob("*.tmp"))  # atomic writes leave no debris


def test_hub_push_timer_is_independent_of_the_local_timer(tmp_path: Path) -> None:
    now = [0.0]
    mgr = CheckpointManager(
        tmp_path / "ck", every_min=15.0, hub_every_min=60.0, hub_repo_id="x/y", clock=lambda: now[0]
    )
    now[0] = 20 * 60
    assert mgr.due() and not mgr.hub_due()
    now[0] = 61 * 60
    assert mgr.hub_due()
    # No repo id configured => never due, and never a surprise network call.
    quiet = CheckpointManager(tmp_path / "q", hub_every_min=60.0, clock=lambda: now[0])
    assert not quiet.hub_due()


def test_resolve_resume_modes(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "r")
    assert mgr.resolve_resume("none") is None
    assert mgr.resolve_resume(None) is None
    assert mgr.resolve_resume("auto") is None  # nothing there yet
    model = make_tiny_model()
    mgr.save(7, model=model, optimizers=[], scaler=None, state=TrainerState(step=7))
    assert mgr.resolve_resume("auto") == mgr.path_for(7)
    with pytest.raises(FileNotFoundError):
        mgr.resolve_resume(str(tmp_path / "nope.pt"))


def test_checkpoint_rejects_a_foreign_format_version(tmp_path: Path) -> None:
    model = make_tiny_model()
    path = save_checkpoint(tmp_path / "c.pt", model, [], None, TrainerState())
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["format_version"] = 99
    torch.save(payload, path)
    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(path, model)


def test_checkpoint_rejects_the_wrong_optimizer_arm(tmp_path: Path) -> None:
    """An AdamW checkpoint has one optimizer state; the Muon arm has two."""
    model = make_tiny_model()
    adamw = build_optimizer_arm(model, "adamw", lr=1e-3)
    path = save_checkpoint(tmp_path / "a.pt", model, adamw, None, TrainerState())
    muon = build_optimizer_arm(make_tiny_model(), "muon", lr=1e-3)
    with pytest.raises(ValueError, match="optimizer state"):
        load_checkpoint(path, model, optimizers=muon)


def test_unwrap_strips_ddp_and_compile_prefixes() -> None:
    inner = make_tiny_model()

    class FakeWrapper(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

    wrapped = FakeWrapper(FakeWrapper(inner))
    assert unwrap_model(wrapped) is inner
    assert not any(k.startswith("module.") for k in unwrapped_state_dict(wrapped))
    assert set(unwrapped_state_dict(wrapped)) == set(inner.state_dict())


# =====================================================================================
# 10. The loop actually learns
# =====================================================================================
def test_loss_decreases_on_a_real_training_run(tiny_train_cfg: TrainConfig, tmp_path: Path) -> None:
    """The point of the whole file: gradients flow, the sign is right, the model learns.

    `SyntheticLoader` emits ``x[t] = (7*x[t-1] + 13) mod 64``, so a working loop drives
    cross-entropy well below ``ln(64) = 4.16`` -- the entropy of guessing uniformly.
    """
    cfg = tiny_train_cfg.model_copy(update={"peak_lr": 6e-3, "max_tokens": 128 * 60})
    trainer = build_trainer(cfg, tmp_path / "learn")
    try:
        loader = make_loader(cfg)
        result = trainer.run(loader, max_steps=60, save_at_end=False)
    finally:
        trainer.close()

    losses = [r["ce_loss"] for r in result.metrics]
    assert len(losses) == 60
    assert all(math.isfinite(v) for v in losses)
    assert losses[0] > math.log(64) * 0.8, "the run did not start near the uniform-guess entropy"
    assert losses[-1] < losses[0] * 0.6, f"loss barely moved: {losses[0]:.3f} -> {losses[-1]:.3f}"
    # The scale should not be collapsing: repeated halving means fp16 overflow (ADR 0001).
    scales = [r["scaler_scale"] for r in result.metrics]
    assert scales[-1] >= scales[0] / 4


@pytest.mark.parametrize("arm", ["adamw", "muon", "adamw_2x"])
def test_every_optimizer_arm_trains(arm: str, tiny_train_cfg: TrainConfig, tmp_path: Path) -> None:
    cfg = tiny_train_cfg.model_copy(update={"optimizer": arm, "max_tokens": 128 * 40})
    trainer = build_trainer(cfg, tmp_path / f"arm-{arm}")
    try:
        result = trainer.run(make_loader(cfg), max_steps=40, save_at_end=False)
    finally:
        trainer.close()
    losses = [r["ce_loss"] for r in result.metrics]
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0], f"{arm} did not reduce the loss"


def test_validation_reports_ce_loss_not_the_regularised_loss(
    tiny_train_cfg: TrainConfig, tmp_path: Path
) -> None:
    trainer = build_trainer(tiny_train_cfg, tmp_path / "val")
    try:
        val = evaluate(trainer.raw_model, make_loader(tiny_train_cfg), 3, torch.device("cpu"))
        assert val.ndim == 0, "evaluate must return a device tensor, not a synced float"
        assert 0.0 < float(val) < 20.0
        result = trainer.run(
            make_loader(tiny_train_cfg),
            val_loader=make_loader(tiny_train_cfg),
            max_steps=8,
            save_at_end=False,
        )
    finally:
        trainer.close()
    # val_every=4 over 8 steps => two history points.
    assert len(result.state.history) == 2
    assert all(math.isfinite(row[3]) for row in result.state.history)
    assert result.state.best_val_loss <= min(row[3] for row in result.state.history)


def test_tokens_and_wall_clock_to_target_can_disagree() -> None:
    """§8 asks for both because they measure different things.

    Arm A reaches the target in fewer tokens; arm B reaches it in less wall clock. That
    is the whole reason the study reports two numbers instead of one.
    """
    #            (step, tokens, seconds, val_loss)
    sample_efficient = [(1, 100, 10.0, 5.0), (2, 200, 20.0, 3.0)]
    fast = [(1, 400, 4.0, 5.0), (2, 800, 8.0, 3.0)]
    assert tokens_to_target_loss(sample_efficient, 4.0) == pytest.approx(150.0)
    assert tokens_to_target_loss(fast, 4.0) == pytest.approx(600.0)
    assert wall_clock_to_target_loss(sample_efficient, 4.0) == pytest.approx(15.0)
    assert wall_clock_to_target_loss(fast, 4.0) == pytest.approx(6.0)
    assert tokens_to_target_loss(sample_efficient, 1.0) is None  # never reached


def test_cli_runs_end_to_end(tiny_model_yaml: Path, tmp_path: Path) -> None:
    """`python -m localmind.train.loop --config ... --synthetic`, which is what the
    justfile's `train-smoke` recipe calls (minus `--synthetic`)."""
    cfg_path = tmp_path / "cli.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "model_config": str(tiny_model_yaml),
                "tokens_per_step": 128,
                "micro_batch_size": 2,
                "seq_len": 32,
                "max_tokens": 128 * 3,
                "peak_lr": 3.0e-3,
                "precision": "fp16",
                "torch_compile": "off",
                "log_every": 1,
                "val_every": 100,
                "out_dir": str(tmp_path / "cli-runs"),
                "ckpt_every_min": 10000.0,
                "hub_push_every_min": None,
            }
        ),
        encoding="utf-8",
    )
    result = train_from_config(cfg_path, resume="auto", synthetic=True, device="cpu")
    assert result.steps_run == 3
    assert result.checkpoint is not None and result.checkpoint.exists()
    assert math.isfinite(result.last_loss)

    # `--resume auto` a second time picks the checkpoint up and runs zero further steps.
    again = train_from_config(cfg_path, resume="auto", synthetic=True, device="cpu")
    assert again.steps_run == 0
    assert again.state.step == 3


def test_optimizer_study_produces_a_conventions_shaped_payload(
    tiny_train_cfg: TrainConfig, tmp_path: Path
) -> None:
    from localmind.train.loop import run_optimizer_study

    cfg = tiny_train_cfg.model_copy(update={"val_every": 4, "val_iters": 2})
    payload = run_optimizer_study(
        cfg,
        loader_factory=lambda c, seed, split: make_loader(c),
        arms=("adamw", "muon"),
        seeds=(0,),
        max_steps=8,
        target_loss=4.0,
        out_dir=tmp_path / "study",
        device="cpu",
    )
    assert set(payload) >= {"name", "hardware", "seeds", "rows", "ci"}
    assert payload["ci"] == "bootstrap95"
    assert [r["arm"] for r in payload["rows"]] == ["adamw", "muon"]
    for row in payload["rows"]:
        # §8: BOTH metrics, always.
        assert "tokens_to_target_mean" in row
        assert "wall_clock_to_target_mean" in row
        assert row["final_val_loss_mean"] is not None


# =====================================================================================
# 11. GPU-only
# =====================================================================================
@pytest.mark.gpu
def test_fp16_autocast_and_scaler_on_cuda(tiny_train_cfg: TrainConfig, tmp_path: Path) -> None:
    """The real T4 path: fp16 autocast + a live GradScaler. Never bf16 (ADR 0001)."""
    if not torch.cuda.is_available():  # pragma: no cover - guarded by the marker
        pytest.skip("no CUDA")
    trainer = Trainer(tiny_train_cfg, device="cuda", out_dir=tmp_path / "gpu", compile_model=False)
    try:
        assert trainer.autocast_enabled and trainer.scaler_enabled
        result = trainer.run(make_loader(tiny_train_cfg), max_steps=5, save_at_end=False)
    finally:
        trainer.close()
    assert all(math.isfinite(r["ce_loss"]) for r in result.metrics)
    assert all("gpu_mem_alloc_gb" in r for r in result.metrics)
    assert result.metrics[-1]["mfu"] > 0.0
