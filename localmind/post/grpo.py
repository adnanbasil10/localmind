"""Phase 5d -- GRPO with verifiable rewards (SS9 5d).

The post-R1 recipe, and the cheapest reinforcement learning in this repo: **no value
network, no reward model, no judge calls**. It works at 31M scale precisely *because* the
reward is verifiable rather than learned -- the grader head must emit strict JSON *and* be
correct against ground truth, and both halves of that are a function call, not a model.

Two design points carry the whole method:

**Group-relative advantage.** PPO needs a critic to tell it what return to expect from a
state. GRPO gets the same baseline for free by sampling a *group* of ``G`` completions
from the same prompt and normalising their rewards against each other::

    A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

The group mean *is* the baseline, so the critic -- which at this scale would be a second
network as large as the policy -- disappears. SS9 pins ``G = 8``. The normalisation is
strictly within a group and never across the batch: two prompts of different difficulty
have different reward scales, and pooling them would let an easy prompt's rewards set the
baseline for a hard one. :func:`group_advantages` enforces that shape.

**The degenerate group.** When every completion in a group earns the same reward -- all
correct, or all wrong -- the standard deviation is zero and there is genuinely no
preference to learn. :func:`group_advantages` returns exact zeros there rather than
dividing by ``eps`` and manufacturing an enormous advantage out of floating-point noise.
That single branch is the difference between GRPO training and GRPO exploding, and it is
asserted directly in the tests.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from localmind.post.sft import (
    IGNORE_INDEX,
    ChatTokenizer,
    Job,
    StageResult,
    label_of,
    parse_teacher_output,
    prompt_messages,
    run_stage,
    safe_decode,
    seed_prompts,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor, nn

    from localmind.post.kd import StudentSampler

__all__ = [
    "ADVANTAGE_EPS",
    "DEFAULT_GROUP_SIZE",
    "GRPOConfig",
    "GRPOResult",
    "Group",
    "RewardBreakdown",
    "RewardConfig",
    "RolloutSample",
    "group_advantages",
    "grpo_loss",
    "main",
    "run_grpo",
    "sample_group",
    "sequence_token_logprobs",
    "verifiable_reward",
]

#: SS9 5d. Restated in `configs/train/grpo.yaml`; never written inline anywhere else.
DEFAULT_GROUP_SIZE = 8

#: Guards the division in `group_advantages`. Only ever reached for a group whose rewards
#: differ but barely; an *exactly* degenerate group is short-circuited to zeros before it.
ADVANTAGE_EPS = 1e-4


# ------------------------------------------------------------------------------------ #
# The verifiable reward
# ------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class RewardBreakdown:
    """A reward you can audit: which half fired, and why.

    Reported as parts rather than one float because "reward went up" is uninformative
    when the two components pull in different directions -- a policy that learns perfect
    JSON and forgets the answer looks identical to one that learns the answer and forgets
    the JSON, if all you log is the sum.
    """

    format_valid: bool
    correct: bool
    total: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_valid": self.format_valid,
            "correct": self.correct,
            "total": self.total,
            "detail": self.detail,
        }


class RewardConfig(BaseModel):
    """Weights for the two verifiable components."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_weight: float = Field(default=0.5, ge=0.0)
    correctness_weight: float = Field(default=0.5, ge=0.0)
    #: SS9 5d reads "format OK *and* correct". Correctness is therefore gated on format:
    #: an answer that happens to contain the right word inside prose has not solved the
    #: task the control plane actually poses, which is "emit a machine-readable verdict".
    require_format_for_correctness: bool = True


def verifiable_reward(
    job: Job, output: str, gold: str, *, cfg: RewardConfig | None = None
) -> RewardBreakdown:
    """Reward = format-valid **AND** correct against ground truth. No model involved.

    ``format_valid`` means the output parses under `localmind.post.sft.parse_teacher_output`
    -- strict JSON with the right keys and value domains for the router and the grader.
    ``correct`` compares the canonicalised label to ``gold``.

    The grader's numeric ``score`` field deliberately does **not** enter correctness: it
    is not verifiable against anything, and rewarding it would smuggle a learned
    judgement into a reward that is supposed to be a pure function. That is the whole
    reason this works at 31M -- SS9: "it works at small scale precisely *because* the
    reward is verifiable rather than learned".
    """
    rc = cfg or RewardConfig()
    canonical = parse_teacher_output(job, output)
    format_valid = canonical is not None
    if not format_valid:
        return RewardBreakdown(False, False, 0.0, "unparseable output")

    predicted = label_of(job, canonical or "")
    correct = predicted == gold
    if rc.require_format_for_correctness and not format_valid:  # pragma: no cover
        correct = False

    total = rc.format_weight * float(format_valid) + rc.correctness_weight * float(correct)
    return RewardBreakdown(format_valid, correct, total, f"predicted={predicted!r} gold={gold!r}")


# ------------------------------------------------------------------------------------ #
# Group-relative advantage -- the thing that removes the value network
# ------------------------------------------------------------------------------------ #
def group_advantages(
    rewards: Tensor, *, eps: float = ADVANTAGE_EPS, unbiased: bool = False
) -> Tensor:
    """Normalise rewards **within each group**: ``A = (r - mean) / (std + eps)``.

    Args:
        rewards: ``(G,)`` for one group or ``(P, G)`` for ``P`` prompts of ``G``
            completions each. The last axis is always the group; there is no mode in
            which this function normalises across prompts, because that would let one
            prompt's difficulty set another prompt's baseline.
        eps: floor on the denominator for groups with tiny but non-zero spread.
        unbiased: ``ddof=1``. Default ``False`` (population std) matches the GRPO paper
            and the reference implementations; at ``G = 8`` the two differ by ~7%, which
            the learning rate absorbs but which would silently change the effective step
            size if it were toggled by accident.

    Returns:
        Advantages with the same shape as ``rewards``.

    A group whose rewards are all identical returns **exact zeros**. That is not an edge
    case to be tolerated, it is the common case early in training (all 8 completions
    malformed) and late in training (all 8 correct), and dividing by ``eps`` there would
    turn floating-point dust into advantages of magnitude 1e4 and destroy the policy.
    """
    import torch

    if rewards.dim() not in (1, 2):
        raise ValueError(f"rewards must be (G,) or (P, G), got {tuple(rewards.shape)}")
    r = rewards.float()
    if r.shape[-1] < 2:
        raise ValueError(f"a group needs at least 2 completions, got {r.shape[-1]}")

    mean = r.mean(dim=-1, keepdim=True)
    std = r.std(dim=-1, unbiased=unbiased, keepdim=True)
    advantages = (r - mean) / (std + eps)
    # Exactly-degenerate groups carry no preference information. Zero them explicitly
    # rather than letting `(r - mean) / eps` fabricate one out of rounding error.
    degenerate = (std <= 0).expand_as(advantages)
    return torch.where(degenerate, torch.zeros_like(advantages), advantages)


# ------------------------------------------------------------------------------------ #
# Rollouts
# ------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class RolloutSample:
    """One sampled completion, its reward, and the ids needed to score it again."""

    prompt_ids: list[int]
    completion_ids: list[int]
    text: str
    reward: RewardBreakdown


@dataclass
class Group:
    """``G`` completions sampled from one prompt -- GRPO's unit of comparison."""

    job: Job
    inputs: dict[str, str]
    gold: str
    samples: list[RolloutSample] = field(default_factory=list)

    @property
    def rewards(self) -> list[float]:
        return [s.reward.total for s in self.samples]

    @property
    def degenerate(self) -> bool:
        """True when every completion earned the same reward: no gradient signal here."""
        return len(set(self.rewards)) <= 1

    def to_dict(self) -> dict[str, Any]:
        n = max(len(self.samples), 1)
        return {
            "job": self.job,
            "gold": self.gold,
            "rewards": self.rewards,
            "degenerate": self.degenerate,
            "format_valid_frac": sum(s.reward.format_valid for s in self.samples) / n,
            "correct_frac": sum(s.reward.correct for s in self.samples) / n,
        }


def sample_group(
    sampler: StudentSampler,
    tokenizer: ChatTokenizer,
    job: Job,
    inputs: Mapping[str, str],
    gold: str,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    max_new_tokens: int = 24,
    reward_cfg: RewardConfig | None = None,
) -> Group:
    """Sample ``group_size`` completions from the policy and score each verifiably.

    The sampler is asked for ``group_size`` *distinct* rollouts, so it must be stochastic;
    a greedy sampler collapses every group to a single point, every group is degenerate,
    and GRPO silently learns nothing. :class:`GRPOResult` surfaces the degenerate fraction
    for exactly this reason, rather than letting a flat reward curve be the only clue.
    """
    clean = {k: v for k, v in inputs.items() if k != "GOLD"}
    prompt_ids = tokenizer.encode_chat(prompt_messages(job, clean), add_generation_prompt=True)
    group = Group(job=job, inputs=clean, gold=gold)
    for g in range(group_size):
        completion_ids = sampler.generate(prompt_ids, max_new_tokens, sample_index=g)
        text = safe_decode(tokenizer, completion_ids).strip()
        group.samples.append(
            RolloutSample(
                prompt_ids=list(prompt_ids),
                completion_ids=list(completion_ids),
                text=text,
                reward=verifiable_reward(job, text, gold, cfg=reward_cfg),
            )
        )
    return group


def _pad_rollouts(
    samples: Sequence[RolloutSample], pad_id: int, *, max_len: int
) -> tuple[Tensor, Tensor]:
    """``(input_ids, labels)`` for a list of rollouts: prompt masked, right padded."""
    import torch

    rows: list[tuple[list[int], list[int]]] = []
    for s in samples:
        prompt, completion = list(s.prompt_ids), list(s.completion_ids)
        # Trim the PROMPT HEAD, never the completion. Right-truncating
        # [*prompt, *completion][:max_len] silently deletes the completion whenever the
        # prompt alone exceeds max_len -- every label becomes IGNORE_INDEX, the loss is
        # taken over nothing, the gradient is zero, and the run reports success. The
        # grader prompt is ~400 tokens, so any seq_len below that turned GRPO into a
        # no-op. `sft.encode_sft_example` already drops the prompt head for this reason.
        budget = max_len - len(completion)
        if budget < 0:
            # Even the completion alone overflows; keep its head and train on what fits.
            prompt, completion = [], completion[:max_len]
        else:
            prompt = prompt[-budget:] if budget else []
        full = [*prompt, *completion]
        start = len(prompt)
        labels = [full[k] if k >= start else IGNORE_INDEX for k in range(len(full))]
        rows.append((full[:-1], labels[1:]))
    width = max(max((len(r[0]) for r in rows), default=1), 1)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    label_ids = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, labels) in enumerate(rows):
        if ids:
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            label_ids[i, : len(labels)] = torch.tensor(labels, dtype=torch.long)
    return input_ids, label_ids


def sequence_token_logprobs(
    model: nn.Module, input_ids: Tensor, labels: Tensor
) -> tuple[Tensor, Tensor]:
    """``(token_logprobs, mask)``, both ``(B, T)``.

    Per-token rather than per-sequence because GRPO's surrogate clips the *token* ratio;
    clipping a summed sequence ratio would let one wild token drag a whole trajectory
    outside the trust region without ever being clipped itself.
    """
    import torch

    logits = model(input_ids).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    mask = labels != IGNORE_INDEX
    safe = labels.masked_fill(~mask, 0)
    token_lp = torch.gather(logprobs, 2, safe.unsqueeze(-1)).squeeze(-1)
    return token_lp * mask.to(token_lp.dtype), mask


# ------------------------------------------------------------------------------------ #
# The loss
# ------------------------------------------------------------------------------------ #
def grpo_loss(
    token_logprobs: Tensor,
    old_token_logprobs: Tensor,
    advantages: Tensor,
    mask: Tensor,
    *,
    clip_eps: float = 0.2,
    kl_coef: float = 0.0,
    ref_token_logprobs: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """The clipped GRPO surrogate. No value network appears anywhere in this function.

    ``L = -(1/N) sum_i (1/|o_i|) sum_t [ min(rho_it A_i, clip(rho_it, 1-e, 1+e) A_i) ]
          + kl_coef * KL``

    with ``rho_it = exp(log pi(o_it) - log pi_old(o_it))`` and ``A_i`` the group-relative
    advantage, constant along a trajectory -- there is no per-token credit assignment to
    be had from a reward that only exists once the sequence is finished.

    The per-sequence token *mean* (rather than a sum) is deliberate: a sum makes a long
    completion's gradient proportional to its length, so the policy learns to be verbose
    for reasons that have nothing to do with the reward.

    The KL term uses the k3 estimator ``exp(r) - r - 1`` with ``r = log pi_ref - log pi``,
    which is non-negative per token; ``kl_coef = 0`` drops the reference entirely, which
    is the cheap configuration SS9 describes and the one that is safe when the reward is
    verifiable and therefore cannot be gamed.
    """
    import torch

    if token_logprobs.shape != old_token_logprobs.shape:
        raise ValueError("policy and old log-probs must have the same shape")
    if advantages.dim() != 1 or advantages.shape[0] != token_logprobs.shape[0]:
        raise ValueError(
            f"advantages must be (B,) matching batch {token_logprobs.shape[0]}, "
            f"got {tuple(advantages.shape)}"
        )
    if clip_eps <= 0:
        raise ValueError(f"clip_eps must be positive, got {clip_eps}")

    m = mask.to(token_logprobs.dtype)
    ratio = torch.exp(token_logprobs - old_token_logprobs)
    adv = advantages.to(token_logprobs.dtype).unsqueeze(-1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_token = -torch.min(unclipped, clipped)

    kl_mean = 0.0
    if kl_coef > 0.0:
        if ref_token_logprobs is None:
            raise ValueError("kl_coef > 0 requires ref_token_logprobs")
        r = ref_token_logprobs - token_logprobs
        kl_per_token = torch.exp(r) - r - 1.0
        per_token = per_token + kl_coef * kl_per_token
        kl_mean = float(((kl_per_token * m).sum() / m.sum().clamp_min(1.0)).detach())

    lengths = m.sum(dim=-1).clamp_min(1.0)
    loss = ((per_token * m).sum(dim=-1) / lengths).mean()

    with torch.no_grad():
        outside = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).to(m.dtype)
        clip_frac = float((outside * m).sum() / m.sum().clamp_min(1.0))
        mean_ratio = float((ratio * m).sum() / m.sum().clamp_min(1.0))
    metrics = {
        "policy_loss": float(loss.detach()),
        "mean_ratio": mean_ratio,
        "clip_frac": clip_frac,
        "mean_advantage": float(advantages.mean()),
        "kl_to_ref": kl_mean,
    }
    return loss, metrics


# ------------------------------------------------------------------------------------ #
# Config and run
# ------------------------------------------------------------------------------------ #
class GRPOConfig(BaseModel):
    """``configs/train/grpo.yaml``."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, protected_namespaces=()
    )

    model_config_path: str = Field(alias="model_config")
    #: The grader is the job with a programmatic reward: strict JSON *and* a ground-truth
    #: label. The rewriter has no verifiable answer, which is why it gets DPO instead.
    job: Literal["grader", "router"] = "grader"
    #: SS9 5d: "Group size 8, no value network."
    group_size: int = Field(default=DEFAULT_GROUP_SIZE, ge=2)
    #: There is no critic to configure. Present as an explicit ``False`` so a reader does
    #: not have to infer the absence of a value network from silence.
    use_value_network: Literal[False] = False

    n_prompts: int = Field(default=2_000, gt=0)
    max_new_tokens: int = Field(default=32, gt=0)
    sampling_temperature: float = Field(default=1.0, gt=0.0)

    clip_eps: float = Field(default=0.2, gt=0.0)
    kl_coef: float = Field(default=0.0, ge=0.0)
    format_weight: float = Field(default=0.5, ge=0.0)
    correctness_weight: float = Field(default=0.5, ge=0.0)

    seq_len: int = Field(default=256, gt=0)
    peak_lr: float = Field(default=1.0e-6, gt=0.0)
    warmup_frac: float = Field(default=0.1, ge=0.0, lt=1.0)
    min_lr_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(default=0.0, ge=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    #: ADR 0001: bf16 absent on purpose.
    precision: Literal["fp16", "fp32"] = "fp16"

    out_dir: str = "artifacts/post/grpo"
    seed: int = 1337

    @property
    def reward_config(self) -> RewardConfig:
        return RewardConfig(
            format_weight=self.format_weight, correctness_weight=self.correctness_weight
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> GRPOConfig:
        import yaml

        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping")
        return cls.model_validate(raw)


@dataclass
class GRPOResult:
    """Stage metrics plus the reward decomposition SS9 5d is actually about."""

    stage_result: StageResult
    mean_reward: float
    format_valid_rate: float
    correct_rate: float
    degenerate_group_frac: float
    group_size: int
    groups: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.stage_result.to_dict(),
            "mean_reward": self.mean_reward,
            "format_valid_rate": self.format_valid_rate,
            "correct_rate": self.correct_rate,
            "degenerate_group_frac": self.degenerate_group_frac,
            "group_size": self.group_size,
            "use_value_network": False,
            "groups": self.groups,
            "warnings": self.warnings,
        }


def run_grpo(
    policy: nn.Module,
    tokenizer: ChatTokenizer,
    sampler: StudentSampler,
    *,
    cfg: GRPOConfig | None = None,
    reference: nn.Module | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    n_prompts: int = 8,
    max_new_tokens: int = 24,
    peak_lr: float = 1.0e-6,
    clip_eps: float = 0.2,
    kl_coef: float = 0.0,
    seq_len: int = 256,
    seed: int = 1337,
) -> GRPOResult:
    """Sample groups, score them verifiably, take one clipped policy step per group.

    A single gradient step per rollout batch keeps the run strictly on-policy, so
    ``pi_old == pi`` when the ratio is formed and every ratio starts at exactly 1. The
    clipped surrogate is still the right objective to write down: the moment anyone
    raises the inner epoch count above one it becomes load-bearing, and an implementation
    that only happened to work at ratio 1 would fail silently then.
    """
    import torch

    g = cfg.group_size if cfg else group_size
    n = cfg.n_prompts if cfg else n_prompts
    mnt = cfg.max_new_tokens if cfg else max_new_tokens
    lr = cfg.peak_lr if cfg else peak_lr
    eps = cfg.clip_eps if cfg else clip_eps
    kc = cfg.kl_coef if cfg else kl_coef
    max_len = cfg.seq_len if cfg else seq_len
    run_seed = cfg.seed if cfg else seed
    reward_cfg = cfg.reward_config if cfg else RewardConfig()
    job_filter: Job = cfg.job if cfg else "grader"

    pool = [p for p in seed_prompts(max(n * 4, n + 8), seed=run_seed) if p[0] == job_filter][:n]
    if not pool:
        raise ValueError(f"no {job_filter!r} prompts in the seed pool")

    groups = [
        sample_group(
            sampler,
            tokenizer,
            job,
            inputs,
            inputs.get("GOLD", ""),
            group_size=g,
            max_new_tokens=mnt,
            reward_cfg=reward_cfg,
        )
        for job, inputs in pool
    ]

    prepared: list[tuple[Tensor, Tensor, Tensor]] = []
    for grp in groups:
        input_ids, labels = _pad_rollouts(grp.samples, tokenizer.pad_id, max_len=max_len)
        rewards = torch.tensor(grp.rewards, dtype=torch.float32)
        prepared.append((input_ids, labels, group_advantages(rewards)))

    ref_cache: list[Tensor | None] = [None] * len(prepared)
    if kc > 0.0 and reference is not None:
        reference.eval()
        with torch.no_grad():
            ref_cache = [
                sequence_token_logprobs(reference, ids, labels)[0] for ids, labels, _ in prepared
            ]

    policy.eval()
    with torch.no_grad():
        old_cache = [sequence_token_logprobs(policy, ids, labels)[0] for ids, labels, _ in prepared]

    def step_fn(step: int) -> tuple[Tensor, dict[str, float]]:
        input_ids, labels, advantages = prepared[step]
        token_lp, mask = sequence_token_logprobs(policy, input_ids, labels)
        return grpo_loss(
            token_lp,
            old_cache[step],
            advantages,
            mask,
            clip_eps=eps,
            kl_coef=kc,
            ref_token_logprobs=ref_cache[step],
        )

    stage = run_stage(
        policy,
        step_fn,
        stage="grpo",
        steps=len(prepared),
        peak_lr=lr,
        cfg=cfg,
        warmup_frac=cfg.warmup_frac if cfg else 0.1,
        min_lr_ratio=cfg.min_lr_ratio if cfg else 1.0,
        grad_clip=cfg.grad_clip if cfg else 1.0,
        seed=run_seed,
    )
    stage.summary["group_size"] = g
    stage.summary["use_value_network"] = False

    all_samples = [s for grp in groups for s in grp.samples]
    n_samples = max(len(all_samples), 1)
    result = GRPOResult(
        stage_result=stage,
        mean_reward=sum(s.reward.total for s in all_samples) / n_samples,
        format_valid_rate=sum(s.reward.format_valid for s in all_samples) / n_samples,
        correct_rate=sum(s.reward.correct for s in all_samples) / n_samples,
        degenerate_group_frac=sum(grp.degenerate for grp in groups) / max(len(groups), 1),
        group_size=g,
        groups=[grp.to_dict() for grp in groups],
    )
    if result.degenerate_group_frac >= 1.0:
        result.warnings.append(
            "every group was degenerate (all G completions scored identically): there is "
            "no learning signal at all. Either the sampler is not stochastic, or the task "
            "is uniformly too easy or too hard for the current policy."
        )
    elif result.degenerate_group_frac > 0.8:
        result.warnings.append(
            f"{result.degenerate_group_frac:.0%} of groups were degenerate; most of the "
            "rollout budget produced zero gradient. Raise the sampling temperature or "
            "re-balance prompt difficulty."
        )
    if result.format_valid_rate < 0.1:
        result.warnings.append(
            f"format-valid rate is {result.format_valid_rate:.1%}: the policy is not "
            "emitting parseable JSON, so the correctness half of the reward is "
            "unreachable. Run more SFT before GRPO."
        )
    return result


# ------------------------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------------------------ #
def main(argv: Sequence[str] | None = None) -> int:
    """``python -m localmind.post.grpo --config configs/train/grpo.yaml --group-size 8``"""
    parser = argparse.ArgumentParser(description="Phase 5d GRPO with verifiable rewards")
    parser.add_argument("--config", required=True)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--tokenizer", default=None, help="path to a trained tokenizer json")
    parser.add_argument("--checkpoint", default=None, help="SFT+KD checkpoint to start from")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="override sampling_temperature; must be > 0 or every group is degenerate",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = GRPOConfig.from_yaml(args.config)
    overrides: dict[str, Any] = {}
    if args.group_size is not None:
        overrides["group_size"] = args.group_size
    if args.temperature is not None:
        overrides["sampling_temperature"] = args.temperature
    if overrides:
        # `model_copy` does not re-validate, so the Field constraints above are not
        # enforced on an override; check the one that matters by hand.
        cfg = cfg.model_copy(update=overrides)
    print(json.dumps(cfg.model_dump(by_alias=True), default=str))
    print(f"group size {cfg.group_size}, no value network, reward = format AND correctness")
    if args.dry_run:
        return 0
    if not args.tokenizer or not args.checkpoint:
        parser.error("--tokenizer and --checkpoint are required unless --dry-run is passed")

    # Everything below used to be a SystemExit telling the reader to run the notebook --
    # which called this same CLI. `run_grpo` existed all along; nothing invoked it.
    import torch

    from localmind.model import LocalMindTransformer, ModelConfig
    from localmind.post.kd import GreedyStudentSampler
    from localmind.tokenizer.tokenizer import Tokenizer

    tokenizer = Tokenizer.load(args.tokenizer)
    # NB: cfg.model_config is pydantic v2's reserved ConfigDict, not this field.
    model_cfg = ModelConfig.from_yaml(cfg.model_config_path)

    def load_checkpoint(role: str) -> LocalMindTransformer:
        model = LocalMindTransformer(model_cfg)
        blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = blob.get("model", blob) if isinstance(blob, dict) else blob
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[grpo] loaded {args.checkpoint} as {role} "
            f"(missing={len(missing)} unexpected={len(unexpected)})"
        )
        return model

    policy = load_checkpoint("policy")
    reference: LocalMindTransformer | None = None
    if cfg.kl_coef > 0.0:
        # Only loaded when the KL tether is switched on. The default `kl_coef = 0` is the
        # cheap configuration SS9 5d describes: a verifiable reward cannot be gamed, so
        # there is nothing for a reference policy to protect against.
        reference = load_checkpoint("reference (frozen)")
        reference.eval()
        reference.requires_grad_(False)
    else:
        print("[grpo] kl_coef=0: no reference policy loaded (verifiable rewards cannot be gamed)")

    # `GreedyStudentSampler` is reused rather than reimplemented, but its name is only
    # half the story: at ``temperature > 0`` it samples from the softmax with a per-rollout
    # RNG seed (``sample_index``), which is the mode GRPO needs.
    #
    # GREEDY DECODING IS INVALID FOR GRPO. `sample_group` asks the sampler for ``G``
    # completions from ONE prompt; at temperature 0 it returns G identical strings, so all
    # G rewards are equal, `group_advantages` correctly returns exact zeros, `grpo_loss`
    # is identically 0, and the whole run "succeeds" having changed nothing. That failure
    # is silent by construction, so it is refused here rather than warned about.
    if cfg.sampling_temperature <= 0.0:
        parser.error(
            "sampling_temperature must be > 0: greedy decoding returns the same completion "
            "for every member of a group, so every group-relative advantage is exactly "
            "zero and GRPO trains on a loss that is identically 0"
        )
    sampler = GreedyStudentSampler(
        model=policy,
        eos_id=tokenizer.eos_id,
        temperature=cfg.sampling_temperature,
        seed=cfg.seed,
    )
    print(
        f"[grpo] job={cfg.job} n_prompts={cfg.n_prompts} group_size={cfg.group_size} "
        f"temperature={cfg.sampling_temperature} max_new_tokens={cfg.max_new_tokens}"
    )

    result = run_grpo(
        policy,
        tokenizer,
        sampler,
        cfg=cfg,
        reference=reference,
        group_size=cfg.group_size,
        n_prompts=cfg.n_prompts,
        max_new_tokens=cfg.max_new_tokens,
        peak_lr=cfg.peak_lr,
        clip_eps=cfg.clip_eps,
        kl_coef=cfg.kl_coef,
        seq_len=cfg.seq_len,
        seed=cfg.seed,
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "grpo_policy.pt"
    torch.save({"model": policy.state_dict(), "stage": "grpo", "group_size": cfg.group_size}, ckpt)
    (out_dir / "grpo_metrics.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8"
    )

    # StageResult exposes initial_loss/final_loss -- not first_loss/last_loss, which
    # silently give `nan -> nan` while training works fine.
    stage = result.stage_result
    print(
        f"[grpo] done: steps={stage.steps} "
        f"loss {stage.initial_loss:.4f} -> {stage.final_loss:.4f} | "
        f"mean reward {result.mean_reward:.4f} "
        f"(format-valid {result.format_valid_rate:.1%}, correct {result.correct_rate:.1%}) | "
        f"degenerate groups {result.degenerate_group_frac:.1%} -> {ckpt}"
    )
    for w in result.warnings:
        print(f"[grpo] WARNING: {w}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
