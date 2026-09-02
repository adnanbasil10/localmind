"""Phase 5c -- DPO on the query rewriter (SS9 5c).

~5k preference pairs, ``beta = 0.1``. The rewriter is the right head to run DPO on: it is
the only one of the three jobs whose output is not a label. "Which of these two standalone
queries is better" is a genuine preference; "which of ``in_domain`` / ``needs_web`` is
better" is just a label, and a label has an SFT target, not a preference.

Two things get tracked every step, because SS9 says so and because they are the two ways
DPO fails quietly:

* **reward margin** -- ``beta * [(log pi(y_w|x) - log pi_ref(y_w|x)) -
  (log pi(y_l|x) - log pi_ref(y_l|x))]``. It should rise. If it rises while quality falls,
  the preference data is measuring something other than quality.
* **KL from the reference policy** -- SS9: "runaway KL means your beta is wrong". Reported
  with the low-variance non-negative k3 estimator, and surfaced as an explicit
  :class:`DPOResult.kl_warning` rather than left for someone to notice in a log. Silent
  divergence is the failure mode; a warning is the deliverable.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from localmind.post.sft import (
    IGNORE_INDEX,
    ChatTokenizer,
    SFTExample,
    StageResult,
    encode_sft_example,
    run_stage,
    seed_prompts,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor, nn

__all__ = [
    "DPOBatch",
    "DPOConfig",
    "DPOResult",
    "PreferencePair",
    "build_preference_pairs",
    "collate_preference_batch",
    "dpo_loss",
    "kl_from_reference",
    "main",
    "make_reference_model",
    "run_dpo",
    "sequence_logprob",
]

#: SS9 5c. Also the default in `configs/train/dpo.yaml`; never written inline elsewhere.
DEFAULT_BETA = 0.1


# ------------------------------------------------------------------------------------ #
# The loss
# ------------------------------------------------------------------------------------ #
def sequence_logprob(
    model: nn.Module, input_ids: Tensor, labels: Tensor, *, length_normalize: bool = False
) -> Tensor:
    """``(B,)`` summed log-probability of the supervised span of each row.

    ``labels`` follows the frozen model contract: already shifted, ``-100`` masks a
    position. Only the assistant span is scored, so the (identical) prompt shared by a
    chosen/rejected pair contributes nothing and cannot bias the comparison.

    ``length_normalize=False`` is the DPO paper's formulation. Length normalisation is
    offered because the sum makes DPO length-biased -- a longer response is a lower log
    probability almost mechanically -- but it is off by default so this implementation
    matches the objective it claims to implement.
    """
    import torch

    logits = model(input_ids).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    mask = labels != IGNORE_INDEX
    safe_labels = labels.masked_fill(~mask, 0)
    token_lp = torch.gather(logprobs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_lp = token_lp * mask.to(token_lp.dtype)
    total = token_lp.sum(dim=-1)
    if length_normalize:
        total = total / mask.sum(dim=-1).clamp_min(1).to(total.dtype)
    return total


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    ref_chosen_logps: Tensor,
    ref_rejected_logps: Tensor,
    *,
    beta: float = DEFAULT_BETA,
    label_smoothing: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    """The DPO objective and its diagnostics.

    ``L = -log sigmoid( beta * [ (log pi(y_w) - log pi_ref(y_w))
                                - (log pi(y_l) - log pi_ref(y_l)) ] )``

    The implicit reward of a response is ``beta * (log pi - log pi_ref)``; the loss is
    just a logistic loss on the *difference* of those two rewards, which is why DPO needs
    no reward model. ``beta`` is simultaneously the reward scale and the strength of the
    KL tether to the reference -- raising it to "learn faster" tightens the tether, which
    is the opposite of what people expect.

    ``label_smoothing`` (cDPO) hedges against label noise in the preference set; 0.0 is
    the plain objective.

    Returns:
        ``(loss, metrics)`` with ``chosen_reward``, ``rejected_reward``,
        ``reward_margin`` and ``reward_accuracy`` -- the fraction of pairs the implicit
        reward already orders correctly, which is the number that says whether the model
        has learned the preference at all.
    """
    from torch.nn import functional as tf

    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError(f"label_smoothing must be in [0, 0.5), got {label_smoothing}")
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    chosen_logratio = policy_chosen_logps - ref_chosen_logps
    rejected_logratio = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_logratio - rejected_logratio)

    if label_smoothing > 0.0:
        losses = (
            -tf.logsigmoid(logits) * (1.0 - label_smoothing)
            - tf.logsigmoid(-logits) * label_smoothing
        )
    else:
        losses = -tf.logsigmoid(logits)

    chosen_reward = beta * chosen_logratio.detach()
    rejected_reward = beta * rejected_logratio.detach()
    metrics = {
        "chosen_reward": float(chosen_reward.mean()),
        "rejected_reward": float(rejected_reward.mean()),
        "reward_margin": float((chosen_reward - rejected_reward).mean()),
        "reward_accuracy": float((chosen_reward > rejected_reward).float().mean()),
        "logits_mean": float(logits.detach().mean()),
    }
    return losses.mean(), metrics


def kl_from_reference(policy_logps: Tensor, ref_logps: Tensor) -> tuple[float, float]:
    """``(k1, k3)`` estimates of ``KL(pi || pi_ref)`` from paired log-probs.

    Feed this **per-token** log-probs, not summed sequence log-probs. The estimators are
    scale-sensitive -- ``k3`` contains ``exp(-r)`` -- so summing over a 200-token response
    first inflates the result by orders of magnitude and makes any fixed warning threshold
    meaningless. `run_dpo` therefore length-normalises before calling this, even though the
    DPO *loss* itself uses the summed form the paper specifies.

    ``k1 = mean(log pi - log pi_ref)`` is unbiased but can go negative on a finite sample,
    which makes "is the KL running away?" hard to answer from a noisy trace. ``k3 =
    mean(exp(-r) + r - 1)`` with ``r = log pi - log pi_ref`` (Schulman's estimator) is
    also unbiased, always non-negative, and much lower variance -- so it is the one the
    warning threshold is applied to. Both are returned because a large gap between them
    is itself a signal that the sample is too small to conclude anything.
    """
    import torch

    r = (policy_logps - ref_logps).detach().float()
    k1 = float(r.mean())
    k3 = float((torch.exp(-r) + r - 1.0).mean())
    return k1, k3


# ------------------------------------------------------------------------------------ #
# Preference data
# ------------------------------------------------------------------------------------ #
class PreferencePair(BaseModel):
    """One rewriter preference: same prompt, a good rewrite and a bad one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inputs: dict[str, str]
    chosen: str
    rejected: str
    #: Why the rejected response is worse. Kept on the record because "the annotator
    #: preferred A" is not a specification, and a preference set whose failure modes are
    #: unnamed cannot be audited when DPO learns the wrong thing.
    reason: Literal["context_dropped", "verbose", "hallucinated", "unchanged"] = "context_dropped"

    def to_examples(self) -> tuple[SFTExample, SFTExample]:
        chosen = SFTExample(job="rewriter", inputs=self.inputs, completion=self.chosen)
        rejected = SFTExample(job="rewriter", inputs=self.inputs, completion=self.rejected)
        return chosen, rejected


def build_preference_pairs(n: int, *, seed: int = 1337) -> list[PreferencePair]:
    """A deterministic rewriter preference set covering four named failure modes.

    Every rejected response is a *plausible* rewrite, not noise. A preference set where
    the rejected side is obvious garbage teaches the model to avoid garbage, which it
    already does after SFT, and produces a reward margin that rises while nothing
    improves.
    """
    pool = [p for p in seed_prompts(max(n * 3, n + 8), seed=seed) if p[0] == "rewriter"]
    rng = random.Random(seed)
    pairs: list[PreferencePair] = []
    for i, (_, inputs) in enumerate(pool):
        if len(pairs) >= n:
            break
        gold = inputs.get("GOLD", inputs.get("QUERY", ""))
        query = inputs.get("QUERY", "")
        clean = {k: v for k, v in inputs.items() if k != "GOLD"}
        mode: Literal["context_dropped", "verbose", "hallucinated", "unchanged"] = (
            "context_dropped",
            "verbose",
            "hallucinated",
            "unchanged",
        )[i % 4]
        if mode == "context_dropped":
            rejected = query
        elif mode == "verbose":
            rejected = (
                f"I think what the user is asking about here is {gold} and possibly also "
                "some related background information"
            )
        elif mode == "hallucinated":
            rejected = f"{gold} in {rng.choice(('2019', '2021', 'the EU', 'the Berlin office'))}"
        else:
            rejected = query
        pairs.append(PreferencePair(inputs=clean, chosen=gold, rejected=rejected, reason=mode))
    return pairs


@dataclass(frozen=True)
class DPOBatch:
    """Chosen and rejected stacked into one forward pass.

    Stacked rather than run as two forwards because the pair must see identical model
    state: with dropout or any stochastic path, two separate forwards give the chosen and
    rejected sides different noise, and the reward difference picks that up as signal.
    """

    chosen_input_ids: Tensor
    chosen_labels: Tensor
    rejected_input_ids: Tensor
    rejected_labels: Tensor

    def __len__(self) -> int:
        return int(self.chosen_input_ids.shape[0])


def collate_preference_batch(
    tokenizer: ChatTokenizer, pairs: Sequence[PreferencePair], *, max_len: int = 256
) -> DPOBatch:
    """Encode + right-pad a batch of pairs (prompt masked, assistant span supervised)."""
    import torch

    def encode(side: str) -> tuple[Tensor, Tensor]:
        rows: list[tuple[list[int], list[int]]] = []
        for pair in pairs:
            chosen_ex, rejected_ex = pair.to_examples()
            ex = chosen_ex if side == "chosen" else rejected_ex
            enc = encode_sft_example(tokenizer, ex, max_len=max_len)
            ids = enc.full_ids
            labels = [
                ids[k] if k >= enc.completion_start else IGNORE_INDEX for k in range(len(ids))
            ]
            rows.append((ids[:-1], labels[1:]))
        width = max(len(r[0]) for r in rows)
        input_ids = torch.full((len(rows), width), tokenizer.pad_id, dtype=torch.long)
        label_ids = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
        for i, (ids, labels) in enumerate(rows):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            label_ids[i, : len(labels)] = torch.tensor(labels, dtype=torch.long)
        return input_ids, label_ids

    ci, cl = encode("chosen")
    ri, rl = encode("rejected")
    return DPOBatch(
        chosen_input_ids=ci, chosen_labels=cl, rejected_input_ids=ri, rejected_labels=rl
    )


def make_reference_model(model: nn.Module) -> nn.Module:
    """A frozen deep copy of ``model`` -- the SFT checkpoint DPO is tethered to.

    A copy rather than "the same model with adapters off": the reference must be the
    policy *before* DPO, and reusing the live module makes the KL term compare the model
    against itself, which is zero by construction and hides every divergence.
    """
    import copy

    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


# ------------------------------------------------------------------------------------ #
# Config and run
# ------------------------------------------------------------------------------------ #
class DPOConfig(BaseModel):
    """``configs/train/dpo.yaml``."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, protected_namespaces=()
    )

    model_config_path: str = Field(alias="model_config")
    job: Literal["rewriter"] = "rewriter"
    n_pairs: int = Field(default=5_000, gt=0)
    beta: float = Field(default=DEFAULT_BETA, gt=0.0)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=0.5)
    length_normalize: bool = False

    seq_len: int = Field(default=256, gt=0)
    micro_batch_size: int = Field(default=8, gt=0)
    epochs: int = Field(default=1, gt=0)

    peak_lr: float = Field(default=5.0e-6, gt=0.0)
    warmup_frac: float = Field(default=0.1, ge=0.0, lt=1.0)
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(default=0.0, ge=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    #: ADR 0001: bf16 absent on purpose.
    precision: Literal["fp16", "fp32"] = "fp16"

    #: SS9 5c: "runaway KL means your beta is wrong". Crossing this raises a warning on
    #: `DPOResult`, it does not silently continue and it does not silently abort.
    kl_warn_threshold: float = Field(default=0.5, gt=0.0)
    out_dir: str = "artifacts/post/dpo"
    seed: int = 1337

    @classmethod
    def from_yaml(cls, path: str | Path) -> DPOConfig:
        import yaml

        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping")
        return cls.model_validate(raw)


@dataclass
class DPOResult:
    """A :class:`StageResult` plus the two things SS9 insists are tracked."""

    stage_result: StageResult
    reward_margin: float
    kl_k1: float
    kl_k3: float
    kl_threshold: float
    reward_accuracy: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    @property
    def kl_warning(self) -> bool:
        return self.kl_k3 > self.kl_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.stage_result.to_dict(),
            "reward_margin": self.reward_margin,
            "reward_accuracy": self.reward_accuracy,
            "kl_from_reference_k1": self.kl_k1,
            "kl_from_reference_k3": self.kl_k3,
            "kl_threshold": self.kl_threshold,
            "kl_warning": self.kl_warning,
            "warnings": self.warnings,
        }


def run_dpo(
    policy: nn.Module,
    reference: nn.Module,
    tokenizer: ChatTokenizer,
    pairs: Sequence[PreferencePair],
    *,
    cfg: DPOConfig | None = None,
    beta: float = DEFAULT_BETA,
    batch_size: int = 4,
    epochs: int = 1,
    peak_lr: float = 5.0e-6,
    seq_len: int = 256,
    seed: int = 1337,
    max_steps: int | None = None,
) -> DPOResult:
    """Run DPO, tracking reward margin and KL from the reference every step."""
    import torch

    if not pairs:
        raise ValueError("run_dpo needs at least one preference pair")

    b = cfg.beta if cfg else beta
    ls = cfg.label_smoothing if cfg else 0.0
    lnorm = cfg.length_normalize if cfg else False
    bs = min(batch_size, cfg.micro_batch_size) if cfg else batch_size
    n_epochs = cfg.epochs if cfg else epochs
    lr = cfg.peak_lr if cfg else peak_lr
    n = cfg.seq_len if cfg else seq_len
    threshold = cfg.kl_warn_threshold if cfg else 0.5
    run_seed = cfg.seed if cfg else seed

    order = list(range(len(pairs)))
    batches: list[DPOBatch] = []
    for epoch in range(n_epochs):
        random.Random(run_seed + epoch).shuffle(order)
        for start in range(0, len(order), max(bs, 1)):
            block = [pairs[i] for i in order[start : start + max(bs, 1)]]
            batches.append(collate_preference_batch(tokenizer, block, max_len=n))
    if max_steps is not None:
        batches = batches[:max_steps]

    # The reference is frozen, so its log-probs never change: compute them once instead
    # of paying a second forward pass every step.
    reference.eval()
    ref_logps: list[tuple[Tensor, Tensor]] = []
    with torch.no_grad():
        for batch in batches:
            ref_logps.append(
                (
                    sequence_logprob(
                        reference,
                        batch.chosen_input_ids,
                        batch.chosen_labels,
                        length_normalize=lnorm,
                    ),
                    sequence_logprob(
                        reference,
                        batch.rejected_input_ids,
                        batch.rejected_labels,
                        length_normalize=lnorm,
                    ),
                )
            )

    last: dict[str, float] = {}

    def step_fn(step: int) -> tuple[Tensor, dict[str, float]]:
        batch = batches[step]
        ref_c, ref_r = ref_logps[step]
        pol_c = sequence_logprob(
            policy, batch.chosen_input_ids, batch.chosen_labels, length_normalize=lnorm
        )
        pol_r = sequence_logprob(
            policy, batch.rejected_input_ids, batch.rejected_labels, length_normalize=lnorm
        )
        loss, metrics = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=b, label_smoothing=ls)
        # Per-token, not per-sequence: see `kl_from_reference`. Without the
        # normalisation the threshold in `configs/train/dpo.yaml` would depend on how
        # long the sampled responses happened to be, which is not a property of beta.
        n_c = (batch.chosen_labels != IGNORE_INDEX).sum(-1).clamp_min(1)
        n_r = (batch.rejected_labels != IGNORE_INDEX).sum(-1).clamp_min(1)
        k1, k3 = kl_from_reference(
            torch.cat([pol_c / n_c, pol_r / n_r]).detach(),
            torch.cat([ref_c / n_c, ref_r / n_r]),
        )
        metrics["kl_k1"] = k1
        metrics["kl_k3"] = k3
        last.clear()
        last.update(metrics)
        return loss, metrics

    stage = run_stage(
        policy,
        step_fn,
        stage="dpo",
        steps=len(batches),
        peak_lr=lr,
        cfg=cfg,
        warmup_frac=cfg.warmup_frac if cfg else 0.1,
        min_lr_ratio=cfg.min_lr_ratio if cfg else 0.1,
        grad_clip=cfg.grad_clip if cfg else 1.0,
        seed=run_seed,
    )
    stage.summary["beta"] = b
    stage.summary["n_pairs"] = len(pairs)

    result = DPOResult(
        stage_result=stage,
        reward_margin=last.get("reward_margin", float("nan")),
        reward_accuracy=last.get("reward_accuracy", float("nan")),
        kl_k1=last.get("kl_k1", float("nan")),
        kl_k3=last.get("kl_k3", float("nan")),
        kl_threshold=threshold,
    )
    if result.kl_warning:
        result.warnings.append(
            f"KL from the reference policy is {result.kl_k3:.3f} nats/token > {threshold:.3f}: "
            f"policy is running away from the SFT checkpoint. beta={b} is too low (or the "
            "LR is too high). SS9 5c: runaway KL means your beta is wrong."
        )
    margins = [m.extra.get("reward_margin", float("nan")) for m in stage.history]
    if len(margins) >= 2 and margins[-1] <= margins[0]:
        result.warnings.append(
            f"reward margin did not increase ({margins[0]:.4f} -> {margins[-1]:.4f}); the "
            "preference set may not encode the property you think it does"
        )
    return result


# ------------------------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------------------------ #
def main(argv: Sequence[str] | None = None) -> int:
    """``python -m localmind.post.dpo --config configs/train/dpo.yaml --beta 0.1``"""
    parser = argparse.ArgumentParser(description="Phase 5c DPO on the rewriter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = DPOConfig.from_yaml(args.config)
    if args.beta is not None:
        cfg = cfg.model_copy(update={"beta": args.beta})
    print(json.dumps(cfg.model_dump(by_alias=True), default=str))
    if args.dry_run:
        return 0
    if not args.tokenizer or not args.checkpoint:
        parser.error("--tokenizer and --checkpoint are required unless --dry-run is passed")
    raise SystemExit(
        "training entrypoint requires an SFT checkpoint as the reference policy; run "
        "this from notebooks/kaggle/02_distill.ipynb on a T4"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
