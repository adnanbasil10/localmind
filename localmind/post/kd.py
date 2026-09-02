"""Phase 5b -- distillation, three arms benchmarked against each other (SS9 5b, ADR 0005).

**Why the arms are shaped this way** (SS3.3, "the cross-tokenizer distillation problem"):
true logit-level KD needs teacher and student to share a vocabulary. Qwen's is ~151k,
ours is 16,384, and the token boundaries do not align -- there is simply no
correspondence between the two logit vectors, so a KL between them is undefined. It is
not a hard problem, it is an *ill-posed* one. Hence:

1. :func:`run_sequence_kd` -- **sequence-level KD (primary)**. Train on the teacher's
   sampled outputs as hard labels. Tokenizer-agnostic by construction: the teacher's
   text is re-tokenised with *our* tokenizer, so the 151k/16k mismatch never arises.
   This is what most real-world distillation actually is.
2. :func:`run_on_policy_correction` -- **on-policy correction**. Sample from the
   *student*, have the teacher score or rewrite, train on the correction. Fixes exposure
   bias: arm 1 only ever shows the model the teacher's own trajectories, so at inference
   the model is one sampling accident away from a distribution it has never been trained
   on. Plausibly the real reason Gemma 3 1B and Llama 3.2 1B work as well as they do.
3. :func:`topk_kl_loss` / :func:`run_logit_kd` -- **same-tokenizer logit KD (optional
   arm)**, LocalMind-100M -> LocalMind-31M, where the vocabularies *do* match because we
   trained both. ``L = alpha*KL(student || teacher_topK) + (1-alpha)*CE``, K=64,
   alpha=0.7.

ADR 0005 is accepted; this module implements it rather than re-arguing it.

The subtle part of arm 3 is renormalisation. The teacher's top-K slice is not a
distribution -- it sums to less than one. Taking ``softmax`` over the K teacher logits
renormalises it onto the K-element support, and the student must be restricted to *the
same* support and renormalised the same way, or the two sides of the KL live on
different measures and the "divergence" is not one. :func:`topk_kl_loss` keeps the wrong
version reachable as ``renormalize_student=False`` precisely so the test can show the
gap instead of describing it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from localmind.post.sft import (
    IGNORE_INDEX,
    ChatTokenizer,
    DeterministicFakeTeacher,
    Job,
    SFTExample,
    StageResult,
    Teacher,
    build_teacher_prompt,
    encode_sft_example,
    pack_sft,
    parse_teacher_output,
    prompt_messages,
    run_stage,
    safe_decode,
    seed_prompts,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor, nn

__all__ = [
    "GreedyStudentSampler",
    "KDArm",
    "KDConfig",
    "OnPolicyReport",
    "StudentSampler",
    "TeacherLogitStore",
    "compare_arms",
    "estimate_topk_bytes",
    "kd_loss",
    "main",
    "run_kd",
    "run_logit_kd",
    "run_on_policy_correction",
    "run_sequence_kd",
    "topk_kl_loss",
]

KDArm = Literal["sequence", "on-policy", "logit"]
KD_ARMS: tuple[KDArm, ...] = ("sequence", "on-policy", "logit")

#: SS9 5b / ADR 0005. Restated in `configs/train/kd.yaml`; these are the fallbacks used
#: when a caller passes no config at all (tests, notebooks).
DEFAULT_TOP_K = 64
DEFAULT_ALPHA = 0.7


# ------------------------------------------------------------------------------------ #
# Arm 3: top-K logit KL
# ------------------------------------------------------------------------------------ #
def topk_kl_loss(
    student_logits: Tensor,
    teacher_topk_values: Tensor,
    teacher_topk_indices: Tensor,
    *,
    temperature: float = 1.0,
    direction: Literal["forward", "reverse"] = "forward",
    renormalize_student: bool = True,
    scale_by_temperature_squared: bool = True,
    mask: Tensor | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """KL between teacher and student, both restricted to the teacher's top-K support.

    Args:
        student_logits: ``(N, V)`` full student logits over *our* 16k vocabulary.
        teacher_topk_values: ``(N, K)`` the teacher's raw logits at its top-K tokens.
        teacher_topk_indices: ``(N, K)`` those tokens' ids, in the shared vocabulary.
        temperature: softens both sides. Applied to logits before any normalisation.
        direction: ``"forward"`` is ``KL(teacher || student)`` -- the standard KD
            objective, equivalent to a cross-entropy against the teacher's soft targets,
            and mode-covering. ``"reverse"`` is ``KL(student || teacher)``, which is
            mode-seeking. SS9 writes the term as ``KL(student || teacher_topK)``; that
            notation is ambiguous about argument order, so both are implemented and the
            default is the one KD universally means. The config chooses; nothing here
            silently picks for you.
        renormalize_student: **the subtle part.** ``True`` restricts the student to the
            teacher's K tokens and renormalises it there, so both arguments of the KL
            are proper distributions on the same K-element support. ``False`` reproduces
            the common bug -- student probabilities taken from the *full* vocabulary
            softmax, which do not sum to one over K -- and is kept reachable only so the
            test can quantify how wrong it is.
        scale_by_temperature_squared: Hinton's ``T^2``, so the KD gradient magnitude does
            not collapse as ``T`` rises. A no-op at ``T = 1``.
        mask: optional ``(N,)`` boolean/0-1 mask; masked-out rows contribute nothing.
        reduction: ``"mean"`` averages over unmasked rows.

    Returns:
        Scalar for ``"mean"``/``"sum"``, else ``(N,)``.

    Note that ``softmax`` over the K teacher logits *is* the correctly renormalised
    restriction of the full teacher softmax: the full normaliser cancels between
    numerator and denominator. That identity is why this function never needs the
    teacher's full logit vector -- which is the whole reason the top-64 dump is 1.2 GB
    instead of 3 TB.
    """
    import torch

    if student_logits.dim() != 2:
        raise ValueError(f"student_logits must be (N, V), got {tuple(student_logits.shape)}")
    if teacher_topk_values.shape != teacher_topk_indices.shape:
        raise ValueError(
            f"teacher values {tuple(teacher_topk_values.shape)} and indices "
            f"{tuple(teacher_topk_indices.shape)} must have the same shape"
        )
    if teacher_topk_values.shape[0] != student_logits.shape[0]:
        raise ValueError("teacher and student must agree on N")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    t_logits = teacher_topk_values.float() / temperature
    log_p = torch.log_softmax(t_logits, dim=-1)

    s_full = student_logits.float() / temperature
    s_topk = torch.gather(s_full, 1, teacher_topk_indices.long())
    if renormalize_student:
        log_q = torch.log_softmax(s_topk, dim=-1)
    else:
        # The bug, made explicit: normalise over all V, then read off K entries. The
        # resulting "distribution" has mass sum < 1, so the sum below is not a KL.
        log_q = s_topk - torch.logsumexp(s_full, dim=-1, keepdim=True)

    if direction == "forward":
        per_row = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    else:
        log_q_norm = log_q if renormalize_student else torch.log_softmax(s_topk, dim=-1)
        per_row = (log_q_norm.exp() * (log_q_norm - log_p)).sum(dim=-1)

    if scale_by_temperature_squared:
        per_row = per_row * (temperature**2)

    if mask is not None:
        per_row = per_row * mask.to(per_row.dtype)

    if reduction == "none":
        return per_row
    if reduction == "sum":
        return per_row.sum()
    denom = (
        mask.to(per_row.dtype).sum() if mask is not None else per_row.new_tensor(per_row.numel())
    )
    return per_row.sum() / denom.clamp_min(1.0)


def kd_loss(
    student_logits: Tensor,
    targets: Tensor,
    teacher_topk_values: Tensor,
    teacher_topk_indices: Tensor,
    *,
    alpha: float = DEFAULT_ALPHA,
    temperature: float = 1.0,
    direction: Literal["forward", "reverse"] = "forward",
    renormalize_student: bool = True,
) -> tuple[Tensor, dict[str, float]]:
    """``L = alpha*KL(teacher_topK || student_topK) + (1-alpha)*CE(student, hard labels)``.

    SS9 5b pins ``alpha = 0.7``: the soft targets carry most of the signal (that is the
    point of distillation) while the CE term keeps the student anchored to the actual
    next token, which matters because the top-K slice is a lossy view of the teacher.

    ``targets`` uses ``-100`` to mask a position, exactly as
    `LocalMindTransformer.forward` does; masked positions are excluded from *both* terms,
    so a padded row cannot dilute the KD signal.
    """
    import torch
    from torch.nn import functional as tf

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    flat_logits = student_logits.reshape(-1, student_logits.shape[-1])
    flat_targets = targets.reshape(-1)
    keep = flat_targets != IGNORE_INDEX

    ce = tf.cross_entropy(flat_logits, flat_targets, ignore_index=IGNORE_INDEX)
    kl = topk_kl_loss(
        flat_logits,
        teacher_topk_values.reshape(-1, teacher_topk_values.shape[-1]),
        teacher_topk_indices.reshape(-1, teacher_topk_indices.shape[-1]),
        temperature=temperature,
        direction=direction,
        renormalize_student=renormalize_student,
        mask=keep,
        reduction="mean",
    )
    loss = alpha * kl + (1.0 - alpha) * ce
    parts = {"kd_kl": float(kl.detach()), "kd_ce": float(ce.detach())}
    if torch.isnan(loss):  # pragma: no cover - defensive
        raise RuntimeError("kd_loss produced NaN; check temperature and teacher logits")
    return loss, parts


def estimate_topk_bytes(
    n_sequences: int,
    seq_len: int,
    k: int = DEFAULT_TOP_K,
    *,
    value_bytes: int = 2,
    index_bytes: int = 2,
) -> int:
    """Bytes needed to store a top-K teacher-logit dump.

    ADR 0005's storage note ("top-64 logits over 20k sequences x 256 tokens is about
    1.2 GB -- dump to HF Hub, do not try to hold it in a Kaggle session") comes out of
    this at the defaults: fp16 values and **uint16 indices**, the latter being legal only
    because our vocabulary is 16,384 and therefore fits in 16 bits. Storing indices as
    int32, as most reference implementations do, doubles the index half of the dump and
    turns 1.2 GB into 2.0 GB -- enough to lose a free Kaggle session to disk quota.
    """
    return n_sequences * seq_len * k * (value_bytes + index_bytes)


@dataclass
class TeacherLogitStore:
    """Top-K teacher logits on disk, in the layout :func:`estimate_topk_bytes` costs.

    ``.npz`` rather than a torch checkpoint so the dump can be read on a machine that
    never installs torch, and uint16 indices because a 16k vocabulary fits in 16 bits.
    """

    path: Path
    k: int = DEFAULT_TOP_K

    @staticmethod
    def from_teacher_logits(logits: Tensor, k: int = DEFAULT_TOP_K) -> tuple[Tensor, Tensor]:
        """``(values, indices)`` of the top-K entries of ``(..., V)`` teacher logits."""
        import torch

        if k <= 0 or k > logits.shape[-1]:
            raise ValueError(f"k must be in [1, V={logits.shape[-1]}], got {k}")
        values, indices = torch.topk(logits, k, dim=-1)
        return values, indices

    def dump(self, values: Tensor, indices: Tensor) -> Path:
        import numpy as np

        if indices.max().item() >= 2**16:
            raise ValueError(
                "uint16 index storage requires vocab_size <= 65536; this dump would "
                "overflow -- widen index_bytes and re-cost the dump before proceeding"
            )
        import torch

        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.path,
            values=values.detach().to(dtype=torch.float16).cpu().numpy(),
            indices=indices.detach().cpu().numpy().astype(np.uint16),
            k=np.asarray(self.k, dtype=np.int32),
        )
        return self.path

    def load(self) -> tuple[Tensor, Tensor]:
        import numpy as np
        import torch

        with np.load(self.path) as data:
            values = torch.from_numpy(np.asarray(data["values"], dtype=np.float32))
            indices = torch.from_numpy(np.asarray(data["indices"], dtype=np.int64))
        return values, indices


# ------------------------------------------------------------------------------------ #
# Arm 2: on-policy correction
# ------------------------------------------------------------------------------------ #
@runtime_checkable
class StudentSampler(Protocol):
    """Sample a continuation from the *student*.

    Deliberately the same shape as the frozen ``GenerationEngine`` Protocol in
    CONVENTIONS.md, so `localmind.inference.engine` satisfies it without an adapter, and
    so this module never has to import the inference package.
    """

    def generate(self, prompt_ids: list[int], max_new_tokens: int, **kw: Any) -> list[int]: ...


@dataclass
class GreedyStudentSampler:
    """Greedy (or temperature-sampled) decoding straight off the model.

    Exists so arm 2 is runnable offline without the inference engine. It is deliberately
    the slow, obviously-correct implementation: no KV cache, one forward per token.
    Production sampling belongs to Phase 6.
    """

    model: nn.Module
    eos_id: int
    temperature: float = 0.0
    seed: int = 1337

    def generate(self, prompt_ids: list[int], max_new_tokens: int, **kw: Any) -> list[int]:
        """``sample_index`` offsets the RNG seed.

        GRPO asks for ``G`` completions from one prompt and needs them to *differ*; a
        sampler that reseeds identically every call returns eight copies of the same
        string, every group is degenerate, and the run silently learns nothing.
        """
        import torch

        sample_index = int(kw.pop("sample_index", 0))
        gen = torch.Generator().manual_seed(self.seed + 7919 * sample_index)
        ids = list(prompt_ids)
        max_len = int(getattr(getattr(self.model, "cfg", None), "max_seq_len", 10**9))
        out: list[int] = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                window = ids[-max_len:]
                logits = self.model(torch.tensor([window], dtype=torch.long)).logits[0, -1]
                if self.temperature <= 0:
                    nxt = int(torch.argmax(logits))
                else:
                    probs = torch.softmax(logits.float() / self.temperature, dim=-1)
                    nxt = int(torch.multinomial(probs, 1, generator=gen))
                if nxt == self.eos_id:
                    break
                ids.append(nxt)
                out.append(nxt)
        return out


@dataclass
class OnPolicyReport:
    """What one round of arm 2 found out about the student.

    ``student_error_rate`` is the number worth watching across rounds: it is the fraction
    of the student's *own* samples the teacher had to correct, and it is the exposure-bias
    measurement arm 1 structurally cannot produce.
    """

    round_index: int
    n_sampled: int
    n_corrected: int
    n_teacher_rejected: int = 0
    examples: list[SFTExample] = field(default_factory=list)

    @property
    def student_error_rate(self) -> float:
        return self.n_corrected / self.n_sampled if self.n_sampled else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "n_sampled": self.n_sampled,
            "n_corrected": self.n_corrected,
            "n_teacher_rejected": self.n_teacher_rejected,
            "student_error_rate": self.student_error_rate,
        }


def collect_on_policy_corrections(
    sampler: StudentSampler,
    teacher: Teacher,
    tokenizer: ChatTokenizer,
    pool: Sequence[tuple[Job, dict[str, str]]],
    *,
    round_index: int = 0,
    max_new_tokens: int = 48,
    fake_teacher: bool | None = None,
) -> OnPolicyReport:
    """Sample the student, ask the teacher to correct it, keep only the corrections.

    Only rows the student got **wrong** are kept. Keeping the agreements too would just
    re-teach what the student already does, diluting the batch with zero-gradient
    examples; the whole value of an on-policy round is the set of states the student
    actually reaches and mishandles.
    """
    if fake_teacher is None:
        fake_teacher = isinstance(teacher, DeterministicFakeTeacher)

    student_texts: list[str] = []
    for job, inputs in pool:
        msgs = prompt_messages(job, {k: v for k, v in inputs.items() if k != "GOLD"})
        prompt_ids = tokenizer.encode_chat(msgs, add_generation_prompt=True)
        new_ids = sampler.generate(prompt_ids, max_new_tokens)
        student_texts.append(safe_decode(tokenizer, new_ids).strip())

    prompts = []
    for (job, inputs), attempt in zip(pool, student_texts, strict=True):
        base = build_teacher_prompt(job, inputs, include_gold=fake_teacher)
        prompts.append(
            f"{base}\nSTUDENT: {attempt}\n"
            "The STUDENT answer above may be wrong. Reply with the corrected answer only."
        )
    replies = teacher.generate(prompts, max_tokens=64, temperature=0.0, seed=round_index)

    examples: list[SFTExample] = []
    corrected = 0
    rejected = 0
    for (job, inputs), attempt, reply in zip(pool, student_texts, replies, strict=True):
        gold = parse_teacher_output(job, reply)
        if gold is None:
            rejected += 1
            continue
        if parse_teacher_output(job, attempt) == gold:
            continue  # student already agrees with the teacher: nothing to learn
        corrected += 1
        examples.append(
            SFTExample(
                job=job,
                inputs={k: v for k, v in inputs.items() if k != "GOLD"},
                completion=gold,
                gold=inputs.get("GOLD"),
                teacher=f"{teacher.name}/on-policy-r{round_index}",
            )
        )
    return OnPolicyReport(
        round_index=round_index,
        n_sampled=len(pool),
        n_corrected=corrected,
        n_teacher_rejected=rejected,
        examples=examples,
    )


# ------------------------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------------------------ #
class KDConfig(BaseModel):
    """``configs/train/kd.yaml``. Keys match what the Kaggle notebook's CLI passes."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, protected_namespaces=()
    )

    model_config_path: str = Field(alias="model_config")
    #: Arm 3's teacher is *our own* 100M sibling -- the only teacher whose vocabulary
    #: matches the student's, which is the entire reason arm 3 is possible at all.
    teacher_model_config: str = "configs/model/100m_teacher.yaml"
    teacher: str = "Qwen/Qwen2.5-3B-Instruct"
    arm: KDArm = "sequence"

    top_k: int = Field(default=DEFAULT_TOP_K, gt=0)
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)
    temperature: float = Field(default=1.0, gt=0.0)
    kl_direction: Literal["forward", "reverse"] = "forward"

    n_examples: int = Field(default=20_000, gt=0)
    seq_len: int = Field(default=256, gt=0)
    micro_batch_size: int = Field(default=8, gt=0)
    epochs: int = Field(default=1, gt=0)

    peak_lr: float = Field(default=3.0e-4, gt=0.0)
    warmup_frac: float = Field(default=0.03, ge=0.0, lt=1.0)
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(default=0.1, ge=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    #: ADR 0001: no ``bf16`` member, so a bf16 config fails validation on load.
    precision: Literal["fp16", "fp32"] = "fp16"

    on_policy_rounds: int = Field(default=3, gt=0)
    on_policy_samples_per_round: int = Field(default=512, gt=0)
    student_temperature: float = Field(default=0.8, ge=0.0)
    logit_dump_dir: str = "artifacts/post/kd/teacher_topk"
    out_dir: str = "artifacts/post/kd"
    seed: int = 1337

    @classmethod
    def from_yaml(cls, path: str | Path) -> KDConfig:
        import yaml

        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping")
        return cls.model_validate(raw)


# ------------------------------------------------------------------------------------ #
# The three arms
# ------------------------------------------------------------------------------------ #
def _rows_from_examples(
    tokenizer: ChatTokenizer,
    examples: Sequence[SFTExample],
    seq_len: int,
    *,
    mask_prompt_tokens: bool = True,
) -> list[Any]:
    encoded = [encode_sft_example(tokenizer, ex, max_len=seq_len) for ex in examples]
    return pack_sft(
        encoded,
        seq_len,
        pad_id=tokenizer.pad_id,
        mask_prompt_tokens=mask_prompt_tokens,
        boundary_masking=True,
    )


def run_sequence_kd(
    model: nn.Module,
    tokenizer: ChatTokenizer,
    examples: Sequence[SFTExample],
    *,
    cfg: KDConfig | None = None,
    seq_len: int = 256,
    batch_size: int = 4,
    peak_lr: float = 3.0e-4,
    epochs: int = 1,
    seed: int = 1337,
    max_steps: int | None = None,
) -> StageResult:
    """Arm 1: train on the teacher's outputs as hard labels.

    Mechanically this *is* SFT -- which is the point. Sequence-level KD is not a special
    loss, it is a special corpus, and pretending otherwise is how people end up believing
    they need aligned vocabularies to distill.
    """
    from localmind.post.sft import run_sft

    n = cfg.seq_len if cfg else seq_len
    rows = _rows_from_examples(tokenizer, examples, n)
    result = run_sft(
        model,
        rows,
        epochs=cfg.epochs if cfg else epochs,
        batch_size=min(batch_size, cfg.micro_batch_size) if cfg else batch_size,
        peak_lr=cfg.peak_lr if cfg else peak_lr,
        seed=cfg.seed if cfg else seed,
        max_steps=max_steps,
    )
    result.stage = "kd/sequence"
    result.summary["arm"] = "sequence"
    result.summary["n_examples"] = len(examples)
    result.summary["tokenizer_agnostic"] = True
    return result


def run_on_policy_correction(
    model: nn.Module,
    tokenizer: ChatTokenizer,
    sampler: StudentSampler,
    teacher: Teacher,
    *,
    cfg: KDConfig | None = None,
    rounds: int = 2,
    samples_per_round: int = 16,
    seq_len: int = 256,
    batch_size: int = 4,
    peak_lr: float = 3.0e-4,
    seed: int = 1337,
    max_new_tokens: int = 24,
    max_steps_per_round: int | None = None,
) -> StageResult:
    """Arm 2: sample -> teacher corrects -> train on the correction, repeated.

    The loop must be repeated rather than run once: after a round of training the
    student's distribution has moved, so the states it reaches -- the whole object of
    study -- have moved too. A single round is just SFT on a differently-sampled corpus.
    """
    from localmind.post.sft import run_sft

    n_rounds = cfg.on_policy_rounds if cfg else rounds
    per_round = cfg.on_policy_samples_per_round if cfg else samples_per_round
    n = cfg.seq_len if cfg else seq_len
    lr = cfg.peak_lr if cfg else peak_lr
    bs = min(batch_size, cfg.micro_batch_size) if cfg else batch_size
    base_seed = cfg.seed if cfg else seed

    merged = StageResult(stage="kd/on-policy", steps=0)
    reports: list[dict[str, Any]] = []
    offset = 0
    for r in range(n_rounds):
        pool = seed_prompts(per_round, seed=base_seed + 1000 * r)
        report = collect_on_policy_corrections(
            sampler, teacher, tokenizer, pool, round_index=r, max_new_tokens=max_new_tokens
        )
        reports.append(report.to_dict())
        if not report.examples:
            continue
        rows = _rows_from_examples(tokenizer, report.examples, n)
        if not rows:
            continue
        sub = run_sft(
            model,
            rows,
            epochs=1,
            batch_size=bs,
            peak_lr=lr,
            seed=base_seed + r,
            max_steps=max_steps_per_round,
        )
        for m in sub.history:
            merged.history.append(
                type(m)(step=m.step + offset, loss=m.loss, lr=m.lr, extra={**m.extra, "round": r})
            )
        offset += sub.steps
        merged.steps = offset

    merged.summary["arm"] = "on-policy"
    merged.summary["rounds"] = reports
    merged.summary["fixes"] = "exposure bias"
    if reports:
        merged.summary["student_error_rate_first"] = reports[0]["student_error_rate"]
        merged.summary["student_error_rate_last"] = reports[-1]["student_error_rate"]
    return merged


def run_logit_kd(
    student: nn.Module,
    teacher_model: nn.Module,
    batches: Sequence[tuple[Tensor, Tensor]],
    *,
    cfg: KDConfig | None = None,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
    temperature: float = 1.0,
    direction: Literal["forward", "reverse"] = "forward",
    peak_lr: float = 3.0e-4,
    seed: int = 1337,
    store: TeacherLogitStore | None = None,
) -> StageResult:
    """Arm 3: true top-K logit KL from LocalMind-100M into LocalMind-31M.

    The teacher is frozen and run under ``no_grad``; in a real run its top-K slices are
    precomputed once and streamed from the dump ADR 0005 sizes (see
    :class:`TeacherLogitStore`), because re-running a 100M forward every epoch costs more
    GPU-hours than the arm is worth.
    """
    import torch

    k = cfg.top_k if cfg else top_k
    a = cfg.alpha if cfg else alpha
    temp = cfg.temperature if cfg else temperature
    dirn = cfg.kl_direction if cfg else direction
    lr = cfg.peak_lr if cfg else peak_lr

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    cached: list[tuple[Tensor, Tensor]] = []
    with torch.no_grad():
        for input_ids, _ in batches:
            t_logits = teacher_model(input_ids).logits
            values, indices = TeacherLogitStore.from_teacher_logits(t_logits, k)
            cached.append((values, indices))
    if store is not None and cached:
        store.dump(cached[0][0], cached[0][1])

    def step_fn(step: int) -> tuple[Tensor, dict[str, float]]:
        input_ids, labels = batches[step]
        values, indices = cached[step]
        logits = student(input_ids).logits
        loss, parts = kd_loss(
            logits,
            labels,
            values,
            indices,
            alpha=a,
            temperature=temp,
            direction=dirn,
            renormalize_student=True,
        )
        return loss, parts

    result = run_stage(
        student,
        step_fn,
        stage="kd/logit",
        steps=len(batches),
        peak_lr=lr,
        cfg=cfg,
        warmup_frac=cfg.warmup_frac if cfg else 0.03,
        min_lr_ratio=cfg.min_lr_ratio if cfg else 0.1,
        grad_clip=cfg.grad_clip if cfg else 1.0,
        seed=cfg.seed if cfg else seed,
    )
    result.summary.update(
        {
            "arm": "logit",
            "top_k": k,
            "alpha": a,
            "temperature": temp,
            "kl_direction": dirn,
            "same_tokenizer_required": True,
            "estimated_dump_bytes": estimate_topk_bytes(
                len(batches), int(batches[0][0].shape[1]), k
            ),
        }
    )
    return result


def run_kd(arm: KDArm, **kwargs: Any) -> StageResult:
    """Dispatch on the arm name the notebook passes with ``--arm``."""
    if arm == "sequence":
        return run_sequence_kd(**kwargs)
    if arm == "on-policy":
        return run_on_policy_correction(**kwargs)
    if arm == "logit":
        return run_logit_kd(**kwargs)
    raise ValueError(f"unknown arm {arm!r}; expected one of {KD_ARMS}")


def compare_arms(
    results: Mapping[str, StageResult],
    *,
    hardware: str = "unknown",
    seeds: Sequence[int] = (1337,),
    scorer: Callable[[str], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """ADR 0005's deliverable: the arms side by side, including a null result.

    Training loss is reported because it is what an offline CPU run can actually
    measure. Downstream quality per arm needs a trained checkpoint and the Phase 10
    harness, so those cells are emitted as ``null`` with ``"status": "not-run"`` rather
    than filled with anything. A losing arm is a deliverable (CONVENTIONS.md rule 2); an
    invented number is not a result at all.
    """
    rows: list[dict[str, Any]] = []
    for name, res in results.items():
        row: dict[str, Any] = {
            "name": name,
            "steps": res.steps,
            "initial_loss": res.initial_loss,
            "final_loss": res.final_loss,
            "improved": res.improved,
            "tokenizer_agnostic": name != "logit",
            **{k: v for k, v in res.summary.items() if isinstance(v, int | float | str | bool)},
        }
        quality = scorer(name) if scorer is not None else None
        row["quality"] = quality if quality is not None else None
        row["quality_status"] = "measured" if quality is not None else "not-run"
        rows.append(row)
    return {
        "name": "kd_arms",
        "hardware": hardware,
        "seeds": [int(s) for s in seeds],
        "rows": rows,
        "ci": "bootstrap95",
        "note": (
            "Arms 1-2 are tokenizer-agnostic (Qwen 151k -> LocalMind 16k). Arm 3 requires "
            "LocalMind-100M, which shares our tokenizer. Loss values are comparable within "
            "an arm across seeds, not across arms: arm 3's objective includes a KL term."
        ),
    }


# ------------------------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------------------------ #
def main(argv: Sequence[str] | None = None) -> int:
    """``python -m localmind.post.kd --config configs/train/kd.yaml --arm sequence``"""
    parser = argparse.ArgumentParser(description="Phase 5b distillation arms")
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=list(KD_ARMS), default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--data", default=None, help="JSONL from localmind.post.sft")
    parser.add_argument("--tokenizer", default=None, help="path to a trained tokenizer json")
    parser.add_argument("--dry-run", action="store_true", help="validate config and exit")
    args = parser.parse_args(argv)

    cfg = KDConfig.from_yaml(args.config)
    overrides: dict[str, Any] = {}
    if args.arm:
        overrides["arm"] = args.arm
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    if args.alpha is not None:
        overrides["alpha"] = args.alpha
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    print(json.dumps({"config": cfg.model_dump(by_alias=True), "arm": cfg.arm}, default=str))
    if cfg.arm == "logit":
        est = estimate_topk_bytes(cfg.n_examples, cfg.seq_len, cfg.top_k)
        print(f"arm 3 teacher-logit dump estimate: {est / 1e9:.2f} GB -> {cfg.logit_dump_dir}")
        print("ADR 0005: push this to HF Hub, do not hold it in the Kaggle session")
    if args.dry_run:
        return 0
    if not args.data or not args.tokenizer:
        parser.error("--data and --tokenizer are required unless --dry-run is passed")
    raise SystemExit(
        "training entrypoint requires a trained tokenizer and a checkpoint; run this "
        "from notebooks/kaggle/02_distill.ipynb on a T4"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
