"""Phase 5a -- supervised fine-tuning on the three production jobs (implementation.md SS9).

The 31M model is not a chatbot. It is the CPU-resident control plane of the RAG system,
and it does exactly three things (`ControlPlane` in CONVENTIONS.md):

===========  ==========================  =========================================
job          input                       output
===========  ==========================  =========================================
``router``   query                       ``in_domain`` / ``out_of_domain`` / ``needs_web``
``rewriter`` query + conversation        a standalone search query
``grader``   query + chunk               ``relevant`` / ``irrelevant`` + score
===========  ==========================  =========================================

SFT data comes from **Qwen2.5-3B-Instruct on a Kaggle T4 via vLLM**. That path cannot run
on this laptop, so it sits behind the injectable :class:`Teacher` Protocol with
:class:`DeterministicFakeTeacher` as the offline stand-in; every test here runs on CPU with
no network and no teacher weights.

What the spec pins down, and where it lives here:

* *apply the chat template (via the tokenizer, not string concatenation)* --
  :func:`encode_sft_example` calls ``tokenizer.encode_chat``, which splices role ids in
  structurally. Nothing in this module builds a prompt with an f-string.
* *mask loss on prompt tokens (train only on assistant spans)* -- :func:`encode_sft_example`
  emits ``-100`` at every target position whose token belongs to the prompt. This is the
  step most implementations get wrong, so it is asserted directly in ``tests/test_post.py``.
* *pack with boundary masking* -- :func:`pack_sft`. See its docstring for the one honest
  compromise: the frozen ``LocalMindTransformer.forward`` signature takes no attention
  mask, so boundary masking is enforced in the **loss** (and the per-position ``doc_ids``
  are carried alongside for the day the model grows a mask argument).
* *cosine decay from 10% of pretrain LR* -- :func:`cosine_lr`, with the 0.1 factor living
  in ``configs/train/sft.yaml`` as ``peak_lr_frac_of_pretrain``, never inline.
* *hand-verify a stratified sample and report agreement with the teacher* --
  :func:`stratified_sample`, :func:`write_verification_sheet`, :func:`agreement_report`.
  Synthetic data that has not been spot-checked is not shippable, so
  :class:`SFTConfig.verify_min_agreement` turns a bad spot-check into a loud warning.

``sft.py`` also hosts the small pieces of training plumbing (:func:`cosine_lr`,
:func:`run_stage`, :class:`StageResult`) that ``kd.py`` / ``dpo.py`` / ``grpo.py`` reuse --
KD arm 1 *is* SFT on teacher outputs, so duplicating the loop would be a lie about the
relationship between the stages.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps torch out of import time
    from torch import Tensor, nn

__all__ = [
    "JOBS",
    "AgreementReport",
    "ChatTokenizer",
    "DeterministicFakeTeacher",
    "EncodedExample",
    "Job",
    "PackedRow",
    "SFTConfig",
    "SFTExample",
    "StageResult",
    "StepMetrics",
    "Teacher",
    "VLLMTeacher",
    "agreement_report",
    "boundary_attention_mask",
    "build_teacher_prompt",
    "cosine_lr",
    "encode_sft_example",
    "generate_sft_dataset",
    "iter_batches",
    "label_of",
    "load_verification_sheet",
    "main",
    "pack_sft",
    "parse_teacher_output",
    "prompt_messages",
    "read_examples",
    "run_sft",
    "run_stage",
    "safe_decode",
    "seed_prompts",
    "stratified_sample",
    "write_examples",
    "write_verification_sheet",
]

# The loss-mask sentinel. `LocalMindTransformer.forward` masks a target position out of
# the cross-entropy iff its label is exactly this value.
IGNORE_INDEX = -100

Job = Literal["router", "rewriter", "grader"]
JOBS: tuple[Job, ...] = ("router", "rewriter", "grader")

#: The teacher named in SS3.3. Only ever loaded on a Kaggle T4 by `VLLMTeacher`.
DEFAULT_TEACHER_MODEL = "Qwen/Qwen2.5-3B-Instruct"

ROUTES: tuple[str, ...] = ("in_domain", "out_of_domain", "needs_web")
GRADES: tuple[str, ...] = ("relevant", "irrelevant")

_SYSTEM_PROMPTS: dict[Job, str] = {
    "router": (
        "You are a query router. Reply with strict JSON of the form "
        '{"route": "in_domain"|"out_of_domain"|"needs_web"} and nothing else.'
    ),
    "rewriter": (
        "You are a query rewriter. Rewrite the user's query into a single standalone "
        "search query that needs no conversation context. Reply with the query only."
    ),
    "grader": (
        "You are a relevance grader. Reply with strict JSON of the form "
        '{"label": "relevant"|"irrelevant", "score": <float in [0,1]>} and nothing else.'
    ),
}


# ------------------------------------------------------------------------------------ #
# Seams: the teacher, and the tokenizer surface SFT actually needs
# ------------------------------------------------------------------------------------ #
@runtime_checkable
class Teacher(Protocol):
    """Batch text generation. The production implementation is vLLM on a Kaggle T4.

    Batch-shaped rather than one-prompt-at-a-time because that is the only shape in which
    50k short outputs finish in the 1-2 hours SS9 budgets: vLLM's throughput comes from
    continuous batching, and a per-prompt loop throws it away.
    """

    @property
    def name(self) -> str: ...

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> list[str]: ...


@runtime_checkable
class ChatTokenizer(Protocol):
    """The tokenizer surface SFT needs, which is slightly wider than the frozen Protocol.

    ``encode_chat`` (a `localmind.tokenizer.Tokenizer` method, not part of the frozen
    CONVENTIONS.md Protocol) is **required**, not optional. The frozen Protocol's
    ``apply_chat_template`` returns a *string*, and re-encoding that string cannot
    recover the role ids: ``encode`` deliberately refuses to recognise ``<|assistant|>``
    inside text so that a user typing it cannot inject a control token. Structural
    splicing is therefore the only correct way to build a supervised chat example, and
    depending on it here is a feature.
    """

    @property
    def vocab_size(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def pad_id(self) -> int: ...

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def encode_chat(
        self, messages: list[dict[str, str]], add_generation_prompt: bool = False
    ) -> list[int]: ...


class VLLMTeacher:
    """Qwen2.5-3B-Instruct on a Kaggle T4 via vLLM (SS3.3 / SS9 5a).

    Cannot run on the dev laptop: ``vllm`` and ``transformers`` are imported inside
    :meth:`generate`, so importing this module never touches the GPU or the network
    (CONVENTIONS.md: no network calls at import time).
    """

    def __init__(
        self,
        model: str = DEFAULT_TEACHER_MODEL,
        *,
        dtype: Literal["float16", "float32"] = "float16",
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        # ADR 0001: the target is a Turing T4 (SM 7.5) with no bf16 tensor cores. The
        # Literal makes "bfloat16" a type error rather than a silent slow path.
        self.model = model
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llm: Any = None
        self._tok: Any = None

    @property
    def name(self) -> str:
        return self.model

    def _ensure_loaded(self) -> None:  # pragma: no cover - needs a GPU
        if self._llm is not None:
            return
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
        from vllm import LLM  # type: ignore[import-not-found]

        self._tok = AutoTokenizer.from_pretrained(self.model)
        self._llm = LLM(
            model=self.model,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> list[str]:  # pragma: no cover - needs a GPU
        from vllm import SamplingParams  # type: ignore[import-not-found]

        self._ensure_loaded()
        rendered = [
            self._tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed)
        outputs = self._llm.generate(rendered, params)
        return [o.outputs[0].text.strip() for o in outputs]


_FIELD_RE = re.compile(r"^([A-Z_]+):[ \t]*(.*)$", re.MULTILINE)


def _fields(prompt: str) -> dict[str, str]:
    return {k: v.strip() for k, v in _FIELD_RE.findall(prompt)}


@dataclass
class DeterministicFakeTeacher:
    """Offline stand-in for the vLLM teacher: same Protocol, no weights, no network.

    It reads the tagged fields :func:`build_teacher_prompt` writes and answers from the
    same construction rule the prompt pool used, so the whole pipeline (prompting,
    parsing, validation, stratified spot-check, packing, training) is exercised end to
    end on CPU.

    Two knobs exist so the *failure* paths are testable rather than theoretical:

    ``malformed_every``
        emit unparseable text every N calls -- proves bad teacher output is dropped
        rather than trained on.
    ``wrong_every``
        emit a confidently wrong label every N calls -- proves the hand-verification
        agreement metric can actually go down. A spot-check that can only ever say
        "100%" is not a spot-check.
    """

    name: str = "fake-deterministic"
    malformed_every: int = 0
    wrong_every: int = 0
    _calls: int = field(default=0, repr=False)

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> list[str]:
        del max_tokens, temperature, seed
        out: list[str] = []
        for prompt in prompts:
            self._calls += 1
            if self.malformed_every and self._calls % self.malformed_every == 0:
                out.append("Sure! Here is my answer: it depends.")
                continue
            wrong = bool(self.wrong_every) and self._calls % self.wrong_every == 0
            out.append(self._answer(_fields(prompt), wrong=wrong))
        return out

    @staticmethod
    def _answer(f: Mapping[str, str], *, wrong: bool) -> str:
        task = f.get("TASK", "router")
        if task == "router":
            route = f.get("GOLD", "in_domain")
            if wrong:
                route = (
                    ROUTES[(ROUTES.index(route) + 1) % len(ROUTES)] if route in ROUTES else route
                )
            return json.dumps({"route": route})
        if task == "grader":
            label = f.get("GOLD", "relevant")
            if wrong:
                label = "irrelevant" if label == "relevant" else "relevant"
            score = 0.9 if label == "relevant" else 0.1
            return json.dumps({"label": label, "score": score})
        gold = f.get("GOLD", f.get("QUERY", ""))
        return f"{gold} (unrelated)" if wrong else gold


# ------------------------------------------------------------------------------------ #
# Prompts and outputs
# ------------------------------------------------------------------------------------ #
class SFTExample(BaseModel):
    """One (prompt, completion) pair, plus the bookkeeping the spot-check needs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    job: Job
    #: Tagged fields, e.g. ``{"QUERY": ..., "CHUNK": ...}``. The prompt is *derived*
    #: from these by `build_teacher_prompt`, never stored pre-rendered, so the wire
    #: format can change without invalidating a generated corpus.
    inputs: dict[str, str]
    #: The teacher's (validated, canonicalised) output -- the training target.
    completion: str
    #: The label the construction rule implies. Present because this corpus is
    #: synthetic-by-construction; in a corpus scraped from real traffic it is ``None``
    #: and the human spot-check is the only ground truth there is.
    gold: str | None = None
    teacher: str = "unknown"

    @property
    def label(self) -> str:
        """The teacher's answer reduced to a stratum key (a route / a grade / ``rewrite``)."""
        return label_of(self.job, self.completion)

    @property
    def verification_label(self) -> str:
        """What a human annotator actually adjudicates on the spot-check sheet.

        For the router and the grader that is the discrete class. The rewriter has no
        class -- `label` is the constant ``"rewrite"``, which is fine for stratification
        and useless for agreement -- so the thing to agree or disagree with is the
        rewritten query itself. Using `label` here would score every rewriter row as a
        match no matter what the teacher wrote, and inflate the reported agreement by
        roughly a third.
        """
        return self.completion if self.job == "rewriter" else self.label


def build_teacher_prompt(job: Job, inputs: Mapping[str, str], *, include_gold: bool = False) -> str:
    """Render the tagged prompt sent to the teacher.

    ``include_gold`` exists only for :class:`DeterministicFakeTeacher`, which has no
    weights and must be told the construction rule's answer to imitate one. The real
    :class:`VLLMTeacher` is called with ``include_gold=False`` -- leaking the label into
    a real teacher's prompt would make the agreement metric measure nothing at all.
    """
    lines = [f"TASK: {job}", _SYSTEM_PROMPTS[job]]
    order = ("HISTORY", "QUERY", "CHUNK")
    for key in order:
        if key in inputs:
            lines.append(f"{key}: {inputs[key]}")
    for key in sorted(set(inputs) - set(order) - {"GOLD"}):
        lines.append(f"{key}: {inputs[key]}")
    if include_gold and "GOLD" in inputs:
        lines.append(f"GOLD: {inputs['GOLD']}")
    return "\n".join(lines)


def prompt_messages(job: Job, inputs: Mapping[str, str]) -> list[dict[str, str]]:
    """The chat messages the *student* is trained on (system + user, no assistant turn)."""
    user = build_teacher_prompt(job, inputs, include_gold=False)
    return [{"role": "system", "content": _SYSTEM_PROMPTS[job]}, {"role": "user", "content": user}]


def parse_teacher_output(job: Job, text: str) -> str | None:
    """Validate + canonicalise a raw teacher completion; ``None`` means "drop this row".

    Router and grader must be strict JSON, because that is what GRPO's verifiable reward
    checks later (SS9 5d) -- accepting prose here would train the model to emit prose and
    then punish it for doing so.
    """
    text = text.strip()
    if not text:
        return None
    if job == "rewriter":
        first = text.splitlines()[0].strip().strip('"')
        return first if 0 < len(first) <= 200 else None
    obj = _strict_json(text)
    if obj is None:
        return None
    if job == "router":
        route = obj.get("route")
        return json.dumps({"route": route}) if route in ROUTES else None
    label = obj.get("label")
    score = obj.get("score")
    if label not in GRADES or not isinstance(score, int | float) or isinstance(score, bool):
        return None
    if not 0.0 <= float(score) <= 1.0:
        return None
    return json.dumps({"label": label, "score": round(float(score), 4)})


def safe_decode(tokenizer: ChatTokenizer, ids: Sequence[int]) -> str:
    """Decode student-sampled ids, tolerating byte sequences that are not valid UTF-8.

    A byte-level BPE vocabulary can represent *any* byte string, so an untrained (or
    merely unlucky) student will sample id sequences whose bytes are not decodable UTF-8
    -- `Tokenizer.decode` then raises `UnicodeDecodeError`. That is correct behaviour for
    a decoder and fatal for a training loop: on-policy KD (arm 2) and GRPO both decode
    the student's own samples, and a single malformed rollout would abort a multi-hour
    Kaggle run. Undecodable output is not an exception here, it is a *data point* -- the
    student produced something unparseable, which the teacher will correct and the
    verifiable reward will score as zero.
    """
    try:
        return tokenizer.decode(list(ids))
    except (UnicodeDecodeError, ValueError, KeyError):
        return ""


def _strict_json(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def label_of(job: Job, completion: str) -> str:
    obj = _strict_json(completion)
    if obj is None:
        return "rewrite" if job == "rewriter" else "malformed"
    if job == "router":
        return str(obj.get("route", "malformed"))
    if job == "grader":
        return str(obj.get("label", "malformed"))
    return "rewrite"


# ------------------------------------------------------------------------------------ #
# The seed-prompt pool
# ------------------------------------------------------------------------------------ #
_DOC_SUBJECTS = (
    "the parental leave policy",
    "the expense reimbursement limit",
    "the incident escalation path",
    "the on-call rotation",
    "the data retention schedule",
    "the vendor onboarding checklist",
    "the security review process",
    "the quarterly budget freeze",
)
_OOD_QUERIES = (
    "write me a haiku about otters",
    "what is 17 times 23",
    "tell me a joke about databases",
    "who won the 1998 world cup",
    "translate 'good morning' into Finnish",
    "what should I cook tonight",
)
_WEB_SUBJECTS = (
    "the current CVE score for log4shell",
    "today's USD to EUR rate",
    "the latest release of postgres",
    "this week's SEC filing deadline",
    "the newest pricing for S3 Glacier",
)
_CHUNK_TEMPLATE = (
    "Section {n}. {subject}: employees must submit the form within {d} business days. "
    "Exceptions require written approval from a director."
)
_DISTRACTOR_TEMPLATE = (
    "Appendix {n}. The office cafeteria serves lunch between 11:30 and 14:00. "
    "Dietary requirements should be flagged to facilities."
)


def seed_prompts(
    n: int, *, seed: int = 1337, job_mix: Mapping[str, float] | None = None
) -> list[tuple[Job, dict[str, str]]]:
    """A deterministic pool of ``n`` (job, inputs) pairs with a known-by-construction gold.

    Seeded and pure: the same ``seed`` yields the same pool, bit for bit
    (CONVENTIONS.md, determinism). This stands in for the real seed corpus (user query
    logs + retrieved chunks); its shape, not its content, is what the rest of the module
    is written against.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    mix = dict(job_mix or {"router": 1 / 3, "rewriter": 1 / 3, "grader": 1 / 3})
    unknown = set(mix) - set(JOBS)
    if unknown:
        raise ValueError(f"job_mix has unknown jobs: {sorted(unknown)}")
    total = sum(mix.values())
    if total <= 0:
        raise ValueError("job_mix weights must sum to a positive number")

    counts = {job: round(n * mix.get(job, 0.0) / total) for job in JOBS}
    drift = n - sum(counts.values())
    counts["router"] += drift
    counts["router"] = max(counts["router"], 0)

    rng = random.Random(seed)
    out: list[tuple[Job, dict[str, str]]] = []
    for job in JOBS:
        for i in range(max(counts[job], 0)):
            out.append((job, _make_inputs(job, i, rng)))
    rng.shuffle(out)
    return out[:n]


def _make_inputs(job: Job, i: int, rng: random.Random) -> dict[str, str]:
    if job == "router":
        bucket = ROUTES[i % 3]
        if bucket == "in_domain":
            subj = rng.choice(_DOC_SUBJECTS)
            query = f"what does the handbook say about {subj}?"
        elif bucket == "out_of_domain":
            query = rng.choice(_OOD_QUERIES)
        else:
            query = f"what is {rng.choice(_WEB_SUBJECTS)} right now?"
        return {"QUERY": query, "GOLD": bucket}

    if job == "rewriter":
        subj = rng.choice(_DOC_SUBJECTS)
        follow = rng.choice(
            ("does it apply to contractors", "how long does it take", "who signs it")
        )
        return {
            "HISTORY": f"user: what does the handbook say about {subj}?",
            "QUERY": follow + "?",
            "GOLD": f"{follow} for {subj}?",
        }

    subj = rng.choice(_DOC_SUBJECTS)
    relevant = i % 2 == 0
    days = rng.randint(3, 30)
    chunk = (
        _CHUNK_TEMPLATE.format(n=i % 40 + 1, subject=subj.capitalize(), d=days)
        if relevant
        else _DISTRACTOR_TEMPLATE.format(n=i % 40 + 1)
    )
    return {
        "QUERY": f"what does the handbook say about {subj}?",
        "CHUNK": chunk,
        "GOLD": "relevant" if relevant else "irrelevant",
    }


def generate_sft_dataset(
    teacher: Teacher,
    n: int,
    *,
    seed: int = 1337,
    job_mix: Mapping[str, float] | None = None,
    batch_size: int = 256,
    max_tokens: int = 64,
    temperature: float = 0.0,
    fake_teacher: bool | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[SFTExample]:
    """Prompt ``teacher`` over the seed pool and keep only rows that parse and validate.

    Rows the teacher botched are **dropped, not repaired**. A repaired row would be a
    row the teacher never actually produced, which quietly decouples the corpus from the
    thing the spot-check measured.
    """
    pool = seed_prompts(n, seed=seed, job_mix=job_mix)
    if fake_teacher is None:
        fake_teacher = isinstance(teacher, DeterministicFakeTeacher)

    out: list[SFTExample] = []
    for start in range(0, len(pool), max(batch_size, 1)):
        block = pool[start : start + max(batch_size, 1)]
        prompts = [build_teacher_prompt(j, inp, include_gold=fake_teacher) for j, inp in block]
        replies = teacher.generate(
            prompts, max_tokens=max_tokens, temperature=temperature, seed=seed + start
        )
        if len(replies) != len(block):
            raise ValueError(f"teacher returned {len(replies)} replies for {len(block)} prompts")
        for (job, inputs), reply in zip(block, replies, strict=True):
            completion = parse_teacher_output(job, reply)
            if completion is None:
                continue
            out.append(
                SFTExample(
                    job=job,
                    inputs={k: v for k, v in inputs.items() if k != "GOLD"},
                    completion=completion,
                    gold=inputs.get("GOLD"),
                    teacher=teacher.name,
                )
            )
        if progress is not None:
            progress(min(start + len(block), len(pool)), len(pool))
    return out


def write_examples(examples: Sequence[SFTExample], path: str | Path) -> Path:
    """Write JSONL. Used by ``--generate-teacher-data`` on Kaggle."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.model_dump(), sort_keys=True) + "\n")
    return p


def read_examples(path: str | Path) -> list[SFTExample]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [SFTExample.model_validate_json(line) for line in fh if line.strip()]


# ------------------------------------------------------------------------------------ #
# Hand verification: stratified sample + agreement (SS9 5a, "never ship unchecked")
# ------------------------------------------------------------------------------------ #
def stratified_sample(examples: Sequence[SFTExample], n: int, *, seed: int = 1337) -> list[int]:
    """Indices of a sample stratified by ``(job, teacher label)``, returned sorted.

    Stratified rather than uniform because the interesting failures are in the rare
    strata: a uniform draw of 300 rows from a corpus that is 4% ``needs_web`` gets ~12
    ``needs_web`` rows, which cannot distinguish 60% accuracy from 95%. Allocation is
    largest-remainder over the strata, so every stratum present in the corpus is present
    in the sheet.
    """
    if n <= 0 or not examples:
        return []
    n = min(n, len(examples))
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        strata[(ex.job, ex.label)].append(i)

    keys = sorted(strata)
    exact = {k: n * len(strata[k]) / len(examples) for k in keys}
    take = {k: min(math.floor(exact[k]), len(strata[k])) for k in keys}
    # Largest-remainder: hand the leftovers to the strata the floor hurt most, and give
    # every non-empty stratum at least one row before any stratum gets a second.
    leftover = n - sum(take.values())
    order = sorted(keys, key=lambda k: (take[k] > 0, -(exact[k] - math.floor(exact[k])), k))
    while leftover > 0:
        progressed = False
        for k in order:
            if leftover == 0:
                break
            if take[k] < len(strata[k]):
                take[k] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break

    rng = random.Random(seed)
    picked: list[int] = []
    for k in keys:
        pool = list(strata[k])
        rng.shuffle(pool)
        picked.extend(pool[: take[k]])
    return sorted(picked)


def write_verification_sheet(
    examples: Sequence[SFTExample], indices: Sequence[int], path: str | Path
) -> Path:
    """Write the rows a human must label, with ``human_label`` left empty on purpose.

    The teacher's own answer is written into ``teacher_label`` but the human field starts
    blank; pre-filling it would anchor the annotator onto the teacher and turn the
    agreement number into a measurement of the annotator's patience.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for i in indices:
            ex = examples[i]
            fh.write(
                json.dumps(
                    {
                        "index": i,
                        "job": ex.job,
                        "inputs": ex.inputs,
                        "teacher_completion": ex.completion,
                        "teacher_label": ex.verification_label,
                        "stratum": f"{ex.job}/{ex.label}",
                        "human_label": "",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return p


def load_verification_sheet(path: str | Path) -> list[dict[str, Any]]:
    """Read a filled sheet back; rows with an empty ``human_label`` are dropped."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("human_label", "")).strip():
                rows.append(row)
    return rows


@dataclass(frozen=True)
class AgreementReport:
    """Teacher-vs-human agreement on the stratified spot-check."""

    n: int
    agreement: float
    kappa: float
    per_stratum: dict[str, tuple[int, float]]
    threshold: float
    #: True when ``agreement < threshold``. The caller is expected to surface this, not
    #: swallow it -- SS9: "Never ship synthetic data you haven't spot-checked."
    below_threshold: bool

    def warning(self) -> str | None:
        if not self.below_threshold:
            return None
        return (
            f"teacher/human agreement {self.agreement:.3f} on n={self.n} is below the "
            f"{self.threshold:.3f} gate -- do not train on this corpus; fix the teacher "
            "prompt or the seed pool first"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "agreement": self.agreement,
            "kappa": self.kappa,
            "per_stratum": {k: {"n": v[0], "agreement": v[1]} for k, v in self.per_stratum.items()},
            "threshold": self.threshold,
            "below_threshold": self.below_threshold,
        }


def agreement_report(
    rows: Sequence[Mapping[str, Any]], *, threshold: float = 0.90
) -> AgreementReport:
    """Agreement + Cohen's kappa between ``teacher_label`` and ``human_label``.

    Kappa as well as raw agreement because raw agreement is inflated by the marginals:
    a router corpus that is 80% ``in_domain`` gets 80% "agreement" from a teacher that
    has learned to say ``in_domain`` and nothing else. ``cohens_kappa`` comes from
    `localmind.eval.stats`, imported lazily so this module stays importable without it.
    """
    if not rows:
        return AgreementReport(0, float("nan"), float("nan"), {}, threshold, True)

    teacher = [str(r["teacher_label"]) for r in rows]
    human = [str(r["human_label"]).strip() for r in rows]
    hits = [int(t == h) for t, h in zip(teacher, human, strict=True)]
    overall = sum(hits) / len(hits)

    by_stratum: dict[str, list[int]] = defaultdict(list)
    for r, hit in zip(rows, hits, strict=True):
        by_stratum[f"{r.get('job', '?')}/{r['teacher_label']}"].append(hit)
    per_stratum = {k: (len(v), sum(v) / len(v)) for k, v in sorted(by_stratum.items())}

    try:
        from localmind.eval.stats import cohens_kappa

        kappa = float(cohens_kappa(teacher, human))
    except Exception:  # pragma: no cover - eval package optional at this layer
        kappa = float("nan")

    return AgreementReport(
        n=len(rows),
        agreement=overall,
        kappa=kappa,
        per_stratum=per_stratum,
        threshold=threshold,
        below_threshold=overall < threshold,
    )


# ------------------------------------------------------------------------------------ #
# Encoding: chat template + prompt masking + packing with boundary masking
# ------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class EncodedExample:
    """One example as flat ids, with the assistant span marked.

    ``full_ids`` is the whole conversation. ``completion_start`` is the index of the
    first *supervised* token in ``full_ids``: everything before it is prompt, including
    the ``<|assistant|>`` marker itself (the model is told to speak, it does not have to
    predict being told).
    """

    full_ids: list[int]
    completion_start: int
    job: Job

    @property
    def n_supervised(self) -> int:
        return max(len(self.full_ids) - self.completion_start, 0)


def encode_sft_example(
    tokenizer: ChatTokenizer, example: SFTExample, *, max_len: int | None = None
) -> EncodedExample:
    """Apply the chat template and locate the assistant span.

    The span is found by *construction*, not by searching for a marker id in the encoded
    stream: encode the prompt with ``add_generation_prompt=True``, encode prompt +
    assistant turn, and assert the former is a prefix of the latter. Searching for the
    last ``<|assistant|>`` id would be wrong the moment a multi-turn example carries a
    prior assistant turn, and it would fail silently.
    """
    msgs = prompt_messages(example.job, example.inputs)
    prompt_ids = tokenizer.encode_chat(msgs, add_generation_prompt=True)
    full_ids = tokenizer.encode_chat(
        [*msgs, {"role": "assistant", "content": example.completion}],
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "chat template is not prefix-stable: encoding the prompt with "
            "add_generation_prompt=True must be a prefix of the full conversation"
        )
    completion_start = len(prompt_ids)
    if max_len is not None and len(full_ids) > max_len:
        # Truncating from the right is the obvious thing and it is wrong: for these jobs
        # the prompt (a query plus a retrieved chunk) dwarfs the answer, so a right
        # truncation deletes the entire assistant span and yields a row whose labels are
        # all -100. Cross-entropy over such a row is 0/0, and a single one of them turns
        # the whole training loss into NaN. Drop the *prompt's* head instead, keeping the
        # BOS token, so the supervised span always survives.
        n_completion = len(full_ids) - completion_start
        if n_completion >= max_len:
            # Pathological: the answer alone overflows. Keep BOS plus as much answer as
            # fits; there is no prompt left to trim.
            full_ids = [full_ids[0], *full_ids[completion_start : completion_start + max_len - 1]]
            completion_start = 1
        else:
            drop = len(full_ids) - max_len
            full_ids = [full_ids[0], *full_ids[1 + drop :]]
            completion_start -= drop
    return EncodedExample(
        full_ids=list(full_ids), completion_start=max(completion_start, 1), job=example.job
    )


@dataclass(frozen=True)
class PackedRow:
    """A packed training row, already shifted for `LocalMindTransformer.forward`.

    ``labels[t]`` is the token that follows ``input_ids[t]``, or ``-100``. ``doc_ids``
    carries the per-position segment id so that a future forward pass taking an
    attention mask can consume :func:`boundary_attention_mask` without repacking.
    """

    input_ids: list[int]
    labels: list[int]
    doc_ids: list[int]

    def __post_init__(self) -> None:
        if not (len(self.input_ids) == len(self.labels) == len(self.doc_ids)):
            raise ValueError("input_ids, labels and doc_ids must be the same length")

    @property
    def n_supervised(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE_INDEX)


#: `doc_ids` value for right-padding. Distinct from every real segment id so
#: `boundary_attention_mask` isolates padding into its own block.
PAD_DOC_ID = -1


def pack_sft(
    encoded: Sequence[EncodedExample],
    seq_len: int,
    *,
    pad_id: int,
    mask_prompt_tokens: bool = True,
    boundary_masking: bool = True,
    drop_last: bool = False,
) -> list[PackedRow]:
    """Concatenate examples into ``seq_len``-long rows with prompt + boundary masking.

    Two independent masks are applied:

    * **prompt masking** (``mask_prompt_tokens``) -- every target that is a prompt token
      becomes ``-100``, so gradient only ever flows through assistant spans (SS9 5a).
    * **boundary masking** (``boundary_masking``) -- the first token of each example is
      masked *as a target*, because that position asks the last token of the previous,
      unrelated example to predict it. Nothing is learnable there.

    Worth stating plainly, because it is the kind of thing that hides a bug: with
    ``mask_prompt_tokens=True`` boundary masking is **subsumed** by prompt masking, since
    every chat example opens with ``<|bos|>`` and a system/user turn, all of which are
    prompt. It becomes load-bearing exactly when prompt masking is off -- packed raw
    continuations, which is what KD arm 1 uses when the teacher output *is* the whole
    sequence. Both flags exist so that case is correct rather than accidentally correct.

    Honest limitation: correct packing would *also* stop attention crossing a boundary,
    and the frozen ``LocalMindTransformer.forward(input_ids, targets, past_kvs,
    use_cache)`` signature accepts no attention mask, so that is not reachable from here.
    Boundary masking is therefore enforced in the **loss** only, and per-position
    ``doc_ids`` are carried on every row (consumable via
    :func:`boundary_attention_mask`) so the attention-side fix is a change in the model,
    not a repack of the corpus. This is an approximation and is reported as one.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    rows: list[PackedRow] = []
    buf_ids: list[int] = []
    buf_lab: list[int] = []
    buf_doc: list[int] = []
    doc = 0

    def flush(pad: bool) -> None:
        nonlocal buf_ids, buf_lab, buf_doc
        if len(buf_ids) < 2 or all(x == IGNORE_INDEX for x in buf_lab[1:]):
            # A row with no supervised target contributes 0/0 to the mean cross-entropy,
            # which is NaN, which poisons the whole batch. Drop it rather than emit it.
            buf_ids, buf_lab, buf_doc = [], [], []
            return
        if len(buf_ids) < seq_len + 1:
            if not pad:
                buf_ids, buf_lab, buf_doc = [], [], []
                return
            missing = seq_len + 1 - len(buf_ids)
            buf_ids.extend([pad_id] * missing)
            buf_lab.extend([IGNORE_INDEX] * missing)
            buf_doc.extend([PAD_DOC_ID] * missing)
        rows.append(
            PackedRow(
                input_ids=buf_ids[:seq_len],
                labels=buf_lab[1 : seq_len + 1],
                doc_ids=buf_doc[:seq_len],
            )
        )
        buf_ids, buf_lab, buf_doc = [], [], []

    for ex in encoded:
        ids = ex.full_ids
        if not ids:
            continue
        for k, tok in enumerate(ids):
            if len(buf_ids) == seq_len + 1:
                flush(pad=False)
            # `buf_lab[j]` is the supervision attached to *token j itself*. The single
            # shift to next-token targets happens once, in `flush`.
            lab = tok
            if mask_prompt_tokens and k < ex.completion_start:
                lab = IGNORE_INDEX
            if boundary_masking and k == 0 and buf_ids:
                lab = IGNORE_INDEX
            buf_ids.append(tok)
            buf_lab.append(lab)
            buf_doc.append(doc)
        doc += 1

    flush(pad=not drop_last)
    return rows


def boundary_attention_mask(doc_ids: Sequence[int]) -> Tensor:
    """Block-diagonal causal mask ``(T, T)``, True where attention is allowed.

    Matches `localmind.data.packing.build_doc_mask` semantics. Not consumed by the
    current forward pass (see :func:`pack_sft`); provided so the attention-side fix is
    available the moment the model grows a mask argument, and so the *intent* of
    "boundary masking" is testable today rather than asserted in prose.
    """
    import torch

    ids = torch.as_tensor(list(doc_ids), dtype=torch.long)
    same = ids[:, None] == ids[None, :]
    causal = torch.ones(len(ids), len(ids), dtype=torch.bool).tril()
    return same & causal


def iter_batches(
    rows: Sequence[PackedRow], batch_size: int, *, shuffle: bool = True, seed: int = 1337
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield ``(input_ids, labels)`` int64 tensors of shape ``(B, T)``."""
    import torch

    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), max(batch_size, 1)):
        block = [rows[i] for i in order[start : start + max(batch_size, 1)]]
        yield (
            torch.tensor([r.input_ids for r in block], dtype=torch.long),
            torch.tensor([r.labels for r in block], dtype=torch.long),
        )


# ------------------------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------------------------ #
class SFTConfig(BaseModel):
    """``configs/train/sft.yaml``. CONVENTIONS.md: no hyperparameter inline in code."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, protected_namespaces=()
    )

    model_config_path: str = Field(alias="model_config")
    teacher: str = DEFAULT_TEACHER_MODEL
    n_examples: int = Field(default=40_000, gt=0)
    job_mix: dict[str, float] = Field(default_factory=lambda: dict.fromkeys(JOBS, 1 / 3))
    seq_len: int = Field(default=1024, gt=0)
    micro_batch_size: int = Field(default=8, gt=0)
    epochs: int = Field(default=2, gt=0)

    #: SS9 5a: "cosine decay from 10% of pretrain LR". The pretrain peak lives in
    #: `configs/train/pretrain.yaml`; it is restated here so the SFT run is
    #: self-describing and hash-loggable without cross-reading another file.
    pretrain_peak_lr: float = Field(default=3.0e-3, gt=0.0)
    peak_lr_frac_of_pretrain: float = Field(default=0.1, gt=0.0, le=1.0)
    schedule: Literal["cosine"] = "cosine"
    warmup_frac: float = Field(default=0.03, ge=0.0, lt=1.0)
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)

    optimizer: Literal["adamw"] = "adamw"
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(default=0.1, ge=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    #: ADR 0001. ``bf16`` is absent from this Literal on purpose: a bf16 config is a
    #: validation error at load time, not a crash on a T4 nine hours in.
    precision: Literal["fp16", "fp32"] = "fp16"

    mask_prompt_tokens: bool = True
    boundary_masking: bool = True
    verify_sample_size: int = Field(default=300, ge=0)
    verify_min_agreement: float = Field(default=0.90, ge=0.0, le=1.0)
    seed: int = 1337
    out_dir: str = "artifacts/post/sft"

    @property
    def peak_lr(self) -> float:
        return self.pretrain_peak_lr * self.peak_lr_frac_of_pretrain

    @model_validator(mode="after")
    def _check_mix(self) -> SFTConfig:
        unknown = set(self.job_mix) - set(JOBS)
        if unknown:
            raise ValueError(f"job_mix has unknown jobs: {sorted(unknown)}")
        if sum(self.job_mix.values()) <= 0:
            raise ValueError("job_mix weights must sum to a positive number")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTConfig:
        import yaml

        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping")
        return cls.model_validate(raw)


# ------------------------------------------------------------------------------------ #
# Shared training plumbing (kd.py / dpo.py / grpo.py import these)
# ------------------------------------------------------------------------------------ #
def cosine_lr(
    step: int, total_steps: int, peak_lr: float, *, warmup_steps: int = 0, min_lr_ratio: float = 0.1
) -> float:
    """Linear warmup then cosine decay to ``peak_lr * min_lr_ratio``.

    Cosine here rather than the WSD schedule Phase 4 uses (ADR 0002): WSD's win is that
    you can branch a decay off a long stable phase without re-running it, which matters
    for a 5.7k-step pretrain and is irrelevant for a 2-epoch SFT of known length.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    min_lr = peak_lr * min_lr_ratio
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class StepMetrics:
    step: int
    loss: float
    lr: float
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class StageResult:
    """What every Phase 5 stage returns: enough to say whether it worked, and by how much."""

    stage: str
    steps: int
    history: list[StepMetrics] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def initial_loss(self) -> float:
        return self.history[0].loss if self.history else float("nan")

    @property
    def final_loss(self) -> float:
        return self.history[-1].loss if self.history else float("nan")

    def mean_loss(self, first_k: int | None = None, last_k: int | None = None) -> float:
        if not self.history:
            return float("nan")
        vals = [m.loss for m in self.history]
        if first_k is not None:
            vals = vals[:first_k]
        if last_k is not None:
            vals = vals[-last_k:]
        return sum(vals) / len(vals)

    @property
    def improved(self) -> bool:
        """Mean of the first quarter vs the last quarter -- a single noisy final step is
        not evidence that anything was learned."""
        if len(self.history) < 4:
            return self.final_loss < self.initial_loss
        k = max(len(self.history) // 4, 1)
        return self.mean_loss(last_k=k) < self.mean_loss(first_k=k)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "steps": self.steps,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "improved": self.improved,
            "history": [
                {"step": m.step, "loss": m.loss, "lr": m.lr, **m.extra} for m in self.history
            ],
            **self.summary,
        }


def build_optimizer(model: nn.Module, lr: float, cfg: Any) -> Any:
    """AdamW over `localmind.train.optim`, with a local fallback.

    ``localmind/train/`` is owned by another task and may be mid-flight, so it is
    imported *inside* this function and a plain ``torch.optim.AdamW`` stands in if the
    symbol is missing. Phase 5 must not be able to break because Phase 4 moved.
    """
    import torch

    betas = tuple(getattr(cfg, "betas", (0.9, 0.95)))
    wd = float(getattr(cfg, "weight_decay", 0.1))
    try:
        from localmind.train.optim import build_adamw

        return build_adamw(model, lr=lr, betas=(betas[0], betas[1]), weight_decay=wd)
    except Exception:  # pragma: no cover - only when train/ is absent or moved
        decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": wd, "lr_scale": 1.0},
                {"params": no_decay, "weight_decay": 0.0, "lr_scale": 1.0},
            ],
            lr=lr,
            betas=(betas[0], betas[1]),
        )


def set_learning_rate(optimizer: Any, lr: float) -> None:
    """`localmind.train.optim.set_lr` if present, else the two-line equivalent."""
    try:
        from localmind.train.optim import set_lr

        set_lr(optimizer, lr)
    except Exception:  # pragma: no cover
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))


def seed_everything(seed: int) -> None:
    """`localmind.train.loop.seed_everything` if present, else the local equivalent."""
    try:
        from localmind.train.loop import seed_everything as _seed

        _seed(seed)
        return
    except Exception:  # pragma: no cover
        pass
    import numpy as np
    import torch

    random.seed(seed)
    # Legacy global seeding on purpose: this is the fallback for
    # `localmind.train.loop.seed_everything`, and it has to reproduce that function's
    # effect on the *global* numpy state, which a local `default_rng` would not.
    np.random.seed(seed % (2**32))  # noqa: NPY002
    torch.manual_seed(seed)


def run_stage(
    model: nn.Module,
    step_fn: Callable[[int], tuple[Tensor, dict[str, float]]],
    *,
    stage: str,
    steps: int,
    peak_lr: float,
    optimizer: Any | None = None,
    cfg: Any | None = None,
    warmup_frac: float = 0.03,
    min_lr_ratio: float = 0.1,
    grad_clip: float = 1.0,
    seed: int = 1337,
    on_step: Callable[[StepMetrics], None] | None = None,
) -> StageResult:
    """The one optimisation loop all four Phase 5 stages share.

    ``step_fn(step)`` returns ``(scalar_loss_to_backward, extra_metrics)``. Everything
    stage-specific -- which loss, which batch, which reference model -- lives in that
    closure, so this function stays small enough to be obviously correct.

    fp32 on CPU: SS9 runs on a T4 where the caller wraps ``step_fn`` in fp16 autocast +
    ``GradScaler`` (ADR 0001, never bf16). There is nothing to scale on CPU and adding a
    scaler here would only make the offline tests slower and less deterministic.
    """
    import torch

    seed_everything(seed)
    opt = optimizer if optimizer is not None else build_optimizer(model, peak_lr, cfg)
    warmup_steps = round(steps * warmup_frac)
    result = StageResult(stage=stage, steps=0)

    model.train()
    for step in range(steps):
        lr = cosine_lr(step, steps, peak_lr, warmup_steps=warmup_steps, min_lr_ratio=min_lr_ratio)
        set_learning_rate(opt, lr)
        opt.zero_grad(set_to_none=True)
        loss, extra = step_fn(step)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip
        )
        opt.step()
        metrics = StepMetrics(
            step=step,
            loss=float(loss.detach()),
            lr=lr,
            extra={**extra, "grad_norm": float(grad_norm)},
        )
        result.history.append(metrics)
        result.steps = step + 1
        if on_step is not None:
            on_step(metrics)
    return result


def run_sft(
    model: nn.Module,
    rows: Sequence[PackedRow],
    *,
    cfg: SFTConfig | None = None,
    epochs: int | None = None,
    batch_size: int = 4,
    peak_lr: float | None = None,
    seed: int = 1337,
    max_steps: int | None = None,
) -> StageResult:
    """Phase 5a: train on assistant spans only, over pre-packed rows.

    ``ModelOutput.loss`` (CE + z-loss) is what gets ``.backward()``; ``ce_loss`` is what
    is reported, because z-loss is a regulariser and not a measure of prediction quality
    (`localmind.model.transformer.ModelOutput`).
    """
    if not rows:
        raise ValueError("run_sft needs at least one packed row")
    n_epochs = epochs if epochs is not None else (cfg.epochs if cfg else 1)
    lr = peak_lr if peak_lr is not None else (cfg.peak_lr if cfg else 3.0e-4)
    bs = batch_size if cfg is None else min(batch_size, cfg.micro_batch_size)

    batches: list[tuple[Tensor, Tensor]] = []
    for epoch in range(n_epochs):
        batches.extend(iter_batches(rows, bs, shuffle=True, seed=seed + epoch))
    if max_steps is not None:
        batches = batches[:max_steps]
    if not batches:
        raise ValueError("no batches produced; check batch_size and rows")

    supervised = sum(r.n_supervised for r in rows)
    total = sum(len(r.labels) for r in rows)
    if supervised == 0:
        raise ValueError(
            "no packed row carries a supervised target: every label is -100. The loss "
            "would be NaN. Check seq_len against the encoded example lengths."
        )

    def step_fn(step: int) -> tuple[Tensor, dict[str, float]]:
        input_ids, labels = batches[step]
        out = model(input_ids, targets=labels)
        if out.loss is None:  # pragma: no cover - forward always returns a loss here
            raise RuntimeError("model returned no loss; were targets passed?")
        ce = float(out.ce_loss.detach()) if out.ce_loss is not None else float(out.loss.detach())
        return out.loss, {"ce_loss": ce}

    result = run_stage(
        model,
        step_fn,
        stage="sft",
        steps=len(batches),
        peak_lr=lr,
        cfg=cfg,
        warmup_frac=cfg.warmup_frac if cfg else 0.03,
        min_lr_ratio=cfg.min_lr_ratio if cfg else 0.1,
        grad_clip=cfg.grad_clip if cfg else 1.0,
        seed=seed,
    )
    result.summary["supervised_token_frac"] = supervised / total if total else 0.0
    result.summary["masked_token_frac"] = 1.0 - (supervised / total if total else 0.0)
    result.summary["n_rows"] = len(rows)
    return result


# ------------------------------------------------------------------------------------ #
# CLI (notebooks/kaggle/02_distill.ipynb calls this exact form)
# ------------------------------------------------------------------------------------ #
def main(argv: Sequence[str] | None = None) -> int:
    """``python -m localmind.post.sft --generate-teacher-data --teacher ... --n ... --out ...``"""
    parser = argparse.ArgumentParser(description="Phase 5a SFT data generation and training")
    parser.add_argument("--generate-teacher-data", action="store_true")
    parser.add_argument("--teacher", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--out", default="artifacts/post/sft_data.jsonl")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--fake-teacher",
        action="store_true",
        help="use DeterministicFakeTeacher (offline smoke test, no GPU)",
    )
    parser.add_argument("--verify-out", default=None, help="write the stratified spot-check sheet")
    args = parser.parse_args(argv)

    cfg = SFTConfig.from_yaml(args.config) if args.config else None
    n = args.n if cfg is None else min(args.n, cfg.n_examples)
    mix = cfg.job_mix if cfg else None

    if not args.generate_teacher_data:
        parser.error("nothing to do: pass --generate-teacher-data")

    teacher: Teacher = (
        DeterministicFakeTeacher() if args.fake_teacher else VLLMTeacher(model=args.teacher)
    )
    examples = generate_sft_dataset(teacher, n, seed=args.seed, job_mix=mix)
    path = write_examples(examples, args.out)
    counts = Counter(f"{e.job}/{e.label}" for e in examples)
    print(f"wrote {len(examples)} examples to {path}")
    for key, count in sorted(counts.items()):
        print(f"  {key}: {count}")

    if args.verify_out:
        size = cfg.verify_sample_size if cfg else 300
        idx = stratified_sample(examples, size, seed=args.seed)
        sheet = write_verification_sheet(examples, idx, args.verify_out)
        print(f"wrote {len(idx)}-row stratified spot-check sheet to {sheet}")
        print("fill in human_label, then call agreement_report() before training on this corpus")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
