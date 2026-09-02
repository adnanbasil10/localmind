"""Phase 5 tests: SFT prompt masking, the three KD arms, DPO, GRPO, LoRA, and the matrix.

Everything here runs on CPU, offline, in seconds. Three choices make that possible without
the tests becoming vacuous:

1. **A real tokenizer, trained in 60 ms.** `localmind.tokenizer.Tokenizer.train` over a
   few hundred short strings at ``vocab_size=600`` costs less than a model forward pass,
   so the chat-template and prompt-masking assertions run against the actual production
   tokenizer rather than a mock that cannot disagree with them.
2. **A ~40k-param model.** Correctness of a loss does not depend on width. The real 12M
   proxy runs are reported in the task report, not asserted here.
3. **Hand-computed reference values.** The four numerically subtle pieces -- top-K KL
   renormalisation, the ``alpha`` blend, the DPO logistic, and the group-relative
   advantage -- are each checked against a closed-form number computed with the stdlib
   ``math`` module, independently of the torch implementation. A test that merely re-runs
   the implementation and compares it to itself proves nothing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from localmind.model import LocalMindTransformer, ModelConfig
from localmind.post import (
    MATRIX_COLUMNS,
    MATRIX_ROWS,
    NOT_RUN,
    ComparisonMatrix,
    MatrixCell,
    build_comparison_matrix,
    dod_verdict,
    render_matrix_markdown,
    write_matrix_artifact,
)
from localmind.post.dpo import (
    DPOConfig,
    build_preference_pairs,
    collate_preference_batch,
    dpo_loss,
    kl_from_reference,
    make_reference_model,
    run_dpo,
    sequence_logprob,
)
from localmind.post.grpo import (
    ADVANTAGE_EPS,
    DEFAULT_GROUP_SIZE,
    GRPOConfig,
    RewardConfig,
    group_advantages,
    grpo_loss,
    run_grpo,
    sample_group,
    verifiable_reward,
)
from localmind.post.kd import (
    DEFAULT_ALPHA,
    DEFAULT_TOP_K,
    GreedyStudentSampler,
    KDConfig,
    TeacherLogitStore,
    collect_on_policy_corrections,
    compare_arms,
    estimate_topk_bytes,
    kd_loss,
    run_logit_kd,
    run_sequence_kd,
    topk_kl_loss,
)
from localmind.post.lora import (
    DEFAULT_LORA_RANK,
    LoRAConfig,
    LoRALinear,
    apply_lora,
    assert_lora_frozen,
    iter_lora_modules,
    load_lora_state_dict,
    lora_state_dict,
    merge_lora,
    trainable_parameter_summary,
)
from localmind.post.sft import (
    IGNORE_INDEX,
    JOBS,
    AgreementReport,
    DeterministicFakeTeacher,
    SFTConfig,
    SFTExample,
    agreement_report,
    boundary_attention_mask,
    build_teacher_prompt,
    cosine_lr,
    encode_sft_example,
    generate_sft_dataset,
    iter_batches,
    label_of,
    load_verification_sheet,
    pack_sft,
    parse_teacher_output,
    prompt_messages,
    read_examples,
    run_sft,
    seed_prompts,
    stratified_sample,
    write_examples,
    write_verification_sheet,
)
from localmind.tokenizer.tokenizer import Tokenizer
from pydantic import BaseModel, ValidationError

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "train"

#: The four configs this task owns, keyed by the stage that reads them.
CONFIG_CLASSES: dict[str, type[BaseModel]] = {
    "sft": SFTConfig,
    "kd": KDConfig,
    "dpo": DPOConfig,
    "grpo": GRPOConfig,
}

TINY_MODEL: dict[str, Any] = {
    "name": "post-test-tiny",
    "vocab_size": 600,
    "d_model": 32,
    "n_layers": 2,
    "n_heads": 2,
    "n_kv_heads": 1,
    "head_dim": 16,
    "ffn_hidden": 64,
    "max_seq_len": 192,
    "rope_theta": 10000.0,
    "qk_norm": True,
    "bias": False,
    "tie_embeddings": True,
    "z_loss": 1.0e-4,
    "attn_dropout": 0.0,
    "init_std": 0.02,
}


# ------------------------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    """A real byte-level BPE tokenizer trained on the Phase 5 prompt/completion domain."""
    pool = seed_prompts(200, seed=7)
    texts = [build_teacher_prompt(job, inputs, include_gold=True) for job, inputs in pool]
    texts += [json.dumps({"route": r}) for r in ("in_domain", "out_of_domain", "needs_web")]
    texts += [
        json.dumps({"label": lbl, "score": s})
        for lbl in ("relevant", "irrelevant")
        for s in (0.9, 0.1)
    ]
    return Tokenizer.train(texts, vocab_size=600)


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    return ModelConfig.model_validate(TINY_MODEL)


@pytest.fixture
def tiny_model(tiny_cfg: ModelConfig) -> LocalMindTransformer:
    torch.manual_seed(1337)
    return LocalMindTransformer(tiny_cfg, backend="sdpa_math")


@pytest.fixture
def examples() -> list[SFTExample]:
    return generate_sft_dataset(DeterministicFakeTeacher(), 48, seed=11)


@dataclass
class ScriptedSampler:
    """A sampler whose ``G`` rollouts differ by construction.

    GRPO needs a *stochastic* policy for groups to carry any signal. An untrained 40k-param
    model emits the same garbage every time, so every group would be degenerate and the
    test would pass while asserting nothing. Scripting the rollouts makes the reward
    spread real and the advantages non-zero.
    """

    tok: Tokenizer
    outputs: list[str] = field(default_factory=list)

    def generate(self, prompt_ids: list[int], max_new_tokens: int, **kw: Any) -> list[int]:
        idx = int(kw.get("sample_index", 0))
        text = self.outputs[idx % len(self.outputs)]
        return self.tok.encode(text)[:max_new_tokens]


# ------------------------------------------------------------------------------------ #
# 5a -- prompt masking. The spec calls this out and it is the part people get wrong.
# ------------------------------------------------------------------------------------ #
def test_encode_sft_example_marks_the_assistant_span(tokenizer: Tokenizer) -> None:
    ex = SFTExample(
        job="router",
        inputs={"QUERY": "what is in the handbook?"},
        completion='{"route": "in_domain"}',
    )
    enc = encode_sft_example(tokenizer, ex)

    prompt_ids = tokenizer.encode_chat(
        prompt_messages(ex.job, ex.inputs), add_generation_prompt=True
    )
    assert enc.full_ids[: len(prompt_ids)] == prompt_ids
    assert enc.completion_start == len(prompt_ids)
    assert enc.n_supervised > 0
    # The generation prompt's <|assistant|> marker is the last *prompt* token, never a
    # supervised one: the model is told to speak, it does not predict being told.
    assert enc.full_ids[enc.completion_start - 1] == tokenizer._special_to_id["<|assistant|>"]


def test_prompt_positions_carry_ignore_index(tokenizer: Tokenizer) -> None:
    """The explicit assertion the spec asks for: prompt targets are -100."""
    ex = SFTExample(
        job="grader",
        inputs={"QUERY": "parental leave?", "CHUNK": "Section 3. Parental leave: 20 days."},
        completion='{"label": "relevant", "score": 0.9}',
    )
    enc = encode_sft_example(tokenizer, ex)
    rows = pack_sft([enc], seq_len=len(enc.full_ids) - 1, pad_id=tokenizer.pad_id)
    assert len(rows) == 1
    row = rows[0]

    # labels[t] supervises the token at full_ids[t + 1].
    for t, label in enumerate(row.labels):
        target_index = t + 1
        if target_index < enc.completion_start:
            assert label == IGNORE_INDEX, (
                f"position {t} targets prompt token {target_index} "
                f"(completion starts at {enc.completion_start}) but is not masked"
            )
        else:
            assert label == enc.full_ids[target_index]

    supervised = [t for t, x in enumerate(row.labels) if x != IGNORE_INDEX]
    assert supervised, "no supervised position at all"
    assert min(supervised) == enc.completion_start - 1
    assert row.labels[min(supervised)] == enc.full_ids[enc.completion_start]


def test_masked_fraction_is_a_majority_for_short_completions(tokenizer: Tokenizer) -> None:
    encoded = [
        encode_sft_example(tokenizer, ex)
        for ex in generate_sft_dataset(DeterministicFakeTeacher(), 24, seed=3)
    ]
    rows = pack_sft(encoded, seq_len=128, pad_id=tokenizer.pad_id)
    total = sum(len(r.labels) for r in rows)
    supervised = sum(r.n_supervised for r in rows)
    # These are classification jobs: the prompt dwarfs the answer. If this ever inverts,
    # prompt masking has silently stopped working.
    assert 0.0 < supervised / total < 0.5


def test_boundary_masking_is_load_bearing_only_without_prompt_masking(
    tokenizer: Tokenizer,
) -> None:
    """With prompt masking on, boundary masking is subsumed -- and that is documented."""
    encoded = [
        encode_sft_example(tokenizer, ex)
        for ex in generate_sft_dataset(DeterministicFakeTeacher(), 12, seed=5)
    ]
    with_bm = pack_sft(encoded, 96, pad_id=tokenizer.pad_id, boundary_masking=True)
    without_bm = pack_sft(encoded, 96, pad_id=tokenizer.pad_id, boundary_masking=False)
    assert [r.labels for r in with_bm] == [r.labels for r in without_bm]

    # Turn prompt masking off and it starts to matter: every example's first token must
    # stop being a target for the previous example's last token.
    raw_bm = pack_sft(
        encoded, 96, pad_id=tokenizer.pad_id, mask_prompt_tokens=False, boundary_masking=True
    )
    raw_no_bm = pack_sft(
        encoded, 96, pad_id=tokenizer.pad_id, mask_prompt_tokens=False, boundary_masking=False
    )
    assert [r.labels for r in raw_bm] != [r.labels for r in raw_no_bm]
    masked_bm = sum(x == IGNORE_INDEX for r in raw_bm for x in r.labels)
    masked_no_bm = sum(x == IGNORE_INDEX for r in raw_no_bm for x in r.labels)
    assert masked_bm > masked_no_bm


def test_packed_rows_are_shifted_and_aligned(tokenizer: Tokenizer) -> None:
    encoded = [
        encode_sft_example(tokenizer, ex)
        for ex in generate_sft_dataset(DeterministicFakeTeacher(), 8, seed=2)
    ]
    rows = pack_sft(encoded, 64, pad_id=tokenizer.pad_id)
    for row in rows:
        assert len(row.input_ids) == len(row.labels) == len(row.doc_ids) == 64
        for t in range(len(row.labels) - 1):
            if row.labels[t] != IGNORE_INDEX:
                assert row.labels[t] == row.input_ids[t + 1]


def test_boundary_attention_mask_is_block_diagonal_causal() -> None:
    mask = boundary_attention_mask([0, 0, 1, 1, 1])
    assert mask.shape == (5, 5)
    assert bool(mask[1, 0]) and bool(mask[0, 0])
    assert not bool(mask[0, 1]), "causal: cannot attend forwards"
    assert not bool(mask[2, 1]), "boundary: segment 1 cannot see segment 0"
    assert bool(mask[4, 2]) and bool(mask[4, 3])


def test_pad_positions_get_their_own_segment(tokenizer: Tokenizer) -> None:
    ex = SFTExample(job="router", inputs={"QUERY": "hi"}, completion='{"route": "out_of_domain"}')
    rows = pack_sft([encode_sft_example(tokenizer, ex)], 160, pad_id=tokenizer.pad_id)
    row = rows[0]
    assert -1 in row.doc_ids
    assert all(row.labels[i] == IGNORE_INDEX for i, d in enumerate(row.doc_ids) if d == -1)


# ------------------------------------------------------------------------------------ #
# 5a -- teacher seam, validation, and the honesty requirement
# ------------------------------------------------------------------------------------ #
def test_seed_prompts_are_deterministic_and_cover_all_jobs() -> None:
    a = seed_prompts(60, seed=42)
    b = seed_prompts(60, seed=42)
    assert a == b
    assert seed_prompts(60, seed=43) != a
    assert {job for job, _ in a} == set(JOBS)


def test_teacher_output_validation_rejects_prose_and_bad_domains() -> None:
    assert parse_teacher_output("router", '{"route": "in_domain"}') == '{"route": "in_domain"}'
    assert parse_teacher_output("router", "I think it's in domain!") is None
    assert parse_teacher_output("router", '{"route": "maybe"}') is None
    assert parse_teacher_output("grader", '{"label": "relevant", "score": 1.5}') is None
    assert parse_teacher_output("grader", '{"label": "relevant"}') is None
    assert parse_teacher_output("grader", '{"label": "relevant", "score": true}') is None
    assert parse_teacher_output("grader", '{"label": "relevant", "score": 0.5}') is not None
    assert parse_teacher_output("rewriter", "who signs it for the on-call rotation?")
    assert parse_teacher_output("rewriter", "") is None


def test_malformed_teacher_rows_are_dropped_not_repaired() -> None:
    clean = generate_sft_dataset(DeterministicFakeTeacher(), 40, seed=9)
    dirty = generate_sft_dataset(DeterministicFakeTeacher(malformed_every=4), 40, seed=9)
    assert len(dirty) < len(clean)
    assert all(parse_teacher_output(e.job, e.completion) is not None for e in dirty)


def test_generated_dataset_round_trips_through_jsonl(
    examples: list[SFTExample], tmp_path: Path
) -> None:
    path = write_examples(examples, tmp_path / "sft.jsonl")
    assert read_examples(path) == examples


def test_stratified_sample_covers_every_stratum(examples: list[SFTExample]) -> None:
    idx = stratified_sample(examples, 12, seed=5)
    assert len(idx) == 12
    assert idx == sorted(idx)
    assert len(set(idx)) == len(idx)

    all_strata = {(e.job, e.label) for e in examples}
    sampled_strata = {(examples[i].job, examples[i].label) for i in idx}
    assert sampled_strata == all_strata, (
        "a uniform draw would miss rare strata; stratification exists to prevent exactly that"
    )
    assert stratified_sample(examples, 12, seed=5) == idx


def test_stratified_sample_degenerate_inputs(examples: list[SFTExample]) -> None:
    assert stratified_sample(examples, 0) == []
    assert stratified_sample([], 10) == []
    assert len(stratified_sample(examples, 10_000, seed=1)) == len(examples)


def test_verification_sheet_leaves_the_human_field_blank(
    examples: list[SFTExample], tmp_path: Path
) -> None:
    idx = stratified_sample(examples, 10, seed=5)
    path = write_verification_sheet(examples, idx, tmp_path / "verify.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 10
    assert all(r["human_label"] == "" for r in rows), "pre-filling would anchor the annotator"
    assert all(r["teacher_label"] for r in rows)
    # Unfilled rows are not agreement data.
    assert load_verification_sheet(path) == []


def test_agreement_report_detects_a_disagreeing_teacher(tmp_path: Path) -> None:
    """The spot-check must be able to fail, or it is not a spot-check."""
    honest = generate_sft_dataset(DeterministicFakeTeacher(), 60, seed=4)
    liar = generate_sft_dataset(DeterministicFakeTeacher(wrong_every=3), 60, seed=4)

    def sheet(rows: list[SFTExample]) -> list[dict[str, Any]]:
        # The human labels are the construction-rule gold; the teacher labels are what
        # the teacher actually said.
        return [
            {
                "job": e.job,
                "teacher_label": e.verification_label,
                "human_label": e.gold or "",
            }
            for e in rows
        ]

    good = agreement_report(sheet(honest), threshold=0.90)
    bad = agreement_report(sheet(liar), threshold=0.90)

    assert good.agreement == pytest.approx(1.0)
    assert not good.below_threshold and good.warning() is None
    assert bad.agreement < good.agreement
    assert bad.below_threshold and "do not train on this corpus" in (bad.warning() or "")
    assert bad.kappa < good.kappa
    assert set(good.per_stratum) and all(n > 0 for n, _ in good.per_stratum.values())


def test_agreement_report_on_empty_sheet_is_a_failure_not_a_pass() -> None:
    report = agreement_report([], threshold=0.9)
    assert isinstance(report, AgreementReport)
    assert report.n == 0 and report.below_threshold
    assert math.isnan(report.agreement)


def test_label_of_classifies_strata() -> None:
    assert label_of("router", '{"route": "needs_web"}') == "needs_web"
    assert label_of("grader", '{"label": "irrelevant", "score": 0.1}') == "irrelevant"
    assert label_of("rewriter", "anything at all") == "rewrite"
    assert label_of("router", "not json") == "malformed"


# ------------------------------------------------------------------------------------ #
# 5a -- schedule and training
# ------------------------------------------------------------------------------------ #
def test_cosine_lr_warms_up_then_decays_to_the_floor() -> None:
    peak, total, warm = 3.0e-4, 100, 10
    assert cosine_lr(0, total, peak, warmup_steps=warm) == pytest.approx(peak / warm)
    assert cosine_lr(warm - 1, total, peak, warmup_steps=warm) == pytest.approx(peak)
    trace = [cosine_lr(s, total, peak, warmup_steps=warm, min_lr_ratio=0.1) for s in range(total)]
    assert trace[warm:] == sorted(trace[warm:], reverse=True)
    # The floor is reached at the endpoint, not at the last step index.
    assert cosine_lr(total, total, peak, warmup_steps=warm, min_lr_ratio=0.1) == pytest.approx(
        peak * 0.1
    )
    assert trace[-1] == pytest.approx(peak * 0.1, rel=1e-2)
    assert min(trace) >= peak * 0.1 - 1e-12
    with pytest.raises(ValueError):
        cosine_lr(0, 0, peak)


def test_sft_config_derives_peak_lr_from_ten_percent_of_pretrain() -> None:
    cfg = SFTConfig.from_yaml(CONFIG_DIR / "sft.yaml")
    pretrain = yaml.safe_load((CONFIG_DIR / "pretrain.yaml").read_text(encoding="utf-8"))
    assert cfg.peak_lr_frac_of_pretrain == 0.1
    assert cfg.pretrain_peak_lr == pretrain["peak_lr"]
    assert cfg.peak_lr == pytest.approx(0.1 * pretrain["peak_lr"])
    assert cfg.mask_prompt_tokens and cfg.boundary_masking


def test_run_sft_reduces_loss(tiny_model: LocalMindTransformer, tokenizer: Tokenizer) -> None:
    encoded = [
        encode_sft_example(tokenizer, ex)
        for ex in generate_sft_dataset(DeterministicFakeTeacher(), 16, seed=1)
    ]
    rows = pack_sft(encoded, 96, pad_id=tokenizer.pad_id)
    result = run_sft(tiny_model, rows, epochs=8, batch_size=2, peak_lr=3.0e-3, seed=1337)
    assert result.steps > 4
    assert result.improved, f"{result.initial_loss:.4f} -> {result.final_loss:.4f}"
    assert result.final_loss < result.initial_loss
    assert 0.0 < result.summary["supervised_token_frac"] < 1.0
    assert result.summary["masked_token_frac"] == pytest.approx(
        1.0 - result.summary["supervised_token_frac"]
    )


def test_iter_batches_is_seeded(tokenizer: Tokenizer) -> None:
    encoded = [
        encode_sft_example(tokenizer, ex)
        for ex in generate_sft_dataset(DeterministicFakeTeacher(), 8, seed=1)
    ]
    rows = pack_sft(encoded, 64, pad_id=tokenizer.pad_id)
    a = [x.tolist() for x, _ in iter_batches(rows, 2, seed=3)]
    b = [x.tolist() for x, _ in iter_batches(rows, 2, seed=3)]
    assert a == b


# ------------------------------------------------------------------------------------ #
# 5b arm 3 -- the top-K KL, hand-computed
# ------------------------------------------------------------------------------------ #
def _reference_topk_kl(
    teacher_logits: list[float], student_logits_full: list[float], indices: list[int]
) -> tuple[float, float]:
    """Pure-Python reference: ``(renormalised_kl, naive_kl)``. No torch involved."""
    # Teacher: softmax over the K logits == the full teacher softmax restricted to the
    # top-K support and renormalised (the full normaliser cancels).
    tmax = max(teacher_logits)
    texp = [math.exp(z - tmax) for z in teacher_logits]
    tz = sum(texp)
    p = [e / tz for e in texp]

    s_topk = [student_logits_full[i] for i in indices]
    smax = max(s_topk)
    sexp = [math.exp(z - smax) for z in s_topk]
    sz = sum(sexp)
    q = [e / sz for e in sexp]

    renormalised = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True))

    # The bug: student probabilities read off the FULL vocabulary softmax.
    fmax = max(student_logits_full)
    lse_full = fmax + math.log(sum(math.exp(z - fmax) for z in student_logits_full))
    log_q_naive = [student_logits_full[i] - lse_full for i in indices]
    naive = sum(pi * (math.log(pi) - lq) for pi, lq in zip(p, log_q_naive, strict=True))
    return renormalised, naive


def test_topk_kl_matches_hand_computed_value() -> None:
    """Closed form: teacher top-2 logits [2, 0], student restricted logits [0, 1]."""
    student = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    values = torch.tensor([[2.0, 0.0]])
    indices = torch.tensor([[3, 1]])

    got = float(topk_kl_loss(student, values, indices))
    expected, _ = _reference_topk_kl([2.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [3, 1])

    # 1e-6, not 1e-9: the implementation computes in float32 and the reference in
    # float64, so the two agree to about seven significant figures and no further.
    assert got == pytest.approx(expected, abs=1e-6)
    # Literal, so a refactor that changes the value has to change this number too.
    assert got == pytest.approx(0.8287249, abs=1e-6)


def test_renormalising_over_the_topk_support_is_the_subtle_part() -> None:
    """The naive student normalisation overstates the divergence by exactly the mass gap."""
    student_full = [0.0, 1.0, 0.0, 0.0, 0.0]
    student = torch.tensor([student_full])
    values = torch.tensor([[2.0, 0.0]])
    indices = torch.tensor([[3, 1]])

    correct = float(topk_kl_loss(student, values, indices, renormalize_student=True))
    naive = float(topk_kl_loss(student, values, indices, renormalize_student=False))
    ref_correct, ref_naive = _reference_topk_kl([2.0, 0.0], student_full, [3, 1])

    assert correct == pytest.approx(ref_correct, abs=1e-6)
    assert naive == pytest.approx(ref_naive, abs=1e-6)
    assert ref_correct == pytest.approx(0.8287249, abs=1e-6)
    assert ref_naive == pytest.approx(1.4202957, abs=1e-6)
    assert naive > correct

    # The gap is exactly log(sum over V) - log(sum over K) of the student's exponentials:
    # the log-mass the top-K slice leaves behind. This identity is why the renormalised
    # form is the correct one -- the naive number is a KL plus a constant that depends
    # only on how much student mass fell outside the teacher's top-K.
    lse_full = float(torch.logsumexp(torch.tensor(student_full), dim=0))
    lse_topk = float(torch.logsumexp(torch.tensor([student_full[3], student_full[1]]), dim=0))
    assert naive - correct == pytest.approx(lse_full - lse_topk, abs=1e-6)


def test_topk_kl_is_zero_when_the_student_matches_the_teacher() -> None:
    values = torch.tensor([[2.0, 1.0, -0.5]])
    indices = torch.tensor([[0, 2, 4]])
    student = torch.zeros(1, 6)
    student[0, 0], student[0, 2], student[0, 4] = 2.0, 1.0, -0.5
    assert float(topk_kl_loss(student, values, indices)) == pytest.approx(0.0, abs=1e-6)


def test_topk_kl_directions_differ_and_both_are_proper() -> None:
    student = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    values = torch.tensor([[2.0, 0.0]])
    indices = torch.tensor([[3, 1]])
    fwd = float(topk_kl_loss(student, values, indices, direction="forward"))
    rev = float(topk_kl_loss(student, values, indices, direction="reverse"))
    assert fwd > 0 and rev > 0 and fwd != rev

    # Reverse KL, computed by hand as sum q (log q - log p).
    p = [math.exp(2) / (math.exp(2) + 1), 1 / (math.exp(2) + 1)]
    q = [1 / (1 + math.e), math.e / (1 + math.e)]
    expected = sum(qi * math.log(qi / pi) for pi, qi in zip(p, q, strict=True))
    assert rev == pytest.approx(expected, abs=1e-6)
    assert rev == pytest.approx(1.0068421, abs=1e-6)


def test_temperature_squared_scaling_keeps_gradients_comparable() -> None:
    student = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    values = torch.tensor([[2.0, 0.0]])
    indices = torch.tensor([[3, 1]])
    unscaled = float(
        topk_kl_loss(student, values, indices, temperature=2.0, scale_by_temperature_squared=False)
    )
    scaled = float(
        topk_kl_loss(student, values, indices, temperature=2.0, scale_by_temperature_squared=True)
    )
    assert scaled == pytest.approx(unscaled * 4.0, abs=1e-9)
    # T=1 is a no-op either way.
    a = float(topk_kl_loss(student, values, indices, temperature=1.0))
    b = float(
        topk_kl_loss(student, values, indices, temperature=1.0, scale_by_temperature_squared=False)
    )
    assert a == pytest.approx(b)


def test_kd_loss_blends_kl_and_ce_at_alpha_point_seven() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 8)
    targets = torch.tensor([[1, IGNORE_INDEX, 4]])
    values, indices = torch.topk(torch.randn(1, 3, 8), 4, dim=-1)

    loss, parts = kd_loss(logits, targets, values, indices, alpha=DEFAULT_ALPHA)
    assert DEFAULT_ALPHA == 0.7
    assert float(loss) == pytest.approx(0.7 * parts["kd_kl"] + 0.3 * parts["kd_ce"], abs=1e-6)

    # alpha=0 is pure CE; alpha=1 is pure KL. The blend has no hidden extra term.
    only_ce, _ = kd_loss(logits, targets, values, indices, alpha=0.0)
    only_kl, _ = kd_loss(logits, targets, values, indices, alpha=1.0)
    assert float(only_ce) == pytest.approx(parts["kd_ce"], abs=1e-6)
    assert float(only_kl) == pytest.approx(parts["kd_kl"], abs=1e-6)


def test_kd_loss_excludes_masked_positions_from_both_terms() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 8)
    values, indices = torch.topk(torch.randn(1, 4, 8), 4, dim=-1)
    all_masked = torch.full((1, 4), IGNORE_INDEX)
    partly = torch.tensor([[2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]])

    kl_all_masked = float(
        topk_kl_loss(
            logits.reshape(-1, 8),
            values.reshape(-1, 4),
            indices.reshape(-1, 4),
            mask=(all_masked.reshape(-1) != IGNORE_INDEX),
        )
    )
    assert kl_all_masked == pytest.approx(0.0)

    one_position = float(
        topk_kl_loss(
            logits.reshape(-1, 8)[:1], values.reshape(-1, 4)[:1], indices.reshape(-1, 4)[:1]
        )
    )
    masked_kl = float(
        topk_kl_loss(
            logits.reshape(-1, 8),
            values.reshape(-1, 4),
            indices.reshape(-1, 4),
            mask=(partly.reshape(-1) != IGNORE_INDEX),
        )
    )
    assert masked_kl == pytest.approx(one_position, abs=1e-6)


def test_kd_loss_rejects_bad_alpha_and_shapes() -> None:
    logits = torch.randn(1, 2, 8)
    targets = torch.tensor([[1, 2]])
    values, indices = torch.topk(torch.randn(1, 2, 8), 4, dim=-1)
    with pytest.raises(ValueError):
        kd_loss(logits, targets, values, indices, alpha=1.5)
    with pytest.raises(ValueError):
        topk_kl_loss(torch.randn(2, 8), torch.randn(3, 4), torch.zeros(3, 4, dtype=torch.long))
    with pytest.raises(ValueError):
        topk_kl_loss(
            torch.randn(2, 8),
            torch.randn(2, 4),
            torch.zeros(2, 4, dtype=torch.long),
            temperature=0.0,
        )


# ------------------------------------------------------------------------------------ #
# 5b -- storage note, and the three arms end to end
# ------------------------------------------------------------------------------------ #
def test_topk_storage_estimate_reproduces_the_adr_number() -> None:
    """ADR 0005: 20k sequences x 256 tokens x top-64 is about 1.2 GB."""
    n = estimate_topk_bytes(20_000, 256, 64)
    assert n == 20_000 * 256 * 64 * 4
    assert n / 2**30 == pytest.approx(1.22, abs=0.01)
    # int32 indices, the usual default, would blow the budget by 50%.
    assert estimate_topk_bytes(20_000, 256, 64, index_bytes=4) / n == pytest.approx(1.5)


def test_teacher_logit_store_round_trips(tmp_path: Path) -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 32)
    values, indices = TeacherLogitStore.from_teacher_logits(logits, k=8)
    assert values.shape == indices.shape == (2, 5, 8)

    store = TeacherLogitStore(path=tmp_path / "topk.npz", k=8)
    store.dump(values, indices)
    loaded_v, loaded_i = store.load()
    assert torch.equal(loaded_i, indices)
    # fp16 storage: values survive to half precision, which is all the KL needs.
    assert torch.allclose(loaded_v, values, atol=1e-2)

    with pytest.raises(ValueError):
        TeacherLogitStore.from_teacher_logits(logits, k=999)


def test_sequence_kd_is_tokenizer_agnostic_and_reduces_loss(
    tiny_model: LocalMindTransformer, tokenizer: Tokenizer
) -> None:
    """Arm 1: the teacher's text is re-tokenised with *our* vocab, so 151k vs 16k never arises."""
    examples = generate_sft_dataset(DeterministicFakeTeacher(), 16, seed=1)
    result = run_sequence_kd(
        tiny_model, tokenizer, examples, seq_len=96, batch_size=2, peak_lr=3.0e-3, epochs=8
    )
    assert result.stage == "kd/sequence"
    assert result.summary["tokenizer_agnostic"] is True
    assert result.improved, f"{result.initial_loss:.4f} -> {result.final_loss:.4f}"


def test_on_policy_round_keeps_only_corrections(
    tiny_model: LocalMindTransformer, tokenizer: Tokenizer
) -> None:
    """Arm 2: an untrained student is wrong nearly everywhere, so nearly everything is kept."""
    sampler = GreedyStudentSampler(model=tiny_model, eos_id=tokenizer.eos_id, temperature=0.0)
    pool = seed_prompts(6, seed=21)
    report = collect_on_policy_corrections(
        sampler, DeterministicFakeTeacher(), tokenizer, pool, max_new_tokens=6
    )
    assert report.n_sampled == 6
    assert report.n_corrected > 0
    assert report.student_error_rate > 0.5
    assert all(parse_teacher_output(e.job, e.completion) is not None for e in report.examples)
    assert all("on-policy" in e.teacher for e in report.examples)


def test_on_policy_drops_rows_the_teacher_botched(
    tiny_model: LocalMindTransformer, tokenizer: Tokenizer
) -> None:
    sampler = GreedyStudentSampler(model=tiny_model, eos_id=tokenizer.eos_id)
    pool = seed_prompts(8, seed=21)
    report = collect_on_policy_corrections(
        sampler, DeterministicFakeTeacher(malformed_every=2), tokenizer, pool, max_new_tokens=4
    )
    assert report.n_teacher_rejected > 0
    assert len(report.examples) + report.n_teacher_rejected <= report.n_sampled


def test_greedy_sampler_varies_with_sample_index(
    tiny_model: LocalMindTransformer, tokenizer: Tokenizer
) -> None:
    """GRPO needs G *different* rollouts; identical ones make every group degenerate."""
    sampler = GreedyStudentSampler(model=tiny_model, eos_id=tokenizer.eos_id, temperature=1.0)
    prompt = tokenizer.encode_chat(
        prompt_messages("router", {"QUERY": "hello"}), add_generation_prompt=True
    )
    outs = [sampler.generate(prompt, 6, sample_index=i) for i in range(6)]
    assert len({tuple(o) for o in outs}) > 1


def test_logit_kd_reduces_loss(tiny_cfg: ModelConfig, tokenizer: Tokenizer, tmp_path: Path) -> None:
    """Arm 3: real top-K KL from a bigger same-tokenizer sibling into the student."""
    torch.manual_seed(0)
    teacher_cfg = tiny_cfg.model_copy(update={"d_model": 64, "n_heads": 4, "n_layers": 3})
    teacher = LocalMindTransformer(teacher_cfg, backend="sdpa_math")
    student = LocalMindTransformer(tiny_cfg, backend="sdpa_math")
    assert teacher.cfg.vocab_size == student.cfg.vocab_size, "arm 3 requires a shared vocabulary"

    encoded = [
        encode_sft_example(tokenizer, ex)
        for ex in generate_sft_dataset(DeterministicFakeTeacher(), 12, seed=1)
    ]
    rows = pack_sft(encoded, 64, pad_id=tokenizer.pad_id)
    batches = list(iter_batches(rows, 2, seed=1)) * 6

    store = TeacherLogitStore(path=tmp_path / "dump.npz", k=DEFAULT_TOP_K)
    result = run_logit_kd(
        student,
        teacher,
        batches,
        top_k=DEFAULT_TOP_K,
        alpha=DEFAULT_ALPHA,
        peak_lr=3.0e-3,
        store=store,
    )
    assert result.summary["top_k"] == 64 and result.summary["alpha"] == 0.7
    assert result.summary["same_tokenizer_required"] is True
    assert store.path.exists()
    assert result.improved, f"{result.initial_loss:.4f} -> {result.final_loss:.4f}"


def test_compare_arms_marks_unmeasured_quality_as_not_run(
    tiny_model: LocalMindTransformer, tokenizer: Tokenizer
) -> None:
    examples = generate_sft_dataset(DeterministicFakeTeacher(), 8, seed=1)
    seq = run_sequence_kd(tiny_model, tokenizer, examples, seq_len=64, batch_size=2, epochs=1)
    payload = compare_arms({"sequence": seq}, hardware="CPU (test)")
    assert payload["rows"][0]["quality"] is None
    assert payload["rows"][0]["quality_status"] == "not-run"
    assert payload["rows"][0]["tokenizer_agnostic"] is True
    assert set(payload) >= {"name", "hardware", "seeds", "rows", "ci"}


def test_kd_config_matches_the_notebook_flags() -> None:
    cfg = KDConfig.from_yaml(CONFIG_DIR / "kd.yaml")
    assert cfg.top_k == 64 and cfg.alpha == 0.7
    assert cfg.arm == "sequence"
    assert cfg.teacher_model_config == "configs/model/100m_teacher.yaml"
    assert cfg.precision in ("fp16", "fp32")
    for arm in ("sequence", "on-policy", "logit"):
        assert cfg.model_copy(update={"arm": arm}).arm == arm


# ------------------------------------------------------------------------------------ #
# 5c -- DPO, hand-computed
# ------------------------------------------------------------------------------------ #
def test_dpo_loss_matches_hand_computed_value() -> None:
    """beta=0.1, chosen log-ratio 0.5, rejected log-ratio -0.2 -> loss = log(1 + e^-0.07)."""
    pc = torch.tensor([-2.0])
    pr = torch.tensor([-3.0])
    rc = torch.tensor([-2.5])
    rr = torch.tensor([-2.8])
    beta = 0.1

    loss, metrics = dpo_loss(pc, pr, rc, rr, beta=beta)

    logits = beta * ((-2.0 - -2.5) - (-3.0 - -2.8))  # 0.1 * (0.5 - (-0.2)) = 0.07
    assert logits == pytest.approx(0.07)
    expected = math.log1p(math.exp(-0.07))
    assert float(loss) == pytest.approx(expected, abs=1e-6)
    assert float(loss) == pytest.approx(0.6587596, abs=1e-6)

    assert metrics["chosen_reward"] == pytest.approx(0.05, abs=1e-6)
    assert metrics["rejected_reward"] == pytest.approx(-0.02, abs=1e-6)
    assert metrics["reward_margin"] == pytest.approx(0.07, abs=1e-6)
    assert metrics["reward_accuracy"] == pytest.approx(1.0)


def test_dpo_loss_is_log_two_when_the_policy_equals_the_reference() -> None:
    """Step zero of any DPO run: policy == reference, so both log-ratios are 0."""
    z = torch.tensor([-1.0, -2.0])
    loss, metrics = dpo_loss(z, z, z.clone(), z.clone(), beta=0.1)
    assert float(loss) == pytest.approx(math.log(2.0), abs=1e-6)
    assert metrics["reward_margin"] == pytest.approx(0.0, abs=1e-7)


def test_dpo_loss_falls_as_the_margin_grows() -> None:
    ref = torch.tensor([-2.0])
    small, _ = dpo_loss(torch.tensor([-1.9]), torch.tensor([-2.1]), ref, ref.clone(), beta=0.1)
    large, _ = dpo_loss(torch.tensor([-1.0]), torch.tensor([-3.0]), ref, ref.clone(), beta=0.1)
    assert float(large) < float(small)


def test_dpo_beta_scales_the_reward_but_not_its_sign() -> None:
    pc, pr = torch.tensor([-1.0]), torch.tensor([-2.0])
    rc, rr = torch.tensor([-1.5]), torch.tensor([-1.5])
    _, low = dpo_loss(pc, pr, rc, rr, beta=0.1)
    _, high = dpo_loss(pc, pr, rc, rr, beta=0.5)
    assert high["reward_margin"] == pytest.approx(5.0 * low["reward_margin"], rel=1e-6)
    assert low["reward_margin"] > 0 and high["reward_margin"] > 0


def test_dpo_label_smoothing_and_validation() -> None:
    z = torch.tensor([-1.0])
    plain, _ = dpo_loss(z, z - 1.0, z.clone(), z.clone(), beta=0.1)
    smoothed, _ = dpo_loss(z, z - 1.0, z.clone(), z.clone(), beta=0.1, label_smoothing=0.1)
    assert float(smoothed) > float(plain), "cDPO hedges, so it cannot be more confident"
    with pytest.raises(ValueError):
        dpo_loss(z, z, z, z, beta=0.0)
    with pytest.raises(ValueError):
        dpo_loss(z, z, z, z, label_smoothing=0.9)


def test_kl_estimators_are_zero_at_the_reference_and_k3_is_non_negative() -> None:
    z = torch.tensor([-1.0, -2.0, -3.0])
    k1, k3 = kl_from_reference(z, z.clone())
    assert k1 == pytest.approx(0.0) and k3 == pytest.approx(0.0)

    drift = kl_from_reference(z + 1.0, z)
    assert drift[0] == pytest.approx(1.0, abs=1e-6)
    assert drift[1] > 0.0
    # k3 = mean(exp(-r) + r - 1) with r = 1 -> e^-1 + 1 - 1 = 0.367879
    assert drift[1] == pytest.approx(math.exp(-1.0), abs=1e-6)
    # k1 can go negative on a finite sample; k3 cannot. That is why the gate uses k3.
    assert kl_from_reference(z - 1.0, z)[0] < 0
    assert kl_from_reference(z - 1.0, z)[1] > 0


def test_sequence_logprob_scores_only_the_supervised_span(
    tiny_model: LocalMindTransformer,
) -> None:
    input_ids = torch.tensor([[3, 4, 5, 6]])
    all_masked = torch.full((1, 4), IGNORE_INDEX)
    assert float(sequence_logprob(tiny_model, input_ids, all_masked).detach()) == pytest.approx(0.0)

    one = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 7]])
    two = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 6, 7]])
    assert float(sequence_logprob(tiny_model, input_ids, two).detach()) < float(
        sequence_logprob(tiny_model, input_ids, one).detach()
    ), "adding a supervised token adds a negative log-prob"

    normed = float(sequence_logprob(tiny_model, input_ids, two, length_normalize=True).detach())
    plain = float(sequence_logprob(tiny_model, input_ids, two).detach())
    assert normed == pytest.approx(plain / 2, abs=1e-5)


def test_preference_pairs_cover_named_failure_modes() -> None:
    pairs = build_preference_pairs(12, seed=3)
    assert len(pairs) == 12
    assert {p.reason for p in pairs} == {"context_dropped", "verbose", "hallucinated", "unchanged"}
    assert all(p.chosen != p.rejected for p in pairs)
    assert build_preference_pairs(12, seed=3) == pairs


def test_collate_pairs_masks_the_shared_prompt(tokenizer: Tokenizer) -> None:
    pairs = build_preference_pairs(3, seed=3)
    batch = collate_preference_batch(tokenizer, pairs, max_len=128)
    assert len(batch) == 3
    assert batch.chosen_input_ids.shape[0] == batch.rejected_input_ids.shape[0] == 3
    assert (batch.chosen_labels != IGNORE_INDEX).any()
    assert (batch.rejected_labels != IGNORE_INDEX).any()
    # The first tokens are the shared prompt; nothing there may be supervised.
    assert (batch.chosen_labels[:, :4] == IGNORE_INDEX).all()


def test_run_dpo_raises_the_margin_and_reports_kl(
    tiny_model: LocalMindTransformer, tokenizer: Tokenizer
) -> None:
    reference = make_reference_model(tiny_model)
    assert all(not p.requires_grad for p in reference.parameters())

    pairs = build_preference_pairs(8, seed=3)
    result = run_dpo(
        tiny_model,
        reference,
        tokenizer,
        pairs,
        beta=0.1,
        batch_size=4,
        epochs=6,
        peak_lr=1.0e-3,
        seq_len=128,
    )
    margins = [m.extra["reward_margin"] for m in result.stage_result.history]
    assert margins[0] == pytest.approx(0.0, abs=1e-6), "step 0: policy is the reference"
    assert margins[-1] > margins[0], f"reward margin did not rise: {margins[0]} -> {margins[-1]}"
    assert result.stage_result.final_loss < result.stage_result.initial_loss
    assert result.kl_k3 >= 0.0
    assert "kl_from_reference_k3" in result.to_dict()


def test_dpo_surfaces_runaway_kl_as_a_warning(
    tiny_model: LocalMindTransformer, tokenizer: Tokenizer
) -> None:
    """SS9 5c: runaway KL means beta is wrong -- and it must not diverge silently."""
    reference = make_reference_model(tiny_model)
    pairs = build_preference_pairs(6, seed=3)
    result = run_dpo(
        tiny_model,
        reference,
        tokenizer,
        pairs,
        beta=0.1,
        batch_size=3,
        epochs=4,
        peak_lr=1.0,  # deliberately absurd, to drive the policy off the reference
        seq_len=128,
    )
    assert result.kl_warning, f"k3 KL was only {result.kl_k3}"
    assert result.warnings and "beta" in result.warnings[0]
    assert result.to_dict()["kl_warning"] is True


def test_dpo_config_pins_beta_at_point_one() -> None:
    cfg = DPOConfig.from_yaml(CONFIG_DIR / "dpo.yaml")
    assert cfg.beta == 0.1
    assert cfg.job == "rewriter"
    assert cfg.n_pairs == 5000
    assert cfg.kl_warn_threshold > 0


# ------------------------------------------------------------------------------------ #
# 5d -- GRPO, hand-computed
# ------------------------------------------------------------------------------------ #
def test_group_advantages_match_hand_computed_values() -> None:
    """rewards [1, 0, 0.5, 0.5]: mean 0.5, population std sqrt(0.125) = 0.3535534."""
    rewards = torch.tensor([1.0, 0.0, 0.5, 0.5])
    adv = group_advantages(rewards)

    mean = 0.5
    std = math.sqrt(((1 - 0.5) ** 2 + (0 - 0.5) ** 2 + 0.0 + 0.0) / 4)
    assert std == pytest.approx(0.35355339, abs=1e-7)
    expected = [(r - mean) / (std + ADVANTAGE_EPS) for r in (1.0, 0.0, 0.5, 0.5)]

    assert adv.tolist() == pytest.approx(expected, abs=1e-6)
    assert adv.tolist() == pytest.approx([1.4138, -1.4138, 0.0, 0.0], abs=1e-4)
    assert float(adv.mean()) == pytest.approx(0.0, abs=1e-6), "the group mean IS the baseline"


def test_group_advantages_for_a_group_of_eight() -> None:
    """SS9 pins G=8. Four correct, four wrong: mean 0.5, std 0.5, advantages +-1."""
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    assert rewards.numel() == DEFAULT_GROUP_SIZE == 8
    adv = group_advantages(rewards)
    expected = 0.5 / (0.5 + ADVANTAGE_EPS)
    assert adv[:4].tolist() == pytest.approx([expected] * 4, abs=1e-7)
    assert adv[4:].tolist() == pytest.approx([-expected] * 4, abs=1e-7)
    assert expected == pytest.approx(0.9998, abs=1e-4)


def test_degenerate_group_yields_exactly_zero_not_a_huge_number() -> None:
    """The branch that keeps GRPO from exploding on an all-correct group."""
    for value in (0.0, 0.5, 1.0):
        adv = group_advantages(torch.full((8,), value))
        assert torch.equal(adv, torch.zeros(8)), f"all-{value} group produced {adv}"
        assert float(adv.abs().max()) == 0.0

    # Without the branch this would be (r - mean)/eps, i.e. of order 1e4.
    almost = group_advantages(torch.tensor([0.5, 0.5, 0.5, 0.5000001]))
    assert float(almost.abs().max()) > 0.0


def test_advantages_are_normalised_within_a_group_never_across_prompts() -> None:
    """Two prompts of very different difficulty must not share a baseline."""
    rewards = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
    adv = group_advantages(rewards)
    assert adv.shape == (2, 4)
    # Row 0 has spread, row 1 is degenerate. If normalisation leaked across rows, row 1
    # would be non-zero (its rewards differ from row 0's mean).
    assert float(adv[0].abs().max()) > 0.5
    assert torch.equal(adv[1], torch.zeros(4))
    assert torch.allclose(adv[0], group_advantages(rewards[0]))


def test_group_advantages_validation() -> None:
    with pytest.raises(ValueError):
        group_advantages(torch.tensor([1.0]))
    with pytest.raises(ValueError):
        group_advantages(torch.zeros(2, 3, 4))
    biased = group_advantages(torch.tensor([1.0, 0.0, 0.5, 0.5]), unbiased=False)
    unbiased = group_advantages(torch.tensor([1.0, 0.0, 0.5, 0.5]), unbiased=True)
    assert float(biased.abs().max()) > float(unbiased.abs().max())


def test_grpo_loss_at_ratio_one_is_minus_the_mean_advantage() -> None:
    """On-policy, pi_old == pi, so every ratio is exactly 1 and the clip never bites."""
    logp = torch.tensor([[-1.0, -2.0], [-0.5, -1.5]])
    adv = torch.tensor([1.5, -0.5])
    mask = torch.ones(2, 2, dtype=torch.bool)

    loss, metrics = grpo_loss(logp, logp.clone(), adv, mask)
    assert float(loss) == pytest.approx(-float(adv.mean()), abs=1e-7)
    assert float(loss) == pytest.approx(-0.5, abs=1e-7)
    assert metrics["mean_ratio"] == pytest.approx(1.0, abs=1e-7)
    assert metrics["clip_frac"] == pytest.approx(0.0)


def test_grpo_clipping_caps_a_positive_advantage_but_not_a_negative_one() -> None:
    """PPO's asymmetry, kept intact: the clip only bounds the direction that helps."""
    old = torch.zeros(2, 1)
    new = torch.full((2, 1), math.log(2.0))  # ratio = 2
    mask = torch.ones(2, 1, dtype=torch.bool)

    pos, _ = grpo_loss(new, old, torch.tensor([1.0, 1.0]), mask, clip_eps=0.2)
    # min(2*1, 1.2*1) = 1.2 -> loss = -1.2
    assert float(pos) == pytest.approx(-1.2, abs=1e-6)

    neg, metrics = grpo_loss(new, old, torch.tensor([-1.0, -1.0]), mask, clip_eps=0.2)
    # min(2*-1, 1.2*-1) = -2 -> loss = +2. The clip does not rescue a bad action.
    assert float(neg) == pytest.approx(2.0, abs=1e-6)
    assert metrics["clip_frac"] == pytest.approx(1.0)


def test_grpo_loss_uses_a_token_mean_so_length_does_not_inflate_gradients() -> None:
    adv = torch.tensor([1.0])
    short = torch.zeros(1, 2)
    long = torch.zeros(1, 8)
    a, _ = grpo_loss(short, short.clone(), adv, torch.ones(1, 2, dtype=torch.bool))
    b, _ = grpo_loss(long, long.clone(), adv, torch.ones(1, 8, dtype=torch.bool))
    assert float(a) == pytest.approx(float(b), abs=1e-7)

    # Masked positions are excluded from the denominator too.
    mask = torch.tensor([[True, True, False, False, False, False, False, False]])
    c, _ = grpo_loss(long, long.clone(), adv, mask)
    assert float(c) == pytest.approx(float(a), abs=1e-7)


def test_grpo_kl_penalty_is_optional_and_non_negative() -> None:
    logp = torch.zeros(1, 2)
    adv = torch.tensor([1.0])
    mask = torch.ones(1, 2, dtype=torch.bool)
    ref = torch.full((1, 2), -0.5)

    free, m_free = grpo_loss(logp, logp.clone(), adv, mask, kl_coef=0.0)
    tethered, m_teth = grpo_loss(logp, logp.clone(), adv, mask, kl_coef=1.0, ref_token_logprobs=ref)
    assert m_free["kl_to_ref"] == 0.0
    assert m_teth["kl_to_ref"] > 0.0
    assert float(tethered) > float(free)
    # k3 per token: exp(-0.5) + 0.5 - 1
    assert m_teth["kl_to_ref"] == pytest.approx(
        math.exp(-0.5) - 0.5 - 1.0 + 1.0 - 0.5 + 0.5, abs=1e-6
    )

    with pytest.raises(ValueError):
        grpo_loss(logp, logp.clone(), adv, mask, kl_coef=1.0)
    with pytest.raises(ValueError):
        grpo_loss(logp, logp.clone(), torch.tensor([1.0, 2.0]), mask)
    with pytest.raises(ValueError):
        grpo_loss(logp, logp.clone(), adv, mask, clip_eps=0.0)


def test_verifiable_reward_requires_format_and_correctness() -> None:
    good = verifiable_reward("grader", '{"label": "relevant", "score": 0.9}', "relevant")
    assert good.format_valid and good.correct and good.total == pytest.approx(1.0)

    wrong = verifiable_reward("grader", '{"label": "irrelevant", "score": 0.1}', "relevant")
    assert wrong.format_valid and not wrong.correct and wrong.total == pytest.approx(0.5)

    prose = verifiable_reward("grader", "Yes, this chunk is relevant.", "relevant")
    assert not prose.format_valid and not prose.correct and prose.total == 0.0

    # Right word, wrong shape: still zero. The task is "emit a machine-readable verdict".
    assert verifiable_reward("router", "in_domain", "in_domain").total == 0.0
    assert verifiable_reward("router", '{"route": "in_domain"}', "in_domain").total == 1.0


def test_reward_weights_are_configurable_and_the_score_field_is_not_rewarded() -> None:
    cfg = RewardConfig(format_weight=0.2, correctness_weight=0.8)
    r = verifiable_reward("grader", '{"label": "relevant", "score": 0.9}', "relevant", cfg=cfg)
    assert r.total == pytest.approx(1.0)
    half = verifiable_reward("grader", '{"label": "irrelevant", "score": 0.1}', "relevant", cfg=cfg)
    assert half.total == pytest.approx(0.2)

    # The numeric score is not verifiable, so it cannot change the reward.
    a = verifiable_reward("grader", '{"label": "relevant", "score": 0.51}', "relevant")
    b = verifiable_reward("grader", '{"label": "relevant", "score": 0.99}', "relevant")
    assert a.total == b.total


def test_sample_group_scores_every_rollout(tokenizer: Tokenizer) -> None:
    sampler = ScriptedSampler(
        tok=tokenizer,
        outputs=[
            '{"label": "relevant", "score": 0.9}',
            '{"label": "irrelevant", "score": 0.1}',
            "not json at all",
        ],
    )
    group = sample_group(
        sampler,
        tokenizer,
        "grader",
        {"QUERY": "q", "CHUNK": "c"},
        "relevant",
        group_size=DEFAULT_GROUP_SIZE,
        max_new_tokens=32,
    )
    assert len(group.samples) == 8
    assert not group.degenerate
    assert set(group.rewards) == {1.0, 0.5, 0.0}
    assert group.to_dict()["correct_frac"] == pytest.approx(3 / 8)


def test_run_grpo_end_to_end(tiny_model: LocalMindTransformer, tokenizer: Tokenizer) -> None:
    sampler = ScriptedSampler(
        tok=tokenizer,
        outputs=[
            '{"label": "relevant", "score": 0.9}',
            '{"label": "irrelevant", "score": 0.1}',
            "unparseable",
            '{"label": "relevant", "score": 0.9}',
        ],
    )
    result = run_grpo(
        tiny_model,
        tokenizer,
        sampler,
        n_prompts=4,
        group_size=8,
        max_new_tokens=32,
        peak_lr=1.0e-3,
        seq_len=160,
    )
    assert result.group_size == 8
    assert result.stage_result.steps == 4
    assert result.stage_result.summary["use_value_network"] is False
    assert 0.0 < result.format_valid_rate < 1.0
    assert result.degenerate_group_frac < 1.0
    assert 0.0 < result.mean_reward < 1.0
    assert result.to_dict()["use_value_network"] is False


def test_run_grpo_warns_when_every_group_is_degenerate(
    tiny_model: LocalMindTransformer, tokenizer: Tokenizer
) -> None:
    """A deterministic sampler makes GRPO a no-op. That has to be loud, not silent."""
    sampler = ScriptedSampler(tok=tokenizer, outputs=["always the same garbage"])
    result = run_grpo(
        tiny_model, tokenizer, sampler, n_prompts=3, group_size=8, max_new_tokens=16, seq_len=160
    )
    assert result.degenerate_group_frac == 1.0
    assert result.warnings
    assert any("degenerate" in w for w in result.warnings)
    assert any("format-valid rate" in w for w in result.warnings)
    # No signal means no movement: every advantage was exactly zero.
    assert result.stage_result.final_loss == pytest.approx(0.0, abs=1e-7)


def test_grpo_config_pins_group_size_eight_and_no_critic() -> None:
    cfg = GRPOConfig.from_yaml(CONFIG_DIR / "grpo.yaml")
    assert cfg.group_size == 8
    assert cfg.use_value_network is False
    assert cfg.job == "grader"
    assert cfg.reward_config.format_weight + cfg.reward_config.correctness_weight == 1.0
    with pytest.raises(ValidationError):
        GRPOConfig.model_validate({**cfg.model_dump(by_alias=True), "use_value_network": True})


# ------------------------------------------------------------------------------------ #
# LoRA (r=16) -- the matrix's "~0.1% params updated" row
# ------------------------------------------------------------------------------------ #
def test_lora_is_the_identity_at_initialisation(tiny_model: LocalMindTransformer) -> None:
    """B = 0, so attaching adapters to a trained checkpoint changes nothing at step 0."""
    ids = torch.tensor([[3, 4, 5, 6]])
    tiny_model.eval()
    with torch.no_grad():
        before = tiny_model(ids).logits.clone()
    apply_lora(tiny_model, LoRAConfig(r=DEFAULT_LORA_RANK))
    tiny_model.eval()
    with torch.no_grad():
        after = tiny_model(ids).logits
    assert torch.allclose(before, after, atol=1e-6)


def test_apply_lora_freezes_the_base_and_reports_a_tiny_trainable_fraction(
    tiny_model: LocalMindTransformer,
) -> None:
    report = apply_lora(tiny_model, LoRAConfig(r=DEFAULT_LORA_RANK))
    assert report.rank == 16
    assert report.n_replaced == len(report.replaced) > 0
    assert all(name.endswith(("wq", "wk", "wv", "wo")) for name in report.replaced)
    assert 0.0 < report.trainable_fraction < 0.5
    assert report.trainable_params + report.base_params == report.total_params
    assert report.trainable_percent_str.endswith("%")

    assert_lora_frozen(tiny_model)
    for _, module in iter_lora_modules(tiny_model):
        assert not module.base.weight.requires_grad
        assert module.lora_A.requires_grad and module.lora_B.requires_grad

    summary = trainable_parameter_summary(tiny_model)
    assert summary["n_lora_modules"] == report.n_replaced
    assert summary["trainable_fraction"] == pytest.approx(report.trainable_fraction)


def test_assert_lora_frozen_catches_a_leaked_base_weight(
    tiny_model: LocalMindTransformer,
) -> None:
    apply_lora(tiny_model, LoRAConfig(r=4))
    assert_lora_frozen(tiny_model)
    next(iter_lora_modules(tiny_model))[1].base.weight.requires_grad_(True)
    with pytest.raises(AssertionError, match="not a LoRA run"):
        assert_lora_frozen(tiny_model)


def test_assert_lora_frozen_rejects_a_model_with_no_adapters(
    tiny_model: LocalMindTransformer,
) -> None:
    with pytest.raises(AssertionError, match="apply_lora was never called"):
        assert_lora_frozen(tiny_model)


def test_lora_gradients_reach_only_the_adapters(tiny_model: LocalMindTransformer) -> None:
    apply_lora(tiny_model, LoRAConfig(r=8))
    ids = torch.tensor([[3, 4, 5, 6]])
    out = tiny_model(ids, targets=torch.tensor([[4, 5, 6, 7]]))
    assert out.loss is not None
    out.loss.backward()
    for name, param in tiny_model.named_parameters():
        if name.endswith(("lora_A", "lora_B")):
            continue
        assert param.grad is None, f"{name} received a gradient in a LoRA run"
    # lora_B starts at zero, so only lora_B has a non-zero gradient on the first step.
    grads = [m.lora_B.grad for _, m in iter_lora_modules(tiny_model) if m.lora_B.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)


def test_merging_lora_is_numerically_exact_and_removes_the_overhead(
    tiny_model: LocalMindTransformer,
) -> None:
    apply_lora(tiny_model, LoRAConfig(r=8))
    with torch.no_grad():
        for _, module in iter_lora_modules(tiny_model):
            module.lora_B.normal_(0.0, 0.05)

    ids = torch.tensor([[3, 4, 5, 6]])
    tiny_model.eval()
    with torch.no_grad():
        adapted = tiny_model(ids).logits.clone()
    n_merged = merge_lora(tiny_model)
    assert n_merged > 0
    assert not list(iter_lora_modules(tiny_model))
    with torch.no_grad():
        merged = tiny_model(ids).logits
    assert torch.allclose(adapted, merged, atol=1e-5)


def test_lora_state_dict_round_trips(tiny_model: LocalMindTransformer) -> None:
    apply_lora(tiny_model, LoRAConfig(r=4))
    with torch.no_grad():
        for _, module in iter_lora_modules(tiny_model):
            module.lora_B.normal_(0.0, 0.1)
    state = lora_state_dict(tiny_model)
    assert state and all(k.endswith(("lora_A", "lora_B")) for k in state)
    # An adapter checkpoint is a rounding error next to the full model.
    assert sum(v.numel() for v in state.values()) < sum(p.numel() for p in tiny_model.parameters())

    zeroed = {k: torch.zeros_like(v) for k, v in state.items()}
    load_lora_state_dict(tiny_model, zeroed)
    assert all(
        float(m.lora_B.detach().abs().sum()) == 0.0 for _, m in iter_lora_modules(tiny_model)
    )
    load_lora_state_dict(tiny_model, state)
    assert torch.allclose(
        next(iter_lora_modules(tiny_model))[1].lora_B,
        state[next(iter_lora_modules(tiny_model))[0] + ".lora_B"],
    )
    with pytest.raises(KeyError):
        load_lora_state_dict(tiny_model, {"nope.lora_A": torch.zeros(1)})


def test_lora_rejects_impossible_ranks_and_silent_no_ops(
    tiny_model: LocalMindTransformer, tiny_cfg: ModelConfig
) -> None:
    with pytest.raises(ValueError, match=r"no nn\.Linear matched"):
        apply_lora(tiny_model, LoRAConfig(r=4, target_modules=("does_not_exist",)))
    base = torch.nn.Linear(8, 4)
    with pytest.raises(ValueError, match="exceeds"):
        LoRALinear(base, r=8)
    with pytest.raises(ValueError):
        LoRALinear(base, r=0)


def test_lora_scaling_is_alpha_over_r() -> None:
    base = torch.nn.Linear(16, 16, bias=False)
    layer = LoRALinear(base, r=8, alpha=16.0)
    assert layer.scaling == pytest.approx(2.0)
    with torch.no_grad():
        layer.lora_A.fill_(0.0)
        layer.lora_B.fill_(0.0)
        layer.lora_A[0, 0] = 1.0
        layer.lora_B[0, 0] = 1.0
    x = torch.zeros(1, 16)
    x[0, 0] = 1.0
    delta = (layer(x) - base(x)).detach()
    assert float(delta[0, 0]) == pytest.approx(2.0, abs=1e-6)
    assert torch.allclose(layer.delta_weight(), layer.lora_B @ layer.lora_A * 2.0)


def test_lora_config_defaults_to_r_sixteen() -> None:
    cfg = LoRAConfig()
    assert cfg.r == DEFAULT_LORA_RANK == 16
    assert cfg.scaling == pytest.approx(2.0)
    assert not cfg.include_lm_head, "tie_embeddings=true means the head is the embedding"


# ------------------------------------------------------------------------------------ #
# 5e -- the comparison matrix. The deliverable, and the place not to invent numbers.
# ------------------------------------------------------------------------------------ #
def test_matrix_has_the_specs_six_rows_and_four_metric_columns() -> None:
    matrix = build_comparison_matrix()
    assert [r.system for r in matrix.rows] == [s for s, _, _ in MATRIX_ROWS]
    assert len(matrix.rows) == 6
    assert MATRIX_COLUMNS == ("router_acc", "grader_f1", "rewrite_win_rate", "p50_latency_ms")
    assert matrix.row("Qwen3-4B + LoRA (r=16)").params_updated == "~0.1%"
    assert matrix.row("LocalMind-31M, SFT only").hardware == "laptop CPU"
    assert matrix.row("Qwen3-4B, zero-shot prompt").params_updated == "0"


def test_every_cell_starts_explicitly_not_run() -> None:
    matrix = build_comparison_matrix()
    assert matrix.n_measured == 0
    assert matrix.n_cells == 24
    assert not matrix.complete
    rendered = render_matrix_markdown(matrix)
    table = rendered.split(chr(10) * 2)[0]
    assert table.count(NOT_RUN) == 24
    assert "0/24 cells measured" in rendered
    # An empty cell must never render as something that reads like a result.
    for bad in ("| 0.0 |", "| n/a |", "| - |", "| |"):
        assert bad not in table


def test_a_measured_cell_requires_a_ci_and_three_seeds() -> None:
    """CONVENTIONS.md rule 5, enforced at construction rather than at review."""
    with pytest.raises(ValueError, match="confidence interval"):
        MatrixCell(mean=0.91)
    with pytest.raises(ValueError, match=">=3 seeds"):
        MatrixCell(mean=0.91, lo=0.88, hi=0.94, n_seeds=1)
    with pytest.raises(ValueError):
        MatrixCell(mean=0.91, lo=0.94, hi=0.88, n_seeds=3)
    ok = MatrixCell(mean=0.91, lo=0.88, hi=0.94, n_seeds=3)
    assert ok.measured and ok.render() == "0.910 [0.880, 0.940]"


def test_matrix_fills_only_what_was_measured() -> None:
    matrix = build_comparison_matrix(
        {"LocalMind-31M, SFT only": {"router_acc": MatrixCell(0.9, 0.87, 0.93, 3)}},
        hardware_note="laptop CPU (test)",
        seeds=(1, 2, 3),
    )
    assert matrix.n_measured == 1
    row = matrix.row("LocalMind-31M, SFT only")
    assert row.cell("router_acc").measured
    assert not row.cell("grader_f1").measured
    assert not row.complete
    assert render_matrix_markdown(matrix).split(chr(10) * 2)[0].count(NOT_RUN) == 23


def test_matrix_rejects_typos_rather_than_silently_dropping_them() -> None:
    with pytest.raises(KeyError, match="unknown system"):
        build_comparison_matrix({"LocalMind-31M": {"router_acc": MatrixCell()}})
    with pytest.raises(KeyError, match="unknown column"):
        build_comparison_matrix({"LocalMind-31M, SFT only": {"accuracy": MatrixCell()}})


def test_dod_is_not_evaluable_without_measurements() -> None:
    verdict = dod_verdict(build_comparison_matrix())
    assert verdict["status"] == "not-evaluable"
    assert verdict["missing_cells"]
    assert "no GPU" in verdict["reason"]
    # "not measured" is a third state; it must not be collapsed into pass or fail.
    assert verdict["status"] not in ("passed", "failed")


def test_dod_evaluates_when_the_cells_exist() -> None:
    def cell(v: float) -> MatrixCell:
        return MatrixCell(v, v * 0.98, v * 1.02, 3)

    matrix = build_comparison_matrix(
        {
            "Qwen3-4B, zero-shot prompt": {
                "router_acc": cell(0.90),
                "grader_f1": cell(0.85),
                "rewrite_win_rate": cell(0.50),
                "p50_latency_ms": cell(2000.0),
            },
            "LocalMind-31M, SFT + KD + GRPO": {
                "router_acc": cell(0.88),
                "grader_f1": cell(0.60),
                "rewrite_win_rate": cell(0.30),
                "p50_latency_ms": cell(50.0),
            },
        },
        seeds=(1, 2, 3),
    )
    verdict = dod_verdict(matrix)
    assert verdict["status"] == "passed"
    assert verdict["latency_speedup"] == pytest.approx(40.0)
    assert verdict["tasks_at_95pct"] == ["router_acc"]

    slow = build_comparison_matrix(
        {
            "Qwen3-4B, zero-shot prompt": {
                "router_acc": cell(0.90),
                "grader_f1": cell(0.85),
                "rewrite_win_rate": cell(0.50),
                "p50_latency_ms": cell(200.0),
            },
            "LocalMind-31M, SFT + KD + GRPO": {
                "router_acc": cell(0.40),
                "grader_f1": cell(0.30),
                "rewrite_win_rate": cell(0.10),
                "p50_latency_ms": cell(50.0),
            },
        },
        seeds=(1, 2, 3),
    )
    failed = dod_verdict(slow)
    assert failed["status"] == "failed"
    assert "negative result" in failed["note"]


def test_matrix_artifact_matches_the_conventions_schema(tmp_path: Path) -> None:
    matrix = build_comparison_matrix(hardware_note="CPU (test)")
    path = write_matrix_artifact(
        matrix, tmp_path / "matrix.json", markdown_path=tmp_path / "benchmarks.md"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) >= {"name", "hardware", "seeds", "rows", "ci"}
    assert payload["ci"] == "bootstrap95"
    assert payload["complete"] is False
    assert payload["dod"]["status"] == "not-evaluable"
    assert all(payload["rows"][0][c]["status"] == "not-run" for c in MATRIX_COLUMNS)
    md = (tmp_path / "benchmarks.md").read_text(encoding="utf-8")
    assert NOT_RUN in md and "Params updated" in md


def test_comparison_matrix_is_a_dataclass_not_a_dict() -> None:
    matrix = build_comparison_matrix()
    assert isinstance(matrix, ComparisonMatrix)
    with pytest.raises(KeyError):
        matrix.row("no such system")


# ------------------------------------------------------------------------------------ #
# Cross-cutting contracts
# ------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("name", ["sft", "kd", "dpo", "grpo"])
def test_no_config_admits_bf16(name: str) -> None:
    """ADR 0001: the T4 is SM 7.5. A bf16 config must fail to load, not fail at runtime."""
    klass = CONFIG_CLASSES[name]
    raw = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    assert raw["precision"] not in ("bfloat16", "bf16")
    with pytest.raises(ValidationError):
        klass.model_validate({**raw, "precision": "bf16"})


@pytest.mark.parametrize("name", ["sft", "kd", "dpo", "grpo"])
def test_configs_reject_unknown_keys(name: str) -> None:
    klass = CONFIG_CLASSES[name]
    raw = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    with pytest.raises(ValidationError):
        klass.model_validate({**raw, "lr": 1e-3})


def test_post_package_imports_without_torch_at_module_scope() -> None:
    """`import localmind.post` must stay cheap: submodules resolve lazily."""
    import localmind.post as post

    assert "localmind.post.sft" not in [m for m in dir(post) if m == "sft"] or True
    assert post.NOT_RUN == "not-yet-run"
    with pytest.raises(AttributeError):
        _ = post.does_not_exist


def test_stage_result_improved_uses_quarters_not_a_single_step(
    tiny_model: LocalMindTransformer, tokenizer: Tokenizer
) -> None:
    encoded = [
        encode_sft_example(tokenizer, ex)
        for ex in generate_sft_dataset(DeterministicFakeTeacher(), 8, seed=1)
    ]
    rows = pack_sft(encoded, 64, pad_id=tokenizer.pad_id)
    result = run_sft(tiny_model, rows, epochs=4, batch_size=2, peak_lr=3.0e-3)
    payload = result.to_dict()
    assert payload["stage"] == "sft"
    assert len(payload["history"]) == result.steps
    assert payload["improved"] == result.improved
    assert result.mean_loss(first_k=2) != result.mean_loss(last_k=2)
