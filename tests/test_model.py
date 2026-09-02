"""Phase 2 model tests.

The three DoD checks from implementation.md §6 are:

* `test_dod_1_count_params_31m`      -- `count_params(cfg) == 30_932_992`
* `test_dod_2_backends_agree_fp32`   -- the three attention backends agree to 1e-3
* `test_dod_3_overfit_100_sequences` -- a ~5M model overfits 100 sequences to
  near-zero loss in under 2 minutes on CPU (marked `slow`)

Everything else guards the four traps the spec calls out (RoPE on V, missing causal
mask, `expand` instead of `repeat_interleave`, a shared RoPE cache) and the fp16
numerics that a T4 makes load-bearing.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from localmind.model import (
    ATTN_BACKENDS,
    CausalSelfAttention,
    LocalMindTransformer,
    ModelConfig,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    apply_rope,
    count_module_params,
    count_norm_params,
    count_params,
    init_weights,
    kv_cache_bytes_per_token,
    non_embedding_params,
    observed_sdpa_backend,
    repeat_kv,
    residual_scale,
    rotate_half,
    run_benchmarks,
    swiglu_hidden_dim,
)
from localmind.model.attention import FUSED_KERNELS
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "model"
CONFIG_PATHS = sorted(CONFIG_DIR.glob("*.yaml"))
CONFIG_IDS = [p.stem for p in CONFIG_PATHS]

# Small config used everywhere a full 31M forward would be wasteful.
TINY = ModelConfig(
    name="tiny",
    vocab_size=97,
    d_model=32,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    head_dim=8,
    ffn_hidden=64,
    max_seq_len=32,
    rope_theta=10000.0,
    qk_norm=True,
    bias=False,
    tie_embeddings=True,
    z_loss=1e-4,
    attn_dropout=0.0,
    init_std=0.02,
)


@pytest.fixture(scope="module")
def cfg_31m() -> ModelConfig:
    return ModelConfig.from_yaml(CONFIG_DIR / "31m.yaml")


# =====================================================================================
# Config + parameter accounting
# =====================================================================================
def test_all_three_configs_exist() -> None:
    assert CONFIG_IDS == ["100m_teacher", "12m_proxy", "31m"]


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=CONFIG_IDS)
def test_count_params_matches_expected_params_both_modes(path: Path) -> None:
    """DoD #1 generalised: assert BOTH counting modes for all three configs.

    `expected_params` in the YAML is the `include_norms=False` value (the number the
    spec's DoD asserts). The honest total adds `count_norm_params`.
    """
    cfg = ModelConfig.from_yaml(path)
    assert cfg.expected_params is not None
    excl = count_params(cfg, include_norms=False)
    incl = count_params(cfg, include_norms=True)
    assert excl == cfg.expected_params
    assert incl == cfg.expected_params + count_norm_params(cfg)
    assert incl > excl  # norms are real parameters even though the DoD number omits them


def test_dod_1_count_params_31m(cfg_31m: ModelConfig) -> None:
    """The literal §6 DoD assert, plus the honest total and the corrected norm count.

    The spec's parameter table says "Norms ~33,000". That is wrong: the true figure is
    9,728. See the docstring of `localmind.model.config.count_params`.
    """
    assert count_params(cfg_31m) == 30_932_992
    assert count_params(cfg_31m, include_norms=False) == 30_932_992
    assert count_params(cfg_31m, include_norms=True) == 30_942_720
    assert count_norm_params(cfg_31m) == 9_728
    assert count_norm_params(cfg_31m) != 33_000

    # And the table's other rows, component by component.
    assert cfg_31m.vocab_size * cfg_31m.d_model == 8_388_608
    attn = cfg_31m.n_layers * (
        2 * cfg_31m.d_model * cfg_31m.q_dim + 2 * cfg_31m.d_model * cfg_31m.kv_dim
    )
    ffn = cfg_31m.n_layers * 3 * cfg_31m.d_model * cfg_31m.ffn_hidden
    assert attn == 5_242_880
    assert ffn == 17_301_504
    assert 8_388_608 + attn + ffn == 30_932_992
    assert non_embedding_params(cfg_31m) == 22_544_384  # the "22.5M non-embedding" claim


@pytest.mark.parametrize("stem", ["12m_proxy", "31m"])
def test_analytic_count_matches_instantiated_module(stem: str) -> None:
    """The hand-derived formula and the real `nn.Module` must not drift apart."""
    cfg = ModelConfig.from_yaml(CONFIG_DIR / f"{stem}.yaml")
    model = LocalMindTransformer(cfg)
    assert model.num_params(include_norms=False) == count_params(cfg, include_norms=False)
    assert model.num_params(include_norms=True) == count_params(cfg, include_norms=True)
    assert count_module_params(model, include_norms=True) == count_params(cfg, include_norms=True)
    assert model.num_params(include_norms=True, include_embedding=False) == non_embedding_params(
        cfg, include_norms=True
    )


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=CONFIG_IDS)
def test_swiglu_hidden_dim_reproduces_config(path: Path) -> None:
    cfg = ModelConfig.from_yaml(path)
    assert swiglu_hidden_dim(cfg.d_model) == cfg.ffn_hidden
    assert cfg.ffn_hidden % 128 == 0


def test_config_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig.model_validate(TINY.model_dump() | {"n_kv_heads": 3})
    with pytest.raises(ValueError, match="must equal d_model"):
        ModelConfig.model_validate(TINY.model_dump() | {"d_model": 64})
    with pytest.raises(ValueError, match="even"):
        ModelConfig.model_validate(TINY.model_dump() | {"head_dim": 9, "d_model": 36, "n_heads": 4})


def test_config_forbids_unknown_keys() -> None:
    with pytest.raises(ValueError):
        ModelConfig.model_validate(TINY.model_dump() | {"activation_fn": "gelu"})


def test_config_is_frozen_and_yaml_roundtrips(cfg_31m: ModelConfig) -> None:
    with pytest.raises(ValueError):
        cfg_31m.d_model = 1024  # type: ignore[misc]
    assert cfg_31m.name == "LocalMind-31M"
    assert cfg_31m.n_rep == 4  # GQA 4:1


def test_no_bf16_and_no_flash_attention_2_in_model_source() -> None:
    """CONVENTIONS.md rules 3 and 4, enforced mechanically.

    Turing (SM 7.5) has no bf16 tensor cores and cannot run FlashAttention-2.
    """
    banned = ("torch.bfloat16", "bfloat16)", "flash_attn", "flash_attn_2", "FlashAttention2")
    for py in (REPO_ROOT / "localmind" / "model").glob("*.py"):
        src = py.read_text(encoding="utf-8")
        for token in banned:
            assert token not in src, f"{py.name} contains banned token {token!r}"


# =====================================================================================
# RMSNorm
# =====================================================================================
def test_rmsnorm_matches_definition_and_skips_mean_subtraction() -> None:
    norm = RMSNorm(16)
    x = torch.randn(3, 5, 16) + 4.0  # deliberately non-zero mean
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)
    torch.testing.assert_close(norm(x), expected, rtol=1e-5, atol=1e-6)
    # LayerNorm would zero the mean; RMSNorm must not.
    assert norm(x).mean(-1).abs().min() > 0.1
    # No bias, exactly one learnable tensor.
    params = list(norm.parameters())
    assert len(params) == 1 and params[0].shape == (16,)


def test_rmsnorm_fp32_reduction_survives_fp16_overflow() -> None:
    """The fp32 cast is load-bearing (§3.2), not a decoration.

    In fp16, `x.pow(2)` overflows at |x| ~ 256. A naive fp16 reduction produces inf and
    then NaN; the fp32 reduction returns finite values.
    """
    norm = RMSNorm(64).half()
    x = torch.full((2, 4, 64), 300.0, dtype=torch.float16)
    assert torch.isinf(x.pow(2).mean(-1)).all(), "premise: fp16 reduction really does overflow"
    out = norm(x)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()


def test_rmsnorm_preserves_input_dtype() -> None:
    norm = RMSNorm(8)
    for dtype in (torch.float32, torch.float16):
        assert norm.to(dtype)(torch.randn(2, 8, dtype=dtype)).dtype == dtype


# =====================================================================================
# RoPE
# =====================================================================================
def test_rotate_half() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    torch.testing.assert_close(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def _rope_tables(head_dim: int, max_len: int, theta: float, dtype: torch.dtype):
    """The module's table formula, at an arbitrary precision (used to separate the
    mathematical property from fp32 table round-off)."""
    exponent = torch.arange(0, head_dim, 2, dtype=dtype) / head_dim
    inv_freq = 1.0 / (theta**exponent)
    t = torch.arange(max_len, dtype=dtype)
    emb = torch.cat((torch.outer(t, inv_freq),) * 2, dim=-1)
    return emb.cos()[None, None], emb.sin()[None, None]


@pytest.mark.parametrize("table", ["float64", "module_fp32"])
def test_rope_attention_score_depends_only_on_relative_position(table: str) -> None:
    """The spec's RoPE test: score(i, j) is a function of (i - j) alone.

    Checked twice. In float64 the identity is exact to 1e-12, which proves the
    rotate-half algebra. With the module's production fp32 tables the residual is
    ~1e-7 relative -- table round-off, not a modelling error.
    """
    head_dim, max_len, theta = 32, 128, 10000.0
    if table == "float64":
        cos, sin = _rope_tables(head_dim, max_len, theta, torch.float64)
        tol = 1e-12
    else:
        rope = RotaryEmbedding(head_dim, max_len, theta=theta).double()
        cos, sin = rope(max_len)
        tol = 1e-6

    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)
    k = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)

    def score(i: int, j: int) -> float:
        """The attention logit for a query at position i against a key at position j."""
        q_rot, _ = apply_rope(q, q, cos[:, :, i : i + 1], sin[:, :, i : i + 1])
        _, k_rot = apply_rope(k, k, cos[:, :, j : j + 1], sin[:, :, j : j + 1])
        return float((q_rot * k_rot).sum())

    for delta in (0, 1, 7, 40):
        ref = score(delta + 3, 3)
        for base in (0, 5, 17, 60):
            got = score(delta + base, base)
            assert math.isclose(got, ref, rel_tol=tol, abs_tol=tol), (
                f"score({delta + base},{base})={got} != score({delta + 3},3)={ref}"
            )

    # Different relative offsets really do give different scores (the test has teeth).
    assert not math.isclose(score(10, 3), score(20, 3), rel_tol=1e-4)


def test_rope_is_norm_preserving() -> None:
    rope = RotaryEmbedding(16, 64)
    cos, sin = rope(64)
    q = torch.randn(2, 3, 64, 16)
    q_rot, _ = apply_rope(q, q.clone(), cos, sin)
    torch.testing.assert_close(q_rot.norm(dim=-1), q.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_rope_is_not_applied_to_v() -> None:
    """Trap #1. With Wq = Wk = 0 attention is uniform, so the output is the running
    mean of V. If RoPE had touched V the running mean would be of *rotated* V and this
    comparison would fail."""
    cfg = TINY
    attn = CausalSelfAttention(cfg, backend="naive").eval()
    with torch.no_grad():
        attn.wq.weight.zero_()
        attn.wk.weight.zero_()
    rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
    b, t = 2, 12
    x = torch.randn(b, t, cfg.d_model)
    out, _ = attn(x, rope)

    v = attn.wv(x).view(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
    v_rep = repeat_kv(v, cfg.n_rep)
    denom = torch.arange(1, t + 1, dtype=v_rep.dtype).view(1, 1, t, 1)
    running_mean = v_rep.cumsum(dim=2) / denom
    expected = attn.wo(running_mean.transpose(1, 2).reshape(b, t, cfg.q_dim))
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)


def test_rope_cache_is_never_shared_across_max_seq_len() -> None:
    """Trap #4. Asking a 128-position table for position 200 must raise, not wrap."""
    short = RotaryEmbedding(16, 128)
    long = RotaryEmbedding(16, 512)
    assert short.cos_cached.shape[2] == 128
    assert long.cos_cached.shape[2] == 512
    with pytest.raises(ValueError, match="max_seq_len"):
        short(200)
    with pytest.raises(ValueError, match="max_seq_len"):
        short(64, offset=100)
    # Overlapping positions agree, so the guard is about bounds, not about the values.
    torch.testing.assert_close(short(128)[0], long(128)[0])
    # A model owns its own table sized to its own max_seq_len.
    model = LocalMindTransformer(TINY)
    assert model.rope.max_seq_len == TINY.max_seq_len


def test_rope_offset_matches_absolute_positions() -> None:
    rope = RotaryEmbedding(16, 64)
    cos_all, sin_all = rope(64)
    cos_tail, sin_tail = rope(8, offset=40)
    torch.testing.assert_close(cos_tail, cos_all[:, :, 40:48])
    torch.testing.assert_close(sin_tail, sin_all[:, :, 40:48])


# =====================================================================================
# Attention: backends, causality, GQA, QK-norm, KV cache
# =====================================================================================
def _tiny_model(**overrides: object) -> LocalMindTransformer:
    cfg = ModelConfig.model_validate(TINY.model_dump() | overrides)
    torch.manual_seed(0)
    model = LocalMindTransformer(cfg)
    init_weights(model, cfg, seed=0)
    return model.eval()


def test_dod_2_backends_agree_fp32() -> None:
    """DoD #2: `naive`, `sdpa_math` and `sdpa_efficient` agree to 1e-3 in fp32."""
    torch.manual_seed(0)
    ids = torch.randint(0, TINY.vocab_size, (3, 16))
    outputs: dict[str, torch.Tensor] = {}
    for backend in ATTN_BACKENDS:
        model = _tiny_model()
        model.set_backend(backend)
        with torch.no_grad():
            outputs[backend] = model(ids).logits
    for a, b in itertools.combinations(ATTN_BACKENDS, 2):
        diff = (outputs[a] - outputs[b]).abs().max().item()
        assert diff < 1e-3, f"{a} vs {b} differ by {diff}"


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_backends_agree_for_every_gqa_ratio(n_kv_heads: int) -> None:
    """MQA (1), GQA (2) and MHA (4) must all give identical results across backends."""
    cfg = ModelConfig.model_validate(TINY.model_dump() | {"n_kv_heads": n_kv_heads})
    rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
    torch.manual_seed(1)
    x = torch.randn(2, 13, cfg.d_model)
    torch.manual_seed(2)
    attn = CausalSelfAttention(cfg).eval()
    outs = {}
    with torch.no_grad():
        for backend in ATTN_BACKENDS:
            outs[backend], _ = attn(x, rope, backend=backend)
    for a, b in itertools.combinations(ATTN_BACKENDS, 2):
        assert (outs[a] - outs[b]).abs().max().item() < 1e-3


@pytest.mark.parametrize("backend", ATTN_BACKENDS)
def test_causal_mask_is_present_in_every_backend(backend: str) -> None:
    """Trap #2. Editing the future must not change the past.

    A missing causal mask makes the training loss look *better*, not worse, which is
    why this needs an explicit test rather than eyeballing a curve.
    """
    model = _tiny_model()
    model.set_backend(backend)  # type: ignore[arg-type]
    torch.manual_seed(0)
    ids = torch.randint(0, TINY.vocab_size, (2, 16))
    cut = 9
    edited = ids.clone()
    edited[:, cut:] = (edited[:, cut:] + 7) % TINY.vocab_size
    with torch.no_grad():
        a = model(ids).logits
        b = model(edited).logits
    torch.testing.assert_close(a[:, :cut], b[:, :cut], rtol=1e-4, atol=1e-5)
    assert (a[:, cut:] - b[:, cut:]).abs().max() > 1e-4  # the edit did something


def test_naive_path_output_differs_from_unmasked_attention() -> None:
    """Direct proof the naive path masks: it must not equal bidirectional attention."""
    cfg = TINY
    torch.manual_seed(0)
    attn = CausalSelfAttention(cfg, backend="naive").eval()
    rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
    b, t = 1, 10
    x = torch.randn(b, t, cfg.d_model)
    with torch.no_grad():
        masked, _ = attn(x, rope)

        q = attn.wq(x).view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = attn.wk(x).view(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = attn.wv(x).view(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        assert attn.q_norm is not None and attn.k_norm is not None
        q, k = attn.q_norm(q), attn.k_norm(k)
        cos, sin = rope(t)
        q, k = apply_rope(q, k, cos, sin)
        k, v = repeat_kv(k, cfg.n_rep), repeat_kv(v, cfg.n_rep)
        att = torch.softmax((q @ k.transpose(-2, -1)) * cfg.head_dim**-0.5, dim=-1)
        unmasked = attn.wo((att @ v).transpose(1, 2).reshape(b, t, cfg.q_dim))
    assert (masked - unmasked).abs().max() > 1e-3
    # The LAST position already attends to every token, so masked and unmasked agree
    # there -- the difference is entirely at the earlier positions, which is exactly
    # the information leak a missing mask would introduce.
    torch.testing.assert_close(masked[:, -1], unmasked[:, -1], rtol=1e-4, atol=1e-5)
    assert (masked[:, 0] - unmasked[:, 0]).abs().max() > 1e-3


def test_repeat_kv_maps_heads_correctly_and_is_contiguous() -> None:
    """Trap #3: `repeat_interleave`, not `expand`.

    `expand` yields a zero-stride view; the fused SDPA kernels want real, contiguous
    memory. Query head h must read KV head h // n_rep.
    """
    x = torch.randn(2, 2, 5, 4)
    out = repeat_kv(x, 3)
    assert out.shape == (2, 6, 5, 4)
    assert out.is_contiguous()
    assert 0 not in out.stride(), "zero stride means this is an expand(), not a repeat"
    assert out.data_ptr() != x.data_ptr()
    for h in range(6):
        torch.testing.assert_close(out[:, h], x[:, h // 3])
    assert repeat_kv(x, 1) is x  # MHA: no copy at all


def test_qk_norm_is_applied_per_head_and_changes_the_output() -> None:
    on = ModelConfig.model_validate(TINY.model_dump() | {"qk_norm": True})
    off = ModelConfig.model_validate(TINY.model_dump() | {"qk_norm": False})
    a = CausalSelfAttention(on)
    b = CausalSelfAttention(off)
    assert a.q_norm is not None and a.k_norm is not None
    assert b.q_norm is None and b.k_norm is None
    # Normalisation is over head_dim, i.e. per head -- not over the whole projection.
    assert a.q_norm.w.shape == (on.head_dim,)
    assert a.k_norm.w.shape == (on.head_dim,)

    b.load_state_dict({k: v for k, v in a.state_dict().items() if "norm" not in k})
    rope = RotaryEmbedding(on.head_dim, on.max_seq_len, on.rope_theta)
    x = torch.randn(1, 8, on.d_model) * 5.0
    with torch.no_grad():
        assert (a.eval()(x, rope)[0] - b.eval()(x, rope)[0]).abs().max() > 1e-3
    assert count_norm_params(on) - count_norm_params(off) == on.n_layers * 2 * on.head_dim


@pytest.mark.parametrize("backend", ATTN_BACKENDS)
def test_kv_cache_incremental_decode_matches_full_forward(backend: str) -> None:
    """Token-at-a-time decoding with a cache must equal one full forward pass.

    This is also a RoPE-offset test: get the offset wrong and positions shift.
    """
    model = _tiny_model()
    model.set_backend(backend)  # type: ignore[arg-type]
    torch.manual_seed(0)
    ids = torch.randint(0, TINY.vocab_size, (2, 12))
    with torch.no_grad():
        full = model(ids).logits
        # prefill on a prefix, then decode the rest one token at a time
        out = model(ids[:, :5], use_cache=True)
        caches = out.kv_caches
        pieces = [out.logits]
        for i in range(5, ids.shape[1]):
            step = model(ids[:, i : i + 1], past_kvs=caches, use_cache=True)
            caches = step.kv_caches
            pieces.append(step.logits)
        incremental = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full, incremental, rtol=1e-3, atol=1e-3)


def test_invalid_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        CausalSelfAttention(TINY, backend="flash2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="backend must be one of"):
        _tiny_model().set_backend("flash2")  # type: ignore[arg-type]


def test_sdpa_backend_that_actually_fired_is_asserted() -> None:
    """Pinning a backend and *hoping* is not the same as verifying it (§3.2)."""
    cfg = TINY
    rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
    x = torch.randn(1, 16, cfg.d_model)

    def make(backend: str):
        attn = CausalSelfAttention(cfg, backend=backend).eval()  # type: ignore[arg-type]

        def run() -> None:
            with torch.no_grad():
                attn(x, rope)

        return run

    assert observed_sdpa_backend(make("sdpa_math")) == "math"
    efficient = observed_sdpa_backend(make("sdpa_efficient"))
    assert efficient in FUSED_KERNELS, (
        f"sdpa_efficient fell back to {efficient!r}; it must dispatch to a fused, "
        "tile-streaming kernel (EFFICIENT_ATTENTION on CUDA, the ATen tiled CPU "
        "kernel on CPU)"
    )
    assert efficient != "math"


@pytest.mark.gpu
def test_efficient_attention_backend_on_cuda() -> None:
    """On a real T4 the memory-efficient CUDA kernel must be the one that fires."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cfg = TINY
    rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta).cuda()
    attn = CausalSelfAttention(cfg, backend="sdpa_efficient").cuda().eval()
    x = torch.randn(1, 16, cfg.d_model, device="cuda")

    def run() -> None:
        with torch.no_grad():
            attn(x, rope)

    assert observed_sdpa_backend(run) == "efficient"


# =====================================================================================
# SwiGLU
# =====================================================================================
def test_swiglu_matches_definition_and_has_three_matrices() -> None:
    ffn = SwiGLU(TINY)
    x = torch.randn(2, 5, TINY.d_model)
    expected = ffn.down(F.silu(ffn.gate(x)) * ffn.up(x))
    torch.testing.assert_close(ffn(x), expected)
    linears = [m for m in ffn.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 3
    assert all(m.bias is None for m in linears)  # bias: false
    assert sum(p.numel() for p in ffn.parameters()) == 3 * TINY.d_model * TINY.ffn_hidden


# =====================================================================================
# Init
# =====================================================================================
def test_init_scales_residual_output_projections() -> None:
    cfg = ModelConfig.from_yaml(CONFIG_DIR / "31m.yaml")
    model = LocalMindTransformer(cfg)
    init_weights(model, cfg, seed=0)

    expected_out_std = cfg.init_std * residual_scale(cfg.n_layers)
    assert math.isclose(residual_scale(8), 1 / math.sqrt(16))

    for name, param in model.named_parameters():
        std = param.detach().float().std().item()
        if name.endswith("attn.wo.weight") or name.endswith("ffn.down.weight"):
            assert math.isclose(std, expected_out_std, rel_tol=0.05), name
        elif name.endswith(".w"):  # RMSNorm scale
            assert torch.all(param == 1.0), name
        else:
            assert math.isclose(std, cfg.init_std, rel_tol=0.05), name


def test_init_is_bit_exact_for_a_given_seed() -> None:
    cfg = TINY
    a, b, c = LocalMindTransformer(cfg), LocalMindTransformer(cfg), LocalMindTransformer(cfg)
    init_weights(a, cfg, seed=1234)
    init_weights(b, cfg, seed=1234)
    init_weights(c, cfg, seed=4321)
    for (na, pa), (_, pb), (_, pc) in zip(
        a.named_parameters(), b.named_parameters(), c.named_parameters(), strict=True
    ):
        assert torch.equal(pa, pb), f"{na} is not reproducible"
        if not na.endswith(".w"):
            assert not torch.equal(pa, pc), f"{na} ignored the seed"


def test_mup_is_documented_as_future_work() -> None:
    """muP and the MoE/MLA/sliding-window stretch goals are deliberately not built."""
    src = (REPO_ROOT / "localmind" / "model" / "init.py").read_text(encoding="utf-8")
    assert "muP" in src and "FUTURE WORK" in src
    for name in ("MoE", "MLA", "sliding-window"):
        assert name in src


# =====================================================================================
# Transformer
# =====================================================================================
def test_forward_shapes_and_loss_components() -> None:
    model = _tiny_model()
    ids = torch.randint(0, TINY.vocab_size, (3, 11))
    out = model(ids)
    assert out.logits.shape == (3, 11, TINY.vocab_size)
    assert out.loss is None

    out = model(ids, ids)
    assert out.loss is not None and out.ce_loss is not None and out.z_loss is not None
    assert out.loss.dtype == torch.float32
    torch.testing.assert_close(out.loss, out.ce_loss + TINY.z_loss * out.z_loss)
    # An untrained model should sit in the neighbourhood of ln(vocab_size): the logits
    # are near-uniform at init, so the loss is close to the uniform-prior entropy.
    assert abs(out.ce_loss.item() - math.log(TINY.vocab_size)) < 1.0


def test_z_loss_is_disabled_when_the_coefficient_is_zero() -> None:
    model = _tiny_model(z_loss=0.0)
    ids = torch.randint(0, model.cfg.vocab_size, (2, 8))
    out = model(ids, ids)
    assert out.loss is not None and out.ce_loss is not None
    torch.testing.assert_close(out.loss, out.ce_loss)


def test_ignore_index_masks_positions_out_of_the_loss() -> None:
    model = _tiny_model()
    ids = torch.randint(0, TINY.vocab_size, (2, 8))
    targets = ids.clone()
    targets[:, 4:] = -100
    out = model(ids, targets)
    assert out.ce_loss is not None and torch.isfinite(out.ce_loss)


def test_embeddings_are_tied() -> None:
    tied = _tiny_model(tie_embeddings=True)
    assert tied.lm_head.weight.data_ptr() == tied.tok_emb.weight.data_ptr()
    untied = _tiny_model(tie_embeddings=False)
    assert untied.lm_head.weight.data_ptr() != untied.tok_emb.weight.data_ptr()
    assert count_params(untied.cfg) - count_params(tied.cfg) == TINY.vocab_size * TINY.d_model


def test_gradients_reach_every_parameter() -> None:
    model = _tiny_model()
    model.train()
    ids = torch.randint(0, TINY.vocab_size, (2, 8))
    out = model(ids, ids)
    assert out.loss is not None
    out.loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradients"
        assert p.grad.abs().sum() > 0, f"{name} has an all-zero gradient"


def test_forward_rejects_sequences_longer_than_max_seq_len() -> None:
    model = _tiny_model()
    ids = torch.randint(0, TINY.vocab_size, (1, TINY.max_seq_len + 1))
    with pytest.raises(ValueError, match="max_seq_len"):
        model(ids)


def test_model_output_is_tuple_unpackable() -> None:
    model = _tiny_model()
    logits, loss, ce, z, caches = model(torch.randint(0, TINY.vocab_size, (1, 4)))
    assert logits.shape[-1] == TINY.vocab_size
    assert loss is None and ce is None and z is None and caches is None


# =====================================================================================
# Benchmarks
# =====================================================================================
def test_kv_cache_bytes_per_token_reproduces_the_interview_number(cfg_31m: ModelConfig) -> None:
    """§6: MHA 16 KB/token vs GQA(8q/2kv) 4 KB/token, computed -- never hardcoded."""
    mha = kv_cache_bytes_per_token(cfg_31m, n_kv_heads=cfg_31m.n_heads, dtype_bytes=2)
    gqa = kv_cache_bytes_per_token(cfg_31m, dtype_bytes=2)
    assert mha == 16 * 1024
    assert gqa == 4 * 1024
    assert mha // gqa == 4 == cfg_31m.n_rep
    # At 2048 context: 32 MB vs 8 MB per sequence.
    assert mha * 2048 == 32 * 1024**2
    assert gqa * 2048 == 8 * 1024**2
    # MQA would be the floor.
    assert kv_cache_bytes_per_token(cfg_31m, n_kv_heads=1) == 2 * 1024
    with pytest.raises(ValueError):
        kv_cache_bytes_per_token(cfg_31m, n_kv_heads=3)


def test_flops_accounting_is_sane(cfg_31m: ModelConfig) -> None:
    from localmind.model import flops_per_forward, flops_per_step

    fwd = flops_per_forward(cfg_31m, 1, 512)
    assert flops_per_step(cfg_31m, 1, 512) == 3 * fwd
    # The quadratic attention term must grow 4x when the sequence doubles.
    quad = lambda t: 4 * cfg_31m.n_layers * t * t * cfg_31m.d_model  # noqa: E731
    assert quad(1024) == 4 * quad(512)
    assert flops_per_forward(cfg_31m, 2, 512) == 2 * fwd


@pytest.mark.slow
def test_benchmark_artifact_matches_conventions_schema(tmp_path: Path) -> None:
    out = tmp_path / "model.json"
    payload = run_benchmarks(
        config_path=CONFIG_DIR / "31m.yaml",
        seq_lens=(64, 128),
        seeds=(0, 1),
        throughput_batch=1,
        throughput_seq=64,
        out_path=out,
        docs_path=None,
    )
    assert set(payload) >= {"name", "hardware", "seeds", "rows", "ci"}
    assert payload["ci"] == "bootstrap95"
    assert "CPU" in payload["hardware"]
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(payload))

    kinds = {r["bench"] for r in payload["rows"]}
    assert kinds == {"kv_cache", "attn_backend", "throughput"}

    kv = {r["variant"]: r for r in payload["rows"] if r["bench"] == "kv_cache"}
    assert kv["MHA"]["kb_per_token"] == 16.0
    assert kv["GQA(8q/2kv)"]["kb_per_token"] == 4.0

    for row in payload["rows"]:
        if row["bench"] == "attn_backend":
            assert row["latency_ms"]["ci_low"] <= row["latency_ms"]["mean"] + 1e-9
            assert row["latency_ms"]["mean"] <= row["latency_ms"]["ci_high"] + 1e-9
            assert row["total_alloc_bytes"]["mean"] > 0


@pytest.mark.slow
def test_naive_attention_memory_is_quadratic_and_efficient_is_linear(
    cfg_31m: ModelConfig,
) -> None:
    """The §6 headline plot, as an assertion.

    Doubling the sequence must roughly 4x the naive backend's allocated bytes and
    roughly 2x the fused backend's.
    """
    from localmind.model import benchmark_attention_backends

    rows = benchmark_attention_backends(
        cfg_31m, seq_lens=(512, 1024), seeds=(0,), batch_size=1, iters=1
    )
    by_key = {(r["backend"], r["seq_len"]): r for r in rows}

    def alloc(backend: str, seq: int) -> float:
        return float(by_key[(backend, seq)]["total_alloc_bytes_samples"][0])

    naive_ratio = alloc("naive", 1024) / alloc("naive", 512)
    eff_ratio = alloc("sdpa_efficient", 1024) / alloc("sdpa_efficient", 512)
    assert 3.0 < naive_ratio < 5.0, f"naive should be ~quadratic, got {naive_ratio:.2f}x"
    assert 1.5 < eff_ratio < 2.8, f"efficient should be ~linear, got {eff_ratio:.2f}x"
    assert eff_ratio < naive_ratio

    # And the fused kernel allocates strictly less at the longer length.
    assert alloc("sdpa_efficient", 1024) < alloc("naive", 1024)
    assert by_key[("sdpa_efficient", 1024)]["observed_kernel"] in FUSED_KERNELS
    assert by_key[("naive", 1024)]["score_matrix_bytes_analytic"] == 1 * 8 * 1024 * 1024 * 4
    assert by_key[("sdpa_efficient", 1024)]["score_matrix_bytes_analytic"] == 0


# =====================================================================================
# DoD #3 -- the overfit test
# =====================================================================================
OVERFIT_CFG = ModelConfig(
    name="LocalMind-overfit-5M",
    vocab_size=4096,
    d_model=256,
    n_layers=5,
    n_heads=4,
    n_kv_heads=2,
    head_dim=64,
    ffn_hidden=768,
    max_seq_len=64,
    rope_theta=10000.0,
    qk_norm=True,
    bias=False,
    tie_embeddings=True,
    z_loss=1e-4,
    attn_dropout=0.0,
    init_std=0.02,
)


#: The DoD's "under 2 minutes on CPU" budget. ONE constant: it cuts the training loop
#: short AND is the assertion threshold, so the two can never drift apart. If the
#: machine is too slow to finish the planned schedule inside it, the loop is cut and
#: the `step == total_steps` assertion fails loudly -- the test can never quietly do
#: less work and still pass.
OVERFIT_BUDGET_S = 118.0


@pytest.mark.slow
def test_dod_3_overfit_100_sequences_to_near_zero_loss_on_cpu() -> None:
    """DoD #3 -- "the single best correctness test for a from-scratch transformer".

    A ~5M-parameter model memorises 100 random sequences. Near-zero loss means every
    piece is wired correctly: if the causal mask leaked, RoPE were applied to V, the
    GQA head mapping were scrambled, or a residual were dropped, this would plateau.
    Random targets mean there is nothing to generalise -- only memorisation passes.

    CPU only, no GPU, under two minutes.

    Sizing (measured on this machine, 12 cores / 8 torch threads):
      * ~270 ms per optimizer step, ~1500-2000 tokens/s. Throughput is the same for
        micro-batches of 20, 50 and 100, so splitting each epoch into 5 micro-batches
        costs nothing and buys 5x the optimizer steps -- which is the binding
        constraint here, not FLOPs.
      * With `seq_len=32` memorisation needs ~250 steps and a full schedule did not
        fit in the budget on a loaded machine, so `seq_len` is 16: it halves both the
        per-step cost and the number of random targets. Still 100 sequences, still
        uniformly random targets, still a genuine memorisation test.
      * Clock-off trajectory (constant LR, seq_len 32) confirmed the architecture can
        memorise at all before any budget tuning: CE 7.72 -> 0.084 and accuracy
        0.0009 -> 0.983 over 200 steps, falling monotonically with no plateau. A
        plateau would have meant a real bug (RoPE on V, a leaking causal mask, a
        scrambled GQA head mapping, a dropped residual); a slope means a budget
        problem, which is what this was.
      * The schedule is a FIXED 140 steps, not a time-budgeted loop, so the final CE
        is deterministic (seeded) at ~0.0061 with 99.87% train accuracy regardless of
        how loaded the machine is -- an 8x margin under the 0.05 threshold. Only the
        wall clock varies with load: 38 s idle, 113 s at the worst contention
        observed here (805 ms/step), both inside the budget.
    """
    assert 4_500_000 < count_params(OVERFIT_CFG, include_norms=True) < 5_500_000

    torch.manual_seed(0)
    model = LocalMindTransformer(OVERFIT_CFG)
    init_weights(model, OVERFIT_CFG, seed=0)
    model.train()

    gen = torch.Generator().manual_seed(1234)
    n_seq, seq_len, micro = 100, 16, 20
    ids = torch.randint(0, OVERFIT_CFG.vocab_size, (n_seq, seq_len), generator=gen)
    targets = torch.randint(0, OVERFIT_CFG.vocab_size, (n_seq, seq_len), generator=gen)

    peak_lr, total_steps, warmup, min_frac = 4e-3, 140, 20, 0.05
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.0)

    start = time.perf_counter()
    step = 0
    while step < total_steps and time.perf_counter() - start < OVERFIT_BUDGET_S:
        order = torch.randperm(n_seq, generator=gen)
        for i in range(0, n_seq, micro):
            if step >= total_steps or time.perf_counter() - start >= OVERFIT_BUDGET_S:
                break
            if step < warmup:
                lr = peak_lr * (step + 1) / warmup
            else:
                frac = (step - warmup) / (total_steps - warmup)
                lr = peak_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * frac)))
            for group in opt.param_groups:
                group["lr"] = lr

            batch = order[i : i + micro]
            out = model(ids[batch], targets[batch])
            assert out.loss is not None
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

    elapsed = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        final = model(ids, targets)
    assert final.ce_loss is not None
    final_ce = final.ce_loss.item()
    print(
        f"\n[DoD 3] overfit: params={count_params(OVERFIT_CFG, include_norms=True):,} "
        f"steps={step} elapsed={elapsed:.1f}s final_full_batch_ce={final_ce:.5f}"
    )

    # The budget cuts the loop, so a machine too slow to finish the schedule fails
    # here rather than silently training less and still passing.
    assert step == total_steps, (
        f"only completed {step}/{total_steps} steps in {OVERFIT_BUDGET_S}s "
        f"({elapsed / max(step, 1) * 1000:.0f} ms/step) -- machine too slow"
    )
    assert elapsed <= OVERFIT_BUDGET_S, f"took {elapsed:.1f}s, budget is {OVERFIT_BUDGET_S}s"
    assert final_ce < 0.05, f"did not overfit: final CE {final_ce:.4f}"
    # And it really did memorise: argmax accuracy on the training set.
    acc = (final.logits.argmax(-1) == targets).float().mean().item()
    assert acc > 0.99, f"train accuracy only {acc:.4f}"
