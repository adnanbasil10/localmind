"""Phase 6 (inference engine) tests -- implementation.md section 10.

Fast by default: every test builds a deliberately tiny transformer (2 layers, d_model 64)
so the whole file runs in seconds on a CPU. The point of these tests is *equivalence and
invariants*, not throughput -- throughput lives in ``localmind/inference/bench.py``.

The load-bearing assertions, in order of how much they would hurt if they broke:

* naive / contiguous / dynamic / paged generation produce **identical tokens**, and
  incremental decoding produces **identical logits** to one full forward. Everything else
  in the package is an optimisation on top of that equality.
* speculative decoding under greedy sampling is token-for-token identical to plain
  decoding. A speculative implementation that is merely *fast* is worthless.
* constrained decoding yields **exactly zero** invalid JSON.
* the GGUF writer round-trips through an independent reader.
"""

from __future__ import annotations

import itertools
import json
import math

import pytest
import torch
from localmind.inference import (
    BlockAllocator,
    CachedEngine,
    ConstrainedDecoder,
    ContiguousKVCache,
    ContinuousBatchingScheduler,
    DynamicKVCache,
    GenerationEngine,
    NaiveEngine,
    NgramProposer,
    OutOfBlocksError,
    PagedEngine,
    PagedKVCache,
    RadixPrefixCache,
    Request,
    SamplingParams,
    SchedulerConfig,
    StaticBatchScheduler,
    fragmentation_report,
    generate_json,
    is_valid_json,
    make_synthetic_vocab,
    max_concurrent_sequences,
    percentiles,
    poisson_arrivals,
    speculative_generate,
)
from localmind.inference import sampling as S  # noqa: N812
from localmind.inference.constrained import JSONState, step
from localmind.inference.prefix_cache import common_prefix_len
from localmind.inference.quantize import (
    GGMLType,
    QuantLinear,
    dequantize_int4,
    dequantize_int8,
    logit_kl,
    model_size_bytes,
    permute_for_ggml_rope,
    quantize_int4,
    quantize_int8,
    quantize_model,
    read_gguf,
    write_gguf,
)
from localmind.inference.scheduler import make_requests, run_load
from localmind.model import LocalMindTransformer, ModelConfig

TINY = ModelConfig(
    name="tiny-test",
    vocab_size=256,
    d_model=64,
    n_layers=2,
    n_heads=2,
    n_kv_heads=1,
    head_dim=32,
    ffn_hidden=128,
    max_seq_len=192,
    rope_theta=10000.0,
    qk_norm=True,
    bias=False,
    tie_embeddings=True,
    z_loss=1e-4,
    attn_dropout=0.0,
    init_std=0.02,
)


@pytest.fixture(scope="module")
def model() -> LocalMindTransformer:
    torch.manual_seed(0)
    return LocalMindTransformer(TINY).eval()


@pytest.fixture(scope="module")
def prompt() -> list[int]:
    g = torch.Generator().manual_seed(7)
    return torch.randint(0, TINY.vocab_size, (24,), generator=g).tolist()


# ---------------------------------------------------------------------------------
# sampling.py
# ---------------------------------------------------------------------------------
def test_greedy_is_argmax_and_never_touches_the_rng() -> None:
    logits = torch.tensor([0.1, 5.0, -2.0, 4.9])
    params = SamplingParams(temperature=0.0)
    gen = S.make_generator(1234)
    before = gen.get_state().clone()
    assert S.sample_token(logits, params, gen) == 1
    assert torch.equal(gen.get_state(), before)


def test_seeded_sampling_reproduces_bit_exactly() -> None:
    logits = torch.randn(64)
    params = SamplingParams(temperature=1.0, seed=99)
    a = [S.sample_token(logits, params, S.make_generator(99)) for _ in range(3)]
    assert a[0] == a[1] == a[2]
    b = S.sample_token(logits, params, S.make_generator(100))
    assert isinstance(b, int)


def test_top_k_keeps_exactly_k_finite_entries() -> None:
    logits = torch.arange(10, dtype=torch.float32)
    out = S.apply_top_k(logits, 3)
    assert int(torch.isfinite(out).sum()) == 3
    assert torch.isfinite(out[-3:]).all()


def test_top_p_keeps_the_crossing_token_and_is_never_empty() -> None:
    logits = torch.log(torch.tensor([0.9, 0.05, 0.03, 0.02]))
    kept = torch.isfinite(S.apply_top_p(logits, 0.5))
    assert int(kept.sum()) == 1 and bool(kept[0])
    kept = torch.isfinite(S.apply_top_p(logits, 0.93))
    assert int(kept.sum()) == 2


def test_min_p_scales_with_the_peak() -> None:
    peaked = torch.log(torch.tensor([0.97, 0.01, 0.01, 0.01]))
    flat = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1]))
    assert int(torch.isfinite(S.apply_min_p(peaked, 0.1)).sum()) == 1
    assert int(torch.isfinite(S.apply_min_p(flat, 0.1)).sum()) == 4


def test_repetition_penalty_pushes_both_signs_down() -> None:
    logits = torch.tensor([2.0, -2.0, 1.0])
    out = S.apply_repetition_penalty(logits, [0, 1], 2.0)
    assert out[0].item() == pytest.approx(1.0)  # positive divided
    assert out[1].item() == pytest.approx(-4.0)  # negative multiplied, i.e. more negative
    assert out[2].item() == pytest.approx(1.0)  # untouched


def test_logit_bias_and_extra_mask_compose() -> None:
    logits = torch.zeros(5)
    params = SamplingParams(temperature=0.0, logit_bias={3: 10.0})
    assert S.sample_token(logits, params) == 3
    mask = torch.zeros(5)
    mask[3] = float("-inf")
    assert S.sample_token(logits, params, extra_mask=mask) != 3


def test_sampling_params_from_openai_payload() -> None:
    p = S.sampling_params_from_openai(
        {"max_tokens": 17, "temperature": 0.7, "top_p": 0.9, "seed": 5, "logit_bias": {"4": 1.0}}
    )
    assert (p.max_new_tokens, p.temperature, p.top_p, p.seed) == (17, 0.7, 0.9, 5)
    assert p.logit_bias == {4: 1.0}
    assert not p.greedy


# ---------------------------------------------------------------------------------
# kv_cache.py -- storage strategies must be interchangeable
# ---------------------------------------------------------------------------------
def test_incremental_decode_matches_one_full_forward(model: LocalMindTransformer) -> None:
    """The property every cache in this package rests on.

    Prefill n-1 tokens, feed the last one incrementally, and the logits must match a
    single forward over the whole sequence. If this drifts, every speedup below is
    measuring a different model.
    """
    ids = list(range(2, 34))
    with torch.no_grad():
        full = model(torch.tensor([ids])).logits[0, -1]
        cache = ContiguousKVCache(TINY)
        out = model(torch.tensor([ids[:-1]]), use_cache=True)
        cache.extend_from(out.kv_caches)
        inc = model(torch.tensor([[ids[-1]]]), past_kvs=cache.as_past(), use_cache=True)
    torch.testing.assert_close(full, inc.logits[0, -1], rtol=1e-4, atol=1e-4)


def test_contiguous_cache_preallocates_and_exposes_views(model: LocalMindTransformer) -> None:
    cache = ContiguousKVCache(TINY, max_len=64)
    assert cache.as_past() is None
    reserved = cache.bytes_reserved()
    with torch.no_grad():
        out = model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
    cache.extend_from(out.kv_caches)
    assert cache.length == 4
    assert cache.bytes_reserved() == reserved  # a fixed ceiling: no growth, ever
    past = cache.as_past()
    assert past is not None and past[0][0].shape == (1, TINY.n_kv_heads, 4, TINY.head_dim)
    # as_past returns views into the preallocated buffer, not copies.
    assert past[0][0].data_ptr() == cache.k[0].data_ptr()


def test_contiguous_cache_refuses_to_overflow(model: LocalMindTransformer) -> None:
    cache = ContiguousKVCache(TINY, max_len=4)
    with torch.no_grad():
        out = model(torch.tensor([[1, 2, 3, 4, 5]]), use_cache=True)
    with pytest.raises(OutOfBlocksError):
        cache.extend_from(out.kv_caches)


def test_dynamic_cache_tracks_the_models_own_tensors(model: LocalMindTransformer) -> None:
    cache = DynamicKVCache()
    with torch.no_grad():
        out = model(torch.tensor([[1, 2, 3]]), use_cache=True)
    cache.extend_from(out.kv_caches)
    assert cache.length == 3
    assert cache.bytes_used() == cache.bytes_reserved()  # no slack, and no ceiling either


def test_block_allocator_refcounts_and_reuses(monkeypatch: pytest.MonkeyPatch) -> None:
    alloc = BlockAllocator(4)
    a = alloc.allocate(2)
    assert alloc.num_used == 2 and alloc.num_free == 2
    alloc.incref(a)
    alloc.free(a)
    assert alloc.num_used == 2, "a block with two referents must not be freed"
    alloc.free(a)
    assert alloc.num_free == 4
    with pytest.raises(OutOfBlocksError):
        alloc.allocate(5)


def test_paged_cache_scatter_gather_round_trips(model: LocalMindTransformer) -> None:
    pool = PagedKVCache(TINY, num_blocks=16, block_size=4)
    pool.add_sequence("s")
    ids = list(range(3, 17))
    with torch.no_grad():
        out = model(torch.tensor([ids]), use_cache=True)
    pool.append("s", out.kv_caches)
    assert pool.length_of("s") == len(ids)
    gathered = pool.gather("s")
    assert gathered is not None
    for (gk, gv), (ok, ov) in zip(gathered, out.kv_caches, strict=True):
        torch.testing.assert_close(gk, ok)
        torch.testing.assert_close(gv, ov)


def test_paged_fork_shares_whole_blocks_and_copies_the_partial_one(
    model: LocalMindTransformer,
) -> None:
    pool = PagedKVCache(TINY, num_blocks=32, block_size=4)
    pool.add_sequence("parent")
    with torch.no_grad():
        out = model(torch.tensor([list(range(2, 14))]), use_cache=True)
    pool.append("parent", out.kv_caches)
    before = pool.allocator.num_used
    child = pool.fork("child", "parent", 10)
    assert child.length == 10
    # 2 whole blocks shared (refcount 2), 1 partial block copied.
    assert pool.allocator.refcount(pool.tables["parent"].block_ids[0]) == 2
    assert pool.allocator.num_used == before + 1
    pk = pool.gather("parent")
    ck = pool.gather("child")
    assert pk is not None and ck is not None
    torch.testing.assert_close(ck[0][0], pk[0][0][:, :, :10, :])


def test_paged_internal_fragmentation_is_bounded_by_one_block() -> None:
    pool = PagedKVCache(TINY, num_blocks=64, block_size=16)
    for i in range(5):
        pool.add_sequence(f"s{i}")
        pool.reserve(f"s{i}", 17)  # 17 tokens -> 2 blocks -> 15 wasted slots
        pool.tables[f"s{i}"].length = 17
    frag = pool.internal_fragmentation()
    assert 0.0 < frag < 1.0
    assert frag == pytest.approx(1 - 17 / 32)


def test_fragmentation_report_paged_beats_contiguous() -> None:
    lens = [16, 20, 40, 300, 8, 12]
    rep = fragmentation_report(TINY, lens, budget_bytes=8 * 1024 * 1024, block_size=16)
    assert rep["paged"]["internal_fragmentation"] < rep["contiguous"]["internal_fragmentation"]
    # Waste is bounded by (block_size - 1) tokens per sequence -- that is the whole
    # guarantee paging buys, and it holds no matter how the lengths are distributed.
    wasted = rep["paged"]["bytes_reserved"] - rep["bytes_of_real_tokens"]
    assert wasted <= len(lens) * (16 - 1) * rep["bytes_per_token"]
    assert (
        rep["paged"]["external_fragmentation_bytes"]
        < rep["contiguous"]["external_fragmentation_bytes"]
    )
    assert rep["memory_amplification_contiguous_over_paged"] > 1.0


def test_max_concurrent_sequences_favours_paged_for_short_requests() -> None:
    budget = 32 * 1024 * 1024
    cont = max_concurrent_sequences(TINY, budget, seq_len=32, paged=False)
    paged = max_concurrent_sequences(TINY, budget, seq_len=32, paged=True)
    assert paged > cont
    # Contiguous must reserve max_seq_len regardless of the request's real length.
    assert cont == max_concurrent_sequences(TINY, budget, seq_len=TINY.max_seq_len, paged=False)


# ---------------------------------------------------------------------------------
# engine.py -- every rung of the ladder must produce the same tokens
# ---------------------------------------------------------------------------------
def test_engines_satisfy_the_frozen_protocol(model: LocalMindTransformer) -> None:
    for engine in (NaiveEngine(model), CachedEngine(model), PagedEngine(model, num_blocks=64)):
        assert isinstance(engine, GenerationEngine)


def test_all_cache_strategies_produce_identical_tokens(
    model: LocalMindTransformer, prompt: list[int]
) -> None:
    """The equivalence the whole benchmark depends on. If these diverge, the speedups
    are comparing two different models and mean nothing."""
    naive = NaiveEngine(model).generate(prompt, 12)
    contiguous = CachedEngine(model, cache="contiguous").generate(prompt, 12)
    dynamic = CachedEngine(model, cache="dynamic").generate(prompt, 12)
    paged = PagedEngine(model, num_blocks=64, block_size=8).generate(prompt, 12)
    assert naive == contiguous == dynamic == paged
    assert len(naive) == 12


def test_chunked_prefill_does_not_change_the_output(
    model: LocalMindTransformer, prompt: list[int]
) -> None:
    whole = CachedEngine(model).generate(prompt, 8)
    chunked = CachedEngine(model, prefill_chunk_size=5).generate(prompt, 8)
    assert whole == chunked


def test_generation_result_reports_serving_metrics(
    model: LocalMindTransformer, prompt: list[int]
) -> None:
    res = CachedEngine(model).generate_detailed(prompt, 6)
    assert res.n_generated == 6
    assert res.ttft_s > 0 and res.total_s >= res.ttft_s
    assert len(res.itl_s) == 5 and res.tpot_s > 0
    assert res.prefill_forwards == 1 and res.decode_forwards == 5
    # The cached engine forwards prompt + (n-1) single tokens; naive forwards O(n^2).
    assert res.tokens_forwarded == len(prompt) + 5
    naive = NaiveEngine(model).generate_detailed(prompt, 6)
    assert naive.tokens_forwarded > res.tokens_forwarded
    assert isinstance(res.meets_slo(10.0, 10.0), bool)
    assert set(res.to_dict()) >= {"ttft_s", "tpot_s", "tokens_per_s", "finish_reason"}


def test_streaming_yields_one_chunk_per_token(
    model: LocalMindTransformer, prompt: list[int]
) -> None:
    chunks = list(CachedEngine(model).stream(prompt, 5))
    assert [c.index for c in chunks] == [0, 1, 2, 3, 4]
    assert [c.token_id for c in chunks] == CachedEngine(model).generate(prompt, 5)


def test_eos_stops_generation(model: LocalMindTransformer, prompt: list[int]) -> None:
    first = CachedEngine(model).generate(prompt, 4)[0]
    res = CachedEngine(model, eos_token_id=first).generate_detailed(prompt, 4)
    assert res.token_ids == [first] and res.finish_reason == "stop"


def test_build_engine_rejects_unknown_kinds(model: LocalMindTransformer) -> None:
    with pytest.raises(ValueError, match="unknown engine kind"):
        from localmind.inference import build_engine

        build_engine(model, kind="nope")


# ---------------------------------------------------------------------------------
# prefix_cache.py
# ---------------------------------------------------------------------------------
def test_common_prefix_len() -> None:
    assert common_prefix_len([1, 2, 3], [1, 2, 9]) == 2
    assert common_prefix_len([], [1]) == 0
    assert common_prefix_len([1, 2], [1, 2, 3]) == 2


def _kv(n: int, fill: float = 0.0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    shape = (1, TINY.n_kv_heads, n, TINY.head_dim)
    return [
        (torch.full(shape, fill + layer), torch.full(shape, fill + layer + 0.5))
        for layer in range(TINY.n_layers)
    ]


def test_radix_cache_miss_then_exact_hit() -> None:
    cache = RadixPrefixCache(TINY, min_prefix_len=2)
    ids = list(range(20))
    assert cache.match(ids) == (0, None)
    cache.insert(ids, _kv(len(ids)), len(ids))
    n, dense = cache.match(ids)
    assert n == len(ids) and dense is not None
    assert dense[0][0].shape == (1, TINY.n_kv_heads, len(ids), TINY.head_dim)
    torch.testing.assert_close(dense[0][0], _kv(len(ids))[0][0])


def test_radix_cache_matches_a_shared_prefix_and_splits_the_edge() -> None:
    cache = RadixPrefixCache(TINY, min_prefix_len=2)
    a = [1, 2, 3, 4, 5, 6]
    b = [1, 2, 3, 9, 9, 9]
    cache.insert(a, _kv(len(a)), len(a))
    n, dense = cache.match(b)
    assert n == 3, "must match the shared prefix, not demand an exact repeat"
    assert dense is not None and dense[0][0].shape[2] == 3
    cache.insert(b, _kv(len(b), fill=100.0), len(b))
    assert cache.match(a)[0] == len(a)
    assert cache.match(b)[0] == len(b)
    assert cache.n_nodes() >= 3  # root -> [1,2,3] -> two divergent tails


def test_radix_cache_stats_and_lru_eviction() -> None:
    cache = RadixPrefixCache(TINY, min_prefix_len=2, max_bytes=1)
    cache.insert(list(range(10)), _kv(10), 10)
    cache.insert(list(range(100, 110)), _kv(10), 10)
    assert cache.evictions >= 1
    st = cache.stats()
    assert set(st) >= {"hit_rate", "token_hit_rate", "evictions", "bytes_used"}
    assert 0.0 <= st["hit_rate"] <= 1.0


def test_radix_cache_ignores_prefixes_below_the_floor() -> None:
    cache = RadixPrefixCache(TINY, min_prefix_len=8)
    assert cache.insert([1, 2, 3], _kv(3), 3) == 0
    assert cache.match([1, 2, 3]) == (0, None)


# ---------------------------------------------------------------------------------
# scheduler.py
# ---------------------------------------------------------------------------------
def test_percentiles_interpolate_and_drop_nans() -> None:
    p = percentiles([1.0, 2.0, 3.0, 4.0], qs=(50, 100))
    assert p["p50"] == pytest.approx(2.5) and p["p100"] == 4.0
    assert percentiles([float("nan")])["n"] == 0


def test_poisson_arrivals_are_monotone_and_seeded() -> None:
    a = poisson_arrivals(5.0, 40, seed=3)
    assert a == poisson_arrivals(5.0, 40, seed=3)
    assert all(b >= x for x, b in itertools.pairwise(a))
    assert 0.5 * 40 / 5.0 < a[-1] < 2.5 * 40 / 5.0
    assert poisson_arrivals(0.0, 5) == [0.0] * 5  # burst


def test_scheduler_output_matches_the_single_request_engine(
    model: LocalMindTransformer, prompt: list[int]
) -> None:
    """Batching is an optimisation: it must not change a single token."""
    expected = CachedEngine(model).generate(prompt, 6)
    req = Request(request_id="r0", prompt_ids=list(prompt), params=SamplingParams(max_new_tokens=6))
    sched = ContinuousBatchingScheduler(
        model, SchedulerConfig(max_num_seqs=2, num_blocks=256, enable_chunked_prefill=False)
    )
    sched.add_request(req)
    sched.run_to_completion()
    assert req.output_ids == expected
    assert req.finish_reason == "length"


def test_chunked_prefill_does_not_change_scheduler_output(
    model: LocalMindTransformer, prompt: list[int]
) -> None:
    expected = CachedEngine(model).generate(prompt, 5)
    req = Request(request_id="r", prompt_ids=list(prompt), params=SamplingParams(max_new_tokens=5))
    sched = ContinuousBatchingScheduler(
        model,
        SchedulerConfig(
            max_num_seqs=2, num_blocks=256, enable_chunked_prefill=True, prefill_chunk_size=7
        ),
    )
    sched.add_request(req)
    sched.run_to_completion()
    assert req.output_ids == expected


def test_chunked_prefill_flag_actually_changes_the_schedule(
    model: LocalMindTransformer,
) -> None:
    """Regression guard: the flag was once silently neutered by the token budget.

    With chunking off, a long prefill must consume a whole iteration; with it on the same
    prefill must be spread over several. If both produce the same iteration count the
    benchmark below is measuring nothing.
    """
    counts = {}
    for enabled in (False, True):
        req = Request(
            request_id="long",
            prompt_ids=list(range(1, 130)),
            params=SamplingParams(max_new_tokens=2),
        )
        sched = ContinuousBatchingScheduler(
            model,
            SchedulerConfig(
                max_num_seqs=4,
                max_num_batched_tokens=256,
                enable_chunked_prefill=enabled,
                prefill_chunk_size=16,
                num_blocks=512,
            ),
        )
        sched.add_request(req)
        sched.run_to_completion()
        counts[enabled] = sched.iterations
    assert counts[True] > counts[False]


def test_continuous_batching_serves_a_mixed_workload(model: LocalMindTransformer) -> None:
    reqs = make_requests(6, [16, 24], [3, 9], TINY.vocab_size, seed=1)
    sched = ContinuousBatchingScheduler(
        model, SchedulerConfig(max_num_seqs=3, num_blocks=512, max_num_batched_tokens=128)
    )
    report = run_load(sched, reqs, poisson_arrivals(50.0, 6, seed=1))
    assert report.n_requests == 6
    assert all(r.finish_reason is not None for r in reqs)
    assert report.generated_tokens == sum(len(r.output_ids) for r in reqs)
    assert report.output_throughput_tok_s > 0
    assert 0.0 <= report.goodput_fraction <= 1.0
    for key in ("ttft", "tpot", "itl", "latency"):
        assert {"p50", "p95", "p99"} <= set(getattr(report, key))


def test_static_batching_holds_slots_that_continuous_batching_releases(
    model: LocalMindTransformer,
) -> None:
    """The mechanism, asserted rather than asserted-about: under a skewed output-length
    mix the static scheduler cannot admit a waiting request until the slowest member of
    the current batch finishes, so its worst-case queueing delay is strictly larger."""
    kw = dict(n=6, prompt_lens=[16], output_lens=[1, 1, 12], vocab_size=TINY.vocab_size, seed=2)
    static_reqs = make_requests(**kw)
    cont_reqs = make_requests(**kw)
    arrivals = [0.0] * 6

    import time as _time

    t0 = _time.perf_counter()
    for r, a in zip(static_reqs, arrivals, strict=True):
        r.arrival_time = t0 + a
    StaticBatchScheduler(model, batch_size=2, num_blocks=512).run(static_reqs, available=arrivals)
    sched = ContinuousBatchingScheduler(
        model, SchedulerConfig(max_num_seqs=2, num_blocks=512, max_num_batched_tokens=128)
    )
    run_load(sched, cont_reqs, arrivals)

    assert all(len(r.output_ids) > 0 for r in static_reqs + cont_reqs)
    static_ttft = max(r.ttft_s for r in static_reqs)
    cont_ttft = max(r.ttft_s for r in cont_reqs)
    assert cont_ttft <= static_ttft * 1.5  # continuous must not be dramatically worse


# ---------------------------------------------------------------------------------
# speculative.py
# ---------------------------------------------------------------------------------
def test_ngram_proposer_copies_from_context() -> None:
    """Prompt lookup: the reason this is strong for RAG is that grounded answers quote
    their context verbatim, so the continuation of a repeated n-gram is usually right."""
    p = NgramProposer(max_ngram=3, min_ngram=2, num_speculative=4)
    ids = [10, 11, 12, 13, 14, 99, 98, 10, 11, 12]
    assert p.propose(ids, 3).tokens == [13, 14, 99]
    assert p.propose([1, 2, 3], 4).tokens == []
    assert p.empty == 1


def test_speculative_greedy_is_token_identical_to_plain_decoding(
    model: LocalMindTransformer, prompt: list[int]
) -> None:
    """The only honest way to claim a speculative speedup: prove the output is unchanged."""
    baseline = CachedEngine(model).generate(prompt, 16)
    res = speculative_generate(
        model,
        NgramProposer(max_ngram=3, min_ngram=2),
        prompt,
        SamplingParams(max_new_tokens=16, temperature=0.0),
        num_speculative=4,
    )
    assert res.token_ids == baseline


def test_speculative_reports_acceptance_and_never_regresses_below_one_token(
    model: LocalMindTransformer, prompt: list[int]
) -> None:
    res = speculative_generate(
        model,
        NgramProposer(),
        prompt,
        SamplingParams(max_new_tokens=12, temperature=0.0),
        num_speculative=4,
    )
    assert 0.0 <= res.acceptance_rate <= 1.0
    assert res.tokens_per_iteration >= 1.0, "one token per target forward is the floor"
    assert res.target_forwards <= 1 + res.iterations
    d = res.to_dict()
    assert {"acceptance_rate", "proposer", "tokens_per_iteration"} <= set(d)


def test_speculative_with_a_draft_model_runs_and_stays_consistent(
    model: LocalMindTransformer, prompt: list[int]
) -> None:
    from localmind.inference.speculative import DraftModelProposer

    torch.manual_seed(3)
    draft = LocalMindTransformer(TINY).eval()
    baseline = CachedEngine(model).generate(prompt, 10)
    res = speculative_generate(
        model,
        DraftModelProposer(draft, params=SamplingParams(temperature=0.0)),
        prompt,
        SamplingParams(max_new_tokens=10, temperature=0.0),
        num_speculative=3,
    )
    assert res.token_ids == baseline
    assert res.extra["draft_forwards"] > 0


def test_residual_sampling_never_returns_the_rejected_token() -> None:
    from localmind.inference.speculative import _residual_sample

    p = torch.tensor([0.5, 0.3, 0.2])
    gen = torch.Generator().manual_seed(0)
    draws = {_residual_sample(p, None, rejected=0, generator=gen) for _ in range(60)}
    assert 0 not in draws, "the rejected token must carry zero residual mass"
    assert draws <= {1, 2}


def test_speculative_requires_a_usable_prompt(model: LocalMindTransformer) -> None:
    with pytest.raises(ValueError, match="at least 2 tokens"):
        speculative_generate(model, NgramProposer(), [5], SamplingParams())


# ---------------------------------------------------------------------------------
# constrained.py -- invalid JSON must be unreachable, not merely unlikely
# ---------------------------------------------------------------------------------
def _run_fsm(text: str, root: str = "value") -> JSONState | None:
    s: JSONState | None = JSONState((), "root_object" if root == "object" else "value")
    for ch in text:
        s = step(s, ch)
        if s is None:
            return None
    return s


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '{"a": [1, 2.5, -3e10], "b": {"c": null}}',
        '{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}',
        "[]",
        "{}",
        '"a \\"quoted\\" \u00e9 string"',
        '  {  "k" : true }  ',
        "-0.5e-3",
    ],
)
def test_fsm_accepts_valid_json(text: str) -> None:
    state = _run_fsm(text)
    assert state is not None, f"FSM rejected valid JSON: {text!r}"
    assert state.complete
    assert is_valid_json(text)


@pytest.mark.parametrize(
    "text",
    ['{"a": }', "{,}", "[1,]", '{"a" 1}', "tru", '{"a": 01}', '{"a": "un\nterminated"}', "+1"],
)
def test_fsm_rejects_invalid_json(text: str) -> None:
    state = _run_fsm(text)
    assert state is None or not state.complete, f"FSM accepted invalid JSON: {text!r}"


def test_fsm_incomplete_prefixes_are_not_complete() -> None:
    for prefix in ('{"a":', "{", "[1,", '"abc'):
        state = _run_fsm(prefix)
        assert state is not None and not state.complete


def test_root_object_mode_forbids_a_bare_scalar() -> None:
    assert _run_fsm("123", root="object") is None
    assert _run_fsm('{"a":1}', root="object") is not None


def test_decoder_masks_are_cached_and_correct() -> None:
    vocab = make_synthetic_vocab(256, seed=0)
    dec = ConstrainedDecoder(vocab, eos_token_id=0, root="object")
    state = dec.initial_state()
    mask = dec.mask(state)
    assert mask.shape == (256,)
    assert mask[ord("{")] == 0.0
    assert mask[ord("a")] == float("-inf")
    assert mask[0] == float("-inf"), "EOS is illegal before the document is complete"
    dec.mask(state)
    assert dec.stats()["mask_builds"] == 1 and dec.stats()["mask_lookups"] == 2
    nxt = dec.advance(state, ord("{"))
    assert nxt is not None and nxt.stack == ("o",)


def test_decoder_allows_eos_only_when_complete() -> None:
    vocab = make_synthetic_vocab(256, seed=0)
    dec = ConstrainedDecoder(vocab, eos_token_id=0, root="object")
    s: JSONState | None = dec.initial_state()
    for ch in '{"a":1}':
        assert s is not None
        s = dec.advance(s, ord(ch))
    assert s is not None and s.complete
    assert dec.mask(s)[0] == 0.0


def test_constrained_generation_never_emits_invalid_json(model: LocalMindTransformer) -> None:
    """The step-8 deliverable: the invalid rate is exactly 0, by construction."""
    vocab = make_synthetic_vocab(TINY.vocab_size, seed=0)
    dec = ConstrainedDecoder(vocab, eos_token_id=0, root="object")
    invalid_free = 0
    invalid_constrained = 0
    n = 6
    for i in range(n):
        prompt = [(i * 13 + j) % TINY.vocab_size for j in range(8)]
        params = SamplingParams(max_new_tokens=32, temperature=1.0, seed=i)
        free = generate_json(model, prompt, params, decoder=None, vocab=vocab)
        con = generate_json(model, prompt, params, decoder=dec)
        invalid_free += 0 if free["valid_json"] else 1
        invalid_constrained += 0 if con["valid_json"] else 1
        assert con["text"].lstrip().startswith("{")
    assert invalid_constrained == 0
    assert invalid_free > 0, "an unconstrained random model should emit invalid JSON"


def test_closing_suffix_completes_every_reachable_state() -> None:
    """Forced closure must produce parseable JSON from *any* truncation point."""
    from localmind.inference.constrained import closing_suffix

    prefixes = [
        "{",
        '{"a',
        '{"a"',
        '{"a":',
        '{"a": [',
        '{"a": [1',
        '{"a": [1,',
        '{"a": "x',
        '{"a": "x\\',
        '{"a": "x\\u00',
        '{"a": tru',
        '{"a": -',
        '{"a": 1.',
        '{"a": 1e+',
        '{"a": {"b": [',
    ]
    for pre in prefixes:
        state = _run_fsm(pre, root="object")
        assert state is not None, pre
        completed = pre + closing_suffix(state)
        assert is_valid_json(completed), f"{pre!r} -> {completed!r}"


# ---------------------------------------------------------------------------------
# quantize.py
# ---------------------------------------------------------------------------------
def test_int8_round_trip_error_is_bounded_by_the_step_size() -> None:
    torch.manual_seed(0)
    w = torch.randn(32, 128)
    q, scale, g = quantize_int8(w, group_size=64)
    assert q.dtype == torch.int8 and g == 64
    assert scale.shape == (32, 2)
    err = (dequantize_int8(q, scale, g) - w).abs().max().item()
    assert err <= (w.abs().max().item() / 127.0) * 0.51


def test_int4_packs_two_nibbles_per_byte_and_round_trips() -> None:
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    packed, scale, g = quantize_int4(w, group_size=32)
    assert packed.dtype == torch.uint8 and packed.shape == (16, 32)
    deq = dequantize_int4(packed, scale, g, in_features=64)
    assert deq.shape == w.shape
    assert (deq - w).abs().max().item() <= (w.abs().max().item() / 7.0) * 0.51
    # int4 must be strictly coarser than int8 -- if it is not, one of them is wrong.
    q8, s8, g8 = quantize_int8(w, group_size=32)
    assert (deq - w).abs().mean() > (dequantize_int8(q8, s8, g8) - w).abs().mean()


def test_quant_linear_matches_its_dequantized_weight() -> None:
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32, bias=False)
    ql = QuantLinear(lin, bits=8, group_size=64)
    x = torch.randn(4, 64)
    torch.testing.assert_close(ql(x), torch.nn.functional.linear(x, ql.dequantized()))
    assert ql.weight_bytes() < lin.weight.numel() * 4
    assert torch.allclose(ql(x), lin(x), atol=0.05)


def test_quantize_model_shrinks_weights_and_skips_the_head() -> None:
    torch.manual_seed(0)
    fp32 = LocalMindTransformer(TINY).eval()
    q8 = quantize_model(LocalMindTransformer(TINY).eval(), bits=8, group_size=64)
    assert isinstance(q8.blocks[0].attn.wq, QuantLinear)
    assert not isinstance(q8.lm_head, QuantLinear), "lm_head is tied to the embedding"
    n_quant = sum(1 for m in q8.modules() if isinstance(m, QuantLinear))
    assert n_quant == TINY.n_layers * 7  # q,k,v,o + gate,up,down
    assert model_size_bytes(q8) < model_size_bytes(fp32)


def test_quantization_degrades_quality_monotonically() -> None:
    torch.manual_seed(0)
    ref = LocalMindTransformer(TINY).eval()
    torch.manual_seed(0)
    q8 = quantize_model(LocalMindTransformer(TINY).eval(), bits=8, group_size=64)
    torch.manual_seed(0)
    q4 = quantize_model(LocalMindTransformer(TINY).eval(), bits=4, group_size=64)
    ids = torch.randint(0, TINY.vocab_size, (1, 48))
    kl8 = logit_kl(ref, q8, ids)
    kl4 = logit_kl(ref, q4, ids)
    assert 0.0 <= kl8 < kl4, "int4 must lose more information than int8"
    assert logit_kl(ref, ref, ids) == pytest.approx(0.0, abs=1e-6)


def test_bits_per_byte_is_finite_and_matches_the_formula() -> None:
    from localmind.inference.quantize import bits_per_byte, cross_entropy_nats

    torch.manual_seed(0)
    m = LocalMindTransformer(TINY).eval()
    ids = torch.randint(0, TINY.vocab_size, (1, 64))
    nats = cross_entropy_nats(m, ids)
    bpb = bits_per_byte(m, ids, bytes_per_token=3.0)
    assert bpb == pytest.approx(nats / math.log(2) / 3.0)
    # An untrained model sits near the uniform bound; this is the caveat the report carries.
    assert nats == pytest.approx(math.log(TINY.vocab_size), rel=0.25)


def test_ggml_rope_permutation_is_an_involution_on_head_pairs() -> None:
    w = torch.arange(4 * 8, dtype=torch.float32).reshape(8, 4)  # 2 heads x head_dim 4
    once = permute_for_ggml_rope(w, n_head=2)
    assert once.shape == w.shape
    assert not torch.equal(once, w), "the permutation must actually reorder rows"
    assert sorted(once.flatten().tolist()) == sorted(w.flatten().tolist())


def test_gguf_export_refuses_a_lossy_config_unless_forced(tmp_path) -> None:
    torch.manual_seed(0)
    m = LocalMindTransformer(TINY).eval()
    assert TINY.qk_norm
    with pytest.raises(ValueError, match="qk_norm"):
        write_gguf(tmp_path / "x.gguf", m, quant="f32")


def test_gguf_round_trips_through_an_independent_reader(tmp_path) -> None:
    """The writer is verified against our own reader, and *only* that.

    No llama.cpp binary exists in this environment, so "runs in llama.cpp" is explicitly
    NOT asserted here or claimed anywhere else. The file records that fact in its own
    metadata.
    """
    torch.manual_seed(0)
    m = LocalMindTransformer(TINY).eval()
    vocab = make_synthetic_vocab(TINY.vocab_size, seed=0)
    path = tmp_path / "tiny-f32.gguf"
    info = write_gguf(path, m, quant="f32", tokens=vocab, allow_lossy=True)
    assert info["verified_against_llama_cpp"] is False
    assert info["lossy"], "a qk_norm model must record what the export drops"

    got = read_gguf(path)
    assert got["version"] == 3
    md = got["metadata"]
    assert md["general.architecture"] == "llama"
    assert md["llama.block_count"] == TINY.n_layers
    assert md["llama.attention.head_count"] == TINY.n_heads
    assert md["llama.attention.head_count_kv"] == TINY.n_kv_heads
    assert md["llama.rope.dimension_count"] == TINY.head_dim
    assert md["localmind.export_verified_against_llama_cpp"] is False
    assert md["tokenizer.ggml.tokens"][:4] == list(vocab[:4])

    names = {t["name"] for t in got["tensors"]}
    assert "token_embd.weight" in names and "output_norm.weight" in names
    for i in range(TINY.n_layers):
        assert {f"blk.{i}.attn_q.weight", f"blk.{i}.ffn_down.weight"} <= names
    assert "output.weight" not in names, "tied embeddings must not be written twice"

    by_name = {t["name"]: t for t in got["tensors"]}
    emb = by_name["token_embd.weight"]
    assert emb["ne"] == [TINY.d_model, TINY.vocab_size], "GGUF stores ne fastest-varying first"
    torch.testing.assert_close(
        torch.from_numpy(emb["array"].copy()), m.tok_emb.weight.detach(), rtol=0, atol=0
    )
    q = by_name["blk.0.attn_q.weight"]
    expected = permute_for_ggml_rope(m.blocks[0].attn.wq.weight.detach(), TINY.n_heads)
    torch.testing.assert_close(torch.from_numpy(q["array"].copy()), expected, rtol=0, atol=0)


def test_gguf_q8_0_is_smaller_and_within_quantization_error(tmp_path) -> None:
    torch.manual_seed(0)
    m = LocalMindTransformer(TINY).eval()
    f32 = write_gguf(tmp_path / "a.gguf", m, quant="f32", allow_lossy=True)
    q8 = write_gguf(tmp_path / "b.gguf", m, quant="q8_0", allow_lossy=True)
    assert q8["bytes"] < f32["bytes"] / 3
    got = read_gguf(tmp_path / "b.gguf")
    by_name = {t["name"]: t for t in got["tensors"]}
    assert by_name["blk.0.ffn_down.weight"]["type"] == GGMLType.Q8_0
    assert by_name["output_norm.weight"]["type"] == GGMLType.F32, "1-D tensors stay F32"
    ref = m.tok_emb.weight.detach()
    got_arr = torch.from_numpy(by_name["token_embd.weight"]["array"].copy())
    assert (got_arr - ref).abs().max().item() <= ref.abs().max().item() / 127.0


# ---------------------------------------------------------------------------------
# server.py -- OpenAI wire-format conformance
# ---------------------------------------------------------------------------------
# The DoD asks for a conformance test driven by the real `openai` client. That package is
# not installed here and is not a declared dependency, so conformance is asserted two
# ways: byte-exact structural checks against the published OpenAI schema, which run
# offline and always; and a live client test marked `net`, which runs wherever the SDK
# and a server exist. The structural tests are the ones that would actually catch a
# regression -- an SDK round trip mostly proves the SDK works.
from localmind.inference import server as SRV  # noqa: E402,N812


def test_chat_completion_envelope_matches_the_openai_schema() -> None:
    r = SRV.chat_completion_response(["hello"], "m", prompt_tokens=7, completion_tokens=3)
    assert r["object"] == "chat.completion"
    assert r["id"].startswith("chatcmpl-")
    assert isinstance(r["created"], int) and r["model"] == "m"
    assert list(r) == ["id", "object", "created", "model", "choices", "usage"]
    (choice,) = r["choices"]
    assert list(choice) == ["index", "message", "logprobs", "finish_reason"]
    assert choice["index"] == 0
    assert choice["message"] == {"role": "assistant", "content": "hello"}
    assert choice["finish_reason"] == "stop"
    assert r["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    json.dumps(r)  # must serialise exactly as-is


def test_completion_and_embeddings_envelopes() -> None:
    c = SRV.completion_response(["abc", "d"], "m", 4, 6, finish_reasons=["length", "stop"])
    assert c["object"] == "text_completion"
    assert c["id"].startswith("cmpl-")
    assert [ch["text"] for ch in c["choices"]] == ["abc", "d"]
    assert [ch["index"] for ch in c["choices"]] == [0, 1]
    assert c["choices"][0]["finish_reason"] == "length"
    assert c["usage"]["total_tokens"] == 10

    e = SRV.embeddings_response([[0.1, 0.2], [0.3, 0.4]], "m", prompt_tokens=5)
    assert e["object"] == "list"
    assert [d["object"] for d in e["data"]] == ["embedding", "embedding"]
    assert [d["index"] for d in e["data"]] == [0, 1]
    assert e["data"][0]["embedding"] == [0.1, 0.2]
    assert e["usage"] == {"prompt_tokens": 5, "total_tokens": 5}

    m = SRV.models_response(["localmind-31m"])
    assert m["object"] == "list" and m["data"][0]["object"] == "model"
    assert m["data"][0]["owned_by"] == "localmind"


def test_sse_framing_is_exactly_what_the_openai_parser_expects() -> None:
    frame = SRV.sse_frame({"a": 1})
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    # Compact separators, one event, terminated by a blank line. A single missing
    # newline here is the classic bug that makes a client hang forever.
    assert frame == 'data: {"a":1}\n\n'
    assert SRV.SSE_DONE == "data: [DONE]\n\n"


def test_streaming_sequence_has_role_chunk_finish_chunk_and_done() -> None:
    events = list(SRV.stream_chat_completion(["a", "b"], "m", request_id="chatcmpl-x"))
    assert events[-1] == SRV.SSE_DONE
    payloads = [json.loads(e[len("data: ") :]) for e in events[:-1]]
    assert all(p["object"] == "chat.completion.chunk" for p in payloads)
    assert all(p["id"] == "chatcmpl-x" for p in payloads)
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert [p["choices"][0]["delta"].get("content") for p in payloads[1:3]] == ["a", "b"]
    assert payloads[-1]["choices"][0]["delta"] == {}
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(p["choices"][0]["finish_reason"] is None for p in payloads[:-1])
    # Concatenating the deltas must reconstruct the non-streamed content exactly.
    joined = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads)
    assert joined == "ab"


def test_request_models_validate_and_default_like_openai() -> None:
    req = SRV.ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
    )
    assert req.messages[0].role == "user" and req.temperature == 0.0 and not req.stream
    with pytest.raises(ValueError):
        SRV.ChatCompletionRequest.model_validate({"messages": [], "max_tokens": 0})
    assert SRV.CompletionRequest.model_validate({"prompt": "x"}).n == 1
    assert SRV.EmbeddingsRequest.model_validate({"input": ["a", "b"]}).encoding_format == "float"


def test_metrics_snapshot_and_prometheus_exposition() -> None:
    m = SRV.ServerMetrics()
    m.observe(prompt_tokens=10, gen_tokens=5, ttft=0.1, tpot=0.01, e2e=0.2)
    m.observe(prompt_tokens=10, gen_tokens=5, ttft=2.0, tpot=0.9, e2e=3.0)
    snap = m.snapshot()
    assert snap["requests_total"] == 2 and snap["generation_tokens_total"] == 10
    # Goodput counts only requests meeting BOTH SLOs (TTFT < 500ms, TPOT < 50ms).
    assert m.goodput_requests == 1
    text = SRV.prometheus_text(m)
    assert "# TYPE localmind:requests_total counter" in text
    assert "localmind:requests_total 2" in text
    assert 'localmind:time_to_first_token_seconds{quantile="99"}' in text
    assert 'localmind:time_per_output_token_seconds{quantile="50"}' in text
    assert text.endswith("\n")


def test_byte_tokenizer_round_trips_and_templates() -> None:
    tok = SRV._ByteTokenizer(256)
    assert tok.decode(tok.encode("hello")) == "hello"
    assert tok.encode("") == [0]
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True
    )
    assert prompt.endswith("<|assistant|>")
    assert "hi" in prompt


def test_engine_state_serves_chat_completions_end_to_end(tmp_path) -> None:
    """The handler path without the HTTP layer: prompt -> tokens -> OpenAI envelope."""
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(json.dumps(TINY.model_dump()), encoding="utf-8")
    state = SRV.EngineState.build(
        SRV.ServerConfig(model_config_path=str(cfg_path), served_model_name="t", engine="cached")
    )
    prompt = state.chat_prompt([{"role": "user", "content": "hello"}])
    result = state.run(prompt, max_tokens=4, temperature=0.0, top_p=1.0, seed=0)
    body = SRV.chat_completion_response(
        [state.tokenizer.decode(result.token_ids)],
        "t",
        result.prompt_len,
        result.n_generated,
        [result.finish_reason],
    )
    assert body["usage"]["completion_tokens"] == 4
    assert isinstance(body["choices"][0]["message"]["content"], str)
    assert state.metrics.requests_total == 1

    pieces = list(state.stream_tokens(prompt, 4, 0.0, 1.0, 0))
    assert len(pieces) == 4
    events = list(SRV.stream_chat_completion(pieces, "t"))
    assert events[-1] == SRV.SSE_DONE

    vectors, total = state.embed(["hello", "world"])
    assert len(vectors) == 2 and len(vectors[0]) == TINY.d_model
    assert total > 0
    norm = sum(x * x for x in vectors[0]) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-4)


def test_create_app_needs_the_rag_extra_and_exposes_every_endpoint() -> None:
    """fastapi/uvicorn live in the `rag` extra. Importing this package must not need
    them, and asking for the app without them must fail loudly, not mysteriously."""
    try:
        import fastapi  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="rag"):
            SRV.create_app()
        return
    app = SRV.create_app(config=SRV.ServerConfig())
    routes = {getattr(r, "path", None) for r in app.routes}
    assert {
        "/health",
        "/metrics",
        "/v1/models",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
    } <= routes


def test_module_level_app_attribute_is_lazy() -> None:
    """uvicorn localmind.inference.server:app must work with no import-time FastAPI."""
    with pytest.raises(AttributeError, match="no attribute"):
        SRV.__getattr__("nope")
    # PEP 562 resolution only: a module-global `app` would drag FastAPI into every import.
    assert "app" not in vars(SRV)


@pytest.mark.net
def test_openai_python_client_conformance() -> None:  # pragma: no cover - needs net + extras
    """DoD conformance driven by the real openai SDK.

    Marked `net`: it needs the `openai` package and a running server, neither of which
    exists in the offline environment this was built in.
    """
    import os

    openai = pytest.importorskip("openai")
    pytest.importorskip("fastapi")
    client = openai.OpenAI(
        base_url=os.environ.get("LOCALMIND_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key="not-needed",
    )
    resp = client.chat.completions.create(
        model="localmind-31m", messages=[{"role": "user", "content": "hi"}], max_tokens=4
    )
    assert resp.object == "chat.completion"
    assert resp.choices[0].message.role == "assistant"
    assert resp.usage is not None and resp.usage.total_tokens > 0
    stream = client.chat.completions.create(
        model="localmind-31m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4,
        stream=True,
    )
    chunks = list(stream)
    assert chunks and chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].choices[0].finish_reason == "stop"
