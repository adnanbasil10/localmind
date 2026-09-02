"""Generation engines: the naive denominator, and the cached engines that beat it.

``GenerationEngine`` here is the **frozen** cross-task Protocol from CONVENTIONS.md.
Everything else in this module is an implementation of it, arranged as the ladder
implementation.md section 10 asks for:

=======================  ====================================================
:class:`NaiveEngine`     no cache at all; re-runs the whole prefix every token
:class:`CachedEngine`    contiguous preallocated cache; prefill once, decode 1x1
:class:`PagedEngine`     the same decode loop over :class:`PagedKVCache` blocks
=======================  ====================================================

Why the naive/cached gap is so large -- the sentence the benchmark exists to prove
--------------------------------------------------------------------------------
**Prefill is compute-bound.** One forward over a T-token prompt does O(T) work per
weight *read*: every weight is loaded once from memory and then reused across all T
token positions, so arithmetic intensity is high and the machine's FLOP/s is the limit.

**Decode is memory-bandwidth-bound.** Generating one token with a cache reads every
weight in the model to do a single token's worth of matmul -- arithmetic intensity
around 1 -- so the limit is how fast bytes move, not how fast they multiply. For
LocalMind-12M in fp32 that is ~47 MB of weights per token no matter what.

Naive generation is neither: it *redoes prefill on every step*. Generating N tokens
after a P-token prompt costs sum over i of forward(P+i) instead of forward(P) + N
memory-bound steps, i.e. O(N * (P + N/2)) token-forwards instead of O(P + N). The
speedup therefore grows roughly linearly with sequence length, which is exactly the
shape the benchmark reports.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from localmind.inference.kv_cache import (
    DEFAULT_BLOCK_SIZE,
    ContiguousKVCache,
    DynamicKVCache,
    LayerKV,
    PagedKVCache,
)
from localmind.inference.sampling import SamplingParams, make_generator, sample_token
from localmind.model import LocalMindTransformer, ModelConfig

__all__ = [
    "CachedEngine",
    "GenerationEngine",
    "GenerationResult",
    "NaiveEngine",
    "PagedEngine",
    "StreamChunk",
    "build_engine",
]


@runtime_checkable
class GenerationEngine(Protocol):
    """FROZEN (CONVENTIONS.md "Shared interfaces"). Do not widen this signature."""

    def generate(self, prompt_ids: list[int], max_new_tokens: int, **kw: Any) -> list[int]: ...


@dataclass
class GenerationResult:
    """One request's tokens plus the serving metrics implementation.md section 10 names.

    ``ttft_s`` is prefill-dominated; ``itl_s`` are the inter-token latencies whose mean
    is ``tpot_s``. Never report a single request's numbers -- ``bench.py`` aggregates
    these into p50/p95/p99 over generated load.
    """

    token_ids: list[int]
    prompt_len: int
    ttft_s: float = 0.0
    itl_s: list[float] = field(default_factory=list)
    total_s: float = 0.0
    finish_reason: str = "length"
    prefill_forwards: int = 0
    decode_forwards: int = 0
    tokens_forwarded: int = 0

    @property
    def n_generated(self) -> int:
        return len(self.token_ids)

    @property
    def tpot_s(self) -> float:
        """Time per output token, decode only -- TTFT deliberately excluded."""
        return sum(self.itl_s) / len(self.itl_s) if self.itl_s else float("nan")

    @property
    def tokens_per_s(self) -> float:
        return self.n_generated / self.total_s if self.total_s > 0 else float("nan")

    def meets_slo(self, ttft_budget_s: float = 0.5, tpot_budget_s: float = 0.05) -> bool:
        """The goodput predicate: TTFT < 500 ms and TPOT < 50 ms (section 10 default SLO)."""
        tpot = self.tpot_s
        return self.ttft_s < ttft_budget_s and (tpot != tpot or tpot < tpot_budget_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_generated": self.n_generated,
            "prompt_len": self.prompt_len,
            "ttft_s": self.ttft_s,
            "tpot_s": self.tpot_s,
            "total_s": self.total_s,
            "tokens_per_s": self.tokens_per_s,
            "finish_reason": self.finish_reason,
            "prefill_forwards": self.prefill_forwards,
            "decode_forwards": self.decode_forwards,
            "tokens_forwarded": self.tokens_forwarded,
        }


@dataclass
class StreamChunk:
    """One streamed token, with the timing the SSE layer reports back."""

    token_id: int
    index: int
    latency_s: float
    finish_reason: str | None = None


class _BaseEngine:
    """Shared plumbing: model handle, stop conditions, and the timing contract."""

    def __init__(
        self,
        model: LocalMindTransformer,
        eos_token_id: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.eval()
        self.cfg: ModelConfig = model.cfg
        self.device = torch.device(device)
        self.eos_token_id = eos_token_id

    # -- helpers --------------------------------------------------------------------
    def _ids(self, ids: Sequence[int]) -> Tensor:
        return torch.tensor([list(ids)], dtype=torch.long, device=self.device)

    def _stop(self, token: int, params: SamplingParams) -> bool:
        if token in params.stop_token_ids:
            return True
        return (
            not params.ignore_eos and self.eos_token_id is not None and token == self.eos_token_id
        )

    @staticmethod
    def _params(max_new_tokens: int, kw: dict[str, Any]) -> SamplingParams:
        params = kw.pop("params", None)
        if isinstance(params, SamplingParams):
            return params.model_copy(update={"max_new_tokens": max_new_tokens})
        allowed = set(SamplingParams.model_fields) - {"max_new_tokens"}
        return SamplingParams(
            max_new_tokens=max_new_tokens, **{k: v for k, v in kw.items() if k in allowed}
        )

    # -- Protocol -------------------------------------------------------------------
    def generate(self, prompt_ids: list[int], max_new_tokens: int, **kw: Any) -> list[int]:
        return self.generate_detailed(prompt_ids, max_new_tokens, **kw).token_ids

    def generate_detailed(
        self, prompt_ids: list[int], max_new_tokens: int, **kw: Any
    ) -> GenerationResult:  # pragma: no cover - overridden
        raise NotImplementedError


class NaiveEngine(_BaseEngine):
    """No KV cache. Every step re-runs the model over the entire prefix.

    This is the denominator for step 1 of implementation.md section 10 and it is *not*
    a strawman: it is precisely what a from-scratch generation loop does before anyone
    thinks about caching.
    """

    def generate_detailed(
        self, prompt_ids: list[int], max_new_tokens: int, **kw: Any
    ) -> GenerationResult:
        params = self._params(max_new_tokens, dict(kw))
        gen = make_generator(params.seed, self.device)
        ids = list(prompt_ids)
        out: list[int] = []
        itls: list[float] = []
        forwards = 0
        tokens_fwd = 0
        t_start = time.perf_counter()
        ttft = 0.0

        with torch.no_grad():
            for step in range(params.max_new_tokens):
                t0 = time.perf_counter()
                if len(ids) > self.cfg.max_seq_len:
                    return GenerationResult(
                        token_ids=out,
                        prompt_len=len(prompt_ids),
                        ttft_s=ttft,
                        itl_s=itls,
                        total_s=time.perf_counter() - t_start,
                        finish_reason="length",
                        prefill_forwards=1,
                        decode_forwards=forwards - 1,
                        tokens_forwarded=tokens_fwd,
                    )
                logits = self.model(self._ids(ids)).logits[0, -1]
                forwards += 1
                tokens_fwd += len(ids)
                token = sample_token(logits, params, gen, ids)
                dt = time.perf_counter() - t0
                if step == 0:
                    ttft = dt
                else:
                    itls.append(dt)
                out.append(token)
                ids.append(token)
                if self._stop(token, params):
                    return GenerationResult(
                        token_ids=out,
                        prompt_len=len(prompt_ids),
                        ttft_s=ttft,
                        itl_s=itls,
                        total_s=time.perf_counter() - t_start,
                        finish_reason="stop",
                        prefill_forwards=1,
                        decode_forwards=forwards - 1,
                        tokens_forwarded=tokens_fwd,
                    )
        return GenerationResult(
            token_ids=out,
            prompt_len=len(prompt_ids),
            ttft_s=ttft,
            itl_s=itls,
            total_s=time.perf_counter() - t_start,
            finish_reason="length",
            prefill_forwards=1,
            decode_forwards=forwards - 1,
            tokens_forwarded=tokens_fwd,
        )


class CachedEngine(_BaseEngine):
    """Prefill once into a preallocated contiguous cache, then decode one token at a time.

    ``prefill_chunk_size`` caps how many prompt tokens enter one forward (step 5,
    chunked prefill). At the single-request level chunking is latency-neutral-to-slightly
    worse; its payoff is in :mod:`localmind.inference.scheduler`, where an unchunked
    2000-token prefill would otherwise monopolise an iteration and spike every other
    user's TPOT.

    ``cache="dynamic"`` keeps the model's own concatenated tensors instead of copying
    into a preallocated buffer -- the A/B that isolates *storage strategy* from *having
    a cache at all*.
    """

    def __init__(
        self,
        model: LocalMindTransformer,
        eos_token_id: int | None = None,
        device: torch.device | str = "cpu",
        prefill_chunk_size: int | None = None,
        cache: str = "contiguous",
        max_len: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(model, eos_token_id, device)
        if cache not in ("contiguous", "dynamic"):
            raise ValueError(f"cache must be 'contiguous' or 'dynamic', got {cache!r}")
        self.cache_kind = cache
        self.prefill_chunk_size = prefill_chunk_size
        self.max_len = int(max_len or model.cfg.max_seq_len)
        self.dtype = dtype

    def _new_cache(self) -> ContiguousKVCache | DynamicKVCache:
        if self.cache_kind == "dynamic":
            return DynamicKVCache()
        return ContiguousKVCache(
            self.cfg, batch_size=1, max_len=self.max_len, dtype=self.dtype, device=self.device
        )

    def generate_detailed(
        self, prompt_ids: list[int], max_new_tokens: int, **kw: Any
    ) -> GenerationResult:
        result = GenerationResult(token_ids=[], prompt_len=len(prompt_ids))
        for chunk in self.stream(prompt_ids, max_new_tokens, _result=result, **kw):
            result.token_ids.append(chunk.token_id)
        return result

    def stream(
        self, prompt_ids: list[int], max_new_tokens: int, **kw: Any
    ) -> Iterator[StreamChunk]:
        """Yield tokens as they are produced. This is what ``/v1/chat/completions`` SSE uses."""
        result: GenerationResult | None = kw.pop("_result", None)
        params = self._params(max_new_tokens, dict(kw))
        gen = make_generator(params.seed, self.device)
        cache = self._new_cache()
        ids = list(prompt_ids)
        t_start = time.perf_counter()
        prefill_forwards = 0
        decode_forwards = 0
        tokens_fwd = 0

        with torch.no_grad():
            # -- prefill (optionally chunked) --------------------------------------
            chunk_size = self.prefill_chunk_size or len(ids)
            logits: Tensor | None = None
            for start in range(0, len(ids), max(1, chunk_size)):
                piece = ids[start : start + max(1, chunk_size)]
                out = self.model(self._ids(piece), past_kvs=cache.as_past(), use_cache=True)
                assert out.kv_caches is not None
                cache.extend_from(out.kv_caches)
                logits = out.logits[0, -1]
                prefill_forwards += 1
                tokens_fwd += len(piece)
            assert logits is not None

            token = sample_token(logits, params, gen, ids)
            ttft = time.perf_counter() - t_start
            ids.append(token)
            stop = self._stop(token, params)
            yield StreamChunk(token, 0, ttft, "stop" if stop else None)

            itls: list[float] = []
            finish = "stop" if stop else "length"
            if not stop:
                for step in range(1, params.max_new_tokens):
                    if cache.length >= self.max_len:
                        finish = "length"
                        break
                    t0 = time.perf_counter()
                    out = self.model(self._ids([token]), past_kvs=cache.as_past(), use_cache=True)
                    assert out.kv_caches is not None
                    cache.extend_from(out.kv_caches)
                    decode_forwards += 1
                    tokens_fwd += 1
                    token = sample_token(out.logits[0, -1], params, gen, ids)
                    dt = time.perf_counter() - t0
                    itls.append(dt)
                    ids.append(token)
                    stop = self._stop(token, params)
                    finish = "stop" if stop else "length"
                    yield StreamChunk(token, step, dt, "stop" if stop else None)
                    if stop:
                        break

        if result is not None:
            result.ttft_s = ttft
            result.itl_s = itls
            result.total_s = time.perf_counter() - t_start
            result.finish_reason = finish
            result.prefill_forwards = prefill_forwards
            result.decode_forwards = decode_forwards
            result.tokens_forwarded = tokens_fwd


class PagedEngine(_BaseEngine):
    """Single-request decode loop backed by :class:`PagedKVCache`.

    Functionally identical output to :class:`CachedEngine` -- the test suite asserts
    token-for-token equality -- but the cache lives in a shared block pool, which is what
    lets :mod:`localmind.inference.scheduler` run many of these concurrently without
    reserving ``max_seq_len`` for each.
    """

    def __init__(
        self,
        model: LocalMindTransformer,
        eos_token_id: int | None = None,
        device: torch.device | str = "cpu",
        num_blocks: int = 512,
        block_size: int = DEFAULT_BLOCK_SIZE,
        dtype: torch.dtype = torch.float32,
        pool: PagedKVCache | None = None,
    ) -> None:
        super().__init__(model, eos_token_id, device)
        self.pool = pool or PagedKVCache(
            self.cfg, num_blocks=num_blocks, block_size=block_size, dtype=dtype, device=device
        )
        self._seq_counter = 0

    def generate_detailed(
        self, prompt_ids: list[int], max_new_tokens: int, **kw: Any
    ) -> GenerationResult:
        params = self._params(max_new_tokens, dict(kw))
        gen = make_generator(params.seed, self.device)
        self._seq_counter += 1
        seq_id = f"paged-{self._seq_counter}"
        self.pool.add_sequence(seq_id)
        ids = list(prompt_ids)
        out_ids: list[int] = []
        itls: list[float] = []
        t_start = time.perf_counter()
        tokens_fwd = 0
        decode_forwards = 0
        finish = "length"
        ttft = 0.0
        try:
            with torch.no_grad():
                past: list[LayerKV] | None = self.pool.gather(seq_id)
                out = self.model(self._ids(ids), past_kvs=past, use_cache=True)
                assert out.kv_caches is not None
                self.pool.append(seq_id, out.kv_caches)
                tokens_fwd += len(ids)
                token = sample_token(out.logits[0, -1], params, gen, ids)
                ttft = time.perf_counter() - t_start
                out_ids.append(token)
                ids.append(token)
                if self._stop(token, params):
                    finish = "stop"
                else:
                    for _ in range(1, params.max_new_tokens):
                        t0 = time.perf_counter()
                        past = self.pool.gather(seq_id)
                        out = self.model(self._ids([token]), past_kvs=past, use_cache=True)
                        assert out.kv_caches is not None
                        self.pool.append(seq_id, out.kv_caches)
                        decode_forwards += 1
                        tokens_fwd += 1
                        token = sample_token(out.logits[0, -1], params, gen, ids)
                        itls.append(time.perf_counter() - t0)
                        out_ids.append(token)
                        ids.append(token)
                        if self._stop(token, params):
                            finish = "stop"
                            break
        finally:
            self.pool.free_sequence(seq_id)
        return GenerationResult(
            token_ids=out_ids,
            prompt_len=len(prompt_ids),
            ttft_s=ttft,
            itl_s=itls,
            total_s=time.perf_counter() - t_start,
            finish_reason=finish,
            prefill_forwards=1,
            decode_forwards=decode_forwards,
            tokens_forwarded=tokens_fwd,
        )


def build_engine(
    model: LocalMindTransformer,
    kind: str = "cached",
    eos_token_id: int | None = None,
    device: torch.device | str = "cpu",
    **kw: Any,
) -> GenerationEngine:
    """Factory used by ``bench.py`` and ``server.py`` so engine choice is one string."""
    if kind == "naive":
        return NaiveEngine(model, eos_token_id, device)
    if kind in ("cached", "contiguous"):
        return CachedEngine(model, eos_token_id, device, cache="contiguous", **kw)
    if kind == "dynamic":
        return CachedEngine(model, eos_token_id, device, cache="dynamic", **kw)
    if kind == "paged":
        return PagedEngine(model, eos_token_id, device, **kw)
    raise ValueError(f"unknown engine kind {kind!r}")
