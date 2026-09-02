"""Iteration-level (continuous) batching, chunked prefill, and the load generator.

implementation.md section 10 steps 4-5.

Static batching -- the thing this replaces
-----------------------------------------
A static batcher collects N requests, runs them together until *every one* has finished,
then takes the next N. The batch is only as fast as its slowest member, so a request
generating 8 tokens sits in the GPU/CPU occupying a slot until the one generating 400
tokens is done. Utilisation collapses as the length distribution widens, and a request
that arrives one iteration after the batch starts waits for the whole batch.

Continuous batching (Yu et al., Orca 2022) schedules at *iteration* granularity: after
every single forward step the scheduler evicts finished sequences and admits waiting ones
into the freed slots. No request waits for an unrelated request to finish.

Chunked prefill (Agrawal et al., Sarathi-Serve 2024)
----------------------------------------------------
Prefill is compute-bound and decode is memory-bound, so a long prefill and many decodes
*coexist* well -- but only if the prefill is broken up. Without chunking, one 1024-token
prompt occupies an entire iteration and every other user's inter-token latency takes a
one-shot hit equal to the whole prefill. That hit lands in the tail, which is why the
metric to report is **p99 TPOT**, not the mean. With chunking, the prefill is spread over
several iterations under a shared token budget and the tail flattens.

A note on how decode is batched here
------------------------------------
True ragged batching needs a paged-attention kernel that walks a block table inside the
attention op. ``localmind/model/attention.py`` is Phase 2-owned and takes a dense
``past_kv`` with one shared length, so this scheduler batches by **exact length bucket**:
running sequences whose cache lengths coincide are stacked into one ``(B, 1)`` forward.
That is numerically exact and requires no model change; its cost is that the achieved
batch size is lower than a real engine would get, and ``achieved_batch_size`` is reported
alongside throughput so the number is never mistaken for a kernel-batched one.
"""

from __future__ import annotations

import itertools
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

from localmind.inference.kv_cache import (
    DEFAULT_BLOCK_SIZE,
    LayerKV,
    OutOfBlocksError,
    PagedKVCache,
)
from localmind.inference.prefix_cache import RadixPrefixCache
from localmind.inference.sampling import SamplingParams, make_generator, sample_token
from localmind.model import LocalMindTransformer

__all__ = [
    "ContinuousBatchingScheduler",
    "LoadReport",
    "Request",
    "RequestState",
    "SchedulerConfig",
    "StaticBatchScheduler",
    "percentiles",
    "poisson_arrivals",
    "run_load",
]


class RequestState(StrEnum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    PREEMPTED = "preempted"


@dataclass
class Request:
    """One in-flight generation, with everything the metrics need attached to it."""

    request_id: str
    prompt_ids: list[int]
    params: SamplingParams = field(default_factory=SamplingParams)
    arrival_time: float = 0.0
    state: RequestState = RequestState.WAITING

    # progress
    prefill_done: int = 0
    output_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None

    # metrics
    schedule_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    token_times: list[float] = field(default_factory=list)
    cached_prefix_len: int = 0

    _generator: torch.Generator | None = field(default=None, repr=False)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def ttft_s(self) -> float:
        if self.first_token_time is None:
            return float("nan")
        return self.first_token_time - self.arrival_time

    @property
    def itls(self) -> list[float]:
        return [b - a for a, b in itertools.pairwise(self.token_times)]

    @property
    def tpot_s(self) -> float:
        gaps = self.itls
        return sum(gaps) / len(gaps) if gaps else float("nan")

    @property
    def latency_s(self) -> float:
        if self.finish_time is None:
            return float("nan")
        return self.finish_time - self.arrival_time

    def meets_slo(self, ttft_budget_s: float, tpot_budget_s: float) -> bool:
        tpot = self.tpot_s
        ok_tpot = tpot != tpot or tpot < tpot_budget_s  # NaN == single-token output
        return self.ttft_s < ttft_budget_s and ok_tpot


class SchedulerConfig(BaseModel):
    """Every scheduling knob, from config -- no magic numbers in the loop (CONVENTIONS.md)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_num_seqs: int = Field(default=8, gt=0)
    #: Token budget per iteration. Prefill chunks and decode steps draw from the same
    #: budget, which is what makes the two workloads share an iteration gracefully.
    max_num_batched_tokens: int = Field(default=256, gt=0)
    enable_chunked_prefill: bool = True
    prefill_chunk_size: int = Field(default=64, gt=0)
    block_size: int = Field(default=DEFAULT_BLOCK_SIZE, gt=0)
    num_blocks: int = Field(default=1024, gt=0)
    enable_prefix_cache: bool = False
    #: Goodput SLO (implementation.md section 10).
    ttft_slo_s: float = Field(default=0.5, gt=0)
    tpot_slo_s: float = Field(default=0.05, gt=0)


@dataclass
class LoadReport:
    """Aggregate serving metrics. Percentiles only -- never a single-request timing."""

    n_requests: int
    wall_s: float
    prompt_tokens: int
    generated_tokens: int
    ttft: dict[str, float]
    tpot: dict[str, float]
    #: Percentiles over EVERY individual inter-token gap, pooled across requests.
    #: Distinct from ``tpot``, which percentiles each request's *mean* gap. A single
    #: scheduler stall shows up in ``itl["p99"]`` and is averaged away in ``tpot``, so
    #: ITL is the metric that can see a prefill blocking someone else's decode.
    itl: dict[str, float]
    latency: dict[str, float]
    output_throughput_tok_s: float
    request_throughput_req_s: float
    goodput_req_s: float
    goodput_fraction: float
    achieved_batch_size: float
    iterations: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


def percentiles(values: Sequence[float], qs: Sequence[float] = (50, 95, 99)) -> dict[str, float]:
    """p50/p95/p99 with linear interpolation. NaNs (single-token outputs) are dropped."""
    clean = sorted(v for v in values if v == v)
    if not clean:
        return {f"p{q:g}": float("nan") for q in qs} | {"mean": float("nan"), "n": 0}
    out: dict[str, float] = {"mean": sum(clean) / len(clean), "n": float(len(clean))}
    for q in qs:
        pos = (q / 100.0) * (len(clean) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(clean) - 1)
        out[f"p{q:g}"] = clean[lo] + (clean[hi] - clean[lo]) * (pos - lo)
    return out


def poisson_arrivals(rate_per_s: float, n: int, seed: int = 0) -> list[float]:
    """Arrival offsets of a Poisson process: exponential inter-arrival gaps.

    A Poisson generator is the point of the exercise -- a fixed-rate loop hides exactly
    the queueing that produces a p99, and the p99 is what a serving team is judged on.
    """
    rng = random.Random(seed)
    t = 0.0
    out: list[float] = []
    for _ in range(n):
        t += rng.expovariate(rate_per_s) if rate_per_s > 0 else 0.0
        out.append(t)
    return out


class _BatchExecutor:
    """Forward + sampling shared by both schedulers, so the *only* difference between
    them is admission policy.

    Comparing a continuous scheduler that batches against a static one that does not
    would credit the batching win to the scheduling policy. Both get the same
    length-bucketed batched decode here; the remaining delta is purely iteration-level
    admission and eviction.
    """

    model: LocalMindTransformer
    cfg: Any
    pool: PagedKVCache
    device: torch.device
    eos_token_id: int | None

    def _forward_one(self, req: Request, piece: Sequence[int]) -> torch.Tensor:
        past = self.pool.gather(req.request_id)
        ids = torch.tensor([list(piece)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(ids, past_kvs=past, use_cache=True)
        assert out.kv_caches is not None
        self.pool.append(req.request_id, out.kv_caches)
        return out.logits[0, -1]

    def _forward_batch(self, group: Sequence[Request]) -> torch.Tensor:
        """One ``(B, 1)`` decode forward over a bucket of equal-length sequences."""
        if len(group) == 1:
            return self._forward_one(group[0], [_last_token(group[0])]).unsqueeze(0)
        pasts = [self.pool.gather(r.request_id) for r in group]
        stacked: list[LayerKV] | None = None
        if pasts[0] is not None:
            stacked = []
            for layer in range(self.cfg.n_layers):
                ks = torch.cat([p[layer][0] for p in pasts if p is not None], dim=0)
                vs = torch.cat([p[layer][1] for p in pasts if p is not None], dim=0)
                stacked.append((ks, vs))
        ids = torch.tensor([[_last_token(r)] for r in group], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(ids, past_kvs=stacked, use_cache=True)
        assert out.kv_caches is not None
        for i, req in enumerate(group):
            self.pool.append(req.request_id, out.kv_caches, batch_index=i)
        return out.logits[:, -1, :]

    def _decode_buckets(self, running: Sequence[Request]) -> list[list[Request]]:
        buckets: dict[int, list[Request]] = defaultdict(list)
        for req in running:
            buckets[self.pool.length_of(req.request_id)].append(req)
        return [group for _, group in sorted(buckets.items())]

    def _emit(self, req: Request, logits: torch.Tensor) -> None:
        token = sample_token(logits, req.params, req._generator, req.all_ids)
        now = time.perf_counter()
        if req.first_token_time is None:
            req.first_token_time = now
        req.token_times.append(now)
        req.output_ids.append(token)
        req.state = RequestState.DECODE
        stopped = token in req.params.stop_token_ids or (
            not req.params.ignore_eos
            and self.eos_token_id is not None
            and token == self.eos_token_id
        )
        if stopped:
            req.finish_reason = "stop"
            req.state = RequestState.FINISHED
        elif (
            len(req.output_ids) >= req.params.max_new_tokens
            or self.pool.length_of(req.request_id) + 1 > self.cfg.max_seq_len
        ):
            req.finish_reason = "length"
            req.state = RequestState.FINISHED


def _last_token(req: Request) -> int:
    return req.output_ids[-1] if req.output_ids else req.prompt_ids[-1]


class ContinuousBatchingScheduler(_BatchExecutor):
    """Iteration-level scheduler over a shared :class:`PagedKVCache` block pool.

    One ``step()`` is one scheduling decision plus the forwards it implies:

    1. evict finished sequences, returning their blocks to the pool;
    2. admit waiting requests while slots and blocks allow (with an optional prefix-cache
       lookup that can skip most of the prefill);
    3. spend the iteration's token budget on prefill chunks;
    4. spend what is left on one decode step for every running sequence, batched by
       exact cache-length bucket.
    """

    def __init__(
        self,
        model: LocalMindTransformer,
        config: SchedulerConfig | None = None,
        eos_token_id: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        prefix_cache: RadixPrefixCache | None = None,
    ) -> None:
        self.model = model.eval()
        self.cfg = model.cfg
        self.config = config or SchedulerConfig()
        self.eos_token_id = eos_token_id
        self.device = torch.device(device)
        self.pool = PagedKVCache(
            self.cfg,
            num_blocks=self.config.num_blocks,
            block_size=self.config.block_size,
            dtype=dtype,
            device=device,
        )
        self.prefix_cache = prefix_cache
        if self.config.enable_prefix_cache and self.prefix_cache is None:
            self.prefix_cache = RadixPrefixCache(self.cfg, device=device, dtype=dtype)

        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.iterations = 0
        self.batch_sizes: list[int] = []
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.preemptions = 0

    # -- queue management -----------------------------------------------------------
    def add_request(self, req: Request) -> None:
        req.state = RequestState.WAITING
        req._generator = make_generator(req.params.seed, self.device)
        self.waiting.append(req)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _admit(self) -> None:
        while self.waiting and len(self.running) < self.config.max_num_seqs:
            req = self.waiting[0]
            need = req.prompt_len + req.params.max_new_tokens
            if self.pool.allocator.num_free * self.config.block_size < need:
                if not self.running:
                    # Nothing to preempt and the pool cannot hold this request at all.
                    req.finish_reason = "insufficient_kv_cache"
                    req.state = RequestState.FINISHED
                    req.finish_time = time.perf_counter()
                    self.waiting.pop(0)
                    self.finished.append(req)
                    continue
                break
            self.waiting.pop(0)
            self.pool.add_sequence(req.request_id)
            req.schedule_time = time.perf_counter()
            req.state = RequestState.PREFILL
            if self.prefix_cache is not None:
                self._try_prefix_cache(req)
            self.running.append(req)

    def _try_prefix_cache(self, req: Request) -> None:
        assert self.prefix_cache is not None
        matched, dense = self.prefix_cache.match(req.prompt_ids)
        # Always leave at least one token to run through the model: the engine needs a
        # forward to produce logits for the next token, and a full-length hit would
        # leave nothing to compute.
        matched = min(matched, req.prompt_len - 1)
        if matched <= 0 or dense is None:
            return
        self.pool.write_dense(req.request_id, dense, matched)
        req.prefill_done = matched
        req.cached_prefix_len = matched

    def _evict_finished(self) -> None:
        still: list[Request] = []
        for req in self.running:
            if req.state is RequestState.FINISHED:
                if self.prefix_cache is not None:
                    dense = self.pool.gather(req.request_id)
                    if dense is not None:
                        self.prefix_cache.insert(req.prompt_ids, dense, req.prompt_len)
                self.pool.free_sequence(req.request_id)
                req.finish_time = time.perf_counter()
                self.finished.append(req)
            else:
                still.append(req)
        self.running = still

    # -- one iteration --------------------------------------------------------------
    def step(self) -> None:
        self._evict_finished()
        self._admit()
        if not self.running:
            self.iterations += 1
            return

        budget = self.config.max_num_batched_tokens
        prefilling = [r for r in self.running if r.state is RequestState.PREFILL]
        decoding = [r for r in self.running if r.state is RequestState.DECODE]

        # 3. prefill.
        #
        # Chunking ON: a prefill takes at most `prefill_chunk_size` tokens from the shared
        # budget and decodes run in the same iteration -- prefill and decode co-schedule
        # (Sarathi-Serve).
        #
        # Chunking OFF: this reproduces the classic prefill-prioritising scheduler. The
        # whole prompt goes through in ONE forward and that iteration runs no decodes at
        # all, so every in-flight request eats the entire prefill as a single stall. That
        # is the defect chunked prefill exists to remove, and it has to be modelled
        # faithfully or the comparison is rigged in chunking's favour.
        prefill_only = False
        for req in prefilling:
            if budget <= 0:
                break
            remaining = req.prompt_len - req.prefill_done
            if self.config.enable_chunked_prefill:
                chunk = min(remaining, budget, self.config.prefill_chunk_size)
            else:
                chunk = remaining
                prefill_only = True
            if chunk <= 0:
                continue
            piece = req.prompt_ids[req.prefill_done : req.prefill_done + chunk]
            logits = self._forward_one(req, piece)
            req.prefill_done += chunk
            budget -= chunk
            self.prefill_tokens += chunk
            if req.prefill_done >= req.prompt_len:
                self._emit(req, logits)
            if prefill_only:
                break

        # 4. decode, batched by exact length bucket
        if budget > 0 and decoding and not prefill_only:
            for group in self._decode_buckets(decoding):
                group = group[: max(1, budget)]
                logits = self._forward_batch(group)
                budget -= len(group)
                self.decode_tokens += len(group)
                self.batch_sizes.append(len(group))
                for i, req in enumerate(group):
                    self._emit(req, logits[i])
                if budget <= 0:
                    break

        self.iterations += 1

    def run_to_completion(self, max_iterations: int = 100_000) -> list[Request]:
        """Drain the queue. Used when every request is already present at t=0."""
        n = 0
        while self.has_work and n < max_iterations:
            self.step()
            n += 1
        self._evict_finished()
        return self.finished


class StaticBatchScheduler(_BatchExecutor):
    """The baseline: fill a batch, run it to completion, then start the next one.

    Deliberately simple and deliberately bad -- it is the denominator for the
    continuous-batching delta, and it is what almost every hand-rolled serving loop does.
    """

    def __init__(
        self,
        model: LocalMindTransformer,
        batch_size: int = 8,
        eos_token_id: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        num_blocks: int = 1024,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        self.model = model.eval()
        self.cfg = model.cfg
        self.batch_size = batch_size
        self.eos_token_id = eos_token_id
        self.device = torch.device(device)
        self.pool = PagedKVCache(
            self.cfg, num_blocks=num_blocks, block_size=block_size, dtype=dtype, device=device
        )
        self.batch_sizes: list[int] = []
        self.iterations = 0

    def run(
        self, requests: Sequence[Request], available: Sequence[float] | None = None
    ) -> list[Request]:
        """Run ``requests`` in fixed batches. ``available`` gates arrival like real load."""
        pending = list(requests)
        done: list[Request] = []
        while pending:
            if available is not None:
                start = time.perf_counter()
                ready = [r for r in pending if r.arrival_time <= start]
                while not ready:
                    time.sleep(0.001)
                    ready = [r for r in pending if r.arrival_time <= time.perf_counter()]
                batch = ready[: self.batch_size]
            else:
                batch = pending[: self.batch_size]
            for r in batch:
                pending.remove(r)
            self._run_batch(batch)
            done.extend(batch)
        return done

    def _run_batch(self, batch: Sequence[Request]) -> None:
        for req in batch:
            req._generator = make_generator(req.params.seed, self.device)
            req.schedule_time = time.perf_counter()
            self.pool.add_sequence(req.request_id)
        try:
            # Prefill each prompt (lengths differ, so one forward each).
            for req in batch:
                logits = self._forward_one(req, req.prompt_ids)
                self._emit(req, logits)
                self.iterations += 1
            # Decode until the SLOWEST member finishes. Requests that completed early keep
            # their slot and no waiting request may take it: that head-of-line blocking is
            # exactly the defect being measured.
            while any(r.state is not RequestState.FINISHED for r in batch):
                alive = [r for r in batch if r.state is not RequestState.FINISHED]
                for group in self._decode_buckets(alive):
                    self.batch_sizes.append(len(group))
                    logits = self._forward_batch(group)
                    for i, req in enumerate(group):
                        self._emit(req, logits[i])
                self.iterations += 1
        finally:
            for req in batch:
                req.finish_time = time.perf_counter()
                self.pool.free_sequence(req.request_id)


def run_load(
    scheduler: ContinuousBatchingScheduler,
    requests: Sequence[Request],
    arrivals: Sequence[float] | None = None,
    max_iterations: int = 200_000,
) -> LoadReport:
    """Feed ``requests`` at Poisson-generated wall-clock offsets and step until drained.

    Arrival is enforced against the real clock rather than simulated, so queueing delay
    is genuine and shows up in TTFT where it belongs.
    """
    t0 = time.perf_counter()
    offsets = list(arrivals) if arrivals is not None else [0.0] * len(requests)
    pending = sorted(zip(offsets, range(len(requests)), strict=True))
    idx = 0
    n_iter = 0
    while (idx < len(pending) or scheduler.has_work) and n_iter < max_iterations:
        now = time.perf_counter() - t0
        while idx < len(pending) and pending[idx][0] <= now:
            req = requests[pending[idx][1]]
            req.arrival_time = t0 + pending[idx][0]
            scheduler.add_request(req)
            idx += 1
        if not scheduler.has_work:
            time.sleep(0.001)  # idle: wait for the next arrival rather than spin a core
            continue
        scheduler.step()
        n_iter += 1
    scheduler._evict_finished()
    wall = time.perf_counter() - t0
    return summarise(list(requests), wall, scheduler)


def summarise(
    requests: Sequence[Request],
    wall_s: float,
    scheduler: ContinuousBatchingScheduler | StaticBatchScheduler | None = None,
    ttft_slo_s: float = 0.5,
    tpot_slo_s: float = 0.05,
) -> LoadReport:
    gen = sum(len(r.output_ids) for r in requests)
    prompt = sum(r.prompt_len for r in requests)
    good = sum(1 for r in requests if r.meets_slo(ttft_slo_s, tpot_slo_s))
    batch_sizes = getattr(scheduler, "batch_sizes", []) if scheduler else []
    iterations = getattr(scheduler, "iterations", 0) if scheduler else 0
    extra: dict[str, Any] = {}
    if isinstance(scheduler, ContinuousBatchingScheduler):
        extra["prefill_tokens"] = scheduler.prefill_tokens
        extra["decode_tokens"] = scheduler.decode_tokens
        extra["kv_gather_calls"] = scheduler.pool.gather_calls
        extra["kv_internal_fragmentation"] = scheduler.pool.internal_fragmentation()
        if scheduler.prefix_cache is not None:
            extra["prefix_cache"] = scheduler.prefix_cache.stats()
    return LoadReport(
        n_requests=len(requests),
        wall_s=wall_s,
        prompt_tokens=prompt,
        generated_tokens=gen,
        ttft=percentiles([r.ttft_s for r in requests]),
        tpot=percentiles([r.tpot_s for r in requests]),
        itl=percentiles([gap for r in requests for gap in r.itls]),
        latency=percentiles([r.latency_s for r in requests]),
        output_throughput_tok_s=gen / wall_s if wall_s > 0 else float("nan"),
        request_throughput_req_s=len(requests) / wall_s if wall_s > 0 else float("nan"),
        goodput_req_s=good / wall_s if wall_s > 0 else float("nan"),
        goodput_fraction=good / len(requests) if requests else float("nan"),
        achieved_batch_size=sum(batch_sizes) / len(batch_sizes) if batch_sizes else 1.0,
        iterations=iterations,
        extra=extra,
    )


def make_requests(
    n: int,
    prompt_lens: Iterable[int],
    output_lens: Iterable[int],
    vocab_size: int,
    seed: int = 0,
    shared_prefix: Sequence[int] = (),
) -> list[Request]:
    """Synthetic request set. ``shared_prefix`` models the RAG system prompt for step 6."""
    rng = random.Random(seed)
    plens = list(prompt_lens)
    olens = list(output_lens)
    out: list[Request] = []
    for i in range(n):
        plen = max(1, plens[i % len(plens)] - len(shared_prefix))
        body = [rng.randrange(vocab_size) for _ in range(plen)]
        out.append(
            Request(
                request_id=f"req-{i}",
                prompt_ids=list(shared_prefix) + body,
                params=SamplingParams(
                    max_new_tokens=olens[i % len(olens)], temperature=0.0, seed=seed + i
                ),
            )
        )
    return out


def check_out_of_blocks(pool: PagedKVCache, need: int) -> None:
    """Raise the scheduler's preemption signal early rather than deep in a forward."""
    if pool.allocator.num_free < need:
        raise OutOfBlocksError(f"need {need} blocks, {pool.allocator.num_free} free")
