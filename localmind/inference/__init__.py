"""LocalMind inference engine (implementation.md section 10, Phase 6).

The whole chapter runs on a laptop CPU. That is the thesis, not a compromise: a 31M-param
model is ~31 MB at int8, and every technique below -- paged attention, continuous
batching, chunked prefill, prefix caching, speculative decoding, constrained decoding,
quantization -- is a *systems* result that does not need a GPU to be real.

The ladder, each rung benchmarked against the one below it
---------------------------------------------------------
======================  =======================================================
:mod:`~.engine`         naive (no cache) -> contiguous cache -> paged cache
:mod:`~.kv_cache`       dynamic / contiguous / paged storage + fragmentation math
:mod:`~.scheduler`      static batching -> continuous batching -> chunked prefill
:mod:`~.prefix_cache`   radix tree over the KV cache (RAG prompts repeat)
:mod:`~.speculative`    n-gram / prompt-lookup, then a draft model
:mod:`~.constrained`    JSON grammar FSM -> invalid JSON becomes unreachable
:mod:`~.quantize`       int8 / int4 weight-only, plus a GGUF exporter
:mod:`~.server`         OpenAI-compatible HTTP surface (FastAPI imported lazily)
:mod:`~.bench`          ``python -m localmind.inference.bench``
======================  =======================================================

:class:`~localmind.inference.engine.GenerationEngine` is the frozen cross-task Protocol
from CONVENTIONS.md. ``server`` is importable without ``fastapi`` installed; only
``create_app`` needs it.

Vocabulary used throughout, in code and in the artifact: **TTFT** (prefill-dominated),
**TPOT/ITL** (decode-dominated), **throughput**, and **goodput** (requests/s meeting an
SLO). All reported as p50/p95/p99 under generated load -- never as a single-request timing.
"""

from localmind.inference.constrained import (
    ConstrainedDecoder,
    JSONState,
    generate_json,
    is_valid_json,
    make_synthetic_vocab,
)
from localmind.inference.engine import (
    CachedEngine,
    GenerationEngine,
    GenerationResult,
    NaiveEngine,
    PagedEngine,
    StreamChunk,
    build_engine,
)
from localmind.inference.kv_cache import (
    DEFAULT_BLOCK_SIZE,
    BlockAllocator,
    ContiguousKVCache,
    DynamicKVCache,
    OutOfBlocksError,
    PagedKVCache,
    SequenceBlockTable,
    fragmentation_report,
    max_concurrent_sequences,
)
from localmind.inference.prefix_cache import RadixPrefixCache
from localmind.inference.sampling import SamplingParams, sample_token
from localmind.inference.scheduler import (
    ContinuousBatchingScheduler,
    LoadReport,
    Request,
    SchedulerConfig,
    StaticBatchScheduler,
    percentiles,
    poisson_arrivals,
    run_load,
)
from localmind.inference.speculative import (
    DraftModelProposer,
    NgramProposer,
    SpeculativeResult,
    speculative_generate,
)

__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "BlockAllocator",
    "CachedEngine",
    "ConstrainedDecoder",
    "ContiguousKVCache",
    "ContinuousBatchingScheduler",
    "DraftModelProposer",
    "DynamicKVCache",
    "GenerationEngine",
    "GenerationResult",
    "JSONState",
    "LoadReport",
    "NaiveEngine",
    "NgramProposer",
    "OutOfBlocksError",
    "PagedEngine",
    "PagedKVCache",
    "RadixPrefixCache",
    "Request",
    "SamplingParams",
    "SchedulerConfig",
    "SequenceBlockTable",
    "SpeculativeResult",
    "StaticBatchScheduler",
    "StreamChunk",
    "build_engine",
    "fragmentation_report",
    "generate_json",
    "is_valid_json",
    "make_synthetic_vocab",
    "max_concurrent_sequences",
    "percentiles",
    "poisson_arrivals",
    "run_load",
    "sample_token",
    "speculative_generate",
]
