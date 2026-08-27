"""LocalMind multimodal ingestion (implementation.md §11, Phase 7).

    parse (route by doc type, log which parser handled what)
      -> extract tables (markdown + normalized Postgres rows, with provenance)
      -> caption figures (indexed text; crop kept for citation display)
      -> chunk (5 strategies: fixed / recursive / semantic / late / contextual)
      -> [optional] contextualize (situating context before embedding)
      -> IngestionResult, including PageImageRef handoff records for
         `localmind/retrieval/colqwen.py` (owned by another task; NOT
         imported here -- see `pipeline.py`'s module docstring)

No submodule imports torch, `localmind.model`, or `localmind.retrieval` at
module scope -- `import localmind.ingestion` works with only numpy + pydantic
(+ this repo's `localmind.eval.stats`, itself numpy-only) installed. Every
external dependency (Docling / PyMuPDF4LLM / Marker / Surya / PaddleOCR /
SmolVLM / Qwen2.5-VL / Postgres) sits behind an injectable Protocol with a
deterministic fake, so the whole pipeline runs offline in tests.

`localmind.ingestion.chunking` is the measurable core of this phase:
`run_strategy_ablation` and `run_size_overlap_ablation` report nDCG@10 on a
**synthetic, code-generated harness** (deterministic bag-of-words embedder,
ground truth defined by construction) -- clearly not a real corpus result,
but enough to exercise and compare the five chunking mechanisms and the size
x overlap grid entirely offline. See `chunking.py`'s module docstring.
"""

from localmind.ingestion.chunking import (
    CHUNK_SIZES,
    OVERLAP_FRACS,
    STRATEGY_NAMES,
    Chunk,
    ChunkStrategyName,
    Embedder,
    FakeHashEmbedder,
    HeatmapResult,
    chunk_contextual,
    chunk_document,
    chunk_fixed,
    chunk_late,
    chunk_recursive,
    chunk_semantic,
    chunk_stats,
    ndcg_at_k,
    rows_to_markdown,
    run_size_overlap_ablation,
    run_strategy_ablation,
)
from localmind.ingestion.contextualize import (
    LOCAL_31M_COST_MODEL,
    QWEN3_4B_API_COST_MODEL,
    ContextualizeComparison,
    Contextualizer,
    ContextualizeRecord,
    CostModel,
    HeuristicContextualizer,
    SimulatedContextualizer,
    benchmark_contextualizers,
    contextualize_chunks,
)
from localmind.ingestion.pipeline import (
    IngestionResult,
    PageImageRef,
    PipelineConfig,
    ingest_document,
    ingest_parsed,
    ingest_text,
)

__all__ = [
    "CHUNK_SIZES",
    "LOCAL_31M_COST_MODEL",
    "OVERLAP_FRACS",
    "QWEN3_4B_API_COST_MODEL",
    "STRATEGY_NAMES",
    "Chunk",
    "ChunkStrategyName",
    "ContextualizeComparison",
    "ContextualizeRecord",
    "Contextualizer",
    "CostModel",
    "Embedder",
    "FakeHashEmbedder",
    "HeatmapResult",
    "HeuristicContextualizer",
    "IngestionResult",
    "PageImageRef",
    "PipelineConfig",
    "SimulatedContextualizer",
    "benchmark_contextualizers",
    "chunk_contextual",
    "chunk_document",
    "chunk_fixed",
    "chunk_late",
    "chunk_recursive",
    "chunk_semantic",
    "chunk_stats",
    "contextualize_chunks",
    "ingest_document",
    "ingest_parsed",
    "ingest_text",
    "ndcg_at_k",
    "rows_to_markdown",
    "run_size_overlap_ablation",
    "run_strategy_ablation",
]
