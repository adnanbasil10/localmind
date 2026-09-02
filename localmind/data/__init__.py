"""LocalMind data package (implementation.md §7, Phase 3).

    stream -> license filter -> language ID -> quality heuristics -> MinHash-LSH
    near-dedup -> PII scrub -> eval-set decontamination (13-gram) -> tokenize
    -> pack to seq_len with doc-boundary attention masking -> uint16 memmap shards

No submodule imports torch, `datasets`, or `datasketch` at module scope -- the package
always imports cleanly; the optional-extra dependencies (`datasets`, `datasketch`) are
imported lazily inside the specific functions that need them (`prepare.hf_streaming_source`,
`dedup.near_dedup`).

Public surface
--------------
`RawDoc`                     one document as it flows through filtering/dedup
`filter_license/_language/_quality`, `scrub_pii`, `scrub_docs`  pipeline stages 1-3, 5
`near_dedup`, `decontaminate`                                   pipeline stages 4, 6 (dedup.py)
`pack_documents`, `write_shard`, `build_doc_mask`                pipeline stages 8-9 (packing.py)
`PackedShardLoader`                                              resumable dataloader (loader.py)
`MixtureConfig`, `prepare_shards`                                config-driven orchestrator (prepare.py)
"""

from localmind.data.dedup import DecontamStats, DedupStats, decontaminate, near_dedup
from localmind.data.filter import (
    QualityThresholds,
    RawDoc,
    detect_language,
    filter_language,
    filter_license,
    filter_quality,
    quality_ok,
    scrub_docs,
    scrub_pii,
)
from localmind.data.loader import PackedShardLoader
from localmind.data.packing import (
    PackedRow,
    ShardMeta,
    build_doc_mask,
    doc_ids_from_boundaries,
    pack_documents,
    rows_per_shard,
    write_shard,
)
from localmind.data.prepare import (
    Manifest,
    MixtureConfig,
    MixtureEntry,
    ShardRecord,
    SourceSpec,
    TokenizerLike,
    hf_streaming_source,
    iter_mixture,
    prepare_shards,
    sources_from_mixture_config,
)

__all__ = [
    "DecontamStats",
    "DedupStats",
    "Manifest",
    "MixtureConfig",
    "MixtureEntry",
    "PackedRow",
    "PackedShardLoader",
    "QualityThresholds",
    "RawDoc",
    "ShardMeta",
    "ShardRecord",
    "SourceSpec",
    "TokenizerLike",
    "build_doc_mask",
    "decontaminate",
    "detect_language",
    "doc_ids_from_boundaries",
    "filter_language",
    "filter_license",
    "filter_quality",
    "hf_streaming_source",
    "iter_mixture",
    "near_dedup",
    "pack_documents",
    "prepare_shards",
    "quality_ok",
    "rows_per_shard",
    "scrub_docs",
    "scrub_pii",
    "sources_from_mixture_config",
    "write_shard",
]
