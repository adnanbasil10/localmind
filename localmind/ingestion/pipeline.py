"""End-to-end ingestion orchestrator (implementation.md §11):

    parse (route by doc type, log which parser handled what)
      -> extract tables (markdown + normalized rows, with provenance)
      -> caption figures (indexed text; crop kept for citation display)
      -> chunk (one of the five strategies in `chunking.py`)
      -> emit `IngestionResult`, including `PageImageRef` handoff records

**ColQwen2 handoff.** The differentiator visual-retrieval arm (implementation.md
§11: "render each page to an image, embed into multi-vector patch embeddings")
belongs to `localmind/retrieval/colqwen.py`, owned by another task built
concurrently. This module's job in that handoff is narrow and one-directional:
`ParsedDocument.page_images` (populated by a real PDF parser rendering pages --
none is installed here) is surfaced as `IngestionResult.page_images`, a list of
`PageImageRef(doc_id, page, uri, width, height)`. `localmind/retrieval/colqwen.py`
is expected to read that list and do its own embedding/indexing; this module
never imports `localmind.retrieval` and never embeds or indexes a page image
itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from localmind.ingestion.chunking import (
    Chunk,
    ChunkStrategyName,
    Embedder,
    chunk_document,
    chunk_stats,
)
from localmind.ingestion.contextualize import Contextualizer
from localmind.ingestion.parse import (
    ParsedDocument,
    ParserRegistry,
    ParserStrategy,
    default_registry,
)
from localmind.ingestion.parse.tables import ExtractedTable, extract_tables_from_blocks
from localmind.ingestion.parse.vlm_caption import (
    Captioner,
    CaptionResult,
    FigureCrop,
    caption_figures,
    figures_from_blocks,
)

logger = logging.getLogger("localmind.ingestion.pipeline")

__all__ = [
    "IngestionResult",
    "PageImageRef",
    "PipelineConfig",
    "ingest_document",
    "ingest_parsed",
    "ingest_text",
]


class PipelineConfig(BaseModel):
    """Ingestion knobs. Controller owns `configs/`, so this ships as a
    pydantic model with sensible defaults rather than a `configs/ingestion.yaml`
    -- construct one directly, or (once the controller adds the YAML file)
    `PipelineConfig.model_validate(yaml.safe_load(...))` works unchanged."""

    model_config = ConfigDict(frozen=True)

    chunk_strategy: ChunkStrategyName = "recursive"
    chunk_size: int = 512
    chunk_overlap_frac: float = 0.0
    parser_strategy: ParserStrategy = "structure"

    @property
    def chunk_overlap(self) -> int:
        return round(self.chunk_size * self.chunk_overlap_frac)


class PageImageRef(BaseModel):
    """One rendered page, ready for `localmind/retrieval/colqwen.py` to embed.
    See module docstring for the handoff contract."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    page: int
    uri: str = ""
    width: int | None = None
    height: int | None = None


class IngestionResult(BaseModel):
    """Everything one document produced: which parser handled it, its tables
    (markdown + provenance), its figure captions, its chunks, and the page
    images handed off for visual retrieval."""

    doc_id: str
    parser: str
    doc_type: str
    n_blocks: int
    tables: list[ExtractedTable] = Field(default_factory=list)
    figure_crops: list[FigureCrop] = Field(default_factory=list)
    figure_captions: list[CaptionResult] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    page_images: list[PageImageRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def stats(self) -> dict[str, float]:
        """Real, measured chunk statistics for this document (see
        `chunking.chunk_stats`)."""
        return chunk_stats(self.chunks)


def ingest_parsed(
    parsed: ParsedDocument,
    *,
    config: PipelineConfig | None = None,
    captioner: Captioner | None = None,
    contextualizer: Contextualizer | None = None,
    embedder: Embedder | None = None,
) -> IngestionResult:
    """Run tables/figures/chunking over an already-parsed document. Split out
    from `ingest_document` so `ingest_text` (the synthetic/offline entry
    point) and tests can skip the filesystem + parser-routing step."""
    cfg = config or PipelineConfig()
    tables = extract_tables_from_blocks(parsed.blocks, doc_id=parsed.doc_id, parser=parsed.parser)
    crops = figures_from_blocks(parsed.blocks, doc_id=parsed.doc_id)
    captions = caption_figures(crops, captioner) if crops else []
    chunks = chunk_document(
        parsed.text,
        parsed.doc_id,
        cfg.chunk_strategy,
        size=cfg.chunk_size,
        overlap=cfg.chunk_overlap,
        embedder=embedder,
        contextualizer=contextualizer,
    )
    page_images = [
        PageImageRef(
            doc_id=parsed.doc_id, page=pi.page, uri=pi.uri, width=pi.width, height=pi.height
        )
        for pi in parsed.page_images
    ]
    logger.info(
        "ingested %s: parser=%s blocks=%d tables=%d figures=%d chunks=%d "
        "(strategy=%s size=%d overlap=%d)",
        parsed.doc_id,
        parsed.parser,
        len(parsed.blocks),
        len(tables),
        len(crops),
        len(chunks),
        cfg.chunk_strategy,
        cfg.chunk_size,
        cfg.chunk_overlap,
    )
    return IngestionResult(
        doc_id=parsed.doc_id,
        parser=parsed.parser,
        doc_type=parsed.doc_type,
        n_blocks=len(parsed.blocks),
        tables=tables,
        figure_crops=crops,
        figure_captions=captions,
        chunks=chunks,
        page_images=page_images,
        warnings=list(parsed.warnings),
    )


def ingest_document(
    path: str | Path,
    *,
    registry: ParserRegistry | None = None,
    config: PipelineConfig | None = None,
    captioner: Captioner | None = None,
    contextualizer: Contextualizer | None = None,
    embedder: Embedder | None = None,
    doc_id: str | None = None,
) -> IngestionResult:
    """Full pipeline from a file on disk: route -> parse -> tables/figures ->
    chunk. `registry` defaults to `default_registry()` (every real backend
    registered, falling back to `PlainTextParser`)."""
    reg = registry or default_registry()
    cfg = config or PipelineConfig()
    parsed = reg.parse(Path(path), doc_id=doc_id, strategy=cfg.parser_strategy)
    return ingest_parsed(
        parsed, config=cfg, captioner=captioner, contextualizer=contextualizer, embedder=embedder
    )


def ingest_text(
    text: str,
    doc_id: str,
    *,
    config: PipelineConfig | None = None,
    captioner: Captioner | None = None,
    contextualizer: Contextualizer | None = None,
    embedder: Embedder | None = None,
) -> IngestionResult:
    """Ingest an in-memory string through the plaintext fallback parser -- no
    filesystem I/O, the primary offline entry point for synthetic documents
    (used by `tests/test_ingestion.py` and this task's verification run)."""
    from localmind.ingestion.parse import PlainTextParser

    parsed = PlainTextParser().parse_text(text, doc_id=doc_id)
    return ingest_parsed(
        parsed, config=config, captioner=captioner, contextualizer=contextualizer, embedder=embedder
    )
