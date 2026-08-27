"""Document parsers (implementation.md §11): route by document type, log which
parser handled what.

Real backends -- Docling (structure-preserving, reading order + table
structure), PyMuPDF4LLM (fast text), Marker (accurate, slower), Surya /
PaddleOCR (scans) -- are **not installed** in this environment and there is
**no network** to fetch them. Every real backend is a thin `Parser`
implementation whose `.parse()` imports its library lazily and lets the
resulting `ImportError` propagate; `ParserRegistry.parse` catches it and falls
through to the next candidate in priority order, ending at `PlainTextParser`
-- a pure-Python fallback that needs nothing but the stdlib and handles plain
text / Markdown. That is what makes the pipeline testable end to end offline,
and it is a zero-code-change story to go from "no libraries installed" to
"real Docling/Marker/Surya running": install the package, the same `parse()`
call now succeeds instead of raising.

Every successful parse is logged via `logging.getLogger("localmind.ingestion.parse")`
with exactly which parser handled which document and why -- the spec asks for
this explicitly.

Circular-import note: `ocr.py` needs `ParsedDocument`/`ParsedBlock` to build
its return value, and this module needs `SuryaOCRParser`/`PaddleOCRParser`
from `ocr.py` to populate the default registry. That's resolved by import
direction: this file defines every core type *before* it imports from `.ocr`
(at the bottom), and `ocr.py` only imports from this package lazily, inside
each `.parse()` method body (never at module scope) -- so there is no cycle
at import time in either direction. `tables.py` and `vlm_caption.py` only need
these types for annotations, so they import them under `TYPE_CHECKING`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DocType",
    "DoclingParser",
    "MarkerParser",
    "NoParserAvailableError",
    "PageImage",
    "ParsedBlock",
    "ParsedDocument",
    "Parser",
    "ParserRegistry",
    "ParserStrategy",
    "PlainTextParser",
    "PyMuPDF4LLMParser",
    "default_registry",
    "sniff_doc_type",
]

logger = logging.getLogger("localmind.ingestion.parse")

DocType = Literal["text", "markdown", "pdf", "pdf_scanned", "unknown"]
ParserStrategy = Literal["fast", "accurate", "structure"]


def sniff_doc_type(path: Path, *, scanned: bool = False) -> DocType:
    """Route purely by extension (+ an explicit `scanned` hint the caller
    supplies -- e.g. from a page-image-ratio heuristic upstream; detecting
    "is this PDF a scan" from bytes alone is out of scope here)."""
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown"):
        return "markdown"
    if suffix == ".txt":
        return "text"
    if suffix == ".pdf":
        return "pdf_scanned" if scanned else "pdf"
    return "unknown"


# --------------------------------------------------------------------------------------
# Core types
# --------------------------------------------------------------------------------------


class ParsedBlock(BaseModel):
    """One structural unit of a parsed document, in reading order."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["heading", "paragraph", "table", "figure", "code"]
    text: str
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    order: int = 0
    level: int | None = None  # heading level, 1-6
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageImage(BaseModel):
    """One rendered page image. This is the ColQwen2 handoff point -- see
    `localmind.ingestion.pipeline.PageImageRef`, which wraps this with a
    `doc_id` for `localmind/retrieval/colqwen.py` (owned by another task) to
    consume. Nothing in this package embeds or indexes page images itself."""

    model_config = ConfigDict(frozen=True)

    page: int
    uri: str = ""
    width: int | None = None
    height: int | None = None


class ParsedDocument(BaseModel):
    """A document after parsing: reading-order blocks, optional page images,
    and `parser` recording exactly which backend produced it (logged, and
    kept on the object so downstream code/tests can assert on it)."""

    doc_id: str
    source_path: str
    doc_type: DocType
    parser: str
    blocks: list[ParsedBlock] = Field(default_factory=list)
    page_images: list[PageImage] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """Flattened document text (blocks joined by blank lines) -- what the
        chunking stage operates on."""
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())


@runtime_checkable
class Parser(Protocol):
    """Structural seam every parser (real or fake) satisfies. `name` is the
    string logged and stamped onto `ParsedDocument.parser` -- the "log which
    parser handled what" requirement is enforced by construction, not by
    convention."""

    name: str

    def can_parse(self, doc_type: DocType) -> bool: ...
    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument: ...


class NoParserAvailableError(RuntimeError):
    """No registered parser could handle a document: every candidate either
    declined (`can_parse` False) or raised (missing library). Raised rather
    than silently mis-parsing binary bytes as text."""


# --------------------------------------------------------------------------------------
# Pure-Python fallback: always available, works on text/Markdown
# --------------------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_FIGURE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]*)\)\s*$")


class PlainTextParser:
    """Pure-Python fallback: paragraphs split on blank lines, classified into
    headings (`# `), figures (`![alt](uri)`), Markdown pipe tables, or plain
    paragraphs. No third-party dependency, so it's always available -- this is
    what keeps the pipeline testable end to end without Docling/Marker/etc."""

    name = "plaintext-fallback"

    def can_parse(self, doc_type: DocType) -> bool:
        return doc_type in ("text", "markdown", "unknown")

    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        return self.parse_text(raw, doc_id=doc_id or path.stem, source_path=str(path))

    def parse_text(self, raw: str, *, doc_id: str, source_path: str = "<memory>") -> ParsedDocument:
        blocks: list[ParsedBlock] = []
        order = 0
        for para in re.split(r"\n\s*\n", raw.strip("\n")):
            para = para.strip("\n")
            if not para.strip():
                continue
            lines = para.splitlines()
            heading = _HEADING_RE.match(lines[0]) if lines else None
            if heading:
                blocks.append(
                    ParsedBlock(
                        kind="heading",
                        text=heading.group(2).strip(),
                        order=order,
                        level=len(heading.group(1)),
                    )
                )
                order += 1
                remainder = "\n".join(lines[1:]).strip()
                if remainder:
                    blocks.append(ParsedBlock(kind="paragraph", text=remainder, order=order))
                    order += 1
                continue
            figure = _FIGURE_RE.match(lines[0]) if len(lines) == 1 else None
            if figure:
                blocks.append(
                    ParsedBlock(
                        kind="figure",
                        text=figure.group(1) or "(untitled figure)",
                        order=order,
                        metadata={"uri": figure.group(2)},
                    )
                )
                order += 1
                continue
            non_blank = [ln for ln in lines if ln.strip()]
            if len(non_blank) >= 2 and all(_TABLE_ROW_RE.match(ln) for ln in non_blank):
                blocks.append(ParsedBlock(kind="table", text=para, order=order))
                order += 1
                continue
            blocks.append(ParsedBlock(kind="paragraph", text=para, order=order))
            order += 1
        return ParsedDocument(
            doc_id=doc_id,
            source_path=source_path,
            doc_type="markdown",
            parser=self.name,
            blocks=blocks,
        )


# --------------------------------------------------------------------------------------
# Real backends: lazy-imported, unreachable in this offline environment
# --------------------------------------------------------------------------------------


class DoclingParser:
    """Structure-preserving PDF parser: reading order + table structure. Wraps
    `docling.document_converter.DocumentConverter`. Not installed here (no
    network); `parse()` raises `ImportError`, caught by `ParserRegistry.parse`."""

    name = "docling"

    def can_parse(self, doc_type: DocType) -> bool:
        return doc_type == "pdf"

    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument:  # pragma: no cover
        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]

        converter = DocumentConverter()
        result = converter.convert(str(path))
        blocks = [
            ParsedBlock(
                kind="table" if getattr(item, "label", "") == "table" else "paragraph",
                text=str(getattr(item, "text", "")),
                page=getattr(item, "page_no", None),
                order=i,
            )
            for i, item in enumerate(result.document.iterate_items())
        ]
        return ParsedDocument(
            doc_id=doc_id or path.stem,
            source_path=str(path),
            doc_type="pdf",
            parser=self.name,
            blocks=blocks,
        )


class PyMuPDF4LLMParser:
    """Fast text extraction via `pymupdf4llm.to_markdown`, re-parsed through
    `PlainTextParser` for block structure. Not installed here; raises
    `ImportError`."""

    name = "pymupdf4llm"

    def can_parse(self, doc_type: DocType) -> bool:
        return doc_type == "pdf"

    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument:  # pragma: no cover
        import pymupdf4llm  # type: ignore[import-not-found]

        markdown = pymupdf4llm.to_markdown(str(path))
        parsed = PlainTextParser().parse_text(
            markdown, doc_id=doc_id or path.stem, source_path=str(path)
        )
        return parsed.model_copy(update={"parser": self.name, "doc_type": "pdf"})


class MarkerParser:
    """Accurate-but-slow PDF parser via `marker`. Not installed here; raises
    `ImportError`."""

    name = "marker"

    def can_parse(self, doc_type: DocType) -> bool:
        return doc_type == "pdf"

    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument:  # pragma: no cover
        from marker.convert import convert_single_pdf  # type: ignore[import-not-found]
        from marker.models import load_all_models  # type: ignore[import-not-found]

        models = load_all_models()
        markdown, _images, _meta = convert_single_pdf(str(path), models)
        parsed = PlainTextParser().parse_text(
            markdown, doc_id=doc_id or path.stem, source_path=str(path)
        )
        return parsed.model_copy(update={"parser": self.name, "doc_type": "pdf"})


# --------------------------------------------------------------------------------------
# Registry + routing
# --------------------------------------------------------------------------------------

_PDF_STRATEGY_ORDER: dict[ParserStrategy, tuple[str, ...]] = {
    "structure": ("docling", "pymupdf4llm", "marker"),
    "fast": ("pymupdf4llm", "docling", "marker"),
    "accurate": ("marker", "docling", "pymupdf4llm"),
}
_SCANNED_ORDER: tuple[str, ...] = ("surya", "paddleocr")
_FALLBACK_ORDER: tuple[str, ...] = ("plaintext-fallback",)


class ParserRegistry:
    """Holds parsers by name; `parse` picks a priority-ordered candidate list
    for the document's type (+ strategy, for PDFs) and tries each in turn,
    catching `ImportError` and falling through -- logging which parser handled
    (or declined) each document."""

    def __init__(self) -> None:
        self._by_name: dict[str, Parser] = {}

    def register(self, parser: Parser) -> None:
        self._by_name[parser.name] = parser

    def candidate_order(
        self, doc_type: DocType, strategy: ParserStrategy = "structure"
    ) -> tuple[str, ...]:
        if doc_type == "pdf":
            return _PDF_STRATEGY_ORDER.get(strategy, _PDF_STRATEGY_ORDER["structure"])
        if doc_type == "pdf_scanned":
            return _SCANNED_ORDER
        return _FALLBACK_ORDER

    def parse(
        self,
        path: Path,
        *,
        doc_id: str | None = None,
        doc_type: DocType | None = None,
        strategy: ParserStrategy = "structure",
    ) -> ParsedDocument:
        dt = doc_type or sniff_doc_type(path)
        order = self.candidate_order(dt, strategy)
        errors: list[str] = []
        for name in order:
            parser = self._by_name.get(name)
            if parser is None or not parser.can_parse(dt):
                continue
            try:
                result = parser.parse(path, doc_id=doc_id)
            except ImportError as exc:
                msg = f"{name}: {type(exc).__name__}: {exc}"
                errors.append(msg)
                logger.info(
                    "parser %r unavailable for %s (%s); trying next candidate", name, path, exc
                )
                continue
            except Exception as exc:  # a parser bug must not crash ingestion
                msg = f"{name}: {type(exc).__name__}: {exc}"
                errors.append(msg)
                logger.warning("parser %r failed on %s: %s", name, path, exc)
                continue
            logger.info(
                "parsed %s with parser=%s doc_type=%s blocks=%d",
                path,
                result.parser,
                dt,
                len(result.blocks),
            )
            return result
        raise NoParserAvailableError(
            f"no parser could handle {path} (doc_type={dt!r}, strategy={strategy!r}); "
            f"tried {order}; errors={errors}"
        )


def default_registry() -> ParserRegistry:
    """The production routing table: every backend registered, real ones first
    in priority order, `PlainTextParser` as the always-available floor."""
    from localmind.ingestion.parse.ocr import PaddleOCRParser, SuryaOCRParser

    registry = ParserRegistry()
    for parser in (
        DoclingParser(),
        PyMuPDF4LLMParser(),
        MarkerParser(),
        SuryaOCRParser(),
        PaddleOCRParser(),
        PlainTextParser(),
    ):
        registry.register(parser)
    return registry
