"""OCR-backed parsers for scanned documents (implementation.md §11): Surya and
PaddleOCR. Neither is installed in this environment and there is no network to
fetch either -- both `.parse()` methods import their library lazily and let
the resulting `ImportError` propagate, so `ParserRegistry.parse` catches it
and falls through to the next candidate. There is no third scanned-document
candidate, so an all-scanned corpus with neither backend installed surfaces as
`NoParserAvailableError` -- the honest answer: this pipeline cannot read pixels
without an OCR engine.

Import-cycle note: this module needs `ParsedDocument`/`ParsedBlock` from the
parent package, and the parent package (`localmind/ingestion/parse/__init__.py`)
imports `SuryaOCRParser`/`PaddleOCRParser` from here (inside `default_registry`)
to build the production registry. To avoid a module-import-time cycle, the
parent-package types are imported here only under `TYPE_CHECKING` (for
annotations) and lazily inside each `parse()` body (for runtime construction)
-- never at this module's top level.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from localmind.ingestion.parse import DocType, ParsedDocument

__all__ = ["PaddleOCRParser", "SuryaOCRParser"]


class SuryaOCRParser:
    """Surya OCR: layout-aware scanned-document reader. Not installed here;
    `parse()` raises `ImportError` (via `pdf2image`/`surya`, whichever is
    missing first)."""

    name = "surya"

    def can_parse(self, doc_type: DocType) -> bool:
        return doc_type == "pdf_scanned"

    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument:  # pragma: no cover
        from pdf2image import convert_from_path  # type: ignore[import-not-found]
        from surya.model.detection.model import (
            load_model as load_det_model,  # type: ignore[import-not-found]
        )
        from surya.model.recognition.model import (
            load_model as load_rec_model,  # type: ignore[import-not-found]
        )
        from surya.ocr import run_ocr  # type: ignore[import-not-found]

        from localmind.ingestion.parse import ParsedBlock, ParsedDocument

        images = convert_from_path(str(path))
        predictions = run_ocr(images, [None] * len(images), load_det_model(), load_rec_model())
        blocks = [
            ParsedBlock(kind="paragraph", text=line.text, page=page_no, order=i)
            for page_no, page in enumerate(predictions)
            for i, line in enumerate(page.text_lines)
        ]
        return ParsedDocument(
            doc_id=doc_id or path.stem,
            source_path=str(path),
            doc_type="pdf_scanned",
            parser=self.name,
            blocks=blocks,
        )


class PaddleOCRParser:
    """PaddleOCR: the other free/local scanned-document reader. Not installed
    here; `parse()` raises `ImportError`."""

    name = "paddleocr"

    def can_parse(self, doc_type: DocType) -> bool:
        return doc_type == "pdf_scanned"

    def parse(self, path: Path, *, doc_id: str | None = None) -> ParsedDocument:  # pragma: no cover
        from paddleocr import PaddleOCR  # type: ignore[import-not-found]
        from pdf2image import convert_from_path  # type: ignore[import-not-found]

        from localmind.ingestion.parse import ParsedBlock, ParsedDocument

        engine = PaddleOCR(use_angle_cls=True, lang="en")
        images = convert_from_path(str(path))
        blocks: list[ParsedBlock] = []
        order = 0
        for page_no, image in enumerate(images):
            for line in engine.ocr(image, cls=True)[0] or []:
                text = line[1][0]
                blocks.append(ParsedBlock(kind="paragraph", text=text, page=page_no, order=order))
                order += 1
        return ParsedDocument(
            doc_id=doc_id or path.stem,
            source_path=str(path),
            doc_type="pdf_scanned",
            parser=self.name,
            blocks=blocks,
        )
