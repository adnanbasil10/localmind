"""Figure/chart captioning (implementation.md §11): SmolVLM-500M on CPU, or
Qwen2.5-VL-3B as a Kaggle GPU batch job. Neither is installed / reachable here
(no GPU, no network) -- both are thin `Captioner` implementations whose
`.caption()` imports its library lazily and lets `ImportError` propagate.

The caption *text* is what gets indexed for retrieval; the figure crop itself
is kept only as a `crop_uri`/`bbox` reference for citation display -- image
bytes are never loaded or decoded in this module.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from localmind.ingestion.parse import ParsedBlock

__all__ = [
    "CaptionResult",
    "Captioner",
    "FakeCaptioner",
    "FigureCrop",
    "Qwen25VLCaptioner",
    "SmolVLMCaptioner",
    "caption_figures",
    "figures_from_blocks",
]


class FigureCrop(BaseModel):
    """A figure/chart region to caption and, later, cite. `crop_uri` points at
    a saved image crop (e.g. a PNG on disk or object store key); its bytes are
    never loaded here."""

    model_config = ConfigDict(frozen=True)

    figure_id: str
    doc_id: str
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    crop_uri: str = ""
    alt_text: str = ""  # e.g. markdown ![alt](...) text, if the parser had one


class CaptionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    figure_id: str
    caption: str
    captioner: str
    latency_ms: float = 0.0


@runtime_checkable
class Captioner(Protocol):
    """Duck-typed seam: `name` is logged/stamped onto every `CaptionResult`."""

    name: str

    def caption(self, figure: FigureCrop) -> str: ...


class SmolVLMCaptioner:
    """SmolVLM-500M on CPU (implementation.md §11's low-cost default). Not
    installed here; `caption()` raises `ImportError`."""

    name = "smolvlm-500m"

    def caption(self, figure: FigureCrop) -> str:  # pragma: no cover
        from PIL import Image  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForVision2Seq,
            AutoProcessor,
        )

        model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(model_id)
        image = Image.open(figure.crop_uri)
        inputs = processor(
            images=image, text="Describe this figure in one sentence.", return_tensors="pt"
        )
        out = model.generate(**inputs, max_new_tokens=64)
        return processor.decode(out[0], skip_special_tokens=True).strip()


class Qwen25VLCaptioner:
    """Qwen2.5-VL-3B, run as a Kaggle GPU batch job (implementation.md §11's
    higher-quality option). Not reachable here (no GPU, no network);
    `caption()` raises `ImportError`."""

    name = "qwen2.5-vl-3b"

    def caption(self, figure: FigureCrop) -> str:  # pragma: no cover
        from PIL import Image  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForVision2Seq,
            AutoProcessor,
        )

        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(model_id)
        image = Image.open(figure.crop_uri)
        inputs = processor(
            images=image, text="Describe this chart in one to two sentences.", return_tensors="pt"
        )
        out = model.generate(**inputs, max_new_tokens=96)
        return processor.decode(out[0], skip_special_tokens=True).strip()


class FakeCaptioner:
    """Deterministic offline double: reuses `alt_text` (if the parser
    extracted one, e.g. Markdown `![alt](uri)`) or falls back to a generic,
    provenance-carrying placeholder. Enough to exercise indexing/citation
    wiring without a real vision model; this is the default when no
    `Captioner` is injected."""

    name = "fake"

    def caption(self, figure: FigureCrop) -> str:
        if figure.alt_text.strip():
            return figure.alt_text.strip()
        loc = f"page {figure.page}" if figure.page is not None else "unknown page"
        return f"Figure from {figure.doc_id} ({loc}); no alt text available."


def caption_figures(
    figures: Sequence[FigureCrop], captioner: Captioner | None = None
) -> list[CaptionResult]:
    """Caption every figure, never raising: a broken/absent real captioner
    falls back to `FakeCaptioner` per-figure so one bad backend doesn't take
    down ingestion, and the fallback is recorded in `captioner` for the log."""
    cap = captioner or FakeCaptioner()
    out: list[CaptionResult] = []
    for fig in figures:
        t0 = time.perf_counter()
        try:
            text = cap.caption(fig)
            used = cap.name
        except Exception as exc:
            text = FakeCaptioner().caption(fig)
            used = f"{cap.name}:fallback({type(exc).__name__})"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        out.append(
            CaptionResult(
                figure_id=fig.figure_id, caption=text, captioner=used, latency_ms=latency_ms
            )
        )
    return out


def figures_from_blocks(blocks: Sequence[ParsedBlock], *, doc_id: str) -> list[FigureCrop]:
    """Pull every figure block (`kind == "figure"`, produced by e.g.
    `![alt](uri)` in `PlainTextParser`) into a `FigureCrop`, tagged
    `doc_id#figN` with provenance carried over from the block."""
    out: list[FigureCrop] = []
    n = 0
    for b in blocks:
        if b.kind != "figure":
            continue
        out.append(
            FigureCrop(
                figure_id=f"{doc_id}#fig{n}",
                doc_id=doc_id,
                page=b.page,
                bbox=b.bbox,
                crop_uri=str(b.metadata.get("uri", "")),
                alt_text=b.text,
            )
        )
        n += 1
    return out
