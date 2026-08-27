"""`retrieve_image`: figure/diagram lookup by id or caption search.

Returns metadata and VLM captions, never raw bytes -- an agent turn should carry
a reference and a caption, not a megabyte of base64. Captions come from the
ingestion pipeline and are therefore untrusted (a PDF author writes the figure
text), so callers treat them like any other retrieved chunk.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field, model_validator

from localmind.agent.state import ImageStore
from localmind.agent.tools.base import Tool, ToolExecutionError

__all__ = ["RetrieveImageArgs", "RetrieveImageTool"]


class RetrieveImageArgs(BaseModel):
    image_id: str | None = Field(default=None, max_length=200, description="Exact image id.")
    query: str | None = Field(
        default=None, max_length=500, description="Caption search, when the id is unknown."
    )
    k: int = Field(default=3, ge=1, le=10, description="Results for caption search.")

    @model_validator(mode="after")
    def _one_of(self) -> RetrieveImageArgs:
        if not self.image_id and not self.query:
            raise ValueError("provide either image_id or query")
        return self


class RetrieveImageTool(Tool):
    """Idempotent: the same id or query always resolves to the same record."""

    name: ClassVar[str] = "retrieve_image"
    description: ClassVar[str] = (
        "Fetch a figure/diagram by id, or search figures by caption. Returns image "
        "metadata and captions (untrusted text), not image bytes."
    )
    args_model: ClassVar[type[BaseModel]] = RetrieveImageArgs
    idempotent: ClassVar[bool] = True

    def __init__(self, store: ImageStore | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.store = store

    def _call(self, args: RetrieveImageArgs) -> dict[str, Any]:
        if self.store is None:
            raise ToolExecutionError("unavailable", "no image store is attached to this agent")
        try:
            if args.image_id:
                record = self.store.get(args.image_id)
                if record is None:
                    raise ToolExecutionError("not_found", f"no image with id {args.image_id!r}")
                records = [record]
            else:
                records = list(self.store.search(args.query or "", args.k))[: args.k]
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(
                "unavailable", f"image store error: {type(exc).__name__}: {exc}", retryable=True
            ) from exc
        return {
            "images": [r.model_dump(mode="json") for r in records],
            "count": len(records),
        }
