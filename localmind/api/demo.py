"""A self-contained demo app: the RAG gateway with a corpus already indexed.

`localmind.api.main:app` deliberately starts with **no retriever** — retrieval is an injected seam,
which is what makes every dependency fake-able in tests. The cost is that the bare app can only ever
refuse, so it is useless as a container entrypoint or a Hugging Face Space (§16).

This module closes that gap without weakening the injection design. It indexes a corpus with BM25,
wraps it in the gateway's own `ScoredDocRetriever`, and exposes a module-level ``app`` so it can be
served directly:

    uvicorn localmind.api.demo:app --host 0.0.0.0 --port 8000

The corpus resolves in this order:

1. ``LOCALMIND_DEMO_CORPUS`` — a path to a JSONL file, if set.
2. The packaged sample corpus, read via ``importlib.resources`` so it works identically from a
   source checkout and from an installed wheel (the file ships inside the wheel).

Generation defaults to the offline extractive stub, so the image needs no model server. Set
``LOCALMIND_GENERATOR=ollama`` (and optionally ``LOCALMIND_OLLAMA_MODEL``) to use a real one.

**What this demonstrates is the pipeline, not model quality.** LocalMind-31M is untrained, so the
agent falls back to its heuristic router/grader/rewriter, and those heuristics reject retrievals the
trained control plane is meant to accept. Expect refusals; that is the honest current state and the
gap the §9 5e comparison matrix exists to quantify.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# `app` is intentionally absent: it is produced lazily by the module-level ``__getattr__`` below,
# so listing it here would be an undefined name to every static checker (ruff F822). It is still
# importable — ``uvicorn localmind.api.demo:app`` is the documented entrypoint.
__all__ = ["build_demo_app", "load_corpus_records"]

_PACKAGED_CORPUS = ("localmind.eval.datasets", "corpus_sample_v1.jsonl")


def _read_jsonl(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_corpus_records(path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """Load corpus records from `path`, else `$LOCALMIND_DEMO_CORPUS`, else the packaged sample."""
    candidate = path or os.environ.get("LOCALMIND_DEMO_CORPUS")
    if candidate:
        p = Path(candidate)
        if not p.is_file():
            raise FileNotFoundError(f"LOCALMIND_DEMO_CORPUS does not exist: {p}")
        return _read_jsonl(p.read_text(encoding="utf-8"))

    # importlib.resources, not __file__: correct from an installed wheel as well as a checkout.
    from importlib.resources import files

    pkg, name = _PACKAGED_CORPUS
    return _read_jsonl(files(pkg).joinpath(name).read_text(encoding="utf-8"))


def build_demo_app(
    records: Iterable[dict[str, Any]] | None = None,
    *,
    cors_origins: tuple[str, ...] | None = None,
):
    """Build the gateway with `records` indexed. Falls back to the packaged sample corpus."""
    from localmind.api.deps import (
        ExtractiveGenerator,
        ScoredDocRetriever,
        Settings,
        build_container,
        build_generator,
    )
    from localmind.api.main import create_app
    from localmind.retrieval import Document
    from localmind.retrieval.bm25 import BM25Index

    rows = list(records) if records is not None else load_corpus_records()
    if not rows:
        raise ValueError("demo corpus is empty")

    # chunk_id is the citable unit; doc_id groups chunks belonging to one source document.
    docs = [
        Document(
            doc_id=r["chunk_id"],
            text=r["text"],
            metadata={"title": r.get("title", ""), "source_doc": r.get("doc_id", "")},
        )
        for r in rows
    ]
    index = BM25Index()
    index.index(docs)

    corpus = {
        r["chunk_id"]: {"text": r["text"], "title": r.get("title", ""), "doc_id": r["chunk_id"]}
        for r in rows
    }

    origins = cors_origins
    if origins is None:
        raw = os.environ.get("LOCALMIND_CORS_ORIGINS", "")
        origins = tuple(o.strip() for o in raw.split(",") if o.strip())

    if os.environ.get("LOCALMIND_GENERATOR", "extractive").lower() == "ollama":
        settings = Settings(generator="ollama", cors_origins=origins)
        generator, _ = build_generator(settings)
    else:
        # Not a language model — see ExtractiveGenerator's docstring. No model server required.
        settings = Settings(generator="extractive", cors_origins=origins)
        generator = ExtractiveGenerator()

    container = build_container(
        settings=settings,
        retriever=ScoredDocRetriever(index, corpus),
        generator=generator,
    )
    return create_app(container)


def __getattr__(name: str):
    """Build `app` lazily, so importing this module never constructs the whole service."""
    if name == "app":
        value = build_demo_app()
        globals()["app"] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
