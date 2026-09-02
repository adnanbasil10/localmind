"""Run the RAG gateway with a real, indexed corpus — the demo §16 asks for.

`uvicorn localmind.api.main:app` builds a container with **no retriever**, because the retriever is
an injected seam rather than an env var. That is correct for testing (every dependency is fake-able)
but it means the plain command can only ever refuse, which is useless as a demo.

This launcher closes that gap without changing the injection design: it indexes the committed sample
corpus with BM25, wraps it in the gateway's own `ScoredDocRetriever` adapter, and pairs it with the
`ExtractiveGenerator` so nothing depends on Ollama being installed or a model being pulled.

    uv run python deploy/demo_server.py            # http://localhost:8000
    uv run python deploy/demo_server.py --port 9000

The answers are extractive spans from the indexed corpus with real citations. This demonstrates the
*pipeline*, not model quality — LocalMind-31M is untrained, and the control plane falls back to the
agent's heuristic router/grader/rewriter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "localmind" / "eval" / "datasets" / "corpus_sample_v1.jsonl"


def load_corpus() -> list[dict]:
    if not CORPUS.exists():
        raise SystemExit(f"corpus not found: {CORPUS}")
    with CORPUS.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_app(cors_origin: str | None, *, ollama_model: str | None = None):
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

    records = load_corpus()

    # chunk_id is the citable unit; doc_id groups chunks from one source document.
    docs = [
        Document(doc_id=r["chunk_id"], text=r["text"], metadata={
            "title": r.get("title", ""), "source_doc": r.get("doc_id", ""),
        })
        for r in records
    ]
    index = BM25Index()
    index.index(docs)

    corpus = {
        r["chunk_id"]: {"text": r["text"], "title": r.get("title", ""), "doc_id": r["chunk_id"]}
        for r in records
    }
    retriever = ScoredDocRetriever(index, corpus)

    if ollama_model:
        settings = Settings(
            generator="ollama",
            ollama_model=ollama_model,
            cors_origins=(cors_origin,) if cors_origin else (),
        )
        generator, gen_name = build_generator(settings)
    else:
        # No model server required. Not a language model — see ExtractiveGenerator's docstring.
        settings = Settings(
            generator="extractive",
            cors_origins=(cors_origin,) if cors_origin else (),
        )
        generator, gen_name = ExtractiveGenerator(), ExtractiveGenerator.name

    container = build_container(
        settings=settings,
        retriever=retriever,
        generator=generator,
    )
    n_docs = len({r["doc_id"] for r in records})
    print(f"[demo] indexed {len(docs)} chunks from {n_docs} documents | generator={gen_name}")
    return create_app(container)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cors-origin", default="http://localhost:3000",
                    help="frontend origin allowed to call this API; empty to disable CORS")
    ap.add_argument("--ollama-model", default=None,
                    help="e.g. mistral:latest — uses a real generator via Ollama. "
                         "Omit for the offline extractive stub (no model server needed).")
    args = ap.parse_args()

    import uvicorn

    app = build_app(args.cors_origin or None, ollama_model=args.ollama_model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
