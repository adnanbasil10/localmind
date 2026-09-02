"""Run the RAG gateway locally with a corpus indexed — a thin CLI over `localmind.api.demo`.

The app factory itself lives in the package (`localmind/api/demo.py`) rather than here, because it
has to ship inside the wheel for the Docker image and the Hugging Face Space to use it. This file is
just the convenient local entrypoint.

    uv run python deploy/demo_server.py                       # http://localhost:8000
    uv run python deploy/demo_server.py --port 9000
    uv run python deploy/demo_server.py --ollama-model mistral:latest
    uv run python deploy/demo_server.py --corpus path/to/corpus.jsonl

Answers come from the indexed corpus with real citations. This demonstrates the *pipeline*, not
model quality — LocalMind-31M is untrained, so the agent falls back to its heuristic
router/grader/rewriter and will often refuse. That is the honest current state.

Equivalent one-liner, no CLI:
    uv run uvicorn localmind.api.demo:app --port 8000
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--corpus", default=None, help="JSONL corpus; defaults to the packaged sample")
    ap.add_argument(
        "--cors-origin",
        default="http://localhost:3000",
        help="frontend origin allowed to call this API; empty to disable CORS",
    )
    ap.add_argument(
        "--ollama-model",
        default=None,
        help="e.g. mistral:latest — uses a real generator via Ollama. "
        "Omit for the offline extractive stub (no model server needed).",
    )
    args = ap.parse_args()

    if args.ollama_model:
        os.environ["LOCALMIND_GENERATOR"] = "ollama"
        os.environ["LOCALMIND_OLLAMA_MODEL"] = args.ollama_model

    import uvicorn
    from localmind.api.demo import build_demo_app, load_corpus_records

    records = load_corpus_records(args.corpus)
    app = build_demo_app(
        records,
        cors_origins=(args.cors_origin,) if args.cors_origin else (),
    )
    n_docs = len({r.get("doc_id") for r in records})
    gen = args.ollama_model or "extractive-stub"
    print(f"[demo] indexed {len(records)} chunks from {n_docs} documents | generator={gen}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
