# LocalMind

**A 31M-parameter decoder-only LM, built from scratch, that earns its place inside a production
agentic RAG system — trained and deployed entirely on free-tier compute, a laptop, and Docker.**

Hard constraint: **total cash cost = $0.** Every model is open-weights, every service is
self-hosted, every GPU-hour comes from a free allowance.

> **Status: under construction.** Results tables below are placeholders until the benchmark runs
> land. Per §20 rule 1 — nothing goes in this README without a benchmark behind it.

## Results

_Benchmark tables go here, above the fold, before any prose. See `docs/benchmarks.md`._

| System | Recall@10 | nDCG@10 | Faithfulness | p95 ms | CPU-s/query |
|---|---|---|---|---|---|
| _pending_ | | | | | |

## Quickstart

```bash
uv venv && uv pip install -e ".[torch,tok,dev]"
just test          # or: uv run pytest -q
just up core       # postgres+pgvector, redis, api
```

## Layout

See `implementation.md` for the full plan and `CONVENTIONS.md` for the build contracts.

## License

Apache-2.0
