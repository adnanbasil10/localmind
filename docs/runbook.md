# Runbook

What to do when something breaks. Written for the person on call at 2am, which is usually you.

## Quick reference

```bash
just up core        # postgres + redis + api        (~1.5 GB)
just up obs         # + phoenix, prometheus, grafana (+1.2 GB)
just up full        # + qdrant, mlflow               (+2.5 GB)
just down           # stop everything
just test-fast      # offline-safe suite, no net/gpu/docker
```

| Service | URL | Notes |
|---|---|---|
| API | http://localhost:8000 | `/health`, `/metrics` |
| Phoenix (traces) | http://localhost:6006 | OTLP gRPC on 4317 |
| Prometheus | http://localhost:9090 | |
| Grafana | http://localhost:3001 | anonymous admin enabled locally |
| MLflow | http://localhost:5000 | `full` profile only |

## Alerts and what they mean

### `p95 latency > SLO`
Almost always reranking. The cross-encoder over top-50 is the single largest CPU cost in the
pipeline, and §12 warns this is where RAG systems silently blow their latency budget.

1. Check the Phoenix trace for a single slow request. Look at span durations, not totals.
2. If `rerank` dominates: reduce the candidate set (50 to 25), or switch `bge-reranker-v2-m3`
   to `-base`. Both are config changes, no redeploy.
3. If `generate` dominates: the Ollama model is too large for the machine, or another process is
   contending for CPU.
4. If `embed` dominates: the embedding cache is likely cold or misconfigured — check its hit rate.

### `eval metric regression`
The nightly eval or the PR gate reports nDCG@10 down >2%.

1. **Do not tune the threshold.** The gate exists to catch exactly this.
2. `just eval-retrieval` locally against the same golden set — confirm it reproduces.
3. Bisect on the retrieval config: the usual causes are a chunking parameter change, a changed
   embedding model, or an index rebuilt with different HNSW parameters.
4. If it is a genuine quality/latency trade someone chose, record it in `docs/benchmarks.md` with
   the reasoning and update the baseline deliberately — never silently.

### `injection block rate dropped`
A guardrail regression is a security regression. Treat it as such.

1. Run the injection case set; identify which categories now pass through.
2. Check whether the classifier model or its threshold changed.
3. Until fixed, enable citation-required mode so the agent refuses rather than answers uncited.

### `refusal rate spiked`
Usually the router sending in-domain queries to `out_of_domain`, or the grader marking everything
irrelevant. Check router accuracy on the golden set first — a bad checkpoint swap is the common
cause.

## Common operations

### Roll back
```bash
docker compose --profile core down
git checkout <last-good-sha>
docker compose --profile core up -d --build
```
Images are tagged with the git SHA, so `docker run ghcr.io/<owner>/localmind-api:<sha>` also works
without a rebuild.

### Reindex from scratch
Reindexing is destructive to the vector store. Confirm the corpus is still on disk first.
```bash
uv run python -m localmind.ingestion.pipeline --corpus <path> --rebuild
uv run python -m localmind.eval.retrieval --config configs/retrieval/default.yaml
```
Always run the retrieval eval after a reindex — an index that built successfully can still be
wrong, and the eval is the only thing that will tell you.

### Roll back a model checkpoint
Checkpoints are on HF Hub, versioned. Pin the revision in the model config rather than pulling
`main`, so a rollback is a config change rather than a re-download.

### Postgres will not start
Usually a stale volume after a schema change.
```bash
docker compose logs postgres | tail -50
# Destructive — deletes all indexed vectors. Only after confirming the corpus can be re-ingested:
docker compose down -v && just up core
```

## Recovering a killed training session

This is the expected case, not an incident. Kaggle enforces a 12-hour hard kill.

1. Find the last checkpoint pushed to HF Hub (pushes happen hourly).
2. Re-run the Kaggle notebook cell unchanged — `--resume auto` picks up the latest.
3. Verify the resume was bit-exact: loss should continue the curve, not jump. A jump means the
   dataloader shard position or RNG state did not restore, which is a bug, not a hiccup.
4. Record the lost hours in `docs/compute_log.md`. A session with no Hub push is a loss, and the
   log is only useful if losses are recorded.

## What is *not* an incident

- **CUDA unavailable on the laptop.** Expected. Everything except training is designed for CPU.
- **`gpu` / `net` / `docker` marked tests skipping.** Expected offline. `just test-fast` is the
  green-on-any-machine suite.
- **The 31M model writing poor prose.** Expected and documented in the model card. It is a router,
  grader and rewriter, not a generator.
