---
title: LocalMind
emoji: 🧠
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# LocalMind — live demo

A 31M-parameter language model, built from scratch, running as the routing / grading /
rewriting control plane of an agentic RAG system — **on 2 free CPU cores, with no GPU.**

The model is ~31 MB at int8. That is the entire point: it fits here.

- Source: <repo URL>
- Benchmarks: `docs/benchmarks.md` in the repo
- Model card: `docs/model_card.md`

Answer generation in the full local stack uses Qwen3-4B via Ollama. This Space runs the
control plane and retrieval only — a 4B generator does not fit the free CPU tier.
