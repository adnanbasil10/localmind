"""Retrieval-layer metrics: Recall@{1,5,10,20}, nDCG@10, MRR, MAP.

Cheap, deterministic, no LLM anywhere in the loop -- this is the layer the CI
gate runs on every pull request (§14: "Runs on every PR in GitHub Actions on
CPU in ~3 minutes").

Relevance is binary and comes from the golden set: ``expected_chunk_ids`` when
present, ``expected_doc_ids`` otherwise.  Questions the golden set marks
unanswerable carry no relevant evidence, so they are excluded from retrieval
metrics and counted separately -- they are scored by the refusal metrics in
``generation.py`` instead.

A dependency-free BM25 retriever is included so the gate has something real to
run before the production retrieval stack lands; any object satisfying
:class:`Retriever` can be substituted.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict

from localmind.eval.datasets.schema import (
    SAMPLE_CORPUS_PATH,
    SAMPLE_GOLDEN_PATH,
    CorpusChunk,
    GoldenDataset,
    GoldenQuestion,
    doc_of_chunk,
    load_corpus,
    load_golden,
)
from localmind.eval.stats import (
    DEFAULT_SEED,
    Estimate,
    MetricRow,
    benchmark_json,
    bootstrap_ci,
)

__all__ = [
    "DEFAULT_KS",
    "BM25Retriever",
    "RetrievalEvalConfig",
    "RetrievalReport",
    "RetrievedItem",
    "Retriever",
    "StaticRunRetriever",
    "average_precision",
    "evaluate_retrieval",
    "load_eval_config",
    "main",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
]

DEFAULT_KS: tuple[int, ...] = (1, 5, 10, 20)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# Seam
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RetrievedItem:
    chunk_id: str
    score: float = 0.0

    @property
    def doc_id(self) -> str:
        return doc_of_chunk(self.chunk_id)


@runtime_checkable
class Retriever(Protocol):
    """The only thing the retrieval eval needs from a retrieval stack."""

    @property
    def name(self) -> str: ...

    def search(self, query: str, k: int) -> Sequence[RetrievedItem]: ...


@dataclass
class StaticRunRetriever:
    """Replays precomputed ranked lists (a TREC-style run file)."""

    runs: dict[str, list[RetrievedItem]]
    name: str = "static-run"
    _current: str | None = field(default=None, repr=False)

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str = "static-run") -> StaticRunRetriever:
        runs: dict[str, list[RetrievedItem]] = {}
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            ranked = rec.get("ranked_chunk_ids") or rec.get("ranked") or []
            scores = rec.get("scores") or [float(len(ranked) - i) for i in range(len(ranked))]
            runs[rec["id"]] = [
                RetrievedItem(chunk_id=c, score=float(s))
                for c, s in zip(ranked, scores, strict=False)
            ]
        return cls(runs=runs, name=name)

    def search(self, query: str, k: int) -> Sequence[RetrievedItem]:
        del query
        key = self._current
        if key is None or key not in self.runs:
            raise KeyError("StaticRunRetriever needs the question id; use search_by_id")
        return self.runs[key][:k]

    def search_by_id(self, qid: str, k: int) -> Sequence[RetrievedItem]:
        return self.runs.get(qid, [])[:k]


# --------------------------------------------------------------------------- #
# A dependency-free BM25 so the gate always has something real to run
# --------------------------------------------------------------------------- #
@dataclass
class BM25Retriever:
    """Okapi BM25 in numpy.  Deterministic, CPU-only, no index server needed.

    Ties are broken by ``chunk_id`` so a run is reproducible bit-for-bit.
    """

    chunks: Sequence[CorpusChunk]
    k1: float = 1.2
    b: float = 0.75
    name: str = "bm25-numpy"
    _vocab: dict[str, int] = field(default_factory=dict, repr=False)
    _tf: list[dict[int, int]] = field(default_factory=list, repr=False)
    _idf: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    _lens: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    def __post_init__(self) -> None:
        docs = [_tokenize(c.text if not c.title else f"{c.title} {c.text}") for c in self.chunks]
        for doc in docs:
            for tok in doc:
                if tok not in self._vocab:
                    self._vocab[tok] = len(self._vocab)
        n_docs = len(docs)
        df = np.zeros(len(self._vocab), dtype=np.float64)
        for doc in docs:
            counts: dict[int, int] = {}
            for tok in doc:
                idx = self._vocab[tok]
                counts[idx] = counts.get(idx, 0) + 1
            self._tf.append(counts)
            for idx in counts:
                df[idx] += 1.0
        self._idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)) if n_docs else df
        self._lens = np.array([float(len(d)) for d in docs], dtype=np.float64)

    @property
    def avg_len(self) -> float:
        return float(self._lens.mean()) if self._lens.size else 0.0

    def search(self, query: str, k: int) -> Sequence[RetrievedItem]:
        q_idx = [self._vocab[t] for t in _tokenize(query) if t in self._vocab]
        if not q_idx:
            return []
        avg = self.avg_len or 1.0
        scores = np.zeros(len(self.chunks), dtype=np.float64)
        for d, counts in enumerate(self._tf):
            norm = self.k1 * (1.0 - self.b + self.b * self._lens[d] / avg)
            total = 0.0
            for idx in q_idx:
                tf = counts.get(idx, 0)
                if tf:
                    total += self._idf[idx] * tf * (self.k1 + 1.0) / (tf + norm)
            scores[d] = total
        order = sorted(
            (d for d in range(len(self.chunks)) if scores[d] > 0.0),
            key=lambda d: (-scores[d], self.chunks[d].chunk_id),
        )
        return [
            RetrievedItem(chunk_id=self.chunks[d].chunk_id, score=float(scores[d]))
            for d in order[:k]
        ]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------- #
# Metrics (binary relevance)
# --------------------------------------------------------------------------- #
def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant set found in the top ``k``."""
    if not relevant:
        raise ValueError("recall is undefined with an empty relevant set")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG: ``DCG@k / IDCG@k`` with ``gain = 1/log2(rank+1)``."""
    if not relevant:
        raise ValueError("nDCG is undefined with an empty relevant set")
    dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(ranked[:k]) if cid in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], cutoff: int | None = None) -> float:
    """``1 / rank`` of the first relevant item; 0.0 if none within the cutoff."""
    if not relevant:
        raise ValueError("MRR is undefined with an empty relevant set")
    window = ranked if cutoff is None else ranked[:cutoff]
    for i, cid in enumerate(window):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(
    ranked: Sequence[str], relevant: set[str], cutoff: int | None = None
) -> float:
    """AP over the returned ranking, normalised by ``min(|relevant|, cutoff)``.

    Dividing by the reachable number of relevant items (rather than the full
    relevant set) keeps AP = 1.0 achievable when the cutoff is smaller than the
    relevant set, which is the honest reading of "perfect ranking at k".
    """
    if not relevant:
        raise ValueError("AP is undefined with an empty relevant set")
    window = list(ranked) if cutoff is None else list(ranked[:cutoff])
    hits = 0
    total = 0.0
    for i, cid in enumerate(window):
        if cid in relevant:
            hits += 1
            total += hits / (i + 1)
    denom = len(relevant) if cutoff is None else min(len(relevant), cutoff)
    return total / denom if denom else 0.0


def relevant_set(q: GoldenQuestion, *, granularity: str = "auto") -> tuple[set[str], str]:
    """The relevance judgements for one question, and the granularity used."""
    if granularity == "doc" or (granularity == "auto" and not q.expected_chunk_ids):
        return set(q.expected_doc_ids), "doc"
    return set(q.expected_chunk_ids), "chunk"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalReport:
    """Retrieval results.  Every metric carries a bootstrap 95% CI."""

    system: str
    metrics: dict[str, Estimate] = field(default_factory=dict)
    per_question: dict[str, dict[str, float]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, dict[str, Estimate]] = field(default_factory=dict)
    elapsed_s: float = 0.0
    seed: int = DEFAULT_SEED
    dataset_name: str = ""
    dataset_hash: str = ""
    granularity: str = "chunk"

    def values(self, metric: str) -> list[float]:
        return [v[metric] for v in self.per_question.values() if metric in v]

    def to_row(self) -> MetricRow:
        return MetricRow(
            name=self.system,
            values=dict(self.metrics),
            extra={"counts": dict(self.counts), "elapsed_s": self.elapsed_s},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "per_question": self.per_question,
            "counts": dict(self.counts),
            "by_category": {
                cat: {m: e.to_dict() for m, e in vals.items()}
                for cat, vals in self.by_category.items()
            },
            "elapsed_s": self.elapsed_s,
            "seed": self.seed,
            "dataset": {
                "name": self.dataset_name,
                "sha256": self.dataset_hash,
                "granularity": self.granularity,
            },
        }

    def to_markdown(self) -> str:
        lines = ["| metric | value (95% CI) |", "|---|---|"]
        for key in _ordered_metric_names(self.metrics):
            lines.append(f"| {key} | {self.metrics[key].format()} |")
        lines += [
            "",
            f"n={self.counts.get('scored', 0)} scored, "
            f"{self.counts.get('skipped_unanswerable', 0)} unanswerable questions excluded "
            f"(scored by refusal accuracy instead); {self.elapsed_s:.2f}s wall.",
        ]
        return "\n".join(lines)


def _ordered_metric_names(metrics: Mapping[str, Estimate]) -> list[str]:
    preferred = ["recall@1", "recall@5", "recall@10", "recall@20", "ndcg@10", "mrr", "map"]
    rest = sorted(k for k in metrics if k not in preferred)
    return [k for k in preferred if k in metrics] + rest


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def evaluate_retrieval(
    dataset: GoldenDataset,
    retriever: Retriever | StaticRunRetriever,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    ndcg_k: int = 10,
    seed: int = DEFAULT_SEED,
    n_resamples: int = 2_000,
    granularity: str = "auto",
    by_category: bool = True,
) -> RetrievalReport:
    """Score a retriever against the golden set.  Deterministic; no LLM calls."""
    ks = tuple(sorted({int(k) for k in ks}))
    depth = max([*ks, ndcg_k])
    per_q: dict[str, dict[str, float]] = {}
    series: dict[str, list[float]] = {}
    skipped = 0
    used_granularity = "chunk"

    start = time.perf_counter()
    for q in dataset.questions:
        rel, gran = relevant_set(q, granularity=granularity)
        if not rel:
            skipped += 1
            continue
        used_granularity = gran
        items = (
            retriever.search_by_id(q.id, depth)
            if isinstance(retriever, StaticRunRetriever)
            else retriever.search(q.question, depth)
        )
        ranked = [it.chunk_id if gran == "chunk" else it.doc_id for it in items]
        if gran == "doc":
            ranked = list(dict.fromkeys(ranked))

        row: dict[str, float] = {}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked, rel, k)
        row[f"ndcg@{ndcg_k}"] = ndcg_at_k(ranked, rel, ndcg_k)
        row["mrr"] = reciprocal_rank(ranked, rel, cutoff=depth)
        row["map"] = average_precision(ranked, rel, cutoff=depth)
        per_q[q.id] = row
        for key, val in row.items():
            series.setdefault(key, []).append(val)
    elapsed = time.perf_counter() - start

    metrics = {
        key: bootstrap_ci(vals, seed=seed, n_resamples=n_resamples) for key, vals in series.items()
    }

    cat_metrics: dict[str, dict[str, Estimate]] = {}
    if by_category:
        for cat in sorted({q.category for q in dataset.questions}):
            ids = [q.id for q in dataset.questions if q.category == cat and q.id in per_q]
            if len(ids) < 2:
                continue
            cat_metrics[cat] = {
                key: bootstrap_ci([per_q[i][key] for i in ids], seed=seed, n_resamples=n_resamples)
                for key in series
            }

    return RetrievalReport(
        system=getattr(retriever, "name", type(retriever).__name__),
        metrics=metrics,
        per_question=per_q,
        counts={
            "questions": len(dataset.questions),
            "scored": len(per_q),
            "skipped_unanswerable": skipped,
        },
        by_category=cat_metrics,
        elapsed_s=elapsed,
        seed=seed,
        dataset_name=f"{dataset.name}@{dataset.version}",
        dataset_hash=dataset.content_hash,
        granularity=used_granularity,
    )


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class RetrievalEvalConfig(BaseModel):
    """The slice of a retrieval config this harness cares about.

    ``extra="ignore"`` on purpose: ``configs/retrieval/*.yaml`` is owned by the
    retrieval phase and will carry many keys that mean nothing here.  Adding
    keys there must never break the eval gate.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = "bm25-numpy"
    top_k: int = 20
    ndcg_k: int = 10
    seed: int = DEFAULT_SEED
    corpus: str | None = None
    dataset: str | None = None
    runs: str | None = None
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    n_resamples: int = 2_000


def load_eval_config(path: str | Path | None) -> RetrievalEvalConfig:
    """Load a retrieval config, tolerating any shape the retrieval phase picks.

    A nested ``eval:`` block wins over top-level keys; unknown keys are ignored.
    A missing file yields defaults so the gate still runs on the committed
    sample corpus.
    """
    if path is None:
        return RetrievalEvalConfig()
    p = Path(path)
    if not p.exists():
        print(f"[eval.retrieval] config {p} not found; using built-in defaults", file=sys.stderr)
        return RetrievalEvalConfig()
    import yaml  # lazy: keeps `import localmind.eval` numpy/pydantic-only

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        print(f"[eval.retrieval] config {p} is not a mapping; using defaults", file=sys.stderr)
        return RetrievalEvalConfig()
    merged: dict[str, Any] = {k: v for k, v in raw.items() if not isinstance(v, dict | list)}
    nested = raw.get("eval")
    if isinstance(nested, dict):
        merged.update(nested)
    return RetrievalEvalConfig.model_validate(merged)


# --------------------------------------------------------------------------- #
# CLI -- `just eval-retrieval`
# --------------------------------------------------------------------------- #
def build_retriever(cfg: RetrievalEvalConfig) -> Retriever | StaticRunRetriever:
    if cfg.runs:
        return StaticRunRetriever.from_jsonl(cfg.runs, name=cfg.name)
    corpus_path = Path(cfg.corpus) if cfg.corpus else SAMPLE_CORPUS_PATH
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus {corpus_path} not found")
    return BM25Retriever(
        chunks=load_corpus(corpus_path), k1=cfg.bm25_k1, b=cfg.bm25_b, name=cfg.name
    )


def hardware_string() -> str:
    from localmind.eval.system import hardware_string as _hw

    return _hw()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m localmind.eval.retrieval",
        description="Deterministic retrieval eval: Recall@k, nDCG@10, MRR, MAP.",
    )
    parser.add_argument("--config", default=None, help="configs/retrieval/*.yaml")
    parser.add_argument("--dataset", default=None, help="golden set JSONL")
    parser.add_argument("--corpus", default=None, help="corpus JSONL")
    parser.add_argument("--runs", default=None, help="precomputed ranked lists JSONL")
    parser.add_argument("--out", default="artifacts/benchmarks/eval_retrieval.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--budget-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)

    cfg = load_eval_config(args.config)
    if args.dataset:
        cfg = cfg.model_copy(update={"dataset": args.dataset})
    if args.corpus:
        cfg = cfg.model_copy(update={"corpus": args.corpus})
    if args.runs:
        cfg = cfg.model_copy(update={"runs": args.runs})
    if args.seed is not None:
        cfg = cfg.model_copy(update={"seed": args.seed})
    if args.top_k is not None:
        cfg = cfg.model_copy(update={"top_k": args.top_k})

    dataset_path = Path(cfg.dataset) if cfg.dataset else SAMPLE_GOLDEN_PATH
    dataset = load_golden(dataset_path, name=dataset_path.stem, version="v1")
    retriever = build_retriever(cfg)
    ks = tuple(k for k in DEFAULT_KS if k <= max(cfg.top_k, 1)) or (1,)

    report = evaluate_retrieval(
        dataset,
        retriever,
        ks=ks,
        ndcg_k=cfg.ndcg_k,
        seed=cfg.seed,
        n_resamples=cfg.n_resamples,
    )

    payload = benchmark_json(
        "retrieval",
        hardware=hardware_string(),
        seeds=[cfg.seed],
        rows=[report.to_row()],
        extra={"detail": report.to_dict(), "config": cfg.model_dump()},
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(report.to_markdown())
    print(f"\nwrote {out}")
    if report.elapsed_s > args.budget_seconds:
        print(
            f"[eval.retrieval] FAIL: {report.elapsed_s:.1f}s exceeds the "
            f"{args.budget_seconds:.0f}s CI budget",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
