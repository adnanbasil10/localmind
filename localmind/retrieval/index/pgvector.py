"""pgvector index engineering: HNSW tuning, binary quantization + rescoring, and filtered ANN.

Postgres/pgvector is not running in this environment (CONVENTIONS.md), so this module has two
halves:

1. **`PgVectorIndex`** — the real production path. Generates correct DDL/SQL for an HNSW index
   (`m=16, ef_construction=64` as specified) and issues it via `psycopg` + the `pgvector`
   extension, lazy-imported so `import localmind.retrieval.index.pgvector` never touches
   either. Its DB-hitting tests are marked `@pytest.mark.docker`; its SQL-generation is pure
   string logic and is tested directly, with no DB needed.

2. **`SimpleHNSW`** and the `*_report`/`*_curve` functions — a from-scratch HNSW (Malkov &
   Yashunin 2016) plus brute-force ground truth, used to produce *real, locally-measured*
   numbers for the recall-vs-latency curve, the binary-quantization-and-rescore trade, and the
   pre-filter-vs-post-filter ANN failure mode — the three things the spec asks to be
   demonstrated with numbers, not asserted in a docstring. This mirrors the BM25 task's
   "implement it yourself once, then wire the fast library for production" pattern, applied to
   ANN indexing instead of lexical scoring.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from localmind.retrieval.colqwen import estimate_storage, quantize_binary


class HNSWConfig(BaseModel):
    """HNSW build + search parameters. `m` and `ef_construction` are build-time (baked into
    the index); `ef_search` is a per-query knob and is what the recall-vs-latency curve sweeps.
    """

    m: int = Field(default=16, gt=0, description="Max graph edges per node per layer.")
    ef_construction: int = Field(default=64, gt=0, description="Build-time candidate list size.")
    ef_search: int = Field(default=40, gt=0, description="Query-time candidate list size.")


# --------------------------------------------------------------------------------------------
# Production path: real pgvector via psycopg (lazy-imported, requires Docker Postgres)
# --------------------------------------------------------------------------------------------


def create_table_sql(table: str, dim: int) -> str:
    return (
        f"CREATE TABLE IF NOT EXISTS {table} ("
        f"doc_id TEXT PRIMARY KEY, embedding VECTOR({dim}), metadata JSONB)"
    )


def create_hnsw_index_sql(
    table: str, config: HNSWConfig, op_class: str = "vector_cosine_ops"
) -> str:
    return (
        f"CREATE INDEX IF NOT EXISTS {table}_hnsw_idx ON {table} "
        f"USING hnsw (embedding {op_class}) "
        f"WITH (m = {config.m}, ef_construction = {config.ef_construction})"
    )


def set_ef_search_sql(config: HNSWConfig) -> str:
    return f"SET hnsw.ef_search = {config.ef_search}"


def search_sql(table: str, top_k: int) -> str:
    """Plain (unfiltered) ANN search: ORDER BY distance, LIMIT top_k."""
    return f"SELECT doc_id FROM {table} ORDER BY embedding <=> %(query)s LIMIT {top_k}"


def search_postfiltered_sql(table: str, top_k: int) -> str:
    """POST-filtering: ANN picks the nearest `top_k` *first*, filter is applied in an outer
    query afterward. This is the failure mode the spec asks to demonstrate: if the filter is
    selective, most (or all) of those top_k nearest neighbors get discarded by the outer
    WHERE, and recall against the true filtered top-k collapses. See
    `filtered_ann_demo` below for the measured numbers.
    """
    return (
        f"SELECT doc_id FROM (SELECT doc_id, metadata FROM {table} "
        f"ORDER BY embedding <=> %(query)s LIMIT {top_k}) sub WHERE metadata @> %(filter)s"
    )


def search_prefiltered_sql(table: str, top_k: int) -> str:
    """PRE-filtering: WHERE is applied *before* the ANN ordering. pgvector's HNSW index has no
    native concept of a filtered subgraph, so a WHERE-before-ORDER-BY query like this one
    either (a) falls back to scanning every row matching the filter and sorting exactly
    (correct, but pays full brute-force cost on the filtered subset — no ANN speed benefit),
    or (b) with `hnsw.iterative_scan` enabled (pgvector >=0.7), walks the graph and skips
    non-matching nodes, revisiting more of the graph than an unfiltered search would. Either
    way, pre-filtering trades latency for the recall that post-filtering silently loses; see
    `filtered_ann_demo` for the measured trade on a selective filter.
    """
    return (
        f"SELECT doc_id FROM {table} WHERE metadata @> %(filter)s "
        f"ORDER BY embedding <=> %(query)s LIMIT {top_k}"
    )


def binary_column_ddl(table: str, dim: int) -> str:
    return f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS embedding_bit BIT({dim})"


def binary_rescore_sql(table: str, hamming_top_k: int, final_top_k: int) -> str:
    """Binary quantization + rescoring: retrieve `hamming_top_k` candidates by cheap Hamming
    distance on the 1-bit column, then re-rank exactly by full-precision cosine distance on
    the real `embedding` column, and keep the final `final_top_k`.
    """
    return (
        f"SELECT doc_id FROM ("
        f"SELECT doc_id, embedding FROM {table} "
        f"ORDER BY embedding_bit <~> %(query_bit)s LIMIT {hamming_top_k}"
        f") candidates ORDER BY embedding <=> %(query)s LIMIT {final_top_k}"
    )


class PgVectorIndex:
    """Thin wrapper issuing the SQL above against a real Postgres+pgvector instance. Every
    method that touches the network lazy-imports `psycopg`/`pgvector` and is only exercised by
    tests marked `@pytest.mark.docker` (see `docker-compose.yml`'s `core` profile).
    """

    def __init__(self, dsn: str, table: str, dim: int, config: HNSWConfig | None = None) -> None:
        self.dsn = dsn
        self.table = table
        self.dim = dim
        self.config = config or HNSWConfig()
        self._conn = None

    def connect(self):
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as e:
            raise ImportError(
                "psycopg and pgvector are required for PgVectorIndex. Install the `rag` "
                "extra and run `docker compose --profile core up -d`."
            ) from e
        conn = psycopg.connect(self.dsn, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        self._conn = conn
        return conn

    def create(self) -> None:
        conn = self._conn or self.connect()
        conn.execute(create_table_sql(self.table, self.dim))
        conn.execute(create_hnsw_index_sql(self.table, self.config))

    def set_ef_search(self, ef_search: int) -> None:
        conn = self._conn or self.connect()
        self.config = self.config.model_copy(update={"ef_search": ef_search})
        conn.execute(set_ef_search_sql(self.config))

    def upsert(
        self, doc_ids: list[str], vectors: np.ndarray, metadata: list[dict] | None = None
    ) -> None:
        conn = self._conn or self.connect()
        metadata = metadata or [{} for _ in doc_ids]
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (doc_id, embedding, metadata) VALUES (%s, %s, %s) "
                f"ON CONFLICT (doc_id) DO UPDATE SET embedding = EXCLUDED.embedding, "
                f"metadata = EXCLUDED.metadata",
                list(zip(doc_ids, vectors.tolist(), metadata, strict=True)),
            )

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[str]:
        conn = self._conn or self.connect()
        rows = conn.execute(
            search_sql(self.table, top_k), {"query": query_vector.tolist()}
        ).fetchall()
        return [row[0] for row in rows]


# --------------------------------------------------------------------------------------------
# Local measurement path: from-scratch HNSW + brute-force ground truth
# --------------------------------------------------------------------------------------------


def brute_force_search(vectors: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    """Exact cosine nearest neighbors (assumes L2-normalized rows). Ground truth for recall."""
    sims = vectors @ query
    k = min(k, len(vectors))
    idx = np.argpartition(-sims, k - 1)[:k] if k > 0 else np.array([], dtype=int)
    return idx[np.argsort(-sims[idx])].tolist()


@dataclass
class SimpleHNSW:
    """From-scratch HNSW (Malkov & Yashunin, 2016). Favors readability/correctness over
    performance — fine at the scale of our synthetic benchmark corpus (hundreds of vectors),
    which is the point: this exists to produce genuine recall-vs-latency measurements, not to
    compete with pgvector's C implementation.
    """

    dim: int
    m: int = 16
    ef_construction: int = 64
    seed: int = 0
    m0: int = field(init=False)
    _level_mult: float = field(init=False)
    _rng: random.Random = field(init=False)
    _vectors: dict[int, np.ndarray] = field(default_factory=dict, init=False)
    _graph: dict[int, dict[int, list[int]]] = field(default_factory=dict, init=False)
    _entry_point: int | None = field(default=None, init=False)
    _max_level: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        self.m0 = self.m * 2
        self._level_mult = 1.0 / math.log(self.m) if self.m > 1 else 1.0
        self._rng = random.Random(self.seed)
        self._vectors = {}
        self._graph = {}

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - float(np.dot(a, b))  # cosine distance; inputs assumed L2-normalized

    def _random_level(self) -> int:
        return int(-math.log(self._rng.random() + 1e-12) * self._level_mult)

    def add(self, item_id: int, vector: np.ndarray) -> None:
        v = (vector / (np.linalg.norm(vector) + 1e-12)).astype(np.float32)
        self._vectors[item_id] = v
        level = self._random_level()
        self._graph[item_id] = {lvl: [] for lvl in range(level + 1)}

        if self._entry_point is None:
            self._entry_point = item_id
            self._max_level = level
            return

        ep = self._entry_point
        for lvl in range(self._max_level, level, -1):
            nearest = self._search_layer(v, [ep], lvl, ef=1)
            if nearest:
                ep = nearest[0][1]

        for lvl in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(v, [ep], lvl, ef=self.ef_construction)
            cap = self.m0 if lvl == 0 else self.m
            neighbors = [idx for _, idx in candidates[:cap]]
            self._graph[item_id][lvl] = neighbors
            for n in neighbors:
                self._graph[n].setdefault(lvl, [])
                if item_id not in self._graph[n][lvl]:
                    self._graph[n][lvl].append(item_id)
                if len(self._graph[n][lvl]) > cap:
                    ranked = sorted(
                        (self._distance(self._vectors[n], self._vectors[x]), x)
                        for x in self._graph[n][lvl]
                    )
                    self._graph[n][lvl] = [x for _, x in ranked[:cap]]
            if candidates:
                ep = candidates[0][1]

        if level > self._max_level:
            self._max_level = level
            self._entry_point = item_id

    def _search_layer(
        self, query: np.ndarray, entry_points: list[int], level: int, ef: int
    ) -> list[tuple[float, int]]:
        visited = set(entry_points)
        results = sorted((self._distance(query, self._vectors[ep]), ep) for ep in entry_points)
        candidates = list(results)
        while candidates:
            dist, current = candidates.pop(0)
            if len(results) >= ef and dist > results[min(ef, len(results)) - 1][0]:
                break
            for neighbor in self._graph.get(current, {}).get(level, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                d = self._distance(query, self._vectors[neighbor])
                if len(results) < ef or d < results[-1][0]:
                    candidates.append((d, neighbor))
                    candidates.sort()
                    results.append((d, neighbor))
                    results.sort()
                    results = results[:ef]
        return results

    def search(self, query: np.ndarray, k: int = 10, ef_search: int = 40) -> list[int]:
        if self._entry_point is None:
            return []
        q = query / (np.linalg.norm(query) + 1e-12)
        ep = self._entry_point
        for lvl in range(self._max_level, 0, -1):
            nearest = self._search_layer(q, [ep], lvl, ef=1)
            if nearest:
                ep = nearest[0][1]
        results = self._search_layer(q, [ep], 0, ef=max(ef_search, k))
        return [idx for _, idx in results[:k]]


def recall_vs_latency_curve(
    vectors: np.ndarray,
    queries: np.ndarray,
    ef_search_values: list[int],
    k: int = 10,
    config: HNSWConfig | None = None,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Sweep `ef_search`, measuring Recall@k against brute-force ground truth and query
    latency (p50/p95) at each value. "Every vector-DB interview question is really about this
    curve" — this is what produces the actual numbers for it.
    """
    config = config or HNSWConfig()
    index = SimpleHNSW(
        dim=vectors.shape[1], m=config.m, ef_construction=config.ef_construction, seed=seed
    )
    for i, vec in enumerate(vectors):
        index.add(i, vec)

    ground_truth = [brute_force_search(vectors, q, k) for q in queries]

    rows = []
    for ef_search in ef_search_values:
        recalls = []
        latencies = []
        for q, truth in zip(queries, ground_truth, strict=True):
            start = time.perf_counter()
            hits = index.search(q, k=k, ef_search=ef_search)
            latencies.append((time.perf_counter() - start) * 1000)
            truth_set = set(truth)
            recalls.append(len(set(hits) & truth_set) / len(truth_set) if truth_set else 1.0)
        latencies_sorted = sorted(latencies)
        rows.append(
            {
                "ef_search": float(ef_search),
                "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
                "p50_ms": _percentile(latencies_sorted, 50),
                "p95_ms": _percentile(latencies_sorted, 95),
            }
        )
    return rows


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, round(pct / 100 * (len(sorted_values) - 1)))
    return sorted_values[idx]


@dataclass(frozen=True, slots=True)
class BinaryQuantizationReport:
    """Recall@k and latency of the two-stage binary-quantize + rescore pipeline compared to
    exact brute-force cosine search, plus the memory saving that motivates it.
    """

    recall_at_k: float
    exact_recall_at_k: (
        float  # always 1.0 by construction; kept for symmetry with QuantizationRecallReport
    )
    retained_fraction: float
    hamming_top_k: int
    final_top_k: int
    p50_ms: float
    p95_ms: float
    storage: dict[str, float]


def binary_quantization_report(
    vectors: np.ndarray, queries: np.ndarray, k: int = 10, hamming_top_k: int = 200
) -> BinaryQuantizationReport:
    """Store 1-bit vectors (32x smaller), retrieve `hamming_top_k` candidates by Hamming
    distance, rescore those at full precision, keep the top `k`. Reports recall retained vs.
    exact brute-force search and the memory saved by the 1-bit representation.
    """
    n, dim = vectors.shape
    packed = quantize_binary(vectors)

    recalls = []
    latencies = []
    for q in queries:
        ground_truth = set(brute_force_search(vectors, q, k))

        start = time.perf_counter()
        q_bits = quantize_binary(q[None, :])[0]
        xor = np.bitwise_xor(packed, q_bits[None, :])
        hamming = np.unpackbits(xor, axis=-1).sum(axis=-1)
        candidate_idx = np.argsort(hamming)[: min(hamming_top_k, n)]
        candidate_sims = vectors[candidate_idx] @ q
        order = np.argsort(-candidate_sims)[:k]
        hits = candidate_idx[order].tolist()
        latencies.append((time.perf_counter() - start) * 1000)

        recalls.append(len(set(hits) & ground_truth) / len(ground_truth) if ground_truth else 1.0)

    latencies_sorted = sorted(latencies)
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    storage = estimate_storage(n_pages=n, patches_per_page=1, dim=dim)
    return BinaryQuantizationReport(
        recall_at_k=mean_recall,
        exact_recall_at_k=1.0,
        retained_fraction=mean_recall / 1.0,
        hamming_top_k=hamming_top_k,
        final_top_k=k,
        p50_ms=_percentile(latencies_sorted, 50),
        p95_ms=_percentile(latencies_sorted, 95),
        storage=storage,
    )


@dataclass(frozen=True, slots=True)
class FilteredANNReport:
    """The pre-filter-vs-post-filter comparison. `post_filter_recall` collapsing toward 0 as
    `selectivity` shrinks, while `pre_filter_recall` stays near 1.0, *is* the demonstration —
    the spec calls this the most common production vector-DB failure. `pre_filter_latency_ms`
    rising relative to the unfiltered ANN latency is the cost side of that trade.
    """

    selectivity: float
    post_filter_recall: float
    pre_filter_recall: float
    post_filter_latency_ms: float
    pre_filter_latency_ms: float
    unfiltered_ann_latency_ms: float


def filtered_ann_demo(
    vectors: np.ndarray,
    matches_filter: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    config: HNSWConfig | None = None,
    seed: int = 0,
) -> FilteredANNReport:
    """Demonstrate why post-filtering breaks recall under a selective filter, and what
    pre-filtering costs instead.

    `matches_filter`: boolean array, shape (n_vectors,) — which documents satisfy the filter
    predicate (e.g. "tenant_id = X"), simulating a metadata WHERE clause.

    - **Post-filter**: run ANN for the *unfiltered* top-k, then drop hits that fail the
      filter. If the filter is selective, most (or all) of an unfiltered top-k can fail it,
      so this measures recall against the *true* filtered top-k — not just "did the filter
      run correctly", but "did the candidates survive filtering at all".
    - **Pre-filter**: restrict the candidate set to matching documents *first*, then search
      only among those. Implemented here as exact brute-force over the filtered subset —
      which is exactly the cost pre-filtering pays when there's no ANN index over the
      filtered subgraph: it degrades toward a linear scan of the (smaller) matching set.
    """
    config = config or HNSWConfig()
    index = SimpleHNSW(
        dim=vectors.shape[1], m=config.m, ef_construction=config.ef_construction, seed=seed
    )
    for i, vec in enumerate(vectors):
        index.add(i, vec)

    matching_idx = np.nonzero(matches_filter)[0]
    selectivity = len(matching_idx) / len(vectors) if len(vectors) else 0.0

    post_recalls, post_latencies = [], []
    pre_recalls, pre_latencies = [], []
    unfiltered_latencies = []

    for q in queries:
        true_filtered_top_k = brute_force_search(vectors[matching_idx], q, k)
        ground_truth = {int(matching_idx[i]) for i in true_filtered_top_k}

        start = time.perf_counter()
        index.search(q, k=k, ef_search=config.ef_search)  # timed for its own sake, unused
        unfiltered_latencies.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        # Retrieve a generously larger unfiltered candidate pool (not just k) before
        # filtering, matching how a real post-filter query over-fetches; still fails when
        # selectivity is low enough that even the larger pool has few/no matches.
        candidate_pool = index.search(q, k=max(k * 5, 50), ef_search=config.ef_search)
        post_hits = [idx for idx in candidate_pool if matches_filter[idx]][:k]
        post_latencies.append((time.perf_counter() - start) * 1000)
        post_recalls.append(
            len(set(post_hits) & ground_truth) / len(ground_truth) if ground_truth else 1.0
        )

        start = time.perf_counter()
        pre_local_hits = brute_force_search(vectors[matching_idx], q, k)
        pre_hits = {int(matching_idx[i]) for i in pre_local_hits}
        pre_latencies.append((time.perf_counter() - start) * 1000)
        pre_recalls.append(
            len(pre_hits & ground_truth) / len(ground_truth) if ground_truth else 1.0
        )

    return FilteredANNReport(
        selectivity=selectivity,
        post_filter_recall=sum(post_recalls) / len(post_recalls) if post_recalls else 0.0,
        pre_filter_recall=sum(pre_recalls) / len(pre_recalls) if pre_recalls else 0.0,
        post_filter_latency_ms=sum(post_latencies) / len(post_latencies) if post_latencies else 0.0,
        pre_filter_latency_ms=sum(pre_latencies) / len(pre_latencies) if pre_latencies else 0.0,
        unfiltered_ann_latency_ms=sum(unfiltered_latencies) / len(unfiltered_latencies)
        if unfiltered_latencies
        else 0.0,
    )
