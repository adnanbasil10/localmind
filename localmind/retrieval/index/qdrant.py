"""Qdrant — comparison ANN arm, added only after pgvector works (per spec: "One store first").

Structurally minimal on purpose: this exists to let the benchmark table include a second
vector-DB engine for comparison, not to re-derive the HNSW tuning story a second time (that
story is `index/pgvector.py`'s job, including the from-scratch HNSW used for local
measurement). `qdrant-client` is lazy-imported; every method that touches a live Qdrant
instance is exercised only by tests marked `@pytest.mark.docker`.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class QdrantHNSWConfig(BaseModel):
    """Mirrors `pgvector.HNSWConfig`'s knobs so the two engines are tuned comparably. Qdrant
    calls the search-time candidate list size `hnsw_ef` rather than `ef_search`.
    """

    m: int = Field(default=16, gt=0)
    ef_construct: int = Field(default=64, gt=0)
    hnsw_ef: int = Field(default=40, gt=0)


class QdrantIndex:
    """Thin wrapper around `qdrant_client`. Same shape as `PgVectorIndex`: `create`, `upsert`,
    `search`, plus `set_hnsw_ef` for the search-time recall/latency knob.
    """

    def __init__(
        self,
        url: str,
        collection: str,
        dim: int,
        config: QdrantHNSWConfig | None = None,
    ) -> None:
        self.url = url
        self.collection = collection
        self.dim = dim
        self.config = config or QdrantHNSWConfig()
        self._client = None

    def connect(self):
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError(
                "qdrant-client is required for QdrantIndex. Install the `rag` extra and run "
                "`docker compose --profile core up -d` (Qdrant is part of the comparison "
                "profile, brought up alongside Postgres)."
            ) from e
        self._client = QdrantClient(url=self.url)
        return self._client

    def create(self) -> None:
        from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

        client = self._client or self.connect()
        client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=self.config.m, ef_construct=self.config.ef_construct),
        )

    def set_hnsw_ef(self, hnsw_ef: int) -> None:
        self.config = self.config.model_copy(update={"hnsw_ef": hnsw_ef})

    def upsert(
        self, doc_ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None
    ) -> None:
        from qdrant_client.models import PointStruct

        client = self._client or self.connect()
        payloads = payloads or [{} for _ in doc_ids]
        points = [
            PointStruct(id=i, vector=vec.tolist(), payload={"doc_id": doc_id, **payload})
            for i, (doc_id, vec, payload) in enumerate(zip(doc_ids, vectors, payloads, strict=True))
        ]
        client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[str]:
        from qdrant_client.models import SearchParams

        client = self._client or self.connect()
        hits = client.search(
            collection_name=self.collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
            search_params=SearchParams(hnsw_ef=self.config.hnsw_ef),
        )
        return [hit.payload["doc_id"] for hit in hits]
