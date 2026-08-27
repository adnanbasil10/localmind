"""Redis-backed caching, Layer 2: the exact response cache.

Key: ``hash(query + filters + config)`` — a canonical JSON encoding of the three, hashed, so two
requests are only considered "the same" if the query text, retrieval filters, and generation
config all match exactly. Metric to report: hit rate and latency saved.

"Latency saved" is measured honestly: at write time (a miss) we record how long the real,
uncached computation actually took; every later hit that reuses that entry is credited with
exactly that recorded latency, not an assumed constant.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel

from localmind.cache.embedding_cache import CacheBackend

__all__ = ["ResponseCache", "ResponseCacheEntry", "ResponseCacheStats"]


class ResponseCacheEntry(BaseModel):
    response: str
    compute_latency_ms: float


class ResponseCacheStats(BaseModel):
    hits: int
    misses: int
    total: int
    hit_rate: float
    latency_saved_ms_total: float
    latency_saved_ms_mean: float


class ResponseCache:
    """Layer 2. Key: ``hash(query + filters + config)``. Value: the full response + the latency
    its original computation took (so hits can report latency actually saved, not a guess)."""

    def __init__(self, backend: CacheBackend, ttl_seconds: int | None = None) -> None:
        self.backend = backend
        self.ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0
        self._latency_saved_ms: list[float] = []

    @staticmethod
    def key_for(
        query: str,
        filters: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> str:
        payload = {"query": query, "filters": filters or {}, "config": config or {}}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return "resp:" + hashlib.sha256(canonical.encode()).hexdigest()

    def get(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> ResponseCacheEntry | None:
        raw = self.backend.get(self.key_for(query, filters, config))
        if raw is None:
            self._misses += 1
            return None
        entry = ResponseCacheEntry.model_validate_json(raw)
        self._hits += 1
        self._latency_saved_ms.append(entry.compute_latency_ms)
        return entry

    def put(
        self,
        query: str,
        response: str,
        compute_latency_ms: float,
        filters: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        entry = ResponseCacheEntry(response=response, compute_latency_ms=compute_latency_ms)
        self.backend.set(
            self.key_for(query, filters, config),
            entry.model_dump_json().encode(),
            ttl_seconds=self.ttl_seconds,
        )

    def get_or_compute(
        self,
        query: str,
        compute_fn: Callable[[], str],
        filters: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> tuple[str, bool]:
        """Returns ``(response, was_hit)``. Calls ``compute_fn`` (the real pipeline) only on a
        miss, timing it so the *next* hit can report an honest latency-saved number."""
        cached = self.get(query, filters, config)
        if cached is not None:
            return cached.response, True
        t0 = time.perf_counter()
        response = compute_fn()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.put(query, response, latency_ms, filters, config)
        return response, False

    @property
    def stats(self) -> ResponseCacheStats:
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total else 0.0
        saved_total = sum(self._latency_saved_ms)
        saved_mean = (saved_total / len(self._latency_saved_ms)) if self._latency_saved_ms else 0.0
        return ResponseCacheStats(
            hits=self._hits,
            misses=self._misses,
            total=total,
            hit_rate=hit_rate,
            latency_saved_ms_total=saved_total,
            latency_saved_ms_mean=saved_mean,
        )
