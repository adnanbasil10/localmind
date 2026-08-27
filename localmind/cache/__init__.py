"""Three Redis-backed cache layers, each behind a small :class:`CacheBackend` Protocol so all of
them are fully testable offline (Redis is not required to import or exercise this package):

- :class:`~localmind.cache.embedding_cache.EmbeddingCache` — Layer 1: ``hash(text)+model`` ->
  embedding. Hit rate, embedding calls avoided.
- :class:`~localmind.cache.response_cache.ResponseCache` — Layer 2:
  ``hash(query+filters+config)`` -> full response. Hit rate, latency saved.
- :class:`~localmind.cache.semantic_cache.SemanticCache` — Layer 3: nearest-neighbour over query
  embeddings, threshold tau. Hit rate **and false-hit rate** — see ``semantic_cache.py`` for the
  tau sweep (:func:`~localmind.cache.semantic_cache.sweep_tau`) that measures the latter honestly.

A fourth layer, KV prefix caching, lives in ``localmind.inference.prefix_cache`` (a different
task) and is out of scope here; its hit rate is surfaced through
``localmind.obs.metrics.Metrics.record_cache(layer="prefix", ...)`` alongside these three.

``import localmind.cache`` must succeed with only numpy/pydantic on the path — ``redis`` is
imported lazily, only inside :class:`~localmind.cache.embedding_cache.RedisCacheBackend.__init__`.
"""

from __future__ import annotations

from localmind.cache.embedding_cache import (
    CacheBackend,
    EmbeddingCache,
    EmbeddingCacheStats,
    InMemoryCacheBackend,
    RedisCacheBackend,
)
from localmind.cache.response_cache import ResponseCache, ResponseCacheEntry, ResponseCacheStats
from localmind.cache.semantic_cache import (
    Embedder,
    HashingEmbedder,
    NearestMatch,
    SemanticCache,
    SemanticCacheResult,
    SemanticCacheStats,
    SweepResult,
    SyntheticQAItem,
    build_synthetic_queryset,
    choose_operating_point,
    run_and_write_benchmark,
    sweep_tau,
)

__all__ = [
    "CacheBackend",
    "Embedder",
    "EmbeddingCache",
    "EmbeddingCacheStats",
    "HashingEmbedder",
    "InMemoryCacheBackend",
    "NearestMatch",
    "RedisCacheBackend",
    "ResponseCache",
    "ResponseCacheEntry",
    "ResponseCacheStats",
    "SemanticCache",
    "SemanticCacheResult",
    "SemanticCacheStats",
    "SweepResult",
    "SyntheticQAItem",
    "build_synthetic_queryset",
    "choose_operating_point",
    "run_and_write_benchmark",
    "sweep_tau",
]
