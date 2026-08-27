"""Redis-backed caching, Layer 1: the embedding cache, plus the shared ``CacheBackend`` Protocol.

Key: ``hash(text) + model``. Metric to report: hit rate and embedding calls avoided (every hit is
one fewer call to the real embedding model — on CPU-bound inference that's the whole point).

Environment note (task brief): Redis is not running in this environment and ``redis`` is an
optional dependency, so every cache in this package is written against a small
:class:`CacheBackend` Protocol with an in-memory fake (:class:`InMemoryCacheBackend`) good enough
to fully exercise and test all three cache layers offline. :class:`RedisCacheBackend` is the real
backend; it imports ``redis`` lazily inside ``__init__`` so ``import localmind.cache`` never
requires the package to be installed. Tests that need a live Redis are marked
``@pytest.mark.docker``.
"""

from __future__ import annotations

import hashlib
import struct
import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

__all__ = [
    "CacheBackend",
    "EmbeddingCache",
    "EmbeddingCacheStats",
    "InMemoryCacheBackend",
    "RedisCacheBackend",
]


@runtime_checkable
class CacheBackend(Protocol):
    """Minimal key-value contract every cache layer is built on.

    Deliberately small: get/set/delete/exists with an optional TTL. Both
    :class:`InMemoryCacheBackend` (used everywhere offline, including in tests) and
    :class:`RedisCacheBackend` (used in ``docker`` profile deployments, per docker-compose.yml's
    ``redis`` service) implement it identically from a caller's point of view.
    """

    def get(self, key: str) -> bytes | None: ...
    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...


class InMemoryCacheBackend:
    """In-process fake of :class:`CacheBackend`. No network, no Docker — fully testable offline.

    ``clock`` is injectable so tests can control TTL expiry deterministically instead of sleeping.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._store: dict[str, tuple[bytes, float | None]] = {}

    def get(self, key: str) -> bytes | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and self._clock() >= expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        expires_at = self._clock() + ttl_seconds if ttl_seconds is not None else None
        self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        return len(self._store)


class RedisCacheBackend:
    """Real :class:`CacheBackend` over Redis, matching docker-compose.yml's ``redis`` service
    (``redis://redis:6379/0`` inside the compose network, ``redis://localhost:6379/0`` from the
    host). ``redis`` is imported lazily so constructing anything else in this package never
    requires it to be installed; only instantiating *this* class does.
    """

    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "RedisCacheBackend requires the `redis` package "
                '(install with `uv pip install -e ".[rag]"`). '
                "Use InMemoryCacheBackend for offline development and tests."
            ) from exc
        self._client = redis.Redis.from_url(url)

    def get(self, key: str) -> bytes | None:
        return self._client.get(key)

    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        if ttl_seconds is not None:
            self._client.setex(key, ttl_seconds, value)
        else:
            self._client.set(key, value)

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def exists(self, key: str) -> bool:
        return bool(self._client.exists(key))


def _encode_vector(vector: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(vector, dtype=np.float32)
    header = struct.pack("<I", arr.size)
    return header + arr.tobytes()


def _decode_vector(raw: bytes) -> np.ndarray:
    (size,) = struct.unpack_from("<I", raw, 0)
    return np.frombuffer(raw, dtype=np.float32, count=size, offset=4).copy()


class EmbeddingCacheStats(BaseModel):
    hits: int
    misses: int
    total: int
    hit_rate: float
    embedding_calls_avoided: int
    """Equal to ``hits``: every hit is one call to the real embedding model that didn't happen."""


class EmbeddingCache:
    """Layer 1. Key: ``hash(text) + model``. Value: the embedding vector.

    Honest metric: hit rate, and embedding calls avoided (== hit count).
    """

    def __init__(
        self,
        backend: CacheBackend,
        model: str,
        ttl_seconds: int | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    @staticmethod
    def key_for(text: str, model: str) -> str:
        digest = hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()
        return f"emb:{model}:{digest}"

    def get(self, text: str) -> np.ndarray | None:
        raw = self.backend.get(self.key_for(text, self.model))
        if raw is None:
            self._misses += 1
            return None
        self._hits += 1
        return _decode_vector(raw)

    def put(self, text: str, embedding: np.ndarray) -> None:
        self.backend.set(
            self.key_for(text, self.model), _encode_vector(embedding), ttl_seconds=self.ttl_seconds
        )

    def get_or_compute(
        self, text: str, embed_fn: Callable[[str], np.ndarray]
    ) -> tuple[np.ndarray, bool]:
        """Returns ``(embedding, was_hit)``. Calls ``embed_fn`` (the real, presumably expensive
        embedding model) only on a miss."""
        cached = self.get(text)
        if cached is not None:
            return cached, True
        vector = np.asarray(embed_fn(text), dtype=np.float32)
        self.put(text, vector)
        return vector, False

    @property
    def stats(self) -> EmbeddingCacheStats:
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total else 0.0
        return EmbeddingCacheStats(
            hits=self._hits,
            misses=self._misses,
            total=total,
            hit_rate=hit_rate,
            embedding_calls_avoided=self._hits,
        )
