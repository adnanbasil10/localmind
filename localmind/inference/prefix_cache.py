"""Automatic prefix caching: a radix tree over KV blocks (implementation.md section 10 step 6).

Why this matters more for RAG than for chat
-------------------------------------------
A RAG prompt is ``[system prompt][retrieved chunk 1..k][user question]``. The system
prompt is byte-identical on *every* request, and retrieved chunks repeat heavily across
a session because the retriever keeps surfacing the same passages. Those tokens are
recomputed from scratch on every request unless something remembers them -- and prefill
is the whole of TTFT, so the saving lands directly on the metric users feel first.

A radix (compressed prefix) tree is the right structure because prompts share *variable
length* prefixes: a plain hash of the full prompt hits only on an exact repeat, while a
radix tree matches the longest common prefix of any depth and shares interior nodes
between prompts that diverge later.

What is stored
--------------
Each edge owns the dense KV for exactly the tokens on that edge, shaped
``(1, n_kv_heads, edge_len, head_dim)`` per layer. A match walks the tree collecting edge
segments and concatenates them once. Eviction is LRU over leaves under a byte budget,
which is the standard policy: an interior node is still reachable by other prompts, so
evicting it would throw away more than it frees.

Honest note on granularity: a production engine keeps the cache in *blocks* and hands the
scheduler block ids, so a hit costs zero copies. Here the tree owns dense tensors and the
scheduler copies them into the paged pool via ``PagedKVCache.write_dense``. The TTFT win
is real and measured; the copy is an overhead a block-level implementation would not pay.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from localmind.inference.kv_cache import LayerKV
from localmind.model.config import ModelConfig

__all__ = ["RadixNode", "RadixPrefixCache", "common_prefix_len"]


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two token sequences."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


@dataclass
class RadixNode:
    """One compressed edge. ``tokens`` is the edge label; ``kv`` is that edge's KV."""

    tokens: list[int] = field(default_factory=list)
    kv: list[LayerKV] | None = None
    children: dict[int, RadixNode] = field(default_factory=dict)
    parent: RadixNode | None = field(default=None, repr=False)
    last_access: float = 0.0
    hits: int = 0

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def nbytes(self) -> int:
        if self.kv is None:
            return 0
        return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in self.kv)


class RadixPrefixCache:
    """Longest-prefix KV reuse with LRU eviction under a byte budget."""

    def __init__(
        self,
        cfg: ModelConfig,
        max_bytes: int = 64 * 1024 * 1024,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        min_prefix_len: int = 8,
    ) -> None:
        self.cfg = cfg
        self.max_bytes = max_bytes
        self.device = torch.device(device)
        self.dtype = dtype
        #: Do not cache trivially short prefixes: the bookkeeping costs more than the
        #: forward it saves.
        self.min_prefix_len = min_prefix_len
        self.root = RadixNode()
        self.bytes_used = 0
        self.queries = 0
        self.hits = 0
        self.matched_tokens = 0
        self.queried_tokens = 0
        self.evictions = 0
        self.inserts = 0

    # -- lookup ---------------------------------------------------------------------
    def match(self, token_ids: Sequence[int]) -> tuple[int, list[LayerKV] | None]:
        """Longest cached prefix of ``token_ids``.

        Returns ``(matched_len, dense_kv)`` where ``dense_kv`` is per-layer
        ``(1, n_kv_heads, matched_len, head_dim)``, or ``(0, None)`` on a miss.
        """
        self.queries += 1
        self.queried_tokens += len(token_ids)
        node = self.root
        i = 0
        segments: list[tuple[RadixNode, int]] = []
        now = time.perf_counter()
        while i < len(token_ids):
            child = node.children.get(token_ids[i])
            if child is None or child.kv is None:
                break
            n = common_prefix_len(child.tokens, token_ids[i:])
            if n == 0:
                break
            segments.append((child, n))
            child.last_access = now
            i += n
            if n < len(child.tokens):
                break
            node = child

        if i == 0:
            return 0, None
        self.hits += 1
        self.matched_tokens += i
        for child, _ in segments:
            child.hits += 1
        return i, self._concat(segments)

    def _concat(self, segments: Sequence[tuple[RadixNode, int]]) -> list[LayerKV]:
        out: list[LayerKV] = []
        for layer in range(self.cfg.n_layers):
            ks = []
            vs = []
            for node, n in segments:
                assert node.kv is not None
                k, v = node.kv[layer]
                ks.append(k[:, :, :n, :])
                vs.append(v[:, :, :n, :])
            out.append((torch.cat(ks, dim=2), torch.cat(vs, dim=2)))
        return out

    # -- insertion ------------------------------------------------------------------
    def insert(self, token_ids: Sequence[int], dense_kv: Sequence[LayerKV], length: int) -> int:
        """Store the KV for ``token_ids[:length]``. Returns the number of tokens newly added."""
        ids = list(token_ids[:length])
        if len(ids) < self.min_prefix_len:
            return 0
        node = self.root
        i = 0
        while i < len(ids):
            first = ids[i]
            child = node.children.get(first)
            if child is None:
                seg = ids[i:]
                new = RadixNode(
                    tokens=seg,
                    kv=self._slice(dense_kv, i, len(ids)),
                    parent=node,
                    last_access=time.perf_counter(),
                )
                node.children[first] = new
                self.bytes_used += new.nbytes()
                self.inserts += 1
                self._evict_if_needed()
                return len(seg)
            n = common_prefix_len(child.tokens, ids[i:])
            if n < len(child.tokens):
                self._split(child, n)
            i += n
            node = child
            if i >= len(ids):
                return 0
        return 0

    def _split(self, node: RadixNode, n: int) -> None:
        """Break ``node``'s edge after ``n`` tokens, pushing the tail into a new child."""
        tail = RadixNode(
            tokens=node.tokens[n:],
            kv=None if node.kv is None else [(k[:, :, n:, :], v[:, :, n:, :]) for k, v in node.kv],
            children=node.children,
            parent=node,
            last_access=node.last_access,
        )
        for c in tail.children.values():
            c.parent = tail
        node.children = {tail.tokens[0]: tail}
        node.tokens = node.tokens[:n]
        if node.kv is not None:
            node.kv = [(k[:, :, :n, :], v[:, :, :n, :]) for k, v in node.kv]

    @staticmethod
    def _slice(dense_kv: Sequence[LayerKV], start: int, stop: int) -> list[LayerKV]:
        return [
            (k[:, :, start:stop, :].clone(), v[:, :, start:stop, :].clone()) for k, v in dense_kv
        ]

    # -- eviction -------------------------------------------------------------------
    def _leaves(self) -> list[RadixNode]:
        out: list[RadixNode] = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n is not self.root and n.is_leaf:
                out.append(n)
            stack.extend(n.children.values())
        return out

    def _evict_if_needed(self) -> None:
        while self.bytes_used > self.max_bytes:
            leaves = self._leaves()
            if not leaves:
                return
            victim = min(leaves, key=lambda n: n.last_access)
            parent = victim.parent
            if parent is None:
                return
            parent.children.pop(victim.tokens[0], None)
            self.bytes_used -= victim.nbytes()
            self.evictions += 1

    def clear(self) -> None:
        self.root = RadixNode()
        self.bytes_used = 0

    # -- reporting ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "queries": self.queries,
            "hits": self.hits,
            "hit_rate": self.hits / self.queries if self.queries else 0.0,
            "matched_tokens": self.matched_tokens,
            "queried_tokens": self.queried_tokens,
            #: The metric that predicts the TTFT saving: fraction of prompt tokens that
            #: never had to be prefilled.
            "token_hit_rate": (
                self.matched_tokens / self.queried_tokens if self.queried_tokens else 0.0
            ),
            "bytes_used": self.bytes_used,
            "max_bytes": self.max_bytes,
            "evictions": self.evictions,
            "inserts": self.inserts,
            "nodes": self.n_nodes(),
        }

    def n_nodes(self) -> int:
        count = 0
        stack = [self.root]
        while stack:
            n = stack.pop()
            count += 1
            stack.extend(n.children.values())
        return count - 1
