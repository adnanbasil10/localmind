"""KV-cache storage: dynamic -> contiguous -> paged (implementation.md section 10, steps 2-3).

Three storage strategies behind one read interface (``as_past`` / ``gather``), so
``engine.py`` and ``scheduler.py`` can be pointed at any of them and benchmarked against
each other without touching the model.

Why this ladder exists
----------------------
**Dynamic** is what you get for free: the model's ``forward`` concatenates the new
``(k, v)`` onto the past and hands back a fresh tensor, so a T-token generation performs
T allocations whose sizes grow linearly -- O(T^2) bytes allocated in total, and the
allocator has to find a *new, larger, contiguous* block every single step.

**Contiguous** preallocates ``[B, n_kv_heads, max_len, head_dim]`` once per layer and
copies only the newly produced tail into it. One allocation for the whole request, a
hard memory ceiling known before the first token, and no allocator search in the decode
loop. Its cost is that every sequence reserves ``max_len`` whether it needs it or not --
which is exactly the waste paged storage exists to remove.

**Paged** (vLLM, Kwon et al. 2023) chops the cache into fixed ``block_size``-token blocks
drawn from a shared pool and gives each sequence a *block table*. A sequence's blocks
need not be adjacent, so external fragmentation goes to zero by construction and the only
remaining waste is at most ``block_size - 1`` tokens in the last block. That is the whole
trick, and it is why an engine can hold several times more concurrent sequences in the
same bytes.

Honest limitation
-----------------
Real paged attention never materialises the gathered cache: a custom kernel walks the
block table inside the attention op. ``localmind/model/attention.py`` is owned by Phase 2
and takes a dense ``past_kv``, so :meth:`PagedKVCache.gather` copies blocks into a dense
tensor before the forward. The *memory* results below are therefore exact and the
*latency* results carry a gather overhead a fused kernel would not pay. This is measured
and reported rather than hidden.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from localmind.model.config import ModelConfig

__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "BlockAllocator",
    "ContiguousKVCache",
    "DynamicKVCache",
    "KVCacheProtocol",
    "LayerKV",
    "OutOfBlocksError",
    "PagedKVCache",
    "SequenceBlockTable",
    "fragmentation_report",
    "max_concurrent_sequences",
]

#: One layer's cache: ``(k, v)``, each ``(B, n_kv_heads, T, head_dim)``. Matches
#: ``localmind.model.attention.KVCache`` exactly -- do not redefine the shape here.
LayerKV = tuple[Tensor, Tensor]

#: vLLM's default and the value implementation.md section 10 step 3 names.
DEFAULT_BLOCK_SIZE = 16


class OutOfBlocksError(RuntimeError):
    """The block pool is exhausted. The scheduler catches this and preempts."""


@runtime_checkable
class KVCacheProtocol(Protocol):
    """What ``engine.py`` needs from a cache: read it, grow it, throw it away."""

    length: int

    def as_past(self) -> list[LayerKV] | None: ...
    def extend_from(self, model_kv: Sequence[LayerKV]) -> None: ...
    def reset(self) -> None: ...


# ---------------------------------------------------------------------------------
# 1. Dynamic -- the free baseline
# ---------------------------------------------------------------------------------
@dataclass
class DynamicKVCache:
    """Holds the model's returned ``(k, v)`` list verbatim. The denominator for step 2.

    Every ``extend_from`` drops the previous tensors and adopts the model's freshly
    concatenated ones, so peak allocation grows with context and the allocator does a
    fresh search on every decode step.
    """

    layers: list[LayerKV] = field(default_factory=list)

    @property
    def length(self) -> int:  # type: ignore[override]
        return 0 if not self.layers else int(self.layers[0][0].shape[-2])

    def as_past(self) -> list[LayerKV] | None:
        return self.layers or None

    def extend_from(self, model_kv: Sequence[LayerKV]) -> None:
        self.layers = list(model_kv)

    def reset(self) -> None:
        self.layers = []

    def bytes_used(self) -> int:
        return sum(
            k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in self.layers
        )

    def bytes_reserved(self) -> int:
        return self.bytes_used()


# ---------------------------------------------------------------------------------
# 2. Contiguous -- preallocate [B, n_kv_heads, max_len, head_dim]
# ---------------------------------------------------------------------------------
class ContiguousKVCache:
    """One preallocated ``[B, n_kv_heads, max_len, head_dim]`` buffer per layer, per k/v.

    ``as_past`` returns ``narrow`` **views** into the buffers, so no copy happens on the
    read side. ``extend_from`` copies only the tokens the model just produced.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        batch_size: int = 1,
        max_len: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.batch_size = batch_size
        self.max_len = int(max_len or cfg.max_seq_len)
        self.dtype = dtype
        self.device = torch.device(device)
        shape = (batch_size, cfg.n_kv_heads, self.max_len, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(cfg.n_layers)]
        self.length = 0

    def as_past(self) -> list[LayerKV] | None:
        if self.length == 0:
            return None
        n = self.length
        return [(self.k[i][:, :, :n, :], self.v[i][:, :, :n, :]) for i in range(self.cfg.n_layers)]

    def extend_from(self, model_kv: Sequence[LayerKV]) -> None:
        """Copy the tail of the model's returned caches into the preallocated buffers."""
        new_len = int(model_kv[0][0].shape[-2])
        if new_len > self.max_len:
            raise OutOfBlocksError(f"contiguous cache holds {self.max_len} tokens, need {new_len}")
        start = self.length
        for i, (k, v) in enumerate(model_kv):
            self.k[i][:, :, start:new_len, :].copy_(k[:, :, start:new_len, :])
            self.v[i][:, :, start:new_len, :].copy_(v[:, :, start:new_len, :])
        self.length = new_len

    def reset(self) -> None:
        self.length = 0

    def bytes_reserved(self) -> int:
        per = self.k[0].numel() * self.k[0].element_size()
        return 2 * self.cfg.n_layers * per

    def bytes_used(self) -> int:
        if self.max_len == 0:
            return 0
        return int(self.bytes_reserved() * self.length / self.max_len)


# ---------------------------------------------------------------------------------
# 3. Paged -- block pool + per-sequence block table (vLLM-style)
# ---------------------------------------------------------------------------------
@dataclass
class SequenceBlockTable:
    """One sequence's view of the pool: which physical blocks hold its tokens."""

    seq_id: str
    block_ids: list[int] = field(default_factory=list)
    length: int = 0
    #: Tokens inherited from a shared prefix. Those blocks are refcounted, not owned.
    shared_prefix_len: int = 0

    def slot(self, position: int, block_size: int) -> tuple[int, int]:
        """Map a logical token position to ``(physical_block_id, offset_in_block)``."""
        if position >= self.length:
            raise IndexError(f"position {position} >= length {self.length}")
        return self.block_ids[position // block_size], position % block_size


class BlockAllocator:
    """A free list over ``num_blocks`` physical blocks, with reference counting.

    Refcounts are what make prefix sharing safe: two sequences that share a prompt point
    at the same physical blocks, and a block returns to the free list only when the last
    referent releases it.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))  # pop() -> low ids first
        self._refs: dict[int, int] = {}
        self.total_allocated = 0
        self.total_freed = 0

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def allocate(self, n: int = 1) -> list[int]:
        if n > len(self._free):
            raise OutOfBlocksError(
                f"requested {n} blocks, {len(self._free)} free of {self.num_blocks}"
            )
        out = [self._free.pop() for _ in range(n)]
        for b in out:
            self._refs[b] = 1
        self.total_allocated += n
        return out

    def incref(self, blocks: Iterable[int]) -> None:
        for b in blocks:
            self._refs[b] = self._refs.get(b, 0) + 1

    def free(self, blocks: Iterable[int]) -> None:
        """Decrement refcounts; return to the free list at zero."""
        for b in blocks:
            n = self._refs.get(b, 0) - 1
            if n <= 0:
                self._refs.pop(b, None)
                self._free.append(b)
                self.total_freed += 1
            else:
                self._refs[b] = n

    def refcount(self, block: int) -> int:
        return self._refs.get(block, 0)


class PagedKVCache:
    """Block-paged KV storage shared by every sequence in the engine.

    Pool layout per layer: ``[num_blocks, n_kv_heads, block_size, head_dim]`` for k and
    for v. A sequence's logical position ``p`` lives at
    ``pool[block_table[p // block_size], :, p % block_size, :]``.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        num_blocks: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)
        self.allocator = BlockAllocator(num_blocks)
        shape = (num_blocks, cfg.n_kv_heads, block_size, cfg.head_dim)
        self.k_pool = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(cfg.n_layers)
        ]
        self.v_pool = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(cfg.n_layers)
        ]
        self.tables: dict[str, SequenceBlockTable] = {}
        self.gather_calls = 0
        self.gathered_tokens = 0

    # -- capacity accounting --------------------------------------------------------
    @property
    def num_blocks(self) -> int:
        return self.allocator.num_blocks

    @property
    def block_bytes(self) -> int:
        """Bytes one physical block costs across all layers (k and v)."""
        per = self.cfg.n_kv_heads * self.block_size * self.cfg.head_dim
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        return 2 * self.cfg.n_layers * per * itemsize

    def bytes_reserved(self) -> int:
        return self.num_blocks * self.block_bytes

    def bytes_allocated(self) -> int:
        """Bytes held by live block tables (whole blocks, so this includes slack)."""
        return self.allocator.num_used * self.block_bytes

    def bytes_used(self) -> int:
        """Bytes actually carrying tokens -- the numerator of the fragmentation ratio."""
        per_token = self.block_bytes / self.block_size
        return int(sum(t.length for t in self.tables.values()) * per_token)

    def internal_fragmentation(self) -> float:
        """Fraction of *allocated* bytes that hold no token. Bounded by 1/block_size."""
        alloc = self.bytes_allocated()
        return 0.0 if alloc == 0 else 1.0 - self.bytes_used() / alloc

    # -- sequence lifecycle ---------------------------------------------------------
    def add_sequence(self, seq_id: str) -> SequenceBlockTable:
        if seq_id in self.tables:
            raise KeyError(f"sequence {seq_id!r} already present")
        table = SequenceBlockTable(seq_id=seq_id)
        self.tables[seq_id] = table
        return table

    def free_sequence(self, seq_id: str) -> None:
        table = self.tables.pop(seq_id, None)
        if table is not None:
            self.allocator.free(table.block_ids)

    def has(self, seq_id: str) -> bool:
        return seq_id in self.tables

    def length_of(self, seq_id: str) -> int:
        return self.tables[seq_id].length

    def blocks_needed(self, seq_id: str, extra_tokens: int) -> int:
        """How many *new* blocks appending ``extra_tokens`` would require."""
        t = self.tables[seq_id]
        have = len(t.block_ids) * self.block_size
        return max(0, math.ceil((t.length + extra_tokens - have) / self.block_size))

    def reserve(self, seq_id: str, extra_tokens: int) -> None:
        need = self.blocks_needed(seq_id, extra_tokens)
        if need:
            self.tables[seq_id].block_ids.extend(self.allocator.allocate(need))

    # -- data movement --------------------------------------------------------------
    def append(self, seq_id: str, model_kv: Sequence[LayerKV], batch_index: int = 0) -> None:
        """Scatter the tokens the model just produced into this sequence's blocks.

        ``model_kv`` is the model's full concatenated cache for the row; only positions
        ``[table.length, new_len)`` are new and get written.
        """
        table = self.tables[seq_id]
        new_len = int(model_kv[0][0].shape[-2])
        n_new = new_len - table.length
        if n_new <= 0:
            return
        self.reserve(seq_id, n_new)
        for layer, (k, v) in enumerate(model_kv):
            k_src = k[batch_index]  # (n_kv_heads, T, head_dim)
            v_src = v[batch_index]
            for j in range(n_new):
                pos = table.length + j
                block = table.block_ids[pos // self.block_size]
                off = pos % self.block_size
                self.k_pool[layer][block, :, off, :] = k_src[:, pos, :]
                self.v_pool[layer][block, :, off, :] = v_src[:, pos, :]
        table.length = new_len

    def gather(self, seq_id: str, batch_dim: bool = True) -> list[LayerKV] | None:
        """Materialise this sequence's cache as dense ``(1, n_kv_heads, L, head_dim)``.

        This copy is the price of not owning the attention kernel; see the module
        docstring. ``gather_calls`` / ``gathered_tokens`` are counted so the overhead is
        reportable rather than invisible.
        """
        table = self.tables[seq_id]
        if table.length == 0:
            return None
        self.gather_calls += 1
        self.gathered_tokens += table.length
        n_blocks = math.ceil(table.length / self.block_size)
        idx = torch.tensor(table.block_ids[:n_blocks], dtype=torch.long, device=self.device)
        out: list[LayerKV] = []
        for layer in range(self.cfg.n_layers):
            # (n_blocks, H, block, D) -> (H, n_blocks*block, D) -> trim to length
            k = self.k_pool[layer].index_select(0, idx).permute(1, 0, 2, 3)
            v = self.v_pool[layer].index_select(0, idx).permute(1, 0, 2, 3)
            k = k.reshape(self.cfg.n_kv_heads, n_blocks * self.block_size, self.cfg.head_dim)
            v = v.reshape(self.cfg.n_kv_heads, n_blocks * self.block_size, self.cfg.head_dim)
            k = k[:, : table.length, :]
            v = v[:, : table.length, :]
            out.append((k.unsqueeze(0), v.unsqueeze(0)) if batch_dim else (k, v))
        return out

    def write_dense(self, seq_id: str, dense_kv: Sequence[LayerKV], n_tokens: int) -> None:
        """Bulk-load ``n_tokens`` of precomputed KV (a prefix-cache hit) into blocks."""
        table = self.tables[seq_id]
        self.reserve(seq_id, n_tokens - table.length)
        for layer, (k, v) in enumerate(dense_kv):
            k_src = k[0] if k.dim() == 4 else k
            v_src = v[0] if v.dim() == 4 else v
            for pos in range(table.length, n_tokens):
                block = table.block_ids[pos // self.block_size]
                off = pos % self.block_size
                self.k_pool[layer][block, :, off, :] = k_src[:, pos, :]
                self.v_pool[layer][block, :, off, :] = v_src[:, pos, :]
        table.length = n_tokens

    # -- prefix sharing -------------------------------------------------------------
    def fork(self, child_id: str, parent_id: str, num_tokens: int) -> SequenceBlockTable:
        """Share the parent's first ``num_tokens`` tokens with a new sequence.

        Only *whole* blocks are shared; a partially filled boundary block is copied so
        the child can keep writing without disturbing the parent (copy-on-write at block
        granularity, which is what vLLM does).
        """
        parent = self.tables[parent_id]
        if num_tokens > parent.length:
            raise ValueError(f"parent holds {parent.length} tokens, asked to fork {num_tokens}")
        whole = num_tokens // self.block_size
        shared = parent.block_ids[:whole]
        self.allocator.incref(shared)
        child = SequenceBlockTable(
            seq_id=child_id,
            block_ids=list(shared),
            length=whole * self.block_size,
            shared_prefix_len=whole * self.block_size,
        )
        self.tables[child_id] = child
        remainder = num_tokens - child.length
        if remainder:
            (dst,) = self.allocator.allocate(1)
            src = parent.block_ids[whole]
            for layer in range(self.cfg.n_layers):
                self.k_pool[layer][dst, :, :remainder, :] = self.k_pool[layer][
                    src, :, :remainder, :
                ]
                self.v_pool[layer][dst, :, :remainder, :] = self.v_pool[layer][
                    src, :, :remainder, :
                ]
            child.block_ids.append(dst)
            child.length = num_tokens
        return child


# ---------------------------------------------------------------------------------
# Fragmentation accounting (the step-3 deliverable)
# ---------------------------------------------------------------------------------
def max_concurrent_sequences(
    cfg: ModelConfig,
    budget_bytes: int,
    seq_len: int,
    paged: bool,
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_len: int | None = None,
    dtype_bytes: int = 4,
) -> int:
    """How many sequences of ``seq_len`` tokens fit in ``budget_bytes``.

    Contiguous must reserve ``max_len`` per sequence up front, because the sequence may
    generate right up to it and the buffer cannot grow. Paged reserves only the blocks
    the tokens actually occupy.
    """
    per_token = 2 * cfg.n_kv_heads * cfg.head_dim * cfg.n_layers * dtype_bytes
    if paged:
        per_seq = math.ceil(seq_len / block_size) * block_size * per_token
    else:
        per_seq = int(max_len or cfg.max_seq_len) * per_token
    return 0 if per_seq == 0 else budget_bytes // per_seq


def fragmentation_report(
    cfg: ModelConfig,
    seq_lens: Sequence[int],
    budget_bytes: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_len: int | None = None,
    dtype_bytes: int = 4,
) -> dict[str, Any]:
    """Contiguous vs paged waste for a concrete mix of sequence lengths.

    Two kinds of waste are separated because they have different causes and different
    fixes:

    * **internal** -- bytes inside a region a sequence owns but has not filled. Paged
      storage bounds this at ``block_size - 1`` tokens per sequence; contiguous storage
      wastes ``max_len - len``, which for a 40-token request in a 1024-token window is
      96% of the reservation.
    * **external** -- free bytes that exist but are unusable because no single free run
      is large enough. Contiguous allocation suffers this whenever variable-length
      requests come and go; paged allocation makes it *identically zero*, because every
      free block is interchangeable.
    """
    ml = int(max_len or cfg.max_seq_len)
    per_token = 2 * cfg.n_kv_heads * cfg.head_dim * cfg.n_layers * dtype_bytes
    lens = [int(n) for n in seq_lens]
    used = sum(lens) * per_token

    cont_alloc = len(lens) * ml * per_token
    paged_alloc = sum(math.ceil(n / block_size) * block_size for n in lens) * per_token

    # External fragmentation model for the contiguous arena: sequences are packed as
    # fixed max_len slots, so leftover space smaller than one slot can never be used.
    cont_slots = budget_bytes // (ml * per_token)
    cont_external = budget_bytes - cont_slots * ml * per_token
    # Paged: leftover smaller than one block. Blocks are interchangeable, so the only
    # unusable tail is the sub-block remainder of the arena itself.
    block_bytes = block_size * per_token
    paged_external = budget_bytes % block_bytes

    return {
        "bench": "fragmentation",
        "n_sequences": len(lens),
        "block_size": block_size,
        "max_len": ml,
        "bytes_per_token": per_token,
        "bytes_of_real_tokens": used,
        "contiguous": {
            "bytes_reserved": cont_alloc,
            "internal_fragmentation": 0.0 if cont_alloc == 0 else 1.0 - used / cont_alloc,
            "external_fragmentation_bytes": cont_external,
            "external_fragmentation": cont_external / budget_bytes if budget_bytes else 0.0,
            "max_concurrent_at_budget": cont_slots,
        },
        "paged": {
            "bytes_reserved": paged_alloc,
            "internal_fragmentation": 0.0 if paged_alloc == 0 else 1.0 - used / paged_alloc,
            "external_fragmentation_bytes": paged_external,
            "external_fragmentation": paged_external / budget_bytes if budget_bytes else 0.0,
            "max_concurrent_at_budget": (
                min(
                    budget_bytes // max(1, math.ceil(max(lens) / block_size) * block_bytes),
                    budget_bytes // block_bytes,
                )
                if lens
                else 0
            ),
        },
        "memory_amplification_contiguous_over_paged": (
            cont_alloc / paged_alloc if paged_alloc else float("nan")
        ),
    }
