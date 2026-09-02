"""Stage 8-9 of the §7 pipeline: pack tokenized documents to `seq_len` and write
memory-mapped `uint16` shards.

Packing concatenates document token streams so a row wastes zero padding except
possibly the very last row of the very last shard (a document is split across a row
boundary rather than padding the row early). Every row also records the offsets where
each document (or document fragment) starts, so a consumer can build a block-diagonal
"don't attend across documents" mask -- or *not*, which is the naive-packing ablation
arm the spec calls for. That choice is deliberately deferred to `loader.py` (read
time): the same shards serve both arms, nothing needs to be packed twice.

`uint16` is valid because every LocalMind tokenizer config has `vocab_size == 16384 <
65536` (see `configs/model/*.yaml`); `write_shard` asserts this rather than assuming it.

No torch, no `datasets`/`datasketch` -- this module only needs numpy + stdlib, so it is
always importable and always tested.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "PackedRow",
    "ShardMeta",
    "build_doc_mask",
    "doc_ids_from_boundaries",
    "pack_documents",
    "rows_per_shard",
    "write_shard",
]


@dataclass(frozen=True, slots=True)
class PackedRow:
    """One packed training row: `seq_len + 1` tokens (so callers can slice
    `x = tokens[:-1]`, `y = tokens[1:]` without stitching across rows).

    `boundaries` are sorted token offsets, always including 0, where a new document (or
    document fragment, if a document was split across a row) starts within this row --
    the input to `doc_ids_from_boundaries` / `build_doc_mask`. `pad_start` is the offset
    where trailing padding begins; equal to `len(tokens)` if the row has none.
    """

    tokens: np.ndarray
    boundaries: list[int]
    pad_start: int


@dataclass(frozen=True, slots=True)
class ShardMeta:
    bin_path: Path
    idx_path: Path
    n_rows: int
    seq_len: int
    content_hash: str


def rows_per_shard(
    seq_len: int, target_bytes: int = 256 * 1024 * 1024, dtype_bytes: int = 2
) -> int:
    """How many `(seq_len + 1)`-token uint16 rows fit in a `target_bytes` shard
    (spec target: ~256 MB per `.bin`)."""
    row_bytes = (seq_len + 1) * dtype_bytes
    return max(1, target_bytes // row_bytes)


def pack_documents(
    token_lists: Iterable[Sequence[int]],
    seq_len: int,
    pad_id: int = 0,
) -> Iterator[PackedRow]:
    """Greedily concatenate tokenized documents into fixed-length `seq_len + 1` rows.

    A document longer than one row is split across rows (no truncation, no early
    padding); only the final row -- emitted once the input stream is exhausted -- may
    be padded, and only up to `seq_len + 1 - len(remaining_tokens)` positions.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    row_len = seq_len + 1
    buf: list[int] = []
    doc_starts: list[int] = []  # offsets into `buf` where a document begins

    def _row_boundaries(limit: int) -> list[int]:
        b = sorted(s for s in doc_starts if s < limit)
        if not b or b[0] != 0:
            b = [0, *b]
        return b

    for doc_tokens in token_lists:
        if len(doc_tokens) == 0:
            continue
        doc_starts.append(len(buf))
        buf.extend(doc_tokens)
        while len(buf) >= row_len:
            row_tokens = buf[:row_len]
            boundaries = _row_boundaries(row_len)
            yield PackedRow(
                tokens=np.asarray(row_tokens, dtype=np.uint16),
                boundaries=boundaries,
                pad_start=row_len,
            )
            buf = buf[row_len:]
            doc_starts = [s - row_len for s in doc_starts if s >= row_len]

    if buf:
        pad_start = len(buf)
        n_pad = row_len - len(buf)
        row_tokens = buf + [pad_id] * n_pad
        boundaries = _row_boundaries(row_len)
        yield PackedRow(
            tokens=np.asarray(row_tokens, dtype=np.uint16),
            boundaries=boundaries,
            pad_start=pad_start,
        )


def doc_ids_from_boundaries(boundaries: Sequence[int], length: int) -> np.ndarray:
    """Expand sorted boundary offsets into a per-position local segment id array of
    shape `(length,)`, e.g. `boundaries=[0, 3]`, `length=5` -> `[0, 0, 0, 1, 1]`.
    """
    seg_ids = np.zeros(length, dtype=np.int32)
    bounds = sorted(boundaries)
    for k, start in enumerate(bounds):
        end = bounds[k + 1] if k + 1 < len(bounds) else length
        seg_ids[start:end] = k
    return seg_ids


def build_doc_mask(boundaries: Sequence[int], length: int) -> np.ndarray:
    """Block-diagonal causal attention-allowed mask, shape `(length, length)`, bool.

    `mask[i, j]` is True iff `j <= i` (causal) and `i`, `j` fall in the same document
    segment. This is the "correct" packing arm; the naive-packing ablation is simply
    *not* applying this (full causal mask over the whole packed row instead) -- see
    `loader.PackedShardLoader(use_doc_boundaries=False)`.
    """
    seg_ids = doc_ids_from_boundaries(boundaries, length)
    same_segment = seg_ids[:, None] == seg_ids[None, :]
    causal = np.tril(np.ones((length, length), dtype=bool))
    return same_segment & causal


def write_shard(
    rows: Sequence[PackedRow],
    shard_path: str | Path,
    *,
    vocab_size: int,
    pad_id: int = 0,
    seed: int = 0,
    extra_meta: dict | None = None,
) -> ShardMeta:
    """Write one memory-mappable shard: `<shard_path>.bin` (raw uint16 token matrix,
    shape `(n_rows, seq_len + 1)`) + `<shard_path>.idx` (JSON metadata: boundaries, pad
    offsets, and a sha256 content hash of the `.bin` bytes -- the manifest's per-shard
    content hash, per the DoD).
    """
    if vocab_size > 65536:
        raise ValueError(f"vocab_size {vocab_size} exceeds uint16 range (must be <= 65536)")
    if not rows:
        raise ValueError("cannot write an empty shard")

    row_len = rows[0].tokens.shape[0]
    for row in rows:
        if row.tokens.shape[0] != row_len:
            raise ValueError("all rows in a shard must share the same length")

    arr = np.stack([row.tokens for row in rows]).astype(np.uint16, copy=False)

    shard_path = Path(shard_path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path = shard_path.with_suffix(".bin")
    idx_path = shard_path.with_suffix(".idx")

    bin_bytes = arr.tobytes()
    bin_path.write_bytes(bin_bytes)
    content_hash = hashlib.sha256(bin_bytes).hexdigest()

    idx = {
        "seq_len": row_len - 1,
        "n_rows": len(rows),
        "pad_id": pad_id,
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "content_hash": content_hash,
        "boundaries": [row.boundaries for row in rows],
        "pad_starts": [row.pad_start for row in rows],
        "seed": seed,
        **(extra_meta or {}),
    }
    idx_path.write_text(json.dumps(idx, indent=2), encoding="utf-8")

    return ShardMeta(
        bin_path=bin_path,
        idx_path=idx_path,
        n_rows=len(rows),
        seq_len=row_len - 1,
        content_hash=content_hash,
    )
