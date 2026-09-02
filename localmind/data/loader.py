"""Stage 10 of the §7 pipeline: the resumable memory-mapped dataloader.

Design note -- why this is trivially, exactly resumable (implementation.md §3.2
gotcha 3): the loader has **no shuffle buffer**. Instead, the sample order for epoch
`e` is `np.random.default_rng(SeedSequence([seed, e])).permutation(total_rows)` -- a
pure function of `(seed, e, total_rows)`. So the *entire* loader state that matters is
`(seed, step)`, where `step` is a monotonically increasing count of samples consumed.
Checkpointing it is one dict; restoring it means "recompute the same permutation,
seek to the same offset" -- no reconstructed queue, no replayed stream. That property
is what makes a 12-hour Kaggle session survivable: `state_dict()` goes into the same
checkpoint as the model/optimizer, and resume yields the identical next batch.

Reads mmap the `.bin` files written by `packing.write_shard` and reconstruct each row's
document-boundary mask from the `.idx` JSON on demand (only for sampled rows -- O(1)
per row, not O(shard)). `use_doc_boundaries` is the packing ablation switch: the exact
same shards serve the block-diagonal arm and the naive (bleeds across documents) arm.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from localmind.data.packing import doc_ids_from_boundaries

__all__ = ["PackedShardLoader"]


@dataclass(frozen=True, slots=True)
class _ShardHandle:
    idx_path: Path
    bin_path: Path
    n_rows: int
    seq_len: int
    pad_id: int
    boundaries: list[list[int]]
    content_hash: str
    mmap: np.ndarray


class PackedShardLoader:
    """Iterate `(x, y, doc_ids)` batches over a directory of packed shards.

    `next(loader)` (or `loader.next_batch()`) returns three `np.ndarray`s of shape
    `(batch_size, seq_len)`: `x`/`y` are int64 token ids (`y` is `x` shifted by one
    within the packed row), and `doc_ids` is int32 local-segment ids per position
    (all-zero, i.e. full causal attention, when `use_doc_boundaries=False`).
    """

    def __init__(
        self,
        shard_dir: str | Path,
        *,
        seed: int,
        batch_size: int,
        use_doc_boundaries: bool = True,
        manifest_name: str = "manifest.json",
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.seed = seed
        self.batch_size = batch_size
        self.use_doc_boundaries = use_doc_boundaries
        self._step = 0
        self._cached_epoch: int | None = None
        self._cached_order: np.ndarray | None = None

        idx_paths = self._discover_shards(manifest_name)
        self._shards: list[_ShardHandle] = [self._load_shard(p) for p in idx_paths]
        if not self._shards:
            raise ValueError(f"no shards found under {self.shard_dir}")

        seq_lens = {s.seq_len for s in self._shards}
        if len(seq_lens) != 1:
            raise ValueError(f"all shards must share seq_len, got {seq_lens}")
        self.seq_len = next(iter(seq_lens))

        self._row_counts = np.array([s.n_rows for s in self._shards], dtype=np.int64)
        self._cum_rows = np.concatenate([[0], np.cumsum(self._row_counts)])
        self.total_rows = int(self._cum_rows[-1])
        self.manifest_hash = self._manifest_fingerprint()

    # -- construction -----------------------------------------------------------
    def _discover_shards(self, manifest_name: str) -> list[Path]:
        manifest_path = self.shard_dir / manifest_name
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return [self.shard_dir / s["idx_file"] for s in manifest["shards"]]
        return sorted(self.shard_dir.glob("*.idx"))

    @staticmethod
    def _load_shard(idx_path: Path) -> _ShardHandle:
        meta = json.loads(idx_path.read_text(encoding="utf-8"))
        bin_path = idx_path.with_suffix(".bin")
        mmap = np.memmap(bin_path, dtype=np.uint16, mode="r")
        mmap = mmap.reshape(meta["n_rows"], meta["seq_len"] + 1)
        return _ShardHandle(
            idx_path=idx_path,
            bin_path=bin_path,
            n_rows=meta["n_rows"],
            seq_len=meta["seq_len"],
            pad_id=meta["pad_id"],
            boundaries=meta["boundaries"],
            content_hash=meta["content_hash"],
            mmap=mmap,
        )

    def _manifest_fingerprint(self) -> str:
        h = hashlib.sha256()
        for shard in self._shards:
            h.update(shard.content_hash.encode())
        return h.hexdigest()

    # -- deterministic, shuffle-buffer-free ordering -----------------------------
    def _order_for_epoch(self, epoch: int) -> np.ndarray:
        if self._cached_epoch != epoch:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
            self._cached_order = rng.permutation(self.total_rows)
            self._cached_epoch = epoch
        assert self._cached_order is not None
        return self._cached_order

    def _locate(self, global_row: int) -> tuple[int, int]:
        shard_id = int(np.searchsorted(self._cum_rows, global_row, side="right") - 1)
        row_in_shard = int(global_row - self._cum_rows[shard_id])
        return shard_id, row_in_shard

    # -- resumability -------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Everything needed to reproduce the identical remaining stream: the seed,
        how many samples have been consumed, and a fingerprint of the shard set so a
        checkpoint can't silently resume against a different (re-generated) dataset."""
        return {"seed": self.seed, "step": self._step, "manifest_hash": self.manifest_hash}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["seed"] != self.seed:
            raise ValueError(f"cannot resume: seed mismatch ({state['seed']} != {self.seed})")
        saved_hash = state.get("manifest_hash")
        if saved_hash is not None and saved_hash != self.manifest_hash:
            raise ValueError("cannot resume: shard set changed since this checkpoint was saved")
        self._step = int(state["step"])

    # -- iteration ------------------------------------------------------------------
    def __iter__(self) -> PackedShardLoader:
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.next_batch()

    def next_batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b = self.batch_size
        xs = np.empty((b, self.seq_len), dtype=np.int64)
        ys = np.empty((b, self.seq_len), dtype=np.int64)
        docs = np.zeros((b, self.seq_len), dtype=np.int32)

        for i in range(b):
            epoch, within_epoch = divmod(self._step, self.total_rows)
            order = self._order_for_epoch(epoch)
            global_row = int(order[within_epoch])
            shard_id, row_id = self._locate(global_row)
            shard = self._shards[shard_id]

            row = np.asarray(shard.mmap[row_id]).astype(np.int64)
            xs[i] = row[:-1]
            ys[i] = row[1:]
            if self.use_doc_boundaries:
                seg_ids = doc_ids_from_boundaries(shard.boundaries[row_id], self.seq_len + 1)
                docs[i] = seg_ids[:-1]
            self._step += 1

        return xs, ys, docs
