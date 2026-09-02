"""§7 Phase 3 orchestrator: the full pipeline end to end.

    stream -> license filter -> language ID -> quality heuristics -> MinHash-LSH
    near-dedup -> PII scrub -> eval-set decontamination -> tokenize -> pack -> shards

`prepare_shards` is deterministic given a seed: the mixture sampler, MinHash, and
packing are all seeded, so re-running against the same sources and eval set reproduces
byte-identical shards (same content hashes). It streams -- `iter_mixture` /
`hf_streaming_source` never materialize a raw corpus on disk; only tokenized, packed
shards are written.

`Tokenizer` is a frozen cross-task interface (CONVENTIONS.md, owned by the tokenizer
task). `TokenizerLike` below depends on it *structurally* -- only the two methods this
module actually calls -- so this module imports neither `localmind.tokenizer` nor torch
at module scope, and works with any object (real or fake) that has `encode` and
`vocab_size`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml
from pydantic import BaseModel, Field

from localmind.data.dedup import DecontamStats, DedupStats, decontaminate, near_dedup
from localmind.data.filter import (
    QualityThresholds,
    RawDoc,
    filter_language,
    filter_license,
    filter_quality,
    scrub_docs,
)
from localmind.data.packing import PackedRow, ShardMeta, pack_documents, rows_per_shard, write_shard

__all__ = [
    "Manifest",
    "MixtureConfig",
    "MixtureEntry",
    "ShardRecord",
    "SourceSpec",
    "TokenizerLike",
    "hf_streaming_source",
    "iter_mixture",
    "prepare_shards",
    "sources_from_mixture_config",
]


class TokenizerLike(Protocol):
    """The subset of `localmind.tokenizer.Tokenizer` this module actually calls.
    Structural (PEP 544): any object with these members satisfies it, including the
    `FakeTokenizer` test doubles used in `tests/test_data.py` -- no import of
    `localmind.tokenizer` required, here or at test time.
    """

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    @property
    def vocab_size(self) -> int: ...


# -------------------------------------------------------------------------------------
# Mixture config (§7 table) -- pydantic, loaded from YAML, no magic numbers in code.
# -------------------------------------------------------------------------------------
class MixtureEntry(BaseModel):
    name: str
    weight: float = Field(gt=0.0, le=1.0)
    license: str
    hf_path: str | None = None
    hf_name: str | None = None
    split: str | None = "train"
    text_field: str = "text"


class MixtureConfig(BaseModel):
    seq_len: int = Field(gt=0)
    seed: int
    allowed_licenses: list[str]
    sources: list[MixtureEntry]

    @classmethod
    def from_yaml(cls, path: str | Path) -> MixtureConfig:
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls.model_validate(raw)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """A runtime mixture source: a name/weight/license plus a zero-arg factory that
    returns a *fresh* iterable of `RawDoc` each call (so the weighted sampler can cycle
    a finite source, and so re-running with the same seed is deterministic).
    """

    name: str
    weight: float
    license: str
    factory: Callable[[], Iterable[RawDoc]]


def hf_streaming_source(entry: MixtureEntry) -> Iterator[RawDoc]:
    """Stream one §7 mixture source via `datasets.load_dataset(..., streaming=True)`.

    Lazy import: `datasets` is an optional `data`-extra dependency, and this function
    makes a real network call, so it must never run at module import time or in the
    default test suite (see `tests/test_data.py`'s `@pytest.mark.net` tests).
    """
    if entry.hf_path is None:
        raise ValueError(f"mixture entry {entry.name!r} has no hf_path to stream from")
    split = entry.split or "train"

    from datasets import load_dataset  # lazy: optional `data` extra, network at call time

    ds = load_dataset(entry.hf_path, entry.hf_name, split=split, streaming=True)
    for row in ds:
        text = row.get(entry.text_field)
        if not text:
            continue
        yield RawDoc(text=text, source=entry.name, license=entry.license)


def sources_from_mixture_config(config: MixtureConfig) -> list[SourceSpec]:
    """Wire each HF-backed `MixtureEntry` to `hf_streaming_source`. Entries with
    `hf_path=None` (e.g. a local domain corpus) are skipped -- the caller supplies its
    own `SourceSpec` for those, since loading local/proprietary data isn't this
    function's concern.
    """
    specs = []
    for entry in config.sources:
        if entry.hf_path is None:
            continue
        specs.append(
            SourceSpec(
                name=entry.name,
                weight=entry.weight,
                license=entry.license,
                factory=lambda e=entry: hf_streaming_source(e),
            )
        )
    return specs


def _cycle(factory: Callable[[], Iterable[RawDoc]]) -> Iterator[RawDoc]:
    """Restart a finite source's iterable forever. Real HF streaming sources are
    effectively infinite in practice; finite local/synthetic sources are expected to
    repeat once exhausted (this is also how the "start with TinyStories alone" §7
    bootstrap works: one small finite corpus, cycled).
    """
    while True:
        produced = False
        for item in factory():
            produced = True
            yield item
        if not produced:
            raise ValueError("mixture source produced zero documents")


def iter_mixture(sources: Sequence[SourceSpec], *, seed: int, n_docs: int) -> Iterator[RawDoc]:
    """Config-driven weighted sampler: draw `n_docs` documents from `sources`, each
    draw picking a source with probability proportional to its `weight`. Deterministic
    given `seed` (a single `np.random.default_rng(seed).choice` draw, independent of
    how fast any individual source iterator advances).
    """
    if not sources:
        raise ValueError("no sources provided")
    weights = np.array([s.weight for s in sources], dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("source weights must sum to a positive number")
    probs = weights / weights.sum()

    rng = np.random.default_rng(seed)
    iterators = [_cycle(s.factory) for s in sources]
    picks = rng.choice(len(sources), size=n_docs, p=probs)
    for idx in picks:
        yield next(iterators[int(idx)])


# -------------------------------------------------------------------------------------
# Manifest
# -------------------------------------------------------------------------------------
class ShardRecord(BaseModel):
    bin_file: str
    idx_file: str
    n_rows: int
    seq_len: int
    content_hash: str


class Manifest(BaseModel):
    seed: int
    seq_len: int
    vocab_size: int
    mixture: list[dict[str, str | float]]
    dedup: dict[str, float | int]
    decontam: dict[str, float | int]
    shards: list[ShardRecord]
    total_rows: int
    total_tokens: int

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")


# -------------------------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------------------------
def prepare_shards(
    sources: Sequence[SourceSpec],
    tokenizer: TokenizerLike,
    output_dir: str | Path,
    *,
    seq_len: int,
    seed: int,
    n_docs: int,
    eval_texts: Sequence[str] = (),
    allowed_licenses: Iterable[str] | None = None,
    allowed_languages: Sequence[str] = ("en",),
    quality_thresholds: QualityThresholds = QualityThresholds(),
    dedup_threshold: float = 0.8,
    dedup_ngram: int = 5,
    decontam_ngram: int = 13,
    enable_near_dedup: bool = True,
    pad_id: int = 0,
    add_bos: bool = False,
    add_eos: bool = True,
    shard_target_bytes: int = 256 * 1024 * 1024,
) -> Manifest:
    """Run the full §7 pipeline and write shards + a manifest to `output_dir`.

    Deterministic given `seed`: the mixture draw, MinHash, and shard layout all derive
    from it, so two runs against the same `sources` and `eval_texts` produce shards
    with identical content hashes. `enable_near_dedup=False` skips the `datasketch`
    stage entirely (useful when that optional dependency isn't installed); every other
    stage always runs.
    """
    if tokenizer.vocab_size > 65536:
        raise ValueError("tokenizer.vocab_size must be <= 65536 to pack into uint16 shards")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_docs: Iterable[RawDoc] = list(iter_mixture(sources, seed=seed, n_docs=n_docs))

    docs: Iterable[RawDoc] = raw_docs
    if allowed_licenses is not None:
        docs = filter_license(docs, allowed_licenses)
    docs = filter_language(docs, allowed_languages)
    docs = filter_quality(docs, quality_thresholds)
    docs = list(docs)

    dedup_stats: DedupStats
    if enable_near_dedup:
        docs, dedup_stats = near_dedup(
            docs, threshold=dedup_threshold, ngram=dedup_ngram, seed=seed
        )
    else:
        dedup_stats = DedupStats(total=len(docs), kept=len(docs), removed=0, ratio=0.0)

    docs = list(scrub_docs(docs))

    decontam_stats: DecontamStats
    docs, decontam_stats = decontaminate(docs, eval_texts, n=decontam_ngram)

    token_lists = [tokenizer.encode(d.text, add_bos=add_bos, add_eos=add_eos) for d in docs]
    token_lists = [t for t in token_lists if t]

    max_rows = rows_per_shard(seq_len, shard_target_bytes)
    shard_records: list[ShardRecord] = []
    buffer: list[PackedRow] = []
    shard_idx = 0
    total_rows = 0
    total_tokens = 0

    def flush() -> None:
        nonlocal buffer, shard_idx, total_rows, total_tokens
        if not buffer:
            return
        meta: ShardMeta = write_shard(
            buffer,
            output_dir / f"shard_{shard_idx:05d}",
            vocab_size=tokenizer.vocab_size,
            pad_id=pad_id,
            seed=seed,
        )
        shard_records.append(
            ShardRecord(
                bin_file=meta.bin_path.name,
                idx_file=meta.idx_path.name,
                n_rows=meta.n_rows,
                seq_len=meta.seq_len,
                content_hash=meta.content_hash,
            )
        )
        total_rows += meta.n_rows
        total_tokens += meta.n_rows * meta.seq_len
        buffer = []
        shard_idx += 1

    for row in pack_documents(token_lists, seq_len, pad_id=pad_id):
        buffer.append(row)
        if len(buffer) >= max_rows:
            flush()
    flush()

    manifest = Manifest(
        seed=seed,
        seq_len=seq_len,
        vocab_size=tokenizer.vocab_size,
        mixture=[{"name": s.name, "weight": s.weight, "license": s.license} for s in sources],
        dedup=asdict(dedup_stats),
        decontam=asdict(decontam_stats),
        shards=shard_records,
        total_rows=total_rows,
        total_tokens=total_tokens,
    )
    manifest.to_json(output_dir / "manifest.json")
    return manifest
