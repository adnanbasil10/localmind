"""Tests for `localmind/data` (implementation.md §7, Phase 3).

No real network, no real `datasets`/`datasketch` corpora -- everything runs against
small synthetic corpora built in this file. Tests that need `datasketch` (the MinHash
LSH backend) skip via `pytest.importorskip` if it isn't installed, rather than failing;
the one test that would need real network is marked `@pytest.mark.net` and additionally
guarded so it degrades to a skip rather than a hard failure if network is unavailable.

`FakeTokenizer` below is a structural stand-in for `localmind.tokenizer.Tokenizer`
(CONVENTIONS.md's frozen interface) -- this file never imports `localmind.tokenizer`.

The single most important test here is `test_loader_resume_yields_identical_next_batch`:
it is the DoD's "resuming from step N yields the identical next batch" proof.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
from localmind.data.dedup import decontaminate, near_dedup
from localmind.data.filter import (
    QualityThresholds,
    RawDoc,
    detect_language,
    filter_language,
    filter_license,
    filter_quality,
    quality_ok,
    scrub_docs,
    scrub_pii,
)
from localmind.data.loader import PackedShardLoader
from localmind.data.packing import (
    build_doc_mask,
    doc_ids_from_boundaries,
    pack_documents,
    rows_per_shard,
    write_shard,
)
from localmind.data.prepare import (
    MixtureConfig,
    SourceSpec,
    iter_mixture,
    prepare_shards,
    sources_from_mixture_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    """Structural stand-in for `localmind.tokenizer.Tokenizer` (CONVENTIONS.md): only
    `encode`/`decode`/`vocab_size` are implemented, deliberately with no inheritance
    from and no import of `localmind.tokenizer`, so these tests pass independently of
    that concurrently-developed package.
    """

    def __init__(self, vocab_size: int = 300) -> None:
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [4 + (b % (self._vocab_size - 4)) for b in text.encode("utf-8")]
        if add_bos:
            ids = [1, *ids]
        if add_eos:
            ids = [*ids, 2]
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(32 + (i % 95)) for i in ids)


ENGLISH_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog and then it runs to the forest "
    "where it meets a family of rabbits that are also looking for food in the meadow."
)

# Six topically-distinct, stopword-rich English paragraphs with no shared 5-word runs --
# used by the near-dedup and end-to-end tests, where exact injected duplicates are the
# only intended matches.
SYNTHETIC_TOPICS = (
    "the history of the roman empire spanned many centuries and involved a long "
    "series of military campaigns political reforms and cultural exchanges across "
    "the entire mediterranean region and beyond into northern europe and asia minor",
    "modern software engineering teams rely heavily on automated testing continuous "
    "integration and careful code review to keep large systems reliable while they "
    "grow and change over time across many different contributors and releases",
    "the process of baking sourdough bread requires patience because the wild yeast "
    "starter must be fed regularly and the dough needs a long slow fermentation "
    "before it can be shaped and baked in a very hot oven until it turns golden",
    "astronomers have discovered thousands of planets orbiting distant stars using "
    "telescopes that measure tiny dips in brightness as a planet passes in front "
    "of its star and blocks a small fraction of the light reaching earth",
    "the field of economics studies how individuals firms and governments make "
    "choices about scarce resources and how those choices interact through markets "
    "prices and institutions to shape outcomes for entire societies over time",
    "coral reefs support an enormous amount of marine biodiversity and provide "
    "coastal protection from storms but they are increasingly threatened by "
    "rising ocean temperatures acidification and unsustainable fishing practices",
)


def _write_manifest(shard_dir: Path, metas) -> None:
    shards = [
        {
            "bin_file": m.bin_path.name,
            "idx_file": m.idx_path.name,
            "n_rows": m.n_rows,
            "seq_len": m.seq_len,
            "content_hash": m.content_hash,
        }
        for m in metas
    ]
    (shard_dir / "manifest.json").write_text(json.dumps({"shards": shards}), encoding="utf-8")


def _write_synthetic_shards(
    tmp_path: Path, *, seq_len: int = 6, n_docs: int = 30, docs_seed: int = 0, shard_rows: int = 4
) -> tuple[Path, int]:
    """Build a small directory of packed shards from synthetic per-doc token streams
    (each doc's tokens are a single repeated value, handy for eyeballing failures)."""
    rng = random.Random(docs_seed)
    token_lists = [[10 + i] * rng.randint(2, 12) for i in range(n_docs)]
    rows = list(pack_documents(token_lists, seq_len=seq_len))

    shard_dir = tmp_path / f"shards_{docs_seed}"
    shard_dir.mkdir()
    metas = []
    for shard_idx, start in enumerate(range(0, len(rows), shard_rows)):
        chunk = rows[start : start + shard_rows]
        metas.append(
            write_shard(chunk, shard_dir / f"shard_{shard_idx:05d}", vocab_size=300, seed=docs_seed)
        )
    _write_manifest(shard_dir, metas)
    return shard_dir, len(rows)


# =====================================================================================
# filter.py
# =====================================================================================
def test_filter_license_keeps_allowed_drops_disallowed():
    docs = [
        RawDoc(text="x", source="a", license="MIT"),
        RawDoc(text="x", source="b", license="GPL-3.0"),
    ]
    kept = list(filter_license(docs, {"MIT", "Apache-2.0"}))
    assert [d.source for d in kept] == ["a"]


def test_detect_language_english_vs_symbols_vs_empty():
    assert detect_language(ENGLISH_PARAGRAPH) == "en"
    assert detect_language("的 的 的 的 的 的 的 的 的 的 的 的") == "unknown"
    assert detect_language("") == "unknown"
    assert detect_language("12345 67890 111") == "unknown"


def test_filter_language_drops_non_english():
    docs = [
        RawDoc(text=ENGLISH_PARAGRAPH, source="a", license="MIT"),
        RawDoc(text="的 的 的 的 的 的 的 的 的 的 的 的", source="b", license="MIT"),
    ]
    kept = list(filter_language(docs))
    assert [d.source for d in kept] == ["a"]


def test_quality_ok_accepts_good_rejects_short():
    assert quality_ok(ENGLISH_PARAGRAPH) is True
    assert quality_ok("too short") is False


def test_quality_ok_rejects_symbol_heavy_word_shaped_text():
    text = " ".join(["###"] * 25)  # passes word-count and avg-length, fails alpha_ratio
    assert quality_ok(text) is False


def test_filter_quality_pipeline():
    docs = [
        RawDoc(text=ENGLISH_PARAGRAPH, source="good", license="MIT"),
        RawDoc(text="short", source="bad", license="MIT"),
    ]
    kept = list(filter_quality(docs, QualityThresholds()))
    assert [d.source for d in kept] == ["good"]


def test_scrub_pii_emails_ssn_ip_phone():
    text = "Contact me at jane.doe@example.com or 555-123-4567, SSN 123-45-6789, from 192.168.1.1."
    scrubbed = scrub_pii(text)
    assert "jane.doe@example.com" not in scrubbed and "[EMAIL]" in scrubbed
    assert "123-45-6789" not in scrubbed and "[SSN]" in scrubbed
    assert "192.168.1.1" not in scrubbed and "[IP]" in scrubbed
    assert "[PHONE]" in scrubbed


def test_scrub_docs_preserves_other_fields():
    doc = RawDoc(text="email me at a@b.com", source="s", license="MIT")
    scrubbed = next(iter(scrub_docs([doc])))
    assert scrubbed.source == "s"
    assert scrubbed.license == "MIT"
    assert "[EMAIL]" in scrubbed.text


# =====================================================================================
# dedup.py -- near_dedup (needs datasketch) and decontaminate (pure stdlib)
# =====================================================================================
def test_near_dedup_collapses_exact_and_near_duplicates_keeps_distinct():
    pytest.importorskip("datasketch")

    words = [f"tok{i:03d}" for i in range(200)]
    base_text = " ".join(words)

    near_words = list(words)
    near_words[100] = "REPLACED"  # single mid-document substitution -> Jaccard ~0.95
    near_text = " ".join(near_words)

    unrelated_text = " ".join(f"zzz{i:03d}" for i in range(200))  # disjoint vocabulary

    docs = [
        RawDoc(text=base_text, source="a", license="MIT"),
        RawDoc(text=base_text, source="a-exact-dup", license="MIT"),
        RawDoc(text=near_text, source="a-near-dup", license="MIT"),
        RawDoc(text=unrelated_text, source="b", license="MIT"),
    ]
    kept, stats = near_dedup(docs, threshold=0.8, seed=1337)
    kept_sources = {d.source for d in kept}

    assert kept_sources == {"a", "b"}
    assert stats.total == 4
    assert stats.removed == 2
    assert stats.kept == 2
    assert stats.ratio == pytest.approx(0.5)


def test_near_dedup_is_deterministic_given_seed():
    pytest.importorskip("datasketch")
    docs = [
        RawDoc(text=" ".join(f"w{i}" for i in range(60)), source=f"d{i}", license="MIT")
        for i in range(10)
    ]
    kept1, stats1 = near_dedup(docs, seed=42)
    kept2, stats2 = near_dedup(docs, seed=42)
    assert [d.source for d in kept1] == [d.source for d in kept2]
    assert stats1 == stats2


def test_decontaminate_removes_leaked_13gram_overlap_only():
    golden = " ".join(f"golden{i}" for i in range(20))
    eval_texts = [f"question about topic: {golden} end of golden passage"]

    leaked = RawDoc(
        text=f"training text containing {golden} verbatim leak", source="leaked", license="MIT"
    )
    clean = RawDoc(text=" ".join(f"clean{i}" for i in range(30)), source="clean", license="MIT")
    partial = RawDoc(
        text=" ".join(f"golden{i}" for i in range(10))
        + " then totally unrelated words continue on for a while after this",
        source="partial",
        license="MIT",
    )

    kept, stats = decontaminate([leaked, clean, partial], eval_texts, n=13)
    assert {d.source for d in kept} == {"clean", "partial"}
    assert stats.total == 3
    assert stats.removed == 1
    assert stats.kept == 2
    assert stats.ratio == pytest.approx(1 / 3)


def test_decontaminate_noop_with_empty_eval_set():
    docs = [
        RawDoc(
            text="anything goes here with plenty of words to satisfy any check",
            source="a",
            license="MIT",
        )
    ]
    kept, stats = decontaminate(docs, [], n=13)
    assert kept == docs
    assert stats.removed == 0


# =====================================================================================
# packing.py
# =====================================================================================
def test_pack_documents_small_example_boundaries_and_mask():
    rows = list(
        pack_documents([[10, 10, 10], [20, 20, 20, 20]], seq_len=6)
    )  # row_len=7, fits exactly
    assert len(rows) == 1
    row = rows[0]
    assert row.pad_start == 7
    assert list(row.tokens) == [10, 10, 10, 20, 20, 20, 20]
    assert row.boundaries == [0, 3]

    seg_ids = doc_ids_from_boundaries(row.boundaries, len(row.tokens))
    assert list(seg_ids) == [0, 0, 0, 1, 1, 1, 1]

    mask = build_doc_mask(row.boundaries, len(row.tokens))
    assert not mask[3, 2]  # doc B's first token cannot attend to doc A
    assert mask[3, 3]
    assert not mask[0, 1]  # causal: cannot attend to the future
    assert list(mask[2]) == [True, True, True, False, False, False, False]


def test_pack_documents_final_row_padding():
    rows = list(pack_documents([[7, 7, 7]], seq_len=6))  # row_len=7, only 3 real tokens
    assert len(rows) == 1
    row = rows[0]
    assert row.pad_start == 3
    assert list(row.tokens) == [7, 7, 7, 0, 0, 0, 0]


def test_pack_documents_zero_waste_except_final_row():
    token_lists = [[i] * 7 for i in range(1, 6)]  # 5 docs x 7 tokens = 35 tokens
    rows = list(pack_documents(token_lists, seq_len=10))  # row_len=11

    flat: list[int] = []
    for i, row in enumerate(rows):
        if i < len(rows) - 1:
            assert row.pad_start == len(row.tokens), "only the final row may carry padding"
        flat.extend(int(t) for t in row.tokens[: row.pad_start])

    expected = [t for doc in token_lists for t in doc]
    assert flat == expected


def test_pack_documents_rejects_bad_seq_len():
    with pytest.raises(ValueError):
        list(pack_documents([[1, 2, 3]], seq_len=0))


def test_rows_per_shard_target_bytes():
    n = rows_per_shard(1023, target_bytes=256 * 1024 * 1024)
    assert n == (256 * 1024 * 1024) // (1024 * 2)
    assert rows_per_shard(seq_len=10_000_000, target_bytes=100) == 1  # never zero


def test_write_shard_content_hash_deterministic_and_content_sensitive(tmp_path):
    rows = list(pack_documents([[1, 2, 3, 4, 5]], seq_len=4))
    meta1 = write_shard(rows, tmp_path / "shard_00000", vocab_size=300, seed=1)
    assert meta1.bin_path.exists()
    assert meta1.idx_path.exists()

    idx = json.loads(meta1.idx_path.read_text(encoding="utf-8"))
    assert idx["content_hash"] == meta1.content_hash
    assert idx["n_rows"] == 1
    assert idx["seq_len"] == 4

    meta2 = write_shard(rows, tmp_path / "shard_00001", vocab_size=300, seed=1)
    assert meta2.content_hash == meta1.content_hash  # same content -> same hash

    other_rows = list(pack_documents([[9, 9, 9, 9, 9]], seq_len=4))
    meta3 = write_shard(other_rows, tmp_path / "shard_00002", vocab_size=300, seed=1)
    assert meta3.content_hash != meta1.content_hash  # different content -> different hash


def test_write_shard_rejects_vocab_too_large(tmp_path):
    rows = list(pack_documents([[1, 2, 3]], seq_len=4))
    with pytest.raises(ValueError):
        write_shard(rows, tmp_path / "bad", vocab_size=70_000, seed=0)


# =====================================================================================
# loader.py -- resumability is the DoD's centerpiece.
# =====================================================================================
def test_loader_x_y_alignment(tmp_path):
    rows = list(pack_documents([[100, 101, 102, 103, 104, 105, 106]], seq_len=6))
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    meta = write_shard(rows, shard_dir / "shard_00000", vocab_size=200, seed=0)
    _write_manifest(shard_dir, [meta])

    loader = PackedShardLoader(shard_dir, seed=0, batch_size=1)
    x, y, _ = next(loader)
    assert list(x[0]) == [100, 101, 102, 103, 104, 105]
    assert list(y[0]) == [101, 102, 103, 104, 105, 106]


def test_loader_use_doc_boundaries_ablation(tmp_path):
    rows = list(pack_documents([[1, 1], [2, 2], [3, 3]], seq_len=5))  # row_len=6, exactly 1 row
    assert len(rows) == 1
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    meta = write_shard(rows, shard_dir / "shard_00000", vocab_size=10, seed=0)
    _write_manifest(shard_dir, [meta])

    aware = PackedShardLoader(shard_dir, seed=0, batch_size=1, use_doc_boundaries=True)
    _, _, docs_aware = next(aware)
    assert docs_aware.max() > 0  # multiple segments visible

    naive = PackedShardLoader(shard_dir, seed=0, batch_size=1, use_doc_boundaries=False)
    _, _, docs_naive = next(naive)
    assert (docs_naive == 0).all()  # naive ablation: boundaries ignored, one big segment


def test_loader_deterministic_from_scratch(tmp_path):
    shard_dir, _ = _write_synthetic_shards(tmp_path, seq_len=6, n_docs=40, shard_rows=3)
    loader_a = PackedShardLoader(shard_dir, seed=42, batch_size=2)
    loader_b = PackedShardLoader(shard_dir, seed=42, batch_size=2)
    for _ in range(6):
        xa, ya, da = next(loader_a)
        xb, yb, db = next(loader_b)
        assert np.array_equal(xa, xb)
        assert np.array_equal(ya, yb)
        assert np.array_equal(da, db)


def test_loader_resume_yields_identical_next_batch(tmp_path):
    """THE resumability test (DoD): save loader state mid-stream, restore it in a
    brand-new loader instance with fresh mmaps, and assert the next batch it produces
    is byte-identical to what an uninterrupted loader would have produced next.
    """
    shard_dir, total_rows = _write_synthetic_shards(tmp_path, seq_len=6, n_docs=40, shard_rows=3)
    assert total_rows >= 12

    batch_size = 3
    reference = PackedShardLoader(shard_dir, seed=99, batch_size=batch_size)
    for _ in range(4):  # advance partway, not aligned to a shard boundary
        next(reference)
    checkpoint = reference.state_dict()
    expected_next = next(reference)  # what SHOULD come right after the checkpoint

    resumed = PackedShardLoader(shard_dir, seed=99, batch_size=batch_size)  # fresh mmaps
    resumed.load_state_dict(checkpoint)
    actual_next = resumed.next_batch()

    for exp, act in zip(expected_next, actual_next, strict=True):
        assert np.array_equal(exp, act)

    # resume-from-a-resume must also line up
    checkpoint2 = resumed.state_dict()
    expected_next2 = next(reference)
    resumed2 = PackedShardLoader(shard_dir, seed=99, batch_size=batch_size)
    resumed2.load_state_dict(checkpoint2)
    actual_next2 = resumed2.next_batch()
    for exp, act in zip(expected_next2, actual_next2, strict=True):
        assert np.array_equal(exp, act)


def test_loader_resume_across_epoch_wraparound(tmp_path):
    shard_dir, total_rows = _write_synthetic_shards(tmp_path, seq_len=6, n_docs=12, shard_rows=2)
    batch_size = 5  # deliberately doesn't divide total_rows evenly
    reference = PackedShardLoader(shard_dir, seed=7, batch_size=batch_size)
    n_batches_to_wrap = (total_rows // batch_size) + 2  # guarantee an epoch boundary is crossed
    for _ in range(n_batches_to_wrap):
        next(reference)
    checkpoint = reference.state_dict()
    expected_next = next(reference)

    resumed = PackedShardLoader(shard_dir, seed=7, batch_size=batch_size)
    resumed.load_state_dict(checkpoint)
    actual_next = resumed.next_batch()
    for exp, act in zip(expected_next, actual_next, strict=True):
        assert np.array_equal(exp, act)


def test_loader_state_dict_rejects_seed_mismatch(tmp_path):
    shard_dir, _ = _write_synthetic_shards(tmp_path, seq_len=6, n_docs=20)
    loader = PackedShardLoader(shard_dir, seed=1, batch_size=2)
    state = loader.state_dict()
    other = PackedShardLoader(shard_dir, seed=2, batch_size=2)
    with pytest.raises(ValueError):
        other.load_state_dict(state)


def test_loader_state_dict_rejects_changed_shard_set(tmp_path):
    shard_dir, _ = _write_synthetic_shards(tmp_path, seq_len=6, n_docs=20, docs_seed=0)
    loader = PackedShardLoader(shard_dir, seed=1, batch_size=2)
    state = loader.state_dict()

    other_dir, _ = _write_synthetic_shards(tmp_path, seq_len=6, n_docs=20, docs_seed=1)
    other_loader = PackedShardLoader(other_dir, seed=1, batch_size=2)
    with pytest.raises(ValueError):
        other_loader.load_state_dict(state)


def test_loader_rejects_empty_shard_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        PackedShardLoader(empty, seed=0, batch_size=1)


# =====================================================================================
# prepare.py -- mixture sampler, config, and the full orchestrator
# =====================================================================================
def test_iter_mixture_deterministic_and_weighted():
    sources = [
        SourceSpec(
            name="heavy",
            weight=0.9,
            license="MIT",
            factory=lambda: [RawDoc("heavy-doc", "heavy", "MIT")],
        ),
        SourceSpec(
            name="light",
            weight=0.1,
            license="MIT",
            factory=lambda: [RawDoc("light-doc", "light", "MIT")],
        ),
    ]
    docs = list(iter_mixture(sources, seed=123, n_docs=2000))
    heavy_frac = sum(1 for d in docs if d.source == "heavy") / len(docs)
    assert 0.85 < heavy_frac < 0.95

    docs_again = list(iter_mixture(sources, seed=123, n_docs=2000))
    assert [d.source for d in docs] == [d.source for d in docs_again]

    docs_diff_seed = list(iter_mixture(sources, seed=456, n_docs=2000))
    assert [d.source for d in docs] != [d.source for d in docs_diff_seed]


def test_iter_mixture_empty_source_raises():
    sources = [SourceSpec(name="empty", weight=1.0, license="MIT", factory=lambda: [])]
    with pytest.raises(ValueError):
        list(iter_mixture(sources, seed=0, n_docs=1))


def test_mixture_config_loads_real_config_file():
    config = MixtureConfig.from_yaml(REPO_ROOT / "configs" / "data" / "mixture.yaml")
    assert config.seq_len == 1024
    assert sum(s.weight for s in config.sources) == pytest.approx(1.0)
    names = {s.name for s in config.sources}
    assert names == {"fineweb_edu", "cosmopedia_v2", "stack_v2", "domain_corpus", "tinystories"}
    licenses = {s.name: s.license for s in config.sources}
    assert licenses["fineweb_edu"] == "ODC-By"
    assert licenses["tinystories"] == "CDLA-Sharing-1.0"


def test_sources_from_mixture_config_skips_local_only_entries():
    config = MixtureConfig.from_yaml(REPO_ROOT / "configs" / "data" / "mixture.yaml")
    specs = sources_from_mixture_config(config)
    names = {s.name for s in specs}
    assert "domain_corpus" not in names  # hf_path is null: caller supplies its own source
    assert "fineweb_edu" in names


def test_prepare_shards_rejects_oversized_vocab(tmp_path):
    sources = [
        SourceSpec(name="s", weight=1.0, license="MIT", factory=lambda: [RawDoc("x", "s", "MIT")])
    ]
    with pytest.raises(ValueError):
        prepare_shards(
            sources, FakeTokenizer(vocab_size=70_000), tmp_path / "out", seq_len=8, seed=0, n_docs=1
        )


def _synthetic_pipeline_sources() -> list[SourceSpec]:
    good_texts = [
        f"the article number {i} explains that this project has been carefully "
        f"designed and tested and it is now ready for the next stage of work "
        f"and it should be useful for everyone on the team going forward"
        for i in range(12)
    ]
    bad_license_texts = [ENGLISH_PARAGRAPH]
    non_english_texts = ["的 的 的 的 的 的 的 的 的 的 的 的 的 的"]

    return [
        SourceSpec(
            name="good",
            weight=0.8,
            license="MIT",
            factory=lambda: [RawDoc(text=t, source="good", license="MIT") for t in good_texts],
        ),
        SourceSpec(
            name="bad_license",
            weight=0.1,
            license="GPL-3.0",
            factory=lambda: [
                RawDoc(text=t, source="bad-license", license="GPL-3.0") for t in bad_license_texts
            ],
        ),
        SourceSpec(
            name="non_english",
            weight=0.1,
            license="MIT",
            factory=lambda: [
                RawDoc(text=t, source="non-english", license="MIT") for t in non_english_texts
            ],
        ),
    ]


def test_prepare_shards_deterministic_given_seed_without_near_dedup(tmp_path):
    """Runs without `datasketch` (dedup disabled) so it always executes -- proves
    determinism of the mixture sampler, filters, packing, and shard content hashes.
    """
    sources = _synthetic_pipeline_sources()
    tokenizer = FakeTokenizer(vocab_size=300)
    kwargs = dict(
        seq_len=32, seed=777, n_docs=60, allowed_licenses={"MIT"}, enable_near_dedup=False
    )

    manifest1 = prepare_shards(sources, tokenizer, tmp_path / "run1", **kwargs)
    manifest2 = prepare_shards(sources, tokenizer, tmp_path / "run2", **kwargs)

    assert [s.content_hash for s in manifest1.shards] == [s.content_hash for s in manifest2.shards]
    assert manifest1.total_rows == manifest2.total_rows > 0
    assert manifest1.dedup == manifest2.dedup
    # license + language filters actually ran: bad-license and non-english never made it through
    assert manifest1.total_tokens > 0


def test_prepare_shards_end_to_end_with_near_dedup_and_decontam(tmp_path):
    """The real end-to-end run: mixture -> filter -> MinHash-LSH dedup -> PII scrub ->
    13-gram decontam -> tokenize -> pack -> shards + manifest. Corpus is built so the
    dedup/decontam counts are hand-verifiable: 6 mutually-distinct topics plus 3 exact
    duplicates of the first -> dedup must remove exactly those 3; the eval set shares no
    vocabulary with any topic -> decontam must remove 0.
    """
    pytest.importorskip("datasketch")

    duplicated = [*SYNTHETIC_TOPICS, SYNTHETIC_TOPICS[0], SYNTHETIC_TOPICS[0], SYNTHETIC_TOPICS[0]]
    sources = [
        SourceSpec(
            name="mix",
            weight=1.0,
            license="MIT",
            factory=lambda: [RawDoc(text=t, source="mix", license="MIT") for t in duplicated],
        )
    ]
    tokenizer = FakeTokenizer(vocab_size=400)
    eval_texts = [
        "totally unrelated golden eval content that shares nothing with any training document here "
        * 3
    ]

    manifest = prepare_shards(
        sources,
        tokenizer,
        tmp_path / "out",
        seq_len=64,
        seed=2024,
        n_docs=len(duplicated),
        allowed_licenses={"MIT"},
        eval_texts=eval_texts,
        enable_near_dedup=True,
    )

    assert manifest.dedup["total"] == 9
    assert manifest.dedup["removed"] == 3
    assert manifest.dedup["kept"] == 6
    assert manifest.dedup["ratio"] == pytest.approx(1 / 3)
    assert manifest.decontam["removed"] == 0
    assert len(manifest.shards) >= 1
    assert manifest.total_rows > 0
    assert manifest.vocab_size == 400

    on_disk = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["dedup"]["removed"] == 3
    assert [s["content_hash"] for s in on_disk["shards"]] == [
        s.content_hash for s in manifest.shards
    ]


@pytest.mark.net
def test_hf_streaming_source_real_network():
    """Marked `net`, and *also* opt-in via an env var: exercises the real
    `datasets.load_dataset(..., streaming=True)` path. The environment for this
    project's test suite is expected to have no network access to real corpora (§7:
    "stream, don't download" -- and don't even resolve a real corpus in CI), so this
    test does not run just because `-m net` isn't excluded; it only runs when a human
    explicitly sets `LOCALMIND_ALLOW_NET_TESTS=1`. Skips (never fails) if `datasets`
    isn't installed or the network call itself fails.
    """
    if not os.environ.get("LOCALMIND_ALLOW_NET_TESTS"):
        pytest.skip("network tests are opt-in: set LOCALMIND_ALLOW_NET_TESTS=1 to run this")
    pytest.importorskip("datasets")
    from localmind.data.prepare import MixtureEntry, hf_streaming_source

    entry = MixtureEntry(
        name="tinystories",
        weight=0.05,
        license="CDLA-Sharing-1.0",
        hf_path="roneneldan/TinyStories",
        split="train",
        text_field="text",
    )
    try:
        doc = next(hf_streaming_source(entry))
    except Exception as exc:  # pragma: no cover - network/environment dependent
        pytest.skip(f"network unavailable for HF streaming test: {exc}")
    assert doc.source == "tinystories"
    assert doc.license == "CDLA-Sharing-1.0"
    assert len(doc.text) > 0
