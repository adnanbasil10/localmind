"""Stage 4 and 6 of the §7 pipeline: MinHash-LSH near-dedup, and 13-gram eval-set
decontamination.

Per the spec, dedup beats filtering for val-loss improvement per unit of effort, so
this is the most heavily tested module in `localmind/data`. Decontamination is "not
optional if you want your numbers believed" -- it removes any training document that
shares a 13-gram with the golden eval set.

`datasketch` (the MinHash-LSH library, in the `data` extra) is imported lazily inside
`near_dedup` so this module -- and the package -- imports cleanly without it. Tests
that exercise `near_dedup` use `pytest.importorskip("datasketch")`.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from localmind.data.filter import RawDoc

__all__ = [
    "DecontamStats",
    "DedupStats",
    "decontaminate",
    "near_dedup",
]

_WORD_RE = re.compile(r"\w+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _word_ngrams(words: Sequence[str], n: int) -> Iterator[str]:
    """Yield whitespace-joined word n-grams; yields nothing if `words` is shorter than `n`."""
    if len(words) < n:
        return
    for i in range(len(words) - n + 1):
        yield " ".join(words[i : i + n])


@dataclass(frozen=True, slots=True)
class DedupStats:
    total: int
    kept: int
    removed: int
    ratio: float  # removed / total -- "the dedup ratio"


@dataclass(frozen=True, slots=True)
class DecontamStats:
    total: int
    kept: int
    removed: int
    ratio: float


def near_dedup(
    docs: Sequence[RawDoc],
    *,
    threshold: float = 0.8,
    num_perm: int = 128,
    ngram: int = 5,
    seed: int = 0,
) -> tuple[list[RawDoc], DedupStats]:
    """MinHash-LSH near-dedup over word 5-grams, Jaccard threshold 0.8 (spec values).

    Single streaming pass: query the LSH index first (has this doc got a near-duplicate
    already kept?), and only insert + keep if not. Deterministic given `seed`, which
    fixes datasketch's MinHash permutation hash functions -- required for `prepare.py`
    to be reproducible.
    """
    from datasketch import MinHash, MinHashLSH  # lazy: optional `data` extra

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[RawDoc] = []
    removed = 0

    for i, doc in enumerate(docs):
        mh = MinHash(num_perm=num_perm, seed=seed)
        for shingle in _word_ngrams(_words(doc.text), ngram):
            mh.update(shingle.encode("utf-8"))
        if lsh.query(mh):
            removed += 1
            continue
        lsh.insert(f"doc-{i}", mh)
        kept.append(doc)

    total = len(docs)
    ratio = removed / total if total else 0.0
    return kept, DedupStats(total=total, kept=len(kept), removed=removed, ratio=ratio)


def decontaminate(
    docs: Sequence[RawDoc],
    eval_texts: Sequence[str],
    *,
    n: int = 13,
) -> tuple[list[RawDoc], DecontamStats]:
    """Drop any training document that shares a 13-gram (word n-gram, spec default)
    with the golden eval set. Pure stdlib -- no network, no optional deps -- so this
    always runs, unlike `near_dedup`.
    """
    eval_ngrams: set[str] = set()
    for text in eval_texts:
        eval_ngrams.update(_word_ngrams(_words(text), n))

    kept: list[RawDoc] = []
    removed = 0
    for doc in docs:
        doc_ngrams = set(_word_ngrams(_words(doc.text), n))
        if eval_ngrams and (doc_ngrams & eval_ngrams):
            removed += 1
            continue
        kept.append(doc)

    total = len(docs)
    ratio = removed / total if total else 0.0
    return kept, DecontamStats(total=total, kept=len(kept), removed=removed, ratio=ratio)
