"""Byte-level BPE trainer: two merge-loop implementations, benchmarked in `bench.py`.

`train_bpe_naive` is the textbook description -- "count adjacent pairs, merge the
most frequent, repeat" -- taken literally: it recomputes every adjacent-pair count
from scratch on every merge step, so it costs O(num_merges * corpus_size).

`train_bpe_incremental` computes the identical sequence of merges but keeps a
running pair-count index (`pair_counts`) plus a reverse index of which words contain
each pair (`pair_to_words`), so a merge step only has to touch the (usually small)
subset of words that actually contained the pair just merged, instead of rescanning
the whole corpus. Both are exercised in `bench.py`, which asserts they agree and
reports the wall-clock speedup.

Trap this module is deliberately built to avoid: BPE merges are *order-dependent*.
`BPETrainResult.merges` is a ranked `list[tuple[int, int]]`, never a set -- rank i
must always be applied before rank i+1 when encoding, because a later merge's pair
may only come into existence *after* an earlier merge has created one of its members.
`tokenizer.py` relies on `merges` being exactly this ranked list, not a set, for that
reason.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

Pair = tuple[int, int]


@dataclass
class BPETrainResult:
    """Output of a BPE training run."""

    merges: list[Pair]  # ranked: merges[i] must be applied strictly before merges[i+1]
    vocab: dict[int, bytes]  # token id -> its byte string (raw bytes 0-255 + every merge)
    num_merges: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_merges = len(self.merges)


def word_pair_counts(word: Sequence[int]) -> Counter[Pair]:
    """Count adjacent-pair occurrences within a single word (a token-id sequence)."""
    return Counter(pairwise(word))


def merge_word(word: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping, left-to-right occurrence of `pair` in `word`.

    Non-overlapping means `merge_word([a, a, a], (a, a), X) == [X, a]`: the first two
    elements merge, and the scan resumes *after* the merged pair, so the third `a`
    never gets a second chance to pair with the token the merge just produced. This
    is the standard BPE convention (matches GPT-2's reference encoder and tiktoken).
    """
    if len(word) < 2:
        return list(word)
    out: list[int] = []
    i = 0
    n = len(word)
    a, b = pair
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def _best_pair(counts: Counter[Pair] | dict[Pair, int]) -> Pair | None:
    """Deterministic argmax over pair counts: highest count wins; ties are broken by
    the numerically smallest pair.

    Without an explicit, fixed tie-break rule, "the most frequent pair" is ambiguous
    whenever two pairs tie, and the resulting merge order -- and therefore the entire
    learned vocabulary -- would depend on `dict`/`Counter` iteration order, which is
    insertion order and not guaranteed to be reproducible across different corpus
    orderings. Fixing the rule here makes training bit-for-bit reproducible for a
    given corpus and vocab size, and makes it possible to assert `train_bpe_naive`
    and `train_bpe_incremental` produce identical merge lists.
    """
    if not counts:
        return None
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _init_words(word_freqs: dict[tuple[int, ...], int]) -> tuple[list[list[int]], list[int]]:
    words = [list(w) for w in word_freqs]
    freqs = list(word_freqs.values())
    return words, freqs


def train_bpe_naive(
    word_freqs: dict[tuple[int, ...], int],
    vocab_size: int,
    base_vocab_size: int = 256,
) -> BPETrainResult:
    """Reference merge loop: recompute all pair counts from scratch every step.

    `word_freqs` maps a unique pre-tokenized "word" (as a tuple of byte values, or of
    previously-assigned token ids on later calls) to how many times it occurred in
    the corpus. `base_vocab_size` is the first id a new merge token may take -- the
    caller (`Tokenizer.train`) sets it past the reserved special-token id range so
    merge ids never collide with them.
    """
    words, freqs = _init_words(word_freqs)
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[Pair] = []
    next_id = base_vocab_size

    while next_id < vocab_size:
        counts: Counter[Pair] = Counter()
        for word, freq in zip(words, freqs, strict=True):
            for pair, c in word_pair_counts(word).items():
                counts[pair] += c * freq
        best = _best_pair(counts)
        if best is None:
            break  # corpus has no adjacent pairs left to merge
        merges.append(best)
        vocab[next_id] = vocab[best[0]] + vocab[best[1]]
        words = [merge_word(w, best, next_id) for w in words]
        next_id += 1

    return BPETrainResult(merges=merges, vocab=vocab)


def train_bpe_incremental(
    word_freqs: dict[tuple[int, ...], int],
    vocab_size: int,
    base_vocab_size: int = 256,
) -> BPETrainResult:
    """Same contract and result as `train_bpe_naive`, computed incrementally.

    Invariant maintained after every step: `pair_counts[p]` equals the corpus-wide
    weighted count of pair `p`, and `pair_to_words[p]` equals the exact set of word
    indices whose current representation contains `p`. Both are updated by fully
    removing, then fully re-adding, the contribution of each *affected* word (a word
    that contained the pair just merged) -- unaffected words are never touched, which
    is the entire source of the speedup over `train_bpe_naive`.
    """
    words, freqs = _init_words(word_freqs)
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[Pair] = []
    next_id = base_vocab_size

    pair_counts: Counter[Pair] = Counter()
    pair_to_words: dict[Pair, set[int]] = defaultdict(set)
    for i, (word, freq) in enumerate(zip(words, freqs, strict=True)):
        for pair, c in word_pair_counts(word).items():
            pair_counts[pair] += c * freq
            pair_to_words[pair].add(i)

    while next_id < vocab_size:
        best = _best_pair(pair_counts)
        if best is None:
            break
        merges.append(best)
        vocab[next_id] = vocab[best[0]] + vocab[best[1]]

        affected = list(pair_to_words.get(best, ()))
        for i in affected:
            word, freq = words[i], freqs[i]

            for pair, c in word_pair_counts(word).items():
                pair_counts[pair] -= c * freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                bucket = pair_to_words.get(pair)
                if bucket is not None:
                    bucket.discard(i)
                    if not bucket:
                        del pair_to_words[pair]

            new_word = merge_word(word, best, next_id)
            words[i] = new_word

            for pair, c in word_pair_counts(new_word).items():
                pair_counts[pair] += c * freq
                pair_to_words[pair].add(i)

        # Defensive no-ops: the loop above already zeroes out `best` exactly.
        pair_counts.pop(best, None)
        pair_to_words.pop(best, None)
        next_id += 1

    return BPETrainResult(merges=merges, vocab=vocab)
