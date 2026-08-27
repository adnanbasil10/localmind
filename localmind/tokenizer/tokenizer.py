"""Byte-level BPE tokenizer: the frozen `Tokenizer` Protocol from CONVENTIONS.md.

    class Tokenizer(Protocol):
        def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...
        def decode(self, ids: list[int]) -> str: ...
        def apply_chat_template(self, messages: list[dict[str, str]],
                                 add_generation_prompt: bool = False) -> str: ...
        @property
        def vocab_size(self) -> int: ...

Id layout (fixed for the lifetime of a trained tokenizer, and baked into every saved
checkpoint): `[0, 256)` are raw UTF-8 bytes, so there is never an UNK token -- any
byte string, valid UTF-8 or not, is representable. `[256, 256 + len(SPECIAL_TOKENS))`
are the reserved special tokens below, in the fixed order `SPECIAL_TOKENS` lists them
in. Everything from `256 + len(SPECIAL_TOKENS)` up to `vocab_size` is a learned BPE
merge, in training order (`self.merges[i]` produced id `first_merge_id + i`).

Anti-injection design (the spec's explicit trap: "a user typing `<|eos|>` must not
inject a control token"): `encode()` *never* recognizes special-token strings inside
`text` -- there is no code path, at any vocab size, by which arbitrary input text can
produce a special-token id. Special ids only ever enter a sequence structurally, via
`add_bos`/`add_eos`, or via `encode_chat`, which splices them in directly rather than
scanning text for their literal spelling. `apply_chat_template` additionally sanitizes
message content so the *string* it returns can't be misread as containing a template
boundary that wasn't actually inserted by the template.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Literal

from localmind.tokenizer.bpe import Pair, merge_word, train_bpe_incremental, train_bpe_naive
from localmind.tokenizer.regex_split import pretokenize

BASE_VOCAB_SIZE = 256

# Bounds the per-pre-token BPE cache each Tokenizer instance keeps (see __init__).
# Natural text repeats the same word/number/punctuation-run constantly, so caching
# `pre_token_bytes -> ids` turns an O(occurrences) cost into O(distinct pre-tokens).
# Bounded (LRU-evicted) so adversarial input with unboundedly many distinct chunks
# can't grow the cache without limit.
DEFAULT_PRETOKEN_CACHE_SIZE = 65536

NAMED_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool_call|>",
    "<|tool_result|>",
)
NUM_RESERVED_SPECIAL_TOKENS = 16
RESERVED_SPECIAL_TOKENS: tuple[str, ...] = tuple(
    f"<|reserved_{i}|>" for i in range(NUM_RESERVED_SPECIAL_TOKENS)
)
# 7 named + 16 reserved = 23 special ids, occupying [256, 279).
SPECIAL_TOKENS: tuple[str, ...] = NAMED_SPECIAL_TOKENS + RESERVED_SPECIAL_TOKENS

# Chat-template role -> the special token that opens that turn. "system" is
# deliberately absent: the spec's special-token list has no <|system|> slot, so
# system content is rendered as an unmarked preamble right after <|bos|> instead
# (see apply_chat_template / encode_chat).
_ROLE_TOKEN: dict[str, str] = {
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool_call": "<|tool_call|>",
    "tool_result": "<|tool_result|>",
}

_TRAIN_ALGORITHMS = {"naive": train_bpe_naive, "incremental": train_bpe_incremental}


class Tokenizer:
    """Byte-level BPE tokenizer with GPT-4/cl100k-style pre-tokenization."""

    def __init__(
        self,
        merges: list[Pair],
        vocab: dict[int, bytes],
        special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
        base_vocab_size: int = BASE_VOCAB_SIZE,
        pretoken_cache_size: int = DEFAULT_PRETOKEN_CACHE_SIZE,
    ) -> None:
        self.merges: list[Pair] = list(merges)  # ranked; index == training order
        self._merge_rank: dict[Pair, int] = {pair: rank for rank, pair in enumerate(self.merges)}

        self.base_vocab_size = base_vocab_size
        self.special_tokens: tuple[str, ...] = tuple(special_tokens)
        self._special_to_id: dict[str, int] = {
            tok: base_vocab_size + i for i, tok in enumerate(self.special_tokens)
        }
        self._id_to_special: dict[int, str] = {v: k for k, v in self._special_to_id.items()}

        self.first_merge_id = base_vocab_size + len(self.special_tokens)
        # merges[rank] always produced id (first_merge_id + rank) -- ids were handed
        # out in training order, so this is a pure re-derivation, not new state.
        self._merge_result_id: dict[Pair, int] = {
            pair: self.first_merge_id + rank for rank, pair in enumerate(self.merges)
        }
        self.id_to_bytes: dict[int, bytes] = dict(vocab)
        self._vocab_size = self.first_merge_id + len(self.merges)

        # Per-instance, bounded LRU cache of pre_token_bytes -> ids. Built here (not
        # as a class-level @lru_cache) so the cache is scoped to this Tokenizer and
        # doesn't keep `self` alive forever via a module-level cache's key tuple.
        # `pretoken_cache_size=0` disables caching outright (goes straight to the
        # uncached path), which `test_cached_and_uncached_paths_agree` in
        # test_tokenizer.py uses to compare cached vs. uncached output.
        self.pretoken_cache_size = pretoken_cache_size
        if pretoken_cache_size > 0:
            self._encode_chunk_cached = lru_cache(maxsize=pretoken_cache_size)(
                self._encode_chunk_bytes_uncached
            )
        else:
            self._encode_chunk_cached = self._encode_chunk_bytes_uncached

        self.bos_id = self._special_to_id["<|bos|>"]
        self.eos_id = self._special_to_id["<|eos|>"]
        self.pad_id = self._special_to_id["<|pad|>"]

    # ---- Protocol surface -------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode `text`. Never recognizes special-token strings inside `text` --
        see the module docstring's anti-injection note."""
        ids = self._encode_ordinary(text)
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        text_parts: list[str] = []
        buf = bytearray()
        for i in ids:
            special = self._id_to_special.get(i)
            if special is not None:
                if buf:
                    text_parts.append(bytes(buf).decode("utf-8", errors="surrogatepass"))
                    buf.clear()
                text_parts.append(special)
                continue
            b = self.id_to_bytes.get(i)
            if b is None:
                raise ValueError(f"unknown token id: {i}")
            buf.extend(b)
        if buf:
            text_parts.append(bytes(buf).decode("utf-8", errors="surrogatepass"))
        return "".join(text_parts)

    def apply_chat_template(
        self, messages: list[dict[str, str]], add_generation_prompt: bool = False
    ) -> str:
        """Render `messages` into the model's chat format, as a literal string.

        This is the single place the wire format is defined: SFT/KD/DPO scripts must
        call this (or `encode_chat`) instead of hand-building prompts with f-strings.
        Every reserved special-token string that appears inside a message's own
        `content` is escaped first (see `_sanitize_for_template`), so the returned
        string can't be misread as containing a template boundary the template didn't
        actually insert.
        """
        parts: list[str] = ["<|bos|>"]
        for msg in messages:
            role = msg["role"]
            content = self._sanitize_for_template(msg.get("content", ""))
            if role == "system":
                parts.append(content)
                continue
            tag = _role_token(role)
            parts.append(f"{tag}\n{content}<|eos|>\n")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)

    # ---- Extra, non-Protocol helpers ---------------------------------------------

    def encode_chat(
        self, messages: list[dict[str, str]], add_generation_prompt: bool = False
    ) -> list[int]:
        """Token ids for `messages`, built structurally rather than by re-parsing a
        rendered string: each message's content goes through the ordinary,
        injection-safe `encode()` path, and role/bos/eos markers are spliced in as
        real special-token ids directly. This is what training code should call to
        get ids; `apply_chat_template` exists to satisfy the Protocol's `-> str`
        signature and for logging/debugging what the model actually sees.
        """
        ids: list[int] = [self.bos_id]
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "system":
                ids.extend(self._encode_ordinary(content))
                continue
            tag = _role_token(role)
            ids.append(self._special_to_id[tag])
            ids.extend(self._encode_ordinary(content))
            ids.append(self.eos_id)
        if add_generation_prompt:
            ids.append(self._special_to_id["<|assistant|>"])
        return ids

    def token_bytes(self, token_id: int) -> bytes:
        """Raw bytes for a non-special token id (base byte or learned merge)."""
        return self.id_to_bytes[token_id]

    def is_special(self, token_id: int) -> bool:
        return token_id in self._id_to_special

    # ---- Training -----------------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        algorithm: Literal["naive", "incremental"] = "incremental",
        special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
        pretoken_cache_size: int = DEFAULT_PRETOKEN_CACHE_SIZE,
    ) -> Tokenizer:
        """Train a byte-level BPE tokenizer over `texts`.

        `texts` should be an iterable of *documents* -- pre-tokenization (and
        therefore merge learning) never spans a document boundary, so concatenating
        unrelated documents into one giant string before calling this would still be
        safe, but passing them separately is both clearer and lets the regex
        pre-tokenizer run per-document.

        `pretoken_cache_size=0` trains a tokenizer with per-pre-token BPE caching
        disabled -- mainly useful for comparing cached vs. uncached encode output in
        tests (`test_cached_and_uncached_paths_agree`), or for memory-constrained
        deployments that would rather not hold the cache.
        """
        first_merge_id = BASE_VOCAB_SIZE + len(special_tokens)
        if vocab_size < first_merge_id:
            raise ValueError(
                f"vocab_size={vocab_size} is too small: need >= {first_merge_id} "
                f"({BASE_VOCAB_SIZE} bytes + {len(special_tokens)} special tokens) "
                "before a single merge can be learned"
            )
        word_freqs = cls._build_word_freqs(texts)
        train_fn = _TRAIN_ALGORITHMS[algorithm]
        result = train_fn(word_freqs, vocab_size=vocab_size, base_vocab_size=first_merge_id)
        return cls(
            result.merges,
            result.vocab,
            special_tokens=special_tokens,
            base_vocab_size=BASE_VOCAB_SIZE,
            pretoken_cache_size=pretoken_cache_size,
        )

    @staticmethod
    def _build_word_freqs(texts: Iterable[str]) -> dict[tuple[int, ...], int]:
        freqs: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            for chunk in pretokenize(text):
                data = chunk.encode("utf-8", errors="surrogatepass")
                freqs[tuple(data)] += 1
        return dict(freqs)

    # ---- Save / load ----------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "base_vocab_size": self.base_vocab_size,
            "special_tokens": list(self.special_tokens),
            "merges": [list(pair) for pair in self.merges],
            "vocab_size": self._vocab_size,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> Tokenizer:
        base_vocab_size = data["base_vocab_size"]
        special_tokens = tuple(data["special_tokens"])
        merges: list[Pair] = [(int(a), int(b)) for a, b in data["merges"]]
        first_merge_id = base_vocab_size + len(special_tokens)
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for rank, (a, b) in enumerate(merges):
            vocab[first_merge_id + rank] = vocab[a] + vocab[b]
        return cls(merges, vocab, special_tokens=special_tokens, base_vocab_size=base_vocab_size)

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    # ---- Internals ------------------------------------------------------------------

    def _encode_chunk_bytes(self, data: bytes) -> list[int]:
        """BPE-encode one pre-token chunk's raw bytes, memoized.

        Ordinary text repeats the same pre-tokens constantly ("the", "a", common
        punctuation runs...), so `encode()` calls this once per *occurrence* but the
        expensive part -- `_encode_chunk_bytes_uncached` -- only actually has to run
        once per *distinct* chunk, via the per-instance LRU cache built in
        `__init__`. This is the standard trick tiktoken itself relies on for
        throughput; see the Fix round 1 section of the task report for the measured
        speedup. `test_cached_and_uncached_paths_agree` in test_tokenizer.py asserts
        this cache never changes what gets encoded, only how fast.
        """
        return list(self._encode_chunk_cached(data))

    def _encode_chunk_bytes_uncached(self, data: bytes) -> tuple[int, ...]:
        """Greedy lowest-rank-first BPE merge over one pre-token chunk's raw bytes.

        Standard algorithm (matches GPT-2's reference encoder / tiktoken): repeatedly
        find the adjacent pair with the lowest merge rank still present anywhere in
        the sequence, and merge *every* non-overlapping occurrence of exactly that
        pair (via `merge_word`) before rescanning.

        Merging all occurrences of the winning pair per pass, rather than one
        occurrence at a time, produces byte-identical output to the naive
        one-at-a-time version: a merge can only ever introduce pairs built from a
        token that didn't exist before that merge, and any such new pair therefore
        has a strictly *later* (higher) rank than the merge that just created it --
        so nothing a merge produces can ever outrank the merge itself while any of
        its own occurrences remain unmerged. The one-at-a-time algorithm would thus
        always reselect the very same pair on every subsequent iteration until no
        occurrence of it is left; doing them all in one `merge_word` call just skips
        the redundant rescans. `test_batch_merge_matches_reference_one_at_a_time`
        checks this equivalence directly against a deliberately naive reference
        implementation, independent of the cache.
        """
        ids: list[int] = list(data)
        merge_rank = self._merge_rank
        while len(ids) >= 2:
            best_pair: Pair | None = None
            best_rank: int | None = None
            for pair in pairwise(ids):
                rank = merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            new_id = self._merge_result_id[best_pair]
            ids = merge_word(ids, best_pair, new_id)
        return tuple(ids)

    def _encode_ordinary(self, text: str) -> list[int]:
        """Encode text with no special-token handling at all: every character is
        routed through regex pre-tokenization + BPE merges. This is what makes it
        safe to call directly on untrusted input -- no substring of `text`, however
        it is spelled, can ever produce a special-token id.
        """
        ids: list[int] = []
        for chunk in pretokenize(text):
            data = chunk.encode("utf-8", errors="surrogatepass")
            ids.extend(self._encode_chunk_bytes(data))
        return ids

    def _sanitize_for_template(self, text: str) -> str:
        """Neutralize literal occurrences of any reserved special-token string
        inside user-supplied message content, so a message body can never forge a
        template boundary in the string `apply_chat_template` returns.
        """
        if not text:
            return text
        out = text
        zero_width_space = chr(0x200B)  # U+200B ZERO WIDTH SPACE
        for tok in self.special_tokens:
            if tok in out:
                # "<|eos|>" -> "<|<ZWSP>eos|>": visually near-identical (a
                # zero-width space inserted after the second character), byte-
                # distinct, and no longer equal to any real special-token string.
                escaped = tok[:2] + zero_width_space + tok[2:]
                out = out.replace(tok, escaped)
        return out


def _role_token(role: str) -> str:
    try:
        return _ROLE_TOKEN[role]
    except KeyError:
        raise ValueError(
            f"unknown chat role: {role!r} (expected one of "
            f"'system', {', '.join(sorted(_ROLE_TOKEN))})"
        ) from None
