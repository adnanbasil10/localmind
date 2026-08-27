"""Byte-level BPE tokenizer for LocalMind (Phase 1).

Public surface:
    Tokenizer      -- implements the frozen CONVENTIONS.md Tokenizer Protocol
                       (encode / decode / apply_chat_template / vocab_size), plus
                       train / save / load / encode_chat.
    SPECIAL_TOKENS -- the 23 reserved special-token strings, in id order.
    pretokenize    -- GPT-4/cl100k-style regex pre-tokenizer.
    train_bpe_naive, train_bpe_incremental -- the two merge-loop implementations,
                       compared in `bench.py`.

No network calls happen anywhere at import time.
"""

from localmind.tokenizer.bpe import BPETrainResult, train_bpe_incremental, train_bpe_naive
from localmind.tokenizer.regex_split import GPT4_SPLIT_PATTERN, pretokenize
from localmind.tokenizer.tokenizer import (
    BASE_VOCAB_SIZE,
    NAMED_SPECIAL_TOKENS,
    RESERVED_SPECIAL_TOKENS,
    SPECIAL_TOKENS,
    Tokenizer,
)

__all__ = [
    "BASE_VOCAB_SIZE",
    "GPT4_SPLIT_PATTERN",
    "NAMED_SPECIAL_TOKENS",
    "RESERVED_SPECIAL_TOKENS",
    "SPECIAL_TOKENS",
    "BPETrainResult",
    "Tokenizer",
    "pretokenize",
    "train_bpe_incremental",
    "train_bpe_naive",
]
