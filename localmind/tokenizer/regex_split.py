"""GPT-4 / cl100k-style regex pre-tokenizer.

Splits raw text into pre-token chunks *before* BPE merging is applied. Byte-pair
merges are never learned or applied across a pre-token boundary -- that is what keeps
whitespace, punctuation, and digit runs from bleeding into unrelated merges, and it is
why this file exists as a separate pass instead of just handing raw bytes to the BPE
trainer.

Rules encoded by the pattern, tried in order at every position:

1. English contractions ('s, 'd, 'm, 't, 'll, 've, 're), case-insensitive.
2. A letter run, optionally preceded by exactly one leading character that is not a
   letter, digit, or newline (so `"Hello, world"` splits as `["Hello", ",", " world"]`
   -- the leading space attaches to the following word, not the preceding punctuation).
3. A digit run capped at 3 digits: `"12345"` -> `["123", "45"]`. This is the cl100k
   rule; splitting integers into <=3-digit groups gives the model a small, closed set
   of "hundreds" chunks instead of needing a merge for every multi-digit number it
   happens to see in training, which measurably improves arithmetic.
4. A run of punctuation/symbol characters, with at most one leading space and any
   trailing newlines folded in.
5. Whitespace: a run of newlines; else trailing whitespace not followed by a
   non-whitespace character (so indentation at the end of a chunk stays attached
   rather than being split from the newline that ends it); else any whitespace run.

This uses the third-party `regex` package, not stdlib `re`: the pattern needs Unicode
property classes (`\\p{L}`, `\\p{N}`) and possessive quantifiers (`?+`, `++`) for
linear-time matching, and `re` supports neither.
"""

from __future__ import annotations

import regex

# cl100k_base's split pattern (the GPT-4 tokenizer's pre-tokenizer), verbatim.
GPT4_SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"
    r"|[^\r\n\p{L}\p{N}]?+\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*"
    r"|\s*[\r\n]"
    r"|\s+(?!\S)"
    r"|\s+"
)

_PATTERN = regex.compile(GPT4_SPLIT_PATTERN)


def pretokenize(text: str) -> list[str]:
    """Split `text` into GPT-4/cl100k-style pre-token chunks.

    The pattern's alternatives partition the input: every code point falls into
    exactly one alternative (letters, digits, whitespace, and "everything else" --
    punctuation, symbols, combining marks, control characters, lone surrogates -- are
    each covered by their own branch), so `"".join(pretokenize(s)) == s` for any `s`.
    That invariant is what makes the tokenizer's overall encode/decode round trip
    exact: see the `test_pretokenize_partitions_*` tests in test_tokenizer.py.
    """
    return _PATTERN.findall(text)
