"""Constrained decoding: a JSON grammar FSM that masks logits (section 10 step 8).

The problem
-----------
Tool calling only works if the model emits parseable JSON. Prompting for it, retrying on
failure, and repairing broken output are all probabilistic patches on a problem that has
a deterministic fix: if a token would make the output unparseable, remove it from the
distribution before sampling. Then invalid JSON is not unlikely -- it is **unreachable**,
and the measured invalid rate is exactly 0 by construction, not by luck.

How it works
------------
:class:`JSONState` is a pushdown automaton over *characters* (a stack for ``{``/``[``
nesting plus a mode for where we are inside a string, number, or literal). A *token* is
accepted only if every one of its characters is a legal transition, so the automaton is
run over the token's text and the token is banned if any character fails.

Computing that for every token in the vocabulary at every step would dominate decoding, so
transitions are memoised per automaton state: the first time a state is seen the full
vocabulary mask is built and cached, and every later visit is a dictionary lookup. JSON
generation revisits a small number of states, so the cache converges quickly -- the
reported tokens/s cost is the steady-state one, with the cold-start cost reported
separately because pretending it is free would be dishonest.

The EOS token is allowed only when the automaton is in an accepting configuration (empty
stack, value complete), which is what stops truncated-but-valid-prefix output.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from localmind.inference.kv_cache import ContiguousKVCache
from localmind.inference.sampling import SamplingParams, make_generator, sample_token
from localmind.model import LocalMindTransformer

__all__ = [
    "TOOL_CALL_SCHEMA_HINT",
    "ConstrainedDecoder",
    "JSONState",
    "generate_json",
    "is_valid_json",
    "make_synthetic_vocab",
    "step",
]

WS = " \t\n\r"
DIGITS = "0123456789"
HEX = DIGITS + "abcdefABCDEF"
ESCAPES = '"\\/bfnrt'

#: Modes in which a number is mid-parse but already a complete number.
_NUM_TERMINAL = frozenset({"num_zero", "num_int", "num_frac", "num_exp"})
#: Modes in which the automaton is between tokens and may accept whitespace.
_WS_OK = frozenset(
    {"value", "value_or_end", "key_or_end", "key", "colon", "after", "done", "root_object"}
)


@dataclass(frozen=True, slots=True)
class JSONState:
    """Pushdown state: container stack, parse mode, and any pending literal characters."""

    stack: tuple[str, ...] = ()
    mode: str = "value"
    pending: str = ""

    @property
    def complete(self) -> bool:
        """True when the text so far is a whole JSON document."""
        return not self.stack and (self.mode in ("after", "done") or self.mode in _NUM_TERMINAL)


def _pop(s: JSONState) -> JSONState | None:
    if not s.stack:
        return None
    return JSONState(s.stack[:-1], "after")


def step(s: JSONState, ch: str) -> JSONState | None:
    """One character transition. ``None`` means the character is illegal here."""
    m = s.mode
    if m in _WS_OK and ch in WS:
        return s

    if m == "root_object":
        if ch == "{":
            return JSONState((*s.stack, "o"), "key_or_end")
        return None

    if m in ("value", "value_or_end"):
        if ch == "{":
            return JSONState((*s.stack, "o"), "key_or_end")
        if ch == "[":
            return JSONState((*s.stack, "a"), "value_or_end")
        if ch == '"':
            return JSONState(s.stack, "string")
        if ch == "-":
            return JSONState(s.stack, "num_int_start")
        if ch == "0":
            return JSONState(s.stack, "num_zero")
        if ch in "123456789":
            return JSONState(s.stack, "num_int")
        if ch == "t":
            return JSONState(s.stack, "literal", "rue")
        if ch == "f":
            return JSONState(s.stack, "literal", "alse")
        if ch == "n":
            return JSONState(s.stack, "literal", "ull")
        if m == "value_or_end" and ch == "]":
            return _pop(s)
        return None

    if m == "key_or_end":
        if ch == '"':
            return JSONState(s.stack, "key_string")
        if ch == "}":
            return _pop(s)
        return None

    if m == "key":
        return JSONState(s.stack, "key_string") if ch == '"' else None

    if m == "colon":
        return JSONState(s.stack, "value") if ch == ":" else None

    if m in ("string", "key_string"):
        if ch == '"':
            return JSONState(s.stack, "after" if m == "string" else "colon")
        if ch == "\\":
            return JSONState(s.stack, m + "_esc")
        if ord(ch) < 0x20:
            return None
        return s

    if m in ("string_esc", "key_string_esc"):
        base = m[: -len("_esc")]
        if ch in ESCAPES:
            return JSONState(s.stack, base)
        if ch == "u":
            return JSONState(s.stack, base + "_u", "hhhh")
        return None

    if m in ("string_u", "key_string_u"):
        if ch in HEX:
            rest = s.pending[1:]
            base = m[: -len("_u")]
            return JSONState(s.stack, base if not rest else m, rest)
        return None

    if m == "literal":
        if s.pending and ch == s.pending[0]:
            rest = s.pending[1:]
            return JSONState(s.stack, "after" if not rest else "literal", rest)
        return None

    if m.startswith("num_"):
        nxt = _step_number(s, ch)
        if nxt is not None:
            return nxt
        if m in _NUM_TERMINAL:
            # The number ends here; re-dispatch this character as a post-value one.
            return step(JSONState(s.stack, "after"), ch)
        return None

    if m == "after":
        top = s.stack[-1] if s.stack else None
        if top == "o":
            if ch == ",":
                return JSONState(s.stack, "key")
            if ch == "}":
                return _pop(s)
        elif top == "a":
            if ch == ",":
                return JSONState(s.stack, "value")
            if ch == "]":
                return _pop(s)
        return None

    return None


def _step_number(s: JSONState, ch: str) -> JSONState | None:
    m = s.mode
    if m == "num_int_start":
        if ch == "0":
            return JSONState(s.stack, "num_zero")
        if ch in "123456789":
            return JSONState(s.stack, "num_int")
        return None
    if m == "num_zero":
        if ch == ".":
            return JSONState(s.stack, "num_frac_start")
        if ch in "eE":
            return JSONState(s.stack, "num_exp_start")
        return None
    if m == "num_int":
        if ch in DIGITS:
            return s
        if ch == ".":
            return JSONState(s.stack, "num_frac_start")
        if ch in "eE":
            return JSONState(s.stack, "num_exp_start")
        return None
    if m == "num_frac_start":
        return JSONState(s.stack, "num_frac") if ch in DIGITS else None
    if m == "num_frac":
        if ch in DIGITS:
            return s
        if ch in "eE":
            return JSONState(s.stack, "num_exp_start")
        return None
    if m == "num_exp_start":
        if ch in "+-":
            return JSONState(s.stack, "num_exp_sign")
        return JSONState(s.stack, "num_exp") if ch in DIGITS else None
    if m == "num_exp_sign":
        return JSONState(s.stack, "num_exp") if ch in DIGITS else None
    if m == "num_exp":
        return s if ch in DIGITS else None
    return None


_CLOSERS = {"o": "}", "a": "]"}


def closing_suffix(s: JSONState) -> str:
    """The shortest string that completes the document from ``s``.

    Masking guarantees no *invalid transition* is ever taken, but it cannot conjure token
    budget: a generation cut off at ``max_new_tokens`` leaves a valid JSON **prefix**,
    which is not valid JSON. Claiming a 0% invalid rate while quietly emitting truncated
    output would be a lie by omission.

    So truncation is closed deterministically instead: finish whatever token is half
    written (a string, an escape, a bare ``tru``), supply a placeholder value if a key or
    colon is dangling, then pop the container stack. The result is always parseable, and
    the caller is told it happened via ``forced_close`` so the rate is reportable.
    """
    m = s.mode
    if m == "root_object":
        return "{}"
    prefix = ""
    if m == "value":
        prefix = "null"
    elif m == "key":
        prefix = '"":null'
    elif m == "key_string":
        prefix = '":null'
    elif m == "key_string_esc":
        prefix = 'n":null'
    elif m == "key_string_u":
        prefix = "0" * len(s.pending) + '":null'
    elif m == "colon":
        prefix = ":null"
    elif m == "string":
        prefix = '"'
    elif m == "string_esc":
        prefix = 'n"'
    elif m == "string_u":
        prefix = "0" * len(s.pending) + '"'
    elif m == "literal":
        prefix = s.pending
    elif m in ("num_int_start", "num_frac_start", "num_exp_start", "num_exp_sign"):
        prefix = "0"
    return prefix + "".join(_CLOSERS[c] for c in reversed(s.stack))


def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
    except (ValueError, TypeError):
        return False
    return True


TOOL_CALL_SCHEMA_HINT = '{"name": <string>, "arguments": {<string>: <value>}}'
"""What the agent's tool-call surface looks like. `root="object"` is enough to guarantee
the envelope; key-level schema enforcement is a strict refinement of the same machinery."""


def make_synthetic_vocab(vocab_size: int, seed: int = 0) -> list[str]:
    """A stand-in vocabulary of token strings.

    No trained LocalMind tokenizer artifact exists in this repo yet, and constrained
    decoding needs token *text* to run the automaton. This builds a deterministic
    vocabulary whose first 128 entries are the ASCII characters (so every JSON structural
    character is reachable as a single token) and whose remainder are random 2-4 character
    pieces drawn from JSON-ish text. Swap in ``Tokenizer.token_bytes`` once a trained
    tokenizer exists -- the decoder takes the vocabulary as an argument precisely so that
    substitution is a one-liner.
    """
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ .,:-"
    vocab = [chr(i) for i in range(128)]
    while len(vocab) < vocab_size:
        n = rng.randint(2, 4)
        vocab.append("".join(rng.choice(alphabet) for _ in range(n)))
    return vocab[:vocab_size]


class ConstrainedDecoder:
    """Memoised token-level masks derived from the character-level JSON automaton."""

    def __init__(
        self,
        vocab: Sequence[str],
        eos_token_id: int | None = None,
        root: str = "value",
        device: torch.device | str = "cpu",
    ) -> None:
        if root not in ("value", "object"):
            raise ValueError(f"root must be 'value' or 'object', got {root!r}")
        self.vocab = list(vocab)
        self.eos_token_id = eos_token_id
        self.root = root
        self.device = torch.device(device)
        self._trans: dict[JSONState, list[JSONState | None]] = {}
        self._masks: dict[JSONState, Tensor] = {}
        self.mask_builds = 0
        self.mask_lookups = 0
        self.mask_build_s = 0.0

    def initial_state(self) -> JSONState:
        return JSONState((), "root_object" if self.root == "object" else "value")

    def _row(self, state: JSONState) -> list[JSONState | None]:
        row = self._trans.get(state)
        if row is not None:
            return row
        t0 = time.perf_counter()
        out: list[JSONState | None] = []
        for text in self.vocab:
            s: JSONState | None = state
            for ch in text:
                s = step(s, ch)
                if s is None:
                    break
            out.append(s)
        self._trans[state] = out
        self.mask_builds += 1
        self.mask_build_s += time.perf_counter() - t0
        return out

    def mask(self, state: JSONState) -> Tensor:
        """Additive ``(vocab,)`` mask: ``0.0`` for legal tokens, ``-inf`` for the rest."""
        self.mask_lookups += 1
        cached = self._masks.get(state)
        if cached is not None:
            return cached
        row = self._row(state)
        m = torch.full((len(self.vocab),), float("-inf"), device=self.device)
        for i, nxt in enumerate(row):
            if nxt is not None:
                m[i] = 0.0
        if self.eos_token_id is not None and 0 <= self.eos_token_id < len(self.vocab):
            # EOS is legal exactly when the document is complete, never otherwise.
            m[self.eos_token_id] = 0.0 if state.complete else float("-inf")
        dead_end = bool(torch.isinf(m).all())
        if dead_end and self.eos_token_id is not None and 0 <= self.eos_token_id < len(self.vocab):
            # Dead end: allow EOS so the loop terminates instead of sampling from all -inf.
            m[self.eos_token_id] = 0.0
        self._masks[state] = m
        return m

    def advance(self, state: JSONState, token_id: int) -> JSONState | None:
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            return state
        return self._row(state)[token_id]

    def stats(self) -> dict[str, Any]:
        return {
            "distinct_states": len(self._trans),
            "mask_builds": self.mask_builds,
            "mask_lookups": self.mask_lookups,
            "mask_build_s": self.mask_build_s,
            "cache_hit_rate": (
                1.0 - self.mask_builds / self.mask_lookups if self.mask_lookups else 0.0
            ),
        }


def generate_json(
    model: LocalMindTransformer,
    prompt_ids: Sequence[int],
    params: SamplingParams,
    decoder: ConstrainedDecoder | None = None,
    vocab: Sequence[str] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Decode with (or, if ``decoder is None``, without) the grammar mask.

    Returns the token ids, the detokenised text, whether it parses, and the timing --
    which is the whole before/after table implementation.md section 10 step 8 asks for.
    """
    dev = torch.device(device)
    cfg = model.cfg
    model.eval()
    cache = ContiguousKVCache(cfg, dtype=dtype, device=dev)
    gen = make_generator(params.seed, dev)
    ids = list(prompt_ids)
    out_ids: list[int] = []
    state = decoder.initial_state() if decoder is not None else None
    t0 = time.perf_counter()

    with torch.no_grad():
        piece: list[int] = ids
        for _ in range(params.max_new_tokens):
            t = torch.tensor([piece], dtype=torch.long, device=dev)
            out = model(t, past_kvs=cache.as_past(), use_cache=True)
            assert out.kv_caches is not None
            cache.extend_from(out.kv_caches)
            extra = None
            if decoder is not None and state is not None:
                extra = decoder.mask(state)
            token = sample_token(out.logits[0, -1], params, gen, ids + out_ids, extra)
            if decoder is not None and state is not None:
                nxt = decoder.advance(state, token)
                if nxt is None:
                    break
                state = nxt
                if (
                    decoder.eos_token_id is not None
                    and token == decoder.eos_token_id
                    and state.complete
                ):
                    out_ids.append(token)
                    break
            out_ids.append(token)
            piece = [token]
            if cache.length >= cfg.max_seq_len - 1:
                break

    elapsed = time.perf_counter() - t0
    pieces = decoder.vocab if decoder is not None else vocab
    eos = decoder.eos_token_id if decoder is not None else None
    text = _detokenise(out_ids, pieces, eos)
    forced = ""
    if decoder is not None and state is not None and not state.complete:
        forced = closing_suffix(state)
        text += forced
    return {
        "token_ids": out_ids,
        "text": text,
        "valid_json": is_valid_json(text),
        "n_generated": len(out_ids),
        "elapsed_s": elapsed,
        "tokens_per_s": len(out_ids) / elapsed if elapsed > 0 else float("nan"),
        "constrained": decoder is not None,
        "final_state_complete": bool(state.complete) if state is not None else None,
        "forced_close": bool(forced),
        "forced_close_suffix": forced,
    }


def _detokenise(ids: Sequence[int], vocab: Sequence[str] | None, eos_token_id: int | None) -> str:
    if vocab is None:
        return ""
    return "".join(vocab[i] for i in ids if i != eos_token_id and 0 <= i < len(vocab))
