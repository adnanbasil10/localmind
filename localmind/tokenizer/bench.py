"""Tokenizer benchmarks: yours (Python, naive vs incremental merge loop) vs
`tiktoken` cl100k_base vs GPT-2.

Run as a module from the repo root:

    uv run python -m localmind.tokenizer.bench

Writes `artifacts/benchmarks/tokenizer.json` (schema per CONVENTIONS.md's
"Reporting" section: `{"name","hardware","seeds","rows":[...],"ci"}`, plus two extra
top-level fields this task's spec asks for that don't fit that per-row shape:
`merge_loop_benchmark` -- the naive-vs-incremental wall-clock speedup -- and the
"Proxy val BPB" column, which every row carries as `null`: it needs a GPU proxy-model
training run that happens in a later phase, not something a laptop-only tokenizer
task can produce.

Metric definitions (measured on a held-out eval split, never shown to the trainer --
see `build_corpus`):
    bytes/token   total UTF-8 bytes of the eval text / tokens produced. Higher means
                  better compression (fewer tokens for the same text).
    fertility     tokens produced / whitespace-split word count of the eval text.
                  Lower means less subword fragmentation (1.0 = one token per word).
    encode MB/s   input MB / best-of-3 wall-clock encode time.

No network call happens at import time or at module load. `tiktoken.get_encoding`
downloads and caches BPE rank files on first use, so the tiktoken/GPT-2 rows need
network *the first time they run in a given environment*; if that fails (offline
sandbox, nothing cached yet), those two rows are written with an `"error"` field
instead of crashing the whole benchmark.
"""

from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path
from typing import Any

from localmind.tokenizer.bpe import train_bpe_incremental, train_bpe_naive
from localmind.tokenizer.tokenizer import BASE_VOCAB_SIZE, Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "tokenizer.json"
SEED = 1234

_WORD_BANK = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "while",
    "researchers",
    "build",
    "a",
    "small",
    "transformer",
    "model",
    "on",
    "a",
    "laptop",
    "and",
    "evaluate",
    "retrieval",
    "augmented",
    "generation",
    "systems",
    "using",
    "byte",
    "pair",
    "encoding",
    "tokenizers",
    "trained",
    "entirely",
    "from",
    "scratch",
    "on",
    "free",
    "compute",
    "budgets",
    "across",
    "kaggle",
    "notebooks",
    "and",
    "a",
    "single",
    "consumer",
    "machine",
]
_CONTRACTIONS = ["I've", "it's", "don't", "we're", "they'll", "you'd", "can't", "isn't", "that's"]
_PUNCT = [".", ",", "!", "?", ";", ":", "-"]
_UNICODE_SAMPLES = (
    "the model replied 你好世界 during the demo",
    "emoji stress test \U0001f680\U0001f525\U0001f389 plus surrounding text",
    "café naïve résumé with latin accents",
    "مرحبا بالعالم right to left script",
    "こんにちは世界 japanese greeting",
    "안녕하세요 korean greeting mixed in",
)


def _synthetic_sentences(n: int, rng: random.Random) -> list[str]:
    """Deterministic (given `rng`), locally-generated sentences: no network, no
    corpus to download. Mixes plain English words with contractions, digit runs of
    varying length (to exercise the 3-digit cap), punctuation, and a slice of
    non-Latin/emoji text so bytes/token isn't measured on ASCII alone.

    Every returned sentence is a distinct string (a bounded-retry loop rejects
    duplicates). This matters beyond cosmetics: `build_corpus` splits this list
    80/20 into train/eval, and `_UNICODE_SAMPLES` is a small fixed pool -- without
    deduplication, the same exact fixed string would very likely land on both sides
    of the split, silently violating the "never train on data overlapping eval"
    trap. See `test_build_corpus_train_eval_disjoint` in test_tokenizer.py.
    """
    seen: set[str] = set()
    out: list[str] = []
    max_attempts = n * 20
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        if rng.random() < 0.08:
            # Give unicode-sample sentences some random word-bank context too, so
            # more than just the 6 fixed strings can survive deduplication.
            context_len = rng.randint(0, 3)
            context = " ".join(rng.choice(_WORD_BANK) for _ in range(context_len))
            sample = rng.choice(_UNICODE_SAMPLES)
            sentence = f"{context} {sample}".strip() if context else sample
        else:
            length = rng.randint(6, 16)
            words = [rng.choice(_WORD_BANK) for _ in range(length)]
            if rng.random() < 0.3:
                words.insert(rng.randrange(len(words) + 1), rng.choice(_CONTRACTIONS))
            if rng.random() < 0.4:
                num = rng.randint(0, 10 ** rng.randint(1, 6))
                words.insert(rng.randrange(len(words) + 1), str(num))
            raw = " ".join(words) + rng.choice(_PUNCT)
            sentence = raw[0].upper() + raw[1:]
        if sentence in seen:
            continue
        seen.add(sentence)
        out.append(sentence)
    return out


def _local_corpus_texts() -> list[str]:
    """Real English/technical-prose text already sitting in the repo -- no download
    needed. Falls back to nothing found (still fine; synthetic text covers it).
    """
    candidates = [REPO_ROOT / "implementation.md", REPO_ROOT / "CONVENTIONS.md"]
    docs_dir = REPO_ROOT / "docs"
    if docs_dir.exists():
        candidates += sorted(docs_dir.rglob("*.md"))
    texts = []
    for path in candidates:
        if path.exists() and path.is_file():
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return texts


def build_corpus(seed: int = SEED, n_synthetic: int = 4000) -> tuple[list[str], str]:
    """Returns `(train_texts, eval_text)` with **no overlap**: `eval_text` is built
    from a disjoint slice of the synthetic sentences, never shown to the trainer.
    This directly guards against the spec's named trap ("training the tokenizer on
    data overlapping your eval set") -- see `test_build_corpus_no_overlap` in
    test_tokenizer.py, which asserts the disjointness holds.
    """
    rng = random.Random(seed)
    local_texts = _local_corpus_texts()
    synthetic = _synthetic_sentences(n_synthetic, rng)
    rng.shuffle(synthetic)

    split = int(len(synthetic) * 0.8)
    train_texts = [*local_texts, *synthetic[:split]]
    eval_text = "\n".join(synthetic[split:])
    return train_texts, eval_text


def _bytes_per_token(encode_fn, text: str) -> float:
    ids = encode_fn(text)
    n_bytes = len(text.encode("utf-8", errors="surrogatepass"))
    return n_bytes / max(len(ids), 1)


def _fertility(encode_fn, text: str) -> float:
    ids = encode_fn(text)
    n_words = max(len(text.split()), 1)
    return len(ids) / n_words


def _encode_mb_s(encode_fn, text: str, repeats: int = 3) -> float:
    n_bytes = len(text.encode("utf-8", errors="surrogatepass"))
    best = min(_time_once(encode_fn, text) for _ in range(repeats))
    return (n_bytes / best) / 1e6 if best > 0 else float("inf")


def _time_once(encode_fn, text: str) -> float:
    t0 = time.perf_counter()
    encode_fn(text)
    return time.perf_counter() - t0


def _row(name: str, vocab: int, encode_fn, eval_text: str) -> dict[str, Any]:
    try:
        return {
            "tokenizer": name,
            "vocab": vocab,
            "bytes_per_token": round(_bytes_per_token(encode_fn, eval_text), 4),
            "fertility": round(_fertility(encode_fn, eval_text), 4),
            "encode_mb_s": round(_encode_mb_s(encode_fn, eval_text), 4),
            "proxy_val_bpb": None,  # needs a GPU proxy-model training run (later phase)
            "error": None,
        }
    except Exception as exc:  # tiktoken rows may need network the first time; never crash
        return {
            "tokenizer": name,
            "vocab": vocab,
            "bytes_per_token": None,
            "fertility": None,
            "encode_mb_s": None,
            "proxy_val_bpb": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _tiktoken_rows(eval_text: str) -> list[dict[str, Any]]:
    try:
        import tiktoken
    except ImportError as exc:
        err = f"{type(exc).__name__}: {exc}"
        return [
            {"tokenizer": "tiktoken cl100k", "vocab": 100256, "error": err},
            {"tokenizer": "GPT-2", "vocab": 50257, "error": err},
        ]
    rows = []
    for name, encoding_name, vocab in (
        ("tiktoken cl100k", "cl100k_base", 100256),
        ("GPT-2", "gpt2", 50257),
    ):
        try:
            enc = tiktoken.get_encoding(encoding_name)
            rows.append(_row(name, enc.n_vocab, enc.encode, eval_text))
        except Exception as exc:
            rows.append(
                {"tokenizer": name, "vocab": vocab, "error": f"{type(exc).__name__}: {exc}"}
            )
    return rows


def _merge_loop_benchmark(
    word_freqs: dict[tuple[int, ...], int], vocab_size: int
) -> dict[str, Any]:
    """Times `train_bpe_naive` against `train_bpe_incremental` on the same corpus
    and target vocab size, and asserts they agree -- the speedup number this
    produces is what the spec explicitly asks to be reported.
    """
    t0 = time.perf_counter()
    naive = train_bpe_naive(word_freqs, vocab_size=vocab_size, base_vocab_size=BASE_VOCAB_SIZE)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    incremental = train_bpe_incremental(
        word_freqs, vocab_size=vocab_size, base_vocab_size=BASE_VOCAB_SIZE
    )
    t_incremental = time.perf_counter() - t0

    assert naive.merges == incremental.merges, "naive/incremental merge loops diverged"

    speedup = t_naive / t_incremental if t_incremental > 0 else float("inf")
    return {
        "vocab_size": vocab_size,
        "naive_seconds": round(t_naive, 4),
        "incremental_seconds": round(t_incremental, 4),
        "speedup_x": round(speedup, 2),
    }


def _fmt(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["Tokenizer", "Vocab", "Bytes/token", "Fertility", "Encode MB/s", "Proxy val BPB"]
    widths = [16, 8, 12, 10, 12, 14]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        if r.get("error"):
            cells = [r["tokenizer"], str(r.get("vocab", "")), f"error: {r['error'][:40]}"]
            print(" | ".join(cells))
            continue
        cells = [
            r["tokenizer"],
            str(r.get("vocab", "")),
            _fmt(r.get("bytes_per_token")),
            _fmt(r.get("fertility")),
            _fmt(r.get("encode_mb_s")),
            _fmt(r.get("proxy_val_bpb")),
        ]
        print(" | ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)))


def run(vocab_size: int = 16384, merge_bench_vocab_size: int = 2048) -> dict[str, Any]:
    train_texts, eval_text = build_corpus()

    print(
        f"[bench] training LocalMind tokenizer: vocab_size={vocab_size} "
        f"on {len(train_texts)} documents..."
    )
    t0 = time.perf_counter()
    tok = Tokenizer.train(train_texts, vocab_size=vocab_size, algorithm="incremental")
    train_seconds = time.perf_counter() - t0
    print(f"[bench] trained in {train_seconds:.1f}s -> vocab_size={tok.vocab_size}")

    rows = [_row("Yours (Python)", tok.vocab_size, tok.encode, eval_text)]
    rows += _tiktoken_rows(eval_text)

    print(f"[bench] merge-loop naive vs incremental at vocab_size={merge_bench_vocab_size}...")
    word_freqs = Tokenizer._build_word_freqs(train_texts)
    merge_bench = _merge_loop_benchmark(word_freqs, merge_bench_vocab_size)
    print(
        f"[bench] naive={merge_bench['naive_seconds']}s "
        f"incremental={merge_bench['incremental_seconds']}s "
        f"speedup={merge_bench['speedup_x']}x"
    )

    report = {
        "name": "tokenizer",
        "hardware": f"{platform.system()} {platform.release()} / {platform.processor()}",
        "seeds": [SEED],
        "rows": rows,
        "ci": "bootstrap95",
        "notes": (
            "Single-run wall clock (no repeated seeds): bootstrap95 CI applies to the "
            "later GPU proxy-model ablations that consume this tokenizer, not to "
            "these deterministic, seed-independent encode/train measurements. "
            "proxy_val_bpb is null on every row -- it needs a GPU proxy-model "
            "training run from a later phase."
        ),
        "merge_loop_benchmark": merge_bench,
        "tokenizer_train_seconds": round(train_seconds, 2),
        "eval_text_bytes": len(eval_text.encode("utf-8", errors="surrogatepass")),
        "train_documents": len(train_texts),
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[bench] wrote {ARTIFACT_PATH}")
    _print_table(rows)
    return report


if __name__ == "__main__":
    run()
