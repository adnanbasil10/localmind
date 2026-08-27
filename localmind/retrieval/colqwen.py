"""ColQwen2 — visual document retrieval, the differentiator arm (§11).

Every other arm requires OCR/parsing/chunking before a page is searchable. ColQwen2 (a
ColPali-family model) embeds a *rendered image* of the page directly into ~1030 patch-level
vectors (128-dim each, one per ViT patch) and scores queries against them with the same
MaxSim late-interaction used by ColBERT (`localmind.retrieval.colbert.maxsim`) — so tables,
figures, and layout are visible to retrieval instead of being lossily flattened to text first.

Indexing is a GPU job (~2 GPU-h/few-thousand pages on a free Kaggle T4) — see
`notebooks/kaggle/03_index_colqwen.ipynb`, which shells out to this module's CLI:

    python -m localmind.retrieval.colqwen index --pdf-dir DIR --out DIR --quantize int8 \\
        --report-recall
    python -m localmind.retrieval.colqwen push --index DIR

`run_index_command` is written correctly but deliberately refuses to run without a CUDA GPU
(see CONVENTIONS.md: no GPU in this dev environment) — it is not attempted here, matching the
project-wide rule that GPU-only deliverables ship as reproducible code + a Kaggle launcher,
not executed results. Everything else in this module (quantization, MaxSim scoring, the
retained-recall measurement, and `push`) needs no GPU and is unit-tested with a deterministic
fake page encoder.

**Storage warning:** ~1030 patches x 128 dims x 4 bytes (float32) = ~527 KB *per page*,
uncompressed — a 10k-page corpus is ~5.3 GB of vectors alone. `quantize_binary` (32x smaller,
1 bit/dim) and `quantize_int8` (4x smaller) make this tractable on a 16 GB laptop;
`quantization_recall_report` measures exactly how much recall that costs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from localmind.retrieval import ScoredDoc
from localmind.retrieval.colbert import MultiVectorEncoder, maxsim
from localmind.retrieval.dense import stable_str_hash

DEFAULT_PATCHES_PER_PAGE = 1030
DEFAULT_PATCH_DIM = 128


@runtime_checkable
class PageEncoder(Protocol):
    """Rendered page image -> one L2-normalized vector per ViT patch, shape (n_patches, dim)."""

    @property
    def patch_dim(self) -> int: ...

    def encode_page(self, image_path: str) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class Page:
    """One page of a source document, identified by (doc_id, page_number). `image_path` points
    at a rendered page image (produced from the source PDF during indexing).
    """

    doc_id: str
    page_number: int
    image_path: str

    @property
    def page_id(self) -> str:
        return f"{self.doc_id}#p{self.page_number}"


# --------------------------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------------------------


def quantize_binary(vectors: np.ndarray) -> np.ndarray:
    """1 bit per dimension (sign of each component), packed 8-to-a-byte. 32x smaller than
    float32. Similarity is then approximated by Hamming distance (`hamming_maxsim_binary`).
    """
    bits = (vectors > 0).astype(np.uint8)
    return np.packbits(bits, axis=-1)


def quantize_int8(vectors: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric int8 quantization: scale = max(|v|)/127 over the whole batch, so a single
    scalar `scale` dequantizes every row. 4x smaller than float32.
    """
    max_abs = float(np.abs(vectors).max()) if vectors.size else 1.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = np.clip(np.round(vectors / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale


def hamming_maxsim_binary(query_bits: np.ndarray, doc_packed: np.ndarray, n_bits: int) -> float:
    """MaxSim computed over binary-quantized vectors: per query token, take the doc patch with
    the *smallest Hamming distance* (bit-agreement stands in for cosine similarity), map that
    distance to a [-1, 1] "similarity" the same way a sign-agreement fraction would, then sum
    over query tokens exactly like `colbert.maxsim`.
    """
    if query_bits.shape[0] == 0 or doc_packed.shape[0] == 0:
        return 0.0
    total = 0.0
    for q in query_bits:
        xor = np.bitwise_xor(doc_packed, q[None, :])
        hamming = np.unpackbits(xor, axis=-1).sum(axis=-1).astype(np.float64)
        sim = 1.0 - 2.0 * hamming / n_bits
        total += float(sim.max())
    return total


def estimate_storage(
    n_pages: int,
    patches_per_page: int = DEFAULT_PATCHES_PER_PAGE,
    dim: int = DEFAULT_PATCH_DIM,
) -> dict[str, float]:
    """Bytes required to store the whole corpus's patch vectors at float32, int8, and binary
    precision, plus the compression ratio each buys. Pure arithmetic — no vectors needed —
    used to make the storage warning concrete for whatever corpus size the caller has.
    """
    n_vectors = n_pages * patches_per_page
    float32_bytes = n_vectors * dim * 4
    int8_bytes = n_vectors * dim * 1
    binary_bytes = n_vectors * ((dim + 7) // 8)
    return {
        "n_pages": float(n_pages),
        "n_vectors": float(n_vectors),
        "float32_bytes": float(float32_bytes),
        "int8_bytes": float(int8_bytes),
        "binary_bytes": float(binary_bytes),
        "int8_compression_ratio": float32_bytes / int8_bytes if int8_bytes else 0.0,
        "binary_compression_ratio": float32_bytes / binary_bytes if binary_bytes else 0.0,
    }


# --------------------------------------------------------------------------------------------
# Recall retained under quantization
# --------------------------------------------------------------------------------------------


def recall_at_k(rankings: dict[str, list[str]], relevant: dict[str, set[str]], k: int) -> float:
    """Fraction of queries (with >=1 relevant doc) whose top-k ranking contains a relevant doc."""
    hits, total = 0, 0
    for query_id, rel in relevant.items():
        if not rel:
            continue
        total += 1
        if set(rankings.get(query_id, [])[:k]) & rel:
            hits += 1
    return hits / total if total else 0.0


@dataclass(frozen=True, slots=True)
class QuantizationRecallReport:
    """Recall@k with full-precision MaxSim vs. quantized MaxSim, and how much of the original
    recall the quantized scheme retains — the number the spec asks us to report next to the
    storage saving, so a reader can see the trade being made, not just the compression ratio.
    """

    scheme: str
    k: int
    recall_full: float
    recall_quantized: float
    retained_fraction: float  # recall_quantized / recall_full, 1.0 == "lossless at top-k"
    storage: dict[str, float]


def quantization_recall_report(
    page_vectors: dict[str, np.ndarray],
    query_vectors: dict[str, np.ndarray],
    relevant: dict[str, set[str]],
    scheme: str = "binary",
    k: int = 5,
) -> QuantizationRecallReport:
    """Measure recall retained after quantizing patch vectors.

    `page_vectors`: {page_id: (n_patches, dim) float array}. `query_vectors`: {query_id:
    (n_query_tokens, dim) float array}, already L2-normalized (as `MultiVectorEncoder` and
    `PageEncoder` both guarantee).
    """
    if scheme not in ("binary", "int8"):
        raise ValueError(f"Unknown quantization scheme: {scheme!r}")

    full_rankings: dict[str, list[str]] = {}
    quant_rankings: dict[str, list[str]] = {}

    quantized: dict[str, tuple] = {}
    n_bits = 0
    for page_id, vecs in page_vectors.items():
        if scheme == "binary":
            quantized[page_id] = (quantize_binary(vecs),)
            n_bits = vecs.shape[-1]
        else:
            q, scale = quantize_int8(vecs)
            quantized[page_id] = (q, scale)

    for query_id, qvecs in query_vectors.items():
        full_scores = {pid: maxsim(qvecs, vecs) for pid, vecs in page_vectors.items()}
        full_rankings[query_id] = [
            pid for pid, _ in sorted(full_scores.items(), key=lambda kv: kv[1], reverse=True)
        ]

        if scheme == "binary":
            q_bits = quantize_binary(qvecs)
            quant_scores = {
                pid: hamming_maxsim_binary(q_bits, packed[0], n_bits)
                for pid, packed in quantized.items()
            }
        else:
            quant_scores = {
                pid: maxsim(qvecs, dequantize_int8(q, scale))
                for pid, (q, scale) in quantized.items()
            }
        quant_rankings[query_id] = [
            pid for pid, _ in sorted(quant_scores.items(), key=lambda kv: kv[1], reverse=True)
        ]

    recall_full = recall_at_k(full_rankings, relevant, k)
    recall_quantized = recall_at_k(quant_rankings, relevant, k)
    retained = (recall_quantized / recall_full) if recall_full > 0 else 1.0

    any_vecs = next(iter(page_vectors.values()))
    storage = estimate_storage(
        n_pages=len(page_vectors),
        patches_per_page=any_vecs.shape[0],
        dim=any_vecs.shape[-1],
    )
    return QuantizationRecallReport(
        scheme=scheme,
        k=k,
        recall_full=recall_full,
        recall_quantized=recall_quantized,
        retained_fraction=retained,
        storage=storage,
    )


# --------------------------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------------------------


class ColQwenIndex:
    """Visual retrieval arm. Query text is encoded to token-level vectors via a
    `MultiVectorEncoder` (the same Protocol ColBERT's query side uses — ColQwen2 is a
    ColPali-family model, so text and image share one late-interaction backbone); pages are
    encoded to patch-level vectors via `PageEncoder`. Both sides are scored with `maxsim`
    (float) or `hamming_maxsim_binary` (if `quantize="binary"`).
    """

    def __init__(
        self,
        page_encoder: PageEncoder,
        query_encoder: MultiVectorEncoder,
        quantize: str | None = None,
    ) -> None:
        if quantize not in (None, "binary", "int8"):
            raise ValueError(f"Unknown quantization scheme: {quantize!r}")
        self.page_encoder = page_encoder
        self.query_encoder = query_encoder
        self.quantize = quantize
        self._page_ids: list[str] = []
        self._raw_vectors: dict[str, np.ndarray] = {}
        self._binary_vectors: dict[str, np.ndarray] = {}
        self._int8_vectors: dict[str, tuple[np.ndarray, float]] = {}

    @property
    def page_ids(self) -> list[str]:
        return list(self._page_ids)

    @property
    def raw_vectors(self) -> dict[str, np.ndarray]:
        return dict(self._raw_vectors)

    def index(self, pages: list[Page]) -> None:
        self._page_ids = [p.page_id for p in pages]
        for page in pages:
            vecs = self.page_encoder.encode_page(page.image_path)
            self._raw_vectors[page.page_id] = vecs
            if self.quantize == "binary":
                self._binary_vectors[page.page_id] = quantize_binary(vecs)
            elif self.quantize == "int8":
                self._int8_vectors[page.page_id] = quantize_int8(vecs)

    def search(self, query: str, top_k: int = 10) -> list[ScoredDoc]:
        query_vecs = self.query_encoder.encode(query)
        scores: dict[str, float] = {}
        for page_id in self._page_ids:
            if self.quantize == "binary":
                n_bits = self._raw_vectors[page_id].shape[-1]
                q_bits = quantize_binary(query_vecs)
                scores[page_id] = hamming_maxsim_binary(
                    q_bits, self._binary_vectors[page_id], n_bits
                )
            elif self.quantize == "int8":
                q, scale = self._int8_vectors[page_id]
                scores[page_id] = maxsim(query_vecs, dequantize_int8(q, scale))
            else:
                scores[page_id] = maxsim(query_vecs, self._raw_vectors[page_id])
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [ScoredDoc(doc_id=pid, score=score) for pid, score in ranked]


# --------------------------------------------------------------------------------------------
# Deterministic fake encoder (tests / offline demo)
# --------------------------------------------------------------------------------------------


@dataclass
class DeterministicFakePageEncoder:
    """Test/demo double for `PageEncoder`. No colpali-engine, no torch, no GPU, no images —
    `image_path` is treated as an opaque deterministic seed (e.g. a synthetic page's own id),
    which is all the offline benchmark needs: consistent, seed-derived patch vectors so
    retrieval logic and quantization recall are exercised meaningfully without a real model.
    """

    patch_dim_: int = DEFAULT_PATCH_DIM
    n_patches: int = 64  # smaller than the real ~1030 to keep tests fast; storage math scales
    seed: int = 0

    @property
    def patch_dim(self) -> int:
        return self.patch_dim_

    def encode_page(self, image_path: str) -> np.ndarray:
        from localmind.retrieval.dense import l2_normalize

        base_seed = (self.seed * 1_000_003 + stable_str_hash(image_path)) % (2**32)
        rng = np.random.default_rng(base_seed)
        vecs = rng.standard_normal((self.n_patches, self.patch_dim_)).astype(np.float32)
        return l2_normalize(vecs)


# --------------------------------------------------------------------------------------------
# CLI: `python -m localmind.retrieval.colqwen index|push ...`
# --------------------------------------------------------------------------------------------


def load_pages_from_pdfs(pdf_dir: str) -> list[Page]:
    """Render every page of every PDF under `pdf_dir` to an image file next to it. Lazy-imports
    `pdf2image` (needs the system `poppler` binary too) — a Kaggle-notebook-only dependency,
    not part of the `rag` extra, since it's only ever needed on the GPU indexing box.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError as e:
        raise ImportError(
            "pdf2image (+ system poppler-utils) is required to render PDF pages. This is "
            "expected to be missing outside the Kaggle indexing job — see "
            "notebooks/kaggle/03_index_colqwen.ipynb."
        ) from e

    pdf_dir_path = Path(pdf_dir)
    render_dir = pdf_dir_path / "_rendered"
    render_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Page] = []
    for pdf_path in sorted(pdf_dir_path.glob("*.pdf")):
        images = convert_from_path(str(pdf_path))
        for i, image in enumerate(images, start=1):
            out_path = render_dir / f"{pdf_path.stem}_p{i}.png"
            image.save(out_path)
            pages.append(Page(doc_id=pdf_path.stem, page_number=i, image_path=str(out_path)))
    return pages


def _load_colqwen_encoders() -> tuple[PageEncoder, MultiVectorEncoder]:
    """Load the real ColQwen2 model (page + query towers) via `colpali-engine`. Lazy-imported;
    downloads weights over the network and needs a GPU to run at any usable speed — both
    unavailable here by design (see CONVENTIONS.md). Only reachable from `run_index_command`,
    which itself refuses to proceed without CUDA.
    """
    try:
        import torch
        from colpali_engine.models import ColQwen2, ColQwen2Processor
    except ImportError as e:
        raise ImportError(
            "colpali-engine and torch are required for real ColQwen2 indexing. Install them "
            "in the Kaggle notebook environment; see notebooks/kaggle/03_index_colqwen.ipynb."
        ) from e

    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v1.0", torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

    class _ColQwenPageEncoder:
        @property
        def patch_dim(self) -> int:
            return DEFAULT_PATCH_DIM

        def encode_page(self, image_path: str) -> np.ndarray:
            from PIL import Image

            image = Image.open(image_path)
            batch = processor.process_images([image]).to(model.device)
            with torch.no_grad():
                embeddings = model(**batch)
            return embeddings[0].float().cpu().numpy()

    class _ColQwenQueryEncoder:
        @property
        def dim(self) -> int:
            return DEFAULT_PATCH_DIM

        def encode(self, text: str) -> np.ndarray:
            batch = processor.process_queries([text]).to(model.device)
            with torch.no_grad():
                embeddings = model(**batch)
            return embeddings[0].float().cpu().numpy()

    return _ColQwenPageEncoder(), _ColQwenQueryEncoder()


def run_index_command(pdf_dir: str, out_dir: str, quantize: str, report_recall: bool) -> None:
    """The real indexing job (§11): render pages, encode with ColQwen2, quantize, write the
    index. Meant to run on a Kaggle T4 via notebooks/kaggle/03_index_colqwen.ipynb — refuses
    to run without CUDA so nobody accidentally starts an hours-long CPU job by mistake.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("torch is required (install the `torch` extra).") from e
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ColQwen2 indexing needs a GPU (~2 GPU-h for a few thousand pages on a free "
            "Kaggle T4). Run notebooks/kaggle/03_index_colqwen.ipynb, not a laptop CPU."
        )

    page_encoder, query_encoder = _load_colqwen_encoders()
    pages = load_pages_from_pdfs(pdf_dir)
    quant_scheme = None if quantize == "none" else quantize
    index = ColQwenIndex(page_encoder, query_encoder, quantize=quant_scheme)
    index.index(pages)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest = {"page_ids": index.page_ids, "quantize": quantize}
    for page_id, vecs in index.raw_vectors.items():
        np.save(out_path / f"{page_id.replace('#', '_')}.npy", vecs)
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if report_recall:
        # A held-out recall check needs real relevance judgments, which don't exist until a
        # human labels some (query, page) pairs for this corpus — out of scope for the
        # indexing job itself. We still report the storage trade unconditionally so the
        # Kaggle run's logs always contain the compression numbers described in this module's
        # docstring.
        storage = estimate_storage(
            n_pages=len(pages),
            patches_per_page=DEFAULT_PATCHES_PER_PAGE,
            dim=DEFAULT_PATCH_DIM,
        )
        (out_path / "storage_report.json").write_text(json.dumps(storage, indent=2))


def run_push_command(index_dir: str) -> str:
    """Bundle a local ColQwen2 index directory into one `.tar.gz`, ready to move off the
    Kaggle GPU box (e.g. as a Kaggle Dataset output) onto the laptop that will query it.
    Needs no GPU or network, unlike `run_index_command`.
    """
    import tarfile

    src = Path(index_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")
    archive_path = src.with_name(src.name + ".tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(src, arcname=src.name)
    return str(archive_path)


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(prog="python -m localmind.retrieval.colqwen")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Render + encode + quantize a PDF directory (GPU).")
    p_index.add_argument("--pdf-dir", required=True)
    p_index.add_argument("--out", required=True)
    p_index.add_argument("--quantize", choices=["none", "binary", "int8"], default="int8")
    p_index.add_argument("--report-recall", action="store_true")

    p_push = sub.add_parser("push", help="Bundle a built index for transfer off the GPU box.")
    p_push.add_argument("--index", required=True, dest="index_dir")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "index":
        run_index_command(args.pdf_dir, args.out, args.quantize, args.report_recall)
    elif args.command == "push":
        archive = run_push_command(args.index_dir)
        print(archive)


if __name__ == "__main__":
    main()
