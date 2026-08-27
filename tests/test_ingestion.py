"""Tests for `localmind.ingestion` (implementation.md §11, Phase 7).

Everything here runs fully offline: no network, no GPU, no real parser/VLM/
embedding libraries, no live Postgres. Real-backend classes (Docling, Marker,
PyMuPDF4LLM, Surya, PaddleOCR, SmolVLM, Qwen2.5-VL) are exercised only for
their documented "raise ImportError, let the registry fall through" contract,
which is true unconditionally in this environment; a couple of tests are
additionally guarded with `@pytest.mark.skipif` so they auto-enable if a real
library is ever installed. Postgres round-trip tests are `@pytest.mark.docker`.
"""

from __future__ import annotations

import ast
import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from localmind.eval.stats import Estimate
from localmind.ingestion import (
    LOCAL_31M_COST_MODEL,
    QWEN3_4B_API_COST_MODEL,
    Chunk,
    ContextualizeComparison,
    Contextualizer,
    CostModel,
    Embedder,
    FakeHashEmbedder,
    HeatmapResult,
    HeuristicContextualizer,
    IngestionResult,
    PipelineConfig,
    SimulatedContextualizer,
    benchmark_contextualizers,
    chunk_contextual,
    chunk_document,
    chunk_fixed,
    chunk_late,
    chunk_recursive,
    chunk_semantic,
    chunk_stats,
    contextualize_chunks,
    ingest_document,
    ingest_text,
    ndcg_at_k,
    run_size_overlap_ablation,
    run_strategy_ablation,
)
from localmind.ingestion.parse import (
    DoclingParser,
    MarkerParser,
    NoParserAvailableError,
    ParsedBlock,
    ParserRegistry,
    PlainTextParser,
    PyMuPDF4LLMParser,
    default_registry,
    sniff_doc_type,
)
from localmind.ingestion.parse.ocr import PaddleOCRParser, SuryaOCRParser
from localmind.ingestion.parse.tables import (
    ExtractedTable,
    FakeSQLConnection,
    create_schema,
    extract_tables_from_blocks,
    insert_tables,
    parse_markdown_table,
)
from localmind.ingestion.parse.vlm_caption import (
    FigureCrop,
    Qwen25VLCaptioner,
    SmolVLMCaptioner,
    caption_figures,
    figures_from_blocks,
)


def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return 0.0 if na == 0.0 or nb == 0.0 else float(np.dot(a, b) / (na * nb))


# ======================================================================================
# Package hygiene: offline-importability, no forbidden imports
# ======================================================================================


def test_import_does_not_pull_in_torch_or_owned_by_other_tasks():
    assert "torch" not in sys.modules, "a prior import already pulled in torch"
    import localmind.ingestion  # noqa: F401

    assert "torch" not in sys.modules
    assert "localmind.model" not in sys.modules
    assert "localmind.retrieval" not in sys.modules


_INGESTION_ROOT = Path(__file__).resolve().parents[1] / "localmind" / "ingestion"
_FORBIDDEN_MODULES = ("torch", "localmind.model", "localmind.retrieval")


def _module_scope_imports(path: Path) -> set[str]:
    """Names imported at module (top) scope only -- lazy imports inside
    function bodies live deeper in the AST and are intentionally excluded."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_no_forbidden_imports_at_module_scope():
    py_files = sorted(_INGESTION_ROOT.rglob("*.py"))
    assert len(py_files) >= 7, "expected __init__/chunking/contextualize/pipeline + parse/*"
    for f in py_files:
        imported = _module_scope_imports(f)
        for forbidden in _FORBIDDEN_MODULES:
            hit = {m for m in imported if m == forbidden or m.startswith(forbidden + ".")}
            assert not hit, f"{f} imports forbidden module(s) {hit} at module scope"


# ======================================================================================
# Parsers
# ======================================================================================


def test_sniff_doc_type():
    assert sniff_doc_type(Path("a.md")) == "markdown"
    assert sniff_doc_type(Path("a.markdown")) == "markdown"
    assert sniff_doc_type(Path("a.txt")) == "text"
    assert sniff_doc_type(Path("a.pdf")) == "pdf"
    assert sniff_doc_type(Path("a.pdf"), scanned=True) == "pdf_scanned"
    assert sniff_doc_type(Path("a.docx")) == "unknown"


def test_plaintext_parser_classifies_blocks():
    text = (
        "# Title Heading\n\n"
        "Intro paragraph with some words.\n\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | 2 |\n\n"
        "![a chart](figs/chart.png)\n\n"
        "Closing paragraph.\n"
    )
    doc = PlainTextParser().parse_text(text, doc_id="doc1")
    kinds = [b.kind for b in doc.blocks]
    assert kinds == ["heading", "paragraph", "table", "figure", "paragraph"]
    assert doc.blocks[0].text == "Title Heading"
    assert doc.blocks[0].level == 1
    assert doc.blocks[2].text.startswith("| a | b |")
    assert doc.blocks[3].text == "a chart"
    assert doc.blocks[3].metadata["uri"] == "figs/chart.png"
    assert doc.parser == "plaintext-fallback"
    assert "Title Heading" in doc.text and "Closing paragraph." in doc.text


def test_plaintext_parser_reads_a_real_file(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# Hello\n\nWorld.\n", encoding="utf-8")
    doc = PlainTextParser().parse(p)
    assert doc.doc_id == "note"
    assert doc.source_path == str(p)
    assert doc.blocks[0].kind == "heading"


def test_registry_routes_text_and_markdown_to_plaintext_fallback(tmp_path):
    registry = ParserRegistry()
    registry.register(PlainTextParser())
    p = tmp_path / "doc.txt"
    p.write_text("Just some plain text.\n", encoding="utf-8")
    result = registry.parse(p)
    assert result.parser == "plaintext-fallback"
    assert result.doc_type == "markdown"  # PlainTextParser always tags markdown internally


@pytest.mark.parametrize(
    "cls", [DoclingParser, PyMuPDF4LLMParser, MarkerParser, SuryaOCRParser, PaddleOCRParser]
)
def test_real_parsers_raise_importerror_when_library_absent(tmp_path, cls):
    """Documents the lazy-import contract: with no real parsing library
    installed (true in this offline environment), calling `.parse()` directly
    raises ImportError so `ParserRegistry.parse` can catch it and fall
    through. This is *not* gated by `pytest.importorskip` -- it is exactly
    the behaviour under test."""
    fake_path = tmp_path / "doc.pdf"
    fake_path.write_bytes(b"%PDF-1.4 not a real pdf")
    with pytest.raises(ImportError):
        cls().parse(fake_path)


def test_registry_falls_through_unavailable_pdf_parsers_to_no_parser_available(tmp_path):
    """With no PDF backend installed, `default_registry()` must try every
    candidate in order (logging each miss) and finally raise
    `NoParserAvailableError` rather than silently mis-parsing binary PDF bytes as
    text."""
    registry = default_registry()
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 not a real pdf")
    with pytest.raises(NoParserAvailableError) as exc_info:
        registry.parse(p)
    msg = str(exc_info.value)
    assert "docling" in msg and "pymupdf4llm" in msg and "marker" in msg


def test_registry_falls_through_unavailable_scanned_ocr_parsers(tmp_path):
    registry = default_registry()
    p = tmp_path / "scan.pdf"
    p.write_bytes(b"%PDF-1.4 scanned")
    with pytest.raises(NoParserAvailableError) as exc_info:
        registry.parse(p, doc_type="pdf_scanned")
    msg = str(exc_info.value)
    assert "surya" in msg and "paddleocr" in msg


def test_default_registry_still_handles_plain_text(tmp_path):
    registry = default_registry()
    p = tmp_path / "readme.txt"
    p.write_text("Hello offline world.\n", encoding="utf-8")
    result = registry.parse(p)
    assert result.parser == "plaintext-fallback"


@pytest.mark.skipif(not _has_module("docling"), reason="docling not installed")
def test_docling_smoke_when_installed(tmp_path):  # pragma: no cover - only runs if installed
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4")
    doc = DoclingParser().parse(p)
    assert doc.parser == "docling"


# ======================================================================================
# Tables
# ======================================================================================


def test_parse_markdown_table_roundtrip():
    md = "| Name | Qty |\n|---|---|\n| widget | 3 |\n| gadget | 7 |\n"
    headers, rows = parse_markdown_table(md)
    assert headers == ["Name", "Qty"]
    assert rows == [["widget", "3"], ["gadget", "7"]]


def test_parse_markdown_table_rejects_malformed_input():
    with pytest.raises(ValueError, match="separator"):
        parse_markdown_table("| Name | Qty |\n| widget | 3 |\n")
    with pytest.raises(ValueError, match="header row"):
        parse_markdown_table("just one line")


def test_extract_tables_from_blocks_with_provenance():
    blocks = [
        ParsedBlock(kind="paragraph", text="intro", order=0),
        ParsedBlock(
            kind="table",
            text="| A | B |\n|---|---|\n| 1 | 2 |\n",
            order=1,
            page=3,
            bbox=(1.0, 2.0, 3.0, 4.0),
        ),
        ParsedBlock(kind="table", text="not a table at all", order=2),  # malformed, skipped
    ]
    tables = extract_tables_from_blocks(blocks, doc_id="doc7", parser="docling")
    assert len(tables) == 1
    t = tables[0]
    assert t.table_id == "doc7#table0"
    assert t.doc_id == "doc7"
    assert t.page == 3
    assert t.bbox == (1.0, 2.0, 3.0, 4.0)
    assert t.headers == ["A", "B"]
    assert t.rows == [["1", "2"]]
    assert t.parser == "docling"
    # round-trip: re-parsing the rendered markdown reproduces the same structure
    headers2, rows2 = parse_markdown_table(t.to_markdown())
    assert (headers2, rows2) == (t.headers, t.rows)


def test_fake_sql_connection_create_schema_and_insert():
    conn = FakeSQLConnection()
    create_schema(conn)
    assert conn.committed == 1
    assert len(conn.statements) == 2

    table = ExtractedTable(
        table_id="docA#table0",
        doc_id="docA",
        page=5,
        bbox=(0.0, 0.0, 10.0, 10.0),
        headers=["Item", "Price"],
        rows=[["Widget", "9.99"], ["Gadget", "19.99"]],
        parser="marker",
    )
    n = insert_tables(conn, [table])
    assert n == 1
    assert conn.tables["docA#table0"]["doc_id"] == "docA"
    assert conn.tables["docA#table0"]["page"] == 5
    assert conn.tables["docA#table0"]["n_rows"] == 2
    assert conn.cells[("docA#table0", 0, 0)]["value"] == "Widget"
    assert conn.cells[("docA#table0", 0, 0)]["header"] == "Item"
    assert conn.cells[("docA#table0", 1, 1)]["value"] == "19.99"
    assert conn.committed == 2


@pytest.mark.docker
def test_postgres_table_roundtrip_requires_live_docker_compose():
    """Real round-trip against `docker-compose.yml`'s `postgres` service
    (profile `core`). Not run by `just test-fast`. Skips cleanly if Postgres
    isn't actually reachable, rather than failing the suite."""
    pytest.importorskip("psycopg")
    from localmind.ingestion.parse.tables import connect_postgres

    try:
        conn = connect_postgres()
    except Exception as exc:  # pragma: no cover - depends on live infra
        pytest.skip(f"postgres not reachable: {exc}")
    create_schema(conn)
    table = ExtractedTable(
        table_id="live-doc#table0",
        doc_id="live-doc",
        headers=["x"],
        rows=[["1"]],
        parser="test",
    )
    assert insert_tables(conn, [table]) == 1


# ======================================================================================
# Figure captioning
# ======================================================================================


def test_figures_from_blocks_and_fake_captioner():
    blocks = [
        ParsedBlock(
            kind="figure", text="a bar chart", order=0, page=2, metadata={"uri": "figs/f0.png"}
        ),
        ParsedBlock(kind="figure", text="", order=1, page=4, metadata={"uri": "figs/f1.png"}),
    ]
    crops = figures_from_blocks(blocks, doc_id="docX")
    assert [c.figure_id for c in crops] == ["docX#fig0", "docX#fig1"]
    assert crops[0].alt_text == "a bar chart"
    assert crops[0].page == 2

    results = caption_figures(crops)  # default: FakeCaptioner
    assert results[0].caption == "a bar chart"  # alt text reused verbatim
    assert "docX" in results[1].caption and "page 4" in results[1].caption
    assert all(r.captioner == "fake" for r in results)


def test_caption_figures_falls_back_when_captioner_raises():
    class _Broken:
        name = "broken-vlm"

        def caption(self, figure: FigureCrop) -> str:
            raise RuntimeError("no GPU here")

    crop = FigureCrop(figure_id="d#fig0", doc_id="d", page=1, alt_text="")
    results = caption_figures([crop], _Broken())
    assert results[0].caption.startswith("Figure from d")
    assert results[0].captioner == "broken-vlm:fallback(RuntimeError)"


@pytest.mark.parametrize("cls", [SmolVLMCaptioner, Qwen25VLCaptioner])
def test_vlm_captioners_raise_importerror_when_library_absent(cls):
    crop = FigureCrop(figure_id="d#fig0", doc_id="d", crop_uri="does/not/exist.png")
    with pytest.raises(ImportError):
        cls().caption(crop)


# ======================================================================================
# Chunking: five strategies
# ======================================================================================


def test_chunk_fixed_basic_and_chunk_id_contract():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_fixed(text, "docF", size=100, overlap=0)
    assert len(chunks) == 10
    assert all(c.size_tokens == 100 for c in chunks)
    assert [c.chunk_id for c in chunks[:3]] == ["docF#0", "docF#1", "docF#2"]
    assert all(c.doc_id == "docF" for c in chunks)
    # chunks tile the source text exactly, char-accurate
    assert text[chunks[0].start : chunks[0].end] == chunks[0].text


def test_chunk_fixed_overlap_repeats_words():
    text = " ".join(f"w{i}" for i in range(30))
    chunks = chunk_fixed(text, "docO", size=10, overlap=3)
    assert len(chunks) >= 2
    tail_of_first = chunks[0].text.split()[-3:]
    head_of_second = chunks[1].text.split()[:3]
    assert tail_of_first == head_of_second
    assert chunks[1].overlap_tokens == 3


def test_chunk_fixed_rejects_invalid_params():
    with pytest.raises(ValueError):
        chunk_fixed("x", "d", size=0)
    with pytest.raises(ValueError):
        chunk_fixed("x y z", "d", size=5, overlap=5)


def test_chunk_recursive_prefers_paragraph_boundaries():
    para1 = " ".join(f"a{i}" for i in range(20))
    para2 = " ".join(f"b{i}" for i in range(28))
    text = f"{para1}\n\n{para2}"
    chunks = chunk_recursive(text, "docR", size=40, overlap=0)
    assert len(chunks) == 2
    assert chunks[0].text.split()[-1] == "a19"  # ends exactly at the paragraph break
    assert chunks[1].text.split()[0] == "b0"
    assert chunks[0].strategy == "recursive"


def test_chunk_semantic_splits_on_topic_shift():
    topic_a = ". ".join([f"Cats are small domestic animals number {i}" for i in range(4)]) + "."
    topic_b = (
        ". ".join([f"Rockets require liquid fuel for launch stage {i}" for i in range(4)]) + "."
    )
    text = f"{topic_a} {topic_b}"
    chunks = chunk_semantic(text, "docS", FakeHashEmbedder(), max_size=1000)
    assert len(chunks) >= 2
    # the split should separate the two topics rather than interleave them
    assert "Cats" in chunks[0].text or "cats" in chunks[0].text.lower()
    assert "Rockets" in chunks[-1].text or "rockets" in chunks[-1].text.lower()


def test_chunk_semantic_handles_single_sentence():
    chunks = chunk_semantic("Only one sentence here.", "docS2", FakeHashEmbedder())
    assert len(chunks) == 1
    assert chunks[0].text == "Only one sentence here."


def test_chunk_late_edge_cases_prove_the_pooling_mechanism():
    text = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima."
    embedder = FakeHashEmbedder()
    naive = embedder.embed([text])[0]  # single-window doc == single chunk here

    # context_weight=0.0 -> late chunking degenerates to naive per-chunk embedding
    pure_local = chunk_late(text, "docL", embedder, size=100, overlap=0, context_weight=0.0)
    assert np.allclose(pure_local[0].embedding, naive.tolist(), atol=1e-9)

    # context_weight=1.0 -> every chunk's embedding IS the whole-document embedding
    pure_doc = chunk_late(text, "docL", embedder, size=5, overlap=0, context_weight=1.0)
    assert len(pure_doc) > 1
    doc_vec = embedder.embed([text])[0]
    for c in pure_doc:
        assert np.allclose(c.embedding, doc_vec.tolist(), atol=1e-9)

    # the default blend (0.3) differs mechanically from the naive per-chunk embedding
    blended = chunk_late(text, "docL", embedder, size=5, overlap=0)
    local_only = embedder.embed([blended[0].text])[0]
    assert not np.allclose(blended[0].embedding, local_only.tolist(), atol=1e-6)
    assert blended[0].strategy == "late_chunking"


def test_chunk_contextual_prepends_situating_context():
    text = "Acme Corp makes widgets. It ships worldwide."
    chunks = chunk_contextual(text, "docC", HeuristicContextualizer(), size=3, overlap=0)
    assert len(chunks) >= 1
    assert chunks[0].context.startswith("This chunk is from a document that begins:")
    assert chunks[0].strategy == "contextual_retrieval"
    assert chunks[0].context in chunks[0].full_text
    assert chunks[0].text in chunks[0].full_text


def test_contextual_retrieval_beats_plain_chunking_on_a_pronoun_reference_example():
    """A deterministic, hand-verifiable demonstration of *why* contextual
    retrieval is the highest-impact strategy: a chunk that only says "It lasts
    twenty four months..." has near-zero lexical overlap with a query naming
    the product, until the situating context reintroduces the product name."""
    para1 = (
        "Acme Corp is a hardware manufacturer based in Springfield. This document "
        "describes the extended warranty program for Acme Corp customers."
    )
    para2 = (
        "It offers coverage for accidental drops and liquid spills. It lasts twenty "
        "four months from the original purchase date. Customers can file a claim "
        "online at any time."
    )
    text = f"{para1}\n\n{para2}"
    query = "What is Acme Corp's extended warranty length?"
    embedder = FakeHashEmbedder()
    q_vec = embedder.embed([query])[0]

    plain = chunk_recursive(text, "docW", size=40, overlap=0)
    contextual = chunk_contextual(text, "docW", HeuristicContextualizer(), size=40, overlap=0)
    assert len(plain) == 2 and len(contextual) == 2
    assert "Acme" not in plain[1].text  # the target chunk really has lost the antecedent

    plain_vec = embedder.embed([plain[1].text])[0]
    contextual_vec = embedder.embed([contextual[1].full_text])[0]

    cos_plain = _cosine(plain_vec, q_vec)
    cos_contextual = _cosine(contextual_vec, q_vec)
    assert cos_contextual > cos_plain
    assert cos_plain < 0.1
    assert cos_contextual > 0.15


def test_chunk_document_dispatches_all_five_strategies():
    text = " ".join(f"tok{i}." if i % 8 == 0 else f"tok{i}" for i in range(300))
    for strategy in ("fixed", "recursive", "semantic", "late", "contextual"):
        chunks = chunk_document(text, "docD", strategy, size=64, overlap=0)
        assert len(chunks) > 0, strategy
        assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_document_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        chunk_document("x", "d", "bogus-strategy")  # type: ignore[arg-type]


def test_chunk_stats_reports_real_numbers():
    chunks = chunk_fixed(" ".join(f"w{i}" for i in range(250)), "docStats", size=100, overlap=10)
    stats = chunk_stats(chunks)
    assert stats["n_chunks"] == float(len(chunks))
    assert stats["max_size"] <= 100
    assert stats["mean_overlap"] >= 0.0


def test_chunk_stats_empty():
    assert chunk_stats([]) == {
        "n_chunks": 0,
        "mean_size": 0.0,
        "min_size": 0.0,
        "max_size": 0.0,
        "mean_overlap": 0.0,
    }


# ======================================================================================
# nDCG
# ======================================================================================


def test_ndcg_perfect_and_empty_and_worst_order():
    assert ndcg_at_k([1, 1, 1], k=3) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0, 0], k=3) == 0.0
    assert ndcg_at_k([], k=3) == 0.0
    # relevant item at rank 2 instead of rank 1
    assert ndcg_at_k([0, 1], k=10) == pytest.approx(1.0 / math.log2(3))


def test_ndcg_truncates_at_k():
    # ten relevant items, k=1 -> only the first counts, and it's a hit -> perfect
    assert ndcg_at_k([1] * 10, k=1) == pytest.approx(1.0)


# ======================================================================================
# Ablation harness (SYNTHETIC -- see chunking.py docstring)
# ======================================================================================


def test_run_strategy_ablation_reports_all_five_with_ci():
    rows = run_strategy_ablation(size=512, overlap=0, seeds=(1, 2, 3))
    assert {r.name for r in rows} == {"fixed", "recursive", "semantic", "late", "contextual"}
    for row in rows:
        est = row.values["ndcg@10"]
        assert isinstance(est, Estimate)  # never a bare number (CONVENTIONS.md rule 5)
        # a t-interval over only 3 seed means is not itself clipped to [0, 1]
        # (small-n CIs can legitimately overshoot); the mean of a metric in
        # [0, 1] must still land in [0, 1], and lo <= mean <= hi always holds.
        assert est.lo <= est.mean <= est.hi
        assert 0.0 <= est.mean <= 1.0
        assert row.extra["harness"] == "synthetic"


def test_run_size_overlap_ablation_small_grid_is_a_real_grid_not_one_number():
    heat = run_size_overlap_ablation(
        strategy="recursive",
        sizes=(256, 512),
        overlap_fracs=(0.0, 0.10),
        seeds=(1, 2, 3),
    )
    assert isinstance(heat, HeatmapResult)
    assert len(heat.cells) == 4
    for size in (256, 512):
        for frac in (0.0, 0.10):
            est = heat.cell(size, frac)
            assert est.lo <= est.mean <= est.hi
            assert 0.0 <= est.mean <= 1.0
    md = heat.to_markdown()
    assert "SYNTHETIC HARNESS" in md
    assert "256" in md and "512" in md


@pytest.mark.slow
def test_run_size_overlap_ablation_full_default_grid():
    """The full {256,512,768,1024} x {0,10%,20%} grid the spec asks for --
    12 cells, each a >=3-seed CI. Marked slow (excluded from `just test-fast`)
    because it is ~6x the small-grid test above."""
    heat = run_size_overlap_ablation()
    assert heat.sizes == [256, 512, 768, 1024]
    assert heat.overlap_fracs == [0.0, 0.10, 0.20]
    assert len(heat.cells) == 12
    md = heat.to_markdown()
    assert md.count("|") > 20  # a real rendered table, not a single number


# ======================================================================================
# Contextualize: injectable seam + cost/latency measurement
# ======================================================================================


def test_contextualizer_protocol_conformance():
    assert isinstance(HeuristicContextualizer(), Contextualizer)
    assert isinstance(SimulatedContextualizer(LOCAL_31M_COST_MODEL), Contextualizer)


def test_heuristic_contextualizer_is_deterministic_and_uses_document_opening():
    doc = "Widgets are great. They come in blue. They also come in red."
    ctx = HeuristicContextualizer()
    out1 = ctx.situate(doc, "irrelevant chunk text")
    out2 = ctx.situate(doc, "different chunk text")
    assert out1 == out2  # only depends on the document, not the chunk
    assert "Widgets are great." in out1


def test_contextualize_chunks_produces_aligned_records():
    class _C:
        chunk_id = "d#0"
        text = "chunk text"

    records = contextualize_chunks("A document about widgets.", [_C()])
    assert len(records) == 1
    assert records[0].chunk_id == "d#0"
    assert records[0].latency_ms >= 0.0
    assert records[0].cost_usd == 0.0
    assert records[0].backend == "heuristic"


def test_benchmark_contextualizers_local_is_free_and_faster_than_api():
    chunks = chunk_fixed("word " * 200, "docB", size=50, overlap=0)
    comparison = benchmark_contextualizers("A document about widgets and gadgets.", chunks)
    assert isinstance(comparison, ContextualizeComparison)
    assert comparison.local_total_cost_usd == 0.0
    assert comparison.api_total_cost_usd > 0.0
    assert comparison.cost_savings_usd == pytest.approx(comparison.api_total_cost_usd)
    assert comparison.simulated_latency_speedup > 1.0
    rendered = comparison.render()
    assert "SIMULATED" in rendered
    assert "qwen3-4b" in rendered


def test_cost_models_are_labelled_illustrative():
    assert isinstance(LOCAL_31M_COST_MODEL, CostModel)
    assert isinstance(QWEN3_4B_API_COST_MODEL, CostModel)
    assert LOCAL_31M_COST_MODEL.simulated_cost_usd_per_call == 0.0
    assert QWEN3_4B_API_COST_MODEL.simulated_cost_usd_per_call > 0.0
    assert "illustrative" in QWEN3_4B_API_COST_MODEL.note


def test_embedder_protocol_conformance():
    assert isinstance(FakeHashEmbedder(), Embedder)
    vecs = FakeHashEmbedder(dim=32).embed(["hello world", "hello world"])
    assert vecs.shape == (2, 32)
    assert np.allclose(vecs[0], vecs[1])  # deterministic
    assert np.linalg.norm(vecs[0]) == pytest.approx(1.0)


# ======================================================================================
# Pipeline: end-to-end on synthetic documents
# ======================================================================================


def test_pipeline_config_overlap_rounding():
    cfg = PipelineConfig(chunk_size=512, chunk_overlap_frac=0.10)
    assert cfg.chunk_overlap == 51  # round(512 * 0.10)


def test_ingest_text_end_to_end_on_a_synthetic_document():
    text = (
        "# Product Manual\n\n"
        "This manual covers the X200 widget in detail across several sections.\n\n"
        "| Spec | Value |\n|---|---|\n| Weight | 200g |\n| Color | Blue |\n\n"
        "![the widget diagram](figs/widget.png)\n\n" + " ".join(f"filler{i}" for i in range(400))
    )
    result = ingest_text(
        text, "manual-1", config=PipelineConfig(chunk_strategy="recursive", chunk_size=128)
    )
    assert isinstance(result, IngestionResult)
    assert result.parser == "plaintext-fallback"
    assert result.n_blocks >= 4

    assert len(result.tables) == 1
    assert result.tables[0].headers == ["Spec", "Value"]

    assert len(result.figure_crops) == 1
    assert len(result.figure_captions) == 1
    assert result.figure_captions[0].caption == "the widget diagram"

    assert len(result.chunks) >= 2
    assert all(c.doc_id == "manual-1" for c in result.chunks)
    assert all(c.chunk_id.startswith("manual-1#") for c in result.chunks)

    # PlainTextParser never renders page images -- the ColQwen2 handoff list
    # is legitimately empty for the offline fallback parser.
    assert result.page_images == []

    stats = result.stats
    assert stats["n_chunks"] == float(len(result.chunks))


def test_ingest_document_reads_a_real_file(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("Hello. " + " ".join(f"w{i}" for i in range(200)), encoding="utf-8")
    result = ingest_document(p, config=PipelineConfig(chunk_strategy="fixed", chunk_size=64))
    assert result.doc_id == "note"
    assert result.parser == "plaintext-fallback"
    assert len(result.chunks) > 1


def test_ingest_document_raises_no_parser_available_for_unbacked_pdf(tmp_path):
    p = tmp_path / "scan.pdf"
    p.write_bytes(b"%PDF-1.4")
    with pytest.raises(NoParserAvailableError):
        ingest_document(p)


def test_ingest_text_with_contextual_strategy_wires_context_into_chunks():
    text = "Acme Corp policy document.\n\n" + " ".join(f"item{i}" for i in range(200))
    result = ingest_text(
        text, "docK", config=PipelineConfig(chunk_strategy="contextual", chunk_size=64)
    )
    assert all(c.context for c in result.chunks)
