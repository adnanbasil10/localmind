"""Tests for `localmind.retrieval` — the four-arm hybrid retrieval stack.

Structure:
  1. Unit tests per arm (BM25, SPLADE, dense, ColBERT, ColQwen2), fusion, reranking, and the
     pgvector/Qdrant index-engineering helpers. All run offline against deterministic fakes.
  2. `test_benchmark_table_and_fusion_comparison` — builds a synthetic-but-nontrivial corpus,
     runs the spec's 9-row benchmark table (3 seeds, bootstrap 95% CIs) and the RRF-vs-tuned-
     fusion comparison, and writes `artifacts/benchmarks/retrieval.json`. Marked `slow`.

Tests needing real model weights are marked `@pytest.mark.net`; tests needing a live
Postgres/Qdrant are marked `@pytest.mark.docker`. Both categories additionally guard with
`pytest.importorskip`/try-except so `uv run pytest tests/test_retrieval.py -q` is fully green
in this offline, no-Docker, no-GPU environment rather than erroring out.
"""

from __future__ import annotations

import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import pytest
from localmind.retrieval import (
    Document,
    RetrievalConfig,
    ScoredDoc,
    ranks_from_scores,
    reciprocal_rank,
)
from localmind.retrieval.bm25 import BM25Config, BM25Index, BM25SIndex, tokenize
from localmind.retrieval.colbert import (
    ColBERTIndex,
    DeterministicFakeMultiVectorEncoder,
    maxsim,
)
from localmind.retrieval.colqwen import (
    ColQwenIndex,
    DeterministicFakePageEncoder,
    Page,
    dequantize_int8,
    estimate_storage,
    hamming_maxsim_binary,
    quantization_recall_report,
    quantize_binary,
    quantize_int8,
    run_index_command,
    run_push_command,
)
from localmind.retrieval.colqwen import (
    main as colqwen_main,
)
from localmind.retrieval.dense import (
    DeterministicFakeEmbedder,
    TwoStageDenseIndex,
    l2_normalize,
    matryoshka_truncate,
    stable_str_hash,
)
from localmind.retrieval.fusion import (
    DEFAULT_RRF_K,
    compare_fusion_strategies,
    minmax_normalize,
    reciprocal_rank_fusion,
    tune_weights,
    weighted_fusion,
)
from localmind.retrieval.index.pgvector import (
    HNSWConfig,
    PgVectorIndex,
    SimpleHNSW,
    binary_quantization_report,
    brute_force_search,
    create_hnsw_index_sql,
    create_table_sql,
    filtered_ann_demo,
    recall_vs_latency_curve,
    search_postfiltered_sql,
    search_prefiltered_sql,
    set_ef_search_sql,
)
from localmind.retrieval.index.qdrant import QdrantHNSWConfig, QdrantIndex
from localmind.retrieval.rerank import (
    DeterministicFakeCrossEncoder,
    DeterministicFakeListwiseReranker,
    rerank_cross_encoder,
    rerank_listwise,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# ==============================================================================================
# BM25 — hand-verified against an independently-computed formula
# ==============================================================================================


def test_bm25_hand_verified_single_term_query():
    """2-document corpus, verified by hand: doc0='a b b' (len 3), doc1='b c c c' (len 4),
    avgdl=3.5, N=2, k1=1.5, b=0.75. Query 'b' is the only shared term, df(b)=2, so
    idf(b) = ln((2-2+0.5)/(2+0.5) + 1) = ln(1.2) = 0.1823215567939546 (computed independently
    with Python's `math.log`, not by calling BM25Index -- see the task report for the
    derivation). Expected scores follow from the BM25 formula applied by hand.
    """
    docs = [Document(doc_id="d0", text="a b b"), Document(doc_id="d1", text="b c c c")]
    index = BM25Index(BM25Config(k1=1.5, b=0.75))
    index.index(docs)

    assert index._avgdl == pytest.approx(3.5)
    assert index._idf["b"] == pytest.approx(0.1823215567939546, abs=1e-12)

    assert index.score("b", 0) == pytest.approx(0.27299484439736516, abs=1e-9)
    assert index.score("b", 1) == pytest.approx(0.17130884530975599, abs=1e-9)


def test_bm25_hand_verified_multi_term_query():
    """Same corpus, query 'b c'. idf(c) = ln((2-1+0.5)/(1+0.5)+1) = ln(2) = 0.6931471805599453
    (df(c)=1, only doc1 contains 'c'). doc0 has no 'c' so its score is unchanged from the
    single-term case; doc1 gains a large contribution from 'c' (f=3, high IDF).
    """
    docs = [Document(doc_id="d0", text="a b b"), Document(doc_id="d1", text="b c c c")]
    index = BM25Index(BM25Config(k1=1.5, b=0.75))
    index.index(docs)

    assert index._idf["c"] == pytest.approx(0.6931471805599453, abs=1e-12)
    assert index.score("b c", 0) == pytest.approx(0.27299484439736516, abs=1e-9)
    assert index.score("b c", 1) == pytest.approx(1.2867181013832312, abs=1e-9)


def test_bm25_k1_zero_ignores_term_frequency():
    """k1=0 collapses the tf-saturation term entirely: a document containing a term once should
    score identically to one containing it many times, since f cancels: f*(k1+1)/(f+0) = k1+1.
    """
    docs = [
        Document(doc_id="once", text="whale ocean current"),
        Document(doc_id="many", text="whale whale whale whale ocean current"),
    ]
    index = BM25Index(BM25Config(k1=0.0, b=0.0))
    index.index(docs)
    assert index.score("whale", 0) == pytest.approx(index.score("whale", 1), abs=1e-9)


def test_bm25_b_controls_length_normalization():
    """b=0 disables length normalization: a short and a long document with identical term
    frequency for the query term must score identically, since dl/avgdl drops out of the
    denominator entirely when b=0.
    """
    docs = [
        Document(doc_id="short", text="fox"),
        Document(doc_id="long", text="fox " + "padding " * 20),
    ]
    index = BM25Index(BM25Config(k1=1.5, b=0.0))
    index.index(docs)
    assert index.score("fox", 0) == pytest.approx(index.score("fox", 1), abs=1e-9)


def test_bm25_search_ranks_matching_doc_first():
    docs = [
        Document(doc_id="relevant", text="the quick brown fox jumps"),
        Document(doc_id="irrelevant", text="completely unrelated content here"),
    ]
    index = BM25Index()
    index.index(docs)
    results = index.search("quick fox", top_k=2)
    assert results[0].doc_id == "relevant"
    assert results[0].score > results[1].score


def test_bm25_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World! v2.0") == ["hello", "world", "v2", "0"]


def test_bm25s_production_wrapper_matches_own_ranking_when_available():
    """bm25s isn't installed in this environment (it's in the `rag` extra); skip cleanly
    rather than error. When it IS installed, the production wrapper should agree with the
    from-scratch implementation on which document is most relevant (exact score equality
    isn't guaranteed -- bm25s may use a slightly different IDF variant -- but the ranking of
    an unambiguous query should still agree).
    """
    pytest.importorskip("bm25s")
    docs = [
        Document(doc_id="relevant", text="the quick brown fox jumps"),
        Document(doc_id="irrelevant", text="completely unrelated content here"),
    ]
    own = BM25Index()
    own.index(docs)
    prod = BM25SIndex()
    prod.index(docs)
    assert own.search("quick fox", top_k=1)[0].doc_id == prod.search("quick fox", top_k=1)[0].doc_id


def test_bm25s_index_without_dependency_raises_clear_error():
    if _import_ok("bm25s"):
        pytest.skip("bm25s is installed; the ImportError path isn't reachable")
    docs = [Document(doc_id="d0", text="a b c")]
    with pytest.raises(ImportError, match="bm25s"):
        BM25SIndex().index(docs)


# ==============================================================================================
# SPLADE
# ==============================================================================================


def test_splade_expansion_bridges_vocabulary_mismatch():
    """A document containing only 'automobile' should be found by a query for 'car', purely
    because the (fake, deterministic) encoder expands 'car' -> 'automobile' -- the mechanism
    SPLADE exists for. A pure term-overlap arm (like plain BM25 without this expansion) would
    score this pair 0.
    """
    from localmind.retrieval.splade import DeterministicExpansionEncoder, SpladeIndex

    encoder = DeterministicExpansionEncoder(expansions={"car": ["automobile"]})
    docs = [
        Document(doc_id="match", text="automobile maintenance schedule"),
        Document(doc_id="nomatch", text="bicycle maintenance schedule"),
    ]
    index = SpladeIndex(encoder)
    index.index(docs)
    results = index.search("car", top_k=2)
    assert results[0].doc_id == "match"
    assert results[0].score > 0


def test_splade_encoder_weights_are_term_counts_plus_expansions():
    from localmind.retrieval.splade import DeterministicExpansionEncoder

    encoder = DeterministicExpansionEncoder(expansions={"car": ["vehicle"]}, expansion_weight=0.5)
    weights = encoder.encode("car car bus")
    assert weights["car"] == 2.0
    assert weights["bus"] == 1.0
    assert weights["vehicle"] == 0.5


def test_fastembed_splade_encoder_without_dependency_raises_clear_error():
    from localmind.retrieval.splade import fastembed_splade_encoder

    if _import_ok("fastembed"):
        pytest.skip("fastembed is installed; the ImportError path isn't reachable")
    with pytest.raises(ImportError, match="fastembed"):
        fastembed_splade_encoder()


@pytest.mark.net
def test_fastembed_splade_encoder_real_model():
    pytest.importorskip("fastembed")
    from localmind.retrieval.splade import fastembed_splade_encoder

    encoder = fastembed_splade_encoder()
    weights = encoder.encode("hello world")
    assert all(w >= 0 for w in weights.values())


# ==============================================================================================
# Dense / Matryoshka
# ==============================================================================================


def test_matryoshka_truncate_renormalizes_to_unit_length():
    rng = np.random.default_rng(0)
    vectors = l2_normalize(rng.standard_normal((5, 1024)).astype(np.float32))
    truncated = matryoshka_truncate(vectors, 256)
    assert truncated.shape == (5, 256)
    norms = np.linalg.norm(truncated, axis=-1)
    np.testing.assert_allclose(norms, np.ones(5), atol=1e-5)


def test_matryoshka_truncate_rejects_dim_larger_than_input():
    vectors = np.zeros((2, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="Cannot truncate"):
        matryoshka_truncate(vectors, 256)


def test_deterministic_fake_embedder_is_reproducible():
    embedder = DeterministicFakeEmbedder(dim_=64, seed=42)
    a = embedder.embed(["hello world"])
    b = embedder.embed(["hello world"])
    np.testing.assert_array_equal(a, b)
    assert a.shape == (1, 64)
    assert np.linalg.norm(a[0]) == pytest.approx(1.0, abs=1e-5)


def test_deterministic_fake_embedder_shares_no_hidden_state_across_seeds():
    a = DeterministicFakeEmbedder(dim_=32, seed=1).embed(["same text"])
    b = DeterministicFakeEmbedder(dim_=32, seed=2).embed(["same text"])
    assert not np.allclose(a, b)


def test_stable_str_hash_is_deterministic_across_calls():
    assert stable_str_hash("photosynthesis") == stable_str_hash("photosynthesis")
    assert stable_str_hash("photosynthesis") != stable_str_hash("black_holes")


def test_two_stage_dense_index_stage1_and_full_agree_on_an_easy_case():
    """When one document overwhelmingly shares the query's tokens, both the truncated stage-1
    pass and the full-precision pass should surface it first.
    """
    embedder = DeterministicFakeEmbedder(dim_=1024, seed=0)
    docs = [
        Document(doc_id="match", text="kubernetes pod scheduler container node"),
        Document(doc_id="other", text="sourdough starter flour water yeast"),
    ]
    index = TwoStageDenseIndex(embedder, stage1_dim=256)
    index.index(docs)
    assert index.search_stage1_only("kubernetes pod scheduler")[0].doc_id == "match"
    assert index.search_full_only("kubernetes pod scheduler")[0].doc_id == "match"
    assert index.search("kubernetes pod scheduler")[0].doc_id == "match"


def test_two_stage_dense_index_stage1_dim_is_smaller_than_full():
    embedder = DeterministicFakeEmbedder(dim_=1024, seed=0)
    index = TwoStageDenseIndex(embedder, stage1_dim=256)
    index.index([Document(doc_id="d0", text="hello world")])
    assert index._stage1_vectors.shape[1] == 256
    assert index._full_vectors.shape[1] == 1024


def test_fastembed_onnx_embedder_without_dependency_raises_clear_error():
    from localmind.retrieval.dense import fastembed_onnx_embedder

    if _import_ok("fastembed"):
        pytest.skip("fastembed is installed; the ImportError path isn't reachable")
    with pytest.raises(ImportError, match="fastembed"):
        fastembed_onnx_embedder()


# ==============================================================================================
# ColBERT
# ==============================================================================================


def test_maxsim_hand_computed():
    """Two query tokens, two doc tokens, orthonormal basis so the dot products are exact.
    q = [[1,0], [0,1]], d = [[1,0], [0.6,0.8]].
    MaxSim = max(q0.d0, q0.d1) + max(q1.d0, q1.d1) = max(1, 0.6) + max(0, 0.8) = 1 + 0.8 = 1.8.
    """
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    assert maxsim(q, d) == pytest.approx(1.8, abs=1e-6)


def test_maxsim_empty_inputs_score_zero():
    assert maxsim(np.zeros((0, 4)), np.zeros((3, 4))) == 0.0
    assert maxsim(np.zeros((3, 4)), np.zeros((0, 4))) == 0.0


def test_colbert_late_interaction_beats_pooling_on_multi_aspect_doc():
    """A document mixing two topics should still be found by a query about just one of them,
    because MaxSim lets each query token pick its own best-matching doc token instead of
    averaging the whole document into one vector first.
    """
    encoder = DeterministicFakeMultiVectorEncoder(dim_=128, seed=0)
    docs = [
        Document(
            doc_id="multi",
            text="raft consensus leader election quorum sourdough starter flour water yeast baking",
        ),
        Document(doc_id="unrelated", text="css flexbox justify content align items layout"),
    ]
    index = ColBERTIndex(encoder)
    index.index(docs)
    results = index.search("raft consensus leader election", top_k=2)
    assert results[0].doc_id == "multi"


# ==============================================================================================
# ColQwen2
# ==============================================================================================


def test_quantize_binary_shape_and_compression():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((10, 128)).astype(np.float32)
    packed = quantize_binary(vectors)
    assert packed.shape == (10, 16)  # 128 bits / 8 bits-per-byte
    assert packed.dtype == np.uint8


def test_quantize_int8_roundtrip_is_low_error():
    rng = np.random.default_rng(0)
    vectors = l2_normalize(rng.standard_normal((20, 128)).astype(np.float32))
    q, scale = quantize_int8(vectors)
    recovered = dequantize_int8(q, scale)
    assert np.abs(recovered - vectors).max() < 0.02


def test_hamming_maxsim_binary_identical_vectors_score_maximally():
    rng = np.random.default_rng(0)
    vecs = l2_normalize(rng.standard_normal((3, 64)).astype(np.float32))
    packed = quantize_binary(vecs)
    score = hamming_maxsim_binary(packed, packed, n_bits=64)
    assert score == pytest.approx(3.0, abs=1e-6)  # each token matches itself: sim=1, summed


def test_estimate_storage_matches_hand_arithmetic():
    report = estimate_storage(n_pages=10, patches_per_page=1030, dim=128)
    n_vectors = 10 * 1030
    assert report["float32_bytes"] == n_vectors * 128 * 4
    assert report["int8_bytes"] == n_vectors * 128
    assert report["binary_bytes"] == n_vectors * 16
    assert report["int8_compression_ratio"] == pytest.approx(4.0)
    assert report["binary_compression_ratio"] == pytest.approx(32.0)


def test_quantization_recall_report_binary_retains_most_recall():
    """Each query is a small perturbation of one page's own patch centroid, so that page is
    genuinely its nearest neighbor (a random, unrelated ground truth would make `recall_full`
    itself noisy and the retained-fraction ratio meaningless -- this keeps the test honest).
    """
    rng = np.random.default_rng(0)
    n_pages, n_patches, dim = 15, 20, 64
    page_vectors = {
        f"page{i}": l2_normalize(rng.standard_normal((n_patches, dim)).astype(np.float32))
        for i in range(n_pages)
    }
    query_vectors = {}
    relevant = {}
    for i in range(8):
        center = page_vectors[f"page{i}"].mean(axis=0, keepdims=True)
        noise = rng.standard_normal((3, dim)).astype(np.float32) * 0.05
        query_vectors[f"q{i}"] = l2_normalize(center + noise)
        relevant[f"q{i}"] = {f"page{i}"}
    report = quantization_recall_report(page_vectors, query_vectors, relevant, scheme="binary", k=5)
    assert report.recall_full >= 0.75  # queries are constructed to be easy
    assert 0.0 <= report.retained_fraction <= 1.05  # a little slack for a small sample
    assert report.storage["binary_compression_ratio"] == pytest.approx(32.0)


def test_colqwen_index_search_finds_matching_page():
    page_encoder = DeterministicFakePageEncoder(patch_dim_=32, n_patches=16, seed=0)
    query_encoder = DeterministicFakeMultiVectorEncoder(dim_=32, seed=0)
    pages = [Page(doc_id="report_a", page_number=1, image_path="report_a_p1")]
    index = ColQwenIndex(page_encoder, query_encoder)
    index.index(pages)
    results = index.search("anything", top_k=1)
    assert results[0].doc_id == "report_a#p1"


def test_colqwen_index_with_binary_quantization_runs_end_to_end():
    page_encoder = DeterministicFakePageEncoder(patch_dim_=32, n_patches=16, seed=0)
    query_encoder = DeterministicFakeMultiVectorEncoder(dim_=32, seed=0)
    pages = [
        Page(doc_id="doc_a", page_number=1, image_path="a1"),
        Page(doc_id="doc_b", page_number=1, image_path="b1"),
    ]
    index = ColQwenIndex(page_encoder, query_encoder, quantize="binary")
    index.index(pages)
    results = index.search("query text", top_k=2)
    assert {r.doc_id for r in results} == {"doc_a#p1", "doc_b#p1"}


def test_run_push_command_bundles_index_directory(tmp_path):
    index_dir = tmp_path / "colqwen_index"
    index_dir.mkdir()
    (index_dir / "manifest.json").write_text("{}")
    (index_dir / "page1.npy").write_bytes(b"\x00" * 16)

    archive_path = run_push_command(str(index_dir))
    assert Path(archive_path).exists()
    assert archive_path.endswith(".tar.gz")

    import tarfile

    with tarfile.open(archive_path) as tar:
        names = tar.getnames()
    assert any("manifest.json" in n for n in names)


def test_run_push_command_missing_directory_raises():
    with pytest.raises(FileNotFoundError):
        run_push_command("this/directory/does/not/exist")


def test_colqwen_cli_push_subcommand(tmp_path, capsys):
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    (index_dir / "manifest.json").write_text("{}")
    colqwen_main(["push", "--index", str(index_dir)])
    out = capsys.readouterr().out
    assert out.strip().endswith(".tar.gz")


def test_run_index_command_refuses_without_gpu():
    """Real ColQwen2 indexing is a Kaggle GPU job (§11); it must refuse to run on a CPU-only
    box rather than silently attempting an hours-long job. Not marked `gpu`: this test asserts
    the *no-GPU refusal* behavior, which is exactly the CPU-only case this dev box already is.
    """
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("This box has a GPU; the no-GPU refusal path isn't reachable")
    with pytest.raises(RuntimeError, match="GPU"):
        run_index_command("some/pdf/dir", "some/out/dir", "int8", False)


# ==============================================================================================
# Fusion
# ==============================================================================================


def test_reciprocal_rank_fusion_hand_computed():
    """arm_a ranks: [x(1), y(2)]; arm_b ranks: [y(1), x(2)]. With k=60:
    score(x) = 1/61 + 1/62 = 0.032573... ; score(y) = 1/61 + 1/62 (symmetric) -> tie.
    Use an asymmetric case instead: arm_a=[x,y], arm_b=[x,z]. score(x)=1/61+1/61=2/61;
    score(y)=1/62; score(z)=1/62. x should win outright.
    """
    arm_a = [ScoredDoc("x", 9.0), ScoredDoc("y", 8.0)]
    arm_b = [ScoredDoc("x", 5.0), ScoredDoc("z", 1.0)]
    fused = reciprocal_rank_fusion({"a": arm_a, "b": arm_b}, k=60)
    scores = {sd.doc_id: sd.score for sd in fused}
    assert scores["x"] == pytest.approx(2 / 61, abs=1e-9)
    assert scores["y"] == pytest.approx(1 / 62, abs=1e-9)
    assert scores["z"] == pytest.approx(1 / 62, abs=1e-9)
    assert fused[0].doc_id == "x"


def test_reciprocal_rank_fusion_ignores_score_magnitude():
    """RRF is rank-based: an arm with huge scores and one with tiny scores must fuse
    identically as long as the *rankings* agree, which is the entire point of using RRF
    instead of a magnitude-based combination across incomparable arm score scales.
    """
    huge = [ScoredDoc("x", 1e9), ScoredDoc("y", 1.0)]
    tiny = [ScoredDoc("x", 0.002), ScoredDoc("y", 0.001)]
    fused_huge = reciprocal_rank_fusion({"a": huge}, k=60)
    fused_tiny = reciprocal_rank_fusion({"a": tiny}, k=60)
    assert [sd.doc_id for sd in fused_huge] == [sd.doc_id for sd in fused_tiny]


def test_retrieval_config_loads_from_the_default_yaml():
    config = RetrievalConfig.from_yaml(REPO_ROOT / "configs" / "retrieval" / "default.yaml")
    assert config.fusion_rrf_k == 60
    assert config.hnsw_m == 16
    assert config.hnsw_ef_construction == 64
    assert config.dense_stage1_dim == 256


def test_retrieval_config_defaults_without_a_file():
    config = RetrievalConfig()
    assert config.bm25_k1 == 1.5
    assert config.rerank_top_k == 5


def test_ranks_from_scores_and_reciprocal_rank():
    scored = [ScoredDoc("a", 3.0), ScoredDoc("b", 2.0), ScoredDoc("c", 1.0)]
    assert ranks_from_scores(scored) == {"a": 1, "b": 2, "c": 3}
    assert reciprocal_rank(scored, {"b"}) == pytest.approx(0.5)
    assert reciprocal_rank(scored, {"zzz"}) == 0.0


def test_minmax_normalize_scales_to_unit_interval():
    normalized = minmax_normalize({"a": 10.0, "b": 20.0, "c": 30.0})
    assert normalized["a"] == 0.0
    assert normalized["b"] == 0.5
    assert normalized["c"] == 1.0


def test_minmax_normalize_constant_scores_all_map_to_one():
    normalized = minmax_normalize({"a": 5.0, "b": 5.0})
    assert normalized == {"a": 1.0, "b": 1.0}


def test_weighted_fusion_weight_zero_excludes_arm():
    arm_a = [ScoredDoc("x", 1.0), ScoredDoc("y", 0.0)]
    arm_b = [ScoredDoc("y", 1.0), ScoredDoc("x", 0.0)]
    fused = weighted_fusion({"a": arm_a, "b": arm_b}, weights={"a": 1.0, "b": 0.0})
    assert fused[0].doc_id == "x"


def test_tune_weights_recovers_the_better_arm_on_a_toy_dev_set():
    """arm_a always ranks the true relevant doc first; arm_b always ranks it last. Weight
    tuning on a 2-query dev set should discover weights favoring arm_a.
    """

    def recall_at_1(ranked, relevant):
        return 1.0 if ranked and ranked[0].doc_id in relevant else 0.0

    dev_results = {
        "q1": {
            "a": [ScoredDoc("rel", 1.0), ScoredDoc("irrel", 0.5)],
            "b": [ScoredDoc("irrel", 1.0), ScoredDoc("rel", 0.5)],
        },
        "q2": {
            "a": [ScoredDoc("rel2", 1.0), ScoredDoc("irrel2", 0.5)],
            "b": [ScoredDoc("irrel2", 1.0), ScoredDoc("rel2", 0.5)],
        },
    }
    qrels = {"q1": {"rel"}, "q2": {"rel2"}}
    best_weights, best_metric = tune_weights(dev_results, qrels, recall_at_1, steps=4)
    assert best_weights["a"] > best_weights["b"]
    assert best_metric == pytest.approx(1.0)


def test_compare_fusion_strategies_reports_a_bool_and_matching_delta_sign():
    def recall_at_1(ranked, relevant):
        return 1.0 if ranked and ranked[0].doc_id in relevant else 0.0

    dev_results = {
        "q1": {
            "a": [ScoredDoc("rel", 1.0), ScoredDoc("irrel", 0.5)],
            "b": [ScoredDoc("rel", 1.0), ScoredDoc("irrel", 0.5)],
        }
    }
    qrels = {"q1": {"rel"}}
    report = compare_fusion_strategies(
        dev_results, qrels, recall_at_1, rrf_k=DEFAULT_RRF_K, steps=2
    )
    assert isinstance(report.tuned_beat_rrf, bool)
    assert report.delta == pytest.approx(report.tuned_metric - report.rrf_metric)


# ==============================================================================================
# Reranking
# ==============================================================================================


def test_rerank_cross_encoder_reorders_by_relevance():
    candidates = [ScoredDoc("b", 0.9), ScoredDoc("a", 0.5)]  # fusion had 'b' ranked first
    doc_lookup = {"a": "cats and dogs are common household pets", "b": "quarterly earnings report"}
    result = rerank_cross_encoder(
        "household pets", candidates, doc_lookup, DeterministicFakeCrossEncoder(), top_k=2
    )
    assert result.ranked[0].doc_id == "a"
    assert result.latency_s >= 0.0


def test_rerank_listwise_reorders_by_relevance():
    candidates = [ScoredDoc("b", 0.9), ScoredDoc("a", 0.5)]
    doc_lookup = {"a": "cats and dogs are common household pets", "b": "quarterly earnings report"}
    result = rerank_listwise(
        "household pets", candidates, doc_lookup, DeterministicFakeListwiseReranker(), top_k=2
    )
    assert result.ranked[0].doc_id == "a"
    assert result.latency_s >= 0.0


def test_rerank_drops_candidates_missing_from_doc_lookup():
    candidates = [ScoredDoc("a", 1.0), ScoredDoc("missing", 0.9)]
    doc_lookup = {"a": "some text"}
    result = rerank_cross_encoder(
        "some text", candidates, doc_lookup, DeterministicFakeCrossEncoder()
    )
    assert [sd.doc_id for sd in result.ranked] == ["a"]


def test_bge_reranker_without_dependency_raises_clear_error():
    from localmind.retrieval.rerank import bge_reranker_v2_m3

    if _import_ok("fastembed"):
        pytest.skip("fastembed is installed; the ImportError path isn't reachable")
    with pytest.raises(ImportError, match="fastembed"):
        bge_reranker_v2_m3()


def test_qwen3_listwise_reranker_without_httpx_raises_clear_error(monkeypatch):
    from localmind.retrieval import rerank as rerank_module

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("no module named httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match="httpx"):
        rerank_module.qwen3_listwise_reranker()


# ==============================================================================================
# Index engineering: pgvector (SQL generation, SimpleHNSW, quantization, filtered ANN)
# ==============================================================================================


def test_create_table_sql_contains_vector_column():
    sql = create_table_sql("chunks", dim=768)
    assert "CREATE TABLE" in sql
    assert "VECTOR(768)" in sql


def test_create_hnsw_index_sql_uses_spec_mandated_defaults():
    config = HNSWConfig()
    assert config.m == 16
    assert config.ef_construction == 64
    sql = create_hnsw_index_sql("chunks", config)
    assert "USING hnsw" in sql
    assert "m = 16" in sql
    assert "ef_construction = 64" in sql


def test_set_ef_search_sql():
    assert set_ef_search_sql(HNSWConfig(ef_search=100)) == "SET hnsw.ef_search = 100"


def test_postfilter_and_prefilter_sql_differ_in_where_clause_position():
    post = search_postfiltered_sql("chunks", top_k=10)
    pre = search_prefiltered_sql("chunks", top_k=10)
    # Post-filter: WHERE comes after the ORDER BY ... LIMIT subquery.
    assert post.index("ORDER BY") < post.index("WHERE")
    # Pre-filter: WHERE comes before the ORDER BY.
    assert pre.index("WHERE") < pre.index("ORDER BY")


def test_hnsw_config_rejects_non_positive_values():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HNSWConfig(m=0)


def test_pgvector_index_connect_without_psycopg_raises_clear_error():
    if _import_ok("psycopg"):
        pytest.skip("psycopg is installed; the ImportError path isn't reachable")
    with pytest.raises(ImportError, match="psycopg"):
        PgVectorIndex(dsn="postgresql://localhost/test", table="chunks", dim=8).connect()


@pytest.mark.docker
def test_pgvector_index_live_roundtrip():
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    dsn = os.environ.get("LOCALMIND_TEST_PG_DSN")
    if not dsn:
        pytest.skip("LOCALMIND_TEST_PG_DSN not set; no live Postgres to test against")
    rng = np.random.default_rng(0)
    vectors = l2_normalize(rng.standard_normal((5, 8)).astype(np.float32))
    index = PgVectorIndex(dsn=dsn, table="localmind_test_chunks", dim=8)
    index.create()
    index.upsert([f"d{i}" for i in range(5)], vectors)
    results = index.search(vectors[0], top_k=1)
    assert results[0] == "d0"


def test_simple_hnsw_search_finds_the_nearest_neighbor_with_generous_ef():
    rng = np.random.default_rng(0)
    n, dim = 50, 16
    vectors = l2_normalize(rng.standard_normal((n, dim)).astype(np.float32))
    index = SimpleHNSW(dim=dim, m=16, ef_construction=64, seed=0)
    for i, v in enumerate(vectors):
        index.add(i, v)
    query = vectors[7]  # searching for a vector that's already in the index
    hits = index.search(query, k=1, ef_search=n)
    assert hits[0] == 7  # its own nearest neighbor is itself (distance 0)


def test_simple_hnsw_is_deterministic_given_a_seed():
    rng = np.random.default_rng(1)
    vectors = l2_normalize(rng.standard_normal((30, 8)).astype(np.float32))
    query = rng.standard_normal(8).astype(np.float32)

    def build_and_search():
        index = SimpleHNSW(dim=8, m=8, ef_construction=32, seed=5)
        for i, v in enumerate(vectors):
            index.add(i, v)
        return index.search(query, k=5, ef_search=20)

    assert build_and_search() == build_and_search()


def test_brute_force_search_matches_argsort():
    rng = np.random.default_rng(0)
    vectors = l2_normalize(rng.standard_normal((20, 4)).astype(np.float32))
    query = l2_normalize(rng.standard_normal((1, 4)).astype(np.float32))[0]
    expected = np.argsort(-(vectors @ query))[:5].tolist()
    assert brute_force_search(vectors, query, 5) == expected


def test_recall_vs_latency_curve_recall_generally_improves_with_larger_ef_search():
    rng = np.random.default_rng(3)
    n, dim = 150, 24
    vectors = l2_normalize(rng.standard_normal((n, dim)).astype(np.float32))
    queries = l2_normalize(rng.standard_normal((10, dim)).astype(np.float32))
    rows = recall_vs_latency_curve(vectors, queries, ef_search_values=[1, 10, 150], k=10)
    recalls = {row["ef_search"]: row["recall_at_k"] for row in rows}
    assert recalls[150.0] >= recalls[1.0]
    assert recalls[150.0] >= 0.95  # exhaustive-ish ef_search should recover ~ground truth
    assert all(row["p50_ms"] >= 0 and row["p95_ms"] >= row["p50_ms"] for row in rows)


def test_binary_quantization_report_retains_most_recall_and_saves_memory():
    rng = np.random.default_rng(4)
    n, dim = 200, 32
    vectors = l2_normalize(rng.standard_normal((n, dim)).astype(np.float32))
    queries = l2_normalize(rng.standard_normal((15, dim)).astype(np.float32))
    report = binary_quantization_report(vectors, queries, k=10, hamming_top_k=100)
    assert report.retained_fraction > 0.7  # loose bound; the JSON artifact has the real number
    assert report.storage["binary_compression_ratio"] == pytest.approx(32.0)


def test_filtered_ann_demo_post_filter_recall_degrades_with_selectivity():
    """The headline finding this function exists to produce, as a real number: post-filtering
    a highly selective filter should recover noticeably less recall than pre-filtering it.
    """
    rng = np.random.default_rng(5)
    n, dim = 400, 32
    vectors = l2_normalize(rng.standard_normal((n, dim)).astype(np.float32))
    queries = l2_normalize(rng.standard_normal((10, dim)).astype(np.float32))
    matches = rng.random(n) < 0.03  # a 3%-selective filter
    report = filtered_ann_demo(vectors, matches, queries, k=10)
    assert report.pre_filter_recall > report.post_filter_recall


def test_qdrant_hnsw_config_mirrors_pgvector_defaults():
    config = QdrantHNSWConfig()
    assert config.m == 16
    assert config.ef_construct == 64


def test_qdrant_index_connect_without_client_raises_clear_error():
    if _import_ok("qdrant_client"):
        pytest.skip("qdrant-client is installed; the ImportError path isn't reachable")
    with pytest.raises(ImportError, match="qdrant"):
        QdrantIndex(url="http://localhost:6333", collection="chunks", dim=8).connect()


@pytest.mark.docker
def test_qdrant_index_live_roundtrip():
    pytest.importorskip("qdrant_client")
    url = os.environ.get("LOCALMIND_TEST_QDRANT_URL")
    if not url:
        pytest.skip("LOCALMIND_TEST_QDRANT_URL not set; no live Qdrant to test against")
    rng = np.random.default_rng(0)
    vectors = l2_normalize(rng.standard_normal((5, 8)).astype(np.float32))
    index = QdrantIndex(url=url, collection="localmind_test", dim=8)
    index.create()
    index.upsert([f"d{i}" for i in range(5)], vectors)
    results = index.search(vectors[0], top_k=1)
    assert results[0] == "d0"


# ==============================================================================================
# Helpers shared by the ImportError-path tests above
# ==============================================================================================


def _import_ok(module_name: str) -> bool:
    import importlib

    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# ==============================================================================================
# Benchmark: synthetic corpus, spec's 9-row table, RRF-vs-tuned-fusion comparison
# ==============================================================================================

_TOPICS = [
    {
        "name": "photosynthesis",
        "canonical": ["photosynthesis", "chlorophyll", "sunlight", "glucose"],
        "paraphrase": ["lightharvesting", "sugarproduction", "greenpigment"],
        "rare_id": "proc7731",
    },
    {
        "name": "black holes",
        "canonical": ["blackhole", "eventhorizon", "gravity", "singularity"],
        "paraphrase": ["collapsedstar", "spacetimewell", "darkobject"],
        "rare_id": "obj2290",
    },
    {
        "name": "http caching",
        "canonical": ["httpcache", "etag", "cachecontrol", "cdn"],
        "paraphrase": ["responsestorage", "edgecaching", "browsercache"],
        "rare_id": "rfc9111",
    },
    {
        "name": "rust ownership",
        "canonical": ["rustownership", "borrowchecker", "lifetime", "move"],
        "paraphrase": ["memorysafety", "compiletimecheck", "aliasrule"],
        "rare_id": "rfc0021",
    },
    {
        "name": "kubernetes pods",
        "canonical": ["kubernetespod", "container", "scheduler", "node"],
        "paraphrase": ["workloadunit", "orchestration", "clusterscheduling"],
        "rare_id": "k8s4471",
    },
    {
        "name": "vitamin d",
        "canonical": ["vitamind", "sunlightskin", "calcium", "bones"],
        "paraphrase": ["sunshinevitamin", "bonehealth", "seruminoids"],
        "rare_id": "nih3302",
    },
    {
        "name": "roman aqueducts",
        "canonical": ["romanaqueduct", "gravityflow", "arches", "waterway"],
        "paraphrase": ["ancientpipeline", "stonechannel", "watersupplyroute"],
        "rare_id": "arch1188",
    },
    {
        "name": "chess openings",
        "canonical": ["chessopening", "sicilian", "kingpawn", "tempo"],
        "paraphrase": ["earlygamestrategy", "firstmoves", "openingtheory"],
        "rare_id": "ecob20",
    },
    {
        "name": "raft consensus",
        "canonical": ["raftconsensus", "leaderelection", "logreplication", "quorum"],
        "paraphrase": ["distributedagreement", "clustersync", "termvoting"],
        "rare_id": "paper2014",
    },
    {
        "name": "css flexbox",
        "canonical": ["cssflexbox", "flexcontainer", "justifycontent", "alignitems"],
        "paraphrase": ["layoutmodel", "onedimensionallayout", "flexiblebox"],
        "rare_id": "w3c4409",
    },
]

_GENERIC_TEMPLATES = [
    "the process converts one form of energy into another and drives the overall system",
    "the structure supports the surrounding components and keeps everything working",
    "the mechanism stores information so it can be retrieved again later",
    "the protocol coordinates multiple independent parts toward one shared outcome",
    "the strategy focuses effort early so later stages go more smoothly",
    "the nutrient supports a key bodily function over the long term",
    "the model distributes available resources among several competing parts",
    "the technique reduces repeated work by remembering previous results",
    "the approach lets independent participants settle on one agreed answer",
    "the layout arranges elements along a single shared direction",
]

_FILLER_VOCAB = [
    "market",
    "weather",
    "travel",
    "garden",
    "music",
    "history",
    "painting",
    "recipe",
    "budget",
    "weekend",
    "traffic",
    "holiday",
    "festival",
    "novel",
    "podcast",
    "furniture",
    "bicycle",
    "camera",
    "coffee",
    "hiking",
]


def _build_corpus_and_queries():
    """Deterministic synthetic corpus (content, not encoders, is seed-independent so all 3
    benchmark seeds compare on exactly the same documents/queries/qrels -- only the fake
    encoders' random token vectors vary by seed).

    Returns (documents, contextual_documents, doc_lookup, contextual_doc_lookup, queries,
    qrels), where `contextual_documents` is the same corpus with every "ambiguous" chunk given
    a short prepended context ("Topic: X.") -- used only by the "+ contextual chunks" row.
    """
    rng = np.random.default_rng(1234)
    documents: list[Document] = []
    contextual_documents: list[Document] = []
    qrels: dict[str, set[str]] = {}
    queries: list[tuple[str, str]] = []

    for i, topic in enumerate(_TOPICS):
        slug = topic["name"].replace(" ", "_")
        canon_ids = [f"{slug}_canon{j}" for j in range(2)]
        para_id = f"{slug}_para"
        ambiguous_id = f"{slug}_ambiguous"

        for cid in canon_ids:
            text = (
                f"{topic['name']} is described here. Key terms: {' '.join(topic['canonical'])}. "
                f"Reference identifier {topic['rare_id']} appears in this document. "
                f"This is the canonical explanation of {topic['name']}."
            )
            documents.append(Document(doc_id=cid, text=text))
            contextual_documents.append(Document(doc_id=cid, text=text))

        para_text = (
            f"An alternate explanation uses different words: {' '.join(topic['paraphrase'])}. "
            f"It covers the same underlying idea without repeating the exact terminology."
        )
        documents.append(Document(doc_id=para_id, text=para_text))
        contextual_documents.append(Document(doc_id=para_id, text=para_text))

        ambiguous_text = _GENERIC_TEMPLATES[i % len(_GENERIC_TEMPLATES)]
        documents.append(Document(doc_id=ambiguous_id, text=ambiguous_text))
        contextual_documents.append(
            Document(doc_id=ambiguous_id, text=f"Topic: {topic['name']}. {ambiguous_text}")
        )

        relevant = set(canon_ids) | {para_id, ambiguous_id}
        exact_qid = f"{slug}_exact"
        para_qid = f"{slug}_paraphrase"
        id_qid = f"{slug}_id"
        queries.append((exact_qid, f"what is {topic['name']} and its {topic['canonical'][0]}"))
        queries.append((para_qid, f"tell me about {' '.join(topic['paraphrase'][:2])}"))
        queries.append((id_qid, f"lookup reference {topic['rare_id']}"))
        qrels[exact_qid] = relevant
        qrels[para_qid] = relevant
        qrels[id_qid] = {canon_ids[0], canon_ids[1]}

    # Multi-aspect documents: each mixes two topics, relevant to both topics' exact/paraphrase
    # queries -- the case ColBERT's per-token MaxSim is expected to handle better than a single
    # pooled dense vector, since the doc's "signal" for either topic alone is diluted by pooling.
    pair_offsets = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (0, 3), (1, 4), (2, 9)]
    for idx, (a, b) in enumerate(pair_offsets):
        topic_a, topic_b = _TOPICS[a], _TOPICS[b]
        doc_id = f"multiaspect_{idx}"
        text = (
            f"This combined note touches on two subjects. First, {' '.join(topic_a['canonical'][:2])}. "
            f"Second, {' '.join(topic_b['canonical'][:2])}. Both aspects matter for this write-up."
        )
        documents.append(Document(doc_id=doc_id, text=text))
        contextual_documents.append(Document(doc_id=doc_id, text=text))
        for topic in (topic_a, topic_b):
            slug = topic["name"].replace(" ", "_")
            qrels[f"{slug}_exact"].add(doc_id)
            qrels[f"{slug}_paraphrase"].add(doc_id)

    # Filler/noise documents: never relevant to any query, pad the corpus toward
    # "nontrivial" size and stress precision (every arm has to actually rank them low).
    for i in range(40):
        words = rng.choice(_FILLER_VOCAB, size=12, replace=True)
        text = " ".join(words.tolist())
        documents.append(Document(doc_id=f"noise_{i}", text=text))
        contextual_documents.append(Document(doc_id=f"noise_{i}", text=text))

    doc_lookup = {d.doc_id: d.text for d in documents}
    contextual_doc_lookup = {d.doc_id: d.text for d in contextual_documents}
    return documents, contextual_documents, doc_lookup, contextual_doc_lookup, queries, qrels


def _synonym_normalize(text: str, synonym_to_canonical: dict[str, str]) -> str:
    tokens = tokenize(text)
    return " ".join(synonym_to_canonical.get(t, t) for t in tokens)


class _SynonymEmbedder:
    """Wraps `DeterministicFakeEmbedder` with a paraphrase->canonical token normalization, so
    the fake embedder's cosine similarity reflects the corpus's designed synonym relationships
    -- standing in for what a real embedding model learns from data. Benchmark-only; not part
    of the production `dense.py` API.
    """

    def __init__(self, base: DeterministicFakeEmbedder, synonyms: dict[str, str]) -> None:
        self.base = base
        self.synonyms = synonyms

    @property
    def dim(self) -> int:
        return self.base.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.base.embed([_synonym_normalize(t, self.synonyms) for t in texts])


class _SynonymMultiVectorEncoder:
    """Same idea as `_SynonymEmbedder`, for the ColBERT arm's per-token encoder."""

    def __init__(self, base: DeterministicFakeMultiVectorEncoder, synonyms: dict[str, str]) -> None:
        self.base = base
        self.synonyms = synonyms

    @property
    def dim(self) -> int:
        return self.base.dim

    def encode(self, text: str) -> np.ndarray:
        return self.base.encode(_synonym_normalize(text, self.synonyms))


def _synonym_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for topic in _TOPICS:
        for para_word, canon_word in zip(topic["paraphrase"], topic["canonical"], strict=False):
            mapping[para_word] = canon_word
    return mapping


def _splade_expansions() -> dict[str, list[str]]:
    expansions: dict[str, list[str]] = {}
    for topic in _TOPICS:
        for para_word, canon_word in zip(topic["paraphrase"], topic["canonical"], strict=False):
            expansions.setdefault(para_word, []).append(canon_word)
    return expansions


def _build_arms(documents: list[Document], seed: int):
    from localmind.retrieval.splade import DeterministicExpansionEncoder, SpladeIndex

    synonyms = _synonym_map()
    bm25 = BM25Index()
    bm25.index(documents)

    dense = TwoStageDenseIndex(
        _SynonymEmbedder(DeterministicFakeEmbedder(dim_=1024, seed=seed), synonyms), stage1_dim=256
    )
    dense.index(documents)

    splade = SpladeIndex(DeterministicExpansionEncoder(expansions=_splade_expansions()))
    splade.index(documents)

    colbert = ColBERTIndex(
        _SynonymMultiVectorEncoder(
            DeterministicFakeMultiVectorEncoder(dim_=128, seed=seed), synonyms
        )
    )
    colbert.index(documents)

    return {"bm25": bm25, "dense": dense, "splade": splade, "colbert": colbert}


def _extend_with_tail(head: list[str], fused_order: list[str], total: int) -> list[str]:
    seen = set(head)
    out = list(head)
    for doc_id in fused_order:
        if len(out) >= total:
            break
        if doc_id not in seen:
            out.append(doc_id)
            seen.add(doc_id)
    return out[:total]


def _recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def _ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked_ids[:k]) if d in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def _mrr(ranked_ids: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked_ids):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _bootstrap_mean_ci(values: list[float], n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
    }


def _bootstrap_percentile_ci(
    values: list[float], pct: float, n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    point = float(np.percentile(arr, pct))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = np.percentile(arr[idx], pct, axis=1)
    return {
        "mean": point,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
    }


_SEEDS = [0, 1, 2]
_K5, _K10, _K20 = 5, 10, 20


def _run_row(row_name: str, arms: dict, doc_lookup: dict[str, str], query_text: str, cross_encoder):
    import time

    bm25, dense, splade, colbert = arms["bm25"], arms["dense"], arms["splade"], arms["colbert"]

    if row_name == "BM25":
        start = time.perf_counter()
        ranked = [sd.doc_id for sd in bm25.search(query_text, top_k=_K20)]
        return ranked, time.perf_counter() - start

    if row_name == "Dense (256d MRL)":
        start = time.perf_counter()
        ranked = [sd.doc_id for sd in dense.search_stage1_only(query_text, top_k=_K20)]
        return ranked, time.perf_counter() - start

    if row_name == "Dense (full)":
        start = time.perf_counter()
        ranked = [sd.doc_id for sd in dense.search_full_only(query_text, top_k=_K20)]
        return ranked, time.perf_counter() - start

    if row_name == "SPLADE":
        start = time.perf_counter()
        ranked = [sd.doc_id for sd in splade.search(query_text, top_k=_K20)]
        return ranked, time.perf_counter() - start

    if row_name == "ColBERT":
        start = time.perf_counter()
        ranked = [sd.doc_id for sd in colbert.search(query_text, top_k=_K20)]
        return ranked, time.perf_counter() - start

    if row_name == "BM25+Dense (RRF)":
        start = time.perf_counter()
        arm_results = {
            "bm25": bm25.search(query_text, top_k=50),
            "dense": dense.search_full_only(query_text, top_k=50),
        }
        fused = reciprocal_rank_fusion(arm_results, k=DEFAULT_RRF_K)
        ranked = [sd.doc_id for sd in fused[:_K20]]
        return ranked, time.perf_counter() - start

    if row_name in ("4-arm RRF", "4-arm RRF + cross-encoder", "+ contextual chunks"):
        start = time.perf_counter()
        arm_results = {
            "bm25": bm25.search(query_text, top_k=50),
            "dense": dense.search_full_only(query_text, top_k=50),
            "splade": splade.search(query_text, top_k=50),
            "colbert": colbert.search(query_text, top_k=50),
        }
        fused = reciprocal_rank_fusion(arm_results, k=DEFAULT_RRF_K)
        if row_name == "4-arm RRF":
            ranked = [sd.doc_id for sd in fused[:_K20]]
            return ranked, time.perf_counter() - start

        top50 = fused[:50]
        rerank_result = rerank_cross_encoder(query_text, top50, doc_lookup, cross_encoder, top_k=5)
        head = [sd.doc_id for sd in rerank_result.ranked]
        fused_order = [sd.doc_id for sd in fused]
        ranked = _extend_with_tail(head, fused_order, total=_K20)
        return ranked, time.perf_counter() - start

    raise ValueError(f"Unknown row: {row_name}")


_ROW_NAMES = [
    "BM25",
    "Dense (256d MRL)",
    "Dense (full)",
    "SPLADE",
    "ColBERT",
    "BM25+Dense (RRF)",
    "4-arm RRF",
    "4-arm RRF + cross-encoder",
    "+ contextual chunks",
]


@pytest.mark.slow
def test_benchmark_table_and_fusion_comparison():
    """Builds the spec's §12 benchmark table on a synthetic-but-nontrivial local corpus (10
    topics x {2 canonical, 1 paraphrase, 1 ambiguous} + 8 multi-aspect + 40 filler docs = 98
    documents; 30 queries), 3 seeds, bootstrap 95% CIs, and a separate RRF-vs-tuned-fusion
    comparison on a held-out dev/test split. Writes `artifacts/benchmarks/retrieval.json`.
    """
    documents, contextual_documents, doc_lookup, contextual_doc_lookup, queries, qrels = (
        _build_corpus_and_queries()
    )
    assert len(documents) >= 80  # "nontrivial" per the spec
    assert len(queries) >= 20

    cross_encoder = DeterministicFakeCrossEncoder()

    # {row_name: {metric_name: [values across (seed, query)]}}
    metric_samples: dict[str, dict[str, list[float]]] = {
        row: {"recall_at_5": [], "recall_at_20": [], "ndcg_at_10": [], "mrr": [], "latency_ms": []}
        for row in _ROW_NAMES
    }

    for seed in _SEEDS:
        arms = _build_arms(documents, seed)
        contextual_arms = _build_arms(contextual_documents, seed)
        for query_id, query_text in queries:
            relevant = qrels[query_id]
            for row_name in _ROW_NAMES:
                if row_name == "+ contextual chunks":
                    ranked, latency_s = _run_row(
                        "4-arm RRF + cross-encoder",
                        contextual_arms,
                        contextual_doc_lookup,
                        query_text,
                        cross_encoder,
                    )
                else:
                    ranked, latency_s = _run_row(
                        row_name, arms, doc_lookup, query_text, cross_encoder
                    )
                samples = metric_samples[row_name]
                samples["recall_at_5"].append(_recall_at_k(ranked, relevant, _K5))
                samples["recall_at_20"].append(_recall_at_k(ranked, relevant, _K20))
                samples["ndcg_at_10"].append(_ndcg_at_k(ranked, relevant, _K10))
                samples["mrr"].append(_mrr(ranked, relevant))
                samples["latency_ms"].append(latency_s * 1000)

    rows_out = []
    for row_name in _ROW_NAMES:
        samples = metric_samples[row_name]
        rows_out.append(
            {
                "config": row_name,
                "recall_at_5": _bootstrap_mean_ci(samples["recall_at_5"]),
                "recall_at_20": _bootstrap_mean_ci(samples["recall_at_20"]),
                "ndcg_at_10": _bootstrap_mean_ci(samples["ndcg_at_10"]),
                "mrr": _bootstrap_mean_ci(samples["mrr"]),
                "p50_ms": _bootstrap_percentile_ci(samples["latency_ms"], 50),
                "p95_ms": _bootstrap_percentile_ci(samples["latency_ms"], 95),
            }
        )

    # Sanity: every metric is a valid fraction/CI-bounded, p95 >= p50, the reranked rows exist.
    for row in rows_out:
        for key in ("recall_at_5", "recall_at_20", "ndcg_at_10", "mrr"):
            assert 0.0 <= row[key]["mean"] <= 1.0 + 1e-9
        assert row["p95_ms"]["mean"] >= row["p50_ms"]["mean"] - 1e-6

    # --- Reranking's added latency, reported explicitly (spec: "quality gain AND added p95") ---
    base_p95 = next(r for r in rows_out if r["config"] == "4-arm RRF")["p95_ms"]["mean"]
    reranked_p95 = next(r for r in rows_out if r["config"] == "4-arm RRF + cross-encoder")[
        "p95_ms"
    ]["mean"]
    reranking_added_p95_ms = reranked_p95 - base_p95
    base_ndcg = next(r for r in rows_out if r["config"] == "4-arm RRF")["ndcg_at_10"]["mean"]
    reranked_ndcg = next(r for r in rows_out if r["config"] == "4-arm RRF + cross-encoder")[
        "ndcg_at_10"
    ]["mean"]

    # --- Contextual chunking's effect on the (deliberately hard) ambiguous documents ---
    without_ctx = next(r for r in rows_out if r["config"] == "4-arm RRF + cross-encoder")[
        "recall_at_20"
    ]["mean"]
    with_ctx = next(r for r in rows_out if r["config"] == "+ contextual chunks")["recall_at_20"][
        "mean"
    ]

    # --- RRF vs. tuned weighted fusion on a held-out dev/test split (separate from the table) ---
    query_ids_sorted = sorted(qid for qid, _ in queries)
    dev_ids = set(query_ids_sorted[0::2])  # even-indexed queries -> dev (tune on this)
    test_ids = set(query_ids_sorted[1::2])  # odd-indexed queries -> test (report on this)

    arms0 = _build_arms(documents, seed=0)
    dev_arm_results = {}
    test_arm_results = {}
    for query_id, query_text in queries:
        arm_results = {
            "bm25": arms0["bm25"].search(query_text, top_k=50),
            "dense": arms0["dense"].search_full_only(query_text, top_k=50),
            "splade": arms0["splade"].search(query_text, top_k=50),
            "colbert": arms0["colbert"].search(query_text, top_k=50),
        }
        (dev_arm_results if query_id in dev_ids else test_arm_results)[query_id] = arm_results

    def ndcg_metric(ranked: list[ScoredDoc], relevant: set[str]) -> float:
        return _ndcg_at_k([sd.doc_id for sd in ranked], relevant, _K10)

    test_qrels = {qid: qrels[qid] for qid in test_ids}
    fusion_report = compare_fusion_strategies(
        {qid: dev_arm_results[qid] for qid in dev_ids if qid in dev_arm_results},
        {qid: qrels[qid] for qid in dev_ids},
        ndcg_metric,
        rrf_k=DEFAULT_RRF_K,
        steps=4,
    )
    # Also evaluate the dev-tuned weights on the held-out test split, which is the number that
    # actually matters (dev-set nDCG is expected to favor tuning; it's fit on that data).
    test_rrf = [
        ndcg_metric(reciprocal_rank_fusion(test_arm_results[qid], k=DEFAULT_RRF_K), test_qrels[qid])
        for qid in test_ids
    ]
    test_tuned = [
        ndcg_metric(
            weighted_fusion(test_arm_results[qid], fusion_report.best_weights), test_qrels[qid]
        )
        for qid in test_ids
    ]
    test_rrf_mean = sum(test_rrf) / len(test_rrf)
    test_tuned_mean = sum(test_tuned) / len(test_tuned)

    # --- Index engineering: real local measurements on this corpus's own embeddings ---------
    # Uses the same synthetic corpus (via a plain, un-normalized DeterministicFakeEmbedder --
    # not the synonym-wrapped one above) so these ANN numbers are grounded in this repo's own
    # code and data, not disconnected random vectors, while pgvector.py's SimpleHNSW stands in
    # for real pgvector (not running in this environment; see that module's docstring).
    index_embedder = DeterministicFakeEmbedder(dim_=128, seed=0)
    doc_vectors = index_embedder.embed([d.text for d in documents])
    query_vectors_ann = index_embedder.embed([q for _, q in queries])

    ann_config = HNSWConfig(m=16, ef_construction=64, ef_search=40)
    curve = recall_vs_latency_curve(
        doc_vectors,
        query_vectors_ann,
        ef_search_values=[1, 5, 10, 20, 40, 80, len(documents)],
        k=10,
        config=ann_config,
    )
    bq_report = binary_quantization_report(doc_vectors, query_vectors_ann, k=10, hamming_top_k=40)

    filter_rng = np.random.default_rng(99)
    filtered_reports = []
    for target_selectivity in (0.5, 0.15, 0.03):
        matches = filter_rng.random(len(documents)) < target_selectivity
        filtered_reports.append(
            filtered_ann_demo(doc_vectors, matches, query_vectors_ann, k=10, config=ann_config)
        )

    hardware = f"{os.cpu_count()}-core CPU, no GPU ({platform.system()} {platform.release()})"
    artifact = {
        "name": "retrieval",
        "hardware": hardware,
        "seeds": _SEEDS,
        "ci": "bootstrap95",
        "corpus": {
            "type": "synthetic",
            "n_documents": len(documents),
            "n_queries": len(queries),
            "notes": (
                "Locally-built synthetic corpus (10 topics x canonical/paraphrase/ambiguous "
                "docs + 8 multi-aspect docs + 40 filler docs), not a public IR benchmark. "
                "Numbers below are real measurements of this repo's code on that corpus, not "
                "citations of a standard dataset's leaderboard."
            ),
        },
        "rows": rows_out,
        "reranking": {
            "cross_encoder": "DeterministicFakeCrossEncoder (token-overlap heuristic; a real "
            "deployment plugs in bge-reranker-v2-m3 via rerank.bge_reranker_v2_m3, "
            "unavailable here: no network for model weights)",
            "ndcg_at_10_gain": reranked_ndcg - base_ndcg,
            "added_p95_ms": reranking_added_p95_ms,
        },
        "contextual_chunking": {
            "recall_at_20_without_context": without_ctx,
            "recall_at_20_with_context": with_ctx,
            "delta": with_ctx - without_ctx,
        },
        "index_engineering": {
            "engine": (
                "SimpleHNSW (from-scratch, index/pgvector.py) standing in for real pgvector, "
                "which is not running in this environment; PgVectorIndex issues the equivalent "
                "SQL/DDL against a live Postgres+pgvector instance (see @pytest.mark.docker "
                "tests) and was not exercised here."
            ),
            "hnsw_config": {"m": ann_config.m, "ef_construction": ann_config.ef_construction},
            "recall_vs_latency_curve": curve,
            "binary_quantization": {
                "recall_at_10_full_precision": bq_report.exact_recall_at_k,
                "recall_at_10_binary_rescored": bq_report.recall_at_k,
                "retained_fraction": bq_report.retained_fraction,
                "hamming_top_k": bq_report.hamming_top_k,
                "p50_ms": bq_report.p50_ms,
                "p95_ms": bq_report.p95_ms,
                "storage": bq_report.storage,
            },
            "filtered_ann": [
                {
                    "selectivity": r.selectivity,
                    "post_filter_recall": r.post_filter_recall,
                    "pre_filter_recall": r.pre_filter_recall,
                    "post_filter_latency_ms": r.post_filter_latency_ms,
                    "pre_filter_latency_ms": r.pre_filter_latency_ms,
                    "unfiltered_ann_latency_ms": r.unfiltered_ann_latency_ms,
                }
                for r in filtered_reports
            ],
        },
        "fusion_comparison": {
            "dev_queries": len(dev_ids),
            "test_queries": len(test_ids),
            "metric": "nDCG@10",
            "dev_rrf": fusion_report.rrf_metric,
            "dev_tuned": fusion_report.tuned_metric,
            "dev_tuned_beat_rrf": fusion_report.tuned_beat_rrf,
            "best_weights": fusion_report.best_weights,
            "test_rrf": test_rrf_mean,
            "test_tuned": test_tuned_mean,
            "test_tuned_beat_rrf": test_tuned_mean > test_rrf_mean,
            "note": (
                "Weights tuned on the dev split; test_* is the held-out number that matters. "
                "See docs/decisions/0006-rrf-over-tuned-fusion.md."
            ),
        },
        "notes": [
            "BM25 row uses BM25Index, the from-scratch implementation (see bm25.py's module "
            "docstring / this task's ADR note): bm25s is unavailable offline, and using the "
            "own implementation keeps every row reproducible without the `rag` extra.",
            "Dense/SPLADE/ColBERT arms use deterministic fake encoders (no real model weights "
            "are downloadable in this environment): DeterministicFakeEmbedder, "
            "DeterministicExpansionEncoder, DeterministicFakeMultiVectorEncoder, wrapped with "
            "a benchmark-only synonym normalizer so paraphrase/canonical term pairs are "
            "treated as similar, standing in for what a trained model would learn from data.",
            "Cross-encoder reranker is DeterministicFakeCrossEncoder (token-overlap), not a "
            "real bge-reranker-v2-m3 checkpoint, for the same offline reason.",
        ],
    }

    out_path = REPO_ROOT / "artifacts" / "benchmarks" / "retrieval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    assert out_path.exists()
    written_back = json.loads(out_path.read_text())
    assert written_back["name"] == "retrieval"
    assert len(written_back["rows"]) == len(_ROW_NAMES)
