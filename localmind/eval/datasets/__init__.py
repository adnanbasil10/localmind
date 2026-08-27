"""Golden datasets: schema, hash pinning, and the committed sample sets.

Files in this directory:

* ``golden_sample_v1.jsonl``  -- a small hand-written golden set that exercises
  every ``category`` value, including ``out-of-domain`` and
  ``adversarial-injection``. It is a *sample*, not the release set: §14 calls
  for 150-300 hand-verified questions over the real corpus, which
  ``generate_golden.py`` produces candidates for.
* ``corpus_sample_v1.jsonl`` -- the tiny corpus those questions are grounded
  in, so the retrieval gate has something real to run from day one.
* ``judge_labels_v1.jsonl``  -- 100 labelled pairwise comparisons used to
  calibrate the LLM judge (§14 makes calibration mandatory).
* ``manifest.json``          -- SHA-256 pins for all three.
"""

from localmind.eval.datasets.schema import (
    CATEGORIES,
    DATASETS_DIR,
    DIFFICULTIES,
    SAMPLE_CORPUS_PATH,
    SAMPLE_GOLDEN_PATH,
    SAMPLE_JUDGE_LABELS_PATH,
    Category,
    CorpusChunk,
    DatasetManifest,
    Difficulty,
    GoldenDataset,
    GoldenQuestion,
    JudgeLabel,
    ManifestEntry,
    canonical_json,
    hash_questions,
    load_corpus,
    load_golden,
    load_judge_labels,
    load_manifest,
    write_jsonl,
)

__all__ = [
    "CATEGORIES",
    "DATASETS_DIR",
    "DIFFICULTIES",
    "SAMPLE_CORPUS_PATH",
    "SAMPLE_GOLDEN_PATH",
    "SAMPLE_JUDGE_LABELS_PATH",
    "Category",
    "CorpusChunk",
    "DatasetManifest",
    "Difficulty",
    "GoldenDataset",
    "GoldenQuestion",
    "JudgeLabel",
    "ManifestEntry",
    "canonical_json",
    "hash_questions",
    "load_corpus",
    "load_golden",
    "load_judge_labels",
    "load_manifest",
    "write_jsonl",
]
