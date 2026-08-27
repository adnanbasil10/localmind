"""Golden-dataset schema, hash pinning, and loaders.

The schema is fixed verbatim by ``implementation.md`` §14::

    id, question, expected_answer, expected_doc_ids, expected_chunk_ids,
    difficulty {easy|medium|hard}, category {factoid|multi-hop|aggregation|
      table|figure|out-of-domain|adversarial-injection}, requires_tools[]

Datasets are *versioned and hash-pinned*: ``manifest.json`` records a SHA-256
over the canonical serialisation of every question, so a silent edit to the
golden set is a test failure rather than a mystery metric shift.

Only ``pydantic`` and the stdlib are imported here -- no numpy, no yaml.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

Difficulty = Literal["easy", "medium", "hard"]
Category = Literal[
    "factoid",
    "multi-hop",
    "aggregation",
    "table",
    "figure",
    "out-of-domain",
    "adversarial-injection",
]

DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")
CATEGORIES: tuple[Category, ...] = (
    "factoid",
    "multi-hop",
    "aggregation",
    "table",
    "figure",
    "out-of-domain",
    "adversarial-injection",
)

DATASETS_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = DATASETS_DIR / "manifest.json"
SAMPLE_GOLDEN_PATH = DATASETS_DIR / "golden_sample_v1.jsonl"
SAMPLE_CORPUS_PATH = DATASETS_DIR / "corpus_sample_v1.jsonl"
SAMPLE_JUDGE_LABELS_PATH = DATASETS_DIR / "judge_labels_v1.jsonl"

_ID_RE = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
_TOOL_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_RELEASE_MIN = 150
_RELEASE_MAX = 300


# --------------------------------------------------------------------------- #
# Question
# --------------------------------------------------------------------------- #
class GoldenQuestion(BaseModel):
    """One hand-verified evaluation question.

    ``expected_doc_ids == []`` is the harness-wide convention for *unanswerable
    from the corpus*: the system is expected to refuse.  §14 is explicit that
    "a system that never refuses is broken, and refusal accuracy is a metric".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str
    expected_answer: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    difficulty: Difficulty
    category: Category
    requires_tools: list[str] = Field(default_factory=list)

    # Provenance and harness detail -- not part of the §14 metric schema, but a
    # golden set that cannot say whether a human looked at it is not a golden
    # set, and injection resistance needs a canary to look for.
    verified: bool = True
    source: str = "hand-written"
    notes: str = ""
    injection_canary: str = ""

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError(f"id must be lowercase kebab/snake case, got {v!r}")
        return v

    @field_validator("question", "expected_answer")
    @classmethod
    def _check_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @field_validator("requires_tools")
    @classmethod
    def _check_tools(cls, v: list[str]) -> list[str]:
        for tool in v:
            if not _TOOL_RE.match(tool):
                raise ValueError(f"tool names must be snake_case identifiers, got {tool!r}")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> GoldenQuestion:
        docs = set(self.expected_doc_ids)
        if len(docs) != len(self.expected_doc_ids):
            raise ValueError(f"{self.id}: duplicate expected_doc_ids")
        if len(set(self.expected_chunk_ids)) != len(self.expected_chunk_ids):
            raise ValueError(f"{self.id}: duplicate expected_chunk_ids")
        if self.category == "out-of-domain" and docs:
            raise ValueError(
                f"{self.id}: out-of-domain questions must have no expected_doc_ids "
                "(they are unanswerable from the corpus by construction)"
            )
        if not docs and self.expected_chunk_ids:
            raise ValueError(
                f"{self.id}: expected_chunk_ids without expected_doc_ids is contradictory"
            )
        for chunk_id in self.expected_chunk_ids:
            if doc_of_chunk(chunk_id) not in docs:
                raise ValueError(
                    f"{self.id}: chunk {chunk_id!r} does not belong to any expected doc "
                    f"(chunk ids must be '<doc_id>#<n>')"
                )
        if self.injection_canary and self.category != "adversarial-injection":
            raise ValueError(
                f"{self.id}: injection_canary is only meaningful for adversarial-injection"
            )
        if self.category == "adversarial-injection" and not self.injection_canary:
            raise ValueError(
                f"{self.id}: adversarial-injection questions need an injection_canary so "
                "injection resistance can be measured"
            )
        return self

    @property
    def should_refuse(self) -> bool:
        """True when the corpus cannot support an answer and refusal is correct."""
        return not self.expected_doc_ids

    @property
    def is_adversarial(self) -> bool:
        return self.category == "adversarial-injection"

    def canonical(self) -> str:
        return canonical_json(self)


def doc_of_chunk(chunk_id: str) -> str:
    """``'faq-billing#3' -> 'faq-billing'``.  Chunk ids are doc-scoped by contract."""
    return chunk_id.split("#", 1)[0]


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
class CorpusChunk(BaseModel):
    """A retrievable unit.  The eval harness never chunks -- ingestion does."""

    model_config = ConfigDict(frozen=True, extra="allow")

    chunk_id: str
    doc_id: str
    text: str
    title: str = ""
    modality: Literal["text", "table", "figure"] = "text"

    @model_validator(mode="after")
    def _check_chunk_belongs(self) -> CorpusChunk:
        if doc_of_chunk(self.chunk_id) != self.doc_id:
            raise ValueError(
                f"chunk_id {self.chunk_id!r} must be '{self.doc_id}#<n>' to match doc_id"
            )
        return self


# --------------------------------------------------------------------------- #
# Judge calibration labels
# --------------------------------------------------------------------------- #
class JudgeLabel(BaseModel):
    """One hand-labelled pairwise preference used to calibrate the LLM judge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str
    context: str
    answer_a: str
    answer_b: str
    human_label: Literal["A", "B", "tie"]
    # Per-answer groundedness labels; these calibrate the *binary* entailment
    # call that faithfulness depends on, separately from the pairwise ranking.
    a_supported: bool | None = None
    b_supported: bool | None = None
    annotator: str = "hand"
    notes: str = ""


# --------------------------------------------------------------------------- #
# Dataset + hash pinning
# --------------------------------------------------------------------------- #
def canonical_json(model: BaseModel) -> str:
    """Stable, key-sorted, whitespace-free JSON -- the unit of hashing."""
    return json.dumps(
        model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def hash_questions(questions: Iterable[GoldenQuestion]) -> str:
    """SHA-256 over id-sorted canonical questions.  Order-insensitive by design."""
    lines = sorted(canonical_json(q) for q in questions)
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class GoldenDataset(BaseModel):
    """A versioned, hash-pinned set of golden questions."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    kind: Literal["release", "sample", "candidates"] = "sample"
    corpus_id: str = ""
    questions: list[GoldenQuestion] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _check_unique_ids(self) -> GoldenDataset:
        ids = [q.id for q in self.questions]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate question ids: {sorted(dupes)}")
        return self

    # -- properties -------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Iterator[GoldenQuestion]:  # type: ignore[override]
        return iter(self.questions)

    @property
    def content_hash(self) -> str:
        return hash_questions(self.questions)

    @property
    def categories(self) -> set[str]:
        return {q.category for q in self.questions}

    @property
    def answerable(self) -> list[GoldenQuestion]:
        return [q for q in self.questions if not q.should_refuse]

    @property
    def unanswerable(self) -> list[GoldenQuestion]:
        return [q for q in self.questions if q.should_refuse]

    def by_id(self, qid: str) -> GoldenQuestion:
        for q in self.questions:
            if q.id == qid:
                return q
        raise KeyError(qid)

    def filter(
        self,
        *,
        categories: Sequence[str] | None = None,
        difficulties: Sequence[str] | None = None,
        answerable_only: bool = False,
    ) -> GoldenDataset:
        qs = list(self.questions)
        if categories is not None:
            allowed = set(categories)
            qs = [q for q in qs if q.category in allowed]
        if difficulties is not None:
            allowed_d = set(difficulties)
            qs = [q for q in qs if q.difficulty in allowed_d]
        if answerable_only:
            qs = [q for q in qs if not q.should_refuse]
        return self.model_copy(update={"questions": qs})

    def category_counts(self) -> dict[str, int]:
        return {c: sum(1 for q in self.questions if q.category == c) for c in CATEGORIES}

    def difficulty_counts(self) -> dict[str, int]:
        return {d: sum(1 for q in self.questions if q.difficulty == d) for d in DIFFICULTIES}

    # -- validation -------------------------------------------------------- #
    def check_coverage(self) -> list[str]:
        """Return human-readable problems.  Empty list means the set is sane."""
        problems: list[str] = []
        missing = [c for c in CATEGORIES if not any(q.category == c for q in self.questions)]
        if missing:
            problems.append(f"categories with no questions: {missing}")
        if not self.unanswerable:
            problems.append(
                "no unanswerable questions: refusal accuracy cannot be measured "
                "(implementation.md section 14)"
            )
        if not any(q.is_adversarial for q in self.questions):
            problems.append("no adversarial-injection questions")
        unverified = [q.id for q in self.questions if not q.verified]
        if unverified:
            problems.append(f"{len(unverified)} unverified questions, e.g. {unverified[:3]}")
        if self.kind == "release" and not _RELEASE_MIN <= len(self) <= _RELEASE_MAX:
            problems.append(
                f"a release golden set must hold {_RELEASE_MIN}-{_RELEASE_MAX} questions, "
                f"got {len(self)}"
            )
        return problems

    def require_valid(self) -> GoldenDataset:
        problems = self.check_coverage()
        if problems:
            raise ValueError(f"golden dataset {self.name}@{self.version} invalid: {problems}")
        return self

    # -- io ---------------------------------------------------------------- #
    def to_jsonl(self) -> str:
        return "".join(f"{canonical_json(q)}\n" for q in self.questions)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_jsonl(), encoding="utf-8")
        return p


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    version: str
    kind: Literal["release", "sample", "candidates", "corpus", "judge-labels"]
    n_records: int
    sha256: str
    description: str = ""


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1"
    entries: dict[str, ManifestEntry] = Field(default_factory=dict)

    def verify(self, base_dir: Path | None = None) -> list[str]:
        """Recompute every pinned hash.  Returns a list of mismatch messages."""
        base = base_dir or DATASETS_DIR
        problems: list[str] = []
        for name, entry in self.entries.items():
            path = base / entry.path
            if not path.exists():
                problems.append(f"{name}: missing file {entry.path}")
                continue
            actual = _hash_file_records(path, entry.kind)
            if actual != entry.sha256:
                problems.append(
                    f"{name}: hash mismatch -- pinned {entry.sha256[:12]}, got {actual[:12]}"
                )
            n = sum(1 for _ in _iter_jsonl(path))
            if n != entry.n_records:
                problems.append(f"{name}: expected {entry.n_records} records, found {n}")
        return problems


def _hash_file_records(path: Path, kind: str) -> str:
    if kind in ("release", "sample", "candidates"):
        return hash_questions(_parse_questions(path))
    digest = hashlib.sha256()
    for line in sorted(
        json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for r in _iter_jsonl(path)
    ):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_manifest(path: str | Path | None = None) -> DatasetManifest:
    p = Path(path) if path is not None else MANIFEST_PATH
    return DatasetManifest.model_validate_json(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc


def _parse_questions(path: str | Path) -> list[GoldenQuestion]:
    return [GoldenQuestion.model_validate(rec) for rec in _iter_jsonl(path)]


def load_golden(
    path: str | Path | None = None,
    *,
    name: str = "golden-sample",
    version: str = "v1",
    kind: Literal["release", "sample", "candidates"] = "sample",
    require_verified: bool = True,
    expect_hash: str | None = None,
) -> GoldenDataset:
    """Load a golden set from JSONL, optionally asserting its pinned hash."""
    p = Path(path) if path is not None else SAMPLE_GOLDEN_PATH
    questions = _parse_questions(p)
    if require_verified:
        unverified = [q.id for q in questions if not q.verified]
        if unverified:
            raise ValueError(
                f"{p}: {len(unverified)} questions are not hand-verified "
                f"(e.g. {unverified[:3]}). §14 requires hand-verifying every one; "
                "pass require_verified=False to inspect candidates."
            )
    ds = GoldenDataset(name=name, version=version, kind=kind, questions=questions)
    if expect_hash is not None and ds.content_hash != expect_hash:
        raise ValueError(
            f"{p}: golden set hash drifted -- pinned {expect_hash[:12]}, "
            f"got {ds.content_hash[:12]}. Re-pin the manifest deliberately."
        )
    return ds


def load_corpus(path: str | Path | None = None) -> list[CorpusChunk]:
    p = Path(path) if path is not None else SAMPLE_CORPUS_PATH
    return [CorpusChunk.model_validate(rec) for rec in _iter_jsonl(p)]


def load_judge_labels(path: str | Path | None = None) -> list[JudgeLabel]:
    p = Path(path) if path is not None else SAMPLE_JUDGE_LABELS_PATH
    return [JudgeLabel.model_validate(rec) for rec in _iter_jsonl(p)]


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(canonical_json(rec) + "\n")
    return p
