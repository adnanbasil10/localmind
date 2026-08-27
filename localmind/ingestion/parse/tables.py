"""Table extraction (implementation.md §11): to Markdown *and* to a normalized
Postgres structure the SQL tool can query, with provenance (doc, page, bbox)
on every table and every cell.

Two halves:
  1. Pure-Python Markdown-table parsing (`extract_tables_from_blocks`), which
     works on any parser's output, including the offline `PlainTextParser`.
  2. A minimal `SQLConnection` seam (`execute`/`commit`) plus `CREATE TABLE`
     DDL and an insert path, normalized as two tables: `ingestion_tables`
     (one row per extracted table) and `ingestion_table_cells` (one row per
     cell, so the SQL tool can `SELECT value FROM ingestion_table_cells WHERE
     table_id = ... AND header = ...` instead of parsing Markdown at query
     time). Postgres is not running in this environment, so `create_schema`
     and `insert_tables` are exercised in ordinary tests against
     `FakeSQLConnection` (an in-memory double); a real `psycopg` round-trip
     against a live `postgres` service (`docker-compose.yml`, profile `core`)
     is `@pytest.mark.docker` and skipped by default.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from localmind.ingestion.parse import ParsedBlock

__all__ = [
    "CREATE_TABLES_SQL",
    "CREATE_TABLE_CELLS_SQL",
    "INSERT_CELL_SQL",
    "INSERT_TABLE_SQL",
    "SCHEMA_STATEMENTS",
    "ExtractedTable",
    "FakeSQLConnection",
    "SQLConnection",
    "connect_postgres",
    "create_schema",
    "extract_tables_from_blocks",
    "insert_tables",
    "parse_markdown_table",
]


# --------------------------------------------------------------------------------------
# Markdown table parsing
# --------------------------------------------------------------------------------------

_CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")
_SEP_CELL_RE = re.compile(r"^:?-{1,}:?$")


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in _CELL_SPLIT_RE.split(line)]


def parse_markdown_table(markdown: str) -> tuple[list[str], list[list[str]]]:
    """Parse a GitHub-flavoured Markdown pipe table into `(headers, rows)`.
    Raises `ValueError` if `markdown` is not a well-formed table (no header +
    separator row)."""
    lines = [ln for ln in markdown.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("markdown table needs a header row and a separator row")
    headers = _split_row(lines[0])
    sep_cells = _split_row(lines[1])
    if len(sep_cells) != len(headers) or not all(_SEP_CELL_RE.match(c) for c in sep_cells):
        raise ValueError("second row is not a markdown table separator (e.g. '---')")
    rows = [_split_row(ln) for ln in lines[2:]]
    return headers, rows


class ExtractedTable(BaseModel):
    """A table plus its provenance. `to_markdown()` re-renders from the
    normalized `headers`/`rows`, so it always round-trips even if the source
    Markdown had irregular spacing."""

    model_config = ConfigDict(frozen=True)

    table_id: str
    doc_id: str
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    caption: str = ""
    headers: list[str]
    rows: list[list[str]]
    parser: str = "plaintext-fallback"

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.headers)

    def to_markdown(self) -> str:
        lines = ["| " + " | ".join(self.headers) + " |", "|" + "---|" * len(self.headers)]
        for row in self.rows:
            padded = (row + [""] * self.n_cols)[: self.n_cols]
            lines.append("| " + " | ".join(padded) + " |")
        return "\n".join(lines)


def extract_tables_from_blocks(
    blocks: Sequence[ParsedBlock], *, doc_id: str, parser: str = "plaintext-fallback"
) -> list[ExtractedTable]:
    """Pull every well-formed Markdown table out of a block sequence, tagging
    each with `doc_id#tableN` and the provenance (`page`, `bbox`) the parser
    attached to the block. Malformed table blocks are skipped, not raised --
    ingestion of the rest of the document must not fail over one bad table."""
    tables: list[ExtractedTable] = []
    n = 0
    for block in blocks:
        if block.kind != "table":
            continue
        try:
            headers, rows = parse_markdown_table(block.text)
        except ValueError:
            continue
        tables.append(
            ExtractedTable(
                table_id=f"{doc_id}#table{n}",
                doc_id=doc_id,
                page=block.page,
                bbox=block.bbox,
                headers=headers,
                rows=rows,
                parser=parser,
            )
        )
        n += 1
    return tables


# --------------------------------------------------------------------------------------
# Postgres: schema + insert path
# --------------------------------------------------------------------------------------

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_tables (
    table_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    page INTEGER,
    bbox_x0 DOUBLE PRECISION,
    bbox_y0 DOUBLE PRECISION,
    bbox_x1 DOUBLE PRECISION,
    bbox_y1 DOUBLE PRECISION,
    caption TEXT NOT NULL DEFAULT '',
    markdown TEXT NOT NULL,
    n_rows INTEGER NOT NULL,
    n_cols INTEGER NOT NULL,
    parser TEXT NOT NULL
)
""".strip()

CREATE_TABLE_CELLS_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_table_cells (
    table_id TEXT NOT NULL REFERENCES ingestion_tables(table_id) ON DELETE CASCADE,
    row_idx INTEGER NOT NULL,
    col_idx INTEGER NOT NULL,
    header TEXT,
    value TEXT,
    PRIMARY KEY (table_id, row_idx, col_idx)
)
""".strip()

SCHEMA_STATEMENTS: tuple[str, ...] = (CREATE_TABLES_SQL, CREATE_TABLE_CELLS_SQL)

INSERT_TABLE_SQL = """
INSERT INTO ingestion_tables
    (table_id, doc_id, page, bbox_x0, bbox_y0, bbox_x1, bbox_y1, caption, markdown, n_rows, n_cols, parser)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (table_id) DO UPDATE SET
    markdown = EXCLUDED.markdown, n_rows = EXCLUDED.n_rows, n_cols = EXCLUDED.n_cols
""".strip()

INSERT_CELL_SQL = """
INSERT INTO ingestion_table_cells (table_id, row_idx, col_idx, header, value)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (table_id, row_idx, col_idx) DO UPDATE SET value = EXCLUDED.value
""".strip()


@runtime_checkable
class SQLConnection(Protocol):
    """Minimal seam onto Postgres. A `psycopg` connection does NOT satisfy
    this directly (it exposes cursors, not `execute`/`commit`) -- wrap it with
    `connect_postgres`, which returns an adapter that does."""

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> Any: ...
    def commit(self) -> None: ...


class FakeSQLConnection:
    """In-memory double: records every statement and mirrors the two tables
    as plain dicts. Lets `create_schema`/`insert_tables` be exercised with
    Postgres not running -- these are the tests that run without the `docker`
    marker."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[Any, ...]]] = []
        self.tables: dict[str, dict[str, Any]] = {}
        self.cells: dict[tuple[str, int, int], dict[str, Any]] = {}
        self.committed = 0

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        params_t = tuple(params or ())
        self.statements.append((sql, params_t))
        norm = " ".join(sql.split())
        if norm.startswith("INSERT INTO ingestion_tables"):
            table_keys: tuple[str, ...] = (
                "table_id",
                "doc_id",
                "page",
                "bbox_x0",
                "bbox_y0",
                "bbox_x1",
                "bbox_y1",
                "caption",
                "markdown",
                "n_rows",
                "n_cols",
                "parser",
            )
            table_row: dict[str, Any] = dict(zip(table_keys, params_t, strict=True))
            self.tables[params_t[0]] = table_row
        elif norm.startswith("INSERT INTO ingestion_table_cells"):
            cell_keys: tuple[str, ...] = ("table_id", "row_idx", "col_idx", "header", "value")
            cell_row: dict[str, Any] = dict(zip(cell_keys, params_t, strict=True))
            self.cells[(params_t[0], params_t[1], params_t[2])] = cell_row

    def commit(self) -> None:
        self.committed += 1


def create_schema(conn: SQLConnection) -> None:
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
    conn.commit()


def insert_tables(conn: SQLConnection, tables: Sequence[ExtractedTable]) -> int:
    """Insert each table (and every cell) via `INSERT ... ON CONFLICT DO
    UPDATE`, so re-ingesting the same document is idempotent. Returns the
    number of tables inserted."""
    n = 0
    for t in tables:
        bbox = t.bbox or (None, None, None, None)
        conn.execute(
            INSERT_TABLE_SQL,
            (
                t.table_id,
                t.doc_id,
                t.page,
                *bbox,
                t.caption,
                t.to_markdown(),
                t.n_rows,
                t.n_cols,
                t.parser,
            ),
        )
        for r, row in enumerate(t.rows):
            for c in range(t.n_cols):
                header = t.headers[c]
                value = row[c] if c < len(row) else ""
                conn.execute(INSERT_CELL_SQL, (t.table_id, r, c, header, value))
        n += 1
    conn.commit()
    return n


class _PsycopgAdapter:
    """Wraps a real `psycopg.Connection` (cursor-based) to satisfy `SQLConnection`."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        with self._conn.cursor() as cur:
            cur.execute(sql, params)

    def commit(self) -> None:
        self._conn.commit()


def connect_postgres(dsn: str | None = None) -> SQLConnection:
    """Real connection, lazily importing `psycopg`. `dsn` defaults to
    `LOCALMIND_PG_DSN` (the env var `docker-compose.yml`'s `api` service is
    given) then to the compose file's `core` profile default credentials. Only
    used by `@pytest.mark.docker` tests and, eventually, production code."""
    import os

    import psycopg  # type: ignore[import-not-found]

    dsn = dsn or os.environ.get(
        "LOCALMIND_PG_DSN", "postgresql://localmind:localmind@localhost:5432/localmind"
    )
    return _PsycopgAdapter(psycopg.connect(dsn))
