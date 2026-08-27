"""`query_database`: read-only, parameterised SQL against an injectable seam.

Two independent controls, because SQL is the highest-blast-radius tool here:

1. **Statement allowlist.** Only a single `SELECT` (or `WITH ... SELECT`) is
   accepted. Multiple statements, comments, and every mutating or
   filesystem-touching keyword are rejected before the driver sees the string.
2. **Parameterisation.** Literal interpolation is never done here; values travel
   in `params` and are bound by the driver.

Combined with the per-route allowlist and the rule that retrieved text can never
originate a tool call, a document that says "run DROP TABLE users" has no path to
execution: it cannot originate the call, and the string would be rejected anyway.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from localmind.agent.state import Database
from localmind.agent.tools.base import Tool, ToolExecutionError

__all__ = ["QueryDatabaseArgs", "QueryDatabaseTool", "SqlRejectedError", "validate_read_only_sql"]

MAX_SQL_CHARS = 2000

_FORBIDDEN = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "attach",
    "detach",
    "copy",
    "pragma",
    "vacuum",
    "call",
    "execute",
    "exec",
    "merge",
    "replace",
    "load_file",
    "into outfile",
    "into dumpfile",
    "pg_read_file",
    "pg_sleep",
    "dblink",
    "lo_import",
    "set ",
)

_COMMENT = re.compile(r"--|/\*|\*/|#")


class SqlRejectedError(ValueError):
    """The statement is not an acceptable read-only query."""


def validate_read_only_sql(sql: str) -> str:
    """Return the normalised statement, or raise `SqlRejectedError` naming the rule."""
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise SqlRejectedError("empty statement")
    if len(text) > MAX_SQL_CHARS:
        raise SqlRejectedError(f"statement exceeds {MAX_SQL_CHARS} characters")
    if ";" in text:
        raise SqlRejectedError("multiple statements are not permitted")
    if _COMMENT.search(text):
        raise SqlRejectedError("SQL comments are not permitted")
    if "\x00" in text:
        raise SqlRejectedError("null byte in statement")
    low = " " + re.sub(r"\s+", " ", text.lower()) + " "
    if not (low.lstrip().startswith("select ") or low.lstrip().startswith("with ")):
        raise SqlRejectedError("only SELECT (or WITH ... SELECT) statements are permitted")
    for word in _FORBIDDEN:
        # Word-boundary matching, so `pg_sleep(10)` and `copy(` are caught even
        # though no space follows the keyword.
        if re.search(r"(?<![\w$])" + re.escape(word.strip()) + r"(?![\w$])", low):
            raise SqlRejectedError(f"forbidden keyword {word.strip()!r} in a read-only query")
    return text


class QueryDatabaseArgs(BaseModel):
    sql: str = Field(
        min_length=1,
        max_length=MAX_SQL_CHARS,
        description="A single read-only SELECT statement. Use ? / %s placeholders for values.",
    )
    params: list[Any] | None = Field(
        default=None, description="Values bound to the placeholders by the driver."
    )
    max_rows: int = Field(default=50, ge=1, le=500, description="Row cap applied after execution.")


class QueryDatabaseTool(Tool):
    """Structured-data lookups. Read-only by construction, therefore idempotent."""

    name: ClassVar[str] = "query_database"
    description: ClassVar[str] = (
        "Run a single parameterised read-only SELECT against the structured store and "
        "return rows. Mutating statements are rejected."
    )
    args_model: ClassVar[type[BaseModel]] = QueryDatabaseArgs
    idempotent: ClassVar[bool] = True

    def __init__(self, database: Database | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.database = database

    def _call(self, args: QueryDatabaseArgs) -> dict[str, Any]:
        if self.database is None:
            raise ToolExecutionError("unavailable", "no database is attached to this agent")
        try:
            sql = validate_read_only_sql(args.sql)
        except SqlRejectedError as exc:
            raise ToolExecutionError("denied", f"query rejected: {exc}") from exc
        try:
            rows = self.database.execute(sql, args.params)
        except Exception as exc:
            raise ToolExecutionError(
                "unavailable", f"database error: {type(exc).__name__}: {exc}", retryable=True
            ) from exc
        materialised = [dict(r) for r in list(rows or [])[: args.max_rows]]
        return {"rows": materialised, "count": len(materialised), "sql": sql}
