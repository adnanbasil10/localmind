"""Tool registry: the six LocalMind tools, in-process and over MCP.

The registry is the single execution path. `graph.py` calls it in-process and
`mcp_server.py` calls it over JSON-RPC, so an MCP client (Claude Desktop, Cursor)
drives exactly the same code -- including the same timeouts, retries, sandbox and
allowlist -- as the agent does.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import Any

from localmind.agent.state import (
    Database,
    Generator,
    ImageStore,
    Retriever,
    ToolError,
    ToolResult,
    WebSearchProvider,
)
from localmind.agent.tools.base import ResultCache, Tool, ToolExecutionError, idempotency_key
from localmind.agent.tools.calculate import CalculateTool, UnsafeExpressionError, safe_eval
from localmind.agent.tools.query_database import QueryDatabaseTool, validate_read_only_sql
from localmind.agent.tools.retrieve_image import RetrieveImageTool
from localmind.agent.tools.search_documents import SearchDocumentsTool, coerce_chunk
from localmind.agent.tools.search_web import (
    CachingWebSearchProvider,
    DuckDuckGoProvider,
    SearchWebTool,
    StaticWebSearchProvider,
)
from localmind.agent.tools.summarize_document import DocumentSource, SummarizeDocumentTool

__all__ = [
    "CachingWebSearchProvider",
    "CalculateTool",
    "DocumentSource",
    "DuckDuckGoProvider",
    "QueryDatabaseTool",
    "ResultCache",
    "RetrieveImageTool",
    "SearchDocumentsTool",
    "SearchWebTool",
    "StaticWebSearchProvider",
    "SummarizeDocumentTool",
    "Tool",
    "ToolExecutionError",
    "ToolRegistry",
    "UnsafeExpressionError",
    "build_default_registry",
    "coerce_chunk",
    "idempotency_key",
    "safe_eval",
    "validate_read_only_sql",
]


class ToolRegistry:
    """Name -> tool, with allowlist enforcement and structured denials."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [self._tools[n].spec() for n in self.names()]

    def call(
        self,
        name: str,
        args: Mapping[str, Any] | None = None,
        *,
        allowlist: Collection[str] | None = None,
    ) -> ToolResult:
        """Execute a tool. Unknown or disallowed tools return a `denied` result."""
        if allowlist is not None and name not in allowlist:
            return ToolResult(
                tool=name,
                ok=False,
                error=ToolError(
                    code="denied",
                    message=f"tool {name!r} is not allowlisted for this route",
                    tool=name,
                ),
                idempotency_key=idempotency_key(name, dict(args or {})),
            )
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool=name,
                ok=False,
                error=ToolError(
                    code="not_found",
                    message=f"unknown tool {name!r}; available: {', '.join(self.names())}",
                    tool=name,
                ),
            )
        return tool.run(args or {})


def build_default_registry(
    *,
    retriever: Retriever | None = None,
    web_provider: WebSearchProvider | None = None,
    database: Database | None = None,
    image_store: ImageStore | None = None,
    generator: Generator | None = None,
    document_source: DocumentSource | None = None,
    classifier: Any | None = None,
    timeout_s: float = 5.0,
    retries: int = 2,
    clock: Any | None = None,
) -> ToolRegistry:
    """All six tools, wired to whatever seams the caller provides.

    Missing seams are not an error: the corresponding tool returns an
    `unavailable` structured error, which the model can read and route around.
    """
    common: dict[str, Any] = {"timeout_s": timeout_s, "retries": retries}
    if clock is not None:
        common["clock"] = clock
    return ToolRegistry(
        [
            SearchDocumentsTool(retriever, **common),
            SearchWebTool(web_provider, **common),
            QueryDatabaseTool(database, **common),
            CalculateTool(**common),
            RetrieveImageTool(image_store, **common),
            SummarizeDocumentTool(
                generator, document_source, classifier, **{**common, "timeout_s": timeout_s * 4}
            ),
        ]
    )
