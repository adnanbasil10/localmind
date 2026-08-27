"""MCP server: the same six tools, exposed over JSON-RPC so any MCP client can drive them.

`python -m localmind.agent.mcp_server` speaks newline-delimited JSON-RPC 2.0 on
stdio -- the MCP stdio transport -- so Claude Desktop, Cursor, or any other MCP
client can drive the LocalMind retrieval stack directly.

Implemented against the wire protocol with the standard library only: no `mcp`
package, no httpx, no import-time network. That keeps `import localmind.agent`
working with just numpy and pydantic, and it makes the whole server testable by
handing dicts to `MCPServer.handle`.

Crucially, this is not a second implementation of the tools. It is a thin
JSON-RPC skin over the *same* `ToolRegistry` the agent uses in-process, so the
sandbox, timeouts, retries, structured errors and SQL validation all apply
identically over MCP.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import IO, Any

from localmind.agent.tools import ToolRegistry, build_default_registry

__all__ = ["MCPServer", "main"]

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "localmind"
SERVER_VERSION = "0.1.0"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class MCPServer:
    """Minimal MCP server over the shared `ToolRegistry`."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        name: str = SERVER_NAME,
        version: str = SERVER_VERSION,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.name = name
        self.version = version
        self.initialized = False

    # -- JSON-RPC plumbing ---------------------------------------------------------------

    @staticmethod
    def _ok(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _err(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        """Handle one JSON-RPC message. Returns None for notifications."""
        if not isinstance(request, Mapping):  # pragma: no cover - defensive
            return self._err(None, INVALID_REQUEST, "request must be an object")
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if not isinstance(method, str):
            return self._err(request_id, INVALID_REQUEST, "missing method")
        if not isinstance(params, Mapping):
            return self._err(request_id, INVALID_PARAMS, "params must be an object")

        if method == "initialize":
            self.initialized = True
            return self._ok(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.name, "version": self.version},
                },
            )
        if method in ("notifications/initialized", "initialized"):
            return None
        if method == "ping":
            return self._ok(request_id, {})
        if method == "tools/list":
            return self._ok(request_id, {"tools": self.registry.specs()})
        if method == "tools/call":
            return self._tools_call(request_id, params)
        return self._err(request_id, METHOD_NOT_FOUND, f"unknown method {method!r}")

    def _tools_call(self, request_id: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str):
            return self._err(request_id, INVALID_PARAMS, "params.name must be a string")
        if not isinstance(arguments, Mapping):
            return self._err(request_id, INVALID_PARAMS, "params.arguments must be an object")

        result = self.registry.call(name, arguments)
        # MCP convention: tool failures are results with isError, not protocol errors,
        # so the model sees the structured error and can act on it.
        if result.ok:
            payload: dict[str, Any] = {
                "tool": result.tool,
                "data": result.data,
                "elapsed_ms": round(result.elapsed_ms, 3),
                "attempts": result.attempts,
                "cached": result.cached,
            }
        else:
            error = result.error
            payload = {
                "tool": result.tool,
                "error": {
                    "code": error.code if error else "internal",
                    "message": error.message if error else "unspecified failure",
                    "retryable": bool(error.retryable) if error else False,
                },
                "elapsed_ms": round(result.elapsed_ms, 3),
                "attempts": result.attempts,
            }
        return self._ok(
            request_id,
            {
                "content": [{"type": "text", "text": json.dumps(payload, default=str)}],
                "isError": not result.ok,
            },
        )

    # -- transport -----------------------------------------------------------------------

    def handle_line(self, line: str) -> str | None:
        """Handle one newline-delimited JSON-RPC message; returns the reply line."""
        text = line.strip()
        if not text:
            return None
        try:
            request = json.loads(text)
        except json.JSONDecodeError as exc:
            return json.dumps(self._err(None, PARSE_ERROR, f"invalid JSON: {exc.msg}"))
        try:
            response = self.handle(request)
        except Exception as exc:
            request_id = request.get("id") if isinstance(request, Mapping) else None
            response = self._err(request_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        return None if response is None else json.dumps(response, default=str)

    def serve_stdio(self, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> None:
        """Blocking stdio loop. This is what an MCP client spawns."""
        src = stdin or sys.stdin
        dst = stdout or sys.stdout
        for line in src:
            reply = self.handle_line(line)
            if reply is not None:
                dst.write(reply + "\n")
                dst.flush()


def main(argv: list[str] | None = None) -> int:
    """Entry point: `python -m localmind.agent.mcp_server`.

    Seams are unset by default, so tools return `unavailable` structured errors
    rather than importing the retrieval stack. Wire a registry in-process (see
    `build_default_registry`) to serve a live index.
    """
    del argv
    MCPServer().serve_stdio()
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
