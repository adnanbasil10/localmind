"""LocalMind agent: a typed state machine over an agentic RAG loop.

    route -> retrieve -> grade -> (rewrite -> retrieve)* -> generate -> verify -> respond

`route`, `rewrite` and `grade` run LocalMind-31M on CPU through the frozen
`ControlPlane` Protocol; `generate` calls Ollama through the `Generator`
Protocol. Every external dependency is injectable, so the whole package imports
and tests with only numpy and pydantic installed -- no torch, no network, no
index. Submodules are imported lazily below for the same reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "Agent",
    "AgentResult",
    "AgentState",
    "Budget",
    "ControlPlane",
    "Generator",
    "MCPServer",
    "Node",
    "RetrievedChunk",
    "Retriever",
    "ToolRegistry",
    "WebSearchProvider",
    "build_default_registry",
    "evaluate_injection_cases",
    "load_injection_cases",
    "safe_eval",
]

_LAZY: dict[str, tuple[str, str]] = {
    "Agent": ("localmind.agent.graph", "Agent"),
    "AgentResult": ("localmind.agent.state", "AgentResult"),
    "AgentState": ("localmind.agent.state", "AgentState"),
    "Budget": ("localmind.agent.state", "Budget"),
    "ControlPlane": ("localmind.agent.state", "ControlPlane"),
    "Generator": ("localmind.agent.state", "Generator"),
    "MCPServer": ("localmind.agent.mcp_server", "MCPServer"),
    "Node": ("localmind.agent.state", "Node"),
    "RetrievedChunk": ("localmind.agent.state", "RetrievedChunk"),
    "Retriever": ("localmind.agent.state", "Retriever"),
    "ToolRegistry": ("localmind.agent.tools", "ToolRegistry"),
    "WebSearchProvider": ("localmind.agent.state", "WebSearchProvider"),
    "build_default_registry": ("localmind.agent.tools", "build_default_registry"),
    "evaluate_injection_cases": ("localmind.agent.guardrails", "evaluate_injection_cases"),
    "load_injection_cases": ("localmind.agent.guardrails", "load_injection_cases"),
    "safe_eval": ("localmind.agent.tools.calculate", "safe_eval"),
}


def __getattr__(name: str) -> Any:
    """Lazy re-export, so `import localmind.agent` stays cheap and dependency-light."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target[0]), target[1])


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from localmind.agent.graph import Agent
    from localmind.agent.guardrails import evaluate_injection_cases, load_injection_cases
    from localmind.agent.mcp_server import MCPServer
    from localmind.agent.state import (
        AgentResult,
        AgentState,
        Budget,
        ControlPlane,
        Generator,
        Node,
        RetrievedChunk,
        Retriever,
        WebSearchProvider,
    )
    from localmind.agent.tools import ToolRegistry, build_default_registry
    from localmind.agent.tools.calculate import safe_eval
