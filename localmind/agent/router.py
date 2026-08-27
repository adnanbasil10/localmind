"""Route node: 3-way domain routing by LocalMind-31M on CPU.

`ControlPlane` is re-exported here (and from `grader.py` / `rewriter.py`) so
callers can import the frozen CONVENTIONS.md contract from any of the three
modules that use it. It is *defined* once, in `state.py`, to avoid an import
cycle -- it is not redefined.
"""

from __future__ import annotations

import re
import time

from pydantic import BaseModel, ConfigDict

from localmind.agent.guardrails import ROUTE_TOOL_ALLOWLIST
from localmind.agent.state import ControlPlane, Route

__all__ = ["ControlPlane", "Route", "RouteDecision", "Router", "heuristic_route"]

_VALID_ROUTES: frozenset[str] = frozenset({"in_domain", "out_of_domain", "needs_web"})

_WEB_HINTS = re.compile(
    r"\b(?:latest|today|todays|current|currently|recent|recently|news|now|this\s+(?:week|month|year)"
    r"|20[2-9]\d|price\s+of|weather|who\s+won|release[ds]?\s+(?:date|version)|up\s*to\s*date)\b",
    re.IGNORECASE,
)

_OUT_OF_DOMAIN_HINTS = re.compile(
    r"\b(?:write\s+me\s+a\s+poem|tell\s+me\s+a\s+joke|your\s+opinion|how\s+do\s+you\s+feel"
    r"|meaning\s+of\s+life|who\s+are\s+you|marry|horoscope)\b",
    re.IGNORECASE,
)


def heuristic_route(query: str) -> Route:
    """Deterministic fallback used when the 31M control plane is unavailable.

    Cheap, explainable, and never a silent failure: `Router` records that the
    fallback fired so a degraded control plane shows up in the trace.
    """
    if _OUT_OF_DOMAIN_HINTS.search(query):
        return "out_of_domain"
    if _WEB_HINTS.search(query):
        return "needs_web"
    return "in_domain"


class RouteDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: Route
    latency_ms: float = 0.0
    source: str = "control_plane"
    detail: str = ""

    @property
    def allowed_tools(self) -> frozenset[str]:
        return ROUTE_TOOL_ALLOWLIST.get(self.route, frozenset())


class Router:
    """Wraps `ControlPlane.route` with validation, latency logging and a fallback."""

    def __init__(self, control_plane: ControlPlane | None = None) -> None:
        self.control_plane = control_plane

    def route(self, query: str) -> RouteDecision:
        if self.control_plane is None:
            return RouteDecision(route=heuristic_route(query), source="heuristic")
        t0 = time.perf_counter()
        try:
            raw = self.control_plane.route(query)
        except Exception as exc:
            return RouteDecision(
                route=heuristic_route(query),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                source="heuristic",
                detail=f"control plane raised {type(exc).__name__}",
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if isinstance(raw, str) and raw in _VALID_ROUTES:
            return RouteDecision(route=raw, latency_ms=latency_ms, source="control_plane")  # type: ignore[arg-type]
        return RouteDecision(
            route=heuristic_route(query),
            latency_ms=latency_ms,
            source="heuristic",
            detail=f"control plane returned invalid route {raw!r}",
        )
