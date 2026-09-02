"""ASGI entrypoint for the RAG gateway.

``deploy/Dockerfile.api``'s CMD is ``uvicorn localmind.api.main:app``, so this module path and the
name ``app`` are fixed. ``app`` is resolved through :pep:`562` ``__getattr__``, exactly as
``localmind.inference.server`` does it, so importing this module never requires FastAPI to be
installed — only *serving* does. ``import localmind.api`` therefore stays green in a checkout
without the ``rag`` extra, and ``tests/test_api.py`` skips the framework-bound cases cleanly
instead of erroring at collection.

Startup does exactly one optional thing: it calls
:func:`~localmind.obs.tracing.init_tracing`, which returns ``False`` and stays a no-op when
``opentelemetry`` is absent or the collector is unreachable. Nothing else is contacted, which is
what lets ``GET /health`` return 200 on a machine with no Postgres, no Redis, no Ollama and no
Phoenix.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from localmind.api.deps import Container, Settings, build_container
from localmind.api.middleware import RequestContextMiddleware
from localmind.api.routes import create_router, install_error_handlers
from localmind.obs.tracing import init_tracing, shutdown_tracing

__all__ = ["create_app"]

DESCRIPTION = """\
The LocalMind RAG gateway: auth, rate limiting, the OpenTelemetry root span, and the agent
entrypoint. Model serving (OpenAI-compatible `/v1/*`) is a separate process — see
`localmind.inference.server`.
"""


def create_app(container: Container | None = None, settings: Settings | None = None) -> Any:
    """Build the FastAPI application. Raises a clear error if the ``rag`` extra is missing."""
    try:
        from contextlib import asynccontextmanager

        from fastapi import FastAPI
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "fastapi/uvicorn are in the 'rag' optional dependency group. "
            "Install with: uv pip install -e '.[rag]'"
        ) from exc

    if container is None:
        settings = settings or Settings.from_env()
        container = build_container(settings)
    resolved = container.settings

    @asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[None]:
        # Never fatal: returns False (and traces to nowhere) without `opentelemetry`, and the OTLP
        # exporter connects lazily, so a missing Phoenix does not block startup.
        container.tracing_enabled = init_tracing(resolved.service_name, resolved.otlp_endpoint)
        try:
            yield
        finally:
            shutdown_tracing()

    app = FastAPI(
        title="LocalMind RAG gateway",
        version=resolved.version,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs" if resolved.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if resolved.docs_enabled else None,
    )
    app.state.container = container
    install_error_handlers(app)
    app.include_router(create_router(container))

    if resolved.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["authorization", "content-type", "x-api-key", "x-request-id"],
            expose_headers=["x-request-id"],
        )

    # Added last, so it is the outermost middleware: every response — including a 404, a CORS
    # preflight and an unhandled exception — carries a request id and lives inside the root span.
    app.add_middleware(RequestContextMiddleware, metrics=container.http_metrics)
    return app


def __getattr__(name: str) -> Any:
    """PEP 562 hook so ``uvicorn localmind.api.main:app`` works without an import-time FastAPI."""
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
