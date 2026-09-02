"""Tool base class: typed args, timeout, retry with backoff, structured errors, idempotency.

Every tool in this package obeys the same contract:

* **Typed pydantic schema.** Args are validated before the tool body runs; a
  validation failure is an `invalid_input` *result*, not an exception.
* **Timeout.** The body runs on a daemon thread and is abandoned if it overruns,
  so a wedged provider cannot wedge the agent (or interpreter shutdown). This
  covers I/O waits and any body that releases the GIL -- which is what a "wedged
  provider" actually is. It does **not** cover CPU-bound work inside a single
  C call that holds the GIL; see `call_with_timeout` for why, and what tools are
  required to do instead.
* **Retry with exponential backoff.** Only retryable failures are retried, and
  the sleep goes through an injectable `Clock`, so tests are instant.
* **Structured error returned TO the model.** `ToolResult.ok is False` with a
  typed `ToolError`; the model gets a sentence it can act on. Nothing raises.
* **Idempotency where possible.** Read-only tools declare `idempotent = True`
  and share a TTL cache keyed by a hash of the tool name plus canonical args, so
  a retried or repeated call returns the same answer without re-doing the work.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from localmind.agent.state import Clock, ErrorCode, ToolError, ToolResult, WallClock

__all__ = [
    "ResultCache",
    "Tool",
    "ToolExecutionError",
    "call_with_timeout",
    "idempotency_key",
]


class ToolExecutionError(Exception):
    """Raised *inside* a tool body; converted to a structured `ToolError` by `Tool.run`."""

    def __init__(self, code: ErrorCode, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code: ErrorCode = code
        self.message = message
        self.retryable = retryable


class _TimeoutError(Exception):
    pass


def call_with_timeout(fn: Callable[[], Any], timeout_s: float) -> Any:
    """Run `fn` on a daemon thread; abandon it if it overruns `timeout_s`.

    Daemon threads (rather than a `ThreadPoolExecutor`) are deliberate: an
    abandoned worker must never block interpreter shutdown.

    **Known limit, stated because it is load-bearing.** This is a watchdog, not a
    preemption primitive. `done.wait(timeout_s)` can only fire when the
    interpreter is free to schedule this thread, so the timeout is real for
    blocking I/O, `time.sleep`, and any extension that releases the GIL -- and
    inert for CPU-bound work inside a *single* C call that holds the GIL
    throughout. CPython bigint arithmetic is the canonical example:
    `round(2, -10**7)` returned `ok=True` after 11 s under a 1.0 s timeout,
    because the waiter never got a turn.

    There is no thread-level fix; interrupting that would need a subprocess.
    Tools whose bodies can be driven into GIL-holding C by their *arguments*
    must therefore bound those arguments before evaluating -- which is what
    `calculate._guarded_pow`, `_guarded_factorial` and `_guarded_round` do. Treat
    this timeout as a backstop for hung providers, never as the containment story
    for untrusted input.
    """
    if timeout_s <= 0:
        raise _TimeoutError(f"timeout budget of {timeout_s}s is exhausted before the call")

    box: dict[str, Any] = {}
    done = threading.Event()

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:
            box["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=target, daemon=True, name="localmind-tool")
    thread.start()
    if not done.wait(timeout_s):
        raise _TimeoutError(f"exceeded {timeout_s:.3f}s timeout")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def idempotency_key(tool: str, args: Mapping[str, Any]) -> str:
    """Stable hash of tool + canonical args. Same inputs -> same key, across processes."""
    payload = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{tool}\x00{payload}".encode()).hexdigest()[:32]


class ResultCache:
    """Tiny TTL cache for idempotent tool results. Injectable clock; bounded size."""

    def __init__(self, ttl_s: float = 300.0, max_entries: int = 256, clock: Clock | None = None):
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.clock: Clock = clock or WallClock()
        self._store: dict[str, tuple[float, ToolResult]] = {}

    def get(self, key: str) -> ToolResult | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        stamped, result = entry
        if self.clock.now() - stamped > self.ttl_s:
            self._store.pop(key, None)
            return None
        return result

    def put(self, key: str, result: ToolResult) -> None:
        if len(self._store) >= self.max_entries:
            oldest = min(self._store, key=lambda k: self._store[k][0])
            self._store.pop(oldest, None)
        self._store[key] = (self.clock.now(), result)

    def clear(self) -> None:
        self._store.clear()


class Tool(ABC):
    """Base class for every LocalMind tool."""

    name: ClassVar[str] = "tool"
    description: ClassVar[str] = ""
    args_model: ClassVar[type[BaseModel]]
    idempotent: ClassVar[bool] = True

    def __init__(
        self,
        *,
        timeout_s: float = 5.0,
        retries: int = 2,
        backoff_base_s: float = 0.05,
        backoff_max_s: float = 2.0,
        clock: Clock | None = None,
        cache: ResultCache | None = None,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.backoff_base_s = float(backoff_base_s)
        self.backoff_max_s = float(backoff_max_s)
        self.clock: Clock = clock or WallClock()
        self.cache = (
            cache
            if cache is not None
            else (ResultCache(clock=self.clock) if self.idempotent else None)
        )

    # -- to implement ------------------------------------------------------------------

    @abstractmethod
    def _call(self, args: Any) -> dict[str, Any]:
        """Tool body. Raise `ToolExecutionError` for expected failures."""

    # -- schema ------------------------------------------------------------------------

    def json_schema(self) -> dict[str, Any]:
        """JSON schema of the arguments, for the MCP `tools/list` response."""
        return self.args_model.model_json_schema()

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.json_schema(),
        }

    # -- execution ---------------------------------------------------------------------

    def _backoff(self, attempt: int, key: str) -> float:
        """Exponential backoff with deterministic per-key jitter (no RNG state)."""
        base = min(self.backoff_max_s, self.backoff_base_s * (2 ** (attempt - 1)))
        jitter = (int(key[:8], 16) % 1000) / 1000.0 * 0.1 * base if key else 0.0
        return base + jitter

    def _error(self, code: ErrorCode, message: str, *, retryable: bool = False) -> ToolError:
        return ToolError(code=code, message=message, tool=self.name, retryable=retryable)

    def run(self, raw_args: Mapping[str, Any] | None = None) -> ToolResult:
        """Validate, execute with timeout+retry, and always return a `ToolResult`."""
        started = time.perf_counter()
        try:
            args = self.args_model(**dict(raw_args or {}))
        except ValidationError as exc:
            return ToolResult(
                tool=self.name,
                ok=False,
                error=self._error("invalid_input", _format_validation_error(exc)),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        key = idempotency_key(self.name, args.model_dump(mode="json"))
        if self.idempotent and self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit.model_copy(
                    update={
                        "cached": True,
                        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    }
                )

        deadline = self.clock.now() + self.timeout_s if self.timeout_s > 0 else 0.0
        last_error: ToolError | None = None
        attempts = 0
        for attempt in range(1, self.retries + 2):
            attempts = attempt
            remaining = self.timeout_s
            if isinstance(self.clock, WallClock):
                remaining = max(0.0, deadline - self.clock.now())
            try:
                data = call_with_timeout(lambda: self._call(args), remaining)
                result = ToolResult(
                    tool=self.name,
                    ok=True,
                    data=dict(data or {}),
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    attempts=attempts,
                    idempotency_key=key,
                )
                if self.idempotent and self.cache is not None:
                    self.cache.put(key, result)
                return result
            except _TimeoutError as exc:
                last_error = self._error("timeout", f"{self.name} {exc}", retryable=True)
            except ToolExecutionError as exc:
                last_error = self._error(exc.code, exc.message, retryable=exc.retryable)
            except Exception as exc:
                last_error = self._error(
                    "internal", f"{type(exc).__name__}: {exc}", retryable=False
                )
            if last_error is None or not last_error.retryable or attempt > self.retries:
                break
            self.clock.sleep(self._backoff(attempt, key))

        assert last_error is not None
        return ToolResult(
            tool=self.name,
            ok=False,
            error=last_error,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            attempts=attempts,
            idempotency_key=key,
        )


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:4]:
        loc = ".".join(str(x) for x in err.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "invalid arguments -- " + "; ".join(parts)
