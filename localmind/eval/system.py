"""System-layer metrics: latency, tokens, tools, iterations -- and the twist.

§14 asks for TTFT, end-to-end p50/p95/p99, tokens in/out, tool success rate and
the iteration-count distribution.  It then makes the point that matters for
this project: **cost per query is $0**, so the interesting resource numbers are
**CPU-seconds per query** and **peak RSS** -- the quantities that actually
constrain a laptop-resident system.

Nothing here needs a model.  A caller instruments its own request path with
:func:`measure_request` (or fills in :class:`RequestRecord` directly) and this
module turns a pile of records into a report where every number has a CI.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from localmind.eval.stats import (
    DEFAULT_SEED,
    Estimate,
    MetricRow,
    benchmark_json,
    bootstrap_ci,
    percentile_stat,
    wilson_ci,
)

__all__ = [
    "RequestRecord",
    "SystemReport",
    "ToolCall",
    "evaluate_system",
    "hardware_string",
    "measure_request",
    "peak_rss_bytes",
]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    ok: bool = True
    duration_s: float = 0.0
    error: str = ""


class RequestRecord(BaseModel):
    """One end-to-end request through the system under test."""

    model_config = ConfigDict(extra="allow")

    id: str
    ttft_s: float | None = None
    total_s: float = 0.0
    cpu_s: float = 0.0
    peak_rss_bytes: int | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    n_iterations: int = 1
    tool_calls: list[ToolCall] = Field(default_factory=list)
    refused: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def tokens_per_s(self) -> float:
        return self.tokens_out / self.total_s if self.total_s > 0 else 0.0


# --------------------------------------------------------------------------- #
# Instrumentation
# --------------------------------------------------------------------------- #
def peak_rss_bytes() -> int | None:
    """Peak resident set size of this process, or ``None`` if unavailable.

    Tries ``resource`` (POSIX), then the Win32 ``psapi`` peak working set, then
    ``psutil`` if it happens to be installed.  Never raises.
    """
    try:
        import resource  # POSIX only; absent on Windows, hence the try block

        maxrss = resource.getrusage(  # type: ignore[attr-defined]
            resource.RUSAGE_SELF  # type: ignore[attr-defined]
        ).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        return int(maxrss) if sys.platform == "darwin" else int(maxrss) * 1024
    except Exception:
        pass
    if os.name == "nt":
        with contextlib.suppress(Exception):
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = (
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                )

            counters = _PMC()
            counters.cb = ctypes.sizeof(_PMC)
            handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                handle, ctypes.byref(counters), counters.cb
            )
            if ok:
                return int(counters.PeakWorkingSetSize)
    with contextlib.suppress(Exception):
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    return None


@contextlib.contextmanager
def measure_request(record_id: str, **fields: Any) -> Iterator[RequestRecord]:
    """Time a request, filling in wall time, CPU time and peak RSS.

    The caller sets ``ttft_s`` the moment the first token lands, and the token
    counts / tool calls as they happen::

        with measure_request("q-001") as rec:
            rec.ttft_s = ...
            rec.tokens_out = ...
    """
    rec = RequestRecord(id=record_id, **fields)
    t0 = time.perf_counter()
    c0 = time.process_time()
    try:
        yield rec
    except Exception as exc:  # record the failure, do not swallow it
        rec.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        rec.total_s = time.perf_counter() - t0
        rec.cpu_s = time.process_time() - c0
        rec.peak_rss_bytes = peak_rss_bytes()


def hardware_string() -> str:
    """The hardware line CONVENTIONS.md requires on every benchmark."""
    bits = [
        platform.system(),
        platform.release(),
        platform.machine(),
        (platform.processor() or "unknown-cpu"),
        f"{os.cpu_count() or '?'}vCPU",
        f"py{platform.python_version()}",
    ]
    return " | ".join(b for b in bits if b)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class SystemReport:
    """Latency / throughput / resource report.  Every number carries a CI."""

    system: str
    metrics: dict[str, Estimate] = field(default_factory=dict)
    iteration_histogram: dict[int, int] = field(default_factory=dict)
    tool_histogram: dict[str, dict[str, int]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    seed: int = DEFAULT_SEED
    hardware: str = ""

    def to_row(self) -> MetricRow:
        return MetricRow(
            name=self.system,
            values=dict(self.metrics),
            extra={
                "counts": dict(self.counts),
                "iteration_histogram": {str(k): v for k, v in self.iteration_histogram.items()},
                "tool_histogram": self.tool_histogram,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "hardware": self.hardware,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "iteration_histogram": {str(k): v for k, v in self.iteration_histogram.items()},
            "tool_histogram": self.tool_histogram,
            "counts": dict(self.counts),
            "seed": self.seed,
        }

    def to_markdown(self) -> str:
        order = [
            "ttft_ms",
            "e2e_p50_ms",
            "e2e_p95_ms",
            "e2e_p99_ms",
            "tokens_in",
            "tokens_out",
            "tokens_per_s",
            "cpu_s_per_query",
            "peak_rss_mb",
            "tool_success_rate",
            "iterations",
            "refusal_rate",
            "error_rate",
        ]
        lines = ["| metric | value (95% CI) |", "|---|---|"]
        for key in order:
            if key in self.metrics:
                lines.append(f"| {key} | {self.metrics[key].format(2)} |")
        for key in sorted(k for k in self.metrics if k not in order):
            lines.append(f"| {key} | {self.metrics[key].format(2)} |")
        hist = ", ".join(f"{k}:{v}" for k, v in sorted(self.iteration_histogram.items()))
        lines += [
            "",
            f"iteration-count distribution: {hist or 'n/a'}",
            "",
            "Cost per query is $0 (everything is local), so the resource line to read is "
            "`cpu_s_per_query` and `peak_rss_mb`, not dollars.",
            "",
            f"hardware: {self.hardware}",
        ]
        return "\n".join(lines)


def evaluate_system(
    records: Sequence[RequestRecord],
    *,
    system: str = "system",
    seed: int = DEFAULT_SEED,
    n_resamples: int = 2_000,
    hardware: str | None = None,
) -> SystemReport:
    """Turn request records into a report where nothing is a bare number."""
    if not records:
        raise ValueError("evaluate_system requires at least one request record")

    ok = [r for r in records if r.ok]
    latencies_ms = [r.total_s * 1000.0 for r in ok]
    metrics: dict[str, Estimate] = {}

    def add(name: str, values: Sequence[float], statistic: Any = None) -> None:
        if not values:
            return
        kwargs = {"statistic": statistic} if statistic is not None else {}
        metrics[name] = bootstrap_ci(values, seed=seed, n_resamples=n_resamples, **kwargs)

    ttfts = [r.ttft_s * 1000.0 for r in ok if r.ttft_s is not None]
    add("ttft_ms", ttfts)
    add("e2e_mean_ms", latencies_ms)
    add("e2e_p50_ms", latencies_ms, percentile_stat(50))
    add("e2e_p95_ms", latencies_ms, percentile_stat(95))
    add("e2e_p99_ms", latencies_ms, percentile_stat(99))
    add("tokens_in", [float(r.tokens_in) for r in ok])
    add("tokens_out", [float(r.tokens_out) for r in ok])
    add("tokens_per_s", [r.tokens_per_s for r in ok if r.total_s > 0])
    add("cpu_s_per_query", [r.cpu_s for r in ok])
    add("iterations", [float(r.n_iterations) for r in ok])
    rss = [r.peak_rss_bytes / (1024 * 1024) for r in ok if r.peak_rss_bytes]
    add("peak_rss_mb", rss)
    if rss:
        # Peak RSS is a maximum, not an average: report the worst case too.
        metrics["peak_rss_mb_max"] = Estimate.degenerate(max(rss), n=len(rss), method="max")

    tool_calls = [tc for r in records for tc in r.tool_calls]
    if tool_calls:
        metrics["tool_success_rate"] = wilson_ci(
            sum(1 for tc in tool_calls if tc.ok), len(tool_calls)
        )
    metrics["error_rate"] = wilson_ci(sum(1 for r in records if not r.ok), len(records))
    metrics["refusal_rate"] = wilson_ci(sum(1 for r in records if r.refused), len(records))

    iteration_histogram: dict[int, int] = {}
    for r in ok:
        iteration_histogram[r.n_iterations] = iteration_histogram.get(r.n_iterations, 0) + 1

    tool_histogram: dict[str, dict[str, int]] = {}
    for tc in tool_calls:
        bucket = tool_histogram.setdefault(tc.name, {"ok": 0, "fail": 0})
        bucket["ok" if tc.ok else "fail"] += 1

    return SystemReport(
        system=system,
        metrics=metrics,
        iteration_histogram=dict(sorted(iteration_histogram.items())),
        tool_histogram=tool_histogram,
        counts={
            "requests": len(records),
            "ok": len(ok),
            "errors": len(records) - len(ok),
            "tool_calls": len(tool_calls),
        },
        seed=seed,
        hardware=hardware or hardware_string(),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m localmind.eval.system",
        description="Summarise request records into latency/resource metrics with CIs.",
    )
    parser.add_argument("--records", required=True, help="JSONL of RequestRecord")
    parser.add_argument("--system", default="system")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", default="artifacts/benchmarks/system.json")
    args = parser.parse_args(argv)

    records = [
        RequestRecord.model_validate_json(line)
        for line in Path(args.records).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = evaluate_system(records, system=args.system, seed=args.seed)
    payload = benchmark_json(
        "system",
        hardware=report.hardware,
        seeds=[args.seed],
        rows=[report.to_row()],
        extra={"detail": report.to_dict()},
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report.to_markdown())
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
