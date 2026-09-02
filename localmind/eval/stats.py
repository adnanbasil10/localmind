"""Statistics backbone for the LocalMind evaluation harness.

Rule 5 of the four rules (``implementation.md`` §20) is *never report a bare
number*.  This module is how that rule is enforced mechanically rather than by
discipline: every aggregate produced here is an :class:`Estimate` carrying a
bootstrap 95% CI, and :class:`Estimate` deliberately refuses to be coerced to
``float`` so a caller cannot accidentally emit a point estimate.

Every other phase's benchmark imports this module.  It depends on ``numpy``
only; ``scipy`` is used opportunistically for Wilcoxon and a correct pure-numpy
implementation is provided when it is absent.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

__all__ = [
    "DEFAULT_CI",
    "DEFAULT_N_RESAMPLES",
    "DEFAULT_SEED",
    "MIN_SEEDS",
    "ComparisonReport",
    "Estimate",
    "MetricRow",
    "PairedBootstrapResult",
    "WilcoxonResult",
    "aggregate_seeds",
    "benchmark_json",
    "bootstrap_ci",
    "bootstrap_paired",
    "cohens_kappa",
    "compare_configs",
    "format_metric",
    "kappa_ci",
    "mean_ci_t",
    "percentile_stat",
    "require_estimate",
    "require_seeds",
    "rows_to_markdown",
    "wilcoxon_signed_rank",
    "wilson_ci",
]

DEFAULT_SEED = 1234
DEFAULT_N_RESAMPLES = 10_000
DEFAULT_CI = 0.95
MIN_SEEDS = 3
"""§20 rule 5: 3+ seeds for anything stochastic."""

_BOOTSTRAP_BLOCK = 2_000


# --------------------------------------------------------------------------- #
# Estimate: a number that cannot be reported without its interval
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Estimate:
    """A point estimate that is inseparable from its confidence interval.

    ``float(estimate)`` raises on purpose.  If you truly want the bare centre
    you must say ``estimate.mean`` out loud, which makes the omission of the
    interval a visible, reviewable choice instead of an accident.
    """

    mean: float
    lo: float
    hi: float
    n: int
    method: str = "bootstrap-percentile"
    ci_level: float = DEFAULT_CI
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"lo ({self.lo}) must not exceed hi ({self.hi})")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError(f"ci_level must be in (0, 1), got {self.ci_level}")
        if self.n < 0:
            raise ValueError(f"n must be non-negative, got {self.n}")

    # -- formatting -------------------------------------------------------- #
    def format(self, digits: int = 3) -> str:
        """``"0.812 [0.774, 0.849]"`` -- the only sanctioned rendering."""
        return f"{self.mean:.{digits}f} [{self.lo:.{digits}f}, {self.hi:.{digits}f}]"

    def __str__(self) -> str:
        return self.format()

    def __format__(self, spec: str) -> str:
        if spec in ("", "s"):
            return self.format()
        if spec.endswith(("f", "g", "e", "%")):
            raise TypeError(
                "refusing to format an Estimate as a bare number "
                f"(spec {spec!r}); use .format(digits) or say .mean explicitly"
            )
        return format(self.format(), spec)

    def __float__(self) -> float:
        raise TypeError(
            "Estimate does not coerce to float: a point estimate without its CI "
            "violates implementation.md section 20 rule 5. Use .mean if that is "
            "really what you want, or .format() to report it."
        )

    # -- interop ----------------------------------------------------------- #
    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "lo": self.lo,
            "hi": self.hi,
            "n": self.n,
            "method": self.method,
            "ci_level": self.ci_level,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Estimate:
        """Parse an estimate, accepting the ``ci_low``/``ci_high`` spelling too.

        Other phases serialise their own benchmark rows; the reporting layer
        has to read all of them, so the bound names are aliased rather than
        forcing every producer to converge on one spelling.
        """
        lo = _first_key(d, "lo", "ci_low", "low", "lower")
        hi = _first_key(d, "hi", "ci_high", "high", "upper")
        if lo is None or hi is None:
            raise KeyError(f"estimate is missing its interval bounds: {sorted(d)}")
        return cls(
            mean=float(d["mean"]),
            lo=float(lo),
            hi=float(hi),
            n=int(d.get("n", 0)),
            method=str(d.get("method", "unknown")),
            ci_level=float(d.get("ci_level", DEFAULT_CI)),
            seed=d.get("seed"),
        )

    @classmethod
    def degenerate(cls, value: float, *, n: int = 1, method: str = "degenerate") -> Estimate:
        """A single observation: the interval is the point.  Honest, not useful."""
        v = float(value)
        return cls(mean=v, lo=v, hi=v, n=n, method=method)


def _first_key(d: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in d:
            return d[name]
    return None


def format_metric(value: Estimate, digits: int = 3) -> str:
    """Format an :class:`Estimate`.  Rejects bare floats loudly."""
    return require_estimate(value).format(digits)


def require_estimate(value: object, *, what: str = "value") -> Estimate:
    """Type guard used at every reporting boundary."""
    if isinstance(value, Estimate):
        return value
    raise TypeError(
        f"{what} must be an Estimate with a confidence interval, got "
        f"{type(value).__name__}. See implementation.md section 20 rule 5."
    )


def require_seeds(seeds: Sequence[int], *, minimum: int = MIN_SEEDS) -> list[int]:
    """Reject stochastic results computed from fewer than ``minimum`` seeds."""
    unique = list(dict.fromkeys(int(s) for s in seeds))
    if len(unique) < minimum:
        raise ValueError(
            f"stochastic results require >= {minimum} distinct seeds "
            f"(implementation.md section 20 rule 5); got {len(unique)}: {unique}"
        )
    return unique


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def _snap(point: float, lo: float, hi: float, tol: float = 1e-9) -> tuple[float, float]:
    """Absorb float noise so a reported interval always contains its estimate.

    Only rounding-scale violations are absorbed (Wilson at p=0 lands on
    ``1.4e-17`` instead of ``0``); a genuinely one-sided bootstrap distribution
    is left alone, because that is real information about skew.
    """
    if 0.0 < lo - point <= tol:
        lo = point
    if 0.0 < point - hi <= tol:
        hi = point
    return lo, hi


def percentile_stat(q: float) -> Callable[..., Any]:
    """A percentile statistic that bootstraps in a vectorised way."""

    def _stat(x: Any, axis: int | None = None) -> Any:
        return np.percentile(x, q, axis=axis, method="linear")

    _stat.__name__ = f"p{q:g}"
    return _stat


def _apply_statistic(block: np.ndarray, statistic: Callable[..., Any]) -> np.ndarray:
    """Apply ``statistic`` row-wise, vectorising when the callable allows it."""
    try:
        out = np.asarray(statistic(block, axis=1), dtype=float)
    except TypeError:
        out = np.asarray([float(statistic(row)) for row in block], dtype=float)
    if out.shape != (block.shape[0],):
        out = np.asarray([float(statistic(row)) for row in block], dtype=float)
    return out


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI,
    statistic: Callable[..., Any] = np.mean,
) -> Estimate:
    """Non-parametric percentile bootstrap CI.

    Deterministic for a given ``seed``: resampling uses
    ``np.random.default_rng(seed)`` and nothing else, so a seeded run
    reproduces bit-exactly (CONVENTIONS.md, determinism).
    """
    x = np.asarray(values, dtype=float).ravel()
    n = int(x.size)
    if n == 0:
        raise ValueError("bootstrap_ci requires at least one observation")
    point = float(np.asarray(statistic(x)))
    if n == 1:
        return Estimate(
            mean=point,
            lo=point,
            hi=point,
            n=1,
            method="degenerate-n1",
            ci_level=ci_level,
            seed=seed,
        )

    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=float)
    done = 0
    while done < n_resamples:
        size = min(_BOOTSTRAP_BLOCK, n_resamples - done)
        idx = rng.integers(0, n, size=(size, n))
        stats[done : done + size] = _apply_statistic(x[idx], statistic)
        done += size

    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.percentile(stats, [100.0 * alpha, 100.0 * (1.0 - alpha)], method="linear")
    lo, hi = _snap(point, float(lo), float(hi))
    return Estimate(
        mean=point,
        lo=lo,
        hi=hi,
        n=n,
        method="bootstrap-percentile",
        ci_level=ci_level,
        seed=seed,
    )


def wilson_ci(successes: int, trials: int, *, ci_level: float = DEFAULT_CI) -> Estimate:
    """Wilson score interval for a proportion -- better than bootstrap at small n."""
    if trials <= 0:
        raise ValueError("wilson_ci requires trials > 0")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes ({successes}) must be within [0, {trials}]")
    z = _norm_ppf(1.0 - (1.0 - ci_level) / 2.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    lo, hi = _snap(p, max(0.0, centre - margin), min(1.0, centre + margin))
    return Estimate(
        mean=p,
        lo=lo,
        hi=hi,
        n=trials,
        method="wilson",
        ci_level=ci_level,
    )


def mean_ci_t(values: Sequence[float], *, ci_level: float = DEFAULT_CI) -> Estimate:
    """Student-t CI on the mean.  Used for across-seed aggregation (tiny n)."""
    x = np.asarray(values, dtype=float).ravel()
    n = int(x.size)
    if n < 2:
        raise ValueError("mean_ci_t requires at least two observations")
    if abs(ci_level - 0.95) > 1e-9:
        raise ValueError("mean_ci_t only tabulates the 95% level; use bootstrap_ci otherwise")
    mean = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(n))
    t = _t_ppf_975(n - 1)
    return Estimate(
        mean=mean,
        lo=mean - t * se,
        hi=mean + t * se,
        n=n,
        method="student-t95",
        ci_level=ci_level,
    )


def aggregate_seeds(
    per_seed_values: Mapping[int, Sequence[float]] | Sequence[Sequence[float]],
    *,
    minimum_seeds: int = MIN_SEEDS,
) -> Estimate:
    """Aggregate a per-question metric measured under >= 3 seeds.

    The seed is the unit of replication, so the CI is a t interval over the
    per-seed means rather than a bootstrap over pooled questions -- pooling
    would hide seed-to-seed variance behind question-to-question variance.
    """
    if isinstance(per_seed_values, Mapping):
        require_seeds(list(per_seed_values.keys()), minimum=minimum_seeds)
        series: list[Sequence[float]] = list(per_seed_values.values())
    else:
        series = list(per_seed_values)
        if len(series) < minimum_seeds:
            raise ValueError(
                f"stochastic results require >= {minimum_seeds} seeds, got {len(series)}"
            )
    means = [float(np.mean(np.asarray(s, dtype=float))) for s in series]
    est = mean_ci_t(means)
    return Estimate(
        mean=est.mean,
        lo=est.lo,
        hi=est.hi,
        n=len(means),
        method=f"student-t95-over-{len(means)}-seeds",
        ci_level=est.ci_level,
    )


# --------------------------------------------------------------------------- #
# Paired tests
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairedBootstrapResult:
    """Paired bootstrap on the same questions: ``a - b``."""

    diff: Estimate
    p_value: float
    n: int
    seed: int

    @property
    def significant(self) -> bool:
        """CI excludes zero at the estimate's level."""
        return self.diff.lo > 0.0 or self.diff.hi < 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "diff": self.diff.to_dict(),
            "p_value": self.p_value,
            "n": self.n,
            "seed": self.seed,
            "significant": self.significant,
        }


def bootstrap_paired(
    a: Sequence[float],
    b: Sequence[float],
    *,
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI,
    statistic: Callable[..., Any] = np.mean,
) -> PairedBootstrapResult:
    """Bootstrap the paired difference by resampling *pairs*, never elements."""
    xa = np.asarray(a, dtype=float).ravel()
    xb = np.asarray(b, dtype=float).ravel()
    if xa.shape != xb.shape:
        raise ValueError(f"paired inputs must align: {xa.shape} vs {xb.shape}")
    n = int(xa.size)
    if n == 0:
        raise ValueError("bootstrap_paired requires at least one pair")
    d = xa - xb
    point = float(np.asarray(statistic(d)))
    if n == 1:
        return PairedBootstrapResult(
            diff=Estimate.degenerate(point, n=1, method="degenerate-n1"),
            p_value=1.0,
            n=1,
            seed=seed,
        )

    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=float)
    done = 0
    while done < n_resamples:
        size = min(_BOOTSTRAP_BLOCK, n_resamples - done)
        idx = rng.integers(0, n, size=(size, n))
        stats[done : done + size] = _apply_statistic(d[idx], statistic)
        done += size

    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.percentile(stats, [100.0 * alpha, 100.0 * (1.0 - alpha)], method="linear")
    lo, hi = _snap(point, float(lo), float(hi))
    # Two-sided achieved significance level: recentre the bootstrap distribution
    # on the null and count draws at least as extreme as the observed effect.
    centred = stats - point
    p_ge = float(np.mean(centred >= abs(point)))
    p_le = float(np.mean(centred <= -abs(point)))
    p = min(1.0, p_ge + p_le)
    return PairedBootstrapResult(
        diff=Estimate(
            mean=point,
            lo=lo,
            hi=hi,
            n=n,
            method="bootstrap-paired-percentile",
            ci_level=ci_level,
            seed=seed,
        ),
        p_value=p,
        n=n,
        seed=seed,
    )


@dataclass(frozen=True)
class WilcoxonResult:
    """Wilcoxon signed-rank test on paired observations."""

    statistic: float
    p_value: float
    n_effective: int
    method: Literal["exact", "normal-approx", "scipy"]
    backend: str = "numpy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n_effective": self.n_effective,
            "method": self.method,
            "backend": self.backend,
        }


def wilcoxon_signed_rank(
    a: Sequence[float],
    b: Sequence[float] | None = None,
    *,
    use_scipy: bool = True,
    exact_max_n: int = 25,
) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test.

    Zero differences are dropped (the classic ``wilcox`` zero method) and the
    reported statistic is ``min(W+, W-)``.  An exact p-value is computed by
    dynamic programming over the null distribution of ``W+`` when there are no
    ties among the non-zero ``|d|`` and ``n <= exact_max_n``; otherwise a
    tie-corrected normal approximation with continuity correction is used.

    ``scipy`` is imported lazily and only used if present -- the pure-numpy
    path is a complete implementation, not a stub.
    """
    xa = np.asarray(a, dtype=float).ravel()
    if b is None:
        d = xa
    else:
        xb = np.asarray(b, dtype=float).ravel()
        if xa.shape != xb.shape:
            raise ValueError(f"paired inputs must align: {xa.shape} vs {xb.shape}")
        d = xa - xb

    d = d[d != 0.0]
    n = int(d.size)
    if n == 0:
        return WilcoxonResult(statistic=0.0, p_value=1.0, n_effective=0, method="exact")

    absd = np.abs(d)
    ranks = _average_ranks(absd)
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    stat = min(w_plus, w_minus)
    has_ties = bool(np.unique(absd).size != n)

    if use_scipy:
        p_scipy = _scipy_wilcoxon_p(d)
        if p_scipy is not None:
            return WilcoxonResult(
                statistic=stat,
                p_value=p_scipy,
                n_effective=n,
                method="scipy",
                backend="scipy",
            )

    if not has_ties and n <= exact_max_n:
        return WilcoxonResult(
            statistic=stat,
            p_value=_wilcoxon_exact_p(ranks, w_plus),
            n_effective=n,
            method="exact",
        )

    return WilcoxonResult(
        statistic=stat,
        p_value=_wilcoxon_normal_p(ranks, absd, w_plus, n),
        n_effective=n,
        method="normal-approx",
    )


def _scipy_wilcoxon_p(d: np.ndarray) -> float | None:
    try:  # pragma: no cover - only exercised where scipy is installed
        from scipy.stats import wilcoxon as _w
    except Exception:
        return None
    try:  # pragma: no cover
        result = _w(d, zero_method="wilcox", alternative="two-sided")
        # scipy wraps `wilcoxon` in an internal axis/nan-policy decorator whose
        # stub-inferred return type (pyright reports it as class "_") loses the
        # `pvalue` attribute; the actual runtime object is scipy's WilcoxonResult /
        # SignificanceResult, which always carries it.
        return float(cast(Any, result).pvalue)
    except Exception:
        return None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties averaged (the mid-rank convention)."""
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _wilcoxon_exact_p(ranks: np.ndarray, w_plus: float) -> float:
    """Exact two-sided p by DP over the null distribution of ``W+``.

    Under H0 each sign is +/- with probability 1/2, so ``W+`` is a sum of
    independent Bernoulli-weighted ranks.  Ranks are doubled to stay integral
    in the presence of half-ranks.
    """
    weights = np.rint(ranks * 2).astype(int)
    total = int(weights.sum())
    counts = np.zeros(total + 1, dtype=np.float64)
    counts[0] = 1.0
    for w in weights:
        shifted = np.zeros_like(counts)
        shifted[w:] = counts[: total + 1 - w]
        counts = counts + shifted
    n_outcomes = 2.0 ** len(weights)
    target = round(float(w_plus) * 2)
    p_le = float(counts[: target + 1].sum()) / n_outcomes
    p_ge = float(counts[target:].sum()) / n_outcomes
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def _wilcoxon_normal_p(ranks: np.ndarray, absd: np.ndarray, w_plus: float, n: int) -> float:
    del ranks  # ranks enter only through w_plus; kept for signature symmetry
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    _, tie_counts = np.unique(absd, return_counts=True)
    var -= float(np.sum(tie_counts.astype(float) ** 3 - tie_counts)) / 48.0
    if var <= 0:
        return 1.0
    sigma = math.sqrt(var)
    delta = w_plus - mu
    cc = 0.5 if delta > 0 else (-0.5 if delta < 0 else 0.0)
    z = (delta - cc) / sigma
    return float(min(1.0, 2.0 * _norm_sf(abs(z))))


@dataclass(frozen=True)
class ComparisonReport:
    """Head-to-head of two configs measured on the same questions."""

    name_a: str
    name_b: str
    metric: str
    estimate_a: Estimate
    estimate_b: Estimate
    paired: PairedBootstrapResult
    wilcoxon: WilcoxonResult

    @property
    def verdict(self) -> str:
        if not self.paired.significant and self.wilcoxon.p_value >= 0.05:
            return f"no significant difference in {self.metric}"
        winner = self.name_a if self.paired.diff.mean > 0 else self.name_b
        return f"{winner} wins on {self.metric}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "a": {"name": self.name_a, **self.estimate_a.to_dict()},
            "b": {"name": self.name_b, **self.estimate_b.to_dict()},
            "paired_bootstrap": self.paired.to_dict(),
            "wilcoxon": self.wilcoxon.to_dict(),
            "verdict": self.verdict,
        }

    def to_markdown(self) -> str:
        return "\n".join(
            [
                f"| config | {self.metric} (95% CI) |",
                "|---|---|",
                f"| {self.name_a} | {self.estimate_a.format()} |",
                f"| {self.name_b} | {self.estimate_b.format()} |",
                f"| delta (a-b) | {self.paired.diff.format()} |",
                "",
                f"paired bootstrap p={self.paired.p_value:.4f} "
                f"(n={self.paired.n}, seed={self.paired.seed}); "
                f"Wilcoxon W={self.wilcoxon.statistic:g} p={self.wilcoxon.p_value:.4f} "
                f"({self.wilcoxon.method}); verdict: {self.verdict}",
            ]
        )


def compare_configs(
    name_a: str,
    values_a: Sequence[float],
    name_b: str,
    values_b: Sequence[float],
    *,
    metric: str = "metric",
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
) -> ComparisonReport:
    """Both paired tests the spec asks for, on the same questions, in one call."""
    return ComparisonReport(
        name_a=name_a,
        name_b=name_b,
        metric=metric,
        estimate_a=bootstrap_ci(values_a, seed=seed, n_resamples=n_resamples),
        estimate_b=bootstrap_ci(values_b, seed=seed, n_resamples=n_resamples),
        paired=bootstrap_paired(values_a, values_b, seed=seed, n_resamples=n_resamples),
        wilcoxon=wilcoxon_signed_rank(values_a, values_b),
    )


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #
def cohens_kappa(
    rater_a: Sequence[Any],
    rater_b: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
) -> float:
    """Cohen's kappa: chance-corrected agreement between two raters.

    ``kappa = (p_o - p_e) / (1 - p_e)`` where ``p_e`` comes from the product of
    the two raters' marginals.  Returns 1.0 when both raters are constant and
    identical (perfect agreement, no variance left to correct for) and 0.0 when
    they are constant and different.
    """
    a = list(rater_a)
    b = list(rater_b)
    if len(a) != len(b):
        raise ValueError(f"rater sequences must align: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        raise ValueError("cohens_kappa requires at least one item")

    cats = list(labels) if labels is not None else sorted(set(a) | set(b), key=repr)
    index = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    conf = np.zeros((k, k), dtype=float)
    for x, y in zip(a, b, strict=True):
        if x not in index or y not in index:
            raise ValueError(f"label outside declared label set: {x!r}/{y!r} not in {cats!r}")
        conf[index[x], index[y]] += 1.0

    p_o = float(np.trace(conf)) / n
    marg_a = conf.sum(axis=1) / n
    marg_b = conf.sum(axis=0) / n
    p_e = float(np.dot(marg_a, marg_b))
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def kappa_ci(
    rater_a: Sequence[Any],
    rater_b: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
    seed: int = DEFAULT_SEED,
    n_resamples: int = 2_000,
    ci_level: float = DEFAULT_CI,
) -> Estimate:
    """Bootstrap CI on Cohen's kappa by resampling *items*, keeping rater pairs."""
    a = list(rater_a)
    b = list(rater_b)
    n = len(a)
    if n != len(b) or n == 0:
        raise ValueError("kappa_ci requires aligned, non-empty rater sequences")
    cats = list(labels) if labels is not None else sorted(set(a) | set(b), key=repr)
    point = cohens_kappa(a, b, labels=cats)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        draws[i] = cohens_kappa([a[j] for j in idx], [b[j] for j in idx], labels=cats)
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = _snap(
        point, *np.percentile(draws, [100.0 * alpha, 100.0 * (1.0 - alpha)], method="linear")
    )
    return Estimate(
        mean=point,
        lo=lo,
        hi=hi,
        n=n,
        method="bootstrap-percentile-kappa",
        ci_level=ci_level,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Reporting boundary (CONVENTIONS.md "Reporting")
# --------------------------------------------------------------------------- #
@dataclass
class MetricRow:
    """One row of a benchmark table.  Values must be Estimates."""

    name: str
    values: dict[str, Estimate] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        for key, value in self.values.items():
            out[key] = require_estimate(value, what=f"row {self.name!r} metric {key!r}").to_dict()
        out.update(self.extra)
        return out


def benchmark_json(
    name: str,
    *,
    hardware: str,
    seeds: Sequence[int],
    rows: Sequence[MetricRow],
    ci: str = "bootstrap95",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the artifact envelope CONVENTIONS.md mandates.

    Every metric value is validated to be an :class:`Estimate`, so a bare
    number cannot reach ``artifacts/benchmarks/*.json`` at all.
    """
    payload: dict[str, Any] = {
        "name": name,
        "hardware": hardware,
        "seeds": [int(s) for s in seeds],
        "rows": [r.to_dict() for r in rows],
        "ci": ci,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def rows_to_markdown(rows: Sequence[MetricRow], columns: Sequence[str], digits: int = 3) -> str:
    """Markdown table with every cell rendered as ``mean [lo, hi]``."""
    header = "| " + " | ".join(["system", *columns]) + " |"
    sep = "|" + "---|" * (len(columns) + 1)
    lines = [header, sep]
    for row in rows:
        cells = [
            format_metric(row.values[col], digits) if col in row.values else "n/a"
            for col in columns
        ]
        lines.append("| " + " | ".join([row.name, *cells]) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Distribution helpers (no scipy)
# --------------------------------------------------------------------------- #
def _norm_sf(z: float) -> float:
    """Upper tail of the standard normal, via the stdlib erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


_ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam), refined by one Halley step."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"_norm_ppf requires 0 < p < 1, got {p}")
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (
            (
                (((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q
                + _ACKLAM_C[4]
            )
            * q
            + _ACKLAM_C[5]
        ) / ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        num = (
            (
                (((_ACKLAM_A[0] * r + _ACKLAM_A[1]) * r + _ACKLAM_A[2]) * r + _ACKLAM_A[3]) * r
                + _ACKLAM_A[4]
            )
            * r
            + _ACKLAM_A[5]
        ) * q
        den = (
            (((_ACKLAM_B[0] * r + _ACKLAM_B[1]) * r + _ACKLAM_B[2]) * r + _ACKLAM_B[3]) * r
            + _ACKLAM_B[4]
        ) * r + 1.0
        x = num / den
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(
            (
                (((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q
                + _ACKLAM_C[4]
            )
            * q
            + _ACKLAM_C[5]
        ) / ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0)
    e = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


_T_975 = {
    1: 12.7062,
    2: 4.3027,
    3: 3.1824,
    4: 2.7764,
    5: 2.5706,
    6: 2.4469,
    7: 2.3646,
    8: 2.3060,
    9: 2.2622,
    10: 2.2281,
    11: 2.2010,
    12: 2.1788,
    13: 2.1604,
    14: 2.1448,
    15: 2.1314,
    16: 2.1199,
    17: 2.1098,
    18: 2.1009,
    19: 2.0930,
    20: 2.0860,
    21: 2.0796,
    22: 2.0739,
    23: 2.0687,
    24: 2.0639,
    25: 2.0595,
    26: 2.0555,
    27: 2.0518,
    28: 2.0484,
    29: 2.0452,
    30: 2.0423,
    40: 2.0211,
    50: 2.0086,
    60: 2.0003,
    80: 1.9901,
    100: 1.9840,
    120: 1.9799,
}


def _t_ppf_975(df: int) -> float:
    """Two-sided 95% t quantile.  Table for small df, normal limit beyond."""
    if df <= 0:
        raise ValueError("df must be positive")
    if df in _T_975:
        return _T_975[df]
    if df > 120:
        return 1.9600
    keys = sorted(_T_975)
    lo = max(k for k in keys if k <= df)
    hi = min(k for k in keys if k >= df)
    if lo == hi:
        return _T_975[lo]
    w = (df - lo) / (hi - lo)
    return _T_975[lo] * (1 - w) + _T_975[hi] * w
