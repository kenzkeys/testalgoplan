"""Statistics that charge for the search, not just the result."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649


def sharpe(r: pd.Series | np.ndarray, ppy: int) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(ppy)) if sd > 0 else 0.0


def max_drawdown(r: pd.Series) -> float:
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def cagr(r: pd.Series, ppy: int) -> float:
    n = len(r)
    return float((1.0 + r).prod() ** (ppy / n) - 1.0) if n else float("nan")


def stationary_bootstrap_idx(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices (mean block length `block`)."""
    p = 1.0 / block
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(n)
    jumps = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for i in range(1, n):
        idx[i] = starts[i] if jumps[i] else (idx[i - 1] + 1) % n
    return idx


def bootstrap_sharpe_ci(r: pd.Series, ppy: int, reps: int = 2000, block: int = 20, seed: int = 7):
    """95% CI of annualised Sharpe and P(Sharpe <= 0), robust to autocorrelation."""
    x = r.to_numpy()
    rng = np.random.default_rng(seed)
    sr = np.empty(reps)
    for k in range(reps):
        s = x[stationary_bootstrap_idx(len(x), block, rng)]
        sd = s.std(ddof=1)
        sr[k] = s.mean() / sd * math.sqrt(ppy) if sd > 0 else 0.0
    return float(np.percentile(sr, 2.5)), float(np.percentile(sr, 97.5)), float((sr <= 0).mean())


def bootstrap_active_p(strategy: pd.Series, benchmark: pd.Series, ppy: int,
                       reps: int = 2000, block: int = 20, seed: int = 11) -> float:
    """P(strategy does not beat benchmark): bootstrap of the daily active return mean."""
    d = (strategy - benchmark).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for k in range(reps):
        means[k] = d[stationary_bootstrap_idx(len(d), block, rng)].mean()
    return float((means <= 0).mean())


def expected_max_sharpe(trials: int, trial_sr_std: float) -> float:
    """Expected maximum Sharpe of `trials` independent zero-skill strategies
    (Bailey and Lopez de Prado). Units: same per-period units as `trial_sr_std`."""
    if trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (trials * math.e))
    return trial_sr_std * ((1.0 - EULER) * z1 + EULER * z2)


def deflated_sharpe(r: pd.Series, trials: int, trial_sr_std: float) -> float:
    """Probability the true Sharpe exceeds what the best of `trials` noise strategies
    would show. All Sharpes per-period (daily). Need > 0.95."""
    x = r.to_numpy()
    n = len(x)
    if n < 30 or x.std(ddof=1) == 0:
        return float("nan")
    sr = x.mean() / x.std(ddof=1)
    sr0 = expected_max_sharpe(trials, trial_sr_std)
    g3 = stats.skew(x)
    g4 = stats.kurtosis(x, fisher=False)
    denom = math.sqrt(max(1e-12, 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(n - 1) / denom))


def concentration_by_year(r: pd.Series) -> tuple[float, str]:
    """Largest single year's share of total positive annual return. > 0.5 fails."""
    yearly = (1.0 + r).groupby(r.index.year).prod() - 1.0
    pos = yearly[yearly > 0]
    if pos.empty:
        return float("nan"), "none"
    share = pos.max() / pos.sum()
    return float(share), str(pos.idxmax())
