"""Strategies as pure functions: prices -> target weights.

Convention, enforced by tests/test_lookahead.py: the weight in row t is computed from
data up to and including the close of day t, and is HELD over day t+1. The backtester
applies the one-day lag; a strategy never sees a return it is paid on.

Every weight row sums to at most 1 in absolute gross terms per leg (long-only gross 1,
long-short +0.5 / -0.5), so all strategies are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rebalance_mask(n: int, every: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[::every] = True
    return m


def select_top(scores: pd.DataFrame, top_frac: float, every: int, long_short: bool,
               min_names: int = 5) -> pd.DataFrame:
    """Equal-weight the top (and optionally bottom) fraction of `scores` on rebalance days.

    Scores that are NaN (no price history) are never selected. Between rebalances the
    target weights are carried forward unchanged.
    """
    n = len(scores)
    mask = _rebalance_mask(n, every)
    w = pd.DataFrame(np.nan, index=scores.index, columns=scores.columns)
    vals = scores.to_numpy()
    out = np.full(vals.shape, np.nan)
    for i in np.flatnonzero(mask):
        row = vals[i]
        ok = np.isfinite(row)
        k_avail = int(ok.sum())
        if k_avail < min_names:
            out[i] = 0.0
            continue
        k = max(1, int(round(k_avail * top_frac)))
        order = np.argsort(np.where(ok, row, -np.inf))
        longs = order[-k:]
        wi = np.zeros(vals.shape[1])
        if long_short:
            order_asc = np.argsort(np.where(ok, row, np.inf))
            shorts = order_asc[:k]
            wi[longs] = 0.5 / k
            wi[shorts] = -0.5 / k
        else:
            wi[longs] = 1.0 / k
        out[i] = wi
    w[:] = out
    return w.ffill().fillna(0.0)


def momentum_scores(prices: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """Return from t-lookback to t-skip. skip>0 drops the most recent days, which
    carry short-term reversal rather than momentum."""
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def xs_momentum(prices, lookback=252, skip=21, top_frac=0.1, every=21, long_short=False):
    return select_top(momentum_scores(prices, lookback, skip), top_frac, every, long_short)


def st_reversal(prices, lookback=5, top_frac=0.1, every=5, long_short=False):
    """Buy the recent losers."""
    return select_top(-(prices / prices.shift(lookback) - 1.0), top_frac, every, long_short)


def ts_momentum(prices, lookback=90, every=7):
    """Each asset is held (equal weight of the universe) when its own trailing
    return is positive, else that slice sits in cash."""
    n_assets = prices.shape[1]
    sig = (prices / prices.shift(lookback) - 1.0 > 0).astype(float)
    mask = _rebalance_mask(len(prices), every)
    sig = sig.where(pd.Series(mask, index=prices.index), np.nan).ffill().fillna(0.0)
    return sig / n_assets


def equal_weight_hold(prices) -> pd.DataFrame:
    """Benchmark: equal weight of every name with a price today."""
    avail = prices.notna().astype(float)
    return avail.div(avail.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def trend_overlay(prices, ma=200):
    """Hold the equal-weight universe only while its synthetic index is above its
    moving average. Mostly a drawdown control."""
    ew = equal_weight_hold(prices)
    idx = (prices.pct_change().fillna(0.0) * ew.shift(1).fillna(0.0)).sum(axis=1)
    level = (1.0 + idx).cumprod()
    on = (level > level.rolling(ma).mean()).astype(float)
    return ew.mul(on, axis=0)


# Pre-declared parameter grids. The size of each grid is what Deflated Sharpe charges.
# Nothing outside these lists may be evaluated without amending docs/00_RESEARCH_PLAN.md.
FAMILIES = {
    "equities": {
        "xs_mom_12_1":     lambda p: xs_momentum(p, 252, 21),
        "xs_mom_6_1":      lambda p: xs_momentum(p, 126, 21),
        "xs_mom_12_1_ls":  lambda p: xs_momentum(p, 252, 21, long_short=True),
        "st_reversal_5d":  lambda p: st_reversal(p, 5, 0.1, 5),
        "trend_ew_200":    lambda p: trend_overlay(p, 200),
    },
    "crypto": {
        "ts_mom_30":       lambda p: ts_momentum(p, 30),
        "ts_mom_90":       lambda p: ts_momentum(p, 90),
        "ts_mom_180":      lambda p: ts_momentum(p, 180),
        "xs_mom_90_7":     lambda p: xs_momentum(p, 90, 7, top_frac=0.3, every=7),
        "xs_mom_30_7":     lambda p: xs_momentum(p, 30, 7, top_frac=0.3, every=7),
        "trend_ew_100":    lambda p: trend_overlay(p, 100),
    },
}
