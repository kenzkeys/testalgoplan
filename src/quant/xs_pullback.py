"""The video's ranked S&P 100 pullback (Amendment 5, `sp_rsi14` and `sp_rsi2`).

At each close: held names exit on a close above SMA(5) or after `max_hold` days; then free
slots fill with eligible names above SMA(250) whose RSI is below the threshold, lowest RSI
first. Weight row t is decided at the close of t and the backtester holds it over t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .swing import rsi


def indicators(px: pd.DataFrame, rsi_n: int, exit_sma: int = 5, trend_sma: int = 250):
    r = px.apply(lambda s: rsi(s.dropna(), rsi_n).reindex(s.index))
    return (r.to_numpy(), (px > px.rolling(trend_sma, min_periods=trend_sma).mean()).to_numpy(),
            (px > px.rolling(exit_sma, min_periods=exit_sma).mean()).to_numpy())


def pullback(px: pd.DataFrame, eligible: pd.DataFrame, rsi_n: int = 14, threshold: float = 30,
             max_hold: int = 10, slots: int = 10, ind=None) -> tuple[pd.DataFrame, list]:
    """Returns (weights, trades) with trades as (entry_row, rows_held, column)."""
    r, up, above_exit = ind if ind is not None else indicators(px, rsi_n)
    ok_px = px.notna().to_numpy()
    elig = eligible.reindex_like(px).fillna(False).to_numpy(bool)
    n, k = px.shape
    w = np.zeros((n, k))
    held: dict[int, int] = {}  # column -> entry row
    trades = []
    for t in range(n):
        for j in list(held):
            age = t - held[j]
            if not ok_px[t, j] or above_exit[t, j] or age >= max_hold:
                trades.append((held[j], age, j))
                del held[j]
        free = slots - len(held)
        if free > 0:
            cand = elig[t] & up[t] & ok_px[t] & (r[t] < threshold)
            for j in list(held):
                cand[j] = False
            idx = np.flatnonzero(cand)
            if len(idx):
                for j in idx[np.argsort(r[t, idx], kind="stable")][:free]:
                    held[int(j)] = t
        for j in held:
            w[t, j] = 1.0 / slots
    for j, t0 in held.items():
        trades.append((t0, n - 1 - t0, j))
    return pd.DataFrame(w, index=px.index, columns=px.columns), trades


def random_twin(px: pd.DataFrame, eligible: pd.DataFrame, trades: list, slots: int,
                rng: np.random.Generator) -> pd.DataFrame:
    """Same entry rows and holding lengths, each name drawn at random from that row's
    eligible names that are priced and not already held."""
    ok = eligible.reindex_like(px).fillna(False).to_numpy(bool) & px.notna().to_numpy()
    n, k = px.shape
    w = np.zeros((n, k))
    busy_until = np.full(k, -1)
    for t0, age, _ in sorted(trades):
        pool = np.flatnonzero(ok[t0] & (busy_until < t0))
        if not len(pool):
            continue
        j = rng.choice(pool)
        end = t0 + max(age, 1)  # weight rows t0..end-1 are held, like the real trade
        w[t0:end, j] = 1.0 / slots
        busy_until[j] = end - 1
    return pd.DataFrame(w, index=px.index, columns=px.columns)
