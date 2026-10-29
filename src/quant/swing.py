"""Daily-bar, single-instrument, long-only trade engine and the SQX-style robustness tests
for docs/00_RESEARCH_PLAN.md Amendment 5.

A strategy is a signal function (OHLCV frame -> entry flags, optional exit flags, both
decided on the close of their bar) plus an exit geometry (entry mode, holding period, ATR
stop). The engine holds one position at a time and ignores signals while in a trade.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd

from . import stats

PPY = 252
COST = 2e-4  # per side, fraction of price


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    """Wilder RSI. Recursive smoothing, so a value never depends on later bars."""
    d = close.diff()
    up = d.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d).clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    out = 100.0 - 100.0 / (1.0 + up / dn)
    return out.where(dn > 0, 100.0).where(up.notna())


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


Signal = Callable[[pd.DataFrame], tuple[pd.Series, "pd.Series | None"]]


@dataclass(frozen=True)
class Strategy:
    name: str
    signal: Signal
    entry: str = "open"            # "open": fill next bar's open; "close": fill this bar's close
    hold: int | None = None        # exit at the close of the hold-th bar of the trade
    stop_atr: float | None = None  # stop = entry price - stop_atr * ATR(14) at the signal bar
    params: tuple = ()

    def with_(self, **kw) -> "Strategy":
        return replace(self, **kw)


class Bars:
    """Plain arrays, so the trade loop does not pay pandas indexing costs."""

    def __init__(self, df: pd.DataFrame):
        self.index = df.index
        self.o, self.h, self.l, self.c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.atr = atr(df).to_numpy()
        self.n = len(df)


def simulate(b: Bars, t: int, entry: str, hold: int | None, stop_atr: float | None,
             cost: float, exits: np.ndarray | None = None):
    """One long trade signalled on the close of bar t.

    Returns (first_held_bar, exit_bar, fill_net, exit_net, reason) or None when the trade
    cannot start inside the data. fill_net and exit_net include costs."""
    n = b.n
    if entry == "open":
        if t + 1 >= n:
            return None
        raw = b.o[t + 1]
    elif entry == "close":
        if t + 1 >= n:  # nothing to hold over
            return None
        raw = b.c[t]
    else:
        raise ValueError(entry)
    j0 = t + 1
    stop = None
    if stop_atr is not None:
        if not np.isfinite(b.atr[t]):
            return None
        stop = raw - stop_atr * b.atr[t]
    fill_net = raw * (1.0 + cost)
    for j in range(j0, n):
        if stop is not None:
            # on an open entry the fill is bar j0's open, which is above the stop by construction
            if b.o[j] <= stop and not (entry == "open" and j == j0):
                return j0, j, fill_net, b.o[j] * (1.0 - cost), "stop_gap"
            if b.l[j] <= stop:
                return j0, j, fill_net, stop * (1.0 - cost), "stop"
        if hold is not None and j - j0 + 1 >= hold:
            return j0, j, fill_net, b.c[j] * (1.0 - cost), "time"
        if exits is not None and exits[j]:
            return j0, j, fill_net, b.c[j] * (1.0 - cost), "signal"
    return j0, n - 1, fill_net, b.c[n - 1] * (1.0 - cost), "end"


def mark(b: Bars, t: int, entry: str, j0: int, exit_bar: int, fill_net: float, exit_net: float,
         daily: np.ndarray, exposure: np.ndarray | None = None) -> None:
    """Add one trade's mark-to-market daily returns. Their product equals exit_net / fill_net."""
    prev = fill_net
    if entry == "close":
        daily[t] += b.c[t] / prev - 1.0
        prev = b.c[t]
    for j in range(j0, exit_bar + 1):
        m = exit_net if j == exit_bar else b.c[j]
        daily[j] += m / prev - 1.0
        prev = m
        if exposure is not None:
            exposure[j] += 1.0


@dataclass
class Run:
    trades: pd.DataFrame
    daily: pd.Series
    exposure: pd.Series


def backtest(df: pd.DataFrame, s: Strategy, cost: float = COST) -> Run:
    entries, exits = s.signal(df)
    ent = entries.fillna(False).to_numpy(bool)
    ex = None if exits is None else exits.fillna(False).to_numpy(bool)
    b = Bars(df)
    daily = np.zeros(b.n)
    expo = np.zeros(b.n)
    rows = []
    free_from = 0  # first signal bar allowed (one position at a time)
    for t in np.flatnonzero(ent):
        if t < free_from:
            continue
        res = simulate(b, t, s.entry, s.hold, s.stop_atr, cost, ex)
        if res is None:
            continue
        j0, k, fn, xn, why = res
        mark(b, t, s.entry, j0, k, fn, xn, daily, expo)
        rows.append({"signal": b.index[t], "t": t, "exit": b.index[k], "bars": k - j0 + 1,
                     "ret": xn / fn - 1.0, "reason": why})
        # an open entry may be signalled on the exit bar; a close entry needs a later close
        free_from = k if s.entry == "open" else k + 1
    return Run(pd.DataFrame(rows, columns=["signal", "t", "exit", "bars", "ret", "reason"]),
               pd.Series(daily, index=df.index), pd.Series(expo, index=df.index))


# ---------------------------------------------------------------- robustness tests

def bootstrap_mean_ci(x: np.ndarray, reps: int = 5000, seed: int = 17) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(len(x), size=(reps, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def random_entry_p(df: pd.DataFrame, s: Strategy, trades: pd.DataFrame, cost: float = COST,
                   sims: int = 1000, seed: int = 23) -> tuple[float, float]:
    """Same trade count, each trade keeping its real holding length and the stop multiple,
    entered at random bars. Returns (p, mean of the random runs' mean trade return)."""
    if trades.empty:
        return float("nan"), float("nan")
    b = Bars(df)
    holds = trades["bars"].to_numpy(int)
    real = trades["ret"].mean()
    lo = int(trades["t"].min())  # the earliest bar on which this strategy could signal
    cache: dict[int, np.ndarray] = {}
    for h in np.unique(holds):
        out = np.full(b.n, np.nan)
        for t in range(lo, b.n - h):
            r = simulate(b, t, s.entry, int(h), s.stop_atr, cost)
            if r is not None:
                out[t] = r[3] / r[2] - 1.0
        cache[int(h)] = out
    rng = np.random.default_rng(seed)
    means = np.empty(sims)
    for k in range(sims):
        tot = 0.0
        for h in holds:
            pool = cache[int(h)]
            v = np.nan
            while not np.isfinite(v):
                v = pool[rng.integers(lo, b.n - h)]
            tot += v
        means[k] = tot / len(holds)
    return float((1 + (means >= real).sum()) / (sims + 1)), float(means.mean())


def shifted_daily(df: pd.DataFrame, s: Strategy, trades: pd.DataFrame, offset: int,
                  cost: float = COST) -> np.ndarray:
    """The strategy's trade schedule (entry bars, holding lengths) moved circularly by
    `offset` bars inside the window. Keeps count, spacing and exposure, breaks timing."""
    b = Bars(df)
    daily = np.zeros(b.n)
    lo = int(trades["t"].min()) if len(trades) else 0
    span = b.n - lo
    for t, h in zip(trades["t"].to_numpy(int), trades["bars"].to_numpy(int)):
        u = lo + (t - lo + offset) % span
        r = simulate(b, u, s.entry, int(h), s.stop_atr, cost)
        if r is not None:
            mark(b, u, s.entry, r[0], r[1], r[2], r[3], daily)
    return daily


def perturb(df: pd.DataFrame, rng: np.random.Generator, scale: float = 0.1) -> pd.DataFrame:
    """Multiply every O/H/L/C by (1 + e), e ~ N(0, scale * ATR/close), then widen high and
    low to contain open and close. Volume is unchanged."""
    a = (atr(df) / df["close"]).bfill().to_numpy()
    out = df.copy()
    for k in ("open", "high", "low", "close"):
        out[k] = df[k].to_numpy() * (1.0 + rng.normal(0.0, 1.0, len(df)) * scale * a)
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    return out


def noise_pass_rate(df: pd.DataFrame, s: Strategy, runs: int = 200, cost: float = COST,
                    seed: int = 29) -> float:
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(runs):
        tr = backtest(perturb(df, rng), s, cost).trades
        ok += bool(len(tr) and tr["ret"].mean() > 0)
    return ok / runs


def monte_carlo(rets: np.ndarray, sims: int = 2000, seed: int = 31) -> dict:
    """Reshuffle the trade order: same final equity, different paths."""
    rng = np.random.default_rng(seed)
    dds, streaks = np.empty(sims), np.empty(sims)
    for k in range(sims):
        r = rng.permutation(rets)
        eq = np.cumprod(1.0 + r)
        dds[k] = (eq / np.maximum.accumulate(eq) - 1.0).min()
        loss = r < 0
        best = cur = 0
        for x in loss:
            cur = cur + 1 if x else 0
            best = max(best, cur)
        streaks[k] = best
    return {"max_dd_p95": float(np.percentile(dds, 5)), "max_dd_median": float(np.median(dds)),
            "losing_streak_p95": float(np.percentile(streaks, 95))}


def video_size(rets: np.ndarray, avg_loss_budget: float = 0.005) -> float:
    """Fraction of equity per trade that makes the average loser cost 0.5% (no leverage)."""
    losers = rets[rets < 0]
    if not len(losers):
        return 1.0
    return float(min(1.0, avg_loss_budget / abs(losers.mean())))


def summary(df: pd.DataFrame, run: Run) -> dict:
    tr, r = run.trades, run.daily
    bh = df["close"].pct_change().fillna(0.0)
    rets = tr["ret"].to_numpy() if len(tr) else np.array([])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    share, year = stats.concentration_by_year(r)
    return {
        "trades": int(len(tr)), "mean_trade": float(rets.mean()) if len(rets) else float("nan"),
        "win_rate": float((rets > 0).mean()) if len(rets) else float("nan"),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "exposure": float((run.exposure > 0).mean()),
        "cagr": stats.cagr(r, PPY), "sharpe": stats.sharpe(r, PPY), "max_dd": stats.max_drawdown(r),
        "bh_cagr": stats.cagr(bh, PPY), "bh_sharpe": stats.sharpe(bh, PPY), "bh_max_dd": stats.max_drawdown(bh),
        "top_year_share": share, "top_year": year,
        "exit_reasons": tr["reason"].value_counts().to_dict() if len(tr) else {},
    }
