"""Backtest of the YouTube EMA(5/8) crossover bot, as specified in docs/00_RESEARCH_PLAN.md
Amendment 2. Trade-level, in R units (R = net price move / ATR(14) at the signal).

Bar t is the signal candle (closed). The order is a market order filled at the open of
t+1. Stop and target are absolute prices fixed from close[t] and ATR[t].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZON = 2000  # bars after which an unresolved trade is closed at the close


def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


@dataclass(frozen=True)
class Spec:
    fast: int = 5
    slow: int = 8
    atr_n: int = 14
    rr: float = 1.5
    warmup: int = 50
    stop_mult: float = 1.0  # stop distance = stop_mult * ATR; R is in units of this distance
    # Amendment 7 levers. Defaults reproduce Amendments 2 and 3 exactly.
    exit: str = "target"    # "target" | "trail" (chandelier, long only) | "time" (long only)
    hold: int = 10          # bars, for exit="time"
    filt: str = "none"      # "none" | "trend" | "costfloor" | "trend+costfloor"
    trend_n: int = 200
    cost_share: float = 0.05  # costfloor: round-trip cost <= this share of the stop distance


def signals(df: pd.DataFrame, spec: Spec = Spec()) -> pd.DataFrame:
    """+1 where EMA(fast) crossed above EMA(slow) on this closed bar, -1 for the mirror."""
    f, s = ema(df["close"], spec.fast), ema(df["close"], spec.slow)
    up = (f > s) & (f.shift(1) < s.shift(1))
    dn = (f < s) & (f.shift(1) > s.shift(1))
    out = pd.DataFrame(index=df.index)
    out["signal"] = np.where(up, 1, np.where(dn, -1, 0))
    out["atr"] = atr(df, spec.atr_n)
    out.iloc[: spec.warmup, out.columns.get_loc("signal")] = 0
    return out


def eligible(df: pd.DataFrame, spec: Spec, filter_cost: float) -> np.ndarray:
    """Filter flag per bar, from that bar's close and earlier data only."""
    c = df["close"]
    ok = np.ones(len(df), dtype=bool)
    if "trend" in spec.filt:
        ok &= (c > c.rolling(spec.trend_n, min_periods=spec.trend_n).mean()).to_numpy()
    if "costfloor" in spec.filt:
        dist = spec.stop_mult * atr(df, spec.atr_n)
        ok &= (dist * spec.cost_share >= 2.0 * filter_cost * c).to_numpy()
    return ok


def simulate_trade(o, h, l, c, atr_v, t: int, side: int, rr: float, cost: float,
                   stop_mult: float = 1.0, exit: str = "target", hold: int = 10):
    """One trade from signal bar t. Returns (R, exit_index, resolved) or None if no next bar."""
    n = len(o)
    j0 = t + 1
    if j0 >= n or not np.isfinite(atr_v[t]) or atr_v[t] <= 0:
        return None
    a = atr_v[t] * stop_mult
    stop = c[t] - side * a
    tp = c[t] + side * rr * a
    fill = o[j0] * (1 + side * cost)
    end = min(n, j0 + HORIZON)
    lo, hi, op = l[j0:end], h[j0:end], o[j0:end]
    if exit != "target":
        if side != 1:
            raise ValueError("trail and time exits are long only")
        if exit == "trail":
            # Level in force during bar j0+i: highest close over t..j0+i-1, minus the distance.
            levels = np.maximum.accumulate(c[t:end])[: end - j0] - a
        elif exit == "time":
            levels = np.full(end - j0, stop)
        else:
            raise ValueError(exit)
        hit = lo <= levels
        last = end - 1 if exit == "trail" else min(end, j0 + hold) - 1
        hit[last - j0 + 1:] = False
        if hit.any():
            i = int(np.argmax(hit))
            k, exit_px = j0 + i, min(levels[i], op[i])
            resolved = True
        else:
            k = last
            exit_px = c[k]
            resolved = exit == "time" and k == j0 + hold - 1
        return (exit_px * (1 - cost) - fill) / a, k, resolved
    if side == 1:
        hit_stop, hit_tp = lo <= stop, hi >= tp
    else:
        hit_stop, hit_tp = hi >= stop, lo <= tp
    any_hit = hit_stop | hit_tp
    if not any_hit.any():
        k = end - 1
        exit_px, resolved = c[k], False
    else:
        i = int(np.argmax(any_hit))
        k = j0 + i
        resolved = True
        if hit_stop[i]:  # stop wins any tie; a gap beyond the stop fills at the open
            exit_px = min(stop, op[i]) if side == 1 else max(stop, op[i])
        else:
            exit_px = tp
    exit_adj = exit_px * (1 - side * cost)
    return side * (exit_adj - fill) / a, k, resolved


def all_bar_outcomes(df: pd.DataFrame, side: int, cost: float, spec: Spec = Spec(),
                     filter_cost: float | None = None) -> pd.DataFrame:
    """Outcome of entering on EVERY bar with the same geometry. Used for the random-entry
    control: the signal's trades are a subset of these. Bars failing the filter are NaN, so
    the filter removes them from the signal and from the control's pool alike.
    `filter_cost` lets a zero-cost run keep the filter of the costed run."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = atr(df, spec.atr_n).to_numpy()
    ok = eligible(df, spec, cost if filter_cost is None else filter_cost)
    rows = np.full((len(df), 3), np.nan)
    for t in range(spec.warmup, len(df) - 1):
        if not ok[t]:
            continue
        res = simulate_trade(o, h, l, c, a, t, side, spec.rr, cost, spec.stop_mult, spec.exit, spec.hold)
        if res is not None:
            rows[t] = (res[0], res[1], res[2])
    out = pd.DataFrame(rows, index=df.index, columns=["R", "exit_idx", "resolved"])
    return out


def backtest(df: pd.DataFrame, cost: float, both_sides: bool = False, spec: Spec = Spec(),
             filter_cost: float | None = None) -> pd.DataFrame:
    """Trade log for the signal. Long only unless both_sides."""
    sig = signals(df, spec)
    log = []
    cache = {}
    for side in ((1, -1) if both_sides else (1,)):
        cache[side] = all_bar_outcomes(df, side, cost, spec, filter_cost)
    idx = df.index
    for t in np.flatnonzero(sig["signal"].to_numpy() != 0):
        side = int(sig["signal"].iloc[t])
        if side not in cache:
            continue
        r = cache[side].iloc[t]
        if np.isnan(r["R"]):
            continue
        log.append({"signal_time": idx[t], "side": side, "R": float(r["R"]),
                    "exit_time": idx[int(r["exit_idx"])], "resolved": bool(r["resolved"])})
    return pd.DataFrame(log), cache
