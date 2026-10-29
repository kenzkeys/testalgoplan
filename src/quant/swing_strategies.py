"""The Amendment 5 rules (docs/00_RESEARCH_PLAN.md), as signal functions plus exit geometry.

Every signal uses bars up to and including its own close; tests/test_harness.py checks
that truncating the data never changes an earlier signal.
"""

from __future__ import annotations

from itertools import product

import pandas as pd

from .swing import Strategy, rsi, sma

MONDAY, THURSDAY = 0, 3  # pandas weekday numbers (SQX counts Sunday = 0, so Thursday is 4 there)


def turnaround_tuesday(threshold: float = 0.0, hold: int = 1, entry: str = "close") -> Strategy:
    """A Monday bar that closes below the prior close by more than `threshold`."""
    def sig(df):
        ret = df["close"].pct_change()
        return (df.index.weekday == MONDAY) & (ret < threshold), None
    name = "tt_moc" if entry == "close" else "tt_open"
    return Strategy(name, _series(sig), entry=entry, hold=hold, params=(threshold, hold))


def rsi_pullback(n: int = 2, threshold: float = 10, exit_sma: int = 5) -> Strategy:
    """Regime: close above SMA(200). Entry: RSI(n) below threshold. Exit: close above SMA(exit_sma)."""
    def sig(df):
        c = df["close"]
        entries = (c > sma(c, 200)) & (rsi(c, n) < threshold)
        return entries, c > sma(c, exit_sma)
    return Strategy("es_rsi2", sig, entry="open", params=(n, threshold, exit_sma))


def volume_pullback(mult: float = 1.5, vol_n: int = 20, hold: int = 5) -> Strategy:
    """A down close on volume above mult x its SMA, inside an uptrend (close above SMA(200))."""
    def sig(df):
        c, v = df["close"], df["volume"]
        return (c < c.shift(1)) & (v > mult * sma(v, vol_n)) & (c > sma(c, 200)), None
    return Strategy("spy_volume", sig, entry="open", hold=hold, params=(mult, vol_n, hold))


def gold_thursday(rsi_threshold: float = 40, stop_atr: float = 1.0, hold: int = 3,
                  trend: bool = False, weekday: int = THURSDAY) -> Strategy:
    """The video's gold strategy: a Thursday bar with RSI(14) below the threshold."""
    def sig(df):
        c = df["close"]
        e = (df.index.weekday == weekday) & (rsi(c, 14) < rsi_threshold)
        if trend:
            e = e & (c > sma(c, 200))
        return e, None
    return Strategy("gold_thursday_trend" if trend else "gold_thursday", _series(sig),
                    entry="open", hold=hold, stop_atr=stop_atr, params=(rsi_threshold, stop_atr, hold))


def _series(fn):
    """Signal functions may build numpy masks from the index; return them as Series."""
    def wrapped(df):
        e, x = fn(df)
        return pd.Series(e, index=df.index), x
    return wrapped


# Main cells, as registered. ensemble_spy is built from three of them in the research script.
FAMILY = {
    "tt_moc": turnaround_tuesday(entry="close"),
    "tt_open": turnaround_tuesday(entry="open"),
    "es_rsi2": rsi_pullback(),
    "spy_volume": volume_pullback(),
    "gold_thursday": gold_thursday(),
    "gold_thursday_trend": gold_thursday(trend=True),
}
ENSEMBLE = ("tt_moc", "es_rsi2", "spy_volume")
TRIALS = 9  # the six above, ensemble_spy, sp_rsi14, sp_rsi2

HOME = {"tt_moc": "SPY", "tt_open": "SPY", "es_rsi2": "SPY", "spy_volume": "SPY",
        "gold_thursday": "GLD", "gold_thursday_trend": "GLD"}
OTHER_MARKETS = {"SPY": ["QQQ", "IWM", "DIA", "EFA"], "GLD": ["SLV", "GDX", "TLT"]}


def parameter_map(name: str) -> list[Strategy]:
    if name in ("tt_moc", "tt_open"):
        entry = "close" if name == "tt_moc" else "open"
        return [turnaround_tuesday(th, h, entry) for th, h in product([0.0, -0.0025, -0.005, -0.01], [1, 2, 3])]
    if name == "es_rsi2":
        return [rsi_pullback(n, th, x) for n, th, x in product([2, 3, 4], [5, 10, 15, 20, 25], [5, 10])]
    if name == "spy_volume":
        return [volume_pullback(m, vn, h) for m, vn, h in product([1.25, 1.5, 2.0], [20, 50], [3, 5, 7])]
    if name in ("gold_thursday", "gold_thursday_trend"):
        trend = name.endswith("trend")
        return [gold_thursday(th, st, h, trend)
                for th, st, h in product([30, 35, 40, 45, 50], [0.5, 1.0, 1.5, 2.0], [2, 3, 4, 5])]
    raise KeyError(name)
