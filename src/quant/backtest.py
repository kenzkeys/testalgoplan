"""Vectorised daily portfolio backtest.

Honesty rules:
* weights decided at close t are held over t+1 (one-day lag applied HERE, once);
* costs are charged on turnover at the day the position changes;
* short exposure pays a carry cost (borrow for equities, funding for crypto).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    one_way_bps: float          # commission + half spread + slippage, per unit turnover
    short_carry_bps_year: float # annual cost on short notional
    periods_per_year: int

    @classmethod
    def for_universe(cls, kind: str) -> "Costs":
        if kind == "equities":
            # Liquid S&P 500 names: ~2bps spread/half-spread, ~3bps impact/slippage.
            return cls(one_way_bps=5.0, short_carry_bps_year=50.0, periods_per_year=252)
        if kind == "crypto":
            # Taker fee ~5-10bps + slippage on majors. Funding on shorts assumed 10%/yr.
            return cls(one_way_bps=10.0, short_carry_bps_year=1000.0, periods_per_year=365)
        raise ValueError(kind)


@dataclass
class Result:
    gross: pd.Series
    net: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame


def run(prices: pd.DataFrame, weights: pd.DataFrame, costs: Costs) -> Result:
    rets = prices.pct_change()
    # A name with no price today cannot be held: zero its weight rather than fill a return.
    w = weights.reindex_like(prices).fillna(0.0).where(prices.notna(), 0.0)
    held = w.shift(1).fillna(0.0)
    gross = (held * rets.fillna(0.0)).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    short_notional = (-held.clip(upper=0.0)).sum(axis=1)
    cost = turnover * costs.one_way_bps / 1e4 + short_notional * costs.short_carry_bps_year / 1e4 / costs.periods_per_year
    net = gross - cost
    return Result(gross=gross, net=net, turnover=turnover, weights=w)
