"""Acceptance protocol: one function that scores every pre-declared strategy on one split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats
from .backtest import Costs, run
from .strategies import FAMILIES, equal_weight_hold, select_top

# (top_frac, every, long_short) for strategies that have a random-selection control.
SELECTION_SPECS = {
    "xs_mom_12_1": (0.1, 21, False), "xs_mom_6_1": (0.1, 21, False),
    "xs_mom_12_1_ls": (0.1, 21, True), "st_reversal_5d": (0.1, 5, False),
    "xs_mom_90_7": (0.3, 7, False), "xs_mom_30_7": (0.3, 7, False),
}


def random_selection_sharpes(prices, spec, costs, lo, hi, ppy, sims=200, seed=3,
                             signal_prices=None) -> np.ndarray:
    """Same holding rule, same costs, but the names are drawn at random. If the real
    signal does not beat this, the signal is decoration."""
    top_frac, every, ls = spec
    eligible = (prices if signal_prices is None else signal_prices).notna()
    rng = np.random.default_rng(seed)
    out = np.empty(sims)
    for k in range(sims):
        scores = pd.DataFrame(rng.random(prices.shape), index=prices.index, columns=prices.columns)
        scores = scores.where(eligible)
        r = run(prices, select_top(scores, top_frac, every, ls), costs).net.loc[lo:hi]
        out[k] = stats.sharpe(r, ppy)
    return out


def evaluate(kind: str, prices: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp,
             only: list[str] | None = None, control_sims: int = 200,
             costs: Costs | None = None, signal_prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """`signal_prices` restricts what may be SELECTED (NaN = not eligible) while
    returns are still earned on the full `prices`, so a name leaving the eligible
    set does not fabricate a gap in its return series."""
    costs = costs or Costs.for_universe(kind)
    ppy = costs.periods_per_year
    fams = FAMILIES[kind]
    names = only or list(fams)
    sig = prices if signal_prices is None else signal_prices
    bench = run(prices, equal_weight_hold(sig), costs).net.loc[lo:hi]

    results = {n: run(prices, fams[n](sig), costs) for n in names}
    daily_sr = np.array([results[n].net.loc[lo:hi].pipe(lambda s: s.mean() / s.std(ddof=1)) for n in names])
    trials = len(fams)  # charge for the whole declared family, not just what was run
    sr_std = float(np.nanstd(daily_sr, ddof=1)) if len(daily_sr) > 1 else float(abs(daily_sr[0]))

    rows = []
    for n in names:
        res = results[n]
        r = res.net.loc[lo:hi]
        g = res.gross.loc[lo:hi]
        ci_lo, ci_hi, p_le0 = stats.bootstrap_sharpe_ci(r, ppy)
        active_p = stats.bootstrap_active_p(r, bench, ppy)
        share, year = stats.concentration_by_year(r)
        dsr = stats.deflated_sharpe(r, trials, sr_std)
        spec = SELECTION_SPECS.get(n)
        if spec:
            ctrl = random_selection_sharpes(prices, spec, costs, lo, hi, ppy, control_sims,
                                            signal_prices=signal_prices)
            ctrl_p = float((ctrl >= stats.sharpe(r, ppy)).mean())
        else:
            ctrl_p = float("nan")
        sr = stats.sharpe(r, ppy)
        checks = {
            "sharpe_ci_above_0": ci_lo > 0,
            "dsr_above_0.95": bool(dsr > 0.95),
            "beats_random_control": (ctrl_p < 0.05) if spec else bool(sr > stats.sharpe(bench, ppy)),
            "beats_benchmark_p<0.05": active_p < 0.05,
            "not_one_year": bool(share <= 0.5) if np.isfinite(share) else False,
        }
        rows.append({
            "strategy": n, "days": len(r), "cagr": stats.cagr(r, ppy), "sharpe": sr,
            "sharpe_ci_lo": ci_lo, "sharpe_ci_hi": ci_hi, "max_dd": stats.max_drawdown(r),
            "bench_sharpe": stats.sharpe(bench, ppy), "bench_max_dd": stats.max_drawdown(bench),
            "active_p": active_p, "dsr": dsr, "control_p": ctrl_p,
            "top_year_share": share, "top_year": year,
            "turnover_per_year": float(res.turnover.loc[lo:hi].mean() * ppy),
            "cost_drag_per_year": float((g - r).mean() * ppy),
            "gross_sharpe": stats.sharpe(g, ppy),
            **{f"ok_{k}": v for k, v in checks.items()},
            "PASS": all(checks.values()),
        })
    return pd.DataFrame(rows).set_index("strategy")
