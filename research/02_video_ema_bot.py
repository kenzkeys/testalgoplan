"""Amendment 2: backtest of the YouTube EMA(5/8) crossover bot. Verdict window = first 60%
of each symbol's bars; the rest is never loaded into any computation.

    PYTHONPATH=src .venv/Scripts/python.exe research/02_video_ema_bot.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quant import intraday as ib

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "intraday"
UNIVERSE = {
    "GBPJPY=X": ("fx", 1.5e-4), "EURUSD=X": ("fx", 0.5e-4),
    "BTC-USD": ("crypto", 10e-4), "ETH-USD": ("crypto", 10e-4),
    "SOL-USD": ("crypto", 10e-4), "XRP-USD": ("crypto", 10e-4),
    "SPY": ("equities", 2e-4), "QQQ": ("equities", 2e-4), "AAPL": ("equities", 2e-4),
    "MSFT": ("equities", 2e-4), "NVDA": ("equities", 2e-4), "TSLA": ("equities", 2e-4),
}
SIMS = 2000


def fetch(sym: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{sym.replace('=', '_')}_{interval}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raw = yf.download(sym, interval=interval, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    idx = pd.to_datetime(df.index)
    df.index = (idx.tz_convert("UTC") if idx.tz is not None else idx).tz_localize(None)
    df = df[~df.index.duplicated()].sort_index()
    df.to_parquet(path)
    return df


def day_bootstrap_ci(trades: pd.DataFrame, reps=2000, seed=5):
    days = trades["signal_time"].dt.date
    groups = [g["R"].to_numpy() for _, g in trades.groupby(days)]
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for k in range(reps):
        pick = rng.integers(len(groups), size=len(groups))
        allr = np.concatenate([groups[i] for i in pick])
        means[k] = allr.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_variant(name: str, interval: str, period: str, both_sides: bool,
                spec: ib.Spec = ib.Spec(), window: tuple[float, float] = (0.0, 0.6),
                sims: int = SIMS, quiet: bool = False, cost_mult: float = 1.0) -> dict:
    """`window` is the (start, end) fraction of each symbol's bars. Default is the 60%
    verdict window; VALIDATE is (0.6, 0.8) and may only be requested per the plan."""
    per, trades_all, ctrl_pool = [], [], {}
    for sym, (cls, cost) in UNIVERSE.items():
        cost *= cost_mult
        df = fetch(sym, interval, period)
        n = len(df)
        df = df.iloc[int(n * window[0]): int(n * window[1])]
        if len(df) < 300:
            if not quiet:
                print(f"  skip {sym}: only {len(df)} bars")
            continue
        net_log, cache = ib.backtest(df, cost, both_sides, spec)
        gross_log, _ = ib.backtest(df, 0.0, both_sides, spec, filter_cost=cost)
        if net_log.empty:
            continue
        net_log["symbol"], net_log["class"] = sym, cls
        trades_all.append(net_log)
        ctrl_pool[sym] = {s: c["R"].dropna().to_numpy() for s, c in cache.items()}
        per.append({
            "symbol": sym, "class": cls, "bars": len(df), "start": str(df.index[0].date()),
            "end": str(df.index[-1].date()), "trades": len(net_log),
            "win_rate": float((net_log["R"] > 0).mean()),
            "mean_R_net": float(net_log["R"].mean()), "mean_R_gross": float(gross_log["R"].mean()),
            "cost_R_per_trade": float(gross_log["R"].mean() - net_log["R"].mean()),
            "unresolved": int((~net_log["resolved"]).sum()),
        })
    trades = pd.concat(trades_all, ignore_index=True)
    obs = float(trades["R"].mean())
    lo, hi = day_bootstrap_ci(trades)

    # Random-entry control: same symbols, same trade counts per symbol and side.
    rng = np.random.default_rng(9)
    counts = trades.groupby(["symbol", "side"]).size()
    ctrl = np.empty(sims)
    for k in range(sims):
        tot, n = 0.0, 0
        for (sym, side), m in counts.items():
            pool = ctrl_pool[sym][side]
            tot += rng.choice(pool, size=m, replace=True).sum()
            n += m
        ctrl[k] = tot / n
    ctrl_p = float((ctrl >= obs).mean())

    by_class = trades.groupby("class")["R"].mean().to_dict()
    pos = trades[trades["R"] > 0].groupby("symbol")["R"].sum()
    top_share = float(pos.max() / pos.sum()) if len(pos) else float("nan")
    checks = {
        "ci_excludes_0_above": lo > 0,
        "beats_random_entry_p<0.05": ctrl_p < 0.05,
        "positive_in_2_of_3_classes": sum(v > 0 for v in by_class.values()) >= 2,
        "no_symbol_over_50pct": top_share <= 0.5,
    }
    return {
        "variant": name, "interval": interval, "trades": int(len(trades)),
        "win_rate": float((trades["R"] > 0).mean()), "breakeven_win_rate_gross": 1.0 / (1.0 + spec.rr),
        "mean_R_net": obs, "ci95_by_day": [lo, hi],
        "mean_R_gross": float(np.average([p["mean_R_gross"] for p in per], weights=[p["trades"] for p in per])),
        "random_entry_mean_R": float(ctrl.mean()), "random_entry_p": ctrl_p,
        "mean_R_by_class": by_class, "top_symbol_share_of_positive_R": top_share,
        "checks": checks, "PASS": all(checks.values()), "per_symbol": per,
    }


def main():
    out = {}
    for name, interval, period, both in [
        ("A_video_long_only_15m", "15m", "60d", False),
        ("B_symmetric_15m", "15m", "60d", True),
        ("C_long_only_1h", "1h", "730d", False),
    ]:
        print(f"\n=== {name}")
        r = run_variant(name, interval, period, both)
        out[name] = r
        print(pd.DataFrame(r["per_symbol"]).round(3).to_string(index=False))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("per_symbol", "checks")})
        print("checks:", r["checks"])
    path = ROOT / "results" / f"{datetime.now():%Y%m%d-%H%M%S}_video_ema_bot.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print("\nsaved", path.name)


if __name__ == "__main__":
    main()
