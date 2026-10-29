"""Amendment 5: the Chart Fanatics SQX video strategies, run through the video's own
robustness workflow as hard gates. TRAIN (first 60% of each symbol) only; VALIDATE opens
once per strategy that passes TRAIN; TEST is never loaded.

    PYTHONPATH=src .venv/Scripts/python.exe research/05_sqx_video.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quant import data, protocol, stats, swing
from quant.backtest import Costs, run
from quant.strategies import equal_weight_hold
from quant.swing_strategies import ENSEMBLE, FAMILY, HOME, OTHER_MARKETS, TRIALS, gold_thursday, parameter_map
from quant.xs_pullback import indicators, pullback, random_twin

ROOT = Path(__file__).resolve().parents[1]
P_BONF = 0.05 / TRIALS
COST = swing.COST


def load(sym: str, window=(0.0, 0.6)) -> tuple[pd.DataFrame, int]:
    """Bars from the start up to the window's end (earlier bars only warm indicators up),
    and the first bar index of the window. Bars after the window are never read."""
    df = pd.read_parquet(ROOT / "data" / "swing" / f"{sym}_1d.parquet")
    n = len(df)
    return df.iloc[: int(n * window[1])], int(n * window[0])


def in_window(run: swing.Run, start: int) -> swing.Run:
    tr = run.trades[run.trades["t"] >= start].reset_index(drop=True)
    return swing.Run(tr, run.daily.iloc[start:], run.exposure.iloc[start:])


def mean_trade(df, s, start=0, cost=COST) -> float:
    tr = in_window(swing.backtest(df, s, cost), start).trades
    return float(tr["ret"].mean()) if len(tr) else float("nan")


def single(name: str) -> dict:
    s = FAMILY[name]
    df, _ = load(HOME[name])
    r1, r2 = swing.backtest(df, s, COST), swing.backtest(df, s, 2 * COST)
    rets = r1.trades["ret"].to_numpy()
    out = swing.summary(df, r1)
    out["window"] = [str(df.index[0].date()), str(df.index[-1].date())]
    out["entry_mode"] = s.entry
    out["ci95_mean_trade"] = swing.bootstrap_mean_ci(rets)
    out["mean_trade_2x"] = float(r2.trades["ret"].mean())
    out["random_entry_p"], out["random_entry_mean_trade"] = swing.random_entry_p(df, s, r1.trades, COST, 1000)
    cells = {str(c.params): mean_trade(df, c) for c in parameter_map(name)}
    out["map_cells"] = cells
    out["map_share_positive"] = float(np.mean([v > 0 for v in cells.values()]))
    other = {}
    for sym in OTHER_MARKETS[HOME[name]]:
        dfo, _ = load(sym)
        other[sym] = mean_trade(dfo, s)
    out["other_markets"] = other
    out["noise_pass_rate"] = swing.noise_pass_rate(df, s, 200, COST)
    out["monte_carlo"] = swing.monte_carlo(rets)
    out["video_size_fraction"] = swing.video_size(rets)
    need_other = len(other) // 2 + 1
    checks = {
        "trades>=100": out["trades"] >= 100,
        "ci_above_0": out["ci95_mean_trade"][0] > 0,
        "positive_at_2x": out["mean_trade_2x"] > 0,
        f"random_p<{P_BONF:.4f}": out["random_entry_p"] < P_BONF,
        "sharpe>=buy_hold": out["sharpe"] >= out["bh_sharpe"],
        "map>=75%": out["map_share_positive"] >= 0.75,
        f"other_markets>={need_other}/{len(other)}": sum(v > 0 for v in other.values()) >= need_other,
        "noise>=80%": out["noise_pass_rate"] >= 0.8,
        "not_one_year": bool(out["top_year_share"] <= 0.5),
    }
    out["checks"] = checks
    out["PASS"] = all(checks.values())
    if name.startswith("gold"):  # reported only: is Thursday special?
        trend = name.endswith("trend")
        out["weekday_check"] = {d: mean_trade(df, gold_thursday(trend=trend, weekday=i))
                                for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])}
    return out


def ensemble(sims: int = 1000) -> dict:
    df, _ = load("SPY")
    sleeves = {k: FAMILY[k] for k in ENSEMBLE}
    runs = {k: swing.backtest(df, s, COST) for k, s in sleeves.items()}
    daily = sum(r.daily for r in runs.values()) / len(runs)
    daily2 = sum(swing.backtest(df, s, 2 * COST).daily for s in sleeves.values()) / len(runs)
    bh = df["close"].pct_change().fillna(0.0)
    sr = stats.sharpe(daily, swing.PPY)
    lo, hi, _ = stats.bootstrap_sharpe_ci(daily, swing.PPY)
    rng = np.random.default_rng(37)
    ctrl = np.empty(sims)
    for k in range(sims):
        d = np.zeros(len(df))
        for name, s in sleeves.items():
            tr = runs[name].trades
            span = len(df) - int(tr["t"].min())
            d += swing.shifted_daily(df, s, tr, int(rng.integers(1, span)), COST)
        ctrl[k] = stats.sharpe(d / len(sleeves), swing.PPY)
    p = float((1 + (ctrl >= sr).sum()) / (sims + 1))
    sleeve_sr = {k: stats.sharpe(r.daily, swing.PPY) for k, r in runs.items()}
    other = {}
    for sym in OTHER_MARKETS["SPY"]:
        dfo, _ = load(sym)
        d = sum(swing.backtest(dfo, s, COST).daily for s in sleeves.values()) / len(sleeves)
        other[sym] = {"sharpe": stats.sharpe(d, swing.PPY),
                      "bh_sharpe": stats.sharpe(dfo["close"].pct_change().fillna(0.0), swing.PPY)}
    count = sum(r.exposure for r in runs.values())
    agree = {int(k): {"days": int((count == k).sum()), "mean_spy_return": float(bh[count == k].mean())}
             for k in range(0, len(sleeves) + 1)}
    share, year = stats.concentration_by_year(daily)
    out = {
        "window": [str(df.index[0].date()), str(df.index[-1].date())],
        "sharpe": sr, "sharpe_ci": [lo, hi], "sharpe_2x": stats.sharpe(daily2, swing.PPY),
        "cagr": stats.cagr(daily, swing.PPY), "max_dd": stats.max_drawdown(daily),
        "bh_sharpe": stats.sharpe(bh, swing.PPY), "bh_cagr": stats.cagr(bh, swing.PPY),
        "bh_max_dd": stats.max_drawdown(bh), "sleeve_sharpe": sleeve_sr,
        "random_sleeves_p": p, "random_sleeves_mean_sharpe": float(ctrl.mean()),
        "other_markets": other, "agreement": agree, "top_year_share": share, "top_year": year,
    }
    checks = {
        "sharpe_ci_above_0": lo > 0,
        "sharpe_above_0_at_2x": out["sharpe_2x"] > 0,
        f"random_p<{P_BONF:.4f}": p < P_BONF,
        "sharpe>=buy_hold": sr >= out["bh_sharpe"],
        "sharpe>=best_sleeve": sr >= max(sleeve_sr.values()),
        "other_markets>=3/4": sum(v["sharpe"] > v["bh_sharpe"] for v in other.values()) >= 3,
        "not_one_year": bool(share <= 0.5),
    }
    out["checks"], out["PASS"] = checks, all(checks.values())
    return out


def xs(rsi_n: int, threshold: float, sims: int = 200) -> dict:
    px_all = data.load_prices("equities")
    lo, hi = protocol.window(px_all.index, "train", "equities")
    px = px_all.loc[:hi]
    vol = data.load_volume("equities").loc[:hi]
    elig = data.liquid_mask(px, vol, 100)
    costs = Costs.for_universe("equities")
    costs2 = Costs(costs.one_way_bps * 2, costs.short_carry_bps_year * 2, costs.periods_per_year)
    bench = run(px, equal_weight_hold(px.where(elig)), costs).net.loc[lo:hi]
    bench2 = run(px, equal_weight_hold(px.where(elig)), costs2).net.loc[lo:hi]
    ind = indicators(px, rsi_n)
    w, trades = pullback(px, elig, rsi_n, threshold, 10, 10, ind)
    res = run(px, w, costs)
    r = res.net.loc[lo:hi]
    r2 = run(px, w, costs2).net.loc[lo:hi]
    grid = {2: [2, 5, 10, 15, 20], 14: [20, 25, 30, 35, 40]}[rsi_n]
    cells, cell_sr = {}, []
    for th in grid:
        for mh in (5, 10, 20):
            wc, _ = pullback(px, elig, rsi_n, th, mh, 10, ind)
            rc = run(px, wc, costs).net.loc[lo:hi]
            cells[f"({th}, {mh})"] = float((rc - bench).mean() * 252)
            cell_sr.append(rc.mean() / rc.std(ddof=1))
    rng = np.random.default_rng(41)
    ctrl = np.array([stats.sharpe(run(px, random_twin(px, elig, trades, 10, rng), costs).net.loc[lo:hi], 252)
                     for _ in range(sims)])
    sr = stats.sharpe(r, 252)
    ci_lo, ci_hi, _ = stats.bootstrap_sharpe_ci(r, 252)
    share, year = stats.concentration_by_year(r)
    out = {
        "window": [str(lo.date()), str(hi.date())], "trades": len(trades),
        "cagr": stats.cagr(r, 252), "sharpe": sr, "sharpe_ci": [ci_lo, ci_hi], "max_dd": stats.max_drawdown(r),
        "exposure": float((w.loc[lo:hi].sum(axis=1)).mean()),
        "bench_sharpe": stats.sharpe(bench, 252), "bench_cagr": stats.cagr(bench, 252),
        "bench_max_dd": stats.max_drawdown(bench),
        "active_p": stats.bootstrap_active_p(r, bench, 252), "active_p_2x": stats.bootstrap_active_p(r2, bench2, 252),
        "dsr": stats.deflated_sharpe(r, TRIALS, float(np.std(cell_sr, ddof=1))),
        "random_p": float((1 + (ctrl >= sr).sum()) / (sims + 1)), "random_mean_sharpe": float(ctrl.mean()),
        "map_active_return": cells, "map_share_positive": float(np.mean([v > 0 for v in cells.values()])),
        "turnover_per_year": float(res.turnover.loc[lo:hi].mean() * 252),
        "top_year_share": share, "top_year": year,
    }
    checks = {
        "sharpe_ci_above_0": ci_lo > 0, "dsr>0.95": bool(out["dsr"] > 0.95),
        f"random_p<{P_BONF:.4f}": out["random_p"] < P_BONF,
        "beats_benchmark_p<0.05": out["active_p"] < 0.05, "beats_benchmark_2x": out["active_p_2x"] < 0.05,
        "map>=75%": out["map_share_positive"] >= 0.75, "not_one_year": bool(share <= 0.5),
    }
    out["checks"], out["PASS"] = checks, all(checks.values())
    return out


def validate_single(name: str) -> dict:
    """VALIDATE, once: bars up to 80% (earlier bars warm indicators), trades signalled in 60-80%."""
    s = FAMILY[name]
    df, start = load(HOME[name], (0.6, 0.8))
    r1 = in_window(swing.backtest(df, s, COST), start)
    r2 = in_window(swing.backtest(df, s, 2 * COST), start)
    p, _ = swing.random_entry_p(df, s, r1.trades, COST, 1000)
    out = {"trades": len(r1.trades), "mean_trade": float(r1.trades["ret"].mean()),
           "mean_trade_2x": float(r2.trades["ret"].mean()), "random_entry_p": p,
           "window": [str(df.index[start].date()), str(df.index[-1].date())]}
    out["PASS_validate"] = bool(out["mean_trade_2x"] > 0 and p < 0.05)
    return out


def validate_ensemble(sims: int = 1000) -> dict:
    """VALIDATE, once: Sharpe above 0 at 2x costs and random-sleeves p < 0.05 on the 60-80% slice."""
    df, start = load("SPY", (0.6, 0.8))
    sleeves = {k: FAMILY[k] for k in ENSEMBLE}
    runs = {k: in_window(swing.backtest(df, s, COST), start) for k, s in sleeves.items()}
    daily = sum(r.daily for r in runs.values()) / len(runs)
    daily2 = sum(in_window(swing.backtest(df, s, 2 * COST), start).daily for s in sleeves.values()) / len(runs)
    sr = stats.sharpe(daily, swing.PPY)
    rng = np.random.default_rng(43)
    ctrl = np.empty(sims)
    for k in range(sims):
        d = np.zeros(len(df))
        for name, s in sleeves.items():
            tr = runs[name].trades
            span = len(df) - int(tr["t"].min())
            d += swing.shifted_daily(df, s, tr, int(rng.integers(1, span)), COST)
        ctrl[k] = stats.sharpe(d[start:] / len(sleeves), swing.PPY)
    p = float((1 + (ctrl >= sr).sum()) / (sims + 1))
    bh = df["close"].pct_change().iloc[start:]
    out = {"window": [str(df.index[start].date()), str(df.index[-1].date())], "sharpe": sr,
           "sharpe_2x": stats.sharpe(daily2, swing.PPY), "random_sleeves_p": p,
           "bh_sharpe": stats.sharpe(bh, swing.PPY), "cagr": stats.cagr(daily, swing.PPY),
           "max_dd": stats.max_drawdown(daily), "trades": {k: len(r.trades) for k, r in runs.items()}}
    out["PASS_validate"] = bool(out["sharpe_2x"] > 0 and p < 0.05)
    return out


def main():
    import sys
    if "--validate-ensemble" in sys.argv:  # only after ensemble_spy passed TRAIN (20261029-202333)
        v = validate_ensemble()
        print(v)
        out = ROOT / "results" / f"{datetime.now():%Y%m%d-%H%M%S}_sqx_video_validate_ensemble.json"
        out.write_text(json.dumps(v, indent=2, default=str))
        print("saved", out.name)
        return
    results = {"trials": TRIALS, "bonferroni_p": P_BONF}
    for name in FAMILY:
        print(f"--- {name}", flush=True)
        r = single(name)
        results[name] = r
        print(f"  {r['trades']} trades, win {r['win_rate']:.1%}, mean {r['mean_trade']:+.4%} "
              f"CI [{r['ci95_mean_trade'][0]:+.4%}, {r['ci95_mean_trade'][1]:+.4%}], 2x {r['mean_trade_2x']:+.4%}")
        print(f"  random-entry p {r['random_entry_p']:.4f} (random mean {r['random_entry_mean_trade']:+.4%}), "
              f"Sharpe {r['sharpe']:.2f} vs B&H {r['bh_sharpe']:.2f}, exposure {r['exposure']:.0%}, "
              f"CAGR {r['cagr']:.2%} vs {r['bh_cagr']:.2%}, maxDD {r['max_dd']:.1%} vs {r['bh_max_dd']:.1%}")
        print(f"  map {r['map_share_positive']:.0%}, other {', '.join(f'{k} {v:+.3%}' for k, v in r['other_markets'].items())}, "
              f"noise {r['noise_pass_rate']:.0%}, top year {r['top_year_share']:.0%}")
        print(f"  checks {r['checks']}  PASS={r['PASS']}", flush=True)
    print("--- ensemble_spy", flush=True)
    e = ensemble()
    results["ensemble_spy"] = e
    print(f"  Sharpe {e['sharpe']:.2f} CI {e['sharpe_ci'][0]:.2f}..{e['sharpe_ci'][1]:.2f}, 2x {e['sharpe_2x']:.2f}, "
          f"B&H {e['bh_sharpe']:.2f}, sleeves {e['sleeve_sharpe']}, random p {e['random_sleeves_p']:.4f}")
    print(f"  agreement {e['agreement']}")
    print(f"  checks {e['checks']}  PASS={e['PASS']}", flush=True)
    for nm, (n, th) in {"sp_rsi14": (14, 30), "sp_rsi2": (2, 10)}.items():
        print(f"--- {nm}", flush=True)
        x = xs(n, th)
        results[nm] = x
        print(f"  trades {x['trades']}, Sharpe {x['sharpe']:.2f} CI {x['sharpe_ci'][0]:.2f}..{x['sharpe_ci'][1]:.2f}, "
              f"bench {x['bench_sharpe']:.2f}, active p {x['active_p']:.3f} (2x {x['active_p_2x']:.3f}), "
              f"random p {x['random_p']:.3f}, DSR {x['dsr']:.3f}, map {x['map_share_positive']:.0%}, "
              f"exposure {x['exposure']:.0%}")
        print(f"  checks {x['checks']}  PASS={x['PASS']}", flush=True)

    passers = [k for k in FAMILY if results[k]["PASS"]]
    for k in ("ensemble_spy", "sp_rsi14", "sp_rsi2"):
        if results[k]["PASS"]:
            print(f"{k} passed TRAIN; its VALIDATE needs its own run per the amendment.")
    results["validate"] = {}
    for k in passers:
        print(f"Opening VALIDATE once for {k}", flush=True)
        results["validate"][k] = v = validate_single(k)
        print(f"  {v}")
    if not passers:
        print("No single-instrument strategy passes TRAIN. VALIDATE stays unopened.")
    out = ROOT / "results" / f"{datetime.now():%Y%m%d-%H%M%S}_sqx_video.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print("saved", out.name)


if __name__ == "__main__":
    main()
