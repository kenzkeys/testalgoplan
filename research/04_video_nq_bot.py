"""Amendment 4: the video's NASDAQ prop-firm bot pipeline, 40 pre-registered variants.

    PYTHONPATH=src .venv/Scripts/python.exe research/04_video_nq_bot.py data/nq_1m/nq_1m.parquet "source name"

The parquet holds 1-minute UTC-indexed open/high/low/close. Steps, as in the plan:
honest backtest (1x and 2x costs) and coin flip on IN-SAMPLE (to 2020), OUT-OF-SAMPLE once
for IN-SAMPLE passers, portfolio of passers, prop-firm sizing. Every variant is appended to
state/research_log.jsonl before it runs.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant import nq
from quant import nq_strategies as ns

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "state" / "research_log.jsonl"
SIMS = 500
IS_END = np.datetime64("2020-12-31")
OOS_START = np.datetime64("2021-01-01")
SIZES = [1, 2, 3, 5, 10]
PROP_RANDOM = 20  # random versions per size in the prop simulation


def log(entry: dict):
    """Append-only. Never rewrite earlier lines."""
    LOG.parent.mkdir(exist_ok=True)
    n = sum(1 for _ in LOG.open()) if LOG.exists() else 0
    entry = {"n": n + 1, "logged_utc": datetime.now(timezone.utc).isoformat(), **entry}
    with LOG.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry["n"]


def versions_tested() -> int:
    if not LOG.exists():
        return 0
    return sum(1 for line in LOG.open() if json.loads(line).get("kind") == "variant")


def score(days, name, rows, sims, seed):
    tf = nq.trades_frame(days, ns.generate(days, name, rows))
    pnl = nq.run(days, tf)
    pnl2 = nq.run(days, tf, 2.0)
    out = {"trades": len(tf), "net_usd": float(pnl.sum()), "net_usd_2x": float(pnl2.sum()),
           "gross_usd": float(nq.run(days, tf, 0.0).sum()),
           "mean_usd": float(pnl.mean()) if len(pnl) else np.nan,
           "win_rate": float((pnl > 0).mean()) if len(pnl) else np.nan}
    out["ci95"] = nq.bootstrap_ci(pnl)
    if len(tf):
        rnd = nq.coin_flip(days, tf, sims, seed)
        out["coin_flip_pct_better"] = float((rnd >= pnl.sum()).mean())
        out["coin_flip_p"] = float((1 + (rnd >= pnl.sum()).sum()) / (sims + 1))
        out["coin_flip_median_usd"] = float(np.median(rnd))
        out["top_year_share"] = nq.year_share(tf["date"], pnl)
        out["max_dd_usd"] = nq.max_drawdown(pnl)
    else:
        out.update(coin_flip_pct_better=np.nan, coin_flip_p=1.0, coin_flip_median_usd=np.nan,
                   top_year_share=1.0, max_dd_usd=0.0)
    return out, tf, pnl


def is_pass(r):
    return bool(r["trades"] >= 100 and r["net_usd"] > 0 and r["net_usd_2x"] > 0 and r["ci95"][0] > 0
                and r["coin_flip_p"] <= 1 / (SIMS + 1) + 1e-12 and r["top_year_share"] <= 0.5)


def oos_pass(r):
    return bool(r["net_usd"] > 0 and r["net_usd_2x"] > 0 and r["coin_flip_p"] < 0.05)


def daily_series(days, rows, tf, pnl):
    s = pd.Series(0.0, index=pd.DatetimeIndex(days.dates[rows]))
    s.loc[pd.DatetimeIndex(tf["date"])] = pnl
    return s


def random_daily(days, rows, tf):
    return [daily_series(days, rows, tf, nq.coin_flip_daily(days, tf, 1000 + k)).to_numpy()
            for k in range(PROP_RANDOM)]


def prop_block(real, rnd):
    """Prop-firm sizing for one daily P&L stream (one contract), with coin flips at every size."""
    res = {}
    for size in SIZES:
        out, used = nq.prop_eval(real, size)
        done = out >= 0
        r_rates = []
        for x in rnd:
            ro, _ = nq.prop_eval(x, size)
            rd = ro >= 0
            r_rates.append(float((ro[rd] == 1).mean()) if rd.any() else np.nan)
        rate = float((out[done] == 1).mean()) if done.any() else np.nan
        res[size] = {
            "starts_resolved": int(done.sum()), "pass_rate": rate,
            "median_days_to_pass": float(np.median(used[out == 1])) if (out == 1).any() else None,
            "attempts_per_pass": (1 / rate) if rate else None,
            "coin_flip_pass_rate_mean": float(np.nanmean(r_rates)),
        }
        print(f"   size {size:2d} MNQ: pass {rate:.1%}  median days "
              f"{res[size]['median_days_to_pass']}  coin flip {res[size]['coin_flip_pass_rate_mean']:.1%}",
              flush=True)
    return res


def main(path: str, source: str):
    df = pd.read_parquet(path)
    days = nq.to_days(df)
    valid = np.array([c is not None for c in days.ctx])
    is_rows = np.flatnonzero(valid & (days.dates <= IS_END))
    oos_rows = np.flatnonzero(valid & (days.dates >= OOS_START))
    print(f"{source}: {len(days.dates)} RTH days {days.dates[0]} to {days.dates[-1]}; "
          f"IN-SAMPLE {len(is_rows)} days, OUT-OF-SAMPLE {len(oos_rows)} days", flush=True)

    results = {}
    for k, name in enumerate(ns.VARIANTS):
        log({"kind": "variant", "amendment": 4, "variant": name, "split": "in_sample",
             "data": source, "rules": "docs/00_RESEARCH_PLAN.md Amendment 4"})
        r, _, _ = score(days, name, is_rows, SIMS, seed=k)
        r["IS_PASS"] = is_pass(r)
        results[name] = {"in_sample": r}
        print(f"{name:22s} n={r['trades']:5d} net=${r['net_usd']:>10,.0f} 2x=${r['net_usd_2x']:>10,.0f} "
              f"gross=${r['gross_usd']:>10,.0f} coin%better={r['coin_flip_pct_better']:.3f} "
              f"pass={r['IS_PASS']}", flush=True)

    survivors = [v for v in ns.VARIANTS if results[v]["in_sample"]["IS_PASS"]]
    total = versions_tested()
    print(f"\nIN-SAMPLE survivors: {survivors or 'none'}  (versions tested in the log so far: {total})")

    both = []
    for k, name in enumerate(survivors):
        log({"kind": "oos_look", "amendment": 4, "variant": name, "split": "out_of_sample",
             "data": source, "versions_tested_so_far": total})
        r, _, _ = score(days, name, oos_rows, SIMS, seed=100 + k)
        r["OOS_PASS"] = oos_pass(r)
        results[name]["out_of_sample"] = r
        print(f"OOS {name:18s} n={r['trades']:5d} net=${r['net_usd']:>10,.0f} "
              f"2x=${r['net_usd_2x']:>10,.0f} coin%better={r['coin_flip_pct_better']:.3f} "
              f"pass={r['OOS_PASS']}")
        if r["OOS_PASS"]:
            both.append(name)

    all_rows = np.flatnonzero(valid)
    summary = {"data": source, "first_day": str(days.dates[0]), "last_day": str(days.dates[-1]),
               "in_sample_days": len(is_rows), "out_of_sample_days": len(oos_rows),
               "variants": len(ns.VARIANTS), "versions_tested_in_log": total,
               "in_sample_survivors": survivors, "passed_both": both}

    if both:
        print(f"\nPortfolio of {both} (1 MNQ each), full history:")
        series, rnd = {}, []
        for v in both:
            tf = nq.trades_frame(days, ns.generate(days, v, all_rows))
            series[v] = daily_series(days, all_rows, tf, nq.run(days, tf))
            rnd.append(random_daily(days, all_rows, tf))
        port = pd.DataFrame(series)
        port["portfolio"] = port.sum(axis=1)
        summary["portfolio"] = {c: {"net_usd": float(port[c].sum()), "max_dd_usd": nq.max_drawdown(port[c].to_numpy()),
                                    "longest_losing_stretch_days": nq.longest_losing_stretch(port[c].to_numpy())}
                                for c in port}
        print(pd.DataFrame(summary["portfolio"]).T.round(0).to_string())
        print("\nProp-firm sizing, portfolio (coin flip = sum of each member's random version):")
        summary["prop"] = prop_block(port["portfolio"].to_numpy(), [sum(x) for x in zip(*rnd)])
    else:
        ranked = sorted(ns.VARIANTS, key=lambda v: (results[v]["in_sample"]["coin_flip_p"],
                                                    -results[v]["in_sample"]["net_usd"]))
        best = ranked[0]
        summary["prop_informational_variant"] = best
        print(f"\nNothing passed both. Prop simulation for information only (NOT a pass), best "
              f"IN-SAMPLE variant {best}, full history:")
        tf = nq.trades_frame(days, ns.generate(days, best, all_rows))
        summary["prop_informational"] = prop_block(daily_series(days, all_rows, tf, nq.run(days, tf)).to_numpy(),
                                                   random_daily(days, all_rows, tf))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results" / f"{stamp}_video_nq_bot.json"
    out.write_text(json.dumps({"summary": summary, "variants": results}, indent=2, default=str))
    log({"kind": "result", "amendment": 4, "file": out.name, "passed_both": both,
         "in_sample_survivors": survivors})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else Path(sys.argv[1]).name)
