"""Amendment 7: filters x exits x timeframe for the video EMA bot (24 cells).

    PYTHONPATH=src .venv/Scripts/python.exe research/07_video_bot_levers.py

First reproduces one Amendment 3 cell to prove the refactor changed nothing. Then scores
every cell on the 60% verdict window, opens VALIDATE (60% to 80%) once for the best
passing cell only, and never loads the final 20%.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

from quant import intraday as ib

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("bot02", HERE / "02_video_ema_bot.py")
bot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bot)

TIMEFRAMES = [("1d", "max"), ("1h", "730d")]
FILTERS = ["none", "trend", "costfloor", "trend+costfloor"]
EXITS = ["target", "trail", "time"]
CELLS = len(TIMEFRAMES) * len(FILTERS) * len(EXITS)
P_BONF = 0.05 / CELLS
SIMS = 1000
AMENDMENT3_REFERENCE = 0.425  # 1d, rr 3.0, stop 2.0 ATR, results/20261029-192300_video_bot_grid.json


def spec_for(filt: str, ex: str) -> ib.Spec:
    return ib.Spec(rr=2.0, stop_mult=2.0, exit=ex, filt=filt, hold=10, trend_n=200)


def main():
    ref = bot.run_variant("repro", "1d", "max", False, ib.Spec(rr=3.0, stop_mult=2.0), (0.0, 0.6), 200, quiet=True)
    print(f"reproduction of Amendment 3 1d_rr3.0_stop2.0: {ref['mean_R_net']:+.3f} (reference +{AMENDMENT3_REFERENCE})")
    assert abs(ref["mean_R_net"] - AMENDMENT3_REFERENCE) < 0.0005, "refactor changed results"

    rows, full = [], {}
    for (tf, period), filt, ex in product(TIMEFRAMES, FILTERS, EXITS):
        name = f"{tf}|{filt}|{ex}"
        spec = spec_for(filt, ex)
        r = bot.run_variant(name, tf, period, False, spec, (0.0, 0.6), SIMS, quiet=True)
        r2 = bot.run_variant(name, tf, period, False, spec, (0.0, 0.6), 1, quiet=True, cost_mult=2.0)
        classes_pos = sum(v > 0 for v in r["mean_R_by_class"].values())
        checks = {
            "ci_above_0": r["ci95_by_day"][0] > 0,
            "random_p_bonferroni": r["random_entry_p"] < P_BONF,
            "classes_2_of_3": classes_pos >= 2,
            "no_symbol_over_50pct": r["top_symbol_share_of_positive_R"] <= 0.5,
            "positive_at_2x_costs": r2["mean_R_net"] > 0,
        }
        r["mean_R_net_2x"] = r2["mean_R_net"]
        r["checks_a7"] = checks
        r["PASS_a7"] = all(checks.values())
        full[name] = r
        rows.append({
            "cell": name, "trades": r["trades"], "win": r["win_rate"], "R_net": r["mean_R_net"],
            "ci_lo": r["ci95_by_day"][0], "ci_hi": r["ci95_by_day"][1], "R_gross": r["mean_R_gross"],
            "R_net_2x": r2["mean_R_net"], "random_R": r["random_entry_mean_R"],
            "edge_vs_random": r["mean_R_net"] - r["random_entry_mean_R"], "p_random": r["random_entry_p"],
            "classes_pos": classes_pos, "top_sym": r["top_symbol_share_of_positive_R"],
            **{f"R_{k}": v for k, v in r["mean_R_by_class"].items()}, "PASS": r["PASS_a7"],
        })
        print(f"{name:26s} n={r['trades']:5d} R_net={r['mean_R_net']:+.3f} random={r['random_entry_mean_R']:+.3f} "
              f"p={r['random_entry_p']:.3f} 2x={r2['mean_R_net']:+.3f} pass={r['PASS_a7']}", flush=True)

    table = pd.DataFrame(rows).set_index("cell")
    pd.set_option("display.width", 260, "display.max_columns", 30)
    print("\n" + table.round(3).sort_values("edge_vs_random", ascending=False).to_string())

    winners = table[table["PASS"]].sort_values("R_net", ascending=False)
    validate = None
    if len(winners):
        best = winners.index[0]
        tf, filt, ex = best.split("|")
        print(f"\nOpening VALIDATE once for {best}")
        validate = bot.run_variant(best, tf, dict(TIMEFRAMES)[tf], False, spec_for(filt, ex), (0.6, 0.8), SIMS)
        validate["PASS_validate"] = bool(validate["mean_R_net"] > 0 and validate["random_entry_p"] < 0.05)
        print({k: validate[k] for k in ("trades", "mean_R_net", "ci95_by_day", "random_entry_p", "PASS_validate")})
    else:
        print(f"\nNo cell passes all five checks (Bonferroni p < {P_BONF:.4f}). VALIDATE stays unopened.")

    out = HERE.parent / "results" / f"{datetime.now():%Y%m%d-%H%M%S}_video_bot_levers.json"
    out.write_text(json.dumps({"reproduction": ref["mean_R_net"], "cells": full, "validate": validate},
                              indent=2, default=str))
    print("saved", out.name)


if __name__ == "__main__":
    main()
