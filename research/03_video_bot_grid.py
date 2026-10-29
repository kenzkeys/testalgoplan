"""Amendment 3: 24-cell timeframe x reward:risk x stop-distance grid for the video bot.

    PYTHONPATH=src .venv/Scripts/python.exe research/03_video_bot_grid.py

Scores every cell on the 60% verdict window. Opens VALIDATE (60% to 80%) once, only for
the best passing cell, and only if one passes. The final 20% is never loaded.
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

TIMEFRAMES = [("15m", "60d"), ("1h", "730d"), ("1d", "max")]
RRS = [1.0, 1.5, 2.0, 3.0]
STOPS = [1.0, 2.0]
CELLS = len(TIMEFRAMES) * len(RRS) * len(STOPS)
P_BONF = 0.05 / CELLS
SIMS = 1000  # enough resolution below 0.0021 to see whether any cell gets close


def main():
    rows, full = [], {}
    for (tf, period), rr, sm in product(TIMEFRAMES, RRS, STOPS):
        name = f"{tf}_rr{rr}_stop{sm}"
        spec = ib.Spec(rr=rr, stop_mult=sm)
        r = bot.run_variant(name, tf, period, False, spec, (0.0, 0.6), SIMS, quiet=True)
        classes_pos = sum(v > 0 for v in r["mean_R_by_class"].values())
        passed = (r["ci95_by_day"][0] > 0 and r["random_entry_p"] < P_BONF
                  and classes_pos >= 2 and r["top_symbol_share_of_positive_R"] <= 0.5)
        r["PASS_bonferroni"] = bool(passed)
        full[name] = r
        rows.append({
            "cell": name, "trades": r["trades"], "win": r["win_rate"],
            "breakeven_win": r["breakeven_win_rate_gross"], "R_net": r["mean_R_net"],
            "ci_lo": r["ci95_by_day"][0], "ci_hi": r["ci95_by_day"][1], "R_gross": r["mean_R_gross"],
            "random_R": r["random_entry_mean_R"], "p_random": r["random_entry_p"],
            "classes_pos": classes_pos, **{f"R_{k}": v for k, v in r["mean_R_by_class"].items()},
            "PASS": passed,
        })
        print(f"{name:22s} n={r['trades']:5d}  R_net={r['mean_R_net']:+.3f}  "
              f"R_gross={r['mean_R_gross']:+.3f}  p_random={r['random_entry_p']:.3f}  pass={passed}", flush=True)

    table = pd.DataFrame(rows).set_index("cell")
    pd.set_option("display.width", 240, "display.max_columns", 30)
    print("\n" + table.round(3).sort_values("R_net", ascending=False).to_string())

    winners = table[table["PASS"]].sort_values("R_net", ascending=False)
    validate = None
    if len(winners):
        best = winners.index[0]
        tf, rr, sm = best.split("_")
        tf_period = dict(TIMEFRAMES)[tf]
        spec = ib.Spec(rr=float(rr[2:]), stop_mult=float(sm[4:]))
        print(f"\nOpening VALIDATE once for {best}")
        validate = bot.run_variant(best, tf, tf_period, False, spec, (0.6, 0.8), SIMS)
        validate["PASS_validate"] = bool(validate["mean_R_net"] > 0 and validate["random_entry_p"] < 0.05)
        print({k: validate[k] for k in ("trades", "mean_R_net", "ci95_by_day", "random_entry_p", "PASS_validate")})
    else:
        print(f"\nNo cell passes (Bonferroni p < {P_BONF:.4f}). VALIDATE stays unopened.")

    out = HERE.parent / "results" / f"{datetime.now():%Y%m%d-%H%M%S}_video_bot_grid.json"
    out.write_text(json.dumps({"cells": full, "validate": validate}, indent=2, default=str))
    print("saved", out.name)


if __name__ == "__main__":
    main()
