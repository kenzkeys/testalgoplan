"""Next step 1 of docs/00_RESEARCH_PLAN.md, as specified in Amendment 1.

Do the two near-misses from the first TRAIN run survive higher costs and a
liquid-only universe? TRAIN only. The specification, including what counts as
survival, was written into the plan before this script was run.

Runs, all on the equities TRAIN window:
  baseline   1x costs, full universe   (must reproduce results/20261020-164623)
  cost_2x    2x costs, full universe   (rule 3: must survive)
  cost_3x    3x costs, full universe   (information only)
  liquid100  1x costs, eligible = top 100 by 63-day median dollar volume

Run from the project root:  .venv/Scripts/python.exe research/01_near_miss_stress.py
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant import data, protocol
from quant.backtest import Costs
from quant.evaluate import evaluate

CANDIDATES = ["st_reversal_5d", "trend_ew_200"]
TOP_N, LIQ_WINDOW = 100, 63
RESULTS = Path(__file__).resolve().parents[1] / "results"

px = data.load_prices("equities")
try:
    vol = data.load_volume("equities")
except FileNotFoundError:
    print("downloading volume for the cached price panel...")
    vol = data.download_volume("equities")
lo, hi = protocol.window(px.index, "train", "equities")
print(f"equities TRAIN {lo.date()} to {hi.date()}  ({px.shape[1]} names)")
print(f"volume coverage inside TRAIN: {vol.loc[lo:hi].notna().mean().mean():.1%} of cells\n")

eligible = data.liquid_mask(px, vol, TOP_N, LIQ_WINDOW)
liquid_px = px.where(eligible)
print(f"eligible names per day in TRAIN: median {int(eligible.loc[lo:hi].sum(axis=1).median())}\n")

base = Costs.for_universe("equities")


def scaled(m):
    return Costs(one_way_bps=base.one_way_bps * m,
                 short_carry_bps_year=base.short_carry_bps_year * m,
                 periods_per_year=base.periods_per_year)


RUNS = {
    "baseline": dict(costs=scaled(1)),
    "cost_2x": dict(costs=scaled(2)),
    "cost_3x": dict(costs=scaled(3)),
    "liquid100": dict(costs=scaled(1), signal_prices=liquid_px),
}

SHOW = ["cagr", "sharpe", "bench_sharpe", "max_dd", "bench_max_dd", "control_p",
        "active_p", "turnover_per_year", "cost_drag_per_year", "gross_sharpe",
        "ok_beats_random_control", "ok_beats_benchmark_p<0.05"]
pd.set_option("display.width", 240, "display.max_columns", 40)
out = {}
for name, kw in RUNS.items():
    df = evaluate("equities", px, lo, hi, **kw)
    out[name] = json.loads(df.to_json(orient="index"))
    print("=" * 100)
    print(name)
    print("=" * 100)
    print(df.loc[CANDIDATES, SHOW].round(3).to_string(), "\n")

print("=" * 100)
print("VERDICT (Amendment 1: must beat control AND benchmark in BOTH cost_2x and liquid100)")
print("=" * 100)
for c in CANDIDATES:
    ok = all(out[r][c]["ok_beats_random_control"] and out[r][c]["ok_beats_benchmark_p<0.05"]
             for r in ("cost_2x", "liquid100"))
    print(f"  {c:<16} {'SURVIVES: eligible for one VALIDATE look' if ok else 'DEAD on free data'}")

RESULTS.mkdir(exist_ok=True)
path = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}_equities_train_near_miss_stress.json"
path.write_text(json.dumps(out, indent=2))
print(f"\nsaved {path.name}")
