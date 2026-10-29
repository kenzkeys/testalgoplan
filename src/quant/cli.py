"""python -m quant.cli download --universe equities|crypto
python -m quant.cli evaluate --universe crypto --split train
python -m quant.cli evaluate --universe crypto --split test --unseal "frozen: <model name and date>"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import data, protocol
from .evaluate import evaluate

RESULTS = Path(__file__).resolve().parents[2] / "results"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="quant")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--universe", required=True, choices=["equities", "crypto"])
    d.add_argument("--start", default="2004-01-01")
    e = sub.add_parser("evaluate")
    e.add_argument("--universe", required=True, choices=["equities", "crypto"])
    e.add_argument("--split", required=True, choices=["train", "validate", "test"])
    e.add_argument("--unseal", default=None)
    e.add_argument("--only", nargs="*", default=None)
    e.add_argument("--sims", type=int, default=200)
    a = ap.parse_args(argv)

    if a.cmd == "download":
        px = data.download(a.universe, a.start)
        print(f"{a.universe}: {px.shape[1]} names, {px.index[0].date()} to {px.index[-1].date()}")
        return 0

    px = data.load_prices(a.universe)
    lo, hi = protocol.window(px.index, a.split, a.universe, a.unseal)
    print(f"{a.universe} {a.split.upper()} {lo.date()} to {hi.date()}  ({px.shape[1]} names)")
    df = evaluate(a.universe, px, lo, hi, a.only, a.sims)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    show = ["cagr", "sharpe", "sharpe_ci_lo", "sharpe_ci_hi", "max_dd", "bench_sharpe",
            "bench_max_dd", "dsr", "control_p", "active_p", "top_year_share", "turnover_per_year",
            "cost_drag_per_year", "gross_sharpe", "PASS"]
    print(df[show].round(3).to_string())
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}_{a.universe}_{a.split}.json"
    out.write_text(json.dumps(json.loads(df.to_json(orient="index")), indent=2))
    print(f"\nsaved {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
