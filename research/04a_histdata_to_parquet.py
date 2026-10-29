"""HistData.com NSXUSD 1-minute zips -> data/nq_1m/nq_1m.parquet for research/04_video_nq_bot.py.

    .venv/Scripts/python.exe -I research/04a_histdata_to_parquet.py data/nq_1m/histdata

Reads every .zip in the folder in memory (no extraction to disk), parses only its .csv
members with quant.nq.load_histdata_csv (generic ASCII, New York local time; see the loader), drops duplicate
minutes and prints coverage per year so gaps are visible before anything is run.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # -I ignores PYTHONPATH

import pandas as pd  # noqa: E402

from quant import nq  # noqa: E402


def main(folder: str):
    frames = []
    for z in sorted(Path(folder).glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".csv"):
                    frames.append(nq.load_histdata_csv(io.BytesIO(zf.read(name))))
                    print(f"{z.name}/{name}: {len(frames[-1]):,} bars")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    print(f"\n{len(df):,} bars {df.index[0]} to {df.index[-1]}; {int(bad.sum())} bars with high/low "
          f"outside open/close (kept, reported)")
    days = nq.to_days(df)
    per_year = pd.Series(1, index=pd.DatetimeIndex(days.dates)).groupby(lambda t: t.year).sum()
    print("RTH days with >= 300 of 390 bars, by year:\n" + per_year.to_string())
    out = ROOT / "data" / "nq_1m" / "nq_1m.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
