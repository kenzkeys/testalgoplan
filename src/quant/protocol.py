"""Chronological train / validate / test split, with the test period locked.

TRAIN (first 60%) is where all work happens. VALIDATE (next 20%) is for choosing among
survivors. TEST (last 20%) refuses to open without a named reason and every opening is
appended to state/test_set_looks.json. A test set looked at twice is a training set.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "state" / "test_set_looks.json"
FRACTIONS = {"train": (0.0, 0.6), "validate": (0.6, 0.8), "test": (0.8, 1.0)}


class SealedError(RuntimeError):
    pass


def window(index: pd.DatetimeIndex, split: str, universe: str = "", unseal_reason: str | None = None,
           log_path: Path | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    if split not in FRACTIONS:
        raise ValueError(split)
    lo, hi = FRACTIONS[split]
    n = len(index)
    start = index[int(n * lo)]
    end = index[min(n - 1, int(n * hi) - 1)]
    if split == "test":
        if not unseal_reason:
            raise SealedError("TEST is sealed. Pass unseal_reason naming the frozen model. One look, ever.")
        path = log_path or LOG
        path.parent.mkdir(exist_ok=True)
        looks = json.loads(path.read_text()) if path.exists() else []
        looks.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "universe": universe, "start": str(start.date()), "end": str(end.date()),
            "reason": unseal_reason,
            "previous_looks_on_this_universe": sum(1 for x in looks if x.get("universe") == universe),
        })
        path.write_text(json.dumps(looks, indent=2))
    return start, end
