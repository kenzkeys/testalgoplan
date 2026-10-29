import importlib.util
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

_p = Path(__file__).resolve().parents[1] / "research" / "06_incubate_tt_moc.py"
_spec = importlib.util.spec_from_file_location("incubate", _p)
inc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inc)
NY = ZoneInfo("America/New_York")

# Fri 2026-11-06 .. Wed 2026-11-11; Monday 11-09 closes below Friday
CLOSES = pd.Series([100.0, 99.0, 101.0, 102.0],
                   index=pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11"]))


def at(day, hh, mm):
    return datetime(*map(int, day.split("-")), hh, mm, tzinfo=NY)


def test_monday_buy_then_tuesday_sell_and_reconcile():
    live_mon = CLOSES.loc[:"2026-11-09"].copy()
    live_mon.iloc[-1] = 99.2  # 15:50 quote, below Friday's 100
    rows, msg = inc.step([], live_mon, at("2026-11-09", 15, 50))
    assert "BUY" in msg and {r["action"] for r in rows} == {"START", "BUY"}
    live_tue = CLOSES.loc[:"2026-11-10"].copy()
    live_tue.iloc[-1] = 100.9
    rows, msg = inc.step(rows, live_tue, at("2026-11-10", 15, 50))
    assert "SELL" in msg
    rows, _ = inc.step(rows, CLOSES, at("2026-11-11", 9, 0))  # outside window: reconcile only
    buy = next(r for r in rows if r["action"] == "BUY")
    assert buy["flip"] == "False" and buy["close_says_buy"] == "True"
    assert abs(float(buy["paper_return"]) - (101 / 99 * (1 - 2e-4) / (1 + 2e-4) - 1)) < 1e-6
    assert abs(float(buy["gap_bps"]) - (99 / 99.2 - 1) * 1e4) < 0.01  # logged to 2 decimals


def test_flip_when_quote_and_close_disagree():
    live = CLOSES.loc[:"2026-11-09"].copy()
    live.iloc[-1] = 100.5  # quote above Friday, but the official close is 99
    rows, msg = inc.step([], live, at("2026-11-09", 15, 50))
    assert "no signal" in msg
    rows, _ = inc.step(rows, CLOSES, at("2026-11-11", 9, 0))
    mon = next(r for r in rows if r["action"] == "NO_SIGNAL")
    assert mon["flip"] == "True"
    assert "flips 1" in inc.report(rows)


def test_missed_days_are_counted_once():
    rows = [inc.blank(date="2026-11-06", action="START")]
    for _ in range(3):
        rows, _ = inc.step(rows, CLOSES, at("2026-11-11", 9, 0))
    assert sum(r["action"] == "MISSED_MONDAY" for r in rows) == 1


def test_outside_window_never_decides():
    rows, msg = inc.step([], CLOSES.loc[:"2026-11-09"], at("2026-11-09", 14, 50))
    assert "outside" in msg and not any(r["action"] in ("BUY", "NO_SIGNAL") for r in rows)
