"""Incubation for the frozen `tt_moc` cell (docs/00_RESEARCH_PLAN.md Amendment 6). Paper only:
nothing here can place an order.

    PYTHONPATH=src .venv/Scripts/python.exe research/06_incubate_tt_moc.py           # scheduled
    PYTHONPATH=src .venv/Scripts/python.exe research/06_incubate_tt_moc.py --report  # summary only

It decides only between 15:45 and 15:58 New York time, so it can be scheduled at fixed local
times on both sides of a daylight-saving change and the run outside the window does nothing
but reconcile. Every run fills in official closes for earlier rows, marks Mondays and exit
days that were missed, and checks each live decision against the decision on official closes.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

LOG = Path(__file__).resolve().parents[1] / "state" / "incubation_tt_moc.csv"
FIELDS = ["date", "action", "quote", "quote_time_ny", "prior_close", "official_close",
          "gap_bps", "close_says_buy", "flip", "paper_return"]
NY = ZoneInfo("America/New_York")
WINDOW = (time(15, 45), time(15, 58))
COST = 2e-4            # per side, as in the backtest
KILL_DD = -0.228       # TRAIN Monte Carlo 95th percentile drawdown
KILL_STREAK = 10       # TRAIN Monte Carlo 95th percentile losing streak


def blank(**kw) -> dict:
    return {f: "" for f in FIELDS} | {k: str(v) for k, v in kw.items()}


def next_session(closes: pd.Series, d: pd.Timestamp):
    later = closes.index[closes.index > d]
    return later[0] if len(later) else None


def step(rows: list[dict], closes: pd.Series, now: datetime) -> tuple[list[dict], str]:
    """Pure core. `closes` are daily closes up to today; when the market is open the last
    value is the latest quote, not a close. Returns the updated rows and a message."""
    today = pd.Timestamp(now.date())
    final = closes[closes.index < today]  # official closes only
    rows = [dict(r) for r in rows]
    if not rows:
        rows.append(blank(date=today.date(), action="START"))
    have = {(r["date"], r["action"]) for r in rows}
    start = pd.Timestamp(rows[0]["date"])

    # Mondays and exit days in the past that the script never decided on
    for d in final.index[final.index >= start]:
        ds = str(d.date())
        if d.weekday() == 0 and not have & {(ds, "BUY"), (ds, "NO_SIGNAL"), (ds, "MISSED_MONDAY")}:
            rows.append(blank(date=ds, action="MISSED_MONDAY"))
    for r in [r for r in rows if r["action"] == "BUY"]:
        nxt = next_session(final, pd.Timestamp(r["date"]))
        if nxt is not None:
            ns = str(nxt.date())
            if not have & {(ns, "SELL"), (ns, "MISSED_SELL")}:
                rows.append(blank(date=ns, action="MISSED_SELL"))
    rows.sort(key=lambda r: (r["date"], r["action"]))

    # reconcile against official closes
    for r in rows:
        d = pd.Timestamp(r["date"])
        if d in final.index and not r["official_close"]:
            r["official_close"] = f"{final[d]:.4f}"
            if r["quote"]:
                r["gap_bps"] = f"{(final[d] / float(r['quote']) - 1) * 1e4:.2f}"
            if d.weekday() == 0:
                prev = final[final.index < d]
                if len(prev):
                    says = bool(final[d] < prev.iloc[-1])
                    r["close_says_buy"] = str(says)
                    if r["action"] in ("BUY", "NO_SIGNAL"):
                        r["flip"] = str(says != (r["action"] == "BUY"))
        if r["action"] == "BUY" and not r["paper_return"] and d in final.index:
            nxt = next_session(final, d)
            if nxt is not None:
                gross = final[nxt] / final[d]
                r["paper_return"] = f"{gross * (1 - COST) / (1 + COST) - 1:.6f}"

    ny = now.astimezone(NY) if now.tzinfo else now.replace(tzinfo=NY)
    if not (WINDOW[0] <= ny.time() <= WINDOW[1]) or ny.weekday() > 4:
        return rows, f"{ny:%a %Y-%m-%d %H:%M} ET is outside the decision window; reconciled only."
    if closes.index[-1] != today:
        return rows, f"{ny:%Y-%m-%d} has no SPY session (holiday). Nothing to decide."
    if any(r["date"] == str(today.date()) and r["action"] in ("BUY", "SELL", "NO_SIGNAL") for r in rows):
        return rows, "already decided today."
    quote, prior = float(closes.iloc[-1]), float(closes.iloc[-2])
    common = dict(date=today.date(), quote=f"{quote:.4f}", quote_time_ny=f"{ny:%H:%M}", prior_close=f"{prior:.4f}")
    last_buy = [r for r in rows if r["action"] == "BUY"]
    if last_buy and next_session(closes, pd.Timestamp(last_buy[-1]["date"])) == today:
        rows.append(blank(action="SELL", **common))
        return rows, f"PAPER SELL SPY at today's close (quote {quote:.2f})."
    if today.weekday() == 0:
        buy = quote < prior
        rows.append(blank(action="BUY" if buy else "NO_SIGNAL", **common))
        return rows, (f"PAPER BUY SPY at today's close (quote {quote:.2f} < prior close {prior:.2f})." if buy
                      else f"Monday, no signal (quote {quote:.2f} >= prior close {prior:.2f}).")
    return rows, f"no decision due (quote {quote:.2f})."


def report(rows: list[dict]) -> str:
    mondays = [r for r in rows if r["date"] and pd.Timestamp(r["date"]).weekday() == 0]
    decided = [r for r in mondays if r["flip"]]
    flips = sum(r["flip"] == "True" for r in decided)
    gaps = [abs(float(r["gap_bps"])) for r in rows if r["gap_bps"]]
    missed = sum(r["action"].startswith("MISSED") for r in rows)
    required = missed + sum(r["action"] in ("BUY", "SELL", "NO_SIGNAL") for r in rows)
    rets = [float(r["paper_return"]) for r in rows if r["paper_return"]]
    eq = pd.Series(rets).add(1).cumprod() if rets else pd.Series([1.0])
    dd = float((eq / eq.cummax() - 1).min())
    streak = best = 0
    for x in rets:
        streak = streak + 1 if x < 0 else 0
        best = max(best, streak)
    lines = [
        f"Mondays decided {len(decided)}, flips {flips}" + (f" ({flips / len(decided):.0%}, pass <= 10%)" if decided else ""),
        f"mean |fill gap| {sum(gaps) / len(gaps):.2f} bps over {len(gaps)} fills (pass <= 5)" if gaps else "no fills reconciled yet",
        f"missed {missed} of {required} required runs" + (f" ({missed / required:.0%}, pass <= 10%)" if required else ""),
        f"paper trades {len(rets)}, mean {sum(rets) / len(rets):+.4%} net, drawdown {dd:.1%}, losing streak {best}"
        if rets else "no completed paper trades yet",
    ]
    if dd <= KILL_DD or best >= KILL_STREAK:
        lines.append("KILL RULE HIT: stop and review (Amendment 6).")
    return "\n".join(lines)


def spy_closes() -> pd.Series:
    import yfinance as yf

    raw = yf.download("SPY", period="3mo", interval="1d", auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    return raw["Close"].dropna()


def main():
    rows = list(csv.DictReader(LOG.open())) if LOG.exists() else []
    if "--report" not in sys.argv:
        rows, msg = step(rows, spy_closes(), datetime.now(NY))
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"[{datetime.now(NY):%Y-%m-%d %H:%M} ET] {msg}")
    print(report(rows))


if __name__ == "__main__":
    main()
