"""The paper trader ("the dealer", Amendment 4 step 8). Never places orders.

    PYTHONPATH=src .venv/Scripts/python.exe -m quant.dealer --variants "orb30|none,gap_fade|ema"
    PYTHONPATH=src .venv/Scripts/python.exe -m quant.dealer --variants "orb30|none" --once

Every minute on weekdays 09:00-16:05 New York it pulls NQ=F bars from yfinance (delayed),
runs each variant with the same code as the backtest on today's closed bars, and logs each
simulated entry and exit once to state/dealer_log.csv. If state/discord_webhook.txt exists
(one line, a Discord webhook URL you created) each event is also posted there.

History for the prior-days context: 1-minute bars for the last 29 days; older days come from
5-minute bars spread over their 5 minutes, which keeps each day's high, low and close exact
(all the context needs before the 14-day noise window, which is fully 1-minute).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import nq
from . import nq_strategies as ns

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "state"
LOG = STATE / "dealer_log.csv"
SEEN = STATE / "dealer_seen.json"
WEBHOOK = STATE / "discord_webhook.txt"
SYMBOL = "NQ=F"


def _yf(interval: str, start=None, end=None, period=None) -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(SYMBOL).history(interval=interval, start=start, end=end, period=period,
                                   auto_adjust=False, prepost=True)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]]
    return df.tz_convert("UTC")


def fetch() -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    # yfinance serves 1-minute bars for the last 30 days, at most 8 days per request
    one = [_yf("1m", start=(now - timedelta(days=min(d + 7, 29))).date(),
               end=(now - timedelta(days=d - 1)).date())
           for d in range(0, 29, 7)]
    one = pd.concat(one)
    one = one[~one.index.duplicated()].sort_index()
    five = _yf("5m", period="60d")
    five = five[five.index < one.index.min()]
    spread = []
    for k in range(5):
        x = five.copy()
        x.index = x.index + pd.Timedelta(minutes=k)
        if k > 0:
            x["open"] = five["close"].to_numpy() if k == 4 else np.nan
        x["close"] = five["close"].to_numpy() if k == 4 else np.nan
        spread.append(x)
    five1 = pd.concat(spread).sort_index()
    five1["close"] = five1["close"].bfill()
    five1["open"] = five1["open"].fillna(five1["close"])
    return pd.concat([five1, one]).sort_index()


def today_view(df: pd.DataFrame, now_ny: pd.Timestamp):
    """(o, h, l, c) of today's closed RTH bars and today's context from prior days."""
    ny = df.index.tz_convert(nq.TZ)
    today = now_ny.normalize()
    hist = df[ny < today]
    tod = df[(ny >= today + pd.Timedelta(hours=9, minutes=30)) & (ny < now_ny.floor("min"))]
    tod_ny = tod.index.tz_convert(nq.TZ)
    m = tod_ny.hour * 60 + tod_ny.minute - (9 * 60 + 30)
    tod = tod[(m >= 0) & (m < nq.N_BARS)]
    days = nq.to_days(hist)
    pad = np.full((1, nq.N_BARS), np.nan)
    d2 = nq.Days(np.r_[days.dates, np.datetime64(today.date())],
                 np.r_[days.o, pad], np.r_[days.h, pad], np.r_[days.l, pad], np.r_[days.c, pad])
    ctx = nq.build_context(d2)[-1]
    grid = tod[["open", "high", "low", "close"]].to_numpy()
    mins = (tod.index.tz_convert(nq.TZ).hour * 60 + tod.index.tz_convert(nq.TZ).minute - 570).to_numpy()
    n = int(mins.max()) + 1 if len(mins) else 0
    full = pd.DataFrame(np.nan, index=range(n), columns=list("ohlc"))
    full.loc[mins] = grid
    full["c"] = full["c"].ffill()
    for k in "ohl":
        full[k] = full[k].fillna(full["c"].shift(1)).fillna(full["c"])
    return tuple(full[k].to_numpy() for k in "ohlc"), ctx, len(days.dates)


def notify(msg: str):
    print(msg, flush=True)
    if WEBHOOK.exists():
        url = WEBHOOK.read_text().strip()
        try:
            requests.post(url, json={"content": msg}, timeout=10)
        except requests.RequestException as e:
            print(f"discord post failed: {e}", flush=True)


def record(row: dict):
    STATE.mkdir(exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def step(variants, now_ny: pd.Timestamp):
    seen = set(json.loads(SEEN.read_text())) if SEEN.exists() else set()
    (o, h, l, c), ctx, n_hist = today_view(fetch(), now_ny)
    day = str(now_ny.date())
    if ctx is None:
        notify(f"[dealer] {day}: only {n_hist} prior days of history, context needs 22. No trades.")
        return
    for v in variants:
        if len(o) == 0:
            break
        t = ns.day_trade(v, o, h, l, c, ctx)
        if t is None:
            continue
        hhmm = lambda b: (pd.Timestamp(day) + pd.Timedelta(minutes=570 + b)).strftime("%H:%M")
        key = f"{day}|{v}|entry"
        if key not in seen:
            seen.add(key)
            side = "LONG" if t.side == 1 else "SHORT"
            notify(f"[dealer PAPER] {v} {side} 1 MNQ at {t.entry_px:.2f} ({hhmm(t.entry_bar)} ET), "
                   f"stop {t.stop:.2f}, target {t.target:.2f}")
            record({"date": day, "variant": v, "event": "entry", "bar_time": hhmm(t.entry_bar),
                    "side": t.side, "price": t.entry_px, "stop": t.stop, "target": t.target, "usd": ""})
        x = nq.simulate_one(o, h, l, c, t)
        key = f"{day}|{v}|exit"
        if x is not None and key not in seen:
            seen.add(key)
            k, px, why, usd = x
            notify(f"[dealer PAPER] {v} exit ({why}) at {px:.2f} ({hhmm(k)} ET): {usd:+.2f} USD per MNQ net")
            record({"date": day, "variant": v, "event": "exit", "bar_time": hhmm(k), "side": t.side,
                    "price": px, "stop": t.stop, "target": t.target, "usd": round(usd, 2)})
    STATE.mkdir(exist_ok=True)
    SEEN.write_text(json.dumps(sorted(seen)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="comma-separated, e.g. 'orb30|none,gap_fade|ema'")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    variants = [v.strip() for v in a.variants.split(",")]
    for v in variants:
        if v not in ns.VARIANTS:
            raise SystemExit(f"unknown variant {v}; choose from {ns.VARIANTS}")
    while True:
        now = pd.Timestamp.now(tz=nq.TZ)
        live = now.weekday() < 5 and (9 * 60) <= now.hour * 60 + now.minute <= (16 * 60 + 5)
        if live or a.once:
            try:
                step(variants, now)
            except Exception as e:  # keep watching; report the problem
                notify(f"[dealer] problem: {type(e).__name__}: {e}")
        if a.once:
            return
        if now.weekday() < 5 and now.hour * 60 + now.minute > 16 * 60 + 5:
            return
        time.sleep(60 - datetime.now().second + 2)


if __name__ == "__main__":
    main()
