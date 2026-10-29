"""The 10 published day-trading strategies and 4 filters of Amendment 4.

Each strategy maps one day's bars (o, h, l, c, possibly a live prefix shorter than 390) and
that day's prior-days context to the day's first Trade, or None. An entry decided at bar j
uses only bars up to j (market orders: up to j - 1), which tests/test_nq.py checks by
truncating the day. Exits may look past the entry; that is the exit rule, not a signal.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .nq import N_BARS, TICK, Days, Trade

LAST = N_BARS - 1
BAR_1500 = 330  # bar starting 15:00


def _stop_entry_oco(o, h, l, up, dn, i0, i1):
    """First bar in [i0, i1) touching a resting buy stop at `up` or sell stop at `dn`.
    Returns (bar, side, fill) with fill at the trigger or at a gapped open; None if neither
    or both trigger inside the same bar (order unknown, skipped)."""
    for i in range(i0, min(i1, len(o))):
        bu = up is not None and h[i] >= up
        bd = dn is not None and l[i] <= dn
        if bu and bd:
            return None
        if bu:
            return i, 1, max(up, o[i])
        if bd:
            return i, -1, min(dn, o[i])
    return None


def orb5(o, h, l, c, ctx):
    if len(o) < 6:
        return None
    d = np.sign(c[4] - o[0])
    if d == 0:
        return None
    e = o[5]
    stop = l[:5].min() if d == 1 else h[:5].max()
    r = d * (e - stop)
    if r <= 0:
        return None
    return Trade(5, e, int(d), stop, e + d * 10 * r)


def orb30(o, h, l, c, ctx):
    if len(o) < 31:
        return None
    rh, rl = h[:30].max(), l[:30].min()
    x = _stop_entry_oco(o, h, l, rh + TICK, rl - TICK, 30, BAR_1500)
    if x is None:
        return None
    j, s, px = x
    return Trade(j, px, s, rl if s == 1 else rh)


def ib_break(o, h, l, c, ctx):
    if len(o) < 61:
        return None
    rh, rl = h[:60].max(), l[:60].min()
    mid, rg = (rh + rl) / 2, rh - rl
    x = _stop_entry_oco(o, h, l, rh + TICK, rl - TICK, 60, BAR_1500)
    if x is None:
        return None
    j, s, px = x
    return Trade(j, px, s, mid, px + s * rg)


def _gap(o, ctx):
    g = o[0] / ctx["prev_close"] - 1
    return g if 0.0025 <= abs(g) <= 0.015 else None


def gap_fade(o, h, l, c, ctx):
    if len(o) < 2:
        return None
    g = _gap(o, ctx)
    if g is None:
        return None
    s = -int(np.sign(g))
    e, pc = o[1], ctx["prev_close"]
    if s * (pc - e) <= 0:
        return None
    return Trade(1, e, s, e - s * abs(o[0] - pc), pc)


def gap_go(o, h, l, c, ctx):
    if len(o) < 6:
        return None
    g = _gap(o, ctx)
    if g is None or np.sign(c[4] - o[0]) != np.sign(g):
        return None
    s = int(np.sign(g))
    e = o[5]
    stop = l[:5].min() if s == 1 else h[:5].max()
    if s * (e - stop) <= 0:
        return None
    return Trade(5, e, s, stop)


def intraday_mom(o, h, l, c, ctx):
    if len(o) < 361:
        return None
    r = c[29] / ctx["prev_close"] - 1
    if r == 0:
        return None
    return Trade(360, o[360], int(np.sign(r)))


def noise_break(o, h, l, c, ctx):
    sig = ctx["sigma"]
    up_base, dn_base = max(o[0], ctx["prev_close"]), min(o[0], ctx["prev_close"])
    for m in range(29, 360, 30):  # closes at 10:00, 10:30, ..., 15:30
        if m + 1 >= len(o):
            return None
        ub, lb = up_base * (1 + sig[m]), dn_base * (1 - sig[m])
        if c[m] > ub:
            return Trade(m + 1, o[m + 1], 1, lb)
        if c[m] < lb:
            return Trade(m + 1, o[m + 1], -1, ub)
    return None


def pdhl_break(o, h, l, c, ctx):
    x = _stop_entry_oco(o, h, l, ctx["prev_high"] + TICK, ctx["prev_low"] - TICK, 0, BAR_1500)
    if x is None:
        return None
    j, s, px = x
    return Trade(j, px, s, px - s * 0.25 * ctx["prev_range"])


def turtle_soup(o, h, l, c, ctx):
    ph, pl = ctx["prev_high"], ctx["prev_low"]
    mid = (ph + pl) / 2
    above = below = False
    for i in range(min(150, len(o) - 1)):
        above |= h[i] > ph
        below |= l[i] < pl
        if above and c[i] < ph:
            e = o[i + 1]
            if mid < e:
                return Trade(i + 1, e, -1, h[: i + 1].max() + TICK, mid)
            above = False
        if below and c[i] > pl:
            e = o[i + 1]
            if mid > e:
                return Trade(i + 1, e, 1, l[: i + 1].min() - TICK, mid)
            below = False
    return None


def _five(o, h, l, c):
    """Closed 5-minute bars of the day so far."""
    k = len(o) // 5
    O = o[: 5 * k : 5]
    H = h[: 5 * k].reshape(k, 5).max(1)
    L = l[: 5 * k].reshape(k, 5).min(1)
    C = c[4 : 5 * k : 5]
    return O, H, L, C


def _rsi2(C):
    d = np.diff(C)
    up, dn = np.maximum(d, 0), np.maximum(-d, 0)
    out = np.full(len(C), np.nan)
    au = ad = None
    for i, (u, w) in enumerate(zip(up, dn), start=1):
        au = u if au is None else au + (u - au) / 2
        ad = w if ad is None else ad + (w - ad) / 2
        out[i] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def rsi2_rev(o, h, l, c, ctx):
    """ATR(14) of 5-minute bars uses this day's bars only (mean true range of the last up
    to 14), since the grid holds regular hours only."""
    O, H, L, C = _five(o, h, l, c)
    if len(C) < 4:
        return None
    rsi = _rsi2(C)
    pc = np.r_[O[0], C[:-1]]
    tr = np.maximum(H - L, np.maximum(abs(H - pc), abs(L - pc)))
    for k in range(3, len(C)):
        if 5 * k + 5 > BAR_1500:
            return None
        s = 1 if rsi[k] < 5 else -1 if rsi[k] > 95 else 0
        if s == 0:
            continue
        j = 5 * k + 5
        if j >= len(o):
            return None
        atr = tr[max(0, k - 13) : k + 1].mean()
        exit_bar = LAST
        for q in range(k + 1, len(C)):
            if q >= 4:
                sma = C[q - 4 : q + 1].mean()
                if (s == 1 and C[q] > sma) or (s == -1 and C[q] < sma):
                    exit_bar = 5 * q + 5
                    break
        if exit_bar > LAST:
            return Trade(j, o[j], s, o[j] - s * 2 * atr)
        return Trade(j, o[j], s, o[j] - s * 2 * atr, exit_bar=exit_bar,
                     exit_at_open=exit_bar < LAST)
    return None


STRATEGIES = {
    "orb5": orb5, "orb30": orb30, "ib_break": ib_break, "gap_fade": gap_fade, "gap_go": gap_go,
    "intraday_mom": intraday_mom, "noise_break": noise_break, "pdhl_break": pdhl_break,
    "turtle_soup": turtle_soup, "rsi2_rev": rsi2_rev,
}


def keep(trade: Trade, ctx: dict, flt: str) -> bool:
    if flt == "none":
        return True
    if flt == "ema":
        return (trade.side == 1) == ctx["ema_up"]
    if flt == "hivol":
        return ctx["hivol"]
    if flt == "lovol":
        return not ctx["hivol"]
    raise ValueError(flt)


FILTERS = ["none", "ema", "hivol", "lovol"]
VARIANTS = [f"{s}|{f}" for s in STRATEGIES for f in FILTERS]


def on_ticks(t: Trade) -> Trade:
    """Resting orders sit on the 0.25 grid: stops rounded away from entry, targets toward it."""
    s = t.side
    stop = np.floor(t.stop / TICK) * TICK if s == 1 else np.ceil(t.stop / TICK) * TICK
    tgt = np.floor(t.target / TICK) * TICK if s == 1 else np.ceil(t.target / TICK) * TICK
    return replace(t, stop=stop, target=tgt)


def day_trade(name: str, o, h, l, c, ctx):
    s, f = name.split("|")
    if ctx is None:
        return None
    t = STRATEGIES[s](o, h, l, c, ctx)
    return on_ticks(t) if t is not None and keep(t, ctx, f) else None


def generate(days: Days, name: str, rows) -> dict:
    out = {}
    for r in rows:
        t = day_trade(name, days.o[r], days.h[r], days.l[r], days.c[r], days.ctx[r])
        if t is not None:
            out[int(r)] = t
    return out
