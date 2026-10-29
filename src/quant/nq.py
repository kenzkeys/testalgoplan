"""1-minute NASDAQ day-trading harness for docs/00_RESEARCH_PLAN.md Amendment 4.

Days are a (n_days, 390) grid of regular-hours bars, New York time, bar i starting at
09:30 + i minutes. Context for day d (prior close, regime flags, noise area) uses only days
before d. Money is per Micro NASDAQ (MNQ) contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

POINT = 2.0        # $ per index point, MNQ
TICK = 0.25
COMMISSION = 1.24  # $ round trip per contract
SLIP_TICKS = 1     # per market or stop fill
N_BARS = 390
MIN_BARS = 300
TZ = "America/New_York"


@dataclass
class Days:
    dates: np.ndarray                 # datetime64[D]
    o: np.ndarray                     # (n, 390)
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    ctx: list = field(default_factory=list)  # per-day dict from prior days only, or None


def to_days(df: pd.DataFrame) -> Days:
    """UTC-indexed 1-minute OHLC -> RTH grid. Missing minutes are flat bars at the prior close."""
    idx = df.index.tz_convert(TZ) if df.index.tz is not None else df.index.tz_localize("UTC").tz_convert(TZ)
    x = df[["open", "high", "low", "close"]].copy()
    x.index = idx
    mins = idx.hour * 60 + idx.minute - (9 * 60 + 30)
    x = x[(mins >= 0) & (mins < N_BARS)]
    mins = x.index.hour * 60 + x.index.minute - (9 * 60 + 30)
    day = x.index.normalize().tz_localize(None)
    x = x.assign(_d=day, _m=mins)
    counts = x.groupby("_d").size()
    keep = counts.index[counts >= MIN_BARS]
    x = x[x["_d"].isin(keep)]
    dates = np.array(sorted(keep), dtype="datetime64[D]")
    pos = {d: i for i, d in enumerate(pd.DatetimeIndex(dates))}
    n = len(dates)
    grid = {k: np.full((n, N_BARS), np.nan) for k in "ohlc"}
    r = x["_d"].map(pos).to_numpy()
    m = x["_m"].to_numpy()
    for k, col in zip("ohlc", ["open", "high", "low", "close"]):
        grid[k][r, m] = x[col].to_numpy()
    # fill gaps with flat bars at the last close; a leading gap takes the first open
    c = pd.DataFrame(grid["c"].T).ffill().T.to_numpy()
    first_open = pd.DataFrame(grid["o"].T).bfill().T.to_numpy()[:, :1]
    c = np.where(np.isnan(c), first_open, c)
    prev_c = np.concatenate([first_open, c[:, :-1]], axis=1)
    miss = np.isnan(grid["o"])
    o = np.where(miss, prev_c, grid["o"])
    h = np.where(miss, prev_c, grid["h"])
    l = np.where(miss, prev_c, grid["l"])
    c = np.where(miss, prev_c, c)
    d = Days(dates, o, h, l, c)
    d.ctx = build_context(d)
    return d


def build_context(d: Days, noise_n: int = 14, vol_n: int = 20, ema_n: int = 20) -> list:
    n = len(d.dates)
    close = d.c[:, -1]
    hi, lo = d.h.max(axis=1), d.l.min(axis=1)
    rng = hi - lo
    ema = pd.Series(close).ewm(span=ema_n, adjust=False).mean().to_numpy()
    move = np.abs(d.c / d.o[:, :1] - 1.0)  # move from the open at each minute
    out = []
    warm = max(noise_n, vol_n, ema_n) + 1
    for i in range(n):
        if i < warm:
            out.append(None)
            continue
        out.append({
            "prev_close": close[i - 1], "prev_high": hi[i - 1], "prev_low": lo[i - 1],
            "prev_range": rng[i - 1],
            "ema_up": bool(ema[i - 1] > ema[i - 6]),
            "hivol": bool(rng[i - 1] >= np.median(rng[i - vol_n:i])),
            "sigma": move[i - noise_n:i].mean(axis=0),
        })
    return out


@dataclass(frozen=True)
class Trade:
    entry_bar: int
    entry_px: float          # a traded price inside the entry bar, before slippage
    side: int                # +1 long, -1 short
    stop: float = np.nan     # absolute price, nan = no stop
    target: float = np.nan   # absolute price, nan = no target
    exit_bar: int = N_BARS - 1
    exit_at_open: bool = False  # True: exit at the open of exit_bar (signal exit)


def simulate_one(o, h, l, c, t: Trade, cost_mult: float = 1.0):
    """Exit of one trade on one day's bars, possibly partial (live).

    Returns (exit_bar, exit_px, reason, usd) or None while the trade is still open.
    """
    n = len(o)
    last = min(t.exit_bar - (1 if t.exit_at_open else 0), N_BARS - 1)
    s, side = t.stop, t.side
    for k in range(t.entry_bar, min(last + 1, n)):
        if np.isfinite(s) and (l[k] <= s if side == 1 else h[k] >= s):
            px = s if k == t.entry_bar else (min(o[k], s) if side == 1 else max(o[k], s))
            return k, px, "stop", _usd(t, px, 1, cost_mult)
        if k > t.entry_bar and np.isfinite(t.target):
            tp = t.target
            if (h[k] >= tp + TICK) if side == 1 else (l[k] <= tp - TICK):
                px = max(o[k], tp) if side == 1 else min(o[k], tp)
                return k, px, "target", _usd(t, px, 0, cost_mult)
    if t.exit_at_open:
        if t.exit_bar < n:
            return t.exit_bar, o[t.exit_bar], "signal", _usd(t, o[t.exit_bar], 1, cost_mult)
        return None
    if last < n:
        return last, c[last], "time", _usd(t, c[last], 1, cost_mult)
    return None


def _usd(t: Trade, exit_px: float, exit_slip: int, m: float) -> float:
    pts = t.side * (exit_px - t.entry_px) - TICK * SLIP_TICKS * (1 + exit_slip) * m
    return pts * POINT - COMMISSION * m


def simulate(days: Days, rows, entry_bar, entry_px, side, stop, target, exit_bar, exit_at_open,
             cost_mult: float = 1.0) -> np.ndarray:
    """Vectorised simulate_one for full days. Returns net $ per trade."""
    rows = np.asarray(rows, int)
    if len(rows) == 0:
        return np.zeros(0)
    j = np.asarray(entry_bar)[:, None]
    e = np.asarray(entry_px, float)
    sd = np.asarray(side)
    S = np.asarray(stop, float)[:, None]
    T = np.asarray(target, float)[:, None]
    X = np.asarray(exit_bar)
    xo = np.asarray(exit_at_open, bool)
    last = np.minimum(X - xo, N_BARS - 1)[:, None]
    O, H, L, C = days.o[rows], days.h[rows], days.l[rows], days.c[rows]
    b = np.arange(N_BARS)[None, :]
    act = (b >= j) & (b <= last)
    long = (sd == 1)[:, None]
    with np.errstate(invalid="ignore"):
        hs = act & np.where(long, L <= S, H >= S)
        ht = act & (b > j) & np.where(long, H >= T + TICK, L <= T - TICK)
    big = N_BARS + 1
    ks = np.where(hs.any(1), hs.argmax(1), big)
    kt = np.where(ht.any(1), ht.argmax(1), big)
    r = np.arange(len(rows))
    o_ks = O[r, np.minimum(ks, N_BARS - 1)]
    o_kt = O[r, np.minimum(kt, N_BARS - 1)]
    S1, T1 = S[:, 0], T[:, 0]
    stop_px = np.where(ks == j[:, 0], S1, np.where(sd == 1, np.minimum(o_ks, S1), np.maximum(o_ks, S1)))
    tgt_px = np.where(sd == 1, np.maximum(o_kt, T1), np.minimum(o_kt, T1))
    xb = np.minimum(X, N_BARS - 1)
    time_px = np.where(xo, O[r, xb], C[r, np.minimum(last[:, 0], N_BARS - 1)])
    use_stop = (ks <= kt) & (ks < big)
    use_tgt = (kt < ks)
    px = np.where(use_stop, stop_px, np.where(use_tgt, tgt_px, time_px))
    exit_slip = np.where(use_tgt, 0, 1)
    pts = sd * (px - e) - TICK * SLIP_TICKS * (1 + exit_slip) * cost_mult
    return pts * POINT - COMMISSION * cost_mult


def trades_frame(days: Days, trades: dict) -> pd.DataFrame:
    """{row: Trade} -> columns for simulate()."""
    rows = sorted(trades)
    tt = [trades[r] for r in rows]
    return pd.DataFrame({
        "row": rows, "date": days.dates[rows] if rows else [],
        "entry_bar": [t.entry_bar for t in tt], "entry_px": [t.entry_px for t in tt],
        "side": [t.side for t in tt], "stop": [t.stop for t in tt], "target": [t.target for t in tt],
        "exit_bar": [t.exit_bar for t in tt], "exit_at_open": [t.exit_at_open for t in tt],
    })


def run(days: Days, tf: pd.DataFrame, cost_mult: float = 1.0) -> np.ndarray:
    return simulate(days, tf["row"], tf["entry_bar"], tf["entry_px"], tf["side"], tf["stop"],
                    tf["target"], tf["exit_bar"], tf["exit_at_open"], cost_mult)


def coin_flip(days: Days, tf: pd.DataFrame, sims: int = 500, seed: int = 0,
              cost_mult: float = 1.0) -> np.ndarray:
    """Total net $ of `sims` random versions: same days and trade count, each trade keeping its
    stop and target distance and its maximum holding time, with a random side and a random
    entry minute (09:31 to 15:30) filled at that bar's open."""
    rng = np.random.default_rng(seed)
    n = len(tf)
    if n == 0:
        return np.zeros(sims)
    rows = tf["row"].to_numpy()
    e0 = tf["entry_px"].to_numpy()
    sdist = np.abs(e0 - tf["stop"].to_numpy())          # nan stays nan
    tdist = np.abs(tf["target"].to_numpy() - e0)
    hold = tf["exit_bar"].to_numpy() - tf["exit_at_open"].to_numpy() - tf["entry_bar"].to_numpy()
    out = np.empty(sims)
    for s in range(sims):
        eb = rng.integers(1, 361, n)
        sd = rng.choice([-1, 1], n)
        ep = days.o[rows, eb]
        xb = np.minimum(eb + hold, N_BARS - 1)
        out[s] = simulate(days, rows, eb, ep, sd, ep - sd * sdist, ep + sd * tdist, xb,
                          np.zeros(n, bool), cost_mult).sum()
    return out


def coin_flip_daily(days: Days, tf: pd.DataFrame, seed: int, cost_mult: float = 1.0) -> np.ndarray:
    """One random version, as net $ per trade (aligned with tf), for the prop simulation."""
    rng = np.random.default_rng(seed)
    n = len(tf)
    rows = tf["row"].to_numpy()
    e0 = tf["entry_px"].to_numpy()
    sdist = np.abs(e0 - tf["stop"].to_numpy())
    tdist = np.abs(tf["target"].to_numpy() - e0)
    hold = tf["exit_bar"].to_numpy() - tf["exit_at_open"].to_numpy() - tf["entry_bar"].to_numpy()
    eb = rng.integers(1, 361, n)
    sd = rng.choice([-1, 1], n)
    ep = days.o[rows, eb]
    return simulate(days, rows, eb, ep, sd, ep - sd * sdist, ep + sd * tdist,
                    np.minimum(eb + hold, N_BARS - 1), np.zeros(n, bool), cost_mult)


def bootstrap_ci(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean. One trade per strategy per day, so trades are days."""
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def max_drawdown(pnl: np.ndarray) -> float:
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(np.maximum(eq, 0)) - eq).max()) if len(eq) else 0.0


def longest_losing_stretch(pnl: np.ndarray) -> int:
    """Longest run of days spent below a previous equity high."""
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.maximum(eq, 0))
    best = cur = 0
    for under in eq < peak:
        cur = cur + 1 if under else 0
        best = max(best, cur)
    return best


def prop_eval(daily: np.ndarray, size: int, target: float = 3000.0, trail: float = 2000.0):
    """Start an evaluation on every day. daily = net $ per day for one contract.

    Pass at balance >= +target; fail when the end-of-day balance falls to trail below its
    running peak (peak starts at 0). Returns (outcome array: 1 pass, 0 fail, -1 unresolved,
    days used)."""
    n = len(daily)
    outcome = np.full(n, -1)
    used = np.zeros(n, int)
    p = daily * size
    for s in range(n):
        eq = np.cumsum(p[s:])
        peak = np.maximum.accumulate(np.maximum(eq, 0))
        hit_p = np.flatnonzero(eq >= target)
        hit_f = np.flatnonzero(eq <= peak - trail)
        kp = hit_p[0] if len(hit_p) else n
        kf = hit_f[0] if len(hit_f) else n
        if kp < kf:
            outcome[s], used[s] = 1, kp + 1
        elif kf < kp:
            outcome[s], used[s] = 0, kf + 1
    return outcome, used


def year_share(dates, pnl) -> float:
    """Largest share of positive annual P&L held by one year."""
    y = pd.Series(pnl, index=pd.DatetimeIndex(dates)).groupby(lambda t: t.year).sum()
    pos = y[y > 0]
    return float(pos.max() / pos.sum()) if len(pos) else 1.0


def load_databento_csv(path) -> pd.DataFrame:
    """Databento ohlcv-1m CSV (ts_event, open, high, low, close, volume, symbol), front month
    chosen per minute as the contract with the most volume that day (a standard continuous-roll proxy)."""
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts_event"], utc=True)
    if "symbol" in df:
        df = df[~df["symbol"].astype(str).str.contains("-")]  # drop calendar spreads
        day = df["ts"].dt.floor("D")
        vol = df.groupby([day, "symbol"])["volume"].transform("sum")
        df = df.assign(_v=vol).sort_values(["ts", "_v"]).drop_duplicates("ts", keep="last")
    return df.set_index("ts")[["open", "high", "low", "close"]].sort_index()


def load_histdata_csv(path) -> pd.DataFrame:
    """HistData.com generic ASCII 1-minute: 'YYYYMMDD HHMMSS;O;H;L;C;V'.

    HistData documents the clock as fixed EST, but for NSXUSD it is New York local time with
    daylight saving: checked 2011-2026 against QQQ daily closes (median daily gap about 2 bp
    as New York time, 16-50 bp in summer as fixed EST) and against NQ=F 5-minute bars. The
    overnight hours that daylight saving makes ambiguous or nonexistent are dropped."""
    df = pd.read_csv(path, sep=";", header=None, names=["t", "open", "high", "low", "close", "v"])
    ts = pd.to_datetime(df["t"], format="%Y%m%d %H%M%S").dt.tz_localize(
        TZ, ambiguous="NaT", nonexistent="NaT")
    df = df.set_index(ts)[["open", "high", "low", "close"]]
    df = df[df.index.notna()]
    return df.tz_convert("UTC").sort_index()
