import numpy as np
import pandas as pd

from quant import intraday as ib


def bars(close, spread=0.0, seed=None):
    close = np.asarray(close, float)
    o = np.r_[close[0], close[:-1]]
    h = np.maximum(o, close) + spread
    l = np.minimum(o, close) - spread
    idx = pd.date_range("2026-01-01", periods=len(close), freq="15min")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": close}, index=idx)


def rw(n=4000, seed=0, vol=0.002):
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(0, vol, n))
    df = bars(c)
    df["high"] *= 1 + np.abs(rng.normal(0, vol / 2, n))
    df["low"] *= 1 - np.abs(rng.normal(0, vol / 2, n))
    return df


def test_signal_prefix_invariance():
    df = rw(1500, seed=1)
    full = ib.signals(df)
    part = ib.signals(df.iloc[:900])
    pd.testing.assert_frame_equal(full.iloc[:900], part, check_exact=False, atol=1e-12)


def test_crossover_definition_matches_transcript():
    """Previous bar fast below slow, last bar fast above slow."""
    c = np.r_[np.linspace(100, 90, 80), np.linspace(90, 110, 40)]
    df = bars(c)
    sig = ib.signals(df)
    f, s = ib.ema(df["close"], 5), ib.ema(df["close"], 8)
    up = sig.index[sig["signal"] == 1]
    assert len(up) >= 1
    for t in up:
        i = df.index.get_loc(t)
        assert f.iloc[i] > s.iloc[i] and f.iloc[i - 1] < s.iloc[i - 1]


def test_hand_checked_long_target_hit():
    # signal bar close 100, ATR 1 -> stop 99, target 101.5. Next open 100, then rallies.
    n = 5
    o = np.array([100, 100, 100.5, 101, 101.4], float)
    h = np.array([100.2, 100.6, 101.0, 101.6, 101.5], float)
    l = np.array([99.8, 99.9, 100.2, 100.9, 101.0], float)
    c = np.array([100, 100.4, 100.9, 101.5, 101.4], float)
    a = np.full(n, 1.0)
    R, k, ok = ib.simulate_trade(o, h, l, c, a, t=0, side=1, rr=1.5, cost=0.0)
    assert ok and k == 3 and abs(R - 1.5) < 1e-12


def test_stop_wins_when_both_inside_one_bar():
    o = np.array([100, 100, 100], float)
    h = np.array([100.1, 102.0, 100], float)  # reaches target 101.5
    l = np.array([99.9, 98.0, 100], float)    # and stop 99
    c = np.array([100, 100, 100], float)
    R, k, ok = ib.simulate_trade(o, h, l, c, np.ones(3), 0, 1, 1.5, 0.0)
    assert ok and abs(R - (-1.0)) < 1e-12


def test_gap_through_stop_fills_at_the_open():
    o = np.array([100, 100, 97.0], float)  # opens 3 below the stop at 99
    h = np.array([100.1, 100.2, 97.5], float)
    l = np.array([99.9, 99.5, 96.5], float)
    c = np.array([100, 99.8, 97.2], float)
    R, k, ok = ib.simulate_trade(o, h, l, c, np.ones(3), 0, 1, 1.5, 0.0)
    assert ok and abs(R - (-3.0)) < 1e-12


def test_short_is_the_mirror():
    o = np.array([100, 100, 99.5], float)
    h = np.array([100.1, 100.2, 99.6], float)
    l = np.array([99.9, 99.0, 98.0], float)  # target 98.5 reached on bar 2
    c = np.array([100, 99.5, 98.1], float)
    R, k, ok = ib.simulate_trade(o, h, l, c, np.ones(3), 0, -1, 1.5, 0.0)
    assert ok and abs(R - 1.5) < 1e-12


def test_cost_is_charged_on_both_sides():
    o = np.array([100, 100, 100], float)
    h = np.array([100.1, 102.0, 100], float)
    l = np.array([99.9, 99.9, 100], float)
    c = np.array([100, 100, 100], float)
    free, *_ = ib.simulate_trade(o, h, l, c, np.ones(3), 0, 1, 1.5, 0.0)
    paid, *_ = ib.simulate_trade(o, h, l, c, np.ones(3), 0, 1, 1.5, 0.001)
    assert np.isclose(free - paid, 0.001 * 100 + 0.001 * 101.5, atol=1e-9)


def test_random_entry_on_driftless_walk_has_no_gross_edge_and_40pct_wins():
    """Break-even win rate for 1R stop / 1.5R target is 40%. A driftless walk must sit there."""
    df = rw(6000, seed=2)
    out = ib.all_bar_outcomes(df, 1, 0.0).dropna()
    out = out[out["resolved"] == 1]
    assert abs((out["R"] > 0).mean() - 0.40) < 0.04
    se = out["R"].std() / np.sqrt(len(out))
    assert abs(out["R"].mean()) < 4 * se * 3  # overlapping trades inflate the true SE


def test_signals_never_trade_before_they_exist():
    df = rw(2000, seed=3)
    log, _ = ib.backtest(df, 0.0)
    assert (log["exit_time"] > log["signal_time"]).all()


# Amendment 7: exits and filters.

def test_trailing_stop_ratchets_on_prior_closes_only():
    # signal close 100, ATR 1 -> initial stop 99. Closes rise to 104, so the level for the
    # bar after the 104 close is 103. That bar's low 102.9 triggers at 103.
    o = np.array([100, 100, 101, 102, 103, 104, 103.5], float)
    h = np.array([100, 101, 102, 103, 104, 104.5, 103.6], float)
    l = np.array([100, 99.5, 100.5, 101.5, 102.5, 103.5, 102.9], float)
    c = np.array([100, 101, 102, 103, 104, 104, 103.0], float)
    R, k, ok = ib.simulate_trade(o, h, l, c, np.ones(7), 0, 1, 2.0, 0.0, 1.0, "trail")
    assert k == 6 and ok and abs(R - (103.0 - 100.0)) < 1e-12


def test_trailing_stop_does_not_use_the_current_bars_close():
    # Bar 2 closes at 110 but its own low (100.2) must be judged against the level from
    # bar 1's close (101 - 1 = 100), not from 110.
    o = np.array([100, 100, 101, 109], float)
    h = np.array([100, 101, 110, 111], float)
    l = np.array([100, 99.5, 100.2, 108.9], float)
    c = np.array([100, 101, 110, 110], float)
    R, k, ok = ib.simulate_trade(o, h, l, c, np.ones(4), 0, 1, 2.0, 0.0, 1.0, "trail")
    assert k == 3  # stopped on bar 3 at the level 109 set by bar 2's close


def test_time_exit_closes_on_the_holdth_bar():
    n = 20
    c = np.linspace(100, 110, n)
    o = np.r_[c[0], c[:-1]]
    h, l = c + 0.1, np.minimum(o, c) - 0.1
    R, k, ok = ib.simulate_trade(o, h, l, c, np.ones(n), 0, 1, 2.0, 0.0, 1.0, "time", hold=5)
    assert k == 5 and ok and abs(R - (c[5] - o[1])) < 1e-12


def test_filters_are_prefix_invariant():
    df = rw(1500, seed=4)
    for filt in ("trend", "costfloor", "trend+costfloor"):
        spec = ib.Spec(filt=filt, trend_n=100, stop_mult=2.0)
        full = ib.eligible(df, spec, 1e-4)[:900]
        part = ib.eligible(df.iloc[:900], spec, 1e-4)
        assert (full == part).all(), filt


def test_filter_removes_bars_from_signal_and_control_alike():
    df = rw(2000, seed=5)
    spec = ib.Spec(filt="trend", trend_n=100, stop_mult=2.0, exit="trail")
    out = ib.all_bar_outcomes(df, 1, 1e-4, spec)
    ok = ib.eligible(df, spec, 1e-4)
    assert out["R"][~ok].isna().all()
    log, _ = ib.backtest(df, 1e-4, False, spec)
    pos = [df.index.get_loc(t) for t in log["signal_time"]]
    assert ok[pos].all()


def test_zero_cost_run_keeps_the_costed_filter():
    df = rw(2000, seed=6)
    spec = ib.Spec(filt="costfloor", stop_mult=2.0)
    a = ib.all_bar_outcomes(df, 1, 0.0, spec, filter_cost=5e-4)["R"].notna()
    b = ib.all_bar_outcomes(df, 1, 5e-4, spec)["R"].notna()
    assert (a == b).all()
