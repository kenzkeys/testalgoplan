import numpy as np
import pandas as pd
import pytest

from quant import swing
from quant.swing import Strategy, backtest, simulate, Bars
from quant.swing_strategies import FAMILY, gold_thursday, parameter_map, turnaround_tuesday
from quant.xs_pullback import pullback, random_twin


def ohlcv(n=1200, seed=0, drift=0.0003, vol=0.01, start="2010-01-04"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    c = 100 * np.cumprod(1 + rng.normal(drift, vol, n))
    o = c * (1 + rng.normal(0, vol / 3, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    v = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)


def flat(n=10, price=100.0):
    idx = pd.bdate_range("2020-01-06", periods=n)  # a Monday
    p = np.full(n, price)
    return pd.DataFrame({"open": p, "high": p + 1, "low": p - 1, "close": p, "volume": 1e6}, index=idx)


def only(t):
    return lambda df: (pd.Series(np.arange(len(df)) == t, index=df.index), None)


def test_open_entry_hold_n_exits_at_close_of_nth_bar():
    df = flat()
    df.loc[df.index[3], "close"] = 110.0
    run = backtest(df, Strategy("x", only(1), entry="open", hold=2), cost=0.0)
    tr = run.trades.iloc[0]
    assert tr["exit"] == df.index[3] and tr["bars"] == 2  # bars 2 (entry) and 3
    assert np.isclose(tr["ret"], 0.10)


def test_close_entry_hold_one_is_next_close():
    df = flat()
    df.loc[df.index[2], "close"] = 105.0
    run = backtest(df, Strategy("x", only(1), entry="close", hold=1), cost=0.0)
    tr = run.trades.iloc[0]
    assert tr["exit"] == df.index[2] and np.isclose(tr["ret"], 0.05)


def test_stop_wins_a_tie_and_gaps_fill_at_open():
    df = flat()
    atr = swing.atr(df).iloc[1]
    df.loc[df.index[3], ["low", "close"]] = [100 - 5 * atr, 120.0]  # stop and huge close same bar
    run = backtest(df, Strategy("x", only(1), entry="open", hold=2, stop_atr=1.0), cost=0.0)
    tr = run.trades.iloc[0]
    assert tr["reason"] == "stop" and np.isclose(tr["ret"], -atr / 100)
    df2 = flat()
    df2.loc[df2.index[3], ["open", "low"]] = [90.0, 89.0]
    tr2 = backtest(df2, Strategy("x", only(1), entry="open", hold=5, stop_atr=1.0), cost=0.0).trades.iloc[0]
    assert tr2["reason"] == "stop_gap" and np.isclose(tr2["ret"], -0.10)


def test_costs_charged_on_both_fills():
    df = flat()
    tr = backtest(df, Strategy("x", only(1), entry="open", hold=2), cost=1e-3).trades.iloc[0]
    assert np.isclose(tr["ret"], (1 - 1e-3) / (1 + 1e-3) - 1)


def test_daily_marks_compound_to_trade_returns():
    df = ohlcv(800, seed=3)
    for s in [FAMILY["tt_moc"], FAMILY["es_rsi2"], FAMILY["gold_thursday"]]:
        run = backtest(df, s)
        assert len(run.trades) > 5
        assert np.isclose(np.prod(1 + run.daily.to_numpy()), np.prod(1 + run.trades["ret"].to_numpy()))
        assert run.exposure.max() <= 1  # one position at a time


def test_signals_inside_a_trade_are_ignored():
    df = flat(20)
    always = lambda d: (pd.Series(True, index=d.index), None)
    tr = backtest(df, Strategy("x", always, entry="close", hold=3), cost=0.0).trades
    assert (tr["t"].diff().dropna() > 3).all()


def test_thursday_means_thursday():
    df = ohlcv(300, seed=4)
    e, _ = gold_thursday(rsi_threshold=101).signal(df)
    assert set(df.index[e.to_numpy()].day_name()) == {"Thursday"}
    e2, _ = turnaround_tuesday(threshold=1.0).signal(df)
    assert set(df.index[e2.to_numpy()].day_name()) == {"Monday"}


def test_random_entry_control_does_not_flag_noise():
    """Driftless data, a rule that fires on a coin flip: p must not be tiny."""
    df = ohlcv(2000, seed=5, drift=0.0)
    coin = np.random.default_rng(1).random(len(df)) < 0.1
    s = Strategy("coin", lambda d: (pd.Series(coin[: len(d)], index=d.index), None), entry="open", hold=3)
    run = backtest(df, s)
    p, _ = swing.random_entry_p(df, s, run.trades, sims=300)
    assert p > 0.01


def test_random_entry_control_finds_a_planted_edge():
    df = ohlcv(2000, seed=6, drift=0.0)
    df.loc[df.index.weekday == 4, "close"] *= 1.01  # Friday closes jump
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    s = Strategy("thu", lambda d: (pd.Series(d.index.weekday == 3, index=d.index), None), entry="close", hold=1)
    run = backtest(df, s)
    p, _ = swing.random_entry_p(df, s, run.trades, sims=300)
    assert p < 0.01


def test_perturb_keeps_bars_valid():
    df = ohlcv(500, seed=7)
    p = swing.perturb(df, np.random.default_rng(0))
    assert (p["high"] >= p[["open", "close"]].max(axis=1) - 1e-12).all()
    assert (p["low"] <= p[["open", "close"]].min(axis=1) + 1e-12).all()
    assert not np.allclose(p["close"], df["close"])


def test_parameter_maps_have_registered_sizes():
    sizes = {"tt_moc": 12, "tt_open": 12, "es_rsi2": 30, "spy_volume": 18, "gold_thursday": 80,
             "gold_thursday_trend": 80}
    for k, n in sizes.items():
        assert len(parameter_map(k)) == n


def test_shifted_schedule_keeps_trade_count():
    df = ohlcv(1500, seed=8)
    s = FAMILY["tt_moc"]
    run = backtest(df, s)
    d = swing.shifted_daily(df, s, run.trades, offset=137)
    assert np.isfinite(d).all() and (d != 0).sum() > 0


def _panel(n=700, k=30, seed=9):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0004, 0.02, (n, k)), axis=0), index=idx)
    elig = pd.DataFrame(True, index=idx, columns=px.columns)
    elig.iloc[:, 20:] = False
    return px, elig


def test_pullback_respects_slots_eligibility_and_max_hold():
    px, elig = _panel()
    w, trades = pullback(px, elig, rsi_n=2, threshold=20, max_hold=4, slots=5)
    assert (w.gt(0).sum(axis=1) <= 5).all()
    assert (w.iloc[:, 20:] == 0).all().all()
    assert max(age for _, age, _ in trades) <= 4
    assert len(trades) > 20


def test_random_twin_matches_exposure():
    px, elig = _panel()
    w, trades = pullback(px, elig, rsi_n=2, threshold=20, max_hold=4, slots=5)
    tw = random_twin(px, elig, trades, 5, np.random.default_rng(0))
    assert (tw.iloc[:, 20:] == 0).all().all()
    assert abs(tw.to_numpy().sum() - w.to_numpy().sum()) / w.to_numpy().sum() < 0.05
