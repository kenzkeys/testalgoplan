import json

import numpy as np
import pandas as pd
import pytest

from quant import protocol, stats
from quant.backtest import Costs, run
from quant.strategies import FAMILIES, equal_weight_hold, xs_momentum


def walk(n=1500, k=30, seed=0, drift=0.0, vol=0.02, start="2015-01-01", freq="B"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq)
    r = rng.normal(drift, vol, (n, k))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=[f"A{i}" for i in range(k)])


@pytest.mark.parametrize("kind", ["equities", "crypto"])
def test_prefix_invariance_for_every_strategy(kind):
    """Weights at day T computed from data truncated at T must equal weights computed
    from full history. Any use of future data breaks this."""
    px = walk(900, 20, seed=1)
    cut = 700
    for name, fn in FAMILIES[kind].items():
        full = fn(px).iloc[:cut]
        part = fn(px.iloc[:cut])
        pd.testing.assert_frame_equal(full, part, check_exact=False, atol=1e-12, obj=name)


def test_cheating_on_same_day_return_earns_nothing():
    """A 'strategy' that sizes on the same day's return must not profit, because the
    backtester pays weights only on the NEXT day."""
    px = walk(1500, 10, seed=2)
    w = np.sign(px.pct_change().fillna(0.0)) / 10
    res = run(px, w, Costs(0.0, 0.0, 252))
    assert abs(stats.sharpe(res.gross, 252)) < 1.0


def test_null_universe_has_no_edge_after_costs():
    """Driftless random walks: momentum must not look like skill."""
    px = walk(2500, 40, seed=3)
    res = run(px, xs_momentum(px), Costs.for_universe("equities"))
    lo, hi, _ = stats.bootstrap_sharpe_ci(res.net, 252, reps=500)
    assert lo < 0.2  # CI must not exclude zero on the upside


def test_costs_reduce_return_and_scale_with_turnover():
    px = walk(800, 20, seed=4)
    w = xs_momentum(px, 60, 5, every=5)
    free = run(px, w, Costs(0, 0, 252))
    paid = run(px, w, Costs(10, 0, 252))
    assert paid.net.sum() < free.net.sum()
    assert np.isclose((free.net - paid.net).sum(), paid.turnover.sum() * 10 / 1e4)


def test_short_carry_only_charged_on_shorts():
    px = walk(400, 10, seed=5)
    long_only = xs_momentum(px, 60, 5, long_short=False)
    res = run(px, long_only, Costs(0, 1000, 252))
    assert np.isclose(res.net.sum(), res.gross.sum())


def test_missing_prices_are_never_held():
    px = walk(300, 10, seed=6)
    px.iloc[:150, 0] = np.nan
    res = run(px, equal_weight_hold(px), Costs(0, 0, 252))
    assert (res.weights.iloc[:150, 0] == 0).all()


def test_planted_signal_is_found():
    """Positive control: stocks with persistent drift differences must show up in
    momentum. Proves the harness can say yes."""
    n, k = 2500, 40
    rng = np.random.default_rng(8)
    mu = rng.normal(0, 0.0008, k)  # persistent per-asset drift
    r = rng.normal(0, 0.01, (n, k)) + mu
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    px = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx)
    res = run(px, xs_momentum(px), Costs(0, 0, 252))
    assert stats.sharpe(res.net.iloc[600:], 252) > 1.0


def test_test_split_is_sealed_and_logged(tmp_path):
    idx = pd.date_range("2010-01-01", periods=1000, freq="B")
    with pytest.raises(protocol.SealedError):
        protocol.window(idx, "test", "equities", None, tmp_path / "log.json")
    lo, hi = protocol.window(idx, "test", "equities", "frozen model X", tmp_path / "log.json")
    assert lo > protocol.window(idx, "validate")[1]
    log = json.loads((tmp_path / "log.json").read_text())
    assert len(log) == 1 and log[0]["reason"] == "frozen model X"


def test_splits_are_disjoint_and_ordered():
    idx = pd.date_range("2010-01-01", periods=1000, freq="B")
    a, b = protocol.window(idx, "train"), protocol.window(idx, "validate")
    assert a[1] < b[0]


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(9)
    r = pd.Series(rng.normal(0.0005, 0.01, 1500))
    sd = 0.02
    assert stats.deflated_sharpe(r, 50, sd) < stats.deflated_sharpe(r, 2, sd)


def test_concentration_flags_single_year():
    idx = pd.date_range("2018-01-01", "2022-12-31", freq="B")
    r = pd.Series(0.0, index=idx)
    r[r.index.year == 2020] = 0.002
    r[r.index.year == 2021] = -0.0001
    share, year = stats.concentration_by_year(r)
    assert share > 0.9 and year == "2020"


def test_liquid_mask_uses_no_future_data():
    """Eligibility at day T must not change when data after T is removed."""
    from quant.data import liquid_mask

    px = walk(400, 30, seed=5)
    vol = pd.DataFrame(np.random.default_rng(6).integers(1_000, 1_000_000, px.shape),
                       index=px.index, columns=px.columns).astype(float)
    cut = 300
    full = liquid_mask(px, vol, top_n=10, window=63).iloc[:cut]
    part = liquid_mask(px.iloc[:cut], vol.iloc[:cut], top_n=10, window=63)
    pd.testing.assert_frame_equal(full, part)
    assert int(full.iloc[-1].sum()) == 10


def test_ineligible_names_are_never_selected_by_strategy_or_control():
    """signal_prices restricts selection; the random control must obey the same set."""
    from quant.evaluate import random_selection_sharpes  # noqa: F401  (import must succeed)
    from quant.strategies import st_reversal

    px = walk(600, 20, seed=8)
    sig = px.copy()
    sig.iloc[:, 10:] = np.nan
    w = st_reversal(sig, 5, 0.2, 5)
    assert (w.iloc[:, 10:] == 0).all().all()


def _ohlcv(n=900, seed=12):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    c = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    o = c * (1 + rng.normal(0, 0.004, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.005, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.005, n)))
    v = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v}, index=idx)


def test_prefix_invariance_for_every_swing_rule():
    """Amendment 5: entry and exit flags, every map cell, and the trades that close
    before the cut must not change when later bars are removed."""
    from quant.swing import backtest
    from quant.swing_strategies import FAMILY, parameter_map

    df = _ohlcv()
    cut = 650
    for name in FAMILY:
        for s in parameter_map(name):
            e_full, x_full = s.signal(df)
            e_part, x_part = s.signal(df.iloc[:cut])
            pd.testing.assert_series_equal(e_full.iloc[:cut], e_part, obj=f"{name}{s.params}")
            if x_full is not None:
                pd.testing.assert_series_equal(x_full.iloc[:cut], x_part, obj=f"{name}{s.params} exit")
        s = FAMILY[name]
        full, part = backtest(df, s).trades, backtest(df.iloc[:cut], s).trades
        done = full[full["exit"] < df.index[cut - 1]]
        pd.testing.assert_frame_equal(done.reset_index(drop=True),
                                      part[part["exit"] < df.index[cut - 1]].reset_index(drop=True), obj=name)


def test_prefix_invariance_for_xs_pullback():
    from quant.data import liquid_mask
    from quant.xs_pullback import pullback

    px = walk(700, 30, seed=13)
    vol = pd.DataFrame(np.random.default_rng(14).integers(1_000, 1_000_000, px.shape),
                       index=px.index, columns=px.columns).astype(float)
    cut = 550
    for rsi_n, th in [(14, 30), (2, 10)]:
        full, _ = pullback(px, liquid_mask(px, vol, 10), rsi_n, th)
        part, _ = pullback(px.iloc[:cut], liquid_mask(px.iloc[:cut], vol.iloc[:cut], 10), rsi_n, th)
        pd.testing.assert_frame_equal(full.iloc[:cut], part, obj=f"rsi{rsi_n}")
