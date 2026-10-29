import numpy as np
import pandas as pd
import pytest

from quant import nq
from quant import nq_strategies as ns


def synth_1m(n_days=60, seed=0, vol=0.0006):
    """Random-walk 1-minute bars 04:00-20:00 New York on weekdays, as a UTC frame."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2019-03-01", periods=n_days)
    idx = []
    for d in days:
        idx.append(pd.date_range(d + pd.Timedelta(hours=4), d + pd.Timedelta(hours=20), freq="1min",
                                 inclusive="left", tz="America/New_York"))
    idx = idx[0].append(idx[1:])
    c = 7000 * np.cumprod(1 + rng.normal(0, vol, len(idx)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 3, len(idx))))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 3, len(idx))))
    q = lambda x: np.round(x / nq.TICK) * nq.TICK
    return pd.DataFrame({"open": q(o), "high": q(h), "low": q(l), "close": q(c)}, index=idx.tz_convert("UTC"))


@pytest.fixture(scope="module")
def days():
    return nq.to_days(synth_1m())


def test_grid_is_rth_new_york(days):
    raw = synth_1m()
    ny = raw.tz_convert("America/New_York")
    first = ny[ny.index.normalize().tz_localize(None) == pd.Timestamp(days.dates[0])]
    first = first.between_time("09:30", "15:59")
    assert days.o[0, 0] == first["open"].iloc[0]
    assert days.c[0, -1] == first["close"].iloc[-1]
    assert days.o.shape[1] == 390


def test_context_uses_prior_days_only():
    raw = synth_1m()
    d1 = nq.to_days(raw)
    i = 40
    day = pd.Timestamp(d1.dates[i])
    ny = raw.index.tz_convert("America/New_York")
    hit = (ny.normalize().tz_localize(None) == day)
    raw2 = raw.copy()
    raw2.loc[hit, ["open", "high", "low", "close"]] *= 1.05
    d2 = nq.to_days(raw2)
    for k in ["prev_close", "prev_high", "ema_up", "hivol"]:
        assert d1.ctx[i][k] == d2.ctx[i][k]
    np.testing.assert_array_equal(d1.ctx[i]["sigma"], d2.ctx[i]["sigma"])
    assert d1.ctx[i + 1]["prev_close"] != d2.ctx[i + 1]["prev_close"]


@pytest.mark.parametrize("name", list(ns.STRATEGIES))
def test_entry_prefix_invariance(days, name):
    """Truncating the day just after the entry bar must give the same entry, and truncating
    before it must give no entry yet (or the same one)."""
    checked = 0
    for r in range(len(days.dates)):
        ctx = days.ctx[r]
        if ctx is None:
            continue
        o, h, l, c = days.o[r], days.h[r], days.l[r], days.c[r]
        full = ns.STRATEGIES[name](o, h, l, c, ctx)
        for cut in [20, 75, 200, 340, 380]:
            part = ns.STRATEGIES[name](o[:cut], h[:cut], l[:cut], c[:cut], ctx)
            if full is not None and full.entry_bar < cut:
                assert part is not None
                assert (part.entry_bar, part.entry_px, part.side, part.stop, part.target) == \
                       (full.entry_bar, full.entry_px, full.side, full.stop, full.target)
            else:
                assert part is None
        if full is not None:
            j = full.entry_bar
            part = ns.STRATEGIES[name](o[: j + 1], h[: j + 1], l[: j + 1], c[: j + 1], ctx)
            assert part is not None and part.entry_bar == j and part.entry_px == full.entry_px
            checked += 1
    assert checked > 0, f"{name} never traded on synthetic data"


@pytest.mark.parametrize("name", list(ns.STRATEGIES))
def test_fills_are_traded_prices(days, name):
    for r in range(len(days.dates)):
        if days.ctx[r] is None:
            continue
        t = ns.STRATEGIES[name](days.o[r], days.h[r], days.l[r], days.c[r], days.ctx[r])
        if t is None:
            continue
        j = t.entry_bar
        assert days.l[r, j] <= t.entry_px <= days.h[r, j]
        if np.isfinite(t.stop):
            assert t.side * (t.entry_px - t.stop) > 0


def flat_day(price=100.0):
    a = np.full(390, price)
    return a.copy(), a.copy(), a.copy(), a.copy()


def test_round_trip_cost():
    o, h, l, c = flat_day()
    x = nq.simulate_one(o, h, l, c, nq.Trade(10, 100.0, 1))
    assert x[2] == "time"
    assert x[3] == pytest.approx(-(2 * nq.TICK * nq.POINT) - nq.COMMISSION)
    x2 = nq.simulate_one(o, h, l, c, nq.Trade(10, 100.0, 1), cost_mult=2)
    assert x2[3] == pytest.approx(2 * x[3])


def test_stop_wins_tie_and_gap_fills_at_open():
    o, h, l, c = flat_day()
    h[12], l[12] = 105, 95
    x = nq.simulate_one(o, h, l, c, nq.Trade(10, 100.0, 1, stop=97, target=103))
    assert x[:3] == (12, 97, "stop")
    o2, h2, l2, c2 = flat_day()
    o2[12] = h2[12] = l2[12] = c2[12] = 90
    x = nq.simulate_one(o2, h2, l2, c2, nq.Trade(10, 100.0, 1, stop=97))
    assert x[:3] == (12, 90, "stop")


def test_target_needs_a_tick_through():
    o, h, l, c = flat_day()
    h[12] = 103
    assert nq.simulate_one(o, h, l, c, nq.Trade(10, 100.0, 1, stop=97, target=103))[2] == "time"
    h[13] = 103 + nq.TICK
    x = nq.simulate_one(o, h, l, c, nq.Trade(10, 100.0, 1, stop=97, target=103))
    assert x[:3] == (13, 103, "target")
    assert x[3] == pytest.approx(3 * nq.POINT - nq.TICK * nq.POINT - nq.COMMISSION)


def test_live_trade_stays_open_on_partial_day():
    o, h, l, c = flat_day()
    assert nq.simulate_one(o[:50], h[:50], l[:50], c[:50], nq.Trade(10, 100.0, 1, stop=97)) is None


def test_vectorised_matches_single(days):
    for name in ns.STRATEGIES:
        rows = [r for r in range(len(days.dates)) if days.ctx[r] is not None]
        tr = ns.generate(days, f"{name}|none", rows)
        tf = nq.trades_frame(days, tr)
        v = nq.run(days, tf)
        for k, (r, t) in enumerate(sorted(tr.items())):
            one = nq.simulate_one(days.o[r], days.h[r], days.l[r], days.c[r], t)
            assert one[3] == pytest.approx(v[k]), (name, r)


def test_coin_flip_on_random_walk_is_not_significant(days):
    rows = [r for r in range(len(days.dates)) if days.ctx[r] is not None]
    tf = nq.trades_frame(days, ns.generate(days, "orb30|none", rows))
    sims = nq.coin_flip(days, tf, sims=60, seed=1)
    real = nq.run(days, tf).sum()
    assert len(sims) == 60 and np.isfinite(sims).all()
    # random walk: the real strategy is just one more coin flip
    assert 0.0 < (sims >= real).mean() < 1.0 or len(tf) < 5


def test_prop_eval():
    daily = np.array([1000, 1000, 1000, 0, 0])
    out, used = nq.prop_eval(daily, 1)
    assert out[0] == 1 and used[0] == 3
    out, used = nq.prop_eval(np.array([500, -1000, -1000, -600, 0]), 1)
    assert out[0] == 0 and used[0] == 3
    out, _ = nq.prop_eval(np.array([100, 100]), 1)
    assert (out == -1).all()


def test_histdata_clock_is_new_york_local():
    import io
    raw = "20240715 093000;1;2;0.5;1.5;0\n20240115 093000;1;2;0.5;1.5;0\n"
    df = nq.load_histdata_csv(io.StringIO(raw))
    assert list(df.index) == [pd.Timestamp("2024-01-15 14:30", tz="UTC"),
                              pd.Timestamp("2024-07-15 13:30", tz="UTC")]


def test_filters():
    t = nq.Trade(5, 100.0, 1)
    assert ns.keep(t, {"ema_up": True, "hivol": False}, "ema")
    assert not ns.keep(t, {"ema_up": False, "hivol": False}, "ema")
    assert ns.keep(t, {"ema_up": False, "hivol": False}, "lovol")
    assert len(ns.VARIANTS) == 40
