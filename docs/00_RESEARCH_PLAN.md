# Research plan (pre-registration)

Universe: current S&P 500 (daily, 2004 on) and a fixed top-10 crypto list (daily, 2014 on).
Data: yfinance adjusted closes, cached in `data/`.

## Rules

1. The strategy grids are fixed in `src/quant/strategies.py` (`FAMILIES`). Nothing outside
   them is evaluated without amending this file first. Deflated Sharpe charges the size of
   the whole family (5 equities, 6 crypto), not just what was run.
2. Splits are chronological 60/20/20 per universe (`protocol.py`). All work happens on TRAIN.
   VALIDATE only chooses among TRAIN survivors. TEST opens once, for a frozen model, and
   the look is logged to `state/test_set_looks.json`.
3. Costs: equities 5 bps one way plus 50 bps/yr borrow on shorts; crypto 10 bps one way plus
   10%/yr funding on shorts. A candidate must also survive at double these costs.
4. A candidate passes only if all five checks hold on TRAIN, then again on VALIDATE:
   - bootstrap 95% CI of net Sharpe excludes 0
   - Deflated Sharpe > 0.95
   - beats its random-selection control (p < 0.05), or for timing overlays beats the
     buy-and-hold Sharpe
   - beats equal-weight buy-and-hold of the same universe (bootstrap p < 0.05)
   - no single year holds more than 50% of the positive annual return
5. Paper trading comes after TEST and measures execution, not edge.

## Known biases (cannot be fixed with free data)

- **Survivorship.** Both universes use today's membership. Delisted equities and dead coins
  are absent. This inflates every long-only return and especially buy-the-losers
  strategies, because the losers that went to zero are missing. Absolute Sharpe numbers are
  upper bounds. Benchmarks share the bias; controls and comparisons are the only usable signal.
- **Crypto history is short and one-regime.** TRAIN is 2014-2021, mostly a bull run.

## First TRAIN run (2026-10-20, grids were declared in code before this run)

Equities, 2004-01 to 2017-08: no pass. `st_reversal_5d` came closest (beats random p=0.00,
beats benchmark p=0.002) but turns over 87x a year and is the strategy most inflated by
survivorship. `trend_ew_200` has Sharpe 1.03 and a max drawdown of 26% against 51% for
buy-and-hold, but does not significantly beat it. Momentum does not beat random selection
(12-1: p=0.86; 6-1: p=0.06). The long-short momentum variant lost money.

Crypto, 2014-09 to 2021-12: no pass. Nothing beats the equal-weight basket (Sharpe 1.75),
and every variant has 45-91% of its positive return in one year.

## Next, in order

1. ~~Survivorship check for `st_reversal_5d` and `trend_ew_200`: run at 2x and 3x costs, and
   restrict to the 100 most liquid names to test capacity.~~ DONE 2026-10-26: both dead
   (Amendment 1).
2. If either survives, run VALIDATE once. If neither does, the honest result for equities
   and crypto on free data is: buy and hold, optionally with a trend filter for drawdown.
3. Only then consider paid, survivorship-free data (Sharadar, Norgate) before any live money.

## Amendment 1 (2026-10-26): exact specification of next step 1, fixed before running

Applies to `st_reversal_5d` and `trend_ew_200` only, on TRAIN only. Script:
`research/01_near_miss_stress.py`.

- **Cost stress.** Rerun with every cost multiplied by 2 and by 3. Rule 3 already
  requires surviving 2x. 3x is reported for information.
- **Capacity.** Each day the eligible universe is the 100 names with the highest
  63-day median dollar volume (adjusted close times volume), using only data up to
  that day's close. The strategy may only select eligible names, and it is judged
  against the equal-weight portfolio of the same eligible 100, at 1x costs.
- **Survival** means, in both the 2x cost run and the capacity run: beats its
  random-selection control (p < 0.05; for `trend_ew_200`, Sharpe above its
  benchmark's) AND beats its benchmark (bootstrap p < 0.05). A candidate failing
  either run is dead on free data, and VALIDATE is not opened for it.
- These runs re-test already-declared strategies and add no new strategy to the
  family, so the Deflated Sharpe trial count stays 5.

### Result (2026-10-26): both dead on free data

`results/20261026-104853_equities_train_near_miss_stress.json`. The 1x baseline
reproduced the 2026-10-20 run exactly, so data and harness are unchanged.

| run | `st_reversal_5d` vs benchmark p | `trend_ew_200` vs benchmark p |
|---|---|---|
| 1x, full universe | 0.002 | 0.905 |
| **2x costs** | **0.074 (fails)** | 0.916 (fails) |
| 3x costs | 0.540 | 0.922 |
| top-100 liquid, 1x | **0.010 (passes)** | 0.861 (fails) |

`st_reversal_5d` beats its random control in every run (p = 0.00) and survives the
liquid universe, but turns over 87x a year and fails the pre-registered 2x cost
rule. `trend_ew_200` never beats buy-and-hold; it only halves drawdown (26% vs 51%).
VALIDATE stays unopened for both.

Note for the paid-data phase: 5 bps one way is likely conservative for the 100 most
liquid names at retail size, which is exactly why the 2x rule exists and why it is
not relaxed after seeing this result. Reversal is also the strategy most inflated by
survivorship (the losers it buys that went to zero are missing). On
survivorship-free data it may be re-registered as a NEW test, with its own costs
fixed in advance; this free-data result does not carry over.

Runtime: 50 minutes, almost all in the 200-sim random-selection control for a
5-day rebalance.

## Amendment 2 (2026-10-29): the YouTube "EMA crossover trading bot", fixed before running

Source: "How to Build a Trading Bot in Python | Full Algorithmic Trading Tutorial"
(transcript supplied by the user). Script: `research/02_video_ema_bot.py`, code in
`src/quant/intraday.py`. The video demonstrates plumbing (OANDA API), not a claimed edge;
the strategy is a demo.

**Rules exactly as coded in the video**

- Bars: 15-minute. Indicators on closed candles only.
- Signal (long): EMA(5) > EMA(8) on the last closed candle AND EMA(5) < EMA(8) on the one
  before it.
- Entry: market buy at the next candle's open (the bot checks in the first 10 seconds of a
  new candle). Stop and target are absolute prices fixed from the signal candle: entry
  reference = signal close, stop = close - ATR(14), target = close + 1.5 * (close - stop).
  Exits are broker-side stop/target. No other exit, no sizing (1 unit), no limit on
  concurrent positions: every signal is a trade.
- The video only codes the long side.

**Backtest conventions (fixed now)**

- EMA is recursive (adjust=False); ATR is Wilder smoothing, length 14, true range uses the
  previous close.
- Fill at next open, adversely adjusted by the per-side cost. Stop and target are checked
  from the entry bar onward; if both lie inside one bar the STOP wins. A bar that opens
  beyond the stop fills at the open. Target fills at the target price.
- A trade unresolved after 2000 bars is closed at that bar's close and counted separately.
- Result unit is R = net price move / ATR(14) at signal, so a clean stop is about -1R.
- Cost per side in bps of price: GBPJPY 1.5, EURUSD 0.5, crypto 10, equities and ETFs 2.
  A zero-cost run is also reported to separate signal from cost.

**Variants (trial count 3 for Deflated Sharpe style reasoning)**

- A: the video exactly (long only, 15m).
- B: symmetric (adds the mirrored short), 15m.
- C: A on 1h bars (more history, 730 days), reported for sample size only.

**Universe (fixed)**: GBPJPY=X, EURUSD=X, BTC-USD, ETH-USD, SOL-USD, XRP-USD, SPY, QQQ,
AAPL, MSFT, NVDA, TSLA. Data: yfinance. No exchange API is reachable from this machine, and
yfinance serves only the last 60 days of 15-minute bars, so variants A and B are an
EXPLORATORY single-window test, not a validated one. Split per symbol: first 60% is the
verdict window; the last 40% stays unopened.

**Pass** (all required, on the 60% window): pooled expectancy CI (bootstrap by calendar day)
excludes 0 after costs; beats the random-entry control with the same stop/target geometry
and same trade count (p < 0.05); positive in at least 2 of the 3 asset classes (FX, crypto,
equities); not more than 50% of positive R from one symbol. Break-even win rate for a 1R
stop and 1.5R target is 40.0% before costs; reported next to the realised win rate.

### Result of Amendment 2 (2026-10-29): fails every variant

`results/20261029-191258_video_ema_bot.json`. Verdict windows only (first 60% of each
symbol's bars); the last 40% was never loaded. 15m variants cover roughly 5 weeks per symbol
(FX and equities 2026-08-06 to 09-25, crypto 08-31 to 10-05), so they are exploratory.

| variant | trades | win rate (break-even 40% gross) | mean R net | 95% CI (by day) | mean R gross | beats random entry |
|---|---|---|---|---|---|---|
| A: video, long only, 15m | 1340 | 34.7% | -0.659 | [-0.764, -0.551] | -0.106 | no, p = 0.9965 |
| B: symmetric, 15m | 2678 | 34.8% | -0.667 | [-0.758, -0.583] | -0.114 | no, p = 0.998 |
| C: long only, 1h, ~17 months | 3844 | 42.4% | -0.188 | [-0.244, -0.128] | +0.026 | no, p = 0.21 |

Cost per trade in R (A): BTC 1.06, ETH 0.80, SOL 0.56, XRP 0.52, GBPJPY 0.58, EURUSD 0.35,
equities 0.08 to 0.36. A stop of one 15-minute ATR is only a few basis points, so the
round-trip cost is a large fraction of the risk taken. Gross of costs the signal is
indistinguishable from entering at random (random-entry control mean R is -0.56 net, close
to the strategy's -0.66 net; the gap is the strategy's slightly negative gross edge).
All asset classes are negative in every variant. Dead on this data.

## Amendment 3 (2026-10-29): timeframe and reward:risk grid for the video bot, fixed before running

Same signal (EMA 5/8 cross on a closed bar), same entry, costs, symbols, conventions and
60% verdict windows as Amendment 2. Script: `research/03_video_bot_grid.py`.

**Grid (24 cells, long only as in the video)**
- Timeframe: 15m (60 days), 1h (730 days), 1d (full yfinance history).
- Reward:risk (target distance / stop distance): 1.0, 1.5, 2.0, 3.0.
- Stop distance: 1.0 x ATR(14) or 2.0 x ATR(14). R is measured in units of the stop
  distance, so a clean stop is about -1R in every cell.

**Multiple testing.** The best of 24 cells will look better than its true edge by chance.
A cell passes only with ALL of: day-bootstrap CI of net mean R entirely above 0;
random-entry control p < 0.05 / 24 = 0.0021 (Bonferroni); net mean R > 0 in at least 2 of 3
asset classes; no symbol above 50% of positive R.

**Then.** Only passing cells, at most the single best by net mean R, go to VALIDATE (the
next 20% of bars, the 60% to 80% slice) once. VALIDATE pass: net mean R > 0 and
random-entry p < 0.05. If no cell passes, VALIDATE stays unopened and the bot is dead on
this data. The final 20% stays sealed in every case.

### Result of Amendment 3 (2026-10-29): no cell passes; VALIDATE stays unopened

`results/20261029-192300_video_bot_grid.json`. 24 cells, 60% verdict windows.

- **15m:** all 8 cells negative net (-0.25R to -0.67R). Gross is about zero in every cell.
- **1h:** 7 of 8 negative net; best is rr 3.0 / stop 2.0 ATR at +0.029R, CI [-0.064, +0.122],
  p_random 0.367.
- **1d:** 7 of 8 positive net, best rr 3.0 / stop 2.0 ATR at **+0.425R**, CI [+0.333, +0.516].
  But **entering at random with the same geometry earns +0.353R**. The signal adds 0.07R
  over random entry, p = 0.035, which fails the pre-registered Bonferroni bar of 0.0021.

Reading: the daily profit is long-only drift in assets that rose over the window (crypto
+0.695R, equities +0.382R in the best cell), not information in the EMA cross. The same
trap as the gold-beta result in the forex project. Wider stops and larger targets help
mainly by making costs a smaller share of R and by holding longer in rising markets.
Dead on this data; nothing here justifies opening VALIDATE.

## Amendment 4 (2026-10-29): the YouTube "AI prop-firm NASDAQ bot" pipeline, fixed before running

Source: transcript supplied by the user (Claude + VS Code + Databento, "the syndicate",
"the dealer"). The video gives a process, not a strategy: honest backtest, build on
2010-2020 and test on 2021-2026, a 500-run coin-flip test, a research log, a portfolio of
survivors, a prop-firm sizing simulation, a paper trader, then live. Its own rule: "if it
can't beat a coin flip, it doesn't trade." Code: `src/quant/nq.py`, `src/quant/nq_strategies.py`,
`src/quant/dealer.py`. Script: `research/04_video_nq_bot.py`. Every variant is appended to
`state/research_log.jsonl` (append-only, never edited) before it is run.

**Data.** 1-minute NASDAQ-100 bars. Source to be fixed by the user (Databento NQ futures, or
a free NAS100 CFD feed); whichever is used is named in the result. Sessions are New York
time, regular hours 09:30-16:00, bars stamped at their start. Days with fewer than 300 of the
390 RTH bars are dropped. If the data starts after 2010, IN-SAMPLE starts at the data start.

**Splits (the video's).** IN-SAMPLE 2010-01-01 to 2020-12-31. OUT-OF-SAMPLE 2021-01-01 to the
end of the data. OOS is opened once, only for variants that pass IN-SAMPLE.

**Instrument and costs.** Micro NASDAQ (MNQ): $2 per point, tick 0.25. Per contract round
trip: $1.24 commission, plus 1 tick slippage on every market or stop fill (entries and
exits). Limit fills (targets) get no slippage but need price to trade one tick through. 2x
cost run doubles both.

**Fill rules (the video's backtest prompt).** Signals use closed bars only. Market orders
fill at the next bar's open. Stop orders fill at the trigger, or at the open if the bar
gaps through it. If stop and target are both inside one bar the stop wins. On the entry
bar only the stop is checked. Every position is flat at the close of the 15:59 bar. One
trade per strategy per day: the first signal.

**The 10 published strategies (the video's prompt 3), with who loses to each**

1. `orb5` (Zarattini, Barbon, Aziz 2023, 5-minute ORB): trade at 09:35 in the direction of the
   first 5-minute candle (skip dojis), stop at its other end, target 10R, else close.
   Loser: late chasers and liquidity providers leaning against the opening imbalance.
2. `orb30` (Crabel opening range): stop entry 1 tick beyond the 09:30-10:00 range until 15:00,
   stop at the other side of the range. Loser: traders fading the first real move of the day.
3. `ib_break` (Steidlmayer initial balance): stop entry beyond the 09:30-10:30 range until
   15:00, stop at the IB midpoint, target one IB range beyond entry. Loser: responsive sellers
   (buyers) defending the first hour's range.
4. `gap_fade`: if the 09:30 open is 0.25%-1.5% from the prior RTH close, enter against the gap
   at the 09:31 open, target the prior close, stop one gap-size beyond entry. Loser: overnight
   news traders who pay up at the open.
5. `gap_go`: same gap; if the first 5-minute candle closes in the gap's direction, enter with
   it at 09:35, stop at that candle's other end. Loser: gap faders.
6. `intraday_mom` (Gao, Han, Li, Zhou 2018): sign of the return from the prior RTH close to
   10:00 sets the side of a trade from the 15:30 open to the close, no stop. Loser: late-day
   hedgers and rebalancers who must trade at the close.
7. `noise_break` (Zarattini, Aziz, Barbon 2024, noise area): boundaries = open x (1 +/- mean
   absolute move-from-open at that minute over the prior 14 days); at the :00/:30 closes from
   10:00 to 15:30 enter beyond a boundary, stop at the opposite boundary. VWAP from the paper
   is left out (CFD data has no real volume). Loser: mean-reversion sellers inside the range.
8. `pdhl_break`: stop entry 1 tick beyond the prior RTH high or low, 09:30-15:00, stop 25% of
   the prior day's range. Loser: traders who sell resistance and buy support.
9. `turtle_soup` (Connors & Raschke 1995, intraday form): 09:30-12:00, after price trades
   beyond the prior day's high (low) a 1-minute close back inside triggers a fade at the next
   open, stop 1 tick beyond today's extreme so far, target the prior day's midpoint. Loser:
   breakout traders caught on a false break.
10. `rsi2_rev` (Connors RSI(2), on 5-minute bars): RSI(2) below 5 (above 95) at a 5-minute
    close from 09:50 to 15:00, enter at the next open, exit at the first 5-minute close
    above (below) the 5-period SMA, stop 2 x ATR(14) of 5-minute bars. Loser: panic sellers
    and stop-runs at short-term extremes.

**Filters (the video's step 4), a fixed grid, not tweaked after seeing results.** For each
strategy: `none`; `ema` (longs only if the 20-day EMA of RTH closes, as of the prior day, is
above its value 5 days earlier, shorts only if below); `hivol` (only if the prior day's RTH
range is at or above its 20-day median); `lovol` (below it). 10 x 4 = **40 variants**. That
is the trial count, and the research log states it next to every winner.

**Coin-flip test (the video's step 5).** 500 random versions per variant: the same days, the
same number of trades, each with the real trade's stop and target distances in points and
its maximum holding time, but a random side and a random entry minute (09:31 to 15:30).
p = (1 + number of random versions with total net P&L at or above the strategy's) / 501.

**IN-SAMPLE pass (all required).** At least 100 trades; net P&L > 0 at 1x and at 2x costs;
day-bootstrap 95% CI of mean net P&L per trade above 0; coin-flip p = 1/501 (no random
version better: the closest to a Bonferroni bar of 0.05/40 that 500 runs can resolve); no
single year above 50% of the positive annual P&L.

**OUT-OF-SAMPLE pass (the video: "it has to be a winner on both").** Net P&L > 0 at 1x and
2x costs and coin-flip p < 0.05.

**Portfolio (step 7)** only of variants passing both; one MNQ each; reported with its worst
drawdown and longest losing stretch against each member. **Prop-firm sizing (step 9)**: a
generic 50K evaluation, not a named firm: +$3,000 target, $2,000 trailing drawdown on
end-of-day balance, no daily loss limit; start on every trading day; 1, 2, 3, 5 and 10 MNQ;
report pass rate, median days to pass and attempts per pass, with a coin-flip version at
every size. If nothing passes, the prop simulation runs on the single best IN-SAMPLE
variant for information only, labelled as not a pass, and there is no portfolio.

**Paper trader (step 8)** watches NQ=F 1-minute bars from yfinance (delayed) every minute on
weekdays 09:00-16:00 ET, runs passing variants with the same code, logs every simulated
entry and exit to `state/dealer_log.csv`, and posts to Discord only if a webhook file exists.
It never places orders. **Live (step 10)** is not built: nothing goes to a broker until a
variant has passed every test above and a paper period.

## Amendment 5 (2026-10-29): the Chart Fanatics "algo trader, no code" (SQX) video, fixed before running

Source: Chart Fanatics interview transcript supplied by the user (StrategyQuant X, "Ultra
Instinct"). The guest's verified-profit claims are not evidence for any rule below. The
video gives a process (two or three entry rules, one or two exits, in-sample and
out-of-sample, a profitable parameter map, Monte Carlo reshuffle, other markets, randomised
price data, incubation, ensemble voting, sizing so the average loss is 0.5% of the
allocation) and four example strategies. Code: `src/quant/swing.py` (engine and robustness
tests), `src/quant/swing_strategies.py` (rules), `src/quant/xs_pullback.py` (the S&P 100
ranked strategy). Script: `research/05_sqx_video.py`.

**Data.** yfinance daily OHLCV, `auto_adjust=True`, cached in `data/swing/`: SPY, QQQ, IWM,
DIA, EFA, GLD, SLV, GDX, TLT. Any bar dated on the download day is dropped (may be partial).
The video trades ES and GC futures; yfinance continuous futures are not back-adjusted, so
roll days would create fake gaps that trigger ATR stops. SPY stands in for ES and GLD for
GC. These ETFs exist today and existed throughout, so the single-instrument tests here are
**free of survivorship bias**, unlike every earlier test in this file.

**Verdict window.** Each symbol's first 60% of bars, sliced off before any indicator is
computed (TRAIN). The rest is not loaded. VALIDATE (60-80%) opens once, only for a
strategy that passes TRAIN. TEST stays sealed.

**Engine conventions (daily bars, one position per strategy, long only).**
- A signal is evaluated on the close of bar t, using bars up to t only.
- Entry `open`: fill at the open of t+1 (the SQX default). Entry `close`: fill at the close
  of t (market-on-close; the live version decides from a quote minutes before the close,
  so this is slightly optimistic and is stated next to every `close` result).
- Holding N bars: exit at the close of the Nth bar of the trade. For `open` entries the
  entry bar is bar 1; for `close` entries bar 1 is t+1.
- Signal exits fire at the close of the first bar of the trade where the exit rule is true.
- ATR stop: fill price minus k x ATR(14) as of bar t (Wilder). Checked through the low on
  every bar on which the position is held; a bar that opens below the stop fills at the
  open. If the stop and an exit both apply on one bar, the stop wins.
- Signals that arrive while a position is open are ignored.
- Cost 2 bps of price per side, applied adversely to every fill. The 2x run uses 4 bps.
- Daily returns are marked to market (flat days earn 0) for Sharpe, drawdown and exposure.

**The family (9 strategies; Bonferroni bar 0.05 / 9 = 0.0056)**

| id | instrument | rule (main cell) |
|---|---|---|
| `tt_moc` | SPY | Turnaround Tuesday: a Monday bar closing below the prior close; buy at Monday's close, exit Tuesday's close (hold 1) |
| `tt_open` | SPY | same signal, buy Tuesday's open, exit Tuesday's close (hold 1) |
| `es_rsi2` | SPY | the video's "long-only S&P mean reversion with a regime": close > SMA(200) and RSI(2) < 10; buy next open; exit at the first close above SMA(5) |
| `spy_volume` | SPY | the ensemble's "volume" member: close < prior close, volume > 1.5 x SMA(20) of volume, close > SMA(200); buy next open; hold 5 |
| `ensemble_spy` | SPY | the video's voting ensemble: `tt_moc`, `es_rsi2`, `spy_volume` each run as a 1/3 sleeve, so agreement stacks size automatically |
| `gold_thursday` | GLD | the video's gold strategy exactly: a Thursday bar with RSI(14) < 40; buy next open (Friday); stop 1 x ATR(14); hold 3 |
| `gold_thursday_trend` | GLD | the above plus close > SMA(200) (added here, one extra trial) |
| `sp_rsi14` | top 100 S&P 500 names | the video's S&P 100 ranked strategy: close > SMA(250) and RSI(14) < 30, most oversold first, at most 10 positions of 10%; exit at a close above SMA(5) or after 10 days |
| `sp_rsi2` | same | same with RSI(2) < 10 (Connors form, one extra trial) |

`sp_*` use the cached S&P 500 close and volume panel (survivorship-biased, upper bound).
"S&P 100" is the 100 names with the highest 63-day median dollar volume as of each close
(`data.liquid_mask`). The video's "close above $10" filter is dropped because adjusted
historical prices make it meaningless, and "volume above 25,000 shares" is implied by the
top-100 filter. Weights are decided at the close and held from the next day (the existing
harness convention) at the rule-3 equities costs.

**Parameter maps (the video's "map of profitable parameters").** Main cell in bold.
- `tt_*`: Monday return threshold {**< 0**, < -0.25%, < -0.5%, < -1%} x hold {**1**, 2, 3}: 12 cells.
- `es_rsi2`: RSI length {**2**, 3, 4} x threshold {5, **10**, 15, 20, 25} x exit SMA {**5**, 10}: 30.
- `spy_volume`: multiple {1.25, **1.5**, 2.0} x volume SMA {**20**, 50} x hold {3, **5**, 7}: 18.
- `gold_*`: RSI(14) threshold {30, 35, **40**, 45, 50} x stop ATR {0.5, **1.0**, 1.5, 2.0} x hold {2, **3**, 4, 5}: 80.
- `sp_rsi14`: threshold {20, 25, **30**, 35, 40} x max hold {5, **10**, 20}: 15.
- `sp_rsi2`: threshold {2, 5, **10**, 15, 20} x max hold {5, **10**, 20}: 15.
- `ensemble_spy`: no map of its own; each member's map is its own check.

**Robustness tests (the video's workflow, made into hard gates).**
- Random-entry control: 1000 runs with the same number of trades, each keeping the real
  trade's holding length and stop multiple, at random entry bars of the same window and
  entry mode. p = (1 + runs with mean net trade return >= the strategy's) / 1001.
- Other markets (same rules, each market's own first 60%, 2 bps): SPY strategies on QQQ,
  IWM, DIA, EFA; gold strategies on SLV, GDX, TLT.
- Randomised price data: 200 runs; every open, high, low and close is multiplied by
  (1 + e), e ~ N(0, 0.1 x ATR(14)/close of that bar on clean data), then high and low are
  widened to contain open and close.
- Monte Carlo reshuffle (reported, used for sizing, not a gate): 2000 random orderings of
  the trade list; 95th percentile max drawdown and longest losing streak.
- Sizing (reported): the video's rule, the fraction of equity that makes the average losing
  trade cost 0.5% of equity, capped at 1 (no leverage).

**PASS on TRAIN, single-instrument strategies (all required):**
1. at least 100 trades;
2. trade-bootstrap 95% CI of mean net trade return entirely above 0;
3. mean net trade return above 0 at 2x costs;
4. random-entry p < 0.0056;
5. daily Sharpe (flat = 0) at least the buy-and-hold Sharpe of the same ETF, same window;
6. at least 75% of parameter-map cells with mean net trade return above 0;
7. mean net trade return above 0 on a majority of the other markets (3 of 4; 2 of 3);
8. at least 80% of randomised-price runs with mean net trade return above 0;
9. no single year above 50% of the positive annual P&L.

**PASS on TRAIN, `ensemble_spy`:** stationary-bootstrap Sharpe CI above 0; Sharpe above 0 at
2x costs; random-sleeves control (each sleeve's whole trade schedule moved by an
independent random circular shift inside the window, which keeps trade count, holding
lengths, spacing and exposure so random trades never overlap within a sleeve; settled
before the first run; 1000 runs) p < 0.0056 on Sharpe; Sharpe at least SPY buy-and-hold and at least its best
single sleeve (otherwise voting adds nothing); Sharpe above buy-and-hold on 3 of 4 other
markets; year concentration as 9. Reported only: mean next-day SPY return on days with 1, 2
and 3 sleeves long, which tests the video's claim that agreement means a better signal.

**PASS on TRAIN, `sp_*`:** the five rule-4 checks (Sharpe CI above 0; Deflated Sharpe above
0.95 with trials = 9 and the trial Sharpe dispersion taken across that strategy's map
cells; random control p < 0.0056; beats the equal-weight top-100 benchmark, p < 0.05; year
concentration), plus beating the benchmark at 2x costs (p < 0.05) and at least 75% of map
cells with positive mean active return against the benchmark. Random control: the same
entry days and holding lengths, names drawn at random from that day's eligible names not
already held; 200 runs (p resolves to 0.005). No noise or other-market test for these.

**VALIDATE (once, passers only):** mean net trade return above 0 at 2x costs and random-entry
p < 0.05 (`ensemble_spy`: Sharpe above 0 at 2x and random-sleeves p < 0.05; `sp_*`: beats
the benchmark at 2x, p < 0.05). Anything that passes VALIDATE goes to incubation (paper
signals, no orders), the video's own next step. Nothing reaches a broker from this amendment.

### Result of Amendment 5 (2026-10-29): one survivor, `tt_moc`; everything else fails

`results/20261029-202333_sqx_video.json` (TRAIN, and VALIDATE for the two TRAIN passers) and
`results/20261029-202447_sqx_video_validate_ensemble.json`. Runtime about 20 minutes. Costs
2 bps per side. Mean trade is net of costs.

| strategy | TRAIN window | trades | win | mean trade (CI) | random p | Sharpe vs B&H | TRAIN | VALIDATE |
|---|---|---|---|---|---|---|---|---|
| `tt_moc` | SPY 1993-2013 | 432 | 55.6% | +0.171% (+0.034, +0.315) | 0.004 | 0.53 vs 0.52 | pass | **pass**: 144 trades, +0.127% (2x +0.087%), p 0.042 |
| `tt_open` | SPY | 432 | 53.0% | +0.051% (-0.069, +0.173) | 0.027 | 0.18 vs 0.52 | fail | unopened |
| `es_rsi2` | SPY | 149 | 79.9% | +0.584% (+0.301, +0.859) | 0.002 | 0.66 vs 0.52 | pass | **fail**: 54 trades, +0.182%, p 0.36 |
| `spy_volume` | SPY | 152 | 57.2% | +0.253% (-0.080, +0.575) | 0.233 | 0.30 vs 0.52 | fail | unopened |
| `gold_thursday` | GLD 2004-2018 | 115 | 55.7% | +0.337% (+0.050, +0.640) | 0.020 | 0.58 vs 0.52 | fail (random p only) | unopened |
| `gold_thursday_trend` | GLD | 25 | 68.0% | +0.574% (-0.191, +1.277) | 0.051 | 0.42 vs 0.52 | fail | unopened |
| `ensemble_spy` | SPY | 733 | | Sharpe 0.71 (0.36, 1.05) | 0.003 | 0.71 vs 0.52 | pass | **fail**: Sharpe 0.39 vs B&H 1.06, p 0.40 |
| `sp_rsi14` | top 100, 2004-2017 | 580 | | Sharpe 0.71 vs bench 0.60 | 0.005 | active p 0.97 | fail | unopened |
| `sp_rsi2` | top 100 | 4877 | | Sharpe 0.92 vs bench 0.60 | 0.005 | active p 0.52 | fail | unopened |

**`tt_moc` detail (TRAIN).** Exposure 8% of days; CAGR 3.5% against 8.5% buy-and-hold, max
drawdown 17% against 55%. Average win +1.06%, average loss -0.94%, profit factor 1.41. Every
one of the 12 parameter cells is positive; positive on QQQ, IWM, DIA and EFA; positive in all
200 randomised-price runs; top year 26% of positive P&L. Monte Carlo reshuffle: 95th
percentile drawdown 22.8% at full size, longest losing streak 10. The video's sizing rule
(average loss = 0.5% of equity) gives 53% of the allocation per trade. Caveats: it is a
market-on-close entry (the live version decides from a quote a few minutes earlier; the
incubation log measures that gap), and the next-open version (`tt_open`) fails, so the
edge, if real, is the Monday-close-to-Tuesday-close hold, overnight included. The VALIDATE
random-entry p of 0.042 clears its 0.05 bar without much room.

**What the rest says.**
- `es_rsi2` and the ensemble were the strongest TRAIN results and both failed VALIDATE.
  In 2013-2020 random entries with the same holding lengths did as well (es_rsi2 p 0.36),
  and a strategy in the market 10-25% of the time cannot keep up with a Sharpe-1.06 bull
  market. This is the in-sample/out-of-sample lesson from the video, in this repo's data.
- The video's gold rule is not noise-fragile (all 80 map cells and all 200 noise runs
  positive, positive on SLV, GDX and TLT) but it does not clear the multiple-testing bar
  (p 0.020 vs 0.0056). Weekday check, mean trade by signal day: Mon -0.02%, Tue +0.03%,
  Wed +0.22%, Thu +0.34%, Fri +0.10%. Adding a trend filter (my addition) cut it to 25
  trades and made it worse.
- The ensemble's voting claim held on TRAIN (mean SPY return on days with 0, 1, 2, 3
  sleeves long: +0.002%, +0.115%, +0.221%, +0.466%, the last on 19 days) but the
  ensemble failed VALIDATE.
- The S&P 100 ranked pullback beats random stock selection but not the equal-weight
  top-100 basket (fails at 1x and 2x). `sp_rsi2` had CAGR 12.2% vs 10.9% for the basket
  with a 17% vs 55% drawdown, active p 0.52; on survivorship-biased data that is not a
  pass, and 2 of 3 map cells failed.

**Next.** `tt_moc` goes to incubation: `research/06_incubate_tt_moc.py` logs paper signals
to `state/incubation_tt_moc.csv` (no orders). TEST (the final 20%, 2020-2026 for SPY)
stays sealed; opening it once for the frozen `tt_moc` main cell is the next decision.

## Amendment 6 (2026-10-29): incubation protocol for `tt_moc`, fixed before the first paper signal

Frozen model: the `tt_moc` main cell exactly (Monday close below the prior close, buy at
Monday's close, sell at the next trading day's close), SPY, no parameter changes. Script:
`research/06_incubate_tt_moc.py`, log `state/incubation_tt_moc.csv`. Paper only; no order
reaches any broker.

**What incubation measures.** At about 21 trades a year, 26 weeks gives roughly 10 trades,
far too few to confirm or deny an edge of +0.13% per trade. So incubation measures
execution, not edge:
1. **Signal agreement**: every Monday's live decision (from the 15:45-15:58 ET quote) against
   the decision on the official closes. A disagreement is a "flip".
2. **Fill gap**: official close minus the decision quote, in bps, on every BUY and SELL day.
3. **Reliability**: Mondays and exit days on which the script did not run ("MISSED").

**Duration.** 26 weeks from the first logged run, then a review. Paper P&L is reported but
is not a pass criterion.

**Pass (to be considered for TEST and then live):** flips on at most 10% of Mondays; mean
absolute fill gap at most 5 bps; MISSED at most 10% of required runs.

**Kill rule (the video's, made concrete):** stop and review if the cumulative paper drawdown
at full size goes beyond 22.8% (the TRAIN Monte Carlo 95th percentile), or after 10
consecutive losing trades (the TRAIN Monte Carlo 95th percentile streak).

TEST (SPY's final 20%) stays sealed during incubation and opens once afterwards for this
frozen cell.

## Amendment 7 (2026-10-29): filters, exits and sizing for the video EMA bot, fixed before running

The levers proposed after Amendment 3, applied to the same signal (EMA 5/8 cross up on a
closed bar, long only, market entry at the next open), same 12 symbols, same per-side costs
and the same 60% verdict windows. Script: `research/07_video_bot_levers.py`.

**Fixed choices.** Initial stop 2.0 x ATR(14) in every cell. This is chosen on cost grounds
(Amendment 2 measured 0.35R to 1.06R of cost per trade at 1 ATR on short bars), and it is
acknowledged that Amendment 3 had already shown 2 ATR cells doing better, so this choice is
not independent of earlier results. R is in units of the initial stop distance.

**Grid (24 cells)**
- Timeframe: 1d (full history), 1h (730 days). 15m is dropped: 60 days of data cannot
  validate anything.
- Filter, evaluated on the signal bar's close:
  - `none`;
  - `trend`: close above its 200-bar simple average;
  - `costfloor`: the round-trip cost is at most 5% of the stop distance, i.e.
    stop distance >= 40 x per-side cost x close;
  - `trend+costfloor`.
- Exit:
  - `target`: fixed target at 2.0R;
  - `trail`: chandelier stop, the highest close since entry (signal close included) minus
    the initial stop distance, ratcheting up only. The level for bar k uses closes up to
    bar k-1 only. No target;
  - `time`: close of the 10th bar of the trade, with the initial stop.

A gap beyond any stop fills at the open; a stop and a target in one bar resolve to the stop.

**Controls.** The random-entry control draws entries only from bars that PASS the same
filter, with the same exit. So it asks whether the EMA cross adds anything beyond the
filter, which is the only thing the bot can claim.

**Pass (all required, on the 60% window):**
- the day-bootstrap CI of net mean R lies entirely above 0;
- random-entry p < 0.05 / 24 = 0.0021;
- net mean R > 0 in at least 2 of 3 asset classes;
- no symbol carries more than 50% of positive R;
- net mean R is still > 0 at 2x costs.

**Then.**
- Only the single best passing cell by net mean R opens VALIDATE (the 60% to 80% slice),
  once. VALIDATE pass: net mean R > 0 and random-entry p < 0.05.
- Only a VALIDATE passer gets the portfolio and sizing layer, which is reported, not
  pass/fail: one position per symbol at a time, 0.5% of equity risked per trade on the
  stop distance, at most 6 open positions, and equity curve, CAGR and max drawdown.
- The final 20% stays sealed.
- If nothing passes, the EMA bot line of research is closed.

### Result of Amendment 7 (2026-10-29): no cell passes; the EMA bot line is closed

`results/20261029-205200_video_bot_levers.json`. The Amendment 3 reference cell reproduced
exactly (+0.425R), so the refactor did not change any earlier result.

- **1d:** every cell is positive net (+0.10R to +0.26R) and survives 2x costs, but every
  cell ties its random-entry control. Edge over random ranges from -0.048R to +0.048R; the
  best is `none|target`, p = 0.063, against the 0.0021 bar. The trend filter halves the
  trade count and leaves the edge over random at about zero (`trend|trail` +0.006R,
  p 0.44). So the filter, not the cross, carries the return: random entries above the
  200-bar average earn +0.245R with the same trailing exit.
- **1h:** net about zero everywhere (-0.04R to +0.06R). No cell has a CI above zero, and
  none beats random entry (best p 0.36).
- **Exits:** trailing raised crypto's share (up to +0.70R) but not the edge over random.
  The 10-bar time exit is the weakest on 1d.
- **Cost floor:** on 1d it removes almost nothing (2042 to 2029 trades). On 1h it removes
  74% of trades.
- **Design quirk, recorded honestly.** The 2x-cost run also doubles the cost floor's
  threshold, so on 1h cost-floor cells the 2x run trades a different, smaller set and can
  show higher R than 1x (`1h|costfloor|trail` +0.139R at 2x vs +0.018R at 1x). No verdict
  depends on it, because none of those cells passes the other checks.

**Conclusion.** Across Amendments 2, 3 and 7 (51 cells), the EMA 5/8 cross never beats
random entry with the same geometry. Every profitable cell is drift in assets that rose,
captured equally well by random long entries. The portfolio and sizing layer was not run,
because it was reserved for a VALIDATE passer and there is none. VALIDATE and TEST remain
unopened. Close this line; the project's live candidate remains `tt_moc` in incubation
(Amendment 6).

### Amendment 4 data note (2026-10-29, written before the first run; no rule changed)

Source: HistData.com NSXUSD (NASDAQ-100 CFD) generic ASCII 1-minute, 26 zips (2010-2025 by
year, 2026-01 to 2026-10 by month), 5,312,262 bars, 2010-11-14 to 2026-10-23. Prices sit on
the 0.25 grid. No volume. HistData documents its clock as fixed EST, but checked against
QQQ daily closes 2011-2026 and NQ=F 5-minute bars it is New York local time with daylight
saving; the loader reads it that way (median daily gap to QQQ about 2 bp). RTH days with
at least 300 of 390 bars: 31 in 2010, 245-251 a year 2011-2022, **140 in 2023** (mid-February
to July 2023 holds only about 210 RTH minutes a day, so those days drop out under the
pre-registered rule), 246 and 247 in 2024-2025, 202 in 2026. IN-SAMPLE therefore starts
2010-11-14 (plus 21 warm-up days for the context).

### Result of Amendment 4 (2026-10-29): no variant passes IN-SAMPLE; OUT-OF-SAMPLE stays unopened

`results/20261029-212131_video_nq_bot.json`, log `results/04_run.log`, research log entries 1-41.
HistData NSXUSD, 3,855 RTH days; IN-SAMPLE 2,498 days (2010-11-15 to 2020-12-31). 40 variants,
500 coin flips each, 1 MNQ.

- 26 of 40 variants are positive gross of costs, 13 after costs, **4 at 2x costs**. Costs are
  the main story: `orb5|none` makes +$8,227 gross and +$2,784 net, so costs take 66% of it
  (the video's own "74% of the profits were fake").
- **No variant has a day-bootstrap 95% CI of mean net P&L per trade above zero.** That gate
  alone fails all 40.
- Coin flip: no variant reaches the pre-registered p = 1/501. Closest: `orb5|ema` p = 0.008
  (+$3,046 net over 10 years, +$2.59 a trade, CI [-1.22, +6.65], 2x costs +$430),
  `orb5|none` 0.010, `gap_go|ema` 0.010 (328 trades, CI [-2.10, +13.41]), `ib_break|lovol` 0.016,
  `gap_go|lovol` 0.016. With 40 variants tried, about 2 would show p <= 0.05 by luck alone; 8 do.
  Random versions lose money on costs (median coin flip for `orb5|ema` -$1,250), so beating
  the coin flip here mostly means losing less to costs, and even that is not significant.
- Weakest: `pdhl_break` loses in every filter (coin flip better 74-98% of the time),
  `intraday_mom` and `rsi2_rev` lose net in every filter.

**Prop simulation, information only, NOT a pass** (`orb5|ema`, full history, generic 50K,
+$3,000 target, $2,000 trailing EOD drawdown, start every day): pass rate 64.9% at 1 MNQ but
a median 1,110 trading days to pass; 46.9% (357 days) at 2; 31.2% (100 days) at 3; 43.5%
(87 days) at 5; 37.3% (19 days) at 10. Random-entry versions pass 16-20% at every size.
Bigger size passes faster but not more often, which is the video's sizing lesson; with no
edge, size only trades waiting time for blow-up risk.

Verdict: on this data the 10 published strategies and the video's filter step do not
produce a tradable edge in MNQ after costs. The video's rule applies: none beats a coin flip
at the bar set before running, so none goes to OUT-OF-SAMPLE, the portfolio, the dealer or
live. The near misses (`orb5|ema`, `gap_go|ema`) are not re-tested with new tweaks: that
would be the overfitting the video warns about. A new test needs a new amendment, ideally on
real NQ futures data (Databento), with the trial count carried forward from this log.
