# Quant equities and crypto research

Equities and crypto strategy research. **Separate from the forex project** in
`Documents\ICT_SMC_Trading_System`: no code, data or notes are shared between the two
folders. Read this file and `docs/00_RESEARCH_PLAN.md` at the start of every session.

## Hard rules (carried over from the forex project, where they were learned)

1. **Never fabricate performance.** A number not produced by a run in this repo does not
   get stated.
2. **Never use em-dashes** in any output: code comments, docs, reports, chat.
3. **No look-ahead.** Every strategy and every universe filter has a prefix-invariance
   test in `tests/test_harness.py`.
4. **Pre-register before running.** Strategies live in `FAMILIES`
   (`src/quant/strategies.py`); anything new is written into the plan as an amendment
   BEFORE it runs, including what counts as passing.
5. **TEST is sealed.** `protocol.window(..., "test")` refuses without an unseal reason
   and logs every look to `state/test_set_looks.json`.
6. **Judge against the right null.** Long-only strategies are compared with equal-weight
   buy-and-hold of the same universe and with random selection, never with zero.

## Where things stand (2026-10-26)

Free yfinance data, current S&P 500 members (survivorship-biased) and a fixed top-10
crypto list. Nothing has passed.

| strategy | TRAIN result |
|---|---|
| 12-1 momentum | does not beat random selection (p = 0.86) |
| 6-1 momentum | p = 0.06 against random, fails |
| 12-1 momentum long-short | loses money |
| 5-day reversal | beats random and benchmark at 1x costs and in the top-100 liquid universe; **fails at 2x costs** (p = 0.074). Dead on free data |
| 200-day trend filter | halves drawdown (26% vs 51%), never beats buy-and-hold |
| crypto, 6 variants | nothing beats the equal-weight basket |

VALIDATE and TEST are unopened for both universes.

**Amendment 5 (2026-10-29, SQX video, ETF daily OHLCV in `data/swing/`, survivorship-free).**
One survivor: `tt_moc` (Turnaround Tuesday on SPY, buy Monday's close after a down Monday,
sell Tuesday's close) passed all 9 TRAIN gates and VALIDATE (p 0.042, +0.127% per trade
net). It is in incubation (`research/06_incubate_tt_moc.py`, paper only); its TEST slice
(2020-2026) is still sealed. `es_rsi2` and `ensemble_spy` passed TRAIN and failed VALIDATE.
Gold Thursday fails the multiple-testing bar (p 0.020). The S&P 100 ranked pullback does
not beat its basket. Details in the plan.

**NASDAQ 1-minute day-trading pipeline (Amendment 4, 2026-10-29): built, NOT yet run.**
10 published strategies x 4 filters = 40 variants, MNQ costs, IN-SAMPLE to 2020 /
OUT-OF-SAMPLE 2021 on, 500-run coin flip, append-only `state/research_log.jsonl`,
portfolio, prop-firm sizing, paper trader `quant.dealer` (no orders). Blocked on 1-minute
data: Dukascopy's free datafeed answers 429 and its docs now point to a requester-pays AWS
tick bucket. Needs a Databento NQ CSV or HistData NSXUSD files from the user.

## Next decision

Free data has given its answer. The remaining options, in order:

1. **Survivorship-free data** (Norgate, about $270 to $788 a year by third-party
   quotes, 21-day free trial; or Sharadar). Re-register reversal and add low-volatility
   as NEW tests with costs fixed in advance. Free-data results do not carry over.
2. **Accept buy-and-hold** (optionally with the trend filter for drawdown) as the
   honest result.

Broker for any live phase: Interactive Brokers Pro (API needs Pro; directed API
orders bill at the fixed $0.005/share rate). Real shares, not CFDs: the forex project
measured a swap markup that taxes every multi-day CFD hold. **Confirm IBKR accepts
the user's country of residence before Phase 3.**

## Commands

```
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m quant.cli evaluate --universe equities --split train
PYTHONPATH=src .venv/Scripts/python.exe research/01_near_miss_stress.py   # ~50 min
PYTHONPATH=src .venv/Scripts/python.exe research/05_sqx_video.py          # ~20 min, Amendment 5
PYTHONPATH=src .venv/Scripts/python.exe research/06_incubate_tt_moc.py    # paper signals; scheduled via scripts/incubate_tt_moc.bat
.venv/Scripts/python.exe research/06_incubate_tt_moc.py --report            # incubation summary (Amendment 6)
PYTHONPATH=src .venv/Scripts/python.exe research/04_video_nq_bot.py data/nq_1m/nq_1m.parquet "source"
PYTHONPATH=src .venv/Scripts/python.exe -m quant.dealer --variants "orb30|none" --once
```

The random-selection control dominates runtime (200 sims; slow on 5-day rebalances).
