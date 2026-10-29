# Quant: equities and crypto research harness

A tested harness that decides whether an equity, ETF or crypto strategy has an edge before
any money is risked. It is a successor to `ICT_SMC_Trading_System`, whose search found no
edge in FX and metals price bars. It keeps the same discipline and applies it to new
markets.

Read `CLAUDE.md` for the rules and the current status, and `docs/00_RESEARCH_PLAN.md` for
every pre-registered test and its result. No number in this repo is meant to be trusted
without the plan entry and the results file behind it.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

## Main commands

```
python -m quant.cli download --universe equities      # or crypto
python -m quant.cli evaluate --universe equities --split train
PYTHONPATH=src python research/05_sqx_video.py         # ETF swing strategies, Amendment 5
PYTHONPATH=src python research/06_incubate_tt_moc.py   # paper signals for tt_moc, run ~15:50 ET
PYTHONPATH=src python research/07_video_bot_levers.py  # EMA bot filters and exits, Amendment 7
```

## Layout

| path | what |
|---|---|
| `src/quant/data.py` | yfinance download and parquet cache, liquidity mask |
| `src/quant/strategies.py`, `backtest.py`, `evaluate.py` | daily portfolio strategies, backtest, acceptance protocol |
| `src/quant/intraday.py` | trade-level engine for the YouTube EMA bot (targets, trailing and time exits, filters) |
| `src/quant/swing.py`, `swing_strategies.py` | single-ETF swing engine and SQX-style robustness tests |
| `src/quant/nq.py`, `nq_strategies.py`, `dealer.py` | NASDAQ 1-minute pipeline and paper trader (blocked on data) |
| `src/quant/stats.py`, `protocol.py` | statistics that charge for the search, and the sealed train/validate/test split |
| `research/` | one script per plan amendment |
| `results/` | JSON output of every run, timestamped |
| `state/` | test-set look log, research log, incubation log |
| `tests/` | look-ahead (prefix invariance), cost, control and protocol tests |

## What the tests guarantee

- Signals and filters are look-ahead free: computing them on data cut at time T gives the
  same values as computing on full history and cutting afterwards.
- Same-day information earns nothing.
- Costs scale with turnover and are charged on both sides.
- Stops that are gapped through fill at the open.
- A planted signal is found.
- The TEST split refuses to open without a reason and logs every look.

## Known limits

- The equity universe uses today's S&P 500 members, so it is survivorship biased and
  absolute returns are upper bounds. Strategies are judged against buy-and-hold and random
  selection of the same universe instead.
- Free intraday data is short: 60 days of 15-minute bars and 730 days of hourly bars.
- No exchange API is reachable from this machine. Only yfinance works.
