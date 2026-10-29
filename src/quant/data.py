"""Price data: download once, cache to parquet, load from cache.

Known limitation, stated here so it cannot be forgotten: both universes are built
from TODAY's membership (current S&P 500, current top-10 crypto). Names that were
delisted, acquired or collapsed are absent, which inflates the absolute return of
any strategy that buys winners. Every strategy is therefore judged against an
equal-weight buy-and-hold of the SAME universe, which shares the bias, and
against a random-selection control on the SAME universe. Absolute numbers from
this data are upper bounds.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

# Spot pairs on yfinance. Stablecoins excluded. Fixed list, chosen before any results.
CRYPTO_TOP10 = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD",
    "DOGE-USD", "ADA-USD", "TRX-USD", "LINK-USD", "AVAX-USD",
]

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500_tickers() -> list[str]:
    cache = DATA_DIR / "universe_sp500.csv"
    if cache.exists():
        return pd.read_csv(cache)["ticker"].tolist()
    resp = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0 (research)"}, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text))[0]
    tickers = sorted(t.replace(".", "-") for t in table["Symbol"].astype(str))
    DATA_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(cache, index=False)
    return tickers


def universe(kind: str) -> list[str]:
    if kind == "crypto":
        return list(CRYPTO_TOP10)
    if kind == "equities":
        return sp500_tickers()
    raise ValueError(f"unknown universe {kind!r}")


def download(kind: str, start: str = "2004-01-01") -> pd.DataFrame:
    import yfinance as yf

    tickers = universe(kind)
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)
    close = raw["Close"].copy()
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index().dropna(how="all")
    # Names with almost no history add nothing and break rank-based selection.
    close = close.loc[:, close.notna().sum() >= 250]
    DATA_DIR.mkdir(exist_ok=True)
    close.to_parquet(DATA_DIR / f"prices_{kind}.parquet")
    return close


def load_prices(kind: str) -> pd.DataFrame:
    path = DATA_DIR / f"prices_{kind}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m quant.cli download --universe {kind}")
    return pd.read_parquet(path)


def download_volume(kind: str) -> pd.DataFrame:
    """Share volume for exactly the cached price panel's names and dates, so the
    prices that results were computed on are never silently replaced."""
    import yfinance as yf

    px = load_prices(kind)
    raw = yf.download(list(px.columns), start=str(px.index[0].date()), auto_adjust=True,
                      progress=False, threads=True)
    vol = raw["Volume"].copy()
    vol.index = pd.to_datetime(vol.index).tz_localize(None)
    vol = vol.reindex(index=px.index, columns=px.columns)
    vol.to_parquet(DATA_DIR / f"volume_{kind}.parquet")
    return vol


def liquid_mask(prices: pd.DataFrame, volume: pd.DataFrame, top_n: int = 100,
                window: int = 63) -> pd.DataFrame:
    """True where a name is among the `top_n` by trailing median dollar volume.

    Row t uses days t-window+1..t only, so it is known at the close of t, the same
    instant strategy weights are decided."""
    dollar_vol = (prices * volume).rolling(window, min_periods=window).median()
    return dollar_vol.rank(axis=1, ascending=False) <= top_n


def load_volume(kind: str) -> pd.DataFrame:
    path = DATA_DIR / f"volume_{kind}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Call quant.data.download_volume({kind!r})")
    return pd.read_parquet(path)


PERIODS_PER_YEAR = {"equities": 252, "crypto": 365}
