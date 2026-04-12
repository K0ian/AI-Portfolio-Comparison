"""
fetch_data.py
-------------
Downloads daily adjusted-close prices and volume for S&P 500 Energy sector
constituents plus SPY (benchmark).  Also pulls a snapshot of key fundamentals
via yfinance .info for use as static features in Model 2.

Design choices:
- auto_adjust=True  → yfinance returns split- and dividend-adjusted 'Close'
- Forward-fill gaps ≤ 3 trading days (e.g. data-vendor holiday quirks)
- Columns with > 5 % missing data are dropped so later models don't silently
  train on stale forward-fills
- Fundamentals are a point-in-time snapshot (today's values held constant).
  This introduces a mild look-ahead for the PE feature, which is flagged as a
  limitation in the model comments.
"""

import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
ENERGY_TICKERS = [
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO",
    "PXD", "OXY", "HAL", "DVN", "BKR", "FANG", "HES", "MRO",
]
BENCHMARK = "SPY"
START_DATE = "2016-01-01"
END_DATE   = "2025-12-31"

FUNDAMENTAL_FIELDS = ["trailingPE", "earningsGrowth", "revenueGrowth", "returnOnEquity"]


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def _download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Return a (date × ticker) DataFrame of adjusted close prices."""
    all_tickers = tickers + [BENCHMARK]
    raw = yf.download(
        all_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns when >1 ticker is requested
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        # Single-ticker edge case — shouldn't happen here but defensive
        prices = raw[["Close"]].copy()
        prices.columns = all_tickers

    return prices


def _clean_prices(prices: pd.DataFrame, max_missing_pct: float = 0.05) -> pd.DataFrame:
    """
    1. Drop tickers with > max_missing_pct NaN rows.
    2. Forward-fill remaining small gaps (max 3 consecutive days).
    3. Drop any leading rows that still contain NaN across the universe.
    """
    # Drop columns that are mostly missing
    min_valid = int((1 - max_missing_pct) * len(prices))
    prices = prices.dropna(axis=1, thresh=min_valid)

    # Forward-fill small gaps caused by exchange holidays / data-vendor issues
    prices = prices.ffill(limit=3)

    # Drop the initial rows that still have NaN (before a stock had listings)
    prices = prices.dropna(how="all")

    return prices


# ---------------------------------------------------------------------------
# Fundamental data (static snapshot)
# ---------------------------------------------------------------------------

def _fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Pull a point-in-time snapshot of key fundamentals using yfinance .info.

    NOTE: These values are *today's* values, not historical time-series.
    They are treated as constant features for every training sample.
    Using today's PE in past training windows introduces a mild look-ahead
    bias for the PE feature specifically; this is documented as a limitation.
    """
    records = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            records[ticker] = {field: info.get(field, np.nan) for field in FUNDAMENTAL_FIELDS}
        except Exception as exc:
            print(f"  [warn] Could not fetch fundamentals for {ticker}: {exc}")
            records[ticker] = {field: np.nan for field in FUNDAMENTAL_FIELDS}

    df = pd.DataFrame(records).T
    df.index.name = "ticker"
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_and_prepare(
    output_path: str = "outputs/cleaned_data.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates data fetching and cleaning.

    Returns
    -------
    stock_prices  : pd.DataFrame  — daily adj-close for energy stocks (date × ticker)
    spy_prices    : pd.DataFrame  — daily adj-close for SPY (date × 'SPY')
    daily_returns : pd.DataFrame  — daily pct-change returns for all assets
    fundamentals  : pd.DataFrame  — static fundamental snapshot (ticker × field)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Prices ---
    print("  Downloading price data via yfinance …")
    all_prices = _download_prices(ENERGY_TICKERS, START_DATE, END_DATE)
    all_prices = _clean_prices(all_prices)

    # Separate benchmark from universe
    spy_prices    = all_prices[["SPY"]].copy()
    available     = [t for t in ENERGY_TICKERS if t in all_prices.columns]
    stock_prices  = all_prices[available].copy()

    dropped = set(ENERGY_TICKERS) - set(available)
    if dropped:
        print(f"  [warn] Tickers dropped due to missing data: {dropped}")

    # --- Daily returns (used downstream) ---
    daily_returns = all_prices.pct_change()
    # Drop the first row (NaN from pct_change) and rows entirely NaN
    daily_returns = daily_returns.iloc[1:].dropna(how="all")

    # --- Fundamentals ---
    print("  Fetching fundamental snapshots …")
    fundamentals = _fetch_fundamentals(available)

    # --- Save first 50 rows of returns ---
    daily_returns.head(50).to_csv(output_path)
    print(f"  Saved first 50 rows of cleaned returns → {output_path}")

    print(
        f"  Universe: {len(available)} energy stocks + SPY  |  "
        f"{len(stock_prices)} trading days  |  "
        f"{stock_prices.index[0].date()} → {stock_prices.index[-1].date()}"
    )

    return stock_prices, spy_prices, daily_returns, fundamentals
