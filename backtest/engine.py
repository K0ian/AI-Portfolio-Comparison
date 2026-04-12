"""
engine.py
---------
Generic, look-ahead-free backtesting engine.

How it avoids look-ahead bias
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Strategy modules produce weights on rebalancing dates — these are based on
information available *at the close of that day*.  To prevent the portfolio
from "trading at the close it observed", weights are shifted forward by one
calendar row using pandas .shift(1).  This means weights set at day T are
first applied to returns on day T+1, simulating execution at the next day's
open (or close — the difference is negligible for daily backtests).

The SPY benchmark is handled the same way: buy at day-0 close,
collect returns from day 1 onward.

Usage
-----
    from backtest.engine import run_backtest, run_spy_benchmark

    portfolio_returns = run_backtest(weights_df, stock_prices)
    spy_returns       = run_spy_benchmark(spy_prices)
"""

import numpy as np
import pandas as pd


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.Series:
    """
    Simulate a strategy defined by a weights DataFrame.

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights on rebalancing dates (date × ticker).
        Non-rebalancing rows need NOT be present — they are forward-filled.
        Values should sum to 1 (long-only, fully invested).

    prices : pd.DataFrame
        Daily adjusted-close prices (date × ticker).

    Returns
    -------
    pd.Series
        Daily portfolio returns with a DatetimeIndex.
        First valid return corresponds to the day *after* the first
        rebalancing date (due to the one-day execution lag).
    """
    # --- Daily returns from prices ---
    daily_ret = prices.pct_change()

    # Restrict to tickers that appear in both weights and prices
    common_tickers = weights.columns.intersection(prices.columns)
    weights    = weights[common_tickers]
    daily_ret  = daily_ret[common_tickers]

    # --- Reindex weights to cover every date in the returns index ---
    # Dates not in weights get NaN (filled forward below)
    weights_full = weights.reindex(daily_ret.index)

    # Forward-fill: hold last weight until next rebalance
    weights_full = weights_full.ffill()

    # --- One-day execution lag (core look-ahead prevention) ---
    # Weights observed at close of day T are applied to returns on day T+1.
    # .shift(1) moves every row down by one position in the date index.
    weights_lagged = weights_full.shift(1)

    # --- Compute daily portfolio return ---
    # P&L on day t = Σᵢ  w_{i, t-1} × r_{i, t}
    port_ret = (weights_lagged * daily_ret).sum(axis=1)

    # Drop leading rows where weights were not yet available (NaN period)
    port_ret = port_ret[weights_lagged.notna().any(axis=1)]
    port_ret = port_ret.dropna()

    return port_ret


def run_spy_benchmark(spy_prices: pd.DataFrame) -> pd.Series:
    """
    Buy-and-hold SPY benchmark.

    Applies the same one-day lag as run_backtest for a fair comparison:
    the first day's return is skipped (invest at close of day 0,
    collect returns from day 1 onward).

    Parameters
    ----------
    spy_prices : pd.DataFrame with a 'SPY' column (or single-column DataFrame)

    Returns
    -------
    pd.Series of daily SPY returns
    """
    col = "SPY" if "SPY" in spy_prices.columns else spy_prices.columns[0]
    spy_ret = spy_prices[col].pct_change().dropna()
    return spy_ret
