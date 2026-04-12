"""
momentum_strategy.py
--------------------
Cross-sectional momentum strategy over the S&P 500 Energy sector universe.

Algorithm
~~~~~~~~~
At every rebalancing date (every `rebalance_freq` trading days):
  1. Compute the N-day trailing return for each stock in the universe.
  2. Rank all stocks by that return (high = strong momentum).
  3. Go long the top `top_n` stocks with equal weights (1/top_n each).

No look-ahead bias:  the signal at date T uses only prices up to and including
date T.  The engine shifts these weights forward by one day before applying
them to returns, simulating next-open execution.

Long-only, fully invested.  No transaction costs (noted as limitation).

Parameters tested:  N = 10, 20, 60 days.  Primary: N = 20.
Top quintile of 16 stocks ≈ 3 stocks  → top_n = 3.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def compute_momentum_weights(
    stock_prices: pd.DataFrame,
    n_days: int = 20,
    rebalance_freq: int = 5,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Build a (rebalance_date × ticker) DataFrame of portfolio weights.

    Only rebalancing dates are populated; the backtesting engine forward-fills
    weights between them.

    Parameters
    ----------
    stock_prices   : daily adjusted-close prices (date × ticker)
    n_days         : momentum lookback window in trading days
    rebalance_freq : how often to rebalance, in trading days (default 5 = weekly)
    top_n          : number of stocks to hold in the long portfolio

    Returns
    -------
    pd.DataFrame with shape (n_rebalance_dates × n_tickers)
    """
    tickers    = stock_prices.columns.tolist()
    all_dates  = stock_prices.index

    # Need at least n_days of history before the first rebalance
    start_idx   = n_days
    rebal_idxs  = range(start_idx, len(all_dates), rebalance_freq)

    rows  = []
    dates = []

    for idx in rebal_idxs:
        date = all_dates[idx]

        # --- Momentum signal: (price_today / price_n_days_ago) - 1 ---
        # prices at position idx vs idx-n_days (pure historical, no look-ahead)
        price_now  = stock_prices.iloc[idx]
        price_past = stock_prices.iloc[idx - n_days]

        signal = (price_now / price_past) - 1.0

        # Drop stocks with missing prices in the window
        signal = signal.dropna()

        if len(signal) < top_n:
            # Not enough valid stocks to form a portfolio — skip this date
            continue

        # --- Rank: highest momentum = highest rank ---
        # nlargest gives the top_n tickers by momentum score
        top_tickers = signal.nlargest(top_n).index

        # Equal-weight the long portfolio
        w = pd.Series(0.0, index=tickers)
        w[top_tickers] = 1.0 / top_n

        rows.append(w)
        dates.append(date)

    weights = pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
    return weights


def run_momentum_strategy(
    stock_prices: pd.DataFrame,
    n_days: int = 20,
    rebalance_freq: int = 5,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Convenience wrapper that logs progress and returns the weights DataFrame.

    To compare lookback windows call compute_momentum_weights directly
    with n_days = 10, 20, or 60.
    """
    print(
        f"  Momentum Strategy  |  lookback={n_days}d  "
        f"rebalance=every {rebalance_freq}d  top_n={top_n}"
    )
    weights = compute_momentum_weights(stock_prices, n_days, rebalance_freq, top_n)
    print(f"  → {len(weights)} rebalancing events generated")
    return weights
