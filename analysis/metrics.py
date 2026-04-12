"""
metrics.py
----------
Computes a comprehensive set of risk-adjusted performance metrics for a
daily-returns series.

Risk-free rate
~~~~~~~~~~~~~~
rf = 4.5 % annualised — approximate average Fed Funds effective rate over the
2023–2024 period (consistent with the test-period evaluation window).
The daily rf is derived from the annual compounded rate:
    rf_daily = (1 + rf_annual)^(1/252) - 1

Metrics returned
~~~~~~~~~~~~~~~~
Annualised Return     — geometric (CAGR)
Annualised Volatility — std of daily returns × √252
Sharpe Ratio          — (CAGR - rf) / ann_vol
Sortino Ratio         — (CAGR - rf) / downside_std; downside = days < rf_daily
Maximum Drawdown      — peak-to-trough drawdown on the cumulative-return curve
Calmar Ratio          — CAGR / |Max Drawdown|
Beta                  — OLS β to SPY over the evaluation period
Alpha (annualised)    — Jensen's α: CAGR - [rf + β × (SPY_CAGR - rf)]
Win Rate              — fraction of trading days with positive return
"""

import numpy as np
import pandas as pd


def compute_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: float = 0.045,
) -> dict:
    """
    Compute performance metrics for a returns series.

    Parameters
    ----------
    returns           : daily portfolio returns (pd.Series with DatetimeIndex)
    benchmark_returns : daily SPY returns aligned to the same period
    rf                : annualised risk-free rate (default 4.5 %)

    Returns
    -------
    dict  — keys are metric names, values are floats
    """
    TRADING_DAYS = 252

    # --- Align series to the intersection of their indices ---
    common_idx = returns.index.intersection(benchmark_returns.index)
    ret  = returns[common_idx].dropna()
    bench = benchmark_returns[common_idx].dropna()

    if len(ret) < 2:
        return {}

    # Daily risk-free rate derived from annualised rate
    rf_daily = (1.0 + rf) ** (1.0 / TRADING_DAYS) - 1.0

    # ------------------------------------------------------------------ #
    # 1. Annualised return (CAGR)
    # ------------------------------------------------------------------ #
    n_years    = len(ret) / TRADING_DAYS
    total_ret  = (1.0 + ret).prod()            # gross cumulative return
    ann_return = total_ret ** (1.0 / n_years) - 1.0

    # ------------------------------------------------------------------ #
    # 2. Annualised volatility
    # ------------------------------------------------------------------ #
    ann_vol = ret.std() * np.sqrt(TRADING_DAYS)

    # ------------------------------------------------------------------ #
    # 3. Sharpe ratio
    # ------------------------------------------------------------------ #
    # Use daily excess returns over the daily rf
    excess_daily = ret - rf_daily
    sharpe = (excess_daily.mean() / ret.std()) * np.sqrt(TRADING_DAYS) if ret.std() > 0 else np.nan

    # ------------------------------------------------------------------ #
    # 4. Sortino ratio
    # ------------------------------------------------------------------ #
    # Downside deviation: std of returns that fall below the daily rf
    downside = ret[ret < rf_daily]
    if len(downside) > 1:
        downside_std = downside.std() * np.sqrt(TRADING_DAYS)
        sortino = (ann_return - rf) / downside_std if downside_std > 0 else np.nan
    else:
        sortino = np.nan

    # ------------------------------------------------------------------ #
    # 5. Maximum drawdown
    # ------------------------------------------------------------------ #
    cumulative  = (1.0 + ret).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown    = (cumulative / rolling_max) - 1.0
    max_dd      = float(drawdown.min())        # negative number

    # ------------------------------------------------------------------ #
    # 6. Calmar ratio
    # ------------------------------------------------------------------ #
    calmar = ann_return / abs(max_dd) if max_dd != 0.0 else np.nan

    # ------------------------------------------------------------------ #
    # 7. Beta to SPY
    # ------------------------------------------------------------------ #
    # β = Cov(r_p, r_b) / Var(r_b)
    # Use numpy covariance on the aligned pair
    cov_matrix  = np.cov(ret.values, bench.values)
    bench_var   = cov_matrix[1, 1]
    beta        = cov_matrix[0, 1] / bench_var if bench_var > 0 else np.nan

    # ------------------------------------------------------------------ #
    # 8. Jensen's alpha (annualised)
    # ------------------------------------------------------------------ #
    # α = CAGR_p - [rf + β × (CAGR_bench - rf)]
    bench_n_years  = len(bench) / TRADING_DAYS
    bench_ann_ret  = (1.0 + bench).prod() ** (1.0 / bench_n_years) - 1.0
    alpha          = ann_return - (rf + beta * (bench_ann_ret - rf)) if not np.isnan(beta) else np.nan

    # ------------------------------------------------------------------ #
    # 9. Win rate
    # ------------------------------------------------------------------ #
    win_rate = float((ret > 0.0).mean())

    return {
        "Annualized Return":      ann_return,
        "Annualized Volatility":  ann_vol,
        "Sharpe Ratio":           sharpe,
        "Sortino Ratio":          sortino,
        "Maximum Drawdown":       max_dd,
        "Calmar Ratio":           calmar,
        "Beta":                   beta,
        "Alpha (Annualized)":     alpha,
        "Win Rate":               win_rate,
    }
