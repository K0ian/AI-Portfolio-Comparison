"""
main.py
-------
Orchestrates the full pipeline:

    fetch data  →  run strategies  →  backtest  →  metrics  →  charts

All outputs are written to the outputs/ directory.

Run with:
    python main.py
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")          # headless — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure project root is on sys.path so sub-packages resolve correctly
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.makedirs("outputs", exist_ok=True)

from data.fetch_data          import fetch_and_prepare
from models.momentum_strategy import run_momentum_strategy
from models.mvo_ml_strategy   import run_mvo_ml_strategy
from backtest.engine          import run_backtest, run_spy_benchmark
from analysis.metrics         import compute_metrics

# ---------------------------------------------------------------------------
# Period boundaries
# ---------------------------------------------------------------------------
TRAIN_START = "2016-01-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2025-12-31"

# Colour palette (consistent across all charts)
COLOURS = {
    "Momentum":      "#1976D2",   # blue
    "MVO+ML":        "#D32F2F",   # red
    "SPY (B&H)":     "#388E3C",   # green
}


# ---------------------------------------------------------------------------
# Charting helpers
# ---------------------------------------------------------------------------

def _style_ax(ax, title: str, ylabel: str):
    """Apply consistent style to a matplotlib Axes."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)


def plot_cumulative_returns(all_returns: dict[str, pd.Series]) -> None:
    """
    Two-panel figure:
      Top    — full period (2016-2025) with a vertical dashed line at train/test split
      Bottom — out-of-sample period only (2023-2025)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Cumulative Returns", fontsize=14, fontweight="bold")

    split_ts = pd.Timestamp(TEST_START)

    # --- Full period ---
    ax = axes[0]
    for name, ret in all_returns.items():
        cum = (1.0 + ret).cumprod()
        ax.plot(cum.index, cum, label=name, color=COLOURS.get(name), linewidth=1.6)
    ax.axvline(split_ts, color="grey", linestyle="--", linewidth=1, alpha=0.8, label="Train / Test split")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}×"))
    _style_ax(ax, "Full Period  (2016 – 2025)", "Portfolio value (×1)")
    ax.legend(fontsize=9, loc="upper left")

    # --- Out-of-sample period ---
    ax = axes[1]
    for name, ret in all_returns.items():
        oos = ret[ret.index >= TEST_START]
        if oos.empty:
            continue
        cum = (1.0 + oos).cumprod()
        ax.plot(cum.index, cum, label=name, color=COLOURS.get(name), linewidth=1.6)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}×"))
    _style_ax(ax, "Out-of-Sample Period  (2023 – 2025)", "Portfolio value (×1)")
    ax.legend(fontsize=9, loc="upper left")

    path = "outputs/cumulative_returns.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_rolling_sharpe(all_returns: dict[str, pd.Series], window: int = 60) -> None:
    """Rolling {window}-day Sharpe ratio for all strategies."""
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)

    for name, ret in all_returns.items():
        roll_sharpe = (
            ret.rolling(window).mean() / ret.rolling(window).std()
        ) * np.sqrt(252)
        ax.plot(roll_sharpe.index, roll_sharpe, label=name,
                color=COLOURS.get(name), linewidth=1.4)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="-")
    ax.axvline(pd.Timestamp(TEST_START), color="grey", linestyle="--",
               linewidth=1, alpha=0.8, label="Train / Test split")
    _style_ax(ax, f"{window}-Day Rolling Sharpe Ratio", "Sharpe Ratio")
    ax.legend(fontsize=9, loc="upper left")

    path = "outputs/rolling_sharpe.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_drawdowns(all_returns: dict[str, pd.Series]) -> None:
    """Underwater equity curve (drawdown) for all strategies."""
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)

    for name, ret in all_returns.items():
        cum = (1.0 + ret).cumprod()
        roll_max = cum.expanding().max()
        dd = (cum / roll_max) - 1.0

        ax.fill_between(dd.index, dd * 100, 0, alpha=0.18, color=COLOURS.get(name))
        ax.plot(dd.index, dd * 100, label=name, color=COLOURS.get(name), linewidth=1.4)

    ax.axvline(pd.Timestamp(TEST_START), color="grey", linestyle="--",
               linewidth=1, alpha=0.8, label="Train / Test split")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    _style_ax(ax, "Portfolio Drawdown", "Drawdown (%)")
    ax.legend(fontsize=9, loc="lower left")

    path = "outputs/drawdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

PERIODS = {
    "Train (2016–2022)": (TRAIN_START, TRAIN_END),
    "Test  (2023–2025)": (TEST_START,  TEST_END),
    "Full  (2016–2025)": (TRAIN_START, TEST_END),
}


def build_metrics_table(
    all_returns: dict[str, pd.Series],
    spy_returns: pd.Series,
) -> pd.DataFrame:
    """Compute metrics for every (strategy, period) combination."""
    rows = []

    for strategy_name, ret in all_returns.items():
        for period_label, (start, end) in PERIODS.items():
            mask_p = (ret.index >= start) & (ret.index <= end)
            mask_b = (spy_returns.index >= start) & (spy_returns.index <= end)

            p_ret = ret[mask_p]
            b_ret = spy_returns[mask_b]

            if p_ret.empty or b_ret.empty:
                continue

            m = compute_metrics(p_ret, b_ret)
            rows.append({
                "Strategy": strategy_name,
                "Period":   period_label,
                **m,
            })

    df = pd.DataFrame(rows).set_index(["Strategy", "Period"])
    return df


def _fmt_pct(v):
    return f"{v * 100:+.2f}%" if pd.notna(v) else "N/A"

def _fmt_ratio(v):
    return f"{v:.3f}" if pd.notna(v) else "N/A"


def print_metrics_table(df: pd.DataFrame) -> None:
    """Pretty-print the metrics table to stdout."""
    pct_cols   = ["Annualized Return", "Annualized Volatility",
                  "Maximum Drawdown", "Alpha (Annualized)", "Win Rate"]
    ratio_cols = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Beta"]

    fmt = df.copy().astype(object)
    for col in pct_cols:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(_fmt_pct)
    for col in ratio_cols:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(_fmt_ratio)

    col_order = [
        "Annualized Return", "Annualized Volatility", "Sharpe Ratio",
        "Sortino Ratio", "Maximum Drawdown", "Calmar Ratio",
        "Beta", "Alpha (Annualized)", "Win Rate",
    ]
    col_order = [c for c in col_order if c in fmt.columns]

    print()
    print(fmt[col_order].to_string())
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  AI PORTFOLIO COMPARISON  —  Backtesting System")
    print("=" * 65)

    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    print("\n[1/5]  Fetching & preparing data …")
    stock_prices, spy_prices, daily_returns, fundamentals = fetch_and_prepare(
        output_path="outputs/cleaned_data.csv"
    )

    # ------------------------------------------------------------------ #
    # 2. Strategies
    # ------------------------------------------------------------------ #
    print("\n[2/5]  Running strategies …")

    print("\n  — Model 1: Momentum (N=20, weekly rebalance, top-3) —")
    momentum_weights = run_momentum_strategy(
        stock_prices, n_days=20, rebalance_freq=5, top_n=3
    )

    print("\n  — Model 2: MVO + Ridge Regression (monthly rebalance) —")
    mvo_weights = run_mvo_ml_strategy(
        stock_prices, fundamentals,
        rebalance_freq=21, forward_days=20,
        ridge_alpha=1.0, min_train_obs=30, max_weight=0.25, cov_window=60,
    )

    # ------------------------------------------------------------------ #
    # 3. Backtests
    # ------------------------------------------------------------------ #
    print("\n[3/5]  Backtesting …")

    mom_returns = run_backtest(momentum_weights, stock_prices)
    mom_returns.name = "Momentum"
    print(f"  Momentum : {len(mom_returns)} daily returns  "
          f"({mom_returns.index[0].date()} → {mom_returns.index[-1].date()})")

    if not mvo_weights.empty:
        mvo_returns = run_backtest(mvo_weights, stock_prices)
        mvo_returns.name = "MVO+ML"
        print(f"  MVO+ML   : {len(mvo_returns)} daily returns  "
              f"({mvo_returns.index[0].date()} → {mvo_returns.index[-1].date()})")
    else:
        mvo_returns = pd.Series(dtype=float, name="MVO+ML")
        print("  MVO+ML   : [no weights generated — skipping]")

    spy_returns = run_spy_benchmark(spy_prices)
    spy_returns.name = "SPY (B&H)"
    print(f"  SPY B&H  : {len(spy_returns)} daily returns  "
          f"({spy_returns.index[0].date()} → {spy_returns.index[-1].date()})")

    # Collect strategies that have data
    all_returns: dict[str, pd.Series] = {}
    for s in [mom_returns, mvo_returns, spy_returns]:
        if not s.empty:
            all_returns[s.name] = s

    # ------------------------------------------------------------------ #
    # 4. Metrics
    # ------------------------------------------------------------------ #
    print("\n[4/5]  Computing metrics …")
    metrics_df = build_metrics_table(all_returns, spy_returns)

    path_csv = "outputs/metrics_table.csv"
    metrics_df.to_csv(path_csv)
    print(f"  Saved {path_csv}")

    print("\n" + "=" * 65)
    print("  PERFORMANCE METRICS")
    print("=" * 65)
    print_metrics_table(metrics_df)

    # ------------------------------------------------------------------ #
    # 5. Charts
    # ------------------------------------------------------------------ #
    print("[5/5]  Generating charts …")
    plot_cumulative_returns(all_returns)
    plot_rolling_sharpe(all_returns)
    plot_drawdowns(all_returns)

    print()
    print("=" * 65)
    print("  DONE  —  all outputs written to  outputs/")
    print("=" * 65)


if __name__ == "__main__":
    main()
