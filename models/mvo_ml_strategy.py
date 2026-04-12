"""
mvo_ml_strategy.py
------------------
Mean-Variance Optimisation + Machine Learning strategy.

Pipeline
~~~~~~~~
Monthly rebalance (every 21 trading days).

At each rebalancing date T:
  1. FEATURES (computed using only data ≤ T — no look-ahead):
       • 20-day momentum
       • 60-day momentum
       • 20-day realised volatility (annualised)
       • 5-day price reversal  (negative of 5-day return — mean-reversion)
       • Trailing P/E ratio    (static snapshot; see look-ahead note below)

  2. TARGET (20-day forward return from date s to s+20):
       Known only at s+20.  Training at T only uses (feature, target) pairs
       where  s + 20 < T  — i.e. every target is fully realised before T.
       This is the expanding-window discipline that prevents look-ahead bias.

  3. MODEL:  Ridge Regression (sklearn) retrained from scratch at every T
       on the full expanding training set.  Features are z-scored with the
       scaler fit on the training set only.

  4. MVO:  scipy.optimize.minimize (SLSQP) to maximise the portfolio Sharpe
       ratio given:
         • Expected returns  = Ridge predictions
         • Covariance matrix = 60-day rolling realised covariance (annualised)
       Constraints: long-only, Σwᵢ = 1, max single weight = 0.25.

Look-ahead note on PE ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~
yfinance .info gives a *current* snapshot, not a time series.  Using today's
PE as a training feature for observations 5–9 years in the past technically
introduces a small look-ahead bias for that one feature.  Documented as a
limitation; all other features are constructed without look-ahead.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FEATURE_COLS = ["mom_20", "mom_60", "vol_20", "reversal_5", "trailingPE"]


# ---------------------------------------------------------------------------
# Feature & target pre-computation
# ---------------------------------------------------------------------------

def _precompute_features(
    stock_prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    rebalance_freq: int = 21,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Compute the 5-factor feature matrix for every stock at every potential
    rebalancing date.  Results stored in a dict keyed by date for O(1) lookup.

    All computations use only historical (past) prices → zero look-ahead.
    """
    all_dates     = stock_prices.index
    tickers       = stock_prices.columns.tolist()
    daily_returns = stock_prices.pct_change()

    features_by_date: dict[pd.Timestamp, pd.DataFrame] = {}

    # We need 60 days of history for the longest lookback; start there
    for idx in range(60, len(all_dates), rebalance_freq):
        date = all_dates[idx]
        rows = {}

        for ticker in tickers:
            p = stock_prices[ticker]

            # Skip if any price is missing in the 60-day window
            if p.iloc[idx - 60 : idx + 1].isna().any():
                continue

            # 20-day momentum  (price return over past 20 trading days)
            mom_20 = (p.iloc[idx] / p.iloc[idx - 20]) - 1.0

            # 60-day momentum
            mom_60 = (p.iloc[idx] / p.iloc[idx - 60]) - 1.0

            # 20-day realised volatility (std of daily returns, annualised)
            # Use returns at positions [idx-19 … idx] → 20 observations
            ret_slice = daily_returns[ticker].iloc[idx - 19 : idx + 1]
            vol_20 = float(ret_slice.std() * np.sqrt(252))

            # 5-day price reversal: -(5-day return) — captures mean reversion
            rev_5 = -((p.iloc[idx] / p.iloc[idx - 5]) - 1.0)

            # P/E ratio from static fundamental snapshot
            pe = (
                float(fundamentals.loc[ticker, "trailingPE"])
                if ticker in fundamentals.index
                else np.nan
            )

            rows[ticker] = {
                "mom_20":     mom_20,
                "mom_60":     mom_60,
                "vol_20":     vol_20,
                "reversal_5": rev_5,
                "trailingPE": pe,
            }

        if len(rows) >= 3:
            features_by_date[date] = pd.DataFrame(rows).T  # shape: ticker × feature
            features_by_date[date].columns = FEATURE_COLS

    return features_by_date


def _precompute_targets(
    stock_prices: pd.DataFrame,
    rebalance_freq: int = 21,
    forward_days: int = 20,
) -> dict[pd.Timestamp, pd.Series]:
    """
    For each potential feature date s, compute the 20-day forward return.

    target[s] = (price[s + forward_days] / price[s]) - 1

    Only stored when price[s + forward_days] actually exists.
    """
    all_dates = stock_prices.index
    tickers   = stock_prices.columns.tolist()

    targets_by_date: dict[pd.Timestamp, pd.Series] = {}

    for idx in range(60, len(all_dates), rebalance_freq):
        future_idx = idx + forward_days
        if future_idx >= len(all_dates):
            break  # no future price available — stop

        date = all_dates[idx]
        p_now    = stock_prices.iloc[idx]
        p_future = stock_prices.iloc[future_idx]

        fwd = {}
        for ticker in tickers:
            if pd.notna(p_now[ticker]) and pd.notna(p_future[ticker]) and p_now[ticker] > 0:
                fwd[ticker] = (p_future[ticker] / p_now[ticker]) - 1.0

        if fwd:
            targets_by_date[date] = pd.Series(fwd)

    return targets_by_date


# ---------------------------------------------------------------------------
# MVO optimizer
# ---------------------------------------------------------------------------

def _mvo_optimize(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    max_weight: float = 0.25,
) -> pd.Series:
    """
    Maximise portfolio Sharpe ratio subject to:
      • long-only  (wᵢ ≥ 0)
      • fully invested  (Σwᵢ = 1)
      • position cap  (wᵢ ≤ max_weight)

    Objective: minimise  −(μ_p / σ_p)
    where  μ_p = wᵀ μ  and  σ_p = √(wᵀ Σ w)

    Falls back to equal-weight if the optimiser fails to converge.
    """
    n       = len(expected_returns)
    mu      = expected_returns.values
    sigma   = cov_matrix.values
    tickers = expected_returns.index

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = w @ mu
        port_vol = np.sqrt(w @ sigma @ w)
        if port_vol < 1e-12:
            return 0.0
        return -(port_ret / port_vol)

    # Gradient of the objective (helps SLSQP converge faster)
    def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
        port_ret = w @ mu
        port_vol = np.sqrt(w @ sigma @ w)
        if port_vol < 1e-12:
            return np.zeros(n)
        d_ret = mu
        d_vol = (sigma @ w) / port_vol
        return -(d_ret * port_vol - port_ret * d_vol) / (port_vol ** 2)

    w0          = np.ones(n) / n                             # warm-start: equal weight
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds      = [(0.0, max_weight)] * n

    result = minimize(
        neg_sharpe,
        w0,
        jac=neg_sharpe_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    if result.success and not np.isnan(result.x).any():
        w_opt = result.x
    else:
        # Fallback: equal weight
        w_opt = w0

    # Clip numerical noise and re-normalise
    w_opt = np.clip(w_opt, 0.0, None)
    if w_opt.sum() > 1e-10:
        w_opt /= w_opt.sum()
    else:
        w_opt = w0

    return pd.Series(w_opt, index=tickers)


# ---------------------------------------------------------------------------
# Main strategy runner
# ---------------------------------------------------------------------------

def run_mvo_ml_strategy(
    stock_prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    rebalance_freq: int = 21,
    forward_days: int = 20,
    ridge_alpha: float = 1.0,
    min_train_obs: int = 30,
    max_weight: float = 0.25,
    cov_window: int = 60,
) -> pd.DataFrame:
    """
    Run the full MVO + Ridge Regression strategy.

    Parameters
    ----------
    stock_prices   : daily adj-close (date × ticker)
    fundamentals   : static fundamentals snapshot (ticker × field)
    rebalance_freq : trading days between rebalances (21 ≈ monthly)
    forward_days   : prediction horizon for the target variable
    ridge_alpha    : regularisation strength for Ridge Regression
    min_train_obs  : minimum (ticker × date) rows before first trade
    max_weight     : maximum single-stock allocation in MVO
    cov_window     : rolling window (days) for realised covariance

    Returns
    -------
    pd.DataFrame  of shape (n_rebalance_dates × n_tickers)
    """
    print("  MVO + ML Strategy  |  rebalance=monthly  ridge_alpha={:.1f}".format(ridge_alpha))

    tickers   = stock_prices.columns.tolist()
    all_dates = stock_prices.index

    # --- Pre-compute features and targets at all monthly dates ---
    print("  Pre-computing features and targets …")
    features_by_date = _precompute_features(stock_prices, fundamentals, rebalance_freq)
    targets_by_date  = _precompute_targets(stock_prices, rebalance_freq, forward_days)

    # Sorted list of all feature dates (used for constructing training set)
    all_feature_dates = sorted(features_by_date.keys())

    # --- Rebalancing dates: same grid as feature dates ---
    # We need enough training data before the first real trade.
    # The training constraint is:  feature_date + forward_days < rebal_date
    # → minimum gap of forward_days between last training sample and T.

    weights_list: list[tuple[pd.Timestamp, pd.Series]] = []

    print(f"  Running expanding-window training for {len(all_feature_dates)} candidate dates …")

    daily_returns = stock_prices.pct_change()

    for i, rebal_date in enumerate(all_feature_dates):
        rebal_idx = all_dates.get_loc(rebal_date)

        # --- Build expanding training set ---
        # Only include (feature_date, target) pairs where
        #   feature_date + forward_days < rebal_date  (no look-ahead)
        X_blocks: list[np.ndarray] = []
        y_blocks: list[np.ndarray] = []

        for feat_date in all_feature_dates:
            feat_idx = all_dates.get_loc(feat_date)

            # Strict look-ahead check: target for feat_date is price at
            # feat_date + forward_days; that must be known before rebal_date.
            if feat_idx + forward_days >= rebal_idx:
                break  # dates are sorted; no later date can satisfy this

            if feat_date not in targets_by_date:
                continue

            feat_df = features_by_date[feat_date]
            target  = targets_by_date[feat_date]

            # Align on common tickers with valid data
            common = feat_df.index.intersection(target.index)
            if len(common) < 3:
                continue

            X_block = feat_df.loc[common, FEATURE_COLS].values
            y_block = target.loc[common].values

            # Fill NaN P/E with cross-sectional median (for this date only)
            pe_col = FEATURE_COLS.index("trailingPE")
            pe_vals = X_block[:, pe_col]
            pe_median = np.nanmedian(pe_vals)
            pe_vals[np.isnan(pe_vals)] = pe_median if not np.isnan(pe_median) else 0.0
            X_block[:, pe_col] = pe_vals

            # Skip rows still containing NaN
            valid_mask = ~np.isnan(X_block).any(axis=1) & ~np.isnan(y_block)
            if valid_mask.sum() < 2:
                continue

            X_blocks.append(X_block[valid_mask])
            y_blocks.append(y_block[valid_mask])

        if not X_blocks:
            continue

        X_train = np.vstack(X_blocks)
        y_train = np.concatenate(y_blocks)

        if len(X_train) < min_train_obs:
            continue  # not enough history yet

        # --- Train Ridge (scaler fit on training data only) ---
        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        model = Ridge(alpha=ridge_alpha)
        model.fit(X_scaled, y_train)

        # --- Predict expected returns at rebal_date ---
        if rebal_date not in features_by_date:
            continue

        feat_now = features_by_date[rebal_date].copy()

        # Fill NaN P/E with cross-sectional median
        pe_col_name = "trailingPE"
        pe_median_now = feat_now[pe_col_name].median()
        feat_now[pe_col_name] = feat_now[pe_col_name].fillna(
            pe_median_now if not np.isnan(pe_median_now) else 0.0
        )

        # Drop rows with remaining NaN
        feat_now = feat_now.dropna()
        if len(feat_now) < 3:
            continue

        X_pred     = scaler.transform(feat_now[FEATURE_COLS].values)
        pred_ret   = model.predict(X_pred)
        exp_ret    = pd.Series(pred_ret, index=feat_now.index)

        # --- Covariance matrix: 60-day rolling realised covariance ---
        # Use only data up to and including rebal_date (no look-ahead)
        ret_window = daily_returns.iloc[max(0, rebal_idx - cov_window + 1) : rebal_idx + 1]
        ret_window = ret_window[feat_now.index]   # align to predicted tickers

        # Drop tickers with more than 20% missing in the window
        min_obs = int(0.8 * len(ret_window))
        ret_window = ret_window.dropna(axis=1, thresh=min_obs)
        ret_window = ret_window.fillna(0.0)       # remaining NaN → 0 return

        common_tickers = exp_ret.index.intersection(ret_window.columns)
        if len(common_tickers) < 3:
            continue

        exp_ret    = exp_ret[common_tickers]
        cov_matrix = ret_window[common_tickers].cov() * 252   # annualise

        # Regularise to avoid near-singular matrix
        cov_matrix = cov_matrix + np.eye(len(common_tickers)) * 1e-6

        # --- MVO ---
        opt_w = _mvo_optimize(exp_ret, cov_matrix, max_weight=max_weight)

        # Expand to full ticker universe (zero weight for excluded tickers)
        full_w = pd.Series(0.0, index=tickers)
        for t in opt_w.index:
            if t in full_w.index:
                full_w[t] = opt_w[t]

        weights_list.append((rebal_date, full_w))

    if not weights_list:
        print("  [warn] MVO+ML strategy produced no weights — check data coverage.")
        return pd.DataFrame(columns=tickers)

    weights = pd.DataFrame(
        [w for _, w in weights_list],
        index=pd.DatetimeIndex([d for d, _ in weights_list]),
    )

    print(f"  → {len(weights)} rebalancing events generated")
    return weights
