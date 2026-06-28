"""
Mean-variance momentum tracking features.

PAPER: "Deterministic Policy Gradient for Learning Equilibrium in Time-Inconsistent
Control Problems" (arxiv 2606.11798) — purely a RL/control paper with no extractable
OHLCV feature. SKIPPED as-is.

SUBSTITUTE (same theme — mean-variance portfolio theory):
  Implements a per-ticker MEAN-VARIANCE EFFICIENCY SCORE family.
  The classical Markowitz mean-variance framework motivates looking at the
  risk-adjusted slope of the return path and how it varies across horizons.
  We compute:
    - Sharpe-slope ratio across multiple trailing windows (rolling mean / std)
    - Sortino-slope ratio (downside std only)
    - Calmar-slope ratio (mean / max-drawdown)
    - Cross-window dispersion of the Sharpe slope (regime signal)

  These capture the *current* risk-efficiency of a stock's recent price path,
  which is distinct from raw momentum or ATR.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_11798_mvmom_tracking",
    "description": (
        "Mean-variance efficiency (Sharpe/Sortino/Calmar slope) features across "
        "multiple horizons; inspired by mean-variance portfolio theory from "
        "arxiv 2606.11798 (time-inconsistent control / mean-variance tracking)."
    ),
    "requires": ["Close"],
    "produces": [
        "mvmom_sharpe_20",
        "mvmom_sharpe_60",
        "mvmom_sharpe_120",
        "mvmom_sortino_20",
        "mvmom_sortino_60",
        "mvmom_calmar_60",
        "mvmom_horizon_spread",
    ],
    "tags": ["momentum", "volatility", "experimental"],
    "version": "1.0",
    "author": "paper:2606.11798",
}


def _rolling_sharpe(log_ret: pd.Series, window: int, ann: float = 252.0) -> pd.Series:
    """Annualised rolling Sharpe (mean / std) of daily log returns."""
    mu = log_ret.rolling(window, min_periods=max(window // 2, 5)).mean()
    sigma = log_ret.rolling(window, min_periods=max(window // 2, 5)).std()
    return (mu / sigma.replace(0, np.nan)) * np.sqrt(ann)


def _rolling_sortino(log_ret: pd.Series, window: int, ann: float = 252.0) -> pd.Series:
    """Annualised rolling Sortino (mean / downside-std)."""
    mu = log_ret.rolling(window, min_periods=max(window // 2, 5)).mean()

    def downside_std(x):
        neg = x[x < 0]
        return np.std(neg) if len(neg) >= 3 else np.nan

    dd_std = log_ret.rolling(window, min_periods=max(window // 2, 5)).apply(
        downside_std, raw=True
    )
    return (mu / dd_std.replace(0, np.nan)) * np.sqrt(ann)


def _rolling_calmar(log_ret: pd.Series, window: int) -> pd.Series:
    """Rolling Calmar ratio: annualised return / max drawdown over the window."""
    mu = log_ret.rolling(window, min_periods=max(window // 2, 5)).mean() * 252.0

    def max_dd(x):
        cum = np.cumsum(x)
        roll_max = np.maximum.accumulate(cum)
        dd = roll_max - cum
        mdd = np.max(dd)
        return mdd if mdd > 0 else np.nan

    mdd = log_ret.rolling(window, min_periods=max(window // 2, 5)).apply(
        max_dd, raw=True
    )
    return mu / mdd.replace(0, np.nan)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    # ---- Rolling Sharpe across horizons ---------------------------------------
    sh20 = _rolling_sharpe(log_ret, 20)
    sh60 = _rolling_sharpe(log_ret, 60)
    sh120 = _rolling_sharpe(log_ret, 120)

    df["mvmom_sharpe_20"] = sh20
    df["mvmom_sharpe_60"] = sh60
    df["mvmom_sharpe_120"] = sh120

    # ---- Rolling Sortino ------------------------------------------------------
    df["mvmom_sortino_20"] = _rolling_sortino(log_ret, 20)
    df["mvmom_sortino_60"] = _rolling_sortino(log_ret, 60)

    # ---- Rolling Calmar -------------------------------------------------------
    df["mvmom_calmar_60"] = _rolling_calmar(log_ret, 60)

    # ---- Cross-horizon spread of Sharpe (regime signal) ----------------------
    # High spread => conflicting signals across horizons (transition/uncertainty)
    sh_stack = pd.concat([sh20, sh60, sh120], axis=1)
    df["mvmom_horizon_spread"] = sh_stack.max(axis=1) - sh_stack.min(axis=1)

    return df
