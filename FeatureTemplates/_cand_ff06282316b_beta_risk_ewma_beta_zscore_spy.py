from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional helper: market index loader
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06282316b_beta_risk_ewma_beta_zscore_spy",
    "description": (
        "EWMA SPY-beta z-score: compute a rolling exponentially-weighted "
        "covariance/variance beta vs SPY (half-life ~20 days), then z-score "
        "the current beta against its trailing 250-day mean and std. "
        "Captures how stretched the stock's current market exposure is relative "
        "to its recent norm. Three columns: raw EWMA beta, its 250-day z-score, "
        "and the magnitude (abs z-score) for convex risk-regime signals."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282316b_beta_risk_ewma_beta_zscore_spy_ewma_beta",
        "ff06282316b_beta_risk_ewma_beta_zscore_spy_zscore",
        "ff06282316b_beta_risk_ewma_beta_zscore_spy_abs_zscore",
    ],
    "tags": ["beta", "risk", "ewma", "market_exposure", "z-score", "spy"],
    "version": "1.0.0",
    "author": "feature-factory ff06282316b",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EWMA_HALFLIFE = 20       # days
_LOOKBACK = 250           # days for z-score history
_EPS = 1e-9


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EWMA SPY-beta and its trailing z-score."""
    out_ewma = "ff06282316b_beta_risk_ewma_beta_zscore_spy_ewma_beta"
    out_zscore = "ff06282316b_beta_risk_ewma_beta_zscore_spy_zscore"
    out_abs = "ff06282316b_beta_risk_ewma_beta_zscore_spy_abs_zscore"

    # Initialise all produced columns to NaN on every code path
    df[out_ewma] = np.nan
    df[out_zscore] = np.nan
    df[out_abs] = np.nan

    if len(df) < 5:
        return df

    # ------------------------------------------------------------------
    # 1. Pull SPY close and align to the stock's dates via merge_asof
    # ------------------------------------------------------------------
    try:
        spy_series = _indexes.index_close("SPY")
        if spy_series is None or len(spy_series) == 0:
            return df
        spy_df = spy_series.reset_index()
        spy_df.columns = ["Date", "_spy_close"]
    except Exception:
        return df

    # Ensure Date column is datetime in both frames
    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    merged = pd.merge_asof(
        work.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Re-align to original df order
    merged = merged.set_index("Date")

    if merged["_spy_close"].isna().all():
        return df

    stock_close = merged["Close"]
    spy_close = merged["_spy_close"]

    # ------------------------------------------------------------------
    # 2. Daily log-returns (causal; first row is NaN)
    # ------------------------------------------------------------------
    stock_ret = np.log(stock_close / stock_close.shift(1))
    spy_ret = np.log(spy_close / spy_close.shift(1))

    # ------------------------------------------------------------------
    # 3. EWMA covariance and variance -> beta
    #    Using pandas EWMA with half-life; min_periods=10 to avoid
    #    noisy estimates at the start.
    # ------------------------------------------------------------------
    alpha = 1.0 - np.exp(-np.log(2.0) / _EWMA_HALFLIFE)  # decay param

    ewma_cov = stock_ret.ewm(
        halflife=_EWMA_HALFLIFE, min_periods=10, adjust=False
    ).cov(spy_ret)

    ewma_var = spy_ret.ewm(
        halflife=_EWMA_HALFLIFE, min_periods=10, adjust=False
    ).var()

    # Guard division
    denom = ewma_var.where(ewma_var.abs() > _EPS, np.nan)
    ewma_beta = ewma_cov / denom

    # ------------------------------------------------------------------
    # 4. Trailing 250-day z-score of ewma_beta
    # ------------------------------------------------------------------
    roll_mean = ewma_beta.rolling(window=_LOOKBACK, min_periods=30).mean()
    roll_std = ewma_beta.rolling(window=_LOOKBACK, min_periods=30).std()

    std_guard = roll_std.where(roll_std > _EPS, np.nan)
    beta_zscore = (ewma_beta - roll_mean) / std_guard

    # ------------------------------------------------------------------
    # 5. Map back to df (align on Date index)
    # ------------------------------------------------------------------
    df_dates = pd.to_datetime(df["Date"])
    df.loc[df.index, out_ewma] = ewma_beta.reindex(df_dates).values
    df.loc[df.index, out_zscore] = beta_zscore.reindex(df_dates).values
    df.loc[df.index, out_abs] = beta_zscore.abs().reindex(df_dates).values

    # Final safety: replace inf/-inf with NaN
    for col in [out_ewma, out_zscore, out_abs]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
