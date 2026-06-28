"""
_p625_capm_resid_moments.py -- Short-horizon CAPM-residual higher moments vs SPY.

Intercept-included rolling CAPM (beta + alpha) regressing stock log returns on SPY
log returns, then higher moments of the residual e = r - alpha - beta*m:
  - residual skewness (21d, 63d)        Boyer, Mitton & Vorkink (2010) RFS
  - residual kurtosis (21d)             Amaya et al. (2015) JFE
  - idiosyncratic-vol share (21d)       Blitz, Huij & Martens (2011) JFE
  - idiosyncratic downside-variance share (63d)

Strictly causal: every statistic at row t uses only rows <= t (rolling windows).
Index returns are aligned to the stock by BACKWARD merge via inner-join on Date.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load the shared index helper (skipped by framework auto-discovery)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _Path(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

METADATA = {
    "name":        "_p625_capm_resid_moments",
    "description": (
        "Short-horizon (21d/63d) CAPM-residual higher moments vs SPY: residual skew, "
        "kurtosis, idio-vol share, idio downside-variance share "
        "(Boyer-Mitton-Vorkink 2010 RFS; Amaya et al. 2015 JFE; Blitz-Huij-Martens 2011 JFE)."
    ),
    "requires":    ["Date", "Close"],
    "produces":    [
        "ires_skew_21",
        "ires_kurt_21",
        "ires_cv_21",
        "ires_skew_63",
        "ires_downvar_63",
    ],
    "tags":        ["volatility", "beta", "experimental"],
    "version":     "1.0",
    "author":      "paper:Boyer-Mitton-Vorkink 2010 RFS; Amaya 2015 JFE; Blitz-Huij-Martens 2011 JFE",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pre-create columns so they always exist even if SPY is unavailable for this ticker.
    for col in METADATA["produces"]:
        df[col] = np.nan

    df_dates = pd.to_datetime(df["Date"]).values

    # Stock log returns, Date-indexed (temporary, never stored on df).
    stock_close = pd.Series(df["Close"].astype(float).values, index=pd.to_datetime(df["Date"]))
    r_full = np.log(stock_close / stock_close.shift(1))

    try:
        idx_close = _indexes.index_close("SPY")  # Series, DatetimeIndex 'Date'
    except Exception:
        return df
    if idx_close is None or idx_close.empty:
        return df

    m_full = np.log(idx_close.astype(float) / idx_close.astype(float).shift(1))

    # Inner-join alignment so we only regress over shared trading dates (backward in spirit:
    # both series are causal log returns; alignment merely intersects calendars).
    r, m = r_full.align(m_full, join="inner")
    if len(r) < 30:
        return df

    def _causal_capm_residual(window: int) -> pd.Series:
        """Residual e = r - alpha_W - beta_W*m using rolling-only CAPM. Strictly causal."""
        mp = int(0.7 * window)
        cov_rm = r.rolling(window, min_periods=mp).cov(m)
        var_m = m.rolling(window, min_periods=mp).var()
        beta = (cov_rm / var_m.replace(0, np.nan)).clip(-5, 5)
        mean_r = r.rolling(window, min_periods=mp).mean()
        mean_m = m.rolling(window, min_periods=mp).mean()
        alpha = mean_r - beta * mean_m
        return r - alpha - beta * m

    # ---- 21-day window ------------------------------------------------------
    W21 = 21
    mp21 = int(0.7 * W21)
    e21 = _causal_capm_residual(W21)

    skew_21 = e21.rolling(W21, min_periods=mp21).skew().clip(-15, 15)
    kurt_21 = e21.rolling(W21, min_periods=mp21).kurt().clip(-5, 200)

    e21_std = e21.rolling(W21, min_periods=mp21).std()
    r_std_21 = r.rolling(W21, min_periods=mp21).std()
    cv_21 = e21_std / (r_std_21 + 1e-9)

    # ---- 63-day window ------------------------------------------------------
    W63 = 63
    mp63 = int(0.7 * W63)
    e63 = _causal_capm_residual(W63)

    skew_63 = e63.rolling(W63, min_periods=mp63).skew().clip(-15, 15)

    # Idiosyncratic downside-variance share: sum(min(e,0)^2) / sum(e^2), rolling.
    e63_sq = e63 * e63
    e63_down_sq = np.minimum(e63, 0.0) ** 2
    sum_sq = e63_sq.rolling(W63, min_periods=mp63).sum()
    sum_down = e63_down_sq.rolling(W63, min_periods=mp63).sum()
    downvar_63 = sum_down / sum_sq.replace(0, np.nan)

    # ---- Reindex aligned results back onto the original df row order --------
    df["ires_skew_21"]    = skew_21.reindex(df_dates).values
    df["ires_kurt_21"]    = kurt_21.reindex(df_dates).values
    df["ires_cv_21"]      = cv_21.reindex(df_dates).values
    df["ires_skew_63"]    = skew_63.reindex(df_dates).values
    df["ires_downvar_63"] = downvar_63.reindex(df_dates).values

    return df
