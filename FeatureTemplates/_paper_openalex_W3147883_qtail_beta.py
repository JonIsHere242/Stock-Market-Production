"""
Quantile tail-beta: ticker return sensitivity to EXTREME market moves, derived from:
  "Impact of the COVID-19 outbreak on the US equity sectors: Evidence from quantile
   return spillovers" (openalex:W3147883316).

The paper extends the Diebold-Yilmaz spillover framework to the quantiles domain,
measuring return spillovers at extreme market quantiles (crash/euphoria states)
separately from the unconditional average.  We proxy this PER TICKER as:
  - qtbeta_lo_N: OLS slope of ticker daily returns on SPY returns, estimated only
    over the lowest-quintile SPY return days within a rolling N-day window.
    Captures downside tail co-movement (left tail beta, distinct from standard beta).
  - qtbeta_hi_N: same but estimated on highest-quintile SPY return days.
    Captures upside tail participation.
  - qtbeta_asym_N: qtbeta_hi - qtbeta_lo (tail asymmetry / skew in market sensitivity).
    Positive = ticker surfs rallies more than it falls in crashes (desirable).
Distinct from standard beta (full-sample average), downside-beta (return-signed,
not quantile-conditioned), and coskewness (3rd-moment statistic, not a conditional slope).
All rolling, causal, no lookahead.
"""

import numpy as np
import pandas as pd

try:
    from _indexes import index_close
    _SPY = index_close("SPY")
except Exception:
    _SPY = pd.Series(dtype="float64")

METADATA = {
    "name":        "_paper_openalex_W3147883_qtail_beta",
    "description": "Tail-quantile conditional beta vs SPY at crash and euphoria market states.",
    "requires":    ["Close"],
    "produces":    [
        "qtbeta_lo_63",
        "qtbeta_hi_63",
        "qtbeta_asym_63",
        "qtbeta_lo_126",
        "qtbeta_hi_126",
        "qtbeta_asym_126",
    ],
    "tags":        ["experimental", "market_regime", "tail_risk", "beta"],
    "version":     "1.0",
    "author":      "paper-mining slate 2",
}


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Vectorised OLS slope y ~ x (no intercept needed for small variations)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan
    xm = x[mask]; ym = y[mask]
    xvar = np.dot(xm - xm.mean(), xm - xm.mean())
    if xvar <= 0:
        return np.nan
    return np.dot(xm - xm.mean(), ym - ym.mean()) / xvar


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Align SPY returns to ticker dates (backward merge_asof)
    ticker_dates = pd.to_datetime(df["Date"] if "Date" in df.columns else df.index)

    if _SPY.empty:
        for w in (63, 126):
            df[f"qtbeta_lo_{w}"] = np.nan
            df[f"qtbeta_hi_{w}"] = np.nan
            df[f"qtbeta_asym_{w}"] = np.nan
        return df

    spy_ret = _SPY.pct_change()  # DatetimeIndex Series
    spy_df = spy_ret.rename("spy_ret").reset_index()  # columns: Date, spy_ret
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    ticker_ret = df["Close"].pct_change().values  # np array

    if "Date" in df.columns:
        date_col = pd.to_datetime(df["Date"])
    else:
        date_col = pd.to_datetime(df.index)

    # Merge SPY returns to ticker dates
    tmp = pd.DataFrame({"Date": date_col})
    tmp = pd.merge_asof(tmp.sort_values("Date"), spy_df.sort_values("Date"),
                        on="Date", direction="backward")
    spy_aligned = tmp["spy_ret"].values  # aligned to ticker row order (after resort)

    n = len(df)
    for window in (63, 126):
        lo_arr = np.full(n, np.nan)
        hi_arr = np.full(n, np.nan)
        q20 = 0.20  # bottom quintile
        q80 = 0.80  # top quintile

        for i in range(window - 1, n):
            spy_w = spy_aligned[i - window + 1 : i + 1]
            tick_w = ticker_ret[i - window + 1 : i + 1]
            valid = np.isfinite(spy_w) & np.isfinite(tick_w)
            if valid.sum() < 10:
                continue
            spy_v = spy_w[valid]
            tick_v = tick_w[valid]
            cutlo = np.quantile(spy_v, q20)
            cuthi = np.quantile(spy_v, q80)
            lo_mask = spy_v <= cutlo
            hi_mask = spy_v >= cuthi
            lo_arr[i] = _ols_slope(spy_v[lo_mask], tick_v[lo_mask])
            hi_arr[i] = _ols_slope(spy_v[hi_mask], tick_v[hi_mask])

        df[f"qtbeta_lo_{window}"] = lo_arr
        df[f"qtbeta_hi_{window}"] = hi_arr
        df[f"qtbeta_asym_{window}"] = hi_arr - lo_arr

    return df
