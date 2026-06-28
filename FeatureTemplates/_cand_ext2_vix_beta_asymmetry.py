"""
Candidate feature block: ext2_vix_beta_asymmetry
Asymmetric sensitivity to VIX up vs. down moves — directional-conditioned OLS beta.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd
import warnings

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path — never via package import)
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
    "name": "ext2_vix_beta_asymmetry",
    "description": (
        "Rolling 120-day OLS of daily stock return on daily VIX change, "
        "estimated SEPARATELY on VIX-up days and VIX-down days. "
        "Produces: beta on VIX-up days (crash fragility), beta on VIX-down days, "
        "and the asymmetry (up_beta minus down_beta). "
        "Extends the parent osap_betatailrisk feature with directional conditioning "
        "to capture whether a stock is disproportionately hurt by fear spikes vs. "
        "fear relief rallies — a distinct signal axis from unconditional VIX beta. "
        "Per-ticker proxy; uses _indexes VIX series merged backward (no lookahead)."
    ),
    "requires": ["Close"],
    "produces": [
        "ext2_vix_beta_asymmetry_up",    # beta on VIX-rising days (crash fragility)
        "ext2_vix_beta_asymmetry_down",  # beta on VIX-falling days
        "ext2_vix_beta_asymmetry_diff",  # asymmetry = up_beta - down_beta
    ],
    "tags": ["vix", "beta", "asymmetry", "macro", "tail-risk", "rolling-ols"],
    "version": "1.0.0",
    "author": (
        "Spec: ext2_vix_beta_asymmetry — Round-3 deep exploration of the "
        "osap_betatailrisk winner vein (SOURCE: project internal spec)."
    ),
}

# ---------------------------------------------------------------------------
# Rolling OLS helper (vectorised via numpy sliding windows)
# ---------------------------------------------------------------------------
_WINDOW = 120  # trading days


def _rolling_ols_slope(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """
    Return rolling OLS slope of y on x over `window` observations.
    Uses a numerically stable incremental approach via numpy stride tricks.
    Output length == len(y); first (window-1) positions are NaN.
    """
    n = len(y)
    out = np.full(n, np.nan)
    if n < window:
        return out

    # Use stride tricks to get (n-window+1, window) views — no Python loop per row
    from numpy.lib.stride_tricks import sliding_window_view
    y_wins = sliding_window_view(y, window)   # shape (n-window+1, window)
    x_wins = sliding_window_view(x, window)

    # Vectorised OLS: slope = cov(x,y) / var(x)
    x_mean = x_wins.mean(axis=1)
    y_mean = y_wins.mean(axis=1)
    x_dev = x_wins - x_mean[:, None]
    y_dev = y_wins - y_mean[:, None]
    cov_xy = (x_dev * y_dev).sum(axis=1)
    var_x  = (x_dev * x_dev).sum(axis=1)

    # Guard divide-by-zero (flat x within window → nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope = np.where(var_x == 0, np.nan, cov_xy / var_x)

    out[window - 1:] = slope
    return out


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-ticker: compute asymmetric VIX-beta features.
    df is one stock, ascending Date, columns include at least Close.
    Adds three columns to df and returns df.
    """
    n = len(df)
    nan_col = np.full(n, np.nan)

    df["ext2_vix_beta_asymmetry_up"]   = nan_col.copy()
    df["ext2_vix_beta_asymmetry_down"] = nan_col.copy()
    df["ext2_vix_beta_asymmetry_diff"] = nan_col.copy()

    if n < _WINDOW + 2:
        return df

    # -----------------------------------------------------------------------
    # 1. Daily stock return (log, using shifted Close — no lookahead)
    # -----------------------------------------------------------------------
    close = df["Close"].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ret = np.where(
            (close[:-1] == 0) | np.isnan(close[:-1]),
            np.nan,
            np.log(close[1:] / close[:-1])
        )
    # prepend NaN so ret[i] = return at bar i (using close[i-1]→close[i])
    ret = np.concatenate([[np.nan], ret])

    # -----------------------------------------------------------------------
    # 2. VIX daily close → daily VIX change
    # -----------------------------------------------------------------------
    try:
        vix_df = _indexes.vix_daily_close()   # DataFrame[["Date","vix_close"]]
        tmp = df[["Date"]].copy()
        tmp["_idx"] = np.arange(n)
        merged = pd.merge_asof(
            tmp.sort_values("Date"),
            vix_df.sort_values("Date"),
            on="Date",
            direction="backward"
        ).sort_values("_idx")
        vix_close = merged["vix_close"].to_numpy(dtype=float)
    except Exception:
        # If VIX unavailable, all outputs stay NaN
        return df

    # VIX change (same day alignment): vix_chg[i] = vix_close[i] - vix_close[i-1]
    vix_chg = np.empty(n)
    vix_chg[0] = np.nan
    vix_chg[1:] = vix_close[1:] - vix_close[:-1]

    # -----------------------------------------------------------------------
    # 3. Build masked series: NaN out the "other" direction each day
    # -----------------------------------------------------------------------
    vix_up   = vix_chg > 0   # VIX rose
    vix_down = vix_chg < 0   # VIX fell  (flat days excluded from both)

    ret_for_up   = np.where(vix_up,   ret,      np.nan)
    x_for_up     = np.where(vix_up,   vix_chg,  np.nan)
    ret_for_down = np.where(vix_down,  ret,      np.nan)
    x_for_down   = np.where(vix_down,  vix_chg,  np.nan)

    # -----------------------------------------------------------------------
    # 4. Rolling OLS over the full window for each direction subset.
    #    We use the raw masked arrays: within each window, OLS is computed
    #    only over the non-NaN (x,y) pairs.
    # -----------------------------------------------------------------------
    # We need a Python-loop per window here because the mask varies per window.
    # However we vectorise the inner OLS with numpy operations.
    # At 700 rows and window=120 this is ~580 iterations — well under 100ms.
    from numpy.lib.stride_tricks import sliding_window_view

    # pad to allow sliding_window_view
    ret_up_wins = sliding_window_view(ret_for_up,   _WINDOW)  # (n-W+1, W)
    x_up_wins   = sliding_window_view(x_for_up,     _WINDOW)
    ret_dn_wins = sliding_window_view(ret_for_down, _WINDOW)
    x_dn_wins   = sliding_window_view(x_for_down,   _WINDOW)

    n_wins = ret_up_wins.shape[0]
    beta_up   = np.full(n_wins, np.nan)
    beta_down = np.full(n_wins, np.nan)

    for i in range(n_wins):
        # VIX-up beta
        ry = ret_up_wins[i]
        rx = x_up_wins[i]
        mask = np.isfinite(ry) & np.isfinite(rx)
        if mask.sum() >= 10:   # require at least 10 VIX-up days in window
            xm = rx[mask] - rx[mask].mean()
            ym = ry[mask] - ry[mask].mean()
            vx = (xm * xm).sum()
            beta_up[i] = np.nan if vx == 0 else (xm * ym).sum() / vx

        # VIX-down beta
        ry = ret_dn_wins[i]
        rx = x_dn_wins[i]
        mask = np.isfinite(ry) & np.isfinite(rx)
        if mask.sum() >= 10:
            xm = rx[mask] - rx[mask].mean()
            ym = ry[mask] - ry[mask].mean()
            vx = (xm * xm).sum()
            beta_down[i] = np.nan if vx == 0 else (xm * ym).sum() / vx

    # Align back to df index (first _WINDOW-1 rows stay NaN)
    start = _WINDOW - 1
    df["ext2_vix_beta_asymmetry_up"]   = np.concatenate([np.full(start, np.nan), beta_up])
    df["ext2_vix_beta_asymmetry_down"] = np.concatenate([np.full(start, np.nan), beta_down])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diff = df["ext2_vix_beta_asymmetry_up"].to_numpy(dtype=float) - \
               df["ext2_vix_beta_asymmetry_down"].to_numpy(dtype=float)
    # Replace inf with nan just in case
    diff = np.where(np.isfinite(diff), diff, np.nan)
    df["ext2_vix_beta_asymmetry_diff"] = diff

    return df
