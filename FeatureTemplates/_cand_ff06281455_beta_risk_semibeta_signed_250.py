"""
Signed semibeta decomposition vs SPY (Bollerslev-Li-Patton style).

Decomposes the 250-day rolling covariance into three sign-quadrant buckets:
  beta_N  : concordant-down  (stock return < 0 AND SPY return < 0)
  beta_P  : concordant-up    (stock return > 0 AND SPY return > 0)
  beta_M  : mixed/discordant (stock and SPY returns have opposite signs)

Each beta_k = sum(r_stock * r_spy | quadrant k) / var(r_spy over full window).
This captures downside co-movement (tail risk), upside co-participation, and
hedging/friction separately -- a strictly different axis from standard OLS beta
or downside-beta.  Per-ticker proxy; economically faithful to the cross-sectional
paper definition but applied in a rolling single-ticker context.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper by file path
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06281455_beta_risk_semibeta_signed_250",
    "description": (
        "Bollerslev-Li-Patton signed semibeta decomposition vs SPY over 250-day "
        "rolling window. Produces three components: concordant-down (beta_N), "
        "concordant-up (beta_P), and discordant/mixed (beta_M). Each = "
        "sum(r_stock * r_spy | sign quadrant) / var(r_spy, full window). "
        "Distinct from plain beta or downside-beta; captures asymmetric tail "
        "co-movement vs index. Per-ticker rolling proxy."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06281455_beta_risk_semibeta_signed_250_beta_N",
        "ff06281455_beta_risk_semibeta_signed_250_beta_P",
        "ff06281455_beta_risk_semibeta_signed_250_beta_M",
    ],
    "tags": ["beta", "semibeta", "risk", "signed", "SPY", "downside", "rolling"],
    "version": "1.0.0",
    "author": "feature-factory ff06281455",
}

_WINDOW = 250
_COL_N = "ff06281455_beta_risk_semibeta_signed_250_beta_N"
_COL_P = "ff06281455_beta_risk_semibeta_signed_250_beta_P"
_COL_M = "ff06281455_beta_risk_semibeta_signed_250_beta_M"


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise outputs to NaN on every code path
    df[_COL_N] = np.nan
    df[_COL_P] = np.nan
    df[_COL_M] = np.nan

    if len(df) < _WINDOW + 1:
        return df

    # ------------------------------------------------------------------
    # Fetch SPY closes and merge backward (lookahead-safe)
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
        if spy_close is None or len(spy_close) == 0:
            return df
        spy_df = spy_close.rename("spy_close").reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    except Exception:
        return df

    merged = pd.merge_asof(
        df[["Date"]].assign(Date=pd.to_datetime(df["Date"])).sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order
    merged.index = df.index
    spy_aligned = merged["spy_close"].values.astype(float)

    # ------------------------------------------------------------------
    # Compute daily returns (log approx via pct_change is fine)
    # ------------------------------------------------------------------
    close = df["Close"].values.astype(float)

    # Stock returns
    r_stock = np.empty(len(close), dtype=float)
    r_stock[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        r_stock[1:] = np.where(
            close[:-1] > 0,
            close[1:] / close[:-1] - 1.0,
            np.nan,
        )

    # SPY returns
    r_spy = np.empty(len(spy_aligned), dtype=float)
    r_spy[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        r_spy[1:] = np.where(
            spy_aligned[:-1] > 0,
            spy_aligned[1:] / spy_aligned[:-1] - 1.0,
            np.nan,
        )

    n = len(df)
    beta_N_arr = np.full(n, np.nan)
    beta_P_arr = np.full(n, np.nan)
    beta_M_arr = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # Rolling semibeta calculation
    # Vectorised: use stride-trick / pandas rolling via apply would be slow;
    # instead build the arrays and iterate only over output indices.
    # We use numpy sliding windows approach: for each t in [WINDOW-1, n-1]
    # slice window indices. To stay fast, precompute cross-product and use
    # cumulative sums (prefix sums per quadrant).
    # ------------------------------------------------------------------

    # Precompute products and quadrant masks
    xy = r_stock * r_spy  # element-wise; NaN propagates

    # Quadrant indicator arrays (True/False), NaN-aware
    # concordant-down: both negative
    mask_N = (r_stock < 0) & (r_spy < 0)
    # concordant-up: both positive
    mask_P = (r_stock > 0) & (r_spy > 0)
    # mixed/discordant: opposite signs
    mask_M = ((r_stock > 0) & (r_spy < 0)) | ((r_stock < 0) & (r_spy > 0))

    # We need valid (non-NaN) data in each window; use masked arrays
    xy_N = np.where(mask_N, xy, 0.0)
    xy_P = np.where(mask_P, xy, 0.0)
    xy_M = np.where(mask_M, xy, 0.0)
    spy2 = np.where(np.isfinite(r_spy), r_spy ** 2, 0.0)

    # Build prefix sums (index i = sum of [0..i-1])
    cs_N = np.nancumsum(np.where(np.isfinite(xy_N) & mask_N, xy_N, 0.0))
    cs_P = np.nancumsum(np.where(np.isfinite(xy_P) & mask_P, xy_P, 0.0))
    cs_M = np.nancumsum(np.where(np.isfinite(xy_M) & mask_M, xy_M, 0.0))
    # Var(r_spy) denominator: use sum of r_spy^2 in window (mean-corrected below)
    # We use full-window sum of r_spy^2 and mean for variance
    cs_spy2 = np.nancumsum(np.where(np.isfinite(r_spy), r_spy ** 2, 0.0))
    cs_spy  = np.nancumsum(np.where(np.isfinite(r_spy), r_spy, 0.0))
    cs_cnt  = np.cumsum(np.isfinite(r_spy).astype(float))

    # Also track valid-pair counts per quadrant for robustness
    cnt_N = np.cumsum(mask_N.astype(float))
    cnt_P = np.cumsum(mask_P.astype(float))
    cnt_M = np.cumsum(mask_M.astype(float))

    for t in range(_WINDOW - 1, n):
        t0 = t - _WINDOW + 1  # window start index (inclusive)

        # Window sums via prefix difference
        sum_N = cs_N[t] - (cs_N[t0 - 1] if t0 > 0 else 0.0)
        sum_P = cs_P[t] - (cs_P[t0 - 1] if t0 > 0 else 0.0)
        sum_M = cs_M[t] - (cs_M[t0 - 1] if t0 > 0 else 0.0)

        spy2_sum = cs_spy2[t] - (cs_spy2[t0 - 1] if t0 > 0 else 0.0)
        spy_sum  = cs_spy[t]  - (cs_spy[t0 - 1]  if t0 > 0 else 0.0)
        cnt      = cs_cnt[t]  - (cs_cnt[t0 - 1]  if t0 > 0 else 0.0)

        if cnt < 2:
            continue

        # Mean-corrected variance of r_spy over window
        spy_var = spy2_sum / cnt - (spy_sum / cnt) ** 2
        if not np.isfinite(spy_var) or spy_var <= 0.0:
            continue

        beta_N_arr[t] = sum_N / spy_var / cnt
        beta_P_arr[t] = sum_P / spy_var / cnt
        beta_M_arr[t] = sum_M / spy_var / cnt

    df[_COL_N] = beta_N_arr
    df[_COL_P] = beta_P_arr
    df[_COL_M] = beta_M_arr

    # Guard against any stray inf
    for col in (_COL_N, _COL_P, _COL_M):
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
