"""
_cand_ext3_semibeta_vs_vix.py
Semibeta vs the VIX: decomposes a stock's covariance with daily VIX changes
into up-VIX and down-VIX components, producing up-semibeta, down-semibeta,
and their difference over a 120-day rolling window.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np
import warnings

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path, as required by sandbox rules)
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext3_semibeta_vs_vix",
    "description": (
        "Semibeta vs the VIX. Over a 120-day rolling window, decomposes the "
        "stock's covariance with daily VIX changes into two regimes: days when "
        "VIX rose (fear spike) and days when VIX fell (fear retreat). "
        "Produces up-VIX semibeta (sensitivity on VIX-spike days), down-VIX "
        "semibeta (sensitivity on VIX-retreat days), and their spread "
        "(down − up), which captures asymmetric tail-risk exposure orthogonal "
        "to standard market beta. Per-ticker proxy; no cross-sectional ranking "
        "is required. Uses _indexes.vix_daily_close() with backward merge to "
        "avoid lookahead."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_semibeta_vs_vix_up",
        "ext3_semibeta_vs_vix_dn",
        "ext3_semibeta_vs_vix_spread",
    ],
    "tags": ["risk", "vix", "semibeta", "tail-risk", "asymmetry"],
    "version": "1.0",
    "author": "Round-4 expansion (osap_betatailrisk); spec ext3_semibeta_vs_vix",
}

# ---------------------------------------------------------------------------
_WINDOW = 120
_MIN_OBS = 30  # minimum observations needed per regime sub-sample


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute up-VIX semibeta, down-VIX semibeta, and their spread.

    Semibeta definition (per Lettau, Maggiori, Weber 2014 approach):
        beta_up  = Cov(r_stock, Δvix | Δvix > 0) / Var(Δvix | Δvix > 0)
        beta_dn  = Cov(r_stock, Δvix | Δvix < 0) / Var(Δvix | Δvix < 0)
        spread   = beta_dn − beta_up

    All computed on a rolling 120-day window, causal (no lookahead).
    """
    n = len(df)
    out_up = np.full(n, np.nan)
    out_dn = np.full(n, np.nan)

    if n < 2:
        df["ext3_semibeta_vs_vix_up"] = np.nan
        df["ext3_semibeta_vs_vix_dn"] = np.nan
        df["ext3_semibeta_vs_vix_spread"] = np.nan
        return df

    # ------------------------------------------------------------------
    # 1. Fetch VIX data and merge asof (backward — no lookahead)
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            vix_df = _indexes.vix_daily_close()
        except Exception:
            vix_df = None

    if vix_df is None or len(vix_df) == 0:
        df["ext3_semibeta_vs_vix_up"] = np.nan
        df["ext3_semibeta_vs_vix_dn"] = np.nan
        df["ext3_semibeta_vs_vix_spread"] = np.nan
        return df

    # Ensure Date columns are datetime for merge_asof
    tmp = df[["Date", "Close"]].copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"])
    vix_df = vix_df.copy()
    vix_df["Date"] = pd.to_datetime(vix_df["Date"])

    merged = pd.merge_asof(
        tmp.sort_values("Date"),
        vix_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order (df is already ascending, but be safe)
    merged = merged.sort_values("Date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Compute daily log returns (stock) and daily VIX change
    # ------------------------------------------------------------------
    close_arr = merged["Close"].values.astype(float)
    vix_arr = merged["vix_close"].values.astype(float)

    # Stock log return: r[t] = log(Close[t] / Close[t-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        stock_ret = np.where(
            close_arr[:-1] > 0,
            np.log(close_arr[1:] / close_arr[:-1]),
            np.nan,
        )
    stock_ret = np.concatenate([[np.nan], stock_ret])

    # VIX change: Δvix[t] = vix[t] - vix[t-1]  (level change, not log)
    dvix = np.concatenate([[np.nan], np.diff(vix_arr)])

    # ------------------------------------------------------------------
    # 3. Rolling semibeta via explicit window loop
    #    (vectorised at the numpy level; outer loop is O(n/1) not O(n^2))
    # ------------------------------------------------------------------
    for i in range(_WINDOW - 1, n):
        r_win = stock_ret[i - _WINDOW + 1 : i + 1]
        d_win = dvix[i - _WINDOW + 1 : i + 1]

        # Drop NaN pairs
        valid = ~(np.isnan(r_win) | np.isnan(d_win))
        r_v = r_win[valid]
        d_v = d_win[valid]

        if len(r_v) < _MIN_OBS:
            continue

        # Split by VIX direction
        up_mask = d_v > 0
        dn_mask = d_v < 0

        # Up semibeta
        if up_mask.sum() >= _MIN_OBS // 3:
            r_up = r_v[up_mask]
            d_up = d_v[up_mask]
            var_up = np.var(d_up, ddof=1)
            if var_up > 0:
                cov_up = np.cov(r_up, d_up, ddof=1)[0, 1]
                out_up[i] = cov_up / var_up

        # Down semibeta
        if dn_mask.sum() >= _MIN_OBS // 3:
            r_dn = r_v[dn_mask]
            d_dn = d_v[dn_mask]
            var_dn = np.var(d_dn, ddof=1)
            if var_dn > 0:
                cov_dn = np.cov(r_dn, d_dn, ddof=1)[0, 1]
                out_dn[i] = cov_dn / var_dn

    # Spread = down - up (positive = hedges better on VIX spikes than retreats)
    with np.errstate(invalid="ignore"):
        out_spread = out_dn - out_up

    # Replace inf with nan (guard)
    out_up = np.where(np.isfinite(out_up), out_up, np.nan)
    out_dn = np.where(np.isfinite(out_dn), out_dn, np.nan)
    out_spread = np.where(np.isfinite(out_spread), out_spread, np.nan)

    # ------------------------------------------------------------------
    # 4. Attach to df (original row order preserved)
    # ------------------------------------------------------------------
    df["ext3_semibeta_vs_vix_up"] = out_up
    df["ext3_semibeta_vs_vix_dn"] = out_dn
    df["ext3_semibeta_vs_vix_spread"] = out_spread

    return df
