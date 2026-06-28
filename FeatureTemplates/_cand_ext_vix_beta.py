from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Load _indexes helper (VIX)
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext_vix_beta",
    "description": (
        "Rolling 120-day OLS slope of daily stock return on daily VIX change "
        "(volatility beta). Negative values indicate vol-sensitive / crash-fragile "
        "stocks that sell off when implied volatility spikes. Produces: raw beta "
        "(ext_vix_beta_slope), its absolute magnitude (ext_vix_beta_abs), and the "
        "20-day change in beta (ext_vix_beta_chg20). Per-ticker proxy; economically "
        "orthogonal to osap_betatailrisk (which uses tail-return conditioning rather "
        "than VIX innovations as the regressor)."
    ),
    "requires": ["Close"],
    "produces": ["ext_vix_beta_slope", "ext_vix_beta_abs", "ext_vix_beta_chg20"],
    "tags": ["volatility", "macro", "vix", "beta", "risk"],
    "version": "1.0.0",
    "author": "Extension/exploration of gate-validated winner (osap_betatailrisk); spec SOURCE: osap_betatailrisk family",
}

# ---------------------------------------------------------------------------
_WINDOW = 120   # OLS window in trading days
_MIN_OBS = 60   # require at least this many valid pairs before outputting


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VIX-innovation beta for one ticker (ascending Date)."""

    # ------------------------------------------------------------------
    # 1. Pull VIX daily close; merge backward so no lookahead
    # ------------------------------------------------------------------
    try:
        vix_df = _indexes.vix_daily_close()   # DataFrame[["Date","vix_close"]]
        vix_df = vix_df.copy()
        vix_df["Date"] = pd.to_datetime(vix_df["Date"])
    except Exception:
        # If index data unavailable, emit NaN columns and return
        df["ext_vix_beta_slope"] = np.nan
        df["ext_vix_beta_abs"] = np.nan
        df["ext_vix_beta_chg20"] = np.nan
        return df

    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = pd.merge_asof(
        work.sort_values("Date"),
        vix_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order (df is passed ascending but we re-sort to be safe)
    work = work.sort_values("Date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Build return (y) and VIX change (x)
    # ------------------------------------------------------------------
    stock_ret = work["Close"].pct_change()          # daily return
    vix_chg = work["vix_close"].diff()              # daily Δ VIX (level points)

    # ------------------------------------------------------------------
    # 3. Rolling OLS slope via the manual formula:
    #    β = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
    # using pandas rolling to stay vectorised and O(n·window).
    # ------------------------------------------------------------------
    n = _WINDOW
    min_p = _MIN_OBS

    # align: both must be finite at same position
    valid_x = np.where(np.isfinite(vix_chg) & np.isfinite(stock_ret), vix_chg, np.nan)
    valid_y = np.where(np.isfinite(vix_chg) & np.isfinite(stock_ret), stock_ret, np.nan)

    x_s = pd.Series(valid_x, index=work.index)
    y_s = pd.Series(valid_y, index=work.index)

    cnt  = x_s.rolling(n, min_periods=min_p).count()
    sum_x  = x_s.rolling(n, min_periods=min_p).sum()
    sum_y  = y_s.rolling(n, min_periods=min_p).sum()
    sum_xy = (x_s * y_s).rolling(n, min_periods=min_p).sum()
    sum_x2 = (x_s * x_s).rolling(n, min_periods=min_p).sum()

    denom = cnt * sum_x2 - sum_x ** 2
    denom = denom.where(denom.abs() > 1e-12, np.nan)   # guard divide-by-zero

    beta = (cnt * sum_xy - sum_x * sum_y) / denom

    # ------------------------------------------------------------------
    # 4. Derived columns
    # ------------------------------------------------------------------
    beta_abs  = beta.abs()
    beta_chg20 = beta - beta.shift(20)

    # ------------------------------------------------------------------
    # 5. Write back into df (same row order as input)
    # ------------------------------------------------------------------
    # work was sorted the same ascending-Date way as df (df is already asc)
    df["ext_vix_beta_slope"] = beta.values
    df["ext_vix_beta_abs"]   = beta_abs.values
    df["ext_vix_beta_chg20"] = beta_chg20.values

    return df
