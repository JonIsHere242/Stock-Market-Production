"""
Coskewness feature block — Harvey & Siddique (2000) per-ticker proxy.

Signal: E[r_it_dm * r_mt_dm^2] / (SD[r_it_dm] * SD[r_mt_dm]^2)
where r_it_dm = de-meaned stock return, r_mt_dm = de-meaned market return.

Original method: 60 months of monthly data, NYSE/AMEX CRSP VW index.
Per-ticker proxy: SPY used as market proxy (cross-sectional VW index unavailable
per-stock). Monthly returns constructed from daily closes (month-end sampling).
Rolling 60-month window (aligned to month frequency, then forward-filled to daily).
Predicted cross-sectional sign: -1 (long LOW coskewness — riskier, higher ERet).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import importlib.util as _ilu
from pathlib import Path as _P

# ---------------------------------------------------------------------------
# Load _indexes helper (SPY as market proxy)
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_coskewness",
    "description": (
        "Harvey & Siddique (2000) coskewness: E[r_i_dm * r_m_dm^2] / "
        "(SD[r_i_dm] * SD[r_m_dm]^2) estimated over a rolling 60-month window "
        "using monthly returns. SPY replaces the original NYSE/AMEX CRSP VW index. "
        "Cross-sectional in origin; implemented as a per-ticker time-series proxy. "
        "Predicted sign -1 (low coskewness stocks earn a risk premium). "
        "Produces: level (60-mo), a shorter 24-mo variant, and momentum (change)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_coskewness_60m",   # primary 60-month rolling coskewness
        "osap_coskewness_24m",   # shorter 24-month variant for dynamics
        "osap_coskewness_chg",   # 6-month change in 60m coskewness (momentum of risk)
    ],
    "tags": ["risk", "coskewness", "market", "monthly", "harvey_siddique"],
    "version": "1.0",
    "author": "Harvey and Siddique (2000); OpenSourceAP Chen-Zimmermann; impl by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling coskewness against SPY on monthly return series."""

    # -----------------------------------------------------------------------
    # 1. Fetch SPY close series aligned to this stock's dates
    # -----------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")  # pd.Series, DatetimeIndex
    except Exception:
        spy_close = None

    # Ensure Date is datetime for merging
    date_col = pd.to_datetime(df["Date"])

    # -----------------------------------------------------------------------
    # 2. Build a daily frame with stock close + spy close
    # -----------------------------------------------------------------------
    stock_close = df["Close"].values.copy().astype(float)

    if spy_close is not None and len(spy_close) > 0:
        spy_df = spy_close.rename("spy_close").reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])

        daily = pd.DataFrame({"Date": date_col, "stock_close": stock_close})
        daily = pd.merge_asof(
            daily.sort_values("Date"),
            spy_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        # Restore original order index
        daily.index = df.index
    else:
        daily = pd.DataFrame(
            {"Date": date_col, "stock_close": stock_close, "spy_close": np.nan},
            index=df.index,
        )

    # -----------------------------------------------------------------------
    # 3. Resample to month-end returns
    # -----------------------------------------------------------------------
    # Use month-end close prices, compute pct_change
    daily_sorted = daily.sort_values("Date").copy()
    daily_sorted["ym"] = daily_sorted["Date"].dt.to_period("M")

    # Take last daily close of each calendar month
    monthly = (
        daily_sorted.groupby("ym", sort=True)[["stock_close", "spy_close"]]
        .last()
    )
    monthly["r_stock"] = monthly["stock_close"].pct_change()
    monthly["r_spy"] = monthly["spy_close"].pct_change()

    # -----------------------------------------------------------------------
    # 4. Rolling coskewness computation helper
    # -----------------------------------------------------------------------
    def _rolling_coskew(r_stock: np.ndarray, r_mkt: np.ndarray, window: int) -> np.ndarray:
        """
        Vectorised rolling coskewness via stride view.
        Returns array of same length as input; first (window-1) entries are NaN.
        Formula: mean(r_i_dm * r_m_dm^2) / (std(r_i_dm) * std(r_m_dm)^2)
        """
        n = len(r_stock)
        out = np.full(n, np.nan)

        if n < window:
            return out

        for end in range(window - 1, n):
            start = end - window + 1
            ri = r_stock[start : end + 1]
            rm = r_mkt[start : end + 1]

            # Mask out NaN rows jointly
            valid = np.isfinite(ri) & np.isfinite(rm)
            if valid.sum() < max(12, window // 2):
                continue

            ri_v = ri[valid]
            rm_v = rm[valid]

            ri_dm = ri_v - ri_v.mean()
            rm_dm = rm_v - rm_v.mean()

            numerator = np.mean(ri_dm * rm_dm ** 2)
            denom_i = ri_dm.std(ddof=1)
            denom_m = rm_dm.std(ddof=1)

            denom = denom_i * (denom_m ** 2)
            if denom == 0 or not np.isfinite(denom):
                continue

            out[end] = numerator / denom

        return out

    # -----------------------------------------------------------------------
    # 5. Compute monthly rolling coskewness (60m and 24m)
    # -----------------------------------------------------------------------
    rs = monthly["r_stock"].values.astype(float)
    rm = monthly["r_spy"].values.astype(float)

    coskew_60 = _rolling_coskew(rs, rm, 60)
    coskew_24 = _rolling_coskew(rs, rm, 24)

    monthly_idx = monthly.index  # PeriodIndex by month

    # -----------------------------------------------------------------------
    # 6. 6-month change in 60m coskewness (momentum of risk loading)
    # -----------------------------------------------------------------------
    coskew_60_series = pd.Series(coskew_60, index=monthly_idx)
    coskew_chg = coskew_60_series - coskew_60_series.shift(6)

    # -----------------------------------------------------------------------
    # 7. Map monthly values back to daily rows (forward-fill within month)
    # -----------------------------------------------------------------------
    # Build a daily->monthly period mapping
    daily_sorted2 = daily_sorted[["Date"]].copy()
    daily_sorted2["ym"] = daily_sorted2["Date"].dt.to_period("M")

    monthly_df = pd.DataFrame(
        {
            "ym": monthly_idx,
            "osap_coskewness_60m": coskew_60,
            "osap_coskewness_24m": coskew_24,
            "osap_coskewness_chg": coskew_chg.values,
        }
    )

    daily_merged = daily_sorted2.merge(monthly_df, on="ym", how="left")
    # Restore to original df order
    daily_merged.index = daily_sorted2.index
    daily_merged = daily_merged.reindex(df.index)

    # -----------------------------------------------------------------------
    # 8. Assign produced columns onto df
    # -----------------------------------------------------------------------
    df["osap_coskewness_60m"] = daily_merged["osap_coskewness_60m"].values
    df["osap_coskewness_24m"] = daily_merged["osap_coskewness_24m"].values
    df["osap_coskewness_chg"] = daily_merged["osap_coskewness_chg"].values

    # Safety: replace any inf that leaked through
    for col in METADATA["produces"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
