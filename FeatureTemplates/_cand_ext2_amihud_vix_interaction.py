"""
Candidate feature block: ext2_amihud_vix_interaction
Illiquidity premium amplified by volatility regime.

Amihud (2002) daily illiquidity = |ret| / (Close * Volume).
Rolling 60-day mean of that ratio scaled by contemporaneous VIX level
captures the "flight-to-liquidity" amplification: in high-VIX regimes,
illiquid names bear extra pricing risk. A 20-day change column captures
the dynamic shift in this loading.

Per-ticker proxy: faithful to the Amihud*VIX interaction; no cross-sectional
rank is needed since the product already carries both the stock-specific
illiquidity level and the market-wide volatility regime.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path, no package import)
# ---------------------------------------------------------------------------
_spec_idx = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec_idx)
_spec_idx.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext2_amihud_vix_interaction",
    "description": (
        "Daily Amihud illiquidity |ret|/(Close*Volume); rolling 60-day mean "
        "multiplied by the contemporaneous VIX level (backward-merged, "
        "lookahead-safe). High product = illiquid stock in a high-fear "
        "regime (flight-to-liquidity amplification). Also produces the "
        "20-day change of that composite. Per-ticker, no cross-sectional "
        "data needed."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ext2_amihud_vix_interaction_level",   # rolling60 Amihud * VIX
        "ext2_amihud_vix_interaction_chg20",   # 20-day change of level
        "ext2_amihud_vix_interaction_raw",     # daily Amihud (unscaled) for transparency
    ],
    "tags": ["illiquidity", "volatility", "vix", "amihud", "regime", "liquidity"],
    "version": "1.0",
    "author": (
        "Spec: Round-3 deep exploration of osap_betatailrisk winner vein. "
        "Amihud illiquidity: Amihud, Y. (2002) 'Illiquidity and Stock Return', "
        "Journal of Financial Markets."
    ),
}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ext2_amihud_vix_interaction_* columns to df (one stock, ascending Date).
    """
    # --- daily return (log or simple; simple is fine for Amihud) -----------
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)

    # simple return: (C_t - C_{t-1}) / C_{t-1}
    ret = np.empty(len(close), dtype=float)
    ret[0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        denom_ret = close[:-1]
        ret[1:] = np.where(denom_ret != 0, (close[1:] - close[:-1]) / denom_ret, np.nan)

    abs_ret = np.abs(ret)

    # --- Amihud daily illiquidity = |ret| / (Close * Volume) ---------------
    dollar_volume = close * volume   # px * shares
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        amihud_raw = np.where(dollar_volume > 0, abs_ret / dollar_volume, np.nan)

    # Scale to more readable units (multiply by 1e6 to get ratio per $M traded)
    amihud_raw = amihud_raw * 1e6

    df["ext2_amihud_vix_interaction_raw"] = amihud_raw

    # --- rolling 60-day mean Amihud ----------------------------------------
    amihud_s = pd.Series(amihud_raw, index=df.index)
    amihud_60 = amihud_s.rolling(window=60, min_periods=30).mean()

    # --- merge VIX (backward, lookahead-safe) ------------------------------
    try:
        vix_df = _indexes.vix_daily_close()
        # vix_df has columns ["Date", "vix_close"]
        df_tmp = df[["Date"]].copy()
        df_tmp["Date"] = pd.to_datetime(df_tmp["Date"])
        vix_df["Date"] = pd.to_datetime(vix_df["Date"])
        vix_df = vix_df.sort_values("Date").reset_index(drop=True)
        df_tmp = df_tmp.reset_index(drop=True)
        merged = pd.merge_asof(df_tmp, vix_df, on="Date", direction="backward")
        vix_vals = merged["vix_close"].to_numpy(dtype=float)
    except Exception:
        vix_vals = np.full(len(df), np.nan)

    # --- interaction: rolling60 Amihud * VIX ------------------------------
    level = amihud_60.to_numpy(dtype=float) * vix_vals

    # replace inf with nan
    level = np.where(np.isfinite(level), level, np.nan)

    df["ext2_amihud_vix_interaction_level"] = level

    # --- 20-day change of the composite ------------------------------------
    level_s = pd.Series(level, index=df.index)
    chg20 = level_s.diff(20)
    df["ext2_amihud_vix_interaction_chg20"] = chg20.to_numpy(dtype=float)

    return df
