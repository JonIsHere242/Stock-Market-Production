"""
ff0703a_volume_price_volume_price_confirmation_vpci_40  --  Volume-Price
Confirmation Indicator (VPCI) block.
Candidate feature (unproven, prefixed with _cand_).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ff0703a_volume_price_volume_price_confirmation_vpci_40",
    "description": (
        "Volume-Price Confirmation Indicator (VPCI), Buff Dormeier style. "
        "VWMA(10) = sum(close*volume,10)/sum(volume,10); SMA(10) = 10-bar "
        "close SMA. VPC-ratio = VWMA10/SMA10 (does price-weighted-by-volume "
        "diverge from plain price average -- >1 means recent volume is "
        "concentrated on up-days). VM = SMA(volume,10)/SMA(volume,40) "
        "(short vs long volume trend). VPCI-1 = VPC-ratio*VM - 1: positive "
        "means volume is confirming the recent price trend, negative means "
        "volume is drying up or running against price (non-confirmation / "
        "distribution risk). Requires >=40 valid bars else NaN. Produces "
        "the raw level, its 5-day change (confirmation momentum), and the "
        "VPC-ratio component alone (price/volume co-movement without the "
        "volume-trend scaling) as an orthogonal decomposition. Per-ticker, "
        "pure OHLCV, causal (no lookahead)."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "ff0703a_vpci40_main",     # VPCI - 1 (VPC-ratio * VM - 1)
        "ff0703a_vpci40_chg5",     # 5-day change in VPCI (confirmation momentum)
        "ff0703a_vpci40_vpcratio", # VWMA10/SMA10 - 1 (price/volume co-movement only)
    ],
    "tags": ["volume", "price-volume", "confirmation", "vpci", "trend"],
    "version": "1.0",
    "author": (
        "ff0703a batch, faithful per-ticker port of Dormeier VPCI "
        "(VWMA10/SMA10 scaled by SMA(vol,10)/SMA(vol,40))."
    ),
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VPCI-based features per ticker.

    Parameters
    ----------
    df : pd.DataFrame
        Single-ticker OHLCV frame, ascending by Date.

    Returns
    -------
    pd.DataFrame
        Original df with three new columns appended.
    """
    n = len(df)

    # Initialise outputs up front so every code path defines them.
    main = np.full(n, np.nan, dtype=np.float64)
    chg5 = np.full(n, np.nan, dtype=np.float64)
    vpcratio = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        df["ff0703a_vpci40_main"] = main
        df["ff0703a_vpci40_chg5"] = chg5
        df["ff0703a_vpci40_vpcratio"] = vpcratio
        return df

    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    s_close = pd.Series(close, index=df.index, dtype=np.float64)
    s_vol = pd.Series(volume, index=df.index, dtype=np.float64)
    s_cv = s_close * s_vol

    WIN_SHORT = 10
    WIN_LONG = 40
    MIN_VALID = 40  # spec: NaN if < 40 valid days

    sma10 = s_close.rolling(WIN_SHORT, min_periods=WIN_SHORT).mean()
    vol_sum10 = s_vol.rolling(WIN_SHORT, min_periods=WIN_SHORT).sum()
    cv_sum10 = s_cv.rolling(WIN_SHORT, min_periods=WIN_SHORT).sum()
    vol_sma10 = s_vol.rolling(WIN_SHORT, min_periods=WIN_SHORT).mean()
    vol_sma40 = s_vol.rolling(WIN_LONG, min_periods=WIN_LONG).mean()

    vol_sum10_np = vol_sum10.to_numpy()
    sma10_np = sma10.to_numpy()
    cv_sum10_np = cv_sum10.to_numpy()
    vol_sma40_np = vol_sma40.to_numpy()
    vol_sma10_np = vol_sma10.to_numpy()

    # VWMA(10) guarded against zero/near-zero volume sum.
    vwma10 = np.where(
        np.abs(vol_sum10_np) > 1e-9,
        cv_sum10_np / vol_sum10_np,
        np.nan,
    )

    # VPC-ratio = VWMA10 / SMA10, guarded.
    vpc_ratio_raw = np.where(
        np.abs(sma10_np) > 1e-9,
        vwma10 / sma10_np,
        np.nan,
    )

    # VM = SMA(volume,10) / SMA(volume,40), guarded.
    vm_raw = np.where(
        np.abs(vol_sma40_np) > 1e-9,
        vol_sma10_np / vol_sma40_np,
        np.nan,
    )

    vpci_raw = vpc_ratio_raw * vm_raw

    # Enforce the >=40 valid-day minimum (uses the long window's coverage;
    # rolling(min_periods=WIN_LONG) already NaNs bars before 40 valid obs,
    # this is an explicit belt-and-suspenders index-based guard too).
    valid_from = WIN_LONG - 1  # 0-indexed position of the 40th bar
    idx = np.arange(n)
    coverage_mask = idx >= valid_from

    main_np = np.where(coverage_mask, vpci_raw - 1.0, np.nan)
    vpcratio_np = np.where(coverage_mask, vpc_ratio_raw - 1.0, np.nan)

    main = main_np
    vpcratio = vpcratio_np

    s_main = pd.Series(main, index=df.index, dtype=np.float64)
    chg5 = s_main.diff(5).to_numpy()

    df["ff0703a_vpci40_main"] = main
    df["ff0703a_vpci40_chg5"] = chg5
    df["ff0703a_vpci40_vpcratio"] = vpcratio

    return df
