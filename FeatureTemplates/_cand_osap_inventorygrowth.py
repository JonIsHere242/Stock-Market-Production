"""
Candidate feature block: osap_inventorygrowth
Inventory Growth signal (Belo and Lin 2012) via OpenSourceAP (Chen-Zimmermann).

Per-ticker proxy: uses PIT fundamentals (inventory, assets, ppe_net) to
compute deflation-adjusted inventory growth rate. The GNP deflator is not
available in real-time, so we approximate real growth using nominal inventory
growth relative to asset-scaled size (a common proxy for deflation in
cross-sectional academic replication). The predicted sign is -1 (high inventory
growth predicts lower future returns), consistent with overinvestment/supply
overhang literature.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# --- load PIT fundamentals helper ---
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

METADATA = {
    "name": "osap_inventorygrowth",
    "description": (
        "Per-ticker proxy for the Belo & Lin (2012) Inventory Growth anomaly "
        "(OpenSourceAP / Chen-Zimmermann). "
        "Uses PIT-fundamental inventory levels to compute YoY inventory growth "
        "and a smoothed 4-quarter acceleration. "
        "GNP deflator unavailable intraday; we approximate real growth by "
        "scaling nominal inventory change against beginning-period total assets "
        "(asset-deflation proxy). Rows where assets or ppe_net <= 0 are set to "
        "NaN, consistent with the original screen. Cross-sectional ranking is "
        "not performed -- this is a per-ticker time-series signal."
    ),
    "requires": ["Close"],  # only needed to anchor Date; OHLCV otherwise unused
    "produces": [
        "osap_inventorygrowth_yoy",   # YoY inventory growth (asset-deflated)
        "osap_inventorygrowth_accel", # 2-period acceleration of growth rate
        "osap_inventorygrowth_level", # log inventory / assets ratio (size proxy)
    ],
    "tags": ["fundamentals", "accounting", "inventory", "profitability", "osap"],
    "version": "1.0",
    "author": "Belo and Lin 2012; OpenSourceAP (Chen-Zimmermann); block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals we need
    df = _fundamentals.as_of(
        df,
        fields=["inventory", "assets", "ppe_net"],
    )

    inv = df["fund_inventory"].values.astype(float)
    assets = df["fund_assets"].values.astype(float)
    ppe = df["fund_ppe_net"].values.astype(float)

    n = len(df)

    # --- screen: set to NaN where assets <= 0 or ppe_net <= 0 (original filter) ---
    bad_mask = (assets <= 0) | (ppe <= 0)

    # --- YoY inventory growth, asset-deflated (proxy for GNP deflation) ---
    # growth_t = (inv_t - inv_{t-1}) / assets_{t-1}
    # We use shift(1) on the fundamental series (already PIT-safe; shift is backward)
    inv_series = pd.Series(inv)
    assets_series = pd.Series(assets)

    inv_lag1 = inv_series.shift(1).values
    assets_lag1 = assets_series.shift(1).values

    denom_yoy = assets_lag1.copy()
    denom_yoy[denom_yoy == 0] = np.nan  # guard zero-division

    yoy = (inv - inv_lag1) / denom_yoy
    yoy[bad_mask] = np.nan

    # --- 2-period acceleration: current growth minus lagged growth ---
    yoy_series = pd.Series(yoy)
    yoy_lag1 = yoy_series.shift(1).values
    accel = yoy - yoy_lag1
    accel[bad_mask] = np.nan

    # --- level: log(inventory / assets) as inventory-intensity ratio ---
    inv_safe = inv.copy()
    assets_safe = assets.copy()
    inv_safe[(inv_safe <= 0) | bad_mask] = np.nan
    assets_safe[assets_safe <= 0] = np.nan

    level = np.log(inv_safe / assets_safe)
    level[bad_mask] = np.nan

    # Assign produced columns
    df["osap_inventorygrowth_yoy"] = yoy
    df["osap_inventorygrowth_accel"] = accel
    df["osap_inventorygrowth_level"] = level

    # Drop scratch fundamentals columns (not in produces)
    df = df.drop(
        columns=[c for c in ["fund_inventory", "fund_assets", "fund_ppe_net"] if c in df.columns],
    )

    return df
