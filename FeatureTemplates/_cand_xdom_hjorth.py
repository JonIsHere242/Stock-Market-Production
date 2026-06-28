"""
Hjorth parameters adapted from EEG signal analysis (Bo Hjorth, 1970) applied
to rolling 60-day windows of daily log returns.

Activity  = var(r)
Mobility  = sqrt(var(dr) / var(r))   where dr = first-difference of r
Complexity = Mobility(dr) / Mobility(r)
           = sqrt(var(ddr)/var(dr)) / sqrt(var(dr)/var(r))

Complexity rises as the return path becomes more erratic / less sinusoidal.
This is a per-ticker time-series implementation of an inherently per-series
signal — no cross-sectional ranking is involved.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "xdom_hjorth",
    "description": (
        "Hjorth parameters (EEG signal analysis, Bo Hjorth 1970) applied to "
        "rolling 60-day windows of daily log returns. "
        "Activity = var(r); Mobility = sqrt(var(dr)/var(r)); "
        "Complexity = Mobility(dr)/Mobility(r), which rises when the return "
        "path becomes more erratic / less smooth. "
        "Per-ticker time-series proxy — no cross-sectional component."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_hjorth_activity_60",
        "xdom_hjorth_mobility_60",
        "xdom_hjorth_complexity_60",
    ],
    "tags": ["cross-domain", "signal-processing", "eeg", "hjorth", "volatility", "complexity"],
    "version": "1.0",
    "author": "Hjorth parameters (EEG signal analysis; Bo Hjorth 1970); cross-domain adaptation",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    window = 60

    # --- log returns ---
    close = df["Close"].replace(0, np.nan)
    r = np.log(close / close.shift(1))          # first-order log return
    dr = r.diff()                                # second-order (first diff of r)
    ddr = dr.diff()                              # third-order (first diff of dr)

    # --- rolling variances (min_periods keeps leading NaNs, avoids false zeros) ---
    var_r   = r.rolling(window, min_periods=window).var()
    var_dr  = dr.rolling(window, min_periods=window).var()
    var_ddr = ddr.rolling(window, min_periods=window).var()

    # --- Activity ---
    activity = var_r.copy()

    # --- Mobility = sqrt(var(dr) / var(r)); guard zero denominator ---
    mob_ratio = var_dr / var_r.replace(0, np.nan)
    mobility = np.where(mob_ratio >= 0, np.sqrt(mob_ratio), np.nan)
    mobility = pd.Series(mobility, index=df.index)

    # --- Complexity = Mobility(dr) / Mobility(r) ---
    # Mobility of dr = sqrt(var(ddr)/var(dr))
    mob_dr_ratio = var_ddr / var_dr.replace(0, np.nan)
    mobility_dr = np.where(mob_dr_ratio >= 0, np.sqrt(mob_dr_ratio), np.nan)
    mobility_dr = pd.Series(mobility_dr, index=df.index)

    mob_safe = mobility.replace(0, np.nan)
    complexity = mobility_dr / mob_safe

    # --- Assign; replace inf with nan ---
    df["xdom_hjorth_activity_60"]   = activity.replace([np.inf, -np.inf], np.nan)
    df["xdom_hjorth_mobility_60"]   = mobility.replace([np.inf, -np.inf], np.nan)
    df["xdom_hjorth_complexity_60"] = complexity.replace([np.inf, -np.inf], np.nan)

    return df
