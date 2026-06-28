"""
Teager-Kaiser Energy Operator (TKEO) applied to log-price.

Causal discrete TKEO: psi[n] = x[n-1]^2 - x[n-2]*x[n]
  where x = log(Close).  This uses only past and current data
  (no look-ahead).  Rolling-20d mean of psi estimates sustained
  instantaneous energy (amplitude * freq^2), spiking during
  bursts of high-amplitude high-frequency price motion.

Per-ticker proxy; cross-sectional ranking is NOT applied here.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_teager_kaiser",
    "description": (
        "Causal Teager-Kaiser Energy Operator applied to log-Close.  "
        "psi[n] = x[n-1]^2 - x[n-2]*x[n], x=log(Close).  "
        "Produces a 20-day rolling mean of psi (instantaneous energy level), "
        "a 5-day rolling mean (short-term burst detector), "
        "and a slope (20d vs 5d) capturing acceleration / deceleration of energy. "
        "Per-ticker proxy; cross-sectional rank not applied."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_teager_kaiser_energy_20",
        "xdom_teager_kaiser_energy_5",
        "xdom_teager_kaiser_slope",
    ],
    "tags": ["cross-domain", "signal-processing", "energy", "volatility", "price"],
    "version": "1.0.0",
    "author": "Teager-Kaiser energy operator (Kaiser 1990; speech/mechanical signals)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Log-price series
    log_close = np.log(df["Close"].replace(0, np.nan).values.astype(np.float64))

    n = len(log_close)
    psi = np.full(n, np.nan)

    if n >= 3:
        # Causal TKEO: psi[n] = x[n-1]^2 - x[n-2]*x[n]
        # Valid from index 2 onwards (needs n-2 and n-1).
        x_cur  = log_close[2:]    # x[n]
        x_lag1 = log_close[1:-1]  # x[n-1]
        x_lag2 = log_close[:-2]   # x[n-2]

        raw = x_lag1 ** 2 - x_lag2 * x_cur
        psi[2:] = raw

    psi_series = pd.Series(psi, index=df.index)

    # 20-day rolling mean of instantaneous energy
    energy_20 = psi_series.rolling(20, min_periods=10).mean()

    # 5-day rolling mean (burst detector)
    energy_5 = psi_series.rolling(5, min_periods=3).mean()

    # Slope: short vs long energy (positive = energy accelerating)
    # Guard against divide-by-zero
    denom = energy_20.abs()
    denom = denom.where(denom > 0, np.nan)
    slope = (energy_5 - energy_20) / denom

    df["xdom_teager_kaiser_energy_20"] = energy_20
    df["xdom_teager_kaiser_energy_5"]  = energy_5
    df["xdom_teager_kaiser_slope"]     = slope

    return df
