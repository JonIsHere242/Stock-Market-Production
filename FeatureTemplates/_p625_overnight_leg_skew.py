import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_overnight_leg_skew",
    "description": "Realized skewness split into overnight (gap) and intraday (session) return legs plus their difference; the lottery/skew premium loads on the overnight leg (Amaya, Christoffersen, Jacobs & Vasquez 2015 JFE; Boyer-Mitton-Vorkink 2010; Gu 2025 Accounting & Finance).",
    "requires":    [],
    "produces":    [
        "oski_on_skew_21",
        "oski_on_skew_63",
        "oski_intr_skew_21",
        "oski_intr_skew_63",
        "oski_skew_gap_21",
        "oski_skew_gap_63",
    ],
    "tags":        ["skewness", "overnight", "behavioral", "experimental"],
    "version":     "1.0",
    "author":      "paper:Amaya/Christoffersen/Jacobs/Vasquez (2015) JFE; Boyer-Mitton-Vorkink (2010); Gu (2025) Acc&Fin",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)

    # Overnight (gap) log-return: log(Open_t / Close_{t-1}); guard non-positive prices.
    prev_close = close.shift(1)
    on_ratio = open_ / prev_close.replace(0.0, np.nan)
    on_ratio = on_ratio.where(on_ratio > 0.0, np.nan)
    on = np.log(on_ratio)

    # Intraday (session) log-return: log(Close_t / Open_t), clipped to [-0.5, 0.5].
    intr_ratio = close / open_.replace(0.0, np.nan)
    intr_ratio = intr_ratio.where(intr_ratio > 0.0, np.nan)
    intr = np.log(intr_ratio).clip(-0.5, 0.5)

    # Intraday lagged by one bar (use only past sessions when measuring at row t).
    intr_lag = intr.shift(1)

    windows = {21: 15, 63: 40}
    for w, mp in windows.items():
        on_skew = on.rolling(w, min_periods=mp).skew()
        intr_skew = intr_lag.rolling(w, min_periods=mp).skew()
        df[f"oski_on_skew_{w}"] = on_skew
        df[f"oski_intr_skew_{w}"] = intr_skew
        df[f"oski_skew_gap_{w}"] = (on_skew - intr_skew).clip(-10.0, 10.0)

    return df
