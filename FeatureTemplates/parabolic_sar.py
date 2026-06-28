import pandas as pd
import numpy as np

METADATA = {
    "name":        "parabolic_sar",
    "description": "Parabolic SAR (Stop and Reverse) trend-following indicator",
    "requires":    ["High", "Low", "Close"],
    "produces":    ["parabolic_sar"],
    "tags":        ["trend", "technical"],
    "version":     "1.0",
    "author":      "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Parabolic SAR using the standard recursion:
    - Initial SAR starts at low[0]
    - Acceleration factor (af) starts at 0.02, increments by 0.02 on new highs/lows
    - Maximum af is capped at 0.2
    - SAR is bounded by prior two bars' lows (in uptrend) or highs (in downtrend)
    """

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)

    # Initialize SAR
    sar = low[0]
    ep = high[0]
    af = 0.02
    sar_values = [sar]

    for i in range(1, len(df)):
        sar = sar + af * (ep - sar)
        if close[i] > close[i - 1]:
            af = min(af + 0.02, 0.2)
        else:
            af = 0.02

        if close[i] > close[i - 1]:
            ep = max(high[i], ep)
        else:
            ep = min(low[i], ep)

        sar = min(sar, low[i], low[i - 1]) if close[i] > close[i - 1] else max(sar, high[i], high[i - 1])
        sar_values.append(sar)

    df["parabolic_sar"] = sar_values
    return df
