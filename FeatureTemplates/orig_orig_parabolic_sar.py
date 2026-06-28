import numpy as np
import pandas as pd

METADATA = {
    "name": "orig_orig_parabolic_sar",
    "description": (
        "Non-standard/simplified Parabolic SAR ported verbatim from AlphaSensitivity "
        "calculate_parabolic_SAR. Stateful iterative loop using Close-vs-prev-Close as the "
        "trend switch (no explicit reversal logic). Init sar=Low[0], ep=High[0], af=0.02."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": ["Parabolic_SAR"],
    "tags": ["price", "trend", "ohlcv", "parabolic_sar", "alphasens_port"],
    "version": "1.0",
    "author": "alphasens port",
}


def compute(df):
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    n = len(df)
    if n == 0:
        df["Parabolic_SAR"] = pd.Series([], dtype=float, index=df.index)
        return df

    sar = low[0]
    ep = high[0]
    af = 0.02
    sar_values = [sar]

    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if close[i] > close[i - 1]:
            af = min(af + 0.02, 0.2)
        else:
            af = 0.02

        if close[i] > close[i - 1]:
            ep = max(high[i], ep)
        else:
            ep = min(low[i], ep)

        sar = (
            min(sar, low[i], low[i - 1])
            if close[i] > close[i - 1]
            else max(sar, high[i], high[i - 1])
        )
        sar_values.append(sar)

    df["Parabolic_SAR"] = sar_values
    return df
