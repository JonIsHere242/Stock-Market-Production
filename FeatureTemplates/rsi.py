import pandas as pd

METADATA = {
    "name":        "rsi",
    "description": "RSI at 14, 28, and 7 periods using Wilder smoothing",
    "requires":    ["Close"],
    "produces":    ["rsi_7", "rsi_14", "rsi_28"],
    "tags":        ["momentum", "technical"],
    "version":     "1.0",
    "author":      "framework demo",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    for period, col in [(7, "rsi_7"), (14, "rsi_14"), (28, "rsi_28")]:
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        df[col] = 100.0 - (100.0 / (1.0 + rs))

    return df
