import numpy as np
import pandas as pd

METADATA = {
    "name":        "returns",
    "description": "Log returns over 1, 5, 10, and 20 trading days",
    "requires":    ["Close"],
    "produces":    ["return_1d", "return_5d", "return_10d", "return_20d"],
    "tags":        ["momentum", "returns"],
    "version":     "1.0",
    "author":      "framework demo",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_close = np.log(df["Close"])
    for period, col in [(1, "return_1d"), (5, "return_5d"), (10, "return_10d"), (20, "return_20d")]:
        df[col] = log_close.diff(period)
    return df
