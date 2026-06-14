import pandas as pd

METADATA = {
    "name":        "vwap_metrics",
    "description": "Lagged VWAP and VWAP distance metrics",
    "requires":    ["Close", "High", "Low", "Volume"],
    "produces": [
        "vwap_14",
        "vwap_std14",
        "vwap_std20",
        "vwap_percent",
        "vwap_percent_from_high",
    ],
    "tags": [
        "vwap",
        "volume",
        "mean_reversion",
    ],
    "version": "1.0",
    "author": "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    expanding_max = close.expanding().max()

    typical_price = (
        high.shift(1)
        + low.shift(1)
        + close.shift(1)
    ) / 3.0

    volume_shifted = volume.shift(1)

    df["vwap_14"] = (
        (
            typical_price * volume_shifted
        ).rolling(14).sum()
        /
        volume_shifted.rolling(14).sum()
    )

    df["vwap_std14"] = (
        df["vwap_14"]
        .rolling(14)
        .std()
    )

    df["vwap_std20"] = (
        df["vwap_14"]
        .rolling(20)
        .std()
    )

    df["vwap_percent"] = (
        (close - df["vwap_14"])
        /
        df["vwap_14"]
        * 100.0
    )

    df["vwap_percent_from_high"] = (
        (df["vwap_14"] - expanding_max)
        /
        expanding_max
        * 100.0
    )

    return df