import pandas as pd
import numpy as np

METADATA = {
    "name":        "volume_flow_metrics",
    "description": "OBV, rolling volume statistics, and weighted velocity metrics",
    "requires":    ["Close", "Volume"],
    "produces": [
        "obv",

        "volume_rolling_28",
        "volume_rolling_90",

        "volume_percent",
        "volume_percent_rolling_90",

        "volume_std",

        "volume_lag_1",

        "weighted_close_change_velocity",
    ],
    "tags": [
        "volume",
        "flow",
        "momentum",
    ],
    "version": "1.0",
    "author": "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:

    close = df["Close"]
    volume = df["Volume"]

    close_shift_1 = close.shift(1)

    df["obv"] = np.where(
        close > close_shift_1,
        volume,
        -volume
    ).cumsum()

    df["volume_rolling_28"] = (
        volume.shift(1)
        .rolling(28)
        .mean()
    )

    df["volume_rolling_90"] = (
        volume.shift(1)
        .rolling(90)
        .mean()
    )

    df["volume_percent"] = (
        (
            volume
            - df["volume_rolling_28"]
        )
        /
        df["volume_rolling_28"]
        * 100.0
    )

    df["volume_percent_rolling_90"] = (
        (
            volume
            - df["volume_rolling_90"]
        )
        /
        df["volume_rolling_90"]
        * 100.0
    )

    df["volume_std"] = (
        volume.shift(1)
        .rolling(28)
        .std()
    )

    df["volume_lag_1"] = volume.shift(1)

    window = 10

    price_change = (
        close.diff()
        .shift(1)
        .fillna(0)
    )

    weights = np.linspace(1, 0, window)
    weights /= weights.sum()

    df["weighted_close_change_velocity"] = (
        price_change
        .rolling(window)
        .apply(
            lambda x: np.dot(x, weights),
            raw=True
        )
    )

    return df