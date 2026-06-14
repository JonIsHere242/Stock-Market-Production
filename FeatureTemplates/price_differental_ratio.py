import pandas as pd

METADATA = {
    "name":        "price_differential_ratio",
    "description": "Computes Price Differential Ratio from High and Low prices",
    "requires":    ["High", "Low"],
    "produces":    ["price_differential_ratio"],
    "tags":        ["experimental", "price_structure"],
    "version":     "1.0",
    "author":      "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    epsilon = 1e-6

    df["price_differential_ratio"] = (
        (0.1673 / (df["High"] + epsilon) - df["Low"])
        / (df["High"] + epsilon)
    )

    return df