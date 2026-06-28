import pandas as pd

METADATA = {
    "name":        "orig_orig_price_differential_ratio_base",
    "description": "Original AlphaSensitivity Price_Differential_Ratio base column from High/Low.",
    "requires":    ["High", "Low"],
    "produces":    ["Price_Differential_Ratio"],
    "tags":        ["price_structure"],
    "version":     "1.0",
    "author":      "alphasens port",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Backup: add_price_differential_ratio (L1867-1870), called at L3594 with epsilon=1e-10.
    # The function default is 1e-6 but the live pipeline (and thus the ground-truth
    # Data/ProcessedData parquets) used epsilon=1e-10 -- match that for faithful parity.
    epsilon = 1e-10

    df["Price_Differential_Ratio"] = (
        (0.1673 / (df["High"] + epsilon) - df["Low"])
        / (df["High"] + epsilon)
    )

    return df
