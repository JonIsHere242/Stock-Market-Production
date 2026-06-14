"""
Composite momentum score built from multi-period log returns.

Depends on: returns.py  (produces return_1d, return_5d, return_10d, return_20d)
The orchestrator will automatically run returns.py before this block.
"""

import pandas as pd

METADATA = {
    "name":        "momentum_score",
    "description": "Weighted composite of 1/5/10/20d log returns, z-scored over a 63-day window",
    "requires":    ["return_1d", "return_5d", "return_10d", "return_20d"],
    "produces":    ["momentum_score_63d"],
    "tags":        ["momentum"],
    "version":     "1.0",
    "author":      "framework demo — demonstrates cross-block dependency",
}

_WEIGHTS = {"return_1d": 0.10, "return_5d": 0.20, "return_10d": 0.30, "return_20d": 0.40}
_WINDOW = 63


def compute(df: pd.DataFrame) -> pd.DataFrame:
    raw = sum(df[col] * w for col, w in _WEIGHTS.items())
    roll = raw.rolling(_WINDOW, min_periods=_WINDOW // 2)
    df["momentum_score_63d"] = (raw - roll.mean()) / roll.std().replace(0, float("nan"))
    return df
