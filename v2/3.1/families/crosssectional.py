"""Cross-sectional (panel) family — the axis the per-ticker stateless model can't
express. Ranks/z-scores a base feature ACROSS tickers on each date (groupby Date),
which is exactly Polars `over('Date')`. kind='panel' so the engine hands it the
whole panel, not one ticker.
"""
from __future__ import annotations

import pandas as pd

from block import FeatureFamily

_BASE = "roll_mean_w20"  # produced by the rolling family — creates a real DAG edge


def _builder(params: dict):
    op = params["op"]
    col = f"xs_{op}_{_BASE}"

    def compute(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("Date")[_BASE]
        if op == "rank":
            r = g.rank(pct=True)
        else:  # zscore
            r = (df[_BASE] - g.transform("mean")) / (g.transform("std") + 1e-12)
        df = df.copy()
        df[col] = r.to_numpy()
        return df

    return [col], compute


FAMILIES = [
    FeatureFamily(
        name="xs",
        param_grid={"op": ["rank", "zscore"]},
        builder=_builder,
        requires=[_BASE],
        tags=["cross_sectional", "panel"],
        kind="panel",
        description="Cross-sectional rank / z-score of a base feature across the universe per date.",
    )
]
