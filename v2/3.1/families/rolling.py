"""Per-ticker rolling-statistic family.

16 features (4 windows x 4 stats) from ~30 LOC. Scale the grids and this one file
stands in for thousands of hand-written blocks — each still individually named,
profiled, cached, and bit-identical-verifiable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from block import FeatureFamily


def _log_returns(close: np.ndarray) -> np.ndarray:
    r = np.zeros_like(close, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = np.diff(np.log(close))
    r[~np.isfinite(r)] = 0.0
    return r


def _builder(params: dict):
    w = int(params["window"])
    stat = params["stat"]
    col = f"roll_{stat}_w{w}"

    def compute(df: pd.DataFrame) -> pd.DataFrame:
        s = pd.Series(_log_returns(df["Close"].to_numpy(float)))
        if stat == "mean":
            v = s.rolling(w).mean()
        elif stat == "std":
            v = s.rolling(w).std()
        elif stat == "zscore":
            v = (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-12)
        elif stat == "skew":
            v = s.rolling(w).skew()
        else:
            raise ValueError(f"unknown stat {stat!r}")
        df[col] = v.to_numpy()
        return df

    return [col], compute


FAMILIES = [
    FeatureFamily(
        name="roll",
        param_grid={"window": [5, 10, 20, 50], "stat": ["mean", "std", "zscore", "skew"]},
        builder=_builder,
        requires=["Close"],
        tags=["price", "rolling", "per_ticker"],
        kind="per_ticker",
        description="Rolling statistics of log returns over a window grid.",
    )
]
