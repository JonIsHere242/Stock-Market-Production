import numpy as np
import pandas as pd

METADATA = {
    "name":        "returns",
    "description": "Log returns over 1, 5, 10, and 20 trading days",
    "requires":    ["Close"],
    # PARITY FIX (audit 2026-06-13): the reference monolith
    # (3__AlphaSensitivity.calculate_price_momentum_features, lines 3096-3098)
    # defines return_5d/10d/20d as Close.pct_change(period) -- NOT log returns --
    # and these are ported faithfully by price_momentum_features.py. This demo
    # block previously also emitted return_5d/10d/20d as LOG returns and, running
    # LAST in topo order, OVERWROTE the faithful pct_change columns (verified: the
    # framework printed "Column 'return_5d' claimed by both ... keeping 'returns'").
    # To stop that corruption, the multi-period LOG returns are now namespaced as
    # logret_* so they no longer collide. return_1d is kept (no reference port
    # produces it, no collision) for momentum_score.py, which is itself a demo block.
    "produces":    ["return_1d", "logret_5d", "logret_10d", "logret_20d"],
    "tags":        ["momentum", "returns"],
    "version":     "1.1",
    "author":      "framework demo",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_close = np.log(df["Close"])
    for period, col in [(1, "return_1d"), (5, "logret_5d"), (10, "logret_10d"), (20, "logret_20d")]:
        df[col] = log_close.diff(period)
    return df
