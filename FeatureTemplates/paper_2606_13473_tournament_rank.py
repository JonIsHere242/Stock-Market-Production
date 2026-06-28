"""
Tournament Rank Features  —  arxiv:2606.13473
"MaxProof: Scaling Mathematical Proof with Generative-Verifier RL and
Population-Level Test-Time Scaling"

The paper uses population-level tournament selection: generate many candidate
proofs, rank them, refine via verifier/ranker, select via tournament. Applied
to OHLCV: treat multiple rolling windows as "candidate models" of the same
underlying signal, run a tournament (rank aggregation) across windows, and
extract the consensus rank score. Stocks that rank consistently high across
many time horizons get high tournament_rank scores.

Method:
  1. For each of K signal primitives × H horizons, compute per-ticker rank
     within its own rolling history using pandas rank-based percentile.
  2. Aggregate ranks via:
     a. Mean rank across all windows (Borda count / consensus)
     b. Min-rank (bottleneck / Condorcet) = worst-case performance
     c. EWM-weighted rank (recency-weighted tournament scoring)
     d. Rank stability: rolling std of Borda rank (low = stable = reliable)
  3. A composite "tournament winner" score.

Uses fully vectorized rolling rank operations — no Python loops over rows.

Produces 8 columns prefixed "trn_".
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_13473_tournament_rank",
    "description": (
        "Tournament rank aggregation features from arxiv:2606.13473 — "
        "multi-window rank consensus using Borda count, Condorcet bottleneck, "
        "and EWM-weighted tournament scoring on OHLCV momentum signals. "
        "Fully vectorized via pandas rolling rank."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "trn_borda_21d",
        "trn_borda_63d",
        "trn_condorcet_21d",
        "trn_condorcet_63d",
        "trn_ewm_rank_21d",
        "trn_ewm_rank_63d",
        "trn_rank_stability",
        "trn_composite",
    ],
    "tags": ["experimental", "momentum", "market_regime"],
    "version": "1.0",
    "author": "paper:2606.13473",
}


def _pct_rank_vectorized(series: pd.Series, window: int) -> pd.Series:
    """
    Percentile rank of current value within rolling window using pandas.
    pandas rolling.rank() is C-level, very fast.
    Returns values in [0, 1].
    """
    mp = max(3, window // 4)
    # rolling().rank(pct=True) returns rank in [1/N, 1] within the window
    return series.rolling(window, min_periods=mp).rank(pct=True)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    C = df["Close"].astype(np.float64)
    V = df["Volume"].astype(np.float64)
    H = df["High"].astype(np.float64)
    L = df["Low"].astype(np.float64)
    O = df["Open"].astype(np.float64)

    # Signal primitives (all causal)
    ret1 = np.log(C / C.shift(1))
    ret5 = np.log(C / C.shift(5))
    ret10 = np.log(C / C.shift(10))
    ret21 = np.log(C / C.shift(21))
    hl = (H - L).replace(0, np.nan)
    body = (C - O) / hl  # bullish body ratio
    vol_chg = np.log(V.replace(0, np.nan)).diff()

    signals = [ret1, ret5, ret10, ret21, body, vol_chg]
    weights = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    weights = weights / weights.sum()

    for w, suffix in [(21, "21d"), (63, "63d")]:
        # Compute percentile rank of each signal within its own rolling window
        rank_cols = []
        for sig in signals:
            r = _pct_rank_vectorized(sig, w)
            rank_cols.append(r)

        rank_df = pd.concat(rank_cols, axis=1)
        rank_df.columns = [f"s{i}" for i in range(len(signals))]

        # Borda count: mean of percentile ranks
        df[f"trn_borda_{suffix}"] = rank_df.mean(axis=1)

        # Condorcet bottleneck: min rank (worst candidate performance)
        df[f"trn_condorcet_{suffix}"] = rank_df.min(axis=1)

        # EWM-weighted rank: more-informative signals (shorter-term) get higher weight
        ewm_rank = (rank_df.to_numpy() * weights[None, :]).sum(axis=1)
        valid = rank_df.notna().any(axis=1)
        df[f"trn_ewm_rank_{suffix}"] = np.where(valid, ewm_rank, np.nan)

    # Rank stability: 1 - rolling std of the 21d Borda score (low std = high stability)
    borda_21 = df["trn_borda_21d"]
    roll_std_borda = borda_21.rolling(21, min_periods=7).std()
    df["trn_rank_stability"] = (1.0 - roll_std_borda.clip(0, 1))

    # Composite: 0.5*borda_21d + 0.3*ewm_21d + 0.2*stability
    borda = df["trn_borda_21d"].fillna(0)
    ewm = df["trn_ewm_rank_21d"].fillna(0)
    stab = df["trn_rank_stability"].fillna(0)
    df["trn_composite"] = 0.5 * borda + 0.3 * ewm + 0.2 * stab
    has_data = df["trn_borda_21d"].notna()
    df["trn_composite"] = np.where(has_data, df["trn_composite"], np.nan)

    return df
