import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_bipower_jump",
    "description": "Bipower-variation continuous/jump decomposition and Patton-Sheppard relative-jump measure (Barndorff-Nielsen & Shephard 2004 JFEC / 2006; Patton & Sheppard 2015 RESTAT).",
    "requires":    [],
    "produces":    [
        "bpv_cont_share_21",
        "bpv_rel_jump_21",
        "bpv_jump_z_21",
        "bpv_cont_share_63",
        "bpv_rel_jump_63",
        "bpv_jump_z_63",
    ],
    "tags":        ["volatility", "jump", "experimental"],
    "version":     "1.0",
    "author":      "paper:Barndorff-Nielsen & Shephard (2004/2006); Patton & Sheppard (2015) RESTAT",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)

    # Log returns; r[t] uses only Close[t] and Close[t-1] (causal).
    prev_close = close.shift(1)
    ratio = close / prev_close.replace(0.0, np.nan)
    ratio = ratio.where(ratio > 0.0, np.nan)   # log undefined for non-positive ratios
    r = np.log(ratio)

    r2 = r * r                                  # squared returns
    abs_r = r.abs()
    # Adjacent-return absolute product within the window: |r_t| * |r_{t-1}|.
    abs_prod = abs_r * abs_r.shift(1)

    mu1_sq = np.pi / 2.0                         # (E|N(0,1)|)^-2 scaling for BV

    for W, mp in ((21, 15), (63, 40)):
        # Realized variance: trailing sum of squared returns.
        rv = r2.rolling(W, min_periods=mp).sum()
        # Bipower variation: scaled trailing sum of adjacent abs-return products.
        bv = mu1_sq * (W / (W - 1.0)) * abs_prod.rolling(W, min_periods=mp).sum()

        rv_safe = rv.where(rv > 0.0, np.nan)     # guard division by zero/negative

        # Continuous share of variation, bounded to [0, 1].
        cont_share = (bv / rv_safe).clip(lower=0.0, upper=1.0)

        # Jump variation = max(RV - BV, 0); relative jump share bounded to [0, 1].
        jump_var = (rv - bv).clip(lower=0.0)
        rel_jump = (jump_var / rv_safe).clip(lower=0.0, upper=1.0)

        # Jump z-score: jump_var normalized by trailing std of lagged BV (causal).
        bv_std = bv.shift(1).rolling(W, min_periods=mp).std()
        jump_z = jump_var / (bv_std + 1e-12)

        df[f"bpv_cont_share_{W}"] = cont_share
        df[f"bpv_rel_jump_{W}"] = rel_jump
        df[f"bpv_jump_z_{W}"] = jump_z

    return df
