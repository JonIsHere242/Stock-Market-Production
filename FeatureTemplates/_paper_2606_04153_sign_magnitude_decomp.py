"""
Sign-magnitude return decomposition features derived from:
  "A new decomposition approach to modeling financial returns:
   Conditioning sign on magnitude" (arXiv 2606.04153).

The paper decomposes returns as:
    r_t = sign(r_t) * |r_t|

and models:
  1. Marginal distribution of magnitude (absolute return) — via rolling moments.
  2. Conditional sign distribution given contemporaneous magnitude — the key
     insight is that larger magnitude returns have more predictable sign based
     on prior volatility state.

Per-ticker rolling features:
  - smag_abs_ret_*d : rolling mean / std of |returns| (magnitude model)
  - smag_abs_norm   : current magnitude normalised by rolling std (magnitude surprise)
  - smag_sign_momentum_*d : rolling fraction of positive returns (sign base rate)
  - smag_sign_given_high_mag : P(sign>0 | |r|>median) — sign conditional on large magnitude
  - smag_vol_sign_pred : sign-prediction from magnitude (high vol = negative sign bias?)
  - smag_asymmetry    : difference in up/down magnitude (leverage effect proxy)
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_04153_sign_magnitude_decomp",
    "description": (
        "Rolling sign-magnitude decomposition: conditional sign probability given "
        "contemporaneous |return| magnitude, leveraged-effect asymmetry, and "
        "magnitude surprise; per-ticker proxy for arXiv 2606.04153."
    ),
    "requires":    ["Close"],
    "produces": [
        "smag_abs_ret_mean_20d",
        "smag_abs_ret_std_20d",
        "smag_abs_norm_20d",
        "smag_sign_momentum_20d",
        "smag_sign_given_high_mag_20d",
        "smag_vol_sign_pred_20d",
        "smag_asymmetry_20d",
        "smag_sign_momentum_5d",
        "smag_abs_norm_5d",
    ],
    "tags":        ["momentum", "volatility", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.04153",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["Close"].pct_change()
    abs_ret = ret.abs()
    sign_ret = np.sign(ret)  # -1, 0, +1

    # ------------------------------------------------------------------ #
    # 1. Magnitude model: rolling mean and std of |returns|
    # ------------------------------------------------------------------ #
    abs_mean_20 = abs_ret.rolling(20, min_periods=10).mean()
    abs_std_20  = abs_ret.rolling(20, min_periods=10).std()

    df["smag_abs_ret_mean_20d"] = abs_mean_20
    df["smag_abs_ret_std_20d"]  = abs_std_20

    # Magnitude surprise: current |r| relative to recent volatility
    df["smag_abs_norm_20d"] = (abs_ret / abs_mean_20.replace(0, np.nan)).clip(0, 10)

    # Short-window equivalent
    abs_mean_5 = abs_ret.rolling(5, min_periods=3).mean()
    df["smag_abs_norm_5d"] = (abs_ret / abs_mean_5.replace(0, np.nan)).clip(0, 10)

    # ------------------------------------------------------------------ #
    # 2. Sign model: rolling fraction of positive returns (sign base rate)
    # ------------------------------------------------------------------ #
    pos_flag = (ret > 0).astype(float)
    df["smag_sign_momentum_20d"] = pos_flag.rolling(20, min_periods=10).mean()
    df["smag_sign_momentum_5d"]  = pos_flag.rolling(5,  min_periods=3).mean()

    # ------------------------------------------------------------------ #
    # 3. Conditional sign given high magnitude
    #    P(sign>0 | |r_{t-1..t-20}| > median) — lagged, no lookahead
    #    Strategy: within the trailing 20-day window of PAST returns,
    #    what fraction of high-magnitude days had positive sign?
    # ------------------------------------------------------------------ #
    n = len(df)
    sign_given_high = np.full(n, np.nan)
    abs_arr = abs_ret.shift(1).values  # shift by 1 so we use PAST days only
    pos_arr = pos_flag.shift(1).values

    for i in range(20, n):
        w_abs = abs_arr[max(0, i - 19): i]  # 20 past days (up to t-1)
        w_pos = pos_arr[max(0, i - 19): i]
        mask  = ~(np.isnan(w_abs) | np.isnan(w_pos))
        if mask.sum() < 8:
            continue
        wa = w_abs[mask]
        wp = w_pos[mask]
        med = np.median(wa)
        high_mask = wa > med
        if high_mask.sum() < 3:
            continue
        sign_given_high[i] = wp[high_mask].mean()

    df["smag_sign_given_high_mag_20d"] = sign_given_high

    # ------------------------------------------------------------------ #
    # 4. Volatility → sign prediction
    #    The paper shows volatility level predicts sign probability.
    #    Proxy: rolling logistic-like transformation of volatility rank.
    #    High recent vol → negative sign bias (leverage effect).
    # ------------------------------------------------------------------ #
    # Vol rank in expanding window to avoid lookahead
    vol_rank = abs_std_20.expanding(min_periods=20).rank(pct=True)
    # High vol rank → predict negative (tanh centred at 0.5)
    df["smag_vol_sign_pred_20d"] = (0.5 - vol_rank).clip(-0.5, 0.5)

    # ------------------------------------------------------------------ #
    # 5. Asymmetry (leverage effect): up-magnitude vs down-magnitude
    #    avg(|r| | r>0) - avg(|r| | r<0) over trailing window
    #    Positive → up moves larger than down moves (momentum regime)
    #    Negative → down moves larger than up moves (leverage regime)
    # ------------------------------------------------------------------ #
    up_mag   = abs_ret.where(ret > 0)
    down_mag = abs_ret.where(ret < 0)
    up_avg   = up_mag.rolling(20,   min_periods=5).mean()
    down_avg = down_mag.rolling(20, min_periods=5).mean()
    df["smag_asymmetry_20d"] = (up_avg - down_avg).clip(-0.1, 0.1)

    return df
