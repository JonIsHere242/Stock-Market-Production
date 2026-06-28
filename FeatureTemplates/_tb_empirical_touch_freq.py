"""
_tb_empirical_touch_freq.py  --  CANDIDATE block for the `tb` (stop-aware) ensemble arm.

Reconstructs the EXACT tb label bar-by-bar (next-day Low <= -1.9% stop vs High >= +3.5% target,
measured against the prior close, stop-checked first -- the identical mechanic in
4__Predictorv4_rank.load_and_label_tickers), then SHIFTS by 1 so every trailing statistic uses
only outcomes strictly before the current bar. Emits the trailing stop/target touch FREQUENCIES,
their vol-robust ASYMMETRY (target - stop), and the realized tb mean.

Axis: the empirical base-rate of the tb label itself, conditioned on the intrabar Low/High ordering
that topq/binary_up/ret_5d never see. The asymmetry form is first-order vol-robust (raw freqs
co-move with vol; their difference isolates the wick-down-then-recover geometry).

Ref: Lopez de Prado, "Advances in Financial Machine Learning" (Wiley 2018), Ch.3 triple-barrier.
Candidate -- NOT promoted until it clears the multi-seed (>=4) ablation gate.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name":        "_tb_empirical_touch_freq",
    "description": "Trailing empirical stop/target touch frequencies of the reconstructed tb label "
                   "(-1.9%/+3.5% barriers vs prior close, stop-first), their target-minus-stop "
                   "asymmetry, and realized tb mean at 63/126d. Lopez de Prado 2018 triple-barrier.",
    "requires":    ["High", "Low", "Close"],
    "produces":    ["tbt_stop_freq_63", "tbt_target_freq_63", "tbt_touch_asym_63",
                    "tbt_realized_mean_63",
                    "tbt_stop_freq_126", "tbt_target_freq_126", "tbt_touch_asym_126"],
    "tags":        ["tb", "path", "downside", "candidate"],
    "version":     "0.1",
    "author":      "alt-target-feature-research 2026-06-26 (tb white-space)",
}

_STOP = -0.019
_TGT = 0.035
# inner thresholds to robustly tag the clamped barrier values against float noise
_STOP_IN = -0.0189
_TGT_IN = 0.0349


def compute(df: pd.DataFrame) -> pd.DataFrame:
    prev_close = df["Close"].shift(1)
    lo = df["Low"] / prev_close - 1.0
    hi = df["High"] / prev_close - 1.0
    c2c = df["Close"] / prev_close - 1.0

    # exact label mechanic (stop checked first, as in the predictor); known at close of bar t
    tb_realized = pd.Series(
        np.where(lo <= _STOP, _STOP, np.where(hi >= _TGT, _TGT, c2c)),
        index=df.index,
    )
    tb_realized[prev_close.isna()] = np.nan

    # SHIFT(1): row t only sees outcomes strictly before t
    tbr = tb_realized.shift(1)
    valid = tbr.notna()
    stop_ind = (tbr <= _STOP_IN).astype(float).where(valid)
    tgt_ind = (tbr >= _TGT_IN).astype(float).where(valid)

    for W in (63, 126):
        mp = W // 2
        sf = stop_ind.rolling(W, min_periods=mp).mean()
        tf = tgt_ind.rolling(W, min_periods=mp).mean()
        df[f"tbt_stop_freq_{W}"] = sf
        df[f"tbt_target_freq_{W}"] = tf
        df[f"tbt_touch_asym_{W}"] = tf - sf

    df["tbt_realized_mean_63"] = tbr.rolling(63, min_periods=31).mean()
    return df
