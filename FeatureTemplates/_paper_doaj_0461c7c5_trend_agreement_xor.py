"""
_paper_doaj_0461c7c5_trend_agreement_xor.py

Explicit sign-agreement / XOR-gate interaction features between short-horizon
and long-horizon trend signals.

Motivated by: "Uncovering feature interdependencies in high-noise environments
with stepwise lookahead decision forests" (DOAJ 0461c7c5).  The paper shows
that less-greedy decision trees that consider PAIRS of splits outperform greedy
trees when XOR-LIKE relationships exist between long-term and short-term
technical indicators in low signal-to-noise environments such as financial price
series.  By materialising the non-linear interaction explicitly, standard greedy
trees can exploit it without the lookahead mechanism.

All computation is strictly causal (only uses rows <= t).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name":        "trend_agreement_xor",
    "description": (
        "XOR-gate sign-agreement features between short and long momentum horizons "
        "(5d/10d vs 50d/200d ROC), MA-cross state, RSI-regime alignment, and a "
        "consensus score; materialises the non-linear pair-interaction the paper "
        "identifies as hardest for greedy trees."
    ),
    "requires":    ["Close"],
    "produces":    [
        "xor_mom_agree_5_50",
        "xor_mom_disagree_5_50",
        "xor_ma_cross_state_10_50",
        "xor_agree_persist_20",
        "xor_short_x_longsign",
        "xor_rsi_regime_agree",
        "xor_trend_consensus",
    ],
    "tags":        ["momentum", "trend", "interaction", "experimental"],
    "version":     "1.0",
    "author":      "paper doaj 0461c7c5 — stepwise lookahead decision forests",
}


# ---------------------------------------------------------------------------
# Internal helper: 14-period RSI (Wilder smoothing). Returns a pd.Series
# aligned to df.index.  NOT emitted — scratch use only.
# ---------------------------------------------------------------------------
def _rsi_14(close: pd.Series) -> pd.Series:
    """14-period Wilder RSI.  NaN for the first 13 rows (insufficient history)."""
    delta    = close.diff()
    gain     = delta.clip(lower=0.0)
    loss     = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    # Guard division by zero: avg_loss == 0 and avg_gain > 0 -> RSI = 100
    rs  = avg_gain / avg_loss.replace(0.0, float("nan"))
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # Where avg_loss was exactly 0 and avg_gain was also 0: price flat -> RSI=50
    # Where avg_loss was 0 but gain > 0: RSI should be 100; already NaN from
    # replace above, so patch those cells separately.
    flat_mask = (avg_loss == 0.0) & (avg_gain == 0.0)
    bull_mask = (avg_loss == 0.0) & (avg_gain > 0.0)
    rsi = rsi.where(~flat_mask, 50.0)
    rsi = rsi.where(~bull_mask, 100.0)
    return rsi


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds XOR-gate trend-agreement columns to df (one stock, ascending by Date).

    Internals
    ---------
    mom5    = ROC  5 bars:  (Close / Close.shift(5)  - 1)
    mom10   = ROC 10 bars:  (Close / Close.shift(10) - 1)
    mom50   = ROC 50 bars:  (Close / Close.shift(50) - 1)
    mom200  = ROC 200 bars: (Close / Close.shift(200)- 1)
    ma10    = simple 10-period trailing MA of Close
    ma50    = simple 50-period trailing MA of Close
    rsi14   = 14-period Wilder RSI (scratch, not emitted)

    NaN propagation rule: if either momentum series in a comparison is NaN
    (insufficient history), the agreement flag is left as NaN, never cast to
    a spurious 0 or 1.
    """

    close = df["Close"]

    # ------------------------------------------------------------------ #
    # Momentum (Rate-of-Change)                                           #
    # ------------------------------------------------------------------ #
    # Guard: where the shifted value is 0, result is NaN (not inf).
    def _roc(series: pd.Series, n: int) -> pd.Series:
        prev = series.shift(n)
        safe_prev = prev.where(prev != 0.0, other=float("nan"))
        return series / safe_prev - 1.0

    mom5   = _roc(close, 5)
    mom10  = _roc(close, 10)
    mom50  = _roc(close, 50)
    mom200 = _roc(close, 200)

    # ------------------------------------------------------------------ #
    # Moving averages (trailing, strictly causal)                        #
    # ------------------------------------------------------------------ #
    ma10 = close.rolling(10,  min_periods=10).mean()
    ma50 = close.rolling(50,  min_periods=50).mean()

    # ------------------------------------------------------------------ #
    # RSI (scratch)                                                       #
    # ------------------------------------------------------------------ #
    rsi14 = _rsi_14(close)

    # ------------------------------------------------------------------ #
    # Helper: NaN-aware sign-agreement flag                              #
    #   Returns 1.0 if sign(a)==sign(b) (both non-zero), 0.0 if not,   #
    #   and NaN where either input is NaN or exactly zero (ambiguous).  #
    # ------------------------------------------------------------------ #
    def _agree(a: pd.Series, b: pd.Series) -> pd.Series:
        """1.0 when signs agree, 0.0 when they disagree, NaN when ambiguous."""
        both_valid = a.notna() & b.notna() & (a != 0.0) & (b != 0.0)
        same_sign  = (np.sign(a) == np.sign(b))
        result = pd.Series(float("nan"), index=df.index)
        result = result.where(~both_valid, same_sign.astype(float))
        return result

    # ------------------------------------------------------------------ #
    # Column 1: xor_mom_agree_5_50                                       #
    #   1.0 if sign(mom5) == sign(mom50), else 0.0 (NaN if ambiguous)   #
    # ------------------------------------------------------------------ #
    xor_mom_agree_5_50 = _agree(mom5, mom50)
    df["xor_mom_agree_5_50"] = xor_mom_agree_5_50

    # ------------------------------------------------------------------ #
    # Column 2: xor_mom_disagree_5_50  (the XOR gate itself)            #
    #   1.0 - agree; 1.0 means the two horizons point in opposite dirs   #
    # ------------------------------------------------------------------ #
    df["xor_mom_disagree_5_50"] = 1.0 - xor_mom_agree_5_50

    # ------------------------------------------------------------------ #
    # Column 3: xor_ma_cross_state_10_50                                 #
    #   1.0 if MA10 > MA50 (fast above slow = bullish trend), else 0.0  #
    # ------------------------------------------------------------------ #
    both_ma_valid = ma10.notna() & ma50.notna()
    ma_cross = pd.Series(float("nan"), index=df.index)
    ma_cross = ma_cross.where(~both_ma_valid, (ma10 > ma50).astype(float))
    df["xor_ma_cross_state_10_50"] = ma_cross

    # ------------------------------------------------------------------ #
    # Column 4: xor_agree_persist_20                                     #
    #   Trailing 20-bar mean of xor_mom_agree_5_50 (NaN-ignoring)       #
    # ------------------------------------------------------------------ #
    df["xor_agree_persist_20"] = (
        xor_mom_agree_5_50
        .rolling(20, min_periods=1)
        .mean()
        # Mask positions where the underlying series hasn't started yet
        # (i.e., all 20 inputs were NaN -> rolling mean would itself be NaN)
        # rolling.mean() already returns NaN when all values are NaN, so no
        # extra work needed.  Suppress partial windows in the very early rows
        # where mom5/mom50 haven't warmed up: use min_periods=1 so we get a
        # reasonable estimate as soon as any valid agree value is present.
    )

    # ------------------------------------------------------------------ #
    # Column 5: xor_short_x_longsign                                    #
    #   Signed interaction: mom5 * sign(mom50)                          #
    #   = magnitude of short momentum, signed by the long-term direction #
    #   NaN where mom5 or mom50 is NaN.                                 #
    # ------------------------------------------------------------------ #
    both_mom_valid = mom5.notna() & mom50.notna()
    signed_interact = pd.Series(float("nan"), index=df.index)
    long_sign = np.sign(mom50)
    # Where mom50 == 0.0, sign is 0 -> product is 0, which is valid (no direction)
    product = mom5 * long_sign
    signed_interact = signed_interact.where(~both_mom_valid, product)
    df["xor_short_x_longsign"] = signed_interact

    # ------------------------------------------------------------------ #
    # Column 6: xor_rsi_regime_agree                                    #
    #   1.0 if (RSI14>50) agrees with (mom200>0), else 0.0              #
    #   Both must be non-NaN and mom200 must be non-zero.               #
    # ------------------------------------------------------------------ #
    rsi_valid  = rsi14.notna()
    mom200_valid = mom200.notna() & (mom200 != 0.0)
    both_rsi_valid = rsi_valid & mom200_valid
    rsi_bullish    = rsi14 > 50.0
    mom200_bullish = mom200 > 0.0
    rsi_agree = pd.Series(float("nan"), index=df.index)
    rsi_agree = rsi_agree.where(
        ~both_rsi_valid,
        (rsi_bullish == mom200_bullish).astype(float)
    )
    df["xor_rsi_regime_agree"] = rsi_agree

    # ------------------------------------------------------------------ #
    # Column 7: xor_trend_consensus                                      #
    #   Mean of the binary agreement flags across three pairs:           #
    #   {5v50, 10v50, 10v200}.  0..1 scale; NaN if all three are NaN.  #
    # ------------------------------------------------------------------ #
    agree_5_50  = xor_mom_agree_5_50            # already computed
    agree_10_50 = _agree(mom10, mom50)
    agree_10_200 = _agree(mom10, mom200)

    # Stack and take row-wise nanmean
    stack = pd.concat(
        [agree_5_50, agree_10_50, agree_10_200],
        axis=1,
        keys=["a", "b", "c"],
    )
    # nanmean: at least one valid value required
    row_counts = stack.notna().sum(axis=1)
    row_sums   = stack.sum(axis=1, skipna=True)
    consensus  = pd.Series(float("nan"), index=df.index)
    has_any    = row_counts > 0
    # Guard division
    safe_counts = row_counts.where(row_counts > 0, other=1)
    consensus   = consensus.where(~has_any, row_sums / safe_counts)
    df["xor_trend_consensus"] = consensus

    return df
