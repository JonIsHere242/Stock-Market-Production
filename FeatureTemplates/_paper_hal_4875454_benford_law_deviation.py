"""
_paper_hal_4875454_benford_law_deviation.py  --  Benford's Law deviation features.

Paper: "The link between abnormal numbers and price movements ... How does Benford's
law predict stock" (HAL open archive, id: 4875454).

Benford's law states that in many naturally-occurring numerical datasets the leading
(first significant) digit d appears with probability log10(1 + 1/d).  Systematic
deviations from this distribution in OHLCV data may flag abnormal/irregular market
behaviour that precedes price moves.

All features are computed from a TRAILING window of past-only data (no lookahead).
The natural candidate series for Benford analysis is daily Volume (spans multiple
orders of magnitude) together with |daily dollar-move| (Close * |pct-change|).

PROXY NOTE: Per-ticker Benford-conformity proxy only — measures statistical
regularity of price/volume digit distributions within a rolling trailing window.
No cross-sectional normalisation applied here.
"""

import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Benford reference distribution: P(leading digit == d) = log10(1 + 1/d)
# for d in {1, 2, ..., 9}
# ---------------------------------------------------------------------------
_BENFORD_PROB = np.array([np.log10(1.0 + 1.0 / d) for d in range(1, 10)])  # length 9


def _leading_digit(arr: np.ndarray) -> np.ndarray:
    """Return the leading (first significant) digit in {1..9} for each element.

    Uses the absolute value; zeros are mapped to NaN (no valid leading digit).
    Returns a float array with NaN where input <= 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_abs = np.log10(np.where(arr > 0, arr, np.nan))
    floored = np.floor(log_abs)
    out = arr / np.where(np.isfinite(floored), 10.0 ** floored, np.nan)
    digits = np.floor(out).astype(float)
    # Edge case: floating-point log10 rounding can give digit == 10
    digits = np.where(digits == 10.0, 1.0, digits)
    digits = np.where((digits >= 1.0) & (digits <= 9.0), digits, np.nan)
    return digits


# ---------------------------------------------------------------------------
# Benford 2-digit reference: P(N1N2) = log10(1 + 1/(10*N1+N2)) for codes {10..99},
# normalised (matches the old _first_two_digit_distribution).
# ---------------------------------------------------------------------------
_CODES2 = np.arange(10, 100, dtype=np.float64)
_BENFORD_2D = np.array([np.log10(1.0 + 1.0 / c) for c in _CODES2])
_BENFORD_2D = _BENFORD_2D / _BENFORD_2D.sum()


def _two_digit_codes(arr: np.ndarray) -> np.ndarray:
    """Per-element leading TWO-digit code in {10..99} (NaN if undefined), for the whole
    series at once — vectorised form of the extraction the old
    _first_two_digit_distribution did inside each window."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_abs = np.log10(np.where(arr > 0, arr, np.nan))
    floored_one = np.floor(log_abs)
    normalized = arr / np.where(np.isfinite(floored_one), 10.0 ** (floored_one - 1), np.nan)
    two_digit = np.floor(normalized)
    return np.where((two_digit >= 10.0) & (two_digit <= 99.0), two_digit, np.nan)


def _onehot(values: np.ndarray, categories: np.ndarray) -> np.ndarray:
    """(n, K) 0/1 indicator of `values` against `categories`; a NaN value → all-zero row."""
    return (values[:, None] == categories[None, :]).astype(np.int64)


def _roll_counts(onehot: np.ndarray, W: int) -> np.ndarray:
    """Trailing-window category counts: row i = sum of indicator rows [i-W+1 .. i].
    Rows i < W-1 (no full window) are NaN. Exact integer counts via cumsum (no float
    drift) so the downstream stats are bit-identical to summing each window from scratch."""
    out = np.full(onehot.shape, np.nan)
    n = onehot.shape[0]
    if n < W:
        return out
    cs = np.cumsum(onehot, axis=0)
    out[W - 1] = cs[W - 1]
    if n > W:
        out[W:] = cs[W:] - cs[:-W]
    return out


def _benford_stats(counts: np.ndarray, benford: np.ndarray):
    """Vectorised (chi2, MAD, KL) vs Benford for every row of `counts` (n, 9).
    n_valid = row sum; rows with n_valid < 9 (or no full window) → NaN. Same arithmetic
    as the old per-window _benford_chi2 / _benford_mad / _benford_kl."""
    n = counts.shape[0]
    chi2 = np.full(n, np.nan)
    mad = np.full(n, np.nan)
    kl = np.full(n, np.nan)
    n_valid = counts.sum(axis=1)                  # NaN on rows with no full window
    ok = n_valid >= 9                             # NaN >= 9 is False → excluded
    if ok.any():
        cc = counts[ok]
        nv = n_valid[ok][:, None]
        expected = benford[None, :] * nv
        with np.errstate(invalid="ignore"):
            chi2[ok] = np.sum((cc - expected) ** 2 / expected, axis=1)
        emp = cc / nv
        mad[ok] = np.mean(np.abs(emp - benford[None, :]), axis=1)
        empk = cc / nv + 1e-9
        empk = empk / empk.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            klv = np.sum(empk * np.log(empk / benford[None, :]), axis=1)
        kl[ok] = np.where(np.isfinite(klv), klv, np.nan)
    return chi2, mad, kl


def _benford_mad2(counts2: np.ndarray, benford2: np.ndarray) -> np.ndarray:
    """Vectorised 2-digit MAD vs Benford for every row of `counts2` (n, 90).
    Rows with n_valid < 30 → NaN (matches the old len(valid) < 30 guard)."""
    n = counts2.shape[0]
    mad2 = np.full(n, np.nan)
    n_valid = counts2.sum(axis=1)
    ok = n_valid >= 30
    if ok.any():
        cc = counts2[ok]
        nv = n_valid[ok][:, None]
        emp = cc / nv
        mad2[ok] = np.mean(np.abs(emp - benford2[None, :]), axis=1)
    return mad2


# ---------------------------------------------------------------------------
METADATA = {
    "name":        "benford_law_deviation",
    "description": (
        "Per-ticker Benford's Law conformity/deviation metrics over rolling "
        "trailing windows of Volume and |dollar-move|; flags abnormal digit "
        "distributions that may precede price moves (HAL:4875454)."
    ),
    "requires":    ["Close", "Volume"],
    "produces":    [
        "bnf_vol_chi2_60",       # chi-square vs Benford, volume, 60-day window
        "bnf_vol_mad_60",        # MAD conformity, volume, 60-day window
        "bnf_vol_kl_60",         # KL divergence vs Benford, volume, 60-day window
        "bnf_vol_chi2_120",      # chi-square vs Benford, volume, 120-day window
        "bnf_dv_mad_60",         # MAD conformity, |dollar-move|, 60-day window
        "bnf_2digit_mad_60",     # 2-digit MAD vs Benford 2-digit, volume, 60-day
        "bnf_anomaly_flag_60",   # 1 if vol MAD exceeds causal rolling 90th-pct threshold
        "bnf_nonconformity_trend", # slope of rolling bnf_vol_mad_60 over past 20 bars
    ],
    "tags":        ["volume", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "paper HAL:4875454 — Benford's Law deviation feature block",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Benford's Law deviation features from trailing windows of Volume
    and |dollar-move| (Close * |pct_change|).

    Windows:
      - 60-day  (primary)
      - 120-day (secondary, coarser)

    Columns added (all prefixed bnf_):
      bnf_vol_chi2_60        chi-square stat vs Benford (volume, w=60)
      bnf_vol_mad_60         mean-abs-deviation of digit proportions (volume, w=60)
      bnf_vol_kl_60          KL divergence (volume, w=60)
      bnf_vol_chi2_120       chi-square stat vs Benford (volume, w=120)
      bnf_dv_mad_60          MAD for |dollar-move| series (w=60)
      bnf_2digit_mad_60      2-digit Benford MAD (volume, w=60)
      bnf_anomaly_flag_60    1 when bnf_vol_mad_60 exceeds causal rolling 90th-pct
      bnf_nonconformity_trend  OLS slope of bnf_vol_mad_60 over past 20 bars
    """
    n = len(df)

    # -----------------------------------------------------------------------
    # 1. Build source series (absolute values, drop zeros)
    # -----------------------------------------------------------------------
    vol = df["Volume"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)

    # |dollar-move|: Close * |daily_pct_change|
    pct_change = np.empty(n, dtype=np.float64)
    pct_change[0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_change[1:] = np.abs((close[1:] - close[:-1]) / close[:-1])
    dollar_move = close * pct_change   # always >= 0 (NaN on row 0)

    # Replace zeros and negatives with NaN so _leading_digit handles them cleanly
    vol_safe = np.where(vol > 0, vol, np.nan)
    dv_safe = np.where(dollar_move > 0, dollar_move, np.nan)

    # Pre-compute leading digits for the full series (vectorised)
    vol_digits = _leading_digit(vol_safe)        # shape (n,)
    dv_digits = _leading_digit(dv_safe)          # shape (n,)

    # -----------------------------------------------------------------------
    # 2. Rolling window statistics
    # -----------------------------------------------------------------------
    W60 = 60
    W120 = 120
    W_TREND = 20   # window for non-conformity trend

    # A window's digit histogram is just a trailing count per category (digits 1..9),
    # so build one-hot indicators ONCE and roll-count them for all rows via cumsum, then
    # evaluate chi2/MAD/KL per row with the same arithmetic as the old per-window helpers.
    # Counts are exact integers (no float drift) → bit-identical to the loop, no Python loop.
    _cats = np.arange(1, 10, dtype=np.float64)
    vol_oh = _onehot(vol_digits, _cats)                       # (n, 9)
    dv_oh = _onehot(dv_digits, _cats)                          # (n, 9)
    vol_2d_oh = _onehot(_two_digit_codes(vol_safe), _CODES2)   # (n, 90)

    chi2_60, mad_60, kl_60 = _benford_stats(_roll_counts(vol_oh, W60), _BENFORD_PROB)
    chi2_120, _u1, _u2 = _benford_stats(_roll_counts(vol_oh, W120), _BENFORD_PROB)
    _u3, dv_mad_60, _u4 = _benford_stats(_roll_counts(dv_oh, W60), _BENFORD_PROB)
    two_digit_mad_60 = _benford_mad2(_roll_counts(vol_2d_oh, W60), _BENFORD_2D)

    # -----------------------------------------------------------------------
    # 3. Anomaly flag: 1 when mad_60 exceeds causal rolling 90th-percentile
    #    The threshold is computed from past values only (shift by 1 before
    #    rolling so t is excluded from its own threshold calculation).
    # -----------------------------------------------------------------------
    mad_series = pd.Series(mad_60)
    # Use expanding window to accumulate causal history; min 30 observations.
    causal_pct90 = (
        mad_series.shift(1)
        .expanding(min_periods=30)
        .quantile(0.90)
        .to_numpy()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anomaly = np.where(
            np.isfinite(mad_60) & np.isfinite(causal_pct90),
            (mad_60 > causal_pct90).astype(float),
            np.nan,
        )

    # -----------------------------------------------------------------------
    # 4. Non-conformity TREND: OLS slope of mad_60 over the past W_TREND bars.
    #    slope > 0 = non-conformity is rising (potentially more abnormal).
    #    Only uses past-and-present values (no lookahead).
    # -----------------------------------------------------------------------
    # Vectorised OLS slope via rolling covariance / rolling variance of x.
    # x_i = i (index within window), y_i = mad_60 value.
    # slope = (n * sum(x*y) - sum(x)*sum(y)) / (n * sum(x^2) - sum(x)^2)
    # For a fixed window of size W the denominator is constant.
    _W = W_TREND
    _x = np.arange(_W, dtype=np.float64)
    _x_mean = _x.mean()
    _x_var = ((_x - _x_mean) ** 2).sum()   # sum of squared deviations

    nonconformity_trend = np.full(n, np.nan)
    if _x_var > 0:
        for i in range(_W - 1, n):
            window_y = mad_60[i - _W + 1: i + 1]
            if not np.any(np.isfinite(window_y)):
                continue
            # Replace NaN with window mean for slope computation
            valid_mask = np.isfinite(window_y)
            if valid_mask.sum() < _W // 2:
                continue   # too few valid points
            y_filled = np.where(valid_mask, window_y, np.nanmean(window_y))
            y_mean = y_filled.mean()
            cov_xy = ((_x - _x_mean) * (y_filled - y_mean)).sum()
            nonconformity_trend[i] = cov_xy / _x_var

    # -----------------------------------------------------------------------
    # 5. Assign to df (ONLY produced columns; do NOT modify existing columns)
    # -----------------------------------------------------------------------
    idx = df.index
    df["bnf_vol_chi2_60"] = pd.array(chi2_60, dtype="Float64")
    df["bnf_vol_mad_60"] = pd.array(mad_60, dtype="Float64")
    df["bnf_vol_kl_60"] = pd.array(kl_60, dtype="Float64")
    df["bnf_vol_chi2_120"] = pd.array(chi2_120, dtype="Float64")
    df["bnf_dv_mad_60"] = pd.array(dv_mad_60, dtype="Float64")
    df["bnf_2digit_mad_60"] = pd.array(two_digit_mad_60, dtype="Float64")
    df["bnf_anomaly_flag_60"] = pd.array(anomaly, dtype="Float64")
    df["bnf_nonconformity_trend"] = pd.array(nonconformity_trend, dtype="Float64")

    return df
