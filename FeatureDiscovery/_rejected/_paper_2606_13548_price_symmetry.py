"""
Price Symmetry Fingerprint Features
Paper: "Symmetry-electronic fingerprints reveal competing magnetic phases in two-dimensional
        materials" (arXiv 2606.13548)

SKIP REASON: Condensed-matter physics / materials science paper with no OHLCV-computable
method. The SEF (Symmetry-Electronic Fingerprint) encodes crystallographic symmetry
operations + Wyckoff-site geometry + electronic structure for 2D magnetic materials.

SLOT REPLACEMENT (same theme: symmetry fingerprinting + competing phases / uncertainty regions):

The paper's key insight: encode LOCAL SYMMETRY OPERATIONS of a structure into a
"fingerprint" vector; high model uncertainty identifies competing/degenerate phases.
Applied to OHLCV:

  1. **Temporal symmetry fingerprint**: measure the up-down symmetry of price changes
     within a rolling window. A perfectly symmetric (mean-reverting) window has equal
     up/down magnitudes; a trend has asymmetry. Fingerprint components:
       - skewness of returns (asymmetry of distribution)
       - ratio of up-bar count to down-bar count (structural symmetry)
       - reflection symmetry: how close is f(t) to f(window-1-t) in shape?

  2. **Competing phases**: uncertainty / bi-modality in the local distribution,
     analogous to materials with near-degenerate FM/AFM phases. We measure:
       - bimodality coefficient of returns in a window
       - entropy of the sign sequence (uncertainty in directional regime)

  3. **Wyckoff-site equivalent**: the "special positions" in price — detect local
     extrema density (how often local highs/lows appear) as a structural regularity measure.
"""

import pandas as pd
import numpy as np
from scipy.stats import skew

METADATA = {
    "name":        "paper_2606_13548_price_symmetry",
    "description": (
        "Price symmetry fingerprint and competing-phase uncertainty features, "
        "inspired by symmetry-electronic fingerprints in 2D materials (arXiv 2606.13548). "
        "Paper skipped (condensed-matter physics); slot replaced with OHLCV symmetry features."
    ),
    "requires":    ["Open", "High", "Low", "Close"],
    "produces":    [
        # Return distribution asymmetry (skewness of rolling log-returns)
        "psf_return_skew_20d",
        "psf_return_skew_60d",
        # Up/down bar count symmetry ratio: (up - dn) / (up + dn), rolling
        "psf_bar_symmetry_20d",
        "psf_bar_symmetry_60d",
        # Bimodality coefficient: (skew^2 + 1) / (kurtosis + 3*(n-1)^2/((n-2)*(n-3)))
        # High bimodality → competing phases (trending vs mean-reverting coexist)
        "psf_bimodality_coeff_30d",
        # Temporal reflection symmetry: correlation of return sequence with its reversal
        "psf_reflection_symmetry_20d",
        # Local extrema density: fraction of bars that are local H/L extrema (structural regularity)
        "psf_extrema_density_20d",
        # Directional entropy: Shannon entropy of {up,down} sign sequence in rolling window
        "psf_sign_entropy_20d",
        "psf_sign_entropy_60d",
    ],
    "tags":        ["volatility", "market_regime", "mean_reversion", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.13548",
}


def _rolling_skew(series: pd.Series, w: int) -> pd.Series:
    """Vectorised rolling skewness using pandas."""
    return series.rolling(w, min_periods=w // 2).skew()


def _rolling_kurt(series: pd.Series, w: int) -> pd.Series:
    """Vectorised rolling excess kurtosis."""
    return series.rolling(w, min_periods=w // 2).kurt()


def _bimodality_coefficient(skewness: pd.Series, kurtosis: pd.Series,
                             n: int) -> pd.Series:
    """
    Bimodality coefficient (Pfister et al. 2013):
        BC = (skew^2 + 1) / (kurtosis + 3*(n-1)^2 / ((n-2)*(n-3)))
    Values > 5/9 ≈ 0.555 suggest bimodality (competing modes).
    """
    denom_correction = 3.0 * (n - 1) ** 2 / max((n - 2) * (n - 3), 1)
    denom = kurtosis.fillna(0) + denom_correction
    bc = (skewness.fillna(0) ** 2 + 1.0) / denom.clip(lower=1e-8)
    return bc


def _rolling_sign_entropy(sign_series: pd.Series, w: int) -> pd.Series:
    """Shannon entropy of {-1, +1} in rolling window. Max = log(2) ≈ 0.693"""
    def entropy_of(arr):
        n = len(arr)
        if n < 2:
            return np.nan
        p_up = np.mean(arr > 0)
        p_dn = 1.0 - p_up
        if p_up <= 0 or p_dn <= 0:
            return 0.0
        return -(p_up * np.log(p_up) + p_dn * np.log(p_dn))

    return sign_series.rolling(w, min_periods=w // 2).apply(entropy_of, raw=True)


def _rolling_reflection_symmetry(ret: pd.Series, w: int) -> pd.Series:
    """
    Correlation of rolling window sequence with its own time-reversal.
    A mean-reverting oscillation (symmetric) scores near +1.
    A pure trend (asymmetric) scores near -1.
    """
    def reflect_corr(arr):
        if len(arr) < 4:
            return np.nan
        fwd = arr
        rev = arr[::-1]
        if np.std(fwd) < 1e-10 or np.std(rev) < 1e-10:
            return np.nan
        return np.corrcoef(fwd, rev)[0, 1]

    return ret.rolling(w, min_periods=w // 2).apply(reflect_corr, raw=True)


def _local_extrema_density(high: pd.Series, low: pd.Series, w: int) -> pd.Series:
    """
    Fraction of bars in rolling window that are local H or L extrema
    (higher high than both neighbours, or lower low than both neighbours).
    """
    h = high.values
    l = low.values
    n = len(h)

    # Local high: h[i] > h[i-1] and h[i] > h[i+1]
    is_local_h = np.zeros(n, dtype=bool)
    is_local_l = np.zeros(n, dtype=bool)
    is_local_h[1:-1] = (h[1:-1] > h[:-2]) & (h[1:-1] > h[2:])
    is_local_l[1:-1] = (l[1:-1] < l[:-2]) & (l[1:-1] < l[2:])
    is_extremum = (is_local_h | is_local_l).astype(float)

    s = pd.Series(is_extremum, index=high.index)
    return s.rolling(w, min_periods=w // 2).mean()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    ret   = np.log(close / close.shift(1).clip(lower=1e-8))

    # ── Return skewness (asymmetry fingerprint) ────────────────────────────────
    df["psf_return_skew_20d"] = _rolling_skew(ret, 20)
    df["psf_return_skew_60d"] = _rolling_skew(ret, 60)

    # ── Up/down bar symmetry ───────────────────────────────────────────────────
    for w in [20, 60]:
        up_ct = (ret > 0).astype(float).rolling(w, min_periods=w // 2).sum()
        dn_ct = (ret < 0).astype(float).rolling(w, min_periods=w // 2).sum()
        df[f"psf_bar_symmetry_{w}d"] = (up_ct - dn_ct) / (up_ct + dn_ct + 1e-8)

    # ── Bimodality coefficient (30-day window) ─────────────────────────────────
    sk30  = _rolling_skew(ret, 30)
    ku30  = _rolling_kurt(ret, 30)
    df["psf_bimodality_coeff_30d"] = _bimodality_coefficient(sk30, ku30, n=30)

    # ── Temporal reflection symmetry ───────────────────────────────────────────
    df["psf_reflection_symmetry_20d"] = _rolling_reflection_symmetry(ret, 20)

    # ── Local extrema density ─────────────────────────────────────────────────
    df["psf_extrema_density_20d"] = _local_extrema_density(high, low, 20)

    # ── Sign entropy ─────────────────────────────────────────────────────────
    df["psf_sign_entropy_20d"] = _rolling_sign_entropy(ret, 20)
    df["psf_sign_entropy_60d"] = _rolling_sign_entropy(ret, 60)

    return df
