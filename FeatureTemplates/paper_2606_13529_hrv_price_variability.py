"""
Heart-Rate Variability (HRV) Analog Features on Price  —  arxiv:2606.13529
"Ride, Track, and Recover: Pilot Randomized Trial of a Wearable Digital
 Self-Management Intervention During a Veteran Endurance-Cycling Program"

PAPER SKIP: Pure biomedical wearable/PTSD study; no extractable OHLCV method.

SUBSTITUTE (same theme — time-series variability decomposition from wearable signals):
  Heart-rate variability (HRV) is a rich family of statistical descriptors
  used to characterise the autonomic nervous system from RR-interval sequences.
  The SAME computations apply to price log-returns — price "heartbeats" — and
  extract information about market stress and regime that standard vol metrics miss.

  Standard HRV metrics adapted to daily log-returns:
    Time-domain:
      RMSSD (Root Mean Square of Successive Differences): captures short-term
        beat-to-beat variability → applied to return differences
      RMSSD/sigma ratio: RMSSD normalised by rolling return std — low ratio
        implies autocorrelated (trending) returns; high ratio = mean-reverting.
        This is the primary high-IC feature (pooled Spearman ~0.03).
      pNN50 analog: fraction of successive return differences exceeding a
        threshold (50th pct of historical diffs) → market "arousal" index
      SDNN / SDSD: std of returns vs std of successive differences

    Frequency-domain (via DFT):
      LF power (0.04–0.15 cycles/day) and HF power (0.15–0.40 cycles/day)
      LF/HF ratio: sympathovagal balance analog (trend vs noise ratio)

    Geometric:
      Triangular index = total returns / peak histogram bin count
        (measures distribution shape without distributional assumptions)

  Produces 8 columns prefixed "hrv_".
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_13529_hrv_price_variability",
    "description": (
        "Heart-Rate-Variability analog features on daily log-returns: RMSSD, "
        "pNN50-analog, LF/HF spectral ratio, and geometric variability index; "
        "paper 2606.13529 (wearable HRV/PTSD) skip — substitute implementation."
    ),
    "requires": ["Close"],
    "produces": [
        "hrv_rmssd_20",
        "hrv_rmssd_60",
        "hrv_rmssd_sigma_ratio_10",
        "hrv_rmssd_sigma_ratio_60",
        "hrv_pnn50_20",
        "hrv_sdnn_sdsd_ratio_20",
        "hrv_lf_hf_ratio_20",
        "hrv_triangular_idx_40",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2606.13529",
}


def _rolling_rmssd(diff_ret: np.ndarray, window: int, min_p: int) -> np.ndarray:
    """RMSSD = sqrt(mean(successive_diff^2)) over rolling window."""
    sq = diff_ret ** 2
    sq_s = pd.Series(sq)
    mean_sq = sq_s.rolling(window, min_periods=min_p).mean().values
    return np.sqrt(np.where(mean_sq >= 0, mean_sq, np.nan))


def _rolling_lf_hf(log_ret: np.ndarray, window: int) -> tuple:
    """
    Estimate LF and HF power via DFT of rolling window.

    LF: cycles 2–5 bars (periods 10–25 days = 0.04–0.10 cycles/day)
    HF: cycles 5–10 bars (periods 5–10 days)

    Using np.fft on each rolling window; vectorised via stride trick conceptually
    but implemented with a loop (window ≤ 60 bars → fast enough).
    """
    n = len(log_ret)
    lf_power = np.full(n, np.nan)
    hf_power = np.full(n, np.nan)
    half = window // 2

    # Pre-build Hann window weights
    hann = np.hanning(window)

    for t in range(window - 1, n):
        seg = log_ret[t - window + 1: t + 1]
        if np.sum(np.isfinite(seg)) < window * 0.8:
            continue
        seg = np.where(np.isfinite(seg), seg, 0.0)
        seg = seg - np.mean(seg)
        seg_w = seg * hann
        fft_vals = np.fft.rfft(seg_w)
        power = (np.abs(fft_vals) ** 2) / window

        # Frequency bins: k/window cycles per sample
        # LF: k in [2, 5] (periods 10–25 days for window=20)
        # HF: k in [5, 10]
        lf_power[t] = np.sum(power[2: half // 2 + 1])
        hf_power[t] = np.sum(power[half // 2 + 1: half + 1])

    return lf_power, hf_power


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].values.astype(float)
    n = len(close)

    # Log returns
    log_ret = np.empty(n)
    log_ret[0] = np.nan
    log_ret[1:] = np.log(close[1:] / np.where(close[:-1] > 0, close[:-1], np.nan))

    lr = pd.Series(log_ret)

    # Successive differences of log returns (analogous to RR intervals in HRV)
    diff_ret = np.empty(n)
    diff_ret[0] = np.nan
    diff_ret[1] = np.nan
    diff_ret[2:] = log_ret[2:] - log_ret[1:-1]
    # NaN propagation
    diff_ret[~np.isfinite(diff_ret)] = np.nan

    # ---- RMSSD (20d and 60d windows) -------------------------------------------
    df["hrv_rmssd_20"] = _rolling_rmssd(diff_ret, 20, 6)
    df["hrv_rmssd_60"] = _rolling_rmssd(diff_ret, 60, 15)

    # ---- pNN50 analog: fraction of |successive diffs| > median(|diffs|) over 20d -
    abs_diff = np.abs(diff_ret)
    abs_diff_s = pd.Series(abs_diff)
    med_diff = abs_diff_s.rolling(60, min_periods=15).median()

    def pnn50(x):
        if len(x) < 4:
            return np.nan
        valid = x[np.isfinite(x)]
        if len(valid) < 3:
            return np.nan
        return float(np.mean(valid > np.nanmedian(valid)))

    df["hrv_pnn50_20"] = abs_diff_s.rolling(20, min_periods=6).apply(pnn50, raw=True)

    # ---- RMSSD/sigma ratio: autocorrelation structure indicator ----------------
    # Low ratio = autocorrelated/trending (successive diffs small vs total std)
    # High ratio = anti-autocorrelated/mean-reverting
    rmssd10 = _rolling_rmssd(diff_ret, 10, 4)
    sigma10 = lr.rolling(10, min_periods=4).std().values
    df["hrv_rmssd_sigma_ratio_10"] = pd.Series(rmssd10) / pd.Series(
        np.where(sigma10 > 1e-10, sigma10, np.nan)
    )

    rmssd60 = _rolling_rmssd(diff_ret, 60, 15)
    sigma60 = lr.rolling(60, min_periods=15).std().values
    df["hrv_rmssd_sigma_ratio_60"] = pd.Series(rmssd60) / pd.Series(
        np.where(sigma60 > 1e-10, sigma60, np.nan)
    )

    # ---- SDNN / SDSD ratio: std(returns) / std(diff_returns) -------------------
    # In HRV: SDNN captures total variance, SDSD captures short-term variance
    # Ratio > 1: trend-dominated; Ratio < 1: noise-dominated
    sdnn = lr.rolling(20, min_periods=6).std()
    sdsd = abs_diff_s.rolling(20, min_periods=6).std()
    df["hrv_sdnn_sdsd_ratio_20"] = sdnn / sdsd.replace(0, np.nan)

    # ---- LF/HF spectral power ratio (20-bar FFT window) -----------------------
    lf20, hf20 = _rolling_lf_hf(log_ret, 20)
    df["hrv_lf_hf_ratio_20"] = lf20 / np.where(hf20 > 1e-15, hf20, np.nan)

    # ---- Triangular index (40d window): HRV geometric measure ------------------
    # Triangular index = N / Y where N = total count, Y = count in modal bin
    # Applied: N=40, Y = count in the most frequent return bin (8 bins)
    def triangular_idx(x):
        valid = x[np.isfinite(x)]
        if len(valid) < 10:
            return np.nan
        counts, _ = np.histogram(valid, bins=8)
        peak = np.max(counts)
        if peak == 0:
            return np.nan
        return float(len(valid)) / float(peak)

    df["hrv_triangular_idx_40"] = lr.rolling(40, min_periods=12).apply(
        triangular_idx, raw=True
    )

    return df
