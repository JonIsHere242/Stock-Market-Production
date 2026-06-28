"""
BallMapper Topological Anomaly + Temporal Fingerprint Features
Paper: "Cross-sectional topological anomaly scores and intraday return predictability
        in the S&P 500: A BallMapper, decoder-conditional VAE, and
        Function-on-Function regression approach"
arXiv: 2606.08586

The paper constructs STOCK-LEVEL topological anomaly scores conditioned on market-level
topology and cross-sectional peer context, then uses function-on-function regression to
show predictive content.  Key findings:
  - Gradual accumulation of return impact over the anomaly history
  - Frequent early reversal of anomaly direction
  - Predictive content distributed across lags (not just lag-1)

OHLCV adaptation:
  We cannot do cross-sectional VAE, but we CAN implement the core IDEAS:
  1. Takens delay embedding (d=3, tau=2) on per-ticker log-returns to build a
     trajectory in embedding space.
  2. BallMapper-inspired ball coverage: fraction of an adaptive epsilon-ball grid
     that the rolling window of embedded trajectories occupies.  High = dispersed
     topology (unusual), low = clustered (familiar).
  3. Temporal anomaly fingerprint: instead of function-on-function regression we
     compute the WEIGHTED SUM of lagged anomaly scores with exponentially decaying
     weights — capturing the "gradual accumulation" finding.
  4. Reversal detection: sign of the short-lag anomaly vs. the medium-lag anomaly,
     which detects the "early reversal" temporal fingerprint.
  5. Reconstruction deviation: local PCA in embedding space — distance from the
     1D principal manifold approximates decoder-VAE reconstruction error.

Produces 7 columns prefixed "bmt_" (BallMapper Topological).
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_08586_ballmapper_topo_anomaly",
    "description": (
        "BallMapper-inspired topological anomaly scores + temporal fingerprint features "
        "from arXiv:2606.08586 — Takens delay embedding on log-returns, rolling ball "
        "coverage (topology dispersal), weighted lag accumulation of anomaly history, "
        "early reversal detector, and local PCA reconstruction deviation."
    ),
    "requires": ["Close", "High", "Low", "Volume"],
    "produces": [
        "bmt_ball_coverage_20",
        "bmt_ball_coverage_60",
        "bmt_anomaly_score",
        "bmt_anomaly_accum",
        "bmt_early_reversal",
        "bmt_pca_recon_dev",
        "bmt_topo_regime",
    ],
    "tags": ["topology", "anomaly", "delay-embedding", "experimental"],
    "version": "1.0",
    "author": "paper:2606.08586",
}


def _delay_embed(series: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """
    Takens delay embedding.
    Returns (n - (dim-1)*tau, dim) array where row i is
    [series[i], series[i-tau], series[i-2*tau], ...].
    Rows that require negative indices are filled with NaN in the output.
    """
    n = len(series)
    start = (dim - 1) * tau
    emb = np.full((n, dim), np.nan)
    for lag_idx in range(dim):
        lag = lag_idx * tau
        if lag == 0:
            emb[start:, lag_idx] = series[start:]
        else:
            emb[start:, lag_idx] = series[start - lag: n - lag]
    return emb


def _ball_coverage(points: np.ndarray, n_bins: int = 6) -> float:
    """
    Fraction of n_bins^d grid cells occupied by `points` (WINDOW, DIM).
    Higher = more topologically dispersed (unusual trajectory).
    """
    if np.any(np.isnan(points)):
        return np.nan
    pmin = points.min(axis=0)
    rng = points.max(axis=0) - pmin
    if np.all(rng == 0):
        return 0.0
    rng = np.where(rng == 0, 1.0, rng)
    normed = (points - pmin) / rng
    bins = np.floor(normed * n_bins).astype(int).clip(0, n_bins - 1)
    # Count unique grid cells by encoding each row as a single mixed-radix
    # integer (base n_bins) then deduping. Identical result to the original
    # len(set(map(tuple, bins.tolist()))) but cheaper for these tiny windows.
    codes = bins[:, 0]
    for j in range(1, bins.shape[1]):
        codes = codes * n_bins + bins[:, j]
    n_unique = len(set(codes.tolist()))
    return float(n_unique) / (n_bins ** points.shape[1])


def _pca_recon_dev(point: np.ndarray, history: np.ndarray) -> float:
    """
    Distance from the leading principal component subspace of `history`.
    Approximates VAE decoder reconstruction error.
    """
    if np.any(np.isnan(point)) or np.any(np.isnan(history)):
        return np.nan
    mu = history.mean(axis=0)
    centered = history - mu
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        pc1 = Vt[0]  # leading PC
        proj = (point - mu) @ pc1
        recon = mu + proj * pc1
        return float(np.linalg.norm(point - recon))
    except np.linalg.LinAlgError:
        return np.nan


def compute(df: pd.DataFrame) -> pd.DataFrame:
    DIM = 3
    TAU = 2
    WIN_SHORT = 20
    WIN_LONG = 60
    START = (DIM - 1) * TAU  # minimum rows needed for embedding

    close = df["Close"].values.astype(float)
    n = len(close)

    # Log returns
    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(close[1:] / np.where(close[:-1] > 0, close[:-1], np.nan))

    # Delay embedding on log returns
    emb = _delay_embed(log_ret, DIM, TAU)  # (n, DIM)

    # Output arrays
    ball_cov_20 = np.full(n, np.nan)
    ball_cov_60 = np.full(n, np.nan)
    anomaly_raw = np.full(n, np.nan)   # Mahalanobis distance in embedding space
    pca_dev = np.full(n, np.nan)

    # Precompute per-row NaN mask of the embedding once (reused per window).
    emb_rownan = np.any(np.isnan(emb), axis=1)
    emb_rowvalid = ~emb_rownan

    for i in range(WIN_LONG + START, n):
        if emb_rownan[i]:
            continue
        pt = emb[i]

        # Skip if too many NaNs
        keep20 = emb_rowvalid[i - WIN_SHORT: i]
        keep60 = emb_rowvalid[i - WIN_LONG: i]
        valid20 = emb[i - WIN_SHORT: i][keep20]
        valid60 = emb[i - WIN_LONG: i][keep60]
        if len(valid20) < 10 or len(valid60) < 20:
            continue

        ball_cov_20[i] = _ball_coverage(valid20)
        ball_cov_60[i] = _ball_coverage(valid60)

        # Mahalanobis anomaly score vs 60d history
        mu60 = valid60.mean(axis=0)
        cov60 = np.cov(valid60.T) + 1e-8 * np.eye(DIM)
        try:
            cov_inv = np.linalg.inv(cov60)
            diff = pt - mu60
            d2 = float(diff @ cov_inv @ diff)
            anomaly_raw[i] = np.sqrt(max(0.0, d2))
        except np.linalg.LinAlgError:
            pass

        # Local-PCA reconstruction deviation (inlined _pca_recon_dev, reusing mu60).
        # valid60 is already NaN-free; mu60 == valid60.mean(axis=0).
        centered = valid60 - mu60
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            pc1 = Vt[0]
            proj = (pt - mu60) @ pc1
            recon = mu60 + proj * pc1
            pca_dev[i] = float(np.linalg.norm(pt - recon))
        except np.linalg.LinAlgError:
            pass

    df["bmt_ball_coverage_20"] = ball_cov_20
    df["bmt_ball_coverage_60"] = ball_cov_60
    df["bmt_anomaly_score"] = anomaly_raw
    df["bmt_pca_recon_dev"] = pca_dev

    # ── Temporal fingerprint: weighted accumulation of anomaly history ──────────
    # Paper finding: predictive content is distributed across recent anomaly history
    # Exp-weighted sum of anomaly scores over lag-1..lag-10
    anom_s = pd.Series(anomaly_raw, index=df.index)
    weights = np.exp(-0.2 * np.arange(1, 11))  # decay factor 0.2
    weights /= weights.sum()
    # Weighted sum = exponential moving average of past anomaly scores (causal)
    accum = np.full(n, np.nan)
    for i in range(WIN_LONG + 10, n):
        lags = anom_s.iloc[i - 10: i].values  # lag 1..10 (past, no lookahead)
        if np.sum(np.isfinite(lags)) >= 6:
            valid_w = np.where(np.isfinite(lags), weights, 0.0)
            valid_w /= max(valid_w.sum(), 1e-12)
            lags_filled = np.where(np.isfinite(lags), lags, 0.0)
            accum[i] = float((lags_filled * valid_w).sum())

    df["bmt_anomaly_accum"] = accum

    # ── Early reversal detector ─────────────────────────────────────────────────
    # Paper finding: frequent early reversal of anomaly direction
    # Proxy: sign agreement between short-lag and medium-lag anomaly changes
    anom_chg_s = anom_s.diff()
    short_sign = np.sign(anom_chg_s.rolling(3, min_periods=2).mean())
    medium_sign = np.sign(anom_chg_s.shift(3).rolling(5, min_periods=3).mean())
    # +1 = continuation, -1 = reversal (early reversal = negative)
    df["bmt_early_reversal"] = (short_sign * medium_sign).astype(float)

    # ── Topological regime: ratio of short/long ball coverage ──────────────────
    # High ratio = trajectory recently MORE dispersed than usual (anomalous topology)
    bc20 = pd.Series(ball_cov_20, index=df.index)
    bc60 = pd.Series(ball_cov_60, index=df.index)
    denom = bc60.rolling(20, min_periods=10).mean().replace(0, np.nan)
    df["bmt_topo_regime"] = bc20 / denom

    return df
