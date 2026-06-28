import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

METADATA = {
    "name": "tda_delay_embedding_anomaly",
    "description": (
        "Per-ticker OHLCV proxy for the cross-sectional topological anomaly scores of Oesterling et al. (2025). "
        "That paper applies Takens delay embedding â†’ BallMapper graph â†’ decoder-conditional VAE across "
        "S&P 500 intraday peer panels, then scores each stock relative to the latent cross-sectional topology. "
        "This proxy: Takens embedding (d=3, Ï„=1) on daily log-returns per stock, then rolling Mahalanobis "
        "anomaly distance from the embedding distribution, local neighbour density in an adaptive Îµ-ball, "
        "isolation score, trajectory curvature (angle between successive step vectors), and a BallMapper-inspired "
        "grid-coverage metric quantifying how densely historical trajectories fill embedding space. "
        "Cross-sectional peer conditioning and VAE scoring are omitted â€” not reproducible per-ticker."
    ),
    "requires": ["Close"],
    "produces": [
        "tda_embed_anomaly_mahal",
        "tda_embed_neighbor_density",
        "tda_embed_isolation_score",
        "tda_embed_trajectory_curl",
        "tda_embed_ball_coverage",
    ],
    "tags": ["topology", "anomaly", "delay-embedding", "tda"],
    "version": "1.0",
    "author": "claude-sonnet-4-6 â€” per-ticker OHLCV proxy; paper requires cross-sectional VAE + BallMapper",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    WINDOW = 60      # rolling history for anomaly scoring
    DIM = 3          # Takens embedding dimension
    EPS_SCALE = 0.5  # neighbour radius = EPS_SCALE Ã— std of pairwise distances to history

    n = len(df)
    close = df["Close"].values.astype(float)

    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(close[1:] / close[:-1])

    # Takens delay embedding: emb[i, lag] = log_ret[i - lag]
    emb = np.full((n, DIM), np.nan)
    for lag in range(DIM):
        src = log_ret[:n - lag] if lag > 0 else log_ret
        emb[lag:, lag] = src

    mahal     = np.full(n, np.nan)
    nbr_den   = np.full(n, np.nan)
    isolation = np.full(n, np.nan)
    curl      = np.full(n, np.nan)
    ball_cov  = np.full(n, np.nan)

    for i in range(WINDOW + DIM, n):
        cur = emb[i]
        if np.any(np.isnan(cur)):
            continue

        hist = emb[i - WINDOW: i]   # (WINDOW, DIM)
        if np.any(np.isnan(hist)):
            continue

        mu   = hist.mean(axis=0)
        diff = cur - mu

        # Mahalanobis distance from rolling embedding distribution
        cov = np.cov(hist.T) + 1e-8 * np.eye(DIM)
        try:
            cov_inv = np.linalg.inv(cov)
            d2 = float(diff @ cov_inv @ diff)
            mahal[i] = np.sqrt(max(0.0, d2))
        except np.linalg.LinAlgError:
            pass

        # Local neighbour density in adaptive Îµ-ball
        dists = cdist(cur.reshape(1, -1), hist)[0]
        std_d = dists.std()
        eps   = EPS_SCALE * std_d if std_d > 0 else 1e-8
        frac  = float((dists < eps).sum()) / WINDOW
        nbr_den[i]   = frac
        isolation[i] = 1.0 - frac

        # Trajectory curvature: 1 âˆ’ cos(angle between consecutive step vectors in embedding space)
        if i >= WINDOW + DIM + 1:
            v1 = emb[i - 1] - emb[i - 2]
            v2 = cur         - emb[i - 1]
            if not (np.any(np.isnan(v1)) or np.any(np.isnan(v2))):
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 0 and n2 > 0:
                    cos_a   = np.clip(v1 @ v2 / (n1 * n2), -1.0, 1.0)
                    curl[i] = float(1.0 - cos_a)   # 0 = straight ahead, 2 = sharp reversal

        # BallMapper proxy: fraction of 5^DIM grid cells occupied by the historical point cloud
        rng = hist.max(axis=0) - hist.min(axis=0)
        if np.all(rng > 0):
            normed   = (hist - hist.min(axis=0)) / rng
            nbins    = 5
            bins     = np.floor(normed * nbins).astype(int).clip(0, nbins - 1)
            n_unique = len(set(map(tuple, bins.tolist())))
            ball_cov[i] = float(n_unique) / (nbins ** DIM)

    df["tda_embed_anomaly_mahal"]    = mahal
    df["tda_embed_neighbor_density"] = nbr_den
    df["tda_embed_isolation_score"]  = isolation
    df["tda_embed_trajectory_curl"]  = curl
    df["tda_embed_ball_coverage"]    = ball_cov

    return df