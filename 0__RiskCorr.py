"""
Asset Correlation & Clustering Pipeline
========================================
v3 — adds fitted sigma^2, detoning, and ASCII-safe logging.

Pipeline:
  1. Load daily returns from Data/PriceData (~600 days per ticker)
  2. Exponentially-weighted correlation matrix (half-life 126d)
  3. Marchenko-Pastur denoising with fitted sigma^2
  4. Detoning — remove the market-wide factor so sector structure dominates
  5. Ward hierarchical clustering on correlation distance
     k chosen by silhouette on factor loadings, capped at MAX_CLUSTERS
  6. Assemble output in same schema as before

Output schema (Correlations.parquet) is unchanged:
  Ticker, Cluster, mean_intragroup_correlation, diff_to_mean_group_corr,
  Marginal_Risk, Diversification_Ratio, Volatility, correlation_0 .. correlation_k
"""

import os
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from Util import get_logger, dprint, Colors

warnings.filterwarnings('ignore')
logger = get_logger("AssetCorrelator")

# ─── Configuration ─────────────────────────────────────────────────────────────
PRICE_DIR       = 'Data/PriceData'
MIN_DAYS        = 252       # require at least 1 trading year
MAX_DAILY_RET   = 0.50      # drop tickers with any >50% single-day return
MAX_ANN_VOL     = 1.00      # drop tickers with annualised vol >100%
WINSOR_SIGMA    = 3.0       # winsorise at +-3 sigma per ticker
EXP_HALFLIFE    = 126       # exponential weighting half-life in days (~6 months)
DATE_COVERAGE   = 0.80      # keep dates where >=80% of tickers are present
TICKER_COVERAGE = 0.90      # keep tickers present on >=90% of kept dates
N_MARKET_MODES  = 1         # number of market-mode eigenvectors to detone
MIN_CLUSTERS    = 5         # floor on cluster count
MAX_CLUSTERS    = 20        # hard cap — never more than this
OUTPUT_FILE     = 'Correlations.parquet'


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load & filter returns
# ─────────────────────────────────────────────────────────────────────────────

def load_and_filter_returns(price_dir=PRICE_DIR):
    """
    Load Close prices from PriceData directory, compute daily returns,
    apply data quality filters, and return a clean (dates x tickers) matrix.

    Returns
    -------
    returns_df : pd.DataFrame, shape (T, N)
    vol_series : pd.Series, ticker -> last 21-day daily std dev
    """
    logger.info(f"Loading price data from '{price_dir}'")
    files = [f for f in os.listdir(price_dir) if f.endswith('.parquet')]
    logger.info(f"Found {len(files)} parquet files")

    close_map = {}
    for f in tqdm(files, desc="Loading prices"):
        ticker = f.split('.')[0]
        try:
            df = pd.read_parquet(os.path.join(price_dir, f))
            if 'Close' not in df.columns or 'Date' not in df.columns:
                continue
            s = df.set_index('Date')['Close'].sort_index()
            s = s[s > 0]
            if len(s) >= MIN_DAYS:
                close_map[ticker] = s
        except Exception as e:
            logger.debug(f"Could not load {ticker}: {e}")

    logger.info(f"{len(close_map)} tickers passed >={MIN_DAYS}-day history filter")

    close_df   = pd.DataFrame(close_map)
    returns_df = close_df.pct_change().iloc[1:]

    # Filter 1: extreme single-day returns
    bad1 = [c for c in returns_df.columns if (returns_df[c].abs() > MAX_DAILY_RET).any()]
    returns_df.drop(columns=bad1, inplace=True)
    logger.info(f"Filter 1 — removed {len(bad1)} tickers with >{MAX_DAILY_RET:.0%} single-day return")

    # Filter 2: high annualised volatility
    last_ann_vol = returns_df.rolling(21).std().iloc[-1] * np.sqrt(252)
    bad2 = last_ann_vol[last_ann_vol > MAX_ANN_VOL].index.tolist()
    returns_df.drop(columns=bad2, inplace=True)
    logger.info(f"Filter 2 — removed {len(bad2)} tickers with >{MAX_ANN_VOL:.0%} annualised vol")

    # Trim to dates with sufficient cross-sectional coverage
    date_cov   = returns_df.notna().mean(axis=1)
    good_dates = date_cov[date_cov >= DATE_COVERAGE].index
    returns_df = returns_df.loc[good_dates]
    logger.info(f"Date trim — kept {len(good_dates)} dates with >={DATE_COVERAGE:.0%} ticker coverage")

    # Filter 3: tickers with insufficient data in the trimmed window
    ticker_cov = returns_df.notna().mean()
    bad3 = ticker_cov[ticker_cov < TICKER_COVERAGE].index.tolist()
    returns_df.drop(columns=bad3, inplace=True)
    logger.info(f"Filter 3 — removed {len(bad3)} tickers with <{TICKER_COVERAGE:.0%} date coverage")

    # Winsorise at +-3 sigma per ticker
    for col in returns_df.columns:
        m, s = returns_df[col].mean(), returns_df[col].std()
        if s > 0:
            returns_df[col] = returns_df[col].clip(lower=m - WINSOR_SIGMA * s,
                                                    upper=m + WINSOR_SIGMA * s)

    returns_df = returns_df.fillna(0)
    logger.info(f"Clean universe: {returns_df.shape[1]} tickers x {returns_df.shape[0]} days")

    vol_series = returns_df.rolling(21).std().iloc[-1]
    return returns_df, vol_series


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Exponentially-weighted correlation matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_weighted_correlation(returns_df, halflife=EXP_HALFLIFE):
    """
    Build an N x N correlation matrix from all available history, with
    exponential decay so recent data gets more weight.

    Returns
    -------
    corr   : np.ndarray (N, N)
    tickers: list[str]
    n_eff  : float  — effective sample size (used by MP denoising)
    """
    tickers = returns_df.columns.tolist()
    T, N    = returns_df.shape
    data    = returns_df.values

    raw_w   = np.exp(-np.arange(T)[::-1] / halflife)
    weights = raw_w / raw_w.sum()
    n_eff   = 1.0 / (weights ** 2).sum()

    logger.info(f"Correlation matrix: {N} assets, T={T} days, "
                f"half-life={halflife}d, n_eff={n_eff:.0f}")

    w_mean  = (data * weights[:, None]).sum(axis=0)
    data_c  = data - w_mean
    sqrt_w  = np.sqrt(weights)[:, None]
    X_w     = data_c * sqrt_w
    cov     = X_w.T @ X_w

    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1.0
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)

    return corr, tickers, n_eff


# ─────────────────────────────────────────────────────────────────────────────
# Step 3a: Fit sigma^2 from the empirical eigenvalue distribution
# ─────────────────────────────────────────────────────────────────────────────

def _mp_pdf(lam, sigma2, q, pts=1000):
    """
    Marchenko-Pastur probability density evaluated at points lam (array).
    sigma2 : noise variance parameter
    q      : T/N ratio
    """
    lam_min = sigma2 * (1.0 - 1.0 / np.sqrt(q)) ** 2
    lam_max = sigma2 * (1.0 + 1.0 / np.sqrt(q)) ** 2
    # pdf is defined only inside [lam_min, lam_max]
    inside = (lam >= lam_min) & (lam <= lam_max)
    pdf    = np.zeros_like(lam, dtype=float)
    x      = lam[inside]
    pdf[inside] = (q / (2.0 * np.pi * sigma2)) * np.sqrt(
        np.clip((lam_max - x) * (x - lam_min), 0, None)
    ) / x
    return pdf


def fit_mp_sigma(eigenvalues, q, sigma2_grid=None):
    """
    Find the sigma^2 that best fits the Marchenko-Pastur distribution to
    the empirical eigenvalue density (minimise SSE between histogrammed
    eigenvalues and the MP pdf).

    Only eigenvalues below the initial lambda+ guess are used for fitting
    (to exclude signal eigenvalues from the fit).

    Returns fitted sigma^2 and the corresponding lambda+ threshold.
    """
    if sigma2_grid is None:
        sigma2_grid = np.arange(0.50, 1.01, 0.01)

    # Use a rough initial threshold to isolate the noise bulk
    lam_guess = (1.0 + 1.0 / np.sqrt(q)) ** 2
    noise_eigs = eigenvalues[eigenvalues <= lam_guess]

    if len(noise_eigs) < 10:
        logger.warning("Too few noise eigenvalues to fit sigma^2 — using sigma^2=1.0")
        return 1.0, lam_guess

    # Empirical density via histogram over the noise bulk
    bins    = min(100, max(20, len(noise_eigs) // 20))
    counts, edges = np.histogram(noise_eigs, bins=bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])

    best_sigma2, best_sse = 1.0, np.inf
    for s2 in sigma2_grid:
        pdf_vals = _mp_pdf(centres, s2, q)
        sse = float(np.sum((counts - pdf_vals) ** 2))
        if sse < best_sse:
            best_sse, best_sigma2 = sse, s2

    lambda_p = best_sigma2 * (1.0 + 1.0 / np.sqrt(q)) ** 2
    logger.info(f"MP fit: best sigma^2={best_sigma2:.3f}, lambda+={lambda_p:.3f} "
                f"(SSE={best_sse:.4f})")
    return best_sigma2, lambda_p


# ─────────────────────────────────────────────────────────────────────────────
# Step 3b: Marchenko-Pastur denoising
# ─────────────────────────────────────────────────────────────────────────────

def denoise_correlation(corr, n_eff):
    """
    Apply Marchenko-Pastur denoising with fitted sigma^2.

    Any eigenvalue above lambda+ = sigma^2 * (1 + 1/sqrt(q))^2 is genuine
    signal.  Noise eigenvalues are replaced with their mean (trace-preserving).

    Returns
    -------
    corr_d   : np.ndarray (N, N)  denoised correlation matrix
    n_signal : int
    lambda_p : float
    """
    N = corr.shape[0]
    q = n_eff / N

    # Eigendecompose first so we can fit sigma^2 from the data
    logger.info(f"Eigendecomposing {N}x{N} correlation matrix (q={q:.4f})...")
    eigenvalues, eigenvectors = np.linalg.eigh(corr)   # ascending order

    # Fit sigma^2 from empirical eigenvalue distribution
    sigma2, lambda_p = fit_mp_sigma(eigenvalues, q)

    # Identify signal vs noise
    signal_mask = eigenvalues > lambda_p
    n_signal    = int(signal_mask.sum())
    noise_vals  = eigenvalues[~signal_mask]
    noise_mean  = noise_vals.mean() if len(noise_vals) > 0 else 1.0

    eigenvalues_d = eigenvalues.copy()
    eigenvalues_d[~signal_mask] = noise_mean

    logger.info(f"Signal eigenvalues: {n_signal}/{N} (above lambda+={lambda_p:.3f})")

    # Reconstruct
    corr_d = eigenvectors @ np.diag(eigenvalues_d) @ eigenvectors.T
    d = np.sqrt(np.diag(corr_d))
    d[d == 0] = 1.0
    corr_d = corr_d / np.outer(d, d)
    np.fill_diagonal(corr_d, 1.0)
    corr_d = np.clip(corr_d, -1, 1)

    return corr_d, n_signal, lambda_p


# ─────────────────────────────────────────────────────────────────────────────
# Step 3c: Detoning — remove the market-wide factor
# ─────────────────────────────────────────────────────────────────────────────

def detone_correlation(corr_d, n_market=N_MARKET_MODES):
    """
    Remove the top n_market eigenvectors ("market modes") from the denoised
    correlation matrix so that sector/industry structure dominates clustering.

    After denoising, the largest eigenvalue corresponds to a nearly uniform
    eigenvector — every stock loads positively on the market.  This creates a
    background correlation floor that makes all assets look similar, drowning
    out sector-level differences.

    Detoning subtracts those contributions:
        C_detoned = C_d - sum_{i=1}^{n_market} lambda_i * v_i @ v_i.T

    then re-normalises the diagonal to 1.

    Returns the detoned correlation matrix (same shape as corr_d).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(corr_d)   # ascending

    # Top n_market eigenvalues/vectors (the market modes)
    market_eigs = eigenvalues[-n_market:]
    market_vecs = eigenvectors[:, -n_market:]

    market_component = np.zeros_like(corr_d)
    for i in range(n_market):
        v = market_vecs[:, i]
        market_component += market_eigs[i] * np.outer(v, v)

    corr_detoned = corr_d - market_component

    # Re-normalise diagonal to 1
    d = np.sqrt(np.diag(corr_detoned))
    d[d == 0] = 1.0
    corr_detoned = corr_detoned / np.outer(d, d)
    np.fill_diagonal(corr_detoned, 1.0)
    corr_detoned = np.clip(corr_detoned, -1, 1)

    logger.info(f"Detoned {n_market} market mode(s), "
                f"largest eigenvalue was {market_eigs[-1]:.2f}")
    return corr_detoned


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Hierarchical clustering
# ─────────────────────────────────────────────────────────────────────────────

def get_factor_loadings(corr_d, n_signal, max_factors=60):
    """
    Extract factor loadings from the top signal eigenvectors of the
    denoised/detoned correlation matrix.  Returns shape (N, k).
    """
    k = min(n_signal, max_factors)
    eigenvalues, eigenvectors = np.linalg.eigh(corr_d)
    top_vals = eigenvalues[-k:]
    top_vecs = eigenvectors[:, -k:]
    # Clip negative eigenvalues that can appear after detoning
    top_vals = np.clip(top_vals, 0, None)
    loadings = top_vecs * np.sqrt(top_vals)
    return loadings


def hierarchical_cluster(corr_d, n_signal,
                          min_k=MIN_CLUSTERS, max_k=MAX_CLUSTERS):
    """
    Ward hierarchical clustering on correlation distance sqrt(0.5*(1-rho)),
    with k chosen by silhouette on factor loadings (capped at max_k).

    Returns
    -------
    labels : np.ndarray (N,)  0-indexed cluster assignments
    k      : int              chosen cluster count
    """
    N = corr_d.shape[0]

    dist_mat = np.sqrt(np.clip(0.5 * (1.0 - corr_d), 0.0, 1.0))
    np.fill_diagonal(dist_mat, 0.0)
    dist_condensed = squareform(dist_mat, checks=False)

    logger.info("Building Ward hierarchical linkage...")
    Z = linkage(dist_condensed, method='ward')

    loadings = get_factor_loadings(corr_d, n_signal)

    logger.info(f"Silhouette search k={min_k}..{max_k} on factor loadings "
                f"(shape {loadings.shape})")
    best_k, best_sil = min_k, -1.0

    for k in tqdm(range(min_k, max_k + 1), desc="Silhouette search"):
        labels_k = fcluster(Z, k, criterion='maxclust') - 1
        sample   = min(3000, N)
        try:
            sil = silhouette_score(loadings, labels_k,
                                   metric='euclidean',
                                   sample_size=sample,
                                   random_state=42)
            logger.debug(f"  k={k:2d}  silhouette={sil:.4f}")
            if sil > best_sil:
                best_sil, best_k = sil, k
        except Exception as e:
            logger.debug(f"  k={k:2d}  silhouette failed: {e}")

    logger.info(f"Chosen k={best_k} (silhouette={best_sil:.4f})")
    labels = fcluster(Z, best_k, criterion='maxclust') - 1
    return labels, best_k


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Output metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_group_correlations(corr_d, tickers, cluster_labels):
    """
    For each asset compute its mean correlation with every cluster.
    Uses the denoised/detoned correlation matrix.
    """
    unique_clusters = sorted(set(cluster_labels))
    result = pd.DataFrame(
        index=tickers,
        columns=[f'correlation_{c}' for c in unique_clusters],
        dtype=float
    )
    for c in tqdm(unique_clusters, desc="Group correlations"):
        idx = np.where(cluster_labels == c)[0]
        result[f'correlation_{c}'] = corr_d[:, idx].mean(axis=1)
    return result


def calculate_risk_metrics(corr_d, vol_series, tickers, cluster_labels):
    """Per-asset risk decomposition within each cluster (equal-weight)."""
    risk_df = pd.DataFrame(index=tickers)
    risk_df['Volatility'] = vol_series.reindex(tickers).fillna(vol_series.mean())

    for c in sorted(set(cluster_labels)):
        idx = np.where(cluster_labels == c)[0]
        if len(idx) < 2:
            continue
        c_tickers = [tickers[i] for i in idx]
        c_corr    = corr_d[np.ix_(idx, idx)]
        c_vols    = risk_df.loc[c_tickers, 'Volatility'].values
        w         = np.ones(len(idx)) / len(idx)
        cov       = c_corr * np.outer(c_vols, c_vols)
        port_vol  = np.sqrt(float(w @ cov @ w))
        if port_vol > 0:
            for i, t in enumerate(c_tickers):
                risk_df.loc[t, 'Marginal_Risk']        = (cov @ w)[i] / port_vol
                risk_df.loc[t, 'Diversification_Ratio'] = c_vols[i] / port_vol

    return risk_df.fillna(0)


def assemble_output(tickers, cluster_labels, group_corr_df, risk_df):
    """
    Build the final output DataFrame with the same schema as before so that
    the EDA notebook and downstream consumers need no changes.
    """
    clustered = pd.DataFrame({'Ticker': tickers, 'Cluster': cluster_labels})
    clustered = clustered.merge(group_corr_df, left_on='Ticker', right_index=True)

    corr_cols = list(group_corr_df.columns)

    # mean_intragroup_correlation: cluster-level mean of own-cluster correlation
    for c in sorted(set(cluster_labels)):
        own_col = f'correlation_{c}'
        mask    = clustered['Cluster'] == c
        clustered.loc[mask, 'mean_intragroup_correlation'] = \
            clustered.loc[mask, own_col].mean()

    # diff_to_mean_group_corr: how far this asset sits from its cluster average
    clustered['diff_to_mean_group_corr'] = clustered.apply(
        lambda row: row[f'correlation_{int(row.Cluster)}']
                    - row['mean_intragroup_correlation'],
        axis=1
    )

    clustered = clustered.merge(risk_df, left_on='Ticker', right_index=True, how='left')

    base_cols = ['Ticker', 'Cluster',
                 'mean_intragroup_correlation', 'diff_to_mean_group_corr']
    risk_cols = [c for c in ['Marginal_Risk', 'Diversification_Ratio', 'Volatility']
                 if c in clustered.columns]

    clustered = clustered[base_cols + risk_cols + corr_cols].round(5).fillna(0)
    return clustered


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def process_data_directory(price_dir=PRICE_DIR):
    """Run the full correlation + clustering pipeline."""
    logger.info("=" * 60)
    logger.info("Asset Correlation Pipeline  (v3 — MP denoised + detoned)")
    logger.info("=" * 60)

    returns_df, vol_series = load_and_filter_returns(price_dir)

    corr_raw, tickers, n_eff = compute_weighted_correlation(returns_df)

    # Denoise: remove random eigenvalues using fitted Marchenko-Pastur threshold
    corr_d, n_signal, _ = denoise_correlation(corr_raw, n_eff)

    # Detone: remove the market-wide factor so sector structure dominates
    corr_d = detone_correlation(corr_d, n_market=N_MARKET_MODES)

    cluster_labels, k = hierarchical_cluster(corr_d, n_signal)

    group_corr_df = calculate_group_correlations(corr_d, tickers, cluster_labels)
    risk_df       = calculate_risk_metrics(corr_d, vol_series, tickers, cluster_labels)
    clustered     = assemble_output(tickers, cluster_labels, group_corr_df, risk_df)

    clustered.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"Saved {len(clustered)} assets in {k} clusters -> '{OUTPUT_FILE}'")

    dprint(f"Cluster summary ({k} clusters):", level="INFO")
    for c in sorted(set(cluster_labels)):
        sub = clustered[clustered['Cluster'] == c]
        dprint(f"  Cluster {c:2d}: {len(sub):4d} assets  "
               f"mean_intra_corr={sub['mean_intragroup_correlation'].mean():.3f}  "
               f"avg_ann_vol={sub['Volatility'].mean() * np.sqrt(252):.1%}",
               level="DETAIL", indent=2)

    return clustered


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Asset Correlation — MP denoised + detoned, Ward hierarchical'
    )
    parser.add_argument('--price-dir',    type=str, default=PRICE_DIR)
    parser.add_argument('--min-clusters', type=int, default=MIN_CLUSTERS)
    parser.add_argument('--max-clusters', type=int, default=MAX_CLUSTERS)
    parser.add_argument('--halflife',     type=int, default=EXP_HALFLIFE)
    parser.add_argument('--n-market',     type=int, default=N_MARKET_MODES,
                        help='Number of market-mode eigenvectors to detone (default 1)')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        logger = get_logger("AssetCorrelator", debug=True)

    MIN_CLUSTERS   = args.min_clusters
    MAX_CLUSTERS   = args.max_clusters
    EXP_HALFLIFE   = args.halflife
    N_MARKET_MODES = args.n_market

    try:
        process_data_directory(args.price_dir)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
