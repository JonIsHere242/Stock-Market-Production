"""
Rebuild Correlations.parquet at monthly anchor dates using only past data.

Replaces the single static ``Correlations.parquet`` (which uses the full
historical correlation matrix and leaks future return structure into past
trading decisions) with a series of point-in-time files keyed by anchor month.

Output: ``Data/Correlations/correlations_YYYYMM.parquet`` per anchor.
The backtester loads the most recent file with anchor <= current date.

Each output matches the original schema:
    Ticker, Cluster, mean_intragroup_correlation, diff_to_mean_group_corr,
    correlation_0..correlation_K

Run:
    python 0__RollingCorrelations.py --start 2025-04-01 --end 2026-05-01
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from Util import get_logger

logger = get_logger("RollingCorrelations")

PRICE_DIR = Path("Data/PriceData")
OUT_DIR = Path("Data/Correlations")
TRAIL_DAYS = 252
MIN_DAYS_REQUIRED = 200


@njit
def _calc_dynamic_weights(volatility, mean_volatility, std_volatility):
    if std_volatility == 0:
        return np.array([0.25, 0.25, 0.25, 0.25])
    z = (volatility - mean_volatility) / std_volatility
    w = np.array([0.25 - 0.10 * z, 0.25 - 0.05 * z, 0.25 + 0.05 * z, 0.25 + 0.10 * z])
    w = np.maximum(np.minimum(w, 1.0), 0.0)
    return w / np.sum(w)


@njit
def _calc_weighted_returns(returns, vol, mean_vol, std_vol):
    out = np.zeros(len(returns))
    for i in range(21, len(returns)):
        w = _calc_dynamic_weights(vol[i], mean_vol[i], std_vol[i])
        out[i] = np.sum(returns[i] * w)
    return out


def _weighted_return_series(close: pd.Series) -> pd.Series:
    """Replicates 0__AssetCorrolator.process_stock_data's Weighted_Return."""
    close = close.ffill().bfill()
    daily = close.pct_change().fillna(0).round(3)
    weekly = close.pct_change(5).fillna(0).round(3)
    monthly = close.pct_change(21).fillna(0).round(3)
    yearly = close.pct_change(252).fillna(0).round(3)

    vol = daily.rolling(21).std().fillna(0)
    mean_vol = vol.rolling(21).mean().fillna(0)
    std_vol = vol.rolling(21).std().fillna(0)

    returns = np.column_stack([daily.values, weekly.values, monthly.values, yearly.values])
    wr = _calc_weighted_returns(returns, vol.values, mean_vol.values, std_vol.values)
    return pd.Series(wr, index=close.index).round(3)


def load_all_weighted_returns(price_dir: Path = PRICE_DIR) -> pd.DataFrame:
    """Load every ticker's full Weighted_Return series into one wide DataFrame."""
    files = sorted(price_dir.glob("*.parquet"))
    logger.info(f"Loading {len(files)} price files for weighted-return matrix")

    cols = {}
    for fp in tqdm(files, desc="Computing weighted returns"):
        ticker = fp.stem
        try:
            df = pd.read_parquet(fp, columns=["Date", "Close"])
            if df.empty or len(df) < 30:
                continue
            df = df.drop_duplicates("Date").sort_values("Date").set_index("Date")
            cols[ticker] = _weighted_return_series(df["Close"])
        except Exception as e:
            logger.debug(f"Skipping {ticker}: {e}")

    wide = pd.DataFrame(cols)
    wide.index = pd.to_datetime(wide.index)
    logger.info(f"Weighted-return matrix shape: {wide.shape}")
    return wide


def _find_optimal_clusters(scaled, min_k=4, max_k=15, sample_size=2000) -> int:
    """Silhouette-optimal cluster count. Sub-samples to keep silhouette tractable on 4k+ tickers."""
    n = scaled.shape[0]
    max_k = min(max_k, n - 1)
    min_k = max(min_k, 4)

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False) if n > sample_size else np.arange(n)

    best_k, best_score = min_k, -1.0
    for k in range(min_k, max_k + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=0, n_init=5).fit_predict(scaled)
            score = silhouette_score(scaled[sample_idx], labels[sample_idx])
            if score > best_score:
                best_k, best_score = k, score
        except Exception as e:
            logger.debug(f"  k={k} failed: {e}")
    logger.info(f"  optimal k={best_k} (silhouette={best_score:.4f})")
    return best_k


def build_anchor(returns_wide: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame | None:
    """Build the correlation+cluster table for a single anchor date.

    Uses returns strictly before ``anchor`` (no equality), trailing TRAIL_DAYS rows.
    """
    window = returns_wide.loc[returns_wide.index < anchor].tail(TRAIL_DAYS)
    if len(window) < MIN_DAYS_REQUIRED:
        logger.warning(f"  {anchor.date()}: only {len(window)} days available, skipping")
        return None

    valid_cols = window.columns[window.notna().sum() >= MIN_DAYS_REQUIRED]
    sub = window[valid_cols].fillna(0)
    if sub.shape[1] < 50:
        logger.warning(f"  {anchor.date()}: only {sub.shape[1]} valid tickers, skipping")
        return None

    logger.info(f"  {anchor.date()}: window={len(sub)}d, tickers={sub.shape[1]}")
    corr = sub.corr().fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(corr.values)
    k = _find_optimal_clusters(scaled)
    clusters = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(scaled)

    out = pd.DataFrame({"Ticker": corr.index.values, "Cluster": clusters})

    # Per-cluster mean correlation columns (matches original 0__AssetCorrolator schema)
    corr_arr = corr.values
    for c in range(k):
        cluster_idx = np.where(clusters == c)[0]
        if len(cluster_idx) == 0:
            out[f"correlation_{c}"] = 0.0
        else:
            out[f"correlation_{c}"] = corr_arr[:, cluster_idx].mean(axis=1)

    # Mean intra-group correlation per cluster
    intra = {}
    for c in range(k):
        cluster_tickers = out.loc[out["Cluster"] == c, "Ticker"].values
        idx = [list(corr.index).index(t) for t in cluster_tickers]
        intra[c] = corr_arr[np.ix_(idx, idx)].mean() if idx else 0.0
    out["mean_intragroup_correlation"] = out["Cluster"].map(intra)

    # Diff between this ticker's correlation to its own cluster vs cluster's mean
    out["diff_to_mean_group_corr"] = out.apply(
        lambda r: r[f"correlation_{int(r['Cluster'])}"] - r["mean_intragroup_correlation"],
        axis=1,
    )

    # Reorder to match original schema
    base_cols = ["Ticker", "Cluster", "mean_intragroup_correlation", "diff_to_mean_group_corr"]
    corr_cols = [f"correlation_{c}" for c in range(k)]
    out = out[base_cols + corr_cols].round(5)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-04-01", help="First anchor month (YYYY-MM-DD)")
    p.add_argument("--end", default="2026-05-01", help="Last anchor month (YYYY-MM-DD)")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    returns_wide = load_all_weighted_returns()

    # Anchor = first day of each month in [start, end]
    anchors = pd.date_range(args.start, args.end, freq="MS")
    logger.info(f"Building {len(anchors)} anchor months")

    for anchor in anchors:
        out_path = out_dir / f"correlations_{anchor.strftime('%Y%m')}.parquet"
        if out_path.exists():
            logger.info(f"  {anchor.date()}: exists, skipping ({out_path.name})")
            continue
        df = build_anchor(returns_wide, anchor)
        if df is not None:
            df.to_parquet(out_path, index=False)
            logger.info(f"  saved {out_path.name} ({df.shape[0]} tickers, {df['Cluster'].nunique()} clusters)")

    logger.info("Done.")


if __name__ == "__main__":
    main()
