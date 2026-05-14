#!/usr/bin/env python3
"""
3__ExperimentalFeatures.py
==========================
Experimental feature builder + tester  (v2 — iterated from first-pass results).

Changes from v1:
  - Dropped 18 chaff features (AUC ~0.5 / MI ~0 / highly correlated duplicates)
  - Kept 15 base survivors
  - Added TIER-2 enhanced versions (z-score, volatility-normalised, smoothed)
  - Added TIER-3 combination features (cross-feature interactions)

Usage:
    python 3__ExperimentalFeatures.py                      # 200 random tickers
    python 3__ExperimentalFeatures.py --n_tickers 500
    python 3__ExperimentalFeatures.py --no_plots
    python 3__ExperimentalFeatures.py --ticker AAPL MSFT
"""

import os
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pointbiserialr
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm

warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["figure.dpi"] = 100
sns.set_style("whitegrid")

PRICE_DIR   = "Data/PriceData"
OUTPUT_DIR  = "Data/EDA"
SAMPLE_N    = 200
RANDOM_SEED = 42

W_SHORT = 10
W_MED   = 20
W_LONG  = 50


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _days_since_event(flag: pd.Series) -> pd.Series:
    arr = flag.values.astype(float)
    out = np.full(len(arr), np.nan)
    last = -1
    for i in range(len(arr)):
        if arr[i] == 1.0:
            last = i
        if last >= 0:
            out[i] = i - last
    return pd.Series(out, index=flag.index)


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """(value − rolling mean) / rolling std"""
    mu  = s.rolling(window).mean()
    sig = s.rolling(window).std().replace(0, np.nan)
    return (s - mu) / sig


def _rolling_percentile_rank(s: pd.Series, window: int) -> pd.Series:
    """Rank of today's value within the trailing window (0–1)."""
    def _rank(x):
        if len(x) < 2 or np.isnan(x[-1]):
            return np.nan
        return float(np.sum(x[:-1] < x[-1]) / (len(x) - 1))
    return s.rolling(window).apply(_rank, raw=True)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_experimental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-tier feature set computed from raw OHLCV.

    TIER 1 – BASE:     raw signals that survived the first round of testing
    TIER 2 – ENHANCED: z-scored / vol-normalised / smoothed versions of T1 features
    TIER 3 – COMBO:    cross-feature interactions

    Dropped from v1 (chaff):
        Parkinson/RogersSatchell/YangZhang (r>0.89 with GarmanKlass),
        RollingVarianceShift (r=0.97 with VolRatioShortLong),
        RollingKurtosis, BollingerWidthRatio, EfficiencyRatio, TrendPersistence,
        DirectionalEntropy, RangeCompressionATR, MomentumRatioShortLong,
        ExtremeReturnFrequency, TailRatio, DirectionalRunLength*,
        TrendCurvature, HurstExponent, MomentumSlope, BodySizeRatio,
        VolumeImbalance, DaysSinceGap, VolumeSpikeRatio
    """
    out = df.copy()

    C = df["Close"]; O = df["Open"]; H = df["High"]
    L = df["Low"];   V = df["Volume"]

    log_ret  = np.log(C / C.shift(1))
    log_co   = np.log(C / O)                        # intraday log-return
    log_opc  = np.log(O / C.shift(1))               # overnight log-return
    hl_range = (H - L).replace(0, np.nan)

    # ── TIER 1 : BASE SURVIVORS ────────────────────────────────────────────────

    # Garman-Klass OHLC volatility (best of the correlated cluster)
    gk_daily = 0.5 * np.log(H / L)**2 - (2 * np.log(2) - 1) * log_co**2
    out["GarmanKlassVolatility"] = np.sqrt(gk_daily.rolling(W_MED).mean().clip(0))

    # Candle wick structure
    upper_wick = H - pd.concat([O, C], axis=1).max(axis=1)
    lower_wick = pd.concat([O, C], axis=1).min(axis=1) - L
    out["UpperWickRatio"] = upper_wick / hl_range
    out["LowerWickRatio"] = lower_wick / hl_range

    # Intraday and overnight returns
    out["IntradayReturn"]  = log_co
    out["OvernightReturn"] = log_opc

    # Intraday and overnight realised vol
    out["IntradayVolatility"]  = log_co.rolling(W_MED).std()
    out["OvernightVolatility"] = log_opc.rolling(W_MED).std()

    # Asymmetric rolling volatility
    def _downside(x):
        neg = x[x < 0]; return neg.std() if len(neg) > 2 else np.nan
    def _upside(x):
        pos = x[x > 0]; return pos.std() if len(pos) > 2 else np.nan

    out["DownsideVolatility"] = log_ret.rolling(W_MED).apply(_downside, raw=True)
    out["UpsideVolatility"]   = log_ret.rolling(W_MED).apply(_upside,   raw=True)

    # Second-order vol (vol of rolling vol)
    vol_short = log_ret.rolling(W_SHORT).std()
    out["VolatilityOfVolatility"] = vol_short.rolling(W_MED).std()

    # Rolling max drawdown over 50-day window
    def _max_dd(x):
        cum  = np.cumprod(1 + x)
        peak = np.maximum.accumulate(cum)
        dd   = (cum - peak) / np.where(peak > 0, peak, 1.0)
        return float(dd.min())
    out["MaxDrawdown"] = log_ret.rolling(W_LONG).apply(_max_dd, raw=True)

    # Volatility entropy: sigma-based bins capture heavy-tail vs thin-tail shape
    def _vol_entropy(x):
        abs_ret = np.abs(x)
        if len(x) < 5:
            return np.nan
        sigma = np.std(abs_ret)
        if sigma == 0:
            return 0.0
        thresholds = [0.0, 0.5 * sigma, 1.0 * sigma, 1.5 * sigma, 2.0 * sigma, np.inf]
        counts = np.array([
            np.sum((abs_ret >= thresholds[i]) & (abs_ret < thresholds[i + 1]))
            for i in range(len(thresholds) - 1)
        ], dtype=float)
        total = counts.sum()
        if total == 0:
            return np.nan
        probs = counts / total
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))
    out["VolatilityEntropy"] = log_ret.abs().rolling(W_MED).apply(_vol_entropy, raw=True)

    # Volume-price features
    out["SignedVolume"]    = V * np.sign(C - O)
    log_vol_ma             = np.log1p(V).rolling(W_MED).mean().replace(0, np.nan)
    out["PriceImpactRatio"] = log_ret.abs() / log_vol_ma

    # Days since an extreme return (log-compressed to reduce right-skew)
    ret_mu  = log_ret.rolling(W_LONG).mean()
    ret_sig = log_ret.rolling(W_LONG).std()
    is_ext  = ((log_ret - ret_mu).abs() > 2 * ret_sig).astype(float)
    out["DaysSinceExtremeReturn"] = np.log1p(
        _days_since_event(is_ext).clip(upper=100)
    )

    # ── TIER 2 : ENHANCED ─────────────────────────────────────────────────────

    # GarmanKlass — regime z-score (is vol elevated vs its own history?)
    gkv = out["GarmanKlassVolatility"]
    out["GKV_ZScore"]    = _rolling_zscore(gkv, W_LONG)
    # GarmanKlass — momentum: is vol expanding or contracting?
    out["GKV_Momentum"]  = gkv / gkv.rolling(W_SHORT).mean().replace(0, np.nan)
    # GarmanKlass — percentile rank (robust to outliers)
    out["GKV_Percentile"] = _rolling_percentile_rank(gkv, W_LONG)

    # LowerWickRatio — how extreme is today's wick vs recent history?
    lwr = out["LowerWickRatio"]
    out["LowerWick_ZScore"]   = _rolling_zscore(lwr, W_MED)
    # LowerWickRatio — 5-day smoothed (reduces daily noise)
    out["LowerWick_Smoothed"] = lwr.rolling(5).mean()

    # UpperWickRatio — same treatment
    uwr = out["UpperWickRatio"]
    out["UpperWick_ZScore"]   = _rolling_zscore(uwr, W_MED)
    out["UpperWick_Smoothed"] = uwr.rolling(5).mean()

    # IntradayReturn — normalise by intraday vol (return in sigma units)
    # Captures: "was this a big move relative to typical daily range?"
    out["IntradayReturn_Normalised"] = (
        log_co / out["IntradayVolatility"].replace(0, np.nan)
    )
    # IntradayReturn — z-score within rolling window
    out["IntradayReturn_ZScore"] = _rolling_zscore(log_co, W_MED)

    # OvernightReturn — normalise by overnight vol (standardised gap surprise)
    ovn_vol = out["OvernightVolatility"].replace(0, np.nan)
    out["OvernightReturn_Normalised"] = log_opc / ovn_vol
    out["OvernightReturn_ZScore"]     = _rolling_zscore(log_opc, W_MED)

    # VolatilityEntropy — z-score vs 50-day history
    ve = out["VolatilityEntropy"]
    out["VolatilityEntropy_ZScore"] = _rolling_zscore(ve, W_LONG)

    # MaxDrawdown — z-score
    out["MaxDrawdown_ZScore"] = _rolling_zscore(out["MaxDrawdown"], W_LONG)

    # SignedVolume — normalised by average dollar volume (scale-independent)
    avg_vol = V.rolling(W_MED).mean().replace(0, np.nan)
    out["SignedVolume_Normalised"] = out["SignedVolume"] / avg_vol

    # UpsideVolatility — percentile rank (regime of asymmetric vol)
    out["UpsideVol_Percentile"] = _rolling_percentile_rank(
        out["UpsideVolatility"].fillna(0), W_LONG
    )

    # VolatilityOfVolatility — z-score (is vol of vol elevated?)
    out["VolOfVol_ZScore"] = _rolling_zscore(out["VolatilityOfVolatility"], W_LONG)

    # ── TIER 3 : COMBINATIONS ─────────────────────────────────────────────────

    # Wick asymmetry: positive → upper wick dominates (sellers pushing back rallies)
    # negative → lower wick dominates (buyers defending dips)
    out["WickAsymmetry"] = out["UpperWickRatio"] - out["LowerWickRatio"]
    out["WickAsymmetry_ZScore"] = _rolling_zscore(out["WickAsymmetry"], W_MED)

    # Upside/downside vol ratio: >1 means the stock has been making bigger up moves
    out["UpsideDownsideRatio"] = (
        out["UpsideVolatility"] / out["DownsideVolatility"].replace(0, np.nan)
    )

    # Volatility momentum: is the vol regime accelerating?
    out["VolatilityMomentum"] = (
        gkv / gkv.rolling(W_MED).mean().replace(0, np.nan)
    )

    # Gap-fill signal: overnight gap that was subsequently filled intraday
    # A big positive overnight gap + negative intraday return = gap fade
    out["GapFillSignal"] = out["OvernightReturn_Normalised"] + out["IntradayReturn_Normalised"]

    # Realised skew proxy: (UpsideVol − DownsideVol) / GKV
    # Captures asymmetry of the return distribution in a scale-free way
    out["RealisedSkewProxy"] = (
        (out["UpsideVolatility"] - out["DownsideVolatility"])
        / gkv.replace(0, np.nan)
    )

    # Intraday return conditional on vol regime:
    # multiplies the normalised return by vol-expansion flag (+1 high, −1 low)
    # "was today a big intraday move AND are we in a high-vol expansion?"
    vol_regime = np.sign(out["GKV_Momentum"] - 1.0)   # +1 expanding, -1 contracting
    out["IntradayReturn_x_VolRegime"] = out["IntradayReturn_Normalised"] * vol_regime

    # Overnight surprise × vol regime
    out["OvernightReturn_x_VolRegime"] = out["OvernightReturn_Normalised"] * vol_regime

    # Entropy × GKV interaction: high entropy + high vol → chaotic regime
    out["Entropy_x_GKV"] = out["VolatilityEntropy_ZScore"] * out["GKV_ZScore"]

    return out


# ── feature manifest with tier labels ─────────────────────────────────────────

TIER1_BASE = [
    "GarmanKlassVolatility",
    "UpperWickRatio",
    "LowerWickRatio",
    "IntradayReturn",
    "OvernightReturn",
    "IntradayVolatility",
    "OvernightVolatility",
    "DownsideVolatility",
    "UpsideVolatility",
    "VolatilityOfVolatility",
    "MaxDrawdown",
    "VolatilityEntropy",
    "SignedVolume",
    "PriceImpactRatio",
    "DaysSinceExtremeReturn",
]

TIER2_ENHANCED = [
    "GKV_ZScore",
    "GKV_Momentum",
    "GKV_Percentile",
    "LowerWick_ZScore",
    "LowerWick_Smoothed",
    "UpperWick_ZScore",
    "UpperWick_Smoothed",
    "IntradayReturn_Normalised",
    "IntradayReturn_ZScore",
    "OvernightReturn_Normalised",
    "OvernightReturn_ZScore",
    "VolatilityEntropy_ZScore",
    "MaxDrawdown_ZScore",
    "SignedVolume_Normalised",
    "UpsideVol_Percentile",
    "VolOfVol_ZScore",
]

TIER3_COMBO = [
    "WickAsymmetry",
    "WickAsymmetry_ZScore",
    "UpsideDownsideRatio",
    "VolatilityMomentum",
    "GapFillSignal",
    "RealisedSkewProxy",
    "IntradayReturn_x_VolRegime",
    "OvernightReturn_x_VolRegime",
    "Entropy_x_GKV",
]

FEATURE_NAMES = TIER1_BASE + TIER2_ENHANCED + TIER3_COMBO


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_featurize(
    price_dir: str = PRICE_DIR,
    n_tickers: int = SAMPLE_N,
    specific_tickers: list | None = None,
) -> pd.DataFrame:

    all_files = sorted([f for f in os.listdir(price_dir) if f.endswith(".parquet")])

    if specific_tickers:
        files = [f"{t}.parquet" for t in specific_tickers
                 if os.path.exists(os.path.join(price_dir, f"{t}.parquet"))]
        if not files:
            raise FileNotFoundError(f"None of the specified tickers found in {price_dir}")
    else:
        rng   = np.random.default_rng(RANDOM_SEED)
        files = list(rng.choice(all_files, size=min(n_tickers, len(all_files)), replace=False))

    dfs = []
    for fname in tqdm(files, desc="Loading & featurizing"):
        path = os.path.join(price_dir, fname)
        try:
            raw = pd.read_parquet(path)
            raw["Date"] = pd.to_datetime(raw["Date"])
            raw = raw.sort_values("Date").reset_index(drop=True)
            if len(raw) < W_LONG + 10:
                continue

            feat = compute_experimental_features(raw)
            feat["_log_ret"]  = np.log(feat["Close"] / feat["Close"].shift(1))
            feat["target"]    = (feat["_log_ret"].shift(-1) >= 0).astype(int)
            feat = feat.iloc[W_LONG:-1].reset_index(drop=True)
            feat = feat.dropna(subset=["target"])
            dfs.append(feat)
        except Exception:
            pass

    if not dfs:
        raise RuntimeError("No valid tickers loaded.")

    data = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined shape : {data.shape}")
    print(f"Tickers loaded : {len(dfs)}")
    print(f"Date range     : {data['Date'].min().date()} → {data['Date'].max().date()}")
    print(f"Target balance : {data['target'].value_counts(normalize=True).round(4).to_dict()}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# UNIVARIATE ANALYSIS  (same pipeline as 10__FeatureTesting.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

def univariate_analysis(data: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    y       = data["target"].values
    results = []
    for col in tqdm(feature_cols, desc="Univariate metrics"):
        x    = data[col].values
        mask = ~(np.isnan(x) | np.isinf(x))
        if mask.sum() < 100:
            continue
        xc, yc = x[mask], y[mask]
        try:    auc = roc_auc_score(yc, xc)
        except: auc = 0.5
        try:    rho, p_sp = spearmanr(xc, yc)
        except: rho, p_sp = 0.0, 1.0
        try:    pb_r, pb_p = pointbiserialr(yc, xc)
        except: pb_r, pb_p = 0.0, 1.0

        # Determine tier
        if col in TIER1_BASE:      tier = "T1-base"
        elif col in TIER2_ENHANCED: tier = "T2-enhanced"
        else:                       tier = "T3-combo"

        results.append(dict(
            feature         = col,
            tier            = tier,
            auc             = auc,
            auc_abs         = abs(auc - 0.5),
            spearman_rho    = rho,
            spearman_p      = p_sp,
            pointbiserial_r = pb_r,
            pointbiserial_p = pb_p,
            missing_rate    = 1.0 - mask.mean(),
            n_valid         = int(mask.sum()),
        ))

    return pd.DataFrame(results).sort_values("auc_abs", ascending=False)


def add_mutual_info(uni: pd.DataFrame, data: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    present = [f for f in feature_cols if f in data.columns]
    sample  = data[present + ["target"]].dropna()
    if len(sample) > 50_000:
        sample = sample.sample(50_000, random_state=RANDOM_SEED)
    X_mi = sample[present].replace([np.inf, -np.inf], np.nan).fillna(0)
    mi   = mutual_info_classif(X_mi, sample["target"].values,
                               random_state=RANDOM_SEED, n_neighbors=5)
    mi_df = pd.DataFrame({"feature": present, "mutual_info": mi})
    return uni.merge(mi_df, on="feature", how="left")


def build_composite(uni: pd.DataFrame) -> pd.DataFrame:
    for col in ["auc_abs", "mutual_info"]:
        if col not in uni.columns:
            continue
        mx = uni[col].max()
        uni[f"{col}_norm"] = uni[col] / mx if mx > 0 else 0.0
    norm_cols = [c for c in uni.columns if c.endswith("_norm")]
    if norm_cols:
        uni["composite"] = uni[norm_cols].mean(axis=1)
        uni = uni.sort_values("composite", ascending=False)
    return uni


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# colour per tier
TIER_COLOUR = {"T1-base": "steelblue", "T2-enhanced": "forestgreen", "T3-combo": "coral"}


def plot_summary(uni: pd.DataFrame, score_col: str = "composite") -> None:
    top = uni.head(35).copy()
    colours = top["tier"].map(TIER_COLOUR).tolist()

    fig, axes = plt.subplots(1, 3, figsize=(24, 12))

    for ax, metric, title in zip(
        axes,
        ["auc_abs", "mutual_info", score_col],
        ["AUC |distance from 0.5|", "Mutual Information", "Composite Score"],
    ):
        if metric not in top.columns:
            ax.text(0.5, 0.5, f"{metric} N/A", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        ax.barh(range(len(top)), top[metric].values, color=colours)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"].values, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title)

    # legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in TIER_COLOUR.items()]
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.suptitle("Experimental Features v2 — Top 35", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_features_v2_top35.png"),
                dpi=120, bbox_inches="tight")
    plt.show()


def plot_feature_deep_dive(data: pd.DataFrame, feature: str,
                           target: str = "target") -> None:
    subset = data[[feature, target]].dropna()
    subset = subset[~np.isinf(subset[feature])]
    if len(subset) < 100:
        return

    lo, hi = subset[feature].quantile(0.01), subset[feature].quantile(0.99)
    bins   = np.linspace(lo, hi, 50)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    up   = subset[subset[target] == 1][feature]
    down = subset[subset[target] == 0][feature]

    axes[0].hist(down.clip(lo, hi), bins=bins, alpha=0.5, label="DOWN",
                 color="red",   density=True)
    axes[0].hist(up.clip(lo, hi),   bins=bins, alpha=0.5, label="UP",
                 color="green", density=True)
    axes[0].set_title(f"{feature} — Distribution by Class")
    axes[0].legend()

    subset_clipped        = subset.copy()
    subset_clipped[feature] = subset_clipped[feature].clip(lo, hi)
    subset_clipped.boxplot(column=feature, by=target, ax=axes[1])
    plt.sca(axes[1])
    plt.title(f"{feature} — Boxplot by Class")
    axes[1].set_xlabel("Target (0=DOWN, 1=UP)")

    try:
        s2 = subset.copy()
        s2["quintile"] = pd.qcut(s2[feature], 5, labels=False, duplicates="drop")
        wr  = s2.groupby("quintile")[target].mean()
        cnt = s2.groupby("quintile")[target].count()
        axes[2].bar(wr.index, wr.values, color="steelblue", alpha=0.7)
        axes[2].axhline(y=s2[target].mean(), color="red",
                        linestyle="--", label="Overall")
        for i, (w, c) in enumerate(zip(wr.values, cnt.values)):
            axes[2].text(i, w + 0.005, f"{w:.3f}\n(n={c})",
                         ha="center", fontsize=8)
        axes[2].set_title(f"{feature} — Win Rate by Quintile")
        axes[2].set_xlabel("Quintile (0=lowest, 4=highest)")
        axes[2].set_ylabel("P(UP)")
        axes[2].legend()
    except Exception as e:
        axes[2].text(0.5, 0.5, f"Cannot quintile:\n{e}",
                     ha="center", va="center", transform=axes[2].transAxes)

    plt.suptitle("", fontsize=1)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(uni: pd.DataFrame, data: pd.DataFrame,
                             top_n: int = 30) -> None:
    top_feats = [f for f in uni.head(top_n)["feature"].tolist()
                 if f in data.columns]
    corr = data[top_feats].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask    = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Correlation — Top {top_n} Experimental Features v2")
    plt.xticks(fontsize=7, rotation=45, ha="right")
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_features_v2_corr.png"),
                dpi=120, bbox_inches="tight")
    plt.show()

    print(f"\nHighly correlated pairs (|r| > 0.85) among top {top_n}:")
    found = False
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.85:
                print(f"  {corr.columns[i]:45s} ↔ "
                      f"{corr.columns[j]:45s}  r={corr.iloc[i, j]:.3f}")
                found = True
    if not found:
        print("  None above threshold.")


def plot_tier_comparison(uni: pd.DataFrame) -> None:
    """Box-plots of composite score by tier — did the enhancements actually help?"""
    if "composite" not in uni.columns:
        return
    _, ax = plt.subplots(figsize=(8, 5))
    tier_order = ["T1-base", "T2-enhanced", "T3-combo"]
    data_by_tier = [uni.loc[uni["tier"] == t, "composite"].dropna().values
                    for t in tier_order]
    bp = ax.boxplot(data_by_tier, labels=tier_order, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, t in zip(bp["boxes"], tier_order):
        patch.set_facecolor(TIER_COLOUR[t])
    ax.set_title("Composite Score Distribution by Feature Tier")
    ax.set_ylabel("Composite Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_features_v2_tier_comparison.png"),
                dpi=120, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ACTIONABLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_actionable_summary(uni: pd.DataFrame) -> None:
    sc = "composite" if "composite" in uni.columns else "auc_abs"
    print("\n" + "=" * 90)
    print("ACTIONABLE SUMMARY  —  EXPERIMENTAL FEATURES v2")
    print("=" * 90)

    # Per-tier breakdown
    for tier in ["T1-base", "T2-enhanced", "T3-combo"]:
        sub = uni[uni["tier"] == tier].sort_values(sc, ascending=False)
        print(f"\n{'─'*90}")
        print(f"  {tier.upper()}")
        print(f"{'─'*90}")
        for _, row in sub.iterrows():
            mi  = row.get("mutual_info", float("nan"))
            dir = "↑" if row["auc"] > 0.5 else "↓"
            print(
                f"  {row['feature']:42s}  AUC={row['auc']:.4f}{dir}  "
                f"MI={mi:.4f}  miss={row['missing_rate']:.1%}  "
                f"score={row.get(sc, 0):.3f}"
            )

    # Overall top recommendations
    print(f"\n{'='*90}")
    print("  PIPELINE CANDIDATES  (composite ≥ 0.3, missing < 5%)")
    print(f"{'='*90}")
    cands = uni[(uni.get(sc, pd.Series(dtype=float)) >= 0.3) &
                (uni["missing_rate"] < 0.05)].sort_values(sc, ascending=False)
    for _, row in cands.iterrows():
        mi  = row.get("mutual_info", float("nan"))
        dir = "↑" if row["auc"] > 0.5 else "↓"
        print(
            f"  {row['tier']:14s}  {row['feature']:42s}  "
            f"AUC={row['auc']:.4f}{dir}  MI={mi:.4f}"
        )

    print(f"\n{'='*90}")
    print(f"  DROP LIST  (composite < 0.10 OR missing > 15%)")
    print(f"{'='*90}")
    drops = uni[(uni.get(sc, pd.Series(dtype=float)) < 0.10) |
                (uni["missing_rate"] > 0.15)].sort_values(sc)
    for _, row in drops.iterrows():
        print(
            f"  {row['tier']:14s}  {row['feature']:42s}  "
            f"score={row.get(sc, 0):.3f}  miss={row['missing_rate']:.1%}"
        )
    print("=" * 90)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(n_tickers: int, no_plots: bool, specific_tickers: list | None) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_and_featurize(PRICE_DIR, n_tickers, specific_tickers)

    feat_cols = [f for f in FEATURE_NAMES if f in data.columns]
    missing   = [f for f in FEATURE_NAMES if f not in data.columns]
    print(f"\nFeatures computed : {len(feat_cols)} / {len(FEATURE_NAMES)}")
    if missing:
        print(f"Not computed      : {missing}")

    uni = univariate_analysis(data, feat_cols)
    print("\nAdding mutual information ...")
    uni = add_mutual_info(uni, data, feat_cols)
    uni = build_composite(uni)

    sc = "composite" if "composite" in uni.columns else "auc_abs"

    display_cols = ["feature", "tier", "auc", "auc_abs",
                    "spearman_rho", "mutual_info", "missing_rate", sc]
    display_cols = [c for c in display_cols if c in uni.columns]

    print("\n=== ALL FEATURES RANKED ===")
    print(uni[display_cols].to_string(index=False))

    if not no_plots:
        plot_summary(uni, sc)
        plot_tier_comparison(uni)
        plot_correlation_heatmap(uni, data, top_n=min(30, len(uni)))
        print("\n--- Deep dives: top 12 features ---")
        for feat in uni.head(12)["feature"].tolist():
            if feat in data.columns:
                plot_feature_deep_dive(data, feat)

    out_path = os.path.join(OUTPUT_DIR, "experimental_features_v2_analysis.csv")
    uni.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    print_actionable_summary(uni)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tickers", type=int, default=SAMPLE_N)
    parser.add_argument("--no_plots",  action="store_true")
    parser.add_argument("--ticker",    nargs="+", default=None)
    args = parser.parse_args()
    main(args.n_tickers, args.no_plots, args.ticker)
