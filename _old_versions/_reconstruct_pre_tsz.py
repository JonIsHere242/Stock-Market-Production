"""Reconstruct the pre-_tsz state of 3__AlphaSensitivity.py by removing ONLY the
blocks added in the 2026-05-30 session (the file also contains prior uncommitted
work that must be preserved, so `git checkout` is NOT a safe revert).

Each added block is removed by exact-string match with an assertion, so any
mismatch fails loudly rather than silently leaving code behind. Output is written
to _old_versions/3__AlphaSensitivity_pre_tsz_20260530.py and verified to contain
zero _tsz markers and to compile."""
import ast, os, sys

SRC = "3__AlphaSensitivity.py"
OUT = "_old_versions/3__AlphaSensitivity_pre_tsz_20260530.py"

with open(SRC, "r", encoding="utf-8") as f:
    s = f.read()

# Each entry is an exact chunk that was ADDED this session; removing it (->"")
# restores the original surrounding code.
CHUNKS = []

# --- helper (inserted after safe_log) ---
CHUNKS.append('''


def rolling_ticker_zscore(s, window=252, min_periods=60, clip=8.0):
    """Within-ticker rolling z-score that re-bases a feature against this stock's
    own recent history -- the "_tsz" trick used to lift mid-importance per-stock
    features (see eda_feature_scan.py / classify_features.py).

    Lookahead-safe: the mean/std baseline is built from STRICTLY PAST values via
    shift(1); the current value s_t is known at the close of day t (used to
    predict t+1). Operates on a single-ticker, date-ordered Series (every caller
    here runs per-ticker).

    The result is clipped to +/-`clip`. This is a no-op for the model's signal --
    XGBoost splits on order and the cross-sectional rank dual is order-based, so
    rank-IC is unchanged -- but it kills the (1-0)/~0 blowup that sparse BINARY
    flags produce when a stock has zero variance in its trailing window."""
    mu = s.shift(1).rolling(window, min_periods=min_periods).mean()
    sd = s.shift(1).rolling(window, min_periods=min_periods).std()
    z = (s - mu) / (sd + 1e-8)
    return z.clip(-clip, clip)''')

# --- Price_Differential regime tsz (inside add_volatility_regime_signals) ---
CHUNKS.append('''

    # Within-ticker rolling z of the regime flags: how unusual this vol regime is
    # for this stock vs its own history. Deep-ranked but high-lift in EDA
    # (topq-label IC IR): High +0.58 (raw -0.07), Low -0.55 (raw -0.09); both
    # orthogonal to everything else. Originals kept. Leak-safe (flags known at t).
    df[f'{base_column}_HighVolatilityRegime_tsz'] = rolling_ticker_zscore(
        df[f'{base_column}_HighVolatilityRegime'])
    df[f'{base_column}_LowVolatilityRegime_tsz'] = rolling_ticker_zscore(
        df[f'{base_column}_LowVolatilityRegime'])''')

# --- G_Volume_Weighted_High_Ratio_tsz (genetic indicators) ---
CHUNKS.append('''
    # Within-ticker rolling z-score of the volume-weighted high ratio. Captures a
    # new price high made on UNUSUALLY LOW volume (vs this stock's own history) =
    # unconfirmed breakout that tends to mean-revert. EDA (eda_feature_scan.py):
    # per-day topq-label IC IR -0.46 -> -0.86 (t=-15, IC negative on 84% of days,
    # stable across both halves), and only ~0.5 corr with the volume-magnitude
    # cluster / 0.04 with top features, so it adds NEW signal. Original kept.
    # Lookahead-safe: base uses High/High_Lag1/Volume (all known at close of t);
    # the z baseline uses strictly PAST values via shift(1).
    _vwhr = df['G_Volume_Weighted_High_Ratio']
    _vwhr_mu = _vwhr.shift(1).rolling(252, min_periods=60).mean()
    _vwhr_sd = _vwhr.shift(1).rolling(252, min_periods=60).std()
    df['G_Volume_Weighted_High_Ratio_tsz'] = (_vwhr - _vwhr_mu) / (_vwhr_sd + 1e-8)''')

# --- Complexity_Invariant_Distance_tsz ---
CHUNKS.append('''
        # Within-ticker rolling z of CID: an unusually large jump in price-series
        # complexity vs this stock's own history. EDA (topq-label IC IR): raw
        # -0.03 -> tsz +0.25, orthogonal. Original kept; leak-safe (window ends at t).
        new_columns['Complexity_Invariant_Distance_tsz'] = rolling_ticker_zscore(
            new_columns['Complexity_Invariant_Distance'])''')

# --- dollar_volume_ratio_252d_tsz (liquidity) ---
CHUNKS.append('''

    # Within-ticker rolling z-score of the 252d dollar-volume ratio.
    # Re-bases the liquidity-surge signal against THIS stock's own recent
    # history. EDA (eda_midband_features.py) showed this lifts the per-day
    # topq-label IC information ratio 0.68 -> 0.93 and de-correlates it from
    # the redundant dollar_volume_zscore sibling. Original feature left intact.
    # Lookahead-safe: the mean/std baseline uses strictly PAST values (shift(1)),
    # and the numerator ratio is known at the close of day t (used to predict t+1).
    _dvr = features['dollar_volume_ratio_252d']
    _dvr_mu = _dvr.shift(1).rolling(252, min_periods=60).mean()
    _dvr_sd = _dvr.shift(1).rolling(252, min_periods=60).std()
    features['dollar_volume_ratio_252d_tsz'] = (_dvr - _dvr_mu) / (_dvr_sd + 1e-8)''')

# --- volume_burst_intensity_5d (liquidity) ---
CHUNKS.append('''

    # Continuous rebuild of the burst signal. The 0-5 count above is 96% zeros
    # and discards magnitude; this is the 5-day mean of the volume/10d-avg ratio,
    # preserving how BIG recent bursts were, not just whether they crossed 3x.
    # EDA (same topq-label IC IR metric): count IR +0.44 -> burst_intensity_5d
    # IR +0.66 (t=+12), and only 0.44 corr with both the count and the existing
    # single-day volume_spike_ratio -> distinct multi-day persistence signal.
    # Original count left intact. Lookahead-safe: vol_10d_avg uses shift(1)
    # (strictly past) and the 5d window ends at day t (known at close of t).
    _spike = current_volume / (vol_10d_avg + 1e-8)
    features['volume_burst_intensity_5d'] = _spike.rolling(5, min_periods=3).mean()''')

# --- atr_percentile_rank_tsz (volatility_atr) ---
CHUNKS.append('''

    # Within-ticker rolling z-score of the ATR percentile rank.
    # The raw percentile is washed out cross-sectionally (per-day topq-label IC
    # IR ~0.02); standardizing vol-extremity against THIS stock's own recent
    # history turns it into real signal (EDA: topq-IC IR 0.02 -> 0.49).
    # Original atr_percentile_rank left intact.
    # Lookahead-safe: baseline uses strictly PAST values (shift(1)); the current
    # percentile is computed only from data through day t (predicting t+1).
    _apr = features['atr_percentile_rank']
    _apr_mu = _apr.shift(1).rolling(252, min_periods=60).mean()
    _apr_sd = _apr.shift(1).rolling(252, min_periods=60).std()
    features['atr_percentile_rank_tsz'] = (_apr - _apr_mu) / (_apr_sd + 1e-8)''')

# --- atr_regime_low_tsz + atr_regime_high_tsz (volatility_atr, round 4) ---
CHUNKS.append('''

    # Within-ticker rolling z-score of the ATR regime flags ("how unusual is this
    # vol regime for this stock vs its own history"). Both orthogonal to
    # atr_percentile_rank_tsz (corr 0.04-0.06) and to each other / the volume
    # cluster. EDA (topq-label IC IR): low-flag -0.06 -> -0.42 (t=-7.3);
    # high-flag +0.00 -> +0.64. Originals kept. Clipped helper avoids the binary
    # (1-0)/~0 blowup; clip is rank-neutral so the IRs are unchanged.
    features['atr_regime_low_tsz'] = rolling_ticker_zscore(features['atr_regime_low'])
    features['atr_regime_high_tsz'] = rolling_ticker_zscore(features['atr_regime_high'])''')

# --- return_3d_tsz (price momentum) ---
CHUNKS.append('''

    # Within-ticker rolling z-score of the 3-day return. Orthogonal short-term
    # reversal signal (corr <0.1 to the volume/ATR clusters): a 3d move that is
    # large RELATIVE to this stock's own recent moves tends to revert.
    # EDA (topq-label IC IR): raw -0.08 -> ts_z -0.20 (t=-3.5). Original kept.
    # Lookahead-safe: return_3d known at close of t; baseline strictly past.
    _r3 = features['return_3d']
    _r3_mu = _r3.shift(1).rolling(252, min_periods=60).mean()
    _r3_sd = _r3.shift(1).rolling(252, min_periods=60).std()
    features['return_3d_tsz'] = (_r3 - _r3_mu) / (_r3_sd + 1e-8)''')

# --- doji_pattern_tsz (price action) ---
CHUNKS.append('''

    # Within-ticker rolling z of the (very sparse, 94% zero) doji flag = a "doji
    # surprise" signal: a doji is more informative for a stock that rarely prints
    # one. EDA (topq-label IC IR): raw -0.51 -> tsz +0.72, orthogonal to all other
    # candidates. Original flag kept. Clipped helper tames the sparse-binary blowup.
    features['doji_pattern_tsz'] = rolling_ticker_zscore(
        pd.Series(features['doji_pattern'], index=df.index))''')

# --- cv_{window}d_regime_high_tsz (cv features) ---
CHUNKS.append('''

        # "Unusual high-CV regime for this stock" surprise. EDA (topq-label IC IR):
        # 10d +0.42, 20d +0.30 (50d only +0.09 -> skipped). Orthogonal. Originals
        # kept. Clipped helper avoids the sparse-binary blowup.
        if window in (10, 20):
            features[f'cv_{window}d_regime_high_tsz'] = rolling_ticker_zscore(
                features[f'cv_{window}d_regime_high'])''')

for i, c in enumerate(CHUNKS):
    n = s.count(c)
    if n != 1:
        sys.exit(f"ABORT: chunk #{i} matched {n} times (expected 1). "
                 f"First 60 chars: {c[:60]!r}")
    s = s.replace(c, "", 1)

for marker in ("_tsz", "rolling_ticker_zscore", "volume_burst_intensity_5d"):
    assert marker not in s, f"marker {marker!r} still present after removal"

ast.parse(s)  # must still compile
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(s)
print(f"OK: wrote {OUT}  ({s.count(chr(10))+1} lines; removed {len(CHUNKS)} blocks)")
