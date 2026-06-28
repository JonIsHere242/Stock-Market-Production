"""
orig_orig_gp_crosssectional.py -- port of the GP-discovered cross-sectional alpha
features from 3__AlphaSensitivity (pre_tsz_20260530 backup).

WHAT THIS BLOCK MATERIALISES PER-TICKER (faithfully):
  emd_features_IMF_1_Std  -- a per-ticker rolling EMD feature, std of the IMF1 proxy
  (close - 5d MA) over a 60-bar window. Verified PASS_F32 vs Data/ProcessedData.
  This block is the SOLE faithful producer of this column: gp_input_features declares
  it in METADATA['produces'] but CULLS it from its output (see its _CULL_2026_06_13),
  so it does NOT actually emit the column. Ownership therefore lives here.

CROSS-SECTIONAL COLUMNS (F1_PinkNoise, F3_Return_Rev, F4_Price_Ratio, N1_WN_LC,
N4_SpectralEnt, composite) -- EMITTED AS NaN PLACEHOLDERS, BY DESIGN:
  In the original AlphaSensitivity these were computed by add_gp_cross_sectional_features
  (backup L558), a SEPARATE post-processing pass that loaded EVERY per-ticker parquet,
  stacked them into one panel, and ranked each input WITHIN EACH DATE ACROSS ALL TICKERS
  (compute_gp_features, backup L468). That requires the full cross-ticker panel.

  The current 3__FeatureFramework.py has NO such deferred / panel post-pass: it runs
  each block's compute(df) once, PER TICKER (one stock at a time), in topo order, then
  writes the result. There is no stacked-panel stage that runs after the per-ticker
  blocks. A per-ticker frame has exactly one row per Date, so the cross-sectional rank
  (rank/(N+1)*2-1 with N=1 -> 0) and zscore (divide by ~0 std) helpers are degenerate
  and would emit meaningless constants, NOT the panel ground truth.

  Rather than emit fake degenerate values, compute() writes these 6 columns as all-NaN.
  This honours the produces contract (every name is added to df) and stops the crash,
  while making it unmistakable that they are not real until a cross-sectional panel pass
  exists. TO ACTUALLY POPULATE THEM: add a post-pass to 3__FeatureFramework.py that, after
  all per-ticker blocks run, stacks the per-ticker outputs into one (Date, Ticker) panel
  and runs the cross-sectional GP math below (_compute_gp_features on the full panel).
  The faithful panel formulas are preserved verbatim in _compute_gp_features() for that
  future pass. NOTE: that pass also needs the 23 GP_REQUIRED_COLS materialised on the
  panel (gp_input_features supplies all 23 except the culled emd_features_IMF_1_Std,
  which this block supplies).

CRASH ROOT CAUSE THAT THIS FIX REMOVES:
  The previous version re-computed all 23 GP_REQUIRED_COLS via _compute_gp_input_features
  and pd.concat'd them onto the incoming df. But gp_input_features runs EARLIER in the
  pipeline and already added those 23 names, so the concat produced DUPLICATE column
  labels. df['entropy'] (etc.) then returned a 2-column DataFrame, .values was 2-D, and
  _gp_apply_cs_rank crashed with "too many indices for array: array is 1-dimensional,
  but 2 were indexed". The framework caught the exception and SKIPPED the whole block,
  so none of its 7 columns materialised. The fix computes only the EMD input it needs,
  into local scratch (no concat onto df, no duplicate labels).

Self-contained: imports only numpy/pandas/scipy. No pipeline imports.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata as _gp_rankdata


METADATA = {
    "name":        "orig_orig_gp_crosssectional",
    "description": ("Per-ticker emd_features_IMF_1_Std (faithful) plus the GP "
                    "cross-sectional alpha names (F1/F3/F4/N1/N4 + composite). The "
                    "cross-sectional names rank within each Date ACROSS ALL TICKERS, "
                    "which the framework has no panel post-pass to do, so they are "
                    "emitted as NaN placeholders (see module docstring)."),
    "requires":    ["Close", "High", "Low", "Volume"],
    "produces": [
        "F1_PinkNoise",
        "F3_Return_Rev",
        "F4_Price_Ratio",
        "N1_WN_LC",
        "N4_SpectralEnt",
        "composite",
        "emd_features_IMF_1_Std",
    ],
    "tags":        ["gp", "cross_sectional", "alpha", "emd"],
    "version":     "1.0",
    "author":      "alphasens port",
}


# ── GP_REQUIRED_COLS order is LOAD-BEARING (copy verbatim from backup L256-266) ──
GP_REQUIRED_COLS = [
    'Return', 'High_Low', 'High_Close', 'Low_Close',
    'time_reversal_asymmetry', 'time_between_extremes', 'volume_entropy',
    'information_rate_of_change',
    'emd_features_IMF_1_Energy', 'emd_features_IMF_1_Std',
    'emd_features_IMF_2_Energy', 'emd_features_IMF_2_Std',
    'entropy', 'noise_WhiteNoise', 'noise_PinkNoise',
    'higuchi_fractal_dimension', 'lyapunov_exponent', 'dfa_alpha',
    'spectral_entropy', 'autocorrelation', 'hurst_exponent',
    'higher_order_moments_Skewness', 'higher_order_moments_Kurtosis',
]


# ── _gp_* helpers (copied verbatim from backup L268-300) ──────────────────────
def _gp_cs_rank(arr):
    r = _gp_rankdata(arr, method='average')
    return r / (len(arr) + 1) * 2.0 - 1.0


def _gp_conditional(x, y):
    return np.where(x > 0, y, -y)


def _gp_safe_div(x, y):
    safe_y = np.where(np.abs(y) > 1e-10, y, 1.0)
    return np.where(np.abs(y) > 1e-10, x / safe_y, 0.0)


def _gp_safe_sqrt(x):
    return np.sqrt(np.abs(x))


def _gp_apply_cs_rank(signal, groups):
    out = np.zeros_like(signal, dtype=np.float64)
    for idx in groups:
        s = signal[idx]
        mask = np.isfinite(s)
        if mask.sum() >= 2:
            tmp = np.zeros(len(s))
            tmp[mask] = _gp_cs_rank(s[mask])
            out[idx] = tmp
    return out


def _gp_apply_cs_zscore(signal, groups):
    out = np.zeros_like(signal, dtype=np.float64)
    for idx in groups:
        s = signal[idx]
        mu = np.nanmean(s)
        sigma = np.nanstd(s)
        out[idx] = (s - mu) / (sigma + 1e-10)
    return out


# ── hurst helpers (copied verbatim from backup L237-247) ──────────────────────
def _hurst_exponent(time_series):
    lags = range(2, 100)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


def _rolling_hurst_exponent(series, window_size):
    series_clean = series.dropna()

    def hurst_window(window):
        return _hurst_exponent(window)
    return series_clean.rolling(window=window_size).apply(hurst_window, raw=True)


# ── PRESERVED REFERENCE (NOT called by compute()) ─────────────────────────────
# Faithful copies of the backup's per-ticker input stage and cross-sectional panel
# stage, kept so a future cross-sectional post-pass in 3__FeatureFramework.py can
# reuse the exact math. compute() below does NOT call _compute_gp_input_features:
# doing so re-creates the 23 GP_REQUIRED_COLS that gp_input_features already added
# upstream, yielding duplicate column labels -> 2-D .values -> the IndexError this
# fix removes. A real panel pass would build a fresh stacked frame, run
# _compute_gp_input_features once per ticker (on clean OHLCV), then _compute_gp_features.

# ── Stage 1: per-ticker GP input features (backup compute_gp_input_features L303-465) ──
def _compute_gp_input_features(df, window=60):
    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    volume = df['Volume']
    prev_c = close.shift(1)
    ret    = close.pct_change().fillna(0)
    min_p  = max(window // 2, 10)

    new_cols = {}

    new_cols['Return']     = ret
    new_cols['High_Low']   = (high - low) / (close + 1e-10)
    new_cols['High_Close'] = (high - prev_c) / (prev_c + 1e-10)
    new_cols['Low_Close']  = (low  - prev_c) / (prev_c + 1e-10)

    new_cols['time_reversal_asymmetry'] = (
        ret ** 2 * ret.shift(1) - ret.shift(1) ** 2 * ret
    ).rolling(window, min_periods=min_p).mean()

    def _tbe(x):
        return (np.argmax(x) - np.argmin(x)) / max(len(x) - 1, 1)
    new_cols['time_between_extremes'] = close.rolling(window, min_periods=min_p).apply(_tbe, raw=True)

    def _vol_ent(x):
        x = x[x > 0]
        if len(x) < 4:
            return 0.0
        p = x / x.sum()
        return float(-np.sum(p * np.log(p + 1e-10)))
    new_cols['volume_entropy'] = volume.rolling(window, min_periods=min_p).apply(_vol_ent, raw=True)

    new_cols['information_rate_of_change'] = ret.diff().abs().rolling(window, min_periods=min_p).mean()

    fast_ma = close.rolling(5, min_periods=2).mean()
    slow_ma = close.rolling(20, min_periods=5).mean()
    imf1 = close - fast_ma
    imf2 = fast_ma - slow_ma
    new_cols['emd_features_IMF_1_Energy'] = (imf1 ** 2).rolling(window, min_periods=min_p).mean()
    new_cols['emd_features_IMF_1_Std']    = imf1.rolling(window, min_periods=min_p).std()
    new_cols['emd_features_IMF_2_Energy'] = (imf2 ** 2).rolling(window, min_periods=min_p).mean()
    new_cols['emd_features_IMF_2_Std']    = imf2.rolling(window, min_periods=min_p).std()

    def _ent(x):
        counts, _ = np.histogram(x, bins=10)
        p = counts / (counts.sum() + 1e-10)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))
    new_cols['entropy'] = ret.rolling(window, min_periods=min_p).apply(_ent, raw=True)

    def _spectral_slope(x):
        if len(x) < 8:
            return 0.0
        vals = np.abs(np.fft.rfft(x - x.mean()))
        freqs = np.fft.rfftfreq(len(x))[1:]
        vals = vals[1:]
        if len(freqs) < 2:
            return 0.0
        slope, _ = np.polyfit(np.log(freqs + 1e-10), np.log(vals + 1e-10), 1)
        return float(slope)
    slopes = ret.rolling(window, min_periods=min_p).apply(_spectral_slope, raw=True)
    new_cols['noise_WhiteNoise'] = np.exp(-(slopes ** 2) / 0.5)
    new_cols['noise_PinkNoise']  = np.exp(-((slopes + 1.0) ** 2) / 0.5)

    def _higuchi(x, kmax=6):
        n = len(x)
        if n < kmax * 2:
            return 1.5
        lm = []
        for k in range(1, kmax + 1):
            lk = 0.0
            for m in range(1, k + 1):
                idxs = np.arange(m - 1, n, k)
                if len(idxs) < 2:
                    continue
                lmk = np.sum(np.abs(np.diff(x[idxs]))) * (n - 1) / (k * (len(idxs) - 1))
                lk += lmk
            lm.append(lk / k)
        if len(lm) < 2:
            return 1.5
        slope, _ = np.polyfit(np.log(range(1, len(lm) + 1)), np.log(np.array(lm) + 1e-10), 1)
        return float(slope)
    new_cols['higuchi_fractal_dimension'] = close.rolling(window, min_periods=min_p).apply(
        _higuchi, raw=True
    )

    def _lyap(x):
        d = np.abs(np.diff(x))
        d = d[d > 1e-10]
        return float(np.mean(np.log(d))) if len(d) > 0 else 0.0
    new_cols['lyapunov_exponent'] = close.rolling(window, min_periods=min_p).apply(_lyap, raw=True)

    def _dfa(x):
        n = len(x)
        if n < 16:
            return 0.5
        y = np.cumsum(x - np.mean(x))
        max_scale = max(4, n // 4)
        scales = np.unique(np.round(np.logspace(1, np.log10(max_scale), 6)).astype(int))
        scales = scales[scales >= 4]
        flucts = []
        for s in scales:
            n_segs = n // s
            if n_segs < 1:
                continue
            f2 = 0.0
            for i in range(n_segs):
                seg = y[i * s:(i + 1) * s]
                t = np.arange(len(seg), dtype=float)
                p = np.polyfit(t, seg, 1)
                f2 += np.mean((seg - np.polyval(p, t)) ** 2)
            flucts.append(np.sqrt(f2 / n_segs))
        if len(flucts) < 2:
            return 0.5
        valid = [(scales[i], flucts[i]) for i in range(len(flucts)) if flucts[i] > 0]
        if len(valid) < 2:
            return 0.5
        ls, lf = zip(*valid)
        alpha, _ = np.polyfit(np.log(ls), np.log(lf), 1)
        return float(np.clip(alpha, -1, 3))
    new_cols['dfa_alpha'] = ret.rolling(window, min_periods=min_p).apply(_dfa, raw=True)

    def _spec_ent(x):
        if len(x) < 8:
            return 0.0
        ps = np.abs(np.fft.rfft(x - x.mean())) ** 2
        ps = ps / (ps.sum() + 1e-10)
        return float(-np.sum(ps * np.log(ps + 1e-10)))
    new_cols['spectral_entropy'] = ret.rolling(window, min_periods=min_p).apply(_spec_ent, raw=True)

    def _ac1(x):
        if len(x) < 3:
            return 0.0
        cc = np.corrcoef(x[:-1], x[1:])
        return float(cc[0, 1]) if np.isfinite(cc[0, 1]) else 0.0
    new_cols['autocorrelation'] = ret.rolling(window, min_periods=min_p).apply(_ac1, raw=True)

    new_cols['hurst_exponent'] = _rolling_hurst_exponent(close, window_size=window)

    new_cols['higher_order_moments_Skewness'] = ret.rolling(window, min_periods=min_p).skew()
    new_cols['higher_order_moments_Kurtosis'] = ret.rolling(window, min_periods=min_p).kurt()

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ── Stage 2: cross-sectional GP features (backup compute_gp_features L468-555) ──
def _compute_gp_features(df, date_col='Date'):
    df = df.copy().reset_index(drop=True)

    groups = [
        grp.index.to_numpy()
        for _, grp in df.groupby(date_col, sort=False)
    ]

    # Cross-sectionally rank all 23 inputs within each date -> x[0..22]
    x = {}
    for i, col in enumerate(GP_REQUIRED_COLS):
        raw = np.nan_to_num(df[col].values.astype(np.float64), nan=0.0)
        x[i] = _gp_apply_cs_rank(raw, groups)

    # F1 -- seed 1017
    f1_raw = x[14] - _gp_conditional(
        np.abs(1.77026646253569) + x[19],
        -0.9931974192883872 * x[3]
    )

    # F3 -- seed 1004
    f3_raw = -0.07320318705361156 * x[0]

    # F4 -- seed 1002
    f4_raw = _gp_safe_div(
        np.abs(0.17726009244632013) + x[2],
        x[3] + (x[7] + 2.6940490745579684 ** 2) + x[8]
    )

    # N1 -- seed 3003
    n1_raw = x[13] + x[3]

    # N4 -- seed 3029
    _denom_n4 = np.abs(-1.2577672553937247 * 3.959124395312415)
    n4_raw = _gp_safe_sqrt(
        np.abs(x[3]) * _gp_safe_div(-1.7102360477772514 + x[18] + x[0], _denom_n4)
    )

    F1 = _gp_apply_cs_rank(f1_raw, groups)
    F3 = _gp_apply_cs_rank(f3_raw, groups)
    F4 = _gp_apply_cs_zscore(f4_raw, groups)
    N1 = _gp_apply_cs_rank(n1_raw, groups)
    N4 = _gp_apply_cs_rank(n4_raw, groups)

    def _safe(a):
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    F1, F3, F4, N1, N4 = _safe(F1), _safe(F3), _safe(F4), _safe(N1), _safe(N4)

    composite = (
        -0.034224 * F1 + 1.296387 * F3 + 0.004473 * F4
        + 0.012287 * N1 + 0.009157 * N4
        - 0.274113 * F1 * F3 + 0.085361 * F1 * F4
        + 0.021151 * F1 * N1 - 0.181167 * F1 * N4
        - 1.295755 * F3 * F4 + 0.355264 * F3 * N1
        + 2.387638 * F3 * N4 - 0.044057 * F4 * N1
        - 0.000961 * F4 * N4 + 0.048930 * N1 * N4
        + 0.002355
    )

    return pd.DataFrame({
        'F1_PinkNoise':   F1,
        'F3_Return_Rev':  F3,
        'F4_Price_Ratio': F4,
        'N1_WN_LC':       N1,
        'N4_SpectralEnt': N4,
        'composite':      _safe(composite),
    }, index=df.index)


# ── per-ticker EMD feature (the one faithful column, backup L350-353) ─────────
def _emd_imf1_std(df, window=60):
    """std of the IMF1 proxy (close - 5d MA) over a rolling 60-bar window.

    Computed in LOCAL scratch from OHLCV only -- it deliberately does NOT touch
    or concat onto any pre-existing pipeline columns, so it cannot create the
    duplicate-label condition that previously crashed the block.
    """
    close   = df['Close']
    min_p   = max(window // 2, 10)
    fast_ma = close.rolling(5, min_periods=2).mean()
    imf1    = close - fast_ma
    return imf1.rolling(window, min_periods=min_p).std()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # The one column this block can faithfully compute per-ticker.
    df['emd_features_IMF_1_Std'] = _emd_imf1_std(df, window=60).to_numpy()

    # The six cross-sectional GP columns need a cross-ticker panel pass that the
    # framework does not run (see module docstring). Emit them as NaN placeholders
    # so the produces contract is honoured and nothing downstream sees fake values.
    # The faithful panel math is preserved in _compute_gp_features() for the day a
    # cross-sectional post-pass is added to 3__FeatureFramework.py.
    for col in ('F1_PinkNoise', 'F3_Return_Rev', 'F4_Price_Ratio',
                'N1_WN_LC', 'N4_SpectralEnt', 'composite'):
        df[col] = np.nan

    return df
