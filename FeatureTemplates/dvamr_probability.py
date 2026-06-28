import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

METADATA = {
    "name":        "dvamr_probability",
    "description": "Dynamic Volatility Adjusted Mean Reversion Probability (DVAMR): a 0-1 confidence score for mean-reversion opportunities derived from VIX dynamics, market stress, and volatility regimes",
    "requires":    ["Close", "mean_reversion_z_score_90_std_1", "vix_acceleration", "vix_close"],
    "produces":    ["dvamr_probability"],
    "tags":        ["mean_reversion", "market_regime", "experimental"],
    "version":     "1.0",
    "author":      "ported from 3__AlphaSensitivity.calculate_dvamr_probability",
}

_LOOKBACK_WINDOW     = 60
_MOMENTUM_WINDOW     = 20   # preserved from source signature (unused in body)
_PROBABILITY_WINDOW  = 120


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dynamic Volatility Adjusted Mean Reversion Probability (DVAMR)

    Produces a probability score (0-1) indicating confidence in mean reversion
    opportunities during volatile market conditions. Higher values indicate
    higher confidence in potential reversion trades based on VIX dynamics,
    market stress, and volatility regimes.

    'daily_return' is computed as a LOCAL variable; the only column added to df
    is 'dvamr_probability'.
    """

    lookback_window    = _LOOKBACK_WINDOW
    probability_window = _PROBABILITY_WINDOW
    # _MOMENTUM_WINDOW preserved in source signature but unused in the body.

    n_rows = len(df)

    # Pull the inputs into plain numpy once. The original looped row-by-row over
    # pandas Series (~15 scalar .iloc / Series-reduction / expanding() calls per
    # row), which is pure pandas dispatch overhead. Everything below computes the
    # *same* fixed-60-window numpy reductions, just batched with sliding windows.
    close = df['Close'].to_numpy(dtype=float)
    mr    = df['mean_reversion_z_score_90_std_1'].to_numpy(dtype=float)
    vix_a = df['vix_acceleration'].to_numpy(dtype=float)
    vix_c = df['vix_close'].to_numpy(dtype=float)

    # Temporary per-ticker return series (kept local — NOT added as a column).
    # Use pandas pct_change to match its exact NaN semantics, then drop to numpy.
    daily_return = df['Close'].pct_change().to_numpy(dtype=float)

    # === BASE GP SIGNAL (vectorized; uses current row, no window) ===
    # denominator = vix_acceleration + vix_close + vix_close  (i.e. + 2*vix_close)
    numerator   = mr - vix_a
    denominator = vix_a + 2.0 * vix_c
    with np.errstate(divide='ignore', invalid='ignore'):
        base_signal = numerator / denominator
    # Faithful to `if denominator != 0`: NaN denom -> NaN != 0 is True -> keeps NaN.
    base_signal = np.where(denominator != 0, base_signal, 0.0)

    # Rows with i < lookback_window keep the bare base signal.
    raw_signal = base_signal.copy()

    lb = lookback_window
    if n_rows > lb:
        # Window for output row i is rows [i-lb : i] (past data only, no leakage).
        # sliding_window_view(a, lb)[j] == a[j : j+lb]; row i maps to j = i-lb.
        # Valid output rows are i = lb .. n_rows-1  -> j = 0 .. n_rows-lb-1.
        k = n_rows - lb
        out = slice(lb, n_rows)
        ret_w   = sliding_window_view(daily_return, lb)[:k]   # == daily_return[i-lb:i]
        mr_w    = sliding_window_view(mr,    lb)[:k]
        close_w = sliding_window_view(close, lb)[:k]

        # === DYNAMIC ADJUSTMENT FACTOR ===
        # 1. Volatility factor (nanstd ddof=1 matches pandas Series.std skipna).
        current_volatility = np.nanstd(ret_w, axis=1, ddof=1) * np.sqrt(252)
        adjustment_factor = 0.8 + 0.4 * np.clip(current_volatility / 0.6, 0.5, 2.0)

        # 2. Up-days factor ((NaN > 0) is False, counted in the denominator).
        up_days_pct = np.mean(ret_w > 0, axis=1)
        adjustment_factor *= np.clip(1.6 - up_days_pct * 2, 0.6, 1.4)

        # 3. Mean-reversion factor.
        mr_score = np.nanmean(mr_w, axis=1)
        adjustment_factor *= np.clip(1.2 - mr_score * 0.5, 0.7, 1.5)

        # 4. Drawdown factor (expanding().max() == cumulative max).
        peak = np.maximum.accumulate(close_w, axis=1)
        current_drawdown = np.min((close_w - peak) / peak, axis=1)
        adjustment_factor *= np.clip(1.0 + np.abs(current_drawdown) * 1.5, 0.8, 1.8)

        # 5. Price-stability factor.
        price_cv = np.nanstd(close_w, axis=1, ddof=1) / np.nanmean(close_w, axis=1)
        current_price_stability = 1.0 / (price_cv + 1e-8)
        adjustment_factor *= np.clip(1.5 - current_price_stability * 0.1, 0.7, 1.3)

        adjustment_factor = np.clip(adjustment_factor, 0.3, 3.0)

        # === VIX REGIME FILTER === (previous-day vix; NaN -> 20)
        current_vix = vix_c[lb - 1:n_rows - 1].copy()   # vix_close[i-1] for i = lb..n-1
        current_vix = np.where(np.isnan(current_vix), 20.0, current_vix)
        # if/elif chain -> np.select picks the first satisfied condition in order.
        vix_regime = np.select(
            [(current_volatility > 0.4) & (current_vix > 15),
             (current_volatility < 0.25) | (current_vix < 12)],
            [1.0, 0.3],
            default=0.7,
        )

        # === DISTRESS AMPLIFIER ===
        stock_return = close_w[:, -1] / close_w[:, 0] - 1.0
        # NOTE: original `elif current_vix > 35` is dead code (the `> 25` branch
        # already catches it). Preserved faithfully: only the *1.5 path applies.
        distress_amp = np.where(current_vix > 25, 1.5, 1.0)
        distress_amp = np.where((stock_return < -0.1) & (current_vix > 20),
                                distress_amp * 1.3, distress_amp)
        distress_amp = np.where(stock_return < -0.3, distress_amp * 1.6, distress_amp)
        distress_amp = np.clip(distress_amp, 0.5, 2.5)

        # === FINAL ENHANCED SIGNAL ===
        raw_signal[out] = base_signal[out] * adjustment_factor * vix_regime * distress_amp

    # === CONVERT TO PROBABILITY (0-1) ===
    # Rolling percentile rank -> sigmoid. Vectorized over the full-history rows;
    # the window is always exactly probability_window wide here.
    probability_signal = np.zeros(n_rows)

    pw = probability_window
    if n_rows > pw:
        # window for row i is raw_signal[i-pw : i]; sliding row j == raw[j:j+pw].
        sig_w = sliding_window_view(raw_signal, pw)[:n_rows - pw]
        current = raw_signal[pw:][:, None]
        percentile_rank = np.sum(sig_w <= current, axis=1) / pw
        probability_signal[pw:] = 1.0 / (1.0 + np.exp(-5 * (percentile_rank - 0.5)))

    # For early periods without enough history, use simple normalization
    for i in range(probability_window):
        if i > 0:
            # Simple min-max normalization on available data
            window_signals = raw_signal[:i + 1]
            if np.std(window_signals) > 0:
                normalized = (raw_signal[i] - np.min(window_signals)) / (np.max(window_signals) - np.min(window_signals))
                probability_signal[i] = np.clip(normalized, 0, 1)
            else:
                probability_signal[i] = 0.5

    # Add only the probability signal to output
    df['dvamr_probability'] = probability_signal

    return df
