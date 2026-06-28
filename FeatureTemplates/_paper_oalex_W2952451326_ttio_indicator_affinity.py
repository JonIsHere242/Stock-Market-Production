"""
Per-ticker technical indicator affinity (TTIO) proxy derived from:
  "Individualized Indicator for All" (openalex:W2952451326).

The paper observes that stocks have different affinities to different technical
indicators — i.e., the same RSI or MACD signal works well for some tickers and
poorly for others. The TTIO framework learns per-stock indicator re-scaling
weights using skip-gram stock embeddings. The per-ticker OHLCV proxy implemented
here instead measures affinity DIRECTLY: for each classical indicator signal,
compute how often it has led to a positive 5-day forward return over the
trailing 120 days. High affinity = this indicator has historically worked for
THIS ticker; low affinity = the indicator is noisy or contrarian for this stock.

This meta-indicator is genuinely per-ticker, fully causal (forward return
computed by shifting the return BACKWARD to label the signal bar), and
informative: a stock where RSI oversold signals have a 70% success rate is
different from one where they have a 30% rate.

Forward-return window: 5 bars (approximately 1 trading week). The 5-day
forward return label used inside the rolling window is causal because at
bar t we only use past labels (for bars t-119 through t-4, i.e. labels whose
5-day window closed before t). Leading NaNs until the lookback fills.

Columns prefixed `ttio_`.
"""
import numpy as np
import pandas as pd

METADATA = {
    "name":        "paper_oalex_W2952451326_ttio_indicator_affinity",
    "description": (
        "Per-ticker rolling (120d) affinity of three classical technical signals "
        "(RSI oversold, MACD bull cross, price-above-MA20) measured as their "
        "historical 5-day-forward hit rate for this specific stock; plus a "
        "composite affinity score. Proxy for the TTIO framework (openalex:W2952451326)."
    ),
    "requires":    ["Close"],
    "produces":    [
        "ttio_rsi_affinity_120d",      # RSI<30 signal hit rate over trailing 120d
        "ttio_macd_affinity_120d",     # MACD>signal hit rate over trailing 120d
        "ttio_ma_affinity_120d",       # close>MA20 hit rate over trailing 120d
        "ttio_composite_affinity_120d",# equal-weight average of the three
        "ttio_affinity_spread_120d",   # max-min affinity spread (indicator dispersion)
    ],
    "tags":        ["experimental", "technical", "meta"],
    "version":     "1.0",
    "author":      "paper-mining slate 3",
}

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
_WINDOW    = 120    # rolling affinity window (trading days)
_FWD       = 5     # forward return horizon (bars)
_MIN_SIGS  = 5     # minimum signal occurrences for a valid affinity estimate

# RSI configuration
_RSI_WIN  = 14
_RSI_OB   = 30.0   # oversold threshold (signal = RSI < 30, then expect bounce)

# MACD configuration
_FAST = 12
_SLOW = 26
_SIG  = 9


# ─────────────────────────────────────────────────────────────────────────────
# Indicator helpers (vectorised, causal)
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI via EMA; fully vectorised, causal."""
    n = len(close)
    rsi = np.full(n, np.nan)
    delta = np.empty(n)
    delta[0] = np.nan
    delta[1:] = close[1:] - close[:-1]

    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Seed with simple average over first period
    if n <= period:
        return rsi

    avg_gain = np.nanmean(gains[1: period + 1])
    avg_loss = np.nanmean(losses[1: period + 1])

    alpha = 1.0 / period
    for i in range(period, n):
        avg_gain = alpha * gains[i] + (1.0 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1.0 - alpha) * avg_loss
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """EMA via pandas; causal."""
    return pd.Series(arr).ewm(span=span, adjust=False).mean().to_numpy()


def _macd_bull(close: np.ndarray, fast: int, slow: int, sig: int) -> np.ndarray:
    """
    Returns a 0/1 array: 1 where MACD line > Signal line (bullish), else 0.
    NaN where either series has insufficient history.
    """
    n = len(close)
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, sig)
    bull = np.where(np.isfinite(macd_line) & np.isfinite(signal_line),
                    (macd_line > signal_line).astype(float), np.nan)
    return bull


def _ma_bull(close: np.ndarray, window: int = 20) -> np.ndarray:
    """1 where close > rolling MA(window), else 0, NaN in warmup."""
    ma = pd.Series(close).rolling(window, min_periods=window).mean().to_numpy()
    bull = np.where(np.isfinite(ma), (close > ma).astype(float), np.nan)
    return bull


# ─────────────────────────────────────────────────────────────────────────────
# Affinity scorer
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_affinity(signal: np.ndarray, fwd_ret_label: np.ndarray,
                      window: int, min_sigs: int) -> np.ndarray:
    """
    For each bar t, compute: among the bars in [t-window+1, t-FWD] where
    signal==1 (or signal triggered), what fraction had a positive 5-day
    forward return?

    fwd_ret_label[i] = return from close[i] to close[i+FWD], pre-computed
    and lagged so that fwd_ret_label[t] is the label CLOSED _before_ bar t
    (causal: only labels where the full fwd window is in the past).

    Returns affinity in [0,1], or NaN if fewer than min_sigs signals.
    """
    n = len(signal)
    affinity = np.full(n, np.nan)

    for t in range(window - 1, n):
        # Bars used: [t-window+1 .. t]. But fwd_ret uses close[i+FWD],
        # so only bars i <= t-FWD have a closed label causally.
        # We restrict to [t-window+1 .. t-FWD].
        end_label = t - _FWD  # inclusive end index for labelled bars
        if end_label < 0:
            continue
        start_idx = max(t - window + 1, 0)
        sig_win = signal[start_idx: end_label + 1]
        lbl_win = fwd_ret_label[start_idx: end_label + 1]

        # Select bars where signal fired
        mask = (sig_win == 1.0) & np.isfinite(lbl_win)
        n_sig = mask.sum()
        if n_sig < min_sigs:
            continue

        affinity[t] = float((lbl_win[mask] > 0).sum()) / float(n_sig)

    return affinity


# ─────────────────────────────────────────────────────────────────────────────
# Public compute
# ─────────────────────────────────────────────────────────────────────────────

def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # ---- Forward return label (causal: fwd_ret[i] = close[i+FWD]/close[i]-1) --
    # At bar t we can only use fwd_ret[i] for i <= t-FWD (fully in the past).
    fwd_ret = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd_ret[: n - _FWD] = np.where(
            close[: n - _FWD] > 0,
            close[_FWD:] / close[: n - _FWD] - 1.0,
            np.nan,
        )
    # fwd_ret[n-FWD .. n-1] = NaN (no closed label yet — correct)

    # ---- Indicator signals (all causal 0/1 arrays) ---------------------------
    # 1. RSI oversold (<30): signal = 1
    rsi = _rsi(close, _RSI_WIN)
    rsi_signal = np.where(np.isfinite(rsi), (rsi < _RSI_OB).astype(float), np.nan)

    # 2. MACD bullish (MACD > signal): signal = 1
    macd_signal = _macd_bull(close, _FAST, _SLOW, _SIG)

    # 3. Price > MA20: signal = 1
    ma_signal = _ma_bull(close, window=20)

    # ---- Rolling affinity per indicator -------------------------------------
    aff_rsi  = _rolling_affinity(rsi_signal,  fwd_ret, _WINDOW, _MIN_SIGS)
    aff_macd = _rolling_affinity(macd_signal, fwd_ret, _WINDOW, _MIN_SIGS)
    aff_ma   = _rolling_affinity(ma_signal,   fwd_ret, _WINDOW, _MIN_SIGS)

    # ---- Composite and spread -----------------------------------------------
    stack = np.stack([aff_rsi, aff_macd, aff_ma], axis=1)  # (n, 3)
    valid_counts = np.sum(np.isfinite(stack), axis=1)

    composite = np.where(
        valid_counts >= 2,
        np.nanmean(stack, axis=1),
        np.nan,
    )
    spread = np.where(
        valid_counts >= 2,
        np.nanmax(stack, axis=1) - np.nanmin(stack, axis=1),
        np.nan,
    )

    def _clean(a: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(a), a, np.nan)

    df["ttio_rsi_affinity_120d"]       = _clean(aff_rsi)
    df["ttio_macd_affinity_120d"]      = _clean(aff_macd)
    df["ttio_ma_affinity_120d"]        = _clean(aff_ma)
    df["ttio_composite_affinity_120d"] = _clean(composite)
    df["ttio_affinity_spread_120d"]    = _clean(spread)

    return df
