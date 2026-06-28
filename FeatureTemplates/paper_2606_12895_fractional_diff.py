"""
Fractional differencing features.

PAPER: "LongSpike: Fractional Order Spiking State Space Models for Efficient
Long Sequence Learning" (arxiv 2606.12895) — pure neural-network architecture,
no extractable OHLCV feature. SKIPPED as-is.

SUBSTITUTE (same theme — fractional-order / long-memory dynamics):
  Implements FRACTIONAL DIFFERENCING of price series.

  Fractional differencing (Hosking 1981, popularised for finance by de Prado 2018)
  applies a fractional-order difference operator d ∈ (0,1) to the log-price series.
  Unlike integer differencing (d=1 → log returns, all memory lost) or no differencing
  (d=0 → levels, non-stationary), fractional differencing at d≈0.3–0.5 preserves
  *long-memory* while still achieving approximate stationarity.

  The Grünwald-Letnikov discretisation:
      Δ^d x_t = Σ_{k=0}^{T} w_k * x_{t-k}
  where w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k

  Feature family:
    frac_diff_d025   — fractionally-differenced log-close, d=0.25 (long memory)
    frac_diff_d040   — d=0.40 (balanced)
    frac_diff_d060   — d=0.60 (closer to stationary)
    frac_diff_mom20  — 20-day momentum of frac_diff_d040 (trend in the fd signal)
    frac_diff_zscore — z-score of frac_diff_d040 over trailing 60 days
    frac_diff_sign   — sign persistence: rolling 10-day sign-run length of frac_diff_d040
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_12895_fractional_diff",
    "description": (
        "Grünwald-Letnikov fractional differencing of log-price at multiple orders d "
        "to preserve long-memory signal; inspired by fractional-order state-space "
        "modelling in arxiv 2606.12895 (LongSpike f-SSM)."
    ),
    "requires": ["Close"],
    "produces": [
        "frac_diff_d025",
        "frac_diff_d040",
        "frac_diff_d060",
        "frac_diff_mom20",
        "frac_diff_zscore",
        "frac_diff_sign_run",
    ],
    "tags": ["momentum", "trend", "experimental"],
    "version": "1.0",
    "author": "paper:2606.12895",
}


def _frac_diff_weights(d: float, threshold: float = 1e-5, max_lags: int = 200) -> np.ndarray:
    """Compute Grünwald-Letnikov weights for fractional difference order d."""
    w = [1.0]
    for k in range(1, max_lags):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w)


def _apply_frac_diff(log_price: np.ndarray, d: float) -> np.ndarray:
    """Apply fractional differencing to a 1-D array of log prices."""
    weights = _frac_diff_weights(d)
    n_lags = len(weights)
    n = len(log_price)
    result = np.full(n, np.nan)

    for i in range(n_lags - 1, n):
        window = log_price[i - n_lags + 1 : i + 1][::-1]   # most-recent first
        if np.any(~np.isfinite(window)):
            continue
        result[i] = float(weights @ window)

    return result


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_price = np.log(df["Close"].replace(0, np.nan).values.astype(float))

    # ---- Core fractional-difference series at three orders --------------------
    fd025 = _apply_frac_diff(log_price, d=0.25)
    fd040 = _apply_frac_diff(log_price, d=0.40)
    fd060 = _apply_frac_diff(log_price, d=0.60)

    df["frac_diff_d025"] = fd025
    df["frac_diff_d040"] = fd040
    df["frac_diff_d060"] = fd060

    # ---- 20-day momentum of the d=0.40 signal ---------------------------------
    s040 = pd.Series(fd040, index=df.index)
    df["frac_diff_mom20"] = s040 - s040.shift(20)

    # ---- Z-score of d=0.40 over trailing 60 days ------------------------------
    mu60 = s040.rolling(60, min_periods=20).mean()
    sd60 = s040.rolling(60, min_periods=20).std()
    df["frac_diff_zscore"] = (s040 - mu60) / sd60.replace(0, np.nan)

    # ---- Sign-run length: how many consecutive same-sign days? ----------------
    signs = np.sign(fd040)

    def _run_length(arr: np.ndarray) -> np.ndarray:
        """Rolling 10-day sign-run length (length of current run up to today)."""
        n = len(arr)
        result = np.full(n, np.nan)
        for i in range(1, n):
            if not np.isfinite(arr[i]):
                continue
            run = 1
            for j in range(i - 1, max(i - 10, -1), -1):
                if np.isfinite(arr[j]) and np.sign(arr[j]) == np.sign(arr[i]):
                    run += 1
                else:
                    break
            result[i] = run
        return result

    df["frac_diff_sign_run"] = _run_length(signs)

    return df
