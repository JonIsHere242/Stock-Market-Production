"""
_paper_2605_22215_path_signature.py  --  Path signature features (level-2 truncated).

Inspired by arXiv 2605.22215 "A Generative Adversarial Graph Neural Network for Synthetic
Time Series Data" (Sig-Graph GAN), which embeds the rough-path SIGNATURE as a structured
temporal summary via iterated integrals. Only the signature transform is implemented here;
the GAN, LSTM, and visibility-graph components are NOT.

CONCEPT
-------
A 2-D path X = (X^1, X^2) parameterised over a window of n steps has a level-2 signature
whose off-diagonal iterated integrals carry real structural signal:

    S^{ij}(X) = integral_s<t  dX^i_s * dX^j_t

For discrete increments dx^i_k, dx^j_k  (k = 1..n):

    S^{12} = sum_{s<t}  dx^1_s * dx^2_t    (accumulated cross-integral, lower-triangular)
    S^{21} = sum_{s<t}  dx^2_s * dx^1_t

Levy area = 0.5*(S^{12} - S^{21})   -- rotation / winding of path, antisymmetric.

The diagonal terms S^{11} and S^{22} are trivially 0.5*(sum dX^i)^2 so they are omitted.

Two path choices are used:
  pv  = (log_close_increment, log_volume_increment)
  pt  = (log_close_increment, normalized_time)       -- time as uniformly-spaced increments

Rolling windows: 20 and 60 bars (trailing, look-ahead free).
"""

import numpy as np
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
METADATA = {
    "name":        "path_signature",
    "description": (
        "Level-2 path signature features (Levy area, off-diagonal iterated integrals, "
        "path length) on (log_close, log_volume) and (log_close, time) paths over "
        "rolling 20- and 60-bar windows."
    ),
    "requires":    ["Close", "Volume"],
    "produces":    [
        "sig_s12_pv_20",
        "sig_s21_pv_20",
        "sig_levy_area_pv_20",
        "sig_path_length_20",
        "sig_s12_pv_60",
        "sig_s21_pv_60",
        "sig_levy_area_pv_60",
        "sig_path_length_60",
        "sig_levy_area_pt_60",
    ],
    "tags":        ["momentum", "volatility", "experimental", "signature"],
    "version":     "1.0",
    "author":      "paper arXiv 2605.22215 — path signature transform only",
}


# ---------------------------------------------------------------------------
# Core computation helpers (operate on raw numpy arrays, no pandas overhead)
# ---------------------------------------------------------------------------

def _levy_area_and_s12_s21(dx: np.ndarray, dy: np.ndarray) -> tuple:
    """
    Compute the level-2 off-diagonal iterated integrals S12, S21 and Levy area
    for a single window segment represented by increment arrays dx, dy.

    S^{12} = sum_{0<=s<t<n}  dx[s] * dy[t]
           = sum_t  dy[t] * (sum_{s<t} dx[s])
           = sum_t  dy[t] * cumsum_dx[t-1]   (cumsum shifted by one)

    Similarly S^{21} = sum_t  dx[t] * cumsum_dy[t-1]

    Returns (s12, s21, levy_area=0.5*(s12-s21)).
    """
    n = len(dx)
    if n < 2:
        return np.nan, np.nan, np.nan

    # cumulative sum of dx/dy up to (but not including) position t
    cum_dx = np.empty(n, dtype=np.float64)
    cum_dy = np.empty(n, dtype=np.float64)
    cum_dx[0] = 0.0
    cum_dy[0] = 0.0
    for i in range(1, n):
        cum_dx[i] = cum_dx[i - 1] + dx[i - 1]
        cum_dy[i] = cum_dy[i - 1] + dy[i - 1]

    s12 = float(np.dot(cum_dx, dy))
    s21 = float(np.dot(cum_dy, dx))
    levy = 0.5 * (s12 - s21)
    return s12, s21, levy


def _path_length(dx: np.ndarray, dy: np.ndarray) -> float:
    """Euclidean path length in (dx, dy) increment space."""
    norms = np.sqrt(dx ** 2 + dy ** 2)
    return float(np.sum(norms))


def _rolling_signature(
    inc1: np.ndarray,
    inc2: np.ndarray,
    window: int,
    want_s12: bool = True,
    want_s21: bool = True,
    want_levy: bool = True,
    want_length: bool = False,
) -> tuple:
    """
    Compute rolling level-2 signature over trailing windows of `window` increments.

    The increment arrays have length (n - 1) where n = len(price series).
    Output arrays have length n with the first `window` rows NaN (need `window`
    increments, meaning `window + 1` prices).

    Returns tuple of requested arrays in order: (s12, s21, levy, length).
    Missing outputs are returned as None.
    """
    m = len(inc1)          # number of increments = n_rows - 1
    n_out = m + 1          # output aligns with original price rows

    out_s12    = np.full(n_out, np.nan) if want_s12 else None
    out_s21    = np.full(n_out, np.nan) if want_s21 else None
    out_levy   = np.full(n_out, np.nan) if want_levy else None
    out_length = np.full(n_out, np.nan) if want_length else None

    # Iterate: position i in the price series corresponds to increment index i-1.
    # A trailing window of `window` increments covers indices [i-window .. i-1].
    # Result is placed at price-row i.
    for i in range(window, n_out):
        dx = inc1[i - window: i]
        dy = inc2[i - window: i]

        # Guard: degenerate window (all NaN or constant)
        if not np.all(np.isfinite(dx)) or not np.all(np.isfinite(dy)):
            continue

        s12, s21, levy = _levy_area_and_s12_s21(dx, dy)
        if want_s12:
            out_s12[i] = s12
        if want_s21:
            out_s21[i] = s21
        if want_levy:
            out_levy[i] = levy
        if want_length:
            out_length[i] = _path_length(dx, dy)

    return out_s12, out_s21, out_levy, out_length


# ---------------------------------------------------------------------------
# Vectorized variant for path length (cheaper via rolling apply)
# ---------------------------------------------------------------------------

def _rolling_path_length_vec(dx: np.ndarray, dy: np.ndarray, window: int) -> np.ndarray:
    """
    Vectorized path length via pandas rolling sum of ||(dx_t, dy_t)||.

    inc arrays have length n-1; output is length n (row 0 .. window are NaN).
    """
    norms = np.sqrt(dx ** 2 + dy ** 2)
    s = pd.Series(norms).rolling(window, min_periods=window).sum().to_numpy()
    # Prepend a NaN so output aligns with price rows (length = n_increments + 1 = n_prices)
    return np.concatenate([[np.nan], s])


# ---------------------------------------------------------------------------
# Public compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add level-2 path signature columns to df.

    Path 1 (pv): channels = (log_close_increment, log_volume_increment)
    Path 2 (pt): channels = (log_close_increment, normalized_time_increment)

    Windows: 20, 60 bars (trailing, look-ahead free).
    """
    n = len(df)

    close  = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    # Guard: replace non-positive with NaN before log
    close_safe  = np.where(close  > 0, close,  np.nan)
    volume_safe = np.where(volume > 0, volume, np.nan)

    # Log-increments (shape n-1)
    log_close  = np.log(close_safe)
    log_volume = np.log(volume_safe)

    dlc = np.diff(log_close)    # log-return increments, length n-1
    dlv = np.diff(log_volume)   # log-volume increments, length n-1

    # Normalized time increments: each step = 1/(window) so that the time
    # channel has unit range over any window.  For rolling purposes we use
    # a fixed dt = 1/60 (longest window) — the Levy area is scale-sensitive
    # but the RELATIVE signal is preserved within each window length.
    dt_20 = np.full(n - 1, 1.0 / 20.0)
    dt_60 = np.full(n - 1, 1.0 / 60.0)

    new: dict = {}

    # ------------------------------------------------------------------
    # Window 20: pv path  (s12, s21, levy_area)
    # ------------------------------------------------------------------
    if n > 20:
        s12_20, s21_20, levy_20, _ = _rolling_signature(
            dlc, dlv, window=20,
            want_s12=True, want_s21=True, want_levy=True, want_length=False,
        )
        new["sig_s12_pv_20"]       = s12_20
        new["sig_s21_pv_20"]       = s21_20
        new["sig_levy_area_pv_20"] = levy_20

        # Path length (vectorized)
        new["sig_path_length_20"] = _rolling_path_length_vec(dlc, dlv, 20)
    else:
        for col in ("sig_s12_pv_20", "sig_s21_pv_20",
                    "sig_levy_area_pv_20", "sig_path_length_20"):
            new[col] = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # Window 60: pv path  (s12, s21, levy_area, path_length)
    # ------------------------------------------------------------------
    if n > 60:
        s12_60, s21_60, levy_60, _ = _rolling_signature(
            dlc, dlv, window=60,
            want_s12=True, want_s21=True, want_levy=True, want_length=False,
        )
        new["sig_s12_pv_60"]       = s12_60
        new["sig_s21_pv_60"]       = s21_60
        new["sig_levy_area_pv_60"] = levy_60

        new["sig_path_length_60"] = _rolling_path_length_vec(dlc, dlv, 60)
    else:
        for col in ("sig_s12_pv_60", "sig_s21_pv_60",
                    "sig_levy_area_pv_60", "sig_path_length_60"):
            new[col] = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # Window 60: pt path  (log_close, time)  — Levy area only
    # Only compute if window 60 is reachable.
    # ------------------------------------------------------------------
    if n > 60:
        _, _, levy_pt_60, _ = _rolling_signature(
            dlc, dt_60, window=60,
            want_s12=False, want_s21=False, want_levy=True, want_length=False,
        )
        new["sig_levy_area_pt_60"] = levy_pt_60
    else:
        new["sig_levy_area_pt_60"] = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # Assign all new columns; guard against inf (replace with NaN)
    # ------------------------------------------------------------------
    for col, arr in new.items():
        series = pd.Series(arr, index=df.index, dtype=np.float64)
        series = series.replace([np.inf, -np.inf], np.nan)
        df[col] = series

    return df
