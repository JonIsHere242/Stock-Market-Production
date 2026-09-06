"""
_cand_ff0703c_path_geometry_vertex_angle_sharpness_mean_60.py

Path-geometry: mean interior vertex angle at turning points (sharpness of
turns) over a rolling 60-bar window.

For each 60-bar window, build a normalised 2D path: x = unit time step
(0,1,2,...,59), y = Close / window-mean(Close).  An interior point i
(1 <= i <= 58 within the window) is a "turn" (local extremum) if y[i] is a
local max or local min relative to its immediate neighbours.  At each turn,
form the two edge vectors that point AWAY from the vertex toward its
neighbours:
    v1 = (x[i-1]-x[i], y[i-1]-y[i])
    v2 = (x[i+1]-x[i], y[i+1]-y[i])
and compute the interior vertex angle theta = arccos( (v1.v2) / (|v1||v2|) ).
This is the standard "polygon interior angle" convention: a very sharp spike
(V-shaped reversal) collapses the two edges toward the same direction and
yields a SMALL angle (near 0); a shallow, almost-straight wiggle yields an
angle near pi.  The feature is the mean of theta over all turns in the
window (sharper turns on average -> smaller mean angle).  Requires >= 3
turns in the window, else NaN.  Requires >= 60 bars, else NaN.

A secondary column reports the fraction of turns that are "sharp"
(theta < pi/2), which captures turn-sharpness DENSITY as a distinct axis
from the mean-magnitude level column (two windows can share the same mean
angle while differing in how many turns are genuinely acute vs. how many
are shallow wiggles).

PROXY NOTE: this is a pure per-ticker OHLCV computation, fully faithful to
the spec's definition (no cross-sectional or external data needed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ff0703c_path_geometry_vertex_angle_sharpness_mean_60",
    "description": (
        "Rolling 60-bar path geometry: normalises the price path within the "
        "window as (unit time step, Close/window-mean), locates interior "
        "local-extremum 'turning points', and computes the interior vertex "
        "angle at each turn via the dot product of the two edge vectors "
        "pointing from the vertex to its neighbours (polygon-interior-angle "
        "convention: sharper spike/V-reversal => smaller angle, near-straight "
        "wiggle => angle near pi). Feature = mean interior angle (radians) "
        "over all turns in the window; NaN if fewer than 3 turns or fewer "
        "than 60 bars. Secondary column = fraction of turns classified "
        "'sharp' (angle < pi/2), capturing turn-sharpness density as a "
        "distinct axis from the mean-magnitude level."
    ),
    "requires": ["Close"],
    "produces": [
        "ff0703c_path_geometry_vertex_angle_sharpness_mean_60",       # mean interior vertex angle (radians), level
        "ff0703c_path_geometry_vertex_angle_sharp_frac_60",           # fraction of turns with angle < pi/2
    ],
    "tags": ["path-geometry", "vertex-angle", "turning-points", "sharpness", "ohlcv"],
    "version": "1.0.0",
    "author": (
        "Spec: ff0703c_path_geometry_vertex_angle_sharpness_mean_60 (path_geometry vein). "
        "Faithful per-ticker implementation (no proxy needed -- pure OHLCV geometry). Impl: Claude."
    ),
}

_WINDOW = 60
_MIN_TURNS = 3
_EPS = 1e-12


def _rolling_vertex_angle(close_arr: np.ndarray, window: int):
    n = len(close_arr)
    mean_angle = np.full(n, np.nan)
    sharp_frac = np.full(n, np.nan)

    if n < window:
        return mean_angle, sharp_frac

    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(close_arr, window_shape=window)  # (n_win, window)
    n_win = windows.shape[0]

    win_mean = windows.mean(axis=1, keepdims=True)
    win_mean_safe = np.where(np.abs(win_mean) < _EPS, np.nan, win_mean)
    y = windows / win_mean_safe  # normalised path, shape (n_win, window)

    # interior points i = 1 .. window-2 (0-indexed), x-step is always 1
    y_prev = y[:, 0:window - 2]
    y_curr = y[:, 1:window - 1]
    y_next = y[:, 2:window]

    dy1 = y_prev - y_curr  # v1 y-component (v1 x-component = -1)
    dy2 = y_next - y_curr  # v2 y-component (v2 x-component = +1)

    dot = (-1.0) * 1.0 + dy1 * dy2
    norm1 = np.sqrt(1.0 + dy1 ** 2)
    norm2 = np.sqrt(1.0 + dy2 ** 2)
    denom = norm1 * norm2
    denom_safe = np.where(denom < _EPS, np.nan, denom)

    cos_theta = np.clip(dot / denom_safe, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # (n_win, window-2)

    is_extremum = ((y_curr > y_prev) & (y_curr > y_next)) | ((y_curr < y_prev) & (y_curr < y_next))

    theta_masked = np.where(is_extremum, theta, np.nan)
    turn_count = is_extremum.sum(axis=1)

    with np.errstate(invalid="ignore"):
        mean_theta = np.nanmean(theta_masked, axis=1)
        sharp_mask = is_extremum & (theta < (np.pi / 2.0))
        sharp_count = sharp_mask.sum(axis=1)
        frac = np.divide(
            sharp_count.astype(np.float64),
            turn_count.astype(np.float64),
            out=np.full(n_win, np.nan),
            where=turn_count > 0,
        )

    valid = turn_count >= _MIN_TURNS
    mean_theta = np.where(valid, mean_theta, np.nan)
    frac = np.where(valid, frac, np.nan)

    mean_angle[window - 1:] = mean_theta
    sharp_frac[window - 1:] = frac

    return mean_angle, sharp_frac


def compute(df: pd.DataFrame) -> pd.DataFrame:
    df["ff0703c_path_geometry_vertex_angle_sharpness_mean_60"] = np.nan
    df["ff0703c_path_geometry_vertex_angle_sharp_frac_60"] = np.nan

    if len(df) < _WINDOW:
        return df

    close_arr = df["Close"].to_numpy(dtype=np.float64)
    mean_angle, sharp_frac = _rolling_vertex_angle(close_arr, _WINDOW)

    df["ff0703c_path_geometry_vertex_angle_sharpness_mean_60"] = mean_angle
    df["ff0703c_path_geometry_vertex_angle_sharp_frac_60"] = sharp_frac

    return df
