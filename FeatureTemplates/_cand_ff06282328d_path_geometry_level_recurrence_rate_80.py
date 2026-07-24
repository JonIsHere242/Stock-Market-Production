from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ff06282328d_path_geometry_level_recurrence_rate_80",
    "description": (
        "Price-level recurrence rate over a rolling 80-bar window. "
        "The window's close range is divided into ~10 equal-width bins; each bar's "
        "close is mapped to one bin. Recurrence rate = fraction of bars whose bin "
        "was already occupied by an earlier bar in the window. High rate => "
        "price oscillates and revisits prior levels (range-bound). Low rate => "
        "price keeps discovering fresh ground (trending). A slope variant "
        "(current 20-bar rate minus 20-bar rate 20 bars ago) captures "
        "whether the stock is becoming more/less range-bound recently."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282328d_path_geometry_level_recurrence_rate_80_rate",
        "ff06282328d_path_geometry_level_recurrence_rate_80_slope",
    ],
    "tags": ["path_geometry", "recurrence", "range_bound", "trending", "price_levels"],
    "version": "1.0.0",
    "author": "feature-factory ff06282328d",
}

# Number of price bins
_N_BINS = 10
# Main window length
_WINDOW = 80
# Short window for slope computation
_SHORT = 20


def _recurrence_rate_1d(closes: np.ndarray, n_bins: int = _N_BINS) -> float:
    """
    Compute recurrence rate for a 1-D array of closes (one window).
    Returns fraction of bars [1..n-1] whose bin was already occupied by an
    earlier bar [0..i-1]. Returns NaN if range is zero or length < 2.
    """
    n = len(closes)
    if n < 2:
        return np.nan
    lo = closes.min()
    hi = closes.max()
    rng = hi - lo
    if rng == 0.0:
        # All bars in same bin => every bar after the first is a recurrence
        return 1.0
    # Map each close to a bin index [0, n_bins-1]
    # Use clip to handle the max value (would otherwise map to n_bins)
    bin_indices = np.floor((closes - lo) / rng * n_bins).astype(np.int32)
    np.clip(bin_indices, 0, n_bins - 1, out=bin_indices)

    seen = np.zeros(n_bins, dtype=np.bool_)
    recurrences = 0
    seen[bin_indices[0]] = True
    for i in range(1, n):
        b = bin_indices[i]
        if seen[b]:
            recurrences += 1
        else:
            seen[b] = True
    return recurrences / (n - 1)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    col_rate = "ff06282328d_path_geometry_level_recurrence_rate_80_rate"
    col_slope = "ff06282328d_path_geometry_level_recurrence_rate_80_slope"

    # Initialise outputs to NaN on every code path
    df[col_rate] = np.nan
    df[col_slope] = np.nan

    n = len(df)
    if n < 2:
        return df

    closes = df["Close"].to_numpy(dtype=np.float64)

    # -- Main 80-bar rolling recurrence rate --
    rate_arr = np.full(n, np.nan)
    for i in range(_WINDOW - 1, n):
        window = closes[i - _WINDOW + 1 : i + 1]
        rate_arr[i] = _recurrence_rate_1d(window)

    df[col_rate] = rate_arr

    # -- Slope: 20-bar rate minus 20-bar rate 20 bars ago --
    # Compute 20-bar recurrence rate at each bar, then diff by 20
    short_arr = np.full(n, np.nan)
    for i in range(_SHORT - 1, n):
        window = closes[i - _SHORT + 1 : i + 1]
        short_arr[i] = _recurrence_rate_1d(window, n_bins=_N_BINS)

    slope_arr = np.full(n, np.nan)
    for i in range(_SHORT, n):
        if not np.isnan(short_arr[i]) and not np.isnan(short_arr[i - _SHORT]):
            slope_arr[i] = short_arr[i] - short_arr[i - _SHORT]

    df[col_slope] = slope_arr

    return df
