from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ff06280034a_freq_stability_allan_intraday_80",
    "description": (
        "Allan deviation (tau=5) over a rolling 80-day window of intraday returns "
        "ln(Close/Open). Non-overlapping tau-blocks of size 5 are formed from the 80-day "
        "window, block means are computed, then Allan deviation = "
        "sqrt(0.5 * mean(diff(block_means)^2)). Produces: level (allan_dev), ratio to "
        "rolling 80d std of intraday return (allan_rel), and 20d z-score of the level "
        "(allan_z). Per-ticker proxy for frequency stability / oscillator noise; orthogonal "
        "to amplitude-only volatility. Because true Allan deviation is defined for "
        "equally-spaced time-domain measurements, using daily intraday returns as the "
        "measurement sequence is a faithful daily-bar proxy."
    ),
    "requires": ["Open", "Close"],
    "produces": [
        "ff06280034a_freq_stability_allan_intraday_80_dev",
        "ff06280034a_freq_stability_allan_intraday_80_rel",
        "ff06280034a_freq_stability_allan_intraday_80_z",
    ],
    "tags": ["volatility", "frequency_stability", "allan_deviation", "intraday"],
    "version": "1.0.0",
    "author": "feature-factory",
}

_COL_DEV = "ff06280034a_freq_stability_allan_intraday_80_dev"
_COL_REL = "ff06280034a_freq_stability_allan_intraday_80_rel"
_COL_Z   = "ff06280034a_freq_stability_allan_intraday_80_z"

_WIN   = 80   # rolling window in bars
_TAU   = 5    # block size for Allan deviation
_N_BLOCKS = _WIN // _TAU   # = 16 non-overlapping blocks inside the window
_Z_WIN = 20   # z-score lookback


def _allan_dev_from_window(arr: np.ndarray) -> float:
    """
    Compute Allan deviation at tau from a 1-D array of length WIN.
    arr must have exactly WIN elements.
    Non-overlapping blocks of size TAU -> block means -> Allan dev.
    """
    # reshape into (N_BLOCKS, TAU), take mean of each block
    blocks = arr.reshape(_N_BLOCKS, _TAU)
    block_means = blocks.mean(axis=1)
    diffs = np.diff(block_means)
    if len(diffs) == 0:
        return np.nan
    val = 0.5 * np.mean(diffs ** 2)
    if val < 0.0 or not np.isfinite(val):
        return np.nan
    return np.sqrt(val)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN (required on every code path)
    df[_COL_DEV] = np.nan
    df[_COL_REL] = np.nan
    df[_COL_Z]   = np.nan

    n = len(df)
    if n < _WIN:
        return df

    # Intraday log return: ln(Close/Open)
    with np.errstate(divide="ignore", invalid="ignore"):
        intra = np.log(df["Close"].values / df["Open"].values)
    intra = np.where(np.isfinite(intra), intra, np.nan)

    dev_arr = np.full(n, np.nan)
    std_arr = np.full(n, np.nan)

    for i in range(_WIN - 1, n):
        window = intra[i - _WIN + 1 : i + 1]  # length WIN
        if np.any(~np.isfinite(window)):
            # Skip if any NaN in the window (conservative; avoids silent reshape errors)
            continue
        dev_arr[i] = _allan_dev_from_window(window)
        s = window.std()
        std_arr[i] = s if s > 0.0 else np.nan

    df[_COL_DEV] = dev_arr

    # Ratio: allan_dev / rolling_std (normalises by amplitude volatility)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = dev_arr / std_arr
    df[_COL_REL] = np.where(np.isfinite(rel), rel, np.nan)

    # 20d z-score of the allan deviation level
    dev_series = pd.Series(dev_arr)
    roll_mean = dev_series.rolling(_Z_WIN, min_periods=_Z_WIN).mean()
    roll_std  = dev_series.rolling(_Z_WIN, min_periods=_Z_WIN).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (dev_series - roll_mean) / roll_std
    df[_COL_Z] = np.where(np.isfinite(z.values), z.values, np.nan)

    return df
