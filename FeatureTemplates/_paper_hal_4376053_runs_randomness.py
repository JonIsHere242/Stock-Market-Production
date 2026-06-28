"""
_paper_hal_4376053_runs_randomness.py  --  Wald-Wolfowitz runs-test randomness suite.

Paper: "Predicting the unpredictable: New experimental evidence on forecasting
random walks" (HAL 4376053).  The study shows subjects use autocorrelation,
amplitude, and reversal count as predictability cues, and that perceived-trend
charts lead to more trend-chasing behaviour.

This block operationalises the key cue — the NUMBER OF RUNS of same-sign returns
over a trailing 60-day window — via the classical Wald-Wolfowitz runs test:
  * run_count_60        : total number of runs (maximal same-sign streaks)
  * run_z_60            : z-statistic  ( <0 = trending/persistent,
                                          >0 = choppy/mean-reverting )
  * run_longest_up_60   : length of the longest consecutive up-streak in window
  * run_longest_down_60 : length of the longest consecutive down-streak in window
  * run_up_fraction_60  : fraction of days with positive returns in window
  * run_current_streak  : signed length of the run ending at t
                          (positive = up-streak, negative = down-streak)

Zero returns are folded into the preceding sign (or treated as positive on the
first observation in a window with no prior sign).  This is the "carry-forward"
convention: a zero return does not break a run.

Degenerate windows where all days are up or all are down yield Var[R] = 0;
run_z_60 is set to NaN in those cases.

Window = 60 bars.  First 59 rows produce NaN for all outputs (expected).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name":        "paper_hal_4376053_runs_randomness",
    "description": (
        "Wald-Wolfowitz runs-test suite (60-day trailing window) quantifying "
        "how random vs trend-persistent the recent return path appears, motivated "
        "by HAL-4376053 experimental evidence on chart-pattern predictability cues."
    ),
    "requires":    ["Close"],
    "produces":    [
        "run_count_60",
        "run_z_60",
        "run_longest_up_60",
        "run_longest_down_60",
        "run_up_fraction_60",
        "run_current_streak",
    ],
    "tags":        ["mean_reversion", "momentum", "randomness", "experimental"],
    "version":     "1.0",
    "author":      "paper HAL 4376053 — runs-test randomness block",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_WINDOW = 60


def _sign_with_carry(ret_arr: np.ndarray) -> np.ndarray:
    """
    Convert a 1-D array of returns into signs {-1, +1}, carrying the last
    known sign forward through zeros so that zeros never break a run.
    The first element defaults to +1 if it is zero.
    """
    n = len(ret_arr)
    signs = np.sign(ret_arr).astype(float)

    # default: first zero -> +1
    if n > 0 and signs[0] == 0.0:
        signs[0] = 1.0

    # carry forward
    for i in range(1, n):
        if signs[i] == 0.0:
            signs[i] = signs[i - 1]

    return signs


def _runs_stats(signs: np.ndarray):
    """
    Compute Wald-Wolfowitz runs statistics from a sign array of {-1, +1}.

    Returns:
        (n_runs, z_stat, longest_up, longest_down, up_fraction)
    """
    n = len(signs)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    n_up = float(np.sum(signs > 0))
    n_dn = float(np.sum(signs < 0))
    total = n_up + n_dn
    if total == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Count runs: a run changes whenever sign differs from previous
    run_changes = np.count_nonzero(signs[1:] != signs[:-1])
    n_runs = float(run_changes + 1)

    # Up-fraction
    up_frac = n_up / total

    # Longest up-run and down-run
    longest_up = 0
    longest_dn = 0
    cur_up = 0
    cur_dn = 0
    for s in signs:
        if s > 0:
            cur_up += 1
            cur_dn = 0
            if cur_up > longest_up:
                longest_up = cur_up
        else:
            cur_dn += 1
            cur_up = 0
            if cur_dn > longest_dn:
                longest_dn = cur_dn

    # Runs-test expected value and variance
    if n_up == 0 or n_dn == 0:
        # All-same-sign => only 1 run, Var = 0 => z undefined
        return n_runs, np.nan, float(longest_up), float(longest_dn), up_frac

    e_r = 2.0 * n_up * n_dn / total + 1.0
    var_r_num = 2.0 * n_up * n_dn * (2.0 * n_up * n_dn - total)
    var_r_den = total * total * (total - 1.0)
    if var_r_den <= 0.0 or var_r_num <= 0.0:
        z = np.nan
    else:
        var_r = var_r_num / var_r_den
        std_r = np.sqrt(var_r)
        z = (n_runs - e_r) / std_r if std_r > 0.0 else np.nan

    return n_runs, z, float(longest_up), float(longest_dn), up_frac


def _current_streak(signs: np.ndarray) -> float:
    """
    Length of the maximal same-sign run ending at the last element.
    Positive value = up-streak, negative = down-streak.
    """
    if len(signs) == 0:
        return np.nan
    last_sign = signs[-1]
    count = 0
    for s in reversed(signs):
        if s == last_sign:
            count += 1
        else:
            break
    return float(last_sign * count)


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Wald-Wolfowitz runs-test columns over a trailing 60-bar window.

    All outputs are NaN for the first (_WINDOW - 1) rows where there is
    insufficient history to fill the full window.
    """
    n = len(df)
    close = df["Close"].to_numpy(dtype=np.float64)

    # Daily log-returns (row 0 is NaN — handled below)
    ret = np.empty(n, dtype=np.float64)
    ret[0] = 0.0  # no prior bar; treat as no-change (will carry forward as +1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret[1:] = np.where(close[:-1] > 0, close[1:] / close[:-1] - 1.0, np.nan)

    # Replace NaN returns with 0 so they get absorbed by the carry-forward rule
    ret = np.where(np.isfinite(ret), ret, 0.0)

    # Pre-build sign array for the full series (carry-forward zeros)
    signs_full = _sign_with_carry(ret)

    # Output arrays
    out_count   = np.full(n, np.nan)
    out_z       = np.full(n, np.nan)
    out_long_up = np.full(n, np.nan)
    out_long_dn = np.full(n, np.nan)
    out_up_frac = np.full(n, np.nan)
    out_streak  = np.full(n, np.nan)

    # Rolling window: require full _WINDOW rows (indices 0..t inclusive, length W)
    for t in range(_WINDOW - 1, n):
        window_signs = signs_full[t - _WINDOW + 1 : t + 1]  # length == _WINDOW

        n_r, z, lu, ld, uf = _runs_stats(window_signs)
        out_count[t]   = n_r
        out_z[t]       = z
        out_long_up[t] = lu
        out_long_dn[t] = ld
        out_up_frac[t] = uf
        out_streak[t]  = _current_streak(window_signs)

    idx = df.index
    df["run_count_60"]        = pd.array(out_count,   dtype="float64")
    df["run_z_60"]            = pd.array(out_z,       dtype="float64")
    df["run_longest_up_60"]   = pd.array(out_long_up, dtype="float64")
    df["run_longest_down_60"] = pd.array(out_long_dn, dtype="float64")
    df["run_up_fraction_60"]  = pd.array(out_up_frac, dtype="float64")
    df["run_current_streak"]  = pd.array(out_streak,  dtype="float64")

    return df
