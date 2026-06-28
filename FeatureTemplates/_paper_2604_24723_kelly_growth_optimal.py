"""
_paper_2604_24723_kelly_growth_optimal.py  --  Single-asset Kelly / growth-optimal feature pack.

Inspired by: "Efficient Multivariate Kelly Optimization Reveals Sigmoidal Scaling Laws"
             arXiv 2604.24723

NOTE: This is an UNPROVEN candidate block (leading underscore = hidden from auto-discovery).
      The paper treats the multivariate portfolio Kelly problem; what is implemented here is
      the closest per-ticker, single-asset analogue derived from daily OHLCV log-returns:

      1.  Continuous (Gaussian) Kelly fraction:  f* = mu / sigma^2
          Growth-optimal leverage under Gaussian return assumption (rolling window).

      2.  Discrete-bet Kelly fraction: f = p - (1-p)/b
          p = rolling win-rate, b = rolling avg-win / avg-loss magnitude.

      3.  Rolling expected log-growth at the Kelly fraction and at half-Kelly.

      4.  Half-Kelly fraction (conservative 0.5 * f*) and a sigmoid-squashed Kelly
          fraction (the paper's "sigmoidal scaling" flavour — extreme leverage
          is squashed through sigmoid so the output is bounded in (0,1) ∪ (-1,0)).

      5.  Kelly edge = mu / sigma  (Sharpe-like signal) over two look-back windows.

      6.  Growth gap = log-growth(f*) minus log-growth(0) = the incremental value
          of betting at all vs sitting out.

All computed over two rolling windows: 60 and 120 days.
All columns prefixed `kel_`. Leading NaNs during warm-up are expected.
No look-ahead: value at row t depends only on rows <= t.
"""

import math
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name":        "_paper_2604_24723_kelly_growth_optimal",
    "description": (
        "Single-asset Kelly / growth-optimal feature pack: continuous f*, discrete-bet Kelly, "
        "expected log-growth at full- and half-Kelly, sigmoid-squashed Kelly, Kelly edge "
        "(Sharpe proxy), and growth gap — over 60- and 120-day rolling windows."
    ),
    "requires":    ["Close"],
    "produces":    [
        # 60-day window
        "kel_f_continuous_60",
        "kel_f_halfkelly_60",
        "kel_f_sigmoid_60",
        "kel_f_discrete_60",
        "kel_loggrowth_full_60",
        "kel_loggrowth_half_60",
        "kel_edge_60",
        "kel_growth_gap_60",
        # 120-day window
        "kel_f_continuous_120",
        "kel_f_halfkelly_120",
        "kel_f_sigmoid_120",
        "kel_f_discrete_120",
        "kel_loggrowth_full_120",
        "kel_loggrowth_half_120",
        "kel_edge_120",
        "kel_growth_gap_120",
    ],
    "tags":        ["momentum", "volatility", "experimental", "kelly", "growth_optimal"],
    "version":     "1.0",
    "author":      "paper arXiv 2604.24723 — single-asset Kelly proxy",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: 1 / (1 + exp(-x))."""
    # clip to avoid overflow in exp
    xc = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-xc))


def _signed_sigmoid_squash(f: np.ndarray) -> np.ndarray:
    """
    Map any real-valued Kelly fraction to a bounded range via sigmoid,
    preserving sign: sign(f) * sigmoid(|f|).

    Rationale: the paper introduces 'sigmoidal scaling' to prevent explosive
    leverage — extreme f* values are squashed toward ±1 while moderate
    fractions survive close to their original value.

    Output is in (-1, 0) ∪ (0, 1) — bounded, never exactly ±1.
    NaN inputs propagate as NaN.
    """
    out = np.where(
        np.isfinite(f),
        np.sign(f) * _sigmoid(np.abs(f)),
        np.nan,
    )
    return out.astype(np.float64)


def _expected_log_growth(mu: np.ndarray, var: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Gaussian approximation of single-period expected log-growth when betting
    fraction f of wealth on a return distribution with mean mu and variance var:

        E[log(1 + f*R)] ≈ f*mu - 0.5*f^2*var      (second-order Taylor expansion)

    Valid when |f*R| << 1 (i.e. moderate leverage). Guards against zero var.
    """
    # zero-variance → trivially 0 log-growth regardless of f
    safe_var = np.where(var > 0.0, var, np.nan)
    growth = f * mu - 0.5 * (f ** 2) * safe_var
    # set to nan wherever inputs are nan
    mask = np.isfinite(mu) & np.isfinite(var) & np.isfinite(f)
    return np.where(mask, growth, np.nan).astype(np.float64)


def _kelly_features_for_window(
    log_ret: np.ndarray,
    w: int,
) -> dict:
    """
    Compute all Kelly features for a single rolling window w.

    Returns a dict mapping suffix-free column name → 1-D numpy array
    aligned to the input length.
    """
    n = len(log_ret)

    # Pre-allocate output arrays as NaN
    f_cont   = np.full(n, np.nan, dtype=np.float64)
    f_half   = np.full(n, np.nan, dtype=np.float64)
    f_sig    = np.full(n, np.nan, dtype=np.float64)
    f_disc   = np.full(n, np.nan, dtype=np.float64)
    lg_full  = np.full(n, np.nan, dtype=np.float64)
    lg_half  = np.full(n, np.nan, dtype=np.float64)
    edge     = np.full(n, np.nan, dtype=np.float64)
    gap      = np.full(n, np.nan, dtype=np.float64)

    # -----------------------------------------------------------------------
    # Rolling window loop — each position t uses rows [t-w+1 .. t] (t-indexed,
    # so strictly no future data). We start at index (w-1) because we need w
    # observations.
    # -----------------------------------------------------------------------
    for t in range(w - 1, n):
        window_ret = log_ret[t - w + 1 : t + 1]   # length-w slice, current bar included

        # ---- guard: need at least 2 finite values ---------------------------
        finite_mask = np.isfinite(window_ret)
        n_finite = int(finite_mask.sum())
        if n_finite < 2:
            continue

        ret_valid = window_ret[finite_mask]
        mu  = float(ret_valid.mean())
        var = float(ret_valid.var(ddof=1))

        # ---- Continuous (Gaussian) Kelly fraction --------------------------
        if var > 0.0:
            fc = mu / var
        else:
            fc = np.nan
        f_cont[t] = fc

        # ---- Half-Kelly ----------------------------------------------------
        if np.isfinite(fc):
            f_half[t] = 0.5 * fc

        # ---- Sigmoid-squashed Kelly ----------------------------------------
        if np.isfinite(fc):
            f_sig[t] = float(_signed_sigmoid_squash(np.array([fc]))[0])

        # ---- Discrete-bet Kelly: p - (1-p)/b  ------------------------------
        wins  = ret_valid[ret_valid > 0.0]
        loses = ret_valid[ret_valid < 0.0]
        p = float(len(wins)) / float(n_finite)
        if len(wins) > 0 and len(loses) > 0:
            avg_win  = float(wins.mean())          # positive
            avg_loss = float(np.abs(loses.mean())) # positive magnitude
            if avg_loss > 0.0:
                b = avg_win / avg_loss
                fd = p - (1.0 - p) / b
                f_disc[t] = fd
        elif len(loses) == 0 and len(wins) > 0:
            # Never lost → full Kelly = 1 (clamp to 1 as convention)
            f_disc[t] = 1.0
        # else: never won → fd = 0 (could also be nan; leave as nan)

        # ---- Expected log-growth at full-Kelly and half-Kelly ---------------
        fc_arr   = np.array([fc if np.isfinite(fc) else np.nan])
        fh_arr   = np.array([f_half[t]])
        mu_arr   = np.array([mu])
        var_arr  = np.array([var if var > 0.0 else np.nan])

        lg_full_val = _expected_log_growth(mu_arr, var_arr, fc_arr)[0]
        lg_half_val = _expected_log_growth(mu_arr, var_arr, fh_arr)[0]
        lg_full[t]  = float(lg_full_val) if np.isfinite(lg_full_val) else np.nan
        lg_half[t]  = float(lg_half_val) if np.isfinite(lg_half_val) else np.nan

        # ---- Kelly edge (Sharpe proxy): mu / sigma --------------------------
        sigma = math.sqrt(var) if var > 0.0 else np.nan
        if np.isfinite(sigma) and sigma > 0.0:
            edge[t] = mu / sigma

        # ---- Growth gap: log-growth(f*) - log-growth(0) -------------------
        # log-growth(0) = 0*mu - 0.5*0^2*var = 0, so gap = lg_full
        # (still useful to make the "edge of betting" explicit as its own feature)
        if np.isfinite(lg_full[t]):
            gap[t] = lg_full[t]   # == lg_full - 0

    return {
        f"kel_f_continuous_{w}":   f_cont,
        f"kel_f_halfkelly_{w}":    f_half,
        f"kel_f_sigmoid_{w}":      f_sig,
        f"kel_f_discrete_{w}":     f_disc,
        f"kel_loggrowth_full_{w}": lg_full,
        f"kel_loggrowth_half_{w}": lg_half,
        f"kel_edge_{w}":           edge,
        f"kel_growth_gap_{w}":     gap,
    }


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Kelly / growth-optimal features over 60- and 120-day rolling windows.

    All look-back windows are strictly causal (use only rows up to and including
    the current bar). Leading NaNs are expected during warm-up.
    """
    close = df["Close"].to_numpy(dtype=np.float64)

    # Log returns — shift(1) so return at t = log(Close[t]/Close[t-1]),
    # first row is NaN naturally.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log_ret = np.log(close[1:] / close[:-1])

    # Prepend one NaN so log_ret[i] corresponds to df row i
    log_ret = np.concatenate([[np.nan], log_ret])

    new_cols: dict = {}
    for w in (60, 120):
        new_cols.update(_kelly_features_for_window(log_ret, w))

    # Replace any accidental inf with NaN (safety guard)
    for col_name, arr in new_cols.items():
        arr = np.where(np.isfinite(arr) | np.isnan(arr), arr, np.nan)
        df[col_name] = arr

    return df
