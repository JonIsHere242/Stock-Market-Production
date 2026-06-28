"""
_paper_2605_16866_heavy_tail_index.py  --  Tail-index estimation and
finite-variance diagnostics per ticker.

Motivated by arXiv 2605.16866 ("Heavy Tails and Predictive Ability
Testing"): when return distributions have infinite (or near-infinite)
variance, the classical CLT-based inference breaks and predictability
tests are unreliable.  These features quantify HOW heavy and HOW
asymmetric the tail is for each stock, giving the model a per-ticker
signal about tail risk and moment existence.

ESTIMATORS
----------
Hill tail-index (alpha):
    Given the top-k order statistics |X|(n) >= ... >= |X|(n-k+1),
    the Hill (1975) MLE of the tail exponent is:
        alpha_hat = k / sum_{i=1}^{k} log(|X|(n-i+1) / |X|(n-k))
    Larger alpha => lighter tail; alpha<2 => infinite variance.
    Computed SEPARATELY for the right and left tails.

Max-to-sum ratio (R_p):
    R_p = max(|x_i|^p) / sum(|x_i|^p)  for p in {1, 2, 4}
    When the p-th moment does NOT exist, R_p -> 1 (dominated by max).
    When it exists, R_p -> 0 (max is negligible vs sum).  A high R_2
    is therefore a classic infinite-variance diagnostic.

NOTE: per-ticker tail-index/heavy-tail diagnostic only.
      All estimates are noisy with 120-250 obs; treat as ordinal.
"""

import numpy as np
import pandas as pd
import warnings

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name":        "paper_2605_16866_heavy_tail_index",
    "description": (
        "Per-ticker heavy-tail index estimation (Hill estimator on right/left tails, "
        "max-to-sum ratio finite-variance diagnostics, tail asymmetry, and an "
        "infinite-variance flag) motivated by arXiv 2605.16866."
    ),
    "requires":    ["Close"],
    "produces":    [
        "htl_hill_right",          # Hill alpha on right tail (large gains)
        "htl_hill_left",           # Hill alpha on left tail (large losses)
        "htl_tail_asymmetry",      # hill_left - hill_right (+ = heavier right loss tail)
        "htl_msr_p1",              # max-to-sum ratio, p=1
        "htl_msr_p2",              # max-to-sum ratio, p=2  (infinite-variance signal)
        "htl_msr_p4",              # max-to-sum ratio, p=4  (infinite 4th-moment signal)
        "htl_infinite_var_flag",   # 1 if htl_msr_p2 > threshold (moment may not exist)
        "htl_stable_alpha_proxy",  # coarse stable-distribution alpha proxy via log-log slope
    ],
    "tags":        ["volatility", "risk", "tail_risk", "experimental"],
    "version":     "1.0",
    "author":      "paper arXiv 2605.16866 — heavy-tail / infinite-variance diagnostics",
}

# ---------------------------------------------------------------------------
# Helper: Hill tail-index estimator (vectorised, expanding-then-sliding)
# ---------------------------------------------------------------------------
_WINDOW      = 120   # rolling window for all estimates
_K_FRAC      = 0.15  # fraction of window observations used as top order statistics
_INF_VAR_THR = 0.35  # msr_p2 threshold above which we flag infinite-variance


def _hill_index(arr: np.ndarray, k: int) -> float:
    """
    Compute the Hill (1975) tail-index estimate for array `arr` using the
    top-k order statistics.  Returns NaN when k < 2 or all values are 0.

    `arr` must contain non-negative values (absolute returns for one tail).
    """
    if k < 2:
        return np.nan
    # Sort ascending; top-k largest are at the end
    s = np.sort(arr)
    top_k   = s[-k:]          # k largest (ascending order inside top_k)
    cutoff  = s[-(k + 1)]     # the (k+1)-th largest = the threshold value
    if cutoff <= 0:
        return np.nan
    log_ratios = np.log(top_k / cutoff)
    if log_ratios.sum() <= 0:
        return np.nan
    return float(k) / log_ratios.sum()


def _msr(arr: np.ndarray, p: float) -> float:
    """
    Max-to-sum ratio: max(|x|^p) / sum(|x|^p).
    Returns NaN when all values are 0.
    """
    pw = np.abs(arr) ** p
    total = pw.sum()
    if total <= 0:
        return np.nan
    return pw.max() / total


def _stable_alpha_proxy(abs_rets: np.ndarray) -> float:
    """
    Coarse stable-distribution alpha estimate via log-log tail slope.

    For a stable law with index alpha, the survival function F̄(x) ~ x^{-alpha}
    for large x.  We fit a simple OLS slope on log(rank/n) ~ -alpha * log(x)
    using the top 20% of observations.
    """
    n = len(abs_rets)
    if n < 10:
        return np.nan
    top = int(max(5, n * 0.20))
    s = np.sort(abs_rets)[-top:]       # top 20% ascending
    # Empirical survival at each order stat: (n - rank) / n, rank 0-indexed from bottom
    # For top `top` stats, ranks are n-top, n-top+1, ..., n-1  (0-indexed)
    ranks = np.arange(n - top, n)
    surv  = (n - ranks - 0.5) / n
    logx  = np.log(s + 1e-12)
    logp  = np.log(surv + 1e-12)
    # OLS slope: logp ~ slope * logx + intercept  =>  alpha = -slope
    # Use only points where s > 0
    mask = s > 0
    if mask.sum() < 3:
        return np.nan
    lx = logx[mask]
    lp = logp[mask]
    cov_xy = np.cov(lx, lp, ddof=1)
    if cov_xy[0, 0] <= 0:
        return np.nan
    slope  = cov_xy[0, 1] / cov_xy[0, 0]
    alpha  = float(-slope)
    # Clip to plausible range [0.3, 4.0]
    return float(np.clip(alpha, 0.3, 4.0))


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    n   = len(df)
    idx = df.index

    # Log returns (NaN for row 0)
    close = df["Close"].to_numpy(dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_ret = np.empty(n)
        log_ret[0] = np.nan
        # Avoid divide-by-zero / log(0) via explicit mask
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                log_ret[i] = np.log(close[i] / close[i - 1])
            else:
                log_ret[i] = np.nan

    # Pre-allocate output arrays
    hill_right       = np.full(n, np.nan)
    hill_left        = np.full(n, np.nan)
    tail_asymmetry   = np.full(n, np.nan)
    msr_p1           = np.full(n, np.nan)
    msr_p2           = np.full(n, np.nan)
    msr_p4           = np.full(n, np.nan)
    inf_var_flag     = np.full(n, np.nan)
    stable_alpha     = np.full(n, np.nan)

    k = max(2, int(_WINDOW * _K_FRAC))   # number of order stats for Hill

    for t in range(_WINDOW - 1, n):
        # Window of log-returns ending at t (no future data)
        window = log_ret[t - _WINDOW + 1 : t + 1]

        # Drop NaN
        valid = window[np.isfinite(window)]
        if len(valid) < _WINDOW // 2:
            continue  # too few observations for stable estimates

        # Split into positive (right tail) and negative (left tail) returns
        pos = valid[valid > 0]
        neg = -valid[valid < 0]   # magnitudes of losses

        k_eff = min(k, len(pos) - 2, len(neg) - 2)

        if k_eff >= 2:
            hr = _hill_index(pos, k_eff)
            hl = _hill_index(neg, k_eff)
            hill_right[t] = hr
            hill_left[t]  = hl
            if np.isfinite(hr) and np.isfinite(hl):
                tail_asymmetry[t] = hl - hr   # positive => left tail heavier

        # Max-to-sum ratios on all absolute returns
        abs_valid = np.abs(valid)
        msr_p1[t] = _msr(abs_valid, 1.0)
        msr_p2[t] = _msr(abs_valid, 2.0)
        msr_p4[t] = _msr(abs_valid, 4.0)

        if np.isfinite(msr_p2[t]):
            inf_var_flag[t] = float(msr_p2[t] > _INF_VAR_THR)

        # Stable alpha proxy (uses absolute returns)
        stable_alpha[t] = _stable_alpha_proxy(abs_valid)

    # Replace any inadvertent inf with NaN
    def _clean(a: np.ndarray) -> np.ndarray:
        a[~np.isfinite(a)] = np.nan
        return a

    df["htl_hill_right"]        = _clean(hill_right)
    df["htl_hill_left"]         = _clean(hill_left)
    df["htl_tail_asymmetry"]    = _clean(tail_asymmetry)
    df["htl_msr_p1"]            = _clean(msr_p1)
    df["htl_msr_p2"]            = _clean(msr_p2)
    df["htl_msr_p4"]            = _clean(msr_p4)
    df["htl_infinite_var_flag"] = _clean(inf_var_flag)
    df["htl_stable_alpha_proxy"]= _clean(stable_alpha)

    return df
