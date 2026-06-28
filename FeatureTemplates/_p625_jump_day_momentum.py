import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

METADATA = {
    "name":        "_p625_jump_day_momentum",
    "description": "Jump-day momentum (characteristic-managed momentum proxy): cumulative log-return over large-|return| 'jump' days vs calm days, their share/difference, and a magnitude-weighted variant (Beckmeyer & Wiedemann 2025 JBF Vol.181 S0378426625001852).",
    "requires":    [],
    "produces":    ["jmom_jumpcum_126", "jmom_calmcum_126", "jmom_jumpshare_126",
                    "jmom_jumpdiff_126", "jmom_magwgt_126"],
    "tags":        ["momentum", "jump", "experimental"],
    "version":     "1.0",
    "author":      "paper:Beckmeyer & Wiedemann (2025) JBF Vol.181 S0378426625001852",
}

W = 126        # formation window length (days)
SKIP = 5       # skip the most recent SKIP days
THRESH = 2.5   # jump threshold in units of rolling std of returns
EPS = 1e-8


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    n = len(close)

    # Daily log returns. log of non-positive prices -> NaN (guarded).
    prev = close.shift(1)
    ratio = close / prev.replace(0.0, np.nan)
    ratio = ratio.where(ratio > 0.0, np.nan)
    r = np.log(ratio.to_numpy(dtype="float64"))   # r[0] is NaN

    # Causal rolling std of returns over the formation window (uses rows <= t).
    r_ser = pd.Series(r)
    roll_std = r_ser.rolling(window=W, min_periods=W).std(ddof=1).to_numpy(dtype="float64")

    # Jump flag at day i: |r_i| > THRESH * rolling_std(r, W) evaluated at day i.
    # rolling_std at i uses returns in [i-W+1, i] (all <= i) -> causal.
    with np.errstate(invalid="ignore"):
        is_jump = np.abs(r) > (THRESH * roll_std)
    is_jump = is_jump & np.isfinite(r) & np.isfinite(roll_std)
    is_calm = (~is_jump) & np.isfinite(r) & np.isfinite(roll_std)

    r0 = np.where(np.isfinite(r), r, 0.0)
    jump_r = np.where(is_jump, r0, 0.0)
    calm_r = np.where(is_calm, r0, 0.0)
    absr = np.where(np.isfinite(r), np.abs(r), 0.0)
    magnum = np.where(np.isfinite(r), r0 * absr, 0.0)   # r_i * |r_i|

    jumpcum = np.full(n, np.nan)
    calmcum = np.full(n, np.nan)
    jumpshare = np.full(n, np.nan)
    jumpdiff = np.full(n, np.nan)
    magwgt = np.full(n, np.nan)

    # Window over which sums are taken at day t: [t-W, t-SKIP] (formation, skip last SKIP).
    # This window must be fully available: need t-W >= 0  =>  t >= W. We also require the
    # rolling std at the window's start to be defined; first valid t is W (length W-SKIP+1).
    first = W
    if n > first:
        # Sliding windows of length (W - SKIP + 1) over the lagged-by-SKIP series.
        # For day t, the contributing days are indices [t-W, t-SKIP] inclusive.
        win_len = W - SKIP + 1
        # Series aligned so that position j corresponds to source index j; we take windows
        # ending at t-SKIP and starting at t-W. Build via sliding_window_view on the base
        # arrays then index by window-end.
        jw = sliding_window_view(jump_r, win_len)   # shape (n-win_len+1, win_len)
        cw = sliding_window_view(calm_r, win_len)
        mw = sliding_window_view(magnum, win_len)
        aw = sliding_window_view(absr, win_len)

        # Window for day t starts at t-W. Window index s = t-W, valid for t in [W, n-1]
        # provided s+win_len-1 = t-W + (W-SKIP) = t-SKIP <= n-1  => t <= n-1+SKIP (always true here).
        # sliding_window_view yields windows for start s in [0, n-win_len]; we need s up to n-1-W.
        # max start needed = (n-1)-W; available max start = n-win_len = n-(W-SKIP+1).
        # n-(W-SKIP+1) >= (n-1)-W  <=> SKIP >= 0, always true.
        t_idx = np.arange(first, n)
        s_idx = t_idx - W
        jumpcum[t_idx] = jw[s_idx].sum(axis=1)
        calmcum[t_idx] = cw[s_idx].sum(axis=1)
        magsum = mw[s_idx].sum(axis=1)
        abssum = aw[s_idx].sum(axis=1)

        jc = jumpcum[t_idx]
        cc = calmcum[t_idx]
        denom = np.abs(jc) + np.abs(cc) + EPS
        jumpshare[t_idx] = jc / denom
        jumpdiff[t_idx] = jc - cc
        with np.errstate(invalid="ignore", divide="ignore"):
            mwv = np.where(abssum > 0.0, magsum / abssum, np.nan)
        magwgt[t_idx] = mwv

    df["jmom_jumpcum_126"] = jumpcum
    df["jmom_calmcum_126"] = calmcum
    df["jmom_jumpshare_126"] = jumpshare
    df["jmom_jumpdiff_126"] = jumpdiff
    df["jmom_magwgt_126"] = magwgt
    return df
