from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ext_volume_shock_reversal",
    "description": (
        "Return reversal following volume shocks. "
        "Identifies volume-shock days where the 20-day z-score of log-volume exceeds 2. "
        "For each bar (causal), computes the rolling 90-day mean of NEXT-DAY returns "
        "that followed UP-day volume shocks (ext_volume_shock_reversal_up) and DOWN-day "
        "volume shocks (ext_volume_shock_reversal_dn), and their asymmetry "
        "(ext_volume_shock_reversal_asym = dn_mean_ret - up_mean_ret). "
        "Conditioning is on shock magnitude (zscore>2), not tercile. "
        "Per-ticker proxy -- cannot rank across stocks -- captures same economic signal as "
        "xdom2_autocorr_volume_return on a different axis (shock-conditioning vs autocorr)."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "ext_volume_shock_reversal_up",
        "ext_volume_shock_reversal_dn",
        "ext_volume_shock_reversal_asym",
    ],
    "tags": ["volume", "reversal", "shock", "behavioral", "mean_reversion"],
    "version": "1.0",
    "author": "Extension/exploration of xdom2_autocorr_volume_return (gate-validated winner)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    if n < 25:
        df["ext_volume_shock_reversal_up"] = np.nan
        df["ext_volume_shock_reversal_dn"] = np.nan
        df["ext_volume_shock_reversal_asym"] = np.nan
        return df

    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    # --- log volume z-score over 20-day rolling window (causal) ---
    log_vol = np.where(volume > 0, np.log(volume), np.nan)

    vol_roll_mean = np.full(n, np.nan)
    vol_roll_std = np.full(n, np.nan)
    win_z = 20
    for i in range(win_z - 1, n):
        w = log_vol[i - win_z + 1 : i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) >= win_z // 2:
            vol_roll_mean[i] = valid.mean()
            vol_roll_std[i] = valid.std(ddof=1) if len(valid) > 1 else np.nan

    vol_zscore = np.where(
        vol_roll_std > 0,
        (log_vol - vol_roll_mean) / vol_roll_std,
        np.nan,
    )

    # --- volume shock flag (zscore > 2) ---
    shock = (vol_zscore > 2).astype(float)  # 1=shock, 0=not

    # --- daily return (lookahead-safe: return[i] = close[i]/close[i-1] - 1) ---
    ret = np.full(n, np.nan)
    ret[1:] = close[1:] / np.where(close[:-1] != 0, close[:-1], np.nan) - 1.0

    # --- UP/DOWN shock classification: price return on the shock day >= 0 vs < 0 ---
    # shock_up[i]=1 if bar i was a volume shock AND close return >= 0
    shock_up = np.where((shock == 1) & (~np.isnan(ret)) & (ret >= 0), 1.0, 0.0)
    shock_dn = np.where((shock == 1) & (~np.isnan(ret)) & (ret < 0), 1.0, 0.0)

    # --- rolling 90d mean next-day return after UP/DN shock days (causal) ---
    # For each day i, we look at the past 90 days [i-90 .. i-1].
    # A shock at day j contributes next-day return = ret[j+1] (which is at most j+1 <= i-1+1=i).
    # So we pair (shock[j], ret[j+1]) and sum over j in [i-90, i-1].
    # This is equivalent to: next_ret[j] = ret[j+1], aggregated when shock[j] fires.
    # Shift next-ret array left by 1 to align: next_ret_aligned[j] = ret[j+1].
    # But since we only look at j < i, the latest pair we can use is (j=i-1, ret[i]),
    # which would read ret[i] -- NOT yet observed at bar i. So we must stop at j <= i-2,
    # i.e., use pairs [i-90 .. i-2].

    # next_ret_arr[j] = return of day j+1 (only valid for j+1 < n)
    next_ret_arr = np.full(n, np.nan)
    next_ret_arr[:-1] = ret[1:]   # next_ret_arr[j] = ret[j+1]; last slot remains NaN

    win_r = 90

    up_mean = np.full(n, np.nan)
    dn_mean = np.full(n, np.nan)

    for i in range(win_r, n):
        # look-back window: j in [i-win_r, i-2] (i-1 excluded to avoid using ret[i])
        start = i - win_r
        end = i - 1  # exclusive slice end
        # slice indices: [start : end] covers j = start .. end-1 = i-2
        su = shock_up[start:end]
        sd = shock_dn[start:end]
        nr = next_ret_arr[start:end]

        valid_mask = ~np.isnan(nr)
        su_mask = (su == 1) & valid_mask
        sd_mask = (sd == 1) & valid_mask

        n_up = su_mask.sum()
        n_dn = sd_mask.sum()

        up_mean[i] = nr[su_mask].mean() if n_up > 0 else np.nan
        dn_mean[i] = nr[sd_mask].mean() if n_dn > 0 else np.nan

    asym = np.where(
        ~np.isnan(up_mean) & ~np.isnan(dn_mean),
        dn_mean - up_mean,
        np.nan,
    )

    df["ext_volume_shock_reversal_up"] = up_mean
    df["ext_volume_shock_reversal_dn"] = dn_mean
    df["ext_volume_shock_reversal_asym"] = asym
    return df
