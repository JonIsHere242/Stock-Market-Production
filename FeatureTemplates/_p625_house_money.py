import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_house_money",
    "description": "House-money vs snake-bitten prior-outcome conditioning via Grinblatt-Han "
                   "capital-gains-overhang reference price (Thaler-Johnson 1990 Mgmt Sci; "
                   "Barberis Huang Santos 2001 QJE).",
    "requires":    [],
    "produces":    ["hms_state_63", "hms_risk_takeup_63", "hms_housemoney_mom_21"],
    "tags":        ["behavioral", "prospect_theory", "experimental"],
    "version":     "1.0",
    "author":      "paper:Thaler&Johnson(1990) MgmtSci; Barberis,Huang,Santos(2001) QJE",
}

# Grinblatt-Han reference-price lookback (days). The survival recursion makes
# weights from the distant past decay to ~0, so a finite window is exact-causal.
_RP_WIN = 60


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    n = len(close)

    c = close.to_numpy(dtype="float64")
    v = vol.to_numpy(dtype="float64")

    # --- turnover proxy in [0,1] (no shares-outstanding available) -----------
    # Grinblatt-Han uses share turnover V_t = volume / shares_out. Without shares
    # we use a self-normalising proxy: today's volume vs its trailing-252d median,
    # squashed into (0,1) so it behaves like a turnover fraction. Trailing only.
    med = vol.rolling(252, min_periods=20).median().to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((med > 0) & np.isfinite(med), v / med, np.nan)
    # logistic squash centred at the median (ratio==1 -> ~0.27 turnover);
    # bounded strictly inside (0,1) so survival products never collapse oddly.
    turn = 1.0 - np.exp(-np.where(np.isfinite(ratio), ratio, 0.0) * 0.5)
    turn = np.clip(turn, 0.0, 0.999)
    valid_turn = np.isfinite(ratio)  # rows whose turnover is real

    # --- Grinblatt-Han reference price via the survival recursion ------------
    # RP_t = sum_{n>=1} w_{t-n} P_{t-n},  w_{t-n} = V_{t-n} * prod_{k=1..n-1}(1-V_{t-n+k})
    # Computed over a trailing window ending at t; weights renormalised to sum 1.
    rp = np.full(n, np.nan, dtype="float64")
    for t in range(n):
        lo = max(0, t - _RP_WIN)
        # candidate past days j in [lo, t-1] (strictly past -> causal)
        if t - lo < 5:
            continue
        idx = np.arange(lo, t)
        pj = c[idx]
        tj = turn[idx]
        vj = valid_turn[idx]
        if not vj.any() or not np.isfinite(pj).all():
            continue
        # survival weight for day j: turnover_j * prod of (1 - turnover) for days AFTER j up to t-1
        one_minus = 1.0 - tj
        # cumulative product of (1-turn) from the END backwards:
        # surv[j] = prod_{m=j+1..end}(1-turn[m])
        rev_cumprod = np.cumprod(one_minus[::-1])[::-1]
        # shift so surv[j] excludes day j itself: surv[j] = rev_cumprod[j+1]
        surv = np.empty_like(rev_cumprod)
        surv[:-1] = rev_cumprod[1:]
        surv[-1] = 1.0
        w = tj * surv
        w = np.where(vj, w, 0.0)
        wsum = w.sum()
        if wsum <= 0 or not np.isfinite(wsum):
            continue
        rp[t] = float(np.dot(w, pj) / wsum)

    rp_s = pd.Series(rp, index=close.index)

    # --- z = capital gains overhang = (Close - RP)/RP ------------------------
    denom = rp_s.replace(0.0, np.nan)
    z = (close - rp_s) / denom
    z = z.replace([np.inf, -np.inf], np.nan)

    # --- s_t = 5-day return ---------------------------------------------------
    c5 = close.shift(5).replace(0.0, np.nan)
    s = close / c5 - 1.0
    s = s.replace([np.inf, -np.inf], np.nan)

    # --- daily return magnitude ----------------------------------------------
    c1 = close.shift(1).replace(0.0, np.nan)
    daily_ret = close / c1 - 1.0
    daily_ret = daily_ret.replace([np.inf, -np.inf], np.nan)
    abs_ret = daily_ret.abs()

    # --- hms_state_63 = clip(z, -1, 1) ---------------------------------------
    df["hms_state_63"] = z.clip(-1.0, 1.0)

    # --- hms_risk_takeup_63 = clip(corr_63(|daily_return|, z), -1, 1) --------
    # trailing 63-day rolling Pearson correlation ending at t (causal)
    corr = abs_ret.rolling(63, min_periods=20).corr(z)
    corr = corr.replace([np.inf, -np.inf], np.nan)
    df["hms_risk_takeup_63"] = corr.clip(-1.0, 1.0)

    # --- hms_housemoney_mom_21 = clip(SMA21(s * 1[z>0]), -1, 1) ---------------
    gain_state = (z > 0).astype(float)
    house_mom_raw = s * gain_state
    house_mom = house_mom_raw.rolling(21, min_periods=5).mean()
    house_mom = house_mom.replace([np.inf, -np.inf], np.nan)
    df["hms_housemoney_mom_21"] = house_mom.clip(-1.0, 1.0)

    return df
