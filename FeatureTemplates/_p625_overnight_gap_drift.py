import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_overnight_gap_drift",
    "description": "Overnight-conditioned intraday drift asymmetry (gap-and-go vs gap-fade conditional first moment); Lou, Polk & Skouras (2019) JFE; intraday gap literature (S1544612325018926).",
    "requires":    [],
    "produces":    ["ogd_gapdrift_asym_21", "ogd_gapdrift_asym_63", "ogd_today_gapdrift"],
    "tags":        ["overnight", "gap", "mean_reversion", "experimental"],
    "version":     "1.0",
    "author":      "paper:Lou,Polk&Skouras(2019)JFE;S1544612325018926",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)

    prev_close = close.shift(1)

    # Overnight return: log(Open_t / Close_{t-1}); guard non-positive prices -> NaN
    safe_open = open_.where(open_ > 0)
    safe_prev = prev_close.where(prev_close > 0)
    on = np.log(safe_open / safe_prev)

    # Intraday return: log(Close_t / Open_t), clipped to [-0.5, 0.5]
    safe_close = close.where(close > 0)
    intr = np.log(safe_close / safe_open).clip(-0.5, 0.5)

    # Gap classification on the overnight return
    up = (on > 0.003).astype(float)
    dn = (on < -0.003).astype(float)

    # Lag the within-day quantities so row t only uses rows <= t-1 for the
    # conditional history (today's intraday return is not known at decision time).
    intr_lag = intr.shift(1)
    up_lag = up.shift(1)
    dn_lag = dn.shift(1)

    intr_up = (intr_lag * up_lag)
    intr_dn = (intr_lag * dn_lag)

    eps = 1e-9
    for w in (21, 63):
        n_up = up_lag.rolling(w).sum()
        n_dn = dn_lag.rolling(w).sum()
        mu_up = intr_up.rolling(w).sum() / (n_up + eps)
        mu_dn = intr_dn.rolling(w).sum() / (n_dn + eps)
        asym = (mu_up - mu_dn).clip(-0.2, 0.2)
        # Require at least 4 gaps of each sign for a meaningful conditional mean.
        asym = asym.where((n_up >= 4) & (n_dn >= 4))
        df[f"ogd_gapdrift_asym_{w}"] = asym

    # Today's signal: lean the overnight gap in the historically learned direction.
    df["ogd_today_gapdrift"] = (
        (on * np.sign(df["ogd_gapdrift_asym_63"])).clip(-0.5, 0.5)
    )

    return df
