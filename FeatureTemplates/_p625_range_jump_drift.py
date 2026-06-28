import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_range_jump_drift",
    "description": ("Range-estimator continuous-vs-jump variance split and "
                    "Parkinson-vs-Rogers-Satchell drift gap (GK-RS close-direction "
                    "excess, overnight/oc jump share, sign-free drift^2). "
                    "Barndorff-Nielsen & Shephard (2004,2006); "
                    "Bollerslev, Li & Todorov (2016) JFE; Rogers-Satchell (1991); "
                    "Garman-Klass (1980); Parkinson (1980)."),
    "requires":    [],
    "produces":    ["rsjp_jump_share_21", "rsjp_jump_share_63",
                    "rsjp_overnight_intraday_63", "rsjp_gk_rs_gap_63",
                    "rsjp_pk_rs_ratio_63", "rsjp_drift2_63"],
    "tags":        ["volatility", "jump", "range", "experimental"],
    "version":     "1.0",
    "author":      ("paper:Barndorff-Nielsen&Shephard 2004/2006; "
                    "Bollerslev,Li&Todorov 2016 JFE; Rogers-Satchell 1991; "
                    "Garman-Klass 1980; Parkinson 1980"),
}

_EPS = 1e-10
_LN2 = np.log(2.0)
_GK_C = 2.0 * _LN2 - 1.0   # Garman-Klass close-open coefficient


def compute(df: pd.DataFrame) -> pd.DataFrame:
    o_raw = df["Open"].astype(float)
    h_raw = df["High"].astype(float)
    l_raw = df["Low"].astype(float)
    c_raw = df["Close"].astype(float)

    # log prices; non-positive raw prices -> NaN (guarded)
    o = np.log(o_raw.where(o_raw > 0.0))
    h = np.log(h_raw.where(h_raw > 0.0))
    l = np.log(l_raw.where(l_raw > 0.0))
    c = np.log(c_raw.where(c_raw > 0.0))
    c_prev = c.shift(1)   # causal: prior bar's log close

    # ---- per-bar range/jump components ------------------------------------
    hl = h - l
    co = c - o
    gk_d = 0.5 * hl * hl - _GK_C * co * co
    rs_d = (h - c) * (h - o) + (l - c) * (l - o)
    park_d = (hl * hl) / (4.0 * _LN2)
    overnight = (o - c_prev) * (o - c_prev)
    oc = co * co

    # ---- trailing means (strictly causal rolling) ------------------------
    def _roll_mean(s: pd.Series, w: int) -> pd.Series:
        return s.rolling(window=w, min_periods=w // 2).mean()

    out = {}
    for w in (21, 63):
        GK = _roll_mean(gk_d, w)
        RS = _roll_mean(rs_d, w)
        OV = _roll_mean(overnight, w)
        OC = _roll_mean(oc, w)
        PK = _roll_mean(park_d, w)

        # jump share: jump variance / total, clipped [0, 1]
        jump_num = OV + OC
        jump_den = OV + OC + RS + _EPS
        out[f"rsjp_jump_share_{w}"] = (jump_num / jump_den).clip(0.0, 1.0)

        if w == 63:
            out["rsjp_overnight_intraday_63"] = (OV / (RS + _EPS)).clip(0.0, 20.0)
            out["rsjp_gk_rs_gap_63"] = ((GK - RS) / (GK + _EPS)).clip(-5.0, 5.0)
            out["rsjp_pk_rs_ratio_63"] = (PK / (RS + _EPS)).clip(0.0, 20.0)
            out["rsjp_drift2_63"] = (PK - RS).clip(lower=0.0).clip(upper=4.0)

    for col in METADATA["produces"]:
        s = out[col]
        df[col] = s.replace([np.inf, -np.inf], np.nan)

    return df
