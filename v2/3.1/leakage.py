"""Panel/cross-sectional leakage detectors that per-ticker truncation cannot see.

The existing causality_check recomputes a feature on a TIME-truncated SINGLE ticker
and asserts past values are unchanged. That is blind to panel leakage: a (date,ticker)
row's value can stay stable under its own series yet still peek across OTHER tickers or
the full sample (global cross-sectional z-scores/ranks), and a universe that only keeps
survivors silently inflates targets. Per arXiv 2411.09218 ("On the (Mis)Use of ML with
Panel Data"), these are the failure modes that single-series truncation never surfaces;
this module screens for them at the panel level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# stats primitives (no external deps; mirrors eval.spearman)
# --------------------------------------------------------------------------- #
def _rank(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a)).astype(float)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    d = float(np.sqrt((x @ x) * (y @ y)))
    return float(x @ y) / d if d > 0 else 0.0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    return _pearson(_rank(x), _rank(y))


def _mean_daily_ic(
    panel: pd.DataFrame, feat: np.ndarray, target_col: str, date_col: str
) -> float:
    """Mean across dates of the per-date cross-sectional Spearman IC."""
    y = panel[target_col].to_numpy(float)
    ics: list[float] = []
    for _, idx in panel.groupby(date_col, sort=True).indices.items():
        fx = feat[idx]
        ty = y[idx]
        m = np.isfinite(fx) & np.isfinite(ty)
        if m.sum() >= 3:
            ics.append(_spearman(fx[m], ty[m]))
    return float(np.mean(ics)) if ics else 0.0


# --------------------------------------------------------------------------- #
# 1. cross-sectional normalization leak (global fit vs walk-forward refit)
# --------------------------------------------------------------------------- #
def xs_normalization_leak(
    panel: pd.DataFrame,
    feature_col: str,
    target_col: str,
    date_col: str = "Date",
    n_folds: int = 5,
) -> dict:
    """IC of a globally normalized feature vs an in-fold (walk-forward) normalized one.

    Both paths standardize each row by its TICKER'S mean/std, then measure mean per-date
    Spearman IC vs the target. The GLOBAL path uses each ticker's WHOLE-SAMPLE mean/std
    (it peeks at that ticker's future), so the normalized value of a PAST row depends on
    bars that had not happened yet. The IN-FOLD path standardizes each test fold using
    only that ticker's TRAILING history up to the fold (walk-forward, causal).

    This per-ticker standardization is non-monotone WITHIN a date (different tickers get
    different offsets/scales), so the two paths reshuffle the per-date cross-sectional
    ordering -- which is exactly why their ICs can diverge. A large positive gap
    (global > in-fold) means whole-sample normalization is leaking the future into the IC.
    """
    panel = panel.sort_values([date_col], kind="stable").reset_index(drop=True)
    raw = panel[feature_col].to_numpy(float)
    tick = panel.get("Ticker")
    tick = tick.to_numpy() if tick is not None else np.zeros(len(panel), int)

    dates = np.sort(panel[date_col].unique())
    n_folds = max(2, min(n_folds, len(dates)))
    edges = np.linspace(0, len(dates), n_folds + 1).astype(int)
    date_idx = {d: i for i, d in enumerate(dates)}
    pos = panel[date_col].map(date_idx).to_numpy()

    # group row indices by ticker once.
    tick_rows: dict = {}
    for i, tk in enumerate(tick):
        tick_rows.setdefault(tk, []).append(i)
    tick_rows = {k: np.asarray(v) for k, v in tick_rows.items()}

    # global path: standardize each row by its ticker's full-sample mean/std (peeks).
    global_z = np.full_like(raw, np.nan)
    for tk, ri in tick_rows.items():
        v = raw[ri]
        fv = v[np.isfinite(v)]
        if fv.size < 2:
            continue
        mu, sd = float(fv.mean()), float(fv.std())
        global_z[ri] = (v - mu) / (sd if sd > 0 else 1.0)
    gmask = np.isfinite(global_z)
    global_ic = _mean_daily_ic(panel[gmask].copy(), global_z[gmask], target_col, date_col)

    # in-fold path: standardize each test fold using only that ticker's trailing history.
    infold = np.full_like(raw, np.nan)
    for k in range(1, n_folds):  # fold 0 has no prior train window -> stays NaN
        te = (pos >= edges[k]) & (pos < edges[k + 1])
        for tk, ri in tick_rows.items():
            tr_i = ri[pos[ri] < edges[k]]
            te_i = ri[te[ri]]
            if te_i.size == 0:
                continue
            tv = raw[tr_i]
            tv = tv[np.isfinite(tv)]
            if tv.size < 2:
                continue
            mu, sd = float(tv.mean()), float(tv.std())
            infold[te_i] = (raw[te_i] - mu) / (sd if sd > 0 else 1.0)

    mask = np.isfinite(infold)
    infold_ic = (
        _mean_daily_ic(panel[mask].copy(), infold[mask], target_col, date_col)
        if mask.any()
        else 0.0
    )

    leak_gap = global_ic - infold_ic
    flag = bool(abs(leak_gap) > 0.01 and abs(global_ic) > abs(infold_ic) + 0.01)
    return {
        "global_ic": float(global_ic),
        "infold_ic": float(infold_ic),
        "leak_gap": float(leak_gap),
        "flag": flag,
    }


# --------------------------------------------------------------------------- #
# 2. survivorship bias (universe drift)
# --------------------------------------------------------------------------- #
def survivorship_bias(
    panel: pd.DataFrame,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
    target_col: str | None = None,
) -> dict:
    """Tickers present only in the LATE window are survivors; if their target is much
    higher than non-survivors', the universe is survivorship-biased."""
    dates = np.sort(panel[date_col].unique())
    n = len(dates)
    if n == 0:
        return {
            "n_tickers": 0,
            "frac_survivors": 0.0,
            "frac_late_entrants": 0.0,
            "survivor_target_edge": None,
            "flag": False,
        }
    early_cut = dates[max(0, int(np.ceil(0.10 * n)) - 1)]
    late_cut = dates[min(n - 1, int(np.floor(0.90 * n)))]

    by_ticker = panel.groupby(ticker_col)[date_col]
    first = by_ticker.min()
    last = by_ticker.max()
    tickers = first.index.to_numpy()

    survivor = (last.to_numpy() >= late_cut)  # present in the last 10% of dates
    late_entrant = (first.to_numpy() > early_cut)  # absent in the first 10% of dates
    survivor_set = set(tickers[survivor])

    n_tickers = int(len(tickers))
    frac_survivors = float(survivor.mean())
    frac_late_entrants = float(late_entrant.mean())

    survivor_target_edge: float | None = None
    flag = False
    if target_col is not None and target_col in panel.columns:
        is_surv_row = panel[ticker_col].isin(survivor_set).to_numpy()
        y = panel[target_col].to_numpy(float)
        sm = is_surv_row & np.isfinite(y)
        nm = (~is_surv_row) & np.isfinite(y)
        if sm.any() and nm.any():
            survivor_target_edge = float(y[sm].mean() - y[nm].mean())
            base = float(np.nanstd(y)) or 1.0
            flag = bool(survivor_target_edge > 0.10 * base)
    return {
        "n_tickers": n_tickers,
        "frac_survivors": frac_survivors,
        "frac_late_entrants": frac_late_entrants,
        "survivor_target_edge": survivor_target_edge,
        "flag": flag,
    }


# --------------------------------------------------------------------------- #
# 3. panel-level look-ahead (the truncation test, generalized to the panel)
# --------------------------------------------------------------------------- #
def panel_lookahead(
    compute_fn,
    panel: pd.DataFrame,
    produces: list[str],
    date_col: str = "Date",
    future_frac: float = 0.2,
    tol: float = 1e-6,
) -> dict:
    """Compute on the full panel and on a past-only prefix; a past row whose produced
    value changes when future dates/tickers are appended is leaking cross-sectionally."""
    full = compute_fn(panel.copy())

    dates = np.sort(panel[date_col].unique())
    keep_n = max(1, int(round((1.0 - future_frac) * len(dates))))
    keep_dates = set(dates[:keep_n])
    past = panel[panel[date_col].isin(keep_dates)].copy()
    trunc = compute_fn(past.copy())

    # align rows present in both runs on a stable key (date + positional within date).
    full_idx = full.index.to_numpy()
    trunc_idx = set(trunc.index.to_numpy())
    common = [i for i in full_idx if i in trunc_idx]

    max_drift = 0.0
    offending: list[str] = []
    for c in produces:
        a = np.asarray(full.loc[common, c].to_numpy(), float)
        b = np.asarray(trunc.loc[common, c].to_numpy(), float)
        m = np.isfinite(a) & np.isfinite(b)
        col_drift = float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0
        nan_mismatch = bool((np.isfinite(a) != np.isfinite(b)).any())
        if col_drift > tol or nan_mismatch:
            offending.append(c)
        max_drift = max(max_drift, col_drift if not nan_mismatch else float("inf"))

    return {
        "max_drift": float(max_drift),
        "causal": bool(not offending),
        "offending": offending,
    }


# --------------------------------------------------------------------------- #
# 4. convenience audit
# --------------------------------------------------------------------------- #
def audit_feature(
    panel: pd.DataFrame,
    feature_col: str,
    target_col: str,
    compute_fn=None,
    produces: list[str] | None = None,
    **kw,
) -> dict:
    """Run all applicable leakage checks and collect which ones fired."""
    date_col = kw.get("date_col", "Date")
    ticker_col = kw.get("ticker_col", "Ticker")

    xs = xs_normalization_leak(
        panel, feature_col, target_col, date_col=date_col, n_folds=kw.get("n_folds", 5)
    )
    surv = survivorship_bias(
        panel, date_col=date_col, ticker_col=ticker_col, target_col=target_col
    )
    out: dict = {"xs_normalization_leak": xs, "survivorship_bias": surv}

    leak_flags: list[str] = []
    if xs["flag"]:
        leak_flags.append("xs_normalization_leak")
    if surv["flag"]:
        leak_flags.append("survivorship_bias")

    if compute_fn is not None and produces:
        pl = panel_lookahead(
            compute_fn,
            panel,
            produces,
            date_col=date_col,
            future_frac=kw.get("future_frac", 0.2),
            tol=kw.get("tol", 1e-6),
        )
        out["panel_lookahead"] = pl
        if not pl["causal"]:
            leak_flags.append("panel_lookahead")

    out["leak_flags"] = leak_flags
    out["clean"] = not leak_flags
    return out


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
def _synthetic_panel(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_dates, n_tickers = 160, 50
    dates = pd.bdate_range("2022-01-03", periods=n_dates)

    # per-ticker persistent LEVEL that is unrelated to the target; the tradeable signal
    # is the WITHIN-ticker deviation from that level. To use it you must remove the level,
    # and the level also DRIFTS over time so a trailing estimate of it is always stale --
    # only a full-sample (peeking) fit recovers the level, and thus the signal, cleanly.
    mu_tk = rng.normal(0.0, 8.0, n_tickers)
    drift_tk = rng.normal(0.0, 0.10, n_tickers)  # per-ticker linear level drift / date

    rows = []
    for ti in range(n_tickers):
        # survivorship: the first 60% of tickers trade the whole window; the rest are
        # LATE-ENTRANT SURVIVORS that only appear in the last 30% of dates and carry a
        # higher forward return (so keeping only survivors inflates the target).
        late = ti >= int(0.60 * n_tickers)
        # non-survivors among the early cohort get DELISTED partway (exit early).
        delisted = (not late) and (ti % 3 == 0)
        start = int(0.70 * n_dates) if late else 0
        stop = int(0.45 * n_dates) if delisted else n_dates
        for di in range(start, stop):
            signal = rng.normal(0.0, 1.0)           # within-ticker tradeable deviation
            eps = rng.normal(0.0, 0.3)
            fwd = 0.01 * signal + (0.02 if late else 0.0) + rng.normal(0.0, 0.01)
            clean_feat = rng.normal(0.0, 1.0)        # honest noise, no level structure
            # LEAKY feature: persistent ticker level + the tradeable signal. The signal
            # only becomes usable once the level mu_tk is removed -- and the GLOBAL
            # normalizer removes it exactly by peeking at the ticker's whole future.
            leaky_feat = mu_tk[ti] + drift_tk[ti] * di + signal + eps
            rows.append((dates[di], f"T{ti:02d}", clean_feat, leaky_feat, fwd))

    df = pd.DataFrame(
        rows, columns=["Date", "Ticker", "clean_feat", "leaky_feat", "fwd_ret"]
    )
    return df.sort_values(["Date", "Ticker"], kind="stable").reset_index(drop=True)


def _leaky_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Look-ahead block: a global rank that shifts when future rows are appended."""
    out = df.copy()
    out["global_rank"] = out["fwd_ret"].rank(pct=True).to_numpy()
    return out


def _clean_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Causal block: per-ticker trailing mean (depends only on a ticker's own past)."""
    out = df.sort_values(["Ticker", "Date"], kind="stable").copy()
    out["roll_mean"] = (
        out.groupby("Ticker")["clean_feat"].transform(lambda s: s.expanding().mean())
    )
    return out.reindex(df.index)


def main() -> None:
    panel = _synthetic_panel()
    print("=== synthetic panel ===")
    print("rows=%d  dates=%d  tickers=%d"
          % (len(panel), panel["Date"].nunique(), panel["Ticker"].nunique()))

    print("\n=== 1. xs_normalization_leak ===")
    clean_xs = xs_normalization_leak(panel, "clean_feat", "fwd_ret")
    leaky_xs = xs_normalization_leak(panel, "leaky_feat", "fwd_ret")
    print("clean_feat :", {k: round(v, 4) if isinstance(v, float) else v for k, v in clean_xs.items()})
    print("leaky_feat :", {k: round(v, 4) if isinstance(v, float) else v for k, v in leaky_xs.items()})
    print("clean flagged?", clean_xs["flag"], " leaky flagged?", leaky_xs["flag"])

    print("\n=== 2. survivorship_bias ===")
    surv = survivorship_bias(panel, target_col="fwd_ret")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in surv.items()})
    print("survivorship flagged?", surv["flag"])

    print("\n=== 3. panel_lookahead ===")
    clean_pl = panel_lookahead(_clean_compute, panel, ["roll_mean"])
    leaky_pl = panel_lookahead(_leaky_compute, panel, ["global_rank"])
    print("clean block:", {k: round(v, 6) if isinstance(v, float) else v for k, v in clean_pl.items()})
    print("leaky block:", {k: round(v, 6) if isinstance(v, float) else v for k, v in leaky_pl.items()})
    print("clean causal?", clean_pl["causal"], " leaky causal?", leaky_pl["causal"])

    print("\n=== 4. audit_feature (leaky + survivorship + lookahead) ===")
    audit = audit_feature(
        panel, "leaky_feat", "fwd_ret",
        compute_fn=_leaky_compute, produces=["global_rank"],
    )
    print("leak_flags:", audit["leak_flags"], " clean?", audit["clean"])

    clean_audit = audit_feature(
        panel, "clean_feat", "fwd_ret",
        compute_fn=_clean_compute, produces=["roll_mean"],
    )
    # clean_feat is signal-free, so the only honest flag here is survivorship (a property
    # of the universe, not the feature). Demonstrate the per-check behavior:
    print("clean_feat leak_flags:", clean_audit["leak_flags"])

    ok = (not clean_xs["flag"]) and leaky_xs["flag"] and surv["flag"] \
        and clean_pl["causal"] and (not leaky_pl["causal"])
    print("\nSMOKE TEST:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
