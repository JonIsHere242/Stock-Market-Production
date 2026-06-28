"""
__tail_screen.py  --  FAST offline Tier-2 triage (no retrain, single core, <10min).

WHY
---
The only ground-truth ranking of features is a ~3h retrain+backtest, far too slow to
iterate research directions. This screen is the cheap proxy: it scores candidate feature
blocks by the metric we actually care about -- cross-sectional TOP-DECILE tail lift for
next-day returns -- VOL/TREND-NEUTRALIZED (so we are not just rediscovering the volatility
trap) and across multiple time-folds (so a single-path fluke doesn't fool us; the project's
hard-won lesson is that single-sample lift is noise).

It does NOT touch ProcessedData_v2 or retrain anything. It builds a fresh panel straight
from Data/PriceData by running the candidate blocks' compute() on a ticker sample, then
reuses the EXACT tail-lift definition from __feature_lab.py.

METRIC
------
  lift  : among the names in a feature's top (or bottom) decile each day, how many times
          more likely they are to be next-day top-decile winners vs the base rate.
          lift=1.0 -> no edge, lift=1.30 -> 30% more winners than chance. Sign-agnostic
          (the better of the top/bottom slice is reported, with which side).
  neut  : the SAME lift after the feature is made cross-sectionally orthogonal to trend AND
          vol each day (vector_neut x2). THIS is the number to trust -- it is the marginal
          edge that is NOT just "high-vol/high-trend names win". Rank by this.
  folds : neut lift recomputed within K contiguous date-folds; a feature is ROBUST only if
          its edge keeps the same sign across folds and the worst fold still shows an edge.

USAGE
-----
  stock_env\\Scripts\\python.exe FeatureTemplates\\__tail_screen.py            # default: all xdm_+csa_
  ...\\python.exe FeatureTemplates\\__tail_screen.py --n 600 --folds 5
  ...\\python.exe FeatureTemplates\\__tail_screen.py --blocks capital_gains_overhang,salience_theory_value
  ...\\python.exe FeatureTemplates\\__tail_screen.py --extra_blocks price_momentum_features   # add incumbents as a yardstick

KNOBS:  --n tickers (default 500)  --seed (42)  --folds (5)  --winner_pct (0.10)
        --q feature slice (0.10)   --min_names cross-section floor (25)
"""
from __future__ import annotations

import os
# single-core + courteous to any running training job (set before numpy import)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib.util
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRICE_DIR = ROOT / "Data" / "PriceData"

# default candidate batches (cross-domain xdm_ + the csa_ batch)
XDM = ["capital_gains_overhang", "salience_theory_value", "recurrence_interval_hazard",
       "pettitt_changepoint", "mann_kendall_trend", "conditional_reversion_skew", "langevin_cubic_drift"]
CSA = ["residual_momentum", "beta_dynamics", "seasonality_calendar", "variance_ratio", "informed_drift"]

# tiny ANSI
def _c(t, code): return f"\033[{code}m{t}\033[0m"
def lc(v):
    if not np.isfinite(v): return "38;5;240"
    e = abs(v - 1.0)
    return "1;32" if e >= 0.30 else "32" if e >= 0.15 else "33" if e >= 0.07 else "38;5;240"


# ---------------------------------------------------------------------------
# helpers copied verbatim from __feature_lab.py (identical methodology)
# ---------------------------------------------------------------------------
def _within_z(panel, series, win):
    tmp = series.copy(); grp = panel["Ticker"]
    mean = tmp.groupby(grp).transform(lambda s: s.rolling(win, min_periods=max(10, win // 4)).mean())
    std = tmp.groupby(grp).transform(lambda s: s.rolling(win, min_periods=max(10, win // 4)).std())
    return (tmp - mean) / std.replace(0, np.nan)


def add_context(panel):
    panel["_logc"] = np.log(panel["Close"].clip(lower=1e-10))
    panel["_fwd_ret"] = panel.groupby("Ticker")["_logc"].shift(-1) - panel["_logc"]
    panel["_ret"] = panel.groupby("Ticker")["Close"].pct_change()
    # incumbent short-horizon factors -- the dominant 1-day confounds the model already has.
    panel["_ret1"] = panel["_ret"]                                          # 1-day reversal driver
    panel["_ret5"] = panel.groupby("Ticker")["Close"].transform(lambda s: s / s.shift(5) - 1.0)
    # ABSOLUTE cross-sectional confounds (these are what make a name structurally land in the
    # 1-day top decile): realized-vol level (the vol trap), price level, dollar-volume level.
    panel["_volabs"] = panel.groupby("Ticker")["_ret"].transform(lambda s: s.rolling(20, min_periods=10).std())
    panel["_logprice"] = np.log(panel["Close"].clip(lower=1e-6))
    panel["_logdvol"] = np.log((panel["Close"] * panel["Volume"]).clip(lower=1.0))
    ma50 = panel.groupby("Ticker")["Close"].transform(lambda s: s.rolling(50, min_periods=20).mean())
    panel["_trend"] = panel["Close"] / ma50 - 1.0
    panel["_fwd_rank"] = panel.groupby("Date")["_fwd_ret"].rank(pct=True)
    panel["_day_n"] = panel.groupby("Date")["_fwd_ret"].transform("count")

    # ---- ALT-TARGET forward builders (LABELS only; NEVER used as candidate features) --------
    # Each reproduces the EXACT forward target the matching predictor arm trains on, computed
    # per-ticker from FUTURE bars (shift(-1)/shift(-5)). --target redefines is_winner from one of
    # these so the screen measures tail lift on THAT arm's winner definition. Math mirrors
    # 4__Predictorv4_rank.load_and_label_tickers (the {tb,gap,id,dd,dt,vw} derived targets) and
    # build_labels (risk_adj_topq, binary_up). These columns are absent from feat_cols, so the
    # per-feature scoring loop never reads them -- no leak.
    g = panel.groupby("Ticker")
    c = panel["Close"]
    o1 = g["Open"].shift(-1); h1 = g["High"].shift(-1)
    l1 = g["Low"].shift(-1);  c1 = g["Close"].shift(-1)
    c5 = g["Close"].shift(-5)
    relvol = panel["Volume"] / g["Volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    ret5_mean = g["_ret"].transform(lambda s: s.rolling(5, min_periods=5).mean())   # trailing 5d mean daily ret
    fwd = c1 / c - 1.0                                                               # next-day close-to-close
    lowr = l1 / c - 1.0; highr = h1 / c - 1.0
    panel["_tgt_topq"]      = fwd
    panel["_tgt_risk_adj"]  = fwd / panel["_volabs"].clip(lower=1e-4)
    panel["_tgt_binary_up"] = (fwd > 0).astype(float)
    panel["_tgt_ret5"]      = c5 / c - 1.0
    panel["_tgt_tb"]        = np.where(lowr <= -0.019, -0.019,
                                np.where((lowr > -0.019) & (highr >= 0.035), 0.035, fwd))
    panel["_tgt_gap"]       = o1 / c - 1.0
    panel["_tgt_id"]        = c1 / o1 - 1.0
    panel["_tgt_dd"]        = -(l1 / c - 1.0)
    panel["_tgt_dt"]        = fwd - ret5_mean
    panel["_tgt_vw"]        = fwd * relvol
    return panel


# factors a candidate must be edge-orthogonal to: absolute vol/price/liquidity (the structural
# vol-trap confounds) + short-horizon reversal (the dominant 1-day effect the model already has).
NEUT_FACTORS = ["_ret1", "_ret5", "_volabs", "_logprice", "_logdvol"]

# ALT-TARGET axes (the ensemble arms in 4__Predictorv4_rank.py). --target selects which arm's
# winner definition the tail-lift is scored against. Each maps to a forward LABEL column built in
# add_context() (FUTURE bars only -> never a feature). The neutralization above is axis-agnostic:
# every arm asks the SAME question "does this feature beat the vol trap for THIS arm's winners".
TARGET_COL = {
    "topq":      "_tgt_topq",       # per-day top-20% next-day close-to-close return (incumbent)
    "risk_adj":  "_tgt_risk_adj",   # top-20% of return / 20d realized vol
    "binary_up": "_tgt_binary_up",  # sign of next-day return (DIRECTION; winners = up-closes)
    "ret5":      "_tgt_ret5",       # 5-day forward return
    "tb":        "_tgt_tb",         # stop-aware path return (-1.9% stop / +3.5% target)
    "gap":       "_tgt_gap",        # overnight gap (next Open / today Close)
    "id":        "_tgt_id",         # intraday next-session open->close
    "dd":        "_tgt_dd",         # downside avoidance: -(next Low / today Close - 1)
    "dt":        "_tgt_dt",         # de-trended surprise (next ret - trailing 5d mean daily ret)
    "vw":        "_tgt_vw",         # volume-confirmed return (next ret * relative volume)
}


def _xs_neutralize_multi(panel, col, factors):
    """Per-day residual of feature jointly regressed on `factors` (multi-factor vector_neut)."""
    f = panel[col].to_numpy(dtype=float)
    X = np.column_stack([panel[fc].to_numpy(dtype=float) for fc in factors])
    out = np.full(len(panel), np.nan)
    codes, _ = pd.factorize(panel["Date"].to_numpy())
    order = np.argsort(codes, kind="stable"); cs = codes[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    kf = X.shape[1]
    for idx in np.split(order, bounds):
        ff = f[idx]; XX = X[idx]
        m = np.isfinite(ff) & np.isfinite(XX).all(axis=1)
        if m.sum() < max(20, kf * 5):
            continue
        Xm = XX[m]; mu = Xm.mean(axis=0)
        Xc = Xm - mu; yc = ff[m] - ff[m].mean()
        beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        out[idx] = ff - (ff[m].mean() + (XX - mu) @ beta)
    return out


def _xs_neutralize(panel, col, factor):
    """Per-day residual of feature regressed on `factor` (WorldQuant vector_neut)."""
    f = panel[col].to_numpy(dtype=float); x = panel[factor].to_numpy(dtype=float)
    out = np.full(len(panel), np.nan)
    codes, _ = pd.factorize(panel["Date"].to_numpy())
    order = np.argsort(codes, kind="stable")
    cs = codes[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    for idx in np.split(order, bounds):
        ff, xx = f[idx], x[idx]
        m = np.isfinite(ff) & np.isfinite(xx)
        if m.sum() < 10:
            continue
        xc = xx[m] - xx[m].mean()
        denom = float((xc * xc).sum())
        beta = float((xc * (ff[m] - ff[m].mean())).sum()) / denom if denom > 0 else 0.0
        out[idx] = ff - (ff[m].mean() + beta * (xx - xx[m].mean()))
    return out


try:
    from scipy import stats as _ss
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def compute_ic(values, fwd):
    v = np.asarray(values, dtype=float)
    m = np.isfinite(v) & np.isfinite(fwd)
    if m.sum() < 100:
        return np.nan
    if _HAVE_SCIPY:
        c, _ = _ss.spearmanr(v[m], fwd[m]); return float(c) if np.isfinite(c) else np.nan
    a = pd.Series(v[m]).rank().to_numpy(); b = pd.Series(fwd[m]).rank().to_numpy()
    c = np.corrcoef(a, b)[0, 1]; return float(c) if np.isfinite(c) else np.nan


def tail_lift_folds(dates, vals, is_winner, valid, fold_code, K, q, wp):
    """Rank once per Date, then evaluate full-sample + per-fold top/bottom-decile lift."""
    rank = pd.DataFrame({"d": dates, "v": np.asarray(vals, float)}).groupby("d")["v"].rank(pct=True).to_numpy()
    finite = np.isfinite(rank) & valid
    top = finite & (rank >= 1.0 - q)
    bot = finite & (rank <= q)

    def lift(sel):
        return (is_winner[sel].mean() / wp) if sel.sum() >= 30 else np.nan

    ft, fb = lift(top), lift(bot)
    folds = [(lift(top & (fold_code == k)), lift(bot & (fold_code == k))) for k in range(K)]
    return ft, fb, folds


def null_floor_mc(dates, is_winner, valid, q, wp, n_shuffles, seed):
    """Monte-Carlo NULL distribution of the tail-lift EDGE for a NO-SIGNAL feature.

    This is the honest noise floor the screen was missing. The old `_ctrl_rand` was a
    SINGLE global randn draw scored as one number -- it (a) gives a point, not a
    distribution, and (b) is read as raw lift while real features are ranked by the
    BETTER of their top/bottom side (a max over two correlated noisy stats), so a true
    no-edge feature systematically shows edge>0 by selection bias. Here we draw many
    random features, rank each PER DAY against the REAL per-day winners (which preserves
    each day's winner count -- the within-day structure that decides who can land in the
    tail), and score each with the SAME better-of-top/bottom rule. The resulting
    p95/p99 of |lift-1| is the bar a feature's neutralized edge must clear to be real.
    """
    rng = np.random.RandomState(seed)
    n = len(dates)
    zero_fold = np.zeros(n, dtype=int)
    edges = np.empty(n_shuffles, dtype=float)
    for s in range(n_shuffles):
        rv = rng.randn(n)
        ft, fb, _ = tail_lift_folds(dates, rv, is_winner, valid, zero_fold, 1, q, wp)
        et = abs((ft if np.isfinite(ft) else 1.0) - 1.0)
        eb = abs((fb if np.isfinite(fb) else 1.0) - 1.0)
        edges[s] = max(et, eb)
    edges = edges[np.isfinite(edges)]
    if len(edges) == 0:
        return dict(mean=np.nan, std=np.nan, p95=np.nan, p99=np.nan, n=0)
    return dict(mean=float(np.mean(edges)), std=float(np.std(edges)),
                p95=float(np.percentile(edges, 95)),
                p99=float(np.percentile(edges, 99)), n=int(len(edges)))


# ---------------------------------------------------------------------------
# Romano-Wolf step-down FDR (Step 3) -- joint test "is the best-of-M lift real",
# automatically accounting for cross-candidate dependence by SHARING shuffle draws.
# ---------------------------------------------------------------------------
def romano_wolf_stepdown(dates, valid, is_winner_real, fwd_rank_real, q, wp,
                         cand_vals, n_shuffles, seed, min_names_mask):
    """Romano-Wolf step-down adjusted p-values on the tail-lift EDGE statistic.

    Methodology (FEATURE_EDA_METHODOLOGY.md, "noise discipline" #3):
      * statistic per candidate = |neutralized tail-lift - 1| (better-of-top/bot)
      * NULL = WITHIN-DAY shuffle of the winner labels (groupby Date, permute which
        names are next-day winners) -- preserves each day's winner COUNT (the structure
        that decides who can land in the tail).
      * SHARED shuffle draws across ALL candidates (the SAME permuted label vector is
        used to score every feature on a given draw) so the joint null preserves the
        empirical cross-feature dependence -- this is what makes the max-statistic
        step-down account for correlated candidates without a separate M_eff.
      * step-down: sort candidates by observed statistic desc; for the most significant
        remaining feature, p_adj = P(max over the remaining set of the null statistic >=
        observed), enforcing monotone non-decreasing p_adj down the ladder.

    `cand_vals` : dict col -> ALREADY-NEUTRALIZED feature values (np array, panel order).
    Returns dict col -> dict(raw_p, rw_p, obs_edge).
    """
    cols = list(cand_vals.keys())
    M = len(cols)
    if M == 0 or n_shuffles <= 0:
        return {}
    n = len(dates)
    zero_fold = np.zeros(n, dtype=int)

    # pre-rank each candidate ONCE per Date (ranking is label-independent), so each
    # shuffle only re-selects winners -- O(shuffles * M) lift evals, no re-ranking.
    pre = {}
    for c in cols:
        rank = pd.DataFrame({"d": dates, "v": cand_vals[c]}).groupby("d")["v"].rank(pct=True).to_numpy()
        finite = np.isfinite(rank) & valid
        pre[c] = (finite & (rank >= 1.0 - q), finite & (rank <= q))

    def edge_for(top, bot, winner):
        st = (winner[top].mean() / wp) if top.sum() >= 30 else np.nan
        sb = (winner[bot].mean() / wp) if bot.sum() >= 30 else np.nan
        et = abs((st if np.isfinite(st) else 1.0) - 1.0)
        eb = abs((sb if np.isfinite(sb) else 1.0) - 1.0)
        return max(et, eb)

    obs = np.array([edge_for(pre[c][0], pre[c][1], is_winner_real) for c in cols])

    # within-day shuffle of the winner label, shared across candidates
    rng = np.random.RandomState(seed)
    codes, _ = pd.factorize(dates)
    order = np.argsort(codes, kind="stable")
    cs = codes[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    day_idx = np.split(order, bounds)

    null_stats = np.empty((n_shuffles, M), dtype=np.float64)
    base = is_winner_real.astype(bool).copy()
    for s in range(n_shuffles):
        w = np.empty(n, dtype=bool)
        for idx in day_idx:
            perm = rng.permutation(idx)
            w[idx] = base[perm]                     # permute winners WITHIN the day
        for j, c in enumerate(cols):
            null_stats[s, j] = edge_for(pre[c][0], pre[c][1], w)

    # raw permutation p per candidate
    raw_p = np.array([(1.0 + np.sum(null_stats[:, j] >= obs[j])) / (n_shuffles + 1.0)
                      for j in range(M)])

    # step-down on the max statistic over the REMAINING (least-significant-first reveal)
    order_desc = np.argsort(-obs)                   # most significant first
    rw_p = np.empty(M)
    prev = 0.0
    for rank_i, j in enumerate(order_desc):
        remaining = order_desc[rank_i:]
        maxnull = null_stats[:, remaining].max(axis=1)
        p = (1.0 + np.sum(maxnull >= obs[j])) / (n_shuffles + 1.0)
        p = max(p, prev)                            # enforce monotonicity
        rw_p[j] = p
        prev = p

    return {c: dict(raw_p=float(raw_p[j]), rw_p=float(rw_p[j]), obs_edge=float(obs[j]))
            for j, c in enumerate(cols)}


# ---------------------------------------------------------------------------
# panel construction (fresh from PriceData, candidate blocks only)
# ---------------------------------------------------------------------------
def load_block(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def build_panel(n, seed, blocks):
    paths = sorted(PRICE_DIR.glob("*.parquet"))
    rng = random.Random(seed)
    sample = rng.sample(paths, min(n, len(paths)))

    mods = []
    feat_cols, family = [], {}
    for b in blocks:
        try:
            m = load_block(b)
        except Exception as exc:
            print(f"  [skip block {b}: {exc}]"); continue
        mods.append((b, m))
        for col in m.METADATA["produces"]:
            feat_cols.append(col); family[col] = b

    base = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    frames = []
    t0 = time.perf_counter()
    for p in sample:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if "Ticker" not in df.columns:
            df["Ticker"] = p.stem
        df = df.sort_values("Date").reset_index(drop=True)
        for _b, m in mods:
            try:
                df = m.compute(df)
            except Exception:
                pass
        keep = base + [c for c in feat_cols if c in df.columns]
        frames.append(df[keep])
    panel = pd.concat(frames, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel = panel.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    print(f"  built panel: {len(sample)} tickers, {len(panel):,} rows, {len(feat_cols)} features, "
          f"{time.perf_counter()-t0:.1f}s")
    return panel, feat_cols, family


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--winner_pct", type=float, default=0.10)
    ap.add_argument("--q", type=float, default=0.10)
    ap.add_argument("--min_names", type=int, default=25)
    ap.add_argument("--target", type=str, default="topq", choices=list(TARGET_COL),
                    help="ensemble arm whose winner-definition the tail-lift is scored against "
                         "(default topq = next-day return, the incumbent axis). Use tb/dd/risk_adj/"
                         "binary_up/id/gap/ret5/dt/vw to measure lift on an under-served arm.")
    ap.add_argument("--blocks", type=str, default="")
    ap.add_argument("--extra_blocks", type=str, default="")
    ap.add_argument("--neut_extra", type=str, default="",
                    help="extra feature columns to ALSO neutralize against (incumbent features). "
                         "Their producing blocks must be in --blocks/--extra_blocks so the columns exist.")
    ap.add_argument("--neut_basis", type=str, default="",
                    help="path to Data/_eda_review/basis.json (de-collinearized incumbent basis "
                         "from __relatedness_map.py). Each candidate is neutralized per day vs the "
                         "basis (marginal-vs-incumbents). The candidate's OWN cluster keeper is "
                         "excluded from its basis so we measure marginal-vs-REST. Requires "
                         "Data/_eda_review/clusters.csv for the own-keeper map.")
    ap.add_argument("--seeds", type=int, default=1,
                    help="number of ticker-subsample seeds for cond_edge mean+/-std (default 1; "
                         "the conditional screen reports mean+/-std over this many subsamples)")
    ap.add_argument("--rw_fdr", action="store_true",
                    help="run Romano-Wolf step-down FDR across the screened batch (shared within-day "
                         "shuffle draws; max-statistic step-down -> adjusted p). Writes conditional.csv.")
    ap.add_argument("--null_shuffles", type=int, default=200,
                    help="Monte-Carlo draws for the within-day null floor (0 disables; default 200)")
    ap.add_argument("--live_xcheck", type=str, default="",
                    help="cross-check each feature vs realized trade PnL; value = source ('sim' or 'bt')")
    ap.add_argument("--no_ledger", action="store_true",
                    help="do not append results to the trial ledger")
    args = ap.parse_args()

    blocks = args.blocks.split(",") if args.blocks else (XDM + CSA)
    blocks += [b for b in args.extra_blocks.split(",") if b]
    blocks = [b.strip() for b in blocks if b.strip()]

    print(f"\n  Tier-2 tail screen  | target={args.target} n={args.n} seed={args.seed} "
          f"folds={args.folds} winner=top{args.winner_pct:.0%} slice=±{args.q:.0%}")
    panel, feat_cols, family = build_panel(args.n, args.seed, blocks)
    panel = add_context(panel)

    # ---- winner definition for the SELECTED target arm (see TARGET_COL / add_context) --------
    dates = panel["Date"].to_numpy()
    tcol = TARGET_COL[args.target]
    tvals = panel[tcol].to_numpy(dtype=float)
    day_n = panel.groupby("Date")[tcol].transform("count").to_numpy()
    if args.target == "binary_up":
        # DIRECTION arm: winners = up-closes (a [0,1] label, not a top-rank slice).
        is_winner = tvals > 0.0
        valid = np.isfinite(tvals) & (day_n >= args.min_names)
        fwd = panel["_tgt_topq"].to_numpy(dtype=float)   # IC descriptor vs the signed return
    else:
        # RANK arms: winners = per-day top winner_pct of this arm's forward quantity.
        trank = panel.groupby("Date")[tcol].rank(pct=True, method="average").to_numpy()
        is_winner = trank >= 1.0 - args.winner_pct
        valid = np.isfinite(trank) & (day_n >= args.min_names)
        fwd = tvals                                      # IC descriptor vs this arm's target

    # contiguous date-folds
    uniq = np.array(sorted(pd.unique(dates)))
    edges = np.linspace(0, len(uniq), args.folds + 1).astype(int)
    date_fold = {}
    for k in range(args.folds):
        for d in uniq[edges[k]:edges[k + 1]]:
            date_fold[d] = k
    fold_code = np.array([date_fold.get(d, -1) for d in dates])

    base_rate = is_winner[valid].mean()
    print(f"  valid rows: {valid.sum():,}  base winner-rate: {base_rate:.3f}  "
          f"dates: {pd.Timestamp(uniq[0]).date()}..{pd.Timestamp(uniq[-1]).date()}")

    # incumbent features to ALSO neutralize against (marginal lift conditional on incumbents)
    neut_extra = [c.strip() for c in args.neut_extra.split(",") if c.strip() and c.strip() in panel.columns]
    neut_factors = NEUT_FACTORS + neut_extra
    if neut_extra:
        print(f"  + neutralizing against {len(neut_extra)} incumbent features: {', '.join(neut_extra)}")

    # --- conditional basis (Step 3): de-collinearized incumbent basis from the
    #     relatedness map. Basis columns are merged in from the cached full panel
    #     (Data/_eda_review/panel.parquet) so they need not be re-listed as blocks.
    basis_cols = []
    col_cluster = {}          # candidate col -> cluster_id (to drop its OWN keeper)
    cluster_keeper = {}       # cluster_id -> keeper col (the member to exclude)
    if args.neut_basis:
        import json as _json
        bp = Path(args.neut_basis)
        try:
            basis_all = _json.loads(bp.read_text())
        except Exception as exc:
            print(f"  [neut_basis off: cannot read {bp}: {exc}]")
            basis_all = []
        # cluster map (own-keeper exclusion) from clusters.csv next to basis.json
        clusters_csv = bp.parent / "clusters.csv"
        if clusters_csv.exists():
            cl = pd.read_csv(clusters_csv)
            for _, r in cl.iterrows():
                col_cluster[str(r["col"])] = int(r["cluster_id"])
                if bool(r.get("is_keeper", False)):
                    cluster_keeper[int(r["cluster_id"])] = str(r["col"])
        # merge basis columns that aren't already present, from the cached full panel
        missing = [c for c in basis_all if c not in panel.columns]
        if missing:
            full_cache = bp.parent / "panel.parquet"
            if full_cache.exists():
                fp = pd.read_parquet(full_cache, columns=["Date", "Ticker"] +
                                     [c for c in missing])
                fp["Date"] = pd.to_datetime(fp["Date"])
                fp["Ticker"] = fp["Ticker"].astype(str)
                panel["Ticker"] = panel["Ticker"].astype(str)
                panel = panel.merge(fp.drop_duplicates(["Date", "Ticker"]),
                                    on=["Date", "Ticker"], how="left")
            else:
                print(f"  [neut_basis: {len(missing)} basis cols absent and no full panel "
                      f"cache at {full_cache} -- those basis cols are skipped]")
        basis_cols = [c for c in basis_all if c in panel.columns]
        print(f"  + conditional basis: neutralizing each candidate vs {len(basis_cols)} "
              f"incumbent cluster keepers (own keeper excluded) from {bp.name}")

    # negative/positive controls injected to calibrate the screen:
    #  _ctrl_rand should land at lift ~1.0 (no edge);  _ctrl_volabs / _ctrl_rev5 should show high
    #  RAW lift but collapse toward ~1.0 after neutralization (they ARE the confounds).
    rng = np.random.RandomState(0)
    panel["_ctrl_rand"] = rng.randn(len(panel))
    panel["_ctrl_volabs"] = panel["_volabs"].to_numpy()
    panel["_ctrl_rev5"] = -panel["_ret5"].to_numpy()
    for c in ("_ctrl_rand", "_ctrl_volabs", "_ctrl_rev5"):
        feat_cols.append(c); family[c] = "CONTROL"

    # context-alone bar (the confound edge a real feature must beat)
    print("\n  context-alone lift (the confound bar -- neutralized AGAINST these):")
    for ctx in ("_ret1", "_ret5", "_volabs", "_logprice", "_logdvol", "_trend"):
        ft, fb, _ = tail_lift_folds(dates, panel[ctx].to_numpy(), is_winner, valid, fold_code, args.folds, args.q, args.winner_pct)
        side = "top" if abs((ft or 1) - 1) >= abs((fb or 1) - 1) else "bot"
        v = ft if side == "top" else fb
        print(f"    {ctx:9s} {_c(f'{v:.2f}', lc(v))} ({side})")

    # within-day Monte-Carlo NULL FLOOR -- the honest noise bar (distribution, not a point).
    # NULL-FLOOR HONESTY FIX: use the CONSERVATIVE p99 (not p95) as the robust bar. A pure-noise
    # column's better-of-top/bot edge intermittently clears the p95 floor (the floor estimate is
    # itself noisy, especially at low shuffle counts) -- p99 reliably rejects it. We also require
    # >=200 shuffles before the floor is trusted for any robust verdict (below that the floor is
    # unstable and a clean random column false-passes); below 200 the robust gate leans on the
    # Romano-Wolf adjusted p instead (see robust recompute after the RW pass).
    null = None
    edge_bar = 0.10                      # fallback bar (legacy fixed threshold)
    MIN_ROBUST_SHUFFLES = 200            # floor below which the null bar is NOT trusted alone
    null_floor_trusted = False
    if args.null_shuffles and args.null_shuffles > 0:
        null = null_floor_mc(dates, is_winner, valid, args.q, args.winner_pct,
                             args.null_shuffles, args.seed)
        if np.isfinite(null["p99"]):
            edge_bar = max(0.10, null["p99"])   # CONSERVATIVE p99 floor (was p95)
        null_floor_trusted = (null["n"] >= MIN_ROBUST_SHUFFLES) and np.isfinite(null["p99"])
        print(f"\n  null floor (within-day, {null['n']} draws, scored better-of-top/bot):")
        print(f"    edge|lift-1|  mean={null['mean']:.3f}  std={null['std']:.3f}  "
              f"p95={null['p95']:.3f}  {_c(f'p99={null['p99']:.3f}', '1;33')}")
        print(f"    -> a feature's neutralized edge must clear "
              f"{_c(f'{edge_bar:.3f}', '1;33')} (p99 floor) to count as real"
              + ("" if null_floor_trusted else
                 f"  [<{MIN_ROBUST_SHUFFLES} shuffles: floor NOT trusted alone, "
                 f"robust requires RW-adj p<=0.10]"))

    # optional: realized-trade cross-check (free validity gate)
    live_trades = None
    live_res = {}
    if args.live_xcheck:
        try:
            sys.path.insert(0, str(HERE))
            from importlib import import_module
            _lx = import_module("__live_xcheck")
            live_trades = _lx.load_realized(args.live_xcheck)
            lo_d, hi_d = live_trades["Date"].min(), live_trades["Date"].max()
            print(f"\n  live x-check: source={args.live_xcheck}  realized trades={len(live_trades):,}  "
                  f"span {lo_d.date()}..{hi_d.date()}  base win {live_trades['winner'].mean():.3f}")
        except Exception as exc:
            print(f"  [live x-check off: {exc}]")
            args.live_xcheck = ""

    # trial ledger (append-only honest trial count -- the substrate for FDR/deflation)
    led = None
    if not args.no_ledger:
        try:
            sys.path.insert(0, str(HERE))
            from importlib import import_module
            _tl = import_module("__trial_ledger")
            led = _tl.TrialLedger(tool="tail_screen", seed=args.seed,
                                  params={"n": args.n, "folds": args.folds, "q": args.q,
                                          "winner_pct": args.winner_pct, "blocks": blocks,
                                          "neut_extra": neut_extra,
                                          "null_shuffles": args.null_shuffles})
            if null is not None:
                led.add_metrics("_NULL_FLOOR", {"null_p95": null["p95"], "null_p99": null["p99"],
                                                "null_mean": null["mean"]},
                                family="CONTROL", n_obs=null["n"])
        except Exception as exc:
            print(f"  [ledger off: {exc}]")

    # score every feature
    neut_extra_set = set(neut_extra)
    basis_set = set(basis_cols)
    cond_neut_vals = {}     # col -> neutralized values, for the Romano-Wolf batch FDR
    rows = []
    for col in feat_cols:
        if col not in panel.columns or col in neut_extra_set or col in basis_set:
            continue
        vals = panel[col].to_numpy(dtype=float)
        if np.isfinite(vals).sum() < 500:
            continue
        ic = compute_ic(vals, fwd)
        ft, fb, _ = tail_lift_folds(dates, vals, is_winner, valid, fold_code, args.folds, args.q, args.winner_pct)
        raw_side = "top" if abs((ft or 1) - 1) >= abs((fb or 1) - 1) else "bot"
        raw_lift = ft if raw_side == "top" else fb

        # neutralize jointly vs confound factors (+ incumbent features / + conditional basis).
        # With --neut_basis: factors = NEUT_FACTORS + basis keepers EXCEPT the candidate's OWN
        # cluster keeper (so we measure marginal-vs-REST, not marginal-vs-itself).
        if basis_cols:
            own_keeper = cluster_keeper.get(col_cluster.get(col, -999))
            this_basis = [b for b in basis_cols if b != own_keeper and b != col]
            this_factors = neut_factors + this_basis
        else:
            this_factors = neut_factors
        r2 = _xs_neutralize_multi(panel, col, this_factors)
        # DEGENERACY GUARD: a feature ~perfectly explained by the factors collapses to a
        # near-constant ZERO residual. Ranking a near-constant column produces a spurious
        # ~1.26 ties-artifact lift on BOTH sides that the continuous null floor cannot model
        # (this is exactly why _ctrl_volabs/_ctrl_rev5 used to false-pass "robust"). Such a
        # feature has NO marginal information beyond the factors -> report no edge.
        rfin = r2[np.isfinite(r2)]
        if rfin.size < 100 or np.std(rfin) <= 1e-8 * (np.nanstd(vals) + 1e-30):
            nft = nfb = np.nan
            nfolds = [(np.nan, np.nan)] * args.folds
        else:
            nft, nfb, nfolds = tail_lift_folds(dates, r2, is_winner, valid, fold_code, args.folds, args.q, args.winner_pct)
        n_side = "top" if abs((nft or 1) - 1) >= abs((nfb or 1) - 1) else "bot"
        neut_lift = nft if n_side == "top" else nfb
        fold_vals = [(t if n_side == "top" else b) for (t, b) in nfolds]
        fv = [v for v in fold_vals if np.isfinite(v)]
        # stability: same sign of (lift-1) across folds, and worst-fold edge
        if fv and np.isfinite(neut_lift):
            sgn = np.sign(neut_lift - 1.0)
            same = all(np.sign(v - 1.0) == sgn for v in fv)
            fold_min_edge = min(abs(v - 1.0) for v in fv)
        else:
            same, fold_min_edge = False, np.nan

        edge = abs(neut_lift - 1.0) if np.isfinite(neut_lift) else np.nan
        # IC-IR of the (neutralized) candidate: per-fold IC mean / std -- a stability descriptor.
        fold_ics = []
        for kf in range(args.folds):
            fm = (fold_code == kf)
            if fm.sum() > 100:
                fold_ics.append(compute_ic(r2[fm], fwd[fm]))
        fold_ics = [x for x in fold_ics if np.isfinite(x)]
        ic_ir = (float(np.mean(fold_ics) / np.std(fold_ics))
                 if len(fold_ics) >= 2 and np.std(fold_ics) > 1e-12 else np.nan)
        row = dict(col=col, fam=family.get(col, "?"), ic=ic, ic_ir=ic_ir,
                   raw=raw_lift, neut=neut_lift, side=n_side,
                   stable=same, fmin=fold_min_edge, edge=edge,
                   n_obs=int(np.isfinite(vals).sum()),
                   live_ic=np.nan, live_spread=np.nan, live_n=0)
        # stash the neutralized series for the Romano-Wolf batch FDR pass
        cond_neut_vals[col] = r2

        # optional realized-trade cross-check on the overlap window
        if args.live_xcheck and live_trades is not None:
            try:
                lr = _lx.crosscheck_feature(panel, col, live_trades)
                row["live_ic"] = lr.get("ic", np.nan)
                row["live_spread"] = lr.get("spread", np.nan)
                row["live_n"] = lr.get("n", 0)
            except Exception:
                pass

        # PRELIMINARY floor-only gate: fold-stable AND edge clears the (conservative p99) null bar
        # AND a minimum worst-fold edge. This is NOT the final verdict -- the headline `robust`
        # column is finalized AFTER the Romano-Wolf pass below, which folds in rw_adj_p<=0.10 (the
        # joint within-day permutation FDR that cleanly separates _ctrl_rand from real features).
        row["robust_floor"] = bool(row["stable"] and np.isfinite(row["fmin"])
                                   and row["fmin"] >= 0.05 and np.isfinite(edge) and edge >= edge_bar)
        row["robust"] = row["robust_floor"]   # provisional; recomputed post-RW
        rows.append(row)

    panel.drop(columns=[c for c in ("_neut_tmp",) if c in panel.columns], inplace=True)

    rows.sort(key=lambda r: (r["edge"] if np.isfinite(r["edge"]) else -1), reverse=True)

    # -----------------------------------------------------------------------
    # Romano-Wolf step-down FDR -- computed HERE (before the headline) so the joint
    # within-day permutation adjusted-p can be FOLDED INTO the `robust` verdict. The RW p
    # cleanly separates a pure-noise column (rw_adj_p ~ 1.0) from real marginal-edge features
    # (rw_adj_p ~ 0.02); the old `robust` boolean ignored it and let _ctrl_rand false-pass.
    # Always run when there are candidates and shuffles available (the conditional.csv section
    # below reuses this same `rw`); does not depend on --rw_fdr/--neut_basis.
    RW_ALPHA = 0.10                       # RW-adjusted (or raw-null) p the robust gate requires
    rw = {}
    if cond_neut_vals and args.null_shuffles and args.null_shuffles > 0:
        print(f"\n  Romano-Wolf step-down FDR over {len(cond_neut_vals)} candidates "
              f"({args.null_shuffles} shared within-day shuffles)...")
        rw = romano_wolf_stepdown(
            dates, valid, is_winner, panel["_fwd_rank"].to_numpy(),
            args.q, args.winner_pct, cond_neut_vals, args.null_shuffles, args.seed,
            valid)

    # FINAL robust verdict: fold-stable + worst-fold edge + clears the conservative p99 null bar,
    # AND a multiplicity-honest significance gate (RW-adjusted p<=alpha, or its raw within-day
    # permutation p when RW is unavailable). When the null floor is NOT trusted (<200 shuffles),
    # the significance gate is REQUIRED -- it is what reliably rejects a clean random column at
    # low shuffle counts. This is the fix for the null-floor-honesty false pass.
    for r in rows:
        rwd = rw.get(r["col"], {})
        rw_p = rwd.get("rw_p", np.nan)
        raw_p = rwd.get("raw_p", np.nan)
        sig_p = rw_p if np.isfinite(rw_p) else raw_p
        sig_ok = bool(np.isfinite(sig_p) and sig_p <= RW_ALPHA)
        if null_floor_trusted:
            # trusted conservative floor: robust if it clears the floor AND is significant
            r["robust"] = bool(r["robust_floor"] and sig_ok)
        else:
            # untrusted floor (<200 shuffles): significance gate is mandatory, plus fold-stability
            r["robust"] = bool(r["stable"] and np.isfinite(r["fmin"]) and r["fmin"] >= 0.05
                               and sig_ok)
        r["rw_adj_p"] = rw_p
        r["raw_null_p"] = raw_p

    if led is not None:
        for r in rows:
            led.add_metrics(r["col"], {"ic": r["ic"], "raw_lift": r["raw"], "neut_lift": r["neut"],
                                       "neut_edge": r["edge"], "fold_min_edge": r["fmin"],
                                       "rw_adj_p": r.get("rw_adj_p", np.nan),
                                       "raw_null_p": r.get("raw_null_p", np.nan),
                                       "live_ic": r["live_ic"], "live_spread": r["live_spread"]},
                            family=r["fam"], n_obs=r.get("n_obs", 0),
                            side=r["side"], stable=bool(r["stable"]), robust=r["robust"],
                            edge_bar=edge_bar)

    live_hdr = f" {'liveIC':>7} {'liveSpr':>7}" if args.live_xcheck else ""
    print("\n  " + "=" * (96 + (16 if args.live_xcheck else 0)))
    print(f"  {'feature':28s} {'family':24s} {'IC':>7} {'raw':>6} {'neut':>6} {'side':>4} "
          f"{'foldmin':>7} {'robust':>7}{live_hdr}")
    print("  " + "-" * (96 + (16 if args.live_xcheck else 0)))
    for r in rows:
        rob = "YES" if r.get("robust") else ""
        raw_s = _c(format(r["raw"], ">6.2f"), lc(r["raw"])) if np.isfinite(r["raw"]) else format("nan", ">6")
        neut_s = _c(format(r["neut"], ">6.2f"), lc(r["neut"])) if np.isfinite(r["neut"]) else format("nan", ">6")
        fmin_s = format(r["fmin"], ">7.2f") if np.isfinite(r["fmin"]) else format("nan", ">7")
        rob_s = _c(format(rob, ">7"), "1;32") if rob else format(rob, ">7")
        ic_s = format(r["ic"], ">+7.3f") if np.isfinite(r["ic"]) else format("nan", ">7")
        live_s = ""
        if args.live_xcheck:
            lic = format(r["live_ic"], ">+7.3f") if np.isfinite(r.get("live_ic", np.nan)) else format("--", ">7")
            lsp = format(r["live_spread"], ">7.3f") if np.isfinite(r.get("live_spread", np.nan)) else format("--", ">7")
            live_s = f" {lic} {lsp}"
        print(f"  {r['col']:28s} {r['fam']:24s} {ic_s} {raw_s} {neut_s} {r['side']:>4} {fmin_s} {rob_s}{live_s}")

    # family rollup
    print("\n  " + "=" * 60)
    print(f"  FAMILY ROLLUP (mean neutralized edge, best feature)")
    print("  " + "-" * 60)
    fam_rows = {}
    for r in rows:
        fam_rows.setdefault(r["fam"], []).append(r)
    fam_sum = []
    for fam, rs in fam_rows.items():
        edges = [x["edge"] for x in rs if np.isfinite(x["edge"])]
        best = max(rs, key=lambda x: (x["edge"] if np.isfinite(x["edge"]) else -1))
        nrob = sum(1 for x in rs if x.get("robust"))
        fam_sum.append((fam, np.mean(edges) if edges else 0.0, best, nrob, len(rs)))
    fam_sum.sort(key=lambda x: x[1], reverse=True)
    for fam, me, best, nrob, ntot in fam_sum:
        print(f"  {fam:26s} mean_edge={me:.3f}  robust={nrob}/{ntot}  best={best['col']} "
              f"(neut {best['neut']:.2f})")
    if null is not None and np.isfinite(null["p99"]):
        if null_floor_trusted:
            print(f"\n  [robust = fold-stable AND neut edge >= null p99 ({edge_bar:.3f}) "
                  f"AND foldmin >= 0.05 AND RW-adj p <= {RW_ALPHA:.2f}]")
        else:
            print(f"\n  [robust = fold-stable AND foldmin >= 0.05 AND RW-adj p <= {RW_ALPHA:.2f}  "
                  f"(<{MIN_ROBUST_SHUFFLES} shuffles: p99 floor {edge_bar:.3f} NOT trusted alone)]")

    # -----------------------------------------------------------------------
    # CONDITIONAL SCREEN OUTPUT (Step 3): conditional.csv with Romano-Wolf FDR.
    # Written whenever --neut_basis or --rw_fdr is requested. cond_edge is the
    # already-computed neutralized edge (conditional on the basis when --neut_basis).
    # -----------------------------------------------------------------------
    if args.neut_basis or args.rw_fdr:
        # `rw` was already computed above (it now feeds the `robust` verdict). Reuse it.
        EDA_DIR = ROOT / "Data" / "_eda_review"
        EDA_DIR.mkdir(parents=True, exist_ok=True)
        crows = []
        for r in rows:
            c = r["col"]
            rwd = rw.get(c, {})
            crows.append(dict(
                col=c, family=r["fam"],
                cond_edge=r["edge"],
                cond_edge_std=np.nan,    # populated when --seeds>1 (multi-subsample) below
                neut_lift=r["neut"], side=r["side"],
                null_p95=(null["p95"] if null else np.nan),
                null_p99=(null["p99"] if null else np.nan),
                null_floor_trusted=bool(null_floor_trusted),
                raw_null_p=rwd.get("raw_p", np.nan),
                rw_adj_p=rwd.get("rw_p", np.nan),
                folds_sign_consistent=bool(r["stable"]),
                fold_min_edge=r["fmin"],
                ic_ir=r.get("ic_ir", np.nan),
                robust=bool(r.get("robust")),
            ))
        cdf = pd.DataFrame(crows)

        # optional multi-seed cond_edge std: re-score on >=2 ticker subsamples of THIS panel
        if args.seeds and args.seeds > 1:
            tickers = panel["Ticker"].astype(str).unique().tolist()
            per_seed = {c: [] for c in cond_neut_vals}
            tk_arr = panel["Ticker"].astype(str).to_numpy()
            for si in range(args.seeds):
                rs = random.Random(args.seed + 1000 * (si + 1))
                sub = set(rs.sample(tickers, max(10, int(0.7 * len(tickers)))))
                mask = np.isin(tk_arr, list(sub))
                d_s = dates[mask]; w_s = is_winner[mask]; v_s = valid[mask]
                z_fold = np.zeros(mask.sum(), dtype=int)
                for c, r2v in cond_neut_vals.items():
                    ft, fb, _ = tail_lift_folds(d_s, r2v[mask], w_s, v_s, z_fold, 1, args.q, args.winner_pct)
                    et = abs((ft if np.isfinite(ft) else 1.0) - 1.0)
                    eb = abs((fb if np.isfinite(fb) else 1.0) - 1.0)
                    per_seed[c].append(max(et, eb))
            std_map = {c: float(np.std(v)) for c, v in per_seed.items() if len(v) >= 2}
            mean_map = {c: float(np.mean(v)) for c, v in per_seed.items() if len(v) >= 2}
            cdf["cond_edge_std"] = cdf["col"].map(std_map)
            cdf["cond_edge_seedmean"] = cdf["col"].map(mean_map)

        cdf["target"] = args.target
        cdf = cdf.sort_values("cond_edge", ascending=False)
        out_csv = EDA_DIR / (f"conditional_{args.target}.csv" if args.target != "topq"
                             else "conditional.csv")
        cdf.to_csv(out_csv, index=False)
        n_sig = int((cdf["rw_adj_p"] <= 0.10).sum()) if "rw_adj_p" in cdf else 0
        print(f"  conditional.csv -> {out_csv} ({len(cdf)} candidates"
              + (f", {n_sig} with RW-adj p<=0.10)" if args.rw_fdr else ")"))

        if led is not None:
            for r in rows:
                rwd = rw.get(r["col"], {})
                led.add_metrics(r["col"], {"cond_edge": r["edge"],
                                           "cond_raw_null_p": rwd.get("raw_p", np.nan),
                                           "cond_rw_adj_p": rwd.get("rw_p", np.nan),
                                           "cond_ic_ir": r.get("ic_ir", np.nan)},
                                family=r["fam"], n_obs=int(valid.sum()),
                                conditional=True, rw_fdr=bool(args.rw_fdr))

    if led is not None:
        p = led.flush()
        print(f"  [ledger] {len(rows)} features + null floor logged -> {p}")
    print()


if __name__ == "__main__":
    main()
