#!/usr/bin/env python3
"""
Book Post-Mortem — "how could I have prevented the loser?" discriminator harness
================================================================================
A REPEATABLE post-trade tool. You paste today's live book (winners + losers), it:

  1. Snapshots a battery of statistical properties for every held name, as of the
     latest available daily bar (the info that WAS knowable when the trade was placed).
  2. Isolates which single property most cleanly separates the winners from the
     losers in *this* book (Spearman vs unrealized P&L, plus loser-outlier z-score).
  3. Applies that separator to the ENTIRE historical trade book (trade_history.parquet
     joined to the intraday fill sim, 8__IntradayFillSim.py) and reports how the
     simulated per-trade distribution shifts if you'd filtered on it.
  4. APPENDS today's per-name snapshot to a ledger parquet so that, over weeks, these
     daily snapshots pool into a real sample. Once the ledger has >1 day, step 2 is
     re-run on the POOLED winner/loser set — that pooled result is the only one worth
     trusting; a single day (esp. today, n_loser=1) is anecdotal.

HONESTY GUARDS (this repo's culture — see project_canbuy_topk_alignment / sizer verdict):
  - A per-trade distribution screen is a RELATIVE signal, NOT a full-sim equity claim.
    Slot concurrency means top-k style filters that look great per-trade routinely LOSE
    in the full slot-based sim. So the equity number here is an INDICATIVE proxy only;
    confirm any promising separator with a real 5__NightlyBackTester_asof.py run.
  - n_loser is printed loudly. Do not read robustness into one day.

Usage
-----
    python 8b__BookPostmortem.py                 # uses the built-in TODAY_BOOK below
    python 8b__BookPostmortem.py --book book.csv # cols: Symbol,Qty,AvgPx,Last  (unreal% derived)
    python 8b__BookPostmortem.py --no-log        # don't append to the ledger

CSV/edit the TODAY_BOOK block each morning, re-run, and the ledger grows itself.
"""

import os
import argparse
from datetime import date as _date

import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

# ── TODAY'S BOOK ────────────────────────────────────────────────────────────────
# Edit this each day (or pass --book a CSV). Symbol, shares, avg entry px, last px.
# Unrealized % is derived = (Last/AvgPx - 1)*100. Position=0 rows are ignored.
TODAY_BOOK = [
    # Symbol, Qty, AvgPx,  Last
    ("MGNI",  33,  20.34,  20.20),
    ("INSM",   6, 109.57, 110.21),
    ("SMMT",  44,  15.45,  15.52),
    ("BIRK",  15,  44.15,  44.68),
    ("PTON", 113,   6.04,   6.11),
    ("VRDN",  37,  18.68,  19.13),
    ("ACVA", 100,   6.68,   6.97),
    ("INSP",  15,  46.15,  49.23),
    ("RHI",   20,  33.30,  35.67),
]
BOOK_DATE = "2026-07-14"   # the trading day this book snapshot is from

V2_DIR      = os.path.join(script_dir, "Data", "ProcessedData_v2")
SIM_TRADES  = os.path.join(script_dir, "Data", "IntradayFillSim", "sim_trades.parquet")
TRADE_HIST  = os.path.join(script_dir, "trade_history.parquet")
LEDGER      = os.path.join(script_dir, "analysis_output", "book_postmortem_ledger.parquet")
REPORT      = os.path.join(script_dir, "analysis_output", "book_postmortem_report.md")
TAIL_REPORT = os.path.join(script_dir, "analysis_output", "book_tail_report.md")


# ── Statistical-property battery (all computable from v2 OHLCV → backtestable) ────
def compute_battery(df: pd.DataFrame) -> pd.DataFrame:
    """Given a v2 frame (Date, OHLCV, +features) sorted ascending, return a frame
    indexed by Date with an interpretable, non-redundant property battery. Every
    property is knowable at the close of that bar (no look-ahead)."""
    d = df.sort_values("Date").reset_index(drop=True).copy()
    o, h, l, c, v = (d["Open"], d["High"], d["Low"], d["Close"], d["Volume"])
    ret = c.pct_change()

    out = pd.DataFrame(index=d["Date"])
    out["close_$"]      = c.values
    # ATR% — use the shipped column if present, else a 14d proxy.
    if "ATR%" in d.columns:
        out["atr_pct"]  = d["ATR%"].values
    else:
        tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        out["atr_pct"]  = (tr.rolling(14).mean() / c * 100).values
    out["rv20"]         = (ret.rolling(20).std() * np.sqrt(252) * 100).values      # ann realized vol %
    if "RSI" in d.columns:
        out["rsi14"]    = d["RSI"].values
    else:
        up = ret.clip(lower=0).rolling(14).mean(); dn = (-ret.clip(upper=0)).rolling(14).mean()
        out["rsi14"]    = (100 - 100 / (1 + up / dn.replace(0, np.nan))).values
    out["mom_5d"]       = (c / c.shift(5)  - 1).values * 100
    out["mom_20d"]      = (c / c.shift(20) - 1).values * 100
    out["mom_60d"]      = (c / c.shift(60) - 1).values * 100
    out["dist_hi_20"]   = (c / h.rolling(20).max()  - 1).values * 100              # <=0, distance below recent high
    out["dist_hi_252"]  = (c / h.rolling(252).max() - 1).values * 100
    out["prox_lo_20"]   = (c / l.rolling(20).min()  - 1).values * 100              # >=0, cushion above recent low
    out["ext_sma20"]    = (c / c.rolling(20).mean() - 1).values * 100              # stretch vs its own 20d mean
    gap = (o / c.shift() - 1)
    out["gap_vol_20"]   = (gap.rolling(20).std() * 100).values                     # overnight gappiness
    out["log_dvol20"]   = np.log10((c * v).rolling(20).mean().replace(0, np.nan)).values
    return out


BATTERY_COLS = ["close_$", "atr_pct", "rv20", "rsi14", "mom_5d", "mom_20d", "mom_60d",
                "dist_hi_20", "dist_hi_252", "prox_lo_20", "ext_sma20", "gap_vol_20", "log_dvol20"]

PRETTY = {
    "close_$": "price level ($)", "atr_pct": "ATR % (volatility)", "rv20": "realized vol 20d (ann%)",
    "rsi14": "RSI(14)", "mom_5d": "momentum 5d (%)", "mom_20d": "momentum 20d (%)",
    "mom_60d": "momentum 60d (%)", "dist_hi_20": "dist below 20d high (%)",
    "dist_hi_252": "dist below 52w high (%)", "prox_lo_20": "cushion above 20d low (%)",
    "ext_sma20": "stretch vs 20d SMA (%)", "gap_vol_20": "overnight gap vol (%)",
    "log_dvol20": "log10 $-volume 20d",
}


def load_book(args) -> pd.DataFrame:
    if args.book:
        b = pd.read_csv(args.book)
        b.columns = [c.strip() for c in b.columns]
    else:
        b = pd.DataFrame(TODAY_BOOK, columns=["Symbol", "Qty", "AvgPx", "Last"])
    b = b[b["Qty"].astype(float) != 0].copy()
    b["unreal_pct"] = (b["Last"].astype(float) / b["AvgPx"].astype(float) - 1) * 100
    b["is_loser"] = (b["unreal_pct"] < 0).astype(int)
    return b.reset_index(drop=True)


def battery_asof(symbol: str, asof: pd.Timestamp | None) -> pd.Series | None:
    fp = os.path.join(V2_DIR, f"{symbol}.parquet")
    if not os.path.exists(fp):
        return None
    df = pd.read_parquet(fp)
    if "Date" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    bat = compute_battery(df)
    if asof is None:
        row = bat.iloc[-1]
    else:
        sub = bat[bat.index <= asof]
        if sub.empty:
            return None
        row = sub.iloc[-1]
    return row


# ── Step 2: discrimination on a labelled set (works for today OR the pooled ledger) ──
def rank_discriminators(feat: pd.DataFrame, target="unreal_pct", loser_col="is_loser") -> pd.DataFrame:
    rows = []
    losers = feat[feat[loser_col] == 1]
    winners = feat[feat[loser_col] == 0]
    for col in BATTERY_COLS:
        x = feat[col].astype(float)
        if x.notna().sum() < 3 or x.nunique() < 2:
            continue
        rho = x.corr(feat[target], method="spearman")
        # Loser-outlier z: how many book-SDs the mean loser sits from the mean winner.
        z = np.nan
        if len(losers) and len(winners) and winners[col].std(ddof=0) > 0:
            z = (losers[col].mean() - winners[col].mean()) / feat[col].std(ddof=0)
        rows.append({
            "property": col, "pretty": PRETTY[col],
            "spearman_vs_pnl": rho,
            "loser_z": z,                       # <0 => losers score LOW on this prop
            "win_mean": winners[col].mean(), "los_mean": losers[col].mean(),
            "abs_signal": np.nanmax([abs(rho) if pd.notna(rho) else 0,
                                     abs(z) / 2 if pd.notna(z) else 0]),
        })
    r = pd.DataFrame(rows).sort_values("abs_signal", ascending=False).reset_index(drop=True)
    return r


# ── Step 3: apply a separator to the full historical book + intraday sim ─────────
def apply_to_history(prop: str, direction: int, cut: float, sim: pd.DataFrame,
                     v2_feats: dict) -> dict:
    """direction=+1 keeps trades with prop >= cut (winner side is HIGH), -1 keeps <= cut.
    Returns baseline vs kept per-trade stats on the intraday-sim net return."""
    s = sim[sim["Status"] == "OK"].copy()
    # attach the property at each trade's EntryDate
    vals = []
    for _, r in s.iterrows():
        bat = v2_feats.get(r["Symbol"])
        if bat is None:
            vals.append(np.nan); continue
        sub = bat[bat.index <= pd.Timestamp(r["EntryDate"])]
        vals.append(sub[prop].iloc[-1] if len(sub) else np.nan)
    s[prop] = vals
    s = s.dropna(subset=[prop, "SimPnLPctNet"])
    keep = s[prop] >= cut if direction > 0 else s[prop] <= cut
    return {"all": s, "kept_mask": keep, "prop": prop, "cut": cut, "direction": direction}


def book_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, win=np.nan, sharpe=np.nan, equity=np.nan)
    sh = x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 1 and x.std(ddof=1) > 0 else 0.0
    eq = np.prod(1 + x / 100.0)     # per-trade compounding proxy (NOT slot-aware)
    return dict(n=len(x), mean=x.mean(), median=np.median(x),
                win=(x > 0).mean() * 100, sharpe=sh, equity=eq)


def build_labeled_history():
    """Join the property battery (at each trade's EntryDate) onto the intraday-sim
    net return for EVERY historical trade → a labeled winner/loser table. This is the
    real loser sample (hundreds of losers) vs the 1-loser live book."""
    sim = pd.read_parquet(SIM_TRADES)
    sim["EntryDate"] = pd.to_datetime(sim["EntryDate"])
    sim = sim[sim["Status"] == "OK"].copy()
    feats = {}
    rows = []
    for _, r in sim.iterrows():
        sym = r["Symbol"]
        if sym not in feats:
            fp = os.path.join(V2_DIR, f"{sym}.parquet")
            if os.path.exists(fp):
                df = pd.read_parquet(fp); df["Date"] = pd.to_datetime(df["Date"])
                feats[sym] = compute_battery(df)
            else:
                feats[sym] = None
        bat = feats[sym]
        if bat is None:
            continue
        sub = bat[bat.index <= pd.Timestamp(r["EntryDate"])]
        if sub.empty:
            continue
        row = {"Symbol": sym, "EntryDate": r["EntryDate"], "ret": r["SimPnLPctNet"]}
        for c in BATTERY_COLS:
            row[c] = sub[c].iloc[-1]
        rows.append(row)
    hist = pd.DataFrame(rows).dropna(subset=["ret"])
    hist["is_loser"] = (hist["ret"] < 0).astype(int)
    return hist, feats


def mine_history():
    """Univariate + multivariate loser pattern-mine over the full historical book."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    print("Building labeled history (battery at entry → sim net return)...")
    hist, feats = build_labeled_history()
    n = len(hist); base_loss = hist["is_loser"].mean() * 100
    print(f"  {n} labeled trades | baseline loser rate {base_loss:.1f}%")

    L = ["# Loser Pattern-Mine — full historical book\n",
         f"Labeled trades: **{n}**  |  baseline loser rate **{base_loss:.1f}%**  "
         f"(loser = intraday-sim net return < 0)\n",
         "> This is your REAL loser sample. Losers you can't separate here you won't separate "
         "in the live book either. Caveat: the XGBoost model already trained on most of these "
         "properties, so weak univariate signal is EXPECTED — the multivariate CV-AUC at the "
         "bottom is the honest 'are losers patternable at all beyond noise' answer.\n"]

    # ── univariate: per-property loser discrimination ───────────────────────────
    from numpy import nan
    rows = []
    for c in BATTERY_COLS:
        x = hist[c].astype(float)
        m = x.notna()
        if m.sum() < 50 or x[m].nunique() < 5:
            continue
        pb = np.corrcoef(x[m], hist["is_loser"][m])[0, 1]      # point-biserial
        try:
            auc = roc_auc_score(hist["is_loser"][m], x[m])
        except Exception:
            auc = nan
        # decile loser-rate spread (top decile minus bottom decile of the property)
        try:
            dq = pd.qcut(x[m], 10, duplicates="drop")
            lr = hist["is_loser"][m].groupby(dq, observed=True).mean() * 100
            spread = lr.iloc[-1] - lr.iloc[0]
            mono = lr.is_monotonic_increasing or lr.is_monotonic_decreasing
        except Exception:
            spread, mono = nan, False
        rows.append({"property": c, "pretty": PRETTY[c], "loser_corr": pb,
                     "auc": max(auc, 1 - auc) if pd.notna(auc) else nan,
                     "auc_dir": "high=loser" if (pd.notna(auc) and auc > 0.5) else "low=loser",
                     "decile_spread_pp": spread, "monotone": mono,
                     "win_mean": x[m][hist["is_loser"][m] == 0].mean(),
                     "los_mean": x[m][hist["is_loser"][m] == 1].mean()})
    uni = pd.DataFrame(rows)
    uni["signal"] = (uni["auc"] - 0.5).abs()
    uni = uni.sort_values("signal", ascending=False).reset_index(drop=True)
    L.append("## Univariate — which entry properties precede losers\n")
    L.append("`auc` folded so >0.5 = separates (0.50 = useless); `auc_dir` says which end is the loser end. "
             "`decile_spread_pp` = loser-rate in the worst property-decile minus the best.\n```")
    L.append(uni[["pretty", "loser_corr", "auc", "auc_dir", "decile_spread_pp",
                  "monotone", "win_mean", "los_mean"]].to_string(index=False,
             float_format=lambda v: f"{v:.3f}"))
    L.append("```\n")

    # decile detail for the single strongest property
    top = uni.iloc[0]["property"]
    x = hist[top].astype(float); m = x.notna()
    dq = pd.qcut(x[m], 10, duplicates="drop")
    tbl = pd.DataFrame({
        "decile": range(1, dq.cat.categories.size + 1),
        "prop_range": [f"{iv.left:.2f}..{iv.right:.2f}" for iv in dq.cat.categories],
        "n": hist["is_loser"][m].groupby(dq, observed=True).size().values,
        "loser_%": (hist["is_loser"][m].groupby(dq, observed=True).mean() * 100).values,
        "mean_ret_%": hist["ret"][m].groupby(dq, observed=True).mean().values,
    })
    L.append(f"### Decile detail — strongest property: **{PRETTY[top]}**\n```")
    L.append(tbl.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    L.append("```\n")

    # ── multivariate: can a model separate losers out-of-sample? ────────────────
    L.append("## Multivariate — are losers patternable AT ALL (out-of-sample)? ★\n")
    hh = hist.sort_values("EntryDate").reset_index(drop=True)
    X = hh[BATTERY_COLS].astype(float).fillna(hh[BATTERY_COLS].astype(float).median())
    y = hh["is_loser"].values
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    aucs = []
    tss = TimeSeriesSplit(n_splits=5)
    for tr, te in tss.split(X):
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        clf.fit(X.iloc[tr], y[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    cv_auc = float(np.mean(aucs)) if aucs else float("nan")
    L.append(f"Time-series-CV out-of-sample loser-AUC (logistic on all 13 props): **{cv_auc:.3f}**\n")
    if cv_auc >= 0.55:
        L.append("→ **There IS residual loser structure** the entry model isn't fully pricing. "
                 "Worth chasing: fit on the pooled live ledger as it grows and A/B a down-size gate.\n")
    elif cv_auc >= 0.52:
        L.append("→ **Marginal.** A whisker above coin-flip — likely too weak to trade, but keep the "
                 "ledger going; if it strengthens on live data it's real.\n")
    else:
        L.append("→ **Losers are NOT separable** from these entry properties out-of-sample (AUC≈0.5). "
                 "The model already prices them; loss is timing/luck, not a pre-trade signature. Don't "
                 "build a filter on this — the lever is elsewhere (target/sizing/exits).\n")

    # coefficient direction (in-sample, standardized) for interpretability
    clf.fit(X, y)
    coefs = pd.Series(clf.named_steps["logisticregression"].coef_[0], index=BATTERY_COLS)
    coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
    L.append("Standardized logistic coefficients (in-sample, + = pushes toward LOSER):\n```")
    L.append("\n".join(f"  {PRETTY[c]:<26} {coefs[c]:+.3f}" for c in coefs.index))
    L.append("```\n")

    out = os.path.join(os.path.dirname(REPORT), "loser_pattern_mine_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n" + "=" * 72)
    print("  LOSER PATTERN-MINE (full historical book)")
    print("=" * 72)
    print(f"  trades={n}  baseline loser rate={base_loss:.1f}%")
    print(f"  top univariate loser-predictors (folded AUC):")
    for _, r in uni.head(5).iterrows():
        print(f"    {r['pretty']:<26} auc={r['auc']:.3f} ({r['auc_dir']})  "
              f"decile_spread={r['decile_spread_pp']:+.1f}pp")
    print(f"  MULTIVARIATE out-of-sample loser-AUC = {cv_auc:.3f}  "
          f"({'patternable' if cv_auc>=0.55 else 'marginal' if cv_auc>=0.52 else 'NOT patternable'})")
    print("=" * 72)
    print(f"  Report : {os.path.relpath(out, script_dir)}")
    print("=" * 72)


# ══════════════════════════════════════════════════════════════════════════════
#  TAIL MODE — quantify the fat-right-tail / convexity that CARRIES the strategy
# ══════════════════════════════════════════════════════════════════════════════
# Different question than --mine-history. That one asks "what precedes a LOSER"
# (settled: no per-name price signature, only regime — model already prices it).
# THIS asks "what precedes a TAIL WINNER (top-decile, the 5-25% moonshots)" and
# "how does the tail FORM during the hold" — because the lever this repo keeps
# landing on is SIZING/REGIME, not rejection. If tail-probability has a per-name
# signature you don't filter, you TILT SIZE toward it. Source is the DEPLOYED book
# (trade_history.parquet PnLPct = max-hold), NOT the bracket-clipped intraday sim
# (that sim has no tail: p99 ~ +5.5%). See project_book_postmortem_harness.

def _gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if n == 0 or x[-1] == 0:
        return np.nan
    c = np.cumsum(x)
    return (n + 1 - 2 * np.sum(c) / c[-1]) / n


def load_deployed_book():
    """The real max-hold trade record. Returns df with PnLPct, EntryPrice, dates."""
    t = pd.read_parquet(TRADE_HIST)
    t["EntryDate"] = pd.to_datetime(t["EntryDate"])
    t["ExitDate"]  = pd.to_datetime(t["ExitDate"])
    return t.dropna(subset=["PnLPct"]).reset_index(drop=True)


# ── Family A: tail-carry / fragility ────────────────────────────────────────────
def tail_carry_report(t: pd.DataFrame, pctl: float, abs_thr: float | None) -> list:
    from scipy import stats
    r = t["PnLPct"].to_numpy(float)
    n = len(r); pos = r[r > 0]
    tail_cut = abs_thr if abs_thr is not None else np.percentile(r, pctl)
    L = ["## A. Tail-carry / fragility — is the edge tail-driven?\n",
         f"Deployed book: **{n}** trades | win {100*(r>0).mean():.1f}% | "
         f"mean **{r.mean():+.3f}%** | median {np.median(r):+.3f}%\n",
         f"skew **{stats.skew(r):+.3f}**  excess-kurtosis **{stats.kurtosis(r):+.3f}**  "
         f"(both >0 → fat right tail / convex). Gini(winner PnL%) **{_gini(pos):.3f}** "
         f"(1=one trade owns all the gains).\n"]
    w, l = r[r > 0], r[r <= 0]
    pr = w.mean() / -l.mean() if len(l) and l.mean() != 0 else np.nan
    L.append(f"Expectancy = P(win)·avgWin − P(loss)·avgLoss = "
             f"{(r>0).mean():.3f}·{w.mean():+.2f}% − {(r<=0).mean():.3f}·{-l.mean():.2f}% "
             f"→ payoff ratio **{pr:.2f}**\n")

    # Pareto tail-carry
    srt = np.sort(r)[::-1]
    L.append("**Tail-carry (share of GROSS-POSITIVE PnL from the top-k% of trades):**\n```")
    for k in (0.01, 0.05, 0.10, 0.25):
        m = max(1, int(n * k))
        L.append(f"  top {k*100:4.1f}% (n={m:4d})  = {srt[:m].sum()/pos.sum()*100:5.1f}% of gains")
    L.append("```\n")

    # kill-the-tail fragility
    L.append("**Kill-the-tail (edge remaining if the top-N winners never happened):**\n```")
    base = r.sum()
    for kk in (1, 3, 5, 10, 20):
        rem = srt[kk:]
        L.append(f"  remove top {kk:2d}: sumPnL {rem.sum():+7.1f}  ({(1-rem.sum()/base)*100:4.0f}% of edge gone)")
    L.append("```\n")

    # tail bins
    bins = [(-1e9, -3, "< -3%"), (-3, 0, "-3..0%"), (0, 3, "0..3%"), (3, 5, "3..5%"),
            (5, 10, "5..10%"), (10, 25, "10..25%"), (25, 1e9, ">= 25%")]
    L.append("**Outcome bins:**\n```")
    for lo, hi, lab in bins:
        m = (r >= lo) & (r < hi)
        L.append(f"  {lab:<9} n={m.sum():4d} ({m.mean()*100:4.1f}%)  ΣPnL {r[m].sum():+8.1f}")
    L.append("```\n")

    # expectancy over time — is the tail thickening or thinning?
    tm = t.copy(); tm["mo"] = tm["EntryDate"].dt.to_period("M")
    L.append(f"**Health monitor — expectancy by month** (tail_share = % of the month's "
             f"gains from trades ≥ {'p%g' % pctl if abs_thr is None else '%g%%' % abs_thr} "
             f"= {tail_cut:+.2f}%):\n```")
    L.append(f"  {'month':<8} {'n':>4} {'win%':>6} {'mean%':>7} {'payoff':>7} {'tail_share%':>11}")
    for mo, g in tm.groupby("mo"):
        rr = g["PnLPct"].to_numpy(float); gpos = rr[rr > 0]
        gw, gl = rr[rr > 0], rr[rr <= 0]
        gpr = gw.mean() / -gl.mean() if len(gl) and gl.mean() != 0 else np.nan
        tsh = rr[rr >= tail_cut].sum() / gpos.sum() * 100 if len(gpos) else np.nan
        L.append(f"  {str(mo):<8} {len(rr):>4} {100*(rr>0).mean():>6.1f} {rr.mean():>+7.2f} "
                 f"{gpr:>7.2f} {tsh:>11.1f}")
    L.append("```\n")
    return L, tail_cut


# ── Family B: tail-winner conditioning scan (mirror of the loser mine) ───────────
def build_labeled_deployed(t: pd.DataFrame):
    """Join the 13-prop entry battery onto each DEPLOYED trade's max-hold PnLPct."""
    feats, rows = {}, []
    for _, r in t.iterrows():
        sym = r["Symbol"]
        if sym not in feats:
            fp = os.path.join(V2_DIR, f"{sym}.parquet")
            if os.path.exists(fp):
                df = pd.read_parquet(fp); df["Date"] = pd.to_datetime(df["Date"])
                feats[sym] = compute_battery(df)
            else:
                feats[sym] = None
        bat = feats[sym]
        if bat is None:
            continue
        sub = bat[bat.index <= r["EntryDate"]]
        if sub.empty:
            continue
        row = {"Symbol": sym, "EntryDate": r["EntryDate"], "ret": r["PnLPct"]}
        for c in BATTERY_COLS:
            row[c] = sub[c].iloc[-1]
        rows.append(row)
    return pd.DataFrame(rows).dropna(subset=["ret"]), feats


def _within_day_frac(hist, col):
    """Fraction of the property's entry-variance that is BETWEEN-day (market-wide).
    ~1 → regime signal (picks DAYS, can't rank names in a book). ~0 → cross-sectional
    (picks NAMES, usable as a size-tilt). Mirrors the 07-16 regime/xsec split."""
    x = hist[[col, "EntryDate"]].dropna()
    tot = x[col].var()
    if not tot or len(x) < 20:
        return np.nan
    return x.groupby("EntryDate")[col].mean().var() / tot


def mine_tail(t: pd.DataFrame, pctl: float, abs_thr: float | None) -> list:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    hist, _ = build_labeled_deployed(t)
    cut = abs_thr if abs_thr is not None else np.percentile(hist["ret"], pctl)
    hist["is_tail"] = (hist["ret"] >= cut).astype(int)
    n, ntail = len(hist), int(hist["is_tail"].sum())
    L = ["## B. Tail-winner conditioning — does a moonshot have an ENTRY signature?\n",
         f"Labeled {n} deployed trades | tail = PnLPct ≥ **{cut:+.2f}%** "
         f"({'p%g' % pctl if abs_thr is None else 'abs'}) → **{ntail}** tail winners "
         f"({100*ntail/n:.1f}%).\n",
         "> Same honesty guards as the loser mine: the entry model already trained on these "
         "props so weak univariate signal is EXPECTED. The permutation floor and OOS multivariate "
         "AUC are the honest 'is there anything beyond noise' answers. `within_day` says whether a "
         "hit picks DAYS (regime → book-level dose lever) or NAMES (cross-sectional → size-tilt lever).\n"]

    rng = np.random.default_rng(0)
    y = hist["is_tail"].to_numpy()
    rows, null_best = [], []
    # permutation noise floor: best folded-AUC over the 13 props under shuffled labels
    for _ in range(200):
        yp = rng.permutation(y); best = 0.0
        for c in BATTERY_COLS:
            x = hist[c].to_numpy(float); m = ~np.isnan(x)
            if m.sum() < 50:
                continue
            try:
                a = roc_auc_score(yp[m], x[m]); best = max(best, max(a, 1 - a))
            except Exception:
                pass
        null_best.append(best)
    floor = float(np.percentile(null_best, 95))

    for c in BATTERY_COLS:
        x = hist[c].to_numpy(float); m = ~np.isnan(x)
        if m.sum() < 50 or np.unique(x[m]).size < 5:
            continue
        try:
            auc = roc_auc_score(y[m], x[m])
        except Exception:
            continue
        folded = max(auc, 1 - auc)
        rows.append({"pretty": PRETTY[c],
                     "auc": folded,
                     "dir": "high→tail" if auc > 0.5 else "low→tail",
                     "beats_floor": "YES" if folded > floor else "-",
                     "within_day": _within_day_frac(hist, c),
                     "win_mean": x[m][y[m] == 1].mean(),
                     "rest_mean": x[m][y[m] == 0].mean()})
    uni = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    L.append(f"**Univariate — entry props preceding a tail winner** "
             f"(permutation noise floor, best-of-13 p95 = **{floor:.3f}**; "
             f"beat it to be real):\n```")
    L.append(uni.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    L.append("```\n")

    # multivariate OOS
    hh = hist.sort_values("EntryDate").reset_index(drop=True)
    X = hh[BATTERY_COLS].astype(float)
    X = X.fillna(X.median())
    yy = hh["is_tail"].to_numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    aucs = []
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        if yy[tr].sum() == 0 or yy[te].sum() == 0:
            continue
        clf.fit(X.iloc[tr], yy[tr])
        aucs.append(roc_auc_score(yy[te], clf.predict_proba(X.iloc[te])[:, 1]))
    cv = float(np.mean(aucs)) if aucs else float("nan")
    L.append(f"**Multivariate out-of-sample tail-AUC (logistic, TimeSeriesSplit-5): {cv:.3f}** ★\n")
    if cv >= 0.55:
        L.append("→ Tail winners ARE forward-patternable. Cross the `within_day` column: low → build a "
                 "SIZE-TILT toward high-tail-prob names; high → it's a regime day-picker (dose). "
                 "Confirm any tilt in a real `5__NightlyBackTester_asof.py` run.\n")
    elif cv >= 0.52:
        L.append("→ Marginal. A whisker above coin-flip — keep it on the watch-list, don't size on it yet.\n")
    else:
        L.append("→ Tail winners are NOT separable from these entry price-props OOS (AUC≈0.5). Same verdict "
                 "as the loser mine: the model prices what's here; the tail is timing/regime, and per-name "
                 "tail-edge (if any) lives in NON-price event/alt-data. Don't build a price filter.\n")
    return L, uni


# ── Family C: MFE / MAE excursion — how the tail forms during the hold ───────────
def excursion_analysis(t: pd.DataFrame, tail_cut: float) -> list:
    """From daily H/L over each trade's EntryDate→ExitDate: Max Favorable / Adverse
    Excursion vs realized. Answers: do winners dip-then-rip, is max-hold clipping the
    tail (realized << MFE), and WHEN does the gain accrue."""
    rows = []
    panels = {}
    for _, r in t.iterrows():
        sym = r["Symbol"]
        if sym not in panels:
            fp = os.path.join(V2_DIR, f"{sym}.parquet")
            if os.path.exists(fp):
                df = pd.read_parquet(fp); df["Date"] = pd.to_datetime(df["Date"])
                panels[sym] = df[["Date", "High", "Low", "Close"]].sort_values("Date")
            else:
                panels[sym] = None
        p = panels[sym]
        ep = float(r["EntryPrice"])
        if p is None or ep <= 0:
            continue
        win = p[(p["Date"] > r["EntryDate"]) & (p["Date"] <= r["ExitDate"])]
        if win.empty:
            continue
        mfe = (win["High"].max() / ep - 1) * 100
        mae = (win["Low"].min()  / ep - 1) * 100
        day_of_high = int(np.argmax(win["High"].to_numpy())) + 1   # 1-indexed hold-day of the peak
        rows.append({"ret": float(r["PnLPct"]), "mfe": mfe, "mae": mae,
                     "day_of_high": day_of_high, "held": len(win)})
    ex = pd.DataFrame(rows)
    if ex.empty:
        return ["## C. MFE/MAE excursion\n_No usable price windows._\n"]

    ex["is_tail"] = (ex["ret"] >= tail_cut).astype(int)
    ex["capture"] = np.where(ex["mfe"] > 0, ex["ret"] / ex["mfe"], np.nan)  # realized / peak
    L = ["## C. MFE / MAE excursion — how the tail forms during the hold\n",
         f"Reconstructed peak/trough over the hold window for **{len(ex)}** trades "
         f"(daily H/L, entry→exit).\n"]

    def blk(sub, lab):
        if sub.empty:
            return f"  {lab:<16} (none)"
        return (f"  {lab:<16} n={len(sub):4d}  MFE {sub['mfe'].mean():+6.2f}%  "
                f"MAE {sub['mae'].mean():+6.2f}%  realized {sub['ret'].mean():+6.2f}%  "
                f"capture {sub['capture'].median():.2f}  peak@day {sub['day_of_high'].median():.0f}")
    L.append("**Excursion by outcome bucket** (capture = realized ÷ MFE, median; "
             "peak@day = median hold-day of the high):\n```")
    L.append(blk(ex, "ALL"))
    L.append(blk(ex[ex["ret"] > 0], "winners"))
    L.append(blk(ex[ex["ret"] <= 0], "losers"))
    L.append(blk(ex[ex["is_tail"] == 1], f"TAIL ≥{tail_cut:+.1f}%"))
    L.append("```\n")

    tail = ex[ex["is_tail"] == 1]
    if len(tail):
        left = (tail["mfe"] - tail["ret"]).mean()
        dipped = (tail["mae"] <= -tail["mae"].abs().median()).mean()  # share that dipped meaningfully first
        L.append("**Tail-winner mechanics (the buckets that carry the book):**\n")
        L.append(f"- Median capture **{tail['capture'].median():.2f}** — max-hold keeps this fraction of "
                 f"the peak; the gap is money left on the table (avg **{left:.2f} pp** of MFE unrealized). "
                 f"A trailing-efficacy exit would target closing that gap (see trailing-efficacy lever).\n")
        L.append(f"- Median peak lands on hold-day **{tail['day_of_high'].median():.0f}** of "
                 f"~{tail['held'].median():.0f} held — {'gains accrue LATE → extending max-hold may catch more tail' if tail['day_of_high'].median() >= tail['held'].median()-0.5 else 'gains accrue EARLY → a trail/target could lock them before decay'}.\n")
        L.append(f"- Median tail MAE **{tail['mae'].median():+.2f}%** — how deep the eventual moonshots dig "
                 f"BEFORE running; any early-exit/stop tighter than this would have cut the winner off.\n")
    return L


def tail_mode(args):
    t = load_deployed_book()
    print(f"Deployed book: {len(t)} trades | tail pctl={args.tail_pctl}"
          + (f" (abs {args.tail_abs}%)" if args.tail_abs is not None else ""))
    L = [f"# Book Tail Post-Mortem — quantifying the convexity\n",
         f"Source: `trade_history.parquet` (DEPLOYED max-hold book, n={len(t)}). "
         f"This is the fat-tail the strategy lives on — NOT the bracket-clipped intraday sim.\n"]
    a_block, tail_cut = tail_carry_report(t, args.tail_pctl, args.tail_abs)
    L += a_block
    print("Family A done. Mining tail-winner entry signature (Family B)...")
    b_block, _ = mine_tail(t, args.tail_pctl, args.tail_abs)
    L += b_block
    print("Family B done. Reconstructing MFE/MAE excursions (Family C)...")
    L += excursion_analysis(t, tail_cut)

    os.makedirs(os.path.dirname(TAIL_REPORT), exist_ok=True)
    with open(TAIL_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("=" * 72)
    print(f"  TAIL REPORT written: {os.path.relpath(TAIL_REPORT, script_dir)}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default=None, help="CSV: Symbol,Qty,AvgPx,Last")
    ap.add_argument("--book-date", default=BOOK_DATE)
    ap.add_argument("--asof", default=None, help="as-of date for the property snapshot (default: latest bar)")
    ap.add_argument("--no-log", action="store_true", help="don't append snapshot to the ledger")
    ap.add_argument("--top-k", type=int, default=3, help="how many separators to test on the full book")
    ap.add_argument("--mine-history", action="store_true",
                    help="Pattern-mine losers across the FULL historical book (real loser sample) and exit.")
    ap.add_argument("--tail", action="store_true",
                    help="TAIL MODE: quantify the fat-right-tail that carries the strategy "
                         "(carry/fragility + tail-winner entry mine + MFE/MAE excursion) and exit.")
    ap.add_argument("--tail-pctl", type=float, default=90,
                    help="percentile of deployed PnLPct that defines a 'tail winner' (default 90)")
    ap.add_argument("--tail-abs", type=float, default=None,
                    help="absolute PnLPct cut for a tail winner (e.g. 10); overrides --tail-pctl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)

    if args.mine_history:
        mine_history()
        return

    if args.tail:
        tail_mode(args)
        return

    asof = pd.Timestamp(args.asof) if args.asof else None
    book = load_book(args)
    print(f"Book {args.book_date}: {len(book)} held names | "
          f"winners={int((book['is_loser']==0).sum())}  losers={int(book['is_loser'].sum())}")

    # ── snapshot properties for each held name ──────────────────────────────────
    feat_rows = []
    for _, r in book.iterrows():
        bat = battery_asof(r["Symbol"], asof)
        if bat is None:
            print(f"  ! no feature data for {r['Symbol']} — skipped from discrimination")
            continue
        row = {"Symbol": r["Symbol"], "unreal_pct": r["unreal_pct"], "is_loser": r["is_loser"],
               "book_date": args.book_date}
        for c in BATTERY_COLS:
            row[c] = float(bat[c]) if pd.notna(bat[c]) else np.nan
        feat_rows.append(row)
    feat = pd.DataFrame(feat_rows)

    L = [f"# Book Post-Mortem — {args.book_date}\n",
         f"Held names: **{len(feat)}**  |  winners **{int((feat['is_loser']==0).sum())}**  "
         f"losers **{int(feat['is_loser'].sum())}**  |  as-of `{asof or 'latest bar'}`\n",
         "> ⚠ One day is anecdotal — especially with "
         f"**{int(feat['is_loser'].sum())} loser(s)**. The pooled-ledger section is the one to trust "
         "as days accumulate.\n"]

    # ── per-name snapshot table ─────────────────────────────────────────────────
    L.append("## Today's book — property snapshot\n```")
    show = feat[["Symbol", "unreal_pct"] + BATTERY_COLS].sort_values("unreal_pct")
    L.append(show.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    L.append("```\n")

    # ── Step 2 on TODAY ─────────────────────────────────────────────────────────
    disc_today = rank_discriminators(feat)
    L.append("## Max-discriminating properties — TODAY (winner vs loser)\n")
    L.append("`spearman_vs_pnl` = rank-corr of the property with unrealized P&L across the book. "
             "`loser_z` = book-standard-deviations the loser sits from the winner mean "
             "(negative → losers score LOW on it).\n```")
    L.append(disc_today.drop(columns="abs_signal").to_string(index=False,
             float_format=lambda v: f"{v:.3f}"))
    L.append("```\n")

    # ── ledger append + pooled discrimination ───────────────────────────────────
    pooled = feat.copy()
    if not args.no_log and len(feat):
        if os.path.exists(LEDGER):
            old = pd.read_parquet(LEDGER)
            old = old[old["book_date"] != args.book_date]     # idempotent re-runs of same day
            led = pd.concat([old, feat], ignore_index=True)
        else:
            led = feat.copy()
        led.to_parquet(LEDGER, index=False)
        pooled = led
        ndays = pooled["book_date"].nunique()
        print(f"Ledger: {len(pooled)} name-snapshots across {ndays} day(s) -> {os.path.relpath(LEDGER, script_dir)}")

    disc = disc_today
    ndays = pooled["book_date"].nunique()
    if ndays > 1:
        disc_pool = rank_discriminators(pooled)
        L.append(f"## Max-discriminating properties — POOLED ({ndays} days, "
                 f"{len(pooled)} name-snapshots, {int(pooled['is_loser'].sum())} losers) ★\n```")
        L.append(disc_pool.drop(columns="abs_signal").to_string(index=False,
                 float_format=lambda v: f"{v:.3f}"))
        L.append("```\n> This pooled table is the robust one. Use it, not today's, to pick a separator.\n")
        disc = disc_pool
    else:
        L.append("## Pooled discrimination\n_Only one day in the ledger so far — pooled = today. "
                 "Re-run this daily; the pooled table appears once ≥2 days are logged._\n")

    # ── Step 3: test the top separators against the full historical book ─────────
    print("Loading full trade book + intraday sim for separator backtest...")
    sim = pd.read_parquet(SIM_TRADES)
    sim["EntryDate"] = pd.to_datetime(sim["EntryDate"])
    syms = sim["Symbol"].unique()
    v2_feats = {}
    for sym in syms:
        fp = os.path.join(V2_DIR, f"{sym}.parquet")
        if os.path.exists(fp):
            df = pd.read_parquet(fp); df["Date"] = pd.to_datetime(df["Date"])
            v2_feats[sym] = compute_battery(df)

    base = book_stats(sim[sim["Status"] == "OK"]["SimPnLPctNet"].to_numpy())
    L.append("## Separator applied to the FULL historical book (intraday-sim net %)\n")
    L.append(f"Baseline (all {base['n']} sim-OK trades): mean **{base['mean']:+.3f}%**, "
             f"median {base['median']:+.3f}%, win {base['win']:.1f}%, Sharpe {base['sharpe']:.2f}, "
             f"equity-proxy ×{base['equity']:.2f}\n")
    L.append("For each top separator we drop the trades on the LOSER side (the side the losers "
             "sat on) and re-measure the kept book. **Equity is a per-trade compounding proxy — "
             "NOT slot-aware; confirm winners with a real asof backtest.**\n")
    L.append("| property | rule | kept n | kept mean % | kept win % | kept Sharpe | Δmean pp | dropped mean % |")
    L.append("|---|---|---|---|---|---|---|---|")

    for _, drow in disc.head(args.top_k).iterrows():
        prop = drow["property"]
        # winner side = high if losers score low (loser_z<0) → keep >= cut; else keep <= cut
        loser_low = (drow["loser_z"] < 0) if pd.notna(drow["loser_z"]) else (drow["spearman_vs_pnl"] > 0)
        direction = +1 if loser_low else -1
        # cut = the book boundary between losers and winners on this prop
        los = pooled[pooled["is_loser"] == 1][prop]; wins = pooled[pooled["is_loser"] == 0][prop]
        if los.notna().sum() == 0:
            continue
        cut = (los.mean() + wins.mean()) / 2 if wins.notna().sum() else los.mean()
        res = apply_to_history(prop, direction, cut, sim, v2_feats)
        s = res["all"]; keep = res["kept_mask"]
        ks = book_stats(s.loc[keep, "SimPnLPctNet"].to_numpy())
        ds = book_stats(s.loc[~keep, "SimPnLPctNet"].to_numpy())
        rule = f"keep {prop} {'≥' if direction>0 else '≤'} {cut:.2f}"
        L.append(f"| {PRETTY[prop]} | {rule} | {ks['n']} | {ks['mean']:+.3f} | {ks['win']:.1f} | "
                 f"{ks['sharpe']:.2f} | {ks['mean']-base['mean']:+.3f} | {ds['mean']:+.3f} |")

    L.append("")
    L.append("### How to read this\n"
             "- A separator earns its keep only if **kept mean > baseline** AND **dropped mean < baseline** "
             "(it's actually removing the bad trades, not just shrinking the book).\n"
             "- Δmean in the low single-digit pp with a big drop in n is usually noise / survivorship — "
             "the per-trade screen LIES vs the slot-based full sim (see project_canbuy_topk_alignment). "
             "Treat a hit here as a *hypothesis to confirm* with `5__NightlyBackTester_asof.py`, not a result.\n"
             "- The point of this file is the **ledger**: re-run it daily, let losers accumulate, and only "
             "act on a separator that stays top-ranked in the POOLED table across many days.\n")

    report = "\n".join(L)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(report)

    # ── console summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TODAY'S TOP SEPARATORS (winner vs loser)")
    print("=" * 72)
    for _, d in disc_today.head(5).iterrows():
        arrow = "losers LOW " if (pd.notna(d["loser_z"]) and d["loser_z"] < 0) else "losers HIGH"
        print(f"  {d['pretty']:<26} spearman={d['spearman_vs_pnl']:+.3f}  "
              f"loser_z={d['loser_z']:+.2f}  ({arrow})")
    print("=" * 72)
    print(f"  Report : {os.path.relpath(REPORT, script_dir)}")
    if not args.no_log:
        print(f"  Ledger : {os.path.relpath(LEDGER, script_dir)}  "
              f"({pooled['book_date'].nunique()} day(s) logged)")
    print("=" * 72)


if __name__ == "__main__":
    main()
