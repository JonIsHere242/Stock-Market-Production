"""Signal autopsy: an honest, layered measurement of the predictor's rank signal.

Read-only. Touches nothing in Data/ except to read. Never runs the backtester.

Why this exists
    The shipped report quotes a single in-book rank-IC around 0.12. That number is
    computed on names the strategy already chose to buy, so it is selection biased,
    and the UpProbability it correlates against is stamped at exit, so it is partly
    circular. The population-level number is roughly 0.02 to 0.03. Both can be true
    at once. This script makes the whole ladder visible in one place so the gap
    between them is a measurement instead of an argument.

What it computes
    0. Positive controls. A leak control (the forward return used as its own score,
       which must return an IC of 1.0), a known-good control (the raw_score column,
       which must land near the UpProbability numbers), and a null control (a
       within-day permutation of UpProbability, which must land inside the null
       band). If any of the three fails, the report is stamped INSTRUMENT INVALID at
       the top and nothing below it should be quoted.
    1. IC term structure. Per-day cross sectional Spearman of UpProbability against
       forward close-to-close return at horizons 1, 2, 3, 5 and 8 trading days, with
       a within-day permutation null band and a Newey-West t-stat on the daily IC
       series. Tells you where in time the signal lives, and whether the roughly
       2-day average hold is sitting on the part of the curve that has anything.
    2. IC through time. 21d and 63d rolling means of the daily IC at horizons 1 and
       3, over the full loaded history rather than just the scoring window, so known
       break dates can be inspected.
    2c. Lagged single-period IC curve. Per-day Spearman of the day-t score against
       the SINGLE day t+l return for l = 1 to 10, plus an exponential fit that gives
       the decay constant tau and the signal half-life in trading days. This is the
       Qian, Sorensen and Hua construction and it is the readout that answers how
       long a forecast stays worth acting on.
    2d. Forecast autocorrelation. Cross sectional Spearman between consecutive days'
       score vectors on the names scored on both days. Sets the turnover floor.
    3. Layered IC at horizon 3. The same measurement on nested populations: full
       panel, scored rows, the can_buy pool, the top of book, and the booked trades.
       A collapse in the inner layers while the outer layer holds is the decay
       signature. The layer table is the point of this script.
    4. Decile table with a noise floor. Within-day decile ranks of UpProbability,
       mean forward return per decile at horizons 1, 3 and 5, bootstrap CI by
       resampling days. Includes an explicit test of the top-decile inversion and a
       shoulder-versus-peak row.
    5. Calibration. Reliability curve at horizon 1 with a Brier score, per quarter,
       so drift is visible.

Every printed statistic carries the name of the population it was measured on and
the n of that population. The population names are defined once in POPULATIONS and
listed with their sizes in the glossary section of the report.

Run
    python auxiliary/signal_autopsy.py
    python auxiliary/signal_autopsy.py --window 500 --out analysis_output/signal_autopsy_500
    python auxiliary/signal_autopsy.py --drift        # nightly quick read, no permutations

Outputs
    CSV artifacts plus a REPORT.md under --out (default analysis_output/signal_autopsy).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)], force=True)
LOG = logging.getLogger("autopsy")

HORIZONS = (1, 2, 3, 5, 8)
LAYER_HORIZON = 3
DECILE_HORIZONS = (1, 3, 5)
LAGS = tuple(range(1, 11))

# Population names used across every table. A statistic is meaningless without the
# population it was measured on, so the name and the n travel together everywhere.
POPULATIONS = {
    "full panel": "every row of the prediction lake inside the window, including "
                  "the block pinned at the 0.30 universe-gate floor",
    "scored rows": "rows with UpProbability strictly above the 0.30 pin, which is "
                   "the model's own ordering and the reference population",
    "can_buy pool": "scored rows that also pass the replicated backtester can_buy "
                    "gate, a superset of the live pool",
    "top of book": "the highest UpProbability names inside the can_buy pool on each "
                   "day, sized by --top_k",
    "booked trades": "entries recorded in TradeHistory.parquet, whose UpProbability "
                     "is stamped at exit and is therefore contaminated",
}

# UpProbability is a per-day rank remap, not a probability. Quality rows land in
# 0.30 + 0.14 * rank, and the top band is remapped into 0.45 to 0.70. Rows that
# fail the universe gate are pinned at exactly 0.30, so the panel carries a very
# large tie block at the floor. PROB_FLOOR is that pin.
PROB_FLOOR = 0.30
FLOOR_EPS = 1e-6


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #
def _read_dir(directory, columns, workers, ticker_col="Ticker"):
    """Read every parquet in a per-ticker lake into one long frame.

    The ticker is taken from the file name when the file does not carry a Ticker
    column. Files that are empty or lack a Date column are skipped loudly.
    """
    files = sorted(f for f in os.listdir(directory) if f.endswith(".parquet"))
    if not files:
        raise RuntimeError("no parquet files under %s" % directory)

    def _load(fn):
        path = os.path.join(directory, fn)
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        if "Date" not in names or pf.metadata.num_rows == 0:
            return None
        tbl = pf.read(columns=[c for c in columns if c in names])
        if ticker_col not in tbl.schema.names:
            tbl = tbl.append_column(
                ticker_col, pa.array([fn[:-8]] * len(tbl), type=pa.string()))
        return tbl

    tables, skipped = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fn, tbl in zip(files, ex.map(_load, files)):
            if tbl is None:
                skipped += 1
            else:
                tables.append(tbl)
    if skipped:
        LOG.warning("%s: skipped %d files with no Date column or zero rows",
                    directory, skipped)
    df = pa.concat_tables(tables, promote_options="default").to_pandas()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_panel(pred_dir, workers):
    cols = ["Date", "Close", "Volume", "UpProbability", "raw_up_prob",
            "pre_cm_up_prob", "raw_score"]
    df = _read_dir(pred_dir, cols, workers)
    missing = [c for c in cols if c not in df.columns]
    if "UpProbability" in missing:
        raise RuntimeError("prediction lake has no UpProbability column")
    if missing:
        LOG.warning("prediction lake is missing columns %s", missing)
    n0 = len(df)
    df = df.dropna(subset=["UpProbability"])
    df = df.drop_duplicates(subset=["Ticker", "Date"], keep="last")
    if len(df) != n0:
        LOG.info("panel: dropped %d rows (null UpProbability or duplicate key)",
                 n0 - len(df))
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def trim_partial_trailing_days(df, frac=0.5):
    """Drop trailing dates that carry far fewer rows than a normal day.

    A mid-refresh lake can hold the newest date for only a slice of the universe.
    Ranking inside such a day misrepresents the cross-section, so trailing days
    under `frac` times the median daily row count are dropped.
    """
    counts = df.groupby("Date").size().sort_index()
    med = float(counts.median())
    dropped = []
    while len(counts) > 1 and counts.iloc[-1] < frac * med:
        bad = counts.index[-1]
        dropped.append((pd.Timestamp(bad).date(), int(counts.iloc[-1])))
        df = df[df["Date"] != bad]
        counts = counts.iloc[:-1]
    for d, n in dropped:
        LOG.warning("dropped partial trailing day %s (%d rows vs median %.0f)",
                    d, n, med)
    return df.reset_index(drop=True), dropped, med


def load_forward_returns(price_dir, horizons, min_date, workers,
                         price_lo, price_hi, max_abs_ret, lags=()):
    """Forward close-to-close returns from the split-back-adjusted price lake.

    Absurd prices are set to NaN rather than dropped, so the h-bar spacing of the
    shift is preserved. Returns whose magnitude exceeds max_abs_ret are set to NaN
    as split artifacts, and the count is reported rather than swallowed.

    `lags` asks for SINGLE-period returns as well: lagret_l is the return earned
    between the close of day t+l-1 and the close of day t+l, which is what the
    lagged IC curve needs. These are non-overlapping across l, unlike fwd_h.
    """
    px = _read_dir(price_dir, ["Date", "Close"], workers)
    px = px.drop_duplicates(subset=["Ticker", "Date"], keep="last")
    px, px_dropped, _ = trim_partial_trailing_days(px)
    px_last = px["Date"].max()
    px = px[px["Date"] >= min_date].sort_values(["Ticker", "Date"])
    bad_px = int(((px["Close"] < price_lo) | (px["Close"] > price_hi)
                  | ~np.isfinite(px["Close"])).sum())
    px.loc[(px["Close"] < price_lo) | (px["Close"] > price_hi), "Close"] = np.nan
    LOG.info("prices: %s rows, %d absurd closes voided (outside [%.2f, %.0f])",
             f"{len(px):,}", bad_px, price_lo, price_hi)

    g = px.groupby("Ticker")["Close"]
    capped = {}
    for h in horizons:
        r = g.shift(-h) / px["Close"] - 1.0
        n_cap = int((r.abs() > max_abs_ret).sum())
        r = r.where(r.abs() <= max_abs_ret)
        px["fwd_%d" % h] = r.astype(np.float32)
        capped[h] = n_cap
    LOG.info("forward returns voided as split artifacts (|r| > %.2f): %s",
             max_abs_ret, capped)

    lag_capped = {}
    for l in lags:
        prev = g.shift(-(l - 1)) if l > 1 else px["Close"]
        r = g.shift(-l) / prev - 1.0
        n_cap = int((r.abs() > max_abs_ret).sum())
        px["lagret_%d" % l] = r.where(r.abs() <= max_abs_ret).astype(np.float32)
        lag_capped[l] = n_cap
    if lags:
        LOG.info("single-period lag returns voided as split artifacts: %s",
                 lag_capped)

    keep = ["Ticker", "Date"] + ["fwd_%d" % h for h in horizons]
    lagkeep = ["Ticker", "Date"] + ["lagret_%d" % l for l in lags]
    lagfwd = px[lagkeep].reset_index(drop=True) if lags else None
    return (px[keep].reset_index(drop=True), lagfwd, bad_px, capped, px_last,
            px_dropped, lag_capped)


# --------------------------------------------------------------------------- #
# IC machinery                                                                 #
# --------------------------------------------------------------------------- #
def _rank(a):
    """Average-method ranks, which is what Spearman needs with ties present."""
    return pd.Series(a).rank(method="average").to_numpy(np.float64)


def daily_ic(dates, probs, rets, min_names):
    """Per-day cross sectional Spearman. Returns (day_index, ic_values, counts)."""
    frame = pd.DataFrame({"d": dates, "p": probs, "r": rets}).dropna()
    out_d, out_ic, out_n = [], [], []
    for d, sub in frame.groupby("d", sort=True):
        n = len(sub)
        if n < min_names:
            continue
        x = _rank(sub["p"].to_numpy())
        y = _rank(sub["r"].to_numpy())
        xs = x - x.mean()
        ys = y - y.mean()
        den = np.sqrt((xs * xs).sum() * (ys * ys).sum())
        if den <= 0:
            continue
        out_d.append(d)
        out_ic.append(float((xs * ys).sum() / den))
        out_n.append(n)
    return (pd.DatetimeIndex(out_d), np.asarray(out_ic, np.float64),
            np.asarray(out_n, np.int64))


def permutation_null(dates, probs, rets, min_names, n_perm, seed):
    """Within-day shuffle null for the daily IC.

    Permuting the probability values inside a day is identical to permuting their
    ranks, so the shuffle is applied to the rank vector and the correlation is a
    single dot product. Ties survive the shuffle, which matters here because most
    of the panel is pinned at the floor.

    Returns (p95 of |per-day null IC|, p95 of |mean-over-days null IC|).
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"d": dates, "p": probs, "r": rets}).dropna()
    per_day = []
    replicate_sum = np.zeros(n_perm, np.float64)
    n_days = 0
    for _, sub in frame.groupby("d", sort=True):
        n = len(sub)
        if n < min_names:
            continue
        x = _rank(sub["p"].to_numpy())
        y = _rank(sub["r"].to_numpy())
        xs = x - x.mean()
        ys = y - y.mean()
        den = np.sqrt((xs * xs).sum() * (ys * ys).sum())
        if den <= 0:
            continue
        mat = np.tile(xs, (n_perm, 1))
        mat = rng.permuted(mat, axis=1)
        ic = (mat @ ys) / den
        per_day.append(ic)
        replicate_sum += ic
        n_days += 1
    if n_days == 0:
        return np.nan, np.nan
    per_day = np.concatenate(per_day)
    replicate_mean = replicate_sum / n_days
    return (float(np.percentile(np.abs(per_day), 95)),
            float(np.percentile(np.abs(replicate_mean), 95)))


def newey_west_t(x, lag):
    """t-stat of the mean of a serially correlated series, Bartlett kernel."""
    x = np.asarray(x, np.float64)
    T = len(x)
    if T < 3:
        return np.nan
    xbar = x.mean()
    e = x - xbar
    s = float((e * e).sum() / T)
    for l in range(1, min(lag, T - 1) + 1):
        g = float((e[l:] * e[:-l]).sum() / T)
        s += 2.0 * (1.0 - l / (lag + 1.0)) * g
    if s <= 0:
        return np.nan
    return float(xbar / np.sqrt(s / T))


# --------------------------------------------------------------------------- #
# can_buy replication                                                          #
# --------------------------------------------------------------------------- #
CANBUY_NOTES = [
    "Replicates 5__NightlyBackTester.py can_buy. The dead 0.45/0.70 variant "
    "can_buy__NO__CONFIG was deleted from the 5v2 source on 2026-08-30.",
    "Inputs are the RFpredictions Close/Volume/UpProbability series, because that "
    "is the feed the backtester actually adds (data_dir default Data/RFpredictions, "
    "EnhancedPandasData). Forward returns still come from PriceDataFull.",
    "All BT_* env knobs are taken at their shipped defaults: floor 0.40, "
    "dollar volume 1e6, 52w 0.85, momentum +0.15/-0.18, volume spike 3.5, "
    "max 20d vol 0.10, percentile band p65 to p98, min RSI 20.",
    "APPROX: rule_201_monitor SSR restriction is not reconstructible read only, so "
    "that gate is skipped. It removes names, so the pool here is a superset.",
    "APPROX: the portfolio-state gates are skipped (open_positions >= max_positions, "
    "and the 30-day warmup after strategy start). Neither is a property of the signal.",
    "APPROX: the 2026-07-25 probe gates (BT_MAX_PROBSTD, BT_MIN_DISTINCT, "
    "BT_PERSIST_N, BT_MIN_PROBMOM) are inert at their defaults and are skipped.",
    "APPROX: backtrader counts bars, this counts calendar rows of the panel. Where a "
    "ticker has a gap in the lake, a rolling window here spans the same number of "
    "panel rows but fewer observations. Coverage is near complete in the window so "
    "the effect is small, but it is not byte identical.",
    "APPROX: gates that sit inside a bare `except: pass` in the original pass here "
    "when their window is too short, matching the swallow. The two gates that hard "
    "reject on short data (fewer than 6 UpProbability bars, fewer than 2 historic "
    "probs) reject here too.",
]


def build_can_buy_mask(panel):
    """Vectorized can_buy over a wide (date x ticker) view of the panel."""
    close = panel.pivot(index="Date", columns="Ticker", values="Close")
    vol = panel.pivot(index="Date", columns="Ticker", values="Volume")
    prob = panel.pivot(index="Date", columns="Ticker", values="UpProbability")

    prior = prob.shift(1)
    n_hist = prior.rolling(45, min_periods=1).count()

    p_low_suff = prior.rolling(45, min_periods=2).quantile(0.65)
    p_high_suff = prior.rolling(45, min_periods=2).quantile(0.98)
    p_low_lim = prior.rolling(45, min_periods=2).quantile(0.90)
    p_high_lim = prior.rolling(45, min_periods=2).quantile(0.99)

    sufficient = n_hist >= 30
    p_low = p_low_suff.where(sufficient, p_low_lim)
    p_high = p_high_suff.where(sufficient, p_high_lim)

    ok = pd.DataFrame(True, index=prob.index, columns=prob.columns)
    ok &= prob.notna()

    # probability bounds, cross sectional floor, and the trailing 5-bar sanity band
    ok &= (prob >= 0.20) & (prob <= 0.80)
    ok &= prob >= 0.40
    ok &= prior.rolling(5, min_periods=5).count() >= 5          # len < 6 hard reject
    r5min = prior.rolling(5, min_periods=5).min()
    r5max = prior.rolling(5, min_periods=5).max()
    ok &= ~((r5min < 0.20).fillna(False)) & ~((r5max > 0.80).fillna(False))

    # price, share volume, dollar volume
    ok &= (close >= 2.00) & (close <= 1650.00)
    ok &= vol >= 10_000
    ok &= (close * vol) >= 1_000_000

    # historic-prob availability: fewer than 5 usable is a hard reject on the
    # limited path, and fewer than 2 is a hard reject on either path
    ok &= n_hist >= 2
    ok &= ~(((~sufficient) & (n_hist < 5)).fillna(False))

    # limited-data path also rejects a >15% single-day drop in the last 10 bars
    daily = close / close.shift(1) - 1.0
    worst10 = daily.rolling(10, min_periods=1).min()
    ok &= ~(((~sufficient) & (worst10 < -0.15)).fillna(False))

    # 52-week proximity, window includes today so the ratio is bounded by 1
    max252 = close.rolling(252, min_periods=1).max()
    ok &= ~(((close / max252) > 0.85).fillna(False))

    # 5-bar momentum band
    ret5 = close / close.shift(5) - 1.0
    ok &= ~((ret5 > 0.15).fillna(False)) & ~((ret5 < -0.18).fillna(False))

    # volume spike against the 20-bar mean, window includes today
    avgvol20 = vol.rolling(20, min_periods=1).mean()
    ok &= ~(((vol / avgvol20) > 3.5).fillna(False))

    # realized volatility of 19 log returns spanning 20 bars, population std
    logret = np.log(close.where(close > 0)).diff()
    vol20 = logret.rolling(19, min_periods=2).std(ddof=0)
    ok &= ~((vol20 > 0.10).fillna(False))

    # simple-average RSI(14) over 14 deltas spanning 15 bars
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rsi = np.where(loss.to_numpy() == 0, 100.0,
                   100.0 - 100.0 / (1.0 + gain.to_numpy() / np.where(
                       loss.to_numpy() == 0, np.nan, loss.to_numpy())))
    rsi = pd.DataFrame(rsi, index=close.index, columns=close.columns)
    ok &= ~((rsi < 20.0).fillna(False))

    # the core condition: today's prob sits inside its own trailing percentile band
    ok &= (prob >= p_low).fillna(False) & (prob < p_high).fillna(False)

    long = ok.stack(future_stack=True).rename("can_buy").reset_index()
    return long


# --------------------------------------------------------------------------- #
# sections                                                                     #
# --------------------------------------------------------------------------- #
def section_term_structure(win, args, lines, label, tag, seed_off=0):
    rows = []
    for h in HORIZONS:
        col = "fwd_%d" % h
        d, ic, n = daily_ic(win["Date"].to_numpy(), win["UpProbability"].to_numpy(),
                            win[col].to_numpy(), args.min_names)
        if len(ic) == 0:
            rows.append(dict(horizon=h, n_days=0))
            continue
        p95_day, p95_mean = permutation_null(
            win["Date"].to_numpy(), win["UpProbability"].to_numpy(),
            win[col].to_numpy(), args.min_names, args.n_perm,
            args.seed + h + seed_off)
        rows.append(dict(
            population=tag,
            horizon=h, n_days=len(ic), mean_names=float(n.mean()),
            mean_ic=float(ic.mean()), std_ic=float(ic.std(ddof=1)),
            se_iid=float(ic.std(ddof=1) / np.sqrt(len(ic))),
            nw_t=newey_west_t(ic, h),
            null_p95_daily=p95_day, null_p95_mean=p95_mean,
            pct_days_positive=float((ic > 0).mean()),
            ic_per_day_ann=float(ic.mean() / h)))
    df = pd.DataFrame(rows)

    lines.append("")
    lines.append(label)
    lines.append("   window %d trading days ending %s, permutation null = %d "
                 "within-day shuffles"
                 % (len(win["Date"].unique()), win["Date"].max().date(),
                    args.n_perm))
    hdr = ("   %-4s %6s %8s %9s %9s %9s %9s %10s %8s %9s"
           % ("h", "days", "names", "meanIC", "stdIC", "NW t", "null95", "nullMean95",
              "frac>0", "IC/day"))
    lines.append(hdr)
    lines.append("   " + "-" * (len(hdr) - 3))
    for _, r in df.iterrows():
        if r.get("n_days", 0) == 0:
            lines.append("   %-4d %6s  no usable days" % (r["horizon"], "0"))
            continue
        verdict = "OVER" if abs(r["mean_ic"]) > r["null_p95_mean"] else "under"
        lines.append("   %-4d %6d %8.0f %9.4f %9.4f %9.2f %9.4f %10.4f %8.2f %9.4f  %s"
                     % (r["horizon"], r["n_days"], r["mean_names"], r["mean_ic"],
                        r["std_ic"], r["nw_t"], r["null_p95_daily"],
                        r["null_p95_mean"], r["pct_days_positive"],
                        r["ic_per_day_ann"], verdict))
    lines.append("   null95     = p95 of |IC| over every day-by-shuffle pair, the "
                 "single-day noise floor")
    lines.append("   nullMean95 = p95 of |mean IC| across shuffle replicates, the "
                 "floor the mean must clear")
    lines.append("   IC/day     = meanIC divided by the horizon, so the horizons "
                 "are comparable per unit of holding time")
    return df


def section_through_time(full, args, lines, outdir):
    pops = {"full": full,
            "scored": full[full["UpProbability"] > PROB_FLOOR + FLOOR_EPS]}
    series = {}
    for pop, frame in pops.items():
        for h in (1, 3):
            d, ic, n = daily_ic(frame["Date"].to_numpy(),
                                frame["UpProbability"].to_numpy(),
                                frame["fwd_%d" % h].to_numpy(), args.min_names)
            s = pd.Series(ic, index=d).sort_index()
            series["ic_%s_h%d" % (pop, h)] = s
            series["r21_%s_h%d" % (pop, h)] = s.rolling(21, min_periods=15).mean()
            series["r63_%s_h%d" % (pop, h)] = s.rolling(63, min_periods=40).mean()
    df = pd.DataFrame(series).sort_index()
    df.index.name = "Date"
    df.to_csv(os.path.join(outdir, "ic_rolling.csv"), float_format="%.6f")

    cols = ["r21_scored_h1", "r63_scored_h1", "r21_scored_h3", "r63_scored_h3",
            "r21_full_h3", "r63_full_h3"]
    lines.append("")
    lines.append("2. IC THROUGH TIME  (rolling mean of the daily IC over the full "
                 "loaded history %s to %s, %d days)"
                 % (df.index.min().date(), df.index.max().date(), len(df)))
    lines.append("   all four series and both populations are in ic_rolling.csv; "
                 "the scored-rows columns are the model's own ordering")
    hdr = ("   %-12s %10s %10s %10s %10s %10s %10s"
           % ("date", "sc r21 h1", "sc r63 h1", "sc r21 h3", "sc r63 h3",
              "all r21 h3", "all r63 h3"))
    lines.append(hdr)
    lines.append("   " + "-" * (len(hdr) - 3))

    def _emit(tag, ts):
        idx = df.index
        near = idx[(idx >= ts - pd.Timedelta(days=12))
                   & (idx <= ts + pd.Timedelta(days=12))]
        if len(near) == 0:
            lines.append("   %-12s  outside the loaded history" % str(ts.date()))
            return
        for t in [near[0], near[len(near) // 2], near[-1]]:
            r = df.loc[t]
            lines.append("   %-12s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f   %s"
                         % ((t.date(),) + tuple(r[c] for c in cols) + (tag,)))

    for bd in args.break_dates:
        _emit("break %s" % bd.date(), bd)
        lines.append("")
    tail = df.dropna(subset=["r63_scored_h3"]).tail(1)
    if len(tail):
        t = tail.index[0]
        r = df.loc[t]
        lines.append("   %-12s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f   latest"
                     % ((t.date(),) + tuple(r[c] for c in cols)))
    q = (df[["ic_scored_h1", "ic_scored_h3", "ic_full_h3"]]
         .groupby(df.index.to_period("Q")).mean())
    lines.append("")
    lines.append("2b. QUARTERLY MEAN DAILY IC")
    hdr = ("   %-9s %8s %14s %14s %14s"
           % ("quarter", "days", "scored h1", "scored h3", "full h3"))
    lines.append(hdr)
    lines.append("   " + "-" * (len(hdr) - 3))
    counts = df["ic_scored_h1"].groupby(df.index.to_period("Q")).count()
    for per, r in q.iterrows():
        lines.append("   %-9s %8d %14.4f %14.4f %14.4f"
                     % (str(per), int(counts.get(per, 0)), r["ic_scored_h1"],
                        r["ic_scored_h3"], r["ic_full_h3"]))
    q.to_csv(os.path.join(outdir, "ic_by_quarter.csv"), float_format="%.6f")
    return df


def _layer_stats(sub, h, args, name, note, seed_off):
    col = "fwd_%d" % h
    mn = args.min_names if name in ("full panel", "scored rows") else args.min_names_thin
    d, ic, n = daily_ic(sub["Date"].to_numpy(), sub["UpProbability"].to_numpy(),
                        sub[col].to_numpy(), mn)
    row = dict(layer=name, rows=int(len(sub)), n_days=int(len(ic)),
               mean_names=float(n.mean()) if len(n) else np.nan,
               mean_ic=float(ic.mean()) if len(ic) else np.nan,
               nw_t=newey_west_t(ic, h) if len(ic) else np.nan,
               mean_fwd_ret=float(sub[col].mean()), note=note)
    if len(ic):
        _, p95m = permutation_null(sub["Date"].to_numpy(),
                                   sub["UpProbability"].to_numpy(),
                                   sub[col].to_numpy(), mn, args.n_perm,
                                   args.seed + 100 + seed_off)
        row["null_p95_mean"] = p95m
    else:
        row["null_p95_mean"] = np.nan
    return row


def section_layers(win, args, lines, outdir):
    h = LAYER_HORIZON
    col = "fwd_%d" % h
    rows = []

    rows.append(_layer_stats(win, h, args, "full panel",
                             "every scored row, floor ties included", 0))

    scored = win[win["UpProbability"] > PROB_FLOOR + FLOOR_EPS]
    rows.append(_layer_stats(scored, h, args, "scored rows",
                             "UpProbability above the 0.30 pin", 1))

    pool = win[win["can_buy"].fillna(False)]
    rows.append(_layer_stats(pool, h, args, "can_buy pool",
                             "replicated gate, see approximations", 2))

    if len(pool):
        top = (pool.sort_values(["Date", "UpProbability", "Ticker"],
                                ascending=[True, False, True])
                   .groupby("Date").head(args.top_k))
    else:
        top = pool
    rows.append(_layer_stats(top, h, args, "top %d of book" % args.top_k,
                             "highest UpProbability inside the pool", 3))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "layer_ic.csv"), index=False,
              float_format="%.6f")

    lines.append("")
    lines.append("3. LAYERED IC AT HORIZON %d  (the decay early-warning table)" % h)
    hdr = ("   %-18s %10s %6s %8s %9s %9s %11s %10s"
           % ("layer", "rows", "days", "names", "meanIC", "NW t", "nullMean95",
              "meanRet"))
    lines.append(hdr)
    lines.append("   " + "-" * (len(hdr) - 3))
    for _, r in df.iterrows():
        if not np.isfinite(r["mean_ic"]):
            lines.append("   %-18s %10d %6d   too few names per day to rank"
                         % (r["layer"], r["rows"], r["n_days"]))
            continue
        verdict = "OVER" if abs(r["mean_ic"]) > r["null_p95_mean"] else "under"
        lines.append("   %-18s %10d %6d %8.1f %9.4f %9.2f %11.4f %9.4f%%  %s"
                     % (r["layer"], r["rows"], r["n_days"], r["mean_names"],
                        r["mean_ic"], r["nw_t"], r["null_p95_mean"],
                        100 * r["mean_fwd_ret"], verdict))
    lines.append("   OVER / under compares |meanIC| against its own within-day "
                 "permutation band, so the thin layers are judged at their own "
                 "power.")

    floor = win[win["UpProbability"] <= PROB_FLOOR + FLOOR_EPS]
    scored = win[win["UpProbability"] > PROB_FLOOR + FLOOR_EPS]
    lines.append("")
    lines.append("3b. WHAT THE FULL-PANEL ROW IS ACTUALLY MEASURING")
    lines.append("   %.2f%% of window rows are pinned at exactly %.2f because they "
                 "failed the universe gate, not because"
                 % (100 * len(floor) / max(len(win), 1), PROB_FLOOR))
    lines.append("   the model ranked them last. Inside a day that whole block "
                 "shares one averaged rank, and it sits BELOW every scored")
    lines.append("   name. So the full-panel IC is dominated by 'gate rejects vs "
                 "gate passers', which is a property of the universe")
    lines.append("   filter rather than of the model's ordering. Read the scored-"
                 "rows layer as the model's population IC.")
    lines.append("   mean %dd forward return: floor block %.4f%% (n=%s) vs scored "
                 "rows %.4f%% (n=%s), spread %+.4f%%"
                 % (h, 100 * float(floor["fwd_%d" % h].mean()),
                    f"{len(floor):,}", 100 * float(scored["fwd_%d" % h].mean()),
                    f"{len(scored):,}",
                    100 * (float(scored["fwd_%d" % h].mean())
                           - float(floor["fwd_%d" % h].mean()))))
    return df


def section_booked(win, args, lines, outdir):
    path = args.trade_history
    lines.append("")
    lines.append("3d. BOOKED TRADES LAYER")
    if not os.path.exists(path):
        lines.append("   %s not present, layer skipped" % path)
        return None
    th = pq.read_table(path).to_pandas()          # read only, never written
    th["EntryDate"] = pd.to_datetime(th["EntryDate"])
    lo, hi = win["Date"].min(), win["Date"].max()
    sub = th[(th["EntryDate"] >= lo) & (th["EntryDate"] <= hi)].copy()
    lines.append("   CONTAMINATED: TradeHistory.UpProbability is stamped at EXIT, "
                 "not at entry.")
    lines.append("   Any correlation below is therefore partly circular and is "
                 "reported for contrast only.")
    lines.append("   file holds %d trades, %d of them entered inside the window "
                 "(%s to %s)" % (len(th), len(sub), lo.date(), hi.date()))
    if len(sub) < 10:
        lines.append("   too few in-window trades to measure")
        return None
    key = win[["Ticker", "Date", "fwd_1", "fwd_3", "fwd_5"]]
    sub = sub.merge(key, left_on=["Symbol", "EntryDate"],
                    right_on=["Ticker", "Date"], how="left")
    matched = sub["fwd_3"].notna().sum()
    lines.append("   %d of %d matched a panel row with a 3-day forward return"
                 % (matched, len(sub)))
    good = sub.dropna(subset=["fwd_3", "UpProbability"])
    if len(good) >= 10:
        pooled = float(pd.Series(_rank(good["UpProbability"].to_numpy())).corr(
            pd.Series(_rank(good["fwd_3"].to_numpy()))))
        d, ic, n = daily_ic(good["EntryDate"].to_numpy(),
                            good["UpProbability"].to_numpy(),
                            good["fwd_3"].to_numpy(), args.min_names_thin)
        lines.append("   pooled Spearman (exit-stamped prob vs fwd 3d) = %.4f on "
                     "%d trades" % (pooled, len(good)))
        lines.append("   per-entry-day IC on days with at least %d trades: mean "
                     "%.4f over %d days"
                     % (args.min_names_thin,
                        float(ic.mean()) if len(ic) else np.nan, len(ic)))
        lines.append("   mean realized 3d forward return on booked entries = %.4f%%"
                     % (100 * good["fwd_3"].mean()))
        lines.append("   realized PnLPct on those trades: mean %.4f%%, median %.4f%%"
                     % (sub["PnLPct"].mean(), sub["PnLPct"].median()))
        good.to_csv(os.path.join(outdir, "booked_trades_window.csv"), index=False)
        return dict(pooled=pooled, n=len(good),
                    mean_ic=float(ic.mean()) if len(ic) else np.nan,
                    mean_ret=float(good["fwd_3"].mean()))
    lines.append("   too few matched trades to measure")
    return None


def section_deciles(win, args, lines, outdir):
    rng = np.random.default_rng(args.seed)
    w = win.copy()
    w["pct"] = w.groupby("Date")["UpProbability"].rank(pct=True, method="average")
    w["decile"] = np.minimum((w["pct"] * 10).astype(int) + 1, 10)

    # day-equal-weighted decile means, winsorized inside each day so one split
    # artifact cannot own a bucket
    for h in DECILE_HORIZONS:
        c = "fwd_%d" % h
        lo = w.groupby("Date")[c].transform(lambda s: s.quantile(0.005))
        hi = w.groupby("Date")[c].transform(lambda s: s.quantile(0.995))
        w["wz_%d" % h] = w[c].clip(lo, hi)

    per_day = {}
    for h in DECILE_HORIZONS:
        per_day[h] = (w.pivot_table(index="Date", columns="decile",
                                    values="wz_%d" % h, aggfunc="mean")
                       .sort_index())

    counts = w.groupby("decile").size()
    days = per_day[DECILE_HORIZONS[0]].index
    n_days = len(days)
    boot_idx = rng.integers(0, n_days, size=(args.n_boot, n_days))

    rows = []
    for dec in range(1, 11):
        row = dict(decile=dec, rows=int(counts.get(dec, 0)),
                   prob_lo=float(w.loc[w["decile"] == dec, "UpProbability"].min())
                   if counts.get(dec, 0) else np.nan,
                   prob_hi=float(w.loc[w["decile"] == dec, "UpProbability"].max())
                   if counts.get(dec, 0) else np.nan)
        for h in DECILE_HORIZONS:
            m = per_day[h].get(dec)
            if m is None or m.notna().sum() < 5:
                row["mean_%d" % h] = np.nan
                row["lo_%d" % h] = np.nan
                row["hi_%d" % h] = np.nan
                continue
            v = m.to_numpy(np.float64)
            row["mean_%d" % h] = float(np.nanmean(v))
            draws = np.nanmean(v[boot_idx], axis=1)
            row["lo_%d" % h] = float(np.nanpercentile(draws, 2.5))
            row["hi_%d" % h] = float(np.nanpercentile(draws, 97.5))
        rows.append(row)
    dec_df = pd.DataFrame(rows)
    dec_df.to_csv(os.path.join(outdir, "decile_table.csv"), index=False,
                  float_format="%.6f")

    lines.append("")
    lines.append("4. DECILE TABLE  (within-day decile of UpProbability, day-equal-"
                 "weighted mean forward return, in percent)")
    lines.append("   returns winsorized inside each day at the 0.5 and 99.5 "
                 "percentile, CI = %d bootstrap draws over days (seed %d)"
                 % (args.n_boot, args.seed))
    lines.append("   DEGENERATE BY CONSTRUCTION: the floor block shares one "
                 "averaged within-day rank, so it lands whole in a single decile")
    lines.append("   and empties the ones below it. Which decile that is moves with "
                 "the day's floor fraction, which is why the floor")
    lines.append("   value 0.300 shows up in more than one bucket. Only the top "
                 "deciles carry a genuine within-day ordering.")
    hdr = ("   %-4s %9s %13s %22s %22s %22s"
           % ("dec", "rows", "prob range", "h=1 mean [95% CI]",
              "h=3 mean [95% CI]", "h=5 mean [95% CI]"))
    lines.append(hdr)
    lines.append("   " + "-" * (len(hdr) - 3))
    for _, r in dec_df.iterrows():
        if r["rows"] == 0:
            lines.append("   %-4d %9d   empty (tie block pushes the mass upward)"
                         % (r["decile"], 0))
            continue
        cells = []
        for h in DECILE_HORIZONS:
            if not np.isfinite(r["mean_%d" % h]):
                cells.append("%22s" % "n/a")
            else:
                cells.append("%8.3f [%6.3f,%6.3f]"
                             % (100 * r["mean_%d" % h], 100 * r["lo_%d" % h],
                                100 * r["hi_%d" % h]))
        lines.append("   %-4d %9d  %5.3f-%5.3f %s %s %s"
                     % (r["decile"], r["rows"], r["prob_lo"], r["prob_hi"],
                        cells[0], cells[1], cells[2]))

    # --- explicit inversion test -----------------------------------------
    lines.append("")
    lines.append("4b. TOP-DECILE INVERSION TEST  (decile 10 minus the mean of "
                 "deciles 8 and 9)")
    inv_rows = []
    for h in DECILE_HORIZONS:
        p = per_day[h]
        if not all(c in p.columns for c in (8, 9, 10)):
            lines.append("   h=%d: deciles 8, 9 or 10 missing, cannot test" % h)
            continue
        diff = p[10] - 0.5 * (p[8] + p[9])
        v = diff.to_numpy(np.float64)
        m = float(np.nanmean(v))
        draws = np.nanmean(v[boot_idx], axis=1)
        lo = float(np.nanpercentile(draws, 2.5))
        hi = float(np.nanpercentile(draws, 97.5))
        if hi < 0:
            verdict = "INVERTED, CI excludes zero"
        elif lo > 0:
            verdict = "NOT inverted, decile 10 leads and the CI excludes zero"
        else:
            verdict = "INDISTINGUISHABLE from zero, CI straddles it"
        lines.append("   h=%d  diff %+7.4f%%  95%% CI [%+7.4f%%, %+7.4f%%]   %s"
                     % (h, 100 * m, 100 * lo, 100 * hi, verdict))
        inv_rows.append(dict(horizon=h, diff=m, lo=lo, hi=hi, verdict=verdict))
    pd.DataFrame(inv_rows).to_csv(os.path.join(outdir, "inversion_test.csv"),
                                  index=False, float_format="%.6f")

    # --- shoulder vs peak -------------------------------------------------
    lines.append("")
    lines.append("4c. SHOULDER VERSUS PEAK")
    pmax = float(w["UpProbability"].max())
    lines.append("   observed UpProbability range in the window: %.4f to %.4f"
                 % (float(w["UpProbability"].min()), pmax))
    band_rows = []
    abs_bands = [("absolute 0.90 to 0.95", 0.90, 0.95),
                 ("absolute above 0.95", 0.95, np.inf)]
    pct_bands = [("pctile 0.90 to 0.95", 0.90, 0.95),
                 ("pctile above 0.95", 0.95, np.inf)]
    for label, a, b in abs_bands:
        sel = w[(w["UpProbability"] >= a) & (w["UpProbability"] < b)]
        band_rows.append(("abs", label, sel))
    for label, a, b in pct_bands:
        sel = w[(w["pct"] >= a) & (w["pct"] < b)]
        band_rows.append(("pct", label, sel))

    hdr = ("   %-24s %9s %22s %22s %22s"
           % ("band", "rows", "h=1 mean [95% CI]", "h=3 mean [95% CI]",
              "h=5 mean [95% CI]"))
    lines.append(hdr)
    lines.append("   " + "-" * (len(hdr) - 3))
    sh_rows = []
    for kind, label, sel in band_rows:
        if len(sel) == 0:
            lines.append("   %-24s %9d   EMPTY, no row in the window falls here"
                         % (label, 0))
            sh_rows.append(dict(band=label, rows=0))
            continue
        cells, rec = [], dict(band=label, rows=int(len(sel)))
        for h in DECILE_HORIZONS:
            g = sel.groupby("Date")["wz_%d" % h].mean().reindex(days)
            v = g.to_numpy(np.float64)
            if np.isfinite(v).sum() < 5:
                cells.append("%22s" % "n/a")
                continue
            m = float(np.nanmean(v))
            draws = np.nanmean(v[boot_idx], axis=1)
            lo = float(np.nanpercentile(draws, 2.5))
            hi = float(np.nanpercentile(draws, 97.5))
            rec["mean_%d" % h] = m
            rec["lo_%d" % h] = lo
            rec["hi_%d" % h] = hi
            cells.append("%8.3f [%6.3f,%6.3f]" % (100 * m, 100 * lo, 100 * hi))
        lines.append("   %-24s %9d %s %s %s"
                     % (label, len(sel), cells[0], cells[1], cells[2]))
        sh_rows.append(rec)
    if pmax < 0.90:
        lines.append("   NOTE: UpProbability is a per-day rank remap capped at "
                     "0.70, so the absolute 0.90 to 0.95 and above 0.95 bands")
        lines.append("   cannot exist by construction. The percentile rows are the "
                     "only readable version of the shoulder-versus-peak question.")
    pd.DataFrame(sh_rows).to_csv(os.path.join(outdir, "shoulder_vs_peak.csv"),
                                 index=False, float_format="%.6f")

    w[["Date", "Ticker", "UpProbability", "pct", "decile", "fwd_1", "fwd_3",
       "fwd_5", "can_buy"]].to_parquet(
        os.path.join(outdir, "window_rows.parquet"), index=False)
    return dec_df, inv_rows, sh_rows


def section_calibration(win, args, lines, outdir):
    w = win.dropna(subset=["fwd_1"]).copy()
    w["y"] = (w["fwd_1"] > 0).astype(np.float64)
    edges = np.linspace(PROB_FLOOR, 0.70, args.n_cal_bins + 1)
    w["bin"] = np.clip(np.digitize(w["UpProbability"], edges) - 1,
                       0, args.n_cal_bins - 1)

    rel = (w.groupby("bin")
             .agg(rows=("y", "size"), mean_prob=("UpProbability", "mean"),
                  realized=("y", "mean"))
             .reindex(range(args.n_cal_bins)))
    rel["bin_lo"] = edges[:-1]
    rel["bin_hi"] = edges[1:]
    rel = rel.reset_index()
    rel.to_csv(os.path.join(outdir, "calibration_curve.csv"), index=False,
               float_format="%.6f")

    brier_all = float(((w["UpProbability"] - w["y"]) ** 2).mean())
    base = float(w["y"].mean())
    brier_base = float(((base - w["y"]) ** 2).mean())

    lines.append("")
    lines.append("5. CALIBRATION AT HORIZON 1  (%d equal-width bins over the "
                 "0.30 to 0.70 remap band)" % args.n_cal_bins)
    lines.append("   UpProbability is a per-day RANK REMAP, not a probability. It "
                 "is expected to be badly calibrated in level;")
    lines.append("   the readable content is whether realized P(up) rises "
                 "monotonically across the bins.")
    lines.append("   overall Brier %.5f, base-rate-only Brier %.5f, base rate "
                 "P(up) = %.4f, rows %s"
                 % (brier_all, brier_base, base, f"{len(w):,}"))
    hdr = "   %-14s %10s %10s %10s %9s" % ("bin", "rows", "meanProb",
                                           "realized", "lift")
    lines.append(hdr)
    lines.append("   " + "-" * (len(hdr) - 3))
    for _, r in rel.iterrows():
        if not np.isfinite(r["rows"]) or r["rows"] == 0:
            continue
        lines.append("   %-14s %10d %10.4f %10.4f %+9.4f"
                     % ("%.3f-%.3f" % (r["bin_lo"], r["bin_hi"]), int(r["rows"]),
                        r["mean_prob"], r["realized"], r["realized"] - base))

    w["quarter"] = w["Date"].dt.to_period("Q").astype(str)
    qrows = []
    for qname, s in w.groupby("quarter", sort=True):
        base_q = float(s["y"].mean())
        cut = s["UpProbability"].quantile(0.9)
        qrows.append(dict(
            quarter=qname,
            rows=len(s),
            base_rate=base_q,
            brier=float(((s["UpProbability"] - s["y"]) ** 2).mean()),
            brier_base=float(((base_q - s["y"]) ** 2).mean()),
            mean_prob=float(s["UpProbability"].mean()),
            top_decile_hit=float(s.loc[s["UpProbability"] >= cut, "y"].mean()),
        ))
    q = pd.DataFrame(qrows)
    q["brier_skill"] = 1.0 - q["brier"] / q["brier_base"]
    q["top_decile_lift"] = q["top_decile_hit"] - q["base_rate"]
    q.to_csv(os.path.join(outdir, "calibration_by_quarter.csv"), index=False,
             float_format="%.6f")

    lines.append("")
    lines.append("5b. CALIBRATION DRIFT BY QUARTER")
    hdr = ("   %-9s %10s %10s %10s %10s %11s %10s %11s"
           % ("quarter", "rows", "meanProb", "baseRate", "Brier", "BrierBase",
              "skill", "top10 lift"))
    lines.append(hdr)
    lines.append("   " + "-" * (len(hdr) - 3))
    for _, r in q.iterrows():
        lines.append("   %-9s %10d %10.4f %10.4f %10.5f %11.5f %10.4f %+11.4f"
                     % (r["quarter"], int(r["rows"]), r["mean_prob"],
                        r["base_rate"], r["brier"], r["brier_base"],
                        r["brier_skill"], r["top_decile_lift"]))
    lines.append("   skill = 1 - Brier / BrierBase. Negative means the stamped "
                 "level is worse than just quoting the base rate.")
    lines.append("   top10 lift = realized P(up) among the day-ranked top decile "
                 "minus the quarter base rate. That is the part that can decay.")
    return rel, q


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Layered read-only measurement of predictor signal quality.")
    p.add_argument("--pred_dir", default=os.path.join("Data", "RFpredictions"))
    p.add_argument("--price_dir", default=os.path.join("Data", "PriceDataFull"))
    p.add_argument("--trade_history", default=os.path.join("Data",
                                                           "TradeHistory.parquet"))
    p.add_argument("--out", default=os.path.join("analysis_output",
                                                 "signal_autopsy"))
    p.add_argument("--window", type=int, default=252,
                   help="Trading days of the panel to score. Default 252.")
    p.add_argument("--prob_col", default="UpProbability",
                   help="Score column to autopsy. raw_up_prob is the "
                        "pre-neutralization value.")
    p.add_argument("--n_perm", type=int, default=200)
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--n_cal_bins", type=int, default=20)
    p.add_argument("--top_k", type=int, default=10,
                   help="Names per day in the top-of-book layer.")
    p.add_argument("--min_names", type=int, default=50,
                   help="Minimum names in a day for the wide layers to be ranked.")
    p.add_argument("--min_names_thin", type=int, default=5,
                   help="Minimum names in a day for the thin layers.")
    p.add_argument("--max_abs_ret", type=float, default=1.0,
                   help="Forward returns above this magnitude are voided as split "
                        "artifacts.")
    p.add_argument("--min_fwd_cov", type=float, default=0.5,
                   help="A panel day counts as scoreable when at least this "
                        "fraction of its rows carry a forward return at the "
                        "longest horizon.")
    p.add_argument("--price_lo", type=float, default=0.01)
    p.add_argument("--price_hi", type=float, default=100000.0)
    p.add_argument("--break_dates", default="2025-07-01,2025-09-01")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_can_buy", action="store_true",
                   help="Skip the can_buy replication (the slowest step).")
    return p


def main():
    args = build_parser().parse_args()
    args.break_dates = [pd.Timestamp(s.strip())
                        for s in args.break_dates.split(",") if s.strip()]
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    caps = []

    LOG.info("loading prediction panel from %s", args.pred_dir)
    panel = load_panel(args.pred_dir, args.workers)
    if args.prob_col != "UpProbability":
        if args.prob_col not in panel.columns:
            raise RuntimeError("prob_col %s not in the prediction lake" % args.prob_col)
        panel["UpProbability"] = panel[args.prob_col]
    panel, dropped_days, med_rows = trim_partial_trailing_days(panel)
    all_days = np.sort(panel["Date"].unique())
    LOG.info("panel: %s rows, %d tickers, %d days (%s to %s)",
             f"{len(panel):,}", panel["Ticker"].nunique(), len(all_days),
             pd.Timestamp(all_days[0]).date(), pd.Timestamp(all_days[-1]).date())

    LOG.info("loading prices from %s", args.price_dir)
    min_date = pd.Timestamp(all_days[0]) - pd.Timedelta(days=30)
    fwd, n_bad_px, capped_ret, px_last, px_dropped = load_forward_returns(
        args.price_dir, HORIZONS, min_date, args.workers,
        args.price_lo, args.price_hi, args.max_abs_ret)

    panel = panel.merge(fwd, on=["Ticker", "Date"], how="left")
    match_rate = float(panel["fwd_1"].notna().mean())
    LOG.info("forward-return join: %.3f of panel rows carry a 1-day forward return",
             match_rate)

    if args.skip_can_buy:
        panel["can_buy"] = False
        caps.append("can_buy replication SKIPPED via --skip_can_buy; the pool and "
                    "top-of-book layers are empty.")
    else:
        LOG.info("replicating can_buy over the panel")
        t1 = time.time()
        cb = build_can_buy_mask(panel[["Date", "Ticker", "Close", "Volume",
                                       "UpProbability"]])
        panel = panel.merge(cb, on=["Date", "Ticker"], how="left")
        LOG.info("can_buy: %s of %s panel rows pass (%.3f%%), %.1fs",
                 f"{int(panel['can_buy'].sum()):,}", f"{len(panel):,}",
                 100 * float(panel["can_buy"].mean()), time.time() - t1)

    # ---- window ----
    # The window is anchored on the last day that is actually MEASURABLE at the
    # longest horizon, not on the last day the panel carries. The price lake can
    # lag the prediction lake, and a window whose newest days have no forward
    # return would silently shrink the sample instead of reporting the gap.
    hmax = max(HORIZONS)
    cov = panel.groupby("Date")["fwd_%d" % hmax].apply(lambda s: float(s.notna().mean()))
    usable = cov.index[cov >= args.min_fwd_cov]
    if len(usable) == 0:
        raise RuntimeError("no panel day has at least %.2f forward-return coverage "
                           "at horizon %d; the price lake and the prediction lake "
                           "do not overlap usefully." % (args.min_fwd_cov, hmax))
    eval_last = pd.Timestamp(usable.max())
    panel_last = pd.Timestamp(all_days[-1])
    n_lost = int((all_days > np.datetime64(eval_last)).sum())
    if n_lost:
        caps.append("the prediction lake runs to %s but the price lake ends %s, so "
                    "the newest %d panel trading days cannot be scored at horizon "
                    "%d and are EXCLUDED. The window is anchored on %s."
                    % (panel_last.date(), pd.Timestamp(px_last).date(), n_lost,
                       hmax, eval_last.date()))
        LOG.warning(caps[-1])

    elig = all_days[all_days <= np.datetime64(eval_last)]
    win_days = elig[-args.window:]
    if len(elig) < args.window:
        caps.append("requested --window %d but only %d scoreable trading days "
                    "exist; all of them were used." % (args.window, len(elig)))
    win = panel[(panel["Date"] >= win_days[0])
                & (panel["Date"] <= win_days[-1])].copy()

    lines = []
    lines.append("=" * 100)
    lines.append("SIGNAL AUTOPSY")
    lines.append("=" * 100)
    lines.append("generated        %s" % time.strftime("%Y-%m-%d %H:%M"))
    lines.append("score column     %s" % args.prob_col)
    lines.append("prediction lake  %s" % os.path.abspath(args.pred_dir))
    lines.append("price lake       %s" % os.path.abspath(args.price_dir))
    lines.append("panel            %s rows, %d tickers, %d days, %s to %s"
                 % (f"{len(panel):,}", panel["Ticker"].nunique(), len(all_days),
                    pd.Timestamp(all_days[0]).date(),
                    pd.Timestamp(all_days[-1]).date()))
    lines.append("window           %d trading days, %s to %s, %s rows"
                 % (len(win_days), pd.Timestamp(win_days[0]).date(),
                    pd.Timestamp(win_days[-1]).date(), f"{len(win):,}"))
    lines.append("price lake ends  %s (prediction lake ends %s, %d newest panel "
                 "days are unscoreable and excluded)"
                 % (pd.Timestamp(px_last).date(), panel_last.date(), n_lost))
    lines.append("median day size  %.0f rows; partial trailing panel days dropped: "
                 "%s; partial trailing price days dropped: %s"
                 % (med_rows, dropped_days if dropped_days else "none",
                    px_dropped if px_dropped else "none"))
    floor_frac = float((win["UpProbability"] <= PROB_FLOOR + FLOOR_EPS).mean())
    lines.append("floor ties       %.2f%% of window rows sit at exactly %.2f "
                 "(universe-gate reject pin)" % (100 * floor_frac, PROB_FLOOR))
    lines.append("prob range       %.4f to %.4f"
                 % (float(win["UpProbability"].min()),
                    float(win["UpProbability"].max())))
    lines.append("fwd join         %.3f of panel rows carry a 1-day forward return"
                 % match_rate)
    lines.append("price hygiene    %d absurd closes voided, forward returns voided "
                 "for |r| > %.2f: %s" % (n_bad_px, args.max_abs_ret, capped_ret))

    ts_full = section_term_structure(
        win, args, lines,
        "1. IC TERM STRUCTURE, FULL PANEL  (per-day cross sectional Spearman of "
        "UpProbability vs forward return)", "full_panel", seed_off=0)
    ts_scored = section_term_structure(
        win[win["UpProbability"] > PROB_FLOOR + FLOOR_EPS], args, lines,
        "1b. IC TERM STRUCTURE, SCORED ROWS ONLY  (UpProbability above the %.2f "
        "pin; this is the model's own ordering)" % PROB_FLOOR,
        "scored_rows", seed_off=1000)
    ts = pd.concat([ts_full, ts_scored], ignore_index=True)
    ts.to_csv(os.path.join(outdir, "term_structure.csv"), index=False,
              float_format="%.6f")
    roll = section_through_time(panel, args, lines, outdir)
    layers = section_layers(win, args, lines, outdir)
    booked = section_booked(win, args, lines, outdir)
    dec_df, inv_rows, sh_rows = section_deciles(win, args, lines, outdir)
    rel, quart = section_calibration(win, args, lines, outdir)

    lines.append("")
    lines.append("6. CAN_BUY REPLICATION, EXACTLY WHAT WAS APPROXIMATED")
    for i, n in enumerate(CANBUY_NOTES, 1):
        lines.append("   %d. %s" % (i, n))

    lines.append("")
    lines.append("7. CAPS AND LIMITS")
    caps.append("permutation null uses %d shuffles per day; the p95 band is itself "
                "estimated from %d replicates." % (args.n_perm, args.n_perm))
    caps.append("bootstrap uses %d day-resamples, seeded at %d, so reruns are "
                "identical." % (args.n_boot, args.seed))
    caps.append("decile mean returns are winsorized inside each day at the 0.5 and "
                "99.5 percentile; rank-IC is untouched by this since it is "
                "rank based.")
    caps.append("forward returns come from PriceDataFull, which is split-back-"
                "adjusted end to end. The Close carried inside RFpredictions is a "
                "frozen snapshot and disagrees with it more the further back you "
                "go, so it is NOT used for returns.")
    caps.append("days with fewer than %d ranked names are excluded from the wide "
                "layers and fewer than %d from the thin layers."
                % (args.min_names, args.min_names_thin))
    for i, c in enumerate(caps, 1):
        lines.append("   %d. %s" % (i, c))

    lines.append("")
    lines.append("runtime %.1fs, artifacts under %s" % (time.time() - t0, outdir))
    lines.append("=" * 100)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(outdir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("# Signal autopsy\n\nRead-only measurement produced by "
                "`signal_autopsy.py`. Nothing under `Data/` was written.\n\n```\n")
        f.write(text)
        f.write("\n```\n")
    LOG.info("report -> %s", os.path.join(outdir, "REPORT.md"))


if __name__ == "__main__":
    main()
