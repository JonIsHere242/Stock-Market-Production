"""OVERNIGHT PART 1: seed-ensemble validation (strategy-level, rigorous).

Motivated by the session finding that the model is ~+-15pp/month seed-sensitive:
a single-seed production model is one draw from a high-variance distribution.
Averaging predict_proba (raw_score) across N seeds reduces that variance BY
CONSTRUCTION. This tests whether it pays off at the strategy level: does an
8-seed ensemble beat the SINGLE-SEED DISTRIBUTION on Sharpe / drawdown / monthly
volatility?

Design: 8 seeds at one cutoff (2025-12-31), production params fixed. Build 2/4/8-
seed ensembles (mean raw_score -> re-rank per day -> UpProbability). Backtest all
8 singles + 3 ensembles (3-wide parallel, COPY backtester, no --force). Aggregate
the single-seed DISTRIBUTION vs the ensembles.

Autonomous: no prompts, resumable (skip-if-exists), retries, incremental. Touches
nothing live. Run: python analysis_output/overnight_ensemble.py
"""
import glob, os, re, subprocess, sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = "Data/_refit_reg"           # regularized single-seed models (separate dir)
ENS = "Data/_ensemble_reg"
os.makedirs(ENS, exist_ok=True)
BACKTESTER = "5__NightlyBackTester_oosvalidate.py"
CUTOFF = "2025-12-31"
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
ENSEMBLE_SIZES = [2, 4, 8]
TOP_FRAC = 0.01
PARALLEL = 3
# REGULARIZED config (depth-5, proper reg) — the sweep's generalizing region.
# With ensembling removing the noise, tests whether regularized params beat the
# production Optuna (depth-8) config — the comparison single-path could not resolve.
PROD = ["--n_estimators", "300", "--max_depth", "5", "--learning_rate", "0.05",
        "--min_child_weight", "5", "--subsample", "0.8",
        "--colsample_bytree", "0.6", "--reg_alpha", "0.5", "--reg_lambda", "2.0"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def seed_dir(s):
    return f"{ROOT}/{CUTOFF.replace('-','')}_s{s}"


def map_pct_rank_to_upprob(pct_rank, top_frac):
    threshold = 1.0 - top_frac
    above = pct_rank >= threshold
    up = np.empty_like(pct_rank, dtype=np.float64)
    up[above] = 0.45 + 0.25 * (pct_rank[above] - threshold) / max(top_frac, 1e-9)
    up[~above] = 0.30 + 0.14 * pct_rank[~above] / max(threshold, 1e-9)
    return np.clip(up, 0.30, 0.70)


def load_seed_scores(s):
    """Compact [Date,Ticker,raw_score(,UpPrediction)] for a seed, cached."""
    cache = f"{ENS}/scores_s{s}.parquet"
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    d = f"{seed_dir(s)}/rf"
    files = sorted(glob.glob(f"{d}/*.parquet"))

    def _one(f):
        t = pq.read_table(f, columns=["Date", "raw_score", "UpPrediction"]).to_pandas()
        t["Ticker"] = os.path.splitext(os.path.basename(f))[0]
        return t
    parts = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(_one, files):
            parts.append(r)
    df = pd.concat(parts, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.to_parquet(cache, index=False)
    return df


def build_ensemble(seeds, out_dir):
    """Average raw_score across `seeds`, re-rank per day, write rf parquets."""
    os.makedirs(f"{out_dir}/rf", exist_ok=True)
    if len(glob.glob(f"{out_dir}/rf/*.parquet")) > 100:
        log(f"ensemble {out_dir}: exists, skip build")
        return
    # average raw scores across seeds on (Date,Ticker); universe from seed[0]
    base = load_seed_scores(seeds[0]).rename(columns={"raw_score": "s0"})
    ens = base[["Date", "Ticker", "UpPrediction", "s0"]].copy()
    for i, s in enumerate(seeds[1:], 1):
        sc = load_seed_scores(s)[["Date", "Ticker", "raw_score"]].rename(columns={"raw_score": f"s{i}"})
        ens = ens.merge(sc, on=["Date", "Ticker"], how="inner")
    score_cols = [c for c in ens.columns if re.fullmatch(r"s\d+", c)]
    ens["ens_raw"] = ens[score_cols].mean(axis=1).astype(np.float32)
    ens["is_univ"] = ens["UpPrediction"] != -1
    ens["rk"] = 0.0
    m = ens["is_univ"].values
    ens.loc[m, "rk"] = ens.loc[m].groupby("Date")["ens_raw"].rank(pct=True, method="average")
    up = map_pct_rank_to_upprob(ens["rk"].values, TOP_FRAC)
    up[~m] = 0.30
    ens["UpProbability_new"] = np.clip(up, 0.01, 0.99).astype(np.float32)
    ens["UpPrediction_new"] = np.where(m & (ens["rk"].values >= 1 - TOP_FRAC), 1,
                                       np.where(m, 0, -1)).astype(np.int8)
    ens["raw_new"] = ens["ens_raw"]
    # write per-ticker, using seed[0] files as template (OHLCV/VIX/etc.)
    tdir = f"{seed_dir(seeds[0])}/rf"
    n = 0
    for tk, g in ens.groupby("Ticker", sort=False):
        tf = f"{tdir}/{tk}.parquet"
        if not os.path.exists(tf):
            continue
        tmpl = pd.read_parquet(tf)
        tmpl["Date"] = pd.to_datetime(tmpl["Date"])
        gg = g[["Date", "UpProbability_new", "UpPrediction_new", "raw_new"]]
        merged = tmpl.merge(gg, on="Date", how="left")
        merged["UpProbability"] = merged["UpProbability_new"].fillna(merged["UpProbability"]).astype(np.float32)
        merged["DownProbability"] = (1.0 - merged["UpProbability"]).astype(np.float32)
        merged["UpPrediction"] = merged["UpPrediction_new"].fillna(merged["UpPrediction"]).astype(np.int8)
        merged["raw_score"] = merged["raw_new"].fillna(merged["raw_score"]).astype(np.float32)
        merged = merged.drop(columns=["UpProbability_new", "UpPrediction_new", "raw_new"])
        merged.to_parquet(f"{out_dir}/rf/{tk}.parquet", index=False)
        n += 1
    log(f"ensemble {out_dir}: wrote {n} tickers from {len(seeds)} seeds")


def parse_backtest(txt):
    txt = ANSI.sub("", txt)
    head = {}
    for key, pat in [("ann", r"Annualized Return %:\s*([-\d.]+)"),
                     ("sharpe", r"Sharpe Ratio:\s*([-\d.]+)"),
                     ("dd", r"Max Drawdown %:\s*([-\d.]+)"),
                     ("psr", r"Probabilistic Sharpe Ratio \(%\):\s*([-\d.]+)"),
                     ("wr", r"Win Rate \(after fees\) %:\s*([-\d.]+)")]:
        mm = re.search(pat, txt)
        if mm:
            head[key] = float(mm.group(1))
    monthly, sec = {}, False
    for line in txt.splitlines():
        if "Strategy Monthly Performance" in line:
            sec = True
            continue
        if sec:
            if "Monthly Excess" in line or "Strategy Yearly" in line:
                break
            mm = re.match(r"\s*(\d{4}-\d{2}):\s+(-?\d+\.?\d*)", line)
            if mm:
                monthly[mm.group(1)] = float(mm.group(2))
    return head, monthly


t0 = time.time()
log(f"PART 1 seed-ensemble: {len(SEEDS)} seeds @ {CUTOFF}, ensembles {ENSEMBLE_SIZES}")

# ---- Phase A: train seed pool (skip existing) ----
for s in SEEDS:
    md = seed_dir(s)
    if os.path.exists(f"{md}/xgb.joblib") and len(glob.glob(f"{md}/rf/*.parquet")) > 100:
        log(f"train seed {s}: exists, skip")
        continue
    log(f"train seed {s} ...")
    try:
        with open(f"{ENS}/train_s{s}.log", "w") as lf:
            subprocess.run([sys.executable, "4__Predictor.py", "--no_tune",
                            "--train_end_date", CUTOFF, "--seed", str(s), *PROD,
                            "--model_dir", md, "--output_dir", f"{md}/rf"],
                           stdout=lf, stderr=subprocess.STDOUT, check=True, timeout=1800)
    except Exception as e:
        log(f"train seed {s} FAILED: {repr(e)[:140]}")
log(f"Phase A (train) {(time.time()-t0)/60:.1f}m")

# ---- Phase B: build ensembles ----
avail = [s for s in SEEDS if os.path.exists(f"{seed_dir(s)}/xgb.joblib")
         and len(glob.glob(f"{seed_dir(s)}/rf/*.parquet")) > 100]
log(f"available seeds: {avail}")
for sz in ENSEMBLE_SIZES:
    if len(avail) >= sz:
        try:
            build_ensemble(avail[:sz], f"{ENS}/ens{sz}")
        except Exception as e:
            log(f"build ens{sz} FAILED: {repr(e)[:160]}")
log(f"Phase B (ensembles) {(time.time()-t0)/60:.1f}m")

# ---- Phase C: backtest singles + ensembles (3-wide parallel) ----
jobs = [(f"seed_s{s}", f"{seed_dir(s)}/rf") for s in avail]
jobs += [(f"ens{sz}", f"{ENS}/ens{sz}/rf") for sz in ENSEMBLE_SIZES
         if len(glob.glob(f"{ENS}/ens{sz}/rf/*.parquet")) > 100]


def launch(name, rf):
    lf = open(f"{ENS}/bt_{name}.log", "w")
    return subprocess.Popen([sys.executable, BACKTESTER, "--data_dir", rf],
                            stdout=lf, stderr=subprocess.STDOUT), lf


results = {}
pending = list(jobs)
running = {}
while pending or running:
    while pending and len(running) < PARALLEL:
        name, rf = pending.pop(0)
        log(f"backtest {name}: launch")
        p, lf = launch(name, rf)
        running[p] = (name, lf)
    time.sleep(5)
    for p in [q for q in running if q.poll() is not None]:
        name, lf = running.pop(p)
        lf.close()
        head, monthly = parse_backtest(open(f"{ENS}/bt_{name}.log").read())
        results[name] = {"head": head, "monthly": monthly}
        log(f"backtest {name}: ann={head.get('ann')} sharpe={head.get('sharpe')} dd={head.get('dd')}")

# retry empties
for name, rf in jobs:
    if not results.get(name) or not results[name]["head"]:
        log(f"retry {name}")
        p, lf = launch(name, rf)
        p.wait()
        lf.close()
        head, monthly = parse_backtest(open(f"{ENS}/bt_{name}.log").read())
        results[name] = {"head": head, "monthly": monthly}
log(f"Phase C (backtests) {(time.time()-t0)/60:.1f}m")

# ---- Phase D: aggregate ----
OOS = ["2026-01", "2026-02", "2026-03", "2026-04"]
singles = {k: v for k, v in results.items() if k.startswith("seed_")}


def oos_vol(monthly):
    vals = [monthly[m] for m in OOS if m in monthly]
    return float(np.std(vals)) if len(vals) > 1 else float("nan")


def oos_compound(monthly):
    p = 1.0
    for m in OOS:
        if m in monthly:
            p *= (1 + monthly[m] / 100.0)
    return (p - 1) * 100.0

lines = ["=" * 86, "SEED-ENSEMBLE VALIDATION — does averaging seeds cut strategy variance?",
         "=" * 86,
         f"{len(singles)} single seeds @ {CUTOFF}, prod params. OOS = Jan-Apr 2026.", "",
         f"  {'model':<10} {'ann%':>8} {'sharpe':>7} {'maxDD%':>7} {'OOS_cmp%':>9} {'OOS_vol':>8}"]
for name in sorted(singles):
    h = singles[name]["head"]
    mo = singles[name]["monthly"]
    lines.append(f"  {name:<10} {h.get('ann', float('nan')):>8.1f} {h.get('sharpe', float('nan')):>7.2f} "
                 f"{h.get('dd', float('nan')):>7.1f} {oos_compound(mo):>9.1f} {oos_vol(mo):>8.2f}")
# single-seed distribution stats
def stat(key):
    xs = [singles[n]["head"].get(key) for n in singles if singles[n]["head"].get(key) is not None]
    return (np.mean(xs), np.std(xs), np.min(xs), np.max(xs)) if xs else (np.nan,) * 4
lines += ["", "  SINGLE-SEED DISTRIBUTION (the variance we're trying to dampen):"]
for key, lab in [("ann", "ann%"), ("sharpe", "sharpe"), ("dd", "maxDD%")]:
    mu, sd, lo, hi = stat(key)
    lines.append(f"    {lab:<8} mean={mu:>7.2f}  std={sd:>6.2f}  range=[{lo:.2f}, {hi:.2f}]")
svols = [oos_vol(singles[n]["monthly"]) for n in singles]
lines.append(f"    OOS monthly vol: mean={np.nanmean(svols):.2f}  (per-model month-to-month std)")
# ensembles
lines += ["", "  ENSEMBLES (should beat the single-seed MEAN on sharpe/DD, lower OOS_vol):",
          f"  {'model':<10} {'ann%':>8} {'sharpe':>7} {'maxDD%':>7} {'OOS_cmp%':>9} {'OOS_vol':>8}"]
for sz in ENSEMBLE_SIZES:
    name = f"ens{sz}"
    if results.get(name) and results[name]["head"]:
        h = results[name]["head"]
        mo = results[name]["monthly"]
        lines.append(f"  {name:<10} {h.get('ann', float('nan')):>8.1f} {h.get('sharpe', float('nan')):>7.2f} "
                     f"{h.get('dd', float('nan')):>7.1f} {oos_compound(mo):>9.1f} {oos_vol(mo):>8.2f}")
lines += ["", "THESIS: ensemble sharpe > single-seed mean sharpe, ensemble maxDD < mean,",
          "ensemble OOS_vol < single-seed mean OOS_vol. If so, ship a seed-ensemble.",
          "=" * 86]
rep = "\n".join(lines)
with open(f"{ENS}/ENSEMBLE_REPORT.txt", "w") as f:
    f.write(rep)
print("\n" + rep, flush=True)
log(f"PART 1 DONE {(time.time()-t0)/60:.1f}m -> {ENS}/ENSEMBLE_REPORT.txt")
