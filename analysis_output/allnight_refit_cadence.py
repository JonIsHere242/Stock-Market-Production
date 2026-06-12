"""ALL-NIGHT autonomous experiment: rigorous multi-seed strategy-level refit-cadence.

Answers the one surviving OOS lever — does keeping the production model FRESH
actually improve strategy-level returns, robust to the strategy's brutal month-to-
month variance? Builds a seed-averaged STALENESS-DECAY curve at the strategy level.

Design (params held FIXED at production config; only cutoff + seed vary):
  cutoffs x seeds = models. Each model trained through its cutoff, full inference,
  then backtested (NO --force; uses the COPY backtester, never the original). A
  model trained through cutoff C yields, from one backtest, its return for every
  later month at increasing staleness (C+1 = 1mo, C+2 = 2mo, ...). Seed-averaging
  separates the freshness signal from the strategy noise.

Fully autonomous: no input prompts, skip-if-exists (resumable), try/except around
every step, INCREMENTAL result writes (partial completion still yields a report).
Touches NOTHING live: models/predictions go to Data/_refit_allnight/, backtests
read via --data_dir, and trade_history.parquet is backed up first.

Run:  python analysis_output/allnight_refit_cadence.py
"""
import glob, os, re, shutil, subprocess, sys, time
import pandas as pd

ROOT = "Data/_refit_allnight"
os.makedirs(ROOT, exist_ok=True)
os.makedirs("Data/_backups", exist_ok=True)
RESULTS_CSV = f"{ROOT}/results.csv"
BACKTESTER = "5__NightlyBackTester_oosvalidate.py"   # COPY — original untouched

# Fixed production params (Data/XGBPipeline/summary.json) — held constant.
PROD = ["--n_estimators", "304", "--max_depth", "8", "--learning_rate", "0.0509",
        "--min_child_weight", "15", "--subsample", "0.886",
        "--colsample_bytree", "0.444", "--reg_alpha", "0.0048",
        "--reg_lambda", "0.0223"]

CUTOFFS = ["2025-11-30", "2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31"]
SEEDS = [1, 2, 3, 4]

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_monthly(txt):
    txt = ANSI.sub("", txt)
    out, in_sec = {}, False
    for line in txt.splitlines():
        if "Strategy Monthly Performance" in line:
            in_sec = True
            continue
        if in_sec:
            if "Monthly Excess" in line or "Strategy Yearly" in line:
                break
            m = re.match(r"\s*(\d{4}-\d{2}):\s+(-?\d+\.?\d*)", line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def months_between(cut, oos_month):
    """Staleness in months: oos_month minus the cutoff's month."""
    c = pd.Period(cut[:7], "M")
    m = pd.Period(oos_month, "M")
    return (m.year - c.year) * 12 + (m.month - c.month)


def append_result(rows):
    df = pd.DataFrame(rows)
    header = not os.path.exists(RESULTS_CSV)
    df.to_csv(RESULTS_CSV, mode="a", header=header, index=False)


# ---- Phase 0: back up trade_history (overnight backtests overwrite it) ----
for f in glob.glob("trade_history.parquet"):
    bak = f"Data/_backups/trade_history_preallnight.parquet"
    if not os.path.exists(bak):
        try:
            shutil.copy(f, bak)
            log(f"backed up {f} -> {bak}")
        except Exception as e:
            log(f"backup warn: {e}")

t0 = time.time()
log(f"START all-night refit-cadence: {len(CUTOFFS)} cutoffs x {len(SEEDS)} seeds "
    f"= {len(CUTOFFS)*len(SEEDS)} models")

# Tags already fully processed (backtested -> in results); their prediction dirs
# may have been cleaned, so DON'T re-train them on resume.
_done_tags = set()
if os.path.exists(RESULTS_CSV):
    try:
        _done_tags = set(pd.read_csv(RESULTS_CSV)["tag"].astype(str).unique())
    except Exception:
        pass

# ---- Phase A: train all models (skip if already built or already backtested) ----
for cut in CUTOFFS:
    for s in SEEDS:
        tag = f"{cut.replace('-','')}_s{s}"
        mdir = f"{ROOT}/{tag}"
        if tag in _done_tags:
            log(f"train {tag}: already backtested, skip")
            continue
        if os.path.exists(f"{mdir}/xgb.joblib") and len(glob.glob(f"{mdir}/rf/*.parquet")) > 100:
            log(f"train {tag}: exists, skip")
            continue
        log(f"train {tag}: cutoff {cut} seed {s} ...")
        try:
            with open(f"{ROOT}/{tag}_train.log", "w") as lf:
                subprocess.run([sys.executable, "4__Predictor.py", "--no_tune",
                                "--train_end_date", cut, "--seed", str(s), *PROD,
                                "--model_dir", mdir, "--output_dir", f"{mdir}/rf"],
                               stdout=lf, stderr=subprocess.STDOUT, check=True, timeout=1800)
        except Exception as e:
            log(f"train {tag} FAILED: {repr(e)[:140]}")

log(f"Phase A (training) done in {(time.time()-t0)/60:.1f}m")

# ---- Phase B: backtest each SEQUENTIALLY (no --force), incremental results ----
already = set()
if os.path.exists(RESULTS_CSV):
    try:
        already = set(pd.read_csv(RESULTS_CSV)["tag"].unique())
    except Exception:
        pass

for cut in CUTOFFS:
    for s in SEEDS:
        tag = f"{cut.replace('-','')}_s{s}"
        mdir = f"{ROOT}/{tag}"
        if tag in already:
            log(f"backtest {tag}: already in results, skip")
            continue
        if not (os.path.exists(f"{mdir}/xgb.joblib") and len(glob.glob(f"{mdir}/rf/*.parquet")) > 100):
            log(f"backtest {tag}: no model/predictions, skip")
            continue
        monthly = {}
        for attempt in (1, 2):   # one retry on transient resource errors
            log(f"backtest {tag} (attempt {attempt}) ...")
            try:
                r = subprocess.run([sys.executable, BACKTESTER, "--data_dir", f"{mdir}/rf"],
                                   capture_output=True, text=True, timeout=3600)
                with open(f"{ROOT}/{tag}_backtest.log", "w") as lf:
                    lf.write(r.stdout)
                monthly = parse_monthly(r.stdout)
                if monthly:
                    break
                log(f"backtest {tag}: empty parse, retrying" if attempt == 1 else
                    f"backtest {tag}: empty after retry")
            except Exception as e:
                log(f"backtest {tag} attempt {attempt} error: {repr(e)[:140]}")
                time.sleep(5)
        rows = [{"tag": tag, "cutoff": cut, "seed": s, "oos_month": m,
                 "ret": v, "staleness_mo": months_between(cut, m)}
                for m, v in monthly.items() if months_between(cut, m) >= 1]
        if rows:
            append_result(rows)
            log(f"backtest {tag}: wrote {len(rows)} OOS-month rows")
        # free disk: predictions no longer needed once parsed
        try:
            shutil.rmtree(f"{mdir}/rf", ignore_errors=True)
        except Exception:
            pass

log(f"Phase B (backtests) done in {(time.time()-t0)/60:.1f}m")

# ---- Phase C: aggregate -> seed-averaged staleness-decay report ----
try:
    res = pd.read_csv(RESULTS_CSV)
except Exception as e:
    log(f"no results to aggregate: {e}")
    sys.exit(0)

lines = ["=" * 92,
         "ALL-NIGHT REFIT-CADENCE — strategy-level, seed-averaged",
         "=" * 92,
         f"models: {res['tag'].nunique()}   OOS rows: {len(res)}   "
         f"params fixed (production); only cutoff + seed vary", ""]

# 1) Staleness-decay curve: avg strategy return by months-stale (across all OOS months/seeds)
lines += ["--- STALENESS DECAY: mean strategy monthly return % by model age ---",
          f"  {'months_stale':<13} {'mean_ret%':>10} {'median%':>9} {'std%':>8} {'n':>5}"]
for st, g in res.groupby("staleness_mo"):
    lines.append(f"  {int(st):<13} {g['ret'].mean():>10.2f} {g['ret'].median():>9.2f} "
                 f"{g['ret'].std():>8.2f} {len(g):>5}")

# 2) Per-OOS-month: return by staleness (seed-averaged), to see fresh vs stale per month
lines += ["", "--- PER OOS MONTH: seed-avg return % by staleness (fresh=1mo) ---"]
for m, g in res.groupby("oos_month"):
    piv = g.groupby("staleness_mo")["ret"].mean()
    cells = "  ".join(f"{int(st)}mo={piv[st]:+.1f}" for st in sorted(piv.index))
    lines.append(f"  {m}:  {cells}")

# 3) FRESH (1mo) vs STALEST-available, per month + compounded
lines += ["", "--- FRESH (1mo) vs STALEST available, seed-averaged ---",
          f"  {'month':<9} {'fresh(1mo)':>11} {'stalest':>9} {'(mo)':>5} {'fresh-stale':>12}"]
fresh_track, stale_track = {}, {}
for m, g in res.groupby("oos_month"):
    piv = g.groupby("staleness_mo")["ret"].mean()
    fresh = piv.get(1)
    stalest_mo = max(piv.index)
    stale = piv.get(stalest_mo)
    fresh_track[m] = fresh
    stale_track[m] = stale
    diff = (fresh - stale) if (fresh is not None and stale is not None) else None
    lines.append(f"  {m:<9} {('' if fresh is None else f'{fresh:+.2f}'):>11} "
                 f"{('' if stale is None else f'{stale:+.2f}'):>9} {int(stalest_mo):>5} "
                 f"{('' if diff is None else f'{diff:+.2f}'):>12}")

def compound(d):
    p = 1.0
    for v in d.values():
        if v is not None:
            p *= (1 + v / 100.0)
    return (p - 1) * 100.0

lines += ["", f"  compounded  fresh={compound(fresh_track):+.2f}%   "
          f"stalest={compound(stale_track):+.2f}%",
          "", "READ: if mean_ret falls monotonically as months_stale rises, refitting helps and",
          "the slope = the cost per month of staleness. If flat/noisy, freshness doesn't matter.",
          "Seed std shows the strategy's intrinsic noise — the fresh-vs-stale gap must exceed it.",
          "=" * 92]

report = "\n".join(lines)
with open(f"{ROOT}/REPORT.txt", "w") as f:
    f.write(report)
print("\n" + report, flush=True)
log(f"DONE in {(time.time()-t0)/60:.1f}m  ->  {ROOT}/REPORT.txt")
