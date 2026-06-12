"""FAST refit-cadence probe (~1hr): is fresh>stale seed-robust on Mar & Apr?

Focused subset of the all-night battery. Reuses already-trained Dec models
(= current live vintage / STALE), trains Feb & Mar cutoffs (= 1-month FRESH for
Mar & Apr), backtests all 6 with a 3-WIDE PARALLEL pool, and compares per seed.

  Mar 2026: STALE = Dec model (3mo) vs FRESH = Feb model (1mo)
  Apr 2026: STALE = Dec model (4mo) vs FRESH = Mar model (1mo)

Params fixed (production); only cutoff + seed vary. Touches nothing live; uses the
COPY backtester via --data_dir. Autonomous, resumable (skip-if-exists), retries.
"""
import glob, os, re, subprocess, sys, time
import pandas as pd

ROOT = "Data/_refit_allnight"
os.makedirs(ROOT, exist_ok=True)
BACKTESTER = "5__NightlyBackTester_oosvalidate.py"   # COPY — original untouched
PARALLEL = 3

PROD = ["--n_estimators", "304", "--max_depth", "8", "--learning_rate", "0.0509",
        "--min_child_weight", "15", "--subsample", "0.886",
        "--colsample_bytree", "0.444", "--reg_alpha", "0.0048",
        "--reg_lambda", "0.0223"]

# (tag, cutoff, seed, role)  Dec = stale (reuse), Feb/Mar = fresh (train)
MODELS = [("20251231_s1", "2025-12-31", 1), ("20251231_s2", "2025-12-31", 2),
          ("20260228_s1", "2026-02-28", 1), ("20260228_s2", "2026-02-28", 2),
          ("20260331_s1", "2026-03-31", 1), ("20260331_s2", "2026-03-31", 2)]

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_monthly(txt):
    txt = ANSI.sub("", txt)
    out, sec = {}, False
    for line in txt.splitlines():
        if "Strategy Monthly Performance" in line:
            sec = True
            continue
        if sec:
            if "Monthly Excess" in line or "Strategy Yearly" in line:
                break
            mm = re.match(r"\s*(\d{4}-\d{2}):\s+(-?\d+\.?\d*)", line)
            if mm:
                out[mm.group(1)] = float(mm.group(2))
    return out


t0 = time.time()
log(f"FAST probe: {len(MODELS)} models, {PARALLEL}-wide backtests")

# ---- Phase A: train missing models (Dec reused if present) ----
for tag, cut, s in MODELS:
    mdir = f"{ROOT}/{tag}"
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
log(f"Phase A done in {(time.time()-t0)/60:.1f}m")


# ---- Phase B: backtest all 6 in a 3-wide parallel pool ----
def launch(tag):
    lf = open(f"{ROOT}/{tag}_fastbt.log", "w")
    p = subprocess.Popen([sys.executable, BACKTESTER, "--data_dir", f"{ROOT}/{tag}/rf"],
                         stdout=lf, stderr=subprocess.STDOUT)
    return p, lf


results = {}
pending = [tag for tag, _, _ in MODELS
           if os.path.exists(f"{ROOT}/{tag}/xgb.joblib")
           and len(glob.glob(f"{ROOT}/{tag}/rf/*.parquet")) > 100]
running = {}
while pending or running:
    while pending and len(running) < PARALLEL:
        tag = pending.pop(0)
        log(f"backtest {tag}: launch ({len(running)+1} running)")
        p, lf = launch(tag)
        running[p] = (tag, lf)
    time.sleep(5)
    for p in [q for q in running if q.poll() is not None]:
        tag, lf = running.pop(p)
        lf.close()
        monthly = parse_monthly(open(f"{ROOT}/{tag}_fastbt.log").read())
        results[tag] = monthly
        log(f"backtest {tag}: done ({'OK' if monthly else 'EMPTY'})")

# retry any empties sequentially
for tag in [t for t, _, _ in MODELS if not results.get(t)]:
    if not (os.path.exists(f"{ROOT}/{tag}/xgb.joblib")):
        continue
    log(f"retry backtest {tag} (sequential)")
    p, lf = launch(tag)
    p.wait()
    lf.close()
    results[tag] = parse_monthly(open(f"{ROOT}/{tag}_fastbt.log").read())

log(f"Phase B done in {(time.time()-t0)/60:.1f}m")


# ---- Phase C: compare fresh vs stale on Mar & Apr ----
def avg(tags, month):
    vals = [results[t].get(month) for t in tags if results.get(t) and results[t].get(month) is not None]
    return (sum(vals) / len(vals), vals) if vals else (None, [])

dec = ["20251231_s1", "20251231_s2"]
feb = ["20260228_s1", "20260228_s2"]
mar = ["20260331_s1", "20260331_s2"]

lines = ["=" * 78, "FAST REFIT-CADENCE PROBE — fresh(1mo) vs stale(Dec) on Mar & Apr",
         "=" * 78, "params fixed; seeds={1,2}. STALE=Dec vintage (current live), FRESH=1mo refit", ""]
for month, fresh_tags, stale_mo, fresh_mo in [("2026-03", feb, 3, 1), ("2026-04", mar, 4, 1)]:
    sm, sv = avg(dec, month)
    fm, fv = avg(fresh_tags, month)
    lines.append(f"--- {month} ---")
    lines.append(f"  STALE (Dec, {stale_mo}mo):  seeds={['%+.2f'%x for x in sv]}  mean={('%+.2f'%sm) if sm is not None else 'NA'}")
    lines.append(f"  FRESH (1mo refit):     seeds={['%+.2f'%x for x in fv]}  mean={('%+.2f'%fm) if fm is not None else 'NA'}")
    if sm is not None and fm is not None:
        lines.append(f"  >> FRESH - STALE = {fm-sm:+.2f} pp  ({'FRESH wins' if fm>sm else 'STALE wins'})")
    lines.append("")
lines += ["Read: if FRESH beats STALE on BOTH months across BOTH seeds, the single-path",
          "finding holds and refit cadence is worth it. If seeds straddle, it's noise.",
          "=" * 78]
report = "\n".join(lines)
with open(f"{ROOT}/FAST_REPORT.txt", "w") as f:
    f.write(report)
print("\n" + report, flush=True)
log(f"DONE in {(time.time()-t0)/60:.1f}m -> {ROOT}/FAST_REPORT.txt")
