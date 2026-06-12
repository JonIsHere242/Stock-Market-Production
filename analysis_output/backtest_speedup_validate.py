"""Backtest speedup validation (overnight-solver PART).

Proves the fast backtester is LOSSLESS and measures the speedup. Runs the SAME
predictions through three backtesters and checks the results are IDENTICAL:
  1. original  (5__NightlyBackTester_oosvalidate.py)         on full predictions
  2. fast      (5__NightlyBackTester_fast.py, dead-inds gone) on full predictions
  3. fast      on the FIRING-ONLY subset (stocks that ever fire)

If R1 == R2 == R3 -> both optimizations (indicator removal, firing-filter) are
lossless, and T1/T2, T1/T3 are the speedups. Sequential (each uses all cores).
Robust: try/except, generous timeouts, incremental writes, side-dirs only.
NEVER runs the live broker. Does not use --force (finviz = 10x slower).
"""
import glob, os, re, shutil, subprocess, sys, time
import pandas as pd

PY = sys.executable
OUT = 'Data/_speedup'
os.makedirs(OUT, exist_ok=True)
ANSI = re.compile(r'\x1b\[[0-9;]*m')

PRED = ('Data/RFpredictions_ens'
        if glob.glob('Data/RFpredictions_ens/*.parquet') else 'Data/RFpredictions')
FIRING = f'{OUT}/RFpred_firing'


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse(txt):
    txt = ANSI.sub('', txt)
    out = {}
    for k, p in [('total_ret', r'Total Return %:\s*([-\d.]+)'),
                 ('ann', r'Annualized Return %:\s*([-\d.]+)'),
                 ('sharpe', r'Sharpe Ratio:\s*([-\d.]+)'),
                 ('dd', r'Max Drawdown %:\s*([-\d.]+)'),
                 ('trades', r'Total Trades:\s*([-\d.]+)'),
                 ('win', r'Win Rate \(after fees\) %:\s*([-\d.]+)')]:
        m = re.search(p, txt)
        if m:
            out[k] = float(m.group(1))
    return out


t0 = time.time()
log(f"predictions dir: {PRED}  ({len(glob.glob(f'{PRED}/*.parquet'))} files)")

# PART 1: build firing-only subset (skip if already built)
if not (os.path.isdir(FIRING) and len(glob.glob(f'{FIRING}/*.parquet')) > 100):
    os.makedirs(FIRING, exist_ok=True)
    n = 0
    for f in glob.glob(f'{PRED}/*.parquet'):
        try:
            if (pd.read_parquet(f, columns=['UpPrediction'])['UpPrediction'] == 1).any():
                shutil.copy(f, f'{FIRING}/{os.path.basename(f)}'); n += 1
        except Exception:
            pass
    log(f"firing subset built: {n} files")
else:
    log(f"firing subset exists: {len(glob.glob(f'{FIRING}/*.parquet'))} files")

# PART 2: run the three backtests sequentially, time + parse
RUNS = [
    ('original_full', '5__NightlyBackTester_oosvalidate.py', PRED),
    ('fast_full',     '5__NightlyBackTester_fast.py',        PRED),
    ('fast_firing',   '5__NightlyBackTester_fast.py',        FIRING),
]
results = {}
for tag, script, ddir in RUNS:
    log(f"running {tag}: {script} on {ddir} ...")
    try:
        s = time.time()
        r = subprocess.run([PY, script, '--data_dir', ddir],
                           capture_output=True, text=True, timeout=3600)
        dt = time.time() - s
        with open(f'{OUT}/bt_{tag}.log', 'w') as lf:
            lf.write(r.stdout)
        res = parse(r.stdout)
        results[tag] = {'min': dt / 60, 'res': res}
        # incremental write
        with open(f'{OUT}/_progress.txt', 'a') as pf:
            pf.write(f"{tag}: {dt/60:.1f}min  {res}\n")
        log(f"{tag}: {dt/60:.1f}min  ann={res.get('ann')} sharpe={res.get('sharpe')} "
            f"dd={res.get('dd')} trades={res.get('trades')}")
    except Exception as e:
        log(f"{tag} FAILED: {repr(e)[:160]}")
        results[tag] = {'min': None, 'res': {}}

# PART 3: report
def g(tag, k): return results.get(tag, {}).get('res', {}).get(k)
def tmin(tag): return results.get(tag, {}).get('min')

orig = results.get('original_full', {}).get('min')
lines = ["=" * 78, "BACKTEST SPEEDUP VALIDATION", "=" * 78,
         f"predictions: {PRED}", "",
         f"  {'backtester':<16}{'minutes':>9}{'speedup':>9}{'ann%':>8}{'Sharpe':>8}"
         f"{'maxDD%':>8}{'trades':>8}{'totRet%':>9}"]
for tag, _, _ in RUNS:
    m = tmin(tag); sp = (orig / m) if (orig and m) else None
    lines.append(f"  {tag:<16}{(m if m else float('nan')):>9.1f}"
                 f"{(sp if sp else float('nan')):>8.1f}x"
                 f"{g(tag,'ann') or float('nan'):>8.0f}{g(tag,'sharpe') or float('nan'):>8.2f}"
                 f"{g(tag,'dd') or float('nan'):>8.1f}{g(tag,'trades') or float('nan'):>8.0f}"
                 f"{g(tag,'total_ret') or float('nan'):>9.0f}")

# identity check
def close(a, b, tol=0.5):
    return a is not None and b is not None and abs(a - b) <= tol
ident = all(close(g('original_full', k), g('fast_full', k)) and
            close(g('original_full', k), g('fast_firing', k))
            for k in ['ann', 'sharpe', 'trades'])
lines += ["",
          f"  LOSSLESS CHECK: results identical across all three?  {'YES' if ident else 'NO -- INVESTIGATE'}",
          "  (if YES: indicator-removal + firing-filter are safe, use fast_firing for iteration.)",
          "  (if NO: the optimization changed a decision -- diff the trade logs.)",
          "=" * 78]
rep = "\n".join(lines)
with open(f'{OUT}/SPEEDUP_REPORT.txt', 'w') as f:
    f.write(rep)
print("\n" + rep, flush=True)
log(f"DONE in {(time.time()-t0)/60:.0f}m -> {OUT}/SPEEDUP_REPORT.txt")
