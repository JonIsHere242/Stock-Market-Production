#!/usr/bin/env python3
"""
PARALLEL focused exit sweep (fast path). Runs the HARD-stop frontier (the winners)
+ tighter stops, 3-wide concurrent, resumable (skips tags already in the CSV).
Trail configs are skipped — already proven worst. Metrics from stdout; per-config
trades -> Data/TradeHistory_sweep_<tag>.parquet. Restores TradeHistory.parquet at end.
"""
import subprocess, sys, os, re, time, csv, shutil, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

RESULTS = os.path.join('analysis_output', 'exit_sweep_results.csv')
BACKUP  = os.path.join('_backups', 'stage2_20260612', 'TradeHistory.parquet')
LIVE_TH = os.path.join('Data', 'TradeHistory.parquet')
WORKERS = 3
os.makedirs('analysis_output', exist_ok=True)
_lock = threading.Lock()

# Fill the hard-stop x TP grid (skip trail = known worst). stop in {1.0,1.5,1.9,2.5,5,8,100}.
NEW = []
for tp in (3.5, 20):
    for stop in (1.0, 1.5, 1.9, 2.5, 5, 8, 100):
        NEW.append(('hard', stop, tp))

ANSI = re.compile(r'\x1b\[[0-9;]*m')
def parse(out):
    o = ANSI.sub('', out)
    def g(pat, cast=float):
        m = re.search(pat, o); return cast(m.group(1)) if m else None
    return dict(
        ann=g(r'Annualized Return:\s+([\-\d.]+)'), maxdd=g(r'Max Drawdown %:\s+([\-\d.]+)'),
        win=g(r'Win Rate \(after fees\) %:\s+([\-\d.]+)'), pf=g(r'Profit Factor:\s+([\-\d.]+)'),
        psr=g(r'Probabilistic Sharpe Ratio \(%\):\s*([\-\d.]+)'), trades=g(r'Total Trades:\s+(\d+)', int))

FIELDS = ['tag','mode','stop','tp','ann','maxdd','win','pf','psr','trades','secs']

def done_tags():
    if not os.path.exists(RESULTS): return set()
    with open(RESULTS) as f: return {r['tag'] for r in csv.DictReader(f)}

def run_one(cfg):
    mode, stop, tp = cfg
    tag = f"{mode}{stop}_tp{tp}"
    cmd = [sys.executable, '-u', '5__NightlyBackTester_bracket.py', '--runpercent', '100',
           '--stop_mode', mode, '--stop_pct', str(stop), '--tp_pct', str(tp), '--run_tag', tag]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        m = parse(p.stdout)
    except Exception as e:
        m = dict(ann=None, maxdd=None, win=None, pf=None, psr=None, trades=None)
        print(f"[par] {tag} FAILED: {e}", flush=True)
    m.update(tag=tag, mode=mode, stop=stop, tp=tp, secs=round(time.time()-t0))
    with _lock:
        wh = not os.path.exists(RESULTS)
        with open(RESULTS, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if wh: w.writeheader()
            w.writerow({k: m.get(k) for k in FIELDS})
    rd = (m['ann']/m['maxdd']) if (m['ann'] and m['maxdd']) else 0
    print(f"[par] {tag:14s} ann={m['ann']} maxDD={m['maxdd']} win={m['win']} retDD={rd:.2f} ({m['secs']}s)", flush=True)
    return tag

def main():
    done = done_tags()
    todo = [c for c in NEW if f"{c[0]}{c[1]}_tp{c[2]}" not in done]
    print(f"[par] {len(NEW)} grid configs, {len(done)} already done, {len(todo)} to run, {WORKERS}-wide", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(run_one, c) for c in todo]
        for _ in as_completed(futs):
            pass
    if os.path.exists(BACKUP):
        try: shutil.copy(BACKUP, LIVE_TH)
        except Exception: pass

    # full frontier
    rows = []
    with open(RESULTS) as f:
        for r in csv.DictReader(f):
            try:
                r['ann'] = float(r['ann']); r['maxdd'] = float(r['maxdd'])
                r['retDD'] = r['ann']/r['maxdd'] if r['maxdd'] else 0
                rows.append(r)
            except (TypeError, ValueError): continue
    print("\n" + "="*82)
    print("  FULL EXIT FRONTIER (sorted by return/drawdown)")
    print("="*82)
    print(f"  {'config':16s} {'ann%':>7} {'maxDD%':>7} {'ret/DD':>7} {'win%':>6} {'PF':>5} {'trades':>7}")
    for r in sorted(rows, key=lambda x: x['retDD'], reverse=True):
        print(f"  {r['tag']:16s} {r['ann']:7.1f} {r['maxdd']:7.1f} {r['retDD']:7.2f} "
              f"{float(r['win']):6.1f} {float(r['pf']):5.2f} {r['trades']:>7}")
    print("="*82)
    # also top by raw return
    print("  TOP 5 BY RAW RETURN:")
    for r in sorted(rows, key=lambda x: x['ann'], reverse=True)[:5]:
        print(f"    {r['tag']:16s} ann={r['ann']:.1f}  maxDD={r['maxdd']:.1f}  retDD={r['retDD']:.2f}")

if __name__ == '__main__':
    main()
