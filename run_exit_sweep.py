#!/usr/bin/env python3
"""
Strategy-level EXIT SWEEP orchestrator.
Runs 5__NightlyBackTester_bracket.py across a grid of exit configs (stop_mode x
stop_pct x tp_pct), parses the metrics from each run, writes incremental results,
and prints the return/risk frontier. Resumable (skips configs already in results).
Restores Data/TradeHistory.parquet from backup after each run (stray writer guard).

Each backtest ~5 min; full grid ~75 min. Metrics come from stdout (not the clobbered
TradeHistory.parquet), so the stray-writer issue doesn't affect results.
"""
import subprocess, sys, os, re, time, csv, shutil

RESULTS = os.path.join('analysis_output', 'exit_sweep_results.csv')
BACKUP  = os.path.join('_backups', 'stage2_20260612', 'TradeHistory.parquet')
LIVE_TH = os.path.join('Data', 'TradeHistory.parquet')
os.makedirs('analysis_output', exist_ok=True)

# (stop_mode, stop_pct, tp_pct) — ordered most-informative first so partial runs are useful
CONFIGS = [
    ('hard', 5,   20),    # TP isolation: does wide TP recover return at low DD?  (the big one)
    ('trail', 3,  20),    # ~= canonical baseline (sanity, expect ~154% / ~22% DD)
    ('hard', 5,   3.5),   # the bracket (sanity, expect ~145% / ~15% DD)
    ('trail', 1.5, 3.5),  # live-approx (trail floor)
    ('hard', 1.9, 3.5),   # live hard-approx
    ('hard', 100, 20),    # no stop + wide TP = max-return ceiling
    ('hard', 8,   20),
    ('trail', 5,  20),
    ('hard', 8,   3.5),
    ('trail', 1.5, 20),
    ('hard', 1.9, 20),
    ('trail', 5,  3.5),
    ('trail', 3,  3.5),
    ('hard', 100, 3.5),   # no stop + tight TP
]

ANSI = re.compile(r'\x1b\[[0-9;]*m')
def parse(out):
    o = ANSI.sub('', out)
    def g(pat, cast=float):
        m = re.search(pat, o)
        return cast(m.group(1)) if m else None
    return dict(
        ann   = g(r'Annualized Return:\s+([\-\d.]+)'),
        maxdd = g(r'Max Drawdown %:\s+([\-\d.]+)'),
        win   = g(r'Win Rate \(after fees\) %:\s+([\-\d.]+)'),
        pf    = g(r'Profit Factor:\s+([\-\d.]+)'),
        psr   = g(r'Probabilistic Sharpe Ratio \(%\):\s*([\-\d.]+)'),
        trades= g(r'Total Trades:\s+(\d+)', int),
    )

FIELDS = ['tag','mode','stop','tp','ann','maxdd','win','pf','psr','trades','secs']

def load_done():
    if not os.path.exists(RESULTS): return set()
    with open(RESULTS) as f:
        return {r['tag'] for r in csv.DictReader(f)}

def main():
    done = load_done()
    print(f"[sweep] {len(CONFIGS)} configs, {len(done)} already done", flush=True)
    for mode, stop, tp in CONFIGS:
        tag = f"{mode}{stop}_tp{tp}"
        if tag in done:
            print(f"[sweep] skip {tag} (done)", flush=True); continue
        cmd = [sys.executable, '-u', '5__NightlyBackTester_bracket.py', '--runpercent', '100',
               '--stop_mode', mode, '--stop_pct', str(stop), '--tp_pct', str(tp), '--run_tag', tag]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            m = parse(p.stdout)
        except Exception as e:
            print(f"[sweep] {tag} FAILED: {e}", flush=True)
            m = dict(ann=None, maxdd=None, win=None, pf=None, psr=None, trades=None)
        m.update(tag=tag, mode=mode, stop=stop, tp=tp, secs=round(time.time()-t0))
        write_header = not os.path.exists(RESULTS)
        with open(RESULTS, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header: w.writeheader()
            w.writerow({k: m.get(k) for k in FIELDS})
        if os.path.exists(BACKUP):
            try: shutil.copy(BACKUP, LIVE_TH)
            except Exception: pass
        rd = (m['ann']/m['maxdd']) if (m['ann'] and m['maxdd']) else None
        print(f"[sweep] {tag:16s} ann={m['ann']} maxDD={m['maxdd']} win={m['win']} "
              f"PF={m['pf']} trades={m['trades']} ret/DD={rd:.2f}" if rd else
              f"[sweep] {tag:16s} ann={m['ann']} maxDD={m['maxdd']} (parse incomplete)", flush=True)

    # frontier
    rows = []
    with open(RESULTS) as f:
        for r in csv.DictReader(f):
            try:
                r['ann'] = float(r['ann']); r['maxdd'] = float(r['maxdd'])
                r['retDD'] = r['ann']/r['maxdd'] if r['maxdd'] else 0
                rows.append(r)
            except (TypeError, ValueError):
                continue
    print("\n" + "="*78)
    print("  EXIT SWEEP FRONTIER  (sorted by return/drawdown — best risk-adjusted first)")
    print("="*78)
    print(f"  {'config':16s} {'ann%':>7} {'maxDD%':>7} {'ret/DD':>7} {'win%':>6} {'PF':>5} {'trades':>7}")
    for r in sorted(rows, key=lambda x: x['retDD'], reverse=True):
        print(f"  {r['tag']:16s} {r['ann']:7.1f} {r['maxdd']:7.1f} {r['retDD']:7.2f} "
              f"{float(r['win']):6.1f} {float(r['pf']):5.2f} {r['trades']:>7}")
    print("="*78)
    print(f"  results: {RESULTS}  | per-config trades: Data/TradeHistory_sweep_<tag>.parquet")

if __name__ == '__main__':
    main()
