#!/usr/bin/env python
"""Run the live-faithful backtest of the SHIPPED configuration with no env setup.

    python run_optimal_backtest.py                 # seed 1, full universe, ~90 s
    python run_optimal_backtest.py --seed 5        # another shuffle seed (1, 2, 5, 13 measured)
    python run_optimal_backtest.py --sample 25     # quick 25% universe smoke run
    python run_optimal_backtest.py --set BT_GATE_GAP_PCT=3.0   # override any knob

What it runs: experimental/backtesters/5v4__ExitLab.py (the lab copy of the 5v2 trigger
arm) in the configuration the broker trades since 2026-08-29:
    trigger entry K 1.5, Parkinson vol, oversub 4 x 3 slots (reach 12), model exits OFF,
    cashadjust double-count FIXED, Tiered commissions,
    gap-frequency pool gate 2 gaps > 4% in 20 bars, scale-out sells HALF at +15%.
Paired 8 seeds vs the pre-08-29 base: AnnRet +45.5pp (t 9.3, 8/8), maxDD -3.5pp (7/8),
merged mean/TRADE +0.66pp (t 8.2, 8/8). Memo project_exit_sweeps_trigger_arm_2026_08_29.

Outputs go under Data/_canbuyv2/optimal/ only. The live trade book and signal pool are
md5-checked before and after. The full printout streams to the terminal and is also
saved next to the trade book. Refuses to start while another backtester is running.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
# Since 2026-08-30 the production nightly IS the optimal configuration (its defaults
# carry every knob below), so this wrapper's job is the SANDBOX REDIRECT: a hand run
# must never overwrite Data/0__Signals.parquet or Data/TradeHistory.parquet, which the
# nightly owns. The knob env is kept explicit so the printout says what ran.
SCRIPT = os.path.join(REPO, '5__NightlyBackTester.py')
OUT_DIR = os.path.join(REPO, 'Data', '_canbuyv2', 'optimal')
LIVE_FILES = [os.path.join(REPO, 'Data', 'TradeHistory.parquet'),
              os.path.join(REPO, 'Data', '0__Signals.parquet'),
              os.path.join(REPO, '_Buy_Signals.parquet')]

OPTIMAL_ENV = {
    'PYTHONPATH': REPO,
    'PYTHONIOENCODING': 'utf-8',
    'PYTHONUTF8': '1',
    'BT_SAMPLE_SEED': '42',
    # entry arm the broker runs
    'BT_LIMIT_ENTRY_K': '1.5',
    'BT_LIMIT_OVERSUB': '4',
    'BT_LIMIT_VOL_EST': 'yz',   # shipped 2026-08-30, rollback parkinson
    # honest accounting and the broker's exit set
    'BT_CASHADJUST_FIX': '1',
    'BT_PROD_EXITS': '1',
    # the 2026-08-29 baseline
    'BT_GATE_GAPFREQ_N': '2',
    'BT_GATE_GAP_PCT': '4.0',
    'BT_RUNNER_TRIG': '15', 'BRACKET_RUNNER_TRIG': '15',
    'BT_RUNNER_KEEP': '0.5', 'BRACKET_RUNNER_KEEP': '0.5',
}


def md5(path):
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def other_backtests_running():
    try:
        out = subprocess.run(['wmic', 'process', 'where', "name='python.exe'", 'get', 'CommandLine'],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    me = os.path.basename(__file__)
    return [l.strip()[:140] for l in out.splitlines()
            if ('NightlyBackTester' in l or 'ExitLab' in l) and me not in l]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=1, help='BT_LIMIT_SHUFFLE seed (default 1)')
    ap.add_argument('--sample', default='100', help='percent of the universe (default 100)')
    ap.add_argument('--tag', default='optimal')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VAL', help='override any env knob')
    ap.add_argument('--data_dir', default=None,
                    help="Pass through to 5__'s --data_dir: backtest a candidate model's "
                         'prediction dir instead of live Data/RFpredictions (read-only).')
    ap.add_argument('--no-guard', action='store_true', help='skip the running-backtester check')
    a = ap.parse_args()

    if not a.no_guard:
        busy = other_backtests_running()
        if busy:
            print('ABORT: a backtester is already running (concurrent runs contend on Data/logging):')
            for b in busy:
                print('   ', b)
            sys.exit(2)

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, f'{a.tag}_shuf{a.seed}')
    env = dict(os.environ)
    env.update(OPTIMAL_ENV)
    for kv in a.set:
        k, _, v = kv.partition('=')
        env[k] = v
    env['BT_LIMIT_SHUFFLE'] = str(a.seed)
    env['BT_V2_TRADEHIST'] = os.path.relpath(stem + '_TradeHistory.parquet', REPO).replace('\\', '/')
    env['BT_V2_SIGNALS'] = os.path.relpath(stem + '_signals.parquet', REPO).replace('\\', '/')

    before = {p: md5(p) for p in LIVE_FILES}
    cmd = [sys.executable, SCRIPT, '--force', '--sample', str(a.sample)]
    if a.data_dir:
        cmd += ['--data_dir', a.data_dir]
    print('== optimal backtest:', ' '.join(cmd))
    print('== knobs:', ' '.join(f'{k}={env[k]}' for k in sorted(OPTIMAL_ENV) if k.startswith(('BT_', 'BRACKET_'))),
          f'BT_LIMIT_SHUFFLE={a.seed}')
    print('== trade book ->', env['BT_V2_TRADEHIST'])
    t0 = time.time()
    log_path = stem + '.log'
    ok = False
    with open(log_path, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            if 'Annualized Return %' in line:
                ok = True
        rc = proc.wait()
    after = {p: md5(p) for p in LIVE_FILES}
    changed = [p for p in LIVE_FILES if before[p] != after[p]]
    print(f'\n== finished in {time.time() - t0:.0f}s, rc={rc}, summary line {"FOUND" if ok else "MISSING (run FAILED, read the log)"}')
    print('== log:', log_path)
    if changed:
        print('!!!!! LIVE FILE CHANGED during the run (restore from experimental/_rollback):', changed)
        sys.exit(3)
    print('== live trade book and signal pool unchanged')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
