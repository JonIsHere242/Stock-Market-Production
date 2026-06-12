"""Night-2 autonomous chain: intraday fill sim + independent-ensemble OOS confirmation.

1) Wait for the TWS 1-min download to finish, run 8__IntradayFillSim.py (real 10:00
   fills vs the backtest's open/close) on the new ensemble trades.
2) Train TWO fully-independent 8-seed ensembles (base seeds 100 & 200 -> members
   100-107 / 200-207), backtest each, and compare their OOS months (Jan-Apr 2026)
   to the user's seed-42 run. If all three land in the same OOS ballpark, the
   ensemble's +60% OOS is multi-path confirmed (not a lucky draw) — the rigor the
   session's noise-finding demands.

Autonomous (overnight-solver pattern): try/except + continue, generous timeouts
(sized after last night's timeout-kill lesson), incremental logs, side-dirs only,
NO live writes, NO broker.
"""
import os, re, subprocess, sys, time

PY = sys.executable
OOS = ['2026-01', '2026-02', '2026-03', '2026-04']
ANSI = re.compile(r'\x1b\[[0-9;]*m')
os.makedirs('Data/_confirm', exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def wait_marker(path, marker, max_hr=2):
    log(f"waiting for '{marker}' in {path}")
    t0 = time.time()
    while True:
        try:
            if os.path.exists(path) and marker in open(path, errors='ignore').read():
                log(f"'{marker}' detected after {(time.time()-t0)/60:.0f}m")
                return True
        except Exception:
            pass
        if time.time() - t0 > max_hr * 3600:
            log("timeout; proceeding")
            return False
        time.sleep(60)


def parse_monthly(txt):
    txt = ANSI.sub('', txt); out = {}; sec = False
    for line in txt.splitlines():
        if 'Strategy Monthly Performance' in line:
            sec = True; continue
        if sec:
            if 'Monthly Excess' in line or 'Strategy Yearly' in line:
                break
            m = re.match(r'\s*(\d{4}-\d{2}):\s+(-?\d+\.?\d*)', line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def parse_head(txt):
    txt = ANSI.sub('', txt); h = {}
    for k, p in [('ann', r'Annualized Return %:\s*([-\d.]+)'),
                 ('sharpe', r'Sharpe Ratio:\s*([-\d.]+)'),
                 ('dd', r'Max Drawdown %:\s*([-\d.]+)'),
                 ('win', r'Win Rate \(after fees\) %:\s*([-\d.]+)')]:
        m = re.search(p, txt)
        if m:
            h[k] = float(m.group(1))
    return h


def compound(m):
    p = 1.0
    for k in OOS:
        if m.get(k) is not None:
            p *= (1 + m[k] / 100.0)
    return (p - 1) * 100.0


t0 = time.time()

# ---- 1) intraday fill sim ----
wait_marker('Data/IntradayTradeSim/_download.log', 'DOWNLOAD COMPLETE', max_hr=2)
log("running intraday fill sim on the new ensemble trades")
try:
    with open('Data/IntradayTradeSim/_fillsim.log', 'w') as lf:
        subprocess.run([PY, '8__IntradayFillSim.py',
                        '--trade-history', 'Data/TradeHistory.parquet'],
                       stdout=lf, stderr=subprocess.STDOUT, timeout=1800)
    log("fill sim done -> Data/IntradayTradeSim/_fillsim.log")
except Exception as e:
    log(f"fill sim error: {repr(e)[:160]}")

# ---- 2) independent-ensemble OOS confirmation ----
# user's seed-42 run (from the backtest they pasted) as the reference path
results = {'ens_s42(user)': {'2026-01': 6.47, '2026-02': 26.14,
                             '2026-03': 8.26, '2026-04': 12.62,
                             'ann': 443.0, 'sharpe': 4.34, 'dd': 16.76}}
for seed in [100, 200]:
    tag = f'ens_s{seed}'
    mdir = f'Data/_confirm/{tag}'
    os.makedirs(mdir, exist_ok=True)
    if not os.path.exists(f'{mdir}/xgb.joblib'):
        log(f"train {tag}: independent 8-seed ensemble (base_seed={seed})")
        try:
            with open(f'Data/_confirm/{tag}_train.log', 'w') as lf:
                subprocess.run([PY, '4.1__Predictor.py', '--seed', str(seed),
                                '--model_dir', mdir, '--output_dir', f'{mdir}/rf'],
                               stdout=lf, stderr=subprocess.STDOUT,
                               timeout=5400, check=True)
        except Exception as e:
            log(f"train {tag} FAILED: {repr(e)[:160]}")
            continue
    else:
        log(f"train {tag}: model exists, skip")
    log(f"backtest {tag}")
    try:
        r = subprocess.run([PY, '5__NightlyBackTester_oosvalidate.py',
                            '--data_dir', f'{mdir}/rf'],
                           capture_output=True, text=True, timeout=3600)
        with open(f'Data/_confirm/{tag}_bt.log', 'w') as lf:
            lf.write(r.stdout)
        mo = parse_monthly(r.stdout); hd = parse_head(r.stdout)
        results[tag] = {**mo, **hd}
        log(f"{tag}: OOS_cmp={compound(mo):+.1f}% ann={hd.get('ann')} "
            f"Sharpe={hd.get('sharpe')} DD={hd.get('dd')}")
    except Exception as e:
        log(f"backtest {tag} error: {repr(e)[:160]}")

# ---- 3) report ----
lines = ["=" * 78,
         "NIGHT-2: independent-ensemble OOS confirmation  (OOS = Jan-Apr 2026)",
         "=" * 78,
         f"  {'ensemble':<14} {'Jan':>7} {'Feb':>7} {'Mar':>7} {'Apr':>7} "
         f"{'OOScmp%':>8} {'ann%':>6} {'Sh':>5} {'DD%':>6}"]
for tag, m in results.items():
    cells = ' '.join(f"{m.get(k, float('nan')):>7.2f}" for k in OOS)
    lines.append(f"  {tag:<14} {cells} {compound(m):>8.1f} "
                 f"{m.get('ann', float('nan')):>6.0f} {m.get('sharpe', float('nan')):>5.2f} "
                 f"{m.get('dd', float('nan')):>6.1f}")
lines += ["",
          "READ: if all independent ensembles land ~+50-65% OOS compounded with all",
          "months green, the ensemble OOS gain is MULTI-PATH CONFIRMED (reliable, not a",
          "lucky single draw) -> promote 4.1__Predictor.py to live. If they scatter",
          "wildly, even the ensemble is noisier than hoped and needs more members.",
          "", "Fill-sim (execution realism) is in Data/IntradayTradeSim/_fillsim.log.",
          "=" * 78]
report = "\n".join(lines)
with open('Data/_confirm/NIGHT2_REPORT.txt', 'w') as f:
    f.write(report)
print("\n" + report, flush=True)
log(f"NIGHT-2 COMPLETE in {(time.time()-t0)/60:.0f}m -> Data/_confirm/NIGHT2_REPORT.txt")
