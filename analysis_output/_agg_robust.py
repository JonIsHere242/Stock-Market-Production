import re, glob, os, statistics
from collections import defaultdict

res = defaultdict(list)
for f in glob.glob('analysis_output/stoprob_t*_s*.log'):
    base = os.path.basename(f)
    m = re.match(r'stoprob_t([0-9.]+)_s([0-9]+)\.log', base)
    if not m:
        continue
    T, seed = m.group(1), m.group(2)
    s = re.sub(r'\x1b\[[0-9;]*m', '', open(f, encoding='utf-8', errors='ignore').read())
    rm = re.search(r'Total Return %:\s*([0-9.\-]+)', s)
    shm = re.search(r'Sharpe Ratio:\s*([0-9.\-]+)', s)
    if rm:
        res[T].append((seed, float(rm.group(1)), float(shm.group(1)) if shm else float('nan')))

def key(t):
    return 999.0 if t == '100' else float(t)

print(f'{"config":>9} {"n":>2} {"mean_ret":>9} {"min":>7} {"max":>7} {"mean_shrp":>9}   per-seed returns')
for T in sorted(res, key=key):
    rets = [r for _, r, _ in res[T]]
    shs = [h for _, _, h in res[T]]
    lab = 'no-stop' if T == '100' else f'trail{T}'
    print(f'{lab:>9} {len(rets):>2} {statistics.mean(rets):>9.1f} {min(rets):>7.1f} {max(rets):>7.1f} {statistics.mean(shs):>9.2f}   {sorted(round(r) for r in rets)}')
