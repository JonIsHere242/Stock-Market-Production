import sys, re, pandas as pd

H, T, log, parquet, sumf = sys.argv[1:6]
s = re.sub(r'\x1b\[[0-9;]*m', '', open(log, encoding='utf-8', errors='ignore').read())

def g(p, d='NA'):
    m = re.search(p, s)
    return m.group(1) if m else d

ret = g(r'Total Return %:\s*([0-9.\-]+)')
sh  = g(r'Sharpe Ratio:\s*([0-9.\-]+)')
sor = g(r'Sortino Ratio:\s*([0-9.\-]+)')
mdd = g(r'Max Drawdown %:\s*([0-9.\-]+)')
win = g(r'Win Rate \(after fees\) %:\s*([0-9.\-]+)')
warn = 'SHORT-LEAK!' if 'STOPFIX-WARN' in s else ''

try:
    d = pd.read_parquet(parquet)
    n = len(d)
    se = int(d.ExitReason.isin(['Trailing Stop', 'Hard Stop']).sum())
    md = float(d.DaysHeld.median())
except Exception as e:
    n = se = md = 'NA'

cfg = f"hard{H}/trail{T}"
line = f"{cfg:<16} ret={ret:>8}  sharpe={sh:>5}  sortino={sor:>5}  maxDD={mdd:>6}  win={win:>5}  trades={str(n):>4}  stopexits={str(se):>4}  medHold={md}  {warn}"
open(sumf, 'a', encoding='utf-8').write(line + "\n")
print(line)
