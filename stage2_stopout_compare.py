#!/usr/bin/env python3
"""Stop-out RATE comparison on the 5-min lake (recorded-fill anchor):
current LIVE bracket (hard 1.9% + ATR trail + 3.5% TP) vs the recommended drop-trail
variants. Directly answers 'will it stop out less?'. Reuses stage2_stopsweep loaders."""
import numpy as np, pandas as pd
from stage2_stopsweep import load5, build_trade  # functions only (guarded main)

tr = pd.read_parquet('Data/TradeHistory.parquet')
tr['EntryDate'] = pd.to_datetime(tr['EntryDate']); tr['ExitDate'] = pd.to_datetime(tr['ExitDate'])
trades = []
for _, t in tr.iterrows():
    b = build_trade(str(t.Symbol).upper(), t.EntryDate, t.ExitDate, float(t.EntryPrice))
    if isinstance(b, dict):
        trades.append(b)
print(f"priced {len(trades)} trades on 5-min lake\n")

def evalc(t, hard_pct, trail, tp_pct):
    entry, O, H, L, atr = t['entry'], t['O'], t['H'], t['L'], t['atr']
    hard = entry * (1 - hard_pct/100) if hard_pct else -1e18
    tp = entry * (1 + tp_pct/100) if tp_pct else 1e18
    if trail == 'atr': tpct = min(max(1.5, 0.75*atr), 4.0)
    elif trail:       tpct = float(trail)
    else:             tpct = None
    hw = entry
    for i in range(len(O)):
        o, h, l = O[i], H[i], L[i]
        if h > hw: hw = h
        trail_lvl = hw*(1 - tpct/100) if tpct is not None else -1e18
        stop = max(hard, trail_lvl)
        which = 'hard' if hard >= trail_lvl else 'trail'
        if l <= stop:
            px = o if o <= stop else stop
            return which, (px/entry - 1)*100
        if h >= tp:
            px = o if o >= tp else tp
            return 'tp', (px/entry - 1)*100
    return 'time', (O[-1]/entry - 1)*100

CONFIGS = [
    ('LIVE  hard1.9 + ATR-trail + TP3.5', 1.9, 'atr', 3.5),
    ('FIX   hard1.9 + NO trail   + TP3.5', 1.9, None, 3.5),
    ('FIX   hard5   + NO trail   + TP3.5', 5.0, None, 3.5),
    ('FIX   hard8   + NO trail   + TP3.5', 8.0, None, 3.5),
]
n = len(trades)
print(f"{'config':38s} {'STOPPED':>8} {'(hard':>6} {'trail)':>7} {'TP':>6} {'time':>6} {'meanPnL':>8} {'win%':>6}")
for name, hp, trl, tp in CONFIGS:
    cnt = {'hard':0,'trail':0,'tp':0,'time':0}; pnls = []
    for t in trades:
        r, p = evalc(t, hp, trl, tp); cnt[r] += 1; pnls.append(p)
    stopped = (cnt['hard']+cnt['trail'])/n*100
    win = np.mean([p > 0 for p in pnls])*100
    print(f"{name:38s} {stopped:7.0f}% {cnt['hard']/n*100:5.0f}% {cnt['trail']/n*100:6.0f}% "
          f"{cnt['tp']/n*100:5.0f}% {cnt['time']/n*100:5.0f}% {np.mean(pnls):+8.3f} {win:5.0f}%")
