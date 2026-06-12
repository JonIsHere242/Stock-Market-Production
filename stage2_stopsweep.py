#!/usr/bin/env python3
"""
Stage-2 STOP-PARAMETER SWEEP.
Precompute each trade's 5-min bar window once (anchored at the recorded fill), then
evaluate a grid of bracket configs over it to find what preserves the recorded edge.

  HARD stop in {none, 1.9, 3, 5} %      TRAIL in {off, 1.5, ATR-dynamic, 3.0} %
  TP fixed at +3.5% (live default); a second table varies TP at the live config.

Baseline (no hard, trail off, no TP) = pure time-exit, must ~= recorded mean (sanity).
Conservative intrabar tie-break: stop checked before TP. Splits/bad bars excluded.
"""
import os
import numpy as np, pandas as pd

FIVE_DIR = os.path.join('Data', 'IntradayData')
ENTRY_HHMM, EXIT_HHMM = 1000, 935

_c = {}
def load5(sym):
    if sym in _c: return _c[sym]
    f = os.path.join(FIVE_DIR, f'{sym}_5min.parquet')
    if not os.path.exists(f): _c[sym] = None; return None
    d = pd.read_parquet(f, columns=['Date','Open','High','Low','Close']).sort_values('Date')
    d['Date'] = pd.to_datetime(d['Date']); d['day'] = d['Date'].dt.normalize()
    g = d.groupby('day').agg(H=('High','max'), L=('Low','min'), C=('Close','last'))
    pc = g['C'].shift(1)
    tr = pd.concat([g['H']-g['L'], (g['H']-pc).abs(), (g['L']-pc).abs()], axis=1).max(axis=1)
    g['atr_pct'] = (tr.rolling(14, min_periods=5).mean() / g['C'] * 100).shift(1)
    _c[sym] = (d, g['atr_pct']); return _c[sym]

def bar_at(sess, hhmm):
    t = sess[sess['Date'].dt.time == pd.Timestamp(f'{hhmm//100:02d}:{hhmm%100:02d}').time()]
    return t.iloc[0] if not t.empty else None

def build_trade(sym, ed, xd, rec_entry):
    got = load5(sym)
    if got is None: return None
    d, atr_pct = got
    es = d[d['day'] == pd.Timestamp(ed).normalize()]
    if es.empty: return None
    hit = es[(es['Low'] <= rec_entry) & (es['High'] >= rec_entry)]
    eb = hit.iloc[0] if not hit.empty else bar_at(es, ENTRY_HHMM)
    if eb is None: eb = es.iloc[0]
    entry = float(rec_entry); ets = eb['Date']
    xs = d[d['day'] == pd.Timestamp(xd).normalize()]
    xb = bar_at(xs, EXIT_HHMM)
    xcut = xb['Date'] if xb is not None else (xs.iloc[0]['Date'] if not xs.empty else d['Date'].max())
    win = d[(d['Date'] >= ets) & (d['Date'] <= xcut)]
    if win.empty: return None
    O,H,L,C = win['Open'].values, win['High'].values, win['Low'].values, win['Close'].values
    # split guard: any open vs prior close outside [0.7,1.4]
    pcv = np.concatenate([[entry], C[:-1]])
    if np.any((O/pcv > 1.4) | (O/pcv < 0.7)): return 'SPLIT'
    ap = atr_pct.get(pd.Timestamp(ed).normalize(), np.nan)
    if not np.isfinite(ap): ap = 2.0
    return dict(entry=entry, O=O, H=H, L=L, atr=float(ap))

def eval_cfg(t, hard_pct, trail_mode, tp_pct):
    entry, O, H, L = t['entry'], t['O'], t['H'], t['L']
    hard = entry*(1-hard_pct/100) if hard_pct else -1e18
    if trail_mode == 'off':   tr = None
    elif trail_mode == '1.5': tr = 1.5
    elif trail_mode == '3.0': tr = 3.0
    else:                     tr = min(max(1.5, 0.75*t['atr']), 4.0)   # 'atr'
    tp = entry*(1+tp_pct/100) if tp_pct else 1e18
    hw = entry
    for i in range(len(O)):
        o,h,l = O[i],H[i],L[i]
        if h > hw: hw = h
        stop = hard
        if tr is not None:
            stop = max(stop, hw*(1-tr/100))
        if l <= stop:
            px = o if o <= stop else stop
            return (px/entry-1)*100, 'stop'
        if h >= tp:
            px = o if o >= tp else tp
            return (px/entry-1)*100, 'tp'
    return (O[-1]/entry-1)*100, 'time'

def main():
    tr = pd.read_parquet(os.path.join('Data','TradeHistory.parquet'))
    tr['EntryDate']=pd.to_datetime(tr['EntryDate']); tr['ExitDate']=pd.to_datetime(tr['ExitDate'])
    if tr['PnLPct'].abs().median()<1: tr['PnLPct']*=100
    trades=[]; rec=[]; nsplit=0; nskip=0
    for _,t in tr.iterrows():
        b=build_trade(str(t.Symbol).upper(), t.EntryDate, t.ExitDate, float(t.EntryPrice))
        if b=='SPLIT': nsplit+=1; continue
        if b is None: nskip+=1; continue
        trades.append(b); rec.append(float(t.PnLPct))
    rec=np.array(rec)
    print(f'usable trades: {len(trades)} | split-excluded {nsplit} | skipped {nskip}')
    print(f'RECORDED baseline mean PnL%: {rec.mean():+.3f} | sum {rec.sum():+.0f}pp | winrate {(rec>0).mean()*100:.1f}%\n')

    HARD=[None,1.9,3.0,5.0]; TRAIL=['off','1.5','atr','3.0']; TP=3.5
    print(f'=== HARD x TRAIL grid (TP +{TP}%, anchored at fills) : cell = mean PnL%/trade ===')
    hdr = 'hard\\trail | ' + ' | '.join(f'{x:>7}' for x in TRAIL)
    print(hdr); print('-'*len(hdr))
    for hp in HARD:
        cells=[]
        for tm in TRAIL:
            pnls=np.array([eval_cfg(t, hp, tm, TP)[0] for t in trades])
            cells.append(pnls.mean())
        lbl = 'none' if hp is None else f'{hp:.1f}%'
        print(f'{lbl:>9} | ' + ' | '.join(f'{c:+7.3f}' for c in cells))
    print('\n(reference: recorded no-stop baseline = '+f'{rec.mean():+.3f})')

    # detail at a few key configs
    print('\n=== detail at key configs ===')
    keys=[(None,'off',None,'NO STOPS (=time exit)'),
          (1.9,'atr',TP,'LIVE (hard 1.9 + ATR trail + TP3.5)'),
          (3.0,'atr',TP,'hard 3.0 + ATR trail + TP3.5'),
          (5.0,'off',TP,'hard 5.0 only + TP3.5'),
          (None,'atr',TP,'trail only (ATR) + TP3.5'),
          (5.0,'off',None,'hard 5.0 only, NO TP')]
    print(f'{"config":<38} | {"mean":>7} | {"sum":>7} | {"win%":>5} | {"stop%":>5} | {"tp%":>5} | {"time%":>5}')
    for hp,tm,tp,lbl in keys:
        res=[eval_cfg(t, hp, tm, tp) for t in trades]
        p=np.array([x[0] for x in res]); rs=[x[1] for x in res]
        sp=np.mean([r=='stop' for r in rs])*100; tpr=np.mean([r=='tp' for r in rs])*100; tm_=np.mean([r=='time' for r in rs])*100
        print(f'{lbl:<38} | {p.mean():+7.3f} | {p.sum():+7.0f} | {(p>0).mean()*100:4.0f}% | {sp:4.0f}% | {tpr:4.0f}% | {tm_:4.0f}%')

    # TP sensitivity at live stops
    print('\n=== TP sensitivity at LIVE stops (hard 1.9 + ATR trail) ===')
    for tp in [None,3.5,5.0,7.0,10.0]:
        p=np.array([eval_cfg(t,1.9,'atr',tp)[0] for t in trades])
        print(f'  TP {("off" if tp is None else str(tp)+"%"):>5}: mean {p.mean():+.3f} | sum {p.sum():+.0f} | win {(p>0).mean()*100:.0f}%')

if __name__=='__main__':
    main()
