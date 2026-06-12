#!/usr/bin/env python3
"""
Stage-2 order sim (tracking check).
Replays each trade in the trade history on the 5-min lake and compares simulated
fills/PnL to the RECORDED fills/PnL — to confirm the 5-min data + replay harness
reproduces reality before we add intraday stop logic.

For each trade (Symbol, EntryDate, ExitDate, EntryPrice, ExitPrice, Quantity, PnLPct):
  - load Data/IntradayData/{Symbol}_5min.parquet
  - ENTRY candidates on EntryDate : session open (09:30) | 10:00 bar (live-broker rule)
  - EXIT  candidate  on ExitDate  : session close (15:55)
  - report how recorded prices line up with the 5-min bars, and how well simulated
    PnL% tracks recorded PnL%.

Usage: python stage2_ordersim.py [--trades Data/TradeHistory.parquet] [--report N]
"""
import os, argparse
import numpy as np, pandas as pd

FIVE_DIR = os.path.join('Data', 'IntradayData')

def load_5min(sym):
    f = os.path.join(FIVE_DIR, f'{sym}_5min.parquet')
    if not os.path.exists(f):
        return None
    d = pd.read_parquet(f, columns=['Date','Open','High','Low','Close','Volume'])
    d['Date'] = pd.to_datetime(d['Date'])
    d['day'] = d['Date'].dt.normalize()
    return d

def session(d, day):
    s = d[d['day'] == pd.Timestamp(day).normalize()]
    return s if not s.empty else None

def bar_at(s, hh, mm):
    """Open of the bar starting at hh:mm (live fill proxy); fallback to first bar."""
    t = s[s['Date'].dt.time == pd.Timestamp(f'{hh:02d}:{mm:02d}').time()]
    return float(t.iloc[0]['Open']) if not t.empty else float(s.iloc[0]['Open'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trades', default=os.path.join('Data', 'TradeHistory.parquet'))
    ap.add_argument('--report', type=int, default=12)
    args = ap.parse_args()

    tr = pd.read_parquet(args.trades)
    tr['EntryDate'] = pd.to_datetime(tr['EntryDate'])
    tr['ExitDate']  = pd.to_datetime(tr['ExitDate'])
    print(f'trades: {len(tr)} | {tr.EntryDate.min().date()} -> {tr.ExitDate.max().date()} | file {args.trades}')

    rows = []
    skip_nofile = skip_nosession = 0
    cache = {}
    for _, t in tr.iterrows():
        sym = str(t['Symbol']).upper()
        if sym not in cache:
            cache[sym] = load_5min(sym)
        d = cache[sym]
        if d is None:
            skip_nofile += 1; continue
        se = session(d, t['EntryDate']); sx = session(d, t['ExitDate'])
        if se is None or sx is None:
            skip_nosession += 1; continue
        e_open = float(se.iloc[0]['Open'])
        e_1000 = bar_at(se, 10, 0)
        x_close = float(sx.iloc[-1]['Close'])
        rec_pct = float(t['PnLPct'])
        # recorded PnLPct may be in % (e.g. 3.2) or fraction (0.032) — detect
        rows.append(dict(
            sym=sym, ed=t['EntryDate'].date(), xd=t['ExitDate'].date(),
            rec_entry=float(t['EntryPrice']), rec_exit=float(t['ExitPrice']),
            e_open=e_open, e_1000=e_1000, x_close=x_close, rec_pct=rec_pct,
            sim_pct_open=(x_close/e_open - 1)*100,
            sim_pct_1000=(x_close/e_1000 - 1)*100,
        ))
    r = pd.DataFrame(rows)
    print(f'priced {len(r)} trades | skipped: no 5min file {skip_nofile}, no session in range {skip_nosession}')
    if r.empty:
        print('nothing priced'); return

    # normalize recorded pct units to % if it looks like a fraction
    if r['rec_pct'].abs().median() < 1:
        r['rec_pct'] = r['rec_pct'] * 100
        print('(recorded PnLPct looked like a fraction -> scaled x100 to %)')

    print('\n=== how do RECORDED fills line up with the 5-min bars? ===')
    de_open = (r['rec_entry'] - r['e_open']).abs() / r['e_open'] * 100
    de_1000 = (r['rec_entry'] - r['e_1000']).abs() / r['e_1000'] * 100
    dx_close = (r['rec_exit'] - r['x_close']).abs() / r['x_close'] * 100
    print(f'  recorded ENTRY vs 5min OPEN(09:30): median |diff| {de_open.median():.3f}%  | within 0.1%: {(de_open<=0.1).mean()*100:.0f}%')
    print(f'  recorded ENTRY vs 5min 10:00 bar  : median |diff| {de_1000.median():.3f}%  | within 0.1%: {(de_1000<=0.1).mean()*100:.0f}%')
    print(f'  recorded EXIT  vs 5min CLOSE(15:55): median |diff| {dx_close.median():.3f}%  | within 0.1%: {(dx_close<=0.1).mean()*100:.0f}%')

    print('\n=== does SIMULATED PnL% track RECORDED PnL%? ===')
    for col, lbl in [('sim_pct_open','open->close'), ('sim_pct_1000','10:00->close')]:
        err = (r[col] - r['rec_pct'])
        corr = r[col].corr(r['rec_pct'])
        print(f'  {lbl:>12}: corr {corr:.4f} | mean err {err.mean():+.3f}pp | median|err| {err.abs().median():.3f}pp '
              f'| within 0.5pp {(err.abs()<=0.5).mean()*100:.0f}%')
    print(f'  recorded mean PnL% {r.rec_pct.mean():+.3f} | sim(open) {r.sim_pct_open.mean():+.3f} | sim(10:00) {r.sim_pct_1000.mean():+.3f}')

    # best-convention sample
    r['err_open'] = (r.sim_pct_open - r.rec_pct).abs()
    print(f'\n=== sample {args.report} trades (open->close convention) ===')
    cols = ['sym','ed','xd','rec_entry','e_open','rec_exit','x_close','rec_pct','sim_pct_open']
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(r[cols].head(args.report).to_string(index=False,
              formatters={c:(lambda v: f'{v:.2f}') for c in ['rec_entry','e_open','rec_exit','x_close','rec_pct','sim_pct_open']}))

if __name__ == '__main__':
    main()
