#!/usr/bin/env python3
"""Reconcile 5-min bars against the daily bars: aggregate each RTH session of the
5-min file (first Open / max High / min Low / last Close / sum Volume) and compare
to Data/PriceData/{T}.parquet for the same dates. Both are IBKR consolidated TRADES,
so OHLC should match to the penny; volume can differ slightly (closing-auction print)."""
import sys, os, glob
import numpy as np, pandas as pd

DAILY_DIR = os.path.join('Data', 'PriceData')
FIVE_DIR  = os.path.join('Data', 'IntradayData')

def agg_daily(t):
    f = os.path.join(FIVE_DIR, f'{t}_5min.parquet')
    if not os.path.exists(f): return None
    d = pd.read_parquet(f, columns=['Date','Open','High','Low','Close','Volume'])
    d['Date'] = pd.to_datetime(d['Date']); d = d.sort_values('Date')
    d['day'] = d['Date'].dt.normalize()
    g = d.groupby('day')
    return g.agg(Open=('Open','first'), High=('High','max'), Low=('Low','min'),
                 Close=('Close','last'), Vol5=('Volume','sum'), nbars=('Close','size'))

def daily(t):
    f = os.path.join(DAILY_DIR, f'{t}.parquet')
    if not os.path.exists(f): return None
    d = pd.read_parquet(f); d['Date'] = pd.to_datetime(d['Date'])
    return d.set_index('Date')

def cmp_one(t):
    a, b = agg_daily(t), daily(t)
    if a is None or b is None:
        print(f'{t}: missing ({"5min" if a is None else "daily"})'); return
    j = a.join(b[['Open','High','Low','Close','Volume']], how='inner', lsuffix='_5', rsuffix='_d')
    if j.empty:
        print(f'{t}: no overlapping dates'); return
    print(f'\n===== {t} =====  overlap days: {len(j)}  ({j.index.min().date()} -> {j.index.max().date()})')
    print(f'  avg 5-min bars/day: {j["nbars"].mean():.1f}  (RTH 09:30-15:55 = 78 expected)')
    for fld in ['Open','High','Low','Close']:
        diff = (j[f'{fld}_5'] - j[f'{fld}_d']).abs()
        pct  = (diff / j[f'{fld}_d'].abs()).replace([np.inf,-np.inf], np.nan)
        within = (pct <= 0.001).mean() * 100   # within 0.1%
        print(f'  {fld:<5}: match<=0.1% {within:5.1f}% of days | mean|d| ${diff.mean():.4f} | max|d| ${diff.max():.3f}')
    vr = (j['Vol5'] / j['Volume']).replace([np.inf,-np.inf], np.nan)
    print(f'  Volume: 5min-sum / daily ratio  median {vr.median():.3f}  (q10 {vr.quantile(.1):.3f}, q90 {vr.quantile(.9):.3f})')
    # one detailed recent day
    row = j.iloc[-1]
    print(f'  --- sample {j.index[-1].date()} ---')
    print(f'      5min agg: O {row.Open_5:.2f}  H {row.High_5:.2f}  L {row.Low_5:.2f}  C {row.Close_5:.2f}  V {row.Vol5:,.0f}')
    print(f'      daily   : O {row.Open_d:.2f}  H {row.High_d:.2f}  L {row.Low_d:.2f}  C {row.Close_d:.2f}  V {row.Volume:,.0f}')

if __name__ == '__main__':
    tickers = sys.argv[1:] or ['AAPL','MSFT','NVDA']
    for t in tickers:
        cmp_one(t.upper())
