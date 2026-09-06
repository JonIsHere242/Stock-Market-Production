"""Rolling 60-day beta vs SPY for every ticker in Data/RFpredictions.

Written for the limit-entry experiment: the entry offset is scaled by the name's own
beta and volatility so that a dip of comparable rarity is asked for on every name,
rather than a flat percentage that never fills on a quiet stock and always fills on a
noisy one. Data/RollingBeta.parquet exists but stops at 2026-05-29, which does not
cover the evaluation window.

Also emits IDIOSYNCRATIC volatility: the rolling std of the residual after the market
component is removed (r - beta * spy_ret). The literature review flagged that
`beta * vol_20d` double-counts market exposure, since 20-day vol already contains the
systematic part -- and that a market-wide drop is the dip least worth fading. idio_vol
is the cleanly motivated alternative depth basis.

Output: Data/_canbuyv2/beta_lookup.parquet  (ticker, Date, beta, idio_vol, tot_vol)
"""
import os, glob, time
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(REPO, 'Data', 'RFpredictions')
OUT  = os.path.join(REPO, 'Data', '_canbuyv2', 'beta_lookup.parquet')
WIN  = 60

def main():
    t0 = time.time()
    spy = pd.read_parquet(os.path.join(REPO, 'Data', 'IndexesFull', 'SPY.parquet'))
    spy.index = pd.to_datetime(spy.index)
    spy_ret = spy['Close'].sort_index().pct_change()

    files = sorted(glob.glob(os.path.join(PRED, '*.parquet')))
    print(f'{len(files)} tickers', flush=True)
    out = []
    for n, fp in enumerate(files, 1):
        tk = os.path.splitext(os.path.basename(fp))[0]
        try:
            px = pd.read_parquet(fp, columns=['Date', 'Close'])
        except Exception:
            continue
        px['Date'] = pd.to_datetime(px['Date'])
        px = px.sort_values('Date')
        r = px['Close'].pct_change()
        sr = px['Date'].map(spy_ret)
        cov = r.rolling(WIN).cov(sr)
        var = sr.rolling(WIN).var()
        beta = (cov / var).replace([np.inf, -np.inf], np.nan)
        resid = r - beta * sr
        idio = resid.rolling(20).std()
        tot = r.rolling(20).std()
        ok = beta.notna() & idio.notna()
        if not ok.any():
            continue
        out.append(pd.DataFrame({'ticker': tk,
                                 'Date': px['Date'].values[ok.values],
                                 'beta': beta.values[ok.values],
                                 'idio_vol': idio.values[ok.values],
                                 'tot_vol': tot.values[ok.values]}))
        if n % 500 == 0:
            print(f'  {n} {time.time()-t0:.0f}s', flush=True)
    df = pd.concat(out, ignore_index=True)
    df['beta'] = df['beta'].clip(0.2, 3.0).astype('float32')
    df['idio_vol'] = df['idio_vol'].astype('float32')
    df['tot_vol'] = df['tot_vol'].astype('float32')
    print('idio/tot ratio: median %.3f' % (df.idio_vol / df.tot_vol).median())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f'wrote {OUT}  {len(df):,} rows  {df.ticker.nunique():,} tickers  {time.time()-t0:.0f}s')
    print(df[['beta','idio_vol','tot_vol']].describe())

if __name__ == '__main__':
    main()
