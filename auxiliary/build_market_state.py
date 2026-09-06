"""Daily SPY market state, as of each session's CLOSE.

Every column here is knowable the evening the trigger table is written, so nothing
downstream can peek. Used to test whether the entry rule should depend on market
direction (e.g. lean to lower-beta names when SPY is strong).

Output: Data/_canbuyv2/market_state.parquet
"""
import os
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'Data', '_canbuyv2', 'market_state.parquet')


def main():
    spy = pd.read_parquet(os.path.join(REPO, 'Data', 'IndexesFull', 'SPY.parquet'))
    spy.index = pd.to_datetime(spy.index)
    spy = spy.sort_index()
    c = spy['Close']
    df = pd.DataFrame({
        'spy_ret1': c.pct_change(),
        'spy_ret5': c.pct_change(5),
        'spy_vs_ma20': c / c.rolling(20).mean() - 1.0,
        'spy_vs_ma50': c / c.rolling(50).mean() - 1.0,
        'spy_vol20': c.pct_change().rolling(20).std(),
    })
    df['spy_vol_z'] = ((df.spy_vol20 - df.spy_vol20.rolling(120).mean())
                       / df.spy_vol20.rolling(120).std())
    df = df.dropna(subset=['spy_ret1'])
    df.index.name = 'Date'
    df = df.reset_index()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print('wrote %s  %d rows  %s .. %s'
          % (OUT, len(df), df.Date.min().date(), df.Date.max().date()))
    print(df.tail(3).to_string(index=False))


if __name__ == '__main__':
    main()
