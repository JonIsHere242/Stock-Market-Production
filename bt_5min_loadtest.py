#!/usr/bin/env python3
"""
Backtrader 5-min load/scale probe.
Loads N of the Data/IntradayData/*_5min.parquet feeds into one Cerebro, runs a
trivial SMA strategy, and measures read time, cerebro run time, and RAM — then
extrapolates LINEARLY to the full 805-name lake. (Backtrader holds full line
buffers per feed+indicator, so real scaling is often SUPER-linear; this gives a
floor, and tells us whether all-805-in-one-cerebro is even viable.)

Usage:  python bt_5min_loadtest.py --n 10
"""
import os, glob, time, argparse
import pandas as pd
import backtrader as bt
import psutil

DATA_DIR = os.path.join('Data', 'IntradayData')
proc = psutil.Process()
def rss_mb(): return proc.memory_info().rss / 1e6


class PandasData(bt.feeds.PandasData):
    params = (('datetime', None), ('open', 'Open'), ('high', 'High'),
              ('low', 'Low'), ('close', 'Close'), ('volume', 'Volume'),
              ('openinterest', -1))


class SMAStrat(bt.Strategy):
    params = (('period', 20),)
    def __init__(self):
        # one indicator per feed = realistic per-asset compute/memory load
        self.smas = [bt.ind.SMA(d.close, period=self.p.period) for d in self.datas]
        self.ticks = 0
    def next(self):
        self.ticks += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10, help='number of feeds to load')
    ap.add_argument('--largest', action='store_true', default=True,
                    help='use the largest (full-depth) files to stress-test')
    args = ap.parse_args()

    fs = sorted(glob.glob(os.path.join(DATA_DIR, '*_5min.parquet')),
                key=os.path.getsize, reverse=True)[:args.n]
    if not fs:
        print('no 5min files found'); return

    rss0 = rss_mb(); t0 = time.time()

    cer = bt.Cerebro(stdstats=False)
    total_bars = 0
    for f in fs:
        df = pd.read_parquet(f, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        total_bars += len(df)
        cer.adddata(PandasData(dataname=df), name=os.path.basename(f).split('_')[0])

    load_t = time.time() - t0; rss_load = rss_mb()

    cer.addstrategy(SMAStrat)
    t1 = time.time()
    res = cer.run()
    run_t = time.time() - t1; rss_run = rss_mb()

    n = len(fs)
    print('=' * 64)
    print(f'  BACKTRADER 5-MIN LOAD PROBE  (n={n} feeds)')
    print('=' * 64)
    print(f'  total bars loaded : {total_bars:,}')
    print(f'  unified timestamps: {res[0].ticks:,}  (cerebro next() calls)')
    print(f'  read+adddata time : {load_t:6.2f} s')
    print(f'  cerebro run time  : {run_t:6.2f} s')
    print(f'  TOTAL time        : {load_t + run_t:6.2f} s')
    print(f'  RSS start/load/run: {rss0:.0f} / {rss_load:.0f} / {rss_run:.0f} MB '
          f'(delta {rss_run - rss0:.0f} MB)')
    print('-' * 64)
    per_t = (load_t + run_t) / n
    per_m = (rss_run - rss0) / n
    print(f'  per-feed          : {per_t:.2f} s, {per_m:.1f} MB')
    print(f'  LINEAR -> 805     : ~{per_t * 805 / 60:.1f} min, ~{per_m * 805 / 1000:.1f} GB RAM')
    print(f'  (backtrader scaling is usually worse than linear; treat as a floor)')
    print('=' * 64)


if __name__ == '__main__':
    main()
