"""Second parity rig: observers on, custom strategy lines, next-only indicator
(once_via_next path), mixed-timeframe coupling, HeikinAshi filter, writer."""
import sys, io, hashlib, datetime, time
import numpy as np, pandas as pd
import backtrader as bt

NDATA, NBARS = 12, 400

class NextOnly(bt.Indicator):
    lines = ('v',)
    params = (('period', 10),)
    def __init__(self):
        self.addminperiod(self.p.period)
    def next(self):
        self.lines.v[0] = max(self.data.get(size=self.p.period)) - min(self.data.get(size=self.p.period))

class St(bt.Strategy):
    lines = ('sig',)
    def __init__(self):
        self.book = []
        self.d_daily = self.datas[:NDATA]
        self.d_week = self.datas[NDATA:]
        self.no = {d: NextOnly(d.close, period=10) for d in self.d_daily}
        self.wsma = {dw: bt.indicators.SMA(dw.close, period=4) for dw in self.d_week}
        self.coup = {d: self.wsma[dw]() for d, dw in zip(self.d_daily, self.d_week)}  # LinesCoupler
        self.cmp = {d: d.close > self.coup[d] for d in self.d_daily}  # LinesOperation on coupled line
        self.hi = {d: bt.indicators.Highest(d.high, period=20) for d in self.d_daily}
    def notify_trade(self, t):
        if t.isclosed: self.book.append(('t', t.data._name, round(t.pnlcomm, 6), len(self)))
    def notify_order(self, o):
        if o.status in (o.Completed, o.Expired, o.Canceled, o.Margin, o.Rejected):
            self.book.append(('o', len(self), o.status, round(o.executed.price, 6), o.executed.size))
    def next(self):
        self.lines.sig[0] = float(sum(self.cmp[d][0] for d in self.d_daily))
        self.book.append(('v', round(self.broker.getvalue(), 4), self.lines.sig[0], round(self.datas[0].close[0], 6),
                          round(float(self.coup[self.d_daily[0]][0]), 6) if len(self.coup[self.d_daily[0]]) else None))
        for d in self.d_daily:
            pos = self.getposition(d)
            if pos.size == 0 and self.cmp[d][0] and d.close[0] >= self.hi[d][0] * 0.985 and self.no[d][0] > 0:
                self.buy(d, size=7, exectype=bt.Order.Stop, price=d.close[0] * 1.002, valid=datetime.timedelta(days=4))
            elif pos.size and d.close[0] < self.coup[d][0]:
                self.sell(d, size=pos.size, exectype=bt.Order.Market)

def build(runonce=True, exactbars=0, ha=False, writer=False):
    rng = np.random.default_rng(7)
    idx = pd.bdate_range('2021-01-01', periods=NBARS)
    c = bt.Cerebro(stdstats=True, runonce=runonce, exactbars=exactbars)
    c.broker.set_cash(2e5); c.broker.setcommission(commission=0.0005)
    c.addobserver(bt.observers.DrawDown); c.addanalyzer(bt.analyzers.TradeAnalyzer); c.addanalyzer(bt.analyzers.DrawDown)
    if writer: c.addwriter(bt.WriterFile, csv=True, out=io.StringIO())
    feeds = []
    for i in range(NDATA):
        close = 50 * np.exp(np.cumsum(rng.normal(0, 0.015, NBARS)))
        df = pd.DataFrame({'open': close * (1 + rng.normal(0, .004, NBARS)), 'high': close * 1.01,
                           'low': close * 0.99, 'close': close, 'volume': 5e5, 'openinterest': 0}, index=idx)
        d = bt.feeds.PandasData(dataname=df)
        if ha: d.addfilter(bt.filters.HeikinAshi)
        c.adddata(d, name=f'D{i}'); feeds.append(d)
    for i, d in enumerate(feeds):
        c.resampledata(d, timeframe=bt.TimeFrame.Weeks, name=f'W{i}')
    c.addstrategy(St)
    return c

cases = [('runonce_obs', {}), ('next_obs', dict(runonce=False)), ('exactbars1_obs', dict(exactbars=1)),
         ('heikin_next', dict(runonce=False, ha=True)), ('writer_runonce', dict(writer=True))]
for name, kw in cases:
    t0 = time.perf_counter()
    c = build(**kw); st = c.run()[0]
    ta = st.analyzers.tradeanalyzer.get_analysis(); dd = st.analyzers.drawdown.get_analysis()
    obs = [round(x, 6) for x in list(st.observers.drawdown.lines.drawdown.array)[:len(st)]]
    fp = hashlib.md5(repr((st.book, dict(ta), dict(dd), obs)).encode()).hexdigest()[:16]
    print(f'{name:16s} {time.perf_counter()-t0:6.2f}s events={len(st.book):5d} bars={len(st)} value={c.broker.getvalue():.4f} fp={fp}')
