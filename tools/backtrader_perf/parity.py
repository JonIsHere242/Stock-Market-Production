"""Parity matrix: run several cerebro configurations on the same synthetic
universe and print a fingerprint per configuration. Run before and after any
package edit; every line must match."""
import sys, array, hashlib, time, datetime
import numpy as np, pandas as pd
import backtrader as bt
from backtrader.utils.date import date2num

NDATA, NBARS = 40, 600

class NpFeed(bt.feeds.PandasData):
    def start(self):
        super().start()
        df = self.p.dataname
        self._np = {c: df[c].to_numpy(dtype='float64') for c in ('open','high','low','close','volume','openinterest')}
        self._np_dt = np.array([date2num(t.to_pydatetime()) for t in df.index], dtype='float64')
        self._n = len(df)
    def preload(self):
        if self._filters or self._ffilters or any(l.mode != 0 for l in self.lines):
            return super().preload()
        for alias in self.getlinealiases():
            src = self._np_dt if alias == 'datetime' else self._np[alias]
            buf = array.array('d'); buf.frombytes(np.ascontiguousarray(src).tobytes())
            getattr(self.lines, alias).array = buf
        self.home(); self._idx = self._n - 1
    def _load(self):
        self._idx += 1
        if self._idx >= self._n: return False
        i = self._idx
        for alias in self.getlinealiases():
            src = self._np_dt if alias == 'datetime' else self._np[alias]
            getattr(self.lines, alias)[0] = float(src[i])
        return True

class St(bt.Strategy):
    params = dict(period=14, stops=True)
    def __init__(self):
        self.atr = {d: bt.indicators.ATR(d, period=self.p.period) for d in self.datas}
        self.sma = {d: bt.indicators.SMA(d.close, period=20) for d in self.datas}
        self.rsi = {d: bt.indicators.RSI(d, period=14) for d in self.datas}
        self.x = {d: bt.indicators.CrossOver(d.close, self.sma[d]) for d in self.datas}
        self.book, self.vals = [], []
    def notify_trade(self, t):
        if t.isclosed: self.book.append((t.data._name, round(t.pnlcomm, 6), len(self)))
    def notify_order(self, o):
        if o.status in (o.Completed, o.Expired, o.Canceled, o.Margin, o.Rejected):
            self.book.append(('o', len(self), o.status, round(o.executed.price, 6), o.executed.size))
    def next(self):
        self.vals.append(round(self.broker.getvalue(), 4))
        held = sum(1 for d in self.datas if self.getposition(d).size > 0)
        for d in self.datas:
            pos = self.getposition(d)
            a = self.atr[d][0]
            if pos.size == 0 and held < 5 and (d.close[0] > d.close[-1] * 1.03 or self.x[d][0] > 0) and self.rsi[d][0] < 70:
                if self.p.stops:
                    self.buy_bracket(data=d, size=10, exectype=bt.Order.Limit, price=d.close[0]*0.995, valid=datetime.timedelta(days=3),
                                     stopprice=d.close[0] - 2*a, limitprice=d.close[0] + 3*a)
                else:
                    self.buy(d, size=10, exectype=bt.Order.Limit, price=d.close[0]*0.995, valid=datetime.timedelta(days=3))
                held += 1
            elif pos.size and not self.p.stops and (d.close[0] < pos.price * 0.97 or d.close[0] > pos.price * 1.05):
                self.close(d)
            elif pos.size and len(self) % 37 == 0:
                self.sell(d, size=5)

def build(feedcls, resample=None, replay=None, stops=True, **cer):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range('2020-01-01', periods=NBARS)
    c = bt.Cerebro(stdstats=False, **cer)
    c.broker.set_cash(1e6); c.broker.setcommission(commission=0.001)
    c.addanalyzer(bt.analyzers.TradeAnalyzer); c.addanalyzer(bt.analyzers.SharpeRatio)
    for i in range(NDATA):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, NBARS)))
        df = pd.DataFrame({'open': close * (1 + rng.normal(0, .005, NBARS)), 'high': close * 1.012,
                           'low': close * 0.988, 'close': close, 'volume': 1e6, 'openinterest': 0}, index=idx)
        d = feedcls(dataname=df)
        if resample: c.resampledata(d, timeframe=bt.TimeFrame.Weeks, name=f'D{i}')
        elif replay: c.replaydata(d, timeframe=bt.TimeFrame.Weeks, name=f'D{i}')
        else: c.adddata(d, name=f'D{i}')
    c.addstrategy(St, stops=stops)
    return c

cases = [
    ('runonce_np', dict(feedcls=NpFeed)),
    ('runonce_pandas', dict(feedcls=bt.feeds.PandasData)),
    ('next_preload', dict(feedcls=NpFeed, runonce=False)),
    ('next_nopreload', dict(feedcls=NpFeed, runonce=False, preload=False)),
    ('exactbars1', dict(feedcls=NpFeed, exactbars=1)),
    ('exactbars-1', dict(feedcls=NpFeed, exactbars=-1)),
    ('resample_w', dict(feedcls=NpFeed, resample=True)),
    ('replay_w', dict(feedcls=NpFeed, replay=True)),
    ('nostops_coo', dict(feedcls=NpFeed, stops=False, cheat_on_open=True)),
]
only = sys.argv[1:] 
for name, kw in cases:
    if only and name not in only: continue
    t0 = time.perf_counter()
    c = build(**kw); st = c.run()[0]
    ta = st.analyzers.tradeanalyzer.get_analysis()
    fp = hashlib.md5(repr((st.book, st.vals, dict(ta))).encode()).hexdigest()[:16]
    print(f'{name:16s} {time.perf_counter()-t0:6.2f}s trades={len(st.book):5d} bars={len(st.vals):4d} value={c.broker.getvalue():.4f} fp={fp}')
