"""Package-overhead benchmark: bulk-preloaded numpy feed (as the project's
EnhancedPandasData does), ATR per data, a 3-slot limit-order strategy.
Prints wall time, a trade-book fingerprint (for parity), and the profile."""
import cProfile, pstats, time, io, sys, array, hashlib, datetime
import numpy as np, pandas as pd
import backtrader as bt
from backtrader.utils.date import date2num

NDATA = int(sys.argv[1]) if len(sys.argv) > 1 else 300
NBARS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
mode = sys.argv[3] if len(sys.argv) > 3 else 'runonce'
prof = (sys.argv[4] if len(sys.argv) > 4 else 'prof') == 'prof'

class NpFeed(bt.feeds.PandasData):
    def start(self):
        super().start()
        df = self.p.dataname
        self._np = {c: df[c].to_numpy(dtype='float64') for c in ('open','high','low','close','volume','openinterest')}
        self._np_dt = np.array([date2num(t.to_pydatetime()) for t in df.index], dtype='float64')
        self._n = len(df)
    def preload(self):
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
    params = dict(period=14)
    def __init__(self):
        self.atr = {d: bt.indicators.ATR(d, period=self.p.period) for d in self.datas}
        self.book = []
    def notify_trade(self, t):
        if t.isclosed: self.book.append((t.data._name, round(t.pnlcomm, 6), len(self)))
    def next(self):
        held = sum(1 for d in self.datas if self.getposition(d).size > 0)
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size == 0 and held < 3 and d.close[0] > d.close[-1] * 1.03:
                self.buy(d, size=10, exectype=bt.Order.Limit, price=d.close[0]*0.995, valid=datetime.timedelta(days=3))
                held += 1
            elif pos.size and (d.close[0] < pos.price * 0.97 or d.close[0] > pos.price * 1.05):
                self.close(d)

def build():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range('2020-01-01', periods=NBARS)
    c = bt.Cerebro(stdstats=False)
    c.broker.set_cash(1e6)
    c.broker.setcommission(commission=0.001)
    for i in range(NDATA):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, NBARS)))
        df = pd.DataFrame({'open': close * (1 + rng.normal(0, .005, NBARS)), 'high': close * 1.01,
                           'low': close * 0.99, 'close': close, 'volume': 1e6, 'openinterest': 0}, index=idx)
        c.adddata(NpFeed(dataname=df), name=f'D{i}')
    c.addstrategy(St)
    return c

c = build()
t0 = time.perf_counter()
if prof:
    pr = cProfile.Profile(); pr.enable()
st = c.run(runonce=(mode == 'runonce'), preload=(mode != 'next_nopreload'))[0]
if prof: pr.disable()
dt = time.perf_counter() - t0
fp = hashlib.md5(repr(st.book).encode()).hexdigest()[:12]
print(f'{mode}: {NDATA} datas x {NBARS} bars = {dt:.2f}s  trades={len(st.book)} value={c.broker.getvalue():.2f} fp={fp}')
if prof:
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(30); print(s.getvalue()[:7000])
