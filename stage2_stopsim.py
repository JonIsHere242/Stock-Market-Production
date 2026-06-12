#!/usr/bin/env python3
"""
Stage-2 intraday STOP simulation (the "what actually happens" sim).
For every trade in the history, place the live broker's OCA bracket at the 10:00
entry and walk the 5-min bars through the hold window to see which leg fires:

  HARD STOP  : entry * (1 - 1.9%)                         (fixed)
  TRAIL      : highwater * (1 - trail%), trail% = min(max(1.5, 0.75*ATR%), 4.0)
  TAKE-PROFIT: entry * (1 + 3.5%)   (= limit + 2x the 1.75% risk; broker default)
  TIME EXIT  : ExitDate ~09:35 open (the pinned recorded convention) if none fire

Mirrors 9_SuperFastBroker.execute_batch(): HARD_STOP_PCT=1.9, _calculate_dynamic_trail,
TP = entry + 2*risk. ATR% is 14-day, derived from the 5-min RTH daily ranges (so it's
self-consistent with the lake, not the extended-hours-contaminated daily file).

Intrabar tie-break is CONSERVATIVE: if a bar touches both a stop and the TP, the stop
fills first (honest/pessimistic). Gaps through a level fill at the bar open.
Caveat: TP/trail are GTC+outsideRth live (can fill after-hours); this RTH sim only
fires them on RTH bars, so it slightly UNDER-counts after-hours stop/TP fills.

Out: analysis_output/stage2_stopsim_trades.parquet + console summary.
Usage: python stage2_stopsim.py [--trades Data/TradeHistory.parquet] [--entry 1000] [--exit 0935]
"""
import os, argparse
import numpy as np, pandas as pd

FIVE_DIR = os.path.join('Data', 'IntradayData')
OUT_DIR  = 'analysis_output'
os.makedirs(OUT_DIR, exist_ok=True)

HARD_STOP_PCT = 1.9
TRAIL_FLOOR   = 1.5
TRAIL_CAP     = 4.0
TP_PCT        = 3.5     # entry * 1.035

_cache = {}
def load5(sym):
    if sym in _cache: return _cache[sym]
    f = os.path.join(FIVE_DIR, f'{sym}_5min.parquet')
    if not os.path.exists(f):
        _cache[sym] = None; return None
    d = pd.read_parquet(f, columns=['Date','Open','High','Low','Close'])
    d['Date'] = pd.to_datetime(d['Date']); d = d.sort_values('Date')
    d['day'] = d['Date'].dt.normalize()
    # daily ranges -> ATR%(14) per day, as-of prior close (no lookahead)
    g = d.groupby('day').agg(H=('High','max'), L=('Low','min'), C=('Close','last'))
    pc = g['C'].shift(1)
    tr = pd.concat([g['H']-g['L'], (g['H']-pc).abs(), (g['L']-pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=5).mean()
    g['atr_pct'] = (atr / g['C'] * 100).shift(1)   # use yesterday's ATR% at today's entry
    _cache[sym] = (d, g['atr_pct'])
    return _cache[sym]

def bar_open_at(sess, hhmm):
    hh, mm = hhmm // 100, hhmm % 100
    t = sess[sess['Date'].dt.time == pd.Timestamp(f'{hh:02d}:{mm:02d}').time()]
    return t.iloc[0] if not t.empty else None

def simulate(sym, ed, xd, entry_hhmm, exit_hhmm, rec_entry=None, anchor='time'):
    got = load5(sym)
    if got is None: return None
    d, atr_pct = got
    e_sess = d[d['day'] == pd.Timestamp(ed).normalize()]
    if e_sess.empty: return None
    if anchor == 'recorded' and rec_entry and rec_entry > 0:
        # anchor at the actual fill: first bar (>=09:30) whose range brackets rec_entry
        hit = e_sess[(e_sess['Low'] <= rec_entry) & (e_sess['High'] >= rec_entry)]
        eb = hit.iloc[0] if not hit.empty else bar_open_at(e_sess, entry_hhmm)
        entry = float(rec_entry)
        entry_ts = eb['Date'] if eb is not None else e_sess.iloc[0]['Date']
    else:
        eb = bar_open_at(e_sess, entry_hhmm)
        entry = float(eb['Open']) if eb is not None else float(e_sess.iloc[0]['Open'])
        entry_ts = eb['Date'] if eb is not None else e_sess.iloc[0]['Date']

    ap = atr_pct.get(pd.Timestamp(ed).normalize(), np.nan)
    if not np.isfinite(ap): ap = 2.0
    trail_pct = min(max(TRAIL_FLOOR, 0.75 * ap), TRAIL_CAP)
    hard = entry * (1 - HARD_STOP_PCT/100)
    tp   = entry * (1 + TP_PCT/100)

    # hold window: from entry bar through ExitDate's exit bar (inclusive of that bar)
    x_day = pd.Timestamp(xd).normalize()
    x_sess = d[d['day'] == x_day]
    xb = bar_open_at(x_sess, exit_hhmm)
    exit_cut_ts = xb['Date'] if xb is not None else (x_sess.iloc[0]['Date'] if not x_sess.empty else d['Date'].max())
    win = d[(d['Date'] >= entry_ts) & (d['Date'] <= exit_cut_ts)]
    if win.empty: return None

    hw = entry; mae = 0.0; mfe = 0.0; prev_c = entry
    for _, b in win.iterrows():
        o, h, l, c = float(b['Open']), float(b['High']), float(b['Low']), float(b['Close'])
        # split / bad-bar guard: a >40% bar-to-bar gap is non-physical -> flag, don't fake a stop
        if prev_c > 0 and (o/prev_c > 1.40 or o/prev_c < 0.70):
            r = _res(sym, ed, xd, entry, prev_c, b['Date'], 'split_artifact', trail_pct, ap, hard, tp, mae, mfe)
            return r
        prev_c = c
        mae = min(mae, (l/entry - 1)*100); mfe = max(mfe, (h/entry - 1)*100)
        hw = max(hw, h)
        trail_level = hw * (1 - trail_pct/100)
        stop_level = max(hard, trail_level)            # tighter of the two fires first
        which_stop = 'hard' if hard >= trail_level else 'trail'
        # conservative: check stop before TP within a bar
        if l <= stop_level:
            px = o if o <= stop_level else stop_level   # gap-through fills at open
            return _res(sym, ed, xd, entry, px, b['Date'], which_stop, trail_pct, ap, hard, tp, mae, mfe)
        if h >= tp:
            px = o if o >= tp else tp
            return _res(sym, ed, xd, entry, px, b['Date'], 'take_profit', trail_pct, ap, hard, tp, mae, mfe)
    # no leg fired -> time exit at the exit-cut bar open
    px = float(win.iloc[-1]['Open'])
    return _res(sym, ed, xd, entry, px, win.iloc[-1]['Date'], 'time_exit', trail_pct, ap, hard, tp, mae, mfe)

def _res(sym, ed, xd, entry, exitpx, exitts, reason, trail_pct, atr_pct, hard, tp, mae, mfe):
    return dict(sym=sym, EntryDate=pd.Timestamp(ed), ExitDate=pd.Timestamp(xd),
                entry=entry, exit=exitpx, exit_ts=exitts, reason=reason,
                pnl_pct=(exitpx/entry - 1)*100, trail_pct=trail_pct, atr_pct=atr_pct,
                hard_lvl=hard, tp_lvl=tp, mae_pct=mae, mfe_pct=mfe,
                hold_bars=None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trades', default=os.path.join('Data', 'TradeHistory.parquet'))
    ap.add_argument('--entry', type=int, default=1000, help='entry time HHMM')
    ap.add_argument('--exit', type=int, default=935, help='time-exit HHMM on ExitDate')
    ap.add_argument('--anchor', choices=['time','recorded'], default='time',
                    help="entry price anchor: 'time'=10:00 bar open, 'recorded'=actual fill price")
    args = ap.parse_args()

    tr = pd.read_parquet(args.trades)
    tr['EntryDate'] = pd.to_datetime(tr['EntryDate']); tr['ExitDate'] = pd.to_datetime(tr['ExitDate'])
    if tr['PnLPct'].abs().median() < 1: tr['PnLPct'] *= 100
    print(f'trades: {len(tr)} | {tr.EntryDate.min().date()} -> {tr.ExitDate.max().date()}')

    out = []; rec_map = []
    nofile = noses = 0
    for _, t in tr.iterrows():
        r = simulate(str(t.Symbol).upper(), t.EntryDate, t.ExitDate, args.entry, args.exit,
                     rec_entry=float(t.EntryPrice), anchor=args.anchor)
        if r is None:
            if load5(str(t.Symbol).upper()) is None: nofile += 1
            else: noses += 1
            continue
        r['rec_pnl_pct'] = float(t.PnLPct)
        out.append(r)
    R = pd.DataFrame(out)
    R.to_parquet(os.path.join(OUT_DIR, 'stage2_stopsim_trades.parquet'), index=False)
    n_split = (R.reason == 'split_artifact').sum()
    n_split_names = R[R.reason == 'split_artifact']['sym'].nunique()
    print(f'simulated {len(R)} trades | skipped: no file {nofile}, no session {noses}')
    print(f'EXCLUDED {n_split} split/bad-bar artifact trades across {n_split_names} names (need split-adjust)\n')
    R = R[R.reason != 'split_artifact'].copy()
    if R.empty: return

    print('=== EXIT REASON BREAKDOWN (clean) ===')
    br = R.groupby('reason').agg(n=('pnl_pct','size'), mean_pnl=('pnl_pct','mean'),
                                 med_pnl=('pnl_pct','median'), winrate=('pnl_pct', lambda s:(s>0).mean()*100))
    br['pct_of_trades'] = br['n']/len(R)*100
    print(br[['n','pct_of_trades','mean_pnl','med_pnl','winrate']].round(2).to_string())

    print('\n=== PnL: STOP-SIM vs RECORDED (aggregate) ===')
    print(f'  sim mean PnL%   : {R.pnl_pct.mean():+.3f}  | sum {R.pnl_pct.sum():+.1f}pp | winrate {(R.pnl_pct>0).mean()*100:.1f}%')
    print(f'  recorded mean   : {R.rec_pnl_pct.mean():+.3f}  | sum {R.rec_pnl_pct.sum():+.1f}pp | winrate {(R.rec_pnl_pct>0).mean()*100:.1f}%')
    print(f'  delta (sim-rec) : {(R.pnl_pct-R.rec_pnl_pct).mean():+.3f}pp/trade')

    print('\n=== STOP PRESSURE (how close did trades come?) ===')
    print(f'  median MAE (worst intraday drawdown / trade): {R.mae_pct.median():.2f}%  | p10 {R.mae_pct.quantile(.1):.2f}%')
    print(f'  median MFE (best intraday gain / trade)     : {R.mfe_pct.median():.2f}%  | p90 {R.mfe_pct.quantile(.9):.2f}%')
    print(f'  trades that touched -1.9% hard level (MAE<=-1.9): {(R.mae_pct<=-1.9).mean()*100:.1f}%')
    stopped = R.reason.isin(['hard','trail'])
    tp_rate = (R.reason == 'take_profit').mean() * 100
    time_rate = (R.reason == 'time_exit').mean() * 100
    print(f'  STOPPED OUT: {stopped.mean()*100:.1f}% of trades | TP hit: {tp_rate:.1f}% | held to time: {time_rate:.1f}%')
    print(f'  avg trail% used: {R.trail_pct.mean():.2f}  (floor 1.5, cap 4.0) | avg ATR%: {R.atr_pct.mean():.2f}')

    print('\n=== sample 10 stopped trades ===')
    s = R[stopped].head(10)
    cols=['sym','EntryDate','reason','entry','exit','pnl_pct','rec_pnl_pct','trail_pct','mae_pct','mfe_pct']
    with pd.option_context('display.width',200):
        print(s[cols].to_string(index=False, formatters={c:(lambda v:f'{v:.2f}') for c in ['entry','exit','pnl_pct','rec_pnl_pct','trail_pct','mae_pct','mfe_pct']}))

if __name__ == '__main__':
    main()
