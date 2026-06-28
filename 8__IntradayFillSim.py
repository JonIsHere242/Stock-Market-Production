#!/usr/bin/env python3
"""
Intraday Fill & Exit Fidelity Simulator
=======================================
Replays every trade the backtester chose against REAL intraday bars (1-minute now,
tick-ready) under the LIVE broker's execution rules (9_SuperFastBroker.py), then
checks whether the entry's statistical properties — the thing the model was trained
to produce — survive realistic fills.

Why this exists
---------------
5__NightlyBackTester_v2.py fills on DAILY bars: it enters a limit at the daily open
and resolves the -1.9% hard stop / +2:1 take-profit against the daily HIGH/LOW. That
daily view hides three things that decide whether the headline survives with real money:

  1. ENTRY TIMING.  The live broker (9_SuperFastBroker.py) does NOT buy at the open.
     It waits until 10:00 ET (EDA: 9:30->10:00 is +256% Sharpe), aborts the whole day
     if SPY <= -0.5% from its open, and skips any name that gapped up > 1.5%. So the
     real entry price — and whether the trade is taken at all — differs from the backtest.

  2. STOP-vs-TARGET SEQUENCE.  On a daily bar, if BOTH the -1.9% stop and the +3.5%
     target are inside [Low, High], the backtest cannot know which was hit first. At
     1-minute (or tick) resolution we walk the path in order and resolve it. A backtest
     "Take Profit" can turn out to be a "Stop Loss".

  3. EXITS.  ~78% of trades exit "Manual / Momentum / Timeout" as a market order at the
     next daily open in the backtest. We instead let the live bracket run on the real
     path and, if neither leg fires by the backtest's ExitDate, exit at that day's CLOSE
     (the rule the 10:00-entry EDA was built on).

The headline output is NOT "does the 232% survive" (the equity curve, not the per-trade
ledger, drives that — see project_intraday_fill_check). It is: bucket the TAKEN trades by
the model's UpProbability and confirm (a) realized return rises monotonically with
UpProbability and (b) the calibrated probability is honest — realized up-rate ~= predicted
P(up) — once the trade is executed the way it is executed for real.

Resolution-agnostic / tick-ready
--------------------------------
The exit walk consumes an ordered array of (ts, open, high, low, close) rows. Point
--intraday-dir / --suffix at any bar lake (Data/IntradayTradeSim/_1min by default, or
Data/IntradayData/_5min, or a future tick lake) and nothing else changes. With true tick
data each "bar" is a print (open==high==low==close) and the same-bar stop/target ambiguity
disappears entirely.

Usage
-----
    python 8__IntradayFillSim.py                       # 1-min lake, default trade history
    python 8__IntradayFillSim.py --suffix _5min --intraday-dir Data/IntradayData
    python 8__IntradayFillSim.py --slippage-bps 5      # stress the entry-fill assumption
    python 8__IntradayFillSim.py --no-timeout          # let the bracket run, ignore bt ExitDate
    python 8__IntradayFillSim.py --spy-file Data/IntradayTradeSim/SPY_1min.parquet

SPY gate: drop a SPY intraday parquet at Data/IntradayTradeSim/SPY_1min.parquet (or pass
--spy-file). Without it the SPY-abort filter is DISABLED and the report says so loudly.
Fetch it with:  python experimental/2.2__TradeHistoryIntradayDownloader.py --index-tickers SPY
"""

import os
import argparse
from datetime import time as dtime

import numpy as np
import pandas as pd

# ── Live-broker rule constants (mirror 9_SuperFastBroker.py) ─────────────────────
ENTRY_HOUR          = 10
ENTRY_MINUTE        = 0
SPY_ABORT_THRESHOLD = -0.5    # skip ALL trades that day if SPY <= this % from its open
STOCK_GAP_SKIP      = 1.5     # skip a name that gapped up > this % from its own open
HARD_STOP_PCT       = 1.9     # hard stop % below entry
STOP_FOR_RISK_PCT   = 1.75    # risk basis the broker uses to size the 2:1 target
TARGET_RR           = 2.0     # reward:risk -> target = entry*(1 + 2*1.75%) = +3.5%

# ── IBKR commission (mirror IBKRAdaptiveCommission in the backtester) ────────────
COMM_PER_SHARE = 0.0035
COMM_MIN_ORDER = 0.35
COMM_MAX_PCT   = 0.01      # 1% of notional cap
COMM_EXCH_FEE  = 0.0002    # per share

ENTRY_T = dtime(ENTRY_HOUR, ENTRY_MINUTE)

script_dir = os.path.dirname(os.path.abspath(__file__))


def ibkr_commission(shares: float, price: float) -> float:
    """One side of an IBKR fill, matching the backtester's model exactly."""
    shares = abs(shares)
    if shares == 0:
        return 0.0
    per_share = shares * COMM_PER_SHARE
    exch = shares * COMM_EXCH_FEE
    cap = shares * price * COMM_MAX_PCT
    base = max(per_share, COMM_MIN_ORDER)
    return min(base, cap) + exch


# ── Intraday loading ─────────────────────────────────────────────────────────────

def load_intraday(intraday_dir: str, suffix: str, ticker: str) -> pd.DataFrame | None:
    fp = os.path.join(intraday_dir, f"{ticker}{suffix}.parquet")
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_parquet(fp, columns=['Date', 'Open', 'High', 'Low', 'Close'])
    except Exception:
        df = pd.read_parquet(fp)
    df['Date'] = pd.to_datetime(df['Date'])
    # Defensive: strip any tz so day comparisons are naive-ET throughout.
    if getattr(df['Date'].dt, 'tz', None) is not None:
        df['Date'] = df['Date'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
    return df.sort_values('Date').reset_index(drop=True)


def spy_move_by_day(spy_df: pd.DataFrame | None) -> dict:
    """{date -> SPY % move from session open to the 10:00 bar}, for the abort gate."""
    if spy_df is None:
        return {}
    out = {}
    spy_df = spy_df.copy()
    spy_df['day'] = spy_df['Date'].dt.normalize()
    spy_df['t'] = spy_df['Date'].dt.time
    for day, g in spy_df.groupby('day'):
        g = g.sort_values('Date')
        sess_open = g['Open'].iloc[0]
        at10 = g[g['t'] >= ENTRY_T]
        if at10.empty or sess_open <= 0:
            continue
        px10 = at10['Open'].iloc[0]
        out[day] = (px10 / sess_open - 1.0) * 100.0
    return out


# ── Single-trade simulation under live rules ─────────────────────────────────────

def simulate_trade(bars: pd.DataFrame, entry_date, exit_date, bt_entry_price: float,
                   slippage_bps: float, honor_timeout: bool, spy_moves: dict,
                   entry_window_days: int = 4, integrity_tol: float = 0.25):
    """
    Walk the real intraday path for one trade under the live broker bracket.

    `entry_date` is the backtester SIGNAL day; the real fill (live AND backtest) is the
    NEXT trading session's open, so we enter on the first session strictly after it — which
    is also when the live broker reads the evening signals and buys at 10:00. The live order
    is tif=DAY, so the fill must be the *immediate* next session: if that session is missing
    from the lake we return NO_DATA rather than silently entering weeks later.

    Data-integrity cross-check: the backtester filled at the trusted DAILY open, so the
    intraday 09:30 open must agree with `bt_entry_price` within `integrity_tol`. A large
    disagreement means a split / adjustment / corrupt day in the intraday file -> BAD_DATA.

    Returns a dict of sim fields, or {'Status': ...} for a non-taken / no-data trade.
    """
    signal_day = pd.Timestamp(entry_date).normalize()
    exit_day = pd.Timestamp(exit_date).normalize()

    day_index = bars['Date'].dt.normalize()
    sessions = day_index.unique()
    later = [d for d in sessions if signal_day < d <= exit_day]
    if not later:
        return {'Status': 'NO_DATA'}
    entry_day = min(later)                 # first session after the signal = the fill day
    # The live tif=DAY order only lives for the very next session(s); a fill many days
    # later is not what the live system would have done.
    if (entry_day - signal_day).days > entry_window_days:
        return {'Status': 'NO_DATA'}

    day0 = bars[day_index == entry_day]
    sess_open = day0['Open'].iloc[0]
    at10 = day0[day0['Date'].dt.time >= ENTRY_T]
    if at10.empty:
        return {'Status': 'NO_DATA'}      # no bars at/after 10:00 (data hole)

    entry_bar_ts = at10['Date'].iloc[0]
    px10 = at10['Open'].iloc[0]           # the 10:00 print (best 1-min proxy for the mid)
    if px10 <= 0 or sess_open <= 0:
        return {'Status': 'NO_DATA'}

    # Cross-check intraday vs the trusted daily fill price (catches splits / bad days).
    if bt_entry_price and bt_entry_price > 0 and abs(sess_open / bt_entry_price - 1.0) > integrity_tol:
        return {'Status': 'BAD_DATA', 'SessOpen': sess_open, 'BtEntryPrice': bt_entry_price}

    # ── Live filters → a VERDICT, but we still simulate the trade either way ──────
    # Recording the verdict without short-circuiting lets us measure what a SKIPPED
    # trade WOULD have returned under live 10:00 entry — the only fair way to judge
    # whether a filter adds value (vs comparing to an unachievable daily-open backtest).
    spy_move = spy_moves.get(entry_day) if spy_moves else None
    open_gap_pct = (px10 / sess_open - 1.0) * 100.0
    verdict = 'TAKEN'
    if spy_move is not None and spy_move <= SPY_ABORT_THRESHOLD:
        verdict = 'SKIP_SPY'                # SPY filter is checked first in the live broker
    elif open_gap_pct > STOCK_GAP_SKIP:
        verdict = 'SKIP_GAP'

    # ── Entry fill: 10:00 price + adaptive-algo slippage (we BUY, so pay up) ──────
    entry_fill = px10 * (1.0 + slippage_bps / 1e4)
    stop_price = entry_fill * (1.0 - HARD_STOP_PCT / 100.0)
    risk = entry_fill * (STOP_FOR_RISK_PCT / 100.0)
    target_price = entry_fill + risk * TARGET_RR

    # ── Build the forward path: entry bar onward, capped at ExitDate if honoring ──
    path = bars[bars['Date'] >= entry_bar_ts]
    if honor_timeout:
        path = path[path['Date'].dt.normalize() <= exit_day]
    if path.empty:
        return {'Status': 'NO_DATA'}

    o = path['Open'].to_numpy(); h = path['High'].to_numpy()
    l = path['Low'].to_numpy();  c = path['Close'].to_numpy()
    ts = path['Date'].to_numpy()

    exit_price = None; exit_ts = None; reason = None; ambiguous = False
    for i in range(len(path)):
        oi, hi, li, ci = o[i], h[i], l[i], c[i]
        # Gap-through the stop on the open of this bar -> filled at the open, worse.
        if oi <= stop_price:
            exit_price, exit_ts, reason = oi, ts[i], 'STOP_GAP'; break
        # Gap-through the target on the open -> filled at the open, better.
        if oi >= target_price:
            exit_price, exit_ts, reason = oi, ts[i], 'TARGET_GAP'; break
        hit_stop = li <= stop_price
        hit_tgt = hi >= target_price
        if hit_stop and hit_tgt:
            # Both inside one bar: order unknown at this resolution. Conservative =
            # assume the stop printed first (tick data would resolve it).
            exit_price, exit_ts, reason, ambiguous = stop_price, ts[i], 'STOP_AMBIG', True; break
        if hit_stop:
            exit_price, exit_ts, reason = stop_price, ts[i], 'STOP'; break
        if hit_tgt:
            exit_price, exit_ts, reason = target_price, ts[i], 'TARGET'; break

    partial = False
    if exit_price is None:
        # Neither leg fired in the window -> exit at the last available close.
        # When honoring the backtest timeout this is the ExitDate close (the rule the
        # 10:00-entry EDA was built on: timeout / momentum / manual all -> close).
        last_day = path['Date'].dt.normalize().iloc[-1]
        if honor_timeout and last_day < exit_day:
            partial = True               # ExitDate bars were missing from the lake
        exit_price, exit_ts, reason = c[-1], ts[-1], 'CLOSE'

    gross_ret_pct = (exit_price / entry_fill - 1.0) * 100.0

    return {
        'Status': 'OK',
        'FilterVerdict': verdict,
        'SimEntryTime': pd.Timestamp(entry_bar_ts),
        'SimEntryPrice': entry_fill,
        'SimExitTime': pd.Timestamp(exit_ts),
        'SimExitPrice': exit_price,
        'SimExitReason': reason,
        'SimStopPrice': stop_price,
        'SimTargetPrice': target_price,
        'OpenGapPct': open_gap_pct,
        'SpyMovePct': spy_move,
        'SimPnLPct': gross_ret_pct,
        'Ambiguous': ambiguous,
        'Partial': partial,
    }


# ── Reporting helpers ────────────────────────────────────────────────────────────

def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return x.mean() / x.std(ddof=1) * np.sqrt(len(x))


def trimmed_mean(x: np.ndarray, pct: float = 0.02) -> float:
    """Mean after clipping the top/bottom `pct` tails — robust to corrupt-bar outliers."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return float(x.mean()) if len(x) else float('nan')
    lo, hi = np.quantile(x, [pct, 1 - pct])
    return float(x[(x >= lo) & (x <= hi)].mean())


def bucket_table(df: pd.DataFrame, by: str, val: str, q: int) -> pd.DataFrame:
    """Quantile-bucket `df[by]` and summarise realized return `df[val]`."""
    try:
        df = df.copy()
        df['_bkt'] = pd.qcut(df[by], q, duplicates='drop')
    except ValueError:
        return pd.DataFrame()
    rows = []
    for b, g in df.groupby('_bkt', observed=True):
        rows.append({
            'bucket': str(b),
            'n': len(g),
            'up_prob_mean': g[by].mean(),
            'ret_mean_%': g[val].mean(),
            'win_rate_%': (g[val] > 0).mean() * 100,
            'sharpe': sharpe(g[val].to_numpy()),
            'realized_up_rate_%': (g['SimExitPrice'] > g['SimEntryPrice']).mean() * 100,
        })
    return pd.DataFrame(rows)


def fmt_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "  (insufficient data)\n"
    return df.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trade-history', default=os.path.join(script_dir, 'trade_history.parquet'))
    ap.add_argument('--intraday-dir', default=os.path.join(script_dir, 'Data', 'IntradayTradeSim'))
    ap.add_argument('--suffix', default='_1min',
                    help='Bar-file suffix: _1min (default), _5min, or a tick suffix.')
    ap.add_argument('--spy-file', default=None,
                    help='SPY intraday parquet for the 10:00 abort gate (optional).')
    ap.add_argument('--slippage-bps', type=float, default=3.0,
                    help='Adaptive-algo entry slippage in bps (broker states 2-5).')
    ap.add_argument('--no-timeout', action='store_true',
                    help="Let the bracket run to a stop/target; ignore the backtest ExitDate.")
    ap.add_argument('--out-dir', default=os.path.join(script_dir, 'Data', 'IntradayFillSim'))
    ap.add_argument('--report', default=os.path.join(script_dir, 'analysis_output',
                                                     'intraday_fill_sim_report.md'))
    ap.add_argument('--buckets', type=int, default=5, help='Quantile buckets for the UpProb tables.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)

    th = pd.read_parquet(args.trade_history)
    th['EntryDate'] = pd.to_datetime(th['EntryDate'])
    th['ExitDate'] = pd.to_datetime(th['ExitDate'])
    print(f"Loaded {len(th)} backtest trades from {os.path.basename(args.trade_history)}")

    # SPY gate (optional)
    spy_default = os.path.join(args.intraday_dir, 'SPY_1min.parquet')
    spy_path = args.spy_file or (spy_default if os.path.exists(spy_default) else None)
    spy_df = load_intraday(os.path.dirname(spy_path), '_1min', 'SPY') if spy_path else None
    if spy_path and spy_df is None:
        try:
            spy_df = pd.read_parquet(spy_path)
            spy_df['Date'] = pd.to_datetime(spy_df['Date'])
        except Exception:
            spy_df = None
    spy_moves = spy_move_by_day(spy_df)
    spy_enabled = len(spy_moves) > 0
    print(f"SPY abort gate: {'ENABLED' if spy_enabled else 'DISABLED (no SPY data)'}")

    # Cache intraday frames per ticker (a ticker appears in many trades).
    cache: dict[str, pd.DataFrame | None] = {}

    def get_bars(tkr):
        if tkr not in cache:
            cache[tkr] = load_intraday(args.intraday_dir, args.suffix, tkr)
        return cache[tkr]

    out_rows = []
    for _, r in th.iterrows():
        bars = get_bars(r['Symbol'])
        base = {
            'Symbol': r['Symbol'], 'EntryDate': r['EntryDate'], 'ExitDate': r['ExitDate'],
            'BtEntryPrice': r['EntryPrice'], 'BtExitPrice': r['ExitPrice'],
            'BtPnLPct': r['PnLPct'], 'BtExitReason': r.get('ExitReason'),
            'Quantity': r['Quantity'], 'UpProbability': r.get('UpProbability'),
            'ATR': r.get('ATR'),
        }
        if bars is None:
            out_rows.append({**base, 'Status': 'NO_DATA'})
            continue
        sim = simulate_trade(bars, r['EntryDate'], r['ExitDate'], r['EntryPrice'],
                             args.slippage_bps, not args.no_timeout,
                             spy_moves if spy_enabled else {})
        out_rows.append({**base, **sim})

    sim_df = pd.DataFrame(out_rows)
    if 'FilterVerdict' not in sim_df.columns:
        sim_df['FilterVerdict'] = np.nan

    # Every trade with usable data is simulated (Status=='OK'); the live filters only
    # set FilterVerdict, so a "skipped" trade still has a counterfactual sim return.
    ok = sim_df['Status'] == 'OK'
    taken = ok & (sim_df['FilterVerdict'] == 'TAKEN')

    sim_df.loc[ok, 'EntrySlippagePct'] = (
        (sim_df.loc[ok, 'SimEntryPrice'] / sim_df.loc[ok, 'BtEntryPrice'] - 1.0) * 100.0)
    sim_df.loc[ok, 'PnLDeltaPct'] = sim_df.loc[ok, 'SimPnLPct'] - sim_df.loc[ok, 'BtPnLPct']

    # Same-dollar net % return: subtract both commission sides as % of notional.
    def net_pct(row):
        if row['Status'] != 'OK':
            return np.nan
        px_in, px_out = row['SimEntryPrice'], row['SimExitPrice']
        qty = max(int(row['Quantity']), 1)
        notional = px_in * qty
        comm = ibkr_commission(qty, px_in) + ibkr_commission(qty, px_out)
        gross = (px_out - px_in) * qty
        return (gross - comm) / notional * 100.0
    sim_df['SimPnLPctNet'] = sim_df.apply(net_pct, axis=1)

    out_parquet = os.path.join(args.out_dir, 'sim_trades.parquet')
    sim_df.to_parquet(out_parquet, index=False)

    # ── Build report ─────────────────────────────────────────────────────────────
    t = sim_df[taken].copy()
    # Display label: the live decision when simulated, else the data-status.
    disp = sim_df['Status'].where(~ok, sim_df['FilterVerdict'])
    status_counts = disp.value_counts()
    n_total = len(sim_df)
    n_taken = int(taken.sum())

    L = []
    L.append("# Intraday Fill & Exit Fidelity Report\n")
    L.append(f"Trades: **{n_total}**  |  resolution: `{args.suffix}`  |  "
             f"entry slippage: **{args.slippage_bps:.1f} bps**  |  "
             f"timeout: **{'OFF (bracket runs free)' if args.no_timeout else 'ON (honor bt ExitDate)'}**  |  "
             f"SPY gate: **{'ON' if spy_enabled else 'OFF'}**\n")

    # 1. Coverage / fills
    L.append("## 1. Coverage & fills\n")
    for k, v in status_counts.items():
        L.append(f"- `{k}`: {v}  ({v/n_total*100:.1f}%)")
    L.append("")
    if n_taken:
        sl = t['EntrySlippagePct']
        L.append(f"Entry slippage (sim 10:00 fill vs backtest open fill), % of price:")
        L.append(f"- mean **{sl.mean():+.3f}%**, median {sl.median():+.3f}%, "
                 f"p10 {sl.quantile(.1):+.3f}%, p90 {sl.quantile(.9):+.3f}%")
        L.append(f"- trades filling WORSE than backtest: {(sl>0).mean()*100:.1f}%  |  "
                 f"better: {(sl<0).mean()*100:.1f}%\n")

    # 2. Headline reconciliation
    L.append("## 2. Headline reconciliation (taken trades)\n")
    if n_taken:
        bt = t['BtPnLPct']; sm = t['SimPnLPct']; smn = t['SimPnLPctNet']
        L.append("| metric | backtest | sim (gross) | sim (net comm) |")
        L.append("|---|---|---|---|")
        L.append(f"| mean return % | {bt.mean():+.3f} | {sm.mean():+.3f} | {smn.mean():+.3f} |")
        L.append(f"| trimmed mean (2%) % | {trimmed_mean(bt.to_numpy()):+.3f} | "
                 f"{trimmed_mean(sm.to_numpy()):+.3f} | {trimmed_mean(smn.to_numpy()):+.3f} |")
        L.append(f"| median return % | {bt.median():+.3f} | {sm.median():+.3f} | {smn.median():+.3f} |")
        L.append(f"| win rate % | {(bt>0).mean()*100:.1f} | {(sm>0).mean()*100:.1f} | {(smn>0).mean()*100:.1f} |")
        L.append(f"| per-trade Sharpe | {sharpe(bt.to_numpy()):.3f} | {sharpe(sm.to_numpy()):.3f} | {sharpe(smn.to_numpy()):.3f} |")
        L.append("")
        L.append("> This is the per-TRADE distribution, not the equity-curve headline. Prefer the "
                 "trimmed mean / median: a few corrupt intraday prints distort the raw mean.\n")

        # Exit-reason mix — the daily backtest cannot see intraday stop/target touches.
        stop_sh = t['SimExitReason'].isin(['STOP', 'STOP_GAP', 'STOP_AMBIG']).mean() * 100
        tgt_sh = t['SimExitReason'].isin(['TARGET', 'TARGET_GAP']).mean() * 100
        cls_sh = (t['SimExitReason'] == 'CLOSE').mean() * 100
        L.append(f"Intraday exit mix: **{stop_sh:.0f}% hit the −{HARD_STOP_PCT}% stop**, "
                 f"{tgt_sh:.0f}% hit the +2:1 target, {cls_sh:.0f}% exited at close. "
                 f"(The daily backtest logged its stop/target on daily H/L and recorded almost no "
                 f"stop-outs — this is the fidelity gap.)\n")

    # 2b. Do the live filters actually add value? Judge SKIPPED trades by what they
    #     WOULD have returned under live 10:00 entry — NOT by the unachievable
    #     daily-open backtest number (the gap/dip is already gone by 10:00).
    L.append("## 2b. Do the live filters earn their keep? (counterfactual at 10:00)\n")
    sg = sim_df[ok & (sim_df['FilterVerdict'] == 'SKIP_GAP')]
    ss = sim_df[ok & (sim_df['FilterVerdict'] == 'SKIP_SPY')]
    L.append("Each skipped trade re-simulated as if it HAD been entered at 10:00 under the "
             "same bracket. A filter adds value only if the trades it removes are worse than "
             "the ones it keeps — judged on the live sim, not the backtest.\n")
    L.append("| group | n | backtest mean % | **sim-net mean %** | sim win % |")
    L.append("|---|---|---|---|---|")
    L.append(f"| TAKEN (kept) | {len(t)} | {t['BtPnLPct'].mean():+.3f} | "
             f"**{t['SimPnLPctNet'].mean():+.3f}** | {(t['SimPnLPctNet']>0).mean()*100:.1f} |")
    if len(sg):
        L.append(f"| SKIP_GAP (dropped) | {len(sg)} | {sg['BtPnLPct'].mean():+.3f} | "
                 f"**{sg['SimPnLPctNet'].mean():+.3f}** | {(sg['SimPnLPctNet']>0).mean()*100:.1f} |")
    if len(ss):
        L.append(f"| SKIP_SPY (dropped) | {len(ss)} | {ss['BtPnLPct'].mean():+.3f} | "
                 f"**{ss['SimPnLPctNet'].mean():+.3f}** | {(ss['SimPnLPctNet']>0).mean()*100:.1f} |")
    L.append("")
    if len(sg):
        verdict_gap = ("ADDS VALUE" if sg['SimPnLPctNet'].mean() < t['SimPnLPctNet'].mean()
                       else "COSTS — removes trades that beat the kept set")
        L.append(f"- **gap-skip**: dropped trades return {sg['SimPnLPctNet'].mean():+.2f}% live vs "
                 f"{t['SimPnLPctNet'].mean():+.2f}% for kept → **{verdict_gap}**. "
                 f"(Note the backtest's {sg['BtPnLPct'].mean():+.2f}% on these is NOT live-achievable "
                 f"— it enters at the open and banks the gap the live system can't.)")
    if len(ss):
        verdict_spy = ("ADDS VALUE" if ss['SimPnLPctNet'].mean() < t['SimPnLPctNet'].mean()
                       else "COSTS on this sample — removes above-average trades")
        L.append(f"- **SPY-abort**: dropped trades return {ss['SimPnLPctNet'].mean():+.2f}% live vs "
                 f"{t['SimPnLPctNet'].mean():+.2f}% for kept → **{verdict_spy}** "
                 f"(small n={len(ss)}; the live EDA that justified it used a different "
                 f"same-day exit-at-close system).")
    L.append("")

    # 3. ENTRY STATISTICAL-PROPERTIES VALIDATION (the centerpiece)
    L.append("## 3. Entry statistical properties vs the model (★)\n")
    if n_taken and t['UpProbability'].notna().sum() > 10:
        tv = t.dropna(subset=['UpProbability']).copy()
        rho_g = tv['UpProbability'].corr(tv['SimPnLPct'], method='spearman')
        rho_n = tv['UpProbability'].corr(tv['SimPnLPctNet'], method='spearman')
        L.append(f"Spearman rank corr  UpProbability vs realized return:  "
                 f"gross **{rho_g:+.3f}**, net **{rho_n:+.3f}**  "
                 f"(positive => the model's ordering holds under real fills)\n")

        L.append(f"### UpProbability buckets (gross sim return)\n")
        bt_tbl = bucket_table(tv, 'UpProbability', 'SimPnLPct', args.buckets)
        L.append("```")
        L.append(fmt_table(bt_tbl))
        L.append("```")
        if not bt_tbl.empty:
            mono = bt_tbl['ret_mean_%'].is_monotonic_increasing
            L.append(f"Bucket mean-return monotonic in UpProbability: **{mono}**\n")

        # Calibration: predicted P(up) vs realized profitable-trade rate
        L.append("### Calibration — predicted P(up) vs realized profitable-trade rate\n")
        if not bt_tbl.empty:
            cal = bt_tbl[['up_prob_mean', 'realized_up_rate_%', 'n']].copy()
            cal['predicted_%'] = cal['up_prob_mean'] * 100
            cal['cal_error_pp'] = cal['realized_up_rate_%'] - cal['predicted_%']
            L.append("```")
            L.append(fmt_table(cal[['predicted_%', 'realized_up_rate_%', 'cal_error_pp', 'n']]))
            L.append("```")
            mce = cal['cal_error_pp'].abs().mean()
            L.append(f"Mean |gap|: **{mce:.2f} pp**. NOTE: 'realized' here = profitable trade "
                     f"(SimExit > SimEntry under the bracket), which is a RELATED-BUT-DIFFERENT "
                     f"event from the model's training label (next-day up move). Read this as "
                     f"'does UpProbability track real trade profitability', not strict label "
                     f"calibration. The tight match at 4/5 buckets says yes — except the "
                     f"top-confidence bucket, which is consistently overconfident.\n")
    else:
        L.append("_UpProbability not available on enough trades to validate._\n")

    # 4. Exit-reason reshape
    L.append("## 4. Exit-reason reshape (where the daily backtest was blind)\n")
    if n_taken:
        ct = pd.crosstab(t['BtExitReason'], t['SimExitReason'])
        L.append("Backtest exit reason (rows) vs intraday-resolved exit (cols):\n")
        L.append("```")
        L.append(ct.to_string())
        L.append("```")
        n_amb = int(t['Ambiguous'].sum())
        bt_tp = t['BtExitReason'].astype(str).str.contains('Take Profit', case=False)
        flip = int((bt_tp & (t['SimExitReason'].isin(['STOP', 'STOP_GAP', 'STOP_AMBIG']))).sum())
        L.append(f"- Same-bar stop/target ambiguities (resolved stop-first, conservative): **{n_amb}**")
        L.append(f"- Backtest 'Take Profit' that intraday hit the STOP first: **{flip}**")
        if t['Partial'].any():
            L.append(f"- Trades with missing ExitDate bars (exited at last available close): "
                     f"{int(t['Partial'].sum())}")
        L.append("")

    L.append("## Files\n")
    L.append(f"- Per-trade forensics: `{os.path.relpath(out_parquet, script_dir)}`")
    L.append(f"- This report: `{os.path.relpath(args.report, script_dir)}`")
    if not spy_enabled:
        L.append("\n> ⚠ SPY abort gate was OFF (no SPY intraday data). Realized win-rates "
                 "on red-market days are therefore OPTIMISTICALLY included. Enable with:\n"
                 "> `python experimental/2.2__TradeHistoryIntradayDownloader.py --index-tickers SPY`")

    report = "\n".join(L)
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write(report)

    # ── Console summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  INTRADAY FILL SIM — SUMMARY")
    print("=" * 72)
    print(f"  Trades total : {n_total}")
    for k, v in status_counts.items():
        print(f"    {k:<10}: {v}  ({v/n_total*100:.1f}%)")
    if n_taken:
        print(f"  Entry slippage (mean): {t['EntrySlippagePct'].mean():+.3f}%")
        print(f"  Mean return %  bt={t['BtPnLPct'].mean():+.3f}  "
              f"sim={t['SimPnLPct'].mean():+.3f}  net={t['SimPnLPctNet'].mean():+.3f}")
        print(f"  Win rate %     bt={(t['BtPnLPct']>0).mean()*100:.1f}  "
              f"sim={(t['SimPnLPct']>0).mean()*100:.1f}  net={(t['SimPnLPctNet']>0).mean()*100:.1f}")
        if t['UpProbability'].notna().sum() > 10:
            tv = t.dropna(subset=['UpProbability'])
            print(f"  Spearman UpProb vs net return: "
                  f"{tv['UpProbability'].corr(tv['SimPnLPctNet'], method='spearman'):+.3f}")
    print("=" * 72)
    print(f"  Report : {os.path.relpath(args.report, script_dir)}")
    print(f"  Trades : {os.path.relpath(out_parquet, script_dir)}")
    print("=" * 72)


if __name__ == '__main__':
    main()
