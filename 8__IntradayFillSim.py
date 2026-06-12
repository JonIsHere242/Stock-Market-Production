#!/usr/bin/env python3
"""
Intraday Fill Reality-Check  (8__IntradayFillSim.py)
====================================================
Replays every trade in trade_history.parquet against real 1-minute bars
(downloaded by 2.2__TradeHistoryIntradayDownloader.py into Data/IntradayTradeSim/)
and recomputes the portfolio metrics under realistic intraday fills, so you can see
how far the backtest's headline numbers move once orders are filled the way the live
broker actually fills them.

The backtest (5__NightlyBackTester.py) fills at the DAILY OPEN. The live broker
(9_SuperFastBroker.py) waits until 10:00 ET to enter and exits at the close. This script
reprices each trade three ways, on ONE shared trade universe (only trades whose entry-
AND exit-day minute data we have), so the columns are directly comparable:

  1. Backtest (recomputed)  — uses trade_history's own EntryPrice/ExitPrice. Validates
                              the metric math by reproducing the known headline.
  2. Real @ 10:00 / close   — entry at the 10:00 ET bar, exit at the closing bar.  <-- the live rule
  3. Real @ open  / close   — entry at the 09:30 open, exit at the close (slippage contrast).

What's held fixed (per the agreed design): the trade set, the entry/exit DATES, and the
dollar notional of each position. Only the FILL PRICES change; share counts are rederived
from the same dollars (shares = floor(EntryPrice*Quantity / real_entry_fill)). It does NOT
re-compound (later position sizes stay at the backtest's levels) and does NOT re-derive
signals or sizing — it isolates the impact of *where you actually got filled*.

Equity curve: realized P&L is booked on each trade's ExitDate (net of a replicated IBKR
Adaptive commission, charged both sides). Path-independent stats (total return, win rate,
profit factor, avg win/loss, EV) reproduce the backtest exactly for column 1; the risk
stats (Sharpe/Sortino/vol/drawdown) are computed off this realized curve consistently
across all three columns — so they are a fair cross-column comparison, not a re-derivation
of the backtester's daily mark-to-market Sharpe.

Usage
-----
    python 8__IntradayFillSim.py
    python 8__IntradayFillSim.py --entry-time 10:00 --initial-capital 10000
    python 8__IntradayFillSim.py --intraday-dir Data/IntradayTradeSim
"""

import os
import argparse
from datetime import time as dtime

import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

DEFAULT_TRADE_HISTORY = 'trade_history.parquet'
DEFAULT_INTRADAY_DIR  = os.path.join('Data', 'IntradayTradeSim')
OUTPUT_DIR            = 'analysis_output'
TRADING_DAYS         = 252

# IBKR Adaptive commission params (mirrors IBKRAdaptiveCommission in 5__NightlyBackTester.py)
COMM_PER_SHARE  = 0.0035
COMM_MIN_ORDER  = 0.35
COMM_MAX_PCT    = 0.01
COMM_EXCH_FEE   = 0.0002


# ── Commission (replicated from the backtester, charged per side) ───────────────

def commission(shares: int, price: float) -> float:
    if shares <= 0 or price <= 0:
        return 0.0
    per_share = shares * COMM_PER_SHARE
    exch      = shares * COMM_EXCH_FEE
    value_cap = shares * price * COMM_MAX_PCT
    base      = max(per_share, COMM_MIN_ORDER)
    return min(base, value_cap) + exch


# ── Intraday bar access ─────────────────────────────────────────────────────────

class IntradayStore:
    """Lazy per-symbol loader for the 1-min parquet files, indexed by calendar date."""

    def __init__(self, intraday_dir: str):
        self.dir = intraday_dir
        self._cache: dict[str, dict] = {}   # symbol -> {date: day_df}

    def _load(self, symbol: str) -> dict:
        fp = os.path.join(self.dir, f"{symbol}_1min.parquet")
        if not os.path.exists(fp):
            return {}
        try:
            df = pd.read_parquet(fp)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            return {d: g for d, g in df.groupby(df['Date'].dt.date)}
        except Exception:
            return {}

    def day_bars(self, symbol: str, day) -> pd.DataFrame | None:
        if symbol not in self._cache:
            self._cache[symbol] = self._load(symbol)
        return self._cache[symbol].get(day)


def fill_open(day_df: pd.DataFrame) -> float | None:
    """Open of the first RTH bar (~09:30)."""
    if day_df is None or day_df.empty:
        return None
    return float(day_df.iloc[0]['Open'])


def fill_at_time(day_df: pd.DataFrame, t: dtime) -> float | None:
    """Open of the first bar at/after time t (fallback: first bar of the day)."""
    if day_df is None or day_df.empty:
        return None
    after = day_df[day_df['Date'].dt.time >= t]
    bar = after.iloc[0] if not after.empty else day_df.iloc[0]
    return float(bar['Open'])


def fill_close(day_df: pd.DataFrame) -> float | None:
    """Close of the last RTH bar (~16:00)."""
    if day_df is None or day_df.empty:
        return None
    return float(day_df.iloc[-1]['Close'])


# ── Trade repricing ─────────────────────────────────────────────────────────────

def reprice(trades: pd.DataFrame, store: IntradayStore, entry_time: dtime) -> pd.DataFrame:
    """
    For each trade, attach real-fill entry/exit prices and the resulting net P&L for
    each fill model. Adds a `covered` flag (entry- and exit-day minute data present).
    """
    rows = []
    for r in trades.itertuples(index=False):
        sym       = r.Symbol
        e_day     = r.EntryDate.date()
        x_day     = r.ExitDate.date()
        bt_entry  = float(r.EntryPrice)
        bt_exit   = float(r.ExitPrice)
        bt_qty    = int(r.Quantity)
        notional  = bt_entry * bt_qty

        e_bars = store.day_bars(sym, e_day)
        x_bars = store.day_bars(sym, x_day)
        covered = (e_bars is not None and not e_bars.empty
                   and x_bars is not None and not x_bars.empty)

        rec = {
            'Symbol': sym, 'EntryDate': r.EntryDate, 'ExitDate': r.ExitDate,
            'bt_entry': bt_entry, 'bt_exit': bt_exit, 'bt_qty': bt_qty,
            'notional': notional, 'covered': covered,
        }

        # ── Column 1: backtest fills (its own recorded prices) ──
        rec.update(_pnl_block('bt', bt_entry, bt_exit, bt_qty, notional))

        # ── Real fills ──
        real_open = fill_open(e_bars) if covered else None
        real_1000 = fill_at_time(e_bars, entry_time) if covered else None
        real_clz  = fill_close(x_bars) if covered else None

        if covered and real_1000 and real_clz:
            sh = int(notional // real_1000)
            rec.update(_pnl_block('r1000', real_1000, real_clz, sh, notional))
        else:
            rec.update(_pnl_block('r1000', np.nan, np.nan, 0, notional))

        if covered and real_open and real_clz:
            sh = int(notional // real_open)
            rec.update(_pnl_block('ropen', real_open, real_clz, sh, notional))
        else:
            rec.update(_pnl_block('ropen', np.nan, np.nan, 0, notional))

        rows.append(rec)

    return pd.DataFrame(rows)


def _pnl_block(prefix: str, entry: float, exit_: float, shares: int, notional: float) -> dict:
    """Gross/net P&L, %, commission and shares for one fill model."""
    if not (entry and exit_ and shares > 0) or np.isnan(entry) or np.isnan(exit_):
        return {f'{prefix}_entry': entry, f'{prefix}_exit': exit_, f'{prefix}_shares': shares,
                f'{prefix}_gross': np.nan, f'{prefix}_comm': np.nan,
                f'{prefix}_net': np.nan, f'{prefix}_pct': np.nan}
    gross = (exit_ - entry) * shares
    comm  = commission(shares, entry) + commission(shares, exit_)
    net   = gross - comm
    pct   = (exit_ / entry - 1.0) * 100.0
    return {f'{prefix}_entry': entry, f'{prefix}_exit': exit_, f'{prefix}_shares': shares,
            f'{prefix}_gross': gross, f'{prefix}_comm': comm,
            f'{prefix}_net': net, f'{prefix}_pct': pct}


# ── Metrics ──────────────────────────────────────────────────────────────────────

def perf_metrics(df: pd.DataFrame, prefix: str, initial_capital: float) -> dict:
    """Portfolio + trade metrics for one fill model, on the covered subset."""
    net   = df[f'{prefix}_net']
    valid = df[net.notna()].copy()
    n = len(valid)
    if n == 0:
        return {'n_trades': 0}

    gross = valid[f'{prefix}_gross']
    comm  = valid[f'{prefix}_comm']
    pct   = valid[f'{prefix}_pct']
    netv  = valid[f'{prefix}_net']

    # ── Realized-PnL equity curve (booked on ExitDate) ──
    by_day = valid.groupby(valid['ExitDate'].dt.normalize())[f'{prefix}_net'].sum().sort_index()
    idx = pd.bdate_range(by_day.index.min(), by_day.index.max())
    eq = initial_capital + by_day.reindex(idx, fill_value=0.0).cumsum()
    ret = eq.pct_change().dropna()

    final_val   = float(eq.iloc[-1])
    total_ret   = final_val / initial_capital - 1.0
    n_days      = max(1, len(idx))
    ann_ret     = (final_val / initial_capital) ** (TRADING_DAYS / n_days) - 1.0

    if len(ret) > 1 and ret.std() > 0:
        sharpe = ret.mean() / ret.std() * np.sqrt(TRADING_DAYS)
        daily_vol = ret.std() * 100
        ann_vol   = ret.std() * np.sqrt(TRADING_DAYS) * 100
    else:
        sharpe = daily_vol = ann_vol = 0.0
    downside = ret[ret < 0]
    sortino = (ret.mean() / downside.std() * np.sqrt(TRADING_DAYS)
               if len(downside) > 1 and downside.std() > 0 else 0.0)

    running_max = eq.cummax()
    dd = (eq - running_max) / running_max
    max_dd = float(dd.min()) * 100

    wins   = netv[netv > 0]
    losses = netv[netv <= 0]
    win_rate_after  = len(wins) / n * 100
    win_rate_before = (gross > 0).mean() * 100
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float('inf')
    total_comm = comm.sum()
    gross_total = gross.sum()
    commission_impact = (total_comm / abs(gross_total) * 100) if gross_total != 0 else 0.0

    # monthly returns from the equity curve
    monthly = eq.resample('ME').last().pct_change().dropna() * 100

    return {
        'n_trades': n,
        'total_return_pct': total_ret * 100,
        'ann_return_pct': ann_ret * 100,
        'final_value': final_val,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd_pct': max_dd,
        'daily_vol_pct': daily_vol,
        'ann_vol_pct': ann_vol,
        'win_rate_after': win_rate_after,
        'win_rate_before': win_rate_before,
        'profit_factor': profit_factor,
        'avg_win_dollar': wins.mean() if len(wins) else 0.0,
        'avg_loss_dollar': losses.mean() if len(losses) else 0.0,
        'avg_win_pct': pct[netv > 0].mean() if (netv > 0).any() else 0.0,
        'avg_loss_pct': pct[netv <= 0].mean() if (netv <= 0).any() else 0.0,
        'ev_per_trade': netv.mean(),
        'total_commission': total_comm,
        'commission_impact_pct': commission_impact,
        'net_pnl_total': netv.sum(),
        '_monthly': monthly,
    }


def account_value_reference(trades_full: pd.DataFrame, initial: float) -> dict:
    """
    Headline portfolio metrics straight from the backtester's AccountValue equity
    curve (the number the user pasted: 232% / -11.22% DD / Sharpe ~4). This is the
    portfolio-engine truth and needs no intraday data. Uses ALL trades.
    """
    av = trades_full.groupby(trades_full['ExitDate'].dt.normalize())['AccountValue'].last()
    idx = pd.bdate_range(av.index.min(), av.index.max())
    eq = av.reindex(idx).ffill().bfill()
    ret = eq.pct_change().dropna()
    final = float(eq.iloc[-1])
    nd = max(1, len(idx))
    rm = eq.cummax()
    max_dd = float(((eq - rm) / rm).min()) * 100
    sharpe = ret.mean() / ret.std() * np.sqrt(TRADING_DAYS) if ret.std() > 0 else 0.0
    dn = ret[ret < 0]
    sortino = ret.mean() / dn.std() * np.sqrt(TRADING_DAYS) if dn.std() > 0 else 0.0
    return {
        'final_value': final,
        'total_return_pct': (final / initial - 1) * 100,
        'ann_return_pct': ((final / initial) ** (TRADING_DAYS / nd) - 1) * 100,
        'sharpe': sharpe, 'sortino': sortino, 'max_dd_pct': max_dd,
        'ann_vol_pct': ret.std() * np.sqrt(TRADING_DAYS) * 100,
        'realized_ledger_pnl': float((trades_full['PnL'] - trades_full['Commission']).sum()),
        'account_growth': final - initial,
    }


def slippage_stats(cov: pd.DataFrame) -> dict:
    """Real-fill vs recorded-fill price differences (the open->10:00 effect)."""
    def pct(real, rec):
        d = (cov[real] / cov[rec] - 1.0) * 100
        return d.replace([np.inf, -np.inf], np.nan).dropna()
    e10 = pct('r1000_entry', 'bt_entry')
    eop = pct('ropen_entry', 'bt_entry')
    xcl = pct('r1000_exit', 'bt_exit')
    return {
        'entry_10_mean': e10.mean(), 'entry_10_med': e10.median(),
        'entry_open_mean': eop.mean(), 'entry_open_med': eop.median(),
        'exit_close_mean': xcl.mean(), 'exit_close_med': xcl.median(),
    }


# ── Reporting ────────────────────────────────────────────────────────────────────

def fmt(v, spec='{:>14.2f}'):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return f"{'—':>14}"
    return spec.format(v)


def build_report(av_ref, m_led, m_r1000, m_ropen, slip, coverage_line, missing, initial) -> str:
    L = []
    L.append("=" * 80)
    L.append("  INTRADAY FILL REALITY-CHECK  -  trade_history.parquet vs real 1-min fills")
    L.append("=" * 80)

    # ── Section 1: portfolio headline straight from AccountValue (the pasted number) ──
    L.append("")
    L.append("  [1] BACKTEST HEADLINE  (from the AccountValue equity curve - the number you pasted)")
    L.append(f"      Total Return        : {av_ref['total_return_pct']:>9.2f}%   (final ${av_ref['final_value']:,.0f} on ${initial:,.0f})")
    L.append(f"      Annualized          : {av_ref['ann_return_pct']:>9.2f}%")
    L.append(f"      Sharpe / Sortino    : {av_ref['sharpe']:>9.2f} / {av_ref['sortino']:.2f}")
    L.append(f"      Max Drawdown        : {av_ref['max_dd_pct']:>9.2f}%      Ann Vol: {av_ref['ann_vol_pct']:.2f}%")

    # ── Section 2: the reconciliation gap the user should know about ──
    gap = av_ref['account_growth'] - av_ref['realized_ledger_pnl']
    L.append("")
    L.append("  [2] LEDGER RECONCILIATION  (heads up - the ledger does NOT match the headline)")
    L.append(f"      Account growth (AccountValue) : ${av_ref['account_growth']:>12,.2f}")
    L.append(f"      Realized P&L of logged trades : ${av_ref['realized_ledger_pnl']:>12,.2f}  (sum of PnL - Commission)")
    L.append(f"      Unexplained gap               : ${gap:>12,.2f}")
    L.append("      => The 232% headline comes from the portfolio equity curve, not the logged")
    L.append("         per-trade prices, which only account for the realized figure above. So this")
    L.append("         reality-check compares fills at the TRADE level (where the minute data bites).")

    # ── Section 3: trade-level fill comparison on the shared covered universe ──
    L.append("")
    L.append("  [3] TRADE-LEVEL FILL COMPARISON  (shared covered universe)")
    L.append(f"      {coverage_line}")
    cols = ("Recorded", "Real 10:00/clz", "Real open/clz")
    L.append("")
    L.append(f"      {'Metric':<26}" + "".join(f"{c:>16}" for c in cols))
    L.append("      " + "-" * (26 + 16 * 3))

    def row(label, key, spec='{:>14.2f}'):
        a = fmt(m_led.get(key), spec); b = fmt(m_r1000.get(key), spec); c = fmt(m_ropen.get(key), spec)
        L.append(f"      {label:<26}{a:>16}{b:>16}{c:>16}")

    row("Total Trades",            'n_trades', '{:>14.0f}')
    row("Realized P&L ($)",        'net_pnl_total')
    row("Win Rate (after fees) %", 'win_rate_after')
    row("Win Rate (before fees)%", 'win_rate_before')
    row("Profit Factor",          'profit_factor')
    row("Avg Win ($)",            'avg_win_dollar')
    row("Avg Loss ($)",           'avg_loss_dollar')
    row("Avg Win (%)",            'avg_win_pct')
    row("Avg Loss (%)",           'avg_loss_pct')
    row("EV per Trade ($)",       'ev_per_trade')
    row("Total Commission ($)",   'total_commission')
    L.append("      --- single $%s-stream compounding (NOT the portfolio engine) ---" % f"{initial:,.0f}")
    row("Compounded Return %",    'total_return_pct')
    row("Sharpe (realized) *",    'sharpe')
    row("Max Drawdown %",         'max_dd_pct')

    # ── Section 4: entry/exit slippage of real fills vs recorded ──
    L.append("")
    L.append("  [4] REAL-FILL SLIPPAGE vs recorded prices  (mean / median %)")
    L.append(f"      Entry @ 10:00 vs recorded : {slip['entry_10_mean']:+.3f}% / {slip['entry_10_med']:+.3f}%")
    L.append(f"      Entry @ open  vs recorded : {slip['entry_open_mean']:+.3f}% / {slip['entry_open_med']:+.3f}%")
    L.append(f"      Exit  @ close vs recorded : {slip['exit_close_mean']:+.3f}% / {slip['exit_close_med']:+.3f}%")

    L.append("")
    L.append("  * Realized Sharpe books P&L on the exit date (no daily mark-to-market), so it is a")
    L.append("    consistent cross-column comparison only; it is NOT the AccountValue Sharpe in [1].")

    # monthly side-by-side (realized streams)
    L.append("")
    L.append("  --- Monthly Realized Return % (single-stream) ---")
    L.append(f"      {'Month':<10}{'Recorded':>16}{'Real 10:00/clz':>16}{'Real open/clz':>16}")
    mb = m_led.get('_monthly', pd.Series(dtype=float))
    m1 = m_r1000.get('_monthly', pd.Series(dtype=float))
    mo = m_ropen.get('_monthly', pd.Series(dtype=float))
    for mth in sorted(set(mb.index) | set(m1.index) | set(mo.index)):
        L.append(f"      {mth.strftime('%Y-%m'):<10}{fmt(mb.get(mth)):>16}{fmt(m1.get(mth)):>16}{fmt(mo.get(mth)):>16}")

    if missing:
        L.append("")
        L.append(f"  Symbols with NO usable intraday data ({len(missing)}): "
                 + ", ".join(missing[:40]) + (" ..." if len(missing) > 40 else ""))

    L.append("=" * 80)
    return "\n".join(L)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Reprice trade_history at real 1-min fills")
    ap.add_argument('--trade-history', default=DEFAULT_TRADE_HISTORY)
    ap.add_argument('--intraday-dir',  default=DEFAULT_INTRADAY_DIR)
    ap.add_argument('--entry-time',    default='10:00', help='ET entry time, HH:MM (default 10:00)')
    ap.add_argument('--initial-capital', type=float, default=10000.0)
    args = ap.parse_args()

    hh, mm = (int(x) for x in args.entry_time.split(':'))
    entry_time = dtime(hh, mm)

    trades = pd.read_parquet(args.trade_history)
    trades['EntryDate'] = pd.to_datetime(trades['EntryDate'])
    trades['ExitDate']  = pd.to_datetime(trades['ExitDate'])
    if 'TradeType' in trades.columns:
        trades = trades[trades['TradeType'] == 'Long'].reset_index(drop=True)

    # Section [1]+[2] reference is computed from AccountValue and needs no intraday data.
    av_ref = account_value_reference(trades, args.initial_capital)

    if not os.path.isdir(args.intraday_dir) or not os.listdir(args.intraday_dir):
        print(f"\n  No intraday data in {args.intraday_dir}.")
        print("  Run:  python 2.2__TradeHistoryIntradayDownloader.py   (TWS must be running)\n")
        return

    store = IntradayStore(args.intraday_dir)
    print(f"\n  Repricing {len(trades)} trades against 1-min data in {args.intraday_dir} ...")
    priced = reprice(trades, store, entry_time)

    # Shared universe: only trades with both entry- and exit-day data AND a valid real fill.
    covered = priced[priced['covered'] & priced['r1000_net'].notna() & priced['ropen_net'].notna()].copy()
    n_total, n_cov = len(priced), len(covered)
    missing_syms = sorted(priced.loc[~priced['covered'], 'Symbol'].unique().tolist())
    coverage_line = (f"Coverage: {n_cov}/{n_total} trades priced "
                     f"({n_cov/n_total*100:.1f}%) across {covered['Symbol'].nunique()} symbols  "
                     f"| entry-time {args.entry_time} ET")

    if n_cov == 0:
        print("\n  No trades had usable entry+exit minute data yet — download more first.\n")
        return

    m_led   = perf_metrics(covered, 'bt',    args.initial_capital)   # recorded-ledger fills
    m_r1000 = perf_metrics(covered, 'r1000', args.initial_capital)
    m_ropen = perf_metrics(covered, 'ropen', args.initial_capital)
    slip    = slippage_stats(covered)

    report = build_report(av_ref, m_led, m_r1000, m_ropen, slip,
                          coverage_line, missing_syms, args.initial_capital)
    print("\n" + report + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_fp = os.path.join(OUTPUT_DIR, 'intraday_fill_sim_report.txt')
    trades_fp = os.path.join(OUTPUT_DIR, 'intraday_fill_sim_trades.parquet')
    with open(report_fp, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    priced.drop(columns=['_monthly'], errors='ignore').to_parquet(trades_fp, index=False)
    print(f"  Report      -> {report_fp}")
    print(f"  Per-trade   -> {trades_fp}\n")


if __name__ == '__main__':
    main()
