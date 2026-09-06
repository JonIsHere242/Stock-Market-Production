#!/usr/bin/env python
"""Trade-book autopsy: the diagnostics layer between 4__Predictor and 5__NightlyBackTester.

Reads the trade table the backtester just saved (Data/TradeHistory.parquet or any
saved variant) and prints nine sections the headline report does not cover:

  1. Exit-reason anatomy        (count / win rate / hold / P&L share per exit path)
  2. Holding-period anatomy     (edge by days held, where the stop bill lands)
  3. Signal health              (entry-stamped IC by month, probability decay)
  4. Pond attribution           (ATR%, price bucket, position weight, sizer efficacy)
  5. Concentration and luck     (symbol concentration, top-N strip, bootstrap CI, halves)
  6. Repeat entries             (re-entry performance, revenge trades after a stop)
  7. Time structure             (weekday, entries per day, cohort dates, autocorrelation)
  8. Drawdown anatomy           (top episodes: depth, length, recovery, time underwater)
  9. Path stats                 (MAE / MFE, overnight vs intraday split, Rule G support)

Design constraints, deliberate:
  * Self-contained: no Util import, so it can run on a box with nothing but
    pandas / numpy / scipy and a trade parquet.
  * Every section is guarded; one failure never kills the report.
  * Deterministic: the bootstrap uses a fixed seed.
  * Sections that need external data (per-ticker OHLC, SPY) degrade to a
    one-line skip note instead of raising.

Standalone:
    python backtest_diagnostics.py Data/TradeHistory_5_1_runner.parquet
    python backtest_diagnostics.py --no-path --no-regime

From 5__NightlyBackTester, print_detailed_results calls run_diagnostics() with the
live daily return series so the drawdown section uses real daily equity instead of
the trade-book reconstruction.
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Formatting helpers (match the report's 30-char label / 10-char value layout)
# ---------------------------------------------------------------------------
GREEN  = "\033[38;2;0;200;0m"
DGREEN = "\033[38;2;0;150;0m"
YELLOW = "\033[38;2;220;220;0m"
ORANGE = "\033[38;2;220;140;0m"
RED    = "\033[38;2;220;0;0m"
GREY   = "\033[38;2;150;150;150m"
BLUE   = "\033[38;2;100;149;237m"
RESET  = "\033[0m"

_USE_COLOR = sys.stdout.isatty() or os.environ.get('FORCE_COLOR')


def _c(text, color):
    if not _USE_COLOR:
        return str(text)
    return f"{color}{text}{RESET}"


def _sign_c(value, text=None):
    """Color a number green/red by sign."""
    s = text if text is not None else f"{value:,.2f}"
    return _c(s, GREEN if value >= 0 else RED)


def metric(value, label, good, bad, lower_is_better=False, fmt="{:.2f}", extra=None):
    """One colorized metric line in the same visual dialect as the main report."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        print(f"{label:<30}{_c('N/A', GREY):<10}")
        return
    v = float(value)
    better = (v <= good) if lower_is_better else (v >= good)
    worse  = (v >= bad)  if lower_is_better else (v <= bad)
    if better:
        color, tag = GREEN, "Good"
    elif worse:
        color, tag = RED, "Weak"
    else:
        color, tag = YELLOW, "Mid"
    val = fmt.format(v)
    tail = f"  {_c(extra, GREY)}" if extra else ""
    pad = max(30, len(label) + 1)
    print(f"{label:<{pad}}{_c(f'{val:<10}', color)}[{_c(tag, color)}]{tail}")


def _header(title):
    print(f"\n{title}:")


def _note(text):
    print(_c(f"  {text}", GREY))


def _table(headers, rows, widths, align=None):
    """Plain aligned table. rows hold (text, color) cells or bare strings."""
    align = align or ['<'] * len(headers)
    head = "  ".join(f"{h:{align[i]}{widths[i]}}" for i, h in enumerate(headers))
    print(_c(head, GREY))
    for row in rows:
        out = []
        for i, cell in enumerate(row):
            if isinstance(cell, tuple):
                text, color = cell
            else:
                text, color = str(cell), None
            padded = f"{text:{align[i]}{widths[i]}}"
            out.append(_c(padded, color) if color else padded)
        print("  ".join(out))


def _pnl_cell(v, fmt="{:+,.0f}", width_pad=""):
    return (fmt.format(v), GREEN if v >= 0 else RED)


def _wr_cell(wr):
    color = GREEN if wr >= 55 else (RED if wr < 45 else YELLOW)
    return (f"{wr:.1f}%", color)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_trades(trade_file):
    t = pd.read_parquet(trade_file)
    if len(t) == 0:
        raise ValueError(f"{trade_file} is empty")
    t = t.copy()
    for col in ('EntryDate', 'ExitDate'):
        t[col] = pd.to_datetime(t[col])
    for col in ('PnL', 'PnLPct', 'EntryPrice', 'ExitPrice', 'ATR',
                'UpProbability', 'EntryUpProbability', 'AccountValue'):
        if col in t.columns:
            t[col] = pd.to_numeric(t[col], errors='coerce')
    t = t.sort_values('EntryDate').reset_index(drop=True)
    return t


def daily_series_from_trades(t):
    """Fallback equity path when no daily return series is passed in: last
    AccountValue per exit date. Coarse (open positions are invisible between
    exits) so the drawdown section labels it as approximate."""
    if 'AccountValue' not in t.columns:
        return None
    eq = t.sort_values('ExitDate').groupby('ExitDate')['AccountValue'].last()
    if len(eq) < 10:
        return None
    return eq.pct_change().dropna()


# ---------------------------------------------------------------------------
# 1. Exit-reason anatomy
# ---------------------------------------------------------------------------

def sec_exit_reasons(t):
    _header("Trade Autopsy - Exit Reason Anatomy")
    if 'ExitReason' not in t.columns:
        _note("ExitReason column absent, skipped")
        return
    gross_win = t.loc[t['PnL'] > 0, 'PnL'].sum()
    gross_loss = abs(t.loc[t['PnL'] < 0, 'PnL'].sum())
    rows = []
    g = t.groupby('ExitReason')
    order = g['PnL'].sum().sort_values(ascending=False).index
    for reason in order:
        sub = g.get_group(reason)
        n = len(sub)
        wr = (sub['PnL'] > 0).mean() * 100
        wins = sub.loc[sub['PnL'] > 0, 'PnL'].sum()
        losses = abs(sub.loc[sub['PnL'] < 0, 'PnL'].sum())
        rows.append([
            str(reason)[:25],
            f"{n}",
            f"{n / len(t) * 100:.0f}%",
            _wr_cell(wr),
            _pnl_cell(sub['PnLPct'].mean(), "{:+.2f}%"),
            f"{sub['DaysHeld'].mean():.1f}",
            _pnl_cell(sub['PnL'].sum()),
            f"{(wins / gross_win * 100) if gross_win else 0:.0f}%",
            f"{(losses / gross_loss * 100) if gross_loss else 0:.0f}%",
        ])
    _table(
        ["Reason", "n", "trades", "win%", "avgP&L%", "hold", "netP&L$", "of wins", "of losses"],
        rows,
        [25, 6, 8, 8, 10, 6, 11, 9, 10],
        ['<', '>', '>', '>', '>', '>', '>', '>', '>'])
    stops = t[t['ExitReason'].astype(str).str.contains('Stop Loss', case=False, na=False)]
    if len(stops):
        _note(f"stop bill: {len(stops)} stops x {stops['PnL'].mean():,.2f}$ avg "
              f"= {stops['PnL'].sum():,.0f}$ ({abs(stops['PnL'].sum()) / gross_win * 100 if gross_win else 0:.0f}% of gross wins)")


# ---------------------------------------------------------------------------
# 2. Holding-period anatomy
# ---------------------------------------------------------------------------

def _hold_bucket(d):
    if d <= 3:
        return str(int(d))
    if d <= 5:
        return "4-5"
    if d <= 8:
        return "6-8"
    return "9+"


_HOLD_ORDER = ["1", "2", "3", "4-5", "6-8", "9+"]


def sec_holding_period(t):
    _header("Trade Autopsy - Holding Period Anatomy")
    t = t.assign(_hb=t['DaysHeld'].apply(_hold_bucket))
    rows = []
    for b in _HOLD_ORDER:
        sub = t[t['_hb'] == b]
        if len(sub) == 0:
            continue
        rows.append([
            f"{b}d",
            f"{len(sub)}",
            _wr_cell((sub['PnL'] > 0).mean() * 100),
            _pnl_cell(sub['PnLPct'].mean(), "{:+.2f}%"),
            _pnl_cell(sub['PnLPct'].median(), "{:+.2f}%"),
            _pnl_cell(sub['PnL'].sum()),
        ])
    _table(["Held", "n", "win%", "avgP&L%", "medP&L%", "netP&L$"],
           rows, [7, 6, 8, 10, 10, 11], ['<', '>', '>', '>', '>', '>'])
    win_hold = t.loc[t['PnL'] > 0, 'DaysHeld'].mean()
    loss_hold = t.loc[t['PnL'] < 0, 'DaysHeld'].mean()
    _note(f"winners held {win_hold:.2f}d, losers held {loss_hold:.2f}d")
    if 'ExitReason' in t.columns:
        stops = t[t['ExitReason'].astype(str).str.contains('Stop Loss', case=False, na=False)]
        if len(stops):
            dist = stops.groupby('_hb')['PnL'].agg(['count', 'sum'])
            parts = [f"day {b}: {int(dist.loc[b, 'count'])} stops {dist.loc[b, 'sum']:+,.0f}$"
                     for b in _HOLD_ORDER if b in dist.index]
            _note("stop-outs by day held: " + " | ".join(parts))


# ---------------------------------------------------------------------------
# 3. Signal health (entry-stamped)
# ---------------------------------------------------------------------------

def sec_signal_health(t):
    _header("Trade Autopsy - Signal Health (entry-stamped)")
    col = 'EntryUpProbability'
    if col not in t.columns or t[col].isna().all():
        _note("EntryUpProbability absent, skipped")
        return
    sub = t.dropna(subset=[col, 'PnLPct'])

    # Monthly IC on the traded book. This is the decay tracker: the panel study
    # (memo project_signal_eda_2026_08_02) found the frozen model died Dec 2025,
    # and pre-cutoff panel ICs are rescoring-flattered. Entry-stamped trade IC is
    # immune to the rescoring problem, so a slide here is real.
    sub = sub.assign(_m=sub['EntryDate'].dt.to_period('M'))
    rows = []
    for m, grp in sub.groupby('_m'):
        if len(grp) < 15:
            ic_txt = (f"n<15", GREY)
        else:
            ic, _ = spearmanr(grp[col], grp['PnLPct'])
            ic_txt = (f"{ic:+.3f}", GREEN if ic > 0.03 else (RED if ic < -0.03 else YELLOW))
        rows.append([
            str(m),
            f"{len(grp)}",
            ic_txt,
            _wr_cell((grp['PnL'] > 0).mean() * 100),
            _pnl_cell(grp['PnLPct'].mean(), "{:+.2f}%"),
        ])
    _table(["Month", "n", "entryIC", "win%", "avgP&L%"],
           rows, [10, 6, 9, 8, 10], ['<', '>', '>', '>', '>'])

    months = sorted(sub['_m'].unique())
    if len(months) >= 6:
        recent = sub[sub['_m'].isin(months[-3:])]
        prior = sub[sub['_m'].isin(months[:-3])]
        ic_r, _ = spearmanr(recent[col], recent['PnLPct'])
        ic_p, _ = spearmanr(prior[col], prior['PnLPct'])
        metric(ic_r - ic_p, "IC drift (last 3m vs prior):", 0.0, -0.05,
               fmt="{:+.3f}", extra=f"recent {ic_r:+.3f} vs prior {ic_p:+.3f}")

    # Probability decay while held: the mechanism behind exit-stamped circularity.
    if 'UpProbability' in t.columns:
        d = t.dropna(subset=['UpProbability', 'EntryUpProbability'])
        if len(d) > 20:
            decay = d['UpProbability'] - d['EntryUpProbability']
            dw = decay[d['PnL'] > 0].mean()
            dl = decay[d['PnL'] <= 0].mean()
            _note(f"prob decay entry to exit: winners {dw:+.3f}, losers {dl:+.3f} "
                  f"(gap is the circularity engine: losers are held while prob rots)")


# ---------------------------------------------------------------------------
# 4. Pond attribution
# ---------------------------------------------------------------------------

def _bucket_table(t, series, labels, title_note=None):
    rows = []
    for lab, mask in labels:
        sub = t[mask]
        if len(sub) == 0:
            continue
        rows.append([
            lab,
            f"{len(sub)}",
            _wr_cell((sub['PnL'] > 0).mean() * 100),
            _pnl_cell(sub['PnLPct'].mean(), "{:+.2f}%"),
            _pnl_cell(sub['PnL'].sum()),
        ])
    _table(["Bucket", "n", "win%", "avgP&L%", "netP&L$"],
           rows, [16, 6, 8, 10, 11], ['<', '>', '>', '>', '>'])
    if title_note:
        _note(title_note)


def sec_ponds(t):
    _header("Trade Autopsy - Pond Attribution")
    # ATR as % of entry price, terciles. The fanout found the high-ATR tercile is
    # an anti-pond (panel IC negative there), so the book should show the same tilt.
    if 'ATR' in t.columns and t['ATR'].notna().sum() > 30:
        d = t.dropna(subset=['ATR', 'EntryPrice'])
        atr_pct = d['ATR'] / d['EntryPrice'] * 100
        q1, q2 = atr_pct.quantile([1 / 3, 2 / 3])
        print(_c("  ATR% of entry price (terciles):", GREY))
        _bucket_table(d, atr_pct, [
            (f"low  <{q1:.1f}%", atr_pct <= q1),
            (f"mid", (atr_pct > q1) & (atr_pct <= q2)),
            (f"high >{q2:.1f}%", atr_pct > q2),
        ])
    # Entry price buckets (sub-$5 was the other anti-pond).
    p = t['EntryPrice']
    print(_c("  Entry price:", GREY))
    _bucket_table(t, p, [
        ("<$5", p < 5),
        ("$5-20", (p >= 5) & (p < 20)),
        ("$20-50", (p >= 20) & (p < 50)),
        ("$50+", p >= 50),
    ])
    # Position weight and sizer efficacy, measured against the ENTRY-day account.
    # The trade book only stamps AccountValue at exit; dividing entry notional by
    # that bakes the trade's own outcome (and the tape during the hold) into the
    # weight and manufactures a fake negative IC (measured 2026-08-03: -0.398
    # exit-stamped vs -0.151 entry-stamped on the same book). The entry account is
    # rebuilt from the exit-stamped curve, forward-filled to the day before entry.
    if 'AccountValue' in t.columns and 'Quantity' in t.columns:
        d = t.dropna(subset=['AccountValue'])
        eq = d.sort_values('ExitDate').groupby('ExitDate')['AccountValue'].last()
        grid = pd.date_range(d['EntryDate'].min(), d['ExitDate'].max(), freq='D')
        curve = eq.reindex(grid).ffill()
        first_acct = float(eq.iloc[0]) if len(eq) else float('nan')
        entry_acct = d['EntryDate'].map(
            lambda dt: curve.get(dt - pd.Timedelta(days=1), first_acct))
        entry_acct = pd.to_numeric(entry_acct, errors='coerce').fillna(first_acct)
        w = d['Quantity'] * d['EntryPrice'] / entry_acct.values * 100
        q1, q2 = w.quantile([1 / 3, 2 / 3])
        print(_c("  Position weight (% of entry-day account, terciles):", GREY))
        _bucket_table(d, w, [
            (f"small <{q1:.1f}%", w <= q1),
            (f"mid", (w > q1) & (w <= q2)),
            (f"large >{q2:.1f}%", w > q2),
        ])
        ic, pv = spearmanr(w, d['PnLPct'])
        metric(ic, "Sizer efficacy IC:", 0.03, -0.03, fmt="{:+.3f}",
               extra=f"rank(weight) vs rank(P&L%), p={pv:.3f}; positive = sizes winners bigger")
        if w.std() < 1.5:
            _note(f"weights are nearly flat (std {w.std():.2f}pp): the book carries almost no")
            _note("cross-sectional sizing, so this IC reads batch-tail cash effects, not policy")


# ---------------------------------------------------------------------------
# 5. Concentration, luck, robustness
# ---------------------------------------------------------------------------

def sec_concentration(t, initial_value):
    _header("Trade Autopsy - Concentration & Luck")
    net = t['PnL'].sum()

    # Symbol concentration.
    by_sym = t.groupby('Symbol')['PnL'].agg(['sum', 'count']).sort_values('sum', ascending=False)
    n_sym = len(by_sym)
    top5 = by_sym['sum'].head(5).sum()
    metric(n_sym, "Distinct symbols traded:", len(t) * 0.5, len(t) * 0.15, fmt="{:.0f}",
           extra=f"{len(t)} trades")
    if net > 0:
        metric(top5 / net * 100, "Top 5 symbols P&L share %:", 25, 60, lower_is_better=True)
    best = ", ".join(f"{s} {by_sym.loc[s,'sum']:+,.0f}$ (n={int(by_sym.loc[s,'count'])})"
                     for s in by_sym.head(3).index)
    worst = ", ".join(f"{s} {by_sym.loc[s,'sum']:+,.0f}$ (n={int(by_sym.loc[s,'count'])})"
                      for s in by_sym.tail(3).index[::-1])
    _note(f"best:  {best}")
    _note(f"worst: {worst}")

    # Luck strip: what survives without the biggest winners.
    ps = np.sort(t['PnL'].values)[::-1]
    for k in (1, 3, 5, 10):
        if len(ps) > k:
            residual = net - ps[:k].sum()
            share = residual / net * 100 if net != 0 else float('nan')
            val = _sign_c(residual, f"{f'{residual:+,.0f}':<12}")
            print(f"{f'Net P&L ex top {k} trades ($):':<30}{val}"
                  f"{_c(f'{share:.0f}% of net survives', GREY)}")

    # Bootstrap on the per-trade distribution. Fixed seed, deterministic.
    r = t['PnLPct'].dropna().values
    if len(r) > 50:
        rng = np.random.default_rng(42)
        idx = rng.integers(0, len(r), size=(10000, len(r)))
        means = r[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        p_neg = (means <= 0).mean() * 100
        ci = _sign_c(lo, f"{f'[{lo:+.2f}%, {hi:+.2f}%]':<20}")
        print(f"{'Bootstrap mean P&L% 95% CI:':<30}{ci}"
              f"{_c('10k resamples, seed 42', GREY)}")
        metric(p_neg, "P(mean trade P&L <= 0) %:", 2.5, 10.0, lower_is_better=True)

    # Trade-level SQN and half-split decay.
    pnl = t['PnL'].dropna().values
    if len(pnl) > 30 and pnl.std(ddof=1) > 0:
        sqn = pnl.mean() / pnl.std(ddof=1) * np.sqrt(len(pnl))
        metric(sqn, "Trade SQN (sqrt-n scaled):", 2.5, 1.6)
    half = len(t) // 2
    a, b = t.iloc[:half], t.iloc[half:]
    if len(a) > 30 and len(b) > 30:
        rows = []
        for lab, sub in (("first half", a), ("second half", b)):
            ic = spearmanr(sub['EntryUpProbability'], sub['PnLPct'])[0] \
                if 'EntryUpProbability' in sub.columns and sub['EntryUpProbability'].notna().sum() > 20 else float('nan')
            rows.append([
                f"{lab} ({sub['EntryDate'].min():%Y-%m} to {sub['EntryDate'].max():%Y-%m})",
                f"{len(sub)}",
                _wr_cell((sub['PnL'] > 0).mean() * 100),
                _pnl_cell(sub['PnLPct'].mean(), "{:+.2f}%"),
                f"{ic:+.3f}" if ic == ic else "N/A",
            ])
        _table(["Window", "n", "win%", "avgP&L%", "entryIC"],
               rows, [34, 6, 8, 10, 9], ['<', '>', '>', '>', '>'])


# ---------------------------------------------------------------------------
# 6. Repeat entries
# ---------------------------------------------------------------------------

def sec_repeats(t):
    _header("Trade Autopsy - Repeat Entries")
    t = t.sort_values('EntryDate').copy()
    t['_nth'] = t.groupby('Symbol').cumcount()
    first = t[t['_nth'] == 0]
    rep = t[t['_nth'] > 0]
    rows = []
    for lab, sub in (("first entry in symbol", first), ("repeat entries", rep)):
        if len(sub) == 0:
            continue
        rows.append([
            lab,
            f"{len(sub)}",
            _wr_cell((sub['PnL'] > 0).mean() * 100),
            _pnl_cell(sub['PnLPct'].mean(), "{:+.2f}%"),
            _pnl_cell(sub['PnL'].sum()),
        ])
    _table(["Cohort", "n", "win%", "avgP&L%", "netP&L$"],
           rows, [24, 6, 8, 10, 11], ['<', '>', '>', '>', '>'])
    # Re-entry directly after a stop-loss exit in the same symbol.
    if 'ExitReason' in t.columns and len(rep):
        prev_reason = t.groupby('Symbol')['ExitReason'].shift(1)
        prev_exit = t.groupby('Symbol')['ExitDate'].shift(1)
        after_stop = t[(t['_nth'] > 0)
                       & prev_reason.astype(str).str.contains('Stop Loss', na=False)
                       & ((t['EntryDate'] - prev_exit).dt.days <= 15)]
        if len(after_stop) >= 10:
            wr = (after_stop['PnL'] > 0).mean() * 100
            _note(f"re-entry within 15d of a stop-out in the same name: n={len(after_stop)}, "
                  f"win {wr:.0f}%, avg {after_stop['PnLPct'].mean():+.2f}%")
        else:
            _note(f"re-entry within 15d of a stop-out: only n={len(after_stop)}, too few to read")


# ---------------------------------------------------------------------------
# 7. Time structure
# ---------------------------------------------------------------------------

def sec_time_structure(t, daily):
    _header("Trade Autopsy - Time Structure")
    wd_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    rows = []
    for wd in range(5):
        sub = t[t['EntryDate'].dt.dayofweek == wd]
        if len(sub) == 0:
            continue
        rows.append([
            wd_names[wd],
            f"{len(sub)}",
            _wr_cell((sub['PnL'] > 0).mean() * 100),
            _pnl_cell(sub['PnLPct'].mean(), "{:+.2f}%"),
            _pnl_cell(sub['PnL'].sum()),
        ])
    _table(["Entry day", "n", "win%", "avgP&L%", "netP&L$"],
           rows, [12, 6, 8, 10, 11], ['<', '>', '>', '>', '>'])

    per_day = t.groupby(t['EntryDate'].dt.normalize()).size()
    _note(f"entries per active day: mean {per_day.mean():.2f}, p90 {per_day.quantile(0.9):.0f}, "
          f"max {per_day.max()} ({per_day.idxmax():%Y-%m-%d})")

    cohort = t.groupby(t['EntryDate'].dt.normalize())['PnL'].agg(['sum', 'count'])
    top = cohort.sort_values('sum', ascending=False)
    best = " | ".join(f"{d:%Y-%m-%d} {r['sum']:+,.0f}$ (n={int(r['count'])})"
                      for d, r in top.head(3).iterrows())
    worst = " | ".join(f"{d:%Y-%m-%d} {r['sum']:+,.0f}$ (n={int(r['count'])})"
                       for d, r in top.tail(3).iloc[::-1].iterrows())
    _note(f"best entry cohorts:  {best}")
    _note(f"worst entry cohorts: {worst}")

    if daily is not None and len(daily) > 40:
        d = daily.dropna()
        ac1 = d.autocorr(lag=1)
        metric(ac1, "Daily return autocorr (lag 1):", 0.05, -0.10, fmt="{:+.3f}")
        up = d > 0
        if up.sum() > 10 and (~up).sum() > 10:
            p_up_after_up = up[up.shift(1) == True].mean() * 100  # noqa: E712
            p_up_after_dn = up[up.shift(1) == False].mean() * 100  # noqa: E712
            _note(f"P(up day) after up {p_up_after_up:.0f}% vs after down {p_up_after_dn:.0f}% "
                  f"(base {up.mean() * 100:.0f}%)")


# ---------------------------------------------------------------------------
# 8. Drawdown anatomy
# ---------------------------------------------------------------------------

def sec_drawdowns(daily, initial_value, approx=False):
    _header("Trade Autopsy - Drawdown Anatomy")
    if daily is None or len(daily) < 40:
        _note("no daily return series available, skipped")
        return
    if approx:
        _note("equity rebuilt from trade-book AccountValue (exit days only), depths are approximate")
    eq = initial_value * (1 + daily).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    underwater = dd < 0

    episodes = []
    start = None
    for i, (dt, uw) in enumerate(underwater.items()):
        if uw and start is None:
            start = i
        elif not uw and start is not None:
            episodes.append((start, i))
            start = None
    if start is not None:
        episodes.append((start, None))  # still underwater at the end

    detail = []
    for s, e in episodes:
        seg = dd.iloc[s:e] if e is not None else dd.iloc[s:]
        trough_i = seg.idxmin()
        detail.append({
            'depth': seg.min() * 100,
            'start': dd.index[max(s - 1, 0)],
            'trough': trough_i,
            'end': dd.index[e] if e is not None else None,
            'length': len(seg),
            'to_trough': int((seg.index <= trough_i).sum()),
        })
    detail.sort(key=lambda x: x['depth'])
    rows = []
    for ep in detail[:5]:
        end_txt = f"{ep['end']:%Y-%m-%d}" if ep['end'] is not None else "ongoing"
        recovery = ep['length'] - ep['to_trough']
        rows.append([
            f"{ep['depth']:.1f}%",
            f"{ep['start']:%Y-%m-%d}",
            f"{ep['trough']:%Y-%m-%d}",
            end_txt,
            f"{ep['to_trough']}d down",
            f"{recovery}d back" if ep['end'] is not None else "-",
        ])
    _table(["Depth", "From peak", "Trough", "Recovered", "Decline", "Recovery"],
           rows, [9, 12, 12, 12, 10, 10], ['>', '<', '<', '<', '>', '>'])
    metric(underwater.mean() * 100, "Time underwater %:", 40, 70, lower_is_better=True,
           extra=f"{int(underwater.sum())} of {len(underwater)} days below a prior peak")


# ---------------------------------------------------------------------------
# 9. Path stats (MAE / MFE, overnight vs intraday) from per-ticker OHLC
# ---------------------------------------------------------------------------

def sec_path(t, data_dir):
    _header("Trade Autopsy - Path (MAE/MFE & Overnight Split)")
    if not os.path.isdir(data_dir):
        _note(f"{data_dir} not found, skipped")
        return

    # All path math runs in BAR space, anchored to the entry bar's open, never to the
    # recorded fill price. Reason, measured 2026-08-03 on the 5.1 runner book: the
    # current lake's bars have been re-adjusted since that backtest ran, so recorded
    # fills sit several percent off today's bars for over half the trades, and even
    # clean names carry ~1% fill drag on open-fill exits. Within one bar segment the
    # adjustment is shared, so excursions and the overnight/intraday split are exact
    # in bar space; only the recorded-fill linkage is noisy, and that gap is printed
    # as its own line (fill gap) instead of silently poisoning the stats.
    recs = []
    n_nofile, n_nobar, n_drift = 0, 0, 0
    open_fill_exits = ('Max Hold Time', 'Manual Exit')
    for sym, grp in t.groupby('Symbol'):
        f = os.path.join(data_dir, f"{sym}.parquet")
        if not os.path.exists(f):
            n_nofile += len(grp)
            continue
        try:
            bars = pd.read_parquet(f, columns=['Date', 'Open', 'High', 'Low', 'Close'])
        except Exception:
            n_nofile += len(grp)
            continue
        bars['Date'] = pd.to_datetime(bars['Date'])
        bars = bars.set_index('Date').sort_index()
        for _, tr in grp.iterrows():
            seg = bars.loc[tr['EntryDate']:tr['ExitDate']]
            if len(seg) < 2 or seg.index[0] != tr['EntryDate']:
                n_nobar += 1
                continue
            ref = float(seg['Open'].iloc[0])
            if ref <= 0:
                n_nobar += 1
                continue
            if abs(ref - tr['EntryPrice']) / tr['EntryPrice'] > 0.02:
                n_drift += 1  # bar series revised since the run; still usable in bar space
            closes, opens = seg['Close'].values, seg['Open'].values
            # entry day: open -> close (intraday); each later day: prior close -> open
            # (overnight) then open -> close (intraday); the exit day path stops at its
            # open, where max-hold and manual exits actually fill.
            overnight = float((opens[1:] - closes[:-1]).sum())
            intraday = float(closes[0] - ref + (closes[1:-1] - opens[1:-1]).sum()) \
                if len(seg) > 2 else float(closes[0] - ref)
            bar_ret = (opens[-1] - ref) / ref * 100
            rec_ret = (tr['ExitPrice'] - tr['EntryPrice']) / tr['EntryPrice'] * 100
            recs.append({
                'pnl_pos': tr['PnL'] > 0,
                'mfe': (seg['High'].max() - ref) / ref * 100,
                'mae': (seg['Low'].min() - ref) / ref * 100,
                'overnight': overnight / ref * 100,
                'intraday': intraday / ref * 100,
                'bar_ret': bar_ret,
                'fill_gap': rec_ret - bar_ret,
                'open_fill': str(tr.get('ExitReason', '')) in open_fill_exits,
            })
    if not recs:
        _note(f"no trades matched to bars ({n_nofile} no file, {n_nobar} missing bars), skipped")
        return
    d = pd.DataFrame(recs)
    _note(f"matched {len(d)} of {len(t)} trades to {data_dir} bars "
          f"({n_nofile} no file, {n_nobar} missing bars; {n_drift} on bars revised since the "
          f"run, fine in bar space)")

    win, loss = d[d['pnl_pos']], d[~d['pnl_pos']]
    rows = []
    for lab, sub in (("all", d), ("winners", win), ("losers", loss)):
        if len(sub) == 0:
            continue
        rows.append([
            lab,
            f"{len(sub)}",
            _pnl_cell(sub['mfe'].mean(), "{:+.2f}%"),
            _pnl_cell(sub['mae'].mean(), "{:+.2f}%"),
            _pnl_cell(sub['overnight'].mean(), "{:+.2f}%"),
            _pnl_cell(sub['intraday'].mean(), "{:+.2f}%"),
            _pnl_cell(sub['fill_gap'].mean(), "{:+.2f}%"),
        ])
    _table(["Cohort", "n", "avgMFE", "avgMAE", "overnight", "intraday", "fill gap"],
           rows, [10, 6, 9, 9, 11, 10, 10], ['<', '>', '>', '>', '>', '>', '>'])
    _note("path to exit-day open = overnight + intraday (bar space, anchored at entry open);")
    _note("fill gap = recorded trade return minus that path (stop/scale-out fills, slippage,")
    _note("bar revisions). The panel study says the alpha rides overnight gaps; the overnight")
    _note("column shows whether the BOOK collects them.")

    # Execution drag isolated where fills should equal the exit-day open exactly.
    of = d[d['open_fill']]
    if len(of) > 30:
        metric(of['fill_gap'].mean(), "Fill gap, open-fill exits %:", 0.0, -0.5,
               fmt="{:+.2f}", extra=f"n={len(of)}; nonzero = slippage model or bar revisions")

    if d['mae'].mean() != 0:
        metric(abs(d['mfe'].mean() / d['mae'].mean()), "Edge ratio (MFE/|MAE|):", 1.5, 1.0)
    if len(loss) > 20:
        rule_g = (loss['mfe'] >= 2.0).mean() * 100
        metric(rule_g, "Losers that saw +2% first %:", 25, 10,
               extra="upper bound on what a Rule G style exit could intercept")
    touched = d[d['mae'] <= -5.0]
    if len(touched) > 10:
        _note(f"trades touching -5% during the hold: {len(touched)}, of which "
              f"{(touched['bar_ret'] > 0).mean() * 100:.0f}% were still positive at the exit-day open")


# ---------------------------------------------------------------------------
# 10. Regime split (SPY 200-day EMA), network-guarded
# ---------------------------------------------------------------------------

def sec_regime(t):
    _header("Trade Autopsy - Regime Split (S&P 500 200d EMA)")
    try:
        import yfinance as yf
        start = (t['EntryDate'].min() - pd.Timedelta(days=420)).strftime('%Y-%m-%d')
        end = (t['ExitDate'].max() + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        spx = yf.download("^GSPC", start=start, end=end, interval="1d",
                          auto_adjust=True, progress=False)
        if len(spx) < 250:
            _note("not enough index history downloaded, skipped")
            return
        close = spx['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        ema = close.ewm(span=200, adjust=False).mean()
        above = (close > ema)
        above.index = pd.to_datetime(above.index).normalize()
        regime = t['EntryDate'].dt.normalize().map(above)
        rows = []
        for lab, mask in (("above 200d EMA", regime == True),    # noqa: E712
                          ("below 200d EMA", regime == False)):  # noqa: E712
            sub = t[mask]
            if len(sub) == 0:
                continue
            rows.append([
                lab,
                f"{len(sub)}",
                _wr_cell((sub['PnL'] > 0).mean() * 100),
                _pnl_cell(sub['PnLPct'].mean(), "{:+.2f}%"),
                _pnl_cell(sub['PnL'].sum()),
            ])
        _table(["Entry regime", "n", "win%", "avgP&L%", "netP&L$"],
               rows, [18, 6, 8, 10, 11], ['<', '>', '>', '>', '>'])
        n_unmapped = int(regime.isna().sum())
        if n_unmapped:
            _note(f"{n_unmapped} entries on dates missing from index data, excluded")
    except Exception as e:
        _note(f"skipped ({type(e).__name__}: {e})")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_diagnostics(trade_file='Data/TradeHistory.parquet', daily_return_series=None,
                    initial_value=10000.0, data_dir=None, skip=()):
    """Print the full autopsy. Every section is independently guarded."""
    data_dir = data_dir or os.environ.get('BT_DIAG_DATA_DIR', 'Data/RFpredictions')
    skip = set(skip) | set(filter(None, os.environ.get('BT_DIAG_SKIP', '').split(',')))
    try:
        t = load_trades(trade_file)
    except Exception as e:
        print(f"\n[trade autopsy skipped: cannot read {trade_file}: {e}]")
        return

    approx_daily = False
    daily = daily_return_series
    if daily is None:
        daily = daily_series_from_trades(t)
        approx_daily = daily is not None

    print("\n" + "=" * 80)
    print(f" Trade Book Autopsy ({os.path.basename(trade_file)}, {len(t)} trades) ".center(80))
    print("=" * 80)

    sections = [
        ('exit', lambda: sec_exit_reasons(t)),
        ('hold', lambda: sec_holding_period(t)),
        ('signal', lambda: sec_signal_health(t)),
        ('ponds', lambda: sec_ponds(t)),
        ('concentration', lambda: sec_concentration(t, initial_value)),
        ('repeats', lambda: sec_repeats(t)),
        ('time', lambda: sec_time_structure(t, daily)),
        ('drawdowns', lambda: sec_drawdowns(daily, initial_value, approx=approx_daily)),
        ('path', lambda: sec_path(t, data_dir)),
        ('regime', lambda: sec_regime(t)),
    ]
    for name, fn in sections:
        if name in skip:
            continue
        try:
            fn()
        except Exception as e:
            print(f"[autopsy section '{name}' failed: {type(e).__name__}: {e}]")


def _find_default_trade_file():
    candidates = sorted(glob.glob('Data/TradeHistory*.parquet'),
                        key=os.path.getmtime, reverse=True)
    if not candidates:
        raise SystemExit("no Data/TradeHistory*.parquet found; pass a path")
    return candidates[0]


def main():
    ap = argparse.ArgumentParser(description="Trade-book autopsy (standalone)")
    ap.add_argument("trade_file", nargs='?', default=None,
                    help="trade parquet (default: newest Data/TradeHistory*.parquet)")
    ap.add_argument("--data-dir", default=None,
                    help="per-ticker OHLC dir for path stats (default Data/RFpredictions)")
    ap.add_argument("--initial", type=float, default=10000.0)
    ap.add_argument("--no-path", action='store_true', help="skip the OHLC path section")
    ap.add_argument("--no-regime", action='store_true', help="skip the SPY regime section")
    args = ap.parse_args()
    skip = []
    if args.no_path:
        skip.append('path')
    if args.no_regime:
        skip.append('regime')
    tf = args.trade_file or _find_default_trade_file()
    run_diagnostics(trade_file=tf, initial_value=args.initial,
                    data_dir=args.data_dir, skip=skip)


if __name__ == "__main__":
    main()
