# 5__NightlyBackTester.py: THE CANONICAL SOURCE since 2026-08-30. Edit THIS file.
#
# The 5v2 -> make_5v4 -> 5v4 -> prod_patches -> 5__ generator chain is RETIRED
# (experimental/backtesters/_retired_generator_20260830/, README inside). It existed so
# research and production could never drift apart; that guarantee now comes from a
# simpler shape: ONE file where every experimental knob is env-gated and every
# production value is an os.environ.setdefault, so a bare run gets the live defaults
# and the A/B rig (experimental/exit_sweeps/ab.py) overrides them per arm, including
# redirecting BT_V2_TRADEHIST / BT_V2_SIGNALS into its sandbox on every run.
#
# HOUSE RULES for editing:
#   1. A new experiment is an env-gated knob, default byte-identical to shipped.
#   2. Shipping = flipping the setdefault at the top of this file, nothing else.
#   3. A bare hand run WRITES THE LIVE POOL (Data/0__Signals.parquet); use
#      run_optimal_backtest.py for sandboxed hand runs, ab.py for A/Bs.
#   4. Never run two copies at once (they contend on Data/logging and live files).
# Verified at the cut: ab.py control seed on this file reproduces the stack_g4t book
# exactly (memo project_depth_estimator_screen_2026_08_30).
#
# HARDCODE PASS 2026-08-31. The file carried 135 env knobs. 101 of them were never set
# by any sweep, by ab.py or by anything else, and several were marked refuted in their
# own comments, so each one only ever evaluated its shipped branch. Those 101 are now
# literals and their losing branches are deleted (8,065 -> 7,560 lines). Whole features
# went with them: the adaptive trigger offset (_adaptive_offset was an unconditional
# identity with BT_ADAPT unset), the re-entry rules, the alternate sizers, the
# vol-scaled stop and trail, the UpProb slope tilt, the market-fill fallback, the beta
# tilt and the BT_W52 gate. What SURVIVES is deliberate: the 11 production setdefaults
# above and the 33 knobs ab.py drives per arm. Rule 1 still holds for those 33 and for
# anything new; it is simply no longer true that every knob here is live.
# Proof: paired ab.py seed 1, control vs this file, trade books byte-identical
# (md5 7cbc019a8c0268c59a13e69602bb3285, AnnRet 71.23 DD 6.71 Sharpe 2.52, 150 trades).
# Pre-pass copy: 5__NightlyBackTester.py.prehardcode.bak (gitignored).
#!/usr/bin/env python 
import os
import time
import logging
import argparse
import array
import random
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
# --- repo-root bootstrap (moved here 2026-08-28) -------------------------------
# Python puts THIS file's directory on sys.path, not the CWD, so `Util` and
# `auxiliary` at the repo root stop resolving once this script lives in a
# subfolder. Compiling still succeeds (imports are not exercised), which is how
# the move looked clean and then failed at run time. Keep this above the first
# repo import.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# ------------------------------------------------------------------------------

from auxiliary._quiet_progress import tqdm
import pyarrow.parquet as pq
import multiprocessing
from numba import njit
import traceback
from collections import Counter
import pandas_market_calendars as mcal
import math
import glob
import warnings
import yfinance as yf
import concurrent.futures
from collections import defaultdict
from scipy.stats import spearmanr
import scipy.stats as stats
from Util import *



from Util import (
    STRATEGY_PARAMS_TUPLE as STRATEGY_PARAMS,  # Note we're using the tuple version for backtrader
)

# ============================== FORK ISOLATION ==============================
# This is the can_buy-v2 fork. It writes its trade book to its OWN path so it can
# never collide with the live nightly run or with a concurrent A/B arm. The repo's
# own history records two backtests sharing Data/TradeHistory.parquet silently
# corrupting each other, so this is not optional.
# == PRODUCTION DEFAULTS (2026-08-30) ==========================================
# This file is GENERATED from experimental/backtesters/5v2__NightlyBackTester_CanBuyV2.py
# by experimental/exit_sweeps/make_5v4.py --prod. Do not edit by hand; edit the generator.
# These are the knobs of the configuration the broker trades since 2026-08-29 (memo
# project_exit_sweeps_trigger_arm_2026_08_29: 8 paired seeds, AnnRet +45.5 t 9.3 8/8,
# maxDD -3.5 7/8, +0.66 per merged trade). setdefault, so any env set by the caller
# still wins; run_optimal_backtest.py uses that to redirect a hand run into a sandbox.
# Outputs, by default, are the LIVE files the morning broker reads: the pool at
# Data/0__Signals.parquet and the nightly's own trade record Data/TradeHistory.parquet.
# Never run two copies at once (they contend on Data/logging and share these files).
os.environ.setdefault('BT_V2_TRADEHIST', 'Data/TradeHistory.parquet')
os.environ.setdefault('BT_V2_SIGNALS', 'Data/0__Signals.parquet')
os.environ.setdefault('BT_SAMPLE_SEED', '42')
os.environ.setdefault('BT_LIMIT_ENTRY_K', '1.5')
os.environ.setdefault('BT_LIMIT_OVERSUB', '4')
os.environ.setdefault('BT_LIMIT_VOL_EST', 'yz')
os.environ.setdefault('BT_CASHADJUST_FIX', '1')
os.environ.setdefault('BT_PROD_EXITS', '1')
os.environ.setdefault('BT_GATE_GAPFREQ_N', '2')
os.environ.setdefault('BT_GATE_GAP_PCT', '4.0')
os.environ.setdefault('BT_SIGNAL_POOL_SIZE', '36')

V2_TRADE_HISTORY = os.environ.get('BT_V2_TRADEHIST', 'Data/_canbuyv2/TradeHistory.parquet')
os.makedirs(os.path.dirname(V2_TRADE_HISTORY), exist_ok=True)
import Util as _Util
# ISOLATION FIX 2026-08-23. Shadowing COMPLETED_TRADES_FILE in THIS module was not
# enough: Util.clear_completed_trades(), read_completed_trades() and
# write_completed_trades() all resolve Util's OWN module-level constant, so every run
# of this fork truncated the live Data/TradeHistory.parquet regardless of the local
# name. Rebind the attribute on the module object itself, which is what those
# functions actually read.
_Util.COMPLETED_TRADES_FILE = V2_TRADE_HISTORY
COMPLETED_TRADES_FILE = V2_TRADE_HISTORY      # shadows the Util import above
# ============================================================================


# THE bracket. Backtest, signals file and live broker all read this one module, so the
# backtest simulates the strategy that is actually traded. See bracket_config.py for
# the audit that made this necessary.
from auxiliary import bracket_config as BRACKET


# ===================== LIMIT-ENTRY EXPERIMENT (env-gated, default OFF) ==========
# BT_LIMIT_ENTRY_K > 0 replaces the Market entry with a resting buy LIMIT one bar
# below the signal close, at depth = K * beta * vol_20d.
#
# Why scale by beta*vol instead of a flat percentage: a flat -5% limit almost never
# fills on a quiet name and fills constantly on a noisy one, so a flat rule silently
# becomes a volatility screen. Scaling to the name's own beta and 20d vol asks every
# name for a dip of comparable rarity.
#
# The order is valid for exactly ONE bar. If it does not fill it is cancelled at the
# top of the next bar, and handle_order_failure() releases the slot (that path already
# existed for margin rejections). K=0 leaves the shipped Market path byte-identical.
BT_LIMIT_ENTRY_K   = float(os.environ.get('BT_LIMIT_ENTRY_K', '0') or 0)
BT_LIMIT_VOL_LOOKBACK = 20   # matches the can_buy volatility gate's window
# Volatility estimator for the trigger depth. The shipped arm uses an equal-weighted
# 20-day close-to-close std, which is slow: a vol shock takes 20 sessions to fully enter
# it, and in the meantime the effective depth is priced off stale vol. Since K sits on a
# slope (-39pp at K=2.0, -67pp at K=2.5), a vol estimate that lags upward drifts the book
# down the bad side of that curve without anyone changing a setting.
#   std20     equal-weighted 20d close-to-close  (shipped)
#   std10/60  window length
#   ewma94    RiskMetrics decay 0.94  (half-life ~11d)
#   ewma85    faster decay 0.85       (half-life ~4d)
#   parkinson high/low range estimator -- reacts the SAME session as the shock
#   maxsl     max(std10, std20) -- takes the shock immediately, decays slowly
BT_LIMIT_VOL_EST = (os.environ.get('BT_LIMIT_VOL_EST') or 'std20').strip().lower()
#   har       HAR-lite: equal-weight mean of Parkinson over 1, 5 and 22 bars (2026-08-30
#             depth-estimator screen, experimental/exit_sweeps/ideation/depth_estimators/:
#             cross-sectional Spearman with the next-day drop +0.011 over parkinson, paired
#             t +8.9 on 689 days, 5 of 6 half-years; K-matched fill rate steadier across
#             names and VIX regimes; reacts faster on crash day one and decays in days,
#             where parkinson stays 3x elevated for 20 sessions after 2020-03-09)
#   yz        Yang-Zhang 20 bars: overnight gap variance + open-to-close variance + RS.
#             The only estimator that sees the overnight gap; same screen: rank slightly
#             WORSE than parkinson (t -18) but deep gap-throughs on crash days ~15pp fewer.
if BT_LIMIT_VOL_EST not in ('std20','std10','std60','ewma94','ewma85','parkinson','maxsl','har','yz'):
    raise ValueError('bad BT_LIMIT_VOL_EST %r' % BT_LIMIT_VOL_EST)
# backtrader's .get(size=N) returns an EMPTY slice until N bars exist, so asking for more
# history than an estimator needs silently disables the trigger on every early bar. This
# repo has already lost a 52-week gate to exactly that. Ask for the minimum each estimator
# actually requires.
_VOL_BARS = {'std10': 10, 'std20': 20, 'std60': 60, 'maxsl': 20,
             'ewma94': 60, 'ewma85': 60, 'parkinson': 20, 'har': 22, 'yz': 21}
# Opens for the Yang-Zhang branch. _vol_estimate's signature is a generator anchor, so
# the call site parks the open slice here instead of passing a fourth argument.
_OPEN_FOR_VOL = {}


def _vol_estimate(cl, hi, lo):
    """cl/hi/lo are oldest-first arrays. Returns a daily vol estimate, or None."""
    import numpy as _np
    if cl is None or len(cl) < 2:
        return None
    e = BT_LIMIT_VOL_EST
    # std20 must be BYTE-IDENTICAL to the shipped arm: N closes -> N-1 returns -> std.
    # Taking the last 20 RETURNS instead would silently use 21 closes and shift the
    # baseline, which would contaminate every estimator comparison below.
    def _sd(n):
        c = cl[-n:]
        return float(_np.std(_np.diff(_np.log(c)))) if len(c) > 1 else None
    if e == 'std20':
        return _sd(20)
    if e == 'std10':
        return _sd(10)
    if e == 'std60':
        return _sd(60)
    if e == 'maxsl':
        a, b = _sd(10), _sd(20)
        return None if (a is None or b is None) else float(max(a, b))
    r = _np.diff(_np.log(cl))
    if len(r) < 10:
        return None
    if e in ('ewma94', 'ewma85'):
        lam = 0.94 if e == 'ewma94' else 0.85
        rr = r[-60:]
        w = lam ** _np.arange(len(rr) - 1, -1, -1)
        return float(_np.sqrt(_np.sum(w * rr * rr) / _np.sum(w)))
    if e == 'parkinson':
        if hi is None or lo is None or len(hi) < 20:
            return None
        h = _np.asarray(hi[-20:], dtype=float); l = _np.asarray(lo[-20:], dtype=float)
        ok = _np.isfinite(h) & _np.isfinite(l) & (l > 0)
        if ok.sum() < 10:
            return None
        q = _np.log(h[ok] / l[ok]) ** 2
        return float(_np.sqrt(q.mean() / (4.0 * _np.log(2.0))))
    if e == 'har':
        # HAR-lite: mean of the 1-, 5- and 22-bar Parkinson sigmas. The 1-bar term is what
        # makes it react on crash day one and what lets it decay once the shock leaves the
        # short windows; the 22-bar term anchors the level. Byte-identical to the screen's
        # har_pk column (experimental/exit_sweeps/ideation/depth_estimators/screen.py).
        if hi is None or lo is None or len(hi) < 22:
            return None
        h = _np.asarray(hi[-22:], dtype=float); l = _np.asarray(lo[-22:], dtype=float)
        ok = _np.isfinite(h) & _np.isfinite(l) & (l > 0)
        if ok.sum() < 12 or not ok[-1]:
            return None
        q = _np.where(ok, _np.log(_np.where(ok, h, 1.0) / _np.where(ok, l, 1.0)) ** 2, _np.nan)
        def _pk(n):
            v = q[-n:]
            v = v[_np.isfinite(v)]
            return float(_np.sqrt(v.mean() / (4.0 * _np.log(2.0)))) if v.size else None
        p1, p5, p22 = _pk(1), _pk(5), _pk(22)
        if p1 is None or p5 is None or p22 is None:
            return None
        return float((p1 + p5 + p22) / 3.0)
    if e == 'yz':
        # Yang-Zhang, 20 bars: needs 21 closes for 20 overnight gaps. Overnight variance +
        # k x open-to-close variance + (1 - k) x Rogers-Satchell, k = 0.34 / (1.34 + 21/19).
        # The screen's yz20 column; the estimator that sees the overnight gap.
        if hi is None or lo is None or len(cl) < 21 or len(hi) < 21:
            return None
        c = _np.asarray(cl[-21:], dtype=float); h = _np.asarray(hi[-20:], dtype=float)
        l = _np.asarray(lo[-20:], dtype=float)
        o = _OPEN_FOR_VOL.get('open')
        if o is None or len(o) < 20:
            return None
        o = _np.asarray(o[-20:], dtype=float)
        ok = _np.isfinite(c[1:]) & _np.isfinite(c[:-1]) & _np.isfinite(h) & _np.isfinite(l) & _np.isfinite(o) & (l > 0) & (o > 0) & (c[:-1] > 0)
        if ok.sum() < 12:
            return None
        cp, cc, hh, ll, oo = c[:-1][ok], c[1:][ok], h[ok], l[ok], o[ok]
        gap = _np.log(oo / cp); oc = _np.log(cc / oo)
        rs = _np.log(hh / cc) * _np.log(hh / oo) + _np.log(ll / cc) * _np.log(ll / oo)
        n = ok.sum()
        k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
        v = gap.var(ddof=1) + k * oc.var(ddof=1) + (1.0 - k) * max(rs.mean(), 0.0)
        return float(_np.sqrt(max(v, 0.0)))
    return None
# Oversubscription. A limit entry only fills on ~19% of the names it is sent to, and an
# unfilled order leaves the slot EMPTY for the day (the book does not re-pick). That cost
# more than the better entry price was worth. BT_LIMIT_OVERSUB > 1 sends limits to
# OVERSUB x free_slots candidates so ~free_slots of them actually fill, restoring
# participation while the book still holds at most max_positions names.
# Exposure backstop: positions are sized 1/max_positions of equity, so surplus fills are
# margin-rejected by the broker and handle_order_failure() releases the slot. Overfill is
# counted below rather than assumed away.
BT_LIMIT_OVERSUB = float(os.environ.get('BT_LIMIT_OVERSUB', '1') or 1)

# LOOK-AHEAD CONTROL. backtrader matches a bar's orders in SUBMISSION order, and this
# fork submits in ranking order, so when more than max_positions triggers are touched on
# the same day the backtest keeps the best-ranked ones. Live, the winner is whichever
# touched FIRST on the clock, which is uncorrelated with rank. Setting a seed here
# shuffles the submission order within each day to price that advantage.
BT_LIMIT_SHUFFLE = int(os.environ.get('BT_LIMIT_SHUFFLE', '0') or 0)
# GATE (2026-08-26). Drop this FRACTION of the day's candidate pool with the widest recent
# range BEFORE the trigger is armed. max_range_10d = 100 * max over the last 10 bars of
# (High - Low) / Close, reconstructed exactly against the research panel. Panel result:
# dropping the widest 15% pays +0.1529pp/trade, z +2.00 vs 40 matched-random drops (39/40),
# H1 +1.0850 / H2 +1.0888, 9 of 12 months, and it does NOT shave the right tail
# (best 5% +12.3317 -> +12.3559), which is what killed the RSI ceiling.
# This prunes the pool; it must NOT reorder it, because rank order drives who gets sent.
BT_GATE_MAXRANGE_DROP = float(os.environ.get('BT_GATE_MAXRANGE_DROP', '0') or 0)
_MKT_STATE = None


def _mkt_up(date):
    """True when SPY closed above its 20-day mean on `date` (or the last session before)."""
    global _MKT_STATE
    if _MKT_STATE is None:
        fp = 'Data/_canbuyv2/market_state.parquet'
        try:
            _m = pd.read_parquet(fp)
            _m['Date'] = pd.to_datetime(_m['Date'])
            _MKT_STATE = _m.set_index('Date')['spy_vs_ma20'].sort_index()
        except Exception as _e:
            logging.error("MARKET-STATE unavailable (%s); treating every day as UP", _e)
            _MKT_STATE = pd.Series(dtype=float)
    if len(_MKT_STATE) == 0:
        return True
    try:
        v = _MKT_STATE.asof(pd.Timestamp(date))
    except Exception:
        return True
    return bool(v >= 0) if v == v else True
# ── PYRAMID add (lab knob). Unset = off. See experimental/exit_sweeps/pyramid/. ────
BT_PYRAMID_FRAC = float(os.environ.get('BT_PYRAMID_FRAC', '0') or 0)
BT_PYRAMID_DAY = int(float(os.environ.get('BT_PYRAMID_DAY', '1') or 1))
BT_PYRAMID_RESERVE_SLOTS = int(float(os.environ.get('BT_PYRAMID_RESERVE_SLOTS', '0') or 0))
BT_PYRAMID_GAP_MAX = float(os.environ.get('BT_PYRAMID_GAP_MAX', '0') or 0)
# ── 5v4 ExitLab knobs. Unset = byte-identical to 5v2. ─────────────────────────
BT_PANIC_GATE = os.environ.get('BT_PANIC_GATE', '')
BT_PANIC_MODE = (os.environ.get('BT_PANIC_MODE') or 'prior').strip().lower()
BT_PANIC_ACTION = (os.environ.get('BT_PANIC_ACTION') or 'skip').strip().lower()
BT_PANIC_KMULT = float(os.environ.get('BT_PANIC_KMULT', '0.5'))
if BT_PANIC_MODE not in ('prior', 'same') or BT_PANIC_ACTION not in ('skip', 'half', 'double', 'kdown'):
    raise ValueError('BT_PANIC_MODE prior|same, BT_PANIC_ACTION skip|half|double|kdown')
BT_SIZE_DIV = float(os.environ.get('BT_SIZE_DIV', '0') or 0)
# Drop candidates whose SIGNAL-DAY close-to-close return exceeds this percent (autopsy
# 2026-08-29: prior-day return top tercile > +1.24% holds 56% of the bottom-5% trades
# and 12% of the top-5%, 0/4 seeds positive). 0 = off.
BT_GATE_PRIORRET_MAX = float(os.environ.get('BT_GATE_PRIORRET_MAX', '0') or 0)
# Gap-frequency gate: drop names with >= N overnight gaps larger than GAP_PCT (abs %)
# over the last 20 bars. 0 = off. Weak-close gate: drop names that closed in the bottom
# quarter of their daily range on more than FRAC of the last 10 bars. 0 = off.
BT_GATE_GAPFREQ_N = int(float(os.environ.get('BT_GATE_GAPFREQ_N', '0') or 0))
_SPY_GAP = None


def _spy_gap(date):
    """SPY overnight gap in percent on `date` (open vs prior close); 0.0 when unknown."""
    global _SPY_GAP
    if _SPY_GAP is None:
        try:
            _s = pd.read_parquet('Data/IndexesFull/SPY.parquet', columns=['Open', 'Close']).sort_index()
            _g = 100.0 * (_s['Open'] / _s['Close'].shift(1) - 1.0)
            _SPY_GAP = {d.date(): float(v) for d, v in _g.dropna().items()}
        except Exception as _e:
            logging.error('SPY gap series unavailable (%s); idio/beta/disagree modes degrade to abs', _e)
            _SPY_GAP = {}
    return _SPY_GAP.get(date, 0.0)
BT_GATE_GAP_PCT = float(os.environ.get('BT_GATE_GAP_PCT', '3.0'))
BT_GATE_WEAKCLOSE = float(os.environ.get('BT_GATE_WEAKCLOSE', '0') or 0)
# Volume-spike gate: drop names with >= N bars in the last 20 whose Volume exceeds
# BT_GATE_VOLSPIKE_X times the 20-bar median volume. 0 = off. ATR-percent gate: drop
# names whose ATR14 / Close (percent) exceeds this. 0 = off.
BT_GATE_VOLSPIKE_N = int(float(os.environ.get('BT_GATE_VOLSPIKE_N', '0') or 0))
_SPY_RET1 = None


def _spy_ret1(date):
    global _SPY_RET1
    if _SPY_RET1 is None:
        try:
            _m = pd.read_parquet('Data/_canbuyv2/market_state.parquet')
            _m['Date'] = pd.to_datetime(_m['Date'])
            _SPY_RET1 = _m.set_index('Date')['spy_ret1'].sort_index()
        except Exception as _e:
            logging.error('market_state unavailable (%s); boost never fires', _e)
            _SPY_RET1 = pd.Series(dtype=float)
    if len(_SPY_RET1) == 0:
        return None
    try:
        v = _SPY_RET1.asof(pd.Timestamp(date))
    except Exception:
        return None
    return float(v) if v == v else None
_SS20 = None


def _ss20(symbol, date):
    global _SS20
    if _SS20 is None:
        try:
            _t = pd.read_parquet('experimental/exit_sweeps/ideation/altdata/universe_ss.parquet', columns=['Date', 'Symbol', 'ss20'])
            _t['Date'] = pd.to_datetime(_t['Date']).dt.date
            _SS20 = {(s, d): float(v) for s, d, v in zip(_t.Symbol, _t.Date, _t.ss20)}
            logging.info('short-share table loaded: %d rows', len(_SS20))
        except Exception as _e:
            logging.error('short-share table unavailable (%s); gate passes everything', _e)
            _SS20 = {}
    return _SS20.get((symbol, date))
_PANIC = None


def _panic_gated(date, mode):
    """True when trigger limits computed at `date`'s close should be withheld."""
    global _PANIC
    if _PANIC is None:
        _p = pd.read_parquet(BT_PANIC_GATE)
        _p['Date'] = pd.to_datetime(_p['Date'])
        _PANIC = _p.set_index('Date')['gate'].astype(bool).sort_index()
        logging.info('PANIC-GATE loaded %s: %d rows, %d gated', BT_PANIC_GATE,
                     len(_PANIC), int(_PANIC.sum()))
    ts = pd.Timestamp(date)
    if mode == 'same':
        # LOOKAHEAD by construction: the gate of the NEXT session, the one the
        # limit rests in. Upper bound for an intraday kill switch, not a policy.
        _nxt = _PANIC.index[_PANIC.index > ts]
        return bool(_PANIC.loc[_nxt[0]]) if len(_nxt) else False
    try:
        v = _PANIC.asof(ts)
    except Exception:
        return False
    return bool(v) if v == v else False


# ── BT_CLOCK_SESSIONS: hold clock in TRADING SESSIONS, not calendar days (2026-08-30) ──
# The shipped clock is (current_date - signal_date).days >= position_timeout, which gives
# a Monday signal 5 sessions of exposure and a Wednesday/Thursday/Friday signal 3. Stop
# autopsy on the 24 shipped-stack books: Tuesday fills are stopped 87.9% of the time at
# +0.5%/trade, Friday fills 44.8% at +3.5% (memo project_stop_autopsy_2026_08_30).
# N > 0 counts sessions from the signal bar instead: the base timeout fires at N sessions
# for EVERY weekday. Implemented by translating sessions into day units
# (sessions x position_timeout / N) at the top of evaluate_sell_conditions, so every
# downstream comparison is reused unchanged: the runner clock scales to 3N sessions
# (15/5), BT_HOLD_EXTEND_UP and the pyramid guard keep working. 0 = OFF, byte-identical.
BT_CLOCK_SESSIONS = int(float(os.environ.get('BT_CLOCK_SESSIONS', '0') or 0))

# ── BT_K_FRI_MULT: deeper trigger for Friday signals (2026-08-30, default OFF) ─────────
# A Friday-signal trigger is priced off Friday's close and rests into Monday with the
# weekend's news unpriced; that cohort is -1.48pp vs the book, 0/24 seeds, negative in
# most months (memo project_stop_autopsy_2026_08_30). This multiplies the DEPTH on
# Friday-signal names only, demanding a bigger discount for a Monday fill instead of
# refusing the day (BT_SKIP_WEEKDAYS=4 is the refusal limit case). 1.0 = OFF,
# byte-identical. Applied before the min/max clamp so the bounds still bind.
BT_K_FRI_MULT = float(os.environ.get('BT_K_FRI_MULT', '1') or 1)

_BETA_LOOKUP = None

def _beta_for(symbol, date):
    """Rolling 60d beta vs SPY, from Data/_canbuyv2/beta_lookup.parquet.

    Falls back to 1.0 when the name/date is absent, which makes the depth reduce to
    K * vol_20d rather than skipping the name.
    """
    global _BETA_LOOKUP
    if _BETA_LOOKUP is None:
        fp = 'Data/_canbuyv2/beta_lookup.parquet'
        try:
            _b = pd.read_parquet(fp)
            _b['Date'] = pd.to_datetime(_b['Date'])
            _cols = ['beta'] + (['idio_vol'] if 'idio_vol' in _b.columns else [])
            _BETA_LOOKUP = {t: g.set_index('Date')[_cols]
                            for t, g in _b.groupby('ticker', sort=False)}
            logging.info("LIMIT-ENTRY: beta lookup loaded, %d tickers", len(_BETA_LOOKUP))
        except Exception as _e:
            logging.error("LIMIT-ENTRY: beta lookup unavailable (%s); beta=1.0", _e)
            _BETA_LOOKUP = {}
    ser = _BETA_LOOKUP.get(symbol)
    if ser is None:
        return 1.0
    try:
        v = ser['beta'].asof(pd.Timestamp(date))
    except Exception:
        return 1.0
    return float(v) if v == v else 1.0


def _idio_vol_for(symbol, date):
    """Rolling 20d std of the market-residual return. None when unavailable."""
    _beta_for(symbol, date)          # force the lookup to load
    ser = _BETA_LOOKUP.get(symbol)
    if ser is None or 'idio_vol' not in getattr(ser, 'columns', []):
        return None
    try:
        v = ser['idio_vol'].asof(pd.Timestamp(date))
    except Exception:
        return None
    return float(v) if (v == v and v > 0) else None
# ===============================================================================



# ── Runner exits (2026-07-29) ────────────────────────────────────────────────
# Defaults come from bracket_config so this backtest simulates what the broker
# trades; BT_* env vars override per-run for experiments. Validated on the 5.1
# lab rig (memo: project_trail_runner_exit_package_2026_07_29): NO take-profit
# leg, hard stop repegged once per day to prior close - TRAIL_EOD_PCT (an EOD
# ratchet, NOT backtrader StopTrail: the 1-min referee refuted intraday-HWM
# trailing 0/9 books), 80% scale-out at +10% with the runner on a longer clock.
def _runner_on() -> bool:
    v = None
    return (v == '1') if v is not None else BRACKET.USE_RUNNER_EXITS

def _runner_trig() -> float:
    return float(os.environ.get('BT_RUNNER_TRIG', BRACKET.RUNNER_TRIG_PCT))

def _runner_keep() -> float:
    return float(os.environ.get('BT_RUNNER_KEEP', BRACKET.RUNNER_KEEP_FRAC))

def _runner_maxhold() -> int:
    return int(float(BRACKET.RUNNER_MAX_HOLD_DAYS))

def _trail_eod() -> float:
    if not _runner_on():
        return 0.0
    return float(os.environ.get('BT_TRAIL_EOD', BRACKET.TRAIL_EOD_PCT))


def _closethru() -> bool:
    """Close-through stop confirmation (see manage_position). Default OFF.

    The stop stays armed on every bar; this only changes the TRIGGER from an intraday
    touch to an end-of-day close through the level."""
    return os.environ.get('BT_CLOSETHRU', '0') == '1'
# ─────────────────────────────────────────────────────────────────────────────

# Silence the verbose dprint output - file logging still works, console is quiet
def dprint(*_):  # noqa: silence verbose debug output, file logging still active
    pass

def _inner_workers():
    """Worker count for the inner data-loading pools.

    When run_parallel_backtests.py fans many backtesters across cores at once,
    it sets BT_INNER_WORKERS low so 12 jobs don't each spawn a 32-proc loader
    pool (=384 procs). Standalone runs keep the old all-cores behavior."""
    v = None
    if v:
        try:
            return max(1, int(v))
        except ValueError:
            pass
    return min(32, multiprocessing.cpu_count())

_COMM_PER_SHARE = 0.0035
_COMM_MIN_ORDER = 0.35


class IBKRAdaptiveCommission(bt.CommInfoBase):

    """
    Interactive Brokers Adaptive Commission for Backtrader
    Fixed version with correct method signatures and no Unicode characters
    """

    params = (
        ('commission_per_share', _COMM_PER_SHARE),  # tiered 0.0035 | fixed 0.005
        ('min_per_order', _COMM_MIN_ORDER),         # tiered $0.35   | fixed $1.00
        ('max_per_order_pct', 0.01),       # 1.0% cap of trade value
        ('exchange_fees', 0.0002),         # $0.0002 per share for SEC/TAF/etc
        ('partial_fill_min', _COMM_MIN_ORDER),  # minimum for partial fills

        # Standard commission info params
        ('stocklike', True),               # Stock-like instrument
        ('commtype', bt.CommInfoBase.COMM_FIXED),  # Use FIXED commission type
        ('percabs', False),                # Commission is absolute, not percentage
    )

    def __init__(self):
        super(IBKRAdaptiveCommission, self).__init__()
        # Debug tracking
        self.total_commission_charged = 0.0
        self.trade_count = 0
        
        # Set commission to 0 since we'll calculate it ourselves
        self.p.commission = 0.0
    
    def calculate_commission(self, size, price):
        """Calculate commission according to IBKR tiered pricing structure"""
        abs_size = abs(size)
        per_share_comm = abs_size * self.p.commission_per_share
        exchange_fee = abs_size * self.p.exchange_fees
        order_value = abs_size * price
        value_cap = order_value * self.p.max_per_order_pct
        
        # Apply minimum commission
        base_commission = max(per_share_comm, self.p.min_per_order)
        
        # Ensure commission doesn't exceed the percentage cap
        commission = min(base_commission, value_cap) + exchange_fee
        
        return commission
    
    def _getcommission(self, size, price, pseudoexec):
        """
        Main commission calculation method that Backtrader calls
        """
        commission = self.calculate_commission(size, price)
        
        if not pseudoexec:
            # Only track real executions, not simulated ones
            self.total_commission_charged += commission
            self.trade_count += 1
            
            logging.info(f"Commission charged: ${commission:.4f} for {abs(size)} shares at ${price:.2f}")
        
        return commission
    
    def get_credit_interest(self, data, pos, dt0):
        """
        FIXED: Correct signature for Backtrader credit interest calculation
        Backtrader calls this with (data, pos, dt0) not (size, price, days, dt0, dt1)
        """
        return 0.0  # No credit interest in this model
    
    def getsize(self, price, cash):
        """Calculate position size that can be bought with available cash"""
        if price <= 0:
            return 0
        return int(cash / price)
    
    def getoperationcost(self, size, price):
        """Total cost of operation including commission"""
        return abs(size) * price + self.calculate_commission(size, price)
    
    def getvaluesize(self, size, price):
        """Return value of the operation (without commission)"""
        return abs(size) * price
    
    def getvalue(self, position, price):
        """Returns the value of a position given a price"""
        return position.size * price
    
    def get_margin(self, price):
        """Return margin needed for single item"""
        return price  # Full price for cash stocks
    
    def profitandloss(self, size, price, newprice):
        """Calculate P&L for a position"""
        return size * (newprice - price)
    
    def cashadjust(self, size, price, newprice):
        """Calculate cash adjustment for position.

        BT_CASHADJUST_FIX (default 0 = legacy, byte-identical to every prior run).

        THE DEFECT. backtrader calls cashadjust unconditionally, once per bar, for every
        open position (bbroker.py:1224) and again on close (bbroker.py:760). The guard
        that makes it a no-op for shares lives INSIDE the base method
        (comminfo.py:251: "if not self._stocklike: ... return 0.0"). This class overrides
        the method and therefore deletes the guard, so params ('stocklike', True) at line
        433 is set, correct, and completely bypassed. The unrealised move is added to cash
        every bar while the position value already carries it, and on the close the
        realised move is added a second time on top of the sale proceeds.

        Consequence: absolute return, drawdown and final equity are roughly doubled.
        Per-trade rows are unaffected (they come from the trade book, not from cash).
        Set to 1 to restore backtrader's stock semantics.
        """
        import os as _os
        if _os.environ.get('BT_CASHADJUST_FIX', '0') == '1':
            return 0.0
        return self.profitandloss(size, price, newprice)
    
    


class SimpleIBKRCommission(bt.CommInfoBase):
    """
    Simplified IBKR commission model for testing
    """
    
    params = (
        ('commission_per_share', 0.0035),
        ('min_commission', 0.35),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
        ('percabs', False),
    )
    
    def __init__(self):
        super(SimpleIBKRCommission, self).__init__()
        self.total_commission = 0.0
        self.trade_count = 0
    
    def _getcommission(self, size, price, pseudoexec):
        """Simple commission calculation"""
        abs_size = abs(size)
        commission = max(abs_size * self.p.commission_per_share, self.p.min_commission)
        
        if not pseudoexec:
            self.total_commission += commission
            self.trade_count += 1
            logging.info(f"Simple commission: ${commission:.4f} for {abs_size} shares")
        
        return commission
    
    def get_credit_interest(self, data, pos, dt0):
        """Fixed signature for credit interest"""
        return 0.0
    









def get_last_trading_date():
    """Get the last trading date from NYSE calendar.

    AS-OF OVERRIDE 2026-07-29. BT_AS_OF=YYYY-MM-DD treats that date as "today", so the
    backtest window becomes [BT_AS_OF - 400d, BT_AS_OF] and a parameter can be scored on
    a window it was not chosen on. The repo rule is that the discovery window inflates
    roughly 10x, so nothing is shippable until it clears held-out anchors.

    This used to live only in experimental/5__NightlyBackTester_asof.py. That fork is
    stale in a way that makes it useless for bracket work: it still runs the
    pre-consolidation bracket (hardcoded 3.0% TRAILING stop, +20% target via
    buy_bracket), never imports bracket_config, and has no BT_PROD_EXITS. BRACKET_STOP_PCT
    and BRACKET_TP_PCT are therefore INERT in it, so an as-of stop-width sweep run there
    would have returned identical numbers for every arm and looked like a clean null.
    Putting the override here scores as-of windows on the engine that actually trades.
    """
    _asof = os.environ.get('BT_AS_OF')
    if _asof:
        nyse = mcal.get_calendar('NYSE')
        anchor = pd.to_datetime(_asof).date()
        sched = nyse.schedule(start_date=anchor - timedelta(days=10), end_date=anchor)
        if sched.empty:
            raise Exception(f"BT_AS_OF={_asof}: no NYSE trading day in the 10 days to it.")
        return sched.index[-1].date()

    nyse = mcal.get_calendar('NYSE')
    today = datetime.now().date()

    schedule = nyse.schedule(start_date=today - timedelta(days=10), end_date=today)
    
    if schedule.empty:
        raise Exception("No trading days found in the past 10 days.")
    
    if today in schedule.index.date:
        today_market_open = schedule.loc[schedule.index.date == today, 'market_open'].iloc[0]
        
        if today_market_open.tzinfo is None:
            today_market_open = today_market_open.replace(tzinfo=timezone.utc)
        
        now_utc = datetime.now(timezone.utc)
        
        if now_utc < today_market_open:
            schedule = schedule[schedule.index.date < today]
    
    if not schedule.empty:
        last_trading_date = schedule.index[-1].date()
        return last_trading_date
    else:
        raise Exception("No trading days found in the past 10 days after excluding today.")

def get_previous_trading_day(current_date, days_back=1):
    """Get the nth previous trading day."""
    nyse = mcal.get_calendar('NYSE')
    current_date = pd.Timestamp(current_date)
    
    end_date = current_date.date()
    start_date = end_date - timedelta(days=days_back * 2)
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    
    valid_days = schedule[schedule.index.date <= end_date]
    if len(valid_days) < days_back:
        raise ValueError(f"Not enough trading days found before {end_date}")
    
    return valid_days.index[-days_back].date()

def get_next_trading_day(current_date):
    """Get the next trading day."""
    nyse = mcal.get_calendar('NYSE')
    current_date = pd.Timestamp(current_date)
    
    start_date = current_date.date() + timedelta(days=1)
    end_date = start_date + timedelta(days=10)
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    
    if schedule.empty:
        raise ValueError(f"No trading days found after {start_date}")
    
    return schedule.index[0].date()

# Optimized data loading
@njit
def calculate_up_prob_variance(up_probs):
    """Calculate variance of UpProbability using numba for speed."""
    if len(up_probs) < 10:  # Need minimum sample size
        return 0.0
    return np.var(up_probs)

def filter_stocks_by_signal_quality(directory, min_variance=0.01, min_up_prob=0.5, variance_weight=1.0):

    all_files = glob.glob(os.path.join(directory, '*.parquet'))
    quality_stocks = []
    
    logging.info(f"Evaluating {len(all_files)} stocks for signal quality...")
    
    with multiprocessing.Pool(processes=_inner_workers()) as pool:
        results = list(tqdm(
            pool.starmap(
                evaluate_stock_quality, 
                [(f, min_variance, min_up_prob, variance_weight) for f in all_files]
            ),
            total=len(all_files),
            desc="Filtering stocks"
        ))
        
    stock_quality_pairs = [(r[0], r[1]) for r in results if r is not None]
    
    stock_quality_pairs.sort(key=lambda x: x[1], reverse=True)
    
    quality_stocks = [pair[0] for pair in stock_quality_pairs]
    
    if len(stock_quality_pairs) > 0:
        top_5 = stock_quality_pairs[:5]
        bottom_5 = stock_quality_pairs[-5:] if len(stock_quality_pairs) >= 5 else stock_quality_pairs
        
        logging.info("Top 5 quality stocks:")
        for file_path, score in top_5:
            stock_name = os.path.basename(file_path).replace('.parquet', '')
            logging.info(f"  {stock_name}: Quality Score = {score:.4f}")
            
        logging.info("Bottom 5 quality stocks:")
        for file_path, score in bottom_5:
            stock_name = os.path.basename(file_path).replace('.parquet', '')
            logging.info(f"  {stock_name}: Quality Score = {score:.4f}")
    
    logging.info(f"Found {len(quality_stocks)} stocks meeting quality criteria")
    return quality_stocks

def evaluate_stock_quality(file_path, min_variance, min_up_prob, variance_weight=1.0):
    try:
        table = pq.read_table(file_path, columns=['UpProbability'])
        df = table.to_pandas()
        
        if len(df) < 60:  # Require at least 60 days of data
            return None
        
        up_probs = df['UpProbability'].values
        
        variance = calculate_up_prob_variance(up_probs)
        
        max_up_prob = np.max(up_probs)
        
        if variance < min_variance or max_up_prob < min_up_prob:
            return None
            
        norm_variance = min(variance / 0.10, 1.0)  # Cap at 1.0
        norm_max_prob = (max_up_prob - 0.5) / 0.5  # Normalize to 0-1 range
        
        quality_score = (norm_variance * variance_weight + norm_max_prob) / (1 + variance_weight)
        
        return (file_path, quality_score)
            
    except Exception as e:
        logging.error(f"Error evaluating {file_path}: {str(e)}")
    
    return None




def load_data(file_path, last_trading_date):
    """Load a single data file with basic validation, without alignment."""
    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # 52-WEEK HIGH FIX, ported verbatim from 5__NightlyBackTester.py 2026-08-26.
        # The fork was still deriving this from data.close.get(size=252) inside
        # backtrader. LineBuffer.get(size=N) slices array[idx-N+1 : idx+1], which is
        # EMPTY for any bar before idx 251, and the caller's `if closes_252 and
        # len(...)` guard then skipped the whole filter with no error. The feed is
        # truncated to 400 calendar days (~275 bars) below, so the gate was inert for
        # roughly the first 251 bars of every run and live on the rest. The fork's gate
        # was therefore not even consistent with itself across its own window, and every
        # A/B measured in it carried that.
        #
        # Needs NO new data: the parquet holds ~744 rows and the truncation happens
        # AFTER this, so the rolling high is computed on full history and carried as a
        # column. Every retained row gets a true look-back high, no partial window.
        _w52 = 252
        _hi = df['Close'].rolling(_w52, min_periods=max(20, _w52 // 3)).max()
        df['Pct52wHigh'] = (df['Close'] / _hi).astype('float64')

        # UNIVERSE-RANK SIDECAR (2026-08-26). Two per-day percentile ranks taken across
        # the WHOLE scored universe: UniRankRaw (raw_score) and UniRankVol (vol_20d).
        # Built by scratchpad/preflight/build_rankaux.py into Data/_rankaux/, which
        # reads RFpredictions and writes nowhere near it. Nothing upstream of 5__ moves.
        #
        # They cannot be derived inside backtrader: a data feed only ever sees its own
        # ticker, and a rank against today's whole cross-section is the one thing
        # sort_buy_candidates has never had. Ranking inside the candidate set instead
        # was tested first and does NOT reproduce the effect at any weight (best paired
        # t 1.65 vs 5.83), because the universe rank is day-NORMALISED: it compresses on
        # days when the top of the cross-section is bunched and spreads when there is a
        # standout, which continuously reweights the tilt by how well the model is
        # separating that day.
        #
        # Missing file or missing date leaves the neutral 0.5, which makes the tilt term
        # a constant for that name and therefore a no-op rather than a silent reorder.
        df['UniRankRaw'] = 0.5
        df['UniRankVol'] = 0.5
        try:
            _aux = os.path.join('Data', '_rankaux',
                                os.path.basename(file_path))
            if os.path.exists(_aux):
                _a = pd.read_parquet(_aux)
                _a['Date'] = pd.to_datetime(_a['Date'])
                df = df.drop(columns=['UniRankRaw', 'UniRankVol']).merge(
                    _a, on='Date', how='left')
                df['UniRankRaw'] = df['UniRankRaw'].fillna(0.5)
                df['UniRankVol'] = df['UniRankVol'].fillna(0.5)
        except Exception:
            df['UniRankRaw'] = 0.5
            df['UniRankVol'] = 0.5

        yesterday = last_trading_date
        start_date = yesterday - timedelta(days=400)
        
        df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= yesterday)]
        
        if len(df) < 252:  # Need at least 1 year of data
            logging.info(f"Skipping {file_path} due to insufficient data: {len(df)} days")
            return None
        
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'UpProbability', 'UpPrediction', 'VIX_Close', 'Pct52wHigh', 'UniRankRaw', 'UniRankVol']
        
        if all(col in df.columns for col in required_columns):
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].round(4).astype(np.float32)
            
            stock_name = os.path.basename(file_path).replace('.parquet', '')
            return (stock_name, df)
        else:
            missing_cols = [col for col in required_columns if col not in df.columns]
            logging.warning(f"Skipping {file_path} due to missing columns: {missing_cols}")

        ##in the UpProbability add an epsilon value to avoid zero values if it is under 0.01
        if 'UpProbability' in df.columns:
            df['UpProbability'] = df['UpProbability'].apply(lambda x: x if x > 0.01 else 0.01)




    except Exception as e:
        logging.error(f"Error loading {file_path}: {str(e)}")
        traceback.print_exc()

    return None



def parallel_load_data(file_paths, last_trading_date, align_start_date=False, retention_pct=95, min_days=270):
    """
    Load data files in parallel and align them by start date.
    
    Parameters:
    -----------
    file_paths : list
        List of file paths to load
    last_trading_date : datetime.date
        The last trading date to consider
    align_start_date : bool, default=True
        Whether to align all datasets to have a common start date
    retention_pct : float, default=95
        Target percentage of stocks to retain after alignment (0-100)
    min_days : int, default=252
        Minimum number of trading days required after alignment
        
    Returns:
    --------
    list of tuples: (stock_name, DataFrame) with aligned data
    """
    # Step 1: Load all data in parallel
    with multiprocessing.Pool(processes=_inner_workers()) as pool:
        results = list(tqdm(
            pool.starmap(load_data, [(fp, last_trading_date) for fp in file_paths]), 
            total=len(file_paths), 
            desc="Loading Files"
        ))
    
    loaded_data = [result for result in results if result is not None]
    
    if not align_start_date or not loaded_data:
        return loaded_data
    
    # Step 2: Analyze start date distribution
    logging.info(f"Loaded {len(loaded_data)} valid datasets. Analyzing for alignment...")
    
    # Get list of all available dates across all datasets
    all_dates = set()
    date_presence = {}  # For each date, how many datasets have it
    
    for _, df in loaded_data:
        dates = set(df['Date'].dt.date)
        all_dates.update(dates)
        for date in dates:
            date_presence[date] = date_presence.get(date, 0) + 1
    
    all_dates = sorted(all_dates)
    
    # Find dates that appear in at least retention_pct% of datasets
    min_datasets = (retention_pct / 100) * len(loaded_data)
    common_dates = [date for date, count in date_presence.items() if count >= min_datasets]
    common_dates.sort()
    
    if not common_dates:
        logging.warning(f"No dates appear in {retention_pct}% of datasets. Using most common date.")
        # Fall back to finding the most common date
        best_date = max(date_presence.items(), key=lambda x: x[1])[0]
        common_dates = [best_date]
    
    # Step 3: Find the earliest date that still gives us enough data points
    best_start_date = common_dates[0]  # Start with earliest common date
    target_len = None
    
    # Try different start dates to see which gives us the most data while keeping desired stocks
    for start_date in common_dates:
        # Calculate how many datasets would have at least min_days after this start date
        valid_datasets = []
        lengths = []
        
        for name, df in loaded_data:
            filtered_df = df[df['Date'].dt.date >= start_date]
            if len(filtered_df) >= min_days:
                valid_datasets.append((name, filtered_df))
                lengths.append(len(filtered_df))
        
        if len(valid_datasets) >= min_datasets:
            # We found a good start date, now find a common length
            if lengths:
                # Sort lengths and find the one that keeps ~95% of stocks
                lengths.sort()
                target_idx = int(len(lengths) * (retention_pct / 100))
                target_len = lengths[target_idx] if target_idx < len(lengths) else lengths[-1]
                
                # If this length is enough, use this start date
                if target_len >= min_days:
                    best_start_date = start_date
                    break
    
    if target_len is None or target_len < min_days:
        logging.warning(f"Could not find a common length of at least {min_days} days. Using {min_days}.")
        target_len = min_days
    
    # Step 4: Align all datasets to the best start date and common length
    aligned_data = []
    for name, df in loaded_data:
        # Filter to start on or after best_start_date
        aligned_df = df[df['Date'].dt.date >= best_start_date]
        
        # Check if we have enough data after applying the start date filter
        if len(aligned_df) >= target_len:
            # Trim to the common length
            aligned_df = aligned_df.iloc[:target_len]
            aligned_data.append((name, aligned_df))
        else:
            logging.info(f"Skipping {name}: Has only {len(aligned_df)} days after alignment (need {target_len})")
    
    # Step 5: Verify alignment
    start_dates = {df['Date'].dt.date.min() for _, df in aligned_data}
    lengths = {len(df) for _, df in aligned_data}
    
    if len(start_dates) == 1 and len(lengths) == 1:
        start_date = next(iter(start_dates))
        length = next(iter(lengths))
        logging.info(f"Perfect alignment achieved: {len(aligned_data)} stocks with {length} trading days")
        logging.info(f"Start date: {start_date}, End dates may vary slightly")
    else:
        logging.warning(f"Imperfect alignment: {len(start_dates)} different start dates, {len(lengths)} different lengths")
        logging.warning(f"Start dates: {start_dates}")
        logging.warning(f"Lengths: {lengths}")
    
    # Calculate what percentage of original stocks were kept
    retention_actual = (len(aligned_data) / len(loaded_data)) * 100
    logging.info(f"Retained {retention_actual:.1f}% of stocks after alignment ({len(aligned_data)} of {len(loaded_data)})")
    
    return aligned_data






def read_trading_data():
    """RETIRED: the per-ticker state ledger is no longer persisted (it used to
    pollute _Buy_Signals.parquet, which is now exclusively the broker's narrowed
    book). Returns an empty ledger-schema frame so legacy callers keep working
    without touching any file. write_trading_data() is a no-op; per-ticker state
    does not carry across runs."""
    return pd.DataFrame(columns=[
        'Symbol', 'LastBuySignalDate', 'LastBuySignalPrice', 'IsCurrentlyBought',
        'ConsecutiveLosses', 'LastTradedDate', 'UpProbability', 'LastSellPrice', 'PositionSize'
    ])




def write_trading_data(df):
    dtype_schema = {
        'Symbol': 'string',
        'LastBuySignalPrice': 'float64',
        'IsCurrentlyBought': 'bool',
        'ConsecutiveLosses': 'int64',
        'UpProbability': 'float64',
        'LastSellPrice': 'float64',
        'PositionSize': 'float64'
    }
    
    df = df.copy()
    
    date_columns = ['LastBuySignalDate', 'LastTradedDate']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    for col, dtype in dtype_schema.items():
        if col not in df.columns:
            if dtype == 'float64':
                df[col] = pd.Series(dtype='float64')
            elif dtype == 'int64':
                df[col] = pd.Series(dtype='int64')
            elif dtype == 'bool':
                df[col] = pd.Series(dtype='bool')
            elif dtype == 'string':
                df[col] = pd.Series(dtype='string')
        else:
            if dtype == 'float64':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            elif dtype == 'int64':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
            elif dtype == 'bool':
                if df[col].dtype != 'bool':
                    df[col] = df[col].astype('bool')
            elif dtype == 'string':
                if df[col].dtype != 'string':
                    df[col] = df[col].astype('string')
    
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    for col in datetime_cols:
        df[col] = df[col].astype('object').where(df[col].notnull(), None)
    # _Buy_Signals.parquet retired - Z_signals.parquet is the canonical signals file

def update_buy_signal(symbol, date, price, up_probability):
    try:
        price = round(float(price), 4)
        up_probability = round(float(up_probability), 4)
        
        df = read_trading_data()
        
        new_data = pd.DataFrame([{
            'Symbol': str(symbol),
            'LastBuySignalDate': pd.Timestamp(date),
            'LastBuySignalPrice': price,
            'IsCurrentlyBought': False,
            'ConsecutiveLosses': 0,
            'LastTradedDate': pd.NaT,
            'UpProbability': up_probability,
            'LastSellPrice': float('nan'),
            'PositionSize': float('nan')
        }])
        
        # Set proper dtypes for the new DataFrame
        new_data = new_data.astype({
            'Symbol': 'string',
            'LastBuySignalPrice': 'float64',
            'IsCurrentlyBought': 'bool',
            'ConsecutiveLosses': 'int64',
            'UpProbability': 'float64',
            'LastSellPrice': 'float64',
            'PositionSize': 'float64'
        })
        
        new_data['LastBuySignalDate'] = pd.to_datetime(new_data['LastBuySignalDate'])
        new_data['LastTradedDate'] = pd.to_datetime(new_data['LastTradedDate'])
        
        df = df[df['Symbol'] != symbol]
        
        for col in new_data.columns:
            if col in df.columns and col not in ['LastBuySignalDate', 'LastTradedDate']:
                df[col] = df[col].astype(new_data[col].dtype)
        
        df = pd.concat([df, new_data], ignore_index=True)
        
        write_trading_data(df)
        
        logging.info(f"Updated buy signal for {symbol} at price {price}")
        
    except Exception as e:
        logging.error(f"Error in update_buy_signal for {symbol}: {str(e)}")
        raise

def mark_position_as_bought(symbol, position_size):
    df = read_trading_data()
    df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = True
    df.loc[df['Symbol'] == symbol, 'PositionSize'] = position_size
    write_trading_data(df)

def update_trade_result(symbol, is_loss, exit_price=None, exit_date=None):
    df = read_trading_data()

    if symbol in df['Symbol'].values:
        if is_loss:
            df.loc[df['Symbol'] == symbol, 'ConsecutiveLosses'] += 1
        else:
            df.loc[df['Symbol'] == symbol, 'ConsecutiveLosses'] = 0
        
        df.loc[df['Symbol'] == symbol, 'LastTradedDate'] = pd.Timestamp(exit_date or datetime.now().date())
        if exit_price is not None:
            df.loc[df['Symbol'] == symbol, 'LastSellPrice'] = exit_price
        df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = False
        write_trading_data(df)







def get_dynamic_signal_columns(index_folder="Data/Indexes"):
    """Scan the folder for parquet files and collect corr/alpha/beta column names."""
    corr_cols, alpha_cols, beta_cols = set(), set(), set()

    # Find all parquet files in the folder
    parquet_files = glob.glob(os.path.join(index_folder, "*.parquet"))

    for file in parquet_files:
        try:
            df = pd.read_parquet(file, engine="pyarrow")
            cols = df.columns.str.lower()  # lower for consistent comparison

            corr_cols.update([col for col in df.columns if "corr" in col.lower()])
            alpha_cols.update([col for col in df.columns if "alpha" in col.lower()])
            beta_cols.update([col for col in df.columns if "beta" in col.lower()])
        except Exception as e:
            print(f" Skipping {file}: {e}")

    return sorted(corr_cols), sorted(alpha_cols), sorted(beta_cols)

class EnhancedPandasData(bt.feeds.PandasData):
    """Enhanced PandasData class that includes ML signals and technical indicators."""

    # Base lines always present.
    #
    # PERF 2026-07-24: 'VIX_Close' and 'atr' were removed here after a repo-wide
    # grep showed neither is ever read off the feed:
    #   - VIX_Close appeared only in this declaration and in required_columns
    #     (a data-quality check, deliberately kept -- the predictor still has to
    #     emit the column).
    #   - the 'atr' line was mapped to None, so it held NaN for the entire run;
    #     the ATR the strategy actually uses is the bt.indicators.ATR built in
    #     __init__ into self.inds[d]['atr'].
    # Every declared line costs a buffer fill at preload plus an advance() and a
    # tick_* setattr per bar, per data -- ~4250 datas x ~250 bars, so two dead
    # lines were ~20% of all per-bar line work. Do not re-add without a reader.
    # 'Pct52wHigh' added 2026-08-26 to match shipped. It HAS a reader (can_buy's
    # 52-week filter), which is the bar for adding a line here. Precomputed in
    # load_data() from the untruncated history, because deriving it from this buffer is
    # what silently disabled the gate on ~88.5% of evaluations.
    base_lines = ('UpProbability', 'Pct52wHigh', 'UniRankRaw', 'UniRankVol')

    # Dynamically add corr/alpha/beta lines from parquet data
    corr_cols, alpha_cols, beta_cols = get_dynamic_signal_columns()

    lines = base_lines + tuple(corr_cols) + tuple(alpha_cols) + tuple(beta_cols)

    # Base params for PandasData fields
    base_params = (
        ('datetime', 'Date'),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
        ('UpProbability', 'UpProbability'),
        ('Pct52wHigh', 'Pct52wHigh'),
        ('UniRankRaw', 'UniRankRaw'),
        ('UniRankVol', 'UniRankVol'),
    )

    # Combine base + dynamic columns into params
    params = base_params + tuple((col, col) for col in corr_cols + alpha_cols + beta_cols)

    # ------------------------------------------------------------------
    # PERF 2026-07-24: numpy-backed, bulk-preloaded feed.
    #
    # Stock backtrader fills a feed one bar at a time: cerebro calls
    # data.preload(), which calls load() per bar, which calls _load(), which
    # does `dataname.iloc[row, col]` for EVERY mapped line. Each of those
    # builds a pandas Series (_ixs -> _box_col_values -> __finalize__). At this
    # project's scale (~4250 tickers x ~720 bars x 9 mapped lines) that is
    # ~27M pandas scalar lookups per backtest, and it profiled as the single
    # biggest cost in the loop (~57%, 2026-06-14).
    #
    # Because cerebro preloads (default preload=True; the run loop then only
    # advance()s the pointer and never calls _load() again), the whole feed can
    # be materialised in ONE numpy -> array('d') copy per line instead of a
    # Python call per bar. preload() below does that.
    #
    # FAITHFULNESS: the fast path is guarded. Anything that makes the stock
    # per-bar load() loop semantically load-bearing -- filters, tzinput
    # conversion, resampling/replay, a bounded QBuffer (exactbars), a bar
    # stack, or non-numeric columns -- falls back to super().preload() and
    # behaves exactly as before. start()/_load() stay numpy-backed so the
    # preload=False path produces identical values without pandas too.
    #
    # The library PandasData class is untouched; all of this lives on our
    # subclass.
    #
    # KILL SWITCH: set BT_FEED_FAST=0 to run the stock backtrader feed path
    # end-to-end (both preload and per-bar _load). Kept so the optimisation can
    # be A/B'd for timing and ruled out instantly if a feed ever misbehaves.
    # ------------------------------------------------------------------

    _FEED_FAST = True

    # date2num for a tz-naive midnight timestamp is exactly float(toordinal()),
    # and toordinal() == days-since-epoch + 719163. Integer arithmetic below
    # 2**53, so the vectorised form is bit-identical -- but only under those
    # conditions, which _np_datetimes() checks before using it.
    _ORDINAL_EPOCH_OFFSET = 719163
    _NS_PER_DAY = 86400000000000

    def _np_datetimes(self, series_or_index):
        """date2num for every row, vectorised when provably exact."""
        try:
            vals = pd.DatetimeIndex(series_or_index)
            if vals.tz is None:
                i8 = vals.asi8
                if len(i8) == 0 or not (i8 % self._NS_PER_DAY).any():
                    # All bars land exactly on midnight -> exact integer path.
                    return (i8 // self._NS_PER_DAY
                            + self._ORDINAL_EPOCH_OFFSET).astype('float64')
        except Exception:
            pass

        # Intraday / tz-aware / anything unexpected: use backtrader's own
        # scalar conversion, one call per bar (still once per run, not per bar
        # of the hot loop).
        return np.fromiter(
            (bt.date2num(ts.to_pydatetime()) for ts in series_or_index),
            dtype='float64', count=len(series_or_index))

    def start(self):
        super(EnhancedPandasData, self).start()
        if not self._FEED_FAST:
            return
        df = self.p.dataname

        # After super().start(), _colmapping values are integer column
        # positions, or None for lines with no source column (e.g. 'atr',
        # which the stock path leaves as the NaN that forward() wrote).
        self._np_cols = {}
        for datafield in self.getlinealiases():
            if datafield == 'datetime':
                continue
            colindex = self._colmapping.get(datafield)
            if colindex is None:
                self._np_cols[datafield] = None
                continue
            self._np_cols[datafield] = np.ascontiguousarray(
                df.iloc[:, colindex].to_numpy(dtype='float64'))

        coldtime = self._colmapping['datetime']
        self._np_dt = self._np_datetimes(
            df.index if coldtime is None else df.iloc[:, coldtime])
        self._np_len = len(self._np_dt)

        # Flat list for the per-bar fallback; O(1) indexing, no dict lookups.
        self._np_fields = [(getattr(self.lines, k), v)
                           for k, v in self._np_cols.items() if v is not None]

    def _load(self):
        # Only reached when cerebro runs with preload=False. Same values as the
        # stock _load(), minus pandas.
        if not self._FEED_FAST:
            return super(EnhancedPandasData, self)._load()
        self._idx += 1
        i = self._idx
        if i >= self._np_len:
            return False
        for line, arr in self._np_fields:
            line[0] = arr[i]
        self.lines.datetime[0] = self._np_dt[i]
        return True

    def preload(self):
        if not self._bulk_preload():
            super(EnhancedPandasData, self).preload()

    def _bulk_preload(self):
        """Fill every line buffer in one shot.

        Returns False (without touching any state) if this feed is not in the
        plain configuration the fast path is valid for, so the caller can fall
        back to the stock per-bar implementation.
        """
        if not self._FEED_FAST:
            return False
        if self._filters or self._ffilters or self._barstack or self._barstash:
            return False
        if self._tzinput:
            return False
        if getattr(self, 'resampling', 0) or getattr(self, 'replaying', 0):
            return False

        aliases = self.getlinealiases()
        lines = [getattr(self.lines, a) for a in aliases]
        if self.lines.size() != len(aliases):
            return False
        # QBuffer (exactbars) backs lines with a bounded deque, not array('d').
        if any(l.mode != bt.linebuffer.LineBuffer.UnBounded for l in lines):
            return False

        n = self._np_len
        dts = self._np_dt

        # Reproduce load()'s date handling: it BREAKS at the first bar past
        # todate (truncate) and DISCARDS-and-continues bars before fromdate.
        if self.todate != float('inf'):
            over = np.flatnonzero(dts[:n] > self.todate)
            if over.size:
                n = int(over[0])
        keep = None
        if self.fromdate != float('-inf'):
            mask = dts[:n] >= self.fromdate
            if not mask.all():
                keep = np.flatnonzero(mask)

        for alias, line in zip(aliases, lines):
            if alias == 'datetime':
                vals = dts[:n]
            else:
                src = self._np_cols.get(alias)
                vals = np.full(n, float('nan')) if src is None else src[:n]
            if keep is not None:
                vals = vals[keep]
            buf = array.array('d')
            buf.frombytes(np.ascontiguousarray(vals, dtype='float64').tobytes())
            line.array = buf

        # Leave the feed exactly as stock preload() does: buffer full, logical
        # index rewound to the start. home() sets idx=-1/lencount=0 per line;
        # buflen() then reports the full buffer, which is what next() uses to
        # tell "preloaded" from "needs load()".
        self.home()
        # A later _load() must report exhaustion, as it would after a per-bar
        # preload that ran off the end.
        self._idx = self._np_len - 1
        return True









class Rule201Monitor:
    def __init__(self, threshold=-9.99, cooldown_days=1):
        self.threshold = threshold
        self.cooldown_days = cooldown_days
        self.violations = set()
        self.trigger_dates = {}
        self.triggerCount = 0
    
    def check_rule_201(self, symbol, prev_close, current_price, current_date):
        if prev_close <= 0:
            return False
            
        daily_return = (current_price / prev_close - 1) * 100
        
        if daily_return <= self.threshold:
            self.violations.add(symbol)
            self.trigger_dates[symbol] = current_date
            self.triggerCount += 1
            return True
        return False
    
    def clear_expired_restrictions(self, current_date):
        expired = []
        
        for symbol, trigger_date in self.trigger_dates.items():
            days_since = (current_date - trigger_date).days
            if days_since > self.cooldown_days:
                expired.append(symbol)
                logging.info(f"Rule 201 cooldown expired for {symbol}")
        
        for symbol in expired:
            self.violations.discard(symbol)
            del self.trigger_dates[symbol]
    
    def is_restricted(self, symbol):
        """Check if a symbol is currently restricted under Rule 201."""
        return symbol in self.violations



class PositionSizer:
    def __init__(self, risk_per_trade=0.6, max_position_pct=0.06, min_position_pct=0.025, reserve_pct=0.20):
        self.risk_per_trade = risk_per_trade  # Risk per trade in percent of account
        self.max_position_pct = max_position_pct  # Maximum position size as % of account
        self.min_position_pct = min_position_pct  # Minimum position size as % of account
        self.reserve_pct = reserve_pct  # Cash reserve percentage
    
    def calculate_position_size(self, account_value, cash, price, atr, max_positions):
        # Target 80% of account invested (20% cash buffer)
        workable_capital = account_value * (1.0 - self.reserve_pct)
        
        # Ensure adequate cash buffer before taking positions
        if cash < account_value * self.reserve_pct:
            return 0
            
        # Calculate position size based on risk management
        risk_amount = account_value * (self.risk_per_trade / 100.0)
        risk_per_share = 2.0 * atr
        
        if risk_per_share <= 0.01:
            risk_per_share = 0.01
            
        shares_by_risk = risk_amount / risk_per_share
        
        # Position limits based on account percentages
        min_position = (account_value * self.min_position_pct) / price
        max_position = (account_value * self.max_position_pct) / price
        
        # Use risk-based sizing within percentage bounds
        position_size = np.clip(shares_by_risk, min_position, max_position)
        
        # Ensure portfolio diversification across max_positions
        max_per_position = workable_capital / (price * (BT_SIZE_DIV if BT_SIZE_DIV > 0 else max_positions))
        position_size = min(position_size, max_per_position)
        position_size = int(position_size)
        
        # Final liquidity check
        required_capital = position_size * price
        if required_capital > cash * 0.90:  # Keep 10% cash buffer within available cash
            position_size = int(cash * 0.90 / price)
            
        return max(0, position_size)


# ─────────────────────────────────────────────────────────────────────────────
# PLUGGABLE POSITION SIZERS  (added 2026-07-28)
#
# The class above (PositionSizer) is the PRODUCTION policy and is deliberately
# left byte-for-byte untouched. Selecting sizer 'default' constructs exactly that
# object, so the default code path is bit-identical -- verified empirically by
# diffing Data/TradeHistory.parquet across a pinned-seed run before/after.
#
# Selection:  --sizer <name>   (CLI, wins)   or   BT_SIZER=<name>   (env)
# Names:      default | equal_weight | inverse_vol
#
# Experimental variants subclass _GuardedSizer and implement _target_notional().
# _GuardedSizer applies the SAFETY guardrails to every variant so an experiment
# can never emit an unfillable or negative order:
#     1. entry gate      -- refuse to open unless cash >= account_value*reserve_pct
#     2. per-name cap    -- notional <= account_value * max_position_pct
#     3. deployment cap  -- notional <= workable capital (account*(1-reserve_pct))
#     4. cash clamp      -- notional <= cash * 0.90 (same 10% intra-cash buffer
#                           the production sizer uses)
#     5. integer shares, never negative
# NOTE: min_position_pct is *not* applied as a guardrail floor, because with the
# live Util.STRATEGY_PARAMS values (min_position_pct = 20, i.e. 2000% of equity)
# a floor would force every variant straight into the deployment cap and every
# knob would be dead. It stays inside PositionSizer where it belongs. See the
# report note about that value making the production ATR/risk leg inert.
# ─────────────────────────────────────────────────────────────────────────────

def _active_sizer_name():
    """Resolve the active sizer name. Read LAZILY (at strategy construction),
    not at import, because --sizer publishes into BT_SIZER during arg parsing.
    BT_SIZING is accepted as an alias for continuity with the deleted
    5.1b__NightlyBackTester_equalwt.py rig (project_sizer_alignment_verdict_2026_06_28)."""
    return 'default'


# Aliases keep one sizing vocabulary across studies: the 2026-06-28 A/B called the
# arms 'inverse-vol' / 'equal-weight' and used BT_SIZING=equal.
_SIZER_ALIASES = {
    'equal': 'equal_weight',
    'equal-weight': 'equal_weight',
    'equalwt': 'equal_weight',
    'eq': 'equal_weight',
    'invvol': 'inverse_vol',
    'inverse-vol': 'inverse_vol',
    'iv': 'inverse_vol',
}


class _GuardedSizer:
    """Base class for experimental sizers: policy picks a notional, guardrails clamp it."""

    name = '_guarded'

    def __init__(self, risk_per_trade=0.6, max_position_pct=0.06, min_position_pct=0.025, reserve_pct=0.20):
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.reserve_pct = reserve_pct

    def _target_notional(self, account_value, cash, price, atr, max_positions):
        raise NotImplementedError

    def calculate_position_size(self, account_value, cash, price, atr, max_positions):
        if not np.isfinite(price) or price <= 0 or not np.isfinite(account_value) or account_value <= 0:
            return 0
        # (1) entry gate -- identical semantics to the production sizer
        if cash < account_value * self.reserve_pct:
            return 0

        notional = self._target_notional(account_value, cash, price, atr, max_positions)
        if not np.isfinite(notional) or notional <= 0:
            return 0

        # (2) per-name cap, (3) deployment cap
        notional = min(notional, account_value * self.max_position_pct)
        notional = min(notional, account_value * (1.0 - self.reserve_pct))
        # (4) cash clamp -- 10% buffer within available cash
        notional = min(notional, cash * 0.90)

        # (5) integer shares, never negative
        return max(0, int(notional / price))


class EqualWeightSizer(_GuardedSizer):
    """CONTROL. Textbook 1/N: every slot gets the same fraction of equity.

    Expected to LOSE -- a prior 16-seed full-sim study refuted equal weight and
    favored band+inverse-vol. It is here as a connectivity control, not a
    candidate. Uses account_value/max_positions (1/N of equity) rather than
    workable_capital/max_positions, so it is measurably distinct from the
    production path (which collapses onto the workable-capital slot cap)."""

    name = 'equal_weight'

    def _target_notional(self, account_value, cash, price, atr, max_positions):
        n = max(1, int(max_positions))
        return account_value / n


class InverseVolSizer(_GuardedSizer):
    """Band + inverse-vol: notional fraction proportional to 1/vol, clipped to a band.

    Reuses the vol input the production sizer already receives (ATR and price are
    both already arguments) -- vol_pct = atr/price. No new data plumbing.
    Constants match the prior sizing experiment (Data/_sizing_exp/bt_sizeexp.py)
    so the arm is comparable to that study; they are NOT tuned here."""

    name = 'inverse_vol'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.target = 0.0018
        self.frac_min = 0.03
        self.frac_max = 0.15

    def _target_notional(self, account_value, cash, price, atr, max_positions):
        vol_pct = (atr / price) if price > 0 else 0.02
        if not np.isfinite(vol_pct) or vol_pct <= 0.002:
            vol_pct = 0.02  # floor: a tiny/NaN vol must not blow the size up
        frac = float(np.clip(self.target / vol_pct, self.frac_min, self.frac_max))
        notional = account_value * frac
        # keep the production diversification cap so the book still spreads over slots
        n = max(1, int(max_positions))
        return min(notional, account_value * (1.0 - self.reserve_pct) / n)


_SIZER_REGISTRY = {
    EqualWeightSizer.name: EqualWeightSizer,
    InverseVolSizer.name: InverseVolSizer,
}


def make_position_sizer(name=None, **kwargs):
    """Factory. 'default'/None -> the untouched production PositionSizer.

    Unknown names raise instead of silently falling back: this repo has been
    burned by disconnected knobs that looked implemented and changed nothing."""
    key = (name if name is not None else _active_sizer_name())
    key = (key or 'default').strip().lower()
    key = _SIZER_ALIASES.get(key, key)
    if key in ('', 'default', 'prod', 'production', 'live'):
        return PositionSizer(**kwargs)
    cls = _SIZER_REGISTRY.get(key)
    if cls is None:
        raise SystemExit(
            f"Unknown position sizer '{name}'. Valid: default, "
            + ", ".join(sorted(_SIZER_REGISTRY))
        )
    return cls(**kwargs)







class TradeRecorder:
    def __init__(self, filename=None):
        filename = filename or V2_TRADE_HISTORY
        self.filename = filename
        self.trades = []
        
    def record_trade(self, trade_data):
        """Record a trade with detailed metadata."""
        self.trades.append(trade_data)
        
    def save_trades(self):
        """Save all recorded trades to a parquet file."""
        if not self.trades:
            logging.info("No trades to save")
            return
            
        df = pd.DataFrame(self.trades)
        
        numeric_cols = [
            'EntryPrice', 'ExitPrice', 'Quantity', 'PnL', 'PnLPct',
            'Commission', 'EntryCommission', 'Slippage', 'ATR', 'UpProbability',
            'EntryUpProbability'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        date_cols = ['EntryDate', 'ExitDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        df.to_parquet(self.filename, index=False)
        logging.info(f"Saved {len(self.trades)} trades to {self.filename}")














class StockSniperStrategy(bt.Strategy):
    # Use the parameters from STRATEGY_PARAMS_TUPLE
    params = STRATEGY_PARAMS
    
    def __init__(self):
        self.inds = {d: {} for d in self.datas}
        for d in self.datas:
            self.inds[d]['atr'] = bt.indicators.ATR(d, period=self.p.atr_period)
            # FAST: removed up_prob_ma3/ma5/roc - computed on every feed every bar but
            # never referenced anywhere. ATR-14 still sets the warmup, so results are
            # identical. (lossless ~indicator-cost reduction)
            self.inds[d]['up_prob'] = d.UpProbability



        # Rest of your initialization code remains the same
        self.order_list = []  # Track pending orders
        self.signal_metadata = {}
        self.bracket_orders = {}
        self.entry_prices = {}  # Track entry prices for positions
        self.position_dates = {}  # Track entry dates for positions
        self.position_bars = {}   # Bar index (len(data)) at signal, for BT_CLOCK_SESSIONS
        # INSTRUMENT FIX 2026-07-30. UpProbability was only ever stamped onto the trade
        # row at EXIT (handle_sell_execution reads data.UpProbability[0] on the exit
        # bar), which makes the report's Signal Rank-IC and UpProb bucket table partly
        # circular: losers are held while the model's probability decays, so the
        # correlation is manufactured by the holding period, not by the signal. This
        # dict carries the probability the ENTRY decision was actually made on and is
        # written to the trade parquet as EntryUpProbability. Read-only capture, so it
        # cannot change the book.
        self.entry_up_prob = {}  # data -> UpProbability at entry-signal time
        self.stop_loss_orders = {}  # Track stop loss orders
        self.take_profit_orders = {}  # Track take profit orders
        self.trailing_stops = {}  # Track trailing stop levels (legacy path only)
        self.stop_levels = {}     # Hard-stop level per position -- READ by
                                  # determine_exit_reason. Before 2026-07-28 no dict
                                  # was ever populated, so "Stop Loss" was unreachable.
        self.pending_entry_limits = {}  # data -> (order, bar) for a resting entry
                                   # LIMIT; cancelled one bar later (limit-entry expt).
        self._limit_sent = 0        # limit-entry census
        self._limit_filled = 0
        self._limit_expired = 0
        self._limit_overfill_days = 0   # days the book held > max_positions
        self._limit_maxheld = 0
        self._limit_rejected = 0
        self._limit_fallback_size = {}   # data -> size awaiting a market fallback
        self._limit_fellback = 0
        self._limit_trigger = {}     # data -> (limit_px, signal_close, depth)
        self._limit_log = []         # one record per limit fill, for the report
        self._mkt_fill_due = 0       # slots the triggers failed to fill yesterday
        self._mkt_filled = 0
        self.pending_bracket = {}  # data -> bracket intent, attached in
                                   # handle_buy_execution once the real fill is known.
        self.deferred_exits = set()  # datas whose model-driven exit had to wait a bar
        self.runner_done = set()     # datas already scaled out (runner exits)
        # --- EOD-repeg instrumentation (fork only). The live broker ratchets ~74% of
        # exits; the backtest reports ~3.7%. These counters localise where it is lost.
        self._pool_export_mode = False
        self._eod_fired = 0        # repeg actually raised a resting stop
        self._eod_blk_target = 0   # skipped: a take-profit sibling exists
        self._eod_blk_nostop = 0   # skipped: no live stop leg to reprice
        self._eod_blk_notraise = 0 # skipped: new level not above the old one
        self._eod_blk_cancel = 0   # skipped: cancel did not clear the leg this bar
        self._eod_rearmed = 0      # stop re-created after an async cancel cleared
        self._eod_nobr = 0         # nostop sub-case: no bracket record at all
        self._eod_nosize = 0       # nostop sub-case: bracket exists, flat position
        self._eod_cancel_datas = {}  # data -> times the cancel stalled
        self._pyr = dict(att=0, filled=0, sizer0=0, margin=0, expired=0, guard=0, orphan=0,
                         resize=0, mismatch=0, done=set(), orders={}, fillbar={})
                                     # because their bracket legs were not cancellable
                                     # yet. See exit_position() for why.
        self.exit_orders = {}        # data -> the close() order currently working, so a
                                     # second exit trigger on the same bar cannot stack
                                     # another sell on top of it.
        self.asset_groups = {}  # Track asset groups for correlation
        self.group_allocations = {}  # Track group allocations
        
        self.trade_history = []  # Detailed trade history for analysis
        self.winning_trades = 0
        self.losing_trades = 0
        self.breakeven_trades = 0
        self.total_win_pnl = 0.0
        self.total_loss_pnl = 0.0
        self.longest_win_streak = 0
        self.longest_loss_streak = 0
        self.current_win_streak = 0
        self.current_loss_streak = 0
        self.recent_outcomes = []  # Store last 10 trade outcomes (1=win, 0=breakeven, -1=loss)
        self.trade_pct_returns = []  # For rolling Sharpe calculation
        
        self.trade_recorder = TradeRecorder(V2_TRADE_HISTORY)
        self._gate_hits = {}      # first-reject census, printed by _gate_census()

        self.open_positions = 0

        # Per-day capital-efficiency tracking: how many of the max_positions
        # slots are filled, and how much of the book is actually deployed vs
        # sitting in cash. Sampled once per trading day at end of next().
        self.daily_positions = []     # open-position count, ONE sample per trading day
        self.daily_deployment = []    # invested capital / total equity (%), same cadence
        self.daily_cap_dates = []     # the date each sample belongs to (dedupe proof)
        self._cap_last_date = None    # guard: next() fires many times per calendar day
        self._next_calls = 0          # raw next() count, to expose the sampling ratio

        self.correlation_df = pd.read_parquet('Correlations.parquet')
        logging.info(f"Loaded correlation dataframe with columns: {list(self.correlation_df.columns)}")

        if 'Ticker' in self.correlation_df.columns:
            self.correlation_df_by_ticker = self.correlation_df.copy()
            self.correlation_df.set_index('Ticker', inplace=True)
            logging.info("Set 'Ticker' column as index in correlation dataframe")
        else:
            verbose = False
            if verbose:
                logging.warning("'Ticker' column not found in correlation dataframe. Available columns: " 
                               f"{list(self.correlation_df.columns)}")

        self.total_groups = self.correlation_df['Cluster'].nunique()
        self.group_allocations = {group: 0 for group in range(self.total_groups)}
        
        # Use the parameters from Util for PositionSizer.
        # PLUGGABLE (2026-07-28): --sizer / BT_SIZER selects the policy. The
        # default resolves to PositionSizer(...) with exactly these kwargs, so
        # the production path is bit-identical to before the seam existed.
        _sizer_name = _active_sizer_name()
        self.position_sizer = make_position_sizer(
            _sizer_name,
            risk_per_trade=self.p.risk_per_trade_pct,
            max_position_pct=self.p.max_position_pct,
            min_position_pct=self.p.min_position_pct,
            reserve_pct=self.p.reserve_percent
        )
        if not isinstance(self.position_sizer, PositionSizer):
            # Loud, because an experimental sizer must never run unnoticed.
            logging.warning(f"[BT_SIZER] EXPERIMENTAL position sizer ACTIVE: {_sizer_name} "
                            f"({type(self.position_sizer).__name__}) -- results are NOT the production config")
            print(f"[BT_SIZER] EXPERIMENTAL position sizer ACTIVE: {_sizer_name}")
        
        # Use the parameters from Util for Rule201Monitor
        self.rule_201_monitor = Rule201Monitor(
            threshold=self.p.rule_201_threshold,
            cooldown_days=self.p.rule_201_cooldown
        )

        # RE-ENTRY COOLDOWN (experimental, 2026-07-31). BT_REENTRY_COOLDOWN_D=<days>
        # blocks buying a symbol for that many calendar days after its last exit.
        # BT_REENTRY_MODE: 'loss' (default) only blocks when that exit was a loss;
        # 'all' blocks after any exit. 0/unset = off, byte-identical to before.
        self.reentry_cooldown_d = 0
        self.reentry_mode = 'loss'
        # BT_REENTRY_MIN_D: start of the blocked window (default 0 = block from the
        # exit day). MIN_D=3 with COOLDOWN_D=7 blocks only gaps of 3..7 days, leaving
        # fast (<=2d) re-entries alone -- the census-aligned "quarantine window" shape.
        self.reentry_min_d = 0
        # BT_REENTRY_SOFT_PENALTY: when set (with the window/mode envs), matching
        # candidates are NOT hard-blocked; their selection score is docked by this
        # amount instead, so they lose the slot only to a near-equal alternative.
        self.reentry_soft_penalty = 0.0
        self._reentry_penalties = 0
        self._last_exit = {}          # symbol -> (exit_date, was_loss, was_stop, exit_px, entry_up)
        self._reentry_blocks = 0      # census: distinct (symbol, day) evaluations blocked
        self._reentry_blocked_keys = set()
        if self.reentry_cooldown_d > 0:
            logging.warning(f"[BT_REENTRY] EXPERIMENTAL re-entry cooldown ACTIVE: "
                            f"{self.reentry_cooldown_d}d mode={self.reentry_mode}")
            print(f"[BT_REENTRY] EXPERIMENTAL re-entry cooldown ACTIVE: "
                  f"{self.reentry_cooldown_d}d mode={self.reentry_mode}")
        
        self.last_trading_date = get_last_trading_date() 
        self.second_last_trading_date = get_previous_trading_day(self.last_trading_date)
        self.trading_lockup_start = get_previous_trading_day(self.last_trading_date, self.p.lockup_days)
        
        self.positions_cleared_for_lockup = False
        self.trading_locked = False
        self.last_logged_date = None
        
        self.monthly_performance = {}  # {YYYY-MM: percent_return}
        self.yearly_performance = {}   # {YYYY: percent_return}
        self.last_month_equity = None
        self.last_year_equity = None
        self.month_high_equity = None
        self.month_low_equity = None
        self.current_month = None
        self.current_year = None
        
        self.day_count = 0
        self.total_bars = 252  # Expected trading days in backtest
        self.progress_bar = tqdm(
            total=self.total_bars,
            desc="Strategy Progress",
            unit="day",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            ncols=100
        )
        

    def _reentry_condition(self, symbol, current_date, data=None):
        """True when the symbol matches the configured re-entry window+mode.
        Modes: all / loss / stop / slowloss (lost via a NON-stop exit, the
        slow-bleed pocket) / higher / lower (vs last exit price)."""
        if self.reentry_cooldown_d <= 0:
            return False
        last = self._last_exit.get(symbol)
        if last is None:
            return False
        exit_date, was_loss, was_stop, exit_price, _prev_up = last
        mode = self.reentry_mode
        if mode == 'loss' and not was_loss:
            return False
        if mode == 'stop' and not was_stop:
            return False
        if mode == 'slowloss' and not (was_loss and not was_stop):
            return False
        if mode in ('higher', 'lower'):
            try:
                px = float(data.close[0]) if data is not None else None
            except (IndexError, TypeError, ValueError):
                px = None
            if px is None or not exit_price or exit_price <= 0:
                return False
            if mode == 'higher' and px <= exit_price:
                return False
            if mode == 'lower' and px >= exit_price:
                return False
        try:
            gap = (current_date - exit_date).days
        except TypeError:
            return False
        return self.reentry_min_d <= gap <= self.reentry_cooldown_d

    def _reentry_blocked(self, symbol, current_date, data=None):
        """Hard entry block for the configured re-entry condition. Off by
        default; also inert when BT_REENTRY_SOFT_PENALTY is set (the condition
        then acts through the selection score instead of a block, so the trade
        survives whenever the alternative candidate is much worse)."""
        if self.reentry_soft_penalty:
            return False
        if not self._reentry_condition(symbol, current_date, data):
            return False
        key = (symbol, current_date)
        if key not in self._reentry_blocked_keys:
            self._reentry_blocked_keys.add(key)
            self._reentry_blocks += 1
        return True

    def _run_rule_201_checks(self):
        """Rule 201 scan: flag names down >=10% (today's open vs prior close) and
        expire finished cooldowns.

        FIX 2026-07-29: called from BOTH prenext() and next(). This used to live
        only in prenext(), which backtrader stops calling once the ATR-14 warmup
        completes, so ~14 bars into the run new violations stopped being detected
        (can_buy kept buying names the live broker would be restricted on) AND
        clear_expired_restrictions stopped running (anything flagged during warmup
        stayed banned for the rest of the backtest)."""
        current_date = self.datetime.date()

        self.rule_201_monitor.clear_expired_restrictions(current_date)

        for d in self.datas:
            if len(d) > 1:  # Need at least 2 data points
                symbol = d._name
                prev_close = d.close[-1]
                current_price = d.open[0]

                self.rule_201_monitor.check_rule_201(
                    symbol, prev_close, current_price, current_date
                )

    def prenext(self):
        """Run Rule 201 checks during warmup bars; next() takes over after warmup."""
        self._run_rule_201_checks()
    








    def next(self):
        """Main strategy logic executed on each bar."""
        current_date = self.datetime.date()

        if BT_LIMIT_ENTRY_K > 0:
            _h = sum(1 for _d in self.datas if self.getposition(_d).size > 0)
            if _h > self._limit_maxheld:
                self._limit_maxheld = _h
            if _h > self.p.max_positions:
                self._limit_overfill_days += 1

        # LIMIT-ENTRY: an entry limit gets exactly one bar. The broker has already
        # matched orders against THIS bar before next() runs, so anything still
        # open here did not fill and is cancelled. handle_order_failure() then
        # releases the slot, exactly as it does for a margin rejection.
        if self.pending_entry_limits:
            for _d, (_o, _bar) in list(self.pending_entry_limits.items()):
                if len(self) > _bar:
                    self.pending_entry_limits.pop(_d, None)
                    if _o is not None and _o.status in (_o.Submitted, _o.Accepted,
                                                        _o.Partial):
                        self.cancel(_o)
                        self._limit_expired += 1
        current_month = current_date.strftime('%Y-%m')
        current_year = current_date.strftime('%Y')

        if any(len(data) == 0 for data in self.datas):
            return

        if self.last_logged_date != current_date:
            logging.info(f"Processing date: {current_date}")
            self.last_logged_date = current_date

        # Capital-efficiency snapshot on the SETTLED book, taken at the TOP of the
        # bar (before today's sell/buy rotation). Counts positions actually held
        # via getposition -- ground truth, NOT the open_positions counter, which
        # increments at buy SUBMISSION (so it also counts a just-submitted buy
        # that fills next bar -> reads max even when one slot is mid-rotation).
        # Deployment uses held market value / equity here; sampling at END of
        # next() instead caught the mid-cycle trough (intraday exits done, next-
        # day entries not yet filled) and understated deployment badly.
        # BUGFIX 2026-07-28: this block used to append on EVERY next() call. With
        # ~4000 feeds whose bar dates are not perfectly aligned, backtrader calls
        # next() ~1.8x per trading day -- the `last_logged_date` guard directly above
        # exists for exactly that reason. The extra calls land MID-ROTATION (the
        # day's exits have settled, its entries have not yet filled), so the series
        # was a blend of settled and half-empty books and every statistic built on it
        # was biased low. Measured effect: avg positions read 4.86 of 10 (utilisation
        # "48.6% [Unacceptable]") when the book demonstrably holds 9 on 233 of 234
        # days. The arithmetic closes independently: 968 trades x 2.175 trading days
        # held / 234 days = 9.00 concurrent.
        # Fix: one sample per trading date, taken on the FIRST next() of that date,
        # which is the settled book before the day's rotation -- the cadence the
        # comment above always intended.
        self._next_calls += 1
        if current_date != self._cap_last_date:
            self._cap_last_date = current_date
            # DAY_COUNT FIX 2026-07-29: day_count counts TRADING DATES, not next()
            # calls. It used to increment at the top of next(), which fires ~1.8x
            # per trading date on unaligned feeds (see the sampling note above), so
            # the annualization exponent (252/day_count), profit_per_day and
            # daily_return were all computed against an inflated denominator.
            # _next_calls keeps the raw call count for the sampling diagnostic.
            self.day_count += 1
            self.progress_bar.update(1)
            _held = [d for d in self.datas if self.getposition(d).size != 0]
            self.daily_positions.append(len(_held))
            self.daily_cap_dates.append(current_date)
            _eq = self.broker.getvalue()
            if _eq > 0:
                _invested = sum(self.getposition(d).size * d.close[0] for d in _held)
                self.daily_deployment.append(max(0.0, _invested / _eq * 100))
            else:
                self.daily_deployment.append(0.0)

        # Rule 201 scan every cycle. prenext() stops being called after the ATR-14
        # warmup, so without this the monitor went dark ~14 bars into the run (see
        # _run_rule_201_checks). Idempotent within a date, so extra cycles are safe.
        self._run_rule_201_checks()

        # SAFETY NET 2026-07-29. A long is only ever meant to go to zero. If one has
        # gone NEGATIVE the book has been over-sold (see exit_position for the mechanism
        # that used to cause it) and nothing downstream will ever touch it again:
        # evaluate_sell_conditions filters on size > 0 and get_buy_candidates skips it,
        # so the short sits there for the rest of the run, frozen, holding a slot in
        # open_positions. Unwind it the moment it appears and log it loudly -- a short
        # in a long-only backtest is a defect, not a position.
        for d in self.datas:
            if self.getposition(d).size < 0:
                logging.error(f"OVER-SOLD BOOK: {d._name} is short "
                              f"{self.getposition(d).size} on {current_date}. Unwinding.")
                self.close(data=d)

        # Retry the exits that had to be deferred a bar. By now check_submitted() has
        # moved the bracket legs into broker.pending, so cancel() can actually take.
        for d in list(self.deferred_exits):
            self.deferred_exits.discard(d)
            if self.getposition(d).size > 0:
                self.exit_position(d)

        # Continue with normal position management
        sell_data = [d for d in self.datas if self.getposition(d).size > 0]
        for d in sell_data:
            self.evaluate_sell_conditions(d, current_date)

        if self.open_positions < self.p.max_positions:
            buy_candidates = self.get_buy_candidates(current_date)
            if buy_candidates or current_date == self.last_trading_date:
                # This will now handle selecting the best stock if needed
                #print(f"Buy candidates: {[(d._name, size, correlation) for d, size, correlation in buy_candidates]}")
                self.process_buy_candidates(buy_candidates, current_date, verbose=False)
        elif current_date == self.last_trading_date:
            # POOL-EXPORT FIX (ported from the legacy 5__, 2026-08-23): the book is full on
            # the final bar, but the live signal pool still has to be written. The send
            # loop re-checks free slots, so this cannot open a trade.
            self._pool_export_mode = True
            try:
                pool = self.get_buy_candidates(current_date)
            finally:
                self._pool_export_mode = False
            self.process_buy_candidates(pool, current_date, verbose=False)

        current_equity = self.broker.getvalue()
        self.update_performance_tracking(current_equity, current_month, current_year)












    def update_performance_tracking(self, current_equity, current_month, current_year):
        # Monthly performance tracking
        if current_month != self.current_month:
            if self.current_month is not None and self.last_month_equity is not None:
                monthly_return = (current_equity / self.last_month_equity - 1) * 100
                self.monthly_performance[self.current_month] = monthly_return
                logging.info(f"Month {self.current_month} performance: {monthly_return:.2f}%")
                logging.info(f"Month high: {self.month_high_equity:.2f}, low: {self.month_low_equity:.2f}")

            self.current_month = current_month
            self.last_month_equity = current_equity
            self.month_high_equity = current_equity
            self.month_low_equity = current_equity
            logging.info(f"Starting new month: {current_month}")
        else:
            if current_equity > self.month_high_equity:
                self.month_high_equity = current_equity
            if current_equity < self.month_low_equity:
                self.month_low_equity = current_equity

        # Yearly performance tracking
        if current_year != self.current_year:
            if self.current_year is not None and self.last_year_equity is not None:
                yearly_return = (current_equity / self.last_year_equity - 1) * 100
                self.yearly_performance[self.current_year] = yearly_return
                logging.info(f"Year {self.current_year} performance: {yearly_return:.2f}%")
            self.current_year = current_year
            self.last_year_equity = current_equity
            logging.info(f"Starting new year: {current_year}")





    def get_buy_candidates(self, current_date):

        buy_candidates = []
        
        for d in self.datas:
            # size != 0, not size > 0: a data carrying an accidental short must not be
            # bought into. Doing so netted the two, left a stub position, and attached a
            # fresh full-size bracket to it -- which then over-sold again next bar.
            #
            # DOUBLE-ENTRY FIX 2026-07-29. position_dates is set at buy SUBMISSION and
            # cleared when the position is flattened, so it is the only marker of an
            # entry that is in flight. Without it this loop re-proposed a ticker whose
            # buy had been submitted but not yet filled -- which happens every day,
            # because next() fires ~1.8x per trading date (unaligned feed bars, see the
            # note in next()) and a market order does not fill until the following bar.
            # The result was two entries for one intended position: open_positions was
            # incremented twice but decremented once when it closed, so the counter
            # ratcheted up all run until it pinned at max_positions and killed trading.
            # Only one of the two got a bracket, too -- the second intent overwrote the
            # first in pending_bracket, leaving half the shares unprotected.
            if self.getposition(d).size != 0 or d in self.position_dates:
                continue

            if self.can_buy(d, current_date):
                size = self.calculate_position_size(d)
                if size <= 0 and self._pool_export_mode:
                    size = 1
                
                if size > 0:
                    correlation = self.get_mean_correlation(
                        d._name, 
                        [data._name for data in self.datas if self.getposition(data).size > 0]
                    )
                    buy_candidates.append((d, size, correlation))

        return buy_candidates
    
    def calculate_position_size(self, data):
        return self.position_sizer.calculate_position_size(
            account_value=self.broker.getvalue(),
            cash=self.broker.getcash(),
            price=data.close[0],
            atr=self.inds[data]['atr'][0],
            max_positions=self.p.max_positions
        )
    








    ##=================================================[relative thresholds]=================================================##
    ##=================================================[relative thresholds]=================================================##
    ##=================================================[relative thresholds]=================================================##
    ##=================================================[relative thresholds]=================================================##
    ##=================================================[relative thresholds]=================================================##
    ##=================================================[relative thresholds]=================================================##
    ##=================================================[relative thresholds]=================================================##
    























    # ==================================================================================
    # ============================  can_buy v2  ========================================
    # ==================================================================================

    def _rsi14(self, data):
        """Wilder-less 14-period RSI, computed EXACTLY as can_buy_v1_shipped computes it.

        Shared by the gate and the ranker so the two can never disagree about a name's
        RSI. The repo has already been bitten once by two components measuring the same
        indicator differently (the predictor gate vs the FilterRubric disagreed by 11 to
        13 points), so this is deliberately a single implementation, not two.

        Returns None when there are not enough bars, which callers treat as "no opinion"
        rather than as a rejection.
        """
        RSI_PERIOD = 14
        try:
            closes = data.close.get(size=RSI_PERIOD + 1)
            if closes is None or len(closes) < RSI_PERIOD + 1:
                return None
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-RSI_PERIOD:])
            avg_loss = np.mean(losses[-RSI_PERIOD:])
            if avg_loss == 0:
                return 100.0
            return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
        except (IndexError, AttributeError, TypeError, ValueError):
            return None

    def _ret5(self, data):
        """5-bar return, matching the shipped momentum gate's arithmetic."""
        try:
            prev = data.close[-5]
            cur = data.close[0]
            if prev and prev > 0 and cur is not None:
                return (cur / prev) - 1.0
        except (IndexError, AttributeError, TypeError, ZeroDivisionError):
            pass
        return None

    def can_buy(self, data, current_date):
        """can_buy v2 = the shipped gate AND an RSI ceiling.

        WHY A CEILING. The shipped function floors RSI at 20 and has no upper bound, so
        it will buy a name that has already run. Measured on 2,526,042 rows scored
        through the live runner bracket, payoff falls monotonically with RSI inside
        every within-day UpProbability quartile, and the RSI 60-101 band is the only one
        that turns negative out of sample.

        WHY 55, and why this is not a fitted number. The per-signal panel picked it
        (+0.092pp, first half +0.095, second half +0.085, 28 of 35 months, p=0.001)
        before any portfolio simulation ran. The 4-slot sweep then found a broad plateau
        from 50 to 60 rather than a spike, and 55 is the most STABLE point on it: 60 has
        a marginally higher pooled mean (+0.6126 vs +0.6065) but splits +0.7465/+0.4449
        across the halves, against 55's +0.6162/+0.5941. Choosing the stable interior
        point over the higher pooled mean is deliberate.

        WHY IT IS A WRAPPER. Calling the untouched shipped function keeps v2 provably
        equal to "v1 AND rsi <= ceiling". Nothing in the shipped path can drift. Setting
        BT_MAXRSI to 101 (or anything >= 100) makes this byte-identical to shipped and
        is the one-line rollback.

        The ceiling is checked FIRST because it is 15 bars of arithmetic, while the
        shipped path runs a percentile loop over up to 100 bars. Rejecting ~38% of
        candidates before that loop makes v2 cheaper than v1, not dearer. Order is
        irrelevant to the result: this is a conjunction of independent conditions.

        NOT INCLUDED, and this is a measured decision. Tightening the 5-day momentum cap
        from 0.15 to 0.05 is worth +0.047pp per signal on its own, but stacked on this
        ceiling it COSTS 0.079pp at 4 slots (+0.5277 vs +0.6065). High RSI and high
        5-day momentum select overlapping names, so the second gate buys nothing and
        starves the pool. Set BT_MOMGAIN=0.05 to re-test it; the default leaves the
        shipped 0.15 alone.
        """
        # ============================ can_buy v4 ============================
        # Two tightenings of existing gates. Nothing else moves. Derived from the
        # three-question stratification (v4_strata.py) and scored on the slot-simulated
        # TRADED book with a paired month-block bootstrap (v4_build.py).
        #
        # 1. RSI CEILING 60. Shipped floors RSI at 20 and has no ceiling. Within the
        #    traded book the 55-60 band is the single best band (+1.678% mean, 18.3%
        #    P(big)); 60-70 is -0.364 and 70+ is -0.837. 60 beats the 55 that v2 shipped,
        #    which cut the best band in half. Alone: +227.7 total, 95% CI [+55.2, +407.6],
        #    the only single change whose CI excludes zero.
        #
        # 2. W52 CAP 0.85 -> 0.75. Distance from the 52-week high is the strongest
        #    MAGNITUDE separator in the book (AUC 0.4145, edge 0.171 on |ret|), and with
        #    a 3% capped downside and an uncapped runner, magnitude is what pays.
        #    Alone: +192.4, CI [-38.6, +413.5].
        #
        # TOGETHER: +251.3 at 4 slots, and POSITIVE AT EVERY SLOT COUNT 2..6
        # (+201.7 / +164.4 / +251.3 / +202.6 / +177.1). The pair is far steadier across
        # slot counts than either gate alone, which is why both ship rather than the one
        # with the tighter CI.
        #
        # THE RANKER IS DELIBERATELY UNCHANGED. Six alternatives were tested and all
        # lost: U-shaped |rank-0.5| -100.7, up-0.3*volratio -49.7, vol_ratio tie-break
        # -15.5, and three additive composites in the earlier round. UpProbability
        # descending is the best ranker measured.
        #
        # NOT INCLUDED despite the strongest separation in the whole study: vol_ratio
        # (AUC 0.3969, edge 0.206 on top-1% vs bottom-1%). It predicts WHICH EXTREME a
        # trade becomes, and the book is not paid for choosing between extremes:
        # vol_ratio<=1.0 scores +31.0 with a CI of [-235.1, +301.6].
        #
        # ROLLBACK: BT_MAXRSI=101 BT_W52=0.85 restores shipped exactly.
        #
        # CAVEAT, stated rather than buried: this is refereed by the daily-bar payoff
        # sim, which charges no spread and no commission. 5__'s own exit model cannot
        # referee it because its EOD ratchet stalls (censused: 88 symbols, median 3
        # repeats). The 5-minute rig is the tiebreaker and has not been run on this.
        MAX_RSI = 101.0   # v4 REFUTED 2026-08-23 (full-universe PnL -91%); 60 to re-arm
        if MAX_RSI < 100:
            rsi = self._rsi14(data)
            if rsi is not None and rsi > MAX_RSI:
                return False
        return self.can_buy_v1_shipped(data, current_date)

    def can_buy_v1_shipped(self, data, current_date):
        """=

        Restored 2026-05-17 after the EDA-driven retune
        (can_buy_may_17th_eda_attempt_FAILED) regressed badly in backtest:
        73.7% -> 23.5% annualised, Sharpe 1.85 -> 0.56.

        Original docstring follows.

        Tuned by 2 x 800-trial Optuna runs over the continuous filter space
        (top_k=3/hold=2 seed=0, top_k=5/hold=1 seed=42). Held-out window:
        2025-05-01 onwards. Only threshold changes where BOTH runs agreed on
        direction AND the cross-seed midpoint moved measurably are shipped;
        params with conflicting directions were left at legacy values.

        Lite-backtest holdout deltas vs legacy can_buy_may15th:
          Sharpe   +1.38 -> +1.76 (k=3/h=2)   |   +1.16 -> +1.19 (k=5/h=1)
          MaxDD    19.4% -> 11.6%             |   17.8% -> 14.2%
          AnnRet   +90%  -> +100%             |   +332% -> +358%

        Single-line rollback: swap can_buy <-> can_buy_may15th in
        get_buy_candidates (line ~1279).
        """

        # Strategy Timing (unchanged)
        MIN_DAYS_BEFORE_TRADING = 30
        TARGET_HISTORIC_PROB_COUNT = 45
        MAX_HISTORIC_LOOKBACK = 100
        MIN_HISTORIC_PROB_THRESHOLD = 30
        MIN_VIABLE_DATA_POINTS = 5

        # Probability Bounds (unchanged -- runs disagreed on direction)
        UP_PROB_MIN_BOUND = 0.2
        UP_PROB_MAX_BOUND = 0.8

        # Cross-sectional UpProbability floor (added 2026-06-18). EDA on 944 real
        # trades: [0.30,0.38) is a stable money-loser (neg PnL both halves, ~42% win);
        # flooring at 0.40 cuts only the junk band (+$4.7k PnL, per-trade Sharpe
        # 0.19->0.32). A 0.45 floor was rejected (would cut the profitable [0.42,0.44)
        # workhorse, -$9k). Env-gated so the canbuy A/B harness can sweep it.
        UP_PROB_FLOOR = 0.4

        # Financial Thresholds
        MIN_CLOSE_PRICE = 2.00       # was 1.50 -- both runs preferred ~2.0
        MAX_CLOSE_PRICE = 1650.00    # was 1000.00 -- both runs preferred ~1650
        MIN_VOLUME_SHARES = 10_000
        # env-gated 2026-07-25: the gate census showed dollar_vol carries real edge
        # (+0.52pp H1 / +1.35pp H2 between kept and killed) but fires on ~1% of
        # marginal rows, i.e. it is valuable and almost never binding. Tightening
        # the high-value/low-fire-rate gates was never tested.
        MIN_DOLLAR_VOLUME = 1000000.0

        # Risk Management - Drop Protection (unchanged)
        MAX_SINGLE_DAY_DROP = -0.15
        RECENT_DROP_LOOKBACK_DAYS = 10

        # Risk Management - 52-Week Position (env-gated for the canbuy_optimizer A/B; default 0.85)
        # v4: 0.85 -> 0.75. See the can_buy v4 header for the measurement.
        # DEFAULT REALIGNED TO SHIPPED 2026-08-26: was 0.85 here against production's
        # 1.01, so the fork ran a gate production does not run and every fork A/B
        # carried it. At 0.85 this removes ~61.9% of the post-mask universe and the
        # measured name effect flips sign (A4). 1.01 is structurally inert, which is
        # what production ships. BT_W52=0.85 restores the old fork behaviour.
        WEEK_52_HIGH_PROXIMITY_LIMIT = 1.01
        WEEK_52_LOOKBACK_DAYS = 252

        # Risk Management - Momentum
        MOMENTUM_LOOKBACK_DAYS = 5
        MAX_MOMENTUM_GAIN = 0.15
        MAX_MOMENTUM_LOSS = -0.18

        # Risk Management - Volume & Volatility
        VOLUME_SPIKE_MULTIPLIER = 3.5
        VOLUME_AVG_LOOKBACK_DAYS = 20
        MAX_VOLATILITY_THRESHOLD = 0.1  # env-gated for canbuy_optimizer A/B; default 0.10
        VOLATILITY_LOOKBACK_DAYS = 20

        # STUCK-PRICE gate (2026-08-18). A name that barely moves cannot reach the
        # +6% target before the 5-day clock runs out, so the slot is dead capital for a
        # week. Measured on the 18,361-row unfiltered candidate panel scored at the LIVE
        # 3.0/6.0 bracket: candidates whose LARGEST intraday range over the last 10
        # sessions stayed under 1.0% of price hit the target 0.0% of the time (0 of 123)
        # and timed out 100.0%, vs a 23.3% baseline timeout rate (t=-2.36). Zero
        # target-hits in EVERY year the gate fired (2024 n=93, 2025 n=30).
        #
        # THRESHOLD SWEEP: Data/_stuckgrid/ (sweep.csv, heatmap.png, cliff.png). A 240-cell
        # grid over measure x window x threshold. The surface is SMOOTH and MONOTONE with a
        # hard CLIFF: below 1.0-1.25% intraday range NOTHING reaches the target; above it
        # the target-hit rate climbs steadily (1.5% -> 1.7%, 2.0% -> 3.7%, 3.0% -> 10.6%).
        # 1.0% is the last threshold that loses nothing tradeable - walked back from the cliff.
        #
        # INTRADAY RANGE, not close-to-close: a flat close can hide a big intraday swing.
        # Of 254 candidates whose 3-day close-to-close move stayed under 0.5%, 104 still
        # ranged >=1% intraday - tradeable names a close-to-close gate would wrongly cut.
        #
        # PERCENT, not cents. An absolute "range < $0.03" version was tested and REJECTED:
        # it flagged NIO, a $5 stock ranging 2-7% a day, because cheap stocks have small
        # absolute ranges. That is a low-price proxy, not a stuck-price detector.
        #
        # Fires on 0.7% of candidates and would have blocked 0 of the 813 live trades:
        # pure tail insurance against a halted / deal-pegged / flatlined name, in the same
        # category as the delisting screen. Set BT_STUCK_RANGE_PCT=0 to disable.
        STUCK_RANGE_PCT = 1.0   # max intraday range under this %...
        STUCK_LOOKBACK_DAYS = 10   # ...across this window => skip

        # Percentile Thresholds (sufficient-data path)
        # Single biggest finding: both runs converged on ~0.65 low bound. The legacy
        # 0.90 only fires when current prob is in the stock's top 10%; 0.65 fires
        # when it's in the top 35% -- captures the bulk of the model's per-stock
        # alpha that the legacy cutoff was excluding.
        SUFFICIENT_DATA_P_LOW = 65.0     # env-gated for param-stress; default 65.0
        SUFFICIENT_DATA_P_HIGH = 98.0   # env-gated for param-stress; default 98.0

        # For limited data (unchanged -- not in optimizer; keep strict on low-data)
        LIMITED_DATA_P_LOW = 90
        LIMITED_DATA_P_HIGH = 99

        # ============ END CONFIGURATION ============

        symbol = data._name

        if not hasattr(self, 'strategy_start_date'):
            self.strategy_start_date = current_date

        self._gate_hits['_evaluated'] = self._gate_hits.get('_evaluated', 0) + 1
        days_since_start = (current_date - self.strategy_start_date).days
        if days_since_start < MIN_DAYS_BEFORE_TRADING:
            return self._g('g01_warmup_30d')

        try:
            current_close = data.close[0]
            current_prob = data.UpProbability[0]
            current_volume = data.volume[0]
        except (IndexError, AttributeError, TypeError):
            return False

        if self.rule_201_monitor.is_restricted(symbol) or (not self._pool_export_mode and self.open_positions >= self.p.max_positions):
            return self._g('g02_rule201_or_slots_full')
        if self._reentry_blocked(symbol, current_date, data):
            return self._g('g02b_reentry_blocked')

        if current_prob < UP_PROB_MIN_BOUND or current_prob > UP_PROB_MAX_BOUND:
            return self._g('g03_prob_bounds_0208')

        # Cross-sectional UpProb floor: skip the model's stable-negative low band.
        if current_prob < UP_PROB_FLOOR:
            return self._g('g04_prob_floor')

        try:
            if len(data.UpProbability) < 6:
                return False
            recent_probs = [data.UpProbability[-i] for i in range(1, 6) if data.UpProbability[-i] is not None]  # LOOKAHEAD FIX 2026-06-10: [i]->[-i] (positive index read FUTURE bars)
            if recent_probs and min(recent_probs) < UP_PROB_MIN_BOUND:
                return False
            if recent_probs and max(recent_probs) > UP_PROB_MAX_BOUND:
                return False
        except Exception:
            pass

        if current_close is None or current_close < MIN_CLOSE_PRICE:
            return self._g('g06_min_close')
        if current_close is None or current_close > MAX_CLOSE_PRICE:
            return self._g('g07_max_close')
        if current_volume is None or current_volume < MIN_VOLUME_SHARES:
            return self._g('g08_min_volume_shares')
        if current_prob is None:
            return False

        dollar_volume = current_close * current_volume
        if dollar_volume < MIN_DOLLAR_VOLUME:
            return self._g('g09_min_dollar_volume')

        # STUCK-PRICE gate. Deliberately placed here: it is pure arithmetic on bars the
        # feed already holds (no percentile build, no correlation lookup), so it runs
        # before the expensive historic-probability loop below and keeps the hot path fast.
        # See the STUCK_* constants for the measurement.
        if STUCK_RANGE_PCT > 0:
            try:
                highs = data.high.get(size=STUCK_LOOKBACK_DAYS)
                lows = data.low.get(size=STUCK_LOOKBACK_DAYS)
                closes_stuck = data.close.get(size=STUCK_LOOKBACK_DAYS)
                if (highs is not None and lows is not None and closes_stuck is not None
                        and len(highs) == len(lows) == len(closes_stuck)
                        and len(closes_stuck) >= STUCK_LOOKBACK_DAYS):
                    max_rng = 0.0
                    for _h, _l, _c in zip(highs, lows, closes_stuck):
                        if _c and _c > 0:
                            _r = ((_h - _l) / _c) * 100.0
                            if _r > max_rng:
                                max_rng = _r
                    if max_rng < STUCK_RANGE_PCT:
                        return self._g('g10_stuck_price')
            except Exception:
                pass

        historic_probs = []
        i = 1
        while len(historic_probs) < TARGET_HISTORIC_PROB_COUNT and i < MAX_HISTORIC_LOOKBACK:
            try:
                prob_val = data.UpProbability[-i]  # LOOKAHEAD FIX 2026-06-10: [i]->[-i] (was reading future bars into the percentile baseline)
                if prob_val is not None:
                    historic_probs.append(float(prob_val))
                i += 1
            except (IndexError, AttributeError, TypeError):
                break

        if len(historic_probs) < MIN_HISTORIC_PROB_THRESHOLD:
            try:
                for lookback in range(1, min(RECENT_DROP_LOOKBACK_DAYS + 1, 100)):
                    try:
                        prev_close = data.close[-lookback]
                        next_close = data.close[-(lookback-1)] if lookback > 1 else data.close[0]
                        if prev_close is not None and next_close is not None and prev_close > 0:
                            daily_return = (next_close / prev_close) - 1
                            if daily_return < MAX_SINGLE_DAY_DROP:
                                return False
                    except (IndexError, AttributeError, TypeError):
                        break
            except Exception:
                return False

            if len(historic_probs) < MIN_VIABLE_DATA_POINTS:
                return False

            if len(historic_probs) > 1:
                p96 = np.percentile(historic_probs, LIMITED_DATA_P_LOW)
                p97_5 = np.percentile(historic_probs, LIMITED_DATA_P_HIGH)
            else:
                return False
        else:
            if len(historic_probs) > 1:
                p96 = np.percentile(historic_probs, SUFFICIENT_DATA_P_LOW)
                p97_5 = np.percentile(historic_probs, SUFFICIENT_DATA_P_HIGH)
            else:
                return False

        # 1. 52-week high filter. Ported from shipped 2026-08-26. Reads the
        # Pct52wHigh line precomputed in load_data() from the UNTRUNCATED history and
        # falls back to the old buffer derivation only if the line is missing or NaN,
        # counting both, so a silent skip can never come back unnoticed.
        # BT_W52_STRICT=1 makes an unavailable value REJECT the candidate instead of
        # waving it through, which is the safe direction for a filter.
        _pct = None
        try:
            _v = float(data.Pct52wHigh[0])
            if np.isfinite(_v) and _v > 0:
                _pct = _v
                self._w52_precomputed = getattr(self, '_w52_precomputed', 0) + 1
        except (IndexError, AttributeError, TypeError, ValueError):
            _pct = None
        if _pct is None:
            try:
                closes_252 = data.close.get(size=WEEK_52_LOOKBACK_DAYS)
                if closes_252 and len(closes_252) > 0:
                    _h = max(closes_252)
                    if _h and _h > 0:
                        _pct = current_close / _h
                        self._w52_fallback = getattr(self, '_w52_fallback', 0) + 1
            except Exception:
                _pct = None
        if _pct is None:
            self._w52_skipped = getattr(self, '_w52_skipped', 0) + 1
        else:
            self._w52_evaluated = getattr(self, '_w52_evaluated', 0) + 1
            if _pct > WEEK_52_HIGH_PROXIMITY_LIMIT:
                self._w52_blocked = getattr(self, '_w52_blocked', 0) + 1
                return self._g('g11_w52_high')

        # 2. Momentum filter
        try:
            prev_close_5 = data.close[-MOMENTUM_LOOKBACK_DAYS]
            if prev_close_5 and prev_close_5 > 0:
                ret_5d = (current_close / prev_close_5) - 1
                if ret_5d > MAX_MOMENTUM_GAIN:
                    return self._g('g12_momentum_gain')
                if ret_5d < MAX_MOMENTUM_LOSS:
                    return self._g('g13_momentum_loss')
        except Exception:
            pass

        # 3. Volume spike filter
        try:
            vols = data.volume.get(size=VOLUME_AVG_LOOKBACK_DAYS)
            if vols is not None and len(vols) > 0:
                avg_vol_20 = np.mean(vols)
                if avg_vol_20 and current_volume / avg_vol_20 > VOLUME_SPIKE_MULTIPLIER:
                    return self._g('g14_volume_spike')
        except Exception:
            pass

        # 4. Volatility filter
        try:
            closes = data.close.get(size=VOLATILITY_LOOKBACK_DAYS)
            if closes is not None and len(closes) > 1:
                returns_20d = np.diff(np.log(closes))
                if len(returns_20d) > 0:
                    vol_20d = np.std(returns_20d)
                    if vol_20d > MAX_VOLATILITY_THRESHOLD:
                        return self._g('g15_volatility_20d')
        except Exception:
            pass

        RSI_PERIOD = 14
        # env-gated 2026-07-25: persist5 passers carry RSI 62.1 vs 54.5 for failers,
        # so the persistence gate may simply BE a trend filter. This is the control.
        MIN_RSI_THRESHOLD = 20.0

        # 5. RSI filter
        try:
            closes_for_rsi = data.close.get(size=RSI_PERIOD + 1)
            if closes_for_rsi is not None and len(closes_for_rsi) >= RSI_PERIOD + 1:
                deltas = np.diff(closes_for_rsi)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains[-RSI_PERIOD:])
                avg_loss = np.mean(losses[-RSI_PERIOD:])
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                if rsi < MIN_RSI_THRESHOLD:
                    return self._g('g16_rsi_floor')
        except Exception:
            pass


        # ---------------- new probe gates (2026-07-25) ----------------
        # All inert by default. Each targets a property of the PREDICTION that
        # can_buy currently ignores entirely -- it looks at price/volume/percentile
        # but never at whether the probability series itself is trustworthy.
        try:
            _hp = historic_probs  # last <=45 own-history probs, already built above
            if _hp:
                _arr = np.asarray(_hp, dtype=float)

                # (E) signal STABILITY: a name whose own probability jitters wildly
                # is being scored inconsistently. Prefer steady scorers.
                _maxstd = 0.0
                if _maxstd > 0 and _arr.size > 5 and float(np.std(_arr)) > _maxstd:
                    return False

                # (H) DEGENERATE history: ~68% of the panel is pinned at the 0.30
                # floor, so for many names the trailing window is a block of ties
                # and the p65/p98 percentile band is meaningless -- p65 == the floor,
                # so "above own p65" is satisfied by literally any non-floor value.
                # Require the own-history to actually carry information.
                _mindist = 0
                if _mindist > 0 and len(set(np.round(_arr, 6))) < _mindist:
                    return False

                # (F) PERSISTENCE: today's qualifying probability should not be a
                # one-day spike. Require N of the last M days to also clear the floor.
                _persist = 0
                if _persist > 0:
                    _win = 5
                    # BT_PERSIST_FLOOR decouples "the score held up" from "the
                    # score was tradeable"; defaults to the trading floor.
                    _pf = float(UP_PROB_FLOOR)
                    if int((_arr[:_win] >= _pf).sum()) < _persist:
                        return False

            # (G) probability MOMENTUM at entry. The sell side already acts on
            # falling probability (momentum <= -0.05 triggers an exit) but the buy
            # side is completely blind to it -- we will happily buy a name whose
            # score is collapsing and then exit it days later for that same reason.
            _minmom = None
            if _minmom is not None and len(data.UpProbability) > 1:
                _prev = data.UpProbability[-1]
                if _prev is not None and (current_prob - float(_prev)) < float(_minmom):
                    return False
        except Exception:
            pass

        # --- Core Buy Condition ---
        if current_prob >= p96 and current_prob < p97_5:
            self._gate_hits['PASS'] = self._gate_hits.get('PASS', 0) + 1
            return True

        return self._g('g17_band_below_p65' if current_prob < p96 else 'g18_band_above_p98')




























    ##===============================[ SELLING ]==================================##
    ##===============================[ SELLING ]==================================##
    ##===============================[ SELLING ]==================================##
    
    
    
    def force_best_signal_for_current_day(self, data=None):
        """Find and save the best possible stock for the current day using IDENTICAL logic to can_buy function."""
        logging.info("Finding best signals using identical can_buy criteria...")
    
        # Use the exact same configuration constants as can_buy
        MIN_DAYS_BEFORE_TRADING = 30
        TARGET_HISTORIC_PROB_COUNT = 45
        MAX_HISTORIC_LOOKBACK = 100
        MIN_HISTORIC_PROB_THRESHOLD = 30
        MIN_VIABLE_DATA_POINTS = 5
    
        UP_PROB_MIN_BOUND = 0.35
        UP_PROB_MAX_BOUND = 0.60
    
        MIN_CLOSE_PRICE = 1.50
        MAX_CLOSE_PRICE = 1000.00
        MIN_VOLUME_SHARES = 10_000
        MIN_DOLLAR_VOLUME = 500_000
    
        MAX_SINGLE_DAY_DROP = -0.15
        RECENT_DROP_LOOKBACK_DAYS = 10
    
        WEEK_52_HIGH_PROXIMITY_LIMIT = 0.85
        WEEK_52_LOOKBACK_DAYS = 252
    
        MOMENTUM_LOOKBACK_DAYS = 5
        MAX_MOMENTUM_GAIN = 0.15
        MAX_MOMENTUM_LOSS = -0.075
    
        VOLUME_SPIKE_MULTIPLIER = 3.5
        VOLUME_AVG_LOOKBACK_DAYS = 20
        MAX_VOLATILITY_THRESHOLD = 0.04
        VOLATILITY_LOOKBACK_DAYS = 20
    
        SUFFICIENT_DATA_P_LOW = 90.0
        SUFFICIENT_DATA_P_HIGH = 95.0
        LIMITED_DATA_P_LOW = 90
        LIMITED_DATA_P_HIGH = 99
    
        # Strategy start date check (same as can_buy)
        current_date = self.datetime.date()
        if not hasattr(self, 'strategy_start_date'):
            self.strategy_start_date = current_date
    
        days_since_start = (current_date - self.strategy_start_date).days
    
        # Create a list to store candidates that pass ALL can_buy criteria
        valid_candidates = []
    
        for d in self.datas:
            if self.getposition(d).size > 0:
                continue  # Skip stocks we already have positions in
            
            symbol = d._name
            if self.rule_201_monitor.is_restricted(symbol):
                continue  # Skip stocks restricted by Rule 201
            if self._reentry_blocked(symbol, current_date, d):
                continue  # Skip stocks inside the experimental re-entry cooldown
            
            # Apply EXACT same filters as can_buy function
            try:
                # 1. Strategy timing filter
                if days_since_start < MIN_DAYS_BEFORE_TRADING:
                    continue
                
                # 2. Current values extraction
                try:
                    current_close = d.close[0]
                    current_prob = d.UpProbability[0]
                    current_volume = d.volume[0]
                except (IndexError, AttributeError, TypeError):
                    continue
                
                # 3. Basic probability bounds
                if current_prob < UP_PROB_MIN_BOUND or current_prob > UP_PROB_MAX_BOUND:
                    continue

                try:
                    #make sure up prob is within the range that shows the best results and not some weird values like 0.1 or 0.9
                    if len(d.UpProbability) < 6:
                        continue
                    recent_probs = [d.UpProbability[-i] for i in range(1, 6) if d.UpProbability[-i] is not None]

                    if recent_probs and min(recent_probs) < UP_PROB_MIN_BOUND:
                        continue
                    if recent_probs and max(recent_probs) > UP_PROB_MAX_BOUND:
                        continue
                except Exception:
                    pass
                
                # 4. Hard rejects (price, volume, prob)
                if current_close is None or current_close < MIN_CLOSE_PRICE:
                    continue
                if current_close > MAX_CLOSE_PRICE:
                    continue
                if current_volume is None or current_volume < MIN_VOLUME_SHARES:
                    continue
                if current_prob is None:
                    continue
                
                # 5. Liquidity check
                dollar_volume = current_close * current_volume
                if dollar_volume < MIN_DOLLAR_VOLUME:
                    continue
                
                # 6. Build historical UpProbability series (identical to can_buy)
                historic_probs = []
                i = 1
                while len(historic_probs) < TARGET_HISTORIC_PROB_COUNT and i < MAX_HISTORIC_LOOKBACK:
                    try:
                        prob_val = d.UpProbability[-i]
                        if prob_val is not None:
                            historic_probs.append(float(prob_val))
                        i += 1
                    except (IndexError, AttributeError, TypeError):
                        break
                    
                # 7. Limited data sanity check (identical to can_buy)
                if len(historic_probs) < MIN_HISTORIC_PROB_THRESHOLD:
                    # Check for recent drops when data is limited
                    drop_detected = False
                    try:
                        for lookback in range(1, min(RECENT_DROP_LOOKBACK_DAYS + 1, 100)):
                            try:
                                prev_close = d.close[-lookback]
                                next_close = d.close[-(lookback-1)] if lookback > 1 else d.close[0]
    
                                if prev_close is not None and next_close is not None and prev_close > 0:
                                    daily_return = (next_close / prev_close) - 1
                                    if daily_return < MAX_SINGLE_DAY_DROP:
                                        drop_detected = True
                                        break
                            except (IndexError, AttributeError, TypeError):
                                break
                    except Exception:
                        drop_detected = True
    
                    if drop_detected:
                        continue
                    
                    # Need minimum viable data
                    if len(historic_probs) < MIN_VIABLE_DATA_POINTS:
                        continue
                    
                    # Use limited data thresholds
                    if len(historic_probs) > 1:
                        p96 = np.percentile(historic_probs, LIMITED_DATA_P_LOW)
                        p97_5 = np.percentile(historic_probs, LIMITED_DATA_P_HIGH)
                    else:
                        continue
                else:
                    # Use standard thresholds for sufficient data
                    if len(historic_probs) > 1:
                        p96 = np.percentile(historic_probs, SUFFICIENT_DATA_P_LOW)
                        p97_5 = np.percentile(historic_probs, SUFFICIENT_DATA_P_HIGH)
                    else:
                        continue
                    
                # 8. 52-week high filter. Uses the Pct52wHigh line precomputed in
                # load_data() from the untruncated history; the old
                # d.close.get(size=252) derivation returned an EMPTY slice before bar
                # 251 and silently skipped the filter on 88.5% of evaluations. This is
                # the LIVE pool-export copy, so it is the one that decided what the
                # broker saw. Ported from shipped 2026-08-26.
                try:
                    _p52 = float(d.Pct52wHigh[0])
                    if np.isfinite(_p52) and _p52 > WEEK_52_HIGH_PROXIMITY_LIMIT:
                        continue
                except (IndexError, AttributeError, TypeError, ValueError):
                    pass
                
                # 9. Momentum filter
                try:
                    prev_close_5 = d.close[-MOMENTUM_LOOKBACK_DAYS]
                    if prev_close_5 and prev_close_5 > 0:
                        ret_5d = (current_close / prev_close_5) - 1
                        if ret_5d > MAX_MOMENTUM_GAIN or ret_5d < MAX_MOMENTUM_LOSS:
                            continue
                except Exception:
                    pass
                
                # 10. Volume spike filter
                try:
                    vols = d.volume.get(size=VOLUME_AVG_LOOKBACK_DAYS)
                    if vols is not None and len(vols) > 0:
                        avg_vol_20 = np.mean(vols)
                        if avg_vol_20 and current_volume / avg_vol_20 > VOLUME_SPIKE_MULTIPLIER:
                            continue
                except Exception:
                    pass
                
                # 11. Volatility filter
                try:
                    closes = d.close.get(size=VOLATILITY_LOOKBACK_DAYS)
                    if closes is not None and len(closes) > 1:
                        returns_20d = np.diff(np.log(closes))
                        if len(returns_20d) > 0:
                            vol_20d = np.std(returns_20d)
                            if vol_20d > MAX_VOLATILITY_THRESHOLD:
                                continue
                except Exception:
                    pass

                # 12. CORE BUY CONDITION (identical to can_buy)
                if current_prob >= p96 and current_prob < p97_5:
                    # This stock passes ALL can_buy criteria
                    size = self.calculate_position_size(d)
                    if size > 0:
                        # Calculate the same quality score as before for ranking
                        quality_score = current_prob * 100  # Simple ranking by probability
                        valid_candidates.append((d, size, 0, quality_score, current_prob, p96, p97_5))
    
            except Exception as e:
                logging.warning(f"Error evaluating candidate {symbol} with can_buy logic: {str(e)}")
    
        # Sort by quality score (highest probability first, just like can_buy preference)
        valid_candidates.sort(key=lambda x: x[3], reverse=True)
    
        if valid_candidates:
            # Log the results
            logging.info(f"Found {len(valid_candidates)} stocks that pass identical can_buy criteria:")
            for i, (d, size, _, quality_score, current_prob, p96, p97_5) in enumerate(valid_candidates[:5]):
                logging.info(f"  {i+1}. {d._name}: UpProb={current_prob:.4f} (threshold: {p96:.4f}-{p97_5:.4f})")
    
            # Return only the data/size/correlation tuples for compatibility
            best_candidates = [(candidate[0], candidate[1], candidate[2]) for candidate in valid_candidates]

            logging.info(f"Selected {len(best_candidates)} signals using identical can_buy criteria")
        else:
            logging.warning("No stocks passed the identical can_buy criteria filters")
            # If no stocks pass the strict criteria, you could choose to:
            # 1. Return empty list (no signals)
            # 2. Use a fallback method
            # For now, return empty to maintain consistency
            pass























    
    
    def process_buy_candidates(self, buy_candidates, current_date, verbose=False):
        """Process buy candidates and execute trades."""
        if verbose:
            dprint("Starting process_buy_candidates", "INFO")
            dprint(f"Current date: {current_date}", "INFO")
            dprint(f"Number of buy candidates: {len(buy_candidates)}", "INFO")
            dprint(f"Last trading date: {self.last_trading_date}", "INFO")
        
        self.force_best_signal_for_current_day()
        if verbose:
            dprint("Completed force_best_signal_for_current_day", "INFO")
    
        buy_candidates = self.sort_buy_candidates(buy_candidates)
        if verbose:
            dprint(f"Sorted buy candidates: {len(buy_candidates)}", "INFO")
    
        # Save the most-recent-day candidate POOL to Data/0__signals.parquet (canonical signals file)
        # FAST: gate to the LAST trading date only. This export block scrapes finviz +
        # runs FinBERT sentiment (save_guaranteed_signals_to_parquet) - on every historical
        # bar that was ~64% of backtest runtime (1225s of SSL reads). It's post-decision
        # (live-signal export), so running it only on the final bar is LOSSLESS for backtest
        # metrics and matches the experimental backtester's fix.
        if buy_candidates and current_date == self.last_trading_date:
            if verbose:
                dprint("Converting candidates to new signal format", "INFO")
            signals = []
            
            # Anchor the signal date to the DATA's last bar, not the wall clock.
            # This block only fires when current_date == self.last_trading_date, so
            # last_trading_date IS the last processed data bar. Using datetime.now()
            # here mis-stamped signals when the pipeline ran after midnight (e.g. a
            # 01:39 AM run gave now=07-01 -> next=07-02 for 06-30 data). Deriving from
            # the data keeps the target correct regardless of run time.
            real_current_date = self.last_trading_date
            next_trading_day = get_next_trading_day(real_current_date)
            if verbose:
                dprint(f"Real current date: {real_current_date}", "INFO")
                dprint(f"Next trading day for signals: {next_trading_day}", "INFO")
            
            # Write a candidate POOL (not just max_positions) so the morning
            # FilterRubric narrowing has a surplus to trim down to the final book.
            # The broker only ever trades the narrowed _Buy_Signals.parquet, never
            # this raw pool.
            # NOTE: this is a CAP, not a target. If fewer candidates qualify on the final
            # bar, the pool is shorter than the cap and raising it changes nothing; the
            # census printed below tells you which case you are in. 36 so that the broker's
            # gap gate (auxiliary/trigger_entry.py, before its reach-12 cut) leaves 12 names.
            SIGNAL_POOL_SIZE = int(os.environ.get('BT_SIGNAL_POOL_SIZE', 36))
            logging.info("SIGNAL POOL: %d qualifying candidates on %s, cap %d -> writing %d%s"
                         % (len(buy_candidates), real_current_date, SIGNAL_POOL_SIZE,
                            min(len(buy_candidates), SIGNAL_POOL_SIZE),
                            "  <- CAP IS BINDING, raise BT_SIGNAL_POOL_SIZE for more"
                            if len(buy_candidates) > SIGNAL_POOL_SIZE else
                            "  <- candidate-limited, raising the cap will NOT widen this"))
            # POOL ORDER: the broker re-ranks the pool by UpProbability (trigger_entry.load_pool),
            # so write the top N by UpProbability, not the top N by BT_SELRULE. The nightly's
            # own execution loop below still runs in selrule order.
            _pool_order = sorted(buy_candidates,
                                 key=lambda t: float(t[0].UpProbability[0]), reverse=True)
            for d, size, correlation in _pool_order[:SIGNAL_POOL_SIZE]:
                price = d.close[0]
                atr = self.inds[d]['atr'][0] if d in self.inds and 'atr' in self.inds[d] else price * 0.02
                
                signal = {
                    'Symbol': d._name,
                    'Price': price,
                    'UpProbability': d.UpProbability[0],
                    'ATR': atr,
                    'Quality': d.UpProbability[0],  # Use probability as quality measure
                    'DollarVolume': d.volume[0] * price,
                    'ThresholdValue': d.UpProbability[0],  # The threshold that triggered this signal
                }
                signals.append(signal)
                if verbose:
                    dprint(f"Created signal for {d._name} at ${price:.2f} with UpProb {d.UpProbability[0]:.3f}", "DETAIL")
            
            # Save to new consolidated format
            try:
                success = save_guaranteed_signals_to_parquet(signals, next_trading_day)
                if success:
                    if verbose:
                        dprint(f"Successfully saved {len(signals)} signals to consolidated format", "SUCCESS")
                else:
                    if verbose:
                        dprint("Failed to save signals to consolidated format", "ERROR")
            except Exception as e:
                if verbose:
                    dprint(f"Error saving to consolidated format: {str(e)}", "ERROR")
        else:
            if verbose:
                dprint("No buy candidates to save to consolidated format", "WARN")
    
        # Execute trades as normal
        # OVERSUBSCRIPTION: with a limit entry most orders do not fill, so cap the number
        # SENT rather than the number of slots. held is ground truth (getposition), not
        # the open_positions counter, which increments at submission.
        if BT_GATE_MAXRANGE_DROP != 0 and len(buy_candidates) > 2:
            def _mr10(_d):
                # EMPTY-SLICE TRAP: backtrader returns an EMPTY array until 10 bars exist.
                # Return -1 so short-history names sort as NARROWEST and are never dropped,
                # and count them so this can never fail silently the way the w52 gate did.
                try:
                    _h = _d.high.get(size=10); _l = _d.low.get(size=10); _c = _d.close.get(size=10)
                except Exception:
                    _h = None
                if not _h or len(_h) < 10:
                    self._gate_nohist = getattr(self, '_gate_nohist', 0) + 1
                    return -1.0
                return max((100.0 * (hi - lo) / cl) for hi, lo, cl in zip(_h, _l, _c) if cl)
            _k = int(round(len(buy_candidates) * abs(BT_GATE_MAXRANGE_DROP)))
            if _k > 0:
                _scored = [(_mr10(_t[0]), _i) for _i, _t in enumerate(buy_candidates)]
                _rev = BT_GATE_MAXRANGE_DROP < 0      # negative frac = drop NARROWEST
                _drop = {_i for _, _i in sorted(_scored,
                          key=lambda z: ((z[0], z[1]) if _rev else (-z[0], z[1])))[:_k]}
                buy_candidates = [_t for _i, _t in enumerate(buy_candidates) if _i not in _drop]
                self._gate_dropped = getattr(self, '_gate_dropped', 0) + len(_drop)
        _oversub = BT_LIMIT_OVERSUB if (BT_LIMIT_ENTRY_K > 0) else 1.0
        _held = sum(1 for _d in self.datas if self.getposition(_d).size > 0)
        _free = max(0, int(self.p.max_positions) - _held)
        _budget = int(math.ceil(_free * _oversub)) if _oversub > 1 else self.p.max_positions
        _sent_today = 0
        if BT_LIMIT_SHUFFLE and BT_LIMIT_ENTRY_K > 0 and _budget < len(buy_candidates):
            # keep the SAME selected set (top _budget by rank), shuffle only the order
            # they are sent in, so this prices fill-order advantage and nothing else.
            _head = list(buy_candidates[:_budget])
            random.Random(BT_LIMIT_SHUFFLE + len(self)).shuffle(_head)
            buy_candidates = _head + list(buy_candidates[_budget:])
        if BT_GATE_PRIORRET_MAX > 0 and buy_candidates:
            def _r1ok(_d):
                try:
                    _c0, _c1 = float(_d.close[0]), float(_d.close[-1])
                except (IndexError, TypeError, ValueError):
                    return True
                if not (_c1 > 0):
                    return True
                return (100.0 * (_c0 / _c1 - 1.0)) <= BT_GATE_PRIORRET_MAX
            _keep = [_t for _t in buy_candidates if _r1ok(_t[0])]
            self._priorret_dropped = getattr(self, '_priorret_dropped', 0) + (len(buy_candidates) - len(_keep))
            self._priorret_seen = getattr(self, '_priorret_seen', 0) + len(buy_candidates)
            buy_candidates = _keep
        if (BT_GATE_GAPFREQ_N > 0 or BT_GATE_WEAKCLOSE > 0 or BT_GATE_VOLSPIKE_N > 0 ) and buy_candidates:
            def _hyg_ok(_d):
                try:
                    _o = _d.open.get(size=21); _h = _d.high.get(size=10); _l = _d.low.get(size=10)
                    _c = _d.close.get(size=21)
                except Exception:
                    return True
                if not _c or len(_c) < 21:
                    self._hyg_nohist = getattr(self, '_hyg_nohist', 0) + 1
                    return True
                if BT_GATE_GAPFREQ_N > 0:
                    _m = 'abs'
                    _need = 64 if _m == 'accel' else 21
                    if _m == 'accel':
                        try:
                            _o = _d.open.get(size=64); _c = _d.close.get(size=64)
                        except Exception:
                            _o = None
                        if not _c or len(_c) < 64:
                            self._hyg_nohist = getattr(self, '_hyg_nohist', 0) + 1
                            return True
                    _n = len(_c)
                    _dates = [_d.datetime.date(-(_n - 1 - j)) for j in range(_n)]
                    _atrp = None
                    if _m == 'atr':
                        try:
                            _h21 = _d.high.get(size=21); _l21 = _d.low.get(size=21)
                            _tr = [max(_h21[i] - _l21[i], abs(_h21[i] - _c[i-1]), abs(_l21[i] - _c[i-1])) for i in range(7, 21)]
                            _atrp = 100.0 * (sum(_tr) / 14.0) / _c[-1] if _c[-1] else None
                        except Exception:
                            _atrp = None
                    _bt = _beta_for(_d._name, self.datetime.date()) if _m == 'beta' else 1.0
                    def _hit(i):
                        if not _c[i-1]:
                            return False
                        g = 100.0 * (_o[i] / _c[i-1] - 1.0)
                        if _m in ('abs', 'accel'):
                            return abs(g) > BT_GATE_GAP_PCT
                        if _m == 'down':
                            return g < -BT_GATE_GAP_PCT
                        if _m == 'up':
                            return g > BT_GATE_GAP_PCT
                        sg = _spy_gap(_dates[i])
                        if _m == 'idio':
                            return abs(g - sg) > BT_GATE_GAP_PCT
                        if _m == 'beta':
                            return abs(g - _bt * sg) > BT_GATE_GAP_PCT
                        if _m == 'disagree':
                            return abs(g) > BT_GATE_GAP_PCT and abs(sg) > 0.2 and (g > 0) != (sg > 0)
                        if _m == 'atr':
                            return _atrp is not None and abs(g) > 1.5 * _atrp
                        return False
                    if _m == 'accel':
                        _rec = sum(1 for i in range(_n - 15, _n) if _hit(i))
                        _lng = sum(1 for i in range(1, _n) if _hit(i))
                        _fire = _rec >= BT_GATE_GAPFREQ_N and (_rec / 15.0) > 2.0 * max(_lng / 63.0, 1.0 / 63.0)
                    else:
                        _fire = sum(1 for i in range(1, _n) if _hit(i)) >= BT_GATE_GAPFREQ_N
                    if _fire:
                        self._hyg_gap = getattr(self, '_hyg_gap', 0) + 1
                        return False
                if BT_GATE_VOLSPIKE_N > 0:
                    try:
                        _v = _d.volume.get(size=20)
                    except Exception:
                        _v = None
                    if _v and len(_v) == 20:
                        _med = sorted(_v)[10]
                        if _med > 0 and sum(1 for x in _v if x > 3.0 * _med) >= BT_GATE_VOLSPIKE_N:
                            self._hyg_vol = getattr(self, '_hyg_vol', 0) + 1
                            return False
                if BT_GATE_WEAKCLOSE > 0:
                    _c10 = _c[-10:]
                    _w = sum(1 for hi, lo, cl in zip(_h, _l, _c10) if hi > lo and (cl - lo) / (hi - lo) < 0.25)
                    if _w / 10.0 > BT_GATE_WEAKCLOSE:
                        self._hyg_weak = getattr(self, '_hyg_weak', 0) + 1
                        return False
                return True
            _n0 = len(buy_candidates)
            buy_candidates = [_t for _t in buy_candidates if _hyg_ok(_t[0])]
            self._hyg_seen = getattr(self, '_hyg_seen', 0) + _n0
        self._panic_today = False
        if BT_PANIC_GATE and _oversub > 1 and _budget > 0 and _panic_gated(current_date, BT_PANIC_MODE):
            self._panic_days = getattr(self, '_panic_days', 0) + 1
            self._panic_today = True
            _nb = {'skip': 0, 'half': int(math.ceil(_budget / 2.0)), 'double': 2 * _budget,
                   'kdown': _budget}[BT_PANIC_ACTION]
            self._panic_cut = getattr(self, '_panic_cut', 0) + (_budget - _nb)
            logging.info('PANIC-GATE %s: budget %d -> %d', current_date, _budget, _nb)
            _budget = _nb
        for d, size, _ in buy_candidates:
            _room = (_sent_today < _budget) if _oversub > 1                 else (self.open_positions < self.p.max_positions)
            if _room:
                if self.check_group_allocation(d):
                    self.execute_buy(d, size, current_date)
                    _sent_today += 1
                    if verbose:
                        dprint(f"Executed buy order for {d._name}", "SUCCESS")
                else:
                    if verbose:
                        dprint(f"Skipped {d._name} due to group allocation limits", "INFO")
            else:
                if verbose:
                    dprint("Max positions reached, stopping buy executions", "INFO")
                break
            







    # ─────────────────────────────────────────────────────────────────────────
    # CANDIDATE-RANKING POLICY  (--selrule / BT_SELRULE)     added 2026-07-28
    #
    # Slots are filled in the order this returns, highest first. Everything else
    # -- can_buy, slot count, position sizing -- is untouched, so a rule can only
    # change WHICH qualifying names get the ~4 slots that recycle each day.
    # 'shipped' is the production order (raw UpProbability) and is bit-identical
    # to the pre-2026-07-28 code path.
    #
    # EVIDENCE (full slot sim, asof fork, BT_SAMPLE_SEED=42, 6 BT_AS_OF anchors
    # spanning 2025-04-30..2026-07-27 with baselines from -18.34 to +132.76 ann):
    #
    #   low_atr        mean +14.07pp  sd  9.15  t=+3.76  p=0.013  6/6 anchors
    #                  positive, worst anchor +2.31. Dose curve 0.25/0.50/0.75/1.00
    #                  climbs monotonically to 0.50 at 6/6 anchors, then variance
    #                  doubles (sd 9.15 -> 19.87) and it goes 4/6. 0.50 is the
    #                  largest dose with no losing anchor -- a minimax default, NOT
    #                  a broad plateau. Do not raise it without re-running the curve.
    #
    #   low_atr_x_corr mean +32.29pp  sd 35.08  t=+2.25  p=0.074  5/6 anchors.
    #                  Bigger but far noisier; uniquely rescues the two windows whose
    #                  baseline is negative (-18.34 -> +2.03, -0.49 -> +53.81).
    #                  Leans on Correlations.parquet, which is 5 months stale and
    #                  covers 3145/4250 tickers (the missing 26% score 0 =
    #                  "uncorrelated"). Refresh that table before trusting this.
    #
    # WHY low_atr works: calculate_position_size allocates risk_amount/(2*ATR), so
    # at equal conviction a lower-ATR name receives a LARGER position. Ranking by
    # ATR-penalised conviction points entries at the names the sizer already backs
    # hardest, so ranking and sizing compound instead of acting independently. It is
    # an entry x sizing interaction, which is why per-pick screens cannot see it --
    # a 691-day per-pick screen measured re-ranking at +-0.4pp/pick and missed this.
    #
    # REFUTED, do not re-try (same rig, 2 anchors each, full slot sim):
    #   replacing the level with a within-ticker normalisation -- up/sigma -78.59pp,
    #   own_z -55.80pp. The cross-sectional LEVEL is load-bearing; transforms are
    #   safe as tilts and lethal as replacements.
    #   dynamic per-name pairwise correlation (30/60 bar, flat/EWM, mean/max,
    #   1.0x/1.5x) -- 12/12 negative, -10.71 to -64.89pp, vs the static cluster
    #   table's +41.80 at the same anchor. Pairwise |corr| over 30-60 bars is
    #   dominated by sampling error; the cluster table wins because averaging over
    #   many names is a low-variance estimate of structural co-movement.
    # ─────────────────────────────────────────────────────────────────────────
    SELRULES = ('shipped', 'low_atr', 'low_atr_x_corr')

    def _selrule_score(self, data, correlation):
        rule = (os.environ.get('BT_SELRULE') or 'shipped').strip().lower()
        up = float(data.UpProbability[0])


        # BT_REENTRY_BOOST (experimental, 2026-07-31): additive selection bonus for
        # FRESH re-entries -- candidates whose last exit was <= BT_REENTRY_BOOST_D
        # (default 2) calendar days ago. The census's strongest bucket is the <=2d
        # re-entry (+1.85%/trade, 60% WR); this prefers them in the ranking instead
        # of blocking anything. 0/unset = off, byte-identical.
        _boost = 0.0
        if _boost:
            _last = getattr(self, '_last_exit', {}).get(data._name)
            if _last:
                try:
                    _gap = (self.datetime.date() - _last[0]).days
                    _ok = 0 <= _gap <= 2
                    if _ok:
                        up += _boost
                        self._reentry_boosts = getattr(self, '_reentry_boosts', 0) + 1
                except TypeError:
                    pass

        # Soft cooldown: dock matching re-entry candidates instead of blocking them.
        if self.reentry_soft_penalty and self._reentry_condition(
                data._name, self.datetime.date(), data):
            up -= self.reentry_soft_penalty
            self._reentry_penalties += 1

        if rule == 'shipped':
            return up

        # ATR% of price: the quantity the position sizer inverts to set size.
        atr_pct = 0.0
        try:
            atr = float(self.inds[data]['atr'][0])
            px = float(data.close[0])
            if px > 0 and np.isfinite(atr):
                atr_pct = atr / px
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
            atr_pct = 0.0

        k = 0.5
        if rule == 'low_atr':
            return up - k * atr_pct
        if rule == 'low_atr_x_corr':
            c = 0.0
            if correlation is not None:
                try:
                    cf = float(correlation)
                    if np.isfinite(cf):
                        c = cf
                except (TypeError, ValueError):
                    c = 0.0
            return (up - k * atr_pct) * (1.0 - c)

        # Unknown name: abort rather than silently scoring as 'shipped'. A typo that
        # no-ops looks exactly like a rule that does nothing, which is how this repo
        # has previously A/B'd disconnected knobs for weeks.
        raise ValueError("Unknown BT_SELRULE %r; expected one of %s"
                         % (rule, ', '.join(self.SELRULES)))

    def _v2_rank_score(self, data, correlation, band, tie):
        """Band-coarsened UpProbability with a bounded secondary tie-break.

        THE PROBLEM THIS SOLVES. UpProbability is the best ranker measured (+0.3635 at
        4 slots against random's -0.0331) and every attempt to blend a second signal
        into it lost, because blending lets a lower-conviction name outrank a higher one.
        The composite up-0.5mom-0.2rsi scored +0.2735, well under shipped.

        THE CONSTRUCTION. Floor UpProbability onto a grid of width `band`, then add a
        secondary term scaled to at most 0.9 x band. The secondary can therefore reorder
        names only INSIDE one grid cell and can never bridge a cell boundary. Names the
        model genuinely separates keep their order; names it scores within `band` of each
        other, which it cannot really distinguish, get ordered by RSI instead of by the
        arbitrary feed order they currently fall in.

        WHY RSI IS THE TIE-BREAK. It is the secondary signal with a monotone effect
        inside every UpProbability quartile, and it is already computed for the gate.

        HONEST SIZE OF THE EFFECT. With the v2 RSI ceiling in place this is worth about
        +0.006pp at 4 slots (+0.6121 vs +0.6065), which is nothing. It earns its place
        for two other reasons: it never lost a cell in testing, and WITHOUT the ceiling
        it is worth +0.081pp (+0.4442 vs +0.3635), so it is the insurance that keeps the
        ranking sane if the gate is ever relaxed. BT_V2_BAND=0 disables it entirely and
        restores the shipped sort byte-for-byte.
        """
        up = self._selrule_score(data, correlation)   # keeps reentry boost / soft cooldown
        try:
            prim = np.floor(float(up) / band) * band
        except (TypeError, ValueError, ZeroDivisionError):
            return up
        sec = 0.5
        if tie == 'rsi':
            r = self._rsi14(data)
            if r is not None:
                sec = 1.0 - min(max(float(r), 0.0), 100.0) / 100.0
        elif tie == 'mom':
            m = self._ret5(data)
            if m is not None:
                sec = 1.0 - min(max((float(m) + 0.2) / 0.4, 0.0), 1.0)
        return prim + sec * band * 0.9

    def sort_buy_candidates(self, buy_candidates):
        # v2 ranking. See _v2_rank_score for the design and the measured effect.
        # BT_V2_BAND=0 (or BT_V2_TIEBREAK=off) restores the shipped sort exactly.
        #
        # Shipped behaviour, kept below as the fallback:
        # 2026-05-15: flipped to descending after EDA showed the asc sort discarded
        # the model's highest-conviction qualifying candidates. Sharpe 0.80 -> 3.66.
        # x = (data, size, correlation); correlation was computed in
        # get_buy_candidates and, before 2026-07-28, discarded here.
        # DEFAULT DISARMED 2026-08-23: worth ~+0.006pp with a gate present, i.e.
        # nothing. Set BT_V2_BAND=0.002 to re-arm.
        # ---------------- BT_VOLTILT (2026-08-26) ----------------
        # Rank on the universe-wide day rank of raw_score plus BT_VOLTILT times the
        # universe-wide day rank of vol_20d. 0 or unset leaves the shipped sort
        # byte-identical, which is the rollback.
        #
        # WHY, and what it is NOT. This is not another ordering formula. Two prior rigs
        # closed ordering work, and this rig agrees with them wherever the input is the
        # candidate set: sixteen candidate-side formulas were tested here and the best
        # reached paired t 1.65. What is different is the INPUT. UniRank* is the first
        # cross-sectional fact the ranker has ever been given; every previous attempt
        # re-sorted the same per-name numbers.
        #
        # MEASURED on the A1 gate ledger through a 3-slot book with the faithful runner
        # bracket (219,655 rows, 800 tickers, 284 days), net of Tiered commission at the
        # $0.35 floor plus 24bp spread:
        #   shipped UpProbability   +0.1158%/trade   eq 1.140   dd 28.8%   z +2.21
        #   w = 0.30                +0.4588%/trade   eq 1.951   dd 23.4%   z +4.50
        #   time thirds             +0.682 / +0.211 / +0.483   (the only arm 3 of 3)
        #   paired subsample        +0.212 mean, t +5.83, 92% of 25 draws
        #
        # HONEST CAVEATS. Found among roughly 48 comparisons. Tail-rate quintiles of
        # vol_20d inside the passing pool are monotone but MEAN-payoff quintiles are
        # not, so the mechanism is only half pinned. It does NOT replicate on a tighter
        # gate (band off, floor 0.55), where the random-ranker floor rises to +0.534 and
        # nothing clears it, which is consistent with gate quality and ranker value
        # being substitutes. The ledger's daily pool is ~47 deep against production's
        # ~250, so its LEVELS are not production's. Deltas only.
        _voltilt = 0.0
        if _voltilt:
            def _vt(x):
                d0 = x[0]
                try:
                    r = float(d0.UniRankRaw[0])
                    v = float(d0.UniRankVol[0])
                except (IndexError, AttributeError, TypeError, ValueError):
                    return -1e9
                if not (np.isfinite(r) and np.isfinite(v)):
                    return -1e9
                self._voltilt_scored = getattr(self, '_voltilt_scored', 0) + 1
                if r == 0.5 and v == 0.5:
                    self._voltilt_neutral = getattr(self, '_voltilt_neutral', 0) + 1
                return r + _voltilt * v
            sorted_candidates = sorted(buy_candidates, key=_vt, reverse=True)
            _band = _tie = None
        else:
            _band = 0.0
            _tie = 'rsi'

        if _band is None:
            pass                      # already sorted by the vol tilt above
        elif _band > 0 and _tie in ('rsi', 'mom'):
            sorted_candidates = sorted(
                buy_candidates,
                key=lambda x: self._v2_rank_score(x[0], x[2], _band, _tie), reverse=True)
        elif _band <= 0 or _tie == 'off':
            sorted_candidates = sorted(
                buy_candidates, key=lambda x: self._selrule_score(x[0], x[2]), reverse=True)
        else:
            # Unknown tie-break name aborts rather than silently scoring as shipped. A
            # typo that no-ops looks exactly like a rule that does nothing, which is how
            # this repo has previously A/B'd disconnected knobs for weeks.
            raise ValueError("Unknown BT_V2_TIEBREAK %r; expected rsi, mom or off" % _tie)

        # BT_SKIP_TOP (default 0 = shipped behaviour, no-op).
        # *** REFUTED 2026-07-25. DO NOT ENABLE. *** Full sim vs base 59.97:
        # skip1 44.21, skip3 22.82, skip15 8.94, skip20 38.44. Dropping even the
        # single highest-probability name costs ~16pp annualised. The top of the
        # book is where the edge is. Kept only as a reproducible negative result.
        # Original rationale follows: drops the N
        # highest-probability qualifying names before slots are filled, to test
        # whether the extreme top of the probability curve is worth buying. The
        # shipped model's own report says it may not be: on the calib slice
        # top-1%-precision 0.2559 sits BELOW top-5%-precision 0.2893 against a
        # 0.2003 base rate. Guarded so it can never empty the book.
        _skip = 0
        if _skip > 0 and len(sorted_candidates) > _skip:
            sorted_candidates = sorted_candidates[_skip:]


        Verbose = False

        if Verbose:
            if sorted_candidates:
                logging.info("Top buy candidates based on UpProbability:")
                for i, (d, size, corr) in enumerate(sorted_candidates[:min(5, len(sorted_candidates))]):
                    logging.info(f"  {i+1}. {d._name}: UpProb={d.UpProbability[0]:.4f}")

        return sorted_candidates
    





    



    def get_mean_correlation(self, candidate_ticker, current_positions):
        try:
            if not current_positions:
                return 0

            if candidate_ticker not in self.correlation_df.index:

                ##logging.warning(f"Ticker {candidate_ticker} not found in correlation data")
                return 0

            correlations = []

            candidate_row = self.correlation_df.loc[candidate_ticker]

            for pos in current_positions:
                if pos not in self.correlation_df.index:
                    continue

                position_cluster = self.correlation_df.loc[pos, 'Cluster']

                cluster_column = f"correlation_{position_cluster}"
                if cluster_column in self.correlation_df.columns:
                    corr_value = candidate_row[cluster_column]
                    correlations.append(abs(corr_value))  # Use absolute correlation

            return np.mean(correlations) if correlations else 0

        except Exception as e:
            logging.error(f"Error calculating correlations: {str(e)}")
            return 0
    




    def check_group_allocation(self, data):
        """Check if adding a position would exceed group allocation limits.
        Also reject stocks that are in the outlier group (-1).
        """
        symbol = data._name

        # Find which group the stock belongs to
        group = None
        try:
            if hasattr(self.correlation_df, 'index') and hasattr(self.correlation_df.index, 'contains'):
                if symbol in self.correlation_df.index:
                    group = int(self.correlation_df.loc[symbol, 'Cluster'])
            elif 'Ticker' in self.correlation_df.columns:
                ticker_row = self.correlation_df[self.correlation_df['Ticker'] == symbol]
                if not ticker_row.empty:
                    group = int(ticker_row['Cluster'].iloc[0])
            else:
                first_col = self.correlation_df.columns[0]
                ticker_row = self.correlation_df[self.correlation_df[first_col] == symbol]
                if not ticker_row.empty:
                    group = int(ticker_row['Cluster'].iloc[0])
        except Exception as e:
            logging.warning(f"Error finding group for {symbol}: {str(e)}")

        # Reject stocks in the outlier group (-1)
        if group == -1:
            logging.info(f"Rejecting {symbol} because it's in the outlier group (-1)")
            return False

        if group is None:
            logging.info(f"No group found for {symbol}, allowing trade")
            return True  # If no group data, allow the trade

        # Check if adding this stock would exceed the group allocation limit
        current_allocation = self.group_allocations.get(group, 0)
        return current_allocation < self.p.max_group_allocation




    def execute_buy(self, data, size, current_date):
        """Replace the old execute_buy with bracket order version"""
        return self.execute_buy_with_bracket(data, size, current_date)
    



    def execute_buy_with_bracket(self, data, size, current_date):
        """Execute buy using the ONE bracket defined in bracket_config.py.

        Before 2026-07-28 this hardcoded a 3% trailing stop and a +20% target, which is
        NOT what 9_SuperFastBroker.py sends. The backtest was therefore simulating a
        different strategy from the one being traded -- and because self.trailing_stops
        was never assigned, it also reported 0.0% stop-outs while the intraday fill sim
        measured ~47-53% on the same book. Both are fixed here: the bracket comes from
        the shared config, and the stop level is recorded so exits can be labelled.
        """
        symbol = data._name
        current_price = data.close[0]

        if BRACKET.LEGACY_BACKTEST:
            # Rollback path: reproduce the pre-consolidation numbers on demand.
            trailing_stop_percent = BRACKET.LEGACY_TRAIL_PCT
            target_price = current_price * (1 + BRACKET.LEGACY_TP_PCT / 100.0)
            stop_kwargs = dict(stopexec=bt.Order.StopTrail,
                               trailpercent=trailing_stop_percent / 100.0)
            stop_level = None
            logging.info(f"BUY BRACKET {symbol} [LEGACY]: Entry=${current_price:.2f}, "
                         f"Size={size}, TrailStop={trailing_stop_percent}%, "
                         f"Target=${target_price:.2f}")
            bracket_orders = self.buy_bracket(
                data=data, size=size, price=current_price,
                exectype=bt.Order.Market, limitprice=target_price,
                limitexec=bt.Order.Limit, **stop_kwargs)
        else:
            # Read through self.p, whose DEFAULTS come from bracket_config via Util,
            # so --take_profit and the optimizer overrides keep working.
            stop_pct = float(os.environ.get('BT_HARD_STOP_PCT',
                             getattr(self.p, 'hard_stop_percent', BRACKET.HARD_STOP_PCT)))
            tp_pct = getattr(self.p, 'take_profit_percent', BRACKET.TAKE_PROFIT_PCT)

            # ANCHORING FIX 2026-07-29. buy_bracket needs absolute child prices at
            # submission time, but a Market parent does not fill until the NEXT bar's
            # open, so those children were priced off the signal-day CLOSE. Audit on the
            # real book: only 10% of trades ended up with a stop near the intended 1.9%,
            # 27% had a stop at or ABOVE the fill, and effective stop distance ran from
            # -2.6% to +7.0%. The backtest was not testing a 1.9% stop at all.
            # 9_SuperFastBroker.py has no such problem: it anchors to the 10:00 mid it
            # is filling at. So submit a bare entry here and attach the legs in
            # handle_buy_execution, off order.executed.price.
            # LIMIT-ENTRY EXPERIMENT. K=0 (default) keeps the shipped Market entry.
            _depth = 0.0
            if BT_LIMIT_ENTRY_K > 0:
                try:
                    _need = _VOL_BARS[BT_LIMIT_VOL_EST]
                    _cl = data.close.get(size=_need)
                    _hi = data.high.get(size=_need)
                    _lo = data.low.get(size=_need)
                    _OPEN_FOR_VOL['open'] = data.open.get(size=_need) if BT_LIMIT_VOL_EST == 'yz' else None
                    _v20 = _vol_estimate(_cl, _hi, _lo)
                    self._v20_by_data = getattr(self, '_v20_by_data', {})
                    self._v20_by_data[data] = _v20
                    if _v20 is not None and _v20 > 0:
                            _bt = _beta_for(symbol, current_date)
                            _basis = _bt * _v20
                            _depth = BT_LIMIT_ENTRY_K * _basis
                            if BT_K_FRI_MULT != 1.0 and current_date.weekday() == 4:
                                _depth *= BT_K_FRI_MULT
                                self._frimult_n = getattr(self, '_frimult_n', 0) + 1
                            if BT_PANIC_GATE and BT_PANIC_ACTION == 'kdown' and getattr(self, '_panic_today', False):
                                _depth *= BT_PANIC_KMULT
                                self._panic_kdown = getattr(self, '_panic_kdown', 0) + 1
                            _depth = min(max(_depth, 0.005), 0.15)
                except Exception as _e:
                    # A silent fallback to Market here would make the whole experiment
                    # read as a perfect null. Count it and shout.
                    self._limit_depth_fail = getattr(self, '_limit_depth_fail', 0) + 1
                    if self._limit_depth_fail <= 5:
                        logging.error("LIMIT-ENTRY depth FAILED for %s (%s); market entry",
                                      symbol, _e)
                    _depth = 0.0

            if _depth > 0:
                _limit_px = current_price * (1.0 - _depth)
                main_order = self.buy(data=data, size=size, exectype=bt.Order.Limit,
                                      price=_limit_px)
                if main_order is None:
                    return None
                self.pending_entry_limits[data] = (main_order, len(self))
                self._limit_trigger[data] = (_limit_px, current_price, _depth)
                self._limit_sent += 1
                logging.info("BUY LIMIT %s: %.2f (-%.2f%% off %.2f), one bar only",
                             symbol, _limit_px, 100 * _depth, current_price)
            else:
                main_order = self.buy(data=data, size=size, exectype=bt.Order.Market)
                if main_order is None:
                    return None
            _vs = getattr(self, '_v20_by_data', {}).get(data)
            self.pending_bracket[data] = dict(stop_pct=stop_pct, tp_pct=tp_pct, size=size, vol=_vs)
            self.entry_prices[data] = current_price      # provisional, corrected on fill
            self.position_dates[data] = current_date
            self.position_bars[data] = len(data)         # signal bar, for the session clock
            # Stamp the ENTRY-time signal (see self.entry_up_prob in __init__).
            self.entry_up_prob[data] = float(data.UpProbability[0])
            self.open_positions += 1
            self.update_group_data(data)
            update_buy_signal(symbol, current_date, current_price, data.UpProbability[0])
            logging.info(f"BUY {symbol}: Size={size}, bracket deferred to fill "
                         f"(-{stop_pct}% / +{tp_pct}% off the ACTUAL fill)")
            return main_order

        if bracket_orders:
            main_order, stop_order, target_order = bracket_orders

            # Store order info for tracking
            self.entry_prices[data] = current_price
            self.position_dates[data] = current_date
            self.position_bars[data] = len(data)         # signal bar, for the session clock
            # Stamp the ENTRY-time signal (see self.entry_up_prob in __init__).
            self.entry_up_prob[data] = float(data.UpProbability[0])
            self.open_positions += 1
            # RECORD THE STOP. determine_exit_reason reads this; leaving it unset is
            # what made "Stop Loss" unreachable and mislabelled 82% of exits.
            if stop_level is not None:
                self.stop_levels[data] = stop_level

            # Track the bracket orders
            self.bracket_orders[data] = {
                'main': main_order,
                'stop': stop_order,
                'target': target_order,
                'entry_price': current_price,
                'stop_type': 'trailing' if BRACKET.LEGACY_BACKTEST else 'hard',
                'stop_level': stop_level,
                'target_price': target_price
            }

            self.update_group_data(data)
            update_buy_signal(symbol, current_date, current_price, data.UpProbability[0])

            return True

        return False




    def update_group_data(self, data):
        try:
            symbol = data._name
            
            if symbol in self.correlation_df.index:
                group = int(self.correlation_df.loc[symbol, 'Cluster'])
                self.asset_groups[symbol] = group
            
            self.update_group_allocations()
            
        except Exception as e:
            logging.error(f"Error updating group data: {str(e)}")
    



    def update_group_allocations(self):
        """Update the allocation percentages for each cluster group based on current positions."""
        # Calculate total portfolio value
        total_value = self.broker.getvalue()

        # Initialize all possible group IDs including:
        # - Regular groups (could start at 0 or 1 depending on clustering algorithm)
        # - Special outlier group (-1)
        # - Any possible group values in asset_groups

        # First reset allocations for numbered groups
        self.group_allocations = {}

        # Initialize all potential group IDs from 0 to total_groups
        for i in range(0, self.total_groups + 1):
            self.group_allocations[i] = 0.0

        # Also add the outlier group (-1)
        self.group_allocations[-1] = 0.0

        # Collect any additional group IDs that might exist in asset_groups
        for symbol, group in self.asset_groups.items():
            if group not in self.group_allocations:
                self.group_allocations[group] = 0.0

        # If portfolio is empty, set equal allocations and return
        if total_value == 0:
            group_count = len([g for g in self.group_allocations.keys() if g >= 0])  # Don't count outlier group
            if group_count > 0:
                equal_alloc = 1.0 / group_count
                for group in self.group_allocations.keys():
                    if group >= 0:  # Only allocate to real groups, not outliers
                        self.group_allocations[group] = equal_alloc
            return

        # Update allocations based on current positions
        for data in self.datas:
            position = self.getposition(data)
            if position.size > 0:
                symbol = data._name

                # Default to outlier group if not found
                group = self.asset_groups.get(symbol, -1)

                # Make sure the group is in our allocations dict (defensive programming)
                if group not in self.group_allocations:
                    logging.warning(f"Found unexpected group ID {group} for {symbol}, adding to allocations")
                    self.group_allocations[group] = 0.0

                # Calculate position value
                position_value = position.size * data.close[0]

                # Update allocation - now safe since we've handled all possible groups
                self.group_allocations[group] += position_value / total_value













    def _runner_manage(self, data, current_date, days_held):
        """Runner exits: at +trig% unrealized, sell all but keep_frac of the position
        and let the runner ride the repegged stop up to runner_maxhold days.

        Returns True when this position is (or just became) a runner, which tells the
        caller to skip the standard momentum/timeout exits for it. Cancel-verify-defer
        identical to exit_position; scale-out + reattached leg sizes sum exactly to
        the position, so both filling together lands flat, never short."""
        symbol = data._name
        position = self.getposition(data)

        if data in self.runner_done:
            if days_held >= _runner_maxhold():
                logging.info(f"RUNNER TIMEOUT {symbol}: held {days_held}d")
                self.exit_position(data)
            return True

        entry = self.entry_prices.get(data)
        if not entry:
            return False
        if data.close[0] < entry * (1.0 + _runner_trig() / 100.0):
            return False

        working = self.exit_orders.get(data)
        if working is not None and working.alive():
            return False

        bracket = self.bracket_orders.get(data)
        legs = [bracket.get('stop'), bracket.get('target')] if bracket else []
        for leg in legs:
            if leg is not None and leg.alive():
                try:
                    self.cancel(leg)
                except Exception as e:
                    logging.warning(f"RUNNER cancel failed for {symbol}: {e}")
        if any(leg is not None and leg.alive() for leg in legs):
            return False  # legs still in broker.submitted; retry next bar

        _kf = _runner_keep()
        keep = max(1, int(position.size * _kf))
        sell_n = position.size - keep
        if sell_n <= 0:
            return False

        self.sell(data=data, size=sell_n, exectype=bt.Order.Market)
        teod = _trail_eod()
        lvl = (data.close[0] * (1 - teod / 100.0) if teod > 0
               else entry * (1 - BRACKET.HARD_STOP_PCT / 100.0))
        new_stop = self.sell(data=data, size=keep, exectype=bt.Order.Stop, price=lvl)
        self.stop_levels[data] = lvl
        self.bracket_orders[data] = {
            'main': None, 'stop': new_stop, 'target': None,
            'entry_price': entry, 'stop_type': 'runner_eod',
            'stop_level': lvl, 'target_price': None}
        self.runner_done.add(data)
        logging.info(f"RUNNER SCALE-OUT {symbol}: sold {sell_n} at +{_runner_trig():g}%, "
                     f"keeping {keep}, stop ${lvl:.2f}, max {_runner_maxhold()}d")
        return True

    def evaluate_sell_conditions(self, data, current_date):
        """Enhanced: Timeout + 5-day Momentum-based exit"""
        symbol = data._name
        position = self.getposition(data)

        if position.size <= 0:
            return

        entry_date = self.position_dates.get(data, current_date)
        days_held = (current_date - entry_date).days

        # Session clock (BT_CLOCK_SESSIONS, see the knob's comment block). Translates
        # sessions since the signal bar into day units so every comparison below
        # (timeout, runner clock, hold extension, pyramid guard) is reused unchanged.
        if BT_CLOCK_SESSIONS > 0:
            _b0 = self.position_bars.get(data)
            if _b0 is not None:
                days_held = (len(data) - _b0) * float(self.p.position_timeout) / BT_CLOCK_SESSIONS

        # ── CLOSE-THROUGH CONFIRMATION (BT_CLOSETHRU=1, default OFF) ──────────────
        # The resting Stop is an intraday trigger: it fires the moment price TOUCHES the
        # level, so a liquidity sweep that closes back above the stop still books a loss.
        # Measured on the 807-trade book: of 403 stop-breach losers, the 178 that CLOSED
        # BACK ABOVE the stop recovered to breakeven 80.9% of the time vs 53.8% for the
        # 225 that closed below, and deferring those one session moved the realised result
        # -3.00% -> -0.74%.
        #
        # This arm replaces the intraday trigger with an END-OF-DAY test. The stop is
        # STILL ARMED ON EVERY BAR - it simply requires the bar to CLOSE through the level
        # rather than merely touch it. A name that closes below the stop is sold at once.
        # Protection is never removed: on close-through days this exits on the same bar as
        # before, and the only positions it holds are wicks that closed back above.
        if _closethru() and position.size > 0:
            lvl = self.stop_levels.get(data)
            if lvl:
                if float(data.close[0]) <= lvl:
                    logging.info(f"CLOSETHRU STOP {symbol}: close "
                                 f"${float(data.close[0]):.2f} <= stop ${lvl:.2f}")
                    # exit_position() owns the whole teardown: it refuses a second
                    # working close, cancels BOTH bracket legs, and defers when a leg is
                    # not yet cancellable. Hand-rolling that here is what produced
                    # OVER-SOLD BOOK (target leg filling against an already-closed long).
                    self.exit_position(data)
                    return
                # No close-through: neutralise the INTRADAY stop leg so a mere touch
                # cannot fire it. Only ever done while the position is still open and
                # the EOD test above is active, so the stop is never unguarded - it is
                # evaluated on this bar's close instead of on the touch.
                br = self.bracket_orders.get(data)
                leg = br.get('stop') if br else None
                if leg is not None and leg.alive():
                    try:
                        self.cancel(leg)
                        if not leg.alive():
                            br['stop'] = None
                    except Exception as e:
                        logging.warning(f"CLOSETHRU cancel failed {symbol}: {e}")

        if BT_PYRAMID_FRAC > 0:
            _pb = self._pyr; _k = (data, self.position_dates.get(data))
            for _r, (_o, _d) in list(_pb['orders'].items()):   # a limit add rests ONE bar
                if _d is data and _o.alive():
                    self.cancel(_o)                       # counted in handle_order_failure
            _ep = self.entry_prices.get(data) or 0.0; _c = float(data.close[0])
            _bars = len(data) - _pb['fillbar'].get(_k, 10 ** 9)
            if _k not in _pb['done'] and 0 <= _bars < BT_PYRAMID_DAY and _c > _ep > 0:
                _pb['done'].add(_k)                       # one decision per position, ever
                _xo = self.exit_orders.get(data)
                if (data in self.runner_done or (_xo is not None and _xo.alive())
                        or days_held + 2 > self.p.position_timeout
                        or _c >= _ep * (1.0 + _runner_trig() / 100.0)):
                    _pb['guard'] += 1
                else:
                    _eq = float(self.broker.getvalue()); _cash = float(self.broker.getcash())
                    _free = max(0, int(self.p.max_positions) - sum(1 for _d in self.datas if self.getposition(_d).size > 0))
                    _cash -= min(_free, BT_PYRAMID_RESERVE_SLOTS) * _eq * (1.0 - self.p.reserve_percent) / int(self.p.max_positions)
                    _n = min(int(BT_PYRAMID_FRAC * position.size), int(0.90 * _cash / _c))
                    _pb['att'] += 1
                    if _n <= 0:
                        _pb['sizer0'] += 1
                    elif BT_PYRAMID_GAP_MAX > 0:
                        _ao = self.buy(data=data, size=_n, exectype=bt.Order.Limit, price=_c * (1.0 + BT_PYRAMID_GAP_MAX / 100.0)); _pb['orders'][_ao.ref] = (_ao, data)
                    else:
                        _ao = self.buy(data=data, size=_n, exectype=bt.Order.Market); _pb['orders'][_ao.ref] = (_ao, data)
        # EOD-repegged stop (runner exits): once per day, raise the resting stop leg
        # to prior close * (1 - TRAIL_EOD_PCT), never lower it. This is the exact
        # instrument the broker creates by re-pegging a GTC STP each morning. Only
        # legs with no TP sibling are repriced (cancelling one OCO leg cancels the
        # other, which would silently strip the take-profit).
        _teod = _trail_eod()
        if _teod > 0:
            br = self.bracket_orders.get(data)
            leg = br.get('stop') if br else None
            if leg is None or not leg.alive():
                # RE-ARM FIX (fork, 2026-08-23). backtrader processes cancel()
                # asynchronously: the bar that issues it still sees leg.alive() True, so
                # no replacement is created; by the NEXT bar the leg is dead and the old
                # code fell into this branch and did nothing. The position was then
                # carrying NO stop for the rest of its life, and the EOD ratchet never
                # ran again on it. Census before the fix, sample 8 seed 42:
                #   opportunities 1649, fired 449 (27.2%), cancel 390, nostop 237.
                # That is 38% of ratchet opportunities lost and ~14% of position-days
                # left unprotected. Re-arm here instead of skipping.
                #
                # The level never goes below the original hard stop and never below the
                # level already reached, so this stays a ratchet and can only tighten.
                if br is not None and position.size > 0:
                    _hard = br.get('stop_level') or 0.0
                    _prev = self.stop_levels.get(data) or 0.0
                    _new = float(data.close[0]) * (1 - _teod / 100.0)
                    _lvl = max(_new, _prev, _hard * 0.0)   # never ratchet down
                    if _lvl > 0 and _closethru():
                        # CLOSETHRU FIX 2026-08-30. This re-arm resurrected the intraday
                        # leg one bar after the close-through block cancelled it, which
                        # left the touch-stop live during the next bar's broker matching
                        # and made the first BT_CLOSETHRU A/B a fake null (147/161 trades
                        # byte-identical, zero CLOSETHRU STOP lines). Under close-through
                        # the ratchet moves the LEVEL the close test reads; no order.
                        br['stop_level'] = _lvl
                        self.stop_levels[data] = _lvl
                        self._eod_rearmed += 1
                    elif _lvl > 0:
                        try:
                            _re = self.sell(data=data, size=position.size,
                                            exectype=bt.Order.Stop, price=_lvl)
                            br['stop'] = _re
                            br['stop_level'] = _lvl
                            self.stop_levels[data] = _lvl
                            self._eod_rearmed += 1
                        except Exception as e:
                            self._eod_blk_nostop += 1
                            logging.warning(f"EOD re-arm failed {symbol}: {e}")
                    else:
                        self._eod_blk_nostop += 1
                else:
                    self._eod_blk_nostop += 1
                    if br is None:
                        self._eod_nobr += 1
                    else:
                        self._eod_nosize += 1
            elif br.get('target') is not None:
                self._eod_blk_target += 1
            else:
                newlvl = float(data.close[0]) * (1 - _teod / 100.0)
                _cover = BT_PYRAMID_FRAC <= 0 or abs(int(leg.created.size)) == int(position.size)
                if newlvl <= (self.stop_levels.get(data) or 0.0) + 1e-9 and _cover:
                    self._eod_blk_notraise += 1
                else:
                    try:
                        self.cancel(leg)
                    except Exception as e:
                        logging.warning(f"EOD repeg cancel failed {symbol}: {e}")
                    if not leg.alive() and _closethru():
                        # CLOSETHRU FIX 2026-08-30: level only, no intraday leg (see the
                        # re-arm branch above for the defect this prevents).
                        br['stop'] = None
                        br['stop_level'] = newlvl
                        self.stop_levels[data] = newlvl
                        self._eod_fired += 1
                    elif not leg.alive():
                        if not _cover:
                            self._pyr['resize'] += 1; newlvl = max(newlvl, self.stop_levels.get(data) or 0.0)
                        new_leg = self.sell(data=data, size=position.size,
                                            exectype=bt.Order.Stop, price=newlvl)
                        br['stop'] = new_leg
                        br['stop_level'] = newlvl
                        self.stop_levels[data] = newlvl
                        self._eod_fired += 1
                    else:
                        # leg still in broker.submitted (entry bar); retry tomorrow
                        self._eod_blk_cancel += 1
                        _k = data._name
                        self._eod_cancel_datas[_k] = self._eod_cancel_datas.get(_k, 0) + 1

        # Runner hook: scaled-out positions ride their own clock and skip the
        # model-driven exits below.
        _lg = (self.bracket_orders.get(data) or {}).get('stop') if BT_PYRAMID_FRAC > 0 else None
        if _lg is not None and _lg.alive() and abs(int(_lg.created.size)) != int(position.size):
            self._pyr['mismatch'] += 1
        if _runner_on():
            if self._runner_manage(data, current_date, days_held):
                return

        try:
            current_prob = data.UpProbability[0]
            momentum = current_prob - data.UpProbability[-1]

            # *** REFUTED 2026-07-25. DO NOT ENABLE. *** Full sim vs base 59.97:
            # 50.22 / 52.85 / 51.41, and 83.44 when combined with the entry gate
            # (vs 102.48 for the entry gate alone). Slot recycling dominates: the
            # freed slot is worth more than the held position even though Manual
            # Exit is per-trade negative. Consistent with the 07-13 exit verdict.
            # BT_EXIT_PERSIST_EXEMPT: the momentum exit ("Manual Exit") loses money
            # in EVERY window tested (-4538 / -3220 / -2473) while Max-Hold exits
            # make money in every window (+6928 / +2076 / +2107). The persistence
            # ENTRY gate raised both together, which is why its net payoff was
            # unreliable. This targets the asymmetry directly: if a name's
            # probability is STILL durably above the floor, do not let a single
            # day's drop kick it out -- let it run to the max-hold clock.
            _exempt = False

            # BT_PROD_EXITS=1 disables every model-driven exit, leaving only the bracket
            # and the age clock. That is exactly what 9_SuperFastBroker.py does: the
            # broker never re-reads UpProbability after entry, so the momentum and
            # probability-drop exits below have no production counterpart. Set this to
            # measure the backtest under the exit policy that actually ships.
            _prod_exits = os.environ.get('BT_PROD_EXITS', '0') == '1'

            # Early exit conditions.
            # NOTE: despite the docstring this is a ONE-day probability change,
            # UpProbability[0] - UpProbability[-1]. The genuine 5-day version lives in
            # evaluate_sell_conditions__UpProbDropCondition, which is never called.
            # BT_MOMEXIT_MODE (2026-08-28). The momentum exit is TWO rules in one coat.
            # 4__Predictor hard-sets UpProbability to exactly 0.30 when a name's feature
            # vector fails the mask (69.1% of panel rows sit at the sentinel). A real
            # probability falling to the sentinel is a drop of at least -0.10, double the
            # -0.05 threshold, so it ALWAYS fires. Measured on the 4-seed books: of 249
            # firings, 117 (47.0%) were sentinel transitions and 132 (53.0%) genuine
            # decay. Shipping the combined rule to the broker would tie live exits to the
            # pipeline's daily mask rate.
            #   full (default) = the measured rule, both halves. Unset is byte-identical.
            #   real = genuine decay only; a transition INTO the sentinel does not count
            #   pin  = data-quality exit only; fire ONLY on entering the sentinel
            #   none = no momentum exit (prob_drop and the age clock still run)
            _mm = 'full'
            _is_pin_txn = False
            try:
                _pv = float(data.UpProbability[-1])
                _is_pin_txn = (abs(float(current_prob) - 0.30) < 1e-9
                               and abs(_pv - 0.30) >= 1e-9)
            except (IndexError, TypeError, ValueError):
                _is_pin_txn = False
            if _mm == 'full':
                _mom_fire = momentum <= -0.05
            elif _mm == 'real':
                _mom_fire = (momentum <= -0.05) and not _is_pin_txn
            elif _mm == 'pin':
                _mom_fire = _is_pin_txn
            elif _mm == 'none':
                _mom_fire = False
            else:
                raise ValueError(f"BT_MOMEXIT_MODE={_mm!r}: expected full|real|pin|none")
            if _mom_fire and not _exempt and not _prod_exits:
                self._momexit_n = getattr(self, '_momexit_n', 0) + 1
                if _is_pin_txn:
                    self._momexit_pin = getattr(self, '_momexit_pin', 0) + 1
                logging.info(f"MOMENTUM SELL {symbol}: 1-day momentum {momentum:.3f} (Current: {current_prob:.3f})")
                return self.exit_position(data)

            if days_held >= self.p.position_timeout:
                # BT_HOLD_EXTEND_UP: grant extra days ONLY while UpProbability is still
                # rising, never past the extended cap. Unset -> unchanged. NOTE this is
                # the FIRST of two timeout checks in this file; the one further down is
                # unreachable because this one returns. Patching that one instead is a
                # silent no-op, which is how the first attempt at this knob failed.
                _ext = 0
                _rising = False
                if _ext > self.p.position_timeout and days_held < _ext:
                    try:
                        _rising = float(data.UpProbability[0]) > float(data.UpProbability[-1])
                    except (IndexError, TypeError, ValueError):
                        _rising = False
                    if _rising:
                        self._holdext_granted = getattr(self, '_holdext_granted', 0) + 1
                        logging.info(f"HOLD EXTENDED {symbol}: day {days_held}, UpProb rising "
                                     f"({float(data.UpProbability[-1]):.3f} -> "
                                     f"{float(data.UpProbability[0]):.3f}), cap {_ext}d")
                if not _rising:
                    logging.info(f"TIMEOUT SELL {symbol}: Held for {days_held} days")
                    return self.exit_position(data)

            # Probability drop check (only after 3+ days). Also production-absent.
            if days_held >= 3 and not _prod_exits:
                recent_probs = [float(data.UpProbability[-i]) for i in range(1, min(11, len(data)))
                               if data.UpProbability[-i] is not None]

                if recent_probs and max(recent_probs) > 0.55 and current_prob < 0.48:
                    logging.info(f"PROB DROP SELL {symbol}: Max recent {max(recent_probs):.3f}, Current {current_prob:.3f}")
                    return self.exit_position(data)

        except (IndexError, AttributeError, TypeError) as e:
            logging.warning(f"No probability data for {symbol} on {current_date}: {e}")

















    def exit_position(self, data):
        """Close position and clean up bracket tracking.

        ORDERING FIX 2026-07-29. This used to submit close() FIRST and cancel the
        bracket legs afterwards, assuming the cancel always takes. It does not.

        backtrader's BackBroker.cancel() is `self.pending.remove(order)` guarded by a
        try/except that returns False on ValueError -- it can only cancel orders that
        have reached broker.pending. An order lands in broker.SUBMITTED first and is
        only promoted to pending by check_submitted() at the top of the NEXT bar's
        broker.next(). The bracket legs are created inside handle_buy_execution, which
        runs from notify_order, i.e. in the same cycle as -- and just before -- next().
        So whenever a model-driven exit fires on the same bar the entry filled, both
        cancel() calls silently no-op, the legs survive, and the book is left with TWO
        live sells against one long. Next bar the stop (or target) fills AND the market
        close fills, and the position goes from +N straight to -N.

        Nothing downstream could recover from that: evaluate_sell_conditions filters on
        size > 0, get_buy_candidates skipped only size > 0, and handle_sell_execution
        only decrements open_positions when the position reaches exactly 0. So each
        over-sell became a permanent naked short holding a permanent slot. Measured on a
        2% universe: 9 stuck shorts by 2025-09-17, open_positions pinned at 10, and not
        one entry for the remaining ten months of the window.

        Fix: cancel first, VERIFY the cancel took, and if it did not, defer the exit one
        bar instead of stacking a second sell on top of live legs. next() retries it.
        """
        symbol = data._name
        position = self.getposition(data)

        if position.size <= 0:
            return

        # A market close does not fill until the next bar, so the position still reads
        # long for the rest of this one and every later exit trigger on this bar would
        # submit ANOTHER full-size sell. Two of them fill together next bar and the
        # position lands at -N. (The deferred-exit retry in next() and
        # evaluate_sell_conditions can both fire on the same bar, which is exactly how
        # this happened.) One working close per position, no more.
        working = self.exit_orders.get(data)
        if working is not None and working.alive():
            return

        bracket = self.bracket_orders.get(data)
        if bracket:
            legs = [bracket.get('stop'), bracket.get('target')]
            for leg in legs:
                if leg is not None and leg.alive():
                    try:
                        self.cancel(leg)
                    except Exception as e:
                        logging.warning(f"Error canceling bracket orders for {symbol}: {e}")

            # cancel() leaves status untouched when it refuses; alive() is the only
            # honest read of whether the leg is really gone.
            if any(leg is not None and leg.alive() for leg in legs):
                logging.info(f"DEFER EXIT {symbol}: bracket legs not cancellable yet "
                             f"(still in broker.submitted); retrying next bar.")
                self.deferred_exits.add(data)
                return

            del self.bracket_orders[data]

        logging.info(f"EXIT POSITION {symbol}: Closing position of size {position.size}")
        self.exit_orders[data] = self.close(data=data)



















            #logging.info(f"Successfully wrote {len(signal_data)} new buy signals and synchronized with live trader")








    def notify_order(self, order):

        if order.status in [order.Completed, order.Partial]:
            self.handle_order_execution(order)
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.handle_order_failure(order)
    

        ##logging of oopen / close close and the comissions/ slippage 



    def handle_order_execution(self, order):

        if order.isbuy():
            self.handle_buy_execution(order)
        elif order.issell():
            self.handle_sell_execution(order)
    
    def handle_buy_execution(self, order):

        data = order.data
        symbol = data._name

        # A buy that this strategy did not originate as an ENTRY -- i.e. the safety-net
        # unwind of an accidental short in next() -- has no bracket intent and no entry
        # date. Recording it as an entry would overwrite the next real position's fill
        # price and hand it a bracket it never asked for.
        if order.ref in self._pyr['orders']:               # PYRAMID add fill
            self._pyr['orders'].pop(order.ref, None); self._pyr['filled'] += 1
            _lg = (self.bracket_orders.get(data) or {}).get('stop')
            if _lg is None or not _lg.alive():            # original stopped out this bar: flatten the residual
                self._pyr['orphan'] += 1; self.exit_position(data)
            return
        if data in self.position_dates:
            self._pyr['fillbar'][(data, self.position_dates.get(data))] = len(data)
        if data not in self.pending_bracket and data not in self.position_dates:
            logging.info(f"UNWIND BUY {symbol}: {order.executed.size} shares, not an entry")
            return

        if self.pending_entry_limits.pop(data, None) is not None:
            self._limit_filled += 1
        _trg = self._limit_trigger.pop(data, None)
        if _trg is not None and BT_LIMIT_ENTRY_K > 0:
            # Everything here is read on the FILL bar, so open/low/high are that
            # session's. data.close[-1] is the signal close the trigger was priced off.
            try:
                _lim, _sig, _dep = _trg
                _op = float(data.open[0]); _lo = float(data.low[0])
                _fx = float(order.executed.price)
                self._limit_log.append(dict(
                    sym=symbol, sig=_sig, trigger=_lim, depth=_dep, openpx=_op,
                    fill=_fx, gap=_op / _sig - 1.0 if _sig else float('nan'),
                    through=bool(_op <= _lim),
                    saved=(_op - _fx) / _op if _op else 0.0,
                    sb_stop=bool(_lo <= _fx * (1.0 - float(getattr(self.p, 'hard_stop_percent',
                                                                  BRACKET.HARD_STOP_PCT)) / 100.0))))
            except Exception as _e:
                logging.warning("LIMIT-ENTRY diag failed for %s: %s", symbol, _e)

        mark_position_as_bought(symbol, order.executed.size)
        logging.info(f"BUY EXECUTED for {symbol}: Price={order.executed.price:.2f}, "
                    f"Size={order.executed.size}, Cost={order.executed.value:.2f}")

        # Correct entry price to actual fill (market orders fill at next-bar open,
        # not the close stored at signal time in execute_buy_with_bracket).
        fill = order.executed.price
        self.entry_prices[data] = fill
        # Survives the sell-side cleanup. If two sell notifications for one position
        # arrive in the same cycle, the first can delete entry_prices before the second
        # is recorded, which would silently drop the leg the same way the old guard did.
        self._entry_px_cache = getattr(self, '_entry_px_cache', {})
        self._buy_fills = getattr(self, '_buy_fills', 0) + 1
        self._buys_per_data = getattr(self, '_buys_per_data', {})
        _k = (data, self.position_dates.get(data))
        self._buys_per_data[_k] = self._buys_per_data.get(_k, 0) + 1
        self._entry_px_cache[data] = fill
        self._entry_dt_cache = getattr(self, '_entry_dt_cache', {})
        self._entry_dt_cache[data] = self.position_dates.get(data, self.datetime.date())
        # Entry leg's ACTUAL commission + fill size, so every sell-leg row can carry a
        # pro-rata share of it (the Commission column is the exit leg only). Survives
        # the sell-side cleanup the same way _entry_px_cache does. Pyramid adds return
        # early above and never reach this line, so a pyramided position carries only
        # the original entry's charge; the report's Tiered recompute is the fallback.
        self._entry_comm_cache = getattr(self, '_entry_comm_cache', {})
        self._entry_comm_cache[data] = (float(order.executed.comm or 0.0),
                                        abs(int(order.executed.size)) or 1)

        # ANCHORING FIX: attach the bracket now that the real fill price is known.
        # This is the whole point of deferring it. Guarded on Completed so a partial
        # fill cannot attach the legs twice.
        intent = self.pending_bracket.pop(data, None)
        if intent is not None and order.status == order.Completed:
            size = abs(int(order.executed.size)) or intent['size']
            stop_level = fill * (1 - intent['stop_pct'] / 100.0)
            target_level = fill * (1 + intent['tp_pct'] / 100.0)
            try:
                stop_order = self.sell(data=data, size=size, exectype=bt.Order.Stop,
                                       price=stop_level)
                # Runner mode: no take-profit leg; the +trig% scale-out in
                # _runner_manage is the profit-taking mechanism (matches the broker).
                if _runner_on():
                    target_order = None
                else:
                    target_order = self.sell(data=data, size=size, exectype=bt.Order.Limit,
                                             price=target_level, oco=stop_order)
                self.stop_levels[data] = stop_level
                self.bracket_orders[data] = {
                    'main': order, 'stop': stop_order, 'target': target_order,
                    'entry_price': fill, 'stop_type': 'hard',
                    'stop_level': stop_level, 'target_price': target_level,
                    'vol': intent.get('vol')}
                logging.info(f"  BRACKET ATTACHED {symbol}: fill=${fill:.2f} -> "
                             f"stop=${stop_level:.2f} (-{intent['stop_pct']}%), "
                             f"target=${target_level:.2f} (+{intent['tp_pct']}%)")
            except Exception as e:
                logging.error(f"  BRACKET ATTACH FAILED for {symbol}: {e}. "
                              f"POSITION IS UNPROTECTED.")
    



    def handle_sell_execution(self, order):
        """Handle the completion of a sell order.

        ACCOUNTING FIX 2026-07-29. The guard used to be `size == 0`, which is only true
        if exactly one sell closed the position. backtrader queues order notifications
        and delivers them at the START of the next cycle, i.e. after the broker has
        already applied EVERY fill of that bar to the position. So when two sells hit
        one long in the same bar, both notifications read the same, already-negative
        size, `== 0` was false for both, and the slot was never given back -- nor was
        the trade ever recorded (which is why Total Trades ran ahead of the rows in
        Data/TradeHistory.parquet).

        position_dates is set at buy submission and deleted in the cleanup below, so it
        marks the entry episode exactly once. Keying off it makes the release idempotent
        and independent of how many sells it took to get flat or past it.
        """
        data = order.data
        symbol = data._name

        # ---- SCALE-OUT LEG ACCOUNTING FIX 2026-08-26 ----------------------------
        # The guard below fires only on the sell that takes the position to zero. A
        # runner scale-out sells (1 - RUNNER_KEEP_FRAC) = 80% first, leaving a positive
        # size, so that leg was NEVER RECORDED. The row written later carried only the
        # runner leg's quantity, so 80% of the gain on every scaled trade vanished from
        # Data/TradeHistory.parquet.
        #
        # This is not a rounding error and it is not neutral. It deletes value ONLY from
        # trades that reached +10%, i.e. exactly the tail carrying 94.8% of profit.
        # Measured on one full-universe seed: sum of recorded PnL -$619.48 against a
        # broker equity path of +$4,636.09, correlation between the two 0.186. Rows with
        # an implied weight under 0.15 of the account (a surviving runner leg against a
        # full position's 0.29) averaged +24.15% where every other row averaged -0.06%.
        #
        # Any A/B scored on this file was biased AGAINST arms that produce more
        # scale-outs. The four scratchpad/armsim/th_s*.parquet seeds inherit it.
        # Fires on EVERY sell leg, and deliberately does NOT read
        # self.getposition(data).size. backtrader queues order notifications and
        # delivers them after the broker has applied every fill of that bar, so when a
        # scale-out and the runner exit land in the same bar BOTH notifications see the
        # already-zero size. Conditioning on remaining size therefore misses exactly the
        # trades that ran furthest. The zero-crossing branch below no longer records;
        # it only releases the slot and cleans up.
        self._sells_seen = getattr(self, '_sells_seen', 0) + 1
        self._sell_status = getattr(self, '_sell_status', {})
        _st = 'Partial' if order.status == order.Partial else 'Completed'
        self._sell_status[_st] = self._sell_status.get(_st, 0) + 1
        _is_final = self.getposition(data).size <= 0
        if not order.executed.size:
            self._sells_zero_size = getattr(self, '_sells_zero_size', 0) + 1
        elif (self.entry_prices.get(data) or getattr(self, '_entry_px_cache', {}).get(data)) is None:
            self._sells_no_entry = getattr(self, '_sells_no_entry', 0) + 1
        if order.executed.size:
            _ep = (self.entry_prices.get(data)
                   or getattr(self, '_entry_px_cache', {}).get(data))
            _ed = (self.position_dates.get(data)
                   or getattr(self, '_entry_dt_cache', {}).get(data))
            if _ep is not None and _ed is not None:
                _q = abs(order.executed.size)
                _xp = order.executed.price
                if not _is_final:
                    self._partial_legs = getattr(self, '_partial_legs', 0) + 1
                self._legs_recorded = getattr(self, '_legs_recorded', 0) + 1
                # Pro-rata share of the entry leg's actual commission: a scale-out
                # and its runner exit split the one buy charge by leg quantity.
                _ec_tot, _ec_sz = getattr(self, '_entry_comm_cache', {}).get(data, (0.0, 0))
                _ecomm = _ec_tot * _q / _ec_sz if _ec_sz else 0.0
                self.trade_recorder.record_trade({
                    'Symbol': symbol,
                    'EntryDate': _ed,
                    'ExitDate': self.datetime.date(),
                    'EntryPrice': _ep,
                    'ExitPrice': _xp,
                    'Quantity': _q,
                    'PnL': (_xp - _ep) * _q,
                    'PnLPct': ((_xp / _ep) - 1) * 100 if _ep else 0.0,
                    'DaysHeld': (self.datetime.date() - _ed).days,
                    'Commission': order.executed.comm,
                    'EntryCommission': _ecomm,
                    'TradeType': 'Long',
                    'ExitReason': ('Scale Out' if not _is_final
                                   else self.determine_exit_reason(data)),
                    'ATR': self.inds[data]['atr'][0],
                    'UpProbability': data.UpProbability[0],
                    'EntryUpProbability': self.entry_up_prob.get(
                        data, float(data.UpProbability[0])),
                    'AccountValue': self.broker.getvalue(),
                })

        if self.getposition(data).size <= 0 and data in self.position_dates:
            self.open_positions = max(0, self.open_positions - 1)

            entry_price = self.entry_prices.get(data)
            exit_price = order.executed.price
            entry_date = self.position_dates.get(data)
            exit_date = self.datetime.date()
            
            if entry_price is not None:
                profit_pct = ((exit_price / entry_price) - 1) * 100
                profit_abs = (exit_price - entry_price) * abs(order.executed.size)
                days_held = (exit_date - entry_date).days if entry_date else 0

                _color = "\033[32m" if profit_pct >= 0 else "\033[31m"
                _sign  = "+" if profit_pct >= 0 else ""

                self.trade_pct_returns.append(profit_pct)
                _total_ret = (self.broker.getvalue() / self.broker.startingcash - 1) * 100
                _n = len(self.trade_pct_returns)
                if _n >= 2:
                    import numpy as _np
                    _arr = _np.array(self.trade_pct_returns)
                    _sharpe = (_arr.mean() / _arr.std(ddof=1)) * (_n ** 0.5) if _arr.std(ddof=1) > 0 else 0.0
                else:
                    _sharpe = 0.0
                _ret_sign = "+" if _total_ret >= 0 else ""
                print(f"{_color}{symbol:<6}  {_sign}{profit_pct:.2f}%\033[0m  |  Total: {_ret_sign}{_total_ret:.2f}%  Sharpe: {_sharpe:.2f}  (n={_n})")
                
                is_win = profit_abs > 0
                is_loss = profit_abs < 0
                
                if is_win:
                    self.winning_trades += 1
                    self.total_win_pnl += profit_abs
                    self.current_win_streak += 1
                    self.current_loss_streak = 0
                    self.recent_outcomes.append(1)
                    if self.current_win_streak > self.longest_win_streak:
                        self.longest_win_streak = self.current_win_streak
                elif is_loss:
                    self.losing_trades += 1
                    self.total_loss_pnl += profit_abs  # This will be negative
                    self.current_loss_streak += 1
                    self.current_win_streak = 0
                    self.recent_outcomes.append(-1)
                    if self.current_loss_streak > self.longest_loss_streak:
                        self.longest_loss_streak = self.current_loss_streak
                else:
                    self.breakeven_trades += 1
                    self.recent_outcomes.append(0)
                
                if len(self.recent_outcomes) > 10:
                    self.recent_outcomes = self.recent_outcomes[-10:]
                
                # ENTRY-STAMPED SIGNAL. If the stamp is missing the row would silently
                # inherit the exit value and look like a clean measurement, so count
                # and log every miss instead (stop() prints the total).
                _entry_up = self.entry_up_prob.get(data)
                if _entry_up is None:
                    self._entry_prob_misses = getattr(self, '_entry_prob_misses', 0) + 1
                    logging.warning(f"ENTRY UPPROB MISSING for {symbol} "
                                    f"(entry {entry_date}); falling back to exit stamp")
                    _entry_up = float(data.UpProbability[0])

                _ec_tot, _ec_sz = getattr(self, '_entry_comm_cache', {}).get(data, (0.0, 0))
                _ecomm = _ec_tot * abs(order.executed.size) / _ec_sz if _ec_sz else 0.0

                # Record in our internal trade history
                trade_data = {
                    'Symbol': symbol,
                    'EntryDate': entry_date,
                    'ExitDate': exit_date,
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'Quantity': abs(order.executed.size),
                    'PnL': profit_abs,
                    'PnLPct': profit_pct,
                    'DaysHeld': days_held,
                    'Commission': order.executed.comm,
                    'EntryCommission': _ecomm,
                    'TradeType': 'Long',
                    'ExitReason': self.determine_exit_reason(data),
                    'ATR': self.inds[data]['atr'][0],
                    # LEGACY, EXIT-STAMPED. Kept byte-for-byte so old reports and any
                    # downstream consumer keep reading the same column.
                    'UpProbability': data.UpProbability[0],
                    # The probability the entry decision was made on. Use THIS for
                    # signal quality; the column above is contaminated by the holding
                    # period. Falls back to the exit value only if the entry stamp is
                    # somehow missing, so the column is never silently null.
                    'EntryUpProbability': _entry_up,
                    'AccountValue': self.broker.getvalue(),
                }
                
                # Add to internal history (report-side; unchanged)
                self.trade_history.append(trade_data)

                # NO LONGER RECORDS HERE. Every sell leg, final one included, is
                # recorded by the size-independent branch at the top of this method.
                # Recording again here would double-count the closing leg.
                
                # COMPLETED-TRADES FIX 2026-07-29: this used to also call
                # Util.add_completed_trade() here, once per sell. That helper does a
                # full read-modify-write of Data/TradeHistory.parquet per trade,
                # which is the SAME path TradeRecorder overwrites in stop(), so
                # none of those per-trade writes ever survived the run. Worse, once
                # the file was corrupt (torn write from a killed run, or a second
                # backtest sharing the path) the read failed on EVERY trade and
                # spammed "Error reading completed trades" for the whole run. The
                # trade is already captured by trade_recorder.record_trade() above;
                # stop() writes the file once at the end of the run.

                is_loss_for_tracking = profit_abs < 0
                update_trade_result(symbol, is_loss_for_tracking, exit_price, exit_date)
                _reason = str(trade_data.get('ExitReason') or '')
                self._last_exit[symbol] = (exit_date, is_loss_for_tracking,
                                           'stop' in _reason.lower(),
                                           float(exit_price) if exit_price else None,
                                           float(_entry_up) if _entry_up is not None else None)
            
            if data in self.entry_prices:
                del self.entry_prices[data]
            if data in self.position_dates:
                del self.position_dates[data]
            self.entry_up_prob.pop(data, None)  # re-entries get a fresh stamp
            if data in self.trailing_stops:
                del self.trailing_stops[data]
            self.exit_orders.pop(data, None)
            self.deferred_exits.discard(data)
            self.runner_done.discard(data)  # re-entries start clean
            if symbol in self.asset_groups:
                del self.asset_groups[symbol]
            
            self.update_group_allocations()
            
            profit_pct = ((exit_price / entry_price) - 1) * 100 if entry_price else 0
            logging.info(f"SELL EXECUTED for {symbol}: Price={exit_price:.2f}, "
                       f"Profit={profit_pct:.2f}%, Value={order.executed.value:.2f}")
        



    def determine_exit_reason(self, data):
        """Determine the reason for exiting a position."""
        current_price = data.close[0]
        entry_price = self.entry_prices.get(data, current_price)
        entry_date = self.position_dates.get(data, self.datetime.date())
        # Hard stop first, then the legacy trailing level. Reading only trailing_stops
        # (never assigned) is what made this whole branch dead.
        trailing_stop = self.stop_levels.get(data) or self.trailing_stops.get(data)

        # Calculate metrics
        days_held = (self.datetime.date() - entry_date).days
        profit_pct = (current_price / entry_price - 1) * 100
        # self.p.take_profit_percent now defaults to the shared bracket (3.5) instead of
        # the stale 20.0, and still honours --take_profit / optimizer overrides.
        take_profit_level = (entry_price * (1 + BRACKET.LEGACY_TP_PCT / 100.0)
                             if BRACKET.LEGACY_BACKTEST
                             else entry_price * (1 + self.p.take_profit_percent / 100.0))

        # Check conditions
        if trailing_stop and current_price <= trailing_stop:
            if profit_pct >= 0:
                return "Trailing Stop (In Profit)"
            else:
                return "Stop Loss"
        elif current_price >= take_profit_level:
            return "Take Profit"
        elif days_held >= self.p.position_timeout:
            return "Max Hold Time"
        elif days_held > 5 and profit_pct < (days_held * self.p.min_daily_return):
            return "Poor Performance"
        else:
            return "Manual Exit"
    


    
    def handle_order_failure(self, order):

        if order in self.order_list:
            self.order_list.remove(order)
            
        reason = "Unknown"
        if order.status == order.Canceled:
            reason = "Canceled"
        elif order.status == order.Margin:
            reason = "Insufficient Margin"
        elif order.status == order.Rejected:
            reason = "Rejected"
        elif order.status == order.Expired:
            reason = "Expired"

        # SLOT LEAK FIX 2026-07-29. execute_buy_with_bracket() increments
        # open_positions at SUBMISSION, but only handle_sell_execution ever decremented
        # it -- and that requires a fill. A buy that dies as Margin/Rejected/Canceled
        # therefore burned a slot for the rest of the run. Margin rejections are routine
        # here: the sizer prices every candidate off the same close[0] and the same
        # pre-trade cash, then the whole batch fills at the next bar's open, so the last
        # name in a batch regularly costs more than the cash left. Give the slot back,
        # and drop the entry bookkeeping that would otherwise be attributed to the next
        # position opened in this ticker.
        if order.ref in self._pyr['orders']:               # PYRAMID add died: no slot to give back
            self._pyr['orders'].pop(order.ref, None); self._pyr['margin' if order.status == order.Margin else 'expired'] += 1
            return
        if order.isbuy():
            data = order.data
            if BT_LIMIT_ENTRY_K > 0 and order.status in (order.Margin, order.Rejected):
                self._limit_rejected += 1
            # FALLBACK: the entry limit expired unfilled. Re-arm at market rather than
            # give the slot back, which is what cost the limit arm its participation.
            # Only on a clean cancel/expiry -- a margin rejection means there was no
            # cash for this name, so re-sending it would just be rejected again.
            _fb = self._limit_fallback_size.pop(data, None)
            if (_fb and order.status in (order.Canceled, order.Expired)
                    and self.getposition(data).size == 0
                    and data in self.pending_bracket):
                _mo = self.buy(data=data, size=_fb, exectype=bt.Order.Market)
                if _mo is not None:
                    self._limit_fellback += 1
                    logging.info("LIMIT->MARKET fallback for %s: %d shares", data._name, _fb)
                    return          # keep the slot and the bracket intent
            if self.getposition(data).size == 0:
                self.open_positions = max(0, self.open_positions - 1)
                self.pending_bracket.pop(data, None)
                self.entry_prices.pop(data, None)
                self.position_dates.pop(data, None)
                logging.info(f"BUY {reason} for {data._name}: slot released "
                             f"(open_positions={self.open_positions})")
    
    def _gate_census(self):
        """Print every gate's fire rate, unconditionally, every run.

        WHY THIS EXISTS. Three of can_buy's financial gates were swept for weeks before
        anyone measured that they fire on 0 to 0.15% of rows, and the 52-week gate ran
        silently dead on ~90% of every backtest for months. Both were twenty minutes of
        counting away from being obvious. A gate that fires on 0.02% of rows is not a
        filter, and there is no way to notice that from a P&L number.

        Counters are incremented in can_buy_v1_shipped via self._g(). Reading this
        before believing any A/B is the cheapest discipline available.
        """
        c = getattr(self, "_gate_hits", None)
        if not c:
            return
        tot = int(c.get("_evaluated", 0)) or 1
        print()
        print("[GATE CENSUS] can_buy evaluations: %d   passed: %d (%.2f%%)"
              % (tot, c.get("PASS", 0), 100.0 * c.get("PASS", 0) / tot))
        rows = sorted(((k, v) for k, v in c.items() if not k.startswith("_") and k != "PASS"),
                      key=lambda kv: -kv[1])
        for k, v in rows:
            flag = "   <-- INERT" if v == 0 else ("   <-- near-inert" if v / tot < 0.001 else "")
            print("  %-28s first-reject %7d  %6.2f%%%s" % (k, v, 100.0 * v / tot, flag))
        _pl, _lr = getattr(self, "_partial_legs", 0), getattr(self, "_legs_recorded", 0)
        print("  [EXIT LEGS] recorded %d sell legs, of which %d were partial "
              "(scale-outs the old zero-crossing guard dropped entirely)" % (_lr, _pl))
        _multi = sum(1 for v in getattr(self, '_buys_per_data', {}).values() if v > 1)
        print("  [LEG AUDIT] sell notifications %d %s | zero-size %d | no-entry-price %d "
              "| legs recorded %d || buy fills %d, positions with MORE THAN ONE buy fill %d"
              % (getattr(self, '_sells_seen', 0), getattr(self, '_sell_status', {}),
                 getattr(self, '_sells_zero_size', 0), getattr(self, '_sells_no_entry', 0),
                 getattr(self, '_legs_recorded', 0), getattr(self, '_buy_fills', 0), _multi))
        self._reconcile_book()

    def _reconcile_book(self):
        """Tie the recorded trade book to the broker's own equity, every run.

        The trade book silently disagreed with the equity path by $5,255 on a $10,000
        account for as long as anyone has been reading it, because the recorder dropped
        every scale-out leg. Nothing in the run said so. This makes the identity

            start + realised PnL - commission + unrealised == broker.getvalue()

        a printed invariant, so the next time a leg goes missing it shows up here
        instead of inside somebody's A/B six weeks later. UNREALISED is legitimate:
        positions open on the last bar are never liquidated, so they carry value with no
        trade row. RESIDUAL is the part that should be zero.
        """
        try:
            rows = getattr(self.trade_recorder, "trades", None)
            if rows is None:
                rows = getattr(self, "trade_history", [])
            realised = sum(float(r.get("PnL", 0) or 0) for r in rows)
            comm = sum(float(r.get("Commission", 0) or 0) for r in rows)
            start = float(self.broker.startingcash)
            value = float(self.broker.getvalue())
            cash = float(self.broker.getcash())
            # cash = start + realised - (cost basis of open positions) - commission,
            # so equity = start + realised - commission + UNREALISED GAIN, where the
            # unrealised gain is market value minus cost basis. Using market value here
            # instead of the gain is wrong by the whole cost basis.
            mkt = cost = 0.0
            nopen = 0
            for _d in self.datas:
                _p = self.getposition(_d)
                if _p and _p.size:
                    nopen += 1
                    try:
                        _px = float(_d.close[0])
                    except (IndexError, TypeError, ValueError):
                        _px = float(_p.price)
                    mkt += _p.size * _px
                    cost += _p.size * float(_p.price)
            unreal = mkt - cost
            resid = value - (start + realised - comm + unreal)
            print("  [BOOK RECONCILIATION] start %.2f + realised %.2f - commission %.2f "
                  "+ unrealised %.2f = %.2f | broker equity %.2f (cash %.2f, "
                  "%d open positions: cost %.2f, market %.2f)"
                  % (start, realised, comm, unreal,
                     start + realised - comm + unreal, value, cash, nopen, cost, mkt))
            print("  [BOOK RECONCILIATION] RESIDUAL %.2f  (%.3f%% of equity)%s"
                  % (resid, 100.0 * resid / max(value, 1e-9),
                     "" if abs(resid) < 1.0 else "   <-- NOT ZERO, the book is losing legs"))
        except Exception as e:
            print("  [BOOK RECONCILIATION] failed: %s" % e)
        _vt = getattr(self, "_voltilt_scored", 0)
        _we, _wsk = getattr(self, "_w52_evaluated", 0), getattr(self, "_w52_skipped", 0)
        if _we or _wsk:
            _t = _we + _wsk
            print("  [W52] evaluated %d of %d (%.1f%%), unavailable %d, blocked %d "
                  "[precomputed %d, buffer-fallback %d, limit %s]"
                  % (_we, _t, 100.0 * _we / max(_t, 1), _wsk,
                     getattr(self, "_w52_blocked", 0), getattr(self, "_w52_precomputed", 0),
                     getattr(self, "_w52_fallback", 0), "1.01"))

    def _g(self, name):
        """Record `name` as the FIRST gate that rejected this evaluation. Always False,
        so call sites read `return self._g('g03_prob_floor')`."""
        c = self._gate_hits
        c[name] = c.get(name, 0) + 1
        return False

    def stop(self):
        self.progress_bar.close()
        self._gate_census()
        _mm_env = None
        if _mm_env:
            _n = getattr(self, '_momexit_n', 0); _p = getattr(self, '_momexit_pin', 0)
            print(f"[BT_MOMEXIT_MODE={_mm_env}] momentum exits fired: {_n} "
                  f"(sentinel transitions among them: {_p})")
            if _n == 0 and _mm_env != 'none':
                print(f"[BT_MOMEXIT_MODE] WARNING: mode={_mm_env} but the rule NEVER "
                      f"fired. Knob disconnected, or no name qualified.")
        if BT_PANIC_GATE:
            print('[PANIC-GATE CENSUS] mode=%s action=%s  gated signal-days=%d  limits withheld=%d  kdown depths=%d'
                  % (BT_PANIC_MODE, BT_PANIC_ACTION, getattr(self, '_panic_days', 0),
                     getattr(self, '_panic_cut', 0), getattr(self, '_panic_kdown', 0)))
        if BT_GATE_PRIORRET_MAX > 0:
            print('[PRIORRET-GATE CENSUS] max=%.2f%%  candidates seen=%d  dropped=%d  deepened=%d (deepen=%s kmult=%.2f)'
                  % (BT_GATE_PRIORRET_MAX, getattr(self, '_priorret_seen', 0), getattr(self, '_priorret_dropped', 0),
                     getattr(self, '_priorret_deep', 0), False, 1.0))
        if BT_GATE_GAPFREQ_N > 0 or BT_GATE_WEAKCLOSE > 0 or BT_GATE_VOLSPIKE_N > 0 :
            print('[HYGIENE CENSUS] gapfreq N=%d pct=%.1f mode=%s weakclose=%.2f volspike N=%d x%.1f atrpct=%.2f  seen=%d  dropped gap=%d weak=%d vol=%d atr=%d  nohist=%d'
                  % (BT_GATE_GAPFREQ_N, BT_GATE_GAP_PCT, 'abs', BT_GATE_WEAKCLOSE, BT_GATE_VOLSPIKE_N, 3.0, 0.0,
                     getattr(self, '_hyg_seen', 0), getattr(self, '_hyg_gap', 0), getattr(self, '_hyg_weak', 0),
                     getattr(self, '_hyg_vol', 0), getattr(self, '_hyg_atr', 0), getattr(self, '_hyg_nohist', 0)))
        if BT_SIZE_DIV > 0:
            print('[SIZE-DIV] per-slot divisor=%.2f (max_positions=%d)' % (BT_SIZE_DIV, int(self.p.max_positions)))
        if BT_PYRAMID_FRAC > 0:
            _pb = self._pyr
            print('[PYRAMID CENSUS] frac=%.2f day=%d reserve_slots=%d gap_max=%.1f  decided=%d guarded=%d attempted=%d filled=%d  '
                  'cash-blocked: sizer=%d margin=%d  limit-expired=%d  orphan(original stopped on the add bar, residual flattened)=%d  '
                  'stop resized for an add=%d  bars where stop size != position size=%d'
                  % (BT_PYRAMID_FRAC, BT_PYRAMID_DAY, BT_PYRAMID_RESERVE_SLOTS, BT_PYRAMID_GAP_MAX, len(_pb['done']), _pb['guard'],
                     _pb['att'], _pb['filled'], _pb['sizer0'], _pb['margin'], _pb['expired'], _pb['orphan'], _pb['resize'], _pb['mismatch']))
        if BT_LIMIT_ENTRY_K > 0:
            _tot = self._limit_sent or 1
            print("[LIMIT-ENTRY CENSUS] K=%.2f  sent=%d  filled=%d (%.1f%%)  expired=%d"
                  " depth_failures=%d"
                  % (BT_LIMIT_ENTRY_K, self._limit_sent, self._limit_filled,
                     100.0 * self._limit_filled / _tot, self._limit_expired,
                     getattr(self, '_limit_depth_fail', 0)))
            print("[LIMIT-ENTRY EXPOSURE] oversub=%.1f  max concurrent held=%d "
                  "(cap %d)  days over cap=%d  margin/rejected=%d"
                  % (BT_LIMIT_OVERSUB, self._limit_maxheld, int(self.p.max_positions),
                     self._limit_overfill_days, self._limit_rejected))
            print("[LIMIT-ENTRY FALLBACK] enabled=%s  converted to market=%d"
                  % (False, self._limit_fellback))
            print("[LIMIT-ENTRY MKT-FILL] enabled=%s  slots market-filled=%d"
                  % (False, self._mkt_filled))
            self._limit_report()

    def _limit_report(self):
        """Diagnostics for the limit-entry arm. Printed once at the end of a run."""
        L = self._limit_log
        if not L:
            print("[LIMIT-ENTRY DIAG] no fills recorded")
            return
        import statistics as _st
        gaps = [r['gap'] for r in L if r['gap'] == r['gap']]
        saved = [r['saved'] for r in L]
        thr = [r for r in L if r['through']]
        sb = [r for r in L if r['sb_stop']]
        print("[LIMIT-ENTRY DIAG] %d fills" % len(L))
        print("   entry discount vs the open : mean %+.3f%%  median %+.3f%%"
              % (100 * _st.fmean(saved), 100 * _st.median(saved)))
        print("   opening gap of filled names: mean %+.2f%%  median %+.2f%%  min %+.2f%%  max %+.2f%%"
              % (100 * _st.fmean(gaps), 100 * _st.median(gaps), 100 * min(gaps), 100 * max(gaps)))
        print("   gapped THROUGH the trigger : %d (%.1f%%)   traded down to it: %d (%.1f%%)"
              % (len(thr), 100.0 * len(thr) / len(L), len(L) - len(thr),
                 100.0 * (len(L) - len(thr)) / len(L)))
        print("   fill bar's own low breached the stop: %d (%.1f%%)  <- stop is not live until the"
              " next bar, so these are modelled optimistically"
              % (len(sb), 100.0 * len(sb) / len(L)))
        edges = [(-9, -.06), (-.06, -.03), (-.03, -.01), (-.01, 0), (0, .01), (.01, 9)]
        names = ['gap < -6%', '-6 to -3%', '-3 to -1%', '-1 to 0%', '0 to +1%', 'gap > +1%']
        print("   fills by opening gap:")
        for (a, b), nm in zip(edges, names):
            k = [r for r in L if a <= r['gap'] < b]
            if not k:
                continue
            print("     %-11s n=%4d (%4.1f%%)  mean discount vs open %+.3f%%"
                  % (nm, len(k), 100.0 * len(k) / len(L), 100 * _st.fmean([r['saved'] for r in k])))
        _tot = (self._eod_fired + self._eod_rearmed + self._eod_blk_target
                + self._eod_blk_nostop + self._eod_blk_notraise + self._eod_blk_cancel)
        _msg = ("[EOD-REPEG CENSUS] opportunities=%d  fired=%d rearmed=%d (%.1f%% active)  "
                "blocked: target=%d nostop=%d notraise=%d cancel=%d"
                % (_tot, self._eod_fired, self._eod_rearmed,
                   ((self._eod_fired + self._eod_rearmed) / _tot * 100) if _tot else 0.0,
                   self._eod_blk_target, self._eod_blk_nostop,
                   self._eod_blk_notraise, self._eod_blk_cancel))
        print(_msg)
        logging.info(_msg)
        _cd = self._eod_cancel_datas
        if _cd:
            _v = sorted(_cd.values(), reverse=True)
            print("[EOD-REPEG] nostop sub-cases: no-bracket=%d flat=%d | cancel stalls over %d symbols, "
                  "max %d repeats, median %d" % (self._eod_nobr, self._eod_nosize, len(_cd), _v[0], _v[len(_v)//2]))

        if self.reentry_cooldown_d > 0:
            print(f"[BT_REENTRY] census: {self._reentry_blocks} (symbol, day) entry "
                  f"evaluations blocked by the {self.reentry_cooldown_d}d/"
                  f"{self.reentry_mode} cooldown")
            logging.info(f"[BT_REENTRY] blocked evaluations: {self._reentry_blocks}")
        _boosts = getattr(self, '_reentry_boosts', 0)
        if _boosts:
            print(f"[BT_REENTRY] boost census: {_boosts} candidate scorings boosted")
            logging.info(f"[BT_REENTRY] boost scorings: {_boosts}")
        _pens = getattr(self, '_reentry_penalties', 0)
        if _pens:
            print(f"[BT_REENTRY] soft-penalty census: {_pens} candidate scorings docked")
            logging.info(f"[BT_REENTRY] soft-penalty scorings: {_pens}")
        _misses = getattr(self, '_entry_prob_misses', 0)
        if _misses:
            print(f"\033[33mWARNING: EntryUpProbability fell back to the exit stamp on "
                  f"{_misses} trade(s). Entry-stamped diagnostics are contaminated by "
                  f"that many rows.\033[0m")
            logging.warning(f"EntryUpProbability fallback count: {_misses}")
        if BT_GATE_MAXRANGE_DROP != 0:
            print("[GATE maxrange] dropped %d candidate-slots at frac %.2f; %d had <10 bars "
                  "of history and were kept" % (getattr(self, '_gate_dropped', 0),
                  BT_GATE_MAXRANGE_DROP, getattr(self, '_gate_nohist', 0)))
        self.trade_recorder.save_trades()
        

##===========================================================[Control]=========================================================##













##===========================================================[Control]=========================================================##













##===========================================================[Control]=========================================================##












##===========================================================[Control]=========================================================##



def save_guaranteed_signals_to_parquet(signals, next_trading_day=None):
    """
    Save signals with market cap and sentiment analysis integrated.
    Generates all columns in one pass while creating the file.
    """
    from finvizfinance.quote import finvizfinance
    # Layout-resilient fundament parser. finvizfinance's own ticker_fundament() is broken
    # against FinViz's current markup (see get_market_cap_data below); this is still
    # imported for ticker_news(), which is unaffected.
    from auxiliary.finviz_compat import (ticker_fundament as finviz_fundament,
                                         market_cap_millions as finviz_cap_millions)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    logger = logging.getLogger(__name__)
    
    if not signals:
        logger.error("CRITICAL: No signals to save! Check your data pipeline.")
        return False
    
    # Load sentiment model once for all signals
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    
    # ── Market cap: FinViz live, PIT panel fallback ──────────────────────────────
    # 2026-07-24: finvizfinance.ticker_fundament() started raising
    # (AttributeError: 'NoneType' has no attribute 'find_all' - their page changed).
    # The old bare `except: return 'Unknown', None` swallowed it, so CapMillions was
    # null for EVERY signal, which silently disabled 7__MacroFilter's micro-cap gate
    # (`if pd.notna(cap) and cap < 952` never fires on a null). That gate is a
    # validated keeper - see analysis_output/MACRO_FILTER_FINDINGS.md and
    # experimental/macrofilter_v2/FINDINGS.md. Now we fall back to the approximate
    # PIT panel instead of emitting a null, and we say so in the log.
    _cap_panel = {}
    try:
        _cp = pd.read_parquet(os.path.join('Data', 'MarketCaps',
                                           'historical_market_caps.parquet'))
        _cp['Date'] = pd.to_datetime(_cp['Date'])
        _last = _cp.sort_values('Date').groupby(_cp['Ticker'].str.upper()).tail(1)
        _cap_panel = {str(r.Ticker).upper(): (float(r.MarketCap) / 1e6, r.Date)
                      for r in _last.itertuples() if np.isfinite(r.MarketCap)}
        _panel_max = max((d for _, d in _cap_panel.values()), default=None)
        logger.info(f"Market-cap PIT panel loaded: {len(_cap_panel):,} tickers, "
                    f"latest {_panel_max.date() if _panel_max is not None else 'n/a'}. "
                    f"(Refresh with: python auxiliary/0__ApproximateMarketCaps.py)")
    except Exception as _e:
        logger.warning(f"Could not load the PIT market-cap panel ({_e}) - micro-cap "
                       f"data will be FinViz-only.")

    _cap_stats = {'finviz': 0, 'panel': 0, 'none': 0, 'finviz_fails': 0,
                  'finviz_nocap': 0, 'fv_industry': 0}
    _FINVIZ_FAIL_LIMIT = 5   # circuit breaker: stop paying network timeouts once it's dead

    def _classify_cap(cap_millions):
        if cap_millions >= 200000:
            return 'Mega'
        elif cap_millions >= 10000:
            return 'Large'
        elif cap_millions >= 2000:
            return 'Mid'
        elif cap_millions >= 300:
            return 'Small'
        elif cap_millions >= 50:
            return 'Micro'
        return 'Nano'

    def get_market_cap_data(ticker):
        """Market cap in $M + bucket + FinViz Sector/Industry.

        Returns (cap_bucket, cap_millions, fv_sector, fv_industry). Cap is ('Unknown',
        None) only when BOTH sources fail - and that is logged per name AND counted in
        the market-cap source census below.

        2026-07-28: the FinViz leg now goes through finviz_compat.ticker_fundament(),
        which parses the CURRENT FinViz markup. The vendored finvizfinance (1.2.0, and
        1.3.0 - byte-identical) looks for `div.quote-links`, which FinViz renamed to
        `a.quote-header_category` links; that is the AttributeError described above. The
        snapshot table holding Market Cap was never actually gone, so this leg works
        again rather than permanently limping on the approximate panel.

        Sector/Industry are pulled from the SAME fetch (they come off the same page, so
        it costs no extra request) and written to the pool. They had silently dropped out
        of the pool entirely when this scrape broke, which blinded 7__MacroFilter's
        Stage-4 concentration cap to its best input - FinViz Industry is what tags
        foreign gold miners that SEC SIC codes miss."""
        fv_sector = fv_industry = ''
        # 1. FinViz live, unless the circuit breaker has tripped for this run
        if _cap_stats['finviz_fails'] < _FINVIZ_FAIL_LIMIT:
            try:
                fundamentals = finviz_fundament(ticker)
                fv_sector = str(fundamentals.get('Sector', '') or '')
                fv_industry = str(fundamentals.get('Industry', '') or '')
                cap_millions = finviz_cap_millions(fundamentals)
                if cap_millions is not None:
                    _cap_stats['finviz'] += 1
                    return (_classify_cap(cap_millions), cap_millions,
                            fv_sector, fv_industry)
                # Reached FinViz fine, but it reports no cap for this name (e.g. a fresh
                # listing). Not a scrape failure - do NOT trip the circuit breaker.
                _cap_stats['finviz_nocap'] += 1
            except Exception as e:
                _cap_stats['finviz_fails'] += 1
                if _cap_stats['finviz_fails'] == 1:
                    logger.warning(f"FinViz fundament scrape failed on {ticker} "
                                   f"({type(e).__name__}: {e}) - falling back to the PIT "
                                   f"market-cap panel. If this is a FinvizLayoutError, "
                                   f"FinViz changed their markup again: fix the selectors "
                                   f"in finviz_compat.py (python auxiliary/finviz_compat.py AAPL).")
                elif _cap_stats['finviz_fails'] == _FINVIZ_FAIL_LIMIT:
                    logger.warning(f"FinViz market cap failed {_FINVIZ_FAIL_LIMIT}x - "
                                   f"circuit breaker tripped, PIT panel only for this run.")

        # 2. PIT panel fallback
        hit = _cap_panel.get(str(ticker).upper())
        if hit is not None:
            cap_millions, asof = hit
            _cap_stats['panel'] += 1
            return _classify_cap(cap_millions), cap_millions, fv_sector, fv_industry

        # 3. Genuinely unknown - downstream gates cannot evaluate this name
        _cap_stats['none'] += 1
        logger.warning(f"[{ticker}] NO market cap from FinViz or the PIT panel - the "
                       f"micro-cap gate cannot evaluate this name.")
        return 'Unknown', None, fv_sector, fv_industry
    
    def get_sentiment_score(ticker):
        """Get sentiment score from news"""
        try:
            stock = finvizfinance(ticker)
            news_df = stock.ticker_news()
            
            if news_df is None or news_df.empty:
                return None
            
            headlines = ' '.join(news_df.head(10)['Title'].tolist())
            inputs = tokenizer(headlines, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = predictions[0].cpu().numpy()
            positive = probs[0]
            negative = probs[1]
            neutral = probs[2]
            
            score = (positive + (neutral * 0.5)) / (positive + negative + neutral)
            return float(score)
        except:
            return None
    
    try:
        if next_trading_day is None:
            current_date = datetime.now().date()
            next_trading_day = get_next_trading_day(current_date)
            logger.info(f"Next trading day: {next_trading_day}")
        
        # Signals written for 9_SuperFastBroker.py. These MUST be the same bracket the
        # broker will place and the same one the backtest just simulated -- before
        # consolidation this block used BT_STOP_PCT (2.0%) and a local 10.0% target,
        # agreeing with neither.
        STOP_LOSS_PERCENT = BRACKET.HARD_STOP_PCT
        TAKE_PROFIT_PERCENT = BRACKET.TAKE_PROFIT_PCT
        
        signal_data = []
        for signal in signals:
            symbol = str(signal['Symbol']).upper()
            up_prob = float(signal['UpProbability']) if signal['UpProbability'] is not None else 0.0
            price = float(signal['Price']) if signal['Price'] is not None else 0.0
            
            atr = signal.get('ATR', price * 0.02)
            if isinstance(atr, str) or atr == 0:
                atr = price * 0.02
            atr = float(atr)
            
            stop_price = price * (1 - STOP_LOSS_PERCENT / 100.0)
            target_price = price * (1 + TAKE_PROFIT_PERCENT / 100.0)
            
            cap_bucket, cap_millions, fv_sector, fv_industry = get_market_cap_data(symbol)
            if fv_industry:
                _cap_stats['fv_industry'] += 1
            sentiment_score = get_sentiment_score(symbol)
            
            signal_record = {
                'Symbol': symbol,
                'Status': 'Pending',
                'TargetDate': pd.Timestamp(next_trading_day),
                
                'CurrentPrice': price,
                'SignalPrice': price,
                'EntryPrice': price,
                'StopPrice': round(stop_price, 4),
                'TargetPrice': round(target_price, 4),
                
                'UpProbability': up_prob,
                'ATR': atr,
                'SignalStrength': up_prob,
                
                'SignalDate': pd.Timestamp(next_trading_day),
                'CreatedDate': pd.Timestamp(datetime.now()),
                'LastUpdated': pd.Timestamp(datetime.now()),
                'LastUpdate': pd.Timestamp(datetime.now()),
                
                'PositionSize': 0,
                'PnL': 0.0,
                'PnLPct': 0.0,
                'ExitPrice': np.nan,
                'EntryDate': pd.NaT,
                'ExitDate': pd.NaT,
                'ExitReason': '',
                'ConsecutiveLosses': 0,
                
                'CapBucket': cap_bucket,
                'CapMillions': cap_millions,
                # 2026-07-28: restored. These two dropped out of the pool when the FinViz
                # fundament scrape broke, which blinded 7__MacroFilter's concentration cap
                # to its BEST input (FinViz Industry tags foreign gold miners that SEC SIC
                # codes miss). industry_group() reads 'FinvizIndustry' first.
                'FinvizSector': fv_sector,
                'FinvizIndustry': fv_industry,
                'Sentiment': sentiment_score
            }
            
            signal_data.append(signal_record)
            
            threshold_used = signal.get('ThresholdValue', signal.get('Threshold', 'Unknown'))
            quality_score = signal.get('Quality', 'Unknown')
            
            try:
                threshold_str = f"{float(threshold_used):.3f}" if threshold_used != 'Unknown' else 'Unknown'
                quality_str = f"{float(quality_score):.1f}" if quality_score != 'Unknown' else 'Unknown'
            except (ValueError, TypeError):
                threshold_str = str(threshold_used)
                quality_str = str(quality_score)
            
            sentiment_str = f"{sentiment_score:.3f}" if sentiment_score is not None else "N/A"
            
            logger.info(f"SIGNAL CREATED: {symbol} | Price: ${price:.2f} | "
                       f"Stop: ${stop_price:.2f} | Target: ${target_price:.2f} | "
                       f"UpProb: {up_prob:.3f} | Cap: {cap_bucket} | Sentiment: {sentiment_str}")

        # ── MARKET-CAP SOURCE CENSUS ────────────────────────────────────────────────
        # A null CapMillions silently disables 7__MacroFilter's micro-cap gate (a validated
        # +11.1pp keeper), and that is exactly how the gate stayed dead for weeks in 2026-07.
        # So the nightly log must state, every run, where every cap came from and how many
        # names have none. Never let this degrade quietly.
        _n_sig = len(signal_data)
        logger.info(f"MARKET-CAP SOURCE CENSUS ({_n_sig} signal(s)): "
                    f"FinViz {_cap_stats['finviz']}, PIT panel {_cap_stats['panel']}, "
                    f"unavailable {_cap_stats['none']} "
                    f"(FinViz scrape failures {_cap_stats['finviz_fails']}, "
                    f"FinViz reachable-but-no-cap {_cap_stats['finviz_nocap']})")
        logger.info(f"  FinViz Industry resolved on {_cap_stats['fv_industry']}/{_n_sig} "
                    f"signal(s) (feeds 7__MacroFilter's concentration cap).")
        if _n_sig and _cap_stats['fv_industry'] == 0:
            logger.warning("  NO signal carries a FinViz Industry - 7__MacroFilter's "
                           "concentration cap loses its best input and falls back to the "
                           "SIC sector map. Run: python auxiliary/finviz_compat.py AAPL")
        if _cap_stats['none']:
            logger.warning(f"{_cap_stats['none']} signal(s) have NO market cap - the "
                           f"downstream micro-cap gate is BLIND on those names. It cannot "
                           f"exclude a micro-cap it cannot measure.")
        if _n_sig and _cap_stats['finviz'] + _cap_stats['panel'] == 0:
            logger.error("MARKET CAP IS UNAVAILABLE FOR THE ENTIRE POOL - 7__MacroFilter's "
                         "micro-cap gate will be COMPLETELY INERT tonight. Fix before "
                         "trading: python auxiliary/finviz_compat.py AAPL  and/or  python "
                         "auxiliary/0__ApproximateMarketCaps.py")
        if _cap_stats['finviz'] == 0 and _cap_stats['finviz_fails']:
            logger.warning("FinViz market cap is DEAD for the whole run (running on the "
                           "approximate PIT panel, which understates caps for names that "
                           "issued shares since the anchor date). Check the parser: "
                           "python auxiliary/finviz_compat.py AAPL ; refresh the panel with: "
                           "python auxiliary/0__ApproximateMarketCaps.py")
        # PROVENANCE / LOOKAHEAD: the FinViz leg is a CURRENT cap, not point-in-time. That
        # is correct here (this function only ever runs while writing the LIVE next-day
        # pool, where "current" == "as of the session"), but any HISTORICAL replay that
        # reuses these CapMillions values is reading today's cap into a past session.
        if _cap_stats['finviz']:
            logger.info(f"  provenance: {_cap_stats['finviz']} cap(s) are FinViz CURRENT "
                        f"(valid for tonight's live pool only - NOT point-in-time; do not "
                        f"reuse this column for historical replay), "
                        f"{_cap_stats['panel']} from the as-of PIT panel.")

        new_signals_df = pd.DataFrame(signal_data)
        
        datetime_columns = ['TargetDate', 'EntryDate', 'ExitDate', 'CreatedDate', 
                           'LastUpdated', 'LastUpdate', 'SignalDate']
        
        for col in datetime_columns:
            if col in new_signals_df.columns:
                new_signals_df[col] = pd.to_datetime(new_signals_df[col])
        
        # SAFETY (2026-08-26): this fork inherited the production pool-export path.
        # On Windows 'Data/0__signals.parquet' and the LIVE 'Data/0__Signals.parquet'
        # are the SAME FILE (os.path.samefile -> True), so any fork run silently
        # overwrote the pool 9_SuperFastBroker.py trades from that morning. Redirect it
        # the same way BT_V2_TRADEHIST redirects the trade book. Default is now the
        # sandbox, NOT production; set BT_V2_SIGNALS explicitly to aim it elsewhere.
        signals_file_path = os.environ.get('BT_V2_SIGNALS',
                                           'Data/_canbuyv2/0__signals_v2.parquet')
        _sig_dir = os.path.dirname(signals_file_path)
        if _sig_dir:
            os.makedirs(_sig_dir, exist_ok=True)

        # Mechanical pre-filter: bake FilterRubric Step-1 (non-web) hard-exclusion
        # audit columns into the dataframe BEFORE the single final write, so the
        # morning analyst sees price/cap/RSI/weekly-vol verdicts without hand-running
        # the rubric. Annotate only - no rows dropped. (see Util.annotate_signals_*)
        if annotate_signals_mechanical_filter is not None:
            try:
                new_signals_df = annotate_signals_mechanical_filter(new_signals_df)
                n_ex = int(new_signals_df["MechExclude"].sum())
                logger.info(f"Mechanical pre-filter: annotated {len(new_signals_df)} signals, "
                            f"{n_ex} flagged MechExclude (rows kept, just flagged)")
                for _, r in new_signals_df[new_signals_df["MechExclude"]].iterrows():
                    logger.info(f"  FLAG {r['Symbol']}: {r['MechReasons']}")
            except Exception as e:
                logger.error(f"Mechanical pre-filter annotation failed (signals still written): {e}")

        new_signals_df.to_parquet(signals_file_path, index=False)

        logger.info(f"SUCCESS: Wrote {len(signal_data)} signals to {signals_file_path}")
        logger.info(f"Signals ready for live trading on {next_trading_day}")
        
        try:
            verification_df = pd.read_parquet(signals_file_path)
            pending_signals = verification_df[verification_df['Status'] == 'Pending']
            logger.info(f"VERIFICATION: File contains {len(pending_signals)} pending signals")
            
            for _, row in pending_signals.iterrows():
                sentiment_disp = f"{row['Sentiment']:.3f}" if pd.notna(row['Sentiment']) else "N/A"
                logger.info(f"  {row['Symbol']}: Cap={row['CapBucket']}, Sentiment={sentiment_disp}, "
                           f"Target=${row['TargetPrice']:.2f}, Stop=${row['StopPrice']:.2f}")
            
        except Exception as e:
            logger.error(f"ERROR: Could not verify saved file: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR saving signals: {e}")
        logger.error(traceback.format_exc())
        return False


















def filter_stocks_by_signal_quality(data_dir, min_variance=0.1, min_up_prob=0.50):
    
    logger = logging.getLogger(__name__)
    filtered_files = []
    all_files = glob.glob(os.path.join(data_dir, '*.parquet'))
    logger.info(f"Found {len(all_files)} stock prediction files")
    
    def meets_criteria(file_path):
        try:
            df = pd.read_parquet(file_path, columns=['up_prob'])
            
            max_up_prob = df['up_prob'].max()
            if max_up_prob < min_up_prob:
                return (False, f"max_up_prob {max_up_prob:.2f} < {min_up_prob:.2f}")
                
            variance = df['up_prob'].var()
            if variance < min_variance:
                return (False, f"variance {variance:.4f} < {min_variance:.4f}")
                
            return (True, f"Passed: max_up_prob={max_up_prob:.2f}, var={variance:.4f}")
        except Exception as e:
            return (False, f"Error: {str(e)}")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(meets_criteria, all_files),
            total=len(all_files),
            desc="Filtering stocks by signal quality"
        ))
    
    filtered_files = []
    rejected_counts = {"max_up_prob": 0, "variance": 0, "error": 0}
    
    for file_path, (meets, reason) in zip(all_files, results):
        file_name = os.path.basename(file_path)
        if meets:
            filtered_files.append(file_path)
            logger.debug(f"Accepted {file_name}: {reason}")
        else:
            if "max_up_prob" in reason:
                rejected_counts["max_up_prob"] += 1
            elif "variance" in reason:
                rejected_counts["variance"] += 1
            else:
                rejected_counts["error"] += 1
            logger.debug(f"Rejected {file_name}: {reason}")
    
    logger.info(f"Filtered to {len(filtered_files)} stocks with quality signals")
    logger.info(f"Rejected: {rejected_counts['max_up_prob']} for low up_prob, " 
                f"{rejected_counts['variance']} for low variance, "
                f"{rejected_counts['error']} due to errors")
    
    return filtered_files




# ------------------------------------------------------------------------------
# Main function and setup routines
# ------------------------------------------------------------------------------




def main():
    """Modified main function that ensures you get signals"""
    logger = get_logger(script_name="5__NightlyBackTester")

    # Mute all console (StreamHandler) output - file handlers keep full detail
    for _handler in logging.root.handlers + logger.handlers:
        if isinstance(_handler, logging.StreamHandler) and not isinstance(_handler, logging.FileHandler):
            _handler.setLevel(logging.WARNING)
    logging.root.setLevel(logging.WARNING)

    start_time = time.time()
    
    try:
        args = arg_parser()                               

        cerebro, data_feeds = setup_backtest_environment(args, logger)
        
        if not data_feeds:
            logger.error("No data feeds available. Exiting.")
            return None
            
        # Run backtest
        strategies = cerebro.run()
        if not strategies:
            logger.error("No strategies were executed.")
            return None
            
        # Process results
        first_strategy = strategies[0]
        results = extract_backtest_results(first_strategy, cerebro, logger)
        
        # Compute execution time
        execution_time = time.time() - start_time
        
        # Display results
        print_detailed_results(results, execution_time)
        
        # Optional plotting
        try_plot_results(cerebro, logger)
        
        # Return summary
        return create_results_summary(results)
    
    except Exception as e:
        logger.error(f"Critical error in backtest: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"\nA critical error occurred: {str(e)}")
        return None






def prepare_data_feed(name_df_tuple):
    """Prepares a single data feed (can be run in parallel)"""
    name, df = name_df_tuple
    data_feed = EnhancedPandasData(dataname=df)
    return name, data_feed



def setup_backtest_environment(args, logger):
    """Set up the backtest environment with WORKING commission model."""
    
    # PERF 2026-07-24: stdstats=False.
    # The default (True) attaches the Broker, Trades and BuySell observers --
    # and BuySell is registered PER DATA, i.e. ~4250 observer objects, each with
    # its own line buffers, next()'d every bar (~2M calls/run). They exist only
    # to draw markers on a plot, and plotting is gated to len(cerebro.datas)<=10
    # so it never fires at full universe. Nothing in this file reads
    # strat.observers / strat.stats -- verified by repo-wide grep.
    # maxcpus only applies to optstrategy() runs, which this file never uses;
    # kept for compatibility with the optimize path.
    cerebro = bt.Cerebro(maxcpus=None, stdstats=False)
    cerebro.broker.set_cash(10000)  # Initial cash

    # FIXED: Use the corrected commission model without emojis
    commission_added = False
    
    try:
        # Try the full IBKR commission model first
        ibkr_commission = IBKRAdaptiveCommission(
            commission_per_share=0.0035,     
            min_per_order=0.35,              
            max_per_order_pct=0.005,         
            exchange_fees=0.0002, leverage=1.0,
        )
        
        cerebro.broker.addcommissioninfo(ibkr_commission)
        cerebro._commission_model = ibkr_commission  # Store for later analysis
        logger.info("SUCCESS: Added IBKR commission model")
        commission_added = True
        
    except Exception as e:
        logger.warning(f"IBKR model failed: {e}")
        
        # Fallback to simple commission
        try:
            simple_commission = SimpleIBKRCommission()
            cerebro.broker.addcommissioninfo(simple_commission)
            cerebro._commission_model = simple_commission
            logger.info("SUCCESS: Added simple commission model")
            commission_added = True
            
        except Exception as e2:
            logger.warning(f"Simple model failed: {e2}")
    
    # Final fallback if both failed
    if not commission_added:
        # Use built-in Backtrader commission
        cerebro.broker.setcommission(commission=0.0035, mult=1.0, margin=None)
        logger.info("FALLBACK: Using built-in commission: $0.0035 per share")

    
    cerebro.broker.set_coo(False)
    cerebro.broker.set_coc(False)

    # OVERSUBSCRIPTION SUPPORT. backtrader's broker cash-checks every order at
    # SUBMISSION (checksubmit defaults True), so resting N limit orders each sized
    # 1/max_positions of equity needs N/max_positions times the account and the surplus
    # is refused before it can rest. Measured at oversub=5: 698 of 1058 orders were
    # refused at submission, not missed on price.
    #
    # With checksubmit off, the orders rest and cash is only checked when one actually
    # executes -- so the book still cannot hold more than it can pay for, but the
    # unfilled orders cost nothing to leave working. That is the intended semantics of
    # "queue 10 and let the 3 that come to me fill".
    if BT_LIMIT_ENTRY_K > 0 and BT_LIMIT_OVERSUB > 1:
        cerebro.broker.set_checksubmit(False)
        logging.info("LIMIT-ENTRY: checksubmit disabled so %.0fx oversubscription can rest",
                     BT_LIMIT_OVERSUB)
    
    # Get data files (read-only override via --data_dir; default = live signals)
    data_dir = getattr(args, 'data_dir', 'Data/RFpredictions')
    file_paths = select_data_files(args, data_dir, logger)
    
    if not file_paths:
        return cerebro, []
    
    # Process data files
    aligned_data = process_data_files(args, file_paths, logger)
    
    if not aligned_data:
        return cerebro, []
    
    # Prepare data feeds in parallel
    logger.info(f"Preparing {len(aligned_data)} data feeds in parallel...")
    start_time = time.time()
    
    # Use process pool for true parallelism (env-capped via BT_INNER_WORKERS when
    # run under the parallel harness, else all cores)
    num_cores = _inner_workers()
    with multiprocessing.Pool(processes=num_cores) as pool:
        prepared_feeds = list(tqdm(
            pool.imap(prepare_data_feed, aligned_data),
            total=len(aligned_data),
            desc="Preparing Data Feeds"
        ))
    
    prep_time = time.time() - start_time
    logger.info(f"Data feed preparation completed in {prep_time:.2f} seconds using {num_cores} cores")
    
    # Add prepared data feeds to cerebro (this part still needs to be sequential)
    add_start_time = time.time()
    for name, data_feed in tqdm(prepared_feeds, desc="Adding Data Feeds to Cerebro"):
        cerebro.adddata(data_feed, name=name)
    
    add_time = time.time() - add_start_time
    logger.info(f"Data feed addition completed in {add_time:.2f} seconds")
    
    # Add analyzers
    add_analyzers(cerebro, logger)
    
    # Add strategy with parameters
    strategy_params = {}
    
    if hasattr(args, 'up_prob') and args.up_prob is not None:
        strategy_params['up_prob_threshold'] = args.up_prob
        strategy_params['up_prob_min_trigger'] = args.up_prob + 0.02

    # Book-size lever: --max_positions was parsed but never reached the strategy.
    # Wire it through so the breadth sweep (top-4 -> top-50) actually takes effect.
    if getattr(args, 'max_positions', None) is not None:
        strategy_params['max_positions'] = args.max_positions
        logger.info(f"Book size overridden: max_positions={args.max_positions}")

    # Strategy-param overrides for the param-robustness / overfitting stress test.
    # Each was parsed but (like max_positions) never reached the strategy. Only the
    # keys the strategy actually consumes are wired; override only when provided.
    for _arg, _pkey in (('position_timeout', 'position_timeout'),
                        ('take_profit',      'take_profit_percent'),
                        ('risk_per_trade',   'risk_per_trade_pct')):
        _v = getattr(args, _arg, None)
        if _v is not None:
            strategy_params[_pkey] = _v
            logger.info(f"Strategy param override: {_pkey}={_v}")

    cerebro.addstrategy(StockSniperStrategy, **strategy_params)
    
    return cerebro, aligned_data





####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####








def select_data_files(args, data_dir, logger):
    """Select data files based on sampling or filtering criteria."""
    
    # First, check if the directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info("Please ensure you have generated prediction data first")
        return []
    
    if args.sample > 0:
        all_files = glob.glob(os.path.join(data_dir, '*.parquet'))
        num_files = len(all_files)
        
        # Check if there are any files
        if num_files == 0:
            logger.error(f"No .parquet files found in directory: {data_dir}")
            logger.info("Available files in directory:")
            try:
                all_files_any = os.listdir(data_dir)
                if all_files_any:
                    for file in all_files_any[:10]:  # Show first 10 files
                        logger.info(f"  - {file}")
                    if len(all_files_any) > 10:
                        logger.info(f"  ... and {len(all_files_any) - 10} more files")
                else:
                    logger.info("  Directory is empty")
            except Exception as e:
                logger.error(f"  Error listing directory contents: {e}")
            
            logger.info("\nPossible solutions:")
            logger.info("1. Run your data generation/prediction script first")
            logger.info("2. Check if prediction files are in a different directory")
            logger.info("3. Verify the data pipeline is working correctly")
            return []
        
        # Calculate number to select with proper bounds checking
        sample_pct = min(100.0, max(0.1, args.sample))  # Clamp between 0.1% and 100%
        num_to_select = max(1, int(round(num_files * sample_pct / 100)))
        
        # Ensure we don't try to sample more files than exist
        num_to_select = min(num_to_select, num_files)

        # Reproducible sampling: when BT_SAMPLE_SEED is set (e.g. by the parallel
        # harness), every job draws the SAME subset, so a book-size sweep compares
        # like-for-like instead of being confounded by different random universes.
        all_files = sorted(all_files)
        _sseed = os.environ.get('BT_SAMPLE_SEED')
        if _sseed is not None:
            try:
                random.seed(int(_sseed))
                logger.info(f"Deterministic sampling seed BT_SAMPLE_SEED={_sseed}")
            except ValueError:
                pass

        file_paths = random.sample(all_files, num_to_select)
        logger.info(f"Selected {len(file_paths)} random files ({sample_pct}% of {num_files})")
        
    else:
        # When sample is 0, use filtering instead
        logger.info("Sample percentage is 0, using quality filtering instead")
        file_paths = filter_stocks_by_signal_quality(
            data_dir, 
            min_variance=args.filter,
            min_up_prob=args.up_prob
        )
        
    if not file_paths:
        logger.error("No stock files found or passed filtering. Exiting.")
        return []
        
    logger.info(f"Processing {len(file_paths)} stock files")
    return file_paths



def process_data_files(args, file_paths, logger):
    """Load and process data files, ensuring proper alignment."""
    last_trading_date = get_last_trading_date()
    logger.info(f"Last trading date: {last_trading_date}")
    
    # Configure alignment parameters (you could add these to args if you want them configurable)
    align_start_date = False  # Set to False to disable alignment
    retention_pct = 90       # Target to keep 95% of stocks
    min_days = 501           # Minimum trading days required
    
    # Load data with alignment
    aligned_data = parallel_load_data(
        file_paths, 
        last_trading_date, 
        align_start_date=align_start_date,
        retention_pct=retention_pct,
        min_days=min_days
    )
    
    if not aligned_data:
        logger.error("No data remains after processing. Exiting.")
        return []
    
    logger.info(f"Final dataset: {len(aligned_data)} stocks with {len(aligned_data[0][1])} trading days")
    return aligned_data




####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####



def add_analyzers(cerebro, logger):
    """Add analyzers to the Cerebro instance."""
    analyzers_to_add = [
        (bt.analyzers.TradeAnalyzer, {"_name": "TradeStats"}),
        (bt.analyzers.DrawDown, {"_name": "DrawDown"}),
        (bt.analyzers.SharpeRatio, {"_name": "SharpeRatio", "riskfreerate": 0.05}),
        (bt.analyzers.SQN, {"_name": "SQN"}),
        (bt.analyzers.Returns, {"_name": "Returns"}),
        (bt.analyzers.VWR, {"_name": "VWR"}),
        (bt.analyzers.TimeReturn, {"_name": "TimeReturn"}),
        (bt.analyzers.PeriodStats, {"_name": "PeriodStats"}),
        (bt.analyzers.Transactions, {"_name": "Transactions"}),
        (bt.analyzers.TradeAnalyzer, {"_name": "TradeAnalyzer"}),
        (bt.analyzers.PositionsValue, {"_name": "PositionsValue"}),
        (bt.analyzers.TimeDrawDown, {"_name": "TimeDrawDown"}),
        (bt.analyzers.PyFolio, {"_name": "PyFolio"})
    ]
    
    for analyzer_class, kwargs in analyzers_to_add:
        try:
            cerebro.addanalyzer(analyzer_class, **kwargs)
            logger.debug(f"Added analyzer: {kwargs.get('_name', 'unnamed')}")
        except Exception as e:
            logger.error(f"Failed to add analyzer {kwargs.get('_name', 'unnamed')}: {str(e)}")


# ------------------------------------------------------------------------------
# Results extraction and processing routines
# ------------------------------------------------------------------------------






def extract_backtest_results(strategy, cerebro, logger):
    """Extract detailed results from the backtest."""
    results = initialize_results_dict(cerebro)
    
    try:
        # Set day count
        try:
            day_count = strategy.day_count
        except AttributeError:
            logger.warning("Could not get day_count from strategy, using 252 as fallback")
            day_count = 252
            
        results['day_count'] = day_count
        
        # Calculate total and annualized returns
        results['total_return'] = (results['final_value'] / results['initial_value'] - 1) * 100
        try:
            results['annualized_return'] = ((results['final_value'] / results['initial_value']) ** (252 / day_count) - 1) * 100
        except Exception as e:
            logger.warning(f"Failed to calculate annualized return: {str(e)}")
        
        # Extract data from analyzers
        analyzer_data = get_analyzer_data(strategy, logger)
        
        # Get strategy-specific data
        if hasattr(strategy, 'monthly_performance'):
            results['monthly_performance'] = strategy.monthly_performance
        
        if hasattr(strategy, 'yearly_performance'):
            results['yearly_performance'] = strategy.yearly_performance

        # Per-trade detail + capital-efficiency inputs. The backtrader analyzer
        # only exposes aggregate dollar P&L, so we pull the strategy's own
        # per-trade records (real per-trade % return) and the daily slot /
        # deployment samples for the strategy-level efficiency metrics.
        if hasattr(strategy, 'trade_history'):
            results['trade_history'] = strategy.trade_history
        # Per-fill limit-entry records (fill vs open) for the fill-quality metrics
        # in the commission section. Already printed raw by _limit_report().
        results['limit_log'] = list(getattr(strategy, '_limit_log', []))
        results['max_positions'] = getattr(strategy.p, 'max_positions', None)
        results['daily_positions'] = list(getattr(strategy, 'daily_positions', []))
        results['daily_deployment'] = list(getattr(strategy, 'daily_deployment', []))
        results['daily_cap_dates'] = list(getattr(strategy, 'daily_cap_dates', []))
        results['next_calls'] = int(getattr(strategy, '_next_calls', 0))

        # Process analyzer data into results
        process_trade_statistics(results, analyzer_data, logger)
        process_drawdown_statistics(results, analyzer_data, logger)
        process_sharpe_ratio(results, analyzer_data, logger)
        process_trade_analyzer_data(results, analyzer_data, logger)
        process_daily_returns_data(results, analyzer_data, logger)
        calculate_risk_of_ruin(results, logger)
        determine_sqn_description(results)
    
        
        return results


    except Exception as e:
        logger.error(f"Error extracting metrics from analyzers: {str(e)}")
    
    return results


def initialize_results_dict(cerebro):
    """Initialize the results dictionary with default values."""
    return {
        # Core metrics
        'initial_value': cerebro.broker.startingcash,
        'final_value': cerebro.broker.getvalue(),
        'total_return': 0,
        'annualized_return': 0,
        'daily_return': 0,
        'sharpe_ratio': 0,
        'sortino_ratio': 0,
        'calmar_ratio': 0,
        'sqn_value': 0,
        'sqn_description': "Unknown",
        'vwr': 0,
        'gain_to_pain_ratio': 0,
        'omega_ratio': 0,
        'information_ratio': 0,
        
        # Risk metrics
        'max_dd': 0,
        'max_dd_duration': 0,
        'avg_dd': 0,
        'avg_dd_duration': 0,
        'ulcer_index': 0,
        'recovery_factor': 0,
        'common_sense_ratio': 0,
        'risk_of_ruin': 1.0,
        'daily_volatility': 0,
        'annualized_volatility': 0,
        'var_95': 0,
        'cvar_95': 0,
        
        # Trade statistics
        'total_closed': 0,
        'won_total': 0,
        'lost_total': 0,
        'won_pnl_total': 0,
        'lost_pnl_total': 0,
        'won_avg': 0,
        'lost_avg': 0,
        'won_max': 0,
        'lost_max': 0,
        'net_total': 0,
        'profit_factor': 0,
        'percent_profitable': 0,
        'risk_reward_ratio': 0,
        'expectancy': 0,
        'kelly_percentage': 0,
        'avg_win_pct': 0,
        'avg_loss_pct': 0,
        'largest_win_pct': 0,
        'largest_loss_pct': 0,
        'avg_profit_per_trade': 0,
        'net_profit_drawdown_ratio': 0,
        
        # Trade management
        'avg_trade_len': 0,
        'longest_trade': 0,
        'shortest_trade': 0,
        'time_in_market_pct': 0,
        'max_consecutive_wins': 0,
        'max_consecutive_losses': 0,
        'current_streak': None,
        'win_loss_count_ratio': 0,
        
        # Advanced metrics
        'positive_days_pct': 0.0,
        'max_pos_streak': 0,
        'max_neg_streak': 0,
        'mfe_avg': 0,
        'mae_avg': 0,
        'mfe_max': 0,
        'mae_max': 0,
        'profit_per_day': 0,
        
        # Strategy specific
        'monthly_performance': {},
        'yearly_performance': {},
        'trade_history': [],
        'max_positions': None,
        'daily_positions': [],
        'daily_deployment': [],
        'daily_cap_dates': [],
        'next_calls': 0,
    }


def get_analyzer_data(strategy, logger):
    """Safely extract data from all analyzers."""
    analyzer_data = {}
    
    # Define the list of analyzers to extract: (key, analyzer_name)
    analyzers = [
        ('trade_stats', 'TradeStats'),
        ('drawdown', 'DrawDown'),
        ('sharpe_ratio', 'SharpeRatio'),
        ('sqn', 'SQN'),
        ('returns', 'Returns'),
        ('vwr', 'VWR'),
        ('time_return', 'TimeReturn'),
        ('period_stats', 'PeriodStats'),
        ('transactions', 'Transactions'),
        ('trade_analyzer', 'TradeAnalyzer'),
        ('positions_value', 'PositionsValue'),
        ('time_drawdown', 'TimeDrawDown')
    ]
    
    # Extract data from each analyzer with error handling
    for key, name in analyzers:
        try:
            analyzer = getattr(strategy.analyzers, name)
            analyzer_data[key] = analyzer.get_analysis()
            
            # Fix for SQN
            if key == 'sqn':
                sqn_value = analyzer_data[key].get('sqn', None)
                if sqn_value is not None:
                    strategy.sqn_value = sqn_value
                    #logger.info(f"SQN value: {sqn_value}")
                else:
                    # Try to calculate SQN manually if analyzer doesn't provide it
                    if 'trade_stats' in analyzer_data and strategy.winning_trades + strategy.losing_trades > 0:
                        trade_results = []
                        won_total = analyzer_data['trade_stats'].get('won', {}).get('total', 0)
                        lost_total = analyzer_data['trade_stats'].get('lost', {}).get('total', 0)
                        won_pnl = analyzer_data['trade_stats'].get('won', {}).get('pnl', {}).get('total', 0)
                        lost_pnl = analyzer_data['trade_stats'].get('lost', {}).get('pnl', {}).get('total', 0)
                        
                        total_trades = won_total + lost_total
                        if total_trades > 0:
                            avg_win = won_pnl / won_total if won_total > 0 else 0
                            avg_loss = lost_pnl / lost_total if lost_total > 0 else 0
                            
                            # Approximate trade results for SQN calculation
                            trade_results = [avg_win] * won_total + [avg_loss] * lost_total
                            
                            if trade_results:
                                mean_r = sum(trade_results) / len(trade_results)
                                std_dev = (sum((r - mean_r) ** 2 for r in trade_results) / len(trade_results)) ** 0.5
                                
                                if std_dev > 0:
                                    strategy.sqn_value = (mean_r / std_dev) * (len(trade_results) ** 0.5)
                                    logger.info(f"Manually calculated SQN value: {strategy.sqn_value}")
            elif key == 'returns':
                strategy.daily_return = analyzer_data[key].get('rtot', 0) / strategy.day_count
            elif key == 'vwr':
                strategy.vwr = analyzer_data[key].get('vwr', 0)
            elif key == 'period_stats':
                strategy.time_in_market_pct = analyzer_data[key].get('inmarket', 0) * 100
                
        except Exception as e:
            logger.warning(f"Failed to get {name} analysis: {str(e)}")
            analyzer_data[key] = {}
    
    return analyzer_data

def process_trade_statistics(results, analyzer_data, logger):
    """Process trade statistics from the analyzer data."""
    try:
        trade_stats = analyzer_data['trade_stats']
        
        results['total_closed'] = trade_stats.get('total', {}).get('closed', 0)
        results['won_total'] = trade_stats.get('won', {}).get('total', 0)
        results['lost_total'] = trade_stats.get('lost', {}).get('total', 0)
        
        results['won_pnl_total'] = trade_stats.get('won', {}).get('pnl', {}).get('total', 0)
        results['lost_pnl_total'] = abs(trade_stats.get('lost', {}).get('pnl', {}).get('total', 0))
        
        results['won_avg'] = trade_stats.get('won', {}).get('pnl', {}).get('average', 0)
        results['lost_avg'] = abs(trade_stats.get('lost', {}).get('pnl', {}).get('average', 0))
        
        results['won_max'] = trade_stats.get('won', {}).get('pnl', {}).get('max', 0)
        results['lost_max'] = abs(trade_stats.get('lost', {}).get('pnl', {}).get('max', 0))
        
        results['net_total'] = trade_stats.get('pnl', {}).get('net', {}).get('total', 0)
        
        # Calculate derived metrics
        if results['lost_pnl_total'] > 0:
            results['profit_factor'] = results['won_pnl_total'] / results['lost_pnl_total']
        else:
            results['profit_factor'] = float('inf')
            
        if results['total_closed'] > 0:
            results['percent_profitable'] = (results['won_total'] / results['total_closed'] * 100)
        
        # Commission metrics from the RECORDED trades.
        # COMMISSION FIX 2026-07-29: the old block guessed the average position as
        # initial_value/50 (~$200 on $10K; the sizer actually caps at 6% = ~$600)
        # and hardcoded $3/trade from the unused FixedCommissionScheme, while the
        # broker actually runs IBKRAdaptiveCommission ($0.0035/share, $0.35 min,
        # i.e. ~$0.35-0.70 per leg), overstating fee drag 4-8x. Every real number
        # is already in trade_history: Commission is the exit leg's actual charge,
        # EntryPrice * Quantity the real notional deployed.
        _th = results.get('trade_history') or []
        _fee_rows = [t for t in _th
                     if (t.get('EntryPrice') or 0) > 0 and (t.get('Quantity') or 0) > 0]
        if _fee_rows:
            _notionals = [t['EntryPrice'] * t['Quantity'] for t in _fee_rows]
            results['avg_trade_price'] = sum(_notionals) / len(_notionals)
            _leg_pcts = [(t.get('Commission') or 0.0) / n * 100
                         for t, n in zip(_fee_rows, _notionals)]
            results['commission_impact_pct'] = sum(_leg_pcts) / len(_leg_pcts)
            # Entry + exit legs (the Commission on a trade row is the exit leg only)
            results['breakeven_threshold_pct'] = results['commission_impact_pct'] * 2
        else:
            results['avg_trade_price'] = 0.0
            results['commission_impact_pct'] = 0.0
            results['breakeven_threshold_pct'] = 0.0

        # BPS BREAKOUT 2026-09-01: rows now carry the entry leg's actual charge
        # (EntryCommission, pro-rata per sell leg), so the round trip is measured
        # instead of estimated as exit x 2. The two legacy keys above keep their old
        # semantics for any downstream consumer; everything new lives in comm_detail
        # and prints in its own dedicated section.
        results['comm_detail'] = compute_commission_detail(
            _fee_rows, results['initial_value'], results.get('day_count') or 252)

        # Gross (before-fee) win rate, measured instead of estimated: trade_history
        # PnL is computed from raw entry/exit prices (no commission in it), so its
        # sign IS the pre-fee outcome. percent_profitable (TradeAnalyzer, pnlcomm)
        # stays the after-fee rate it always was.
        if _th:
            results['gross_win_rate'] = (
                sum(1 for t in _th if (t.get('PnL') or 0) > 0) / len(_th) * 100)
        elif results['total_closed'] > 0:
            # No per-trade history available: report the net rate rather than invent
            results['gross_win_rate'] = results['percent_profitable']
        else:
            results['gross_win_rate'] = 0.0
            
        if results['lost_avg'] > 0:
            results['risk_reward_ratio'] = abs(results['won_avg'] / results['lost_avg'])
        else:
            results['risk_reward_ratio'] = float('inf')
        






        p_win = results['percent_profitable'] / 100
        results['expectancy'] = (p_win * results['won_avg']) + ((1 - p_win) * -results['lost_avg'])
        
        if results['risk_reward_ratio'] > 0:
            results['kelly_percentage'] = ((p_win) - ((1 - p_win) / results['risk_reward_ratio'])) * 100
        


        ## its like EV but kelly assumes that the % profit is unrelated to how often you win 
        ## your biggest winners will be more rare than your average winners 


        if results['total_closed'] > 0:
            results['avg_profit_per_trade'] = results['net_total'] / results['total_closed']
        
        # Percentage metrics
        # ------------------------------------------------------------------
        # Portfolio-relative avg loss (P&L as a fraction of the whole book).
        # Kept ONLY to scale the daily VaR/CVaR thresholds, which are daily
        # portfolio percentages. NOT displayed -- it understates per-trade moves.
        _iv = results['initial_value']
        results['avg_loss_pct_portfolio'] = (results['lost_avg'] / _iv * 100) if _iv else 0

        # REAL per-trade % returns: each trade's price move on the capital
        # actually deployed in THAT trade, independent of book size. A +43% move
        # on a quarter-size position reads +43% here, not ~+10%. This matches the
        # live per-trade log (profit_pct). Win/loss split by the price move sign.
        trade_hist = results.get('trade_history') or []
        win_pcts = [t['PnLPct'] for t in trade_hist if t.get('PnLPct', 0) > 0]
        loss_pcts = [t['PnLPct'] for t in trade_hist if t.get('PnLPct', 0) < 0]
        if trade_hist and (win_pcts or loss_pcts):
            results['avg_win_pct'] = (sum(win_pcts) / len(win_pcts)) if win_pcts else 0
            results['avg_loss_pct'] = abs(sum(loss_pcts) / len(loss_pcts)) if loss_pcts else 0
            results['largest_win_pct'] = max(win_pcts) if win_pcts else 0
            results['largest_loss_pct'] = abs(min(loss_pcts)) if loss_pcts else 0
        else:
            # Legacy fallback (portfolio-relative) when no per-trade history exists
            results['avg_win_pct'] = (results['won_avg'] / _iv * 100) if _iv else 0
            results['avg_loss_pct'] = results['avg_loss_pct_portfolio']
            results['largest_win_pct'] = (results['won_max'] / _iv * 100) if _iv else 0
            results['largest_loss_pct'] = (results['lost_max'] / _iv * 100) if _iv else 0
        
        if results['lost_total'] > 0:
            results['win_loss_count_ratio'] = results['won_total'] / results['lost_total']
        else:
            results['win_loss_count_ratio'] = float('inf')
        
        if results['day_count'] > 0:
            results['profit_per_day'] = results['net_total'] / results['day_count']
            
    except Exception as e:
        logger.warning(f"Error processing trade statistics: {str(e)}")




def process_drawdown_statistics(results, analyzer_data, logger):
    """Process drawdown statistics from the analyzer data."""
    try:
        drawdown = analyzer_data['drawdown']
        
        results['max_dd'] = drawdown.get('max', {}).get('drawdown', 0)
        results['max_dd_duration'] = drawdown.get('max', {}).get('len', 0)
        results['avg_dd'] = drawdown.get('average', {}).get('drawdown', 0)
        results['avg_dd_duration'] = drawdown.get('average', {}).get('len', 0)
        
        # Calculate derived metrics
        if results['max_dd'] > 0:
            results['calmar_ratio'] = results['annualized_return'] / results['max_dd']
        else:
            results['calmar_ratio'] = float('inf')
        
        if results['max_dd'] > 0:
            results['recovery_factor'] = results['total_return'] / results['max_dd']
        else:
            results['recovery_factor'] = float('inf')
        
        if results['total_return'] > 0 and results['max_dd'] > 0:
            results['common_sense_ratio'] = results['total_return'] / results['max_dd_duration']
        
        if results['max_dd'] > 0:
            results['net_profit_drawdown_ratio'] = results['net_total'] / (results['max_dd'] * results['initial_value'] / 100)
        else:
            results['net_profit_drawdown_ratio'] = float('inf')
    except Exception as e:
        logger.warning(f"Error processing drawdown statistics: {str(e)}")


def process_sharpe_ratio(results, analyzer_data, logger):
    """Process Sharpe ratio from the analyzer data."""
    try:
        results['sharpe_ratio'] = analyzer_data['sharpe_ratio'].get('sharperatio', 0)
    except Exception as e:
        logger.warning(f"Error processing Sharpe ratio: {str(e)}")


def process_trade_analyzer_data(results, analyzer_data, logger):
    """Process trade analyzer data for streaks and trade lengths."""
    try:
        trade_analyzer = analyzer_data['trade_analyzer']
        streak_data = trade_analyzer.get('streak', {})
        won_streak = streak_data.get('won', {})
        lost_streak = streak_data.get('lost', {})
        
        results['max_consecutive_wins'] = won_streak.get('longest', 0)
        results['max_consecutive_losses'] = lost_streak.get('longest', 0)
        
        if 'current' in streak_data:
            if streak_data['current'] > 0:
                results['current_streak'] = f"{streak_data['current']} wins"
            elif streak_data['current'] < 0:
                results['current_streak'] = f"{abs(streak_data['current'])} losses"
        
        trade_len = trade_analyzer.get('len', {})
        results['avg_trade_len'] = trade_len.get('average', 0)
        results['longest_trade'] = trade_len.get('max', 0)
        results['shortest_trade'] = trade_len.get('min', 0)
        
        mfe_stats = trade_analyzer.get('mfe', {})
        mae_stats = trade_analyzer.get('mae', {})
        
        results['mfe_avg'] = mfe_stats.get('average', 0)
        results['mfe_max'] = mfe_stats.get('max', 0)
        results['mae_avg'] = mae_stats.get('average', 0)
        results['mae_max'] = mae_stats.get('max', 0)
    except Exception as e:
        logger.warning(f"Error processing trade analyzer data: {str(e)}")



# Omega ratio - CORRECT calculation



def process_daily_returns_data(results, analyzer_data, logger):
    """Process daily returns data for volatility and related metrics."""
    try:
        daily_returns = []
        for date, ret in analyzer_data['time_return'].items():
            if isinstance(ret, (int, float)):
                daily_returns.append(ret)

        if not daily_returns:
            return

        # Store the dated daily-return series so best/worst day-week-month extremes
        # (and how far they deviate from a normal trading period) can be reported later.
        try:
            dated = {pd.to_datetime(d): r
                     for d, r in analyzer_data['time_return'].items()
                     if isinstance(r, (int, float))}
            if dated:
                results['daily_return_series'] = pd.Series(dated).sort_index()
        except Exception:
            pass

        # Sortino ratio: TARGET downside deviation, computed over ALL n days with
        # each day's excess over the target clipped at zero, then annualized and put
        # in % to match annualized_return units.
        # SORTINO FIX 2026-07-29: the old version took np.std() of the negative days
        # only, which (a) divides by the down-day count instead of n, inflating the
        # deviation ~1.6x at a 40% down-day rate, and (b) centers on the losers' own
        # mean, so a uniform -1.2%/day loser scored ~0 downside risk and Sortino
        # blew up toward infinity. Target matches the numerator's 5% annual.
        _daily_target = (1.05 ** (1.0 / 252)) - 1
        _excess_vs_target = np.array(daily_returns) - _daily_target
        downside_deviation = float(
            np.sqrt(np.mean(np.minimum(_excess_vs_target, 0.0) ** 2)) * np.sqrt(252) * 100)
        if downside_deviation > 0:
            results['sortino_ratio'] = (results['annualized_return'] - 5) / downside_deviation
        else:
            results['sortino_ratio'] = float('inf')
        
        # Gain to pain ratio
        sum_of_positive_returns = sum(max(0, r) for r in daily_returns)
        sum_of_negative_returns = abs(sum(min(0, r) for r in daily_returns))
        if sum_of_negative_returns > 0:
            results['gain_to_pain_ratio'] = sum_of_positive_returns / sum_of_negative_returns
        else:
            results['gain_to_pain_ratio'] = float('inf')
        
        # Ulcer index calculation
        # daily_returns are fractional (0.01 = 1%) - use geometric compounding
        equity_curve = list(results['initial_value'] * np.cumprod(1 + np.array(daily_returns)))
        drawdowns = []
        peak = equity_curve[0]
        for value in equity_curve:
            if value > peak:
                peak = value
                drawdowns.append(0)
            else:
                dd_pct = (peak - value) / peak * 100
                drawdowns.append(dd_pct)
        results['ulcer_index'] = np.sqrt(np.mean(np.array(drawdowns) ** 2))

        # Volatility metrics
        results['daily_volatility'] = np.std(daily_returns) * 100
        results['annualized_volatility'] = results['daily_volatility'] * np.sqrt(252)

        # Override backtrader's SharpeRatio (bt.analyzers.SharpeRatio inflates the
        # Sharpe when the strategy spends many days in cash - near-zero portfolio
        # changes make std artificially tiny). Recompute directly from the daily
        # TimeReturn series, which already includes 0-return idle days.
        if np.std(daily_returns, ddof=1) > 0:
            _daily_rf = (1.05 ** (1.0 / 252)) - 1
            _excess   = np.array(daily_returns) - _daily_rf
            results['sharpe_ratio'] = float(
                np.mean(_excess) / np.std(_excess, ddof=1) * np.sqrt(252)
            )

        # VaR and CVaR
        if len(daily_returns) > 5:
            results['var_95'] = np.percentile(daily_returns, 5) * 100
            cvar_values = [r for r in daily_returns if r < results['var_95'] / 100]
            if cvar_values and results['var_95'] < 0:
                results['cvar_95'] = np.mean(cvar_values) * 100

            # Tail Ratio: P95 daily return / |P5 daily return| - > 1.0 means fat right tail
            p95 = np.percentile(daily_returns, 95)
            p5  = np.percentile(daily_returns, 5)
            if p5 < 0:
                results['tail_ratio'] = p95 / abs(p5)


        def calculate_omega_ratio_inline(returns, threshold=0.0):
            """Calculate true Omega ratio inline."""
            if not returns or len(returns) == 0:
                return 0.0

            returns_array = np.array(returns)
            probability = 1.0 / len(returns_array)

            # Probability-weighted gains above threshold
            weighted_gains = np.sum(np.maximum(0, returns_array - threshold) * probability)

            # Probability-weighted losses below threshold  
            weighted_losses = np.sum(np.maximum(0, threshold - returns_array) * probability)

            return weighted_gains / weighted_losses if weighted_losses > 0 else float('inf')


        
        # Calculate Omega ratio with 0% threshold
        results['omega_ratio'] = calculate_omega_ratio_inline(daily_returns, threshold=0.0)
        
        # Optional: Also calculate with risk-free rate threshold
        daily_risk_free = 0.05 / 252  # 5% annual risk-free rate converted to daily
        results['omega_ratio_rf'] = calculate_omega_ratio_inline(daily_returns, threshold=daily_risk_free)



        # Streak and positive days analysis
        if len(daily_returns) > 20:
            results['positive_days_pct'] = sum(1 for r in daily_returns if r > 0) / len(daily_returns) * 100
            
            pos_streak = 0
            max_pos_streak = 0
            neg_streak = 0
            max_neg_streak = 0
            
            for r in daily_returns:
                if r > 0:
                    pos_streak += 1
                    neg_streak = 0
                    max_pos_streak = max(pos_streak, max_pos_streak)
                else:
                    neg_streak += 1
                    pos_streak = 0
                    max_neg_streak = max(neg_streak, max_neg_streak)
                    
            results['max_pos_streak'] = max_pos_streak
            results['max_neg_streak'] = max_neg_streak

            # Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012)
            # P(true annualized SR > 1.0), corrected for skewness and kurtosis
            dr_arr  = np.array(daily_returns)
            sr_hat  = np.mean(dr_arr) / np.std(dr_arr, ddof=1)   # daily SR
            sr_star = 1.0 / np.sqrt(252)                          # daily equiv of annual 1.0
            gamma3  = stats.skew(dr_arr)
            gamma4  = stats.kurtosis(dr_arr, fisher=False)        # raw kurtosis (normal = 3)
            variance_sr = 1 - gamma3 * sr_hat + (gamma4 - 1) / 4 * sr_hat ** 2
            if variance_sr > 0:
                psr_stat = (sr_hat - sr_star) * np.sqrt(len(daily_returns) - 1) / np.sqrt(variance_sr)
                results['psr'] = float(stats.norm.cdf(psr_stat))

        # Serenity Ratio = (Annualized Excess Return) / Ulcer Index
        # Rewards fast recovery; penalises lingering drawdowns unlike Calmar's single-event max-DD
        if results.get('ulcer_index', 0) > 0:
            results['serenity_ratio'] = (results.get('annualized_return', 0) - 5.0) / results['ulcer_index']

    except Exception as e:
        logger.warning(f"Error calculating advanced metrics from daily returns: {str(e)}")






























def calculate_risk_of_ruin(results, logger):
    """Calculate risk of ruin based on win rate and risk/reward ratio."""
    try:
        if results['percent_profitable'] > 0 and results['risk_reward_ratio'] > 0:
            win_rate_decimal = results['percent_profitable'] / 100
            edge = win_rate_decimal - (1 - win_rate_decimal) / results['risk_reward_ratio']
            if edge > 0:
                results['risk_of_ruin'] = ((1 - edge) / (1 + edge)) ** 20
            else:
                results['risk_of_ruin'] = 1.0
    except Exception as e:
        logger.warning(f"Error calculating risk of ruin: {str(e)}")


def determine_sqn_description(results):
    """Determine the SQN description based on the SQN value."""
    sqn_descriptions = {
        (float('-inf'), 0): "Negative",
        (0, 1.6): "Poor",
        (1.6, 2.0): "Below Average",
        (2.0, 2.5): "Average",
        (2.5, 3.0): "Good",
        (3.0, 5.0): "Excellent",
        (5.0, 7.0): "Superb",
        (7.0, float('inf')): "Holy Grail Potential"
    }
    
    for (low, high), desc in sqn_descriptions.items():
        if low <= results['sqn_value'] < high:
            results['sqn_description'] = desc
            break





# ------------------------------------------------------------------------------
# Printing routines for results
# ------------------------------------------------------------------------------

def print_detailed_results(results, execution_time):
    """Print detailed results to the console with colorized output."""
    print("\n" + "=" * 80)
    print(" Stock Sniper Strategy Backtest Results ".center(80))
    print("=" * 80)

    # Core Performance Metrics
    print("\nCore Performance Metrics:")
    print(colorize_output(results['total_return'], "Total Return %:", 50, 10))
    print(colorize_output(results['annualized_return'], "Annualized Return %:", 25, 10))
    print(colorize_output(results['final_value'], "Final Portfolio Value:", results['initial_value'] * 1.5, results['initial_value'] * 1.1))
    print(colorize_output(results['initial_value'], "Initial Portfolio Value:", results['initial_value'], results['initial_value']))
    print(colorize_output(results['sharpe_ratio'], "Sharpe Ratio:", 1.5, 0.75))
    print(colorize_output(results['sortino_ratio'], "Sortino Ratio:", 2.0, 1.0))
    print(colorize_output(results['calmar_ratio'], "Calmar Ratio:", 2.0, 0.5))
    print(colorize_output(results['gain_to_pain_ratio'], "Gain to Pain Ratio:", 1.5, 1.0))
    print(colorize_output(results['omega_ratio'], "Omega Ratio:", 1.5, 1.0))
    print(colorize_output(results['omega_ratio_rf'], "Omega Ratio (Risk-Free):", 1.5, 1.0))
    
    # SQN Metrics (Original and Enhanced)
    #print(colorize_output(results['sqn_value'], "SQN:", 3.0, 1.6))

    ##SQN CURRENTLY BROKERN - FIX LATER also the std on the postive returns is making this low when it should be high


    if not results['vwr'] == 0.0 or results['vwr'] == None:
        print(colorize_output(results['vwr'], "Variability-Weighted Return:", 5, 0.5))
    
    # Add enhanced SQN if available
    if 'enhanced_modified_sqn' in results:
        print(colorize_output(results['enhanced_modified_sqn'], "Modified SQN (% normalized):", 3.0, 1.6))
        

    # Compute capture ratios before printing so they appear in Risk Metrics
    _compute_capture_ratios(results)

    # Risk metrics (includes capture ratios)
    print_risk_metrics(results)

    # Trade statistics
    print_trade_statistics(results)

    # Commission drag + entry fill quality, in bps (dedicated section 2026-09-01)
    print_commission_and_fill_costs(results)

    # Capital efficiency / book utilization (strategy-level)
    print_capital_efficiency(results)

    # Statistical quality metrics (PSR, Serenity, Tail Ratio)
    print_signal_quality_metrics(results)

    # Trade management metrics
    print_trade_management_metrics(results)

    # Advanced trade quality metrics
    print_advanced_trade_metrics(results)

    # NET-NEW edge diagnostics: exit-path attribution, signal-conditional edge,
    # distribution/tail (Cornish-Fisher VaR/ES, CDaR), concentration, deflated Sharpe.
    print_edge_diagnostics(results)

    # Best/worst day-week-month + how far those extremes deviate from normal
    print_trade_history_extremes(results)

    # Trade-book autopsy: exit-reason / holding-period / signal-health / pond /
    # concentration / path (MAE-MFE, overnight split) / regime diagnostics.
    # Lives in backtest_diagnostics.py so it can also run standalone on any
    # saved trade parquet. Guarded: a failure there never kills this report.
    try:
        from backtest_diagnostics import run_diagnostics
        run_diagnostics(trade_file=COMPLETED_TRADES_FILE,
                        daily_return_series=results.get('daily_return_series'),
                        initial_value=results.get('initial_value', 10000.0))
    except Exception as _diag_e:
        print(f"[trade autopsy skipped: {_diag_e}]")

    # Enhanced strategy consistency metrics (if available)
    #print_enhanced_consistency_metrics(results)
    
    # Position sizing recommendations (if available)
    #print_position_sizing_recommendations(results)
    
    # Monthly and yearly performance
    print_period_performance(results)
    
    # System interpretation (if enhanced metrics available)
    
    # Execution time and trade data notice
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print(f"Trade data saved to {V2_TRADE_HISTORY} for further analysis")





def _compute_capture_ratios(results):
    """Download S&P 500 monthly data and store up/down capture ratios in results."""
    strat_monthly = results.get('monthly_performance', {})
    if not strat_monthly:
        return
    try:
        months   = sorted(strat_monthly.keys())
        sy, sm   = map(int, months[0].split('-'))
        ey, em   = map(int, months[-1].split('-'))
        start_dt = datetime(sy, sm, 1) - timedelta(days=5)
        end_dt   = datetime(ey, em, 28) + timedelta(days=10)

        sp500 = yf.download("^GSPC",
                            start=start_dt.strftime('%Y-%m-%d'),
                            end=end_dt.strftime('%Y-%m-%d'),
                            interval="1d", auto_adjust=True, progress=False)
        if len(sp500) == 0:
            return

        daily_ret = sp500['Close'].pct_change().dropna()
        market_monthly = {}
        for (yr, mo), grp in daily_ret.groupby([daily_ret.index.year, daily_ret.index.month]):
            market_monthly[f"{yr}-{mo:02d}"] = ((1 + grp).cumprod().iloc[-1] - 1) * 100

        common = set(strat_monthly.keys()) & set(market_monthly.keys())
        if len(common) < 3:
            return

        up_s, up_m, dn_s, dn_m = [], [], [], []
        for month in common:
            sp = market_monthly[month]
            st = strat_monthly[month]
            if sp > 0:
                up_s.append(st); up_m.append(sp)
            elif sp < 0:
                dn_s.append(st); dn_m.append(sp)

        def _gm(vals):
            return (np.prod([1 + v / 100 for v in vals]) ** (1 / len(vals)) - 1) * 100

        if len(up_m) >= 2:
            gm_u = _gm(up_m)
            if gm_u != 0:
                results['up_capture'] = (_gm(up_s) / gm_u) * 100
        if len(dn_m) >= 2:
            gm_d = _gm(dn_m)
            if gm_d != 0:
                results['down_capture'] = (_gm(dn_s) / gm_d) * 100
    except Exception:
        pass  # silently skip on network failure


def print_risk_metrics(results):
    """Print risk metrics with dynamic, context-aware thresholds."""
    print("\nRisk Metrics:")
    
    # Get key metrics for dynamic calculations
    sortino = results.get('sortino_ratio', 0)
    calmar = results.get('calmar_ratio', 0)
    annual_return = results.get('annualized_return', 0)
    # Use the PORTFOLIO-relative avg loss here (VaR/CVaR are daily portfolio %).
    # The displayed avg_loss_pct is now a real per-trade move and is a different
    # (larger) scale, so it must not drive these thresholds.
    avg_loss_pct = results.get('avg_loss_pct_portfolio', results.get('avg_loss_pct', 0))

    # ===== Dynamic Threshold Logic =====
    # Annualized Volatility Thresholds
    if sortino > 2 and calmar > 5:  # Exceptional risk-adjusted returns
        vol_good = 30.0  # Green if <30%
        vol_bad = 50.0   # Red if >50%
    elif annual_return > 150:  # Ultra-high return strategy
        vol_good = 40.0
        vol_bad = 60.0
    else:  # Standard thresholds
        vol_good = 20.0
        vol_bad = 40.0

    # Daily Volatility (derived from annualized thresholds)
    daily_vol_multiplier = 1/15.8  # ≈ sqrt(252 trading days)
    daily_good = vol_good * daily_vol_multiplier
    daily_bad = vol_bad * daily_vol_multiplier

    # ===== Updated Print Statements =====
    print(colorize_output(results['max_dd'], "Max Drawdown %:", 10, 25, lower_is_better=True))
    print(colorize_output(results['max_dd_duration'], "Max Drawdown Duration (days):", 
                         results['max_consecutive_wins'] * 10, 
                         results['max_consecutive_wins'] * 15, 
                         lower_is_better=True))
    print(colorize_output(results['ulcer_index'], "Ulcer Index:", 1, 3, 
                         lower_is_better=True, unicorn_multiplier=10000.0))
    print(colorize_output(results['recovery_factor'], "Recovery Factor:", 3.0, 1.0))
    print(colorize_output(results['common_sense_ratio'], "Common Sense Ratio:", 0.5, 0.2))
    print(colorize_output(results['risk_of_ruin'], "Risk of Ruin:", 0.001, 0.05, 
                         lower_is_better=True, unicorn_multiplier=10000.0))
    
    # Updated Volatility Lines with Dynamic Thresholds
    print(colorize_output(results['daily_volatility'], "Daily Volatility %:", 
                         daily_good, daily_bad, lower_is_better=True))
    
    print(colorize_output(results['annualized_volatility'], "Annualized Volatility %:", 
                         vol_good, vol_bad, lower_is_better=True))
    
    # VaR/CVaR thresholds scaled to strategy performance
    var_cvar_multiplier = 2 if annual_return > 100 else 1  # Aggressive vs conservative
    print(colorize_output(results['var_95'], "Daily VaR (95%):", 
                         avg_loss_pct * 1.2 * var_cvar_multiplier, 
                         avg_loss_pct * 2.5 * var_cvar_multiplier, 
                         lower_is_better=True))
    
    print(colorize_output(results['cvar_95'], "Daily CVaR (95%):",
                         avg_loss_pct * 1.5 * var_cvar_multiplier,
                         avg_loss_pct * 3.0 * var_cvar_multiplier,
                         lower_is_better=True))

    # Market Capture Ratios (computed by _compute_capture_ratios before printing)
    if results.get('up_capture') is not None:
        # > 120% outpaces S&P in rallies; > 240% is unicorn
        print(colorize_output(results['up_capture'], "Up-Capture Ratio (%):",
                              good_threshold=120.0, bad_threshold=80.0,
                              unicorn_multiplier=2.0))
    if results.get('down_capture') is not None:
        # Negative = gains when market falls; unicorn triggers at value <= 50/10000 = 0.005
        print(colorize_output(results['down_capture'], "Down-Capture Ratio (%):",
                              good_threshold=50.0, bad_threshold=100.0,
                              lower_is_better=True, unicorn_multiplier=10000.0))

def _cornish_fisher_tail(returns, alpha=0.05):
    """Modified VaR + modified ES via the Cornish-Fisher (Zangari 1996 / Boudt et al.)
    quantile expansion - corrects the Gaussian tail for skewness and excess kurtosis.

    Returns (mVaR, mES, gaussian_VaR, in_valid_domain) as FRACTIONAL daily returns
    (negative = loss). mES is the tail-average of the modified quantile, integrated
    numerically over (0, alpha] to avoid a fragile closed form. Returns Nones on too
    little data. `in_valid_domain` flags whether |skew| is inside the range where the
    CF quantile stays monotonic (|S| <~ 0.83); outside it the estimate is unreliable.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 20:
        return None, None, None, True
    mu = float(r.mean()); sigma = float(r.std(ddof=1))
    if sigma == 0:
        return None, None, None, True
    S = float(stats.skew(r)); K = float(stats.kurtosis(r))  # K = EXCESS kurtosis

    def _cf_q(a):
        z = stats.norm.ppf(a)
        zcf = (z + (z**2 - 1) * S / 6.0
                 + (z**3 - 3*z) * K / 24.0
                 - (2*z**3 - 5*z) * S**2 / 36.0)
        return mu + sigma * zcf

    mvar = _cf_q(alpha)
    grid = np.linspace(max(alpha / 200.0, 1e-4), alpha, 64)
    mes = float(np.mean([_cf_q(a) for a in grid]))
    gaussian_var = mu + sigma * stats.norm.ppf(alpha)
    in_domain = abs(S) <= 6 * (np.sqrt(2) - 1)  # ~0.828
    return mvar, mes, gaussian_var, in_domain


def _conditional_drawdown_at_risk(daily_returns, initial_value=10000.0, alpha=0.95):
    """CDaR(alpha): the CVaR of the drawdown distribution - average of the worst
    (1-alpha) fraction of drawdowns along the equity path. A tail-of-drawdowns
    measure that is far more robust than the single-event Max Drawdown.
    Returns a positive percentage (drawdown depth)."""
    r = np.asarray(daily_returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 10:
        return None
    eq = initial_value * np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak  # fractional drawdown at each point, >= 0
    thresh = np.percentile(dd, alpha * 100)
    worst = dd[dd >= thresh]
    if worst.size == 0:
        return float(dd.max() * 100)
    return float(worst.mean() * 100)


def _load_trade_table():
    """Read the just-saved trade table for trade-level diagnostics. Decoupled and
    fully guarded so a missing/locked file never breaks the report."""
    try:
        df = pd.read_parquet(V2_TRADE_HISTORY)
        return df if len(df) else None
    except Exception:
        return None


def _print_signal_ic_block(trades, col, label, n_buckets=10):
    """Rank-IC plus an avg-return bucket table for one probability column.

    INSTRUMENT FIX 2026-07-30. The old block did this once, on 'UpProbability', which
    the strategy stamps at EXIT. Losers are held while the model's probability decays,
    so a chunk of that correlation is manufactured by the holding period rather than by
    the signal. The report is now printed twice, exit-stamped (legacy 'UpProbability')
    and entry-stamped, and the gap between the two ICs is the size of the circularity.

    Buckets are cut on the RANK of the column (ties broken by row order), so the split
    is always an even n_buckets-way split even when the raw values are tie-heavy (the
    exit-stamped column has a mass at 0.30 that used to collapse qcut to 7 uneven
    buckets). Tie mass is still visible: consecutive buckets print the same range.

    Returns the Spearman IC, or None when the column is unusable.
    """
    if col not in trades.columns:
        print(f"[{label} IC: column '{col}' absent from the trade table]")
        return None
    sub = trades[[col, 'PnLPct']].apply(pd.to_numeric, errors='coerce').dropna()
    dropped = len(trades) - len(sub)
    if dropped:
        print(f"WARNING: {label} IC dropped {dropped} of {len(trades)} rows "
              f"(null or non-numeric).")
    if len(sub) < 10:
        print(f"[{label} IC: only {len(sub)} usable rows, skipped]")
        return None
    up, pl = sub[col], sub['PnLPct']
    ic, pval = spearmanr(up, pl)
    print(colorize_output(float(ic), f"{label} IC:", 0.05, 0.0))
    try:
        g = sub.assign(_b=pd.qcut(up.rank(method='first'), n_buckets,
                                  labels=False)).groupby('_b', observed=True)
        agg = g['PnLPct'].agg(['mean', 'count'])
        lo, hi = g[col].min(), g[col].max()
        for b, row in agg.iterrows():
            print(colorize_output(
                float(row['mean']),
                f"{label} {lo[b]:.3f}-{hi[b]:.3f} n={int(row['count'])}:",
                1.0, -0.5))
    except Exception as e:
        print(f"[{label} bucket table skipped: {e}]")
    print(f"{label + ' IC stats:':<30}n={len(sub)}  p={pval:.4f}  "
          f"distinct={up.nunique()}  range=[{up.min():.4f}, {up.max():.4f}]")
    return float(ic)


def print_edge_diagnostics(results):
    """Edge diagnostics: signal quality, exit-path P&L, tail/distribution, robustness.
    Flat colorized metric lines matching the rest of the report. Each block guarded so
    a failure cannot abort the report. See
    analysis_output/BACKTEST_METRICS_RESEARCH_2026_06_30.md for definitions."""
    trades = _load_trade_table()

    # ---------- Signal & exit path ----------
    try:
        if trades is not None:
            print("\nEdge Diagnostics - Signal & Exit:")
            ic_exit = _print_signal_ic_block(
                trades, 'UpProbability', 'Exit')
            ic_entry = _print_signal_ic_block(
                trades, 'EntryUpProbability', 'Entry')
            if ic_exit is not None and ic_entry is not None:
                # Positive delta = the legacy number was flattered by holding losers
                # while their probability decayed. This is the circularity, in IC units.
                print(colorize_output(float(ic_exit - ic_entry),
                                      "Circularity (exit-entry IC):", 0.0, 0.02,
                                      lower_is_better=True))
            if 'ExitReason' in trades.columns:
                ex = trades.groupby('ExitReason')['PnL'].sum().sort_values(ascending=False)
                # Short display names so every label fits the standard 30-char column.
                short = {'Trailing Stop (In Profit)': 'Trail Stop',
                         'Poor Performance': 'Poor Perf'}
                for reason, net in ex.items():
                    name = short.get(str(reason), str(reason))[:15]
                    print(colorize_output(float(net), f"Exit {name} P&L ($):",
                                          1000.0, -500.0))
    except Exception as e:
        print(f"[edge signal/exit skipped: {e}]")

    # ---------- Tail & risk ----------
    try:
        print("\nEdge Diagnostics - Tail & Risk:")
        dser = results.get('daily_return_series')
        daily = np.asarray(dser.values, dtype=float) if dser is not None else None
        if daily is not None and daily.size > 20:
            print(colorize_output(float(stats.skew(daily)), "Daily Return Skew:", 0.3, -0.3))
            print(colorize_output(float(stats.kurtosis(daily)), "Daily Excess Kurtosis:", 1.0, 5.0,
                                  lower_is_better=True))
            mvar, mes, gvar, ok = _cornish_fisher_tail(daily, 0.05)
            if mvar is not None:
                print(colorize_output(abs(mvar * 100), "Modified VaR 95% loss %:", 2.0, 4.0,
                                      lower_is_better=True))
                print(colorize_output(abs(mes * 100), "Modified ES 95% loss %:", 3.0, 5.0,
                                      lower_is_better=True))
            cdar = _conditional_drawdown_at_risk(daily, results.get('initial_value', 10000.0), 0.95)
            if cdar is not None:
                print(colorize_output(cdar, "CDaR 95% loss %:", results.get('max_dd', 14) * 0.6,
                                      results.get('max_dd', 14), lower_is_better=True))
        if trades is not None:
            r = trades['PnLPct'].astype(float).values
            print(colorize_output(float(stats.skew(r)), "Per-Trade Return Skew:", 0.3, -0.3))
            aw = r[r > 0].mean() if (r > 0).any() else 0.0
            al = r[r < 0].mean() if (r < 0).any() else 0.0
            wr = (r > 0).mean() * 100
            if (aw - al) != 0:
                be = -al / (aw - al) * 100
                print(colorize_output(be, "Break-Even Win Rate %:", 45.0, 55.0, lower_is_better=True))
                print(colorize_output(wr - be, "Win Rate Margin (pp):", 5.0, 0.0))
            tot = trades['PnL'].sum()
            ps = np.sort(trades['PnL'].values)[::-1]
            k = max(1, int(0.1 * len(trades)))
            share = ps[:k].sum() / tot * 100 if tot != 0 else float('nan')
            print(colorize_output(share, "Top 10% Trade P&L Share %:", 90.0, 150.0, lower_is_better=True))
    except Exception as e:
        print(f"[edge tail/risk skipped: {e}]")

    # ---------- Robustness (opt-in: needs BT_DSR_TRIALS = number of configs searched) ----------
    try:
        n_trials = None
        dser = results.get('daily_return_series')
        if n_trials and dser is not None:
            N = float(n_trials)
            daily = np.asarray(dser.values, dtype=float)
            daily = daily[~np.isnan(daily)]
            n = daily.size
            sr = (results.get('sharpe_ratio', 0) or 0) / np.sqrt(252)
            g3 = float(stats.skew(daily)); g4 = float(stats.kurtosis(daily, fisher=False))
            denom = np.sqrt(max(1e-9, 1 - g3 * sr + ((g4 - 1) / 4.0) * sr**2))
            sr_std_env = None
            v = (float(sr_std_env)) ** 2 if sr_std_env else (denom / np.sqrt(n - 1)) ** 2
            euler = 0.5772156649
            sr0 = np.sqrt(v) * ((1 - euler) * stats.norm.ppf(1 - 1.0 / N)
                                + euler * stats.norm.ppf(1 - 1.0 / (N * np.e)))
            dsr = float(stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom)) * 100
            print("\nEdge Diagnostics - Robustness:")
            print(colorize_output(dsr, f"Deflated Sharpe (N={int(N)}) %:", 95.0, 90.0))
    except Exception as e:
        print(f"[edge robustness skipped: {e}]")


def _tiered_leg_comm(q, px):
    """One leg on IBKR Tiered: $0.0035/sh, $0.35 min, 1% cap, + $0.0002/sh fees."""
    base = max(q * _COMM_PER_SHARE, _COMM_MIN_ORDER)
    return min(base, q * px * 0.01) + q * 0.0002


def _fixed_leg_comm(q, px):
    """One leg on IBKR Fixed: $0.005/sh, $1.00 min, 1% cap, exchange fees bundled."""
    return min(max(q * 0.005, 1.00), q * px * 0.01)


def compute_commission_detail(rows, initial_value, day_count):
    """Measure commission drag in bps of entry notional, leg pair by leg pair.

    Uses the recorded charges: Commission is the exit leg's actual charge and
    EntryCommission the leg's pro-rata share of the buy. Rows from parquets
    written before 2026-09-01 have no EntryCommission, so the entry order is
    regrouped (Symbol, EntryDate) and repriced on the Tiered schedule, which
    keeps the $0.35 minimum from being applied once per scale-out leg. The
    Fixed what-if prices the same fills on the old plan to show what the
    2026-08-26 plan switch is worth on this exact book. Exit legs are separate
    orders (scale-out and runner sell independently), so per-leg minimums on
    the exit side are real, not an artifact.
    """
    d = {'n': 0}
    if not rows:
        return d
    groups = {}
    for t in rows:
        groups.setdefault((t.get('Symbol'), t.get('EntryDate')), []).append(t)
    recs = []
    total_comm = total_fixed = total_notional = 0.0
    floor_bound = floor_total = 0
    for legs in groups.values():
        q_tot = sum(t['Quantity'] for t in legs)
        ep = legs[0]['EntryPrice']
        rec_entry = sum(float(t.get('EntryCommission') or 0.0) for t in legs)
        entry_comm = rec_entry if rec_entry > 0 else _tiered_leg_comm(q_tot, ep)
        fixed_entry = _fixed_leg_comm(q_tot, ep)
        floor_total += 1
        if q_tot * _COMM_PER_SHARE < _COMM_MIN_ORDER:
            floor_bound += 1
        for t in legs:
            q = t['Quantity']
            notional = ep * q
            xp = float(t.get('ExitPrice') or ep)
            exit_comm = float(t.get('Commission') or 0.0)
            e_comm = entry_comm * q / q_tot
            rt = e_comm + exit_comm
            recs.append({'sym': t.get('Symbol'), 'date': t.get('ExitDate'),
                         'rt_usd': rt, 'rt_bps': rt / notional * 1e4,
                         'entry_bps': e_comm / notional * 1e4,
                         'exit_bps': exit_comm / notional * 1e4,
                         'pnl': float(t.get('PnL') or 0.0)})
            total_comm += rt
            total_notional += notional
            total_fixed += fixed_entry * q / q_tot + _fixed_leg_comm(q, xp)
            floor_total += 1
            if q * _COMM_PER_SHARE < _COMM_MIN_ORDER:
                floor_bound += 1

    bps = np.array([r['rt_bps'] for r in recs])
    usd = np.array([r['rt_usd'] for r in recs])
    gross_pnl = sum(r['pnl'] for r in recs)
    flipped = sum(1 for r in recs if r['pnl'] > 0 and r['pnl'] - r['rt_usd'] <= 0)
    d.update(
        n=len(recs),
        recorded_entry_legs=sum(1 for t in rows if (t.get('EntryCommission') or 0) > 0),
        rt_bps_mean=float(bps.mean()),
        rt_bps_median=float(np.median(bps)),
        rt_bps_p25=float(np.percentile(bps, 25)),
        rt_bps_p75=float(np.percentile(bps, 75)),
        rt_bps_weighted=total_comm / total_notional * 1e4 if total_notional else 0.0,
        entry_bps_mean=float(np.mean([r['entry_bps'] for r in recs])),
        exit_bps_mean=float(np.mean([r['exit_bps'] for r in recs])),
        rt_usd_mean=float(usd.mean()),
        rt_usd_median=float(np.median(usd)),
        total_comm_usd=total_comm,
        fixed_total_usd=total_fixed,
        fixed_saving_pct=(1 - total_comm / total_fixed) * 100 if total_fixed else 0.0,
        drag_pp_yr=(total_comm / initial_value * 100) * 252 / max(day_count, 1),
        comm_vs_gross_pct=total_comm / gross_pnl * 100 if gross_pnl > 0 else float('nan'),
        floor_bound_pct=100.0 * floor_bound / floor_total if floor_total else 0.0,
        fee_flipped=flipped,
        fee_flipped_pct=100.0 * flipped / len(recs),
        best=recs[int(np.argmin(bps))],
        worst=recs[int(np.argmax(bps))],
    )
    return d


def print_commission_and_fill_costs(results):
    """Dedicated commission and fill-cost section, denominated in bps of entry
    notional (memo rule: bps, never ticks). Split out of Trade Statistics on
    2026-09-01 after the account's Fixed -> Tiered switch (confirmed 2026-08-26)."""
    GREY = "\033[38;2;150;150;150m"
    RESET = "\033[0m"
    d = results.get('comm_detail') or {}
    # Labels must stay under colorize_output's 30-char column or the values jag.
    print("\nCommission & Fill Costs (bps of entry notional):")
    print(f"{GREY}Tiered ${_COMM_PER_SHARE:.4f}/sh, ${_COMM_MIN_ORDER:.2f} min, "
          f"1% cap, +$0.0002/sh fees; live since 2026-08-26{RESET}")
    if not d.get('n'):
        print("  no recorded trades with commission data")
        return
    _rec = d.get('recorded_entry_legs', 0)
    if _rec < d['n']:
        print(f"{GREY}entry legs measured {_rec}/{d['n']}, rest repriced on Tiered{RESET}")

    # Round-trip commission per trade (both legs, measured)
    print(colorize_output(d['rt_bps_mean'], "Round-Trip Comm Mean (bps):", 15.0, 40.0, lower_is_better=True))
    print(colorize_output(d['rt_bps_median'], "Round-Trip Comm Med (bps):", 15.0, 40.0, lower_is_better=True))
    print(colorize_output(d['rt_bps_p25'], "Round-Trip Comm p25 (bps):", 10.0, 30.0, lower_is_better=True))
    print(colorize_output(d['rt_bps_p75'], "Round-Trip Comm p75 (bps):", 25.0, 60.0, lower_is_better=True))
    print(colorize_output(d['rt_bps_weighted'], "RT Comm $-Weighted (bps):", 15.0, 40.0, lower_is_better=True))
    print(colorize_output(d['entry_bps_mean'], "Entry Leg Mean (bps):", 8.0, 20.0, lower_is_better=True))
    print(colorize_output(d['exit_bps_mean'], "Exit Leg Mean (bps):", 8.0, 20.0, lower_is_better=True))
    print(colorize_output(d['rt_usd_mean'], "Round-Trip Comm Mean ($):", 1.00, 2.50, lower_is_better=True))
    print(colorize_output(d['rt_usd_median'], "Round-Trip Comm Med ($):", 1.00, 2.50, lower_is_better=True))
    _b, _w = d['best'], d['worst']
    print(f"{'Cheapest Round Trip:':<30}{GREY}{_b['rt_bps']:.1f} bps  "
          f"{_b['sym']} exit {_b['date']}{RESET}")
    print(f"{'Priciest Round Trip:':<30}{GREY}{_w['rt_bps']:.1f} bps  "
          f"{_w['sym']} exit {_w['date']}{RESET}")

    # Aggregate drag
    print(f"{'Total Commission ($):':<30}{GREY}{d['total_comm_usd']:,.2f} over "
          f"{d['n']} legs{RESET}")
    print(colorize_output(d['drag_pp_yr'], "Comm Drag (pp/yr):", 2.0, 10.0, lower_is_better=True))
    if d['comm_vs_gross_pct'] == d['comm_vs_gross_pct']:
        print(colorize_output(d['comm_vs_gross_pct'], "Comm as % of Gross P&L:", 5.0, 25.0, lower_is_better=True))
    print(colorize_output(d['floor_bound_pct'], "Orders at $0.35 Min (%):", 20.0, 80.0, lower_is_better=True))
    print(colorize_output(d['fee_flipped_pct'], "Winners Flipped by Fees (%):", 1.0, 5.0, lower_is_better=True))

    # Fixed-vs-Tiered what-if: the same fills priced on the old plan
    print(colorize_output(d['fixed_saving_pct'], "Tiered Saving vs Fixed (%):", 40.0, 10.0))
    print(f"{'Same Book on Fixed ($):':<30}{GREY}{d['fixed_total_usd']:,.2f} "
          f"vs {d['total_comm_usd']:,.2f} on Tiered{RESET}")

    # Fee effect on the win rate (moved here from Trade Statistics)
    if results.get('gross_win_rate') and results.get('percent_profitable'):
        print(colorize_output(results['gross_win_rate'], "Win Rate (before fees) %:", 60, 40))
        print(colorize_output(results['gross_win_rate'] - results['percent_profitable'],
                              "Win Rate Lost to Fees (pp):", 0.5, 3.0, lower_is_better=True))
    print(colorize_output(results['commission_impact_pct'] * 100, "Exit Leg Impact (bps):", 8.0, 20.0, lower_is_better=True))
    print(colorize_output(results['breakeven_threshold_pct'] * 100, "Breakeven Threshold (bps):", 15.0, 40.0, lower_is_better=True))

    # Entry fill quality from the limit-entry arm: fill price vs that session's
    # open, positive = filled below the open. The raw census still prints from
    # _limit_report() at the end of the run.
    L = results.get('limit_log') or []
    if L:
        saved = np.array([r['saved'] * 1e4 for r in L])
        best_i, worst_i = int(np.argmax(saved)), int(np.argmin(saved))
        print("\nEntry Fill Quality (vs session open, bps):")
        print(colorize_output(float(saved.mean()), "Fill vs Open Mean (bps):", 10.0, 0.0))
        print(colorize_output(float(np.median(saved)), "Fill vs Open Median (bps):", 5.0, 0.0))
        print(colorize_output(100.0 * float((saved > 0).mean()), "Fills Better Than Open (%):", 60.0, 40.0))
        print(f"{'Best Fill:':<30}{GREY}{saved[best_i]:+.1f} bps  {L[best_i]['sym']}{RESET}")
        print(f"{'Worst Fill:':<30}{GREY}{saved[worst_i]:+.1f} bps  {L[worst_i]['sym']}{RESET}")
        _thr = sum(1 for r in L if r['through'])
        print(f"{'Gapped Through Trigger:':<30}{GREY}{_thr} of {len(L)} fills "
              f"({100.0 * _thr / len(L):.1f}%){RESET}")


def print_trade_statistics(results):
    """Print trade statistics with colorized output."""
    print("\nTrade Statistics:")
    print(colorize_output(results['total_closed'], "Total Trades:", 50, 10))
    print(colorize_output(results['percent_profitable'], "Win Rate (after fees) %:", 60, 40))
    # Commission impact, breakeven threshold and the before-fee win rate moved to
    # the dedicated Commission & Fill Costs section (2026-09-01).

    # Dollar thresholds scale with account size - a $10K account shouldn't be punished for small positions
    _iv = results['initial_value']
    avg_win_good = max(_iv / 150, 30)   # ~$67 on $10K
    avg_win_bad  = max(_iv / 600, 10)   # ~$17 on $10K
    print(colorize_output(results['won_avg'], "Avg. Winning Trade ($):", avg_win_good, avg_win_bad))
    # Losing trade up to 85% of winning trade is fine at 60%+ win rate
    print(colorize_output(results['lost_avg'], "Avg. Losing Trade ($):", results['won_avg'] * 0.75, results['won_avg'] * 1.0, lower_is_better=True))
    # Per-trade % move on capital deployed (size-independent). Thresholds are a
    # first pass for a ~1-day-hold strategy and easy to retune once the real
    # per-trade distribution is observed.
    print(colorize_output(results['avg_win_pct'], "Avg. Winning Trade (% move):", 3.0, 1.0))
    print(colorize_output(results['avg_loss_pct'], "Avg. Losing Trade (% move):", results['avg_win_pct'] * 0.8, results['avg_win_pct'] * 1.1, lower_is_better=True))
    # Largest win: good = 5% of account, bad = 0.5% of account
    print(colorize_output(results['won_max'], "Largest Win ($):", _iv / 20, _iv / 200))
    # Largest loss: good ≤ 50% of largest win, bad ≥ 90%
    print(colorize_output(results['lost_max'], "Largest Loss ($):", results['won_max'] * 0.5, results['won_max'] * 0.9, lower_is_better=True))
    print(colorize_output(results['largest_win_pct'], "Largest Win (% move):", 15.0, 5.0))
    print(colorize_output(results['largest_loss_pct'], "Largest Loss (% move):", results['largest_win_pct'] * 0.5, results['largest_win_pct'] * 2.0, lower_is_better=True))
    print(colorize_output(results['avg_profit_per_trade'], "Avg. Trade P&L:", 50, 0))
    print(colorize_output(results['profit_factor'], "Profit Factor:", 2.5, 1.0))

    # EV Per Trade: win_rate * avg_win - loss_rate * avg_loss (do NOT divide by trade count)
    win_rate_dec = results['percent_profitable'] / 100
    results['Expected_Value_PerTrade'] = (win_rate_dec * results['won_avg']) - ((1 - win_rate_dec) * results['lost_avg'])
    ev_good = max(_iv / 500, 10)   # ~$20 on $10K
    ev_bad  = max(_iv / 2000, 3)   # ~$5  on $10K
    print(colorize_output(results['Expected_Value_PerTrade'], "EV Per Trade ($):", ev_good, ev_bad))

    print(colorize_output(results['net_profit_drawdown_ratio'], "Net Profit / Drawdown Ratio:", 3.0, 1.0))


def print_capital_efficiency(results):
    """Print strategy-level capital-efficiency / book-utilization metrics.

    These surface things the per-trade stats hide: how full the book runs day
    to day and how much capital is actually deployed vs sitting in cash
    (reserves + sub-full sizing). Averages are over the ACTIVE window -- from
    the first day a position is held onward -- so the leading warm-up period,
    when lagged signals are still accumulating and nothing trades, doesn't drag
    the numbers down.
    """
    daily_positions = results.get('daily_positions') or []
    daily_deployment = results.get('daily_deployment') or []
    max_pos = results.get('max_positions')

    if not daily_positions or not max_pos:
        return  # nothing to report (older run or no tracking data)

    # Active window starts the first day we actually hold something.
    first_active = next((i for i, n in enumerate(daily_positions) if n > 0), None)
    if first_active is None:
        return  # never took a position

    active_positions = daily_positions[first_active:]
    active_deployment = daily_deployment[first_active:] if daily_deployment else []
    n_active = len(active_positions)

    avg_positions = sum(active_positions) / n_active
    slot_util = avg_positions / max_pos * 100
    peak_positions = max(active_positions)
    pct_full_book = sum(1 for n in active_positions if n >= max_pos) / n_active * 100
    pct_flat = sum(1 for n in active_positions if n == 0) / n_active * 100
    avg_deployment = (sum(active_deployment) / len(active_deployment)) if active_deployment else 0.0

    # Utilisation is a CONSEQUENCE of turnover, not a free dial: at steady state
    #     avg positions held  ~=  entries per day  x  days held per position
    # so 48% utilisation on a 10-slot book with ~1.2-day holds needs ~4 entries/day to
    # be arithmetically possible, and raising it means holding longer or entering more
    # often -- not "filling empty slots". Decomposing it here stops the % being read as
    # idle capacity that could simply be switched on.
    # (Audited 2026-07-28 against an independent reconstruction from the trade book;
    # the counter is correct. A reconstruction that treats a position as held THROUGH
    # its ExitDate double-counts, because the exit fills at that bar's open and the book
    # is sampled after the broker has executed it -- that error reads ~9.0 instead of
    # ~4.9 on ~1-day holds.)
    n_trades_for_util = results.get('total_closed') or results.get('total_trades') or 0
    entries_per_day = (n_trades_for_util / n_active) if n_active else 0.0
    implied_hold = (avg_positions / entries_per_day) if entries_per_day > 0 else 0.0

    print("\nCapital Efficiency (active trading window):")
    print(f"{'Max Position Slots:':<30}{max_pos}")
    print(colorize_output(avg_positions, "Avg. Positions Held:", max_pos * 0.8, max_pos * 0.4))
    print(colorize_output(slot_util, "Slot Utilization %:", 80, 50))
    print(colorize_output(avg_deployment, "Avg. Capital Deployed %:", 80, 50))
    print(colorize_output(pct_full_book, "Days at Full Book %:", 50, 15))
    print(colorize_output(pct_flat, "Days Flat (in cash) %:", 5, 20, lower_is_better=True))
    print(f"{'Peak Concurrent Positions:':<30}{peak_positions} / {max_pos}")
    if entries_per_day > 0:
        print(f"{'Utilisation Identity:':<30}{entries_per_day:.2f} entries/day x "
              f"{implied_hold:.2f} days held = {avg_positions:.2f} of {max_pos} slots")

    # Sampling integrity. next() fires more than once per trading day when feed bar
    # dates are unaligned; if this ratio drifts from ~1.00 the per-day dedupe has
    # regressed and every number above is biased low. Printed only when it is wrong.
    # BT_CAP_DUMP=<path> writes the raw per-day series so the counter can be audited
    # against an independent reconstruction from the trade book. Off unless asked.
    _dump = None
    if _dump:
        try:
            import csv as _csv
            _dates = results.get('daily_cap_dates') or []
            with open(_dump, 'w', newline='') as _fh:
                _w = _csv.writer(_fh)
                _w.writerow(['date', 'positions_held', 'deployment_pct'])
                for _i, _n in enumerate(daily_positions):
                    _w.writerow([_dates[_i] if _i < len(_dates) else '',
                                 _n,
                                 daily_deployment[_i] if _i < len(daily_deployment) else ''])
            print(f"{'  cap dump:':<30}{_dump} ({len(daily_positions)} rows)")
        except Exception as _e:
            print(f"{'  cap dump FAILED:':<30}{_e}")

    n_calls = results.get('next_calls') or 0
    n_days = len(results.get('daily_cap_dates') or daily_positions)
    if n_days and n_calls and abs(n_calls / n_days - 1.0) > 0.02:
        print(f"{'  sampling:':<30}{n_calls} next() calls over {n_days} trading days "
              f"({n_calls / n_days:.2f}x) - deduped to 1 sample/day")


def print_trade_management_metrics(results):
    """Print trade management metrics with colorized output."""
    print("\nTrade Management Metrics:")
    print(colorize_output(results['avg_trade_len'], "Avg. Holding Period (days):", 1, 5, lower_is_better=True))
    print(colorize_output(results['longest_trade'], "Longest Trade (days):", 15, 25, lower_is_better=True))
    print(colorize_output(results['shortest_trade'], "Shortest Trade (days):", 1, 5))
    print(colorize_output(results['max_consecutive_wins'], "Max Consecutive Wins:", 5, 3))
    print(colorize_output(results['max_consecutive_losses'], "Max Consecutive Losses:", max(1, results['max_consecutive_wins'] - 1), results['max_consecutive_wins'] + 1, lower_is_better=True))
    print(f"{'Current Streak:':<30}{results['current_streak'] if results['current_streak'] else 'None'}")
    print(colorize_output(results['win_loss_count_ratio'], "Win/Loss Count Ratio:", 1.5, 0.8))
    print(colorize_output(results['risk_reward_ratio'], "Risk/Reward Ratio:", 2.5, 1.0))
    print(colorize_output(results['kelly_percentage'], "Kelly %:", 20, 5))


def print_advanced_trade_metrics(results):
    """Print advanced trade quality metrics with colorized output."""
    print("\nAdvanced Trade Quality Metrics:")
    print(colorize_output(results['positive_days_pct'], "Percentage of Positive Days:", 50, 20))
    print(colorize_output(results['max_pos_streak'], "Max Pos Streak:", 5, 3))
    print(colorize_output(results['max_neg_streak'], "Max Neg streak:", results['max_pos_streak'], results['max_pos_streak'] * 10, lower_is_better=True))
    print(colorize_output(results['profit_per_day'], "Profit per Day ($):", 20, 5))


def print_trade_history_extremes(results):
    """Print the best/worst day, week, and month, then show how far each of
    those extremes deviated from a normal period of trading.

    This is a recency-bias / out-of-sample sanity check: it reveals whether the
    headline return leans on a handful of outlier periods, and quantifies just
    how unusual those periods were.

    Robust to variable week/month length. Trading weeks aren't always 5 days
    (holidays make them 4 or 3), and a backtest can start or end mid-week:
      * boundary stub periods (a partial first/last week/month that is just an
        artifact of where the window happens to fall) are dropped, while genuine
        interior holiday-shortened weeks are kept;
      * the trading-day count is printed for every week/month; and
      * the "how wild" sigma is LENGTH-ADJUSTED -- each period is scored against
        the daily distribution scaled by sqrt(its own trading days), so a big
        3-day week and a big 5-day week are judged on equal footing instead of
        being mixed into one length-confounded weekly std.
    """
    series = results.get('daily_return_series')
    if series is None or len(series) < 20:
        return

    # ANSI helpers. Pad the *plain* text before colouring so the columns stay
    # vertically aligned (escape codes have zero display width but count in
    # f-string padding, so they must sit outside the padded field).
    GREEN = "\033[38;2;0;200;0m"
    RED   = "\033[38;2;220;0;0m"
    GREY  = "\033[38;2;150;150;150m"
    RESET = "\033[0m"

    def pct_field(value, width=10):
        color = GREEN if value >= 0 else RED
        return f"{color}{f'{value * 100:+.2f}%':<{width}}{RESET}"

    def severity_tag(z):
        az = abs(z)
        if az >= 4.0:
            return "Extreme outlier"
        if az >= 2.5:
            return "Very unusual"
        if az >= 1.5:
            return "Notable"
        return "Normal"

    # Daily distribution is the uniform source of truth (a day is a day), so all
    # length adjustment is derived from it rather than from length-confounded
    # weekly/monthly stds.
    daily_mean = series.mean()
    daily_std  = series.std()
    daily_typ  = series.abs().median()

    def _trim_boundary_stubs(ret, counts):
        """Drop only a partial first/last bin (a stub created by where the
        backtest window starts/ends). Interior weeks are always real -- a
        holiday-shortened interior week is a legitimate week and is kept."""
        if len(ret) <= 2:
            return ret, counts
        full = counts.median()          # typical full-length bin (e.g. 5 for weeks)
        keep = np.ones(len(ret), dtype=bool)
        if counts.iloc[0] < full:
            keep[0] = False
        if counts.iloc[-1] < full:
            keep[-1] = False
        return ret[keep], counts[keep]

    # Build week/month return curves from the same daily series so every figure
    # is internally consistent. Drop empty bins, then trim boundary stubs.
    def build(freq):
        ret = (1 + series).resample(freq).prod() - 1
        counts = series.resample(freq).count()
        mask = counts > 0
        ret, counts = ret[mask], counts[mask]
        return _trim_boundary_stubs(ret, counts)

    week_ret,  week_cnt  = build('W')
    month_ret, month_cnt = build('ME')
    day_cnt = pd.Series(1, index=series.index)   # every day is a full 1-day bin

    # (name, returns, day-counts, date formatter, prefix, show trading-day count)
    periods = [
        ("Day",   series,    day_cnt,   lambda d: d.strftime('%Y-%m-%d'), "on ",          False),
        ("Week",  week_ret,  week_cnt,  lambda d: d.strftime('%Y-%m-%d'), "week ending ", True),
        ("Month", month_ret, month_cnt, lambda d: d.strftime('%Y %B'),    "",             True),
    ]

    def days_note(k, show):
        return f" ({k} trading days)" if show else ""

    # --- Best & worst periods -------------------------------------------------
    print("\nTrade History - Best & Worst Periods:")
    for name, s, cnt, fmt_date, prefix, show in periods:
        if s.empty:
            continue
        for which, val, dt in (("Best", s.max(), s.idxmax()),
                               ("Worst", s.min(), s.idxmin())):
            k = int(cnt.loc[dt])
            print(f"{f'{which} {name}:':<30}{pct_field(val)}"
                  f"{GREY}{prefix}{fmt_date(dt)}{days_note(k, show)}{RESET}")

    # --- How wild were those extremes vs a normal trading period? -------------
    print("\nDeviation From Normal Trading (sigma scaled by sqrt of period days):")
    print(f"{'Daily baseline:':<30}{GREY}mean {daily_mean * 100:+.2f}%/day, "
          f"sigma {daily_std * 100:.2f}%, typical move {daily_typ * 100:.2f}%{RESET}")
    if daily_std <= 0:
        return
    for name, s, cnt, fmt_date, prefix, show in periods:
        if s.empty:
            continue
        for which, val, dt in (("Best", s.max(), s.idxmax()),
                               ("Worst", s.min(), s.idxmin())):
            k = int(cnt.loc[dt])
            # CLT z-score of the period's return vs k iid daily moves.
            z = (val - k * daily_mean) / (daily_std * np.sqrt(k))
            color = GREEN if val >= 0 else RED
            z_str = f"{z:+.1f} sigma"
            print(f"{f'  {which} {name}:':<30}{color}{f'{z_str:<12}'}{RESET}"
                  f"{GREY}{days_note(k, show).strip() + '  ' if show else ''}"
                  f"[{severity_tag(z)}]{RESET}")


def print_signal_quality_metrics(results):
    """Print statistical validity and return-distribution quality metrics."""
    keys = ['psr', 'serenity_ratio', 'tail_ratio']
    if not any(results.get(k) is not None for k in keys):
        return

    print("\nStatistical Quality Metrics:")

    if results.get('psr') is not None:
        # P(true annualized SR > 1.0) corrected for skewness & kurtosis - > 99.91% is iron-clad
        print(colorize_output(results['psr'] * 100, "Probabilistic Sharpe (%):",
                              good_threshold=97.0, bad_threshold=90.0,
                              unicorn_multiplier=1.031))  # unicorn at >= 99.91%

    if results.get('serenity_ratio') is not None:
        # (Annual Return - RF) / Ulcer Index; penalises lingering drawdowns unlike Calmar
        print(colorize_output(results['serenity_ratio'], "Serenity Ratio:",
                              good_threshold=20.0, bad_threshold=5.0,
                              unicorn_multiplier=50.0))   # unicorn at >= 1000

    if results.get('tail_ratio') is not None:
        # P95 daily return / |P5 daily return|; > 1.0 = fat right tail, > 3.0 is unicorn
        print(colorize_output(results['tail_ratio'], "Tail Ratio (P95/|P5|):",
                              good_threshold=1.5, bad_threshold=0.9,
                              unicorn_multiplier=2.0))    # unicorn at >= 3.0



def calculate_period_returns(prices):
    """Calculate monthly and yearly returns from a price series."""
    # Convert to dataframe if it's a series
    if isinstance(prices, pd.Series):
        prices = pd.DataFrame(prices)
    
    # Make sure we have a datetime index
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    
    # Calculate daily returns
    daily_returns = prices.pct_change().dropna()
    
    # Monthly returns
    monthly_returns = {}
    
    # Group by year and month
    monthly_grouped = daily_returns.groupby([daily_returns.index.year, daily_returns.index.month])
    
    for (year, month), group in monthly_grouped:
        month_name = f"{year}-{month:02d}"
        # Calculate compounded return for the month
        monthly_return = ((1 + group.iloc[:, 0]).cumprod().iloc[-1] - 1) * 100
        monthly_returns[month_name] = monthly_return
    
    # Yearly returns
    yearly_returns = {}
    
    # Group by year
    yearly_grouped = daily_returns.groupby(daily_returns.index.year)
    
    for year, group in yearly_grouped:
        # Calculate compounded return for the year
        yearly_return = ((1 + group.iloc[:, 0]).cumprod().iloc[-1] - 1) * 100
        yearly_returns[str(year)] = yearly_return
    
    # Calculate annualized return
    total_days = (prices.index[-1] - prices.index[0]).days
    total_years = total_days / 365.25
    total_return = (prices.iloc[-1, 0] / prices.iloc[0, 0] - 1) * 100
    annualized_return = ((1 + total_return/100) ** (1/total_years) - 1) * 100
    
    return {
        'monthly_performance': monthly_returns,
        'yearly_performance': yearly_returns,
        'annualized_return': annualized_return
    }









def print_period_performance(results, start_date=None, end_date=None):
    """
    Print monthly and yearly performance metrics with REALISTIC market-beating expectations.
    
    Threshold Philosophy:
    - S&P 500 averages ~12% annually (~1% monthly)
    - Your strategy should consistently beat this or why bother?
    - 0% months are unacceptable - you're not beating cash
    - 1%+ monthly is where you should be as a minimum
    
    Parameters:
    -----------
    results : dict
        Dictionary containing performance metrics
    start_date : str or datetime
        Start date for market data (default: 2 years before today)
    end_date : str or datetime
        End date for market data (default: today)
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
        
    if start_date is None:
        # Default to 2 years before end date
        if 'monthly_performance' in results and results['monthly_performance']:
            # Extract start date from the first month in results
            first_month = min(results['monthly_performance'].keys())
            year, month = map(int, first_month.split('-'))
            start_date = datetime(year, month, 1)
        else:
            start_date = end_date - timedelta(days=2*365)
    elif isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    # Download S&P 500 data for comparison
    try:
        sp500_data = yf.download(
            "^GSPC",
            start=start_date.strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            interval="1d",
            auto_adjust=True,
            progress=False
        )
        
        if len(sp500_data) == 0:
            print("Warning: No S&P 500 data available for the specified period.")
            market_results = None
        else:
            market_results = calculate_period_returns(sp500_data['Close'])
    except Exception as e:
        market_results = None
    
    # ------------------------------------------------------------------
    # Warm-up detection: leading months where the strategy held flat (no
    # trades) because lagged can_buy signals were still accumulating. The
    # strategy did nothing in these months, so a 0.00% return and any
    # "alpha gap" vs the market are NOT indicative of the system -- the gap
    # is just the market drifting while we sit in cash. Tag them instead of
    # scoring them, and drop them from the alpha view.
    # ------------------------------------------------------------------
    WARMUP_EPS = 1e-6  # a no-trade month returns exactly 0.0 (equity == cash)
    GRAY = "\033[38;2;150;150;150m"
    RESET = "\033[0m"

    def _warmup_line(label, value, note):
        return f"{label:<30}{GRAY}{value:<10.2f}{RESET}[{GRAY}{note}{RESET}]"

    warmup_months = set()
    if results['monthly_performance']:
        for _m in sorted(results['monthly_performance'].keys()):
            if abs(results['monthly_performance'][_m]) < WARMUP_EPS:
                warmup_months.add(_m)
            else:
                break  # only the leading flat run is warm-up; stop at first traded month

    # REALISTIC MONTHLY PERFORMANCE THRESHOLDS
    print("\nStrategy Monthly Performance (%):")
    if warmup_months:
        print(f"({len(warmup_months)} leading warm-up month(s) shown, not scored)")

    if results['monthly_performance']:
        months = sorted(results['monthly_performance'].keys())
        # Running account value alongside the % so the equity PATH is visible, not just
        # the per-month rate -- a flat stretch followed by a late run looks identical to
        # steady compounding when you only see percentages.
        _acct = float(results.get('initial_value', 10000.0))
        _month_acct = {}

        for month in months:
            perf = results['monthly_performance'][month]
            month_label = datetime.strptime(month, '%Y-%m').strftime('%Y %B')
            _acct *= (1.0 + perf / 100.0)
            _month_acct[month] = _acct

            if month in warmup_months:
                print(_warmup_line(f"{month_label}:", perf, "Warm-up - no trades"))
                continue

            print(colorize_output(perf, f"{month_label}:",
                                good_threshold=1.8,      # 1.8%+ is good (20%+ annualized)
                                bad_threshold=0.6,       # <0.6% is poor (7% annualized)
                                lower_is_better=False,
                                extra=f"${_acct:,.0f}"))

    # MARKET COMPARISON - Only show if we have market data
    if market_results and market_results['monthly_performance']:
        print("\nMonthly Excess vs S&P 500 (%):")
        if warmup_months:
            print("(warm-up months excluded)")

        market_months = sorted(market_results['monthly_performance'].keys())

        for month in months:
            if month in market_results['monthly_performance']:
                strategy_perf = results['monthly_performance'][month]
                market_perf = market_results['monthly_performance'][month]
                relative_perf = strategy_perf - market_perf
                month_label = datetime.strptime(month, '%Y-%m').strftime('%Y %B')

                if month in warmup_months:
                    print(_warmup_line(f"{month_label}:", relative_perf, "Warm-up - excluded"))
                    continue

                print(colorize_output(relative_perf, f"{month_label}:",
                                    good_threshold=1.0,      # 1%+ monthly alpha is good
                                    bad_threshold=0.0,       # Negative alpha is poor
                                    lower_is_better=False,
                                    extra=f"${_month_acct.get(month, float('nan')):,.0f}"
                                          if _month_acct.get(month) is not None else None))
    
    # REALISTIC YEARLY PERFORMANCE THRESHOLDS
    if results['yearly_performance']:
        print("\nStrategy Yearly Performance (%):")
        
        years = sorted(results['yearly_performance'].keys())
        for year in years:
            perf = results['yearly_performance'][year]
            
            
            print(colorize_output(perf, f"{year}:", 
                                good_threshold=22,       # 22%+ is good
                                bad_threshold=12,        # <12% is poor (market average)
                                lower_is_better=False))
        
        # Strategy annualized return
        print(colorize_output(results['annualized_return'], "Strategy Annualized Return:", 
                            good_threshold=22,           # 22%+ is good
                            bad_threshold=12,            # <12% is poor
                            lower_is_better=False))
    
    # YEARLY EXCESS RETURNS vs MARKET
    if market_results and market_results['yearly_performance']:
        print("\nYearly Excess vs S&P 500 (%):")
        
        market_years = sorted(market_results['yearly_performance'].keys())
        for year in years:
            if year in market_results['yearly_performance']:
                strategy_perf = results['yearly_performance'][year]
                market_perf = market_results['yearly_performance'][year]
                relative_perf = strategy_perf - market_perf
                
                
                print(colorize_output(relative_perf, f"{year}:", 
                                    good_threshold=10,       # 10%+ annual alpha is good
                                    bad_threshold=3,         # <3% annual alpha is poor
                                    lower_is_better=False))
        
        # Relative annualized return
        relative_annualized = results['annualized_return'] - market_results['annualized_return']
        print(colorize_output(relative_annualized, "Excess Annualized Return:", 
                            good_threshold=10,           # 10%+ annual alpha is good
                            bad_threshold=3,             # <3% annual alpha is poor
                            lower_is_better=False))

# ------------------------------------------------------------------------------
# Optional routines: plotting, buy signal check, and logging summary
# ------------------------------------------------------------------------------

def try_plot_results(cerebro, logger):
    """Attempt to plot the results if the dataset is small enough."""
    try:
        if len(cerebro.datas) <= 10:
            plt.style.use('dark_background')
            plt.rcParams['figure.facecolor'] = '#1e1e1e'
            plt.rcParams['axes.facecolor'] = '#1e1e1e'
            plt.rcParams['grid.color'] = '#333333'
            
            cerebro.plot(style='candlestick',
                         barup='green',
                         bardown='red',
                         volup='green',
                         voldown='red',
                         grid=True,
                         subplot=True)
    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")

def create_results_summary(results):
    """Return a summary dictionary of the key backtest results."""
    summary = {
        'total_return': results['total_return'],
        'annualized_return': results['annualized_return'],
        'sharpe_ratio': results['sharpe_ratio'],
        'sortino_ratio': results['sortino_ratio'],
        'calmar_ratio': results['calmar_ratio'],
        'max_drawdown': results['max_dd'],
        'win_rate': results['percent_profitable'],
        'profit_factor': results['profit_factor'],
        'total_trades': results['total_closed'],
        'avg_trade_pnl': results['avg_profit_per_trade'],
        'risk_reward_ratio': results['risk_reward_ratio'],
        'sqn': results['sqn_value']
    }
    return summary



def arg_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stock Sniper Trading Strategy")
    parser.add_argument("--sample", type=float, default=100, help="Percentage of random files to backtest (0-100)")
    parser.add_argument("--runpercent", type=float, default=None, help="Alias for --sample: percent of the universe to backtest, e.g. --runpercent 95. If set, overrides --sample.")
    parser.add_argument("--data_dir", default="Data/RFpredictions", help="Directory of per-ticker prediction parquets to backtest (read-only override; default = live signals). Use to validate a candidate model without clobbering live Data/RFpredictions.")
    parser.add_argument("--filter", type=float, default=0.01,help="Minimum UpProbability variance for stock filtering")
    parser.add_argument("--up_prob", type=float, default=0.68,help="UpProbability threshold for buy signals")
    parser.add_argument("--force", action='store_true', help="Force the script to run even if data is not up to last trading date")
    parser.add_argument("--recommend", action='store_true', default=False, help="Recommend basic system changes based on the backtest risk metrics")
    parser.add_argument("--best", action='store_true', default=False, help="Generate best buy signals for the current or last trading day")
    parser.add_argument("--num_signals", type=int, default=4, help="Number of best signals to generate (default: 4)")
    
    # Add new optimization related arguments
    parser.add_argument("--optimize", action='store_true', default=False, help="Run in optimization mode to find best parameters")
    parser.add_argument("--optimize_param", type=str, action='append', default=None, 
                       help="Parameters to optimize (can be used multiple times, e.g. --optimize_param up_prob_threshold --optimize_param max_positions)")
    parser.add_argument("--runs", type=int, default=10, help="Number of optimization runs (default: 10)")
    
    # Add individual parameter arguments for more granular control
    parser.add_argument("--max_positions", type=int, help="Maximum number of concurrent positions")
    parser.add_argument("--risk_per_trade", type=float, help="Risk per trade percentage")
    parser.add_argument("--stop_loss_atr", type=float, help="Stop loss ATR multiple")
    parser.add_argument("--trailing_stop_atr", type=float, help="Trailing stop ATR multiple")
    parser.add_argument("--take_profit", type=float, help="Take profit percentage")
    parser.add_argument("--position_timeout", type=int, help="Maximum days to hold a position")
    parser.add_argument("--selrule", type=str, default=None,
                       choices=["shipped", "low_atr", "low_atr_x_corr"],
                       help="Candidate-ranking policy (experiment seam). 'shipped' = rank by "
                            "raw UpProbability, bit-identical to the pre-2026-07-28 path (this "
                            "is what runs with no arg). 'low_atr' = up - 0.5*(ATR/price): "
                            "+14.07pp mean annualised over 6 asof anchors, t=+3.76, p=0.013, "
                            "positive at 6/6. 'low_atr_x_corr' = the same, times (1 - "
                            "cluster_correlation): +32.29pp mean but sd 35.08, 5/6, and it "
                            "depends on a Correlations.parquet that is 5 months stale. "
                            "Dose settable via BT_SELRULE_K (default 0.5 -- the largest dose "
                            "with no losing anchor; raising it doubles the variance). Also "
                            "settable via BT_SELRULE env var; the CLI arg wins.")
    parser.add_argument("--sizer", type=str, default=None,
                       help="Position-sizing policy (experiment seam). 'default' = production "
                            "PositionSizer, bit-identical (this is what runs with no arg). "
                            "Experimental: equal_weight (1/N control, REFUTED by the 16-seed "
                            "2026-06-28 A/B), inverse_vol (band+inverse-vol, matches "
                            "Data/_sizing_exp/bt_sizeexp.py). Also settable via BT_SIZER env var; "
                            "the CLI arg wins. Unknown names abort rather than silently no-op.")
    args = parser.parse_args()
    # --runpercent is an alias for --sample (percent of the universe to backtest).
    if getattr(args, 'runpercent', None) is not None:
        args.sample = args.runpercent
    # --sizer publishes into BT_SIZER: the strategy is constructed by backtrader
    # deep inside cerebro.run() and never sees `args`, so env is the transport.
    # CLI wins over a pre-set env var; absent the arg, the env value stands.
    if getattr(args, 'sizer', None):
        os.environ['BT_SIZER'] = str(args.sizer)
    if getattr(args, 'selrule', None):
        os.environ['BT_SELRULE'] = str(args.selrule)
    return args



if __name__ == "__main__":
    # STARTUP RESET FIX 2026-07-29: clear_completed_trades() READS the parquet
    # before truncating it, so a corrupt file (torn write from a killed run, or
    # two backtests sharing Data/TradeHistory.parquet) made the clear a silent
    # no-op and the corrupt file survived into the run. It returns False on that
    # read failure and None on success. On failure, delete the file outright;
    # TradeRecorder recreates it cleanly in stop().
    try:
        _cleared = clear_completed_trades()
    except Exception:
        _cleared = False
    if _cleared is False:
        try:
            os.remove(COMPLETED_TRADES_FILE)
            print(f"Removed unreadable {COMPLETED_TRADES_FILE}; "
                  f"it will be recreated at the end of the run.")
        except FileNotFoundError:
            pass
        except OSError as _e:
            print(f"WARNING: could not remove corrupt {COMPLETED_TRADES_FILE}: {_e}. "
                  f"Close any process holding it open and rerun.")
    main()

