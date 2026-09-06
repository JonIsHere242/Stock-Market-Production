#!/usr/bin/env python
"""Trigger-entry table: the ONE place a limit-entry trigger price is computed.

WHY THIS MODULE EXISTS
----------------------
The trigger price was implemented three times independently: in the backtester fork
(`5v2__NightlyBackTester_CanBuyV2.py`, `_vol_estimate` / `_beta_for`), in the paper
shadow (`9p__PaperTriggerBroker.py`, `build_triggers`) and now in the live broker.
Three copies of one formula is exactly the shape of the RSI defect this repo already
paid for, where the predictor gate and the FilterRubric measured the same indicator and
disagreed by 11 to 13 points. The paper broker's whole purpose is to measure slippage
against the price the live broker will actually rest at, and that measurement is void if
the two compute different triggers. So both import this.

THE FORMULA
-----------
    depth   = clamp(K * beta**BETA_EXP * vol_20d, MIN_DEPTH, MAX_DEPTH)
    trigger = prior_close * (1 - depth)

vol_20d is an equal-weighted 20-day close-to-close standard deviation, computed from
N closes to N-1 returns. That N-vs-N+1 detail is load-bearing: taking the last 20
RETURNS instead silently uses 21 closes and shifts every trigger, which would decouple
this from the backtested arm it is meant to reproduce.

beta is a rolling 60d beta vs SPY read from the beta lookup table, `asof` the last close
date. A missing name falls back to beta 1.0, which degrades the depth to K * vol_20d
rather than dropping the candidate.

WHY beta*vol AND NOT A FLAT PERCENTAGE
--------------------------------------
A flat -5% limit almost never fills on a quiet name and fills constantly on a noisy one,
so a flat rule silently becomes a volatility screen. Scaling to the name's own beta and
20-day vol asks every name for a dip of comparable rarity.

WHAT IS MEASURED, AND WHAT IS NOT
---------------------------------
Ten shuffle seeds across reach 15/24/30 put the trigger arm at +0.7730%/trade +/- 0.2198
against a shipped control of +0.7360%, i.e. an edge of +0.0370pp at t 0.53, 5 of 10 runs.
The RETURN edge is inside noise and must not be sized as real. What is not inside noise
is the risk profile: every one of those ten runs landed max drawdown between 17.83% and
23.92% against the control's 33.25%, with zero overlap, on 33.4-47.5% of capital deployed
against 58.48%. Reach itself is also inside noise (24 vs 15 is 0.23 pooled sd, 24 vs 30 is
0.49), so 24 is a convenience, not an optimum.

Every one of those numbers is GROSS OF SPREAD: the v2 fork carried an `IBKRSlippageModel`
that was never attached to cerebro, and it was deleted as dead code on 2026-08-30. Nothing
in the backtester models the spread.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL_FILE   = os.path.join(REPO, 'Data', '0__Signals.parquet')
NARROW_FILE = os.path.join(REPO, '_Buy_Signals.parquet')
PRED_DIR    = os.path.join(REPO, 'Data', 'RFpredictions')
BETA_FILE   = os.path.join(REPO, 'Data', '_canbuyv2', 'beta_lookup.parquet')

# Defaults reproduce FULL_reach24.parquet, which is
# BT_LIMIT_ENTRY_K=1.5 BT_LIMIT_OVERSUB=8 at 3 slots.
DEFAULT_K      = 1.5
DEFAULT_REACH  = 24
# beta^1. Tested 2026-08-28 as a full 2x2 against beta^0: dropping the beta term costs
# -1.891 Sharpe and is worse in 5 of 5 paired seeds (13.08 -> 23.31 max drawdown). An
# offline screen had said the opposite at 53 SE, because it scored FORECAST accuracy
# against next-day drop while beta is doing PORTFOLIO work - it demands a proportionally
# larger discount from names that move because the market moved. Do not drop it.
BETA_EXP       = 1.0
MIN_DEPTH      = 0.005
MAX_DEPTH      = 0.15
VOL_LOOKBACK   = 20

# Gap-frequency pool gate, SHIPPED 2026-08-29 as the baseline (N=2 at 4%). Before the
# reach cut, drop any pool name with >= GAP_GATE_N overnight gaps of
# |Open / prior Close - 1| > GAP_GATE_PCT in its last GAP_GATE_LOOKBACK sessions (the
# signal day's own open counts). Byte-equivalent to `_hyg_ok` in
# experimental/backtesters/5v4__ExitLab.py (BT_GATE_GAPFREQ_N / BT_GATE_GAP_PCT).
#
# EVIDENCE, 5v4 live-faithful trigger arm, paired shuffle seeds, cashadjust fixed, model
# exits off (memo project_exit_sweeps_trigger_arm_2026_08_29, chains 8, 11, 13, A):
#   N=2 pct=4.0 alone (prunes ~7%):  AnnRet +19.8 t 2.0 3/4, maxDD -2.1 4/4, mean/TRADE +0.28.
#   N=2 pct=4.0 + sell half at +15%: 8 seeds, AnnRet +39.5 (4 seeds, t 4.6) and 4/4 again
#                                    on seeds 3/7/23/42 (98/58/95/79 vs 56/9/36/23), maxDD
#                                    lower on all 8, merged mean/TRADE +0.54 t 4.4.
#   N=2 pct=3.0 (prunes ~21%): AnnRet +22.2 t 4.5 4/4 but skips 8 of 12 live names.
#   N=1 pct=3.0 (prunes 51%):  LOSES on every seed. Anything past ~25% pruned collapses,
#                              and a SECOND prune stacked on this one collapses too.
# The gate must run on the WIDE pool before head(reach) so the cut refills, exactly as
# the sim's budget loop refills; applied to an already-cut 12 it pruned 8 of 12 on the
# 2026-08-27 pool. Rollback: TRIGGER_GAP_GATE_N=0 (pool byte-identical to before).
GAP_GATE_N        = int(float(os.environ.get('TRIGGER_GAP_GATE_N', '2') or 0))
GAP_GATE_PCT      = float(os.environ.get('TRIGGER_GAP_GATE_PCT', '4.0'))
GAP_GATE_LOOKBACK = 20
LAST_GAP_CENSUS   = {}     # filled by apply_gap_gate on every load_pool call

# Volatility estimator for the trigger depth. 'parkinson' shipped 2026-08-28.
#
# EVIDENCE, full universe, reach 12, K 1.5, paired on BT_LIMIT_SHUFFLE (n=9):
#   Sharpe   2.36 -> 2.77   +0.41   t +4.33   better in 9 of 9 seeds
#   Ulcer    5.22 -> 4.28   -0.94   t -2.43
#   Max DD  15.57 -> 12.92  -2.65   t -2.35
#   Return  inside noise (t +1.58) - this is a RISK improvement, not a return one.
#
# MECHANISM. Not a deeper bid (mean sigma moves 0.03661 -> 0.03621, ratio 0.989), not gap
# avoidance (gap-through share unchanged, t +0.18), and not reduced exposure (it takes
# MORE trades and deploys MORE capital). It is estimator ACCURACY: a high-low range
# carries more information about the day's path than two endpoints, so the same depth
# budget is aimed better. Coefficient of variation 1.300 -> 0.938, and Spearman against
# next-day drop from prior close 0.3896 -> 0.4193 (n=189,299 name-days).
#
# Speed is NOT the mechanism: the two fastest-reacting estimators both lost badly
# (ewma85 Sharpe 1.70, maxsl 2.38 vs parkinson 2.91). Garman-Klass and Rogers-Satchell
# score within 1.4 SE of parkinson - the axis is exhausted, not merely improved.
# 'yz' (Yang-Zhang) SHIPPED 2026-08-30. Evidence, 5v4 live-faithful stack, 8 paired
# seeds (1,2,3,5,7,13,23,42), memo project_depth_estimator_screen_2026_08_30:
#   Sharpe +0.32 t 3.9 8/8, AnnVol -1.06 8/8, mean/TRADE +0.21 t 3.4 8/8, median +0.22
#   8/8, bottom-5% sum +3.0 8/8, big losers -1.5/seed 7/8, AnnRet +7.1 t 2.0 5/8;
#   cost: trades -9.75/seed and deployment -2.7pp.
# MECHANISM: Yang-Zhang adds the OVERNIGHT-GAP variance Parkinson cannot see, so gappy
# names must offer a deeper discount before the bid rests. The no-gate control showed
# yz alone recovers essentially the gap gate's whole benefit (94.0 vs 94.2 seed-1
# AnnRet) by pricing per name what the gate handles by deletion; on top of the gate it
# still adds the deltas above. Depth runs ~15% deeper than parkinson on the median
# name, so expect somewhat fewer fills. Rollback: TRIGGER_VOL_EST=parkinson here and
# BT_LIMIT_VOL_EST=parkinson in the backtester.
VOL_EST        = (os.environ.get('TRIGGER_VOL_EST') or 'yz').strip().lower()
_LN2x4         = 4.0 * np.log(2.0)


def yang_zhang_sigma(opens, high, low, closes, lookback=VOL_LOOKBACK):
    """Yang-Zhang volatility: overnight + open-to-close + Rogers-Satchell.

    BYTE-COMPATIBLE with `_vol_estimate(..., 'yz')` in the 5v2 fork: needs lookback+1
    closes for `lookback` overnight gaps, >= 12 usable bars, sample variances (ddof=1),
    k = 0.34 / (1.34 + (n+1)/(n-1)), RS term floored at zero. None when it cannot be
    computed, so the caller can fall back and say so.
    """
    if opens is None or high is None or low is None or closes is None:
        return None
    if len(closes) < lookback + 1 or len(high) < lookback or len(opens) < lookback:
        return None
    c = np.asarray(closes[-(lookback + 1):], dtype=float)
    h = np.asarray(high[-lookback:], dtype=float)
    l = np.asarray(low[-lookback:], dtype=float)
    o = np.asarray(opens[-lookback:], dtype=float)
    ok = (np.isfinite(c[1:]) & np.isfinite(c[:-1]) & np.isfinite(h) & np.isfinite(l)
          & np.isfinite(o) & (l > 0) & (o > 0) & (c[:-1] > 0))
    n = int(ok.sum())
    if n < 12:
        return None
    cp, cc, hh, ll, oo = c[:-1][ok], c[1:][ok], h[ok], l[ok], o[ok]
    gap = np.log(oo / cp)
    oc = np.log(cc / oo)
    rs = np.log(hh / cc) * np.log(hh / oo) + np.log(ll / cc) * np.log(ll / oo)
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    v = gap.var(ddof=1) + k * oc.var(ddof=1) + (1.0 - k) * max(float(rs.mean()), 0.0)
    return float(np.sqrt(max(v, 0.0)))


def parkinson_sigma(high, low, lookback=VOL_LOOKBACK):
    """Parkinson high-low range volatility. None when it cannot be computed.

    Kept BYTE-COMPATIBLE with `_vol_estimate(..., 'parkinson')` in
    experimental/backtesters/5v2__NightlyBackTester_CanBuyV2.py, including the
    >=10 usable bars requirement and the finite/positive mask. The backtest that
    justified this estimator ran that implementation; if the two drift, the live
    trigger stops being the thing that was measured.
    """
    if high is None or low is None or len(high) < lookback:
        return None
    h = np.asarray(high[-lookback:], dtype=float)
    l = np.asarray(low[-lookback:], dtype=float)
    ok = np.isfinite(h) & np.isfinite(l) & (l > 0)
    if ok.sum() < 10:
        return None
    q = np.log(h[ok] / l[ok]) ** 2
    return float(np.sqrt(q.mean() / _LN2x4))


def close_to_close_sigma(closes, lookback=VOL_LOOKBACK):
    """Equal-weighted close-to-close deviation - the pre-2026-08-28 estimator.

    N closes -> N-1 returns. Retained as the fallback for names whose High/Low are
    missing, and as the documented rollback.
    """
    c = np.asarray(closes[-lookback:], dtype=float)
    if len(c) < 2 or not np.all(np.isfinite(c)) or (c <= 0).any():
        return None
    return float(np.std(np.diff(np.log(c))))


def _noop(msg='', level='INFO'):
    pass


def gap_count(opens, closes, pct=None, lookback=GAP_GATE_LOOKBACK):
    """Overnight gaps |Open[i] / Close[i-1] - 1| > pct over the last `lookback` sessions.

    Uses lookback + 1 bars so the newest bar's own open is the last gap, which is what
    the sim's `get(size=21)` does. Returns None with fewer than lookback + 1 bars
    (the sim passes such names through and counts them as nohist). A non-finite or
    zero prior close skips that pair, matching `if _c[i-1] and abs(...)` where a NaN
    close makes the comparison False.
    """
    pct = GAP_GATE_PCT if pct is None else float(pct)
    n = int(lookback) + 1
    if opens is None or closes is None or len(opens) < n or len(closes) < n:
        return None
    o = np.asarray(opens[-n:], dtype=float)
    c = np.asarray(closes[-n:], dtype=float)
    cnt = 0
    for i in range(1, n):
        pc = c[i - 1]
        if not (np.isfinite(pc) and pc > 0 and np.isfinite(o[i])):
            continue
        if abs(100.0 * (o[i] / pc - 1.0)) > pct:
            cnt += 1
    return int(cnt)


def apply_gap_gate(df, log=_noop, n=None, pct=None, pred_dir=PRED_DIR):
    """Drop gap-frequency offenders from the WIDE pool. No-op when the gate is off.

    Always stamps GapCount20 on the returned frame and fills LAST_GAP_CENSUS, so a
    caller can census pool_width / skipped_by_gap / survivors on every run. Names with
    no readable history pass through (as in the sim) and are listed as nohist.
    """
    n = GAP_GATE_N if n is None else int(n)
    pct = GAP_GATE_PCT if pct is None else float(pct)
    census = dict(n=n, pct=pct, lookback=GAP_GATE_LOOKBACK, pool_width=int(len(df)),
                  skipped=[], survivors=int(len(df)), nohist=[], mismatch=[])
    LAST_GAP_CENSUS.clear()
    LAST_GAP_CENSUS.update(census)
    if n <= 0:
        log('gap gate       : OFF (TRIGGER_GAP_GATE_N=0)')
        return df
    counts = []
    for r in df.itertuples():
        sym = r.Symbol
        g = None
        fp = os.path.join(pred_dir, f'{sym}.parquet')
        if os.path.exists(fp):
            try:
                px = pd.read_parquet(fp, columns=['Date', 'Open', 'Close']).sort_values('Date')
                g = gap_count(px['Open'].to_numpy(float), px['Close'].to_numpy(float), pct)
            except Exception as e:
                log(f'  {sym:<7} gap gate: unreadable ({e}); passing through', 'WARN')
        counts.append(g)
        if g is None:
            census['nohist'].append(sym)
        elif g >= n:
            census['skipped'].append(sym)
        # Audit: the nightly stamps the same count with the same function. Disagreement
        # means the pool was built on different bars (or a different pct) than the
        # broker is reading now. Loud, never fatal.
        stamped = getattr(r, 'GapCount20', None)
        if stamped is not None and stamped == stamped and g is not None and int(stamped) != g:
            census['mismatch'].append(f'{sym}:{int(stamped)}!={g}')
    out = df.copy()
    out['GapCount20'] = counts
    out = out[~out['Symbol'].isin(census['skipped'])].reset_index(drop=True)
    census['survivors'] = int(len(out))
    LAST_GAP_CENSUS.update(census)
    for sym in census['skipped']:
        log(f'  {sym:<7} SKIP: gap gate ({n}+ gaps > {pct:g}% in {GAP_GATE_LOOKBACK} bars)', 'WARN')
    log(f'gap gate       : N={n} pct={pct:g}% lookback={GAP_GATE_LOOKBACK} '
        f'pool_width={census["pool_width"]} skipped_by_gap={len(census["skipped"])} '
        f'{census["skipped"]} survivors={census["survivors"]} nohist={len(census["nohist"])}')
    if census['mismatch']:
        log(f'  gap gate: stamped GapCount20 disagrees with recomputed for '
            f'{census["mismatch"]}; pool built on different bars?', 'WARN')
    return out


def load_pool(reach=DEFAULT_REACH, log=_noop, pool_file=None):
    """Top `reach` names by UpProbability from the signal pool.

    Reads the WIDE pool (`Data/0__Signals.parquet`), not the post-rubric narrowed book,
    because the trigger arm's whole design is to rest limits on more names than it
    intends to hold and let the dip choose. Falls back to the narrowed book only when
    the pool is absent, which is a degraded mode worth seeing in the log.

    The gap-frequency gate (apply_gap_gate) runs HERE, on the wide pool and before
    the reach cut, so a skipped name is replaced by the next-ranked survivor the way
    the sim's oversub budget refills. Gating after the cut is a different, worse rule.

    Returns (DataFrame, target_date).
    """
    src = pool_file or (POOL_FILE if os.path.exists(POOL_FILE) else NARROW_FILE)
    if not os.path.exists(src):
        raise FileNotFoundError(f'no signal pool at {src}')
    log(f'pool file      : {src}')
    log(f'  modified     : {datetime.fromtimestamp(os.path.getmtime(src)):%Y-%m-%d %H:%M:%S}')
    df = pd.read_parquet(src)
    log(f'  rows         : {len(df)}   columns: {len(df.columns)}')
    if 'UpProbability' not in df.columns:
        raise ValueError(f'{src} has no UpProbability column')
    pool_width = len(df)
    df = apply_gap_gate(df, log)
    available = len(df)
    df = df.sort_values('UpProbability', ascending=False).head(reach).reset_index(drop=True)
    tgt = pd.to_datetime(df['TargetDate']).dt.date.max() if 'TargetDate' in df else None
    log(f'  TargetDate   : {tgt}')
    log(f'  taking top {len(df)} by UpProbability (reach={reach})')
    # A pool shorter than `reach` does not fail, it silently becomes a different arm:
    # asking for reach 24 and getting 12 is reach 12, and the caller would never know.
    # Measured 2026-08-26, the live pool held 12 rows against a requested 24. Reach is
    # inside the noise band anyway, so this is a census line and not an abort, but it
    # must be visible in the log of every run.
    if available < reach:
        _why = (f' The gap gate removed {pool_width - available} of {pool_width}; '
                f'widen BT_SIGNAL_POOL_SIZE so survivors >= reach.'
                if available < pool_width else '')
        log(f'  REACH SHORTFALL: pool holds {available} rows, reach={reach} requested. '
            f'This run is effectively reach={available}.{_why}', 'WARN')
    return df, tgt


def _load_beta_table(log=_noop):
    if not os.path.exists(BETA_FILE):
        log('beta lookup    : MISSING -- every beta defaults to 1.0', 'WARN')
        return None
    b = pd.read_parquet(BETA_FILE)
    b['Date'] = pd.to_datetime(b['Date'])
    tbl = {t: g.set_index('Date') for t, g in b.groupby('ticker', sort=False)}
    log(f'beta lookup    : {len(tbl)} tickers, '
        f'modified {datetime.fromtimestamp(os.path.getmtime(BETA_FILE)):%Y-%m-%d %H:%M}')
    return tbl


def build_triggers(df, k=DEFAULT_K, log=_noop, beta_exp=BETA_EXP):
    """One row per candidate with its depth and trigger price.

    Rows that cannot be priced carry a non-empty `skip` and a null trigger rather than
    being dropped, so a caller can always census what it lost and why. Silently
    shortening the table is how a pool quietly becomes a different pool.
    """
    beta_tbl = _load_beta_table(log)
    rows = []
    for r in df.itertuples():
        sym = r.Symbol
        fp = os.path.join(PRED_DIR, f'{sym}.parquet')
        if not os.path.exists(fp):
            log(f'  {sym:<7} SKIP: no RFpredictions file', 'WARN')
            rows.append(dict(symbol=sym, skip='no RFpredictions file'))
            continue
        try:
            px = pd.read_parquet(fp, columns=['Date', 'Open', 'High', 'Low', 'Close'])
        except Exception:
            try:
                px = pd.read_parquet(fp, columns=['Date', 'High', 'Low', 'Close'])
            except Exception as e:
                px = None
                _err = e
        if px is None:
            e = _err
            log(f'  {sym:<7} SKIP: unreadable ({e})', 'WARN')
            rows.append(dict(symbol=sym, skip=f'unreadable: {e}'))
            continue
        px['Date'] = pd.to_datetime(px['Date'])
        px = px.sort_values('Date')
        cl = px['Close'].to_numpy(float)[-(VOL_LOOKBACK + 1):]
        if len(cl) < 12:
            log(f'  {sym:<7} SKIP: only {len(cl)} closes', 'WARN')
            rows.append(dict(symbol=sym, skip='short history'))
            continue
        hi = px['High'].to_numpy(float)[-VOL_LOOKBACK:] if 'High' in px.columns else None
        lo = px['Low'].to_numpy(float)[-VOL_LOOKBACK:] if 'Low' in px.columns else None
        op = px['Open'].to_numpy(float)[-VOL_LOOKBACK:] if 'Open' in px.columns else None
        # Estimator chain: the shipped estimator first, then parkinson, then
        # close-to-close, each fallback LOGGED so part of the book can never sit on a
        # different rule silently. cl carries VOL_LOOKBACK+1 closes, which is exactly
        # what yang_zhang_sigma needs for VOL_LOOKBACK overnight gaps.
        vol, vol_src = None, VOL_EST
        if VOL_EST == 'yz':
            vol = yang_zhang_sigma(op, hi, lo, cl)
        if vol is None or vol <= 0:
            vol = parkinson_sigma(hi, lo)
            vol_src = 'parkinson' if VOL_EST == 'parkinson' else 'parkinson-fallback'
            if VOL_EST == 'yz' and vol is not None and vol > 0:
                log(f'  {sym:<7} yang-zhang unavailable (Open missing or thin); '
                    f'falling back to parkinson', 'WARN')
        if vol is None or vol <= 0:
            # Missing or unusable High/Low. Degrade to the old estimator rather than
            # dropping the name, but say so: a silent fallback would put part of the
            # book on a different rule from the rest of it.
            vol = close_to_close_sigma(cl)
            vol_src = 'std20-fallback'
            log(f'  {sym:<7} range estimators unavailable (High/Low missing or flat); '
                f'falling back to close-to-close', 'WARN')
        if vol is None or vol <= 0:
            log(f'  {sym:<7} SKIP: zero volatility', 'WARN')
            rows.append(dict(symbol=sym, skip='zero vol'))
            continue
        beta, beta_src = 1.0, 'default'
        if beta_tbl and sym in beta_tbl:
            try:
                v = beta_tbl[sym]['beta'].asof(px['Date'].iloc[-1])
                if v == v:
                    beta, beta_src = float(v), 'lookup'
            except Exception:
                pass
        raw = k * (beta ** beta_exp) * vol
        depth = min(max(raw, MIN_DEPTH), MAX_DEPTH)
        clamp = 'FLOOR' if raw < MIN_DEPTH else ('CEIL' if raw > MAX_DEPTH else '')
        pc = float(cl[-1])
        rows.append(dict(
            symbol=sym, rank=r.Index + 1, up_prob=round(float(r.UpProbability), 4),
            prior_close=round(pc, 4), prior_close_date=str(px['Date'].iloc[-1].date()),
            beta=round(beta, 3), beta_src=beta_src, vol_20d=round(vol, 5),
            vol_src=vol_src,
            raw_depth_pct=round(100 * raw, 3), depth_pct=round(100 * depth, 3),
            clamped=clamp, trigger=round(pc * (1 - depth), 4), skip=''))
        if clamp:
            log(f'  {sym:<7} depth {100*raw:.2f}% clamped to {100*depth:.2f}% ({clamp})', 'WARN')
    return pd.DataFrame(rows)


def tradable(plan):
    """The subset of a trigger table that can actually be rested, in rank order."""
    if plan is None or plan.empty or 'trigger' not in plan.columns:
        return plan.iloc[0:0] if plan is not None else pd.DataFrame()
    ok = plan[(plan['skip'] == '') & plan['trigger'].notnull()]
    return ok.sort_values('rank').reset_index(drop=True)
