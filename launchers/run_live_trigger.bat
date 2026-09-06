@echo off
REM ============================================================================
REM  LIVE trigger-entry broker. THIS PLACES REAL ORDERS ON THE LIVE ACCOUNT.
REM ----------------------------------------------------------------------------
REM  Machine is ET-2, so 08:00 local == 10:00 ET.
REM
REM  PRE-FLIGHT. Do these in order, today, before running this file:
REM    1. python 4__Predictor.py ... --infer_tail 500        (predictions)
REM    2. python 5__NightlyBackTester.py --force --selrule low_atr   (24-wide pool)
REM    3. python signal_filter.py                            (MechExclude)
REM    4. python rubric_veto.py --show                       (or record <=4 vetoes)
REM    5. run_dry_trigger.bat                                (places nothing)
REM  A pool dated before today ABORTS: STALE_ABORT_TDAYS is 1 and the branch is
REM  `stale_tdays < STALE_ABORT_TDAYS`, so ONE trading day of staleness is already
REM  the abort band. There is no warn-and-proceed window. Regenerate the pool.
REM
REM  ENTRY TIME IS 10:00 ET AND SHOULD STAY THERE. Arming earlier is significantly
REM  worse: the 09:45 cell measured -0.438pp (t -1.96) and is the only robust cell
REM  in the whole arming grid. 10:15 looked better (+0.279) but flips significance
REM  depending on how unresolved brackets are treated, so it is not a measured
REM  number. Do not move this clock without re-running that grid.
REM
REM  The broker waits for its own entry time, so launching early is safe: it holds
REM  the TWS connection and fires at 10:00 ET. It then RUNS UNTIL 15:45 ET
REM  monitoring every armed name. Do not close the window; a taskkill drops the
REM  watch (the breadcrumb at Data\trigger_monitor.json records where it got to).
REM
REM  WHAT THIS MODE IS FOR. Not return. Across ten shuffle seeds the trigger arm
REM  beat the shipped control by +0.037pp/trade at t 0.53 (5 of 10), which is
REM  nothing. What it buys is drawdown: 17.83-23.92% against the control's 33.25%,
REM  zero overlap across every seed, on roughly three quarters of the capital.
REM
REM  ARM MODE: virtual. NOTHING is sent at 10:00. The trigger table is held in
REM  memory, streaming quotes are watched, and ONE real limit order goes out the
REM  moment a name reaches its own trigger. At most free_slots orders ever exist
REM  on the book at once, and the order is a LIMIT AT THE TRIGGER, so a touch can
REM  never fill worse than the modelled entry price.
REM
REM  WHAT VIRTUAL COSTS, and it is not zero. The measured arm rested its bids and
REM  was filled AT the trigger, which EARNS the half-spread. Virtual reacts instead
REM  of resting, so it pays (a) up to --trigger-poll seconds of latency, in which a
REM  dip can touch and rebound unseen, and (b) the spread. Neither is in the
REM  +0.7730%/trade figure. The watcher now reports mean bps vs the armed trigger
REM  at session end, which is the first direct measurement of that gap - read it.
REM  Pass --trigger-arm resting to run the arm that was actually measured,
REM  accepting ~20 simultaneous live orders.
REM
REM  --trigger-max-spread-bps is deliberately NOT set, which matches the arm that
REM  was actually measured. The known hazard: on 2026-08-25 the paper shadow armed
REM  DRUG at a 214.3 bps median spread against an edge of about 43 bps, and nothing
REM  in the arming step looks at spread. At reach 24 you reach further down the
REM  pool, so expect wider names than at reach 12. Add e.g.
REM  --trigger-max-spread-bps 80 to screen that, accepting it is a change from the
REM  measured arm.
REM
REM  THE EXIT FLAGS ARE NOT OPTIONAL. --exp-stop-limit --exp-moc-exit are the
REM  validated exit-order stack (STP LMT floor + stranded-stop sweep, MOC age
REM  exit) that the old production launch command has always passed. An earlier
REM  version of this file omitted both, which would have silently dropped them.
REM
REM  ROLLBACK: pass --no-trigger-entry. Trigger entry is the DEFAULT since
REM  2026-08-28; that flag restores the marketable-Adaptive path byte-for-byte,
REM  which reads the NARROWED _Buy_Signals.parquet instead of the pool.
REM ============================================================================
cd /d C:\Users\Masam\Desktop\Stock-Market
call stock_env\Scripts\activate.bat
python 9_SuperFastBroker.py ^
  --trigger-entry --trigger-arm virtual --trigger-poll 5 ^
  --trigger-reach 12 --trigger-k 1.5 --trigger-cutoff 15:45 ^
  --trigger-max-spread-bps 80 ^
  --trigger-status-min 15 ^
  --exp-stop-limit --exp-moc-exit ^
  --port 7496 >> LIVE_TRIGGER_launch.log 2>&1
