@echo off
REM ============================================================================
REM  DRY structural check. PLACES NOTHING, on any account, ever.
REM ----------------------------------------------------------------------------
REM  --dry-run gates EVERY order path: entries, exits, stop repairs, repegs and
REM  scale-outs. It connects to TWS, loads the pool, applies the mechanical filter
REM  and the rubric veto, prices every trigger, arms the monitor and reports what
REM  WOULD fire. Nothing is transmitted.
REM
REM  READ THIS IF YOU ARE USED TO THE OLD FLAGS. Neither --no-exits nor
REM  --dry-run-exits is safe for a structural check:
REM    --no-exits       gates the max-hold/TP machinery ONLY. Entries run.
REM    --dry-run-exits  says so outright in its own help: "Entries still run
REM                     normally."
REM  The handoff that shipped this feature called
REM    9_SuperFastBroker.py --trigger-entry --skip-wait --no-exits --port 7496
REM  a check that "places nothing, ever". It places real entry orders on a live
REM  account the moment the pool is fresh and a slot is free, which is exactly the
REM  state its own regeneration step leaves you in. Use this file instead.
REM
REM  --skip-wait runs immediately rather than waiting for 10:00 ET, so this is
REM  useful at any hour. Quotes outside RTH are thin, so distances-to-trigger in
REM  the watch table will be noisy; the point is that the plumbing works, not that
REM  the prices are tradeable.
REM
REM  Point at paper TWS with --port 7497 if it is running. Belt and braces: the
REM  dry-run gate is what makes this safe, not the port.
REM ============================================================================
cd /d C:\Users\Masam\Desktop\Stock-Market
call stock_env\Scripts\activate.bat
python 9_SuperFastBroker.py ^
  --dry-run --skip-wait ^
  --trigger-entry --trigger-arm virtual --trigger-poll 5 ^
  --trigger-reach 12 --trigger-k 1.5 --trigger-cutoff 15:45 ^
  --trigger-max-spread-bps 80 ^
  --trigger-status-min 1 ^
  --exp-stop-limit --exp-moc-exit ^
  --port 7496
