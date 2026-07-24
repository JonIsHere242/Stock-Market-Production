# Trading System Runner - PRODUCTION (hardened 2026-06-12)
# Previous version backed up to _backups\trading_system_v1_20260612.ps1
# Manual modes: .\trading_system.ps1 -Mode morning   (or evening)

param(
    [ValidateSet("auto", "morning", "evening")]
    [string]$Mode = "auto"
)

$Host.UI.RawUI.WindowTitle = "Trading System - Production Runner"

$basePath       = "C:\Users\Masam\Desktop\Stock-Market"
$pythonExe      = "$basePath\stock_env\Scripts\python.exe"
$logDir         = "$basePath\Data\logging"
$logFile        = "$logDir\__trading_system.log"
$ibkrHost       = "127.0.0.1"
$ibkrPort       = 7496              # TWS live (verified in 9_SuperFastBroker.py and 2__PriceDownloader.py)
$ibkrLauncher   = ""                # optional: path to TWS/Gateway shortcut for auto-start attempt
$brokerLaunchET = "09:57"           # hand off to broker just before its internal 10:00 ET gate
$brokerCutoffET = "10:30"           # stop retrying after this
$lockFile       = "$logDir\.trading_system.lock"
$alertFile      = "$logDir\BROKER_ALERT.txt"

if (!(Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

function Write-Log($message, $color = "White") {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $message"
    Write-Host $line -ForegroundColor $color
    try { Add-Content -Path $logFile -Value $line } catch {}
}

$script:alertPopupsShown = @{}
function Send-Alert($message) {
    Write-Log "ALERT: $message" "Red"
    try { Add-Content -Path $alertFile -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $message" } catch {}
    try { 1..3 | ForEach-Object { [console]::Beep(1000, 300) } } catch {}
    # Non-blocking popup: beeps + the alert file went unnoticed for 4 straight days
    # (07-06..07-10 stale-book incident). One popup per distinct alert type per run
    # so the 20s retry loops don't spawn dozens of windows.
    $key = $message.Substring(0, [Math]::Min(40, $message.Length))
    if (-not $script:alertPopupsShown.ContainsKey($key)) {
        $script:alertPopupsShown[$key] = $true
        try {
            $safe = $message -replace "[`"']", ""
            $inner = "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('$safe', 'TRADING SYSTEM ALERT', 'OK', 'Warning') | Out-Null"
            $b64 = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($inner))
            Start-Process powershell -ArgumentList "-NoProfile", "-WindowStyle", "Hidden", "-EncodedCommand", $b64 | Out-Null
        } catch {}
    }
}

function Get-ETNow {
    [System.TimeZoneInfo]::ConvertTime([datetime]::Now,
        [System.TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time"))
}

function Get-ETToday($hhmm) {
    $et = Get-ETNow
    [datetime]::ParseExact("$($et.ToString('yyyy-MM-dd')) $hhmm", "yyyy-MM-dd HH:mm", $null)
}

function Test-IBKR {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $async = $client.BeginConnect($ibkrHost, $ibkrPort, $null, $null)
        $ok = $async.AsyncWaitHandle.WaitOne(1500, $false)
        if ($ok -and $client.Connected) { $client.Close(); return $true }
        $client.Close(); return $false
    } catch { return $false }
}

function Run-Stage($name, $file, $argString) {
    $scriptPath = Join-Path $basePath $file
    if (!(Test-Path $scriptPath)) {
        Write-Log "ERROR: script not found: $file" "Red"
        return @{ ok = $false; out = ""; code = -1 }
    }
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $transcript = Join-Path $logDir ("stage_" + ($name -replace "\s", "") + "_$stamp.log")
    Write-Log "Starting $name  ($file $argString)" "Yellow"
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    Push-Location $basePath
    try {
        if ($argString) {
            & $pythonExe $scriptPath ($argString -split " ") 2>&1 | Tee-Object -FilePath $transcript | Out-Host
        } else {
            & $pythonExe $scriptPath 2>&1 | Tee-Object -FilePath $transcript | Out-Host
        }
        $code = $LASTEXITCODE
    } finally { Pop-Location }
    $timer.Stop()
    $out = ""
    if (Test-Path $transcript) { $out = Get-Content $transcript -Raw }
    $secs = [math]::Round($timer.Elapsed.TotalSeconds, 1)
    if ($code -eq 0) {
        Write-Log "$name finished exit=0 (${secs}s)" "Green"
    } else {
        Write-Log "$name FAILED exit=$code (${secs}s)  transcript: $(Split-Path $transcript -Leaf)" "Red"
    }
    return @{ ok = ($code -eq 0); out = $out; code = $code }
}

# Broker outcome classification. SPY_ABORT is the EDA risk gate working as
# designed (Sharpe -6.50 on weak days) - it is a SUCCESS state, never retried.
function Get-BrokerOutcome($result) {
    $o = $result.out
    if ($o -match "SPY ABORT|Market conditions gate FAILED") { return "SPY_ABORT" }
    if ($o -match "All orders transmitted") { return "TRANSMITTED" }
    # Broker's own stale-book fail-safe fired. Checked BEFORE the exit-code/no-orders
    # branches: 07-08 and 07-09 this banner was misclassified as RAN_NO_ORDERS and the
    # log said "no entries qualified" while the real story was a dead nightly pipeline.
    if ($o -match "STALE SIGNAL BOOK|REFUSING TO TRADE") { return "STALE_BOOK" }
    if (-not $result.ok) { return "CONNECT_FAIL" }
    if ($o -match "Traceback|Connection refused|ConnectionRefused|TimeoutError|API connection failed") { return "CONNECT_FAIL" }
    if ($o -match "Connecting to IB" -and $o -notmatch "Connected\.") { return "CONNECT_FAIL" }
    if ($o -match "Snapshotting Account") { return "RAN_NO_ORDERS" }
    return "UNKNOWN"
}

# ---------------------------------------------------------------------------
# Single-instance lock (shared parquet writes corrupt under concurrent runs)
# ---------------------------------------------------------------------------
if (Test-Path $lockFile) {
    $age = (Get-Date) - (Get-Item $lockFile).LastWriteTime
    if ($age.TotalHours -lt 2) {
        Write-Log "Another run holds the lock ($([int]$age.TotalMinutes)m old). Exiting." "Red"
        exit 1
    }
    Write-Log "Stale lock ($([int]$age.TotalHours)h old) - removing." "Yellow"
    Remove-Item $lockFile -Force
}
Set-Content -Path $lockFile -Value $PID

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if ($Mode -eq "auto") {
    # Wide windows, not exact-hour matches: with StartWhenAvailable the scheduler fires
    # missed triggers whenever the machine comes back, and the old `-eq 17` resolved a
    # catch-up start at 18:05 to "do nothing" — one silent way the nightly pipeline
    # never ran (07-06..07-10 stale-book incident). Morning 05:00-11:59; evening
    # 15:00-03:59 (a small-hours catch-up run still finishes ~70 min later, clear of
    # the 07:28 morning task and its lock).
    $hour = (Get-Date).Hour
    if ($hour -ge 5 -and $hour -le 11) { $Mode = "morning" }
    elseif ($hour -ge 15 -or $hour -le 3) { $Mode = "evening" }
}

Clear-Host
Write-Log "==========================================" "Cyan"
Write-Log "TRADING SYSTEM PRODUCTION RUN  mode=$Mode" "Cyan"
Write-Log "==========================================" "Cyan"

try {
    if ($Mode -eq "morning") {
        Write-Log "=== MORNING BROKER MODE ===" "Cyan"

        # Market closed on weekends: skip instead of funneling Monday's book early and
        # tripping the freshness gate's alert (the funnel/broker used to run Saturdays
        # for nothing - e.g. 06-27).
        $etDay = (Get-ETNow).DayOfWeek
        if ($etDay -eq 'Saturday' -or $etDay -eq 'Sunday') {
            Write-Log "Weekend ($etDay) - market closed, nothing to do this morning." "Yellow"
            exit 0
        }

        # Catch-up starts: past the broker cutoff there is nothing useful or safe
        # left to do this morning — do not funnel midday, do not launch the broker.
        if ((Get-ETNow) -ge (Get-ETToday $brokerCutoffET)) {
            Send-Alert "Morning run started after the $brokerCutoffET ET cutoff (catch-up start?). No trades today."
            exit 1
        }

        # Pre-flight at 07:28 gives ~30 min of alerts to fix a dead TWS manually
        if (Test-IBKR) {
            Write-Log "IBKR pre-flight OK: port $ibkrPort responding." "Green"
        } else {
            Send-Alert "IBKR NOT responding on port $ibkrPort at task start. Start TWS/Gateway now."
            if ($ibkrLauncher -and (Test-Path $ibkrLauncher)) {
                Write-Log "Attempting IBKR launch: $ibkrLauncher (login may still be required)" "Yellow"
                Start-Process $ibkrLauncher
            }
        }

        # Funnel: idempotent, no IBKR needed, self-waits to 09:35 ET internally
        $funnel = Run-Stage "Signal Funnel" "7__MacroFilter.py" ""
        if ($funnel.out -notmatch "FUNNEL COMPLETE") {
            Send-Alert "Funnel did not report FUNNEL COMPLETE. Inspect transcript before trusting the book."
        }

        # HARD GATE (2026-07-10): the broker only ever gets a book dated for TODAY's
        # session. On 07-07 the funnel's stale-pool refusal was alert-only, the handoff
        # went ahead, and the broker traded the 07-06 book. A non-fresh book now means
        # NO broker launch, full stop. (Hand-vetted books pass too: they are stamped
        # with today's TargetDate.)
        $bookCheckPy = @"
import datetime, pandas as pd
from zoneinfo import ZoneInfo
try:
    df = pd.read_parquet(r'$basePath\_Buy_Signals.parquet')
    col = next((c for c in ('TargetDate','SignalDate','CreatedDate','LastUpdated') if c in df.columns), None)
    book = pd.to_datetime(df[col], errors='coerce').max().date() if col is not None else None
    today = datetime.datetime.now(ZoneInfo('America/New_York')).date()
    print('FRESH' if book == today else 'STALE book=%s today=%s' % (book, today))
except Exception as e:
    print('ERROR %s' % e)
"@
        $bookCheck = (& $pythonExe -c $bookCheckPy) -join " "
        if ($bookCheck -match "^FRESH") {
            Write-Log "Book freshness OK: _Buy_Signals.parquet targets today's session." "Green"
        } else {
            Send-Alert "SIGNAL BOOK NOT FRESH ($bookCheck). The nightly pipeline likely did not run. SKIPPING the broker - NO ORDERS today. Regenerate after the close: .\trading_system.ps1 -Mode evening"
            exit 1
        }

        $launch = Get-ETToday $brokerLaunchET
        $cutoff = Get-ETToday $brokerCutoffET
        Write-Log "Broker window: launch $brokerLaunchET ET, hard cutoff $brokerCutoffET ET." "White"

        # Hold until launch, polling IBKR so a dead session is caught BEFORE 10:00
        while ((Get-ETNow) -lt $launch) {
            if (-not (Test-IBKR)) {
                Send-Alert "IBKR down while waiting ($((Get-ETNow).ToString('HH:mm')) ET). Fix before $brokerLaunchET ET."
            }
            Start-Sleep -Seconds 30
        }

        # Past launch: require the port before handing money to the broker
        while (-not (Test-IBKR)) {
            if ((Get-ETNow) -ge $cutoff) {
                Send-Alert "Cutoff $brokerCutoffET ET reached and IBKR never came up. NO TRADES TODAY."
                exit 1
            }
            Send-Alert "Past launch time, IBKR still down. Retrying in 20s."
            Start-Sleep -Seconds 20
        }

        # Broker: retry ONLY on clear connection failures. Never after orders went
        # out, and never after a SPY abort - re-running past the gate is the exact
        # manual override the EDA prices at Sharpe -6.50.
        $attempt = 0
        do {
            $attempt++
            Write-Log "Broker attempt $attempt at $((Get-ETNow).ToString('HH:mm:ss')) ET" "Cyan"
            $broker = Run-Stage "Daily Broker" "9_SuperFastBroker.py" ""
            $outcome = Get-BrokerOutcome $broker
            switch ($outcome) {
                "TRANSMITTED" {
                    $syms = ([regex]::Matches($broker.out, "\[(\w+)\] Staging") |
                             ForEach-Object { $_.Groups[1].Value }) -join ", "
                    Write-Log "ORDERS LIVE: $syms" "Green"
                }
                "SPY_ABORT" {
                    Write-Log "SPY abort gate fired - intentional no-trade day. This is the system working." "Yellow"
                }
                "RAN_NO_ORDERS" {
                    Write-Log "Broker ran clean - no entries qualified today (gaps/spreads)." "Yellow"
                }
                "STALE_BOOK" {
                    Send-Alert "Broker REFUSED a stale signal book - the nightly pipeline did not produce fresh signals. NO ORDERS placed. Regenerate after the close: .\trading_system.ps1 -Mode evening"
                }
                "CONNECT_FAIL" {
                    if ((Get-ETNow) -lt $cutoff) {
                        Send-Alert "Broker connection failure on attempt $attempt. Retrying in 20s."
                        Start-Sleep -Seconds 20
                    }
                }
                default {
                    Send-Alert "Broker outcome UNKNOWN (output unrecognized). NOT retrying - check transcript manually."
                }
            }
        } while ($outcome -eq "CONNECT_FAIL" -and (Get-ETNow) -lt $cutoff)

        if ($outcome -eq "CONNECT_FAIL") {
            Send-Alert "Cutoff reached - broker never completed a run. NO TRADES TODAY."
        }
    }
    elseif ($Mode -eq "evening") {
        Write-Log "=== EVENING DATA PROCESSING MODE ===" "Cyan"

        # Partial-bar guard: a StartWhenAvailable catch-up start during regular trading
        # hours would pull an in-progress daily bar and contaminate features/predictions.
        # Exit instead — the regular 17:00 trigger fires again within 24h. (Normal 17:00
        # local starts are after the 16:00 ET close and never hit this.)
        $etNow = Get-ETNow
        if ($etNow.DayOfWeek -ne 'Saturday' -and $etNow.DayOfWeek -ne 'Sunday' -and
            $etNow -ge (Get-ETToday "09:30") -and $etNow -lt (Get-ETToday "16:05")) {
            Send-Alert "Evening pipeline start landed INSIDE market hours ($($etNow.ToString('HH:mm')) ET). Daily bars are not final - skipping this run; the next 17:00 trigger will handle it."
            exit 1
        }

        # Stage 2 pulls prices through IBKR (port 7496) - pre-flight it too.
        # Wait up to 30 min with alerts; then proceed and let output checks catch it.
        $waited = 0
        while (-not (Test-IBKR) -and $waited -lt 1800) {
            Send-Alert "IBKR not responding - price refresh (stage 2) will fail. Retrying for $([int]((1800 - $waited) / 60)) more min."
            Start-Sleep -Seconds 120
            $waited += 120
        }
        if (Test-IBKR) { Write-Log "IBKR pre-flight OK: port $ibkrPort responding." "Green" }

        $stages = @(
            @{ Name = "Ticker Downloader";  File = "1__TickerDownloader.py";  Args = "--ImmediateDownload" },
            @{ Name = "Price Downloader";   File = "2__PriceDownloader.py";   Args = "--RefreshMode" },
            # Non-OHLCV data panels (SEC fundamentals + Form 4 insider + sector map) the feature
            # framework & macro filter consume. --refresh-all re-pulls every raw SEC source first
            # (fetchers self-skip files <20h old). Runs AFTER prices (needs the PriceData universe)
            # and BEFORE the feature framework (which reads Data/Fundamentals + Data/Insider).
            @{ Name = "Data Panels";        File = "build_data_panels.py";    Args = "all --refresh-all" },
            # --exclude: 4 graph/spectral blocks (vvg/vaq_vg/rvg_wl/volume_spectral_splatter)
            # whose 29 cols the ship predictor drops anyway (--drop_feature_patterns) -- they
            # were ~16% of framework compute for ZERO model effect (benchmark 2026-06-19, no
            # dependents). Computing-then-dropping was pure waste.
            @{ Name = "Feature Framework";  File = "3__FeatureFramework.py";  Args = "--all --exclude vvg vaq_vg rvg_wl volume_spectral_splatter" },
            @{ Name = "Predictor";          File = "4__Predictor.py";         Args = "--predict_only --input_dir Data/ProcessedData_v2 --target_column percent_change_close --drop_features_exact percent_change_close,VIX_Close --drop_feature_patterns tda_embed,mp3_,vvg_,vaq_,rvg_wl,volume_spectral_splatter,vg_ --model_dir Data/_ship_v2/model" },
            # SPY-200EMA adaptive pred-space factor neutralization (beta/atr/log-dollar-vol,
            # per-day, zero fitted params): dose 0.15 above the SPY 200d EMA, 0.30 below,
            # then swap into Data/RFpredictions (raw kept at _raw).
            # Validated 07-12: live window 80.2% vs 52.2% for fixed 0.30 (2 seeds); fixed-0.30
            # history: pinned 4-seed +51pp (07-01), live-window 4-seed +19-24pp (07-03).
            # FAIL-SAFE: stale/missing SPY flag degrades that day to dose_below (=0.30, the old
            # default); on gate failure it exits non-zero and leaves preds RAW.
            @{ Name = "Neutralize Preds";   File = "4.5__NeutralizePreds.py"; Args = "--mode spy200v2 --dose_above 0.15 --dose_below 0.30" },
            @{ Name = "Nightly BackTester"; File = "5__NightlyBackTester.py"; Args = "--force" }
        )
        # Pin the backtester tie-break seed: unseeded runs swing +-20-40pp on identical preds
        # (project_backtest_determinism_2026_06_29) and make nightly reports incomparable.
        $env:BT_SAMPLE_SEED = "42"
        $okCount = 0
        for ($i = 0; $i -lt $stages.Count; $i++) {
            $s = $stages[$i]
            $r = Run-Stage $s.Name $s.File $s.Args
            $stageOk = $r.ok
            # 2__PriceDownloader swallows exceptions and exits 0 - verify by output
            if ($s.File -eq "2__PriceDownloader.py" -and $r.out -match "No tickers were successfully processed") {
                $stageOk = $false
                Send-Alert "Price Downloader exited 0 but downloaded NOTHING. Downstream stages will run on stale prices."
            }
            if ($stageOk) { $okCount++ }
            else { Send-Alert "$($s.Name) failed (exit=$($r.code)). Continuing to next stage." }
            if ($i -lt $stages.Count - 1) {
                Write-Log "Waiting 20s (RAM cleanup)..." "Gray"
                Start-Sleep -Seconds 20
            }
        }
        $col = "Red"; if ($okCount -eq $stages.Count) { $col = "Green" }
        Write-Log "Evening pipeline: $okCount/$($stages.Count) stages OK." $col

        # End-to-end freshness proof: the whole point of this mode is a 12-pool dated
        # for the NEXT session. Verify it, don't assume it — 07-06..07-10 the pipeline
        # simply never ran and nothing noticed until the broker refused to trade.
        $poolCheckPy = @"
import datetime, pandas as pd
from zoneinfo import ZoneInfo
try:
    df = pd.read_parquet(r'$basePath\Data\0__signals.parquet')
    pool = pd.to_datetime(df['TargetDate'], errors='coerce').max().date()
    today = datetime.datetime.now(ZoneInfo('America/New_York')).date()
    print('FRESH pool=%s' % pool if pool >= today else 'STALE pool=%s today=%s' % (pool, today))
except Exception as e:
    print('ERROR %s' % e)
"@
        $poolCheck = (& $pythonExe -c $poolCheckPy) -join " "
        if ($poolCheck -match "^FRESH") {
            Write-Log "Signal pool verified fresh: $poolCheck" "Green"
        } else {
            Send-Alert "EVENING PIPELINE ENDED WITHOUT A FRESH SIGNAL POOL ($poolCheck). Tomorrow morning will NOT trade unless this is fixed tonight."
        }
    }
    else {
        Write-Log "Outside scheduled hours (7 or 17) and no -Mode given. Nothing to run." "Yellow"
        Write-Log "Use: .\trading_system.ps1 -Mode morning   (or evening)" "Yellow"
    }
}
finally {
    Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
    Write-Log "Runner finished." "Cyan"
}
