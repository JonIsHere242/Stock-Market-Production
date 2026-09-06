<#
================================================================================
  install_client.ps1  --  Stock-Market execution client bootstrap
================================================================================

  Sets up the thin broker client on a customer machine: creates a Python venv,
  installs the pinned dependencies, scaffolds the working directories, verifies
  every import, and confirms the machine can actually reach IBKR TWS.

  This is the EXECUTION client only. It does not download prices, build
  features, or train a model. It reads the daily book (_Buy_Signals.parquet)
  and routes those orders through IBKR.

  Safe to re-run. It never overwrites an existing book or position ledger.

  USAGE
    .\client\install_client.ps1                  # install, verify against live TWS (7496)
    .\client\install_client.ps1 -Paper           # verify against paper TWS (7497)
    .\client\install_client.ps1 -Port 4001       # IB Gateway
    .\client\install_client.ps1 -Force           # delete and rebuild the venv
    .\client\install_client.ps1 -SkipIbkrCheck   # install only, do not touch TWS
    .\client\install_client.ps1 -Python "C:\Path\to\python.exe"

  Requirements: Python 3.11 or newer, and IBKR TWS or Gateway installed with
  API access enabled (Configure > API > Settings > Enable ActiveX and Socket
  Clients, and add 127.0.0.1 to Trusted IPs).
================================================================================
#>

[CmdletBinding()]
param(
    [switch]$Paper,
    [int]$Port = 0,
    [switch]$Force,
    [switch]$SkipIbkrCheck,
    [string]$Python
)

$ErrorActionPreference = "Stop"

# The client root is one level up from this script (client\install_client.ps1),
# but tolerate the script being copied next to the broker instead.
$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $root "9_SuperFastBroker.py"))) { $root = $PSScriptRoot }
$venvDir = Join-Path $root "client_env"
$venvPy  = Join-Path $venvDir "Scripts\python.exe"
$reqFile = Join-Path $PSScriptRoot "requirements-client.txt"

if ($Port -eq 0) { if ($Paper) { $Port = 7497 } else { $Port = 7496 } }

function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Info($m) { Write-Host "    $m" -ForegroundColor Gray }
function Ok($m)   { Write-Host "    [OK] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "    [!]  $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "`n[X] $m" -ForegroundColor Red; exit 1 }

$bar = "=" * 72
Write-Host $bar -ForegroundColor Cyan
Write-Host "  Stock-Market execution client -- setup" -ForegroundColor Cyan
Write-Host "  $root" -ForegroundColor DarkGray
Write-Host $bar -ForegroundColor Cyan

# ---------------------------------------------------------------------------
#  1. Locate Python
# ---------------------------------------------------------------------------
Step "Locating Python (3.11 or newer)"

function Test-Python($exe, $argList) {
    try {
        $v = & $exe @argList -c "import sys; print(sys.version_info[0], sys.version_info[1], sep='.')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v) { return $v.Trim() }
    } catch { }
    return $null
}

$pyExe = $null; $pyArgs = @()
if ($Python) {
    if (-not (Test-Path $Python)) { Die "Interpreter not found: $Python" }
    $pyExe = $Python
} else {
    $candidates = @(
        @{ exe = "py";     args = @("-3.13") },
        @{ exe = "py";     args = @("-3") },
        @{ exe = "python"; args = @() }
    )
    foreach ($c in $candidates) {
        if (Get-Command $c.exe -ErrorAction SilentlyContinue) {
            $v = Test-Python $c.exe $c.args
            if ($v) { $pyExe = $c.exe; $pyArgs = $c.args; break }
        }
    }
}
if (-not $pyExe) { Die "No working Python found. Install Python 3.11+ from python.org (tick 'Add python.exe to PATH'), then re-run." }

$pyVer = Test-Python $pyExe $pyArgs
$maj, $min = $pyVer.Split('.')
if ([int]$maj -lt 3 -or ([int]$maj -eq 3 -and [int]$min -lt 11)) {
    Die "Python $pyVer is too old. Need 3.11 or newer."
}
Ok "Python $pyVer ($pyExe $($pyArgs -join ' '))"

# ---------------------------------------------------------------------------
#  2. Virtual environment
# ---------------------------------------------------------------------------
Step "Virtual environment (client_env)"
if ($Force -and (Test-Path $venvDir)) {
    Warn "-Force: removing existing venv"
    Remove-Item -Recurse -Force $venvDir
}
if (Test-Path $venvPy) {
    Info "Reusing existing venv"
} else {
    Info "Creating venv ..."
    & $pyExe @pyArgs -m venv $venvDir
    if (-not (Test-Path $venvPy)) { Die "venv creation failed." }
}
Ok "venv ready at $venvDir"

# ---------------------------------------------------------------------------
#  3. Dependencies
# ---------------------------------------------------------------------------
Step "Installing pinned dependencies (about 312 MB, 2 to 4 minutes)"
if (-not (Test-Path $reqFile)) { Die "requirements-client.txt not found next to this script." }
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -r $reqFile
if ($LASTEXITCODE -ne 0) { Die "Dependency install failed. Scroll up for the pip error." }
Ok "dependencies installed"

# ---------------------------------------------------------------------------
#  4. Working directories
# ---------------------------------------------------------------------------
Step "Scaffolding working directories"
$made = 0
foreach ($d in @("Data", "Data\logging", "Data\_backups")) {
    $p = Join-Path $root $d
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null; $made++ }
}
Ok "directories ready ($made created)"

# ---------------------------------------------------------------------------
#  5. Required files present
# ---------------------------------------------------------------------------
Step "Checking client files"
$missing = @()
foreach ($f in @("9_SuperFastBroker.py", "Util.py", "auxiliary\bracket_config.py")) {
    if (-not (Test-Path (Join-Path $root $f))) { $missing += $f }
}
if ($missing.Count -gt 0) { Die "Missing client files: $($missing -join ', ')" }
Ok "all client files present"

$bookPath = Join-Path $root "_Buy_Signals.parquet"
if (Test-Path $bookPath) { Ok "_Buy_Signals.parquet present" }
else { Warn "_Buy_Signals.parquet not here yet. You receive today's book each morning before the 10:00 ET entry." }

# ---------------------------------------------------------------------------
#  6. Verify imports
# ---------------------------------------------------------------------------
Step "Verifying the install"
$verify = @'
import os
import sys

# This script is executed from TEMP, so Python puts TEMP on sys.path, not the
# client root. Put the working directory first so Util and auxiliary resolve.
sys.path.insert(0, os.getcwd())

bad = []
for m in ["numpy","pandas","pyarrow","ib_insync","pandas_market_calendars",
          "exchange_calendars","backtrader","plotly","pytz","nest_asyncio"]:
    try:
        __import__(m)
    except Exception as e:
        bad.append((m, str(e).splitlines()[0]))
for m, e in bad:
    print("  MISSING  %s: %s" % (m, e))
if bad:
    sys.exit(1)
import numpy, pandas
print("  numpy %s | pandas %s" % (numpy.__version__, pandas.__version__))
try:
    from Util import get_logger, PositionSizer
    from auxiliary import bracket_config as B
except Exception as e:
    print("  IMPORT FAILED: %s: %s" % (type(e).__name__, e))
    sys.exit(1)
print("  bracket: stop %.1f pct | target %.1f pct | max hold %d days"
      % (B.HARD_STOP_PCT, B.TAKE_PROFIT_PCT, B.MAX_HOLD_DAYS))
print("VERIFY_OK")
'@
$vf = Join-Path $env:TEMP "smclient_verify.py"
Set-Content -Path $vf -Value $verify -Encoding utf8
Push-Location $root
& $venvPy $vf
$vExit = $LASTEXITCODE
Pop-Location
Remove-Item $vf -ErrorAction SilentlyContinue
if ($vExit -ne 0) { Die "Verification failed. The client will not run until the errors above are fixed." }
Ok "all imports clean"

# ---------------------------------------------------------------------------
#  7. IBKR connectivity
# ---------------------------------------------------------------------------
if ($SkipIbkrCheck) {
    Step "IBKR check SKIPPED (-SkipIbkrCheck)"
} else {
    Step "Checking IBKR on 127.0.0.1:$Port"
    $tcp = $null
    try { $tcp = Test-NetConnection -ComputerName "127.0.0.1" -Port $Port -WarningAction SilentlyContinue } catch { }
    if (-not $tcp -or -not $tcp.TcpTestSucceeded) {
        Warn "Nothing listening on port $Port."
        Warn "Start TWS (or IB Gateway) and log in, then in TWS:"
        Warn "  Configure > API > Settings > tick 'Enable ActiveX and Socket Clients'"
        Warn "  confirm the Socket port is $Port, and add 127.0.0.1 to Trusted IPs"
        Warn "Then re-run:  .\client\install_client.ps1 -Port $Port"
    } else {
        Ok "port $Port is open"
        $probeLines = @(
            'import sys',
            'import ib_insync as ibi',
            'PORT = ' + $Port,
            'ib = ibi.IB()',
            'try:',
            '    ib.connect("127.0.0.1", PORT, clientId=77, timeout=15)',
            'except Exception as e:',
            '    print("  CONNECT FAILED: %s" % e)',
            '    sys.exit(1)',
            'accts = ib.managedAccounts()',
            'print("  connected. accounts: %s" % (", ".join(accts) if accts else "none reported"))',
            'nav = None',
            'for tag in ib.accountValues():',
            '    if tag.tag == "NetLiquidation" and tag.currency == "USD":',
            '        nav = float(tag.value)',
            '        break',
            'if nav is not None:',
            '    print("  net liquidation: %s USD" % format(nav, ",.0f"))',
            'ib.disconnect()',
            'print("IBKR_OK")'
        )
        $pf = Join-Path $env:TEMP "smclient_ibkr.py"
        Set-Content -Path $pf -Value $probeLines -Encoding utf8
        & $venvPy $pf
        $iExit = $LASTEXITCODE
        Remove-Item $pf -ErrorAction SilentlyContinue
        if ($iExit -eq 0) { Ok "IBKR API handshake succeeded" }
        else { Warn "Port is open but the API handshake failed. Check 'Enable ActiveX and Socket Clients' and Trusted IPs in TWS." }
    }
}

# ---------------------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------------------
Write-Host "`n$bar" -ForegroundColor Cyan
Write-Host "  CLIENT INSTALL COMPLETE" -ForegroundColor Green
Write-Host $bar -ForegroundColor Cyan
Write-Host @"

  Daily routine
  -------------
  1. Drop today's _Buy_Signals.parquet into:
       $root

  2. Check the book BEFORE trading (refuses a stale or empty book):
       .\client_env\Scripts\python.exe client\check_book.py

  3. Start TWS and log in. Leave it running.

  4. Launch the broker (it waits until 10:00 ET on its own):
       .\client_env\Scripts\python.exe 9_SuperFastBroker.py --exp-stop-limit --exp-moc-exit --exp-entry-escalate-min 3

  Paper trading first is strongly recommended: add --port 7497 to step 4 and
  point TWS at a paper login.

"@ -ForegroundColor White
