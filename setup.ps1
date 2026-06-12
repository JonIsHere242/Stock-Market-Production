<#
================================================================================
  setup.ps1  --  one-shot bootstrap for the Stock-Market trading pipeline
================================================================================

  Creates the Python virtual environment (stock_env), installs every dependency
  the pipeline needs (auto-detecting CPU vs CUDA for torch), scaffolds the Data/
  directory tree, drops API-key templates, protects your secrets with a
  .gitignore, and verifies the install can import the key packages.

  Safe to re-run. It never overwrites real API-key files or existing data.

  USAGE
    .\setup.ps1                 # full setup, auto-detect GPU
    .\setup.ps1 -Cpu            # force CPU-only torch
    .\setup.ps1 -Gpu            # force CUDA (cu118) torch
    .\setup.ps1 -Force          # delete & recreate the venv from scratch
    .\setup.ps1 -SkipTorch      # skip torch (sentiment path won't run)
    .\setup.ps1 -SkipBacktrader # skip the git-based backtrader installs
    .\setup.ps1 -ColdStart      # after setup, run the WHOLE pipeline end to end
                                #   (stages 1->5, can take hours) then map it
    .\setup.ps1 -Python "C:\Path\to\python.exe"   # use a specific interpreter

  Every run (with or without -ColdStart) regenerates PIPELINE_MAP.md -- a live
  map of all stages plus the actual data each one has produced.

  Requirements: Python >= 3.11 on PATH (or pass -Python), and git on PATH
  unless you pass -SkipBacktrader.
================================================================================
#>

[CmdletBinding()]
param(
    [switch]$Gpu,
    [switch]$Cpu,
    [switch]$Force,
    [switch]$SkipTorch,
    [switch]$SkipBacktrader,
    [switch]$ColdStart,
    [string]$Python
)

$ErrorActionPreference = "Stop"
$root      = $PSScriptRoot
$venvDir   = Join-Path $root "stock_env"
$venvPy    = Join-Path $venvDir "Scripts\python.exe"
$reqFile   = Join-Path $root "requirements.txt"

# --- pinned git deps (match the known-good environment) ----------------------
$gitDeps = @(
    "git+https://github.com/mementum/backtrader.git@b853d7c90b6721476eb5a5ea3135224e33db1f14",
    "git+https://github.com/ultra1971/backtrader_ib_insync@20a450a1908a6866e0529d5bde4bbe52b576babf#egg=backtrader_ib_insync"
)

# ----------------------------------------------------------------------------
#  Pretty logging
# ----------------------------------------------------------------------------
function Step($m)  { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Info($m)  { Write-Host "    $m" -ForegroundColor Gray }
function Ok($m)    { Write-Host "    [OK] $m" -ForegroundColor Green }
function Warn($m)  { Write-Host "    [!]  $m" -ForegroundColor Yellow }
function Die($m)   { Write-Host "`n[X] $m" -ForegroundColor Red; exit 1 }

$banner = "=" * 72
Write-Host $banner -ForegroundColor Cyan
Write-Host "  Stock-Market pipeline -- environment setup" -ForegroundColor Cyan
Write-Host "  $root" -ForegroundColor DarkGray
Write-Host $banner -ForegroundColor Cyan

# ----------------------------------------------------------------------------
#  1. Locate a suitable Python interpreter
# ----------------------------------------------------------------------------
Step "Locating Python (>= 3.11)"

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
    # Prefer the py launcher (3.13, then any 3.x), then plain python.
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
if (-not $pyExe) { Die "No working Python found. Install Python 3.11+ or pass -Python <path>." }

$pyVer = Test-Python $pyExe $pyArgs
$major, $minor = $pyVer.Split('.')
if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 11)) {
    Die "Python $pyVer is too old. Need >= 3.11."
}
Ok "Using Python $pyVer  ($pyExe $($pyArgs -join ' '))"

# ----------------------------------------------------------------------------
#  2. Create / reuse the virtual environment
# ----------------------------------------------------------------------------
Step "Virtual environment (stock_env)"
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

# ----------------------------------------------------------------------------
#  3. Upgrade pip tooling
# ----------------------------------------------------------------------------
Step "Upgrading pip / setuptools / wheel"
& $venvPy -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Die "Failed to upgrade pip tooling." }
Ok "pip tooling current"

# ----------------------------------------------------------------------------
#  4. Install curated core requirements
# ----------------------------------------------------------------------------
Step "Installing core requirements"
if (-not (Test-Path $reqFile)) { Die "requirements.txt not found next to setup.ps1." }
& $venvPy -m pip install -r $reqFile
if ($LASTEXITCODE -ne 0) { Die "Core requirements install failed." }
Ok "core packages installed"

# ----------------------------------------------------------------------------
#  5. torch -- CPU vs CUDA
# ----------------------------------------------------------------------------
if ($SkipTorch) {
    Step "torch -- SKIPPED (-SkipTorch). The sentiment path in stage 5 will not run."
} else {
    Step "Installing torch"
    $useGpu = $false
    if ($Gpu)      { $useGpu = $true;  Info "Forced GPU (-Gpu)" }
    elseif ($Cpu)  { $useGpu = $false; Info "Forced CPU (-Cpu)" }
    else {
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            $useGpu = $true;  Info "NVIDIA GPU detected -> CUDA (cu118) wheel"
        } else {
            $useGpu = $false; Info "No NVIDIA GPU detected -> CPU wheel"
        }
    }
    if ($useGpu) { $idx = "https://download.pytorch.org/whl/cu118" }
    else         { $idx = "https://download.pytorch.org/whl/cpu" }
    & $venvPy -m pip install torch --index-url $idx
    if ($LASTEXITCODE -ne 0) { Warn "torch install failed (sentiment path will be unavailable). Continuing." }
    else { Ok "torch installed ($(if ($useGpu) {'CUDA cu118'} else {'CPU'}))" }
}

# ----------------------------------------------------------------------------
#  6. backtrader (git-pinned, required by stage 5 + Util)
# ----------------------------------------------------------------------------
if ($SkipBacktrader) {
    Step "backtrader -- SKIPPED (-SkipBacktrader). Stage 5 backtester will NOT import."
} else {
    Step "Installing backtrader (pinned git commits)"
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Warn "git not on PATH -- skipping backtrader. Stage 5 will fail until installed."
        Warn "Install git, then re-run:  .\setup.ps1 -SkipTorch"
    } else {
        foreach ($dep in $gitDeps) {
            Info "pip install $dep"
            & $venvPy -m pip install $dep
            if ($LASTEXITCODE -ne 0) { Warn "Failed: $dep" } else { Ok "installed" }
        }
    }
}

# ----------------------------------------------------------------------------
#  7. Scaffold the Data/ directory tree
# ----------------------------------------------------------------------------
Step "Scaffolding Data/ directories"
$dataDirs = @(
    "logging",
    "PriceData", "PriceDataFull", "PriceData_1Min",
    "ProcessedData", "ProcessedDataFull", "PreparedData",
    "RFpredictions", "ModelData", "SimpleModel", "Checkpoints",
    "TickerCikData", "TestResults", "Correlations", "MarketCaps",
    "FundamentalData", "Indexes", "IndexesFull", "News",
    "FRED", "FINRA", "ShortInterest", "CFTC_COT", "SEC",
    "Treasury", "KenFrench", "Shiller", "Wikipedia", "UnifiedPanel"
)
$made = 0
foreach ($d in $dataDirs) {
    $p = Join-Path $root "Data\$d"
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null; $made++ }
}
Ok "Data/ tree ready ($made created, $($dataDirs.Count - $made) already present)"

# ----------------------------------------------------------------------------
#  8. API-key templates  (never overwrite real keys)
# ----------------------------------------------------------------------------
Step "API-key templates"

function Write-IfMissing($path, $content) {
    if (Test-Path $path) { return $false }
    Set-Content -Path $path -Value $content -Encoding utf8
    return $true
}

$fredExample = Join-Path $root ".fred_api_key.example"
$claudExample = Join-Path $root "Claud-API-KEY.txt.example"
Write-IfMissing $fredExample  "# Put your FRED API key on the first line (no quotes).`r`n# Get one free at https://fred.stlouisfed.org/docs/api/api_key.html`r`n# Then rename this file to  .fred_api_key" | Out-Null
Write-IfMissing $claudExample "# Put your Anthropic API key on the first line (sk-ant-...).`r`n# Then rename this file to  Claud-API-KEY.txt" | Out-Null
Info "templates: .fred_api_key.example, Claud-API-KEY.txt.example"

if (-not (Test-Path (Join-Path $root ".fred_api_key"))) {
    Warn "No .fred_api_key -- FRED macro fetch will be skipped until you add one."
} else { Ok ".fred_api_key present" }
if (-not (Test-Path (Join-Path $root "Claud-API-KEY.txt"))) {
    Warn "No Claud-API-KEY.txt -- 7__MacroFilter LLM overlay will run in mock/skip mode."
} else { Ok "Claud-API-KEY.txt present" }

# ----------------------------------------------------------------------------
#  9. Protect secrets with a .gitignore  (safety net)
# ----------------------------------------------------------------------------
Step "Securing secrets (.gitignore)"
$giPath = Join-Path $root ".gitignore"
$secretBlock = @(
    "",
    "# --- added by setup.ps1: never commit secrets, venv, data, or caches ---",
    "stock_env/",
    "Data/",
    "__pycache__/",
    "*.pyc",
    ".fred_api_key",
    "Claud-API-KEY.txt",
    ".env",
    "*.log"
)
if (-not (Test-Path $giPath)) {
    Set-Content -Path $giPath -Value $secretBlock -Encoding utf8
    Ok ".gitignore created (your API keys are now protected from git)"
} else {
    $existing = Get-Content $giPath -Raw
    if ($existing -notmatch [regex]::Escape("Claud-API-KEY.txt")) {
        Add-Content -Path $giPath -Value ($secretBlock -join "`r`n") -Encoding utf8
        Ok ".gitignore updated with secret-protection block"
    } else {
        Info ".gitignore already protects the key files"
    }
}

# ----------------------------------------------------------------------------
#  10. Verify imports
# ----------------------------------------------------------------------------
Step "Verifying install"
$check = @'
import importlib, sys
mods = ["numpy","pandas","scipy","sklearn","xgboost","optuna","pyarrow",
        "numba","yfinance","ib_insync","matplotlib","seaborn","plotly",
        "streamlit","pykalman","networkx","ts2vg","watchdog","nest_asyncio",
        "xmltodict","pyperclip","anthropic","transformers","win32clipboard"]
opt = ["torch","backtrader"]
bad = []
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: bad.append((m,str(e).splitlines()[0]))
miss_opt = []
for m in opt:
    try: importlib.import_module(m)
    except Exception: miss_opt.append(m)
for m,e in bad: print(f"  MISSING  {m}: {e}")
for m in miss_opt: print(f"  optional not installed: {m}")
print(f"CORE_OK={len(bad)==0}")
sys.exit(1 if bad else 0)
'@
$verifyFile = Join-Path $env:TEMP "stockmkt_verify.py"
Set-Content -Path $verifyFile -Value $check -Encoding utf8
& $venvPy $verifyFile
$verifyExit = $LASTEXITCODE
Remove-Item $verifyFile -ErrorAction SilentlyContinue

# ----------------------------------------------------------------------------
#  11. Cold-start full pipeline run  (-ColdStart)
# ----------------------------------------------------------------------------
#  Runs every stage from scratch (1 -> 2 -> 3 -> 4 -> 5) so the whole system is
#  exercised end to end and we get a real, current map of what it produces.
#  This is LONG (price download + feature build + train can take hours) and
#  needs network access. It refreshes data in place -- it never deletes Data/.
if ($ColdStart) {
    if ($verifyExit -ne 0) {
        Warn "Skipping -ColdStart: core packages failed to import (fix the MISSING lines first)."
    } else {
        Step "COLD START -- running the full pipeline end to end (this can take a while)"
        $logDir = Join-Path $root "Data\logging"
        if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
        $csLog = Join-Path $logDir ("cold_start_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
        Info "Full transcript -> $csLog"

        $coldStages = @(
            @{ name = "1 TickerDownloader"; script = "1__TickerDownloader.py"; args = @("--ImmediateDownload") },
            @{ name = "2-5 run_pipeline";   script = "run_pipeline.py";        args = @("--force", "--retrain") }
        )
        $csOk = $true
        foreach ($cs in $coldStages) {
            $scriptPath = Join-Path $root $cs.script
            if (-not (Test-Path $scriptPath)) { Warn "missing $($cs.script) -- skipping"; continue }
            Info ">>> $($cs.name):  python $($cs.script) $($cs.args -join ' ')"
            $t0 = Get-Date
            & $venvPy $scriptPath @($cs.args) 2>&1 | Tee-Object -FilePath $csLog -Append
            $code = $LASTEXITCODE
            $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 0)
            if ($code -eq 0) { Ok "$($cs.name) done ($secs s)" }
            else { Warn "$($cs.name) exited $code ($secs s) -- see $csLog"; $csOk = $false }
        }
        if ($csOk) { Ok "Cold start finished -- full pipeline ran end to end." }
        else       { Warn "Cold start finished with errors -- the map below shows what was produced." }
    }
}

# ----------------------------------------------------------------------------
#  12. Generate the pipeline map  (always -- cheap filesystem scan)
# ----------------------------------------------------------------------------
$mapper = Join-Path $root "map_pipeline.py"
if (Test-Path $mapper) {
    Step "Generating pipeline map (PIPELINE_MAP.md)"
    & $venvPy $mapper | Out-Null   # writes PIPELINE_MAP.md; stdout (the verbose map) suppressed
    if (Test-Path (Join-Path $root "PIPELINE_MAP.md")) { Ok "PIPELINE_MAP.md written -- full map of stages + live data inventory" }
} else {
    Warn "map_pipeline.py not found -- skipping map generation"
}

# ----------------------------------------------------------------------------
#  Summary
# ----------------------------------------------------------------------------
Write-Host "`n$banner" -ForegroundColor Cyan
if ($verifyExit -eq 0) {
    Write-Host "  SETUP COMPLETE -- all core packages import cleanly." -ForegroundColor Green
} else {
    Write-Host "  SETUP FINISHED WITH WARNINGS -- see MISSING lines above." -ForegroundColor Yellow
}
Write-Host $banner -ForegroundColor Cyan
Write-Host @"

  Next steps
  ----------
  1. Add API keys (optional but recommended):
       - rename .fred_api_key.example     -> .fred_api_key      (paste FRED key)
       - rename Claud-API-KEY.txt.example -> Claud-API-KEY.txt  (paste Anthropic key)

  2. Pull the free macro/fundamentals data:
       .\stock_env\Scripts\python.exe fetch_all_data.py

  3. Run the daily pipeline (freshness-aware, recommended):
       .\stock_env\Scripts\python.exe run_pipeline.py
     or force every stage:
       .\stock_env\Scripts\python.exe run_pipeline.py --force

  4. See the full map any time (stages + live data inventory):
       open PIPELINE_MAP.md   (or re-run: python map_pipeline.py)

  For a from-scratch end-to-end run, re-run setup with:  .\setup.ps1 -ColdStart

  Pipeline order: 1 TickerDownloader -> 2 PriceDownloader -> 3 AlphaSensitivity
                  -> 4 Predictor -> 5 NightlyBackTester -> 7 MacroFilter -> 9 Broker

"@ -ForegroundColor White
