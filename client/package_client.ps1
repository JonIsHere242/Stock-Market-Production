<#
================================================================================
  package_client.ps1  --  build the customer bundle
================================================================================

  Assembles the thin execution client into a zip you can hand to a customer.
  Run this from the repo root on YOUR machine.

  It copies only the files the client actually needs (about 300 KB) and
  deliberately excludes the research pipeline, the model, the data lake, the
  feature templates, and every API key.

  USAGE
    .\client\package_client.ps1
    .\client\package_client.ps1 -OutDir D:\handoff
    .\client\package_client.ps1 -IncludeBook      # also ship today's book
================================================================================
#>

[CmdletBinding()]
param(
    [string]$OutDir = "",
    [switch]$IncludeBook
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
if ($OutDir -eq "") { $OutDir = Join-Path $repo "_dist" }

function Info($m) { Write-Host "    $m" -ForegroundColor Gray }
function Ok($m)   { Write-Host "    [OK] $m" -ForegroundColor Green }
function Die($m)  { Write-Host "`n[X] $m" -ForegroundColor Red; exit 1 }

Write-Host "`n==> Packaging execution client" -ForegroundColor Cyan

# The complete client manifest. If you add a runtime import to Util.py or the
# broker, it belongs here too, or the customer install breaks on import.
$manifest = @(
    "9_SuperFastBroker.py",
    "Util.py",
    "auxiliary\bracket_config.py",
    "client\install_client.ps1",
    "client\requirements-client.txt",
    "client\check_book.py",
    "client\CLIENT_SETUP.md"
)
if ($IncludeBook) { $manifest += "_Buy_Signals.parquet" }

$stamp   = Get-Date -Format "yyyyMMdd"
$stageDir = Join-Path $env:TEMP "smclient_stage_$stamp"
if (Test-Path $stageDir) { Remove-Item -Recurse -Force $stageDir }
New-Item -ItemType Directory -Path $stageDir -Force | Out-Null

$missing = @()
foreach ($rel in $manifest) {
    $src = Join-Path $repo $rel
    if (-not (Test-Path $src)) { $missing += $rel; continue }
    $dst = Join-Path $stageDir $rel
    $dstParent = Split-Path -Parent $dst
    if (-not (Test-Path $dstParent)) { New-Item -ItemType Directory -Path $dstParent -Force | Out-Null }
    Copy-Item $src $dst
    Info $rel
}
if ($missing.Count -gt 0) { Die "Manifest files missing from the repo: $($missing -join ', ')" }

# Empty working dirs so the client has somewhere to write on first run.
foreach ($d in @("Data", "Data\logging", "Data\_backups")) {
    New-Item -ItemType Directory -Path (Join-Path $stageDir $d) -Force | Out-Null
}

# Safety net: never ship a secret. Fail loudly rather than quietly excluding.
$leaks = Get-ChildItem $stageDir -Recurse -File | Where-Object {
    $_.Name -match "api.?key|\.env$|\.pem$|_keys$" -or $_.Name -eq ".fred_api_key"
}
if ($leaks) { Die "Refusing to package, secret-looking files staged: $($leaks.Name -join ', ')" }
Ok "no secrets in the bundle"

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }
$zip = Join-Path $OutDir "StockMarketClient_$stamp.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $stageDir "*") -DestinationPath $zip
Remove-Item -Recurse -Force $stageDir

$sizeKb = [math]::Round((Get-Item $zip).Length / 1KB, 0)
Ok "bundle written: $zip ($sizeKb KB)"

Write-Host @"

  Send the customer this zip. On their machine:
    1. Unzip anywhere, for example C:\StockMarket
    2. Open PowerShell in that folder
    3. .\client\install_client.ps1

  Read client\CLIENT_SETUP.md with them on the call.

"@ -ForegroundColor White
