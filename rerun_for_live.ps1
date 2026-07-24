# rerun_for_live.ps1 — reproduce the full live pipeline for a fresh, correctly-dated signal.
# Run this AFTER the 07-01 close so fresh data -> features -> predict -> backtest -> signal produces a
# TRUE next-session (07-02) signal from 07-01 data (fixes the stale-by-a-session issue).
# Matches the ship monday_pipeline exactly. Backs up the live folder + signal first (rollback-safe).
# The BROKER step is intentionally NOT here — you run the trade-signals/FilterRubric narrowing yourself.

$ErrorActionPreference = "Stop"
Set-Location "c:/Users/Masam/Desktop/Stock-Market"
$env:PYTHONPATH = "c:/Users/Masam/Desktop/Stock-Market"
$stamp = Get-Date -Format "yyyyMMdd_HHmm"

Write-Host "=== 0) BACK UP live folder + signal ($stamp) ===" -ForegroundColor Cyan
if (Test-Path Data/RFpredictions)      { Rename-Item Data/RFpredictions      "RFpredictions_bak_$stamp" }
if (Test-Path Data/0__signals.parquet) { Copy-Item   Data/0__signals.parquet "Data/0__signals_bak_$stamp.parquet" }
New-Item -ItemType Directory -Force Data/RFpredictions | Out-Null

Write-Host "=== 1) Pull fresh prices ===" -ForegroundColor Cyan
python 2__PriceDownloader.py --RefreshMode

Write-Host "=== 2) FeatureFramework (your exact exclude list) ===" -ForegroundColor Cyan
python 3__FeatureFramework.py --all --exclude vvg vaq_vg rvg_wl volume_spectral_splatter --workers 32

Write-Host "=== 3) Predictor predict_only (ship model) -> RFpredictions ===" -ForegroundColor Cyan
python 4__Predictor.py --input_dir Data/ProcessedData_v2 --predict_only --model_dir Data/_ship_v2/model --output_dir Data/RFpredictions

Write-Host "=== 3.5) Neutralize preds (spy200v2 0.15/0.30) + swap into RFpredictions ===" -ForegroundColor Cyan
# spy200v2 validated 07-12 (live 80.2% vs 52.2% fixed-0.30, 2 seeds); fixed-dose history
# 07-01/07-03. Fail-safe: stale SPY flag degrades to dose_below 0.30; on gate failure it
# exits non-zero and leaves RFpredictions RAW (plain ship signal).
python 4.5__NeutralizePreds.py --mode spy200v2 --dose_above 0.15 --dose_below 0.30

Write-Host "=== 4) Production backtester --force (backtest + live-signal export) ===" -ForegroundColor Cyan
$env:BT_SAMPLE_SEED = "42"
python 5__NightlyBackTester.py --force

Write-Host "=== 5) Verify fresh signal date ===" -ForegroundColor Cyan
python -c "import pandas as pd; d=pd.read_parquet('Data/0__signals.parquet'); print('rows',len(d),'| TargetDate',d['TargetDate'].max(),'| created',d['CreatedDate'].max())"

Write-Host "`nDONE. Confirm TargetDate is the session you intend, then run your FilterRubric/broker narrowing." -ForegroundColor Green
Write-Host "Rollback:  Remove-Item Data/RFpredictions -Recurse -Force; Rename-Item RFpredictions_bak_$stamp Data/RFpredictions; Copy-Item Data/0__signals_bak_$stamp.parquet Data/0__signals.parquet -Force" -ForegroundColor Yellow
