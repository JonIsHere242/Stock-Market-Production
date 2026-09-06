# rerun_for_live.ps1 - reproduce the full live pipeline for a fresh, correctly-dated signal.
# Run this AFTER the 07-01 close so fresh data -> features -> predict -> backtest -> signal produces a
# TRUE next-session (07-02) signal from 07-01 data (fixes the stale-by-a-session issue).
# Matches the ship monday_pipeline exactly. Backs up the live folder + signal first (rollback-safe).
# The BROKER step is intentionally NOT here - you run the trade-signals/FilterRubric narrowing yourself.

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

Write-Host "=== 3) Predictor predict_only + Phase-13 neutralization + Phase-14 tilt (ship model) -> RFpredictions ===" -ForegroundColor Cyan
# --neutralize = the old 4.5 stage, folded into the predictor 2026-07-27 (runs in-memory at
# the end of inference; each row keeps its un-neutralized value in `raw_up_prob`).
# spy200v2 validated 07-12 (live 80.2% vs 52.2% fixed-0.30, 2 seeds); fixed-dose history
# 07-01/07-03. Fail-safe: stale SPY flag degrades to dose_below 0.30; on gate failure it
# writes RAW preds and exits non-zero (plain ship signal).
#
# The conviction-momentum tilt is Phase 14 of the SAME call as of 2026-07-29 and needs no
# flag: it is the default. It was the separate step 3.5 here (4.6__ConvictionMomentum.py,
# now retired to _archive/) until the fold-in, and running that file after this line would
# now tilt an already-tilted book twice. Pass --no_conviction_momentum for an ablation.
# up' = clip(up + k*mean_spans(up - EMA_span(up)), .30, .70) per ticker, causal, applied to
# the FINAL (post-neutralization) UpProbability = a rising-conviction tilt. Full sim main
# window: ann 59.4->98.7, Sharpe 1.48->2.40, maxDD 27.1->12.2; 8-seed baseline null
# 59.4-62.1 so it clears the BEST baseline seed by +36pp with zero overlap.
# !! Regime-dependent: wins while the level signal is degraded, loses while it is healthy.
# Fail-safe: on a tilt gate failure it ships UN-TILTED preds and exits non-zero.
# Rollback: per-row pre-tilt value in `pre_cm_up_prob` on every row.
python 4__Predictor.py --input_dir Data/ProcessedData_v2 --predict_only --model_dir Data/_ship_v2/model --output_dir Data/RFpredictions --neutralize --neut_mode spy200v2 --neut_dose_above 0.15 --neut_dose_below 0.30

Write-Host "=== 4) Production backtester --force (backtest + live-signal export) ===" -ForegroundColor Cyan
$env:BT_SAMPLE_SEED = "42"
# Pin the candidate-ranking rule. Validated +14.07pp over 6 asof anchors (t=+3.76,
# 6/6 positive) and the arm every 2026-07-29 exit-package result was measured on.
# Without this pin, scheduled runs rank by raw UpProbability ('shipped') while hand
# runs may not, and the two books silently diverge.
$env:BT_SELRULE = "low_atr"
python 5__NightlyBackTester.py --force

Write-Host "=== 5) Verify fresh signal date ===" -ForegroundColor Cyan
python -c "import pandas as pd; d=pd.read_parquet('Data/0__signals.parquet'); print('rows',len(d),'| TargetDate',d['TargetDate'].max(),'| created',d['CreatedDate'].max())"

Write-Host "`nDONE. Confirm TargetDate is the session you intend, then run your FilterRubric/broker narrowing." -ForegroundColor Green
Write-Host "Rollback:  Remove-Item Data/RFpredictions -Recurse -Force; Rename-Item RFpredictions_bak_$stamp Data/RFpredictions; Copy-Item Data/0__signals_bak_$stamp.parquet Data/0__signals.parquet -Force" -ForegroundColor Yellow
