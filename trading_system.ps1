# Trading System Runner - PRODUCTION VERSION
# Runs all trading scripts sequentially with proper delays

$Host.UI.RawUI.WindowTitle = "Trading System - Production Runner"

# Verified working paths
$basePath = "C:\Users\Masam\Desktop\Stock-Market"
$pythonExe = "$basePath\stock_env\Scripts\python.exe"
$logFile = "$basePath\Data\logging\__trading_system.log"

function Write-Log($message, $color = "White") {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $message"
    Write-Host $logMessage -ForegroundColor $color
    
    try {
        Add-Content -Path $logFile -Value $logMessage
    }
    catch {
        Write-Host "Warning: Could not write to log file" -ForegroundColor Yellow
    }
}

function Run-Script($script) {
    $scriptPath = Join-Path $basePath $script.File
    
    Write-Log "Starting $($script.Name)..." "Yellow"
    Write-Log "Script: $($script.File)" "Gray"
    Write-Log "Args: $($script.Args)" "Gray"
    
    if (!(Test-Path $scriptPath)) {
        Write-Log "ERROR: Script not found: $scriptPath" "Red"
        return $false
    }
    
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    
    try {
        # Build argument list
        if ($script.Args) {
            $arguments = @($scriptPath) + ($script.Args -split ' ')
        } else {
            $arguments = @($scriptPath)
        }
        
        Write-Log "Executing: $pythonExe $($arguments -join ' ')" "Cyan"
        
        # Run the Python script and capture output
        $process = Start-Process -FilePath $pythonExe -ArgumentList $arguments -WorkingDirectory $basePath -Wait -PassThru -NoNewWindow
        
        $timer.Stop()
        $elapsed = [math]::Round($timer.Elapsed.TotalSeconds, 1)
        
        if ($process.ExitCode -eq 0) {
            Write-Log "$($script.Name) COMPLETED successfully ($elapsed seconds)" "Green"
            return $true
        } else {
            Write-Log "$($script.Name) FAILED with exit code $($process.ExitCode) ($elapsed seconds)" "Red"
            return $false
        }
    }
    catch {
        $timer.Stop()
        Write-Log "$($script.Name) CRASHED: $($_.Exception.Message)" "Red"
        return $false
    }
}

function Wait-WithCountdown($seconds, $message) {
    Write-Log $message "Yellow"
    for ($i = $seconds; $i -gt 0; $i--) {
        Write-Host "  Waiting $i seconds (RAM cleanup)..." -ForegroundColor Gray
        Start-Sleep -Seconds 1
    }
    Write-Log "Wait complete, continuing..." "Green"
}

# Create log directory if needed
$logDir = Split-Path $logFile -Parent
if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

Clear-Host

# Check what time it is
$currentTime = Get-Date
$currentHour = $currentTime.Hour

Write-Log "=========================================" "Cyan"
Write-Log "TRADING SYSTEM PRODUCTION RUN" "Cyan"
Write-Log "=========================================" "Cyan"
Write-Log "Start time: $($currentTime.ToString('yyyy-MM-dd HH:mm:ss'))" "White"
Write-Log "Current hour: $currentHour" "White"

# Define all scripts with their arguments
$allScripts = @(
    @{ Name = "Ticker Downloader"; File = "1__TickerDownloader.py"; Args = "--ImmediateDownload" },
    @{ Name = "Bulk Price Downloader"; File = "2__BulkPriceDownloader.py"; Args = "--RefreshMode" },
    @{ Name = "Alpha Sensitivity"; File = "3__AlphaSensitivity.py"; Args = "--runpercent 100" },
    @{ Name = "Predictor"; File = "4__Predictor.py"; Args = "--predict --model xgb" },
    @{ Name = "Nightly BackTester"; File = "5__NightlyBackTester.py"; Args = "--force" }
)

# Determine which scripts to run based on time
if ($currentHour -eq 17) {
    Write-Log "=== EVENING DATA PROCESSING MODE ===" "Cyan"
    $scriptsToRun = $allScripts
} elseif ($currentHour -eq 7) {
    Write-Log "=== MORNING BROKER MODE ===" "Cyan"
    $scriptsToRun = @(@{ Name = "Daily Broker"; File = "8__DailyBroker.py"; Args = "" })
} else {
    Write-Log "=== FULL PIPELINE TEST MODE ===" "Cyan"
    Write-Log "Running all scripts in test mode..." "Yellow"
    $scriptsToRun = $allScripts
}

# Execute scripts
$pipelineStart = Get-Date
$successCount = 0
$totalScripts = $scriptsToRun.Count

Write-Log "Pipeline will execute $totalScripts scripts" "White"
Write-Log "=========================================" "White"

for ($i = 0; $i -lt $scriptsToRun.Count; $i++) {
    $script = $scriptsToRun[$i]
    $scriptNumber = $i + 1
    
    Write-Log "[$scriptNumber/$totalScripts] Starting $($script.Name)" "Cyan"
    
    $result = Run-Script $script
    
    if ($result) { 
        $successCount++ 
        Write-Log "[$scriptNumber/$totalScripts] SUCCESS: $($script.Name)" "Green"
    } else {
        Write-Log "[$scriptNumber/$totalScripts] FAILED: $($script.Name)" "Red"
    }
    
    # Wait between scripts (except after the last one)
    if ($i -lt ($scriptsToRun.Count - 1)) {
        Write-Log "=========================================" "Gray"
        Wait-WithCountdown 20 "Waiting 20 seconds before next script (RAM cleanup)..."
        Write-Log "=========================================" "Gray"
    }
}

# Pipeline summary
$pipelineEnd = Get-Date
$totalTime = [math]::Round(($pipelineEnd - $pipelineStart).TotalMinutes, 1)

Write-Log "=========================================" "Cyan"
Write-Log "PIPELINE EXECUTION SUMMARY" "Cyan"
Write-Log "=========================================" "Cyan"
Write-Log "Total scripts: $totalScripts" "White"
Write-Log "Successful: $successCount" "Green"
Write-Log "Failed: $($totalScripts - $successCount)" "Red"
Write-Log "Total time: $totalTime minutes" "White"
Write-Log "End time: $($pipelineEnd.ToString('yyyy-MM-dd HH:mm:ss'))" "White"

if ($successCount -eq $totalScripts) {
    Write-Log "ALL SCRIPTS COMPLETED SUCCESSFULLY!" "Green"
} else {
    Write-Log "SOME SCRIPTS FAILED - Check logs above" "Red"
}

Write-Log "=========================================" "Cyan"

# Final pause
Write-Host ""
Write-Host "=== EXECUTION COMPLETE ===" -ForegroundColor Cyan
Write-Host "Success rate: $successCount/$totalScripts scripts" -ForegroundColor White
Write-Host "Total time: $totalTime minutes" -ForegroundColor White
Write-Host "Log file: $logFile" -ForegroundColor Yellow
Write-Host ""
Write-Host "Window will close in 60 seconds or press any key to exit..." -ForegroundColor Yellow

# Wait for either a key press or timeout
$timeout = 60
$timer = [System.Diagnostics.Stopwatch]::StartNew()

while ($timer.Elapsed.TotalSeconds -lt $timeout) {
    if ($Host.UI.RawUI.KeyAvailable) {
        $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
        Write-Log "Script terminated by user" "Gray"
        break
    }
    Start-Sleep -Milliseconds 500
    
    # Show countdown every 10 seconds
    $remaining = [math]::Round($timeout - $timer.Elapsed.TotalSeconds)
    if ($remaining % 10 -eq 0 -and $remaining -gt 0) {
        Write-Host "Closing in $remaining seconds..." -ForegroundColor Gray
    }
}

$timer.Stop()
Write-Log "Trading system runner completed" "Cyan"