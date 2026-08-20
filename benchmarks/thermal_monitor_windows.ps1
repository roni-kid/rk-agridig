<#
.SYNOPSIS
    RK AgriDig — Windows-side thermal monitor (companion to Task 1.4).

.DESCRIPTION
    WSL2's Linux kernel has no access to real hardware thermal sensors
    (confirmed: sensors-detect fails with "Module cpuid not found" under
    WSL2 — this is a kernel-level limitation, not fixable from inside WSL).

    This script works around that by reading the REAL CPU temperature from
    the Windows host directly, via the MSAcpi_ThermalZoneTemperature WMI
    class. It polls at a fixed interval, logs each reading, and produces
    the same avg/max/min + ADTC threshold summary as benchmarks/thermal_monitor.py,
    so results are directly comparable.

    Run this in an ELEVATED (Administrator) PowerShell window — the WMI
    namespace this reads from requires admin rights (confirmed: standard
    user gets "Access denied", HRESULT 0x80041003).

    Run this ALONGSIDE inference in WSL, not instead of it — start this
    script first, then in a separate WSL terminal run your inference loop
    (e.g. benchmarks/run_profiler.sh or repeated llama-cli calls), then
    let this script finish its duration.

.PARAMETER DurationSeconds
    How long to monitor for, in seconds. Default 300 (5 minutes), matching
    the ADTC build plan's Task 1.4 spec.

.PARAMETER IntervalSeconds
    Seconds between temperature samples. Default 5, matching the build plan.

.PARAMETER OutputDir
    Directory to write results into. Default: benchmarks\results relative
    to the project root (assumes this script lives in benchmarks\).

.EXAMPLE
    # Run as Administrator:
    .\thermal_monitor_windows.ps1
    .\thermal_monitor_windows.ps1 -DurationSeconds 300 -IntervalSeconds 5
#>

param(
    [int]$DurationSeconds = 300,
    [double]$IntervalSeconds = 5,
    [string]$OutputDir = $(Join-Path (Split-Path $PSScriptRoot -Parent) "benchmarks\results")
)

$ErrorActionPreference = "Stop"

$AdtcDisqualificationTempC = 85.0
$AdtcTargetCeilingC = 80.0

# --- Require elevation ---
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator." -ForegroundColor Red
    Write-Host "  The MSAcpi_ThermalZoneTemperature WMI class requires elevated rights on this system" -ForegroundColor Red
    Write-Host "  (confirmed: non-admin query returns 'Access denied', HRESULT 0x80041003)." -ForegroundColor Red
    Write-Host "  Right-click PowerShell / Windows Terminal and choose 'Run as Administrator', then re-run this script." -ForegroundColor Yellow
    exit 1
}

# --- Ensure output directory exists ---
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}
$logPath = Join-Path $OutputDir "thermal_logs_windows.txt"
$summaryPath = Join-Path $OutputDir "thermal_summary_windows.json"

Write-Host ("=" * 70)
Write-Host "RK AgriDig -- Windows-side Thermal Monitor (Task 1.4 companion)"
Write-Host ("=" * 70)
Write-Host "Duration:  $DurationSeconds s"
Write-Host "Interval:  $IntervalSeconds s"
Write-Host "ADTC disqualification threshold: $AdtcDisqualificationTempC C"
Write-Host "ADTC target ceiling: $AdtcTargetCeilingC C"
Write-Host ""
Write-Host "Reading real hardware temperature via WMI (root/WMI, MSAcpi_ThermalZoneTemperature)."
Write-Host "Make sure your WSL2 inference workload is running NOW in a separate terminal."
Write-Host ""

# --- Verify the sensor actually returns data before committing to a full run ---
function Get-CpuTempC {
    try {
        $zones = Get-CimInstance -Namespace "root/WMI" -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop
        if (-not $zones) {
            return $null
        }
        # Some systems report multiple thermal zones (chassis, CPU, etc).
        # Take the max reading as the conservative "worst case" value, same
        # policy as the Linux thermal_monitor.py script.
        $tempsC = $zones | ForEach-Object { ($_.CurrentTemperature - 2732) / 10.0 }
        return ($tempsC | Measure-Object -Maximum).Maximum
    }
    catch {
        return $null
    }
}

$testReading = Get-CpuTempC
if ($null -eq $testReading) {
    Write-Host "ERROR: MSAcpi_ThermalZoneTemperature returned no usable data on this system." -ForegroundColor Red
    Write-Host "  This can happen even with admin rights if the manufacturer doesn't populate" -ForegroundColor Red
    Write-Host "  this WMI class. Consider LibreHardwareMonitor's REST API as a fallback." -ForegroundColor Yellow
    exit 1
}
Write-Host ("Sensor check OK -- current reading: {0:N1} C" -f $testReading) -ForegroundColor Green
Write-Host ""

# --- Monitoring loop ---
$readings = New-Object System.Collections.Generic.List[PSObject]
$startTime = Get-Date
$logLines = New-Object System.Collections.Generic.List[string]
$logLines.Add("# RK AgriDig Windows Thermal Log -- started $($startTime.ToUniversalTime().ToString('o'))")
$logLines.Add("# source=WMI:MSAcpi_ThermalZoneTemperature")
$logLines.Add("# adtc_disqualification_threshold_c=$AdtcDisqualificationTempC")
$logLines.Add("# timestamp_utc,elapsed_s,temp_c")

$alertFired = $false

try {
    while ($true) {
        $elapsed = ((Get-Date) - $startTime).TotalSeconds
        if ($elapsed -ge $DurationSeconds) { break }

        $temp = Get-CpuTempC
        $ts = (Get-Date).ToUniversalTime().ToString('o')

        if ($null -ne $temp) {
            $readings.Add([PSCustomObject]@{ elapsed_s = [math]::Round($elapsed,1); temp_c = [math]::Round($temp,1) })
            $tempStr = "{0:N1}" -f $temp
        } else {
            $tempStr = "N/A"
        }

        $logLines.Add("$ts,$([math]::Round($elapsed,1)),$tempStr")
        Write-Host ("[{0,6:N1}s] Temp: {1} C" -f $elapsed, $tempStr)

        if ($null -ne $temp) {
            if ($temp -ge $AdtcDisqualificationTempC -and -not $alertFired) {
                Write-Host ""
                Write-Host ("ALERT: Temperature {0:N1} C EXCEEDS ADTC disqualification threshold ({1} C)!" -f $temp, $AdtcDisqualificationTempC) -ForegroundColor Red
                Write-Host ""
                $alertFired = $true
            }
            elseif ($temp -ge $AdtcTargetCeilingC) {
                Write-Host ("   Above target ceiling of {0} C" -f $AdtcTargetCeilingC) -ForegroundColor Yellow
            }
        }

        Start-Sleep -Seconds $IntervalSeconds
    }
}
catch [System.Management.Automation.PipelineStoppedException] {
    Write-Host "`nInterrupted by user -- writing partial results..."
}

# --- Write raw log ---
# NOTE: Set-Content -Encoding utf8 (and utf8NoBOM, which doesn't exist in
# Windows PowerShell 5.1 at all -- only PowerShell 6+) both have BOM issues
# on this system. Writing directly via .NET's UTF8Encoding($false) avoids
# the BOM on every PowerShell version, and is the fix confirmed to work on
# Windows PowerShell 5.1 specifically (the version this project runs on --
# see "Windows PowerShell" banner, not "PowerShell 7").
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($logPath, ($logLines -join "`n") + "`n", $utf8NoBom)

# --- Compute summary ---
$validTemps = $readings | ForEach-Object { $_.temp_c }
$summary = [ordered]@{
    timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
    duration_requested_s = $DurationSeconds
    duration_actual_s = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 1)
    interval_s = $IntervalSeconds
    source = "WMI:MSAcpi_ThermalZoneTemperature (Windows host, elevated)"
    sample_count = $readings.Count
    avg_temp_c = $null
    max_temp_c = $null
    min_temp_c = $null
    adtc_disqualification_threshold_c = $AdtcDisqualificationTempC
    adtc_target_ceiling_c = $AdtcTargetCeilingC
    exceeded_disqualification_threshold = $null
    exceeded_target_ceiling = $null
    note = "Captured from Windows host during WSL2 inference workload, since WSL2's kernel has no direct hardware sensor access (coretemp module unavailable). Sensor reads via WMI ACPI thermal zone, which reflects overall system/CPU package temperature."
}

if ($validTemps.Count -gt 0) {
    $stats = $validTemps | Measure-Object -Average -Maximum -Minimum
    $summary.avg_temp_c = [math]::Round($stats.Average, 2)
    $summary.max_temp_c = [math]::Round($stats.Maximum, 2)
    $summary.min_temp_c = [math]::Round($stats.Minimum, 2)
    $summary.exceeded_disqualification_threshold = ($stats.Maximum -ge $AdtcDisqualificationTempC)
    $summary.exceeded_target_ceiling = ($stats.Maximum -ge $AdtcTargetCeilingC)
}

$summary | ConvertTo-Json -Depth 5 | Out-String | ForEach-Object { [System.IO.File]::WriteAllText($summaryPath, $_.TrimEnd() + "`n", $utf8NoBom) }

# --- Print summary ---
Write-Host ""
Write-Host ("=" * 70)
Write-Host "Thermal Monitoring Summary"
Write-Host ("=" * 70)
if ($validTemps.Count -gt 0) {
    Write-Host ("  Avg temp: {0} C" -f $summary.avg_temp_c)
    Write-Host ("  Max temp: {0} C" -f $summary.max_temp_c)
    Write-Host ("  Min temp: {0} C" -f $summary.min_temp_c)
    Write-Host ("  Exceeded 85 C disqualification threshold: {0}" -f $summary.exceeded_disqualification_threshold)
    Write-Host ("  Exceeded 80 C target ceiling: {0}" -f $summary.exceeded_target_ceiling)
} else {
    Write-Host "  No valid temperature samples were collected."
}
Write-Host ""
Write-Host "  Raw log:      $logPath"
Write-Host "  Summary JSON: $summaryPath"
Write-Host ("=" * 70)