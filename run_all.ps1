# NeuralSphere-V1: reproduce the whole experiment on Windows.
# Usage (from a PowerShell window in this folder):
#     powershell -ExecutionPolicy Bypass -File .\run_all.ps1            # full run (~1-2 h on 4 cores)
#     powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Quick     # skips the long studies
param([switch]$Quick)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Creating virtual environment (.venv) and installing packages ..."
    python -m venv .venv
    & $py -m pip install --upgrade pip
    & $py -m pip install -r requirements.txt
}
# ngspice: tools\ngspice (official ngspice-47 build) or the copy bundled with Qucs-S (tools\qucs-s\bin)
$ng = Get-ChildItem .\tools -Recurse -Filter ngspice_con.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $ng) { throw "ngspice_con.exe not found under .\tools - see README section 'Install'" }
$env:NGSPICE = $ng.FullName
Write-Host "Using ngspice: $env:NGSPICE"
Set-Location scripts
$steps = @("s00_validate_ngspice.py", "s01_stageA_single_neuron.py", "s02_train_software.py",
           "s03_stageBC_build_verify.py", "s04_stageD_inference.py")
if (-not $Quick) { $steps += @("s05_stageE_in_loop.py", "s06_nonidealities.py") }
$steps += @("s07_transient_energy.py", "s08_wta.py", "s10_qucs_schematic.py", "s09_figures.py")
foreach ($s in $steps) {
    Write-Host "`n===== $s ====="
    & ..\$py $s
    if ($LASTEXITCODE -ne 0) { throw "$s failed" }
}
Set-Location ..
Write-Host "`nDone. Results in .\results, figures in .\plots. Start the demo with:  .\.venv\Scripts\python.exe demo\demo_app.py"
