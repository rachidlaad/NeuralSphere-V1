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
# Each step is a script followed by its arguments. The Stage E (s05) invocations reproduce the
# archived reference exactly: chips A,B 10 iterations at lr 2; chip C 15 iterations at lr 2 and at lr 0.5.
$steps = @("s00_validate_ngspice.py", "s01_stageA_single_neuron.py", "s02_train_software.py",
           "s03_stageBC_build_verify.py", "s04_stageD_inference.py")
if (-not $Quick) {
    $steps += @("s05_stageE_in_loop.py 10 chipA_6bit_gauss5pct_vos5mV,chipB_4bit_gauss10pct_vos10mV 2.0",
                "s05_stageE_in_loop.py 15 chipC_2bit_gauss20pct_vos20mV 2.0",
                "s05_stageE_in_loop.py 15 chipC_2bit_gauss20pct_vos20mV 0.5",
                "s06_nonidealities.py",
                "s12_opa196_stageD.py", "s13_noise.py", "s07_transient_energy.py opa196")
}
$steps += @("s07_transient_energy.py", "s08_wta.py", "s10_qucs_schematic.py", "s11_qucs_native_check.py", "s09_figures.py")
foreach ($s in $steps) {
    Write-Host "`n===== $s ====="
    $parts = $s -split " "
    $scriptArgs = @($parts | Select-Object -Skip 1)
    & ..\$py $parts[0] @scriptArgs
    if ($LASTEXITCODE -ne 0) { throw "$($parts[0]) failed" }
}
Set-Location ..
Write-Host "`nDone. Results in .\results, figures in .\plots. Start the demo with:  .\.venv\Scripts\python.exe demo\demo_app.py"
