# Download the Texas Instruments OPAx196 PSpice macromodel used by s12/s13 and verify it.
# The TI file is copyrighted ("All rights reserved"), so it is NOT stored in this repository;
# this script fetches the exact version used for the published results.
# Usage (from the project root):  powershell -ExecutionPolicy Bypass -File .\tools\get_opa196_model.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$raw  = Join-Path $root "tools\vendor_models\download_opa196_raw"
$dst  = Join-Path $root "tools\vendor_models\opa196_extracted"
$zip  = Join-Path $raw "SBOMAF9E.ZIP"
$url  = "https://www.ti.com/lit/zip/sbomaf9"
$zipSha = "9c264813a36536f743c166b6a101904b76ed08f049f2eeb127a438cc6e8ccdf5"
$libSha = "31142f5dbd71b758d887ee8a411692fc4c1309c8f47100af4ba5bed7e173c430"

New-Item -ItemType Directory -Force $raw, $dst | Out-Null
if (-not (Test-Path $zip)) {
    Write-Host "Downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
}
$h = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
if ($h -ne $zipSha) {
    Write-Warning "SHA-256 of the zip is $h, expected $zipSha. TI may have published a newer model version; results can differ slightly."
}
Expand-Archive -Path $zip -DestinationPath $dst -Force
$lib = Join-Path $dst "OPAx196.LIB"
$hl = (Get-FileHash $lib -Algorithm SHA256).Hash.ToLower()
if ($hl -eq $libSha) { Write-Host "OK: $lib (model 'Final 1.5', 29 Aug 2022) verified." }
else { Write-Warning "OPAx196.LIB SHA-256 is $hl, expected $libSha (different model version)." }
