$ErrorActionPreference = 'Continue'
$tools = $PSScriptRoot
$log = "$tools\setup_ngspice_log.txt"
function L($m) { "$(Get-Date -Format HH:mm:ss) $m" | Out-File $log -Append -Encoding utf8 }
"" | Out-File $log -Encoding utf8
$ng7z = "$tools\ngspice-47_64.7z"
if ((Test-Path $ng7z) -and ((Get-Item $ng7z).Length -lt 1000000)) { Remove-Item $ng7z }
if (-not (Test-Path $ng7z)) {
  L "downloading ngspice-47_64.7z from SourceForge"
  & curl.exe -L --fail -A "Mozilla/5.0" -o $ng7z "https://sourceforge.net/projects/ngspice/files/ng-spice-rework/47/ngspice-47_64.7z/download" 2>&1 | Select-Object -Last 2 | ForEach-Object { L "curl: $_" }
}
L ("ngspice 7z size: " + (Get-Item $ng7z -ErrorAction SilentlyContinue).Length)
& 7z x $ng7z "-o$tools\ngspice" -y 2>&1 | Select-Object -Last 2 | ForEach-Object { L $_ }
$ngexe = Get-ChildItem "$tools\ngspice" -Recurse -Filter ngspice_con.exe -ErrorAction SilentlyContinue | Select-Object -First 1
L ("ngspice_con: " + $ngexe.FullName)
if ($ngexe) { & $ngexe.FullName -v 2>&1 | Select-Object -First 3 | ForEach-Object { L "NG> $_" } }
if ($ngexe) { & $ngexe.FullName -b "$tools\validate_divider.cir" 2>&1 | ForEach-Object { L "NG> $_" } }
# list Qucs-S bundled simulators
Get-ChildItem "$tools\qucs-s\bin" -Filter *.exe | ForEach-Object { L ("qucs bin: " + $_.Name) }
New-Item -ItemType Directory -Force "$tools\split" | Out-Null
& 7z a -mx0 -v2m "$tools\split\ngsrc.7z" "$tools\ngspice-47.tar.gz" -y 2>&1 | Select-Object -Last 1 | ForEach-Object { L $_ }
L "NGSPICE SETUP DONE"
