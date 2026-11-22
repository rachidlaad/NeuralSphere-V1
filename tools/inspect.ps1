$log = "$PSScriptRoot\inspect_log.txt"
"=== $(Get-Date) ===" | Out-File $log
"OS: " + [Environment]::OSVersion.VersionString | Out-File $log -Append
foreach ($c in 'python','python3','py','ngspice','ngspice_con','qucs-s','winget','curl.exe','tar.exe','7z') {
  $cmd = Get-Command $c -ErrorAction SilentlyContinue
  "$c => " + ($(if ($cmd) { $cmd.Source } else { 'NOT FOUND' })) | Out-File $log -Append
}
try { (& py -0p 2>&1) | Out-File $log -Append } catch { "py launcher error" | Out-File $log -Append }
try { (& python --version 2>&1) | Out-File $log -Append } catch { "python error" | Out-File $log -Append }
Get-ChildItem "$env:LOCALAPPDATA\Programs\Python","C:\Python*","$env:USERPROFILE\anaconda3","$env:USERPROFILE\miniconda3" -ErrorAction SilentlyContinue | Select-Object FullName | Out-File $log -Append
"DONE" | Out-File $log -Append
