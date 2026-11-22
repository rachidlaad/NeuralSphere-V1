$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
$tools = $PSScriptRoot
$log = "$tools\setup_log.txt"
function L($m) { "$(Get-Date -Format HH:mm:ss) $m" | Out-File $log -Append -Encoding utf8 }
"" | Out-File $log -Encoding utf8
L "root=$root"

# 1. ngspice-47 Windows 64-bit binaries (official SourceForge distribution)
$ng7z = "$tools\ngspice-47_64.7z"
if (-not (Test-Path $ng7z)) {
  L "downloading ngspice-47_64.7z"
  & curl.exe -L --fail -A "Mozilla/5.0" -o $ng7z "https://sourceforge.net/projects/ngspice/files/ng-spice-rework/46/ngspice-47_64.7z/download" 2>&1 | Out-Null
}
L ("ngspice 7z size: " + (Get-Item $ng7z -ErrorAction SilentlyContinue).Length)
& 7z x $ng7z "-o$tools\ngspice" -y 2>&1 | Select-Object -Last 3 | ForEach-Object { L $_ }
$ngexe = Get-ChildItem "$tools\ngspice" -Recurse -Filter ngspice_con.exe -ErrorAction SilentlyContinue | Select-Object -First 1
L ("ngspice_con: " + $ngexe.FullName)

# 2. Qucs-S portable package (official GitHub release, no admin rights needed)
$qz = "$tools\Qucs-S-26.1.1-win64.zip"
if (-not (Test-Path $qz)) {
  L "downloading Qucs-S portable"
  & curl.exe -L --fail -o $qz "https://github.com/ra3xdh/qucs_s/releases/download/26.1.1/Qucs-S-26.1.1-win64.zip" 2>&1 | Out-Null
}
L ("qucs zip size: " + (Get-Item $qz -ErrorAction SilentlyContinue).Length)
Expand-Archive -Path $qz -DestinationPath "$tools\qucs-s" -Force -ErrorAction SilentlyContinue
$qexe = Get-ChildItem "$tools\qucs-s" -Recurse -Filter qucs-s.exe -ErrorAction SilentlyContinue | Select-Object -First 1
L ("qucs-s.exe: " + $qexe.FullName)

# 3. Python virtual environment
if (-not (Test-Path "$root\.venv\Scripts\python.exe")) { & C:\Python312\python.exe -m venv "$root\.venv" 2>&1 | ForEach-Object { L $_ } }
& "$root\.venv\Scripts\python.exe" -m pip install --upgrade pip 2>&1 | Select-Object -Last 1 | ForEach-Object { L $_ }
& "$root\.venv\Scripts\python.exe" -m pip install numpy scipy matplotlib pandas scikit-learn 2>&1 | Select-Object -Last 2 | ForEach-Object { L $_ }
& "$root\.venv\Scripts\python.exe" -c "import numpy,scipy,matplotlib,pandas,sklearn;print('versions',numpy.__version__,scipy.__version__,matplotlib.__version__,pandas.__version__,sklearn.__version__)" 2>&1 | ForEach-Object { L $_ }

# 4. Validate ngspice with a physical circuit (voltage divider, 5V, 10k/4.7k)
$net = "$tools\validate_divider.cir"
@"
* validation: 5 V source, 10k over 4.7k divider
V1 in 0 DC 5
R1 in mid 10k
R2 mid 0 4.7k
.op
.control
op
print v(mid) i(V1)
.endc
.end
"@ | Out-File $net -Encoding ascii
if ($ngexe) { & $ngexe.FullName -b $net 2>&1 | ForEach-Object { L "NG> $_" } }

# 5. Split the source tarball into small chunks for transfer to the cloud build machine
New-Item -ItemType Directory -Force "$tools\split" | Out-Null
& 7z a -mx0 -v2m "$tools\split\ngsrc.7z" "$tools\ngspice-47.tar.gz" -y 2>&1 | Select-Object -Last 1 | ForEach-Object { L $_ }
L "SETUP DONE"
