# Launch TrOCR val eval (limit 64 as fast first measurement) via CIM detach.
# Usage: powershell -File launch_trocr_eval.ps1
$root = $PSScriptRoot
$exe = Join-Path $root '.venv\Scripts\python.exe'
$cmd = 'cmd /c ""' + $exe + '" src/trocr_eval.py --limit 64 --out runs/pretrained_trocr > "' + $root + '\runs\trocr_val_eval.log" 2>&1"'
$null = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = $cmd
  CurrentDirectory = $root
}
Write-Output "WMI-launched trocr eval"
