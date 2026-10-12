param(
  [Parameter(Mandatory=$true)][string]$PyArgs,
  [Parameter(Mandatory=$true)][string]$LogPath
)
$root = $PSScriptRoot
$exe = Join-Path $root '.venv\Scripts\python.exe'
$cmd = 'cmd /c ""' + $exe + '" ' + $PyArgs + ' > "' + $LogPath + '" 2>&1"'
$null = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = $cmd
  CurrentDirectory = $root
}
Write-Output "WMI-launched: $cmd"
