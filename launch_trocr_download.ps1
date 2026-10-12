$root = $PSScriptRoot
Set-Location $root
$pyargs = @('src/trocr_download.py')
$p = Start-Process -FilePath "$root\.venv\Scripts\python.exe" `
  -ArgumentList $pyargs `
  -WorkingDirectory $root -WindowStyle Hidden `
  -RedirectStandardOutput "$root\runs\trocr_download.log" `
  -RedirectStandardError "$root\runs\trocr_download.err.log" `
  -PassThru
Write-Output "started trocr download pid $($p.Id)"
