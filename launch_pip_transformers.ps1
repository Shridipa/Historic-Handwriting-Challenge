$root = $PSScriptRoot
Set-Location $root
$pyargs = @('-m', 'pip', 'install', 'transformers')
$p = Start-Process -FilePath "$root\.venv\Scripts\python.exe" `
  -ArgumentList $pyargs `
  -WorkingDirectory $root -WindowStyle Hidden `
  -RedirectStandardOutput "$root\runs\pip_transformers.log" `
  -RedirectStandardError "$root\runs\pip_transformers.err.log" `
  -PassThru
Write-Output "started pip install transformers pid $($p.Id)"
