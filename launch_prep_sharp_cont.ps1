$root = $PSScriptRoot
Set-Location $root
$pyargs = @(
  'src/train_crnn.py',
  '--experiment', 'prep_sharp_h80_cont',
  '--output-dir', 'runs/prep_sharp_h80_cont',
  '--epochs', '10',
  '--batch-size', '6',
  '--lr', '5e-4',
  '--height', '80',
  '--adaptive-width',
  '--max-width', '1024',
  '--val-split', '0.1',
  '--init-checkpoint', 'runs/prep_sharp_h80/checkpoints/best.pt',
  '--patience', '3',
  '--preprocess-variant', 'bgcontrast_sharp'
)
$p = Start-Process -FilePath "$root\.venv\Scripts\python.exe" `
  -ArgumentList $pyargs `
  -WorkingDirectory $root -WindowStyle Hidden `
  -RedirectStandardOutput "$root\runs\prep_sharp_h80_cont_train.log" `
  -RedirectStandardError "$root\runs\prep_sharp_h80_cont_train.err.log" `
  -PassThru
Write-Output "started prep_sharp_h80_cont pid $($p.Id)"
