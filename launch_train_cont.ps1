$root = $PSScriptRoot
Set-Location $root
$args = @(
  'src/train_crnn.py',
  '--experiment', 'convergence_h80_cont',
  '--output-dir', 'runs/convergence_h80_cont',
  '--epochs', '11',
  '--batch-size', '6',
  '--lr', '5e-4',
  '--height', '80',
  '--adaptive-width',
  '--max-width', '1024',
  '--val-split', '0.1',
  '--init-checkpoint', 'runs/convergence_h80/checkpoints/best.pt',
  '--patience', '3',
  '--preprocess-variant', 'bgcontrast'
)
$p = Start-Process -FilePath "$root\.venv\Scripts\python.exe" `
  -ArgumentList $args `
  -WorkingDirectory $root -WindowStyle Hidden `
  -RedirectStandardOutput "$root\runs\convergence_h80_cont_train.log" `
  -RedirectStandardError "$root\runs\convergence_h80_cont_train.err.log" `
  -PassThru
Write-Output "started training pid $($p.Id)"
