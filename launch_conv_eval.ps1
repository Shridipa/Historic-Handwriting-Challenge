$root = $PSScriptRoot
Set-Location $root
$pyargs = @(
  'src/beam_search.py',
  '--checkpoint', 'runs/convergence_h80_cont/checkpoints/best.pt',
  '--mode', 'val',
  '--beams', '0',
  '--lm-path', 'runs/lm/fusion_lm.pkl',
  '--lm-weights', '0.25',
  '--lm-beam', '25',
  '--out', 'runs/beam_search/converged_cont'
)
$p = Start-Process -FilePath "$root\.venv\Scripts\python.exe" `
  -ArgumentList $pyargs `
  -WorkingDirectory $root -WindowStyle Hidden `
  -RedirectStandardOutput "$root\runs\beam_conv_cont.log" `
  -RedirectStandardError "$root\runs\beam_conv_cont.err.log" `
  -PassThru
Write-Output "started converged-ckpt beam+LM eval pid $($p.Id)"
