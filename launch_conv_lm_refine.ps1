$root = $PSScriptRoot
Set-Location $root
$pyargs = @(
  'src/beam_search.py',
  '--checkpoint', 'runs/convergence_h80_cont/checkpoints/best.pt',
  '--mode', 'val',
  '--beams', '0',
  '--lm-path', 'runs/lm/fusion_lm.pkl',
  '--lm-weights', '0.2,0.25,0.3',
  '--lm-beam', '25',
  '--out', 'runs/beam_search/converged_cont'
)
$p = Start-Process -FilePath "$root\.venv\Scripts\python.exe" `
  -ArgumentList $pyargs `
  -WorkingDirectory $root -WindowStyle Hidden `
  -RedirectStandardOutput "$root\runs\beam_conv_lm_refine.log" `
  -RedirectStandardError "$root\runs\beam_conv_lm_refine.err.log" `
  -PassThru
Write-Output "started converged-ckpt lambda refine pid $($p.Id)"
