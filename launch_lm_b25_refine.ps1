$root = $PSScriptRoot
Set-Location $root
$pyargs = @(
  'src/beam_search.py',
  '--checkpoint', 'runs/adaptive_1024_bgcontrast_h80/checkpoints/best.pt',
  '--mode', 'val',
  '--beams', '0',
  '--lm-path', 'runs/lm/fusion_lm.pkl',
  '--lm-weights', '0.15,0.25,0.3',
  '--lm-beam', '25',
  '--out', 'runs/beam_search/lm_b25_refine'
)
$p = Start-Process -FilePath "$root\.venv\Scripts\python.exe" `
  -ArgumentList $pyargs `
  -WorkingDirectory $root -WindowStyle Hidden `
  -RedirectStandardOutput "$root\runs\beam_lm_b25_refine.log" `
  -RedirectStandardError "$root\runs\beam_lm_b25_refine.err.log" `
  -PassThru
Write-Output "started beam lm b25 refine pid $($p.Id)"
