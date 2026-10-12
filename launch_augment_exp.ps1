# Phase 7 one-variable experiment: augmentation only.
# Same init/protocol as convergence_h80_cont (bgcontrast, NOT sharp), single variable = --augment.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File launch_augment_exp.ps1
$root = $PSScriptRoot
$exe = Join-Path $root '.venv\Scripts\python.exe'
$train = 'src/train_crnn.py --experiment augment_all_h80 --output-dir runs/augment_all_h80 --epochs 11 --batch-size 6 --lr 5e-4 --height 80 --adaptive-width --max-width 1024 --val-split 0.1 --init-checkpoint runs/convergence_h80/checkpoints/best.pt --patience 3 --preprocess-variant bgcontrast --augment rotate,shear,stretch,penweight,blur,noise,contrast'
$cmd = 'cmd /c ""' + $exe + '" ' + $train + ' > "' + $root + '\runs\augment_all_h80_train.log" 2>&1"'
$null = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = $cmd
  CurrentDirectory = $root
}
Write-Output "WMI-launched augment_all_h80 training"
