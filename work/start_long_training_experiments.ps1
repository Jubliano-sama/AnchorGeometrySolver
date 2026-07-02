param(
    [string]$Root = "C:\Users\tommi\Documents\Codex\2026-06-26\vkabbv-smartclicker-gui-https-github-com"
)

$ErrorActionPreference = "Stop"
Set-Location $Root

$out = Resolve-Path ".\outputs"
$python = Resolve-Path ".\.venv_cuda_seed\Scripts\python.exe"
$script = Resolve-Path ".\work\anchor_solver_ml_distance_completion.py"

$experiments = @(
    @{
        Name = "bigA_h224_l8"
        Hidden = "224"
        Layers = "8"
        Batch = "32"
        LR = "0.0015"
        WD = "0.0001"
        EDM = "0.035"
        Seed = "2026062701"
        Dropout = "0.035"
    },
    @{
        Name = "bigB_h192_l10"
        Hidden = "192"
        Layers = "10"
        Batch = "32"
        LR = "0.0016"
        WD = "0.00015"
        EDM = "0.045"
        Seed = "2026062702"
        Dropout = "0.035"
    }
)

foreach ($exp in $experiments) {
    $prefix = "anchor_solver_ml_distance_completion_$($exp.Name)"
    $stdout = Join-Path $out "$prefix.train.log"
    $stderr = Join-Path $out "$prefix.train.err.log"
    $args = @(
        $script,
        "--device", "cuda",
        "--amp",
        "--epochs", "60",
        "--steps-per-epoch", "120",
        "--batch-size", $exp.Batch,
        "--hidden", $exp.Hidden,
        "--layers", $exp.Layers,
        "--dropout", $exp.Dropout,
        "--lr", $exp.LR,
        "--weight-decay", $exp.WD,
        "--grad-clip", "1.5",
        "--random-fraction", "0.55",
        "--eval-cases", "100",
        "--solver-iterations", "80",
        "--polish-iterations", "80",
        "--weak-polish-iterations", "80",
        "--predicted-sigma", "0.52",
        "--predicted-sigma-slope", "0.75",
        "--edm-weight", $exp.EDM,
        "--closest-predicted-pairs-per-anchor", "5",
        "--log-every", "20",
        "--seed", $exp.Seed,
        "--prefix", $prefix
    )
    $process = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Write-Output "$($exp.Name) pid=$($process.Id) stdout=$stdout stderr=$stderr"
}
