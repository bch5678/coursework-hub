# Run every experiment against one frozen dataset run, then build the comparison table.
#
#   .\scripts\run_all_experiments.ps1                      # local synthetic fixture
#   .\scripts\run_all_experiments.ps1 -Preset real -RunDir /srv/derived/bn5212/mortality_v1 `
#       -ClinicalSource /srv/derived/bn5212/clinical_features.csv.gz -Device cuda
#
# The synthetic preset produces no results, only proof that the code path works.

param(
    [ValidateSet("synthetic", "real")]
    [string]$Preset = "synthetic",
    [string]$RunDir,
    [string]$ClinicalSource,
    [string]$Device = "",
    [string]$RunId = "",
    [string]$ConfigDir = "",
    [switch]$CrossValidate,
    [int]$Folds = 5,
    [string[]]$Experiments = @("clinical_only", "cxr_only", "concat_fusion", "metra_joint", "cross_attention")
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$train = Join-Path $root ".venv\Scripts\bn5212-train.exe"
$crossval = Join-Path $root ".venv\Scripts\bn5212-crossval.exe"
$summarize = Join-Path $root ".venv\Scripts\bn5212-summarize.exe"
if (-not (Test-Path $train)) {
    throw "bn5212-train not found. Run: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e `".[test]`""
}
# Cross-validation is the right estimator when a single split holds too few
# events for its metric to carry information; -CrossValidate selects it.
$runner = if ($CrossValidate) { $crossval } else { $train }

$configDir = if ($ConfigDir) { $ConfigDir } elseif ($Preset -eq "synthetic") { "configs\synthetic" } else { "configs" }
if (-not $RunId) { $RunId = (Get-Date -Format "yyyyMMdd-HHmmss") }

if ($Preset -eq "synthetic" -and -not $RunDir -and -not $ConfigDir) {
    $RunDir = "..\bn5212-data-pipeline\demo\png\processed"
    if (-not (Test-Path (Join-Path $RunDir "SUCCESS.json"))) {
        throw "Synthetic fixture missing. Build it first - see docs/USAGE.md, step 1 of the local run."
    }
}
if ($Preset -eq "real" -and -not $RunDir) {
    throw "-RunDir is required for the real preset"
}

Write-Host "preset=$Preset  run_dir=$RunDir  run_id=$RunId" -ForegroundColor Cyan
Write-Host ""

$completed = @()
foreach ($experiment in $Experiments) {
    $config = Join-Path $configDir "$experiment.json"
    if (-not (Test-Path $config)) {
        Write-Host "skip $experiment (no $config)" -ForegroundColor DarkYellow
        continue
    }

    $arguments = @("--config", $config, "--run-dir", $RunDir, "--run-id", $RunId)
    if ($CrossValidate) { $arguments += @("--folds", $Folds) }
    if ($Device) { $arguments += @("--device", $Device) }
    if ($ClinicalSource) {
        $arguments += @("--set", "data.clinical_provider=table",
                        "--set", "data.clinical_source=$ClinicalSource")
    }

    Write-Host "=== $experiment ===" -ForegroundColor Green
    & $runner @arguments
    if ($LASTEXITCODE -ne 0) { throw "$experiment failed with exit code $LASTEXITCODE" }
    $completed += "outputs\$experiment\$RunId"
    Write-Host ""
}

if ($completed.Count -gt 0) {
    & $summarize @completed --output "outputs\comparison-$RunId"
    Write-Host ""
    Write-Host "Comparison table: outputs\comparison-$RunId.md" -ForegroundColor Cyan
    Get-Content "outputs\comparison-$RunId.md"
    Write-Host ""
    Write-Host "These are validation numbers for model selection only." -ForegroundColor DarkGray
    Write-Host "Reported results come from benchmark-evaluation." -ForegroundColor DarkGray
}
