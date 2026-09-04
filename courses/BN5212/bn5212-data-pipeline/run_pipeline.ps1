param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PipelineArgs)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $PipelinePython = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
    & $PipelinePython run_pipeline.py @PipelineArgs --test-loader
    if ($LASTEXITCODE -ne 0) { throw "Pipeline failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
