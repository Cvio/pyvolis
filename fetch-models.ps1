# Downloads the models listed in MODELS.md into models\, each in its right
# folder with the settings files pyvolis needs. This, like setup.ps1, uses the
# internet; pyvolis itself never runs it and never goes online.
#
#     .\fetch-models.ps1 -List             what is in place and what would be downloaded; downloads nothing
#     .\fetch-models.ps1                   the models worth having, and the ones to measure on a larger GPU
#     .\fetch-models.ps1 -Group use        only the ones worth having here (use | bigger | untested | tested | all)
#     .\fetch-models.ps1 -Only Qwen3-ASR   only the entries whose name contains this
#     .\fetch-models.ps1 -Yes              don't ask before downloading
#
# What is already in place is left alone, and an engine.toml or pyvolis.toml
# that exists is never overwritten. Gated models (Cohere Transcribe) need
# their terms accepted on huggingface.co and .\fetch-model.ps1 -Login first.
# The list itself, with every source, is at the top of scripts\fetch_models.py.

param(
    [string[]]$Group,
    [string[]]$Only,
    [switch]$List,
    [switch]$Yes
)
$repo = $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "STOP: there is no environment at $(Join-Path $repo '.venv'). Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Not "$ErrorActionPreference = Stop": see fetch-model.ps1.
$argsList = @((Join-Path $repo "scripts\fetch_models.py"))
foreach ($g in $Group) { $argsList += @("--group", $g) }
foreach ($o in $Only) { $argsList += @("--only", $o) }
if ($List) { $argsList += "--list" }
if ($Yes) { $argsList += "--yes" }
& $python @argsList
exit $LASTEXITCODE
