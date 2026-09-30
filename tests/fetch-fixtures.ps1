# Downloads the test clips into tests\fixtures\ (not committed): the first 10
# FLEURS test clips for Spanish, Persian, Arabic and English, with their
# reference transcripts. A development tool that uses the internet; pyvolis
# itself never does. Your own recordings go in tests\fixtures\user\.
#
#     .\tests\fetch-fixtures.ps1

$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "STOP: there is no environment at $(Join-Path $repo '.venv'). Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}
& $python (Join-Path $repo "scripts\fetch_fixtures.py")
exit $LASTEXITCODE
