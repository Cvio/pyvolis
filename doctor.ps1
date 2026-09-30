# Checks this machine's pyvolis environment: native files present and
# unaltered, sherpa-onnx on the right onnxruntime.dll, torch on the GPU,
# llama-cpp-python, and the CUDA DLLs each library loaded. setup.ps1 runs it;
# run it again whenever something that used to work stops.
#
#     .\doctor.ps1
#
# Writes logs\doctor.json. See pyvolis\doctor.py for what each check means.

$repo = $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "STOP: there is no environment at $(Join-Path $repo '.venv'). Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}
Push-Location $repo
try {
    & $python -m pyvolis.doctor
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
