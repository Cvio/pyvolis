# Builds the copy-to-run folder (P12): dist\pyvolis\, with pyvolis.exe, the
# libraries it needs, and config\, prompts\, models\ and the settings beside
# it. Zip that folder, unzip it on another Windows PC, double-click
# pyvolis.exe: no Python and no internet needed there.
#
#     .\build.ps1               the whole folder, models included
#     .\build.ps1 -NoModels     the program only (models\ holds just its README files)
#
# Run .\setup.ps1 first. Uses PyInstaller (one-folder mode, pyvolis.spec),
# installed from uv.lock's "build" group. The GPU build of the translator is
# what ships when wheels\cuda\ has one (build-llama.ps1 -Cuda); it needs an
# NVIDIA GPU on the other PC.

param([switch]$NoModels)
$repo = $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "STOP: there is no environment at $(Join-Path $repo '.venv'). Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}
function Fail($message) { Write-Host "STOP: $message" -ForegroundColor Red; exit 1 }

$env:UV_CACHE_DIR = Join-Path $repo ".uv\cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $repo ".uv\python"
$env:UV_LINK_MODE = "copy"
$env:UV_PYTHON_PREFERENCE = "only-managed"

Push-Location $repo
try {
    Write-Host "== PyInstaller (uv.lock's build group)"
    # --inexact: add the build group without removing anything setup.ps1 installed.
    & uv sync --locked --group build --inexact
    if ($LASTEXITCODE -ne 0) { Fail "uv sync --group build failed" }
    $cudaWheel = Get-ChildItem (Join-Path $repo "wheels\cuda") -Filter "llama_cpp_python-*.whl" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($cudaWheel) {
        # uv sync puts the CPU wheel from uv.lock back; the GPU one is what ships.
        & uv pip install --python $python --reinstall --no-deps $cudaWheel.FullName
        if ($LASTEXITCODE -ne 0) { Fail "installing $($cudaWheel.FullName) failed" }
    }

    Write-Host "`n== Building dist\pyvolis\ (a few minutes)"
    $log = Join-Path $repo "logs\pyinstaller.log"
    New-Item -ItemType Directory -Force (Join-Path $repo "logs") | Out-Null
    & $python -m PyInstaller --noconfirm --clean --distpath dist --workpath build pyvolis.spec *> $log
    if ($LASTEXITCODE -ne 0) { Fail "PyInstaller failed; see $log" }

    Write-Host "`n== Settings, prompts and models beside the program"
    $layout = @((Join-Path $repo "scripts\layout_build.py"), (Join-Path $repo "dist\pyvolis"))
    if ($NoModels) { $layout += "--no-models" }
    & $python @layout
    if ($LASTEXITCODE -ne 0) { Fail "laying out dist\pyvolis\ failed" }

    Write-Host "`n== Checking the built folder (pyvolis.exe --doctor)"
    & (Join-Path $repo "dist\pyvolis\pyvolis.exe") --doctor
    if ($LASTEXITCODE -ne 0) { Fail "the built folder failed its checks (above)" }
}
finally {
    Pop-Location
}
Write-Host "`nBuilt: $(Join-Path $repo 'dist\pyvolis')" -ForegroundColor Green
