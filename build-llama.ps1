# Builds the llama-cpp-python wheel pyvolis uses, into wheels\. Run it only to
# change the version or the build options; setup.ps1 installs the wheel that
# is already in wheels\ and needs no compiler.
#
#     .\build-llama.ps1            CPU build (what wheels\ holds now)
#
# Why a build: llama-cpp-python publishes Windows wheels only up to 0.3.19
# (CPU) and 0.3.4 (CUDA 12.4, which doesn't run on an RTX 5090, and predates
# Qwen3). 0.3.35 is published as source only.
#
# Needs Visual Studio Build Tools (C++), which include CMake, and the internet
# to fetch the source from PyPI. GGML_NATIVE=OFF keeps the build portable to
# other CPUs (copy-to-run); OpenMP and curl are off, as in Rust volis's build.

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$version = "0.3.35"
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
if (-not $vs) { Write-Host "STOP: Visual Studio Build Tools (C++) are not installed." -ForegroundColor Red; exit 1 }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$cmake = Join-Path $vs "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
$work = Join-Path $repo ".uv\build"
New-Item -ItemType Directory -Force $work | Out-Null
$sdist = Join-Path $work "llama_cpp_python-$version.tar.gz"
if (-not (Test-Path $sdist)) {
    $url = (Invoke-RestMethod "https://pypi.org/pypi/llama-cpp-python/$version/json").urls |
        Where-Object { $_.filename -like "*.tar.gz" } | Select-Object -First 1 -ExpandProperty url
    Invoke-WebRequest $url -OutFile $sdist
}
$cmd = @"
call "$vcvars" >nul && set "PATH=$cmake;%PATH%" && set "CMAKE_ARGS=-DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF" && set "UV_CACHE_DIR=$repo\.uv\cache" && uv build --wheel --python "$repo\.venv\Scripts\python.exe" --out-dir "$repo\wheels" "$sdist"
"@
cmd /c $cmd
exit $LASTEXITCODE
