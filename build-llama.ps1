# Builds the llama-cpp-python wheel pyvolis uses, into wheels\. Run it only to
# change the version or the build options; setup.ps1 installs the wheel that
# is already in wheels\ and needs no compiler.
#
#     .\build-llama.ps1            CPU build, into wheels\
#     .\build-llama.ps1 -Cuda      GPU build, into wheels\cuda\ (needs the CUDA Toolkit 12.x:
#                                  12.x so it shares the CUDA files PyTorch already ships)
#
# Why a build: llama-cpp-python publishes Windows wheels only up to 0.3.19
# (CPU) and 0.3.4 (CUDA 12.4, which doesn't run on an RTX 5090, and predates
# Qwen3). 0.3.35 is published as source only.
#
# Needs Visual Studio Build Tools (C++), which include CMake, and the internet
# to fetch the source from PyPI. GGML_NATIVE=OFF keeps the build portable to
# other CPUs (copy-to-run); OpenMP and curl are off, as in Rust volis's build.

param([switch]$Cuda)
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
$out = Join-Path $repo "wheels"
$gpu = "OFF"
$cudaFlags = ""
$extra = ""
if ($Cuda) {
    $toolkits = "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA"
    $toolkit = Get-ChildItem $toolkits -Directory -Filter "v12.*" -ErrorAction SilentlyContinue |
        Sort-Object Name | Select-Object -Last 1
    if (-not $toolkit) { Write-Host "STOP: no CUDA Toolkit 12.x in $toolkits" -ForegroundColor Red; exit 1 }
    $out = Join-Path $out "cuda"
    $gpu = "ON"
    # CUDA 12.9's headers refuse compilers newer than Visual Studio 2022; the flag lifts that
    # check. The result is tested against the CPU build (parity/translate.py).
    $cudaFlags = "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
    # Ninja, because the Visual Studio generator needs the toolkit's MSBuild integration.
    $ninja = Join-Path $vs "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
    $extra = "set `"CUDA_PATH=$($toolkit.FullName)`" && set `"PATH=$($toolkit.FullName)\bin;$ninja;!PATH!`" && set `"CMAKE_GENERATOR=Ninja`" && "
}
$cmd = @"
call "$vcvars" >nul && set "PATH=$cmake;!PATH!" && $extra set "CMAKE_ARGS=-DGGML_NATIVE=OFF -DGGML_CUDA=$gpu $cudaFlags -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF" && set "UV_CACHE_DIR=$repo\.uv\cache" && uv build --wheel --python "$repo\.venv\Scripts\python.exe" --out-dir "$out" "$sdist"
"@
cmd /v:on /c $cmd
exit $LASTEXITCODE
