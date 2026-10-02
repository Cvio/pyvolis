# Downloads a model from Hugging Face into models\asr\, models\mt\ or models\tts\, in a
# folder named after the model. This, like setup.ps1, uses the internet;
# pyvolis itself never runs it and never goes online.
#
#     .\fetch-model.ps1 openai/whisper-large-v3-turbo -Role asr
#     .\fetch-model.ps1 unsloth/gemma-3-4b-it-GGUF -Role mt -Include "*Q4_K_M.gguf"
#     .\fetch-model.ps1 oddadmix/whisper-large-v3-turbo-arabic-dialectal -Role asr -Name whisper-large-v3-turbo-arabic-dialectal-hf
#                                     (-Name when the model's own name is already a folder)
#     .\fetch-model.ps1 csukuangfj/vits-piper-ar_JO-kareem-medium -Role tts
#                                     (a Piper voice: engine.toml is written, and a raw Piper
#                                     model is made loadable; check the engine.toml afterwards)
#     .\fetch-model.ps1 -Login        once, for gated models (a token from
#                                     https://huggingface.co/settings/tokens)
#
# ASR folders get weights (safetensors when the repo has them), configs,
# tokenizer files and the model card. For a GGUF repo, -Include picks the
# quantisation when there is more than one. The work is done by
# scripts\fetch_model.py; its token and download cache stay in .uv\hf\,
# which is never part of the built app.

param(
    [Parameter(Position = 0)][string]$Id,
    [ValidateSet("asr", "mt", "tts")][string]$Role,
    [string[]]$Include,
    [string]$Name,
    [switch]$Login
)
$repo = $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "STOP: there is no environment at $(Join-Path $repo '.venv'). Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Not "$ErrorActionPreference = Stop": Windows PowerShell 5.1 turns anything a
# program writes to stderr (Hugging Face's progress and warnings) into an error
# when output is redirected, which would stop a working download
# (model-converter's README). The exit code decides instead.
$argsList = @((Join-Path $repo "scripts\fetch_model.py"))
if ($Login) {
    $argsList += "--login"
} else {
    if (-not $Id -or -not $Role) {
        Write-Host "STOP: give a model id and -Role asr, mt or tts, e.g. .\fetch-model.ps1 openai/whisper-large-v3-turbo -Role asr" -ForegroundColor Red
        exit 1
    }
    $argsList += @($Id, "--role", $Role)
    foreach ($pattern in $Include) { $argsList += @("--include", $pattern) }
    if ($Name) { $argsList += @("--name", $Name) }
}
& $python @argsList
exit $LASTEXITCODE
