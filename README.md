# pyvolis

Offline speech-to-speech translation, in Python: the same app as
[Rust volis](../volis/README.md), plus models you can drop in straight from Hugging Face
without converting them. It never uses the internet. Everything it needs lives in this folder.

Status: milestone P1 of `pyvolis-build.md`. So far pyvolis finds and lists models, captures
from the microphone or a file, and cuts speech into utterances; it doesn't recognise or
translate yet.

## Setup (development)

Needs 64-bit Windows, [uv](https://docs.astral.sh/uv/) 0.11.x (`winget install astral-sh.uv`),
and an internet connection **for setup only**.

```powershell
.\setup.ps1
```

This puts Python 3.12, every package (PyTorch with CUDA 12.8 is several GB) and uv's cache
inside this folder (`.uv\`, `.venv\`), then runs `.\doctor.ps1`. Nothing is installed anywhere
else. If security software is removing DLLs, give it an exclusion for this folder and run
`.\setup.ps1 -Reinstall`.

Then:

```powershell
.\.venv\Scripts\python.exe -m pyvolis --report
```

## Models

`models\` beside the app, the same layout as Rust volis:

| Folder | What goes there |
|---|---|
| `models\vad\silero_vad.onnx` | the voice activity detector |
| `models\asr\<name>\` | a recognizer: a Rust volis folder with `engine.toml`, or a Hugging Face download (Whisper, wav2vec2/MMS, Cohere Transcribe and other speech models `transformers` supports) |
| `models\tts\<name>\` | a Piper voice with `engine.toml`, as in Rust volis |
| `models\mt\` | translators: one `.gguf` at the top (the one Rust volis uses too), and any others **one folder per model** (Rust volis refuses two `.gguf` files at the top) |

To add a model from Hugging Face:

```powershell
.\fetch-model.ps1 openai/whisper-large-v3-turbo -Role asr
.\fetch-model.ps1 unsloth/gemma-3-4b-it-GGUF -Role mt -Include "*Q4_K_M.gguf"
```

It downloads into a folder named after the model and never needs `engine.toml`. For a gated
model, accept its terms on the model's page first, then run `.\fetch-model.ps1 -Login` once.

A downloaded folder may hold an optional `pyvolis.toml` to override what was detected:

```toml
name = "Whisper large-v3-turbo Spanish (fine-tune)"
languages = ["es"]
varieties = ["es-MX"]
device = "cuda"          # "cuda" or "cpu"
dtype = "float16"        # transformers only
trust_remote_code = false
```

Languages come from the model card's `language:` when there is one. Otherwise the model is
listed as "languages unknown" and offered for every language.

## Settings

`volis.toml` is shared with Rust volis, same keys and meanings. Settings only pyvolis has go in
`pyvolis.toml` beside it (Rust volis refuses sections it doesn't know). All optional:

```toml
[vad]
pre_roll_ms = 600   # audio kept from before each utterance; 0 = off
```

```powershell
.\.venv\Scripts\python.exe -m pyvolis --devices        # names for [audio] in volis.toml
.\tests\fetch-fixtures.ps1                               # test clips (development)
.\.venv\Scripts\python.exe scripts\vad_cuts.py --wav   # cut points with and without pre-roll
.\.venv\Scripts\python.exe scripts\vad_cuts.py --mic 20 --wav
```

## Checking against Rust volis

Copy `machine.example.yaml` to `machine.yaml` and point it at `volis.exe`, then:

```powershell
.\.venv\Scripts\python.exe parity\report.py
```
