# pyvolis

Offline speech-to-speech translation, in Python: the same app as
[Rust volis](../volis/README.md), plus models you can drop in straight from Hugging Face
without converting them. It never uses the internet. Everything it needs lives in this folder.

Status: milestone P5 of `pyvolis-build.md`. pyvolis has its window: live translation from the
microphone with voice output, and file mode with a timeline and export. Turn-taking, pairing
and the shared machine arrive at P6, P9 and P10.

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

## The window

```powershell
.\.venv\Scripts\python.exe -m pyvolis
```

Pick what is spoken and what to translate into; the recognizer list is ordered for that
language (tuned for the variety, general, other varieties, then models that don't say which
languages they know). **Start** listens to the microphone, shows each sentence and its
translation, and speaks the translation. Half-duplex mutes the microphone while pyvolis speaks;
turn it off only with headphones.

**File** mode: open a recording (File > Open, or drop it on the window), choose Real time or
Fast, and Start. Each row is a sentence; clicking a row plays that stretch of the recording.
Pause and Stop work at any point, and **Export** writes the folder described below. Arabic and
Persian are laid out right to left.

`--report --load` loads every model in turn and prints where it runs and the memory it takes.

## Translating a file

```powershell
.\.venv\Scripts\python.exe -m pyvolis --file talk.m4a --from es-MX --to en --fast
```

WAV, MP3, M4A, FLAC, OGG and Opus open as they are. The file goes through the same pipeline as
the microphone. `--asr <folder>` and `--mt <id>` pick the models (as `--report` lists them),
`--fast` runs as fast as the models allow (otherwise at playing speed), and `--export <dir>`
says where to write; the default is `exports\<file>-<date>\`:

| File | What it holds |
|---|---|
| `transcript.txt`, `translation.txt` | one sentence per line, side by side |
| `source.srt`, `translation.srt` | subtitles with timings |
| `events.jsonl` | every event in order; the first line is the full configuration (models and their file hashes, prompt, settings), the last the summary |

If a reference sits beside the file, `talk.m4a.ref.json` with `{"transcript": "...",
"translation": "..."}` (or `talk.m4a.ref.srt` for the transcript), the run ends with the
transcript's CER and the translation's chrF, cleaned as model-bench cleans them.
`scripts\make_fixture_file.py es_419 en` builds such a file from the test clips.

## Settings

`volis.toml` is shared with Rust volis, same keys and meanings. Settings only pyvolis has go in
`pyvolis.toml` beside it (Rust volis refuses sections it doesn't know). All optional:

```toml
[vad]
pre_roll_ms = 600   # audio kept from before each utterance; 0 = off

[guards]            # drop text recognizers invent on silence or noise
vad_probability = true
min_peak_probability = 0.8
repeats = true
stock_phrases = true   # the phrases are in config\hallucinations.toml

[translate]
model = ""          # as --report lists it; "" = the .gguf at the top of models\mt\
prompt = "default"  # a file in prompts\
```

```powershell
.\.venv\Scripts\python.exe -m pyvolis --devices        # names for [audio] in volis.toml
.\.venv\Scripts\python.exe -m pyvolis --listen --seconds 30 --compare
.\.venv\Scripts\python.exe -m pyvolis --translate "¿Dónde está la estación?" --from es --to en
.\.venv\Scripts\python.exe scripts	ranscribe.py es_419 ar_eg   # every model on the fixtures
.\tests\fetch-fixtures.ps1                               # test clips (development)
.\.venv\Scripts\python.exe scripts\vad_cuts.py --wav   # cut points with and without pre-roll
.\.venv\Scripts\python.exe scripts\vad_cuts.py --mic 20 --wav
```

## The translator build

llama-cpp-python is compiled here, once, by `.\build-llama.ps1` (needs Visual Studio Build
Tools), and the wheel is kept in `wheels\`, so `setup.ps1` needs no compiler. It is a CPU
build, as Rust volis's translator is.

## Checking against Rust volis

Copy `machine.example.yaml` to `machine.yaml` and point it at `volis.exe`, then:

```powershell
.\.venv\Scripts\python.exe parity\report.py
```
