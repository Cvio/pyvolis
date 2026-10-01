# pyvolis

Offline speech-to-speech translation, in Python: the same app as
[Rust volis](../volis/README.md), plus models you can drop in straight from Hugging Face
without converting them. It never uses the internet. Everything it needs lives in this folder.

Status: milestone P8 of `pyvolis-build.md`. pyvolis has its window: live translation from the
microphone, taking turns or listening continuously, with voice output; file mode with a
timeline and export; text shown while you speak, and translation that knows what was said
before. Pairing and the shared machine arrive at P9 and P10.

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
languages they know). **Start** then waits for you, in one of two modes, which you can switch
while it runs:

- **Take turns:** the microphone is closed until you press Space; press again to finish (or
  choose "Hold Space while speaking"). Everything said in the turn is translated and spoken
  when it ends. Taking a new turn cuts off a reply that is still being spoken. The key works
  while the window has focus, and never also clicks the button that has focus.
- **Listen continuously:** hands-free; every pause ends an utterance.

Each sentence and its translation is shown, and the translation spoken. Half-duplex mutes the
microphone while pyvolis speaks; turn it off only with headphones. `--listen` on the command
line always listens continuously.

**File** mode: open a recording (File > Open, or drop it on the window), choose Real time or
Fast, and Start. Each row is a sentence; clicking a row plays that stretch of the recording.
Pause and Stop work at any point, and **Export** writes the folder described below. Arabic and
Persian are laid out right to left.

Three checkboxes change how it works:

- **Show text while speaking (streaming):** in continuous and file mode, the utterance is
  transcribed about once a second as it grows. Words that two passes agree on are committed
  (and whole sentences translated at once); the current guess after them is shown lighter and
  may change. It roughly doubles the recognition work.
- **Translate with the earlier sentences as context:** each sentence is translated knowing the
  last four and their translations, so "her", "it" and the like come out right.
- **Revise earlier translations when what follows changes them:** after each sentence the last
  three are translated again together, and a short earlier sentence whose translation changes
  is replaced ("He fell." becomes "It fell." once "The system isn't responding" is heard). The
  row is highlighted briefly, marked "revised", and keeps the earlier wording in its tooltip.
  A sentence that has been spoken aloud is never revised, so this is for captions and files,
  with "Speak translations" off. It costs one more translation per sentence.
- **Join short fragments to what follows:** a few words with no full stop wait up to 1.5 s
  for the rest before they are translated.

**Glossary:** names and terms to keep exactly as they are, separated by commas. It applies
from the next sentence and isn't saved.

`--report --load` loads every model in turn and prints where it runs and the memory it takes.

## Translating a file

```powershell
.\.venv\Scripts\python.exe -m pyvolis --file talk.m4a --from es-MX --to en --fast
```

WAV, MP3, M4A, FLAC, OGG and Opus open as they are. The file goes through the same pipeline as
the microphone. `--asr <folder>` and `--mt <id>` pick the models (as `--report` lists them),
`--fast` runs as fast as the models allow (otherwise at playing speed), and `--export <dir>`
says where to write; the default is `exports\<file>-<date>\`. `--streaming` /
`--no-streaming`, `--context off|carry|revision`, `--no-hold` and `--glossary "Name, Term"` override
the settings for one run. The export holds:

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
sparse = true          # a word or two for several seconds of "speech"
min_words_per_second = 0.33
sparse_min_seconds = 3.0

[asr]
streaming = false   # show text while speaking (continuous and file mode)
interval_s = 1.0    # how often the growing utterance is transcribed

[context]
mode = "carry"      # "off" = each sentence alone, as Rust volis does; "revision" = carry,
                    # and earlier translations are replaced when later sentences change them
sentences = 4       # how many earlier sentences, at most
token_budget = 400  # and never more than this many tokens of them
revise_sentences = 3     # revision: how many are translated again together
revise_max_age_s = 30.0  # revision: never a sentence that ended longer ago than this
revise_max_words = 8     # revision: nor one longer than this

[fragments]
hold = true         # join a short fragment to what follows
min_words = 4       # shorter than this, with no final punctuation, is a fragment
hold_ms = 1500

[translate]
model = ""          # as --report lists it; "" = the .gguf at the top of models\mt\
prompt = "default"  # a file in prompts\; "rust" = exactly what Rust volis sends
device = "auto"     # "auto" = the GPU if the GPU build is installed, "cpu", or "cuda"
```

```powershell
.\.venv\Scripts\python.exe -m pyvolis --devices        # names for [audio] in volis.toml
.\.venv\Scripts\python.exe -m pyvolis --listen --seconds 30 --compare
.\.venv\Scripts\python.exe -m pyvolis --translate "¿Dónde está la estación?" --from es --to en
.\.venv\Scripts\python.exe scripts\transcribe.py es_419 ar_eg   # every model on the fixtures
.\tests\fetch-fixtures.ps1                               # test clips (development)
.\.venv\Scripts\python.exe scripts\vad_cuts.py --wav   # cut points with and without pre-roll
.\.venv\Scripts\python.exe scripts\vad_cuts.py --mic 20 --wav
```

## The translator build

llama-cpp-python is compiled here, once, by `.\build-llama.ps1` (needs Visual Studio Build
Tools), and the wheel is kept in `wheels\`, so `setup.ps1` needs no compiler. It is a CPU
build, as Rust volis's translator is.

For translation on the GPU (several times faster), build the GPU wheel on the machine:
install NVIDIA's CUDA Toolkit 12.8 or 12.9 (not 13; Custom install, driver components
unticked), then `.\build-llama.ps1 -Cuda` and `.\setup.ps1`. The wheel goes in `wheels\cuda\`
(344 MB, not committed), and running it needs no toolkit: it uses the CUDA files PyTorch ships.
`.\doctor.ps1` then reports "GPU offload: yes".

`scripts\obey_check.py` measures whether a translator and prompt translate questions and
requests or answer them.

## Checking against Rust volis

Copy `machine.example.yaml` to `machine.yaml` and point it at `volis.exe`, then:

```powershell
.\.venv\Scripts\python.exe parity\report.py
```
