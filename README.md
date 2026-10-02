# pyvolis

Offline speech-to-speech translation, in Python: the same app as
[Rust volis](../volis/README.md), plus models you can drop in straight from Hugging Face
without converting them. It never uses the internet. Everything it needs lives in this folder.

Status: milestone P11 of `pyvolis-build.md`. pyvolis has its window: live translation from the
microphone, taking turns or listening continuously, with voice output; file mode with a
timeline and export; text shown while you speak, and translation that knows what was said
before; two PCs can pair, with pyvolis or Rust volis at either end; and two people can share
one machine, a key each.

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
| `models\asr\<name>\` | a recognizer: a Rust volis folder with `engine.toml`; a Hugging Face download (Whisper, wav2vec2/MMS, Cohere Transcribe and other speech models `transformers` supports); a GGUF speech model with its audio encoder (`mmproj-*.gguf`: Qwen3-ASR, Gemma 4, Voxtral); or a LoRA adapter folder (`adapter_config.json`) beside its base |
| `models\tts\<name>\` | a Piper voice with `engine.toml`, as in Rust volis |
| `models\mt\` | translators: one `.gguf` at the top (the one Rust volis uses too), and any others **one folder per model** (Rust volis refuses two `.gguf` files at the top): a `.gguf`, a safetensors language model, or a LoRA adapter `.gguf` with a `pyvolis.toml` naming its base |

To add a model from Hugging Face:

```powershell
.\fetch-model.ps1 openai/whisper-large-v3-turbo -Role asr
.\fetch-model.ps1 unsloth/gemma-3-4b-it-GGUF -Role mt -Include "*Q4_K_M.gguf"
.\fetch-model.ps1 ggml-org/Qwen3-ASR-0.6B-GGUF -Role asr -Include "*Q8_0.gguf"   # the model and its audio encoder
.\fetch-model.ps1 Qwen/Qwen3-0.6B -Role mt                                       # a safetensors translator
.\fetch-model.ps1 csukuangfj/vits-piper-ar_JO-kareem-medium -Role tts            # a Piper voice
```

**To get every model that has been tested and is worth having,** in the right folders with
their settings files, run `.\fetch-models.ps1` (`-List` first, to see what it would download
and how big it is). [MODELS.md](MODELS.md) has the list, each model's scores, and what each
kind of folder needs.

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

**LoRA adapters,** to try a tune without merging or converting it:

- *For a recognizer:* put the adapter folder (`adapter_config.json`, `adapter_model.safetensors`)
  in `models\asr\` beside its base model's folder. It appears as its own recognizer. The base
  is found by name (`openai/whisper-small` means the folder `whisper-small`); if yours is
  named differently, say `base = "<folder>"` in the adapter folder's `pyvolis.toml`.
- *For a GGUF translator:* put the adapter `.gguf` (from llama.cpp's `convert_lora_to_gguf.py`)
  in its own folder in `models\mt\` with a `pyvolis.toml`:

  ```toml
  base = "qwen3-1.7b-q4_k_m.gguf"   # the translator it is for, as --report lists it
  scale = 1.0                        # how strongly it is applied
  ```

**A GGUF speech model** needs both its `.gguf` and its `mmproj-*.gguf` in one folder. A model
that only transcribes (Qwen3-ASR) is given the audio alone; a general model that also hears
(Gemma 4, Voxtral) is asked to transcribe. To change the wording, or to say which kind a
model is, put `prompt = "..."` in the folder's `pyvolis.toml` (`""` = the audio alone,
`{language}` = the language's name).

## The window

```powershell
.\.venv\Scripts\python.exe -m pyvolis
```

Pick what is spoken and what to translate into; the recognizer list is ordered for that
language (tuned for the variety, general, other varieties, then models that don't say which
languages they know). **Start** then waits for you, in one of three modes, which you can switch
while it runs:

- **Take turns:** the microphone is closed until you press Space; press again to finish (or
  choose "Hold Space while speaking"). Everything said in the turn is translated and spoken
  when it ends. Taking a new turn cuts off a reply that is still being spoken. The key works
  while the window has focus, and never also clicks the button that has focus.
- **Listen continuously:** hands-free; every pause ends an utterance.
- **Shared machine:** two people who speak different languages, one PC, a key each (the left
  and right arrows; `[shared] left_key` / `right_key` in `volis.toml`). Press your key, speak,
  press it again: your words are spoken in the other person's language. Each side has its own
  language (or regional variety), its own recognizer ("Heard by": any model that covers the
  language, a downloaded one included) and the voice its words are spoken in ("Spoken by": a
  voice in the other person's language). One person at a time: during a turn the other key
  does nothing, and while a reply is worked on or spoken neither does. **Escape** cancels a
  turn or silences what it produced. The active side fills with colour and a heavy border, and
  a side that can't take a turn says why in its column. The arrow keys belong to a text box
  while one has the focus; click anywhere else to give them back. Not while paired.

Each sentence and its translation is shown, and the translation spoken. Half-duplex mutes the
microphone while pyvolis speaks; turn it off only with headphones. `--listen` on the command
line always listens continuously.

**Pair with another PC:** two PCs, one conversation. Each translates what its own person says
and sends only the text; the other PC shows it and speaks it. Tick the box on both, press
Start on both, then on either one type the address the other shows under "This PC" (or pick it
from "Found") and press Connect. It works with Rust volis at the other end. Taking turns, a
press of the turn key asks the other PC for the floor and the microphone opens only when it
answers; the indicator says when the other person is talking. Listening continuously while
paired needs headsets, and the window says so in red for as long as it is true. If the link
drops, both ends say why and the floor is released. Only IP addresses are accepted: a name
would have to be looked up, and pyvolis never does that. Revision is off while paired.

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
  A sentence that has been spoken aloud is never revised, so on its own this is for captions
  and files, with "Speak translations" off. It costs one more translation per sentence.
- **Wait for the next sentence before speaking a short one:** with revision and "Speak
  translations" both on, a short sentence (8 words or fewer) is not spoken until the next one
  has been heard, so it is spoken as revised. It waits 2 s at most, and not at all at the end
  of a turn; a long sentence never waits. Off by default: speech is never delayed.
- **Add vowel marks to Arabic before it is spoken:** Arabic is written without its short
  vowels, and the voices mispronounce some words without them. On, a small model predicts the
  marks first, as Piper itself does; about 0.3 s more per sentence. It needs
  `models\tashkeel\libtashkeel_model.ort` (`.\fetch-models.ps1 -Only tashkeel`). Only Arabic
  is affected. Off by default.
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
hold_speech = false      # revision with the voice on: a short sentence waits for the next before it is spoken
hold_speech_s = 2.0      # ...this long at most

[tts]
diacritize = false  # Arabic: predict the vowel marks before the voice pronounces the text

[fragments]
hold = true         # join a short fragment to what follows
min_words = 4       # shorter than this, with no final punctuation, is a fragment
hold_ms = 1500

[translate]
model = ""          # as --report lists it; "" = the .gguf at the top of models\mt\
prompt = "default"  # a file in prompts\; "rust" = exactly what Rust volis sends
device = "auto"     # "auto" = the GPU if the GPU build is installed, "cpu", or "cuda"
```

Pairing is set in `volis.toml`, shared with Rust volis: `[peer] enabled`, `listen_addr`
(default `0.0.0.0:47800`; two programs on one PC need different ports), `peer_addr` (the last
address typed), `display_name` (default: the computer's name) and `discovery`.

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
