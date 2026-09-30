# Instructions: pyvolis, the Python volis with an open model pipeline

> Read pyvolis-build.md and follow it. The Rust volis and model-converter folders are added for reading only. Don't change anything in them. Start with the summary it asks for, then wait for me. Here are the other two folders.
```
/add-dir D:\AI_Data\projects\volis
/add-dir D:\AI_Data\projects\model-converter
```
For Claude Code. Build **pyvolis**, a Python application that does everything the Rust volis does
today, and adds what the Rust version can't:

1. **Drop-in models.** `hf download` a speech recognition or translation model into `models\`,
   and pyvolis runs it. No ONNX conversion.
2. **Streaming translation of an audio file.** Open a recording and watch it transcribed and
   translated as a stream, with translation that uses what came earlier and can revise itself.
   This is the main test of the app's translation ability.
3. **A stronger ASR pipeline:** streaming recognition, no clipped first words, no hallucinated
   text on silence, and punctuation kept.

The user has created a new, empty repo for pyvolis. Work only there.

**The Rust volis stays.** It works very well and the user will keep maintaining it for other uses.
Don't change anything in its repo. Read it, because **for every existing feature, the Rust code is
the specification**.

Read this whole file before writing any code. Then read, in the Rust repo: `CLAUDE.md`, `SPEC.md`,
`ARCHITECTURE.md`, `HANDOFF.md`, `MODELS.md`, `DIALECTS.md`, and the source files each milestone
names. Also read `model-converter`'s `README.md`, section "Things that went wrong". It records
hard-won facts about onnxruntime and sherpa-onnx on Windows that apply here too.

---

## How to work on this

- **Before writing any code, summarise back** what you understood: what's ported, what's new,
  which backends exist, and the milestone order. List anything in this file that looks wrong,
  unclear or impossible. The user reads it to confirm nothing was missed.
- Build in the milestone order at the end. Each milestone ends with a check. **Stop after each
  milestone, show its check passing, and wait for the user to say go on.**
- **Verify before you rely.** This file names libraries and features that change quickly:
  llama.cpp audio support, `transformers` model classes, chat templates, packaging. Where it
  says "verify", check the installed version's actual behaviour and report what you found before
  building on it. If something doesn't exist, stop and say so. Don't invent an API.
- If a Rust behaviour seems wrong, port it as it is, and note it in `HANDOFF.md`. Deciding to
  change it is the user's call, for both apps.
- Commit after each milestone. Tests pass at every commit.

---

## Terms

- **ASR:** speech recognition, turning speech into text.
- **VAD:** voice activity detection, deciding which parts of the audio are speech. volis uses
  Silero VAD.
- **Segment ASR:** transcribes one finished utterance at a time (Whisper, Parakeet).
- **Streaming ASR:** transcribes while audio is still arriving, producing provisional text that
  firms up.
- **Committed text:** text the pipeline has decided is final and won't change. **Provisional
  text:** its current best guess, which may still change.
- **LocalAgreement:** a rule for turning a non-streaming model like Whisper into a streaming one.
  Transcribe the growing audio buffer again every short interval, and commit only the words
  that two consecutive transcriptions agree on.
- **Pre-roll:** a short stretch of audio kept from just *before* the VAD says speech started, so
  the first syllable isn't cut off.
- **GGUF:** llama.cpp's model file format. volis's translator is a GGUF.
- **Chat template:** each language model family's own way of laying out a conversation (roles,
  special markers). It's stored inside a GGUF's metadata, and in a Hugging Face model's
  `tokenizer_config.json`.
- **Variety:** a language plus a region, written as a tag like `es-MX` or `ar-IQ`. The table is
  in the Rust `src/varieties.rs`.
- **Carry-forward context:** each new sentence is translated knowing what came before it.
- **Revision:** re-translating recent sentences once later ones arrive, and replacing their
  earlier translation if it changes.

---

## Hard constraints

These come from the Rust `CLAUDE.md` and SPEC §2, adapted only where they name Rust tools. Put
them in pyvolis's own `CLAUDE.md`, word for word, so later sessions keep them.

1. **Never requires, attempts or depends on internet access at runtime.** No downloads, no update
   checks, no telemetry, no model hub lookups. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are
   set **before** any Hugging Face library is imported, in the entry point, and a test proves no
   library opens a connection to anywhere but the local network (see "Offline enforcement").
   Local network sockets for paired mode are allowed, under the same test as Rust: *would it still
   work with the WAN cable unplugged and no DNS server anywhere on the segment?*
2. **No separate services.** Every model runs in-process. No Ollama, no local servers, no helper
   processes that serve models. (A Python thread is fine. A child process running inference is
   not.)
3. **No JavaScript toolchain.**
4. **No OS-managed folders.** Never read or write `%APPDATA%`, `%LOCALAPPDATA%`, `~/.cache`,
   Hugging Face's cache, or PyTorch's cache. Every path comes from one `app_root()` function. Set
   `HF_HOME`, `TORCH_HOME` and similar variables to folders **inside** the app, so a library that
   tries to cache something stays inside the app.
5. **Model files keep their published names,** in folders named after the model. A folder made
   by `hf download` is exactly what the user sees in Explorer.
6. **Copy-to-run:** zip the finished folder, unzip on a fresh Windows machine with no internet and
   no Python, double-click, and it works. This is heavier than the Rust version because of
   PyTorch, and that's accepted. It still has to work.

Plus the Rust working rules: every error about a file names the absolute path tried; one path
function; no settings no milestone needs.

---

## Compatibility with the Rust volis

A user moves between the two apps, so these stay the same:

- **`volis.toml`.** pyvolis reads the same file with the same keys and meanings, and keeps the
  user's comments when saving (use `tomlkit`). New pyvolis-only settings go in sections the Rust
  version doesn't have, `[pyvolis.*]`. Check first whether the Rust config parser rejects unknown
  sections (`config.rs`, `deny_unknown_fields`). If it does, pyvolis uses its own
  `pyvolis.toml` for new settings instead, and says so in `HANDOFF.md`. **A config saved by
  pyvolis must still load in Rust volis.**
- **`models\` layout.** Model folders with `engine.toml`, the ones Rust uses, work in pyvolis
  unchanged. New kinds of folders (downloaded Hugging Face folders, GGUF speech models) are
  pyvolis-only. Check how Rust's discovery treats a folder it can't use: it must report it and
  carry on, never crash. If it would crash, stop and tell the user.
- **The paired-mode wire protocol, version 2** (`wire.rs`, `floor.rs`, `peer.rs`,
  `discovery.rs`), unchanged. **pyvolis must pair with Rust volis.** Context and revisions never
  cross the wire. Only final translated text does, as today.
- **The translation guards** in `translate.rs`: echo, reciting the prompt, and implausibly long
  output. Ported with their tests.
- **The command line:** `--report`, `--devices`, `--listen` (`--seconds`, `--wav`, `--compare`),
  and `--print-prompt` and `--translate` if Rust has them (M7.8). Same output format where the
  feature is the same.

---

## Part 1: existing features to port

Everything Rust volis does today, milestones M0 to M7.7 in its `SPEC.md` §13, with the Rust module
named for each. One Python module per Rust module, with the same name, so anyone can find the
matching code.

| Feature | Rust source |
|---|---|
| Paths from the app's own folder | `paths.rs` |
| Config load, and save keeping comments | `config.rs` |
| Model discovery, `engine.toml` rules, ranking by variety | `models.rs`, `varieties.rs`, `report.rs` |
| Audio capture and devices, resampling to 16 kHz mono | `audio.rs` |
| Silero VAD, cutting utterances | `vad.rs` |
| Segment ASR for Whisper and Parakeet, the ring buffer, `--compare` | `asr.rs`, `ring.rs`, `compare.rs` |
| Translation on its own thread, the prompt, output cleaning, the three guards | `translate.rs` |
| Voice output, playback, the half-duplex gate | `tts.rs`, `playback.rs` |
| Pipeline messages, `--listen` | `pipeline.rs`, `listen.rs` |
| The window | `gui.rs` |
| Continuous and turn-based modes, toggle and hold turn keys | `pipeline.rs`, `gui.rs` |
| Paired mode: wire protocol, floor token, peer link, discovery | `wire.rs`, `floor.rs`, `peer.rs`, `discovery.rs` |
| Shared-machine mode, a recognizer per side, Escape to cancel | `shared.rs` |
| Varieties through all three stages; Whisper told only the language part | `varieties.rs`, `asr.rs`, `translate.rs` |

Port the behaviour **and the tests.** The Rust version has unit tests for the pure parts:
`floor.rs`, `shared.rs`, `wire.rs`, `models::rank`, the guards, and `gui::Session`. Port each to
`pytest`.

The window's state logic stays separate from the drawing, as in Rust (`gui::Session`): plain
Python with no Qt in it, unit-tested. The Qt code only draws it.

---

## Part 2: model backends (new)

### The idea

Every model is loaded through a **backend**, chosen from what's in its folder. The pipeline only
ever talks to two interfaces:

```python
class SegmentAsr:      # one finished utterance in, text out
    def transcribe(self, audio: np.ndarray, language: str) -> AsrResult: ...

class StreamAsr:       # audio in chunks, provisional and committed text out
    def feed(self, chunk: np.ndarray) -> AsrUpdate: ...
    def finish(self) -> AsrUpdate: ...

class Translator:
    def translate(self, request: TranslationRequest) -> TranslationResult: ...
```

`AsrResult` carries text, word timings where the backend has them, and the backend's own
confidence where available. `TranslationRequest` carries the text, source and target varieties,
and the context (Part 5).

Every backend for segment ASR can also be used as streaming ASR through LocalAgreement (Part 4).
A backend with real streaming support may implement `StreamAsr` directly.

### How a folder is recognised

Discovery looks at each folder in `models\asr\` and `models\mt\` and decides its backend:

| Folder contains | Backend | Role |
|---|---|---|
| `engine.toml` (the Rust layout) | as that file says (`whisper`, `nemo_transducer`, …) | through sherpa-onnx, exactly as Rust |
| `config.json` with a speech `model_type` (`whisper`, `wav2vec2`, and others the installed `transformers` supports) and weights (`*.safetensors` or `pytorch_model.bin`) | `transformers` | ASR |
| a `.gguf` language model | `llamacpp` | translator |
| a `.gguf` speech model with its audio encoder file (`mmproj-*.gguf`) | `llamacpp-audio` | ASR (see P11; verify first) |
| `config.json` for a causal language model, with safetensors | `transformers` | translator (optional, P11) |
| `adapter_config.json` (a LoRA adapter) | attached to the base model it names | see "LoRA adapters" |

**No `engine.toml` is needed for downloaded folders.** pyvolis reads what it needs from the
model's own files. An **optional** `pyvolis.toml` inside a model folder can override anything
detected: display name, languages, varieties, device, and backend settings. The format is below.
If a folder can't be recognised, `--report` shows it as unusable, with the reason and the
absolute path, and carries on.

**Languages.** Many Hugging Face models don't declare their languages in a machine-readable way.
Read them from the model card's metadata (`README.md` front matter, `language:`) if present.
Otherwise mark the model as **"languages unknown"**, and offer it for any language, with that
label shown next to it in the window. The user can set them in the folder's `pyvolis.toml`.

`pyvolis.toml` in a model folder:

```toml
name = "Whisper large-v3-turbo Persian (fine-tune)"
languages = ["fa"]
varieties = ["fa-IR"]
device = "cuda"          # "cuda" or "cpu"; default: cuda if available
dtype = "float16"        # transformers only
trust_remote_code = false
```

### Backend: `sherpa` (existing ONNX folders)

sherpa-onnx **1.13.8** (the version Rust links), plus `sherpa-onnx-core` 1.13.8 listed by name in
`pyproject.toml`. `model-converter`'s README explains why: without it, sherpa-onnx picks up an
old `onnxruntime.dll` from Windows and crashes. Configure it exactly as `asr.rs` does.

**Parity check:** on the same audio, pyvolis's sherpa backend gives the same text as Rust volis.

### Backend: `transformers` (any downloaded Hugging Face ASR model)

- Load from the local folder only, with offline mode on.
- **Whisper family:** use the model's generation with **language and task forced**
  (`transcribe`, never `translate`). Whisper is only ever told the language part of a variety,
  as in Rust. Return word timestamps where the model supports them (needed for LocalAgreement and
  subtitles).
- **CTC models** (wav2vec2, MMS and similar): the matching processor and model class, and
  greedy decoding. For MMS, select the language adapter by the ISO code the model uses. Map from
  volis's codes and say so in the log.
- **Other speech models** the installed `transformers` supports through its speech recognition
  pipeline: allowed. Report in `--report` which class loaded each model.
- **`trust_remote_code`** is **off** unless the folder's `pyvolis.toml` turns it on. When on, the
  model's code files must already be in the folder. Offline mode prevents fetching them. If
  loading fails for that reason, say so plainly.
- **GPU:** CUDA when available and allowed, otherwise CPU. `float16` on GPU, `float32` on CPU.
  One model loaded once, reused.
- **Long audio:** Whisper hears 30 seconds at most. The pipeline never sends it more (the Rust
  25-second split rule applies).

**Verify at P2:** a Whisper fine-tune downloaded with `hf download` runs with no files added,
and a wav2vec2 or MMS model runs too. Report which model classes the installed `transformers`
supports for speech recognition.

### Backend: `llamacpp` (any GGUF translator)

- `llama-cpp-python`. **Verify** which of its builds works on this machine with CUDA (for the 4070
  and 5090) and on CPU. The Rust version runs its translator on the CPU. pyvolis may use the GPU,
  and every timing it reports says which was used.
- **Chat format from the model, never hardcoded.** Build the prompt with the chat template stored
  in the GGUF's metadata (`tokenizer.chat_template`), rendered with the system text and the user
  text. This replaces Rust's Qwen-only `prompt_for`. It's what lets Gemma, Aya, Command R7B and
  others drop in.
  - **Qwen3:** thinking must be off. Render the template with thinking disabled if the template
    supports it, otherwise append the empty `<think>\n\n</think>\n\n` block exactly as Rust does.
  - **Parity check:** for a Qwen3 GGUF, the rendered prompt is **byte-identical** to Rust's
    `prompt_for` (compare with `volis --print-prompt` if it exists, otherwise with the string built
    by the Rust code, reproduced in a test). If it isn't identical, find out why before going on.
- **Decoding:** greedy, as Rust.
- **Several translators installed.** `models\mt\` may hold several GGUF files or folders. Rust
  allows exactly one. pyvolis lists them all and the user picks.
- **The three guards** apply to every translator.

### LoRA adapters

- **GGUF:** a LoRA adapter converted to GGUF (`convert_lora_to_gguf.py` in llama.cpp) can be loaded
  onto its base translator at runtime. **Verify** `llama-cpp-python` supports it. A folder with an
  adapter GGUF and a `pyvolis.toml` naming its base appears as its own translator in the list.
- **transformers:** an adapter folder (`adapter_config.json`) is attached to its base model with
  `peft` at load time. It appears as its own ASR model in the list.

This is what makes tuning practical: train a LoRA, drop the adapter folder in, test it, with no
merge and no conversion needed for testing.

### Memory and devices

- Show each loaded model's memory use (GPU and system) in the window and in `--report --load`.
- Refuse to load a model that won't fit, with a clear message naming the model, its size and
  what's free. Never crash the app on out-of-memory.
- Models load on a background thread. The window stays responsive and shows "Loading…".

---

## Part 3: the ASR pipeline (improved)

The same flow as Rust, capture → VAD → ASR → translation → voice, with these improvements. Each
one is switchable in settings, so its effect can be measured.

1. **Pre-roll.** Keep a rolling buffer of the most recent audio, and prepend **300 ms** (setting) of
   audio from before the VAD's speech start to every utterance. In testing, first words were
   clipped: "Se me murió" came out as "Si me murió", and "Oigan" as "Oiga". Measure with and
   without pre-roll on the same files, and report the difference.
2. **Hallucination guards,** for text an ASR produces from silence or noise:
   - drop output from segments whose VAD speech probability is low throughout;
   - drop or trim **repeated phrases** (the same phrase three or more times in a row);
   - drop known stock phrases Whisper invents on silence, kept in a list in a config file,
     per language, that the user can edit. For example "Thank you for watching", "Subtítulos
     realizados por…", and "Repito" at the end of a clip, which appeared in testing;
   - log every dropped text with the reason, so a wrong drop can be spotted.
3. **Punctuation kept.** Pass the ASR's punctuation through to translation untouched. Never strip
   it before translating. Questions lost their "¿…?" in testing, and the translations became
   statements. Also report per model how often its output has **no punctuation at all**. That
   says whether it can mark questions.
4. **Language forced** on every ASR call from the side's variety, as Rust. Never auto-detect.
5. **Word timestamps** kept wherever the backend gives them, for subtitles and for LocalAgreement.

---

## Part 4: streaming recognition (new)

- **LocalAgreement for segment models.** While speech continues, re-transcribe the growing buffer
  every **1.0 s** (setting). Commit the longest prefix of words that the last **two** transcriptions
  agree on. Show the rest as provisional. When the VAD ends the utterance, commit everything.
  Trim the buffer at committed sentence ends, so it never exceeds Whisper's 30 seconds.
- **Show committed and provisional text differently** in the window: provisional in a lighter
  style, committed in normal text.
- **Only committed text goes to the translator** (except in revision mode, Part 5).
- **Cost control.** Re-transcribing every second multiplies ASR work. Show the ASR's real-time
  factor (processing time ÷ audio time), and warn when it goes above 0.8, because streaming can't
  keep up beyond 1.0.
- A backend with its own streaming support can implement `StreamAsr` directly, instead of
  LocalAgreement.

Streaming is available with the microphone and with files. Segment mode (the Rust behaviour)
remains available and is the default for the microphone until streaming is proven.

---

## Part 5: translation with context (new)

Two modes, chosen in settings. **Carry-forward is the default.**

### Carry-forward context

Each new committed sentence is translated with the recent conversation as context:

- The last **N** committed source sentences and their translations (setting, default 4, and never
  more than a token budget, default 400 tokens). They're placed in the prompt as **earlier turns**
  of the chat: user turn = source, assistant turn = translation. That's the most natural form for
  every chat model, and the model continues the pattern.
- A **session glossary:** names and terms the user adds in the window ("Susie Wolff", "Bellas
  Artes"), sent in the system text as "Keep these names and terms exactly: …". Optional.
- Context is **untrusted text** (it came from ASR). It never changes the instructions, and the
  guards still run on every output. The "recites the prompt" guard must also recognise recited
  **context**, not just the system text.

### Short fragments

A committed sentence of **fewer than 4 words** (setting) with no final punctuation is **held** for
up to **1.5 s** (setting) and joined to the next one before translating. In testing, "Yo manejo."
was translated alone as "I handle it" when the next words, "Mi carro…", made the meaning "I'll
drive". Measure the effect with and without.

### Revision mode

After each new committed sentence:

- re-translate the last **K** sentences together (setting, default 3) with the context above;
- if an earlier sentence's translation changes, send a **Revision** event: the window replaces it,
  highlights it briefly, and keeps the earlier version in its history;
- never revise a sentence **older than 30 s** (setting), or one that has already been **spoken
  aloud**;
- count revisions per session. A mode that rewrites constantly is worse to read, and the count
  shows it.

In paired mode, only the final translation of each sentence crosses the wire, and revisions stay
local. Revision mode is off while paired, and the window says so.

### Prompts

- The default system text is Rust's, from `system_prompt` in `translate.rs`, unchanged.
- **Prompt files** in `prompts\` can replace it, one file per variant. The user picks one in
  settings. This is where tested improvements will go, such as spoken-language guidance ("'¿no?'
  at the end of a clause means 'right?', never a negation"). pyvolis just loads the file. Deciding
  which prompt is best is model-bench's job.

---

## Part 6: streaming translation of an audio file (new, the main test)

### What the user does

1. **File → Open**, or drag a file onto the window. WAV, MP3, M4A, FLAC and OGG open with **no
   extra tools**. Decode with `PyAV`, whose wheels include the decoders. Resample to 16 kHz mono.
2. Choose the **source variety** (what's spoken in the file) and the **target variety**. The ASR and
   translator dropdowns are ranked by variety, as everywhere else.
3. Choose **Real time** (the audio is fed at playing speed, and optionally played through the
   speakers so the user hears the original while reading) or **Fast** (as fast as the models allow).
4. **Start**, **Pause**, **Stop**. A progress bar shows position in the file.

### How it works

The file becomes an **audio source**, replacing the microphone. Everything after it is the **same
pipeline** as live use: VAD, pre-roll, streaming or segment ASR, context, fragment holding,
revision, guards. So the file test measures exactly what live use will do.

- Voice output is **off** by default in file mode, and can be switched on.
- The half-duplex gate doesn't apply, since the microphone is closed.

### What the user sees

A timeline, one row per sentence: its time in the file, the source text (provisional or
committed), the translation, and a marker if it was revised. Clicking a row plays that stretch
of audio. Arabic and Persian display right to left, including rows that mix in English words or
numbers.

A status line shows: ASR real-time factor, translation time per sentence (median and 90th
percentile), sentences, revisions, and dropped hallucinations.

### What it saves

On **Export**, to a folder the user chooses (default `exports\<file>-<date>\` inside the app):

- `transcript.txt` and `translation.txt`;
- `source.srt` and `translation.srt`, subtitles with timings;
- `events.jsonl`, **every event in order**: partial text, commits, dropped text with reasons,
  translations, revisions (old and new), and timings. First line: the full configuration
  (models with their folder names and file hashes, prompt file, every setting). This file is
  what model-bench and a colleague use to reproduce and check a run.

### Quick scoring

If a file has a reference next to it (`<file>.ref.json` with `{"transcript": "...",
"translation": "..."}`, or an SRT with the same name and `.ref.srt`), show at the end: CER of the
transcript and chrF of the translation, with the same text cleaning model-bench uses (Arabic and
Persian rules in `model-bench.md`). This is a quick check, not the proof. The proof is
model-bench.

### Command line

```powershell
pyvolis --file talk.m4a --from es-MX --to en --asr <folder> --mt <file> --fast --export out\
```

No window. It writes the same export folder and exits with a non-zero code on failure. model-bench
will drive pyvolis through this.

---

## Offline enforcement

- The entry point sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`, `HF_HOME`
  and `TORCH_HOME` (inside the app) before importing anything else.
- A test patches Python's socket connection function to fail on any address outside the local
  network. It then loads one model of each backend and translates one sentence. It must pass.
  If any library tries to reach the internet, the test names it.
- **Getting models is a separate command,** not part of the app: `fetch-model.ps1 <hf-id>` runs
  `hf download` into `models\asr\` or `models\mt\` with the folder named after the model. It's the
  only part of the repo that uses the internet. The app never calls it.

---

## Performance: keeping the window responsive

Python runs one piece of Python code at a time. The heavy work happens inside C++ and CUDA
libraries, which usually let other threads run while they work. If one doesn't, the window
freezes while it runs.

- Give the pipeline the same thread layout as Rust (`ARCHITECTURE.md`, "Threading"): capture,
  ASR, translation on its own thread, voice, playback, and the window. Streaming ASR and revision
  add work, so they get their own threads too.
- Show the same latency readout as Rust: recognised, translated, first audio.
- At P5, measure whether the window stays responsive during ASR and translation. If it doesn't,
  stop and report which library holds up other threads. **Don't** fix it by moving a model into a
  separate process. That breaks the "no separate services" rule, and the user decides.

Somewhat slower than Rust is fine. Slower without anyone knowing isn't. Every milestone check that
involves timing prints pyvolis's and Rust's numbers side by side, where Rust can run the same
thing.

## Checking against the Rust version

`parity/` holds scripts that run the same input through both apps and compare. The path to the
Rust `volis.exe` goes in `machine.yaml` (not committed), like `model-converter`'s.

- **Discovery:** `--report` from both, on the same `models\`, agrees on every Rust-format folder.
- **Recognition:** same sherpa-onnx version, same settings, so the text is **identical**. A
  difference means a setting differs. Find it.
- **Translation:** for a Qwen3 GGUF with the default prompt, the prompt is byte-identical and the
  output identical. If the llama.cpp versions differ slightly, allow a few differing lines and
  print them all for the user.
- **Wire protocol:** pyvolis and Rust volis pair and exchange turns both ways.

## Packaging and the environment

- **Development:** uv, with Python and every package inside the repo (`.uv\`, `.venv\`), set up by
  `setup.ps1`. Copy `model-converter`'s approach and its reasons.
- **Python 3.12.**
- **Main packages:** PySide6, sounddevice, soxr, PyAV, numpy, tomlkit, sherpa-onnx 1.13.8 with
  sherpa-onnx-core 1.13.8, llama-cpp-python, torch (CUDA 12.8 or newer build: the 5090 needs it,
  and it runs on the 4070 too), transformers, peft.
- **A doctor script, `doctor.ps1`,** like `model-converter`'s `0_doctor.py`, checks at setup and on
  request:
  - that every native library is present and unaltered (security software has removed DLLs
    before on this machine);
  - that sherpa-onnx loaded onnxruntime **1.28.2** from the environment, not another copy;
  - that torch sees the GPU;
  - that `llama-cpp-python` loads, and whether it has GPU support;
  - that no two libraries loaded **conflicting CUDA runtime DLLs.** torch and llama-cpp-python's
    CUDA build each bring their own. Load both in one process and report exactly what was loaded
    from where.
- **Distribution (P12):** a folder the user can zip and copy. Try **PyInstaller in one-folder mode**
  first. PyTorch makes it large (several GB), and that's accepted. If PyInstaller can't collect
  torch or the CUDA libraries reliably, fall back to shipping the uv-managed Python and a
  relocatable environment inside the folder, with a small launcher. Report which approach worked
  and why.
- **Packaging traps to check at P12:**
  - **DLLs not collected:** sherpa-onnx's `onnxruntime.dll`, llama.cpp's libraries, torch's CUDA
    libraries, PortAudio for `sounddevice`, and PyAV's decoders all have to end up in the folder.
    Check each one is present, and that the copy **loaded** is the one in the folder. The
    converter's `engine_worker.py` shows how to check which `onnxruntime.dll` got loaded.
  - **Security software:** the converter's README describes McAfee and Trellix removing DLLs, and a
    PyInstaller folder is exactly what they flag. If a DLL goes missing on the test machine, say so.
    Don't work around the scanner.
  - **The app root:** the built app and running from source find `models\` differently. Test both.

## Layout

```
pyvolis/
  CLAUDE.md  README.md  HANDOFF.md
  pyproject.toml  uv.lock  setup.ps1  doctor.ps1  fetch-model.ps1  build.ps1
  machine.example.yaml   copy to machine.yaml: path to Rust volis.exe, for parity checks
  parity/                scripts comparing pyvolis with Rust volis
  pyvolis/
    __main__.py  cli.py  paths.py  config.py
    models.py  varieties.py  report.py
    audio.py  filesource.py  vad.py
    asr/          __init__.py  sherpa.py  hf.py  llamacpp_audio.py  streaming.py  guards.py
    translate/    __init__.py  llamacpp.py  hf.py  context.py  guards.py  prompts.py
    tts.py  playback.py
    pipeline.py  listen.py  shared.py
    wire.py  floor.py  peer.py  discovery.py
    export.py  scoring.py
    gui/          session.py (no Qt)  window.py  filemode.py  …
  prompts/        default.txt (Rust's system text) and variants
  config/         hallucinations.yaml, other editable lists
  tests/
  models/  volis.toml   for development; not committed except README.txt files
  exports/        not committed
```

---

## Test material

Tests need speech. Voices aren't committed.

- `tests/fetch-fixtures.ps1` downloads a small, fixed set of clips (FLEURS: Spanish, Persian,
  Arabic, English; 10 each), with their reference transcripts and the aligned FLORES+ English
  translations, into `tests/fixtures/` (not committed). It's a development tool that uses the
  internet, like `fetch-model.ps1`.
- The user's own recordings from the Mexican Spanish tests can be added to
  `tests/fixtures/user/`. They're the best check of first-word clipping and question marks.
- Every milestone check that compares before and after (pre-roll, fragment holding, streaming)
  runs on the same fixture files, and prints both results side by side.

---

## Milestones

The order puts the file test and the new pipeline early, because they're the priority. The
existing live features are ported around them.

**P0 — skeleton, config, discovery, report.** Paths, the environment variables, `volis.toml`
compatibility, discovery for all three kinds of folder, `--report`, `doctor.ps1`,
`fetch-model.ps1`, and the offline test.
Check: `--report` on the user's current `models\` lists the same Rust models as Rust volis does,
plus any downloaded folders with their detected backend. A config saved by pyvolis loads in Rust
volis. The offline test passes.

**P1 — audio in: microphone, file, VAD, pre-roll.** Device capture as `audio.rs`, the file source
with PyAV, Silero VAD, and pre-roll.
Check: `--devices` matches Rust. A test file is cut into utterances. With pre-roll on, each
utterance starts before the first syllable. Print the cut points both ways.

**P2 — ASR backends: sherpa and transformers.** Both backends, segment mode, `--compare`, and the
hallucination guards.
Check: the sherpa backend matches Rust volis's text on the fixtures. A Whisper fine-tune and a CTC
model, each fetched with `fetch-model.ps1` and not touched afterwards, both transcribe. Guards: a
file of silence and noise produces no text, and each drop is logged.

**P3 — translators: any GGUF.** Chat templates from metadata, Qwen3 thinking off, several
translators, the guards, the prompt files, and `--translate` / `--print-prompt`.
Check: Qwen3 1.7B's prompt is byte-identical to Rust's, and its translations match Rust's on 20
sentences. A second family (Gemma or Aya, fetched as GGUF) translates with its own template, with
no code changes.

**P4 — the file mode, command line first.** `--file` end to end in segment mode, with the export
folder, `events.jsonl` and quick scoring.
Check: a fixture file runs end to end and writes every export file, and the scores print. **This
is the first point where the user can test models on files.** Stop and let them try it.

**P5 — the window, with file mode.** The main window ported from `gui.rs`, plus the file-mode
timeline, playback of rows, and export. Live microphone use with voice output (`tts.rs`,
`playback.rs`, the half-duplex gate).
Check: the user opens a file in the window, watches it translate, clicks a row to hear it, and
exports. Live conversation works as it does in Rust. Arabic and Persian show right to left.

**P6 — continuous and turn-based modes.** As Rust M6.
Check: the Rust M6 checks, repeated.

**P7 — streaming ASR and carry-forward context.** LocalAgreement, committed and provisional text,
the context in the prompt, the glossary, fragment holding.
Check, on the fixtures and the user's recordings: first with each feature off, then on. Print CER,
chrF and latency for each. Streaming shows provisional text within about 1.5 s of speech. Context
fixes the "Yo manejo" case, or the report says it doesn't.

**P8 — revision mode.** Re-translation of recent sentences, revision events, history, and limits.
Check: on a fixture file, revisions happen, are shown and logged, and never touch spoken or
too-old sentences. The revision count is reported.

**P9 — paired mode.** As Rust M7.
Check: pyvolis with pyvolis, then **pyvolis with Rust volis**, in both directions. A killed link
releases the floor with a clear message on both ends.

**P10 — shared-machine mode, a recognizer per side, varieties.** As Rust M7.5 to M7.7, with the new
backends available per side.
Check: the English and Mexican Spanish shared-mode tests the user ran on Rust, with the same
models, give the same results. Then with a downloaded model on one side.

**P11 — more backends, each verified first.**
- `llamacpp-audio`: GGUF speech models (Qwen3-ASR, Voxtral, Gemma 4) through llama.cpp's audio
  input. **Verify first** that `llama-cpp-python` exposes audio input for these models. If it
  doesn't, stop and report what would be needed. Don't build a workaround that runs a separate
  process.
- LoRA adapters for GGUF and for transformers.
- `transformers` translators (safetensors language models on the GPU).
Check: each backend that exists runs one fixture file end to end.

**P12 — portability.** The distributable folder.
Check: zip it with `models\` and config, move it to a Windows machine with no internet and no
Python, unzip, double-click. Live mode and file mode both work, and Windows' Resource Monitor
shows no connections except paired mode on the local network.

## Documents

Same set as Rust, kept short:
- `README.md`: setup, adding a model with `fetch-model.ps1`, file mode;
- `CLAUDE.md`: the constraints and working rules;
- `HANDOFF.md`: status, and **every difference from Rust volis**, each with the reason.

Link to the Rust repo's documents for the wire protocol, varieties and dialects. Don't copy them.
