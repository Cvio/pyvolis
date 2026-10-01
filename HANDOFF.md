# Handoff: where pyvolis stands

Last updated 2026-09-30. Read `CLAUDE.md` first, then `pyvolis-build.md`.

## Status

- **P0 done:** paths and the offline environment, `volis.toml` load and save, discovery for
  `engine.toml` folders, Hugging Face folders and GGUF translators, `--report`, `doctor.ps1`,
  `fetch-model.ps1`, `parity\report.py`, and the offline test.
- **P1 done:** `--devices` (identical to Rust's output), microphone capture (`audio.py`), the
  file source (`filesource.py`, PyAV), Silero VAD with the pre-roll (`vad.py`), `pyvolis.toml`
  with `[vad].pre_roll_ms`, `tests\fetch-fixtures.ps1`, and `scripts\vad_cuts.py` (cut points
  with and without the pre-roll, from a file, the fixtures or the microphone).
- **P2 done, one check partly open (below):** recognizers through sherpa-onnx (`asr/sherpa.py`)
  and transformers (`asr/hf.py`: Whisper, CTC with MMS adapters, Cohere Transcribe), the
  hallucination guards (`asr/guards.py`, `config/hallucinations.toml`), `ring.py`, `compare.py`,
  the pipeline core (`pipeline.py`) and `--listen` with `--seconds`, `--wav`, `--compare`.
  Scripts: `scripts/transcribe.py` (every model on the fixtures: CER, share of outputs without
  punctuation, real-time factor), `parity/asr.py` (Rust volis vs pyvolis on the same audio).
- **P3 done:** GGUF translators through llama-cpp-python (`translate/llamacpp.py`), prompts from
  each model's own chat template, Rust's cleaning and three guards (`translate/guards.py`),
  prompt files (`prompts/`), several translators (`[translate]` in `pyvolis.toml`), and
  `--translate` / `--print-prompt`. `parity/translate.py` compares with Rust volis.
- Next: P4 (file mode on the command line). Wait for the go-ahead.

## Environment, as verified at P0

- Python 3.12.10 (uv-managed), torch 2.11.0+cu128, transformers 5.18.0, sherpa-onnx 1.13.8 with
  onnxruntime 1.28.2 from the environment, PySide6 6.11.2, PyAV 19.0.0, peft 0.21.1.
- The development machine (RTX 4070 Laptop, 8 GB) has no CUDA toolkit, no CMake and no MSVC on
  PATH. Which matters at P3: llama-cpp-python is only published on PyPI as source.
- transformers 5.18's speech seq2seq mapping includes `whisper`, `cohere_asr`, `qwen3_asr`,
  `voxtral`, `granite_speech`, `canary`, `moonshine` and more; its CTC mapping includes `wav2vec2`,
  `hubert`, `wavlm`, `parakeet_ctc` and more. Qwen3-ASR and Voxtral may therefore run through
  `transformers` without llama.cpp's audio input (P11).

## Development models

`models\` holds **hard links** to Rust volis's model files (`target\release\models\`), made by
`parity\link_models.py`: the same bytes, no copies, and deleting pyvolis's side never touches
Rust's. `models\asr\whisper-small\` is a hard link to model-converter's download of
`openai/whisper-small` (without its `fetched.json`). `volis.toml` is a copy of Rust's.

## Differences from Rust volis, with the reasons

1. **pyvolis-only settings go in `pyvolis.toml`, not `[pyvolis.*]` in `volis.toml`.** Rust's
   `Config` has `#[serde(deny_unknown_fields)]` at the top level, so any extra section would
   stop Rust volis loading the shared file. No milestone so far needs a pyvolis setting, so the
   file doesn't exist yet.
2. **An ASR folder without `engine.toml` is examined, not skipped.** Rust skips it with a
   warning; pyvolis recognises Hugging Face and GGUF folders, and lists anything it can't
   recognise as an error with the reason. TTS folders keep Rust's rule.
3. **Several translators.** Rust uses the one `.gguf` at the top of `models\mt\` and refuses to
   start if there are two. pyvolis lists that file plus one or more per subfolder, and its report
   warns when there are two at the top.
4. **A GGUF with no chat template is listed as unusable,** because pyvolis builds prompts from the
   model's own template (Rust hardcodes Qwen's). The Qwen3 file Rust uses has one.
5. **`--listen` and the window** don't exist yet; they arrive at P2 and P5. With no arguments
   pyvolis says the window arrives at P5.
6. **Logs** go to `logs\pyvolis.log.<date>` (Rust: `volis.log.<date>`), plus `logs\doctor.json`.
7. **Varieties error message** names both `pyvolis/varieties.py` and Rust's `src/varieties.rs`,
   because the tables must stay the same (a test pins pyvolis's to Rust's).
8. **Devices are PortAudio's WASAPI list** (sounddevice). cpal's default Windows host is WASAPI,
   so names and order match Rust's; the "F32" in each device line is assumed (WASAPI shared
   mode always delivers float32), where cpal asks the device.
9. **Resampling uses soxr, not sherpa-onnx's linear resampler.** A better filter for the same
   job. Recognition of 16 kHz input (the fixtures) is unaffected; audio from a 48 kHz
   microphone reaches the recognizer slightly differently than in Rust.
10. **The pre-roll length is a setting**, `[vad].pre_roll_ms` in `pyvolis.toml`, default 600 ms
    (Rust's fixed value; the build file assumed 300 ms, but Rust already found 300 too short for
    "¿Cuántos años tienes?"). 0 turns it off. Segments also record where the detector itself
    said speech began, so the lead-in can be measured.
11. **The VAD reset** pops queued segments one by one: sherpa-onnx's Python binding has no
    `clear()`. Same effect.
12. **File input is new** (`filesource.py`): WAV, MP3, M4A/AAC, FLAC, OGG/Vorbis, Opus, ALAC,
    decoded by PyAV, averaged to mono and resampled like the microphone.

13. **`--compare` manages GPU memory.** Rust keeps every engine loaded. pyvolis loads every
    engine if they fit; if the GPU models don't, they take turns (each loaded for its run and
    released) while the CPU (sherpa) models stay loaded. Runs are always one after another.
14. **Hallucination guards are new** (`[guards]` in `pyvolis.toml`, each switchable). The
    speech-probability guard uses a second Silero model from the same `silero_vad.onnx`, run by
    sherpa-onnx at `min_peak_probability` (0.8): sherpa-onnx's Python binding doesn't expose
    Silero's probabilities, but it applies a threshold to each window, which answers "did any
    window clearly contain speech?". (PyTorch can't run an `.onnx` file, which the user had
    been told it could.)
15. **The console is UTF-8** (`__main__.py`), so Arabic and Persian reach it intact.
16. **Whisper through sherpa is created with `tail_paddings=0`,** as Rust passes; the Python
    binding's default is -1.

17. **llama-cpp-python is built here** (`build-llama.ps1`, CPU, kept in `wheels\`). Its
    published Windows wheels stop at 0.3.19 (CPU) and 0.3.4 (CUDA 12.4, too old for Qwen3 and
    for the RTX 5090); 0.3.35 is source only. Rust runs its translator on the CPU too.
18. **Flash attention is on.** llama.cpp defaults to "auto", which is on for the CPU, and that
    is what Rust's llama-cpp-2 gets; llama-cpp-python turns it off unless told. With it off,
    9 of 25 translations differed from Rust's; with it on, all 25 are identical.
19. **Prompts come from each GGUF's chat template** (Rust hardcodes Qwen's ChatML). For Qwen3
    the rendered prompt is byte-identical to Rust's `prompt_for` (tests/test_translate.py),
    with thinking turned off through the template's `enable_thinking`.
20. **Several translators**: `[translate].model` in `pyvolis.toml` (default: the file at the
    top of `models\mt\`, Rust's); `[translate].prompt` picks a prompt file.

## P3 findings

- **Translation parity:** 25 sentences (14 Spanish->English, 11 English->Spanish, Rust's own
  transcripts from the cable run), Qwen3 1.7B, default prompt: **25 of 25 identical**, on the
  CPU, about 0.7 to 2.1 s a sentence (Rust's timings on the same sentences were the same range).
- **A second family**, Gemma 3 4B (`unsloth/gemma-3-4b-it-GGUF`, Q4_K_M), translates with its
  own template (no system role; it writes its own `<bos>`) and no code changes: 1.9 to 3.0 s a
  sentence on the CPU. Asked for Mexican Spanish it wrote "¿Qué onda, carnal?".
- **GPU for the translator** needs a CUDA build of llama-cpp-python, which needs the CUDA
  Toolkit (12.8 or newer, for the 5090) installed on the building machine. Not installed here;
  the user's decision.
- **llama-cpp-python 0.3.35 ships `mtmd.dll` and a `mtmd_cpp` module** (llama.cpp's multimodal
  library): the starting point for P11's audio-input check.
- **Recognition parity through the cable can't be made exact.** With both ends at 16 kHz the
  audio still arrives scaled (gain 0.9896) and not sample-exact, so Whisper's P2 differences
  stand as explained; Parakeet matched exactly on every Spanish utterance in P2. Exact Whisper
  parity would need Rust to transcribe a file itself (its test binary, built outside the Rust
  repo) - the user's call.

## P2 findings

- **Recognition parity with Rust (`parity/asr.py`).** Rust volis listens to the VB-Audio cable
  while the Spanish fixtures play into it; pyvolis transcribes the WAVs Rust wrote. Parakeet:
  identical on every utterance. Whisper (all three int8 ONNX folders): 13 of ~33 differ, by a
  word or a capital. Cause, shown: Rust's WAVs are 16-bit, and changing the audio by less than
  one 16-bit step flips int8 Whisper's text ("Los personas" -> "Las zonas"; scaling by
  32767/32768 gives Rust's exact "Las personas"). So the WAV route proves Parakeet's
  configuration and can't prove Whisper's. Setting the cable to 16 kHz didn't make it exact
  (see P3 findings).
- **transformers 5.18 speech classes** (seq2seq): canary, cohere_asr, fun_asr_nano,
  granite_speech(_plus), kyutai_speech_to_text, moonshine(_streaming), qwen3_asr,
  seamless_m4t(_v2), speech-encoder-decoder, speech_to_text, speecht5, vibevoice_asr, voxtral,
  voxtral_realtime, whisper. CTC: data2vec-audio, granite_speech5_ctc, hubert, lasr_ctc,
  parakeet_ctc, sew(-d), unispeech(-sat), wav2vec2(-bert, -conformer), wavlm.
- **Fixtures (`scripts/transcribe.py`, 10 FLEURS clips each):** Spanish CER 0.5 to 1.9% for every
  model. Arabic: Cohere Transcribe 2.1%, the oddadmix Whisper fine-tune 3.7% in float16 through
  transformers but 8.4% as Rust's int8 ONNX (which cuts sentences short), base Whisper turbo
  5.2% **with no punctuation on any Arabic output**, MMS 5.8% (CTC: never punctuates).
- **A hole in the guards:** noise shaped like speech (band-limited, pulsing at syllable rate)
  passes Silero, and Whisper answers it with invented text: once "¡Suscríbete al canal!"
  (dropped by the stock-phrase guard), once a lone "y" that no guard catches. A possible fourth
  guard, for the user to decide: text implausibly short for the length of the segment.
- **Persian ("fa")** was missing from the varieties table; added in both apps (below).

## Changes made in Rust volis too, by the user's decision

- 2026-09-30: the translation prompt's dialect sentence read "aloud,using" and carried
  indentation and trailing spaces; fixed in `translate.rs`.
- 2026-09-30: `VOICE_NOTE_LEFT` in `config.rs` wrote its second comment line with 31 leading
  spaces; fixed, with a test, in both apps.
- 2026-09-30: Persian added to the varieties table in both apps: `fa` (Persian) and `fa-IR`
  (Persian (Iran), "Iranian Persian" in the prompt).

The Rust changes are left uncommitted for the user to review and commit, and `volis.exe` is
not rebuilt (Rust volis doesn't know `fa` until it is).
