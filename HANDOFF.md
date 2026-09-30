# Handoff: where pyvolis stands

Last updated 2026-09-30. Read `CLAUDE.md` first, then `pyvolis-build.md`.

## Status

- **P0 done:** paths and the offline environment, `volis.toml` load and save, discovery for
  `engine.toml` folders, Hugging Face folders and GGUF translators, `--report`, `doctor.ps1`,
  `fetch-model.ps1`, `parity\report.py`, and the offline test.
- Next: P1 (audio in: microphone, file, VAD, pre-roll). Wait for the go-ahead.

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
5. **`--devices`, `--listen` and the window** don't exist yet; they arrive at P1, P2 and P5. With
   no arguments pyvolis says the window arrives at P5.
6. **Logs** go to `logs\pyvolis.log.<date>` (Rust: `volis.log.<date>`), plus `logs\doctor.json`.
7. **Varieties error message** names both `pyvolis/varieties.py` and Rust's `src/varieties.rs`,
   because the tables must stay the same (a test pins pyvolis's to Rust's).

## Rust behaviour ported as it is, for the user to decide

- **`VOICE_NOTE_LEFT` in `config.rs`** carries the source indentation into the file: when
  `left_voice` is added to a `volis.toml` that lacks it, the second comment line is written with
  31 leading spaces. pyvolis writes the same bytes. It only happens when the key was missing.
- (Fixed in Rust by the user's decision, 2026-09-30: the dialect sentence of the translation
  prompt read "aloud,using" and carried indentation and trailing spaces. P3 will match the
  fixed prompt.)
