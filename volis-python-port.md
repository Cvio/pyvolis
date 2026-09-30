# Instructions: a Python version of volis

For Claude Code. Build a second, independent implementation of volis in Python, called
**pyvolis**, in its own folder beside the Rust one:

```
Desktop\
  volis\        the Rust version. Stays as it is, and stays the reference.
  pyvolis\     the Python version. New.
```

**"Two versions" means both keep working.** Don't change or delete anything in the Rust repo.
This file is the only thing that crosses between them, plus the model files, which both read in
the same layout.

Read this whole file before writing any code. Then read, in the Rust repo: `CLAUDE.md`,
`SPEC.md`, `ARCHITECTURE.md`, `HANDOFF.md`, and the source files each milestone below names.
**The Rust source is the specification.** Where this file and the Rust code disagree about
behaviour, the Rust code wins. Stop and say so.

---

## How to work on this

- Build in the milestone order below. Each milestone ends with a check. Stop after each one,
  show the check passing, and wait for the user to say go on.
- Before P0, summarise back what you understood: what's being ported, what stays the same, what
  changes, and which checks compare against the Rust version.
- If a behaviour in the Rust code seems wrong, **port it as it is** and note it in
  `HANDOFF.md`. Fixing it is a separate decision for the Rust side too. The two versions must
  not quietly drift apart.
- Commit after each milestone passes.

---

## What stays exactly the same

These are what make the two versions interchangeable. A user must be able to copy `models/` and
`volis.toml` from one to the other and have it work.

1. **The hard constraints in the Rust `CLAUDE.md` (SPEC §2), all six.** Restate them in
   pyvolis's own `CLAUDE.md`, adapted only where they name Rust tools:
   - Never requires, attempts or depends on the internet. Local network sockets for paired mode
     are allowed, under the same "WAN cable unplugged, no DNS anywhere" test.
   - No separate services, no localhost servers, no Ollama. Every model runs in-process.
   - No JavaScript toolchain.
   - No OS-managed folders (`%APPDATA%`, `~/.cache` and the like). Everything resolves from the
     app's own folder.
   - Model files keep their published names, in folders named after the model.
   - Copy-to-run: zip the finished folder, unzip on a fresh Windows machine with no internet,
     double-click, and it works.
2. **The `models/` layout and `engine.toml` rules.** They're the same, including rejecting
   unknown keys, `varieties`, and the ranking of tuned, general and other-variety models
   (`models.rs`, `models::rank`).
3. **`volis.toml`.** Same keys, same meaning, same defaults, and the same rule that saving from
   the window keeps the user's comments (`config.rs`, `save_selections`). Use `tomlkit`, which
   preserves comments, like `toml_edit` does.
4. **The paired-mode wire protocol, version 2** (`wire.rs`, `floor.rs`, `peer.rs`,
   `discovery.rs`). Same messages, same fields, the same limits on untrusted input, the same UDP
   discovery on port 47801, IP addresses only (never resolve a hostname), and loss of the link
   noticed by pings stopping. **A Python instance must pair with a Rust instance.** That's one of
   the acceptance checks.
5. **The translation prompt, byte for byte** (`translate.rs`: `prompt_for`, `system_prompt`),
   including the empty `<think></think>` block, the variety sentence, and greedy decoding.
6. **The three refusal guards in `translate.rs`:** echo, reciting the prompt, and implausibly
   long output. Each was found in a live session. Port them with their tests.
7. **Behaviour the user has already tested:** continuous and turn-based modes, the turn key in
   toggle and hold styles, the half-duplex gate (the mic is off while volis speaks), shared-machine
   mode with a key per side and a recognizer per side, Escape to cancel, varieties through all
   three stages, and Whisper only ever told the language part of a variety.
8. **The command line:** `--report`, `--devices`, `--listen` (`--seconds`, `--wav`,
   `--compare`), and, if the Rust side has them by then, `--print-prompt` and `--translate`.
   Same output format, so the two can be compared line by line.

## What changes

| Rust version | Python version | Why |
|---|---|---|
| `sherpa-onnx` crate 1.13.8 | `sherpa-onnx` Python package **1.13.8**, plus `sherpa-onnx-core` 1.13.8 | Same C++ engine, same version, so recognition should match exactly. `model-converter` already found that `sherpa-onnx-core` must be listed by name, or the package picks up an old `onnxruntime.dll` from Windows and crashes. Reuse that finding. |
| `llama-cpp-2` 0.1.156, CPU only (no CUDA features in `Cargo.toml`) | `llama-cpp-python`, CPU build | Choose the release whose bundled llama.cpp is closest to the one in `llama-cpp-sys-2` 0.1.156. P3's check proves it: same GGUF, same input, same output. |
| `eframe` / `egui` | **PySide6** (Qt) | Native widgets, no web view, no JavaScript. Mature threading model: worker threads talk to the window through Qt signals, the same idea as `PipelineMsg`. |
| `cpal` | `sounddevice` | Capture and playback, with device names listed the same way. |
| Resampling in Rust | `soxr` | High-quality resampling to 16 kHz mono at the capture boundary, as in Rust. |
| `std::net` / threads | `socket` / `threading` | Plain sockets, no framework. |
| One `volis.exe` | A folder built with **PyInstaller** (one-folder mode) containing `pyvolis.exe` | Double-click to run, no Python install needed on the target machine. |
| `paths::app_root()` from `current_exe()` | One `app_root()` function: the folder containing `pyvolis.exe` when built, the project folder when run from source | The same rule: exactly one path function. Never the current working directory. |

**Development environment:** uv, with Python and all packages inside the project folder, exactly
as `model-converter`'s `setup.ps1` does (`.uv\`, `.venv\`, `uv sync --locked`). Copy that
approach, and its reasons, rather than inventing a new one.

## Performance: the one real risk

Python runs one piece of Python code at a time. The heavy work (recognition, translation,
speech) happens inside C++ libraries, and those usually let other threads run while they work.
But if one doesn't, the window freezes while it runs.

- Give the pipeline the same thread layout as the Rust version (`ARCHITECTURE.md`, "Threading"):
  capture, recognition, translation on its own thread, speech, playback, and the window.
- Show the same latency readout as the Rust version: recognised, translated, first audio.
- At P5, measure: does the window stay responsive while translating? If not, stop and report
  which library holds up other threads. **Don't** fix it with a separate process. That's the
  "no separate services" rule. The user decides.

It's fine if Python is somewhat slower. It's not fine if it's slower without anyone knowing.
Every milestone check that involves timing prints both versions' numbers side by side.

---

## Layout

```
pyvolis/
  CLAUDE.md             the constraints and working rules, adapted from the Rust version
  README.md             setup and use, short, for a less technical user
  HANDOFF.md            status, differences from Rust, open questions
  pyproject.toml  uv.lock  setup.ps1  build.ps1
  volis_py/
    paths.py  config.py  models.py  varieties.py  report.py
    audio.py  vad.py  asr.py  compare.py  ring.py  wav.py
    translate.py  tts.py  playback.py
    pipeline.py  listen.py  shared.py
    wire.py  floor.py  peer.py  discovery.py
    gui/                PySide6 window
    cli.py  __main__.py
  tests/
  parity/               scripts that compare against the Rust version
  models/  volis.toml   for development; not committed except README.txt files
```

One Python module per Rust module, with the same name. Anyone comparing the two can find the
matching code at once.

---

## Checking against the Rust version

`parity/` holds scripts that run the same input through both versions and compare. The Rust
`volis.exe` path goes in a `machine.yaml`, not committed, like `model-converter`'s.

- **Model discovery:** `--report` from both, on the same `models/` folder, lists the same models
  with the same `ok` or broken status and the same reasons.
- **Recognition:** a folder of test WAVs, transcribed by both. Same sherpa-onnx version, same
  settings, so the text must be **identical**. Any difference means a setting differs. Find it.
- **Translation:** a fixed list of sentences in both directions, including `es-MX` and `ar-IQ`
  targets, through both. If the Rust side has `--translate`, use it. The output should be
  **identical** (same GGUF, greedy decoding). If llama.cpp versions differ slightly, allow a
  small number of differing lines, and print them all for the user to read.
- **Prompt:** if the Rust side has `--print-prompt`, the Python prompt must match it byte for
  byte, for every pair of tags in `varieties.rs`.
- **Wire protocol:** a Python instance and a Rust instance pair, exchange turns in both
  directions, and a killed link is reported on both ends.

---

## Milestones

Named P0 to P9 so they're never confused with the Rust milestones. Each follows the matching Rust
milestone in `SPEC.md` §13. Read that section for each one before starting it.

**P0 — skeleton, paths, config, discovery, report** (Rust M0: `paths.rs`, `config.rs`,
`models.rs`, `varieties.rs`, `report.rs`).
Check: `--report` matches the Rust version's on the same folder. Unit tests for `engine.toml`
parsing (unknown key rejected, bad `varieties` rejected), for `rank`, and for loading a
`volis.toml` with and without every optional section.

**P1 — audio capture and VAD** (Rust M1: `audio.rs`, `vad.rs`).
Check: `--devices` lists the same devices. `--listen --seconds 20 --wav` cuts speech into the
same kind of segments as Rust. Compare segment counts on the same recording played into the mic.

**P2 — recognition** (Rust M2: `asr.rs`, `ring.rs`, `compare.rs`). Parakeet and Whisper, the
ring buffer, and `--compare`.
Check: the recognition parity script. Identical text on every test WAV.

**P3 — translation** (Rust M3: `translate.rs`). On its own thread. The exact prompt, the output
cleaning, and the three guards, with the Rust tests ported.
Check: the translation and prompt parity scripts.

**P4 — speech and playback** (Rust M4: `tts.rs`, `playback.rs`). The half-duplex gate, and the
voice chosen by language and variety.
Check, by hand: speak, and hear the translation. pyvolis never hears its own output.

**P5 — the window** (Rust M5: `gui.rs`, `pipeline.rs`, `listen.rs`). The pipeline reports only
through messages, and the window's state logic has no Qt in it and is unit-tested, as with
`gui::Session`.
Check: the window runs a conversation. The latency readout shows both versions' numbers on the
same sentences. The window stays responsive during translation (see "Performance").

**P6 — continuous and turn-based modes** (Rust M6). Toggle and hold, the mic closed between
turns, a reply held while a turn is taken. The turn key never reaches a widget.
Check, by hand: the Rust M6 checks, repeated.

**P7 — paired mode** (Rust M7: `wire.rs`, `floor.rs`, `peer.rs`, `discovery.rs`).
Check: Python with Python, **then Python with Rust**, both directions. A killed link
force-releases the floor with a clear message on both ends.

**P8 — shared-machine mode, per-side recognizers, varieties** (Rust M7.5–M7.7: `shared.rs`).
Check, by hand: the English and Mexican Spanish tests the user already ran on Rust, with the same
models. Same transcripts and translations.

**P9 — portability** (Rust M9). `build.ps1` makes the PyInstaller folder. Copy in `models/` and
`volis.toml`, zip it, move it to a Windows machine with no internet and no Python, unzip,
double-click.
Check: it works, and it attempts no network connections. Watch with Windows' Resource Monitor,
Network tab, filtered to `pyvolis.exe`. The only allowed traffic is paired mode on the local
network.

**Not ported:** Rust M8 (streaming recognition) hasn't been built yet. When it is, it's ported
the same way.

## PyInstaller traps to check at P9

- **DLLs not collected.** sherpa-onnx's `onnxruntime.dll`, llama.cpp's libraries and PortAudio
  for `sounddevice` all need to end up in the folder. Check each is present, and that the one
  loaded is the one in the folder (the converter's `engine_worker.py` shows how to check which
  `onnxruntime.dll` got loaded).
- **Security software.** The converter's README describes McAfee and Trellix removing DLLs. A
  PyInstaller folder is exactly the kind of thing they flag. If a DLL goes missing on the test
  machine, say so. Don't work around the scanner.
- **The app root.** A built app and running from source find `models/` differently. Test both.

## Documents

pyvolis gets the same document set as Rust, kept short: `README.md` (setup and use),
`CLAUDE.md` (constraints and working rules), `HANDOFF.md` (status, and **every known difference
from the Rust version**). For everything else (models, `engine.toml`, the wire protocol,
dialects) link to the Rust repo's documents. Don't copy them. Each fact lives in one place.
