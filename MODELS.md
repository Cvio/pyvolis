# Models: what has been tested, how it scored, how to get it

Every model pyvolis has been run with, including the ones that weren't worth keeping. What a
recognizer, translator and voice *are*, and how a folder describes itself with `engine.toml`,
is in [Rust volis's MODELS.md](../volis/MODELS.md); this page doesn't repeat it. How pyvolis
finds a model you drop in yourself is in [README.md](README.md#models).

## Getting them

`fetch-models.ps1` downloads the list on this page into the right folders and writes the
settings file each one needs (`engine.toml` for sherpa-onnx models and voices, `pyvolis.toml`
for GGUF speech models). It uses the internet; pyvolis itself never does.

Before you run it:

- **Cohere Transcribe is gated.** Accept its terms at
  <https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026> in your browser, then
  run `.\fetch-model.ps1 -Login` once. Without that, everything else still downloads and the
  script says which one didn't.
- **The whole default list is about 50 GB** on a machine that has none of it. `-List` shows
  the exact figure and downloads nothing.

```powershell
.\fetch-models.ps1 -List              # what is in place, what would be downloaded, how big
.\fetch-models.ps1                    # groups "use" and "bigger" (below)
.\fetch-models.ps1 -Group use         # only what is worth having on an 8 GB GPU
.\fetch-models.ps1 -Group all         # everything on this page
.\fetch-models.ps1 -Only Qwen3-ASR    # only entries whose name contains this
.\.venv\Scripts\python.exe -m pyvolis --report    # afterwards: every model should say ok
```

It asks once before downloading. What is already in place (every file, at its published size)
is skipped, so running it again retries only what failed. A settings file that already exists
is never overwritten. The list itself, with every source, is the `LIST` at the top of
[scripts/fetch_models.py](scripts/fetch_models.py): add a model there and it is on the list.

The groups:

| Group | Meaning | Downloaded by default |
|---|---|---|
| `use` | tested on the development laptop (RTX 4070, 8 GB) and worth having | yes |
| `bigger` | tested, but too large for 8 GB; to be measured on a larger GPU | yes |
| `untested` | a sibling of a tested model; this one has never been run | no |
| `tested` | tested and not kept: a worse score, or only there to prove a backend | no |

To fetch one model that isn't on the list, use `.\fetch-model.ps1 <id> -Role asr|mt`
(README). Voices are not covered by `fetch-model.ps1`; see "Voices" below.

## How the numbers were measured

All on the development laptop, with the GPU build of llama.cpp. They are small tests: read
them as "clearly better", "about the same" or "clearly worse", not to the decimal.

- **CER / WER** (recognizers): character and word error rate against the reference
  transcript, lower is better. "Clips" = 10 FLEURS clips a language (`scripts/transcribe.py`).
  "File" = one two-minute recording of FLEURS sentences through the whole pipeline
  (`scripts/p7_check.py`, `scripts/p11_check.py`). All of it is read speech in a quiet room:
  none of it says how a model hears *you*, or a dialect.
- **RTF**: time to transcribe divided by the length of the audio. 0.1 = ten times faster
  than speech.
- **chrF** (translators): overlap with the FLORES+ reference translation, higher is better.
  Ten sentences a language (`scripts/mt_bench.py`), so one or two points is noise.
- **obey**: of 24 sentences that tempt a chat model to answer instead of translate ("how do
  you say ...", "what is the capital of France"), how many were translated, alone / after two
  earlier sentences (`scripts/obey_check.py`).

## Recognizers (`models\asr\`)

| Model | Group | Runs on | Disk | Spanish CER / WER | Arabic CER / WER | RTF | Notes |
|---|---|---|---|---|---|---|---|
| Gemma 4 E4B (Q4_K_M + audio encoder) | use | llama.cpp audio, GPU | 1.0 GB + the translator's file | 1.1% / 3.4% (file) | **1.5% / 6.5%** (file) | 0.08 | The best Arabic measured. The same file is the Gemma 4 translator; loaded as both it is in memory twice. No word timings. |
| Cohere Transcribe Arabic 07-2026 | use | transformers, GPU, 4.1 GB | 4.1 GB | not tested | 1.9% / 7.4% (file), 2.1% (clips) | 0.19 | Arabic and English only. Gated. The best dedicated Arabic recognizer measured. |
| Qwen3-ASR 0.6B (Q8) | use | llama.cpp audio, GPU | 1.0 GB | 1.6% / 6.0% (file) | 4.1% / 16.2% (file) | **0.02** | The fastest recognizer installed. No word timings. |
| whisper-large-v3-turbo-es (adriszmar) | use | transformers, GPU, 1.6 GB | 3.2 GB | **0.9% / 3.0%** (file) | Spanish only | 0.34 | The Spanish recognizer every check uses. |
| whisper-large-v3-turbo-arabic-dialectal (oddadmix), safetensors | use | transformers, GPU, 1.6 GB | 3.2 GB | Arabic only | 3.7% (clips) | not recorded | Tuned on dialects; the clips are Standard Arabic, so this test undersells it. Not yet tested on dialect speech. |
| MMS 1B (adapters: ar, en, fa, es) | use | transformers, GPU, 1.9 GB | 3.9 GB | 1.3% / 6.0% (file) | 5.8% (clips) | 0.04 | Never writes punctuation or capitals, so sentences are cut by pauses alone. The only one here with a Persian adapter besides Whisper. |
| Parakeet TDT 0.6B v3 (int8) | use | sherpa-onnx, CPU | 0.7 GB | 0.5 to 1.9% (clips) | no Arabic | not recorded | Rust volis's. Detects the language itself; spells numbers out. Identical output to Rust. |
| Whisper large-v3-turbo (int8) | use | sherpa-onnx, CPU | 1.0 GB | 0.5 to 1.9% (clips) | 5.2% (clips), no punctuation on any Arabic output | not recorded | Rust volis's general recognizer. |
| Qwen3-ASR 1.7B (Q8) | untested | llama.cpp audio | 2.5 GB | | | | The larger sibling of Qwen3-ASR 0.6B. Same code path. |
| Voxtral Mini 3B (Q4_K_M) | untested | llama.cpp audio | 3.2 GB | | | | Asked in words, as Gemma 4 is. |
| whisper-small | tested | transformers, GPU | 1.0 GB | not tested | 7.4% / 22.7% (file) | 0.13 | Clearly worse than everything above. Kept on the list only as the base of the adapter below. |
| whisper-algerian-darja-small (LoRA on whisper-small) | tested | transformers + peft | 0.1 GB | | 12.6% / 43.1% (file) | 0.13 | Proves that a LoRA adapter loads and is applied. Worse than its base on this recording, as expected: the recording is Egyptian read speech, the adapter is for Algerian. |

Two more recognizers are in `models\asr\` on the development machine and are **not
downloadable**: `whisper-large-v3-turbo-es-adriszmar` and
`whisper-large-v3-turbo-arabic-dialectal` are the two fine-tunes above converted to int8 ONNX
for Rust volis by [model-converter](../model-converter/README.md). pyvolis runs the originals
instead, which score better (Arabic clips: 3.7% as published, 8.4% as int8 ONNX, which cuts
sentences short). To make them, follow model-converter's README.

**To use one:** pick it under **Recognizer** in the window, or `--asr <folder name>` on the
command line, or `[asr] engine = "<folder name>"` in `volis.toml` (Rust volis can only load the
two sherpa-onnx ones, so set a pyvolis-only one in the window instead if you also run Rust).

What each kind of folder needs, which the script takes care of:

- **sherpa-onnx** (Parakeet, Whisper int8): the `.onnx` files and an `engine.toml`.
- **transformers** (Whisper, MMS, Cohere): the folder exactly as Hugging Face publishes it. No
  settings file.
- **GGUF speech model** (Qwen3-ASR, Gemma 4, Voxtral): the model `.gguf` and its audio encoder
  `mmproj-*.gguf` in one folder, and optionally a `pyvolis.toml` with a display name and the
  languages. **Never an `engine.toml`**: that sends the folder to the sherpa-onnx loader and
  the model shows as broken.
- **Gemma 4 as a recognizer** uses the translator's own `.gguf`. The script hard-links it from
  `models\mt\gemma-4-E4B-it-GGUF\` (no extra disk) and downloads only the 1.0 GB audio encoder.
- **LoRA adapter**: `adapter_config.json` and `adapter_model.safetensors`, with the base
  model's folder beside it.

The language lists in the two GGUF `pyvolis.toml` files are from the model cards. Only
English, Spanish and Arabic have been run.

## Translators (`models\mt\`)

Text only, FLEURS transcripts against FLORES+ references, pyvolis's default prompt with
carry-forward context, 2026-10-01. All are Q4_K_M GGUF files run by llama.cpp on the GPU.

| Model | Group | Disk | es>en | ar>en | fa>en | en>es | mean chrF | obey | ms / sentence | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3 1.7B | use | 1.1 GB | 59.2 | 64.3 | 58.8 | 58.3 | 60.2 | 24 / 24 | **261** | Rust volis's translator: the one `.gguf` at the top of `models\mt\`. The smallest and fastest, 3 to 6 chrF below the rest. |
| Gemma 3 4B | use | 2.5 GB | **65.9** | 67.7 | 65.5 | 60.0 | 64.8 | 24 / 24 | 444 | The best all-rounder that fits 8 GB with a recognizer loaded. |
| Gemma 4 E4B | use | 5.0 GB | 64.3 | 71.3 | 63.8 | 60.9 | 65.1 | 24 / 23 | 586 | Clearly better Arabic than the other small ones. Also a recognizer (above). |
| TranslateGemma 4B | use | 2.5 GB | 64.2 | 66.9 | 65.9 | 60.0 | 64.3 | 23 / 24 | 511 | Trained only to translate. Its own prompt format: the prompt file and the glossary are not used. The one revision mode helps most. |
| Gemma 4 12B | bigger | 7.1 GB | 62.2 | **73.1** | 66.6 | **62.7** | **66.1** | 24 / 24 | 5909 * | The best Arabic and the best mean measured. |
| Gemma 3 12B | bigger | 7.3 GB | 63.1 | 70.3 | **66.9** | 60.5 | 65.2 | 24 / 24 | 9313 * | |
| TranslateGemma 12B | bigger | 7.3 GB | 62.3 | 68.4 | 64.9 | 61.2 | 64.2 | 24 / 24 | 8040 * | |
| Qwen3 8B | tested | 5.0 GB | 64.3 | 67.3 | 62.3 | 59.9 | 63.4 | 24 / 23 | 867 | No better than the 4B models and slower. |
| Qwen3 0.6B (safetensors) | tested | 1.5 GB | 58.8 on the Spanish file | | | | | | 1974 | Through transformers, not llama.cpp: proves that backend. Five times slower a sentence than a GGUF of a larger model. |

\* didn't fit the laptop's 8 GB card and spilled into ordinary memory. These times say nothing
about a larger GPU, which is why the three are in the `bigger` group: their quality is
measured, their speed is not.

What holds across the table: every model from 4B up is 3 to 6 chrF above Qwen3 1.7B, and the
12B models are not clearly better than the 4B ones except on Arabic.

**To use one:** pick it under **Translator** in the window, or `--mt <id>` (as `--report`
lists it, for example `gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf`), or `[translate] model =
"<id>"` in `pyvolis.toml`. `""` means the `.gguf` at the top of `models\mt\`.

What the folder needs: the `.gguf` alone, in a folder of its own. **Only one `.gguf` may sit at
the top level of `models\mt\`** (Rust volis refuses to start with two); the script puts Qwen3
1.7B there, under the name Rust expects, and every other translator in a folder.

**LoRA adapters for a GGUF translator** work (an adapter of strength 0 translates exactly as
its base, 14 of 14 sentences; a stronger one changes the output), but only synthetic adapters
have been tried: no trained one exists for an installed translator. README says how to add
one.

## Voices (`models\tts\`)

Voices have no score here. Each was listened to, and each speaks through the pipeline.

| Voice | Group | Language | Disk | Notes |
|---|---|---|---|---|
| vits-piper-en_US-lessac-medium | use | English (US) | 64 MB | Rust volis's. |
| vits-piper-es_MX-claude-high | use | Spanish (Mexico) | 67 MB | Rust volis's. Chosen first for Spanish (Mexico). |
| vits-piper-es_ES-carlfm-x_low | use | Spanish (Spain) | 25 MB | Rust volis's. |
| vits-piper-ar_JO-kareem-medium | use | Arabic, male | 64 MB | Mispronounces some words written without vowel marks, as every Arabic Piper voice here does. |
| arabic-emirati-female-model | use | Arabic (Emirati), female | 127 MB | Published as a raw Piper model, which sherpa-onnx can't load. The script converts it (below). MIT licence. |
| vits-piper-ar_JO-SA_dii-high | tested | Arabic, male | 64 MB | Works. **Its licence is non-commercial**; read its README before using it, which is why it isn't downloaded by default. |

A voice folder needs four things: the `.onnx` model **with sherpa-onnx's fields inside it**,
`tokens.txt`, the `espeak-ng-data` folder, and an `engine.toml` (`backend = "vits"`). The
sherpa-onnx releases and the `csukuangfj/vits-piper-*` repos on Hugging Face come with the
first three; the script writes the `engine.toml`.

A voice from anywhere else (`<name>.onnx` and `<name>.onnx.json`, as Piper itself publishes
them) has none of the other three. The script makes them: `tokens.txt` from the phoneme table
in the `.json`, a copy of the model named `<name>.sherpa.onnx` with the fields added (the
original is left untouched), and `espeak-ng-data` copied from another installed voice. To do
that for a voice not on the list, add an entry with `piper_language=` to the `LIST`.

The three Arabic voices declare no variety: `ar-AE` and `ar-JO` are not in the varieties
table, and an unknown variety disables the folder.

**To use one:** nothing to set. The voice is chosen from the language translated into, tuned
for the variety first. In shared machine mode each side has a **Spoken by** picker.

## Also required

`models\vad\silero_vad.onnx` (2 MB), the voice activity detector, from the sherpa-onnx
releases. The script fetches it. It must keep that name.

## Where each comes from

| Model | Source |
|---|---|
| silero_vad.onnx, Parakeet, Whisper int8 | [sherpa-onnx ASR releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) |
| English and Spanish voices | [sherpa-onnx TTS releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models) |
| Qwen3 1.7B | [unsloth/Qwen3-1.7B-GGUF](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF), saved as `qwen3-1.7b-q4_k_m.gguf` as Rust volis does |
| whisper-large-v3-turbo-es | [adriszmar/whisper-large-v3-turbo-es](https://huggingface.co/adriszmar/whisper-large-v3-turbo-es) |
| whisper-large-v3-turbo-arabic-dialectal-hf | [oddadmix/whisper-large-v3-turbo-arabic-dialectal](https://huggingface.co/oddadmix/whisper-large-v3-turbo-arabic-dialectal) |
| cohere-transcribe-arabic-07-2026 | [CohereLabs/cohere-transcribe-arabic-07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) (gated) |
| mms-1b-all | [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) |
| Qwen3-ASR 0.6B, 1.7B | [ggml-org/Qwen3-ASR-0.6B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF), [ggml-org/Qwen3-ASR-1.7B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF) |
| Voxtral Mini 3B | [ggml-org/Voxtral-Mini-3B-2507-GGUF](https://huggingface.co/ggml-org/Voxtral-Mini-3B-2507-GGUF) |
| Gemma 4 E4B (translator and audio encoder), Gemma 4 12B | [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF), [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) |
| Gemma 3 4B, 12B | [unsloth/gemma-3-4b-it-GGUF](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF), [unsloth/gemma-3-12b-it-GGUF](https://huggingface.co/unsloth/gemma-3-12b-it-GGUF) |
| TranslateGemma 4B, 12B | [mradermacher/translategemma-4b-it-GGUF](https://huggingface.co/mradermacher/translategemma-4b-it-GGUF), [mradermacher/translategemma-12b-it-GGUF](https://huggingface.co/mradermacher/translategemma-12b-it-GGUF) |
| Qwen3 8B, Qwen3 0.6B | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF), [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| whisper-small, the Darja adapter | [openai/whisper-small](https://huggingface.co/openai/whisper-small), [touati-kamel/whisper-algerian-darja-small](https://huggingface.co/touati-kamel/whisper-algerian-darja-small) |
| Arabic voices | [csukuangfj/vits-piper-ar_JO-kareem-medium](https://huggingface.co/csukuangfj/vits-piper-ar_JO-kareem-medium), [vadimbelsky/arabic-emirati-female-piper](https://huggingface.co/vadimbelsky/arabic-emirati-female-piper), [csukuangfj/vits-piper-ar_JO-SA_dii-high](https://huggingface.co/csukuangfj/vits-piper-ar_JO-SA_dii-high) |

Each model has its own licence. Check it before you pass a copy on.

## Not measured yet

- Any recognizer on dialect speech, on a noisy room, or on a real conversation.
- Persian recognition (MMS and Whisper have it; there is no Persian reference run).
- The 12B translators' speed on a GPU they fit.
- Qwen3-ASR 1.7B and Voxtral.
- Streaming and shared machine mode with a GGUF speech model.
- A trained (not synthetic) LoRA adapter on a translator.
