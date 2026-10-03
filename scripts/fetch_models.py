"""Download the models in MODELS.md into models/, each in its right folder with
the files pyvolis needs to find, load and run it. Run through
fetch-models.ps1. A development tool: it uses the internet, and the app never
calls it.

The list below is the one place the sources are written down. Each entry is
in a group:

  use       tested here and worth having: downloaded by default
  bigger    tested here but too large for the development laptop's 8 GB card;
            to be measured on a larger GPU: downloaded by default
  untested  the code path is tested with a sibling model, this one is not
  tested    tested and not kept (a worse score, or only there to prove a
            backend): never downloaded unless asked for

What is already in place (every file there, at the published size) is left
alone. A settings file (engine.toml, pyvolis.toml) that exists is never
overwritten.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
os.environ["HF_HOME"] = str(REPO / ".uv" / "hf")
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError  # noqa: E402

import fetch_model  # noqa: E402
from fetch_model import piper_engine  # noqa: E402

MODELS = REPO / "models"
DOWNLOADS = REPO / ".uv" / "downloads"  # archives, removed once unpacked
SHERPA_ASR = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
SHERPA_TTS = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
GROUPS = ("use", "bigger", "untested", "tested")
DEFAULT_GROUPS = ("use", "bigger")


@dataclass
class Model:
    role: str  # vad | asr | mt | tts | tashkeel
    folder: str  # under models/<role>/; "" for a file at the role's top level
    group: str
    what: str
    repo: str = ""  # a Hugging Face repo...
    include: tuple[str, ...] = ()  # ...and which of its files (as fetch-model.ps1 -Include)
    file: str = ""  # or one file of the repo, saved at the role's top level...
    save_as: str = ""  # ...under this name
    url: str = ""  # or a file or .tar.bz2 that is not on Hugging Face
    members: tuple[str, ...] = ()  # of an archive: the files to keep; () = the whole folder
    expect: tuple[str, ...] = ()  # for a url: the files that mean "already here"
    link: str = ""  # a file of another model to hard-link here instead of downloading it twice
    write: dict[str, str] = field(default_factory=dict)  # settings files, written when absent
    piper_language: str = ""  # a raw Piper voice: make it loadable by sherpa-onnx
    gated: bool = False

    @property
    def id(self) -> str:
        return f"{self.role}/{self.folder or self.save_as or Path(self.url).name}"

    @property
    def target(self) -> Path:
        return MODELS / self.role / self.folder if self.folder else MODELS / self.role


GGUF_SPEECH = ("# Settings for this model in pyvolis. No engine.toml here: that file is for\n"
               "# sherpa-onnx models, and would send this one to the wrong loader. The two\n"
               "# .gguf files (the model and its audio encoder, mmproj-*) are found by name.\n")
# Only English, Spanish and Arabic were measured; the rest is from the model cards.
SPEECH_LANGUAGES = 'languages = ["en", "es", "ar", "fa", "de", "fr", "it", "pt", "ru", "nl", "pl"]\n'

LIST = [
    # ------------------------------------------------------------ what Rust volis runs on
    Model("vad", "", "use", "Silero VAD, the voice activity detector (required)",
          url=f"{SHERPA_ASR}/silero_vad.onnx", expect=("silero_vad.onnx",)),
    Model("tashkeel", "", "use", "libtashkeel, the Arabic vowel-marking model Piper uses ([tts] diacritize)",
          url="https://raw.githubusercontent.com/rhasspy/piper-phonemize/master/etc/libtashkeel_model.ort",
          expect=("libtashkeel_model.ort",)),
    Model("mt", "", "use", "Qwen3 1.7B Q4_K_M, the translator Rust volis uses (the one .gguf at the top of mt)",
          repo="unsloth/Qwen3-1.7B-GGUF", file="Qwen3-1.7B-Q4_K_M.gguf", save_as="qwen3-1.7b-q4_k_m.gguf"),
    Model("asr", "parakeet-tdt-0.6b-v3-onnx-int8", "use", "Parakeet TDT 0.6B v3, ONNX int8 (sherpa-onnx, CPU)",
          url=f"{SHERPA_ASR}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
          members=("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"),
          expect=("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"),
          write={"engine.toml": 'name = "Parakeet TDT 0.6B v3 (ONNX int8, CPU)"\nkind = "segment"\nbackend = "nemo_transducer"\n'
                                'languages = ["es", "en", "de", "fr", "it", "pt", "nl", "pl", "ru"]\n\n[files]\n'
                                'encoder = "encoder.int8.onnx"\ndecoder = "decoder.int8.onnx"\n'
                                'joiner  = "joiner.int8.onnx"\ntokens  = "tokens.txt"\n'}),
    Model("asr", "whisper-large-v3-turbo-onnx-int8", "use", "Whisper large-v3-turbo, ONNX int8 (sherpa-onnx, CPU)",
          url=f"{SHERPA_ASR}/sherpa-onnx-whisper-turbo.tar.bz2",
          members=("turbo-encoder.int8.onnx", "turbo-decoder.int8.onnx", "turbo-tokens.txt"),
          expect=("turbo-encoder.int8.onnx", "turbo-decoder.int8.onnx", "turbo-tokens.txt"),
          write={"engine.toml": 'name = "Whisper large-v3-turbo (ONNX int8, CPU)"\nkind = "segment"\nbackend = "whisper"\n'
                                'languages = ["es", "en", "de", "fr", "it", "pt", "nl", "pl", "ru", "ar", "fa"]\n\n'
                                '[files]\nencoder = "turbo-encoder.int8.onnx"\ndecoder = "turbo-decoder.int8.onnx"\n'
                                'tokens  = "turbo-tokens.txt"\n'}),
    Model("tts", "vits-piper-en_US-lessac-medium", "use", "Piper voice, English (US)",
          url=f"{SHERPA_TTS}/vits-piper-en_US-lessac-medium.tar.bz2",
          expect=("en_US-lessac-medium.onnx", "tokens.txt", "espeak-ng-data/phontab"),
          write={"engine.toml": piper_engine("Piper en_US lessac (medium)", "en", "en_US-lessac-medium.onnx", "en-US")}),
    Model("tts", "vits-piper-es_ES-carlfm-x_low", "use", "Piper voice, Spanish (Spain)",
          url=f"{SHERPA_TTS}/vits-piper-es_ES-carlfm-x_low.tar.bz2",
          expect=("es_ES-carlfm-x_low.onnx", "tokens.txt", "espeak-ng-data/phontab"),
          write={"engine.toml": piper_engine("Piper es_ES carlfm (x_low)", "es", "es_ES-carlfm-x_low.onnx", "es-ES")}),
    Model("tts", "vits-piper-es_MX-claude-high", "use", "Piper voice, Spanish (Mexico)",
          url=f"{SHERPA_TTS}/vits-piper-es_MX-claude-high.tar.bz2",
          expect=("es_MX-claude-high.onnx", "tokens.txt", "espeak-ng-data/phontab"),
          write={"engine.toml": piper_engine("Piper es_MX claude (high)", "es", "es_MX-claude-high.onnx", "es-MX")}),

    # ------------------------------------------------------------ recognizers, pyvolis only
    Model("asr", "whisper-large-v3-turbo", "use", "Whisper large-v3-turbo as published, all languages (transformers, GPU)",
          repo="openai/whisper-large-v3-turbo"),
    Model("asr", "whisper-large-v3-turbo-es", "use", "Whisper large-v3-turbo tuned for Spanish (transformers, GPU)",
          repo="adriszmar/whisper-large-v3-turbo-es"),
    Model("asr", "whisper-large-v3-turbo-arabic-dialectal-hf", "use",
          "Whisper large-v3-turbo tuned for dialectal Arabic (transformers, GPU)",
          repo="oddadmix/whisper-large-v3-turbo-arabic-dialectal"),
    Model("asr", "cohere-transcribe-arabic-07-2026", "use",
          "Cohere Transcribe Arabic (transformers, GPU; gated: accept its terms, then fetch-model.ps1 -Login)",
          repo="CohereLabs/cohere-transcribe-arabic-07-2026", gated=True),
    Model("asr", "mms-1b-all", "use", "MMS 1B with the Arabic, English, Persian and Spanish adapters (transformers, GPU)",
          repo="facebook/mms-1b-all",
          include=("model.safetensors", "adapter.ara.safetensors", "adapter.eng.safetensors",
                   "adapter.fas.safetensors", "adapter.spa.safetensors")),
    Model("asr", "Qwen3-ASR-0.6B-GGUF", "use", "Qwen3-ASR 0.6B Q8 and its audio encoder (llama.cpp audio)",
          repo="ggml-org/Qwen3-ASR-0.6B-GGUF", include=("*Q8_0.gguf",),
          write={"pyvolis.toml": GGUF_SPEECH + 'name = "Qwen3-ASR 0.6B (Q8)"\n' + SPEECH_LANGUAGES}),
    Model("asr", "gemma-4-E4B-it-GGUF", "use",
          "Gemma 4 E4B as a recognizer: its audio encoder, and a hard link to the translator's file",
          repo="unsloth/gemma-4-E4B-it-GGUF", include=("mmproj-F16.gguf",),
          link="mt/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf",
          write={"pyvolis.toml": GGUF_SPEECH + 'name = "Gemma 4 E4B (Q4_K_M, audio)"\n' + SPEECH_LANGUAGES + (
              "\n# No prompt line: pyvolis's own wording is used (\"Transcribe this audio exactly\n"
              "# as spoken, in {language}. Output only the transcript, ...\"). The wording on\n"
              "# Google's model card was tried and scored worse on the test recordings\n"
              "# (Arabic CER 3.4% against 1.5%, Spanish 1.7% against 1.1%). To try another:\n"
              '# prompt = "Transcribe the following speech segment in {language} into {language} text."\n')}),
    Model("asr", "Qwen3-ASR-1.7B-GGUF", "tested", "Qwen3-ASR 1.7B (Q8)",
          repo="ggml-org/Qwen3-ASR-1.7B-GGUF", include=("*Q8_0.gguf",),
          write={"pyvolis.toml": GGUF_SPEECH + 'name = "Qwen3-ASR 1.7B (Q8)"\n' + SPEECH_LANGUAGES}),
    Model("asr", "whisper-small", "tested", "Whisper small (transformers); the base of the LoRA adapter below",
          repo="openai/whisper-small"),
    Model("asr", "whisper-algerian-darja-small", "tested",
          "a LoRA adapter on whisper-small (Algerian Darja); needs whisper-small beside it",
          repo="touati-kamel/whisper-algerian-darja-small",
          include=("adapter_config.json", "adapter_model.safetensors")),

    # ------------------------------------------------------------ translators, pyvolis only
    Model("mt", "gemma-3-4b-it-GGUF", "use", "Gemma 3 4B Q4_K_M",
          repo="unsloth/gemma-3-4b-it-GGUF", include=("*Q4_K_M.gguf",)),
    Model("mt", "gemma-4-E4B-it-GGUF", "use", "Gemma 4 E4B Q4_K_M",
          repo="unsloth/gemma-4-E4B-it-GGUF", include=("gemma-4-E4B-it-Q4_K_M.gguf",)),
    Model("mt", "translategemma-4b-it-GGUF", "use", "TranslateGemma 4B Q4_K_M",
          repo="mradermacher/translategemma-4b-it-GGUF", include=("*.Q4_K_M.gguf",)),
    Model("mt", "gemma-4-12b-it-GGUF", "bigger", "Gemma 4 12B Q4_K_M (best Arabic measured; needs more than 8 GB of GPU memory)",
          repo="unsloth/gemma-4-12b-it-GGUF", include=("gemma-4-12b-it-Q4_K_M.gguf",)),
    Model("mt", "gemma-3-12b-it-GGUF", "bigger", "Gemma 3 12B Q4_K_M (needs more than 8 GB of GPU memory)",
          repo="unsloth/gemma-3-12b-it-GGUF", include=("gemma-3-12b-it-Q4_K_M.gguf",)),
    Model("mt", "translategemma-12b-it-GGUF", "bigger", "TranslateGemma 12B Q4_K_M (needs more than 8 GB of GPU memory)",
          repo="mradermacher/translategemma-12b-it-GGUF", include=("*.Q4_K_M.gguf",)),
    Model("mt", "Qwen3-8B-GGUF", "tested", "Qwen3 8B Q4_K_M (no better than the 4B models, and slower)",
          repo="Qwen/Qwen3-8B-GGUF", include=("Qwen3-8B-Q4_K_M.gguf",)),
    Model("mt", "Qwen3-0.6B", "tested", "Qwen3 0.6B safetensors (only there to prove the transformers translator)",
          repo="Qwen/Qwen3-0.6B"),

    # ------------------------------------------------------------ Arabic voices
    Model("tts", "vits-piper-ar_JO-kareem-medium", "use", "Piper voice, Arabic, male",
          repo="csukuangfj/vits-piper-ar_JO-kareem-medium",
          write={"engine.toml": piper_engine("Piper ar_JO kareem (medium)", "ar", "ar_JO-kareem-medium.onnx")}),
    Model("tts", "arabic-emirati-female-model", "use", "Piper voice, Arabic (Emirati), female; a raw Piper model, converted here",
          repo="vadimbelsky/arabic-emirati-female-piper", piper_language="Arabic",
          write={"engine.toml": piper_engine(
              "Piper ar_AE Emirati female", "ar", "arabic-emirati-female-model.sherpa.onnx",
              note="# The download is a raw Piper model. sherpa-onnx needs a few fields inside the model file,\n"
                   "# so this is a copy with them added; arabic-emirati-female-model.onnx is the original, untouched.\n")}),
    Model("tts", "vits-piper-ar_JO-SA_dii-high", "tested",
          "Piper voice, Arabic, male. Works, but its licence is non-commercial: read its README before using it",
          repo="csukuangfj/vits-piper-ar_JO-SA_dii-high",
          write={"engine.toml": piper_engine("Piper ar dii (high)", "ar", "ar_JO-SA_dii-high.onnx")}),
]


# ---------------------------------------------------------------- what an entry needs


class Skip(Exception):
    """This entry can't be fetched; the reason is the message."""


def http_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "pyvolis-fetch-models"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.headers.get("Content-Length") or 0)


def repo_files(api: HfApi, m: Model) -> dict[str, int]:
    """The files to take from a model's repo, with their sizes."""
    try:
        info = api.model_info(m.repo, files_metadata=True)
    except GatedRepoError:
        raise Skip(f"{m.repo} is gated. Accept its terms at https://huggingface.co/{m.repo} in your browser, "
                   "then run .\\fetch-model.ps1 -Login, then this again.") from None
    except RepositoryNotFoundError:
        raise Skip(f"no model {m.repo} on Hugging Face (or it is private or gated: .\\fetch-model.ps1 -Login)") from None
    sizes = {s.rfilename: s.size or 0 for s in info.siblings or []}
    if m.file:
        if m.file not in sizes:
            raise Skip(f"{m.repo} has no file {m.file}")
        return {m.file: sizes[m.file]}
    try:
        wanted = fetch_model.choose(list(sizes), m.role, list(m.include))
    except SystemExit as e:
        raise Skip(str(e)) from None
    return {f: sizes[f] for f in wanted}


def plan(api: HfApi, m: Model) -> tuple[int, dict[str, int]]:
    """(bytes still to download, the repo files still to fetch). 0 = in place."""
    if m.url:
        if all((m.target / name).is_file() for name in m.expect):
            return 0, {}
        return http_size(m.url), {}
    files = repo_files(api, m)
    if m.file:
        if (m.target / m.save_as).is_file():
            return 0, {}
        others = [p.name for p in m.target.glob("*.gguf")] if m.target.is_dir() else []
        if others:
            raise Skip(f"{m.target} already has {others[0]} at its top level, and Rust volis refuses two. "
                       "Move it into a folder of its own, or leave this one out.")
        return files[m.file], files
    missing = {f: size for f, size in files.items()
               if not (m.target / f).is_file() or (size and (m.target / f).stat().st_size != size)}
    return sum(missing.values()), missing


# ---------------------------------------------------------------- fetching


def download(url: str, to: Path) -> None:
    to.parent.mkdir(parents=True, exist_ok=True)
    part = to.with_name(to.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "pyvolis-fetch-models"})
    with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
        total, done, shown = int(response.headers.get("Content-Length") or 0), 0, -1
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total and (percent := done * 100 // total) // 10 != shown:
                shown = percent // 10
                print(f"    {percent}% of {total / 1e6:.0f} MB", flush=True)
    part.replace(to)


def fetch_url(m: Model) -> None:
    name = Path(m.url).name
    if not name.endswith(".tar.bz2"):
        download(m.url, m.target / name)
        return
    archive = DOWNLOADS / name
    download(m.url, archive)
    unpacked = DOWNLOADS / name.removesuffix(".tar.bz2")
    shutil.rmtree(unpacked, ignore_errors=True)
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(DOWNLOADS, filter="data")
    if not unpacked.is_dir():
        raise Skip(f"{archive} did not unpack to {unpacked}")
    m.target.mkdir(parents=True, exist_ok=True)
    if m.members:
        for member in m.members:
            shutil.copy2(unpacked / member, m.target / member)
    else:
        shutil.copytree(unpacked, m.target, dirs_exist_ok=True)
    shutil.rmtree(unpacked)
    archive.unlink()


def fetch_repo(m: Model, missing: dict[str, int]) -> None:
    if m.file:
        m.target.mkdir(parents=True, exist_ok=True)
        got = Path(hf_hub_download(m.repo, m.file, local_dir=DOWNLOADS / "hf"))
        shutil.move(got, m.target / m.save_as)
        return
    wanted = list(missing)
    if m.link:
        source, here = MODELS / m.link, m.target / Path(m.link).name
        if source.is_file() and not here.exists():
            m.target.mkdir(parents=True, exist_ok=True)
            os.link(source, here)  # the same bytes: no more disk
            print(f"    linked {here.name} to {source}")
        elif not here.exists():
            wanted.append(Path(m.link).name)  # the translator isn't installed: fetch the file here
    if wanted:
        snapshot_download(m.repo, local_dir=m.target, allow_patterns=wanted)


def finish(m: Model) -> None:
    """Whatever the download itself doesn't bring: settings, and a raw Piper
    voice made loadable."""
    if m.piper_language:
        try:
            fetch_model.convert_piper(m.target, m.piper_language)
        except SystemExit as e:
            raise Skip(str(e)) from None
    for name, text in m.write.items():
        path = m.target / name
        if not path.exists():
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"    wrote {path}")


# ---------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", default=[], help="use, bigger, untested, tested or all")
    parser.add_argument("--only", action="append", default=[], help="a folder name, or part of one")
    parser.add_argument("--list", action="store_true", help="show what would be downloaded; download nothing")
    parser.add_argument("--yes", action="store_true", help="don't ask before downloading")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    groups = set(args.group or DEFAULT_GROUPS)
    if unknown := groups - set(GROUPS) - {"all"}:
        print(f"STOP: no group {sorted(unknown)}; the groups are {', '.join(GROUPS)}, all", file=sys.stderr)
        return 1
    if args.only:
        chosen = [m for m in LIST if any(o.lower() in m.id.lower() for o in args.only)]
        if not chosen:
            print(f"STOP: nothing in the list matches {args.only}", file=sys.stderr)
            return 1
    else:
        chosen = [m for m in LIST if "all" in groups or m.group in groups]

    api, todo, failed = HfApi(), [], []
    print(f"{'':2}{'GROUP':9} {'MODEL':48} SIZE")
    for m in chosen:
        try:
            size, missing = plan(api, m)
        except Skip as e:
            failed.append((m, str(e)))
            print(f"! {m.group:9} {m.id:48} {e}")
            continue
        except OSError as e:
            failed.append((m, f"could not reach its source: {e}"))
            print(f"! {m.group:9} {m.id:48} could not reach its source: {e}")
            continue
        settings = [n for n in m.write if not (m.target / n).exists()]
        if size or missing:
            todo.append((m, size, missing))
            note = " (gated: needs .\\fetch-model.ps1 -Login)" if m.gated else ""
            print(f"+ {m.group:9} {m.id:48} {size / 1e9:6.2f} GB to download{note}")
        else:
            needs = m.piper_language and not list(m.target.glob("*.sherpa.onnx"))
            if settings or needs:
                todo.append((m, 0, {}))
            print(f"  {m.group:9} {m.id:48} in place" + (f", {', '.join(settings)} to write" if settings else ""))
        print(f"  {'':9} {m.what}")

    total = sum(size for _, size, _ in todo)
    free = shutil.disk_usage(REPO).free
    print(f"\n{len([t for t in todo if t[1]])} to download, {total / 1e9:.1f} GB; "
          f"{free / 1e9:.0f} GB free on {REPO.anchor}")
    if args.list or not todo:
        return 1 if failed and not args.list else 0
    if total * 1.1 > free:  # an archive is on disk beside what it unpacks to, briefly
        print(f"STOP: not enough room on {REPO.anchor} for {total / 1e9:.1f} GB", file=sys.stderr)
        return 1
    if total and not args.yes and input(f"Download {total / 1e9:.1f} GB into {MODELS}? [y/N] ").strip().lower() != "y":
        print("nothing downloaded")
        return 1

    for m, size, missing in todo:
        print(f"\n=== {m.id}" + (f" ({size / 1e9:.2f} GB)" if size else ""))
        try:
            if size or missing:
                fetch_url(m) if m.url else fetch_repo(m, missing)
            finish(m)
        except Skip as e:
            failed.append((m, str(e)))
            print(f"!!! {e}", file=sys.stderr)
        except GatedRepoError:
            failed.append((m, "gated"))
            print(f"!!! {m.repo} is gated. Accept its terms at https://huggingface.co/{m.repo} in your browser, "
                  "then run .\\fetch-model.ps1 -Login, then this again.", file=sys.stderr)
        except Exception as e:  # one failed download must not stop the rest
            failed.append((m, str(e)))
            print(f"!!! {m.id} failed: {e}", file=sys.stderr)

    if failed:
        print("\nNot done:")
        for m, why in failed:
            print(f"  {m.id}: {why}")
        print("Run this again to retry just those.")
        return 1
    print("\ndone. Check them with: .\\.venv\\Scripts\\python.exe -m pyvolis --report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
