"""The P11 check: each new backend runs a fixture recording end to end.

    .venv\\Scripts\\python.exe scripts\\p11_check.py

Recognizers, on the Spanish and the Arabic fixture recordings (transcript CER
and WER against the reference, and the recognizer's real-time factor):

  - llamacpp-audio: a GGUF speech model through llama.cpp's audio input
    (Qwen3-ASR, and Gemma 4 with its audio encoder);
  - a LoRA adapter on a transformers recognizer, beside its base alone.

Translators, on the Spanish recording with one recognizer (chrF against the
reference, and the time per sentence):

  - a transformers translator (a safetensors language model);
  - a GGUF translator with a LoRA adapter: a synthetic adapter of strength 0
    must translate exactly as its base does, and one of strength above 0 must
    not. The adapters are made by scripts\\make_test_lora.py in a folder this
    script adds to models\\mt\\ and removes again.

A model that isn't installed is skipped, and said to be.
"""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pyvolis import paths  # noqa: E402

paths.apply_offline_environment(paths.app_root())
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

import logging  # noqa: E402

logging.disable(logging.WARNING)

from pyvolis import events as ev, scoring  # noqa: E402
from pyvolis.config import Config, PyvolisConfig  # noqa: E402
from pyvolis.filesource import read_16k_mono  # noqa: E402
from pyvolis.pipeline import ArraySource, Options, Pipeline, run_to_end  # noqa: E402

FILES = REPO / "tests" / "fixtures" / "files"
BASE_MT = "qwen3-1.7b-q4_k_m.gguf"
TEST_LORA = "p11-test-lora"


def run_file(root: Path, name: str, source: str, asr: str, mt: str = "", translate: bool = False) -> dict | None:
    path = FILES / name
    if not path.is_file():
        print(f"  (no {path}; run scripts\\make_fixture_file.py)")
        return None
    reference = json.loads(path.with_name(path.name + ".ref.json").read_text(encoding="utf-8"))
    config = Config.parse(f'[asr]\nengine = "{asr}"\n[languages]\nsource = "{source}"\ntarget = "en"\n')
    pyconfig, _ = PyvolisConfig.load(paths.pyvolis_config_file(root))
    events: queue.Queue = queue.Queue()
    options = Options(translate=translate, mt=mt, streaming=False, context="carry", prompt="default")
    out = run_to_end(Pipeline(root, config, options, events, ArraySource(read_16k_mono(path)), pyconfig), events)
    if out.errors:
        return {"error": "; ".join(out.errors)}
    transcript = " ".join(e.text for e in out.of(ev.SentenceMsg))
    translated = {e.id: e.text for e in out.of(ev.Translated)}
    translation = " ".join(translated.get(s.id, "") for s in out.of(ev.SentenceMsg))
    stats = out.of(ev.Summary)[-1].stats
    loaded = {e.role: e for e in out.of(ev.ModelLoaded)}
    result = {
        "cer": scoring.cer(reference["transcript"], transcript, source),
        "wer": scoring.wer(reference["transcript"], transcript, source),
        "rtf": stats["asr_rtf"], "sentences": len(out.of(ev.SentenceMsg)), "dropped": len(out.of(ev.Dropped)),
        "asr_device": loaded["recognizer"].device if "recognizer" in loaded else "?",
        "translations": [translated.get(s.id, "") for s in out.of(ev.SentenceMsg)],
    }
    if translate:
        result["chrf"] = scoring.chrf(reference["translation"], translation, "en")
        result["ms"] = stats["translate_ms_median"]
        result["not_translated"] = stats["not_translated"]
        result["mt_device"] = loaded["translator"].device if "translator" in loaded else "?"
    return result


def recognizers(root: Path) -> bool:
    ok = True
    print("\nRecognizers (no translation)")
    rows = [
        ("llamacpp-audio", "Qwen3-ASR-0.6B-GGUF", ("es", "ar")),
        ("llamacpp-audio", "gemma-4-E4B-it-GGUF", ("es", "ar")),
        ("for comparison", "whisper-small", ("ar",)),
        ("LoRA on whisper-small", "whisper-algerian-darja-small", ("ar",)),
    ]
    for kind, folder, languages in rows:
        if not (paths.asr_dir(root) / folder).is_dir():
            print(f"  {kind}: {folder} is not installed; skipped")
            continue
        for language in languages:
            name = {"es": "es_419-to-en.wav", "ar": "ar_eg-to-en.wav"}[language]
            r = run_file(root, name, language, folder)
            if r is None:
                continue
            if "error" in r:
                ok = False
                print(f"  FAIL {kind:22s} {folder:30s} [{language}] {r['error']}")
                continue
            good = r["sentences"] > 0 and r["cer"] < 60
            ok &= good
            print(f"  {'ok  ' if good else 'FAIL'} {kind:22s} {folder:30s} [{language}] CER {r['cer']:5.1f}%  "
                  f"WER {r['wer']:5.1f}%  RTF {r['rtf']}  on the {r['asr_device'].upper()}  "
                  f"{r['sentences']} sentences, {r['dropped']} dropped")
    return ok


def translators(root: Path) -> bool:
    ok = True
    asr = "whisper-large-v3-turbo-es"
    print(f"\nTranslators (Spanish recording, recognised by {asr})")
    mt_dir = paths.mt_dir(root)
    base = run_file(root, "es_419-to-en.wav", "es", asr, BASE_MT, translate=True)

    def show(label: str, r: dict | None, good: bool = True) -> None:
        nonlocal ok
        if r is None:
            return
        if "error" in r:
            ok = False
            print(f"  FAIL {label}: {r['error']}")
            return
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {label:44s} chrF {r['chrf']:4.1f}  {r['ms']} ms a sentence on the "
              f"{r['mt_device'].upper()}  {r['not_translated']} not translated")

    show(f"GGUF, the base: {BASE_MT}", base)
    if (mt_dir / "Qwen3-0.6B").is_dir():
        r = run_file(root, "es_419-to-en.wav", "es", asr, "Qwen3-0.6B", translate=True)
        show("transformers: Qwen3-0.6B (safetensors)", r, bool(r) and "error" not in r and r["chrf"] > 30)
    else:
        print("  transformers translator: Qwen3-0.6B is not installed; skipped")

    folder = mt_dir / TEST_LORA
    try:
        for strength in (0.0, 0.02):
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir()
            subprocess.run([sys.executable, str(REPO / "scripts" / "make_test_lora.py"), str(mt_dir / BASE_MT),
                            str(folder / "test-adapter.gguf"), "--strength", str(strength)], check=True,
                           capture_output=True)
            (folder / "pyvolis.toml").write_text(f'base = "{BASE_MT}"\n', encoding="utf-8")
            r = run_file(root, "es_419-to-en.wav", "es", asr, f"{TEST_LORA}/test-adapter.gguf", translate=True)
            if r and base and "error" not in r and "error" not in base:
                same = sum(a == b for a, b in zip(r["translations"], base["translations"]))
                total = len(base["translations"])
                good = same == total if strength == 0 else same < total
                show(f"GGUF + LoRA adapter, strength {strength:g}: {same} of {total} as the base", r, good)
            else:
                show(f"GGUF + LoRA adapter, strength {strength:g}", r)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return ok


def main() -> int:
    root = paths.app_root()
    ok = recognizers(root)
    ok &= translators(root)
    print("\nall passed" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
