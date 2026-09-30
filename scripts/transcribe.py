"""Transcribe the FLEURS fixtures with every recognizer that covers their
language, through the same guards the pipeline uses, and print per model:
character error rate against the reference, how often the output has no
punctuation at all (whether it can mark questions), and the real-time factor.

    .venv\\Scripts\\python.exe scripts\\transcribe.py [es_419 ar_eg fa_ir en_us] [--model NAME] [--show]

GPU models are loaded one at a time and released after. The CER here is a
quick look: case, punctuation and spacing ignored. model-bench is the proof.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pyvolis import paths  # noqa: E402

paths.apply_offline_environment(paths.app_root())
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

from pyvolis import asr, models  # noqa: E402
from pyvolis.asr.guards import Guards, SpeechScorer, load_phrases, normalise  # noqa: E402
from pyvolis.filesource import read_16k_mono  # noqa: E402

LANG = {"es_419": "es", "ar_eg": "ar", "fa_ir": "fa", "en_us": "en"}
PUNCTUATION = set(".,;:!?¿¡،؛؟…\"'«»()-")


def cer(ref: str, hyp: str) -> float:
    r, h = normalise(ref).replace(" ", ""), normalise(hyp).replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sets", nargs="*", default=["es_419", "ar_eg"])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--show", action="store_true", help="print every transcript")
    args = parser.parse_args()
    root = paths.app_root()
    guards = Guards(load_phrases(paths.hallucinations_file(root)),
                    scorer=SpeechScorer(paths.vad_model_file(root), 0.8))
    engines = [e for e in models.discover(paths.asr_dir(root), models.Role.ASR)
               if isinstance(e, models.Engine) and e.enabled()]
    if args.model:
        engines = [e for e in engines if e.dir_name in args.model]
    rows = []
    for name in args.sets:
        language = LANG[name]
        folder = root / "tests" / "fixtures" / "fleurs" / name
        refs = [json.loads(x) for x in (folder / "refs.jsonl").read_text(encoding="utf-8").splitlines()]
        clips = [(read_16k_mono(folder / r["file"]), r["raw_transcript"]) for r in refs]
        for engine in engines:
            if engine.languages_known and language not in engine.languages:
                continue
            try:
                recognizer = asr.load(engine)
            except asr.AsrError as e:
                print(f"{name} {engine.dir_name}: cannot load: {e}")
                continue
            errors, bare, spent, heard, dropped = [], 0, 0.0, 0.0, 0
            for audio, ref in clips:
                began = time.perf_counter()
                result = recognizer.transcribe(audio, language)
                spent += time.perf_counter() - began
                heard += len(audio) / 16000
                verdict = guards.check(result.text, language, audio)
                dropped += verdict.dropped
                errors.append(cer(ref, verdict.text))
                bare += not any(c in PUNCTUATION for c in verdict.text)
                if args.show:
                    print(f"  [{engine.dir_name}] {verdict.text}")
            recognizer.close()
            rows.append((name, engine.dir_name, engine.backend, sum(errors) / len(errors),
                         bare / len(clips), spent / heard, dropped))
            print(f"{name:7s} {engine.dir_name:44s} CER {rows[-1][3]:6.1%}  no punctuation "
                  f"{rows[-1][4]:5.0%}  RTF {rows[-1][5]:.2f}  guard drops {dropped}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
