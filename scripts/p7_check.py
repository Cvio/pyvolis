"""The P7 check: each feature off, then on, on the same recording.

    .venv\\Scripts\\python.exe scripts\\p7_check.py [--file F --from es --to en --asr FOLDER --mt ID] [--realtime]

Runs a fixture recording (or your own, with its .ref.json beside it) through
the pipeline once per configuration and prints, side by side: transcript CER
and WER, translation chrF, the recognizer's real-time factor (all passes
counted), the translation time per sentence, and the number of sentences.
With --realtime it adds a paced streaming run, which is the one that can say
how soon text appears after speech starts.

Then the context cases: sentences whose translation depends on the one
before, translated alone and with carry-forward context.
"""

from __future__ import annotations

import argparse
import queue
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
from pyvolis import translate as tr  # noqa: E402
from pyvolis.config import Config, PyvolisConfig  # noqa: E402
from pyvolis.filesource import read_16k_mono  # noqa: E402
from pyvolis.pipeline import ArraySource, Options, Pipeline, run_to_end  # noqa: E402
from pyvolis.translate import prompts  # noqa: E402

CONFIGURATIONS = [
    ("segment, no context, no holding", dict(streaming=False, context="off", hold=False)),
    ("+ carry-forward context", dict(streaming=False, context="carry", hold=False)),
    ("+ fragment holding", dict(streaming=False, context="carry", hold=True)),
    ("+ streaming (every pass)", dict(streaming=True, context="carry", hold=True)),
]

# Pairs where the second sentence can't be translated right without the first.
CONTEXT_CASES = [
    ("es", "en", ["¿Dónde está María?", "No la he visto desde ayer."]),
    ("es", "en", ["Mi carro está en el taller.", "Me lo entregan el martes."]),
    ("es", "en", ["Yo manejo.", "Mi carro está afuera."]),
    ("en", "es", ["The bank closed early today.", "It reopens on Monday."]),
]


def run(root, audio, reference, source, target, asr, mt, name, options, realtime=False) -> dict:
    config = Config.parse(f'[asr]\nengine = "{asr}"\n[languages]\nsource = "{source}"\ntarget = "{target}"\n')
    events: queue.Queue = queue.Queue()
    ev.restart_clock()
    pipeline = Pipeline(root, config, Options(mt=mt, **options), events, ArraySource(audio, realtime=realtime),
                        PyvolisConfig())
    out = run_to_end(pipeline, events)
    stats = out.of(ev.Summary)[-1].stats
    sentences = out.of(ev.SentenceMsg)
    translations = {e.id: e.text for e in out.of(ev.Translated)}
    row = {
        "name": name,
        "cer": scoring.cer(reference.transcript, " ".join(s.text for s in sentences), source),
        "wer": scoring.wer(reference.transcript, " ".join(s.text for s in sentences), source),
        "chrf": scoring.chrf(reference.translation, " ".join(translations.get(s.id, "") for s in sentences), target),
        "rtf": stats["asr_rtf"], "passes": stats["asr_passes"], "mt": stats["translate_ms_median"],
        "sentences": stats["sentences"], "held": stats["fragments_held"], "refused": stats["not_translated"],
        "first": stats["first_text_ms_median"], "partials": len(out.of(ev.Partial)), "errors": out.errors,
    }
    return row


def show(row: dict) -> None:
    first = f"{row['first']} ms" if row["first"] is not None else "-"
    print(f"  {row['name']:<34} CER {row['cer']:4.1f}%  WER {row['wer']:4.1f}%  chrF {row['chrf']:4.1f}  "
          f"ASR RTF {row['rtf']:.2f} ({row['passes']} passes)  translate {row['mt']} ms  "
          f"{row['sentences']} sentences, {row['held']} held, {row['refused']} refused  first text {first}", flush=True)
    for error in row["errors"]:
        print(f"      error: {error}")


def context_cases(root, mt: str) -> None:
    translator = tr.load(tr.choose(root, mt), prompts.load(paths.prompts_dir(root)))
    print(f"\nContext cases ({translator.name}): the second sentence alone, then with the first as context")
    for source, target, (first, second) in CONTEXT_CASES:
        earlier = translator.translate(tr.TranslationRequest(first, source, target)).text
        alone = translator.translate(tr.TranslationRequest(second, source, target)).text
        with_context = translator.translate(
            tr.TranslationRequest(second, source, target, [tr.Turn(first, earlier)])).text
        print(f"  {first}  ->  {earlier}")
        print(f"    {second}\n      alone:        {alone}\n      with context: {with_context}"
              + ("" if alone != with_context else "   (no change)"))
    translator.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=REPO / "tests/fixtures/files/es_419-to-en.wav")
    parser.add_argument("--from", dest="source", default="es")
    parser.add_argument("--to", dest="target", default="en")
    parser.add_argument("--asr", default="whisper-large-v3-turbo-es")
    parser.add_argument("--mt", default="qwen3-1.7b-q4_k_m.gguf")
    parser.add_argument("--realtime", action="store_true", help="also a paced streaming run, for the latency")
    parser.add_argument("--skip-cases", action="store_true")
    args = parser.parse_args()
    root = paths.app_root()
    reference = scoring.find_reference(args.file)
    if reference is None or not reference.transcript or not reference.translation:
        raise SystemExit(f"STOP: {args.file} needs a .ref.json with transcript and translation beside it "
                         "(scripts\\make_fixture_file.py builds one)")
    audio = read_16k_mono(args.file)
    print(f"{args.file.name}: {len(audio) / 16000:.0f} s, {args.source} -> {args.target}, {args.asr} + {args.mt}")
    for name, options in CONFIGURATIONS:
        show(run(root, audio, reference, args.source, args.target, args.asr, args.mt, name, options))
    if args.realtime:
        show(run(root, audio, reference, args.source, args.target, args.asr, args.mt,
                 "streaming, real time", dict(streaming=True, context="carry", hold=True), realtime=True))
    if not args.skip_cases:
        context_cases(root, args.mt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
