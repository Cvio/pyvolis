"""The P8 check: revision mode.

    .venv\\Scripts\\python.exe scripts\\p8_check.py [--mt ID ...] [--skip-files]

1. Dialogues: pairs of sentences where the first can only be translated
   right once the second is heard. Each is translated as it would be live
   (carry-forward), then revised; the check says whether the revision made
   the first one right, left it, or changed it for nothing.
2. Files, through the whole pipeline: the dialogues spoken by a voice
   (scripts\\make_dialogue_file.py) and the read-speech fixture, in carry and
   revision mode, with the count of revisions, the scores and the time; then
   the dialogue file with the voice on (a spoken sentence is never revised)
   and with the age limit at zero (nor is one too old).
"""

from __future__ import annotations

import argparse
import queue
import re
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
from pyvolis.translate import context as ctx  # noqa: E402
from pyvolis.translate import prompts  # noqa: E402
from pyvolis.translate.revision import Done, Reviser  # noqa: E402

# (source, target, first sentence, second sentence, the first is right when it matches this)
DIALOGUES = [
    ("es", "en", "Yo manejo.", "Mi carro está afuera.", r"\bdriv"),
    ("es", "en", "Está lista.", "La cena ya está en la mesa.", r"\bit('s| is) ready|dinner"),
    ("es", "en", "¿La viste?", "Es la mejor película del año.", r"see it|watch it|seen it"),
    ("es", "en", "Se cayó.", "El sistema no responde desde las nueve.", r"\bit (crashed|went down|is down|fell|has crashed|dropped)|\bdown\b"),
    ("es", "en", "Es muy rico.", "Mi tío tiene tres casas y un yate.", r"\bhe('s| is) very (rich|wealthy)"),
    ("es", "en", "No lo encuentro.", "Mi hermano no contesta el teléfono.", r"find him"),
    ("es", "en", "Está cerrada.", "La farmacia abre a las nueve.", r"\bit('s| is) closed"),
    ("en", "es", "It's cold.", "The soup has been sitting out for an hour.", r"est[aá] fr[ií]a"),
    ("en", "es", "It's broken.", "The window won't close.", r"est[aá] rota"),
    ("en", "es", "You look tired.", "All three of you should go home.", r"\b(se ven|est[aá]n|parecen|lucen)\b"),
    ("en", "es", "She's here.", "The doctor will see you now.", r"doctora|ella|aqu[ií]"),
    ("en", "es", "They're delicious.", "I baked the cookies this morning.", r"delicios[ao]s|ric[ao]s"),
]


def dialogues(root: Path, mts: list[str]) -> None:
    pyconfig, _ = PyvolisConfig.load(paths.pyvolis_config_file(root))
    py = pyconfig.context
    for mt in mts:
        entry = tr.choose(root, mt)
        translator = tr.load(entry, prompts.load(paths.prompts_dir(root)), pyconfig.translate.device)
        fixed = kept_right = kept_wrong = broken = 0
        print(f"\nDialogues, {entry.id} on the {translator.device.upper()}")
        try:
            for source, target, first, second, right in DIALOGUES:
                history = ctx.History(py.sentences, py.token_budget)
                reviser = Reviser(py.revise_sentences, py.revise_max_age_s, py.revise_max_words)
                shown = []
                for i, text in enumerate((first, second)):
                    result = translator.translate(tr.TranslationRequest(text, source, target,
                                                                        history.context(translator.count_tokens)))
                    history.add(text, result.text)
                    shown.append(result.text)
                    reviser.add(Done(f"{i + 1}.1", text, source, result.text, float(i), False))
                changes = reviser.revise(translator, history.context(translator.count_tokens), target, [])
                before = shown[0]
                after = changes[0].new if changes else before
                was, now = bool(re.search(right, before.lower())), bool(re.search(right, after.lower()))
                outcome = ("fixed" if now and not was else "right already" if now and was
                           else "BROKEN" if was and not now else "still wrong")
                fixed += outcome == "fixed"
                kept_right += outcome == "right already"
                kept_wrong += outcome == "still wrong"
                broken += outcome == "BROKEN"
                arrow = f" -> {after}" if changes else "   (not revised)"
                print(f"  {outcome:13s} {first} | {second}\n                {before}{arrow}")
        finally:
            translator.close()
        print(f"  {fixed} fixed by revision, {kept_right} right already, {kept_wrong} still wrong, {broken} broken, "
              f"of {len(DIALOGUES)}")


def run_file(root: Path, path: Path, source: str, target: str, asr: str, mt: str, mode: str, speak: bool = False,
             max_age: float | None = None) -> dict:
    config = Config.parse(f'[asr]\nengine = "{asr}"\n[languages]\nsource = "{source}"\ntarget = "{target}"\n')
    pyconfig, _ = PyvolisConfig.load(paths.pyvolis_config_file(root))
    if max_age is not None:
        pyconfig.context.revise_max_age_s = max_age
    events: queue.Queue = queue.Queue()
    options = Options(translate=True, speak=speak, mt=mt, context=mode, streaming=False)
    pipeline = Pipeline(root, config, options, events, ArraySource(read_16k_mono(path)), pyconfig=pyconfig)
    collected = run_to_end(pipeline, events)
    if collected.errors:
        raise SystemExit("STOP: " + "; ".join(collected.errors))
    translated = {e.id: e.text for e in collected.of(ev.Translated)}
    spoken = [e.id for e in collected.of(ev.SpeakingStarted)]
    revised = collected.of(ev.Revised)
    for e in revised:
        translated[e.id] = e.new
    sentences = collected.of(ev.SentenceMsg)
    stats = collected.of(ev.Summary)[-1].stats
    return {"sentences": sentences, "translated": translated, "revised": revised, "spoken": spoken, "stats": stats}


def files(root: Path, asr: str, mt: str) -> None:
    folder = root / "tests" / "fixtures" / "files"
    import json

    print(f"\nFiles through the pipeline, {asr} + {mt or 'the default translator'}")
    for name, source, target in (("dialogue-es.wav", "es", "en"), ("es_419-to-en.wav", "es", "en")):
        path = folder / name
        if not path.is_file():
            print(f"  (no {path}; run scripts\\make_dialogue_file.py and tests\\fetch-fixtures.ps1)")
            continue
        ref = path.with_name(path.name + ".ref.json")
        reference = json.loads(ref.read_text(encoding="utf-8")) if ref.is_file() else {}
        for mode in ("carry", "revision"):
            out = run_file(root, path, source, target, asr, mt, mode)
            text = " ".join(out["translated"].get(s.id, "") for s in out["sentences"])
            chrf = f"chrF {scoring.chrf(reference['translation'], text, target):.1f}" if reference.get("translation") else ""
            s = out["stats"]
            print(f"  {name}, {mode:8s}: {len(out['sentences'])} sentences, {len(out['revised'])} revisions "
                  f"in {s['revision_passes']} passes (median {s['revise_ms_median']} ms), translate median "
                  f"{s['translate_ms_median']} ms  {chrf}")
            for e in out["revised"]:
                print(f"      {e.id}: {e.old}\n        -> {e.new}")
    path = folder / "dialogue-es.wav"
    if path.is_file():
        out = run_file(root, path, "es", "en", asr, mt, "revision", speak=True)
        both = sorted(set(e.id for e in out["revised"]) & set(out["spoken"]))
        print(f"  with the voice on: {len(out['spoken'])} sentences spoken, {len(out['revised'])} revisions, "
              f"{len(both)} of a spoken sentence  {'ok' if not out['revised'] else 'FAIL'}")
        out = run_file(root, path, "es", "en", asr, mt, "revision", max_age=0.0)
        print(f"  with the age limit at 0 s: {len(out['revised'])} revisions  {'ok' if not out['revised'] else 'FAIL'}")


def main() -> int:
    root = paths.app_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mt", nargs="*", default=[""])
    parser.add_argument("--asr", default="whisper-large-v3-turbo-es")
    parser.add_argument("--skip-files", action="store_true")
    parser.add_argument("--skip-dialogues", action="store_true")
    args = parser.parse_args()
    if not args.skip_dialogues:
        dialogues(root, args.mt)
    if not args.skip_files:
        for mt in args.mt:
            files(root, args.asr, mt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
