"""Compare translators on text, without audio.

    .venv\\Scripts\\python.exe scripts\\mt_bench.py [--mt ID ...]     (default: every usable translator)

For each translator, with the default prompt and carry-forward context:

  - quality: chrF against the FLORES+ references for the fixture sentences
    (10 each: Spanish, Egyptian Arabic and Persian into English; English into
    Spanish), translating the reference transcripts so recognition plays no part;
  - obedience: scripts\\obey_check.py's sentences, alone and with context;
  - dialogues: scripts\\p8_check.py's pairs: the first sentence right at once,
    right after revision, or made worse;
  - time per sentence and where it ran.

Ten sentences a language is a small sample: differences of a point or two
of chrF mean nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pyvolis import paths  # noqa: E402

paths.apply_offline_environment(paths.app_root())
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

import logging  # noqa: E402

logging.disable(logging.WARNING)

from obey_check import CASES, EARLIER, verdict  # noqa: E402
from p8_check import DIALOGUES  # noqa: E402

from pyvolis import models, scoring  # noqa: E402
from pyvolis import translate as tr  # noqa: E402
from pyvolis.config import PyvolisConfig  # noqa: E402
from pyvolis.translate import context as ctx  # noqa: E402
from pyvolis.translate import prompts  # noqa: E402
from pyvolis.translate.revision import Done, Reviser  # noqa: E402

SETS = [("es_419", "es", "en"), ("ar_eg", "ar", "en"), ("fa_ir", "fa", "en"), ("en_us", "en", "es")]


def translate(translator, text, source, target, context=()):
    try:
        result = translator.translate(tr.TranslationRequest(text, source, target, list(context)))
        return result.text, result.seconds
    except tr.Refused as e:
        return f"(refused: {e.guard})", 0.0


def quality(translator, py, times: list[float]) -> dict[str, float]:
    scores = {}
    for name, source, target in SETS:
        path = REPO / "tests" / "fixtures" / "fleurs" / name / "refs.jsonl"
        if not path.is_file():
            continue
        refs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        refs = [r for r in refs if r.get("references", {}).get(target)]
        history = ctx.History(py.sentences, py.token_budget)
        outputs = []
        for ref in refs:
            text, seconds = translate(translator, ref["raw_transcript"], source, target,
                                      history.context(translator.count_tokens))
            if seconds:
                times.append(seconds)
                history.add(ref["raw_transcript"], text)
            outputs.append("" if text.startswith("(refused") else text)
        scores[f"{source}>{target}"] = scoring.chrf(" ".join(r["references"][target] for r in refs),
                                                    " ".join(outputs), target)
    return scores


def obedience(translator) -> tuple[int, int]:
    counts = []
    for with_context in (False, True):
        passed = 0
        for source, target, text, translated, answered in CASES:
            output, _ = translate(translator, text, source, target, EARLIER[(source, target)] if with_context else ())
            passed += verdict(output, translated, answered)
        counts.append(passed)
    return counts[0], counts[1]


def dialogue_results(translator, py) -> tuple[int, int, int]:
    """(right at once, right after revision, made worse by revision)."""
    at_once = after = worse = 0
    for source, target, first, second, right in DIALOGUES:
        history = ctx.History(py.sentences, py.token_budget)
        reviser = Reviser(py.revise_sentences, py.revise_max_age_s, py.revise_max_words)
        shown = []
        for i, text in enumerate((first, second)):
            output, _ = translate(translator, text, source, target, history.context(translator.count_tokens))
            history.add(text, output)
            shown.append(output)
            reviser.add(Done(f"{i + 1}.1", text, source, output, float(i), False))
        changes = reviser.revise(translator, history.context(translator.count_tokens), target, [])
        was = bool(re.search(right, shown[0].lower()))
        now = bool(re.search(right, (changes[0].new if changes else shown[0]).lower()))
        at_once += was
        after += now
        worse += was and not now
    return at_once, after, worse


def main() -> int:
    root = paths.app_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mt", nargs="*", default=[])
    args = parser.parse_args()
    pyconfig, _ = PyvolisConfig.load(paths.pyvolis_config_file(root))
    found = [t for t in models.discover_translators(paths.mt_dir(root)) if isinstance(t, models.Translator)]
    entries = [tr.choose(root, m) for m in args.mt] or [t for t in found if t.enabled()]
    prompt = prompts.load(paths.prompts_dir(root))
    rows = []
    for entry in entries:
        print(f"... {entry.id}", flush=True)
        try:
            translator = tr.load(entry, prompt, pyconfig.translate.device)
        except tr.TranslateError as e:
            print(f"    cannot load: {e}")
            continue
        try:
            times: list[float] = []
            scores = quality(translator, pyconfig.context, times)
            alone, with_context = obedience(translator)
            at_once, after, worse = dialogue_results(translator, pyconfig.context)
            rows.append((entry, translator.device, scores, alone, with_context, at_once, after, worse,
                         statistics.median(times) * 1000 if times else 0))
        finally:
            translator.close()
    pairs = [f"{s}>{t}" for _, s, t in SETS]
    print(f"\n{'translator':44s} {'GB':>4s} {'dev':4s} " + " ".join(f"{p:>9s}" for p in pairs)
          + f" {'mean':>5s}  {'obey':>7s} {'obey+ctx':>8s}  {'dialogues: at once / revised / worse':s}  ms/sentence")
    for entry, device, scores, alone, with_context, at_once, after, worse, ms in rows:
        mean = statistics.mean(scores.values()) if scores else 0
        print(f"{entry.id[:44]:44s} {entry.size_bytes / 1e9:4.1f} {device:4s} "
              + " ".join(f"{scores.get(p, 0):9.1f}" for p in pairs)
              + f" {mean:5.1f}  {alone:4d}/{len(CASES)} {with_context:5d}/{len(CASES)}  "
              + f"{at_once:2d} / {after:2d} / {worse:d} of {len(DIALOGUES)}".ljust(37) + f"{ms:6.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
