"""Translation parity: for Qwen3 with the default prompt, pyvolis translates
each sentence as Rust volis does.

    .venv\\Scripts\\python.exe parity\\translate.py [es_419 en_us]

Rust volis translates only what it hears, so its parity copy runs --listen
on the VB-Audio cable while fixture clips play into it (as parity\\asr.py),
and logs every transcript and translation. pyvolis then translates exactly
the text Rust translated, with the same GGUF, and the outputs are compared,
refusals included. Exact audio isn't needed here, so the cable may be at any
rate.

The prompt is byte-identical (tests/test_translate.py). llama-cpp-python
bundles a different llama.cpp from Rust's llama-cpp-2, so a few differing
lines are allowed: every one is printed for the user to judge.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "parity"))

from asr import CABLE_IN, LANGUAGE, Rust, play  # noqa: E402

from pyvolis import audio, paths  # noqa: E402
from pyvolis import translate as tr  # noqa: E402
from pyvolis.filesource import read_16k_mono  # noqa: E402
from pyvolis.translate import prompts  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str]) -> int:
    sets = argv or ["es_419", "en_us"]
    devices = {d.name: d for d in audio.list_output_devices()}
    if CABLE_IN not in devices:
        raise SystemExit(f"STOP: no output device named {CABLE_IN!r}; install the VB-Audio Virtual Cable")
    rate = int(devices[CABLE_IN].default_config.split(", ")[1].split()[0])
    entry = tr.choose(REPO, "qwen3-1.7b-q4_k_m.gguf")
    translator = tr.load(entry, prompts.load(paths.prompts_dir(REPO)))
    rows = []
    for name in sets:
        source, target = LANGUAGE[name]
        folder = REPO / "tests" / "fixtures" / "fleurs" / name
        refs = [json.loads(x) for x in (folder / "refs.jsonl").read_text(encoding="utf-8").splitlines()]
        print(f"\n== {name}: Rust listening as \"{source}\", translating into \"{target}\"")
        rust = Rust(source, target, compare=False)
        try:
            if not rust.listening.wait(180):
                raise SystemExit("STOP: Rust volis never started listening:\n" + "\n".join(rust.lines[-30:]))
            for i, ref in enumerate(refs):
                play(read_16k_mono(folder / ref["file"]), devices[CABLE_IN].index, rate)
                # Until every utterance heard so far is translated or refused.
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    time.sleep(1.0)
                    heard = set(rust.transcripts)
                    if heard and heard <= set(rust.translations) | set(rust.refusals):
                        break
                    if not heard and rust.translations:
                        break
                print(f"  clip {i + 1}: {len(rust.translations)} translated, {len(rust.refusals)} refused")
        finally:
            rust.stop()
        # Every utterance Rust translated, and every one it refused.
        for index in sorted(set(rust.translations) | set(rust.refusals)):
            text, theirs = rust.translations.get(index, (rust.transcripts.get(index, ""), None))
            if not text:
                continue
            request = tr.TranslationRequest(text, source, target)
            try:
                result = translator.translate(request)
                ours, seconds, device = result.text, result.seconds, result.device
            except tr.Refused as e:
                ours, seconds, device = None, 0.0, translator.device
                if theirs is not None:
                    ours = f"(refused: {e.guard})"
            rows.append((name, index, text, theirs, ours, seconds, device))
    translator.close()

    same = sum(1 for r in rows if r[3] == r[4])
    print(f"\n{same} of {len(rows)} translations identical (Rust on the CPU; pyvolis on the "
          f"{rows[0][6].upper() if rows else '?'})")
    for name, index, text, theirs, ours, seconds, _ in rows:
        mark = "same" if theirs == ours else "DIFF"
        shown = "(refused)" if theirs is None else theirs
        line = f"  {mark} {name} {index:>2}  {text}\n        Rust:    {shown}"
        if theirs != ours:
            line += f"\n        pyvolis: {'(refused)' if ours is None else ours}"
        print(line + f"   [{seconds * 1000:.0f} ms]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
