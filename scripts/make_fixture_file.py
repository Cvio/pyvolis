"""Join one language's FLEURS fixture clips into a single recording, with a
reference beside it, for file mode (`--file`).

    .venv\\Scripts\\python.exe scripts\\make_fixture_file.py es_419 en [--clips 10]

Writes tests\\fixtures\\files\\<lang>-to-<target>.wav (clips 1.2 s apart, as a
conversation) and <name>.wav.ref.json with the reference transcript (FLEURS)
and translation (FLORES+, from tests\\fetch-fixtures.ps1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pyvolis.filesource import read_16k_mono  # noqa: E402
from pyvolis.pipeline import write_wav  # noqa: E402

GAP_SECONDS = 1.2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lang", help="fixture folder, e.g. es_419")
    parser.add_argument("target", help="reference language: en, es, ar, ar-IQ, fa")
    parser.add_argument("--clips", type=int, default=10)
    parser.add_argument("--out", type=Path, help="the .wav to write (default: tests/fixtures/files/...)")
    args = parser.parse_args()
    folder = REPO / "tests" / "fixtures" / "fleurs" / args.lang
    refs = [json.loads(x) for x in (folder / "refs.jsonl").read_text(encoding="utf-8").splitlines()][: args.clips]
    if any("references" not in r for r in refs):
        raise SystemExit("STOP: the fixtures have no FLORES+ references; run .\\tests\\fetch-fixtures.ps1")
    gap = np.zeros(int(GAP_SECONDS * 16_000), np.float32)
    audio = np.concatenate([part for r in refs for part in (read_16k_mono(folder / r["file"]), gap)])
    out = args.out or REPO / "tests" / "fixtures" / "files" / f"{args.lang}-to-{args.target}.wav"
    write_wav(out, audio)
    reference = {
        "transcript": " ".join(r["raw_transcript"] for r in refs),
        "translation": " ".join(r["references"][args.target] for r in refs),
    }
    out.with_name(out.name + ".ref.json").write_text(json.dumps(reference, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out} ({len(audio) / 16000:.1f} s) and its .ref.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
