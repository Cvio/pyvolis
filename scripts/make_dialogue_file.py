"""Make a recording of the revision check's Spanish dialogues, spoken by an
installed voice, for file mode.

    .venv\\Scripts\\python.exe scripts\\make_dialogue_file.py

Writes tests\\fixtures\\files\\dialogue-es.wav: each pair of sentences from
scripts\\p8_check.py, the two 0.9 s apart (two utterances) and 2.5 s before the
next pair, with the transcript in dialogue-es.wav.ref.json. A synthetic voice,
so it says nothing about recognition; it is there so revision can be checked
through the whole pipeline, on sentences that depend on what follows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pyvolis import paths  # noqa: E402

paths.apply_offline_environment(paths.app_root())

from p8_check import DIALOGUES  # noqa: E402

from pyvolis import models, tts  # noqa: E402
from pyvolis.pipeline import write_wav  # noqa: E402


def main() -> int:
    import soxr

    root = paths.app_root()
    voices = [e for e in models.discover(paths.tts_dir(root), models.Role.TTS) if isinstance(e, models.Engine)]
    voice = tts.Voice(tts.for_language(voices, "es"))
    pieces, said = [np.zeros(8_000, np.float32)], []
    for source, _target, first, second, _right in DIALOGUES:
        if source != "es":
            continue
        for text, gap in ((first, 0.9), (second, 2.5)):
            speech = voice.speak(text)
            pieces.append(soxr.resample(speech.samples, speech.sample_rate, 16_000).astype(np.float32))
            pieces.append(np.zeros(int(gap * 16_000), np.float32))
            said.append(text)
    out = root / "tests" / "fixtures" / "files" / "dialogue-es.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    audio = np.concatenate(pieces)
    write_wav(out, audio)
    out.with_name(out.name + ".ref.json").write_text(
        json.dumps({"transcript": " ".join(said)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out}: {len(audio) / 16_000:.0f} s, {len(said)} sentences, voice {voice.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
