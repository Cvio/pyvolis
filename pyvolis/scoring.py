"""Quick scoring of a file run against a reference: CER of the transcript and
chrF of the translation, with model-bench's text cleaning (textclean.py,
copied from it) and its chrF (sacrebleu's CHRF, default settings).

A quick check, not the proof: model-bench is the proof. Scores are over the
whole file, the transcript and the translation each joined into one text, so
how the recognizer cut sentences doesn't change them.

A reference sits next to the audio file:
  <file>.ref.json   {"transcript": "...", "translation": "..."}, either may be absent
  <file>.ref.srt    the transcript, as subtitles (timings ignored)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .textclean import clean


class ReferenceError(Exception):
    pass


@dataclass
class Reference:
    transcript: str | None = None
    translation: str | None = None
    source: Path | None = None  # the file the reference came from


def find_reference(audio: Path) -> Reference | None:
    """The reference next to `audio`, or None. A malformed one is an error naming it."""
    json_path = audio.with_name(audio.name + ".ref.json")
    srt_path = audio.with_name(audio.name + ".ref.srt")
    # Also accept talk.ref.json beside talk.m4a.
    for candidate in (json_path, audio.with_suffix(".ref.json")):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise ReferenceError(f"cannot read {candidate.absolute()}: {e}") from e
            if not isinstance(data, dict):
                raise ReferenceError(f"{candidate.absolute()} must hold an object with transcript/translation")
            return Reference(data.get("transcript"), data.get("translation"), candidate)
    for candidate in (srt_path, audio.with_suffix(".ref.srt")):
        if candidate.is_file():
            return Reference(srt_text(candidate.read_text(encoding="utf-8-sig")), None, candidate)
    return None


def srt_text(srt: str) -> str:
    """The words of an SRT: numbers and timing lines dropped, cues joined."""
    lines = []
    for line in srt.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line or line.isdigit() or re.match(r"^\d\d:\d\d:\d\d[,.]\d+ -->", line):
            continue
        lines.append(line)
    return " ".join(lines)


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str, language: str) -> float:
    """Character error rate in percent, spaces ignored, after cleaning (as
    model-bench's cer_counts)."""
    r = clean(reference, language).replace(" ", "")
    h = clean(hypothesis, language).replace(" ", "")
    return 100.0 * edit_distance(r, h) / max(len(r), 1)


def chrf(reference: str, hypothesis: str, language: str) -> float:
    from sacrebleu.metrics.chrf import CHRF

    return CHRF().corpus_score([clean(hypothesis, language)], [[clean(reference, language)]]).score
