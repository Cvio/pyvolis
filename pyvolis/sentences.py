"""Splitting a transcript into sentences.

New in pyvolis. Rust translates each utterance the VAD cut, which can be half
a sentence or three. pyvolis translates sentences: everything downstream
(context, fragment holding, revision, subtitles, the timeline) is per
sentence, and every sentence has a stable id, "<utterance>.<sentence>".

A sentence ends at final punctuation (. ? ! … and the Arabic ؟) followed by a
space or the end. Text with no final punctuation at all, which is what MMS
and some Whisper models produce, stays one sentence per utterance: Rust's
behaviour. Times come from word timings when the recognizer gives them, and
are otherwise shared out over the utterance by length, marked approximate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .asr import Word

# A sentence ends at one or more of these, then closing quotes or brackets.
_END = re.compile(r"[.?!…؟]+[\"'»”’)\]]*(?=\s|$)")
# A full stop after these is not the end of a sentence.
ABBREVIATIONS = {
    "sr", "sra", "srta", "dr", "dra", "lic", "ing", "prof", "etc", "ee", "uu", "ud", "uds", "av", "núm",
    "mr", "mrs", "ms", "st", "vs", "no", "approx", "e.g", "i.e",
}


@dataclass
class Sentence:
    id: str  # "<utterance>.<n>", n from 1
    utterance: int
    text: str
    start: float  # seconds on the source's timeline (the file's, for a file)
    end: float
    approximate: bool  # times shared out by length, not from word timings


def split_text(text: str) -> list[str]:
    """The sentences of `text`, with their punctuation, in order."""
    text = " ".join(text.split())
    parts, start = [], 0
    for match in _END.finditer(text):
        before = text[start : match.start()].split()
        last = before[-1].lower().rstrip(".") if before else ""
        if match.group().startswith(".") and (last in ABBREVIATIONS or (len(last) == 1 and last.isalpha())):
            continue  # "Sr. Pérez", "EE. UU.", an initial
        parts.append(text[start : match.end()].strip())
        start = match.end()
    rest = text[start:].strip()
    if rest:
        parts.append(rest)
    return [p for p in parts if p]


def split(
    text: str, utterance: int, start: float, end: float, words: list[Word] | None = None
) -> list[Sentence]:
    """Sentences with ids and times. `start` and `end` are the utterance's
    times on the source timeline; word times are relative to its start."""
    pieces = split_text(text)
    if not pieces:
        return []
    timed = _times_from_words(pieces, words, start) if words else None
    approximate = timed is None
    if timed is None:
        total = sum(len(p) for p in pieces) or 1
        timed, at = [], start
        for p in pieces:
            length = (end - start) * len(p) / total
            timed.append((at, at + length))
            at += length
    return [
        Sentence(f"{utterance}.{n}", utterance, p, round(a, 3), round(b, 3), approximate)
        for n, (p, (a, b)) in enumerate(zip(pieces, timed), 1)
    ]


def _times_from_words(pieces: list[str], words: list[Word], offset: float):
    """Each sentence's span from the words it contains, matched by word count.
    None if the counts don't line up (the recognizer's words and its text
    disagree), in which case the caller falls back to shares by length."""
    counts = [len(p.split()) for p in pieces]
    if sum(counts) != len(words):
        return None
    out, i = [], 0
    for count in counts:
        span = words[i : i + count]
        out.append((offset + span[0].start, offset + span[-1].end))
        i += count
    return out
