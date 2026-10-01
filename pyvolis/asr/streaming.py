"""Streaming recognition from a segment recognizer: LocalAgreement.

New in pyvolis. A whole-utterance model (Whisper, Parakeet, Cohere) becomes a
streaming one by transcribing the growing audio again every short interval
and committing only the words two consecutive transcriptions agree on. What
is committed never changes; the rest is provisional and may.

    agreement = LocalAgreement(recognizer, "es")
    update = agreement.update(audio_so_far)   # every second while speech goes on
    update = agreement.finish(whole_audio)    # when the detector ends the utterance

Each call transcribes the audio from the trim point onward. When the
recognizer gives word timings, the buffer is trimmed at the end of a
committed sentence once it is long, so it never grows towards Whisper's 30
seconds and each pass stays cheap. Without timings nothing is trimmed (the
detector cuts an utterance at 20 s anyway).

The cost is real: every pass is a full transcription of the buffer. The
pipeline reports the recognizer's real-time factor, and above 1.0 streaming
can't keep up.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

import numpy as np

from ..audio import SAMPLE_RATE
from . import Word

# Trim the buffer once it is this long and a committed sentence end allows it.
TRIM_ABOVE_SECONDS = 10.0
SENTENCE_END = (".", "?", "!", "…", "؟")


@dataclass
class AsrUpdate:
    """What one pass changed."""

    committed: list[Word] = field(default_factory=list)  # newly committed words, in order
    provisional: str = ""  # the current best guess for what follows
    seconds: float = 0.0  # what the pass cost
    timed: bool = False  # the committed words carry real times (seconds from the utterance's start)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.committed)


def _key(word: str) -> str:
    """How two transcriptions' words are compared: case and punctuation aside."""
    word = unicodedata.normalize("NFKC", word).casefold()
    return "".join(c for c in word if c.isalnum())


def agreed_prefix(a: list[str], b: list[str]) -> int:
    """How many leading words two hypotheses share."""
    count = 0
    for x, y in zip(a, b):
        if _key(x) != _key(y):
            break
        count += 1
    return count


class LocalAgreement:
    def __init__(self, recognizer, language: str, words_at_the_end: bool = False) -> None:
        self.recognizer = recognizer
        self.language = language
        # Word timings on the last pass (for a file's subtitles); a pass in
        # between asks for them only when the buffer is long enough to trim.
        self.words_at_the_end = words_at_the_end
        self.offset = 0  # samples trimmed from the front of the utterance
        self.done = 0  # words at the head of the current buffer already committed
        self.previous: list[str] = []  # the last hypothesis for the current buffer
        self.passes = 0
        self.seconds = 0.0

    def _transcribe(self, audio: np.ndarray, timestamps: bool):
        result = self.recognizer.transcribe(audio[self.offset :], self.language, timestamps)
        self.passes += 1
        self.seconds += result.seconds
        base = self.offset / SAMPLE_RATE
        if result.words and len(result.words) == len(result.text.split()):
            words = [Word(w.text, base + float(w.start), base + float(w.end)) for w in result.words]
            return words, True, result.seconds
        # No usable timings: the words of the text, untimed.
        return [Word(t, 0.0, 0.0) for t in result.text.split()], False, result.seconds

    def update(self, audio: np.ndarray) -> AsrUpdate:
        """One pass over the utterance so far. Commits what this transcription
        and the last one agree on, beyond what is already committed."""
        long = (len(audio) - self.offset) / SAMPLE_RATE >= TRIM_ABOVE_SECONDS
        words, timed, seconds = self._transcribe(audio, timestamps=long)
        texts = [w.text for w in words]
        agreed = agreed_prefix(texts, self.previous)
        upto = max(self.done, min(agreed, len(words)))
        newly = words[self.done : upto]
        self.done = upto
        self.previous = texts
        update = AsrUpdate(newly, " ".join(texts[upto:]), seconds, timed)
        if timed:
            self._trim(words, len(audio))
        return update

    def finish(self, audio: np.ndarray) -> AsrUpdate:
        """The detector ended the utterance: one last pass, and everything
        beyond what is committed is committed."""
        words, timed, seconds = self._transcribe(audio, timestamps=self.words_at_the_end)
        newly = words[self.done :]
        self.done, self.previous = len(words), [w.text for w in words]
        return AsrUpdate(newly, "", seconds, timed)

    def _trim(self, words: list[Word], length: int) -> None:
        """Cut the buffer at the end of the last committed sentence, once the
        buffer is long. The words before the cut leave the accounting."""
        if (length - self.offset) / SAMPLE_RATE < TRIM_ABOVE_SECONDS:
            return
        last = max((i for i in range(self.done) if words[i].text.rstrip("\"'»”’)]").endswith(SENTENCE_END)),
                   default=None)
        if last is None:
            return
        cut = int(words[last].end * SAMPLE_RATE)
        if cut <= self.offset or cut >= length:
            return
        self.offset = cut
        self.done -= last + 1
        self.previous = self.previous[last + 1 :]
