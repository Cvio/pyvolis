"""Hallucination guards: text a recognizer produced from silence or noise.

New in pyvolis. Four guards, each switchable in pyvolis.toml `[guards]`:

  vad_probability  drop the text of a segment whose Silero speech probability
                   never reaches `min_peak_probability` (default 0.8). The
                   segment passed the detector at `[vad].threshold`, but only
                   just: typically breath, a door, a keyboard.
  repeats          the same phrase three or more times in a row is cut to one
                   ("gracias gracias gracias gracias..."), and dropped if
                   nothing else is left.
  stock_phrases    phrases Whisper invents on silence, per language, from
                   config/hallucinations.toml (user-editable).
  sparse           far too few words for the speech detected: under
                   `min_words_per_second` (default one word per 3 s) over
                   `sparse_min_seconds` (3 s) or more. Noise shaped like
                   speech gets past the detector, and Whisper answers 6 s of
                   it with a lone "y". Only detected speech is counted, never
                   the pauses inside a turn, so one slow word isn't dropped.

Every drop and every trim is reported with its reason, and logged by the
caller, so a wrong one can be spotted. Punctuation is never touched here:
it goes to the translator exactly as the recognizer wrote it.
"""

from __future__ import annotations

import re
import tomllib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import varieties

REPEAT_TIMES = 3  # "three or more times in a row"
MAX_PHRASE_WORDS = 10


class GuardError(Exception):
    pass


@dataclass
class Verdict:
    """What the guards made of one transcript."""

    text: str  # what survives; "" when dropped
    dropped: bool = False
    reasons: list[str] = field(default_factory=list)  # every drop or trim, in words

    @property
    def changed(self) -> bool:
        return bool(self.reasons)


def normalise(text: str) -> str:
    """Case, punctuation and spacing removed, Arabic diacritics too, for matching."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")  # combining marks
    text = "".join(c if (c.isalnum() or c.isspace()) else " " for c in text)
    return " ".join(text.split())


@dataclass
class Phrases:
    whole: set[str] = field(default_factory=set)
    prefix: list[str] = field(default_factory=list)
    trailing: list[str] = field(default_factory=list)


def load_phrases(path: Path) -> dict[str, Phrases]:
    """config/hallucinations.toml, by language. A missing file means no stock
    phrases; a malformed one is an error naming it."""
    if not path.is_file():
        return {}
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise GuardError(f"cannot parse {path.absolute()}: {e}") from e
    out = {}
    for language, section in raw.items():
        if not isinstance(section, dict):
            raise GuardError(f"{path.absolute()}: [{language}] must be a table")
        unknown = set(section) - {"whole", "prefix", "trailing"}
        if unknown:
            raise GuardError(f"{path.absolute()}: [{language}] has unknown key `{sorted(unknown)[0]}`")
        out[language.lower()] = Phrases(
            whole={normalise(p) for p in section.get("whole", [])},
            prefix=[normalise(p) for p in section.get("prefix", [])],
            trailing=[normalise(p) for p in section.get("trailing", [])],
        )
    return out


def collapse_repeats(text: str, times: int = REPEAT_TIMES) -> tuple[str, str | None]:
    """Cut any phrase of up to MAX_PHRASE_WORDS words repeated `times` or more
    times in a row down to one occurrence. Returns the text and a description
    of what was cut, or None."""
    words = text.split()
    keys = [normalise(w) for w in words]
    cut = None
    for n in range(1, MAX_PHRASE_WORDS + 1):
        i = 0
        while i + n * times <= len(words):
            phrase = keys[i : i + n]
            if not any(phrase):
                i += 1
                continue
            count = 1
            while keys[i + count * n : i + (count + 1) * n] == phrase:
                count += 1
            if count >= times:
                shown = " ".join(words[i : i + n])
                cut = f'"{shown}" repeated {count} times, kept once'
                del words[i + n : i + count * n]
                del keys[i + n : i + count * n]
            i += 1
    return " ".join(words), cut


class Guards:
    def __init__(
        self,
        phrases: dict[str, Phrases],
        vad_probability: bool = True,
        repeats: bool = True,
        stock_phrases: bool = True,
        scorer: SpeechScorer | None = None,
        sparse: bool = True,
        min_words_per_second: float = 1 / 3,
        sparse_min_seconds: float = 3.0,
    ) -> None:
        self.use_sparse = sparse
        self.min_words_per_second = min_words_per_second
        self.sparse_min_seconds = sparse_min_seconds
        self.phrases = phrases
        self.use_vad = vad_probability and scorer is not None
        self.use_repeats = repeats
        self.use_phrases = stock_phrases
        self.scorer = scorer

    def check(self, text: str, language: str, audio: np.ndarray | None = None,
              speech_seconds: float | None = None) -> Verdict:
        """`speech_seconds` is how much speech the detector heard (for a turn,
        its segments added up); without it, the audio's own length."""
        verdict = Verdict(text.strip())
        if not verdict.text:
            return verdict
        if speech_seconds is None and audio is not None:
            speech_seconds = len(audio) / 16_000
        if self.use_vad and audio is not None and not self.scorer.has_clear_speech(audio):
            verdict.reasons.append(
                f"speech probability never reached {self.scorer.threshold:.2f} in the segment"
            )
            return _drop(verdict)
        if self.use_repeats:
            collapsed, cut = collapse_repeats(verdict.text)
            if cut:
                verdict.reasons.append(cut)
                verdict.text = collapsed
        if self.use_phrases:
            phrases = self.phrases.get(varieties.language_of(language).lower())
            if phrases:
                key = normalise(verdict.text)
                if key in phrases.whole:
                    verdict.reasons.append("the whole text is a stock phrase recognizers invent on silence")
                    return _drop(verdict)
                for p in phrases.prefix:
                    if p and key.startswith(p):
                        verdict.reasons.append(f'starts with the stock phrase "{p}"')
                        return _drop(verdict)
                for p in phrases.trailing:
                    if p and key != p and key.endswith(" " + p):
                        verdict.text = _cut_trailing(verdict.text, p)
                        verdict.reasons.append(f'ended with the stock phrase "{p}", cut off')
        if not normalise(verdict.text):
            return _drop(verdict)
        if self.use_sparse and speech_seconds is not None and speech_seconds >= self.sparse_min_seconds:
            words = len(verdict.text.split())
            if words < speech_seconds * self.min_words_per_second:
                verdict.reasons.append(
                    f"{words} word(s) for {speech_seconds:.1f} s of detected speech: too few to be what was said"
                )
                return _drop(verdict)
        return verdict


def _drop(verdict: Verdict) -> Verdict:
    verdict.dropped = True
    verdict.text = ""
    return verdict


def _cut_trailing(text: str, phrase: str) -> str:
    """Remove the last words of `text` that normalise to `phrase`, keeping the
    rest (and its punctuation) exactly as written."""
    words = text.split()
    n = len(phrase.split())
    return " ".join(words[:-n]).rstrip(" ,;:") if len(words) > n else text


class SpeechScorer:
    """Scores a finished segment with a second Silero model from the same
    silero_vad.onnx, run by sherpa-onnx at a higher threshold: does any window
    of it clearly contain speech? sherpa-onnx doesn't expose Silero's
    probabilities, but it does apply a threshold to each window, which is all
    this question needs. A few milliseconds per utterance."""

    def __init__(self, model: Path, threshold: float) -> None:
        import sherpa_onnx

        from ..vad import WINDOW

        silero = sherpa_onnx.SileroVadModelConfig()
        silero.model = str(model)
        silero.threshold = threshold
        silero.min_silence_duration = 0.0
        silero.min_speech_duration = 0.0
        silero.window_size = WINDOW
        silero.max_speech_duration = 20.0
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad = silero
        config.sample_rate = 16_000
        config.num_threads = 1
        config.provider = "cpu"
        self._model = sherpa_onnx.VadModel.create(config)
        self.threshold = threshold
        self._window = WINDOW

    def has_clear_speech(self, audio: np.ndarray) -> bool:
        self._model.reset()
        audio = np.asarray(audio, dtype=np.float32)
        for start in range(0, len(audio) - self._window + 1, self._window):
            if self._model.is_speech(audio[start : start + self._window]):
                return True
        return False
