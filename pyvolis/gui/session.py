"""What the events do to the window. No Qt in this file: it is plain Python,
unit-tested, and `window.py` only draws it. Port of Rust `gui::Session`.

One difference in shape from Rust: a caption there is an utterance with one
translation; here a row is a sentence (pyvolis translates per sentence), and
the same rows are the live captions and the file timeline. Turn-taking,
pairing and the shared machine join at P6, P9 and P10.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import events as ev

# Lines kept in the pane (Rust's MAX_LINES).
MAX_LINES = 200

STOPPED, STARTING, LISTENING = "stopped", "starting", "listening"


@dataclass
class Row:
    """One sentence and everything that happened to it afterwards."""

    id: str
    utterance: int
    start: float  # seconds on the source's timeline
    end: float
    # The languages it was spoken and translated in, kept on the row because
    # the settings may have changed since.
    source_lang: str
    target_lang: str
    source: str
    target: str | None = None
    problem: str | None = None  # why there is no translation, when there won't be
    speech_ms: int = 0
    asr_ms: int = 0
    translate_ms: int | None = None
    first_audio_ms: int | None = None
    spoken: bool = False  # reached the sound card (P8: never revised after)
    revised: bool = False
    history: list[str] = field(default_factory=list)  # earlier translations (P8)
    approximate: bool = False  # times shared out by length

    kind = "row"


@dataclass
class Nothing:
    """Speech in which the recognizer found no words: shown, not hidden."""

    index: int
    speech_ms: int
    start: float = 0.0
    kind = "nothing"


@dataclass
class DroppedLine:
    """Text the hallucination guards removed, with why."""

    index: int
    text: str
    reasons: list[str]
    start: float = 0.0
    kind = "dropped"


@dataclass
class ComparisonLine:
    comparison: object
    kind = "comparison"


@dataclass
class LoadedModel:
    role: str
    name: str
    device: str
    gpu_bytes: int
    cpu_bytes: int


class Session:
    """The window's state, as changed by pipeline events."""

    def __init__(self) -> None:
        self.state = STOPPED
        self.loading = ""  # what is being loaded, while STARTING
        self.speaking = False
        self.level_db: float | None = None
        self.has_level = False
        self.lines: list = []
        self.last_error: str | None = None
        self.languages = ("", "")
        self.comparing = False
        self.speaks = False
        self.models: dict[str, LoadedModel] = {}  # by role
        self.progress: tuple[float, float] | None = None  # file: position, duration
        self.stats: dict = {}
        self.worst_stall_ms = 0
        self._utterances: dict[int, tuple[int, int]] = {}  # index -> speech_ms, asr_ms

    def begin(self, source: str, target: str, comparing: bool, speaks: bool) -> None:
        """A run is starting with these settings."""
        self.last_error = None
        self.languages = (source, target)
        self.comparing = comparing
        # Comparing turns speech off.
        self.speaks = speaks and not comparing
        self.models = {}
        self.progress = None
        self.stats = {}
        self.worst_stall_ms = 0
        self.state = STARTING
        self.loading = "models"

    def clear(self) -> None:
        self.lines = []
        self._utterances = {}

    def apply(self, event) -> None:
        if isinstance(event, ev.Loading):
            self.state, self.loading = STARTING, event.what
        elif isinstance(event, ev.ModelLoaded):
            self.models[event.role] = LoadedModel(event.role, event.name, event.device, event.gpu_bytes, event.cpu_bytes)
        elif isinstance(event, ev.Listening):
            self.state, self.loading = LISTENING, ""
            self.last_error = None
        elif isinstance(event, ev.Level):
            self.level_db, self.has_level = event.db, True
        elif isinstance(event, ev.Progress):
            self.progress = (event.position, event.duration)
        elif isinstance(event, ev.Final):
            self._utterances[event.index] = (event.speech_ms, event.asr_ms)
        elif isinstance(event, ev.SentenceMsg):
            speech_ms, asr_ms = self._utterances.get(event.utterance, (0, 0))
            self._push(Row(event.id, event.utterance, event.start, event.end, event.lang,
                           # Until the translation says what it is in.
                           self.languages[1], event.text, speech_ms=speech_ms, asr_ms=asr_ms,
                           approximate=event.approximate))
        elif isinstance(event, ev.Translated):
            if (row := self.row(event.id)) is not None:
                row.target, row.target_lang, row.translate_ms = event.text, event.lang, event.translate_ms
                row.problem = None
        elif isinstance(event, ev.NotTranslated):
            if (row := self.row(event.id)) is not None:
                row.problem = event.reason
        elif isinstance(event, ev.Revised):
            if (row := self.row(event.id)) is not None:
                row.history.append(event.old)
                row.target, row.revised = event.new, True
        elif isinstance(event, ev.NothingRecognized):
            self._push(Nothing(event.index, event.speech_ms, event.start))
        elif isinstance(event, ev.Dropped):
            self._push(DroppedLine(event.index, event.text, event.reasons, event.start))
        elif isinstance(event, ev.SpeakingStarted):
            self.speaking = True
            if (row := self.row(event.id)) is not None:
                row.first_audio_ms, row.spoken = event.first_audio_ms, True
        elif isinstance(event, ev.SpeakingEnded):
            self.speaking = False
        elif isinstance(event, ev.ComparisonMsg):
            self._push(ComparisonLine(event.comparison))
        elif isinstance(event, ev.Stall):
            self.worst_stall_ms = max(self.worst_stall_ms, event.late_ms)
        elif isinstance(event, ev.Summary):
            self.stats = event.stats
        elif isinstance(event, ev.Error):
            self.last_error = event.message
        elif isinstance(event, ev.Stopped):
            self.state, self.loading = STOPPED, ""
            self.speaking = False
            self.level_db, self.has_level = None, False

    def row(self, sentence_id: str) -> Row | None:
        for line in reversed(self.lines):
            if isinstance(line, Row) and line.id == sentence_id:
                return line
        return None

    def rows(self) -> list[Row]:
        return [line for line in self.lines if isinstance(line, Row)]

    def latest_timed(self) -> Row | None:
        """The most recent row with every stage it went through, for the
        latency readout."""
        for line in reversed(self.lines):
            if isinstance(line, Row) and line.translate_ms is not None:
                return line
        return None

    def counts(self) -> dict:
        rows = self.rows()
        return {
            "sentences": len(rows),
            "translated": sum(1 for r in rows if r.target),
            "not_translated": sum(1 for r in rows if r.problem),
            "revisions": sum(len(r.history) for r in rows),
            "dropped": sum(1 for line in self.lines if isinstance(line, DroppedLine)),
        }

    def memory(self) -> tuple[int, int]:
        """Bytes the loaded models take: (GPU, system)."""
        return (sum(m.gpu_bytes for m in self.models.values()), sum(m.cpu_bytes for m in self.models.values()))

    def _push(self, line) -> None:
        self.lines.append(line)
        if len(self.lines) > MAX_LINES:
            del self.lines[: len(self.lines) - MAX_LINES]


def has_rtl(text: str) -> bool:
    """Whether text holds right-to-left script (Hebrew, Arabic, Persian and
    their presentation forms), which decides a row's direction."""
    return any(
        "֐" <= c <= "ࣿ" or "יִ" <= c <= "﷿" or "ﹰ" <= c <= "﻿"
        for c in text
    )


def clock(seconds: float) -> str:
    """A position as m:ss or h:mm:ss."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
