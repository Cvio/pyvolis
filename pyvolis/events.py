"""Everything the pipeline reports, as events: one set for the window,
`--listen`, file mode and `events.jsonl`.

Port of Rust's `PipelineMsg`, extended. The whole set is defined now, so the
export format doesn't change as streaming (P7) and revision (P8) arrive;
an event a milestone doesn't produce yet is simply never sent.

Times: `wall` is seconds since the pipeline started (every event has it, set
when the event is made). `start`/`end` are seconds on the source's timeline:
for a file, the position in the file. Sentence ids are "<utterance>.<n>".
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any

_T0 = time.monotonic()


def _now() -> float:
    return round(time.monotonic() - _T0, 3)


def restart_clock() -> None:
    """`wall` counts from here (the start of a run)."""
    global _T0
    _T0 = time.monotonic()


@dataclass
class Event:
    wall: float = field(default_factory=_now, kw_only=True)

    def to_json(self) -> str:
        data: dict[str, Any] = {"t": type(self).__name__}
        for f in dataclasses.fields(self):
            data[f.name] = _plain(getattr(self, f.name))
        return json.dumps(data, ensure_ascii=False)


def _plain(value):
    if dataclasses.is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, float):
        return round(value, 3)
    if hasattr(value, "item"):  # numpy scalars
        return _plain(value.item())
    return value


# ---------------------------------------------------------------- run

@dataclass
class Configuration(Event):
    """First line of events.jsonl: everything needed to reproduce the run."""

    settings: dict


@dataclass
class Loading(Event):
    """A model is being loaded; the window shows "Loading..."."""

    what: str


@dataclass
class ModelLoaded(Event):
    """A model is in memory: what it is, where, and how much it takes."""

    role: str  # "recognizer" | "translator" | "voice"
    name: str
    device: str  # "cpu" | "cuda"
    gpu_bytes: int = 0
    cpu_bytes: int = 0
    seconds: float = 0.0


@dataclass
class Level(Event):
    """Peak input level over the last 200 ms, in dBFS; None = digital silence."""

    db: float | None


@dataclass
class Listening(Event):
    """The source is open; --seconds counts from here."""


@dataclass
class Mode(Event):
    """The mode the pipeline is in: "continuous" or "turn"."""

    kind: str


@dataclass
class TurnStarted(Event):
    """A turn is open: the microphone is live. In shared-machine mode,
    `side` is whose turn it is ("left" or "right")."""

    side: str = ""


@dataclass
class SharedSide(Event):
    """Shared-machine mode: whether one side's recognizer is ready, and if
    not, why. Shown in that side's column."""

    side: str
    problem: str = ""


@dataclass
class TurnEnded(Event):
    """The turn is over and the microphone closed; its words are being
    recognised, translated and spoken."""


@dataclass
class TurnCancelled(Event):
    """A turn, or what it was producing, was cancelled; nothing from it is spoken."""


@dataclass
class Progress(Event):
    """A file source's position (seconds) and length."""

    position: float
    duration: float


@dataclass
class Stopped(Event):
    pass


@dataclass
class Error(Event):
    message: str


@dataclass
class Summary(Event):
    """Last line: the run's numbers (the status line)."""

    stats: dict


# ---------------------------------------------------------------- recognition

@dataclass
class SpeechStarted(Event):
    start: float = 0.0


@dataclass
class Partial(Event):
    """P7, streaming: provisional text of the utterance so far."""

    utterance: int
    text: str  # provisional: the current best guess, which may still change
    committed: str = ""  # everything committed in this utterance so far
    pending: str = ""  # committed words that don't yet make a whole sentence


@dataclass
class Final(Event):
    """An utterance's transcript, after the guards."""

    index: int
    text: str
    lang: str
    speech_ms: int
    asr_ms: int
    start: float = 0.0
    end: float = 0.0
    words: list | None = None


@dataclass
class NothingRecognized(Event):
    index: int
    speech_ms: int
    start: float = 0.0


@dataclass
class Dropped(Event):
    """Text the hallucination guards removed, with every reason."""

    index: int
    text: str
    reasons: list[str]
    start: float = 0.0


@dataclass
class ComparisonMsg(Event):
    comparison: Any


# ---------------------------------------------------------------- sentences and translation

@dataclass
class SentenceMsg(Event):
    """A committed sentence, as it goes to the translator."""

    id: str
    utterance: int
    text: str
    lang: str
    start: float
    end: float
    approximate: bool = False  # times shared out by length, not from word timings


@dataclass
class Held(Event):
    """P7: a short fragment held to be joined to the next sentence."""

    id: str
    text: str


@dataclass
class Translated(Event):
    id: str
    text: str
    lang: str
    translate_ms: int
    device: str  # "cpu" or "cuda"
    model: str


@dataclass
class NotTranslated(Event):
    id: str
    reason: str
    guard: str = ""  # "echo" | "recited" | "too long" | "" for errors and drops


@dataclass
class SpeakingStarted(Event):
    """Speech reached the sound card. `first_audio_ms` runs from the moment
    the utterance was cut, through recognition, translation and synthesis."""

    id: str
    first_audio_ms: int
    voice: str = ""


@dataclass
class SpeakingEnded(Event):
    pass


@dataclass
class Revised(Event):
    """P8: an earlier sentence's translation, replaced."""

    id: str
    old: str
    new: str


@dataclass
class Stall(Event):
    """The pipeline's own threads were held up (a library holding Python's
    lock): how late a 50 ms timer fired."""

    late_ms: int


RECOGNITION = (Partial, Final, NothingRecognized, Dropped)
TRANSLATION = (SentenceMsg, Held, Translated, NotTranslated, Revised)


# ---------------------------------------------------------------- paired mode (P9)


@dataclass
class PeerMsg(Event):
    """The connection to the other PC changed. `kind`: "waiting" (listening,
    not connected), "connecting", "connected", "disconnected" (was, or tried
    to be; still listening) or "unavailable" (the port could not be opened)."""

    kind: str
    port: int = 0
    addr: str = ""  # "ip:port" of the other PC
    name: str = ""
    speaks: str = ""
    sends: str = ""
    reason: str = ""


@dataclass
class FloorChanged(Event):
    """Who holds the floor: "free", "asking", "me" or "them" (with their name)."""

    holder: str
    name: str = ""


@dataclass
class FloorRefused(Event):
    """A turn was asked for and did not happen, and why. The microphone stayed closed."""

    why: str


@dataclass
class Remote(Event):
    """An utterance from the other PC, already translated into this PC's
    language. `source_text` is for display only."""

    sender: str
    lang: str
    text: str
    source_lang: str
    source_text: str


@dataclass
class Sent(Event):
    """A local translation reached the other PC."""

    id: str
    to: str


@dataclass
class NotSent(Event):
    """A local translation did not reach the other PC, and why."""

    id: str
    reason: str


@dataclass
class Discovered(Event):
    """Other Volis PCs heard on the local network: (name, host, port) each."""

    found: list
