"""`volis.toml` - user selections only, by directory name.

Port of Rust `config.rs`. The same file, the same keys, the same defaults and
the same strictness: a key that isn't one of Rust's is an error, because it is
almost always a typo, and because Rust volis would refuse the file anyway
(`#[serde(deny_unknown_fields)]` at every level, including the top). That is
also why pyvolis never writes a section of its own here: a `volis.toml` saved
by pyvolis must still load in Rust volis. pyvolis-only settings get their own
file, `pyvolis.toml`, once a milestone needs one.

A key left out of a section takes its default. A missing file is not an error
(the defaults apply); a malformed one is.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomlkit
import tomlkit.items


class ConfigError(Exception):
    pass


@dataclass
class Asr:
    # Directory name under models/asr/. Empty = nothing selected yet.
    engine: str = ""


@dataclass
class Languages:
    source: str = "es"
    target: str = "en"


@dataclass
class Mode:
    kind: str = "turn"  # "continuous" | "turn" | "shared"
    # Window-focused key only. Never a global hotkey.
    turn_key: str = "Space"
    turn_style: str = "toggle"  # "toggle" | "hold"


@dataclass
class Audio:
    # Empty = system default.
    input_device: str = ""
    output_device: str = ""


@dataclass
class Vad:
    threshold: float = 0.5
    min_silence_ms: int = 500
    min_speech_ms: int = 250


@dataclass
class Tts:
    enabled: bool = True
    # Gate capture while speaking. False is for headphones only.
    half_duplex: bool = True


@dataclass
class Peer:
    enabled: bool = False
    listen_addr: str = "0.0.0.0:47800"
    # e.g. "192.168.50.2:47800". Empty = not dialling anyone.
    peer_addr: str = ""
    # Empty = hostname.
    display_name: str = ""
    # mDNS/broadcast convenience; manual entry always available.
    discovery: bool = True


@dataclass
class Shared:
    """Shared-machine mode: two people, one machine, a key each."""

    left_language: str = "en"
    right_language: str = "es"
    # The voice that speaks what the LEFT person said, so a voice in the
    # RIGHT person's language. Folder name under models/tts/; "" = first match.
    left_voice: str = ""
    # The voice that speaks what the RIGHT person said, in the LEFT language.
    right_voice: str = ""
    left_key: str = "ArrowLeft"
    right_key: str = "ArrowRight"
    # The recognizer that hears each side: folder under models/asr/; "" = best.
    left_asr: str = ""
    right_asr: str = ""


# Values Rust's enums accept, per (section, key).
ENUMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("mode", "kind"): ("continuous", "turn", "shared"),
    ("mode", "turn_style"): ("toggle", "hold"),
    ("context", "mode"): ("off", "carry"),  # pyvolis.toml
}


@dataclass
class Config:
    asr: Asr = field(default_factory=Asr)
    languages: Languages = field(default_factory=Languages)
    mode: Mode = field(default_factory=Mode)
    audio: Audio = field(default_factory=Audio)
    vad: Vad = field(default_factory=Vad)
    tts: Tts = field(default_factory=Tts)
    peer: Peer = field(default_factory=Peer)
    shared: Shared = field(default_factory=Shared)

    @classmethod
    def parse(cls, text: str) -> Config:
        """Parse TOML text with Rust's rules. Raises ConfigError."""
        try:
            data = tomlkit.parse(text).unwrap()
        except Exception as e:  # tomlkit raises several parse error types
            raise ConfigError(str(e)) from e
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        config = cls()
        sections = {f.name for f in dataclasses.fields(cls)}
        for name, value in data.items():
            if name not in sections:
                raise ConfigError(
                    f"unknown section [{name}], expected one of: {', '.join(sorted(sections))}"
                )
            if not isinstance(value, dict):
                raise ConfigError(f"[{name}] must be a table")
            section = getattr(config, name)
            setattr(config, name, _fill(name, section, value))
        return config

    @classmethod
    def load(cls, path: Path) -> tuple[Config, bool]:
        """Read `volis.toml`. A missing file gives the defaults and False, so
        the caller can say where the file was expected. A malformed file is an
        error naming the absolute path: silently falling back to defaults would
        hide the user's selections."""
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls(), False
        except OSError as e:
            raise ConfigError(f"failed to read {path.absolute()}: {e}") from e
        try:
            return cls.parse(text), True
        except ConfigError as e:
            raise ConfigError(f"failed to parse {path.absolute()}: {e}") from e

    def save_selections(self, path: Path) -> None:
        """Write back the selections the window can change, leaving everything
        else in the file as the user wrote it: comments, order, other keys.

        The same keys, in the same order, as Rust's `save_selections`. A file
        that does not parse is not overwritten.
        """
        try:
            existing = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            existing = ""
        except OSError as e:
            raise ConfigError(f"failed to read {path.absolute()}: {e}") from e
        try:
            doc = tomlkit.parse(existing)
        except Exception as e:
            raise ConfigError(f"failed to parse {path.absolute()}; not overwriting it: {e}") from e

        _set(doc, "asr", "engine", self.asr.engine)
        _set(doc, "languages", "source", self.languages.source)
        _set(doc, "languages", "target", self.languages.target)
        _set(doc, "audio", "input_device", self.audio.input_device)
        _set(doc, "audio", "output_device", self.audio.output_device)
        _set(doc, "mode", "kind", self.mode.kind)
        _set(doc, "mode", "turn_style", self.mode.turn_style)
        _set(doc, "tts", "enabled", self.tts.enabled)
        _set(doc, "tts", "half_duplex", self.tts.half_duplex)
        _set(doc, "peer", "enabled", self.peer.enabled)
        _set(doc, "peer", "peer_addr", self.peer.peer_addr)
        _set(doc, "shared", "left_language", self.shared.left_language)
        _set(doc, "shared", "right_language", self.shared.right_language)
        _set_explained(doc, "shared", "left_voice", self.shared.left_voice, VOICE_NOTE_LEFT)
        _set(doc, "shared", "left_asr", self.shared.left_asr)
        _set(doc, "shared", "right_asr", self.shared.right_asr)
        _set_explained(doc, "shared", "right_voice", self.shared.right_voice, VOICE_NOTE_RIGHT)

        try:
            path.write_text(tomlkit.dumps(doc), encoding="utf-8", newline="")
        except OSError as e:
            raise ConfigError(f"failed to write {path.absolute()}: {e}") from e


# ---------------------------------------------------------------- pyvolis.toml


@dataclass
class PyVad:
    # Audio restored before each utterance. Rust's fixed value is 600 ms;
    # 0 turns the pre-roll off, to measure what it does.
    pre_roll_ms: int = 600


@dataclass
class PyGuards:
    """Hallucination guards (asr/guards.py); each can be switched off to
    measure what it does."""

    vad_probability: bool = True
    # A segment whose Silero speech probability never reaches this is dropped.
    min_peak_probability: float = 0.8
    repeats: bool = True
    stock_phrases: bool = True
    # Too few words for the speech detected (noise shaped like speech).
    sparse: bool = True
    min_words_per_second: float = 0.33
    sparse_min_seconds: float = 3.0


@dataclass
class PyAsr:
    # Streaming recognition (LocalAgreement): provisional text while speech
    # goes on. Off for now: segment mode is the default until streaming is proven.
    streaming: bool = False
    # Seconds between passes over the growing utterance.
    interval_s: float = 1.0


@dataclass
class PyContext:
    # "carry": translate each sentence with the earlier ones as context.
    # "off": each sentence alone. ("revision" arrives at P8.)
    mode: str = "carry"
    sentences: int = 4  # how many earlier sentences, at most
    token_budget: int = 400  # and never more than this many tokens of them


@dataclass
class PyFragments:
    # Hold a short sentence with no final punctuation and join it to the next.
    hold: bool = True
    min_words: int = 4  # shorter than this is a fragment
    hold_ms: int = 1500


@dataclass
class PyTranslate:
    # The translator, as --report lists it: "qwen3-1.7b-q4_k_m.gguf" (a file at
    # the top of models/mt/) or "folder/file.gguf". Empty = the file at the
    # top, the one Rust volis uses, or else the first usable one.
    model: str = ""
    # A prompt file in prompts/, by name.
    prompt: str = "default"
    # "auto" = the GPU when the translator's build and the machine have one;
    # "cpu" = always the CPU, as Rust volis; "cuda" = the GPU or an error.
    device: str = "auto"


@dataclass
class PyvolisConfig:
    """`pyvolis.toml` at the app root: settings Rust volis doesn't have.

    Separate from volis.toml because Rust refuses unknown sections there.
    Same rules as volis.toml: unknown keys are errors, a missing key takes its
    default, a missing file means all defaults.
    """

    vad: PyVad = field(default_factory=PyVad)
    guards: PyGuards = field(default_factory=PyGuards)
    translate: PyTranslate = field(default_factory=PyTranslate)
    asr: PyAsr = field(default_factory=PyAsr)
    context: PyContext = field(default_factory=PyContext)
    fragments: PyFragments = field(default_factory=PyFragments)

    @classmethod
    def load(cls, path: Path) -> tuple[PyvolisConfig, bool]:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls(), False
        except OSError as e:
            raise ConfigError(f"failed to read {path.absolute()}: {e}") from e
        try:
            data = tomlkit.parse(text).unwrap()
        except Exception as e:
            raise ConfigError(f"failed to parse {path.absolute()}: {e}") from e
        config = cls()
        sections = {f.name for f in dataclasses.fields(cls)}
        try:
            for name, value in data.items():
                if name not in sections:
                    raise ConfigError(
                        f"unknown section [{name}], expected one of: {', '.join(sorted(sections))}"
                    )
                if not isinstance(value, dict):
                    raise ConfigError(f"[{name}] must be a table")
                setattr(config, name, _fill(name, getattr(config, name), value))
        except ConfigError as e:
            raise ConfigError(f"failed to parse {path.absolute()}: {e}") from e
        return config, True


def save_pyvolis_selections(path: Path, pyconfig: PyvolisConfig) -> None:
    """Write back what the window can change in pyvolis.toml (the translator
    and the prompt), keeping everything else and every comment, as
    `Config.save_selections` does for volis.toml."""
    try:
        existing = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = ""
    except OSError as e:
        raise ConfigError(f"failed to read {path.absolute()}: {e}") from e
    try:
        doc = tomlkit.parse(existing)
    except Exception as e:
        raise ConfigError(f"failed to parse {path.absolute()}; not overwriting it: {e}") from e
    _set(doc, "translate", "model", pyconfig.translate.model)
    _set(doc, "translate", "prompt", pyconfig.translate.prompt)
    _set(doc, "asr", "streaming", pyconfig.asr.streaming)
    _set(doc, "context", "mode", pyconfig.context.mode)
    _set(doc, "fragments", "hold", pyconfig.fragments.hold)
    try:
        path.write_text(tomlkit.dumps(doc), encoding="utf-8", newline="")
    except OSError as e:
        raise ConfigError(f"failed to write {path.absolute()}: {e}") from e


def _fill(section_name: str, section: Any, values: dict[str, Any]) -> Any:
    """A copy of `section` with `values` applied, checking names and types."""
    fields = {f.name: f for f in dataclasses.fields(section)}
    updates = {}
    for key, value in values.items():
        if key not in fields:
            raise ConfigError(
                f"unknown key `{key}` in [{section_name}], expected one of: "
                f"{', '.join(fields)}"
            )
        default = getattr(section, key)
        updates[key] = _check_type(section_name, key, default, value)
    return dataclasses.replace(section, **updates)


def _check_type(section: str, key: str, default: Any, value: Any) -> Any:
    where = f"[{section}].{key}"
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true or false, not {value!r}")
    elif isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"{where} must be a whole number of 0 or more, not {value!r}")
    elif isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} must be a number, not {value!r}")
        value = float(value)
    elif isinstance(default, str):
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string, not {value!r}")
        allowed = ENUMS.get((section, key))
        if allowed and value not in allowed:
            raise ConfigError(f"{where} = {value!r}; expected one of: {', '.join(allowed)}")
    return value


# The comments written above the voice keys when they are added, because which
# voice each key means is easy to get backwards. Byte for byte Rust's
# VOICE_NOTE_LEFT/RIGHT.
VOICE_NOTE_LEFT = (
    "# left_voice speaks what the LEFT person said, so it is a voice in the RIGHT\n"
    "# person's language. A folder name under models/tts/; empty = first match.\n"
)
VOICE_NOTE_RIGHT = "# right_voice speaks what the RIGHT person said, in the LEFT person's language.\n"


def _table(doc: tomlkit.TOMLDocument, name: str) -> tomlkit.items.Table:
    """The section, added as an ordinary `[section]` at the end if absent,
    never as an inline table: it has to read like the rest of the file."""
    if name not in doc:
        doc.add(name, tomlkit.table())
    return doc[name]


def _set(doc: tomlkit.TOMLDocument, table: str, key: str, value: Any) -> None:
    """Set one key, keeping any comment that sat beside the old value."""
    section = _table(doc, table)
    new = tomlkit.item(value)
    old = section.get(key) if key in section else None
    if isinstance(old, tomlkit.items.Item):
        new.trivia.indent = old.trivia.indent
        new.trivia.comment_ws = old.trivia.comment_ws
        new.trivia.comment = old.trivia.comment
        new.trivia.trail = old.trivia.trail
    section[key] = new


def _set_explained(
    doc: tomlkit.TOMLDocument, table: str, key: str, value: Any, note: str
) -> None:
    """Like `_set`, but a key added for the first time gets `note` above it."""
    section = _table(doc, table)
    if key in section:
        _set(doc, table, key, value)
        return
    new = tomlkit.item(value)
    # tomlkit writes an item's indent verbatim before its key, so the note goes
    # there: the same place Rust's toml_edit puts a key's prefix decor.
    new.trivia.indent = note
    section.add(key, new)
