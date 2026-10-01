"""The paired-mode wire protocol (SPEC §9): newline-delimited JSON over TCP.

Port of Rust `wire.rs`, version 2, unchanged: pyvolis pairs with Rust volis.
Only text crosses it: never audio, never models, never context, never a
revision.

Everything that arrives is untrusted. `decode` bounds every length, refuses
a line that is not UTF-8, refuses an unknown protocol version, and strips
control characters from every string, so what comes out is only ever
displayed or spoken. Received text is never a command, a path, or an
instruction to a model, and `source_text` is never translated again.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass

PROTO = 2  # 2: languages may be varieties (es-MX)
MAX_LINE = 16 * 1024  # bytes, newline included
MAX_TEXT = 2_000  # characters
MAX_NAME = 64
MAX_LANG = 16
U64_MAX = 2**64 - 1


@dataclass(frozen=True)
class Hello:
    name: str
    speaks: str  # the language this end's person speaks
    sends: str  # the language this end translates into, and so sends
    proto: int = PROTO


@dataclass(frozen=True)
class Utterance:
    seq: int
    lang: str  # decides which voice speaks it; the receiver never infers it
    text: str
    source_lang: str
    source_text: str  # display only; never translated again


@dataclass(frozen=True)
class FloorRequest:
    seq: int


@dataclass(frozen=True)
class FloorGrant:
    seq: int


@dataclass(frozen=True)
class FloorRelease:
    seq: int


@dataclass(frozen=True)
class Ping:
    pass


@dataclass(frozen=True)
class Pong:
    pass


@dataclass(frozen=True)
class Bye:
    reason: str | None = None  # how a second connection learns why it was refused


Wire = Hello | Utterance | FloorRequest | FloorGrant | FloorRelease | Ping | Pong | Bye
_KINDS = {c.__name__: c for c in (Hello, Utterance, FloorRequest, FloorGrant, FloorRelease, Ping, Pong, Bye)}
# What serde requires of each variant: field -> its JSON type.
_FIELDS = {
    Hello: {"name": str, "speaks": str, "sends": str, "proto": int},
    Utterance: {"seq": int, "lang": str, "text": str, "source_lang": str, "source_text": str},
    FloorRequest: {"seq": int}, FloorGrant: {"seq": int}, FloorRelease: {"seq": int},
    Ping: {}, Pong: {}, Bye: {},
}


class WireError(Exception):
    """Why a line was refused."""


class TooLong(WireError):
    def __init__(self) -> None:
        super().__init__(f"a message longer than {MAX_LINE} bytes arrived")


class NotUtf8(WireError):
    def __init__(self) -> None:
        super().__init__("a message that is not UTF-8 text arrived")


class UnknownProto(WireError):
    def __init__(self, proto: int) -> None:
        self.proto = proto
        super().__init__(f"the other side speaks protocol version {proto}; this Volis speaks version {PROTO}. "
                         "Use the same Volis version on both PCs.")


class Malformed(WireError):
    def __init__(self, why: str) -> None:
        super().__init__(f"a message Volis does not understand arrived: {why}")


def encode(message: Wire) -> str:
    """One message as a line, newline included: the JSON Rust's serde writes."""
    data: dict = {"t": type(message).__name__}
    for name in message.__dataclass_fields__:
        value = getattr(message, name)
        if isinstance(message, Bye) and value is None:
            continue  # a bare {"t":"Bye"}
        data[name] = value
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"


def decode(line: bytes) -> Wire:
    """Parse and check one received line. A trailing newline is allowed."""
    if len(line) > MAX_LINE:
        raise TooLong()
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError:
        raise NotUtf8() from None
    try:
        value = json.loads(text.rstrip(), parse_constant=_no_constants)
    except (ValueError, RecursionError) as e:
        raise Malformed(str(e)) from None
    if not isinstance(value, dict):
        raise Malformed("not a JSON object")

    # Any message carrying a version other than ours is refused, not only
    # Hello, so a newer peer is told plainly rather than half-understood.
    if "proto" in value:
        proto = value["proto"]
        if not _is_u64(proto):
            raise Malformed('"proto" is not a number')
        if proto != PROTO:
            raise UnknownProto(proto)

    kind = _KINDS.get(value.get("t")) if isinstance(value.get("t"), str) else None
    if kind is None:
        raise Malformed(f'unknown message type {value.get("t")!r}')
    fields = {}
    for name, wanted in _FIELDS[kind].items():
        if name not in value:
            raise Malformed(f'missing field "{name}"')
        got = value[name]
        if wanted is int and not _is_u64(got) or wanted is str and not isinstance(got, str):
            raise Malformed(f'"{name}" is not {"a number" if wanted is int else "text"}')
        fields[name] = got
    if kind is Bye:
        reason = value.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise Malformed('"reason" is not text')
        fields["reason"] = reason
    return _sanitise(kind(**fields))


def _no_constants(name: str):
    raise ValueError(f"{name} is not JSON")


def _is_u64(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= U64_MAX


def _sanitise(message: Wire) -> Wire:
    """Bound and clean every string in a received message."""
    if isinstance(message, Hello):
        return Hello(bounded(message.name, MAX_NAME, "name"), lang(message.speaks), lang(message.sends), message.proto)
    if isinstance(message, Utterance):
        return Utterance(message.seq, lang(message.lang), bounded(message.text, MAX_TEXT, "text"),
                         lang(message.source_lang), bounded(message.source_text, MAX_TEXT, "source_text"))
    if isinstance(message, Bye) and message.reason is not None:
        return Bye(bounded(message.reason, MAX_TEXT, "reason"))
    return message


def is_control(c: str) -> bool:
    return unicodedata.category(c) == "Cc"


def bounded(text: str, limit: int, field: str) -> str:
    """Control characters become spaces, runs of whitespace collapse, and
    anything over `limit` characters is refused rather than cut: a truncated
    sentence would be spoken as if it were the whole one."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:  # half of a surrogate pair, which JSON can spell and UTF-8 can't hold
        raise NotUtf8() from None
    cleaned = " ".join("".join(" " if is_control(c) else c for c in text).split())
    if len(cleaned) > limit:
        raise Malformed(f'"{field}" is longer than {limit} characters')
    return cleaned


def lang(code: str) -> str:
    """A language code: letters and hyphens, short."""
    if code and len(code) <= MAX_LANG and all(c == "-" or (c.isascii() and c.isalpha()) for c in code):
        return code
    raise Malformed(f"{json.dumps(code)} is not a language code")
