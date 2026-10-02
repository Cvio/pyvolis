"""Shared-machine mode: two people who speak different languages use one
machine, each with their own key. Port of Rust `shared.rs`.

The key says who is talking, and so which language they speak. Each side has
its own recognizer, chosen for that side's language, so a model tuned for one
language can hear one person while another hears the other.

This module is the rule that turns "which side pressed" into a direction: the
recognizer and language to hear it with, the language to translate into, and
the voice to speak it with. It has no window, device or model in it, so every
case is tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import models, varieties
from .config import Shared
from .models import Engine, Ranked

LEFT, RIGHT = "left", "right"
SIDES = (LEFT, RIGHT)


def other(side: str) -> str:
    return RIGHT if side == LEFT else LEFT


def language(side: str, settings: Shared) -> str:
    """This side's language in the settings."""
    return settings.left_language if side == LEFT else settings.right_language


def voice(side: str, settings: Shared) -> str:
    """The voice that speaks this side's words, which is a voice in the other
    side's language."""
    return settings.left_voice if side == LEFT else settings.right_voice


def asr(side: str, settings: Shared) -> str:
    """The recognizer configured for this side: a folder name, or empty for
    the best match."""
    return settings.left_asr if side == LEFT else settings.right_asr


def key(side: str, settings: Shared) -> str:
    return settings.left_key if side == LEFT else settings.right_key


@dataclass(frozen=True)
class Direction:
    """Everything one turn needs to know about where its words go."""

    side: str
    asr: str  # the recognizer's folder name under models/asr/
    source: str  # what the speaker is saying, passed to the recognizer as its language
    target: str  # what it is translated into and spoken in
    voice: str  # the voice's folder name under models/tts/


@dataclass
class Resolved:
    """A direction that can be used, possibly with something the side's
    column should show, such as a configured voice that has gone missing."""

    direction: Direction
    warnings: list[str] = field(default_factory=list)


class SharedError(Exception):
    """Why a side can have no turn. Shown in that side's column."""


def recognizer_for(side: str, settings: Shared, recognizers: list[Engine], preferred: str) -> Engine:
    """The recognizer that hears one side, or SharedError saying why there is none.

    The configured one when set; it must be installed, complete, and cover the
    side's language, and it is never swapped for another. When unset, the best
    match: `preferred` (the main recognizer) if it is among the best-fitting,
    otherwise the first of those."""
    tag = language(side, settings).strip()
    configured = asr(side, settings).strip()
    if configured:
        engine = next((e for e in recognizers if e.dir_name == configured), None)
        if engine is None:
            raise SharedError(f'the recognizer "{configured}" is not installed')
        if not engine.enabled():
            raise SharedError(f"{engine.name} {_why_unusable(engine)}")
        if not declares(engine, tag):
            raise SharedError(f'{engine.name} does not list "{tag}" among its languages, so it cannot hear this side')
        return engine
    # The best-fitting group (models.rank); within it, the main recognizer if
    # it is there, so an unset side doesn't load a second model for nothing.
    ranked = models.rank(tag, recognizers)
    if not ranked:
        raise SharedError(f'no installed recognizer lists "{tag}" among its languages')
    best = ranked[0]
    for r in ranked:
        if r.fit != best.fit:
            break
        if r.engine.dir_name == preferred:
            return r.engine
    return best.engine


def _why_unusable(engine: Engine) -> str:
    return f"can't be used: {engine.unusable}" if engine.unusable else \
        "is missing files: " + ", ".join(engine.missing_files())


def recognizers_for(tag: str, recognizers: list[Engine]) -> list[Ranked]:
    """The usable recognizers that cover `tag`, for a side's picker."""
    return models.rank(tag, recognizers)


def direction(side: str, settings: Shared, recognizers: list[Engine], preferred: str, voices: list[Engine]) -> Resolved:
    """Work out where one turn's words go, or raise SharedError saying why
    there can be no turn.

    `recognizers` and `voices` are every discovered model, broken ones
    included; only usable ones are chosen. `preferred` is the main recognizer,
    the first choice for a side that has none configured."""
    source = language(side, settings).strip()
    target = language(other(side), settings).strip()
    if not source or not target:
        raise SharedError("choose a language for both sides")
    # A tag the table doesn't know is refused here, where the column shows it.
    for tag in (source, target):
        try:
            varieties.require(tag)
        except ValueError as e:
            raise SharedError(str(e)) from None
    if source == target:
        raise SharedError(f'both sides are set to "{source}"; choose a different language for each')

    recognizer = recognizer_for(side, settings, recognizers, preferred)
    warnings = []
    if not takes_language(recognizer):
        warnings.append(f"{recognizer.name} decides the language itself rather than being told it, so it may "
                        "mishear a short sentence")
    if not recognizer.languages_known:
        warnings.append(f"{recognizer.name} doesn't say which languages it knows; it may not know "
                        f"{varieties.display_name(source)}")

    configured = voice(side, settings).strip()
    chosen = next((v for v in voices if v.dir_name == configured and v.enabled() and declares(v, target)), None)
    if chosen is None:
        ranked = voices_for(target, voices)
        if not ranked:
            raise SharedError(f'no installed voice speaks "{target}", so this side\'s words could not be spoken')
        chosen = ranked[0].engine
        if configured:
            warnings.append(f'the voice "{configured}" is not installed, or does not speak "{target}"; using '
                            f'"{chosen.dir_name}" instead')
    return Resolved(Direction(side, recognizer.dir_name, source, target, chosen.dir_name), warnings)


def voices_for(tag: str, voices: list[Engine]) -> list[Ranked]:
    """The usable voices that speak `tag`, best first, for a side's voice picker."""
    return models.rank(tag, voices)


def takes_language(engine: Engine) -> bool:
    """Whether the recognizer is told the language. Parakeet (a NeMo
    transducer) decides it itself; Whisper and everything pyvolis runs through
    transformers are told."""
    return engine.backend != "nemo_transducer"


def declares(engine: Engine, tag: str) -> bool:
    """Whether a model covers the language of a tag (`es` for `es-MX`). A
    model that doesn't say which languages it knows (pyvolis only) is taken
    at the user's word."""
    if not engine.languages_known:
        return True
    wanted = varieties.language_of(tag).lower()
    return any(lang.lower() == wanted for lang in engine.languages)
