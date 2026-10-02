"""Translation: the interface the pipeline talks to, and loading.

Port of the shape of Rust `translate.rs`. Every translator (llama.cpp GGUF
now; transformers at P11) implements `Translator`, and every output passes
the same cleaning and the same three guards (`guards.py`), whichever model
produced it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .. import varieties
from ..models import Translator as TranslatorEntry
from . import guards
from .prompts import PromptFile

log = logging.getLogger(__name__)


class TranslateError(Exception):
    """A translator that can't load or can't translate. Names the model."""


class Refused(TranslateError):
    """An output a guard rejected: not captioned, not spoken."""

    def __init__(self, guard: str, message: str) -> None:
        super().__init__(message)
        self.guard = guard  # "echo" | "recited" | "too long" | "empty"


@dataclass
class Turn:
    """One earlier exchange, for carry-forward context (P7)."""

    source: str
    translation: str
    lang: str = ""  # the source's language, when a session translates both ways


@dataclass
class TranslationRequest:
    text: str
    source: str  # language or variety tag
    target: str
    context: list[Turn] = field(default_factory=list)  # P7
    glossary: list[str] = field(default_factory=list)  # P7


@dataclass
class TranslationResult:
    text: str
    raw: str  # before cleaning
    seconds: float
    device: str  # "cpu" or "cuda": every timing says which was used


class Translator(Protocol):
    name: str
    device: str

    def translate(self, request: TranslationRequest) -> TranslationResult: ...

    def prompt(self, request: TranslationRequest) -> str:
        """The exact prompt that would be sent (--print-prompt)."""
        ...

    def close(self) -> None: ...


def load(entry: TranslatorEntry, prompt: PromptFile, device: str = "auto") -> Translator:
    if not entry.path.exists():
        raise TranslateError(
            f"translation model not found: {entry.path.absolute()}\npyvolis never downloads models; "
            "place the GGUF at that exact path (see README.md) and run again."
        )
    if entry.missing:
        raise TranslateError(f'translator "{entry.id}" is incomplete; missing: {", ".join(entry.missing)}')
    if entry.unusable:
        raise TranslateError(f'translator "{entry.id}" can\'t be used: {entry.unusable}')
    if entry.lora is not None and not entry.lora.is_file():
        raise TranslateError(f'translator "{entry.id}": its LoRA adapter is not at {entry.lora.absolute()}')
    if entry.backend == "llamacpp":
        from .llamacpp import LlamaTranslator

        return LlamaTranslator(entry, prompt, device)
    if entry.backend == "transformers":
        from .hf import TransformersTranslator

        return TransformersTranslator(entry, prompt, device)
    raise TranslateError(f'translator "{entry.id}" has backend "{entry.backend}", which pyvolis can\'t run yet')


def system_text(prompt: PromptFile, request: TranslationRequest) -> str:
    """The system turn for a request: the prompt file's text for the pair,
    then the session glossary when there is one."""
    from .context import glossary_line

    text = prompt.system_text(request.source, request.target)
    glossary = glossary_line(request.glossary)
    return f"{text}\n{glossary}" if glossary else text


def translate_checked(translator, request: TranslationRequest, generate) -> TranslationResult:
    """Rust's `Translator::translate` around any backend's raw generation:
    unknown tags refused before any prompt is built, then clean, then the three
    guards. `generate(request)` returns the raw model output."""
    text = request.text.strip()
    if not text:
        return TranslationResult("", "", 0.0, translator.device)
    # A tag the table doesn't know would put "into xx-YY" in the prompt.
    for tag in (request.source, request.target):
        try:
            varieties.require(tag)
        except ValueError as e:
            raise TranslateError(str(e)) from None
    began = time.perf_counter()
    raw = generate(request)
    seconds = time.perf_counter() - began
    cleaned = guards.clean(raw)
    if cleaned != raw.strip():
        log.debug("translation cleaned from %r to %r", raw, cleaned)
    # Reciting the instructions, the glossary or the context is no translation.
    context = [t.source for t in request.context] + [t.translation for t in request.context]
    wrapper = translator.prompt_file.wrapper(request.source, request.target)
    context.append(wrapper)
    if guards.leaks_the_prompt(cleaned, system_text(translator.prompt_file, request), context) \
            or guards.contains_the_wrapper(cleaned, wrapper):
        raise Refused(
            "recited",
            "the model recited its own instructions instead of translating. This happens when the "
            "recognizer hands it nonsense; the utterance is dropped rather than captioned and spoken.",
        )
    if guards.is_implausibly_long(text, cleaned):
        raise Refused(
            "too long",
            f"the model produced {len(cleaned)} characters for a {len(text)} character utterance, "
            "which is an answer or a ramble rather than a translation.",
        )
    if guards.is_echo(text, cleaned):
        raise Refused(
            "echo",
            f"the translation came back as the {request.source} text unchanged, so it was neither "
            "captioned nor spoken",
        )
    return TranslationResult(cleaned, raw, seconds, translator.device)


def model_file(entry: TranslatorEntry) -> Path:
    return entry.path


def choose(root: Path, wanted: str) -> TranslatorEntry:
    """The translator to use: `wanted` by its --report id, or else the file at
    the top of models/mt/ (Rust volis's), or else the first usable one. A
    translator asked for by name that isn't there is an error; pyvolis never
    substitutes another."""
    from .. import models, paths

    found = [t for t in models.discover_translators(paths.mt_dir(root)) if isinstance(t, TranslatorEntry)]
    if wanted:
        for t in found:
            if t.id == wanted:
                return t
        ids = ", ".join(t.id for t in found) or "none"
        raise TranslateError(f'translator "{wanted}" is not in {paths.mt_dir(root)} (found: {ids})')
    usable = [t for t in found if t.enabled()]
    top = [t for t in usable if t.top_level]
    if top or usable:
        return (top or usable)[0]
    raise TranslateError(
        f"no usable translation model in {paths.mt_dir(root)}\npyvolis never downloads models; put a "
        ".gguf there (see README.md) and run again."
    )
