"""Speech recognition: the two interfaces the pipeline talks to, and loading.

Port of the shape of Rust `asr.rs`. Whole-utterance recognizers (Whisper,
Parakeet, MMS, Cohere Transcribe) implement `SegmentAsr`; streaming
recognizers implement `StreamAsr` (P7). Which backend runs a model is decided
by discovery from the folder's contents (`models.py`); the pipeline never
knows.

Backends:
  sherpa        engine.toml folders, through sherpa-onnx exactly as Rust (asr/sherpa.py)
  transformers  downloaded Hugging Face folders (asr/hf.py)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..models import Engine

log = logging.getLogger(__name__)


class AsrError(Exception):
    """A model that can't load or can't transcribe. Names the model and folder."""


class AsrMemoryError(AsrError):
    """A model refused because it would not fit in memory."""


@dataclass
class Word:
    text: str
    start: float  # seconds from the start of the utterance
    end: float


@dataclass
class AsrResult:
    text: str
    # Word timings, where the backend has them (Whisper through transformers,
    # CTC models). None: the backend gives none.
    words: list[Word] | None = None
    # The backend's own confidence, 0..1, where it has one.
    confidence: float | None = None
    # The language the model was told (never detected).
    language: str = ""
    seconds: float = 0.0  # wall-clock time to transcribe


@dataclass
class Memory:
    gpu_bytes: int = 0
    cpu_bytes: int = 0
    device: str = "cpu"


class SegmentAsr(Protocol):
    """One finished utterance in, text out."""

    name: str

    def transcribe(self, audio: np.ndarray, language: str, timestamps: bool = True) -> AsrResult:
        """`audio` is a complete utterance, 16 kHz mono float32 in [-1, 1].
        `language` may be a variety; only its language part reaches the model.
        `timestamps=False` says word timings aren't needed: for Whisper they
        cost about a second a call, which live use and most streaming passes
        shouldn't pay."""
        ...

    def prepare(self, language: str) -> None:
        """Get ready for `language` without delay (shared-machine mode calls it
        for both sides at start). Nothing to do for most backends."""
        ...

    def memory(self) -> Memory: ...

    def close(self) -> None:
        """Release the model's memory."""
        ...


class StreamAsr(Protocol):
    """Audio in chunks, provisional and committed text out. Implemented at P7."""

    def feed(self, chunk: np.ndarray) -> object: ...

    def finish(self) -> object: ...


def load(engine: Engine) -> SegmentAsr:
    """Load the recognizer a discovered folder describes. Refuses anything
    discovery marked disabled or unusable, naming why, rather than failing
    inside a library."""
    missing = engine.missing_files()
    if missing:
        raise AsrError(
            f'model "{engine.dir_name}" is incomplete; these files are not in {engine.dir}: '
            f"{', '.join(missing)}"
        )
    if engine.unusable:
        raise AsrError(f'model "{engine.dir_name}" can\'t be used: {engine.unusable}')
    if engine.from_engine_toml:
        from . import sherpa

        return sherpa.load(engine)
    if engine.backend == "transformers":
        from . import hf

        return hf.load(engine)
    if engine.backend == "llamacpp-audio":
        from . import llamacpp_audio

        return llamacpp_audio.load(engine)
    raise AsrError(f'model "{engine.dir_name}" has backend "{engine.backend}", which pyvolis can\'t run yet')
