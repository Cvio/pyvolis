"""The sherpa backend: engine.toml folders through sherpa-onnx 1.13.8, configured
exactly as Rust `asr.rs` configures them, so the text is identical.

Parakeet (nemo_transducer) decides the language itself and ignores the one
it's told. Whisper has the language baked in when its recognizer is created,
and creating one loads the model, so a recognizer is kept per language, as in
Rust (shared-machine mode alternates two languages every turn).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .. import varieties
from ..audio import SAMPLE_RATE
from ..models import Engine
from . import AsrError, AsrResult, Memory

log = logging.getLogger(__name__)

# Rust's NUM_THREADS: measured on the i7-12700H laptop, 6 threads (the
# performance cores) beat both 2 and 12.
NUM_THREADS = 6


def load(engine: Engine):
    if engine.backend == "nemo_transducer":
        return NemoTransducerAsr(engine)
    if engine.backend == "whisper":
        return WhisperAsr(engine)
    raise AsrError(f'"{engine.dir_name}" is a {engine.backend} model, not a recognizer')


def _file(engine: Engine, role: str) -> str:
    for f in engine.files:
        if f.role == role:
            return str(f.path)
    raise AsrError(f"{engine.dir / 'engine.toml'} does not declare [files].{role}")


def _decode(recognizer, audio: np.ndarray) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, np.asarray(audio, dtype=np.float32))
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def _file_bytes(engine: Engine) -> int:
    return sum(f.path.stat().st_size for f in engine.files if f.present and f.path.suffix == ".onnx")


class NemoTransducerAsr:
    """Parakeet TDT and other NeMo transducers."""

    def __init__(self, engine: Engine) -> None:
        import sherpa_onnx

        self.name = engine.dir_name
        self._engine = engine
        try:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=_file(engine, "encoder"),
                decoder=_file(engine, "decoder"),
                joiner=_file(engine, "joiner"),
                tokens=_file(engine, "tokens"),
                num_threads=NUM_THREADS,
                provider="cpu",
                model_type="nemo_transducer",
            )
        except Exception as e:
            raise AsrError(f"sherpa-onnx refused to load the transducer model in {engine.dir}: {e}") from e

    def transcribe(self, audio: np.ndarray, language: str) -> AsrResult:
        began = time.perf_counter()
        try:
            text = _decode(self._recognizer, audio)
        except Exception as e:
            raise AsrError(f'"{self.name}" failed to transcribe: {e}') from e
        return AsrResult(text, language=varieties.language_of(language), seconds=time.perf_counter() - began)

    def prepare(self, language: str) -> None:
        pass

    def memory(self) -> Memory:
        return Memory(cpu_bytes=_file_bytes(self._engine))

    def close(self) -> None:
        self._recognizer = None


class WhisperAsr:
    """Whisper through sherpa-onnx: one recognizer per language used."""

    def __init__(self, engine: Engine) -> None:
        self.name = engine.dir_name
        self._engine = engine
        self._encoder = _file(engine, "encoder")
        self._decoder = _file(engine, "decoder")
        self._tokens = _file(engine, "tokens")
        self._recognizers: dict[str, object] = {}
        # Whichever language the first utterance asks for replaces this.
        first = engine.languages[0] if engine.languages else "en"
        self._recognizers[first] = self._build(first)
        log.info(
            '"%s" is Whisper: it pads every utterance to 30 s internally, so a 1 s utterance '
            "costs about what a 20 s one does",
            self.name,
        )

    def _build(self, language: str):
        import sherpa_onnx

        try:
            return sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=self._encoder,
                decoder=self._decoder,
                tokens=self._tokens,
                language=language,
                task="transcribe",
                num_threads=NUM_THREADS,
                provider="cpu",
                # Rust passes 0; the Python binding's default is -1.
                tail_paddings=0,
                enable_token_timestamps=False,
                enable_segment_timestamps=False,
            )
        except Exception as e:
            raise AsrError(
                f'sherpa-onnx refused to load the Whisper model in {self._engine.dir} for language "{language}": {e}'
            ) from e

    def _recognizer(self, language: str):
        if not language:
            return next(iter(self._recognizers.values()))
        if language not in self._recognizers:
            began = time.perf_counter()
            self._recognizers[language] = self._build(language)
            log.info(
                '"%s": built a recognizer for "%s" in %d ms; %d language(s) now loaded',
                self.name, language, (time.perf_counter() - began) * 1000, len(self._recognizers),
            )
        return self._recognizers[language]

    def transcribe(self, audio: np.ndarray, language: str) -> AsrResult:
        # Whisper takes plain language codes only; the variety matters through
        # which model hears it, not here.
        language = varieties.language_of(language)
        log.info('  "%s" is told the language "%s"', self.name, language)
        began = time.perf_counter()
        try:
            text = _decode(self._recognizer(language), audio)
        except AsrError:
            raise
        except Exception as e:
            raise AsrError(f'"{self.name}" failed to transcribe: {e}') from e
        return AsrResult(text, language=language, seconds=time.perf_counter() - began)

    def prepare(self, language: str) -> None:
        self._recognizer(varieties.language_of(language))

    def memory(self) -> Memory:
        return Memory(cpu_bytes=_file_bytes(self._engine) * max(1, len(self._recognizers)))

    def close(self) -> None:
        self._recognizers.clear()
