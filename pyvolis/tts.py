"""Speech synthesis through sherpa-onnx's OfflineTts. Port of Rust `tts.rs`.

Which voice speaks is decided by language, not by a setting: the voice is the
discovered model that declares the target language, best fit first (tuned for
the variety, then general, then other varieties). Never a voice in another
language: being told no English voice is installed is more useful than
hearing Spanish.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from . import models
from .models import Engine

log = logging.getLogger(__name__)

# Rust's NUM_THREADS: synthesis runs while the recognizer may be working.
NUM_THREADS = 2


class TtsError(Exception):
    pass


@dataclass
class Speech:
    samples: np.ndarray
    sample_rate: int  # the voice's own; resampling to the device is playback's job

    def duration_ms(self) -> int:
        return len(self.samples) * 1000 // self.sample_rate if self.sample_rate else 0


class Voice:
    def __init__(self, engine: Engine) -> None:
        import sherpa_onnx

        missing = engine.missing_files()
        if missing:
            raise TtsError(f'voice "{engine.dir_name}" is incomplete; these are not in {engine.dir}: {", ".join(missing)}')
        if engine.backend != "vits":
            raise TtsError(f'"{engine.dir_name}" is a {engine.backend} model, not a voice')
        files = {f.role: str(f.path) for f in engine.files}
        began = time.perf_counter()
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=files["model"],
            lexicon=files.get("lexicon", ""),
            tokens=files["tokens"],
            # Piper voices pronounce through espeak-ng and are silent nonsense without it.
            data_dir=str(engine.data_dir.path) if engine.data_dir else "",
        )
        model = sherpa_onnx.OfflineTtsModelConfig()
        model.vits = vits
        model.num_threads = NUM_THREADS
        model.provider = "cpu"
        # One sentence at a time keeps latency down.
        config = sherpa_onnx.OfflineTtsConfig(model=model, max_num_sentences=1)
        try:
            self._tts = sherpa_onnx.OfflineTts(config)
        except Exception as e:
            raise TtsError(f"sherpa-onnx refused to load the voice in {engine.dir}: {e}") from e
        self.sample_rate = int(self._tts.sample_rate)
        if self.sample_rate <= 0:
            raise TtsError(f"the voice in {engine.dir} reports a sample rate of {self.sample_rate}")
        self.name = engine.dir_name
        log.info('loaded voice "%s" (%s) in %d ms: %d Hz, %d speaker(s)', engine.dir_name, engine.name,
                 (time.perf_counter() - began) * 1000, self.sample_rate, self._tts.num_speakers)

    def speak(self, text: str) -> Speech:
        text = text.strip()
        if not text:
            return Speech(np.zeros(0, np.float32), self.sample_rate)
        try:
            audio = self._tts.generate(text)
        except Exception as e:
            raise TtsError(f'"{self.name}" produced no audio for {text!r}: {e}') from e
        samples = np.asarray(audio.samples, dtype=np.float32)
        return Speech(samples, int(audio.sample_rate) or self.sample_rate)


def choose(engines: list[Engine], language: str) -> Engine:
    """The voice for a language or variety: best fit first. Raises TtsError if
    none speaks it, naming what is installed."""
    ranked = models.rank(language, engines)
    if ranked:
        return ranked[0].engine
    usable = [e for e in engines if e.enabled()]
    found = ", ".join(f"{e.dir_name} ({'/'.join(e.languages)})" for e in usable) or "none"
    raise TtsError(f'no installed voice speaks "{language}". Voices found: {found}')


def for_language(engines: list[Engine], language: str) -> Engine:
    """`choose`, logging the choice when more than one voice fits."""
    best = choose(engines, language)
    ranked = models.rank(language, engines)
    if len(ranked) > 1:
        log.info('%d voices speak "%s"; using "%s" (%s)', len(ranked), language, best.dir_name,
                 ranked[0].fit.label(language))
    return best
