"""Voice activity detection: Silero through sherpa-onnx.

Port of Rust `vad.rs`. sherpa's VoiceActivityDetector does the segmenting
itself: it is fed fixed windows and hands back complete speech segments with
their start offset. The `[vad]` keys in volis.toml map straight onto its
config.

On top of it, the pre-roll: the detector reports a segment from the point
where it became confident someone was speaking, and a soft onset comes before
that point (the breath of "Hola", the "¿D" of "¿Dónde", a quiet "No"). The
audio just before each segment is kept and put back on the front of it. Rust
restores a fixed 600 ms; in pyvolis the amount is `[vad].pre_roll_ms` in
pyvolis.toml (default 600, the Rust value; 0 turns it off) so its effect can
be measured.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE

# Samples per call into the detector. Silero is trained on 512-sample windows
# at 16 kHz and the upstream examples feed exactly that.
WINDOW = 512

# Seconds of audio the detector may buffer internally.
BUFFER_SECONDS = 30.0

# Longest single segment before the detector cuts one anyway.
MAX_SPEECH_SECONDS = 20.0

# Rust's PRE_ROLL_MS: 300 ms fixed "¿Dónde"; "¿Cuántos años tienes?" needed 600.
DEFAULT_PRE_ROLL_MS = 600


class VadError(Exception):
    pass


@dataclass
class Segment:
    """One complete utterance as the detector cut it."""

    start_sample: int  # offset of the first sample, from the start of capture
    samples: np.ndarray  # 16 kHz mono float32, the utterance only
    # Where the detector itself said speech began: start_sample plus the
    # lead-in the pre-roll restored. pyvolis only, for measuring the pre-roll.
    detected_sample: int = -1

    def start_ms(self) -> int:
        return self.start_sample * 1000 // SAMPLE_RATE

    def duration_ms(self) -> int:
        return len(self.samples) * 1000 // SAMPLE_RATE

    def end_ms(self) -> int:
        return self.start_ms() + self.duration_ms()

    def lead_in_ms(self) -> int:
        return (self.detected_sample - self.start_sample) * 1000 // SAMPLE_RATE


@dataclass
class VadSettings:
    model: Path
    threshold: float = 0.5
    min_silence_ms: int = 500
    min_speech_ms: int = 250
    pre_roll_ms: int = DEFAULT_PRE_ROLL_MS


class Segmenter:
    """Feeds audio to Silero and yields the utterances it cuts."""

    def __init__(self, settings: VadSettings) -> None:
        import sherpa_onnx

        model = settings.model.absolute()
        if not model.is_file():
            raise VadError(
                f"VAD model not found: {model}\npyvolis never downloads models; place "
                "silero_vad.onnx at that exact path (see README.md) and run again."
            )
        silero = sherpa_onnx.SileroVadModelConfig()
        silero.model = str(model)
        silero.threshold = settings.threshold
        silero.min_silence_duration = settings.min_silence_ms / 1000.0
        silero.min_speech_duration = settings.min_speech_ms / 1000.0
        silero.window_size = WINDOW
        silero.max_speech_duration = MAX_SPEECH_SECONDS
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad = silero
        config.sample_rate = SAMPLE_RATE
        config.num_threads = 1
        config.provider = "cpu"
        if not config.validate():
            raise VadError(f"sherpa-onnx refused the VAD configuration for {model}")
        try:
            self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=BUFFER_SECONDS)
        except Exception as e:
            raise VadError(f"sherpa-onnx refused to create the voice activity detector from {model}: {e}") from e

        self.pre_roll = settings.pre_roll_ms * SAMPLE_RATE // 1000
        # Recent audio exactly as the detector received it, reaching back past
        # the longest segment plus the silence that ends it.
        history_seconds = MAX_SPEECH_SECONDS + settings.min_silence_ms / 1000 + settings.pre_roll_ms / 1000 + 1.0
        self._history_cap = int(history_seconds * SAMPLE_RATE)
        self._history: deque[np.ndarray] = deque()
        self._history_len = 0
        self._pending = np.zeros(0, np.float32)
        self._fed = 0  # samples accepted since creation or the last reset
        self._last_end = 0  # where the previous segment ended
        self._origin = 0  # capture position of the first sample since reset

    def push(self, pcm: np.ndarray) -> list[Segment]:
        """Feed 16 kHz mono audio. Returns the utterances it completed."""
        pending = np.concatenate([self._pending, np.asarray(pcm, np.float32)])
        whole = len(pending) - len(pending) % WINDOW
        for start in range(0, whole, WINDOW):
            self._accept(pending[start : start + WINDOW])
        self._pending = pending[whole:]
        return self._drain()

    def flush(self) -> list[Segment]:
        """End of input: push out any utterance still being accumulated."""
        real_end = self._origin + self._fed + len(self._pending)
        if len(self._pending):
            # Pad the last partial window; the detector only accepts full ones.
            padded = np.zeros(WINDOW, np.float32)
            padded[: len(self._pending)] = self._pending
            self._pending = np.zeros(0, np.float32)
            self._accept(padded)
        self._vad.flush()
        # A segment holds only audio that was captured, never the padding.
        segments = self._drain()
        for segment in segments:
            captured = max(0, real_end - segment.start_sample)
            segment.samples = segment.samples[:captured]
        return [s for s in segments if len(s.samples)]

    def speech_in_progress(self) -> bool:
        """True while the detector believes someone is speaking."""
        return self._vad.is_speech_detected()

    def current(self) -> Segment | None:
        """The utterance still being spoken, from its pre-roll to the audio
        fed so far, or None when nobody is speaking. Streaming recognition
        transcribes this while the speech goes on; its first sample is the
        one the finished segment will start at."""
        if not self._vad.is_speech_detected():
            return None
        start = int(self._vad.current_segment.start)
        if start < 0:
            return None
        oldest = self._fed - self._history_len
        lead = min(self.pre_roll, max(0, start - self._last_end), max(0, start - oldest))
        samples = self._history_slice(start - lead, self._fed)
        return Segment(self._origin + start - lead, samples, detected_sample=self._origin + start)

    def reset(self, origin: int) -> None:
        """Forget all state and queued segments. `origin` is the capture
        position of the next sample that will be fed (the half-duplex gate
        holds the detector reset while the voice speaks)."""
        self._pending = np.zeros(0, np.float32)
        # Rust calls clear() then reset(); the Python binding has no clear(),
        # so the queued segments are popped instead.
        while not self._vad.empty():
            self._vad.pop()
        self._vad.reset()
        self._history.clear()
        self._history_len = 0
        self._fed = 0
        self._last_end = 0
        self._origin = origin

    def _accept(self, window: np.ndarray) -> None:
        self._vad.accept_waveform(window)
        self._history.append(window)
        self._history_len += len(window)
        self._fed += len(window)
        while self._history_len - len(self._history[0]) >= self._history_cap:
            self._history_len -= len(self._history.popleft())

    def _history_slice(self, start: int, end: int) -> np.ndarray:
        """History samples at detector positions [start, end)."""
        oldest = self._fed - self._history_len
        if end <= start:
            return np.zeros(0, np.float32)
        flat = np.concatenate(self._history) if len(self._history) > 1 else self._history[0]
        return flat[start - oldest : end - oldest]

    def _drain(self) -> list[Segment]:
        segments = []
        while not self._vad.empty():
            front = self._vad.front
            start = max(0, int(front.start))
            body = np.asarray(front.samples, dtype=np.float32)
            # The pre-roll, but never back past the previous segment, and
            # never further than history reaches.
            oldest = self._fed - self._history_len
            lead = min(self.pre_roll, max(0, start - self._last_end), max(0, start - oldest))
            samples = np.concatenate([self._history_slice(start - lead, start), body])
            self._last_end = start + len(body)
            segments.append(
                Segment(self._origin + start - lead, samples, detected_sample=self._origin + start)
            )
            self._vad.pop()
        return segments
