"""Playback, and the half-duplex gate that keeps pyvolis from hearing itself.
Port of Rust `playback.rs`.

With the voice playing through speakers and the microphone live, the detector
triggers on the translated speech and the recognizer transcribes it: a loop.
The fix is not echo cancellation; it is not listening while speaking. So
playback owns a `Gate`. It closes when speech is queued for the sound card and
opens 150 ms after the last sample has played. While it is closed the pipeline
discards captured audio and holds the detector reset.

In turn-based mode (P6) the user decides when the microphone is live:
`PlaybackControl.begin_turn` cuts off any reply and holds what arrives until
`end_turn`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import numpy as np
import soxr

from . import audio

log = logging.getLogger(__name__)

TAIL_SECONDS = 0.150  # how long after the last sample the microphone stays shut
POLL_SECONDS = 0.010  # how often the player checks whether the queue drained
MAX_QUEUED_SECONDS = 60  # past this something is wrong; drop rather than grow


class PlaybackError(Exception):
    pass


class Gate:
    """Whether the microphone should be ignored right now. Shared between the
    player, which closes it, and the pipeline, which reads it on every chunk."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled  # False: half-duplex off, for headphones
        self._speaking = False
        self._open_at = 0.0  # monotonic time the tail expires; 0 = no tail
        self._lock = threading.Lock()

    def is_closed(self) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            return self._speaking or (self._open_at > 0 and time.monotonic() < self._open_at)

    def speaking_started(self) -> None:
        with self._lock:
            self._speaking = True

    def speaking_ended(self) -> None:
        with self._lock:
            self._open_at = time.monotonic() + TAIL_SECONDS
            self._speaking = False

    def open_now(self) -> None:
        """No tail: a reply was cut short because the user took a turn."""
        with self._lock:
            self._open_at = 0.0
            self._speaking = False


class Player:
    """A running output stream. `close()` stops playback and releases the device."""

    def __init__(self, device_name: str, gate: Gate, on_ended=None) -> None:
        sd = audio._sd()
        device = select_output_device(device_name)
        info = audio.on_audio_thread(sd.query_devices, device.index)
        self.channels = int(info["max_output_channels"])
        self.device_rate = int(info["default_samplerate"])
        self.name = device.name
        self.gate = gate
        self._on_ended = on_ended
        self._queue: deque[np.ndarray] = deque()  # mono blocks at the device rate
        self._queued = 0  # samples in the queue
        self._lock = threading.Lock()
        self._hold = False  # a user's turn: nothing plays, what's queued waits
        self._stop = threading.Event()

        def callback(outdata, frames, time_info, status) -> None:  # device thread
            out = np.zeros(frames, np.float32)
            if not self._hold:
                with self._lock:
                    filled = 0
                    while filled < frames and self._queue:
                        block = self._queue[0]
                        take = min(len(block), frames - filled)
                        out[filled : filled + take] = block[:take]
                        filled += take
                        if take == len(block):
                            self._queue.popleft()
                        else:
                            self._queue[0] = block[take:]
                    self._queued -= filled
            # The same mono sample goes to every channel.
            outdata[:] = out[:, None]

        def open_stream():
            stream = sd.OutputStream(device=device.index, channels=self.channels,
                                     samplerate=self.device_rate, dtype="float32", callback=callback)
            stream.start()
            return stream

        try:
            self._stream = audio.on_audio_thread(open_stream)  # see audio.on_audio_thread
        except Exception as e:
            raise PlaybackError(f'cannot open output device "{device.name}": {e}') from e
        self._thread = threading.Thread(target=self._watch, name="pyvolis-playback", daemon=True)
        self._thread.start()
        log.info('playback started: "%s" at %d Hz, %d ch', device.name, self.device_rate, self.channels)

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        """Queue speech. Returns at once. The gate closes here, before the
        first sample reaches the speakers."""
        if len(samples) == 0:
            return
        if sample_rate != self.device_rate:
            samples = soxr.resample(np.asarray(samples, np.float32), sample_rate, self.device_rate)
        samples = np.asarray(samples, np.float32)
        with self._lock:
            if self._queued + len(samples) > self.device_rate * MAX_QUEUED_SECONDS:
                log.warning("playback queue is full (%d s); dropping an utterance of %d ms",
                            MAX_QUEUED_SECONDS, len(samples) * 1000 // self.device_rate)
                return
            # A reply arriving during the user's turn waits, and must not close
            # the gate on the turn's audio. Only once the audio is certainly queued.
            if not self._hold:
                self.gate.speaking_started()
            self._queue.append(samples)
            self._queued += len(samples)

    def queued(self) -> int:
        with self._lock:
            return self._queued

    def pause(self, paused: bool) -> None:
        """Hold what is queued without discarding it (file mode's Pause)."""
        self._hold = paused

    def clear(self) -> None:
        """Forget anything queued."""
        with self._lock:
            self._queue.clear()
            self._queued = 0

    def control(self) -> PlaybackControl:
        return PlaybackControl(self)

    def _watch(self) -> None:
        """Watch for the queue draining, off the device thread."""
        was_speaking = False
        while not self._stop.is_set():
            if self._hold:
                if was_speaking:
                    was_speaking = False
                    self._ended()
            elif self.queued() > 0:
                was_speaking = True
            elif was_speaking:
                was_speaking = False
                self.gate.speaking_ended()
                log.debug("speaking ended; the microphone reopens in %d ms", TAIL_SECONDS * 1000)
                self._ended()
            time.sleep(POLL_SECONDS)

    def _ended(self) -> None:
        if self._on_ended is not None:
            self._on_ended()

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        def close_stream() -> None:
            self._stream.stop()
            self._stream.close()

        try:
            audio.on_audio_thread(close_stream)
        finally:
            self.gate.speaking_ended()
            log.info('playback stopped: "%s"', self.name)


class PlaybackControl:
    """What taking a turn does to playback."""

    def __init__(self, player: Player) -> None:
        self._player = player

    def begin_turn(self) -> None:
        """Cut off any reply mid-word, open the gate at once, hold what arrives."""
        p = self._player
        p._hold = True
        with p._lock:
            p._queue.clear()
            p._queued = 0
        p.gate.open_now()

    def stop(self) -> None:
        """Stop speaking now and forget anything queued."""
        p = self._player
        with p._lock:
            p._queue.clear()
            p._queued = 0

    def end_turn(self) -> None:
        """The turn is over: play anything that arrived during it."""
        p = self._player
        if p.queued() > 0:
            p.gate.speaking_started()
        p._hold = False


def select_output_device(name: str) -> audio.Device:
    """Empty name = the system default; a named device that isn't present is
    an error naming it."""
    devices = audio.list_output_devices()
    if not name.strip():
        for device in devices:
            if device.is_default:
                return device
        raise PlaybackError("no default output device; are speakers or headphones connected?")
    for device in devices:
        if device.name == name:
            return device
    raise PlaybackError(f'output device "{name}" was not found; run with --devices to list what is available')
