"""Audio capture: device enumeration and a capture thread that hands the rest
of the pipeline 16 kHz mono float32.

Port of Rust `audio.rs`, on sounddevice (PortAudio) instead of cpal.
Resampling happens here, once, at the capture boundary. Every stage
downstream may assume 16 kHz mono float32 in [-1.0, 1.0] and nothing else.

Devices are the Windows WASAPI ones: cpal's default host on Windows is WASAPI,
so these are the names Rust volis lists and `[audio].input_device` holds.
(PortAudio also lists each device under MME, DirectSound and WDM-KS, where
MME truncates names to 31 characters.)

Two threads, as in Rust:

* PortAudio's callback runs on the device's own thread. It downmixes to mono
  and hands the samples off. It never resamples and never blocks: if the queue
  is full it drops the chunk and counts it.
* The capture thread owns the resampler and forwards 16 kHz chunks downstream.
"""

from __future__ import annotations

import concurrent.futures
import logging
import queue
import sys
import threading
from dataclasses import dataclass

import numpy as np
import soxr

log = logging.getLogger(__name__)

# The one sample rate the pipeline speaks. Chosen by the ASR and VAD models.
SAMPLE_RATE = 16_000

# Chunks of device audio queued between the callback and the capture thread.
DEVICE_QUEUE_CHUNKS = 64

# How often the capture thread wakes to check the stop flag while idle.
POLL_SECONDS = 0.1


class AudioError(Exception):
    pass


@dataclass
class Device:
    """An audio device as offered to the user, for input or output."""

    name: str
    is_default: bool
    # Native configuration, as text for display, in cpal's words.
    default_config: str | None
    index: int = -1


# Every PortAudio call that opens, starts, stops or closes a stream runs on
# this one thread, whoever asks, and PortAudio itself is initialised on it
# (sounddevice initialises PortAudio when it is first imported). On Windows
# (WASAPI through PortAudio), a stream fails to start ("Unanticipated host
# error") when it is opened on a different thread from the one that
# initialised PortAudio, and pyvolis opens streams from the pipeline's thread
# (capture, the voice) and the window's (playing a row, the original).
# Callbacks still run on PortAudio's own device threads.
_audio_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pyvolis-audio")


def _sd():
    """sounddevice, imported (and so PortAudio initialised) on the audio thread."""

    def load():
        import sounddevice

        return sounddevice

    return on_audio_thread(load)


def on_audio_thread(fn, *args, **kwargs):
    """Run a PortAudio call on the audio thread and return its result (or
    raise its exception) here."""
    if threading.current_thread().name.startswith("pyvolis-audio"):
        return fn(*args, **kwargs)
    return _audio_thread.submit(fn, *args, **kwargs).result()


def _wasapi() -> dict:
    sd = _sd()
    for index, api in enumerate(sd.query_hostapis()):
        if "WASAPI" in api["name"]:
            return {"index": index, **api}
    if sys.platform == "win32":
        raise AudioError("PortAudio lists no Windows WASAPI host API")
    # Elsewhere, PortAudio's default host API plays the part cpal's does.
    index = sd.default.hostapi
    return {"index": index, **sd.query_hostapis(index)}


def _list(kind: str) -> list[Device]:
    return on_audio_thread(_list_now, kind)


def _list_now(kind: str) -> list[Device]:
    sd = _sd()
    api = _wasapi()
    channels_key = f"max_{kind}_channels"
    default = api[f"default_{kind}_device"]
    devices = []
    for device in sd.query_devices():
        if device["hostapi"] != api["index"] or device[channels_key] < 1:
            continue
        # WASAPI shared mode always hands over float32 at the mix format's
        # channels and rate, which is what cpal reports as the default.
        config = f"{device[channels_key]} ch, {int(device['default_samplerate'])} Hz, F32"
        devices.append(Device(device["name"], device["index"] == default, config, device["index"]))
    return devices


def list_input_devices() -> list[Device]:
    return _list("input")


def list_output_devices() -> list[Device]:
    """Output devices beside input ones, as in Rust: the two lists are read and
    printed together."""
    return _list("output")


def select_input_device(name: str) -> Device:
    """Empty name = the system default. A named device that is not present is
    an error naming what was asked for; pyvolis never picks another one."""
    devices = list_input_devices()
    if not name.strip():
        for device in devices:
            if device.is_default:
                return device
        raise AudioError("no default input device; is a microphone connected?")
    for device in devices:
        if device.name == name:
            return device
    raise AudioError(
        f'input device "{name}" was not found; run with --devices to list what is available'
    )


def downmix(data: np.ndarray, channels: int) -> np.ndarray:
    """Average interleaved channels down to mono float32. A trailing partial
    frame is dropped, never averaged from half a frame."""
    data = np.asarray(data, dtype=np.float32).reshape(-1)
    if channels <= 1:
        return data
    frames = len(data) // channels
    return data[: frames * channels].reshape(frames, channels).mean(axis=1, dtype=np.float32)


def to_float(data: np.ndarray) -> np.ndarray:
    """Integer PCM to float32 in [-1, 1], the way cpal's FromSample does."""
    if data.dtype.kind == "f":
        return data.astype(np.float32, copy=False)
    if data.dtype.kind == "i":
        return (data.astype(np.float32) / float(-np.iinfo(data.dtype).min)).astype(np.float32)
    if data.dtype.kind == "u":
        info = np.iinfo(data.dtype)
        middle = (info.max + 1) / 2
        return ((data.astype(np.float32) - middle) / middle).astype(np.float32)
    raise AudioError(f"unsupported sample type {data.dtype}")


class Resampler:
    """Streaming resampler to 16 kHz mono (soxr). Rust uses sherpa-onnx's
    linear resampler here; soxr is a better filter (see HANDOFF.md)."""

    def __init__(self, from_rate: int) -> None:
        self.from_rate = from_rate
        self._stream = (
            None if from_rate == SAMPLE_RATE else soxr.ResampleStream(from_rate, SAMPLE_RATE, 1, dtype="float32")
        )

    def __call__(self, mono: np.ndarray, last: bool = False) -> np.ndarray:
        if self._stream is None:
            return mono.astype(np.float32, copy=False)
        return self._stream.resample_chunk(mono.astype(np.float32, copy=False), last=last)


def resample(mono: np.ndarray, from_rate: int) -> np.ndarray:
    """Resample a whole buffer to 16 kHz."""
    return Resampler(from_rate)(mono, last=True)


class CaptureHandle:
    """A running capture. Call `stop()` so the device is released
    deterministically."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped_chunks = 0  # chunks the callback had to discard

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def __enter__(self) -> CaptureHandle:
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def spawn_capture(device_name: str, out: queue.Queue) -> CaptureHandle:
    """Open `device_name` (empty = system default) and put 16 kHz mono float32
    chunks into `out` until the handle is stopped. A device that can't be
    opened is an error from this call, not a silent no-op."""
    sd = _sd()
    device = select_input_device(device_name)
    info = on_audio_thread(sd.query_devices, device.index)
    channels = int(info["max_input_channels"])
    rate = int(info["default_samplerate"])
    handle = CaptureHandle()
    raw: queue.Queue = queue.Queue(DEVICE_QUEUE_CHUNKS)

    def callback(indata, frames, time_info, status) -> None:  # device thread
        mono = downmix(indata, channels)
        if len(mono) == 0:
            return
        try:
            raw.put_nowait(mono.copy())
        except queue.Full:
            handle.dropped_chunks += 1

    def open_stream():
        stream = sd.InputStream(
            device=device.index, channels=channels, samplerate=rate, dtype="float32", callback=callback
        )
        stream.start()
        return stream

    def close_stream() -> None:
        stream.stop()
        stream.close()  # releases the device

    try:
        stream = on_audio_thread(open_stream)
    except Exception as e:  # PortAudioError and friends
        raise AudioError(f'cannot open input device "{device.name}": {e}') from e

    def pump() -> None:
        resampler = Resampler(rate)
        try:
            while not handle._stop.is_set():
                try:
                    chunk = raw.get(timeout=POLL_SECONDS)
                except queue.Empty:
                    continue
                resampled = resampler(chunk)
                if len(resampled) == 0:
                    continue
                try:
                    out.put_nowait(resampled)
                except queue.Full:
                    log.debug("pipeline is behind; dropped a 16 kHz chunk")
        finally:
            on_audio_thread(close_stream)
            if handle.dropped_chunks:
                log.warning(
                    "capture dropped %d chunk(s): the machine could not keep up", handle.dropped_chunks
                )
            log.info('capture stopped: "%s"', device.name)

    handle._thread = threading.Thread(target=pump, name="pyvolis-capture", daemon=True)
    handle._thread.start()
    log.info('capture started: "%s" at %d Hz, %d ch -> %d Hz mono', device.name, rate, channels, SAMPLE_RATE)
    return handle
