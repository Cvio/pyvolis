"""Audio files as a source: WAV, MP3, M4A, FLAC, OGG and anything else FFmpeg
reads, decoded with PyAV (its wheels include the decoders) and brought to
16 kHz mono float32 the same way the microphone is: averaged to mono, then
resampled with `audio.Resampler`.

New in pyvolis (Rust volis has no file input). The file becomes an audio
source in place of the microphone, so everything after it is the live
pipeline. Pacing (real time or fast) and pause arrive with file mode at P4.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, Resampler


class FileSourceError(Exception):
    pass


def read_16k_mono(path: Path) -> np.ndarray:
    """The whole file as 16 kHz mono float32. Errors name the absolute path."""
    import av

    where = path.absolute()
    if not path.is_file():
        raise FileSourceError(f"no audio file at {where}")
    try:
        container = av.open(str(path))
    except Exception as e:  # av.error.* for unreadable or unknown formats
        raise FileSourceError(f"cannot open {where}: {e}") from e
    with container:
        streams = container.streams.audio
        if not streams:
            raise FileSourceError(f"{where} has no audio stream")
        stream = streams[0]
        # Any sample type, planar or packed, becomes planar float32 at the
        # file's own rate and layout; the channels are averaged here and the
        # rate changed by the same resampler the microphone uses.
        to_float = av.AudioResampler(format="fltp")
        resampler = None
        parts: list[np.ndarray] = []

        def take(frames) -> None:
            nonlocal resampler
            for frame in frames:
                data = frame.to_ndarray()  # (channels, samples)
                if resampler is None:
                    resampler = Resampler(frame.sample_rate)
                parts.append(resampler(data.mean(axis=0, dtype=np.float32)))

        try:
            for frame in container.decode(stream):
                take(to_float.resample(frame))
            take(to_float.resample(None))  # flush
        except Exception as e:
            raise FileSourceError(f"cannot decode {where}: {e}") from e
        if resampler is not None:
            parts.append(resampler(np.zeros(0, np.float32), last=True))
    if not parts:
        return np.zeros(0, np.float32)
    return np.clip(np.concatenate(parts), -1.0, 1.0).astype(np.float32)


def chunks(audio: np.ndarray, size: int = 1600) -> Iterator[np.ndarray]:
    """The audio in capture-sized pieces (100 ms by default), as the
    microphone would deliver it."""
    for start in range(0, len(audio), size):
        yield audio[start : start + size]


def duration_seconds(audio: np.ndarray) -> float:
    return len(audio) / SAMPLE_RATE
