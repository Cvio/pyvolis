"""Port of the tests in Rust `audio.rs`, plus the file source."""

from pathlib import Path

import numpy as np
import pytest

from pyvolis import audio, filesource
from pyvolis.audio import SAMPLE_RATE


def test_downmix_averages_the_channels():
    stereo = np.array([1.0, 0.0, 0.5, 0.5, -1.0, 1.0], np.float32)
    assert audio.downmix(stereo, 2).tolist() == [0.5, 0.5, 0.0]


def test_downmix_passes_mono_through():
    assert audio.downmix(np.array([0.25, -0.25], np.float32), 1).tolist() == [0.25, -0.25]


def test_downmix_ignores_a_trailing_partial_frame():
    assert audio.downmix(np.array([1.0, 1.0, 0.0], np.float32), 2).tolist() == [1.0]


def test_downmix_takes_the_device_callbacks_frames_by_channels_shape():
    frames = np.array([[1.0, 0.0], [0.5, 0.5]], np.float32)  # sounddevice's layout
    assert audio.downmix(frames, 2).tolist() == [0.5, 0.5]


def tone(rate: int, seconds: float, hz: float = 1000.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * hz * t) * 0.5).astype(np.float32)


def crossings_hz(x: np.ndarray, seconds: float) -> float:
    return np.count_nonzero(np.signbit(x[1:]) != np.signbit(x[:-1])) / 2 / seconds


def test_resampling_preserves_the_tone_and_the_duration():
    out = audio.resample(tone(48_000, 0.5), 48_000)
    assert abs(len(out) - SAMPLE_RATE // 2) < SAMPLE_RATE // 100
    assert abs(crossings_hz(out, 0.5) - 1000) < 20


def test_streaming_resampling_matches_whole_buffer_resampling_in_length():
    x = tone(44_100, 1.0)
    r = audio.Resampler(44_100)
    pieces = [r(x[i : i + 441]) for i in range(0, len(x), 441)] + [r(np.zeros(0, np.float32), last=True)]
    assert abs(sum(map(len, pieces)) - SAMPLE_RATE) < 32


def test_integer_samples_become_normalised_floats():
    mono = audio.to_float(np.array([32767, 0, -32768], np.int16))
    assert abs(mono[0] - 1.0) < 1e-3 and mono[1] == 0.0 and abs(mono[2] + 1.0) < 1e-3


def test_an_unknown_input_device_is_an_error_naming_it():
    try:
        audio.list_input_devices()
    except audio.AudioError:
        pytest.skip("no audio host API on this machine")
    with pytest.raises(audio.AudioError, match="Definitely Not A Microphone"):
        audio.select_input_device("Definitely Not A Microphone")


# ---------------------------------------------------------------- file source


def encode(path: Path, samples: np.ndarray, rate: int, channels: int) -> None:
    """Write `samples` (mono) to any container PyAV can encode, duplicated to
    `channels`."""
    import av

    codec = {".wav": "pcm_s16le", ".flac": "flac", ".mp3": "libmp3lame", ".m4a": "aac", ".ogg": "vorbis",
             ".opus": "libopus"}[path.suffix]
    layout = "mono" if channels == 1 else "stereo"
    with av.open(str(path), "w") as out:
        stream = out.add_stream(codec, rate=rate, layout=layout)
        if codec == "vorbis":
            stream.codec_context.options = {"strict": "experimental"}
        data = np.tile(samples, (channels, 1)).astype(np.float32)
        frame_size = stream.codec_context.frame_size or 1024
        fmt = stream.codec_context.format.name
        for start in range(0, data.shape[1], frame_size):
            block = data[:, start : start + frame_size]
            frame = av.AudioFrame.from_ndarray(block if fmt.endswith("p") else block.T.reshape(1, -1).copy(),
                                               format="fltp" if fmt.endswith("p") else "flt", layout=layout)
            frame.sample_rate = rate
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


@pytest.mark.parametrize(
    "suffix,rate,channels",
    [(".wav", 16_000, 1), (".wav", 48_000, 2), (".flac", 44_100, 2), (".mp3", 44_100, 2),
     (".m4a", 48_000, 2), (".ogg", 48_000, 2), (".opus", 48_000, 1)],
)
def test_every_common_format_decodes_to_16k_mono(tmp_path, suffix, rate, channels):
    path = tmp_path / f"tone{suffix}"
    encode(path, tone(rate, 1.0), rate, channels)
    decoded = filesource.read_16k_mono(path)
    assert decoded.dtype == np.float32
    # Lossy codecs add a little padding; the length is right to within 100 ms.
    assert abs(len(decoded) - SAMPLE_RATE) < SAMPLE_RATE // 10, len(decoded)
    middle = decoded[SAMPLE_RATE // 4 : 3 * SAMPLE_RATE // 4]
    assert abs(crossings_hz(middle, 0.5) - 1000) < 30
    assert 0.3 < np.abs(middle).max() <= 0.6  # stereo averaged, not summed


def test_a_missing_or_broken_file_is_an_error_naming_the_absolute_path(tmp_path):
    with pytest.raises(filesource.FileSourceError, match=str(tmp_path.absolute()).replace("\\", "\\\\")):
        filesource.read_16k_mono(tmp_path / "absent.wav")
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"not audio at all")
    with pytest.raises(filesource.FileSourceError, match="junk.mp3"):
        filesource.read_16k_mono(junk)
