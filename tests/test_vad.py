"""Port of the tests in Rust `vad.rs`. Rust marks the model tests `#[ignore]`
and takes paths from the environment; here they run whenever
models/vad/silero_vad.onnx and the fixtures (tests/fetch-fixtures.ps1) are
present, and are skipped otherwise."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pyvolis import paths
from pyvolis.audio import SAMPLE_RATE
from pyvolis.config import ConfigError, PyvolisConfig
from pyvolis.filesource import chunks, read_16k_mono
from pyvolis.vad import Segment, Segmenter, VadError, VadSettings

ROOT = paths.app_root()
MODEL = paths.vad_model_file(ROOT)
FIXTURES = ROOT / "tests" / "fixtures" / "fleurs" / "es_419"

needs_model = pytest.mark.skipif(not MODEL.is_file(), reason=f"no VAD model at {MODEL}")
needs_speech = pytest.mark.skipif(
    not MODEL.is_file() or not (FIXTURES / "refs.jsonl").is_file(),
    reason="needs the VAD model and tests/fetch-fixtures.ps1",
)


def settings(pre_roll_ms: int = 600, model: Path = MODEL) -> VadSettings:
    return VadSettings(model=model, threshold=0.5, min_silence_ms=500, min_speech_ms=250, pre_roll_ms=pre_roll_ms)


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    """The Spanish fixtures joined into one long recording with short gaps:
    many segments, some close together, history trimmed as it runs."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from vad_cuts import conversation

    return conversation()[0]


def run(segmenter: Segmenter, audio: np.ndarray, size: int = 700) -> list[Segment]:
    out = []
    for chunk in chunks(audio, size):
        out += segmenter.push(chunk)
    return out


def test_a_missing_model_names_the_absolute_path():
    with pytest.raises(VadError) as e:
        Segmenter(settings(model=Path("models/vad/definitely-not-here.onnx")))
    message = str(e.value)
    first = message.splitlines()[0].removeprefix("VAD model not found: ")
    assert "definitely-not-here.onnx" in message and Path(first).is_absolute()
    assert "never downloads" in message and "http" not in message


def test_segment_timings_are_derived_from_the_sample_rate():
    s = Segment(SAMPLE_RATE, np.zeros(SAMPLE_RATE // 2, np.float32))
    assert (s.start_ms(), s.duration_ms(), s.end_ms()) == (1000, 500, 1500)


@needs_model
def test_silence_produces_no_segments():
    segmenter = Segmenter(settings())
    silence = np.zeros(10 * SAMPLE_RATE, np.float32)
    # A deterministic low-level hiss at about -60 dBFS, as Rust's test.
    noise, hiss = 0.0, np.empty(10 * SAMPLE_RATE, np.float32)
    for i in range(len(hiss)):
        noise = (noise * 7.0 + 0.3) % 1.0
        hiss[i] = (noise - 0.5) * 0.002
    segments = run(segmenter, silence, 1024) + run(segmenter, hiss, 1024) + segmenter.flush()
    assert segments == [], [(s.start_ms(), s.duration_ms()) for s in segments]


@needs_speech
@pytest.mark.parametrize("pre_roll_ms", [0, 300, 600])
def test_pre_roll_splices_exactly_and_never_overlaps(speech, pre_roll_ms):
    segments = run(Segmenter(settings(pre_roll_ms)), speech)
    assert segments, "no speech found"
    previous_end = 0
    for i, s in enumerate(segments):
        end = s.start_sample + len(s.samples)
        assert end <= len(speech), f"segment {i} runs past the end"
        assert np.array_equal(s.samples, speech[s.start_sample : end]), f"segment {i} is not its audio"
        assert s.start_sample >= previous_end, f"segment {i} starts inside the previous one"
        assert 0 <= s.lead_in_ms() <= pre_roll_ms
        previous_end = end


@needs_speech
def test_the_pre_roll_moves_starts_earlier_and_nothing_else(speech):
    off = run(Segmenter(settings(0)), speech)
    on = run(Segmenter(settings(600)), speech)
    assert [s.detected_sample for s in off] == [s.detected_sample for s in on]
    assert [s.start_sample + len(s.samples) for s in off] == [s.start_sample + len(s.samples) for s in on]
    assert all(b.start_sample <= a.start_sample for a, b in zip(off, on))
    assert sum(b.lead_in_ms() for b in on) > 0


@needs_speech
def test_a_reset_keeps_segments_on_the_capture_timeline(speech):
    first = len(speech) // 3
    gated = first + SAMPLE_RATE
    segmenter = Segmenter(settings())
    segments = run(segmenter, speech[:first])
    segmenter.reset(gated)
    segments += run(segmenter, speech[gated:]) + segmenter.flush()
    after = [s for s in segments if s.start_sample >= gated]
    assert after, "no speech after the reset"
    for s in after:
        assert np.array_equal(s.samples, speech[s.start_sample : s.start_sample + len(s.samples)])
    starts = [s.start_sample for s in segments]
    assert starts == sorted(set(starts)), f"timestamps went backwards: {starts}"

    # Stop mid-utterance after the reset: flush() hands back the part spoken,
    # exactly, ending where the audio stopped.
    target = next(s for s in after if s.duration_ms() >= 1500)
    cut = target.start_sample + len(target.samples) - SAMPLE_RATE // 5
    segmenter = Segmenter(settings())
    run(segmenter, speech[:first])
    segmenter.reset(gated)
    tail = run(segmenter, speech[gated:cut]) + segmenter.flush()
    last = tail[-1]
    assert last.start_sample + len(last.samples) == cut
    assert np.array_equal(last.samples, speech[last.start_sample : cut])


@needs_speech
def test_vad_cuts_speech_out_of_a_recording():
    for ref in (FIXTURES / "refs.jsonl").read_text(encoding="utf-8").splitlines()[:3]:
        audio = read_16k_mono(FIXTURES / json.loads(ref)["file"])
        segmenter = Segmenter(settings())
        segments = run(segmenter, audio, 1024) + segmenter.flush()
        assert segments
        total_ms = len(audio) * 1000 // SAMPLE_RATE
        assert sum(s.duration_ms() for s in segments) <= total_ms
        assert all(s.end_ms() <= total_ms + 1 for s in segments)


# ---------------------------------------------------------------- pyvolis.toml


def test_pyvolis_toml_is_optional_and_strict(tmp_path):
    config, found = PyvolisConfig.load(tmp_path / "pyvolis.toml")
    assert not found and config.vad.pre_roll_ms == 600
    (tmp_path / "pyvolis.toml").write_text("[vad]\npre_roll_ms = 0\n", encoding="utf-8")
    assert PyvolisConfig.load(tmp_path / "pyvolis.toml")[0].vad.pre_roll_ms == 0
    (tmp_path / "pyvolis.toml").write_text("[vad]\npre_roll = 300\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="pre_roll"):
        PyvolisConfig.load(tmp_path / "pyvolis.toml")
