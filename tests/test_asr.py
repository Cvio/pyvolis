"""The recognizers, on the fixtures. Like Rust's `#[ignore]` model tests, these
run only when the models and fixtures are present (skipped otherwise). GPU
models are loaded one at a time and released."""

import json
import queue

import numpy as np
import pytest

from pyvolis import asr, models, paths
from pyvolis.config import Config, PyvolisConfig
from pyvolis.filesource import read_16k_mono
from pyvolis.pipeline import ArraySource, Options, Pipeline, run_to_end

ROOT = paths.app_root()
ENGINES = {e.dir_name: e for e in models.discover(paths.asr_dir(ROOT), models.Role.ASR) if isinstance(e, models.Engine)}
FIXTURES = ROOT / "tests" / "fixtures" / "fleurs"


def clip(name: str, i: int = 0):
    folder = FIXTURES / name
    if not (folder / "refs.jsonl").is_file():
        pytest.skip("run tests/fetch-fixtures.ps1")
    ref = json.loads((folder / "refs.jsonl").read_text(encoding="utf-8").splitlines()[i])
    return read_16k_mono(folder / ref["file"]), ref["raw_transcript"]


def engine(name: str) -> models.Engine:
    if name not in ENGINES or not ENGINES[name].enabled():
        pytest.skip(f"model {name} is not installed")
    return ENGINES[name]


# Spanish in, Spanish out, from every kind of backend: no engine may quietly
# translate, and each hears the language it is told.
@pytest.mark.parametrize(
    "name,fixture,language,expect",
    [
        ("parakeet-tdt-0.6b-v3-onnx-int8", "es_419", "es", "viajeros"),
        ("whisper-large-v3-turbo-onnx-int8", "es_419", "es-MX", "viajeros"),
        ("whisper-large-v3-turbo-es", "es_419", "es-MX", "viajeros"),
        ("mms-1b-all", "es_419", "es", "viajeros"),
        ("cohere-transcribe-arabic-07-2026", "ar_eg", "ar", "المحيط"),
        ("whisper-large-v3-turbo-arabic-dialectal-hf", "ar_eg", "ar-IQ", "المحيط"),
    ],
)
def test_each_backend_transcribes_its_language(name, fixture, language, expect):
    audio, _ref = clip(fixture)
    recognizer = asr.load(engine(name))
    try:
        result = recognizer.transcribe(audio, language)
    finally:
        recognizer.close()
    assert expect in result.text, result.text
    assert result.language == language.split("-")[0], "only the language part reaches the model"


def test_whisper_through_transformers_gives_word_timings():
    audio, _ = clip("es_419")
    recognizer = asr.load(engine("whisper-large-v3-turbo-es"))
    try:
        words = recognizer.transcribe(audio, "es").words
    finally:
        recognizer.close()
    assert words and all(a.start <= b.start for a, b in zip(words, words[1:]))
    assert words[-1].end <= len(audio) / 16000 + 0.5


def test_mms_refuses_a_language_it_has_no_adapter_for():
    recognizer = asr.load(engine("mms-1b-all"))
    try:
        with pytest.raises(asr.AsrError, match="adapter"):
            recognizer.transcribe(np.zeros(16000, np.float32), "de")
    finally:
        recognizer.close()


def test_a_disabled_model_is_refused_naming_the_missing_file(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "config.json").write_text('{"model_type": "whisper"}', encoding="utf-8")
    [broken] = models.discover(tmp_path, models.Role.ASR)
    with pytest.raises(asr.AsrError, match="incomplete"):
        asr.load(broken)


def test_the_pipeline_turns_a_file_into_final_transcripts():
    """A fixture through the whole pipeline (VAD, pre-roll, recognizer,
    guards) from a file source, as the microphone would feed it."""
    audio, _ = clip("es_419")
    name = "parakeet-tdt-0.6b-v3-onnx-int8"
    engine(name)
    config = Config.parse(f'[asr]\nengine = "{name}"\n[languages]\nsource = "es"\n')
    events: queue.Queue = queue.Queue()
    gap = np.zeros(16000, np.float32)
    source = ArraySource(np.concatenate([gap, audio, gap]))
    out = run_to_end(Pipeline(ROOT, config, Options(translate=False), events, source, PyvolisConfig()), events)
    assert not out.errors, out.errors
    assert out.finals and "viajeros" in " ".join(f.text for f in out.finals)


def test_silence_and_noise_produce_no_text():
    """P2's guard check: silence, hiss, loud bursts (a door, a knock) and mains
    hum give no text at all."""
    name = "whisper-large-v3-turbo-onnx-int8"
    engine(name)
    sr = 16000
    rng = np.random.default_rng(7)
    parts = [np.zeros(sr * 3, np.float32), (rng.standard_normal(sr * 4) * 0.003).astype(np.float32),
             (0.05 * np.sin(2 * np.pi * 120 * np.arange(sr * 3) / sr)).astype(np.float32)]
    for _ in range(4):
        parts += [(rng.standard_normal(4000) * 0.3).astype(np.float32), np.zeros(sr, np.float32)]
    config = Config.parse(f'[asr]\nengine = "{name}"\n[languages]\nsource = "es"\n')
    events: queue.Queue = queue.Queue()
    out = run_to_end(Pipeline(ROOT, config, Options(translate=False), events, ArraySource(np.concatenate(parts)),
                              PyvolisConfig()), events)
    assert not out.errors, out.errors
    assert out.finals == [], [f.text for f in out.finals]


def test_a_hallucination_is_dropped_and_reported_with_its_reason(monkeypatch, caplog):
    """A recognizer that answers real speech with a stock phrase: the pipeline
    drops it, sends a Dropped event with the text and reason, and logs it."""
    audio, _ = clip("es_419")
    name = "parakeet-tdt-0.6b-v3-onnx-int8"
    engine(name)

    class Invents:
        name = "invents"

        def transcribe(self, pcm, language, timestamps=True):
            return asr.AsrResult("Gracias por ver el video.", language="es")

        def prepare(self, language):
            pass

        def memory(self):
            return asr.Memory()

        def close(self):
            pass

    monkeypatch.setattr(asr, "load", lambda e: Invents())
    config = Config.parse(f'[asr]\nengine = "{name}"\n[languages]\nsource = "es"\n')
    events: queue.Queue = queue.Queue()
    gap = np.zeros(16000, np.float32)
    with caplog.at_level("INFO"):
        out = run_to_end(Pipeline(ROOT, config, Options(translate=False), events, ArraySource(np.concatenate([gap, audio, gap])),
                                  PyvolisConfig()), events)
    assert out.finals == []
    assert out.dropped and out.dropped[0].text == "Gracias por ver el video." and out.dropped[0].reasons
    assert "dropped: the whole text is a stock phrase" in caplog.text
