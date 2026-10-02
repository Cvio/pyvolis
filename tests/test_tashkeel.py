"""Arabic vowel marks before the voice ([tts] diacritize): the port of
piper-phonemize's tashkeel.cpp, and where the speaking thread applies it."""

import queue
from types import SimpleNamespace

import numpy as np
import pytest

from pyvolis import paths, tashkeel
from pyvolis.config import PyvolisConfig
from pyvolis.pipeline import SpeakThread

MODEL = paths.tashkeel_model_file(paths.app_root())
needs_model = pytest.mark.skipif(not MODEL.is_file(), reason=f"no {MODEL} (.\\fetch-models.ps1 -Only tashkeel)")
FATHA, DAMMA, SHADDA = "َ", "ُ", "ّ"


def test_marks_already_in_the_text_are_removed_first():
    assert tashkeel.strip("مَرْحَبًا بِكُمْ") == "مرحبا بكم"
    assert not set(tashkeel.strip("أَيْنَ مَحَطَّةُ")) & tashkeel.HARAKAT


def test_each_character_is_followed_by_its_predicted_mark():
    # 5 = fatha, 17 = damma, 7 = fatha + shadda; 0 (padding), 1 (unknown) and 8 add nothing
    assert tashkeel.mark("abcdef", [5, 17, 7, 0, 1, 8]) == f"a{FATHA}b{DAMMA}c{FATHA}{SHADDA}def"


def test_text_past_the_predictions_is_kept_unmarked():
    assert tashkeel.mark("abc", [5]) == f"a{FATHA}bc"


def test_long_text_is_cut_at_spaces_and_nothing_is_lost():
    text = " ".join(["كلمة"] * 200)
    parts = tashkeel.pieces(text)
    assert "".join(parts) == text
    assert all(len(p) <= tashkeel.MAX_INPUT_CHARS for p in parts) and len(parts) > 1
    assert all(p.endswith(" ") for p in parts[:-1])
    assert tashkeel.pieces("x" * 700, 315) == ["x" * 315, "x" * 315, "x" * 70], "no space to cut at"
    assert tashkeel.pieces("") == []


def test_a_missing_model_is_named_by_its_absolute_path(tmp_path):
    with pytest.raises(tashkeel.TashkeelError) as e:
        tashkeel.Tashkeel(tmp_path / "libtashkeel_model.ort")
    assert str((tmp_path / "libtashkeel_model.ort").absolute()) in str(e.value)


@needs_model
def test_the_model_marks_arabic_and_leaves_the_letters_alone():
    marker = tashkeel.Tashkeel(MODEL)
    text = "أريد أن أذهب إلى المطار غدا."
    marked = marker.run(text)
    assert tashkeel.strip(marked) == text, "only marks are added"
    assert len(set(marked) & tashkeel.HARAKAT) >= 3
    assert marker.run(marked) == marked, "marks are predicted again, not doubled"
    long = " ".join([text] * 20)
    assert tashkeel.strip(marker.run(long)) == long, "past 315 characters too"


# ---------------------------------------------------------------- the speaking thread


class Voice:
    name = "fake"

    def __init__(self, said: list) -> None:
        self.said = said

    def speak(self, text: str):
        self.said.append(text)
        return SimpleNamespace(samples=np.zeros(160, np.float32), sample_rate=16_000, duration_ms=lambda: 10)


class Marker:
    def run(self, text: str) -> str:
        return f"<{text}>"


def speaker(monkeypatch, tmp_path, diacritize: bool, marker=Marker()):
    py = PyvolisConfig()
    py.tts.diacritize = diacritize
    seen: queue.Queue = queue.Queue()
    pipeline = SimpleNamespace(root=tmp_path, pyconfig=py, emit=seen.put, generation=0)
    player = SimpleNamespace(play=lambda samples, rate: None, queued=lambda: 0, close=lambda: None)
    thread = SpeakThread(pipeline, player)
    said: list[str] = []
    monkeypatch.setattr(thread, "_voice", lambda language, folder="": Voice(said))
    thread._tashkeel = marker
    return thread, said, seen


def test_only_arabic_is_marked_and_only_when_the_setting_is_on(monkeypatch, tmp_path):
    thread, said, _ = speaker(monkeypatch, tmp_path, True)
    thread.submit("1.1", "مرحبا", "ar", 0.0)
    thread.submit("2.1", "مرحبا", "ar-IQ", 0.0)
    thread.submit("3.1", "سلام", "fa", 0.0)
    thread.submit("4.1", "Hello", "en", 0.0)
    thread.finish(wait=True)
    assert said == ["<مرحبا>", "<مرحبا>", "سلام", "Hello"]

    thread, said, _ = speaker(monkeypatch, tmp_path, False)
    thread.submit("1.1", "مرحبا", "ar", 0.0)
    thread.finish(wait=True)
    assert said == ["مرحبا"]


def test_without_the_model_arabic_is_still_spoken_and_the_reason_is_given_once(monkeypatch, tmp_path):
    thread, said, seen = speaker(monkeypatch, tmp_path, True, marker=None)
    thread.submit("1.1", "مرحبا", "ar", 0.0)
    thread.submit("2.1", "مرحبا", "ar", 0.0)
    thread.finish(wait=True)
    assert said == ["مرحبا", "مرحبا"]
    errors = [e.message for e in list(seen.queue) if type(e).__name__ == "Error"]
    assert len(errors) == 1 and "without vowel marks" in errors[0]
    assert str(paths.tashkeel_model_file(tmp_path).absolute()) in errors[0]


def test_off_by_default():
    assert PyvolisConfig().tts.diacritize is False
