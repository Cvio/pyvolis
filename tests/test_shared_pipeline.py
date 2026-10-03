"""The shared machine through the whole pipeline (P10): a stand-in
microphone, translator and recognizers, the real threads. The direction rules
are tested in tests/test_shared.py."""

import queue
import threading
import time

import numpy as np
import pytest
from test_paired import FakeMic, Running, Upper, is_a, speech  # noqa: F401 - speech is a fixture

from pyvolis import asr, models, paths
from pyvolis import events as ev
from pyvolis import shared as sh
from pyvolis.asr import AsrResult
from pyvolis.config import Config, PyvolisConfig
from pyvolis.pipeline import Options, Pipeline

LEFT_ASR, RIGHT_ASR = "parakeet-tdt-0.6b-v3-onnx-int8", "whisper-large-v3-turbo-onnx-int8"


class Hears:
    """A recognizer that answers with a fixed text and records what it was asked."""

    def __init__(self, folder: str, heard: list, text: dict) -> None:
        self.name, self.folder, self.heard, self.text = folder, folder, heard, text

    def transcribe(self, audio, language, timestamps=True):
        self.heard.append((self.folder, language))
        return AsrResult(self.text[self.folder], None, None, language, 0.01)

    def prepare(self, language):
        self.heard.append((self.folder, "prepare", language))

    def memory(self):
        return asr.Memory()

    def close(self):
        pass


def shared_run(monkeypatch, audio, texts, right_asr=RIGHT_ASR, right_language="es-MX"):
    folder = paths.asr_dir(paths.app_root())
    if not all((folder / name / "engine.toml").is_file() for name in (LEFT_ASR, RIGHT_ASR)):
        pytest.skip("needs the development models")
    heard: list = []
    monkeypatch.setattr(asr, "load", lambda engine: Hears(engine.dir_name, heard, texts))
    config = Config.parse(
        f'[asr]\nengine = "{LEFT_ASR}"\n[mode]\nkind = "shared"\n[tts]\nenabled = false\n'
        f'[shared]\nleft_language = "en"\nright_language = "{right_language}"\n'
        f'left_asr = "{LEFT_ASR}"\nright_asr = "{right_asr}"\n')
    run = Running.__new__(Running)
    run.events, run.seen, run.mic, run.translator = queue.Queue(), [], FakeMic(audio), Upper()
    run.pipeline = Pipeline(paths.app_root(), config, Options(), run.events, run.mic, PyvolisConfig())
    run.pipeline._translator = lambda: run.translator
    run.pipeline.start()
    engines = [e for e in models.discover(folder, models.Role.ASR) if isinstance(e, models.Engine)]

    def direction(side):
        return sh.Direction(side, sh.recognizer_for(side, config.shared, engines, LEFT_ASR).dir_name,
                            sh.language(side, config.shared), sh.language(sh.other(side), config.shared), "")

    return run, heard, direction


def test_each_side_is_heard_by_its_own_recognizer_in_its_own_language(monkeypatch, speech):  # noqa: F811
    run, heard, direction = shared_run(monkeypatch, speech,
                                       {LEFT_ASR: "where is the station?", RIGHT_ASR: "está muy cerca de aquí, amigo."})
    try:
        ready = [run.wait_for("a side ready", is_a(ev.SharedSide)) for _ in range(2)]
        assert [(e.side, e.problem) for e in ready] == [("left", ""), ("right", "")]
        run.wait_for("listening", is_a(ev.Listening))
        assert run.wait_for("mode", is_a(ev.Mode)).kind == "shared"
        assert run.mic.opened == 0, "the microphone is closed until a key is pressed"

        for side, said, translated, lang in (("left", "where is the station?", "[es-MX] WHERE IS THE STATION?", "es-MX"),
                                             ("right", "está muy cerca de aquí, amigo.", "[en] ESTÁ MUY CERCA DE AQUÍ, AMIGO.", "en")):
            run.pipeline.begin_shared_turn(direction(side))
            assert run.wait_for("turn", is_a(ev.TurnStarted)).side == side
            time.sleep(0.6)
            run.pipeline.end_turn()
            sentence = run.wait_for("sentence", is_a(ev.SentenceMsg))
            done = run.wait_for("translation", is_a(ev.Translated))
            assert (sentence.text, sentence.lang, done.text, done.lang) == (said, direction(side).source, translated, lang)

        transcribed = [h for h in heard if len(h) == 2]
        assert transcribed == [(LEFT_ASR, "en"), (RIGHT_ASR, "es-MX")], "the key decides the recognizer and the language"
        assert (LEFT_ASR, "prepare", "en") in heard and (RIGHT_ASR, "prepare", "es-MX") in heard
        first, second = run.translator.requests
        assert (first.source, first.target, second.source, second.target) == ("en", "es-MX", "es-MX", "en")
        # The second person is translated knowing what the first said, turned
        # round so that it is a pair in their direction.
        assert [(t.source, t.translation) for t in second.context] == [
            ("[es-MX] WHERE IS THE STATION?", "where is the station?")]
        assert run.mic.opened == 2 and run.mic._stop.is_set()
    finally:
        run.stop()


def test_escape_during_a_turn_means_nothing_is_recognised_translated_or_spoken(monkeypatch, speech):  # noqa: F811
    run, heard, direction = shared_run(monkeypatch, speech, {LEFT_ASR: "hello there, friend.", RIGHT_ASR: "hola amigo, ¿cómo estás hoy?"})
    try:
        run.wait_for("listening", is_a(ev.Listening))
        run.pipeline.begin_shared_turn(direction("left"))
        run.wait_for("turn", is_a(ev.TurnStarted))
        time.sleep(0.4)
        run.pipeline.cancel()
        run.wait_for("cancelled", is_a(ev.TurnCancelled))
        assert run.mic._stop.is_set(), "the microphone closed"
        # The other person can go at once, and only their turn produces anything.
        run.pipeline.begin_shared_turn(direction("right"))
        assert run.wait_for("turn", is_a(ev.TurnStarted)).side == "right"
        time.sleep(0.6)
        run.pipeline.end_turn()
        run.wait_for("translation", is_a(ev.Translated))
        assert [h for h in heard if len(h) == 2] == [(RIGHT_ASR, "es-MX")]
        assert len(run.translator.requests) == 1
    finally:
        run.stop()


def test_a_cancel_drops_what_is_already_on_its_way(monkeypatch, speech):  # noqa: F811
    run, heard, direction = shared_run(monkeypatch, speech, {LEFT_ASR: "hello there, friend.", RIGHT_ASR: "hola amigo, ¿cómo estás hoy?"})
    gate = threading.Event()
    original = run.translator.translate

    def slow(request):
        gate.wait(5)  # the translation is in flight when Escape is pressed
        return original(request)

    run.translator.translate = slow
    try:
        run.wait_for("listening", is_a(ev.Listening))
        run.pipeline.begin_shared_turn(direction("left"))
        run.wait_for("turn", is_a(ev.TurnStarted))
        time.sleep(0.6)
        run.pipeline.end_turn()
        run.wait_for("sentence", is_a(ev.SentenceMsg))
        run.pipeline.cancel()
        run.wait_for("cancelled", is_a(ev.TurnCancelled))
        gate.set()
        time.sleep(0.5)
        while not run.events.empty():
            run.seen.append(run.events.get())
        assert not any(isinstance(e, ev.Translated) for e in run.seen), "a cancelled turn's translation was shown"
    finally:
        gate.set()
        run.stop()


def test_a_side_whose_recognizer_is_missing_says_so_and_the_other_keeps_working(monkeypatch, speech):  # noqa: F811
    run, heard, direction = shared_run(monkeypatch, speech, {LEFT_ASR: "hello there, friend."},
                                       right_asr="deleted-model", right_language="es")
    try:
        sides = {e.side: e.problem for e in (run.wait_for("a side", is_a(ev.SharedSide)) for _ in range(2))}
        assert sides["left"] == "" and "deleted-model" in sides["right"]
        run.wait_for("listening", is_a(ev.Listening))
        run.pipeline.begin_shared_turn(sh.Direction("left", LEFT_ASR, "en", "es", ""))
        run.wait_for("turn", is_a(ev.TurnStarted))
        time.sleep(0.6)
        run.pipeline.end_turn()
        assert run.wait_for("translation", is_a(ev.Translated)).lang == "es"
    finally:
        run.stop()


def test_one_model_chosen_for_both_sides_loads_once(monkeypatch, speech):  # noqa: F811
    loads = []
    run, heard, direction = shared_run(monkeypatch, speech, {RIGHT_ASR: "hola amigo, ¿cómo estás hoy?"}, right_asr=RIGHT_ASR)
    run.stop()
    monkeypatch.setattr(asr, "load", lambda engine: loads.append(engine.dir_name) or Hears(engine.dir_name, [], {}))
    config = Config.parse(
        f'[asr]\nengine = "{RIGHT_ASR}"\n[mode]\nkind = "shared"\n[tts]\nenabled = false\n'
        f'[shared]\nleft_language = "en"\nright_language = "es"\nleft_asr = "{RIGHT_ASR}"\nright_asr = "{RIGHT_ASR}"\n')
    events: queue.Queue = queue.Queue()
    pipeline = Pipeline(paths.app_root(), config, Options(), events, FakeMic(speech), PyvolisConfig())
    pipeline._translator = Upper
    pipeline.start()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not isinstance(events.get(timeout=20), ev.Listening):
            pass
        assert loads == [RIGHT_ASR]
    finally:
        pipeline.stop()
        pipeline.join(10)


def test_a_shared_machine_never_pairs():
    config = Config.parse('[peer]\nenabled = true\n[mode]\nkind = "shared"\n')
    assert not Pipeline(paths.app_root(), config, Options(), queue.Queue(), FakeMic(np.zeros(16)), PyvolisConfig()).paired
