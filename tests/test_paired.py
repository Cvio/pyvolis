"""Paired mode through the whole pipeline (P9): two pipelines on this PC, over
loopback, with a stand-in microphone, recognizer and translator. The
connection itself is tested in tests/test_peer.py."""

import queue
import sys
import threading
import time

import numpy as np
import pytest

from pyvolis import asr, paths, peer
from pyvolis import events as ev
from pyvolis import translate as tr
from pyvolis.asr import AsrResult
from pyvolis.config import Config, PyvolisConfig
from pyvolis.pipeline import Options, Pipeline

SECOND = 16_000


class FakeMic:
    """Audio from memory, delivered the way a microphone delivers it: live,
    from the moment it is opened."""

    lossless = False
    duration = None

    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio
        self.opened = 0
        self._stop = threading.Event()
        self._thread = None

    def start(self, out: queue.Queue) -> None:
        self.opened += 1
        self._stop.clear()

        def feed() -> None:
            for start in range(0, len(self.audio), 1600):
                if self._stop.is_set():
                    return
                out.put(self.audio[start:start + 1600])
                time.sleep(0.002)

        self._thread = threading.Thread(target=feed, daemon=True)
        self._thread.start()

    def done(self) -> bool:
        return False

    def stop(self) -> None:
        self._stop.set()


class Says:
    def __init__(self, text: str) -> None:
        self.text, self.name = text, "says"

    def transcribe(self, audio, language, timestamps=True):
        return AsrResult(self.text, None, None, language, 0.01)

    def prepare(self, language):
        pass

    def memory(self):
        return asr.Memory()

    def close(self):
        pass


class Upper:
    """A translator that shouts the source back, so the test can tell a
    translation from its source."""

    device, name = "cpu", "upper"

    def __init__(self) -> None:
        self.requests = []

    def translate(self, request):
        self.requests.append(request)
        return tr.TranslationResult(f"[{request.target}] {request.text.upper()}", "", 0.01, "cpu")

    def count_tokens(self, text):
        return len(text.split())

    def close(self):
        pass


class Running:
    def __init__(self, monkeypatch, name, speaks, sends, port, audio, text, context="revision"):
        config = Config.parse(
            f'[asr]\nengine = "parakeet-tdt-0.6b-v3-int8"\n[languages]\nsource = "{speaks}"\ntarget = "{sends}"\n'
            f'[mode]\nkind = "turn"\n[tts]\nenabled = false\n'
            f'[peer]\nenabled = true\nlisten_addr = "127.0.0.1:{port}"\ndisplay_name = "{name}"\ndiscovery = false\n')
        self.events: queue.Queue = queue.Queue()
        self.seen: list = []
        self.mic = FakeMic(audio)
        self.translator = Upper()
        pyconfig = PyvolisConfig()
        pyconfig.context.mode = context
        self.pipeline = Pipeline(paths.app_root(), config, Options(), self.events, self.mic, pyconfig)
        self.pipeline._recognizer = lambda engines: Says(text)
        self.pipeline._translator = lambda: self.translator
        self.pipeline.start()

    def wait_for(self, what: str, want, seconds: float = 20):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                event = self.events.get(timeout=0.05)
            except queue.Empty:
                continue
            self.seen.append(event)
            if want(event):
                return event
        pytest.fail(f"never saw {what}; saw {[type(e).__name__ for e in self.seen]}")

    def stop(self) -> None:
        self.pipeline.stop()
        self.pipeline.join(10)


@pytest.fixture
def speech():
    sys.path.insert(0, str(paths.app_root() / "scripts"))
    try:
        from vad_cuts import conversation

        return conversation()[0][: 8 * SECOND]
    except SystemExit:
        pytest.skip("needs tests/fetch-fixtures.ps1")


def is_a(cls, **fields):
    return lambda e: isinstance(e, cls) and all(getattr(e, k) == v for k, v in fields.items())


def test_a_turn_reaches_the_other_pc_sentence_by_sentence_then_the_floor_goes_back(monkeypatch, speech):
    a = Running(monkeypatch, "laptop-a", "es", "en", 47971, speech, "hola amigo. ¿dónde está la estación?")
    b = Running(monkeypatch, "laptop-b", "en", "es", 47972, speech, "hello.")
    try:
        a.wait_for("a listening", is_a(ev.Listening))
        b.wait_for("b listening", is_a(ev.Listening))
        assert a.mic.opened == 0, "taking turns: the microphone is closed until the floor is granted"
        a.pipeline.connect(peer.parse_address("127.0.0.1:47972"))
        a.wait_for("a connected", is_a(ev.PeerMsg, kind="connected"))
        b.wait_for("b connected", is_a(ev.PeerMsg, kind="connected"))

        a.pipeline.begin_turn()
        a.wait_for("a's turn", is_a(ev.TurnStarted))
        assert a.mic.opened == 1
        b.wait_for("b sees a has the floor", is_a(ev.FloorChanged, holder="them", name="laptop-a"))
        # B presses its key meanwhile: refused, and its microphone never opens.
        b.pipeline.begin_turn()
        b.wait_for("b refused", is_a(ev.FloorRefused))
        time.sleep(1.0)
        assert b.mic.opened == 0, "b's microphone opened while a held the floor"

        a.pipeline.end_turn()
        first = a.wait_for("first sentence sent", is_a(ev.Sent))
        second = a.wait_for("second sentence sent", is_a(ev.Sent))
        assert (first.id, second.id, first.to) == ("1.1", "1.2", "laptop-b")
        a.wait_for("a's floor released", is_a(ev.FloorChanged, holder="free"))

        b.wait_for("b's floor free", is_a(ev.FloorChanged, holder="free"))
        remotes = [e for e in b.seen if isinstance(e, ev.Remote)]
        assert [(r.lang, r.text, r.source_lang, r.source_text) for r in remotes] == [
            ("en", "[en] HOLA AMIGO.", "es", "hola amigo."),
            ("en", "[en] ¿DÓNDE ESTÁ LA ESTACIÓN?", "es", "¿dónde está la estación?")]
        kinds = [type(e).__name__ for e in b.seen]
        assert kinds.index("Remote") < len(kinds) - 1 - kinds[::-1].index("FloorChanged"), \
            "the floor came back after the sentences, not before"
        # Nothing but the final translation crossed: B translated nothing, and
        # context stayed on A (the second sentence was translated knowing the first).
        assert b.translator.requests == []
        assert [t.source for t in a.translator.requests[1].context] == ["hola amigo."]
        assert not any(isinstance(e, ev.Revised) for e in a.seen), "revision is off while paired"

        # The link dies. B is told, by name, and its floor is free.
        a.stop()
        gone = b.wait_for("b disconnected", is_a(ev.PeerMsg, kind="disconnected"))
        assert "laptop-a" in gone.reason
    finally:
        a.stop()
        b.stop()


def test_a_turn_with_no_speech_hands_the_floor_straight_back(monkeypatch):
    silence = np.zeros(3 * SECOND, np.float32)
    a = Running(monkeypatch, "laptop-a", "es", "en", 47973, silence, "")
    b = Running(monkeypatch, "laptop-b", "en", "es", 47974, silence, "")
    try:
        a.wait_for("a listening", is_a(ev.Listening))
        b.wait_for("b listening", is_a(ev.Listening))
        a.pipeline.connect(peer.parse_address("127.0.0.1:47974"))
        a.wait_for("a connected", is_a(ev.PeerMsg, kind="connected"))
        a.pipeline.begin_turn()
        a.wait_for("a's turn", is_a(ev.TurnStarted))
        time.sleep(0.5)
        a.pipeline.end_turn()
        a.wait_for("nothing recognised", is_a(ev.NothingRecognized))
        a.wait_for("a's floor released", is_a(ev.FloorChanged, holder="free"))
        b.wait_for("b sees a's turn", is_a(ev.FloorChanged, holder="them"))
        b.wait_for("b's floor free", is_a(ev.FloorChanged, holder="free"))
    finally:
        a.stop()
        b.stop()


def test_a_killed_link_mid_turn_closes_the_microphone_and_says_why(monkeypatch, speech):
    a = Running(monkeypatch, "laptop-a", "es", "en", 47975, speech, "hola amigo, ¿cómo estás hoy?")
    b = Running(monkeypatch, "laptop-b", "en", "es", 47976, speech, "hello.")
    try:
        a.wait_for("a listening", is_a(ev.Listening))
        b.wait_for("b listening", is_a(ev.Listening))
        a.pipeline.connect(peer.parse_address("127.0.0.1:47976"))
        a.wait_for("a connected", is_a(ev.PeerMsg, kind="connected"))
        a.pipeline.begin_turn()
        a.wait_for("a's turn", is_a(ev.TurnStarted))
        b.stop()
        gone = a.wait_for("a disconnected", is_a(ev.PeerMsg, kind="disconnected"))
        assert "laptop-b" in gone.reason
        a.wait_for("a's turn ended", is_a(ev.TurnEnded))
        after = a.seen[a.seen.index(gone):]
        assert any(is_a(ev.FloorChanged, holder="free")(e) for e in after), "the floor was force-released"
        assert a.mic.opened == 1 and a.mic._stop.is_set(), "the microphone closed with the link"
        not_sent = a.wait_for("the sentence was not sent", is_a(ev.NotSent))
        assert "not connected" in not_sent.reason
    finally:
        a.stop()
        b.stop()


def test_a_file_and_a_comparison_never_pair():
    config = Config.parse('[peer]\nenabled = true\n')
    from pyvolis.pipeline import ArraySource

    assert not Pipeline(paths.app_root(), config, Options(), queue.Queue(), ArraySource(np.zeros(16)), PyvolisConfig()).paired
    assert not Pipeline(paths.app_root(), config, Options(compare=True), queue.Queue(), FakeMic(np.zeros(16)),
                        PyvolisConfig()).paired
    assert Pipeline(paths.app_root(), config, Options(), queue.Queue(), FakeMic(np.zeros(16)), PyvolisConfig()).paired
    assert not Pipeline(paths.app_root(), config, Options(pair=False), queue.Queue(), FakeMic(np.zeros(16)),
                        PyvolisConfig()).paired
