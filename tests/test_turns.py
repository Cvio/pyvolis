"""Turn-based mode (P6): port of Rust's turn tests from `gui.rs` (the turn
state and the turn key) and `pipeline.rs` (trimming and splitting a turn).
`scripts/turn_check.py` is the check against real devices."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from pyvolis import events as ev
from pyvolis.gui import session as ses
from pyvolis.gui.session import KeyEdges, Session, TurnKey, turn_key_action
from pyvolis.pipeline import MAX_PART, transcribe_parts, trim_and_split
from pyvolis.vad import Segment


def turn_session(speaks: bool = True) -> Session:
    s = Session()
    s.begin("es", "en", False, speaks)
    s.apply(ev.Listening())
    s.apply(ev.Mode("turn"))
    return s


def sentence(s: Session, sid: str, text: str = "Hola.") -> None:
    utterance = int(sid.split(".")[0])
    s.apply(ev.Final(utterance, text, "es", 1500, 120, 0.0, 1.5))
    s.apply(ev.SentenceMsg(sid, utterance, text, "es", 0.0, 1.5))


# ---------------------------------------------------------------- the turn's state


def test_a_turn_goes_ready_recording_processing_ready():
    s = turn_session()
    assert s.turn == ses.IDLE
    s.apply(ev.TurnStarted())
    assert s.turn == ses.RECORDING
    s.apply(ev.Level(-20.0))
    s.apply(ev.TurnEnded())
    assert s.turn == ses.PROCESSING
    assert s.level_db is None, "the microphone is closed"
    sentence(s, "1.1")
    s.apply(ev.Translated("1.1", "Hello.", "en", 300, "cpu", "qwen"))
    assert s.turn == ses.PROCESSING, "the reply has not been spoken yet"
    s.apply(ev.SpeakingStarted("1.1", 900))
    assert s.turn == ses.PROCESSING, "it is being spoken"
    s.apply(ev.SpeakingEnded())
    assert s.turn == ses.IDLE


def test_a_silent_turn_or_a_failure_ends_processing():
    for ending in (ev.NothingRecognized(1, 2000), ev.Dropped(1, "Gracias.", ["a stock phrase"]),
                   ev.Error("synthesis failed")):
        s = turn_session()
        s.apply(ev.TurnStarted())
        s.apply(ev.TurnEnded())
        s.apply(ending)
        assert s.turn == ses.IDLE, ending


def test_a_refused_translation_ends_the_turn():
    s = turn_session()
    s.apply(ev.TurnStarted())
    s.apply(ev.TurnEnded())
    sentence(s, "1.1", "Me scables.")
    assert s.turn == ses.PROCESSING
    s.apply(ev.NotTranslated("1.1", "echo", "echo"))
    assert s.turn == ses.IDLE


def test_without_speech_a_translation_ends_the_turn():
    s = turn_session(speaks=False)
    s.apply(ev.TurnStarted())
    s.apply(ev.TurnEnded())
    sentence(s, "1.1")
    s.apply(ev.Translated("1.1", "Hello.", "en", 300, "cpu", "qwen"))
    assert s.turn == ses.IDLE


def test_a_turn_of_several_sentences_ends_after_the_last_is_spoken():
    """pyvolis: a turn can hold several sentences. The voice going quiet
    between two of them must not end the turn."""
    s = turn_session()
    s.apply(ev.TurnStarted())
    s.apply(ev.TurnEnded())
    s.apply(ev.Final(1, "Yo manejo. Mi carro está aquí.", "es", 3000, 200, 0.0, 3.0))
    s.apply(ev.SentenceMsg("1.1", 1, "Yo manejo.", "es", 0.0, 1.0))
    s.apply(ev.SentenceMsg("1.2", 1, "Mi carro está aquí.", "es", 1.0, 3.0))
    s.apply(ev.Translated("1.1", "I'll drive.", "en", 300, "cpu", "qwen"))
    s.apply(ev.SpeakingStarted("1.1", 900))
    s.apply(ev.SpeakingEnded())
    assert s.turn == ses.PROCESSING, "the second sentence is still being translated"
    s.apply(ev.Translated("1.2", "My car is here.", "en", 400, "cpu", "qwen"))
    assert s.turn == ses.PROCESSING, "and has yet to be spoken"
    s.apply(ev.SpeakingStarted("1.2", 2100))
    s.apply(ev.SpeakingEnded())
    assert s.turn == ses.IDLE


def test_a_cancelled_turn_and_a_mode_change_return_to_ready():
    s = turn_session()
    s.apply(ev.TurnStarted())
    s.apply(ev.TurnCancelled())
    assert s.turn == ses.IDLE
    s.apply(ev.TurnStarted())
    s.apply(ev.Mode("continuous"))
    assert (s.mode, s.turn) == ("continuous", ses.IDLE)
    s.apply(ev.Stopped())
    assert s.mode is None


def test_the_indicator_tells_the_three_states_apart():
    s = turn_session()
    assert ses.indicator(s, "Space", False)[0] == "READY - press Space to talk"
    assert ses.indicator(s, "Space", True)[0] == "READY - hold Space to talk"
    s.apply(ev.TurnStarted())
    recording = ses.indicator(s, "Space", False)
    assert recording[0] == "RECORDING - press Space to finish"
    assert ses.indicator(s, "Space", True)[0] == "RECORDING - release Space to finish"
    s.apply(ev.TurnEnded())
    processing = ses.indicator(s, "Space", False)
    assert processing[0] == "PROCESSING..."
    s.speaking = True
    assert ses.indicator(s, "Space", False)[0] == "SPEAKING"
    idle = ses.indicator(turn_session(), "Space", False)
    assert len({idle[1], recording[1], processing[1]}) == 3, "three states, three colours"
    s.apply(ev.Mode("continuous"))
    s.speaking = False
    assert ses.indicator(s, "Space", False)[0] == "LISTENING"


# ---------------------------------------------------------------- the turn key


def test_holding_the_key_is_one_press_however_the_repeats_arrive():
    key = TurnKey()
    assert key.update(KeyEdges(pressed=True), True).pressed, "the first press counts"
    for _ in range(5):
        assert not key.update(KeyEdges(pressed=True), True).pressed, "a repeat is not a new press"
    up = key.update(KeyEdges(released=True), True)
    assert up.released
    assert key.update(KeyEdges(pressed=True), True).pressed, "after a release, a press counts again"


def test_losing_focus_with_the_key_down_releases_it():
    key = TurnKey()
    key.update(KeyEdges(pressed=True), True)
    edges = key.update(KeyEdges(), False)
    assert edges.released, "the release would never arrive"
    assert not key.update(KeyEdges(), False).released, "and only once"


def test_a_tap_is_a_press_and_a_release():
    key = TurnKey()
    edges = key.update(KeyEdges(pressed=True, released=True), True)
    assert edges.pressed and edges.released
    assert key.update(KeyEdges(pressed=True), True).pressed, "the key is up again afterwards"


def test_toggle_style_starts_on_a_press_and_stops_on_the_next():
    assert turn_key_action("toggle", ses.IDLE, KeyEdges(pressed=True), False, True) == ("begin", False)
    assert turn_key_action("toggle", ses.RECORDING, KeyEdges(pressed=True), False, True) == ("end", False)
    assert turn_key_action("toggle", ses.RECORDING, KeyEdges(released=True), False, True) == (None, False)


def test_hold_style_records_only_while_the_key_is_held():
    assert turn_key_action("hold", ses.IDLE, KeyEdges(pressed=True), False, True) == ("begin", True)
    assert turn_key_action("hold", ses.RECORDING, KeyEdges(), True, True) == (None, True)
    assert turn_key_action("hold", ses.RECORDING, KeyEdges(released=True), True, True) == ("end", False)
    # Losing focus with the key down: the microphone must not stay open.
    assert turn_key_action("hold", ses.RECORDING, KeyEdges(), True, False) == ("end", False)


# ---------------------------------------------------------------- trimming and splitting


def segment_at(start: int, length: int) -> Segment:
    return Segment(start, np.zeros(length, np.float32))


def test_a_turn_is_trimmed_to_its_speech():
    origin = 5_000
    segments = [segment_at(origin + 1_000, 2_000), segment_at(origin + 10_000, 2_000)]
    speech, parts = trim_and_split(origin, 20_000, segments, MAX_PART)
    assert speech == (1_000, 12_000), "silence at either end is trimmed"
    assert parts == [(0, 11_000)], "the pause inside is kept: the user decided the boundary"


def test_a_long_turn_is_split_at_its_pauses():
    segments = [segment_at(1_000, 2_000), segment_at(4_000, 2_000), segment_at(10_000, 2_000)]
    speech, parts = trim_and_split(0, 20_000, segments, 6_000)
    assert speech == (1_000, 12_000)
    assert parts == [(0, 5_000), (9_000, 11_000)]
    assert all(b - a <= 6_000 for a, b in parts)


def test_a_turn_with_no_speech_has_nothing_to_send():
    assert trim_and_split(0, 20_000, [], MAX_PART) is None


def test_a_segment_running_past_the_turn_is_clipped_to_it():
    speech, parts = trim_and_split(0, 5_000, [segment_at(3_000, 4_000)], MAX_PART)
    assert speech == (3_000, 5_000) and parts == [(0, 2_000)]


def test_a_split_turn_is_transcribed_part_by_part_and_joined():
    from pyvolis.asr import AsrResult, Word

    class Counts:
        def transcribe(self, pcm, language, timestamps=True):
            return AsrResult(f"{len(pcm)} muestras.", [Word("x", 0.0, 0.5)], None, "es", 0.1)

    pcm = np.zeros(48_000, np.float32)
    result = transcribe_parts(Counts(), pcm, [(0, 16_000), (32_000, 48_000)], "es-MX")
    assert result.text == "16000 muestras. 16000 muestras."
    assert [w.start for w in result.words] == [0.0, 2.0], "word times move onto the utterance's timeline"
    assert transcribe_parts(Counts(), pcm, None, "es").text == "48000 muestras."


# ---------------------------------------------------------------- the key and the window


def test_the_turn_key_never_reaches_a_focused_widget():
    """SPEC §8: Space must not also press whatever button has focus."""
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from pyvolis import paths
    from pyvolis.config import Config, PyvolisConfig
    from pyvolis.gui.window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow(paths.app_root(), Config(), PyvolisConfig())
    window.save = lambda: None
    window.show()
    window.activateWindow()
    app.processEvents()
    clicks, sent = [], []
    window.start_button.clicked.disconnect()
    window.start_button.clicked.connect(lambda: clicks.append(1))

    class FakePipeline:
        def begin_turn(self):
            sent.append("begin")

        def end_turn(self):
            sent.append("end")

    try:
        # Not in a turn-mode run: Space is an ordinary key and clicks the focused button.
        window.start_button.setFocus()
        QTest.keyClick(window.start_button, Qt.Key.Key_Space)
        assert clicks == [1], "outside a turn-mode run the key belongs to the widgets"

        # A live run, taking turns: the key is pyvolis's.
        window.pipeline, window.source = FakePipeline(), None
        window.isActiveWindow = lambda: True  # offscreen windows are never "active"
        window.session.apply(ev.Listening())
        window.session.apply(ev.Mode("turn"))
        window.config.mode.turn_style = "toggle"
        QTest.keyClick(window.start_button, Qt.Key.Key_Space)
        assert clicks == [1], "Space pressed the focused button"
        assert sent == ["begin"]
        window.session.apply(ev.TurnStarted())
        QTest.keyClick(window.start_button, Qt.Key.Key_Space)
        assert sent == ["begin", "end"] and clicks == [1]
        # Other keys are left for the widgets.
        QTest.keyClick(window.start_button, Qt.Key.Key_Return)
        assert sent == ["begin", "end"]
    finally:
        window.pipeline = None
        window.close()
