"""The shared machine in the window (P10): the two columns, the pickers, the
two keys and Escape. Offscreen; the pipeline is a stand-in that records what
it was asked."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from pyvolis import events as ev  # noqa: E402
from pyvolis import paths  # noqa: E402
from pyvolis.config import Config, PyvolisConfig  # noqa: E402
from pyvolis.gui.window import MainWindow  # noqa: E402


class FakePipeline:
    def __init__(self) -> None:
        self.asked: list = []

    def begin_shared_turn(self, direction):
        self.asked.append(("begin", direction))

    def end_turn(self):
        self.asked.append(("end",))

    def cancel(self):
        self.asked.append(("cancel",))

    def prepare_shared(self, settings):
        self.asked.append(("prepare", settings.left_language, settings.right_language, settings.left_asr,
                           settings.right_asr))

    def set_mode(self, mode):
        self.asked.append(("mode", mode))

    def stop(self):
        pass

    def join(self, timeout=None):
        pass


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    config = Config.parse('[mode]\nkind = "shared"\n[shared]\nleft_language = "en"\nright_language = "es-MX"\n')
    w = MainWindow(paths.app_root(), config, PyvolisConfig())
    if not w.engines or not w.voices:
        w.close()
        pytest.skip("needs the development models")
    w.save = lambda: None  # a test never touches the user's settings
    w.isActiveWindow = lambda: True  # offscreen, no window is ever the active one
    yield w
    w.pipeline = None
    w.close()


def running(window) -> FakePipeline:
    window.pipeline = FakePipeline()
    window.session.begin("en", "es-MX", False, True, False)
    for event in (ev.Listening(), ev.Mode("shared")):
        window.session.apply(event)
    window.refresh()
    return window.pipeline


def press(window, key) -> None:
    window._key_filter.eventFilter(window, QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))


def feed(window, *events) -> None:
    for event in events:
        window.session.apply(event)
    window.refresh()


def test_the_two_columns_say_who_speaks_what_and_which_key(window):
    window.refresh()
    assert window.mode_shared.isChecked() and not window.shared_view.isHidden()
    left, right = window.sides["left"], window.sides["right"]
    assert left["title"].text() == "English - ←"
    assert right["title"].text().startswith("Spanish (Mexico)") and right["title"].text().endswith("→")
    assert left["status"].text() == "Press Start"
    # Pairing and the shared machine exclude each other, and say why.
    assert not window.pair.isEnabled() and "Shared machine" in window.pair.toolTip()


def test_a_voice_list_holds_voices_in_the_other_persons_language(window):
    left_voices = [window.sides["left"]["voice"].itemData(i) for i in range(window.sides["left"]["voice"].count())]
    right_voices = [window.sides["right"]["voice"].itemData(i) for i in range(window.sides["right"]["voice"].count())]
    spanish = {v.dir_name for v in window.voices if "es" in v.languages}
    english = {v.dir_name for v in window.voices if "en" in v.languages}
    assert left_voices[0] == "" and set(left_voices[1:]) == spanish, "the left person is spoken in Spanish"
    assert right_voices[0] == "" and set(right_voices[1:]) == english
    # A side set to Spanish (Mexico) offers the Mexico-tuned voice first, labelled.
    first = window.sides["left"]["voice"].itemText(1)
    if any("es-MX" in v.varieties for v in window.voices):
        assert "tuned for Spanish (Mexico)" in first, first


def test_each_key_starts_and_ends_only_its_own_turn(window):
    pipeline = running(window)
    assert window.sides["left"]["status"].text() == "Ready - press ←"
    press(window, Qt.Key.Key_Left)
    kind, direction = pipeline.asked[-1]
    assert (kind, direction.side, direction.source, direction.target) == ("begin", "left", "en", "es-MX")
    assert direction.asr and direction.voice, "the recognizer and the voice are decided before the microphone opens"

    feed(window, ev.TurnStarted("left"))
    assert "Listening" in window.sides["left"]["status"].text()
    assert "4px solid white" in window.sides["left"]["frame"].styleSheet(), "the active side is unmistakable"
    assert "Wait" in window.sides["right"]["status"].text()
    assert not window.sides["left"]["language"].isEnabled(), "settings are locked until the turn is over"

    before = len(pipeline.asked)
    press(window, Qt.Key.Key_Right)
    assert len(pipeline.asked) == before, "the other key does nothing during a turn"
    press(window, Qt.Key.Key_Left)
    assert pipeline.asked[-1] == ("end",)

    feed(window, ev.TurnEnded())
    press(window, Qt.Key.Key_Left)
    press(window, Qt.Key.Key_Right)
    assert pipeline.asked[-1] == ("end",), "neither key does anything while the turn is worked on"
    assert window.sides["left"]["status"].text() == "Working..."


def test_escape_cancels_a_turn_and_is_left_alone_otherwise(window):
    pipeline = running(window)
    press(window, Qt.Key.Key_Escape)
    assert pipeline.asked == [], "nothing to cancel: Escape still belongs to whatever has the focus"
    feed(window, ev.TurnStarted("right"))
    press(window, Qt.Key.Key_Escape)
    assert pipeline.asked == [("cancel",)]
    feed(window, ev.TurnCancelled())
    assert window.sides["right"]["status"].text() == "Ready - press →"


def test_rows_say_whose_words_they_are(window):
    running(window)
    feed(window, ev.TurnStarted("right"), ev.TurnEnded(), ev.Final(1, "Hola.", "es-MX", 800, 300, 0.0, 1.0),
         ev.SentenceMsg("1.1", 1, "Hola.", "es-MX", 0.0, 1.0), ev.Translated("1.1", "Hello.", "en", 200, "cuda", "qwen"))
    cells = [window.table.item(0, c).text() for c in range(3)]
    assert cells[0].startswith("→") and cells[1:] == ["Hola.", "Hello."]


def test_a_side_that_cannot_take_a_turn_says_why_before_anyone_presses(window):
    pipeline = running(window)
    window.config.shared.right_asr = "deleted-model"
    window.refresh()
    assert "deleted-model" in window.sides["right"]["note"].text()
    press(window, Qt.Key.Key_Right)
    assert pipeline.asked == [], "no turn starts for a side that can't be heard"
    press(window, Qt.Key.Key_Left)
    assert pipeline.asked[-1][0] == "begin", "the other side keeps working"
    window.config.shared.right_asr = ""
    feed(window, ev.SharedSide("right", "the model would not load"))
    assert "would not load" in window.sides["right"]["note"].text()


def test_changing_a_sides_language_refills_its_pickers_and_prepares_the_recognizers(window):
    pipeline = running(window)
    combo = window.sides["right"]["language"]
    combo.setCurrentIndex(combo.findData("ar"))
    assert window.config.shared.right_language == "ar"
    assert pipeline.asked[-1][:3] == ("prepare", "en", "ar"), "loaded now, not on the next key press"
    assert window.sides["right"]["title"].text().startswith("Arabic")
    voices = window.sides["left"]["voice"]
    listed = {voices.itemData(i) for i in range(1, voices.count())}
    assert listed == {v.dir_name for v in window.voices if "ar" in v.languages}


def test_a_text_box_with_the_focus_keeps_its_arrow_keys(window, monkeypatch):
    pipeline = running(window)
    monkeypatch.setattr(QApplication, "focusWidget", staticmethod(lambda: window.glossary))
    press(window, Qt.Key.Key_Left)
    assert pipeline.asked == []
    monkeypatch.setattr(QApplication, "focusWidget", staticmethod(lambda: window.table))
    press(window, Qt.Key.Key_Left)
    assert pipeline.asked[-1][0] == "begin"


def test_switching_to_the_shared_machine_while_running_tells_the_pipeline(window):
    window.mode_turn.setChecked(True)
    pipeline = running(window)
    window.source = None
    window.mode_shared.setChecked(True)
    assert pipeline.asked[-1] == ("mode", "shared") and window.config.mode.kind == "shared"
