"""The window draws what the session holds. Runs with Qt's offscreen platform,
loads no models and saves no settings. `scripts/window_check.py` is the full
run (a file through the real models, with the responsiveness measurement)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem  # noqa: E402

from pyvolis import events as ev  # noqa: E402
from pyvolis import paths  # noqa: E402
from pyvolis.config import Config, PyvolisConfig  # noqa: E402
from pyvolis.gui.window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    w = MainWindow(paths.app_root(), Config(), PyvolisConfig())
    w.save = lambda: None  # a test never touches the user's settings
    yield w
    w.close()


def feed(window, *events) -> None:
    for event in events:
        window.session.apply(event)
    window.refresh()


def cells(window, row: int) -> list[str]:
    return [window.table.item(row, c).text() for c in range(window.table.columnCount())]


def test_a_sentence_becomes_a_row_and_its_translation_fills_in(window):
    feed(window, ev.Listening(), ev.Final(1, "¿Dónde está la estación?", "es", 1500, 120, 62.0, 63.5),
         ev.SentenceMsg("1.1", 1, "¿Dónde está la estación?", "es", 62.0, 63.5))
    assert cells(window, 0)[:3] == ["1:02", "¿Dónde está la estación?", ""]
    feed(window, ev.Translated("1.1", "Where is the station?", "en", 340, "cpu", "qwen"))
    assert cells(window, 0) == ["1:02", "¿Dónde está la estación?", "Where is the station?", "340 ms"]
    assert "recognised 120 ms" in window.latency.text() and "translated 340 ms" in window.latency.text()


def test_arabic_and_persian_cells_are_laid_out_right_to_left(window):
    feed(window, ev.SentenceMsg("1.1", 1, "تغادر بين 36 و 37 Uber إلى المطار.", "ar", 0.0, 2.0),
         ev.Translated("1.1", "They leave between 36 and 37.", "en", 300, "cpu", "qwen"),
         ev.SentenceMsg("2.1", 2, "می‌روم به خانه.", "fa", 3.0, 4.0))
    delegate = window.table.itemDelegate()

    def direction(row: int, column: int):
        option = QStyleOptionViewItem()
        delegate.initStyleOption(option, window.table.model().index(row, column))
        return option.direction

    assert direction(0, 1) == Qt.LayoutDirection.RightToLeft, "Arabic, with numbers and an English word in it"
    assert direction(2, 1) == Qt.LayoutDirection.RightToLeft, "Persian"
    assert direction(0, 2) != Qt.LayoutDirection.RightToLeft, "its English translation stays left to right"


def test_refusals_drops_and_silence_are_shown_not_hidden(window):
    feed(window, ev.SentenceMsg("1.1", 1, "Me scables.", "es", 0.0, 1.0),
         ev.NotTranslated("1.1", "the model recited its own instructions", "recited"),
         ev.Dropped(2, "Gracias por ver el video.", ["a stock phrase"], 5.0),
         ev.NothingRecognized(3, 900, 9.0))
    assert "recited" in cells(window, 0)[2] and "not translated" in cells(window, 0)[3]
    assert "dropped: a stock phrase" in cells(window, 1)[2]
    assert "no words" in cells(window, 2)[1]


def test_a_revision_is_marked_and_keeps_its_history(window):
    feed(window, ev.SentenceMsg("1.1", 1, "Yo manejo.", "es", 0.0, 1.0),
         ev.Translated("1.1", "I handle it.", "en", 300, "cpu", "qwen"), ev.Revised("1.1", "I handle it.", "I'll drive."))
    assert cells(window, 0)[2] == "I'll drive." and "revised" in cells(window, 0)[3]
    assert "I handle it." in window.table.item(0, 2).toolTip()


def test_loading_memory_errors_and_progress_reach_the_screen(window):
    feed(window, ev.Loading("recognizer whisper"))
    assert window.indicator.text() == "LOADING recognizer whisper..."
    feed(window, ev.ModelLoaded("recognizer", "whisper", "cuda", 1_600_000_000, 0), ev.Listening(),
         ev.Progress(30.0, 120.0), ev.Error("no installed voice speaks \"ja\""))
    assert "GPU 1.6 GB" in window.memory.text() and "on the GPU" in window.memory.text()
    assert window.progress.format() == "0:30 / 2:00"
    assert "ja" in window.error.text()


def test_file_mode_speaks_only_when_asked_and_never_saves_that(window):
    window.config.tts.enabled = True
    window.mode_changed()
    assert window.speak.isChecked(), "live: the saved setting"
    window.source_file.setChecked(True)
    assert not window.speak.isChecked(), "a file: off by default"
    assert window.config.tts.enabled, "and the saved setting is untouched"
