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


def test_provisional_text_is_drawn_lighter_below_the_committed_rows(window):
    feed(window, ev.SentenceMsg("1.1", 1, "Se me murió el perro.", "es", 0.0, 2.0),
         ev.Partial(1, "ayer por la", committed="Se me murió el perro. Fue", pending="Fue"))
    assert window.table.rowCount() == 2
    label = window.table.cellWidget(1, 1)
    assert "Fue" in label.text() and "<i>ayer por la</i>" in label.text() and "color:" in label.text()
    feed(window, ev.Final(1, "Se me murió el perro. Fue ayer.", "es", 4000, 900, 0.0, 4.0),
         ev.SentenceMsg("1.2", 1, "Fue ayer.", "es", 2.0, 4.0))
    assert window.table.rowCount() == 2 and window.table.cellWidget(1, 1) is None
    assert cells(window, 1)[1] == "Fue ayer."


def test_a_held_fragment_is_shown_waiting(window):
    feed(window, ev.Held("1.1", "Yo manejo"))
    assert "held" in cells(window, 0)[3]
    feed(window, ev.SentenceMsg("1.1", 1, "Yo manejo Mi carro está aquí.", "es", 0.0, 3.0))
    assert cells(window, 0)[1] == "Yo manejo Mi carro está aquí." and "held" not in cells(window, 0)[3]


def test_the_glossary_reaches_a_running_pipeline(window):
    got = []

    class FakePipeline:
        def set_glossary(self, terms):
            got.append(terms)

        def stop(self):
            pass

        def join(self, timeout=None):
            pass

    window.pipeline = FakePipeline()
    window.glossary.setText("Susie Wolff, Bellas Artes")
    window.glossary_changed()
    window.pipeline = None
    assert got == [["Susie Wolff", "Bellas Artes"]]


def test_a_revision_is_highlighted_only_briefly(window):
    from PySide6.QtCore import Qt

    feed(window, ev.SentenceMsg("1.1", 1, "Yo manejo.", "es", 0.0, 1.0),
         ev.Translated("1.1", "I manage.", "en", 300, "cpu", "qwen"), ev.Revised("1.1", "I manage.", "I'll drive."))
    item = window.table.item(0, 2)
    assert item.text() == "I'll drive." and item.data(Qt.ItemDataRole.BackgroundRole) is not None
    window.session.row("1.1").revised_at -= 60  # a minute later
    window.refresh()
    assert window.table.item(0, 2).data(Qt.ItemDataRole.BackgroundRole) is None
    assert "revised" in cells(window, 0)[3] and "I manage." in window.table.item(0, 2).toolTip()


def test_the_revise_box_builds_on_context(window):
    window.use_context.setChecked(True)
    window.revise.setChecked(True)
    assert window.context_mode() == "revision"
    window.use_context.setChecked(False)
    assert not window.revise.isEnabled() and window.context_mode() == "off"
    window.use_context.setChecked(True)
    window.revise.setChecked(False)
    assert window.revise.isEnabled() and window.context_mode() == "carry"


# ---------------------------------------------------------------- P9: the peer panel


def test_the_peer_panel_shows_this_pcs_addresses_and_the_connection(window):
    window.addresses = [("Ethernet", "169.254.49.46"), ("Wi-Fi", "192.168.1.156")]
    window.my_name = "laptop-a"
    window.pair.setChecked(True)
    window.refresh()
    text = window.peer_me.text()
    assert "laptop-a" in text and "192.168.1.156" in text and "169.254.49.46" in text and "(no router)" in text
    assert not window.connect_button.isEnabled(), "nothing to connect with until a run is listening"

    window.session.begin("es", "en", False, True, True)
    window.pipeline = object()  # running
    feed(window, ev.Listening(), ev.Mode("turn"), ev.PeerMsg("waiting", port=47800))
    assert "Listening on port 47800" in window.peer_status.text() and window.connect_button.isEnabled()
    assert window.pair_label.text() == "Not paired"

    feed(window, ev.PeerMsg("connected", name="laptop-b", addr="192.168.1.20:47800", speaks="en", sends="es-MX"),
         ev.FloorChanged("them", "laptop-b"))
    status = window.peer_status.text()
    assert "Paired with laptop-b (192.168.1.20)" in status and "Floor: laptop-b's" in status
    assert "Mexican Spanish" in status or "Spanish" in status
    assert window.pair_label.text() == "Paired with laptop-b" and window.connect_button.text() == "Disconnect"
    assert "laptop-b IS TALKING" in window.indicator.text()

    feed(window, ev.Remote("laptop-b", "es-MX", "¿Dónde está la estación?", "en", "Where is the station?"))
    assert cells(window, 0) == ["<<", "Where is the station?", "¿Dónde está la estación?", "from laptop-b"]

    feed(window, ev.PeerMsg("disconnected", port=47800, reason="laptop-b went silent: nothing heard for 6 s."))
    assert "Not connected" in window.peer_status.text() and "went silent" in window.peer_status.text()
    assert window.pair_label.text() == "Not paired"
    window.pipeline = None


def test_paired_and_continuous_says_headsets_for_as_long_as_it_is_true(window):
    window.pair.setChecked(True)
    window.session.begin("es", "en", False, True, True)
    window.pipeline = object()
    feed(window, ev.Listening(), ev.Mode("continuous"))
    assert not window.headsets.isHidden() and "Headsets required" in window.headsets.text()
    feed(window, ev.Mode("turn"))
    assert window.headsets.isHidden()
    window.pipeline = None


def test_revision_is_off_while_paired_and_the_window_says_so(window):
    window.use_context.setChecked(True)
    window.pair.setChecked(True)
    window.refresh()
    assert not window.revise.isEnabled() and "off while paired" in window.revise.text()
    window.pair.setChecked(False)
    window.refresh()
    assert window.revise.isEnabled() and "off while paired" not in window.revise.text()


def test_a_name_is_never_dialled(window):
    window.pair.setChecked(True)
    window.session.begin("es", "en", False, True, True)
    dialled = []

    class FakePipeline:
        def connect(self, address):
            dialled.append(address)

        def stop(self):
            pass

        def join(self, timeout=None):
            pass

    window.pipeline = FakePipeline()
    feed(window, ev.Listening(), ev.PeerMsg("waiting", port=47800))
    window.connect_peer("laptop-b.local")
    assert dialled == [] and "never looked up" in window.session.last_error
    window.connect_peer("192.168.1.20")
    assert [str(a) for a in dialled] == ["192.168.1.20:47800"]
    window.pipeline = None
