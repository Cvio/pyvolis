"""The window (PySide6). It only draws `Session` (session.py) and turns
clicks into pipeline starts and stops; what the events mean lives there.

Port of Rust `gui.rs`'s main window: recognizer, language and device pickers,
start and stop, the caption pane, the latency readout and the compare toggle.
New in pyvolis: the translator picker, each model's memory, and file mode
(open or drop a file, real time or fast, pause, a timeline whose rows play
their stretch of audio when clicked, export).

Models load on the pipeline's thread, never this one, so the window stays
responsive and shows "Loading...". Turn-taking, pairing and the shared machine
join at P6, P9 and P10.
"""

from __future__ import annotations

import dataclasses
import logging
from html import escape
import queue
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QAction, QBrush, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QRadioButton, QStyledItemDelegate, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from .. import audio, export, filerun, models, paths, scoring, varieties
from .. import events as ev
from ..audio import SAMPLE_RATE
from ..config import Config, ConfigError, PyvolisConfig, save_pyvolis_selections
from ..filesource import FileSourceError, read_16k_mono
from ..pipeline import CONTINUOUS, TURN, ArraySource, Collected, Options, Pipeline
from ..translate.context import parse_glossary
from . import session as ses

log = logging.getLogger(__name__)

REFRESH_MS = 100  # Rust's REFRESH
METER_FLOOR_DB = -60.0
COLUMNS = ["Time", "Source", "Translation", "Notes"]
MUTED = QColor(130, 130, 130)
PROBLEM = QColor(190, 60, 40)
REVISED = QColor(255, 244, 200)
REVISED_TEXT = QColor(20, 20, 20)  # readable on the highlight in a dark theme too
REVISED_SECONDS = 4.0  # how long a replaced translation stays highlighted


class DirectionDelegate(QStyledItemDelegate):
    """Lays a cell out right to left when its text is Arabic, Persian or
    Hebrew. Alignment alone isn't enough: with a left-to-right base direction
    the final full stop lands on the wrong side, and English words or numbers
    inside the sentence fall in the wrong order."""

    def initStyleOption(self, option, index) -> None:  # noqa: N802 - Qt's name
        super().initStyleOption(option, index)
        if ses.has_rtl(option.text):
            option.direction = Qt.LayoutDirection.RightToLeft
            option.displayAlignment = Qt.AlignmentFlag.AlignLeading | Qt.AlignmentFlag.AlignTop


# [mode].turn_key in volis.toml uses egui's key names (it is Rust's file).
KEY_NAMES = {
    "Space": Qt.Key.Key_Space, "Enter": Qt.Key.Key_Return, "Tab": Qt.Key.Key_Tab, "Escape": Qt.Key.Key_Escape,
    "ArrowLeft": Qt.Key.Key_Left, "ArrowRight": Qt.Key.Key_Right, "ArrowUp": Qt.Key.Key_Up,
    "ArrowDown": Qt.Key.Key_Down, "Backspace": Qt.Key.Key_Backspace, "Insert": Qt.Key.Key_Insert,
    "Delete": Qt.Key.Key_Delete, "Home": Qt.Key.Key_Home, "End": Qt.Key.Key_End,
    "PageUp": Qt.Key.Key_PageUp, "PageDown": Qt.Key.Key_PageDown,
}


def qt_key(name: str):
    """The Qt key for a [mode].turn_key name, or None if it isn't one."""
    if name in KEY_NAMES:
        return KEY_NAMES[name]
    sequence = QKeySequence.fromString(name)
    return sequence[0].key() if sequence.count() == 1 else None


class TurnKeyFilter(QObject):
    """Takes the turn key out of the application's input before any widget
    sees it, and says what it did. A focused button treats Space as a click;
    the turn key must never also press whatever has focus. Auto-repeats of a
    held key are swallowed too, and are not presses. Installed on the
    application, so it works wherever the focus is inside the window."""

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.window = window

    def eventFilter(self, _watched, event) -> bool:  # noqa: N802 - Qt's name
        kind = event.type()
        if kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease, QEvent.Type.ShortcutOverride):
            w = self.window
            if event.key() != w.turn_key or not w.turn_key_active() or not w.isActiveWindow():
                return False
            if kind == QEvent.Type.ShortcutOverride:
                event.accept()  # ours: no shortcut or button may claim it
                return True
            if not event.isAutoRepeat():
                w.turn_key_event(ses.KeyEdges(pressed=kind == QEvent.Type.KeyPress,
                                              released=kind == QEvent.Type.KeyRelease))
            return True
        return False


def run(root: Path, config: Config) -> int:
    app = QApplication.instance() or QApplication([])
    try:
        pyconfig, _ = PyvolisConfig.load(paths.pyvolis_config_file(root))
    except ConfigError as e:
        QMessageBox.critical(None, "pyvolis", str(e))
        return 1
    window = MainWindow(root, config, pyconfig)
    window.show()
    return app.exec()


class MainWindow(QMainWindow):
    def __init__(self, root: Path, config: Config, pyconfig: PyvolisConfig) -> None:
        super().__init__()
        self.root, self.config, self.pyconfig = root, config, pyconfig
        self.session = ses.Session()
        self.events: queue.Queue = queue.Queue()
        self.collected = Collected()
        self.pipeline: Pipeline | None = None
        self.source: ArraySource | None = None
        self.file_path: Path | None = None
        self.file_audio: np.ndarray | None = None
        self.player = None  # for playing rows and the original
        self.paused = False
        self.scores = ""
        self._drawn = 0  # lines already in the table
        self._changed: set[int] = set()
        self._redraw = False  # set by a background thread that has something to show
        self.turn_key = qt_key(config.mode.turn_key)
        self.turn_key_state = ses.TurnKey()
        self.holding = False

        self.setWindowTitle("pyvolis")
        self.resize(1180, 760)
        self.setAcceptDrops(True)
        self._build()
        self.rediscover()
        self._key_filter = TurnKeyFilter(self)
        QApplication.instance().installEventFilter(self._key_filter)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.tick)
        self._timer.start(REFRESH_MS)
        self.refresh()

    # -------------------------------------------------------------- layout

    def _build(self) -> None:
        open_action = QAction("&Open audio file...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_dialog)
        export_action = QAction("&Export...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export)
        menu = self.menuBar().addMenu("&File")
        menu.addAction(open_action)
        menu.addAction(export_action)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.indicator = QLabel()
        self.indicator.setMinimumWidth(330)
        self.meter = QProgressBar()
        self.meter.setRange(int(METER_FLOOR_DB), 0)
        self.meter.setFormat("%v dBFS")
        self.meter.setFixedWidth(180)
        top = QHBoxLayout()
        for widget in (self.start_button, self.pause_button, self.indicator):
            top.addWidget(widget)
        top.addStretch(1)
        top.addWidget(QLabel("Microphone"))
        top.addWidget(self.meter)

        self.source_live = QRadioButton("Microphone")
        self.source_file = QRadioButton("File")
        self.source_live.setChecked(True)
        self.source_live.toggled.connect(self.mode_changed)
        self.file_speak = False  # file mode's own "speak" choice: off by default, never saved
        self.file_label = QLabel("no file: File > Open, or drop one on the window")
        self.file_label.setStyleSheet("color: gray")
        self.open_button = QPushButton("Open...")
        self.open_button.clicked.connect(self.open_dialog)
        self.realtime = QRadioButton("Real time")
        self.fast = QRadioButton("Fast")
        self.fast.setChecked(True)
        self.play_original = QCheckBox("Play the original")
        self.play_original.setToolTip("In real time, hear the recording through the speakers while reading.")
        self.export_button = QPushButton("Export...")
        self.export_button.clicked.connect(self.export)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        file_row = QHBoxLayout()
        for widget in (self.source_live, self.source_file, self.open_button, self.file_label):
            file_row.addWidget(widget)
        file_row.addStretch(1)
        speed = QWidget()
        speed_row = QHBoxLayout(speed)
        speed_row.setContentsMargins(0, 0, 0, 0)
        for widget in (self.realtime, self.fast):
            speed_row.addWidget(widget)
        for widget in (speed, self.play_original, self.export_button):
            file_row.addWidget(widget)

        self.recognizer = QComboBox()
        self.translator = QComboBox()
        self.source_lang = QComboBox()
        self.target_lang = QComboBox()
        self.input_device = QComboBox()
        self.output_device = QComboBox()
        self.speak = QCheckBox("Speak translations")
        self.half_duplex = QCheckBox("Half-duplex (mute the microphone while speaking)")
        self.half_duplex.setToolTip("Turn off only when using headphones: with speakers, pyvolis would hear itself.")
        self.compare = QCheckBox("Compare recognizers (no translation or speech)")
        self.streaming = QCheckBox("Show text while speaking (streaming)")
        self.streaming.setToolTip("Transcribes the growing utterance every second; words two passes agree on are "
                                  "committed, the rest is shown lighter and may change. Costs more recognition.")
        self.use_context = QCheckBox("Translate with the earlier sentences as context")
        self.revise = QCheckBox("Revise earlier translations when what follows changes them")
        self.revise.setToolTip("After each sentence the last few are translated again together; a short earlier "
                               "sentence whose translation changes is replaced, and the row says so. A sentence "
                               "that has been spoken aloud is never revised, so with Speak translations on this "
                               "changes nothing. Costs one more translation per sentence.")
        self.use_context.toggled.connect(lambda on: self.revise.setEnabled(on and not self.running()))
        self.hold_fragments = QCheckBox("Join short fragments to what follows")
        self.glossary = QLineEdit()
        self.glossary.setPlaceholderText("Names and terms to keep exactly, separated by commas")
        self.glossary.editingFinished.connect(self.glossary_changed)
        self.mode_turn = QRadioButton("Take turns")
        self.mode_continuous = QRadioButton("Listen continuously")
        key = self.config.mode.turn_key
        self.style_toggle = QRadioButton(f"Press {key} to start, again to stop")
        self.style_hold = QRadioButton(f"Hold {key} while speaking")
        mode_box, style_box = QWidget(), QWidget()
        mode_row, style_row = QVBoxLayout(mode_box), QVBoxLayout(style_box)
        for row in (mode_row, style_row):
            row.setContentsMargins(0, 0, 0, 0)
        mode_row.addWidget(self.mode_turn)
        style_row.setContentsMargins(18, 0, 0, 0)
        for widget in (self.style_toggle, self.style_hold):
            style_row.addWidget(widget)
        mode_row.addWidget(style_box)
        mode_row.addWidget(self.mode_continuous)
        self.style_box, self.mode_box = style_box, mode_box
        form = QFormLayout()
        form.addRow("Mode", mode_box)
        form.addRow("Spoken", self.source_lang)
        form.addRow("Translate into", self.target_lang)
        form.addRow("Recognizer", self.recognizer)
        form.addRow("Translator", self.translator)
        form.addRow("Microphone", self.input_device)
        form.addRow("Speakers", self.output_device)
        form.addRow(self.speak)
        form.addRow(self.half_duplex)
        form.addRow(self.compare)
        form.addRow(self.streaming)
        form.addRow(self.use_context)
        form.addRow(self.revise)
        form.addRow(self.hold_fragments)
        form.addRow("Glossary", self.glossary)
        self.memory = QLabel()
        self.memory.setWordWrap(True)
        form.addRow(self.memory)
        settings = QGroupBox("Settings")
        settings.setLayout(form)
        settings.setFixedWidth(400)
        self.settings_box = settings
        # The mode, unlike every other setting, can change while running.
        self.locked_while_running = [self.source_lang, self.target_lang, self.recognizer, self.translator,
                                     self.input_device, self.output_device, self.speak, self.half_duplex,
                                     self.compare, self.streaming, self.use_context, self.revise,
                                     self.hold_fragments]
        for widget in (self.streaming, self.use_context, self.revise, self.hold_fragments):
            widget.toggled.connect(self.save)
        for widget in (self.mode_turn, self.mode_continuous, self.style_toggle, self.style_hold):
            widget.toggled.connect(self.mode_controls_changed)
        for combo in (self.source_lang, self.target_lang):
            for variety in varieties.TABLE:
                combo.addItem(variety.display, variety.tag)
        self.source_lang.currentIndexChanged.connect(self.languages_changed)
        self.target_lang.currentIndexChanged.connect(self.save)
        for widget in (self.recognizer, self.translator, self.input_device, self.output_device):
            widget.currentIndexChanged.connect(self.save)
        for widget in (self.speak, self.half_duplex):
            widget.toggled.connect(self.save)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setWordWrap(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setItemDelegate(DirectionDelegate(self.table))
        self.table.cellClicked.connect(self.row_clicked)

        self.latency = QLabel()
        self.status = QLabel()
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #b83a26")
        for label in (self.latency, self.status):
            # Its own lines, never squeezed: a wrapped label under a stretching
            # table was cut off at the window's lower edge.
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setSizePolicy(label.sizePolicy().horizontalPolicy(), label.sizePolicy().Policy.Fixed)

        body = QHBoxLayout()
        body.addWidget(settings)
        right = QVBoxLayout()
        right.addWidget(self.table, 1)
        right.addWidget(self.progress)
        body.addLayout(right, 1)
        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(file_row)
        layout.addLayout(body, 1)
        for widget in (self.latency, self.status, self.error):
            layout.addWidget(widget, 0)
        layout.setContentsMargins(9, 6, 9, 10)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    # -------------------------------------------------------------- settings

    def rediscover(self) -> None:
        """Read models/ and the devices again, and fill the pickers."""
        self._filling = True
        self.engines = [e for e in models.discover(paths.asr_dir(self.root), models.Role.ASR)
                        if isinstance(e, models.Engine)]
        self.translators = [t for t in models.discover_translators(paths.mt_dir(self.root))
                            if isinstance(t, models.Translator)]
        _select(self.source_lang, self.config.languages.source)
        _select(self.target_lang, self.config.languages.target)
        self._fill_recognizers()
        self.translator.clear()
        for t in self.translators:
            label = t.name if t.enabled() else f"{t.name} (unusable)"
            self.translator.addItem(f"{label}  [{t.id}]", t.id)
            if not t.enabled():
                self.translator.setItemData(self.translator.count() - 1, t.unusable or ", ".join(t.missing),
                                            Qt.ItemDataRole.ToolTipRole)
        wanted = self.pyconfig.translate.model or next((t.id for t in self.translators if t.top_level), "")
        _select(self.translator, wanted)
        try:
            inputs, outputs = audio.list_input_devices(), audio.list_output_devices()
        except audio.AudioError as e:
            inputs, outputs = [], []
            self.session.last_error = str(e)
        for combo, devices, chosen in ((self.input_device, inputs, self.config.audio.input_device),
                                       (self.output_device, outputs, self.config.audio.output_device)):
            combo.clear()
            combo.addItem("(system default)", "")
            for device in devices:
                combo.addItem(device.name + ("  *" if device.is_default else ""), device.name)
            _select(combo, chosen)
        self.speak.setChecked(self.config.tts.enabled)
        self.half_duplex.setChecked(self.config.tts.half_duplex)
        self.streaming.setChecked(self.pyconfig.asr.streaming)
        self.use_context.setChecked(self.pyconfig.context.mode in ("carry", "revision"))
        self.revise.setChecked(self.pyconfig.context.mode == "revision")
        self.hold_fragments.setChecked(self.pyconfig.fragments.hold)
        (self.mode_continuous if self.config.mode.kind == CONTINUOUS else self.mode_turn).setChecked(True)
        (self.style_hold if self.config.mode.turn_style == "hold" else self.style_toggle).setChecked(True)
        self._filling = False

    def _fill_recognizers(self) -> None:
        """Ranked for the spoken language: tuned, general, other varieties,
        then models whose languages are unknown, each labelled."""
        tag = self.source_lang.currentData() or self.config.languages.source
        self.recognizer.clear()
        ranked = models.rank(tag, self.engines)
        for r in ranked:
            self.recognizer.addItem(f"{r.engine.name}  ({r.fit.label(tag)}, {r.engine.backend})", r.engine.dir_name)
        listed = {r.engine.dir_name for r in ranked}
        for engine in self.engines:
            if engine.dir_name not in listed and not engine.enabled():
                why = engine.unusable or "missing: " + ", ".join(engine.missing_files())
                self.recognizer.addItem(f"{engine.name}  (unusable)", engine.dir_name)
                index = self.recognizer.count() - 1
                self.recognizer.setItemData(index, why, Qt.ItemDataRole.ToolTipRole)
                self.recognizer.model().item(index).setEnabled(False)
        if not _select(self.recognizer, self.config.asr.engine) and self.recognizer.count():
            self.recognizer.setCurrentIndex(0)  # the best fit for this language

    def languages_changed(self) -> None:
        if getattr(self, "_filling", True):
            return
        self._filling = True
        self._fill_recognizers()
        self._filling = False
        self.save()

    def mode_controls_changed(self) -> None:
        """The mode or the turn style was changed. It applies at once, even
        while running, and is saved."""
        if getattr(self, "_filling", True):
            return
        before = (self.config.mode.kind, self.config.mode.turn_style)
        kind = CONTINUOUS if self.mode_continuous.isChecked() else TURN
        style = "hold" if self.style_hold.isChecked() else "toggle"
        if (kind, style) == before:
            return
        self.config.mode.kind, self.config.mode.turn_style = kind, style
        self.holding = False
        if kind != before[0] and self.pipeline is not None and self.source is None:
            self.pipeline.set_mode(kind)
        self.save()
        self.refresh()

    def glossary_changed(self) -> None:
        """The glossary applies from the next sentence, even mid-run. It is
        the session's, and isn't saved."""
        if self.pipeline is not None:
            self.pipeline.set_glossary(parse_glossary(self.glossary.text()))

    def turn_key_active(self) -> bool:
        """Whether the turn key is ours right now: a live run, taking turns."""
        return (self.pipeline is not None and self.source is None and self.session.mode == TURN
                and self.session.state == ses.LISTENING)

    def turn_key_event(self, raw: ses.KeyEdges) -> None:
        edges = self.turn_key_state.update(raw, self.isActiveWindow())
        self._turn_key(edges)

    def _turn_key(self, edges: ses.KeyEdges) -> None:
        if not self.turn_key_active():
            return
        action, self.holding = ses.turn_key_action(self.config.mode.turn_style, self.session.turn, edges,
                                                   self.holding, self.isActiveWindow())
        if action == "begin":
            self.pipeline.begin_turn()
        elif action == "end":
            self.pipeline.end_turn()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        # Losing focus with the key down: the release would never arrive.
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow():
            self._turn_key(self.turn_key_state.update(ses.KeyEdges(), False))
        super().changeEvent(event)

    def mode_changed(self) -> None:
        """Microphone or file. The "Speak translations" box shows each mode's
        own choice: the saved setting live, off by default for a file."""
        self._filling = True
        self.speak.setChecked(self.file_speak if self.source_file.isChecked() else self.config.tts.enabled)
        self._filling = False
        self.refresh()

    def context_mode(self) -> str:
        """[context].mode as the two boxes say: revision builds on context."""
        if not self.use_context.isChecked():
            return "off"
        return "revision" if self.revise.isChecked() else "carry"

    def save(self) -> None:
        """Keep the selections, as Rust does: volis.toml for what Rust shares,
        pyvolis.toml for the translator."""
        if getattr(self, "_filling", True):
            return
        if self.source_file.isChecked():
            self.file_speak = self.speak.isChecked()
        c = self.config
        c.asr.engine = self.recognizer.currentData() or ""
        c.languages.source = self.source_lang.currentData() or c.languages.source
        c.languages.target = self.target_lang.currentData() or c.languages.target
        c.audio.input_device = self.input_device.currentData() or ""
        c.audio.output_device = self.output_device.currentData() or ""
        if not self.source_file.isChecked():
            c.tts.enabled = self.speak.isChecked()
        c.tts.half_duplex = self.half_duplex.isChecked()
        self.pyconfig.translate.model = self.translator.currentData() or ""
        self.pyconfig.asr.streaming = self.streaming.isChecked()
        self.pyconfig.context.mode = self.context_mode()
        self.pyconfig.fragments.hold = self.hold_fragments.isChecked()
        try:
            c.save_selections(paths.config_file(self.root))
            save_pyvolis_selections(paths.pyvolis_config_file(self.root), self.pyconfig)
        except ConfigError as e:
            self.session.last_error = str(e)

    # -------------------------------------------------------------- running

    def running(self) -> bool:
        return self.pipeline is not None

    def toggle(self) -> None:
        if self.running():
            self.stop()
        elif self.source_file.isChecked():
            self.start_file()
        else:
            self.start_live()

    def _begin(self, source, speak: bool) -> None:
        comparing = self.compare.isChecked()
        self.session.begin(self.config.languages.source, self.config.languages.target, comparing, speak)
        self.session.clear()
        self.collected = Collected()
        self.scores = ""
        self.table.setRowCount(0)
        self._drawn = 0
        self.paused = False
        ev.restart_clock()
        self.events = queue.Queue()
        self.options = Options(compare=comparing, translate=not comparing, speak=speak,
                               mt=self.translator.currentData() or "",
                               glossary=parse_glossary(self.glossary.text()))
        self.pipeline = Pipeline(self.root, dataclasses.replace(self.config), self.options, self.events, source,
                                 self.pyconfig).start()

    def start_live(self) -> None:
        self.source = None
        self._begin(None, self.speak.isChecked())

    def start_file(self) -> None:
        if self.file_audio is None:
            self.open_dialog()
            if self.file_audio is None:
                return
        realtime = self.realtime.isChecked()
        self.source = ArraySource(self.file_audio, realtime=realtime)
        # Voice output is off by default in file mode; the box turns it on.
        self._begin(self.source, self.file_speak)
        if realtime and self.play_original.isChecked():
            self._play(self.file_audio)

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            if self.source is not None:
                self.source.resume()
        if self.player is not None:
            self.player.clear()

    def toggle_pause(self) -> None:
        if self.source is None or not self.running():
            return
        self.paused = not self.paused
        (self.source.pause if self.paused else self.source.resume)()
        if self.player is not None:
            self.player.pause(self.paused)

    def tick(self) -> None:
        """Every 100 ms: take the pipeline's events, then redraw."""
        changed = False
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            changed = True
            self.collected.events.append(event)
            self.session.apply(event)
            if isinstance(event, ev.Stopped):
                self._finished()
        if changed or self._redraw:
            self._redraw = False
            self.refresh()

    def _finished(self) -> None:
        if self.pipeline is not None:
            self.pipeline.join(2.0)
        self.pipeline = None
        if self.file_path is not None and self.source is not None:
            # Scoring a long file takes a moment; not on the window's thread.
            path, collected = self.file_path, self.collected
            source, target = self.config.languages.source, self.config.languages.target

            def score() -> None:
                try:
                    reference = scoring.find_reference(path)
                    if reference is not None:
                        self.scores = filerun.scores(reference, collected, source, target)
                except scoring.ReferenceError as e:
                    self.session.last_error = str(e)
                self._redraw = True

            threading.Thread(target=score, name="pyvolis-score", daemon=True).start()

    # -------------------------------------------------------------- file mode

    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open audio file", str(self.file_path.parent if self.file_path else self.root),
            "Audio (*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.aac *.mp4 *.mka *.webm);;All files (*)")
        if path:
            self.open_file(Path(path))

    def open_file(self, path: Path) -> None:
        try:
            self.file_audio = read_16k_mono(path)
        except FileSourceError as e:
            self.session.last_error = str(e)
            self.refresh()
            return
        self.file_path = path
        self.source_file.setChecked(True)
        self.session.last_error = None
        self.refresh()

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt's name
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls and not self.running():
            self.open_file(Path(urls[0].toLocalFile()))

    def _play(self, samples: np.ndarray) -> None:
        from .. import playback

        try:
            if self.player is None:
                self.player = playback.Player(self.config.audio.output_device, playback.Gate(False))
            self.player.clear()
            self.player.pause(False)
            self.player.play(samples, SAMPLE_RATE)
        except playback.PlaybackError as e:
            self.session.last_error = str(e)

    def row_clicked(self, table_row: int, _column: int) -> None:
        """Clicking a row plays that stretch of the file."""
        item = self.table.item(table_row, 0)
        span = item.data(Qt.ItemDataRole.UserRole) if item else None
        if span is None or self.file_audio is None or self.source is None:
            return
        start, end = span
        self._play(self.file_audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)])

    def export(self) -> None:
        if not self.collected.events or self.file_path is None or self.source is None:
            self.session.last_error = "Nothing to export yet: open a file and run it first."
            self.refresh()
            return
        default = export.default_folder(self.root, self.file_path)
        default.parent.mkdir(parents=True, exist_ok=True)
        folder = QFileDialog.getExistingDirectory(self, "Export to folder", str(default.parent))
        if folder:
            self.export_to(Path(folder) / default.name if Path(folder) == default.parent else Path(folder))

    def export_to(self, folder: Path) -> None:
        prompt_file = paths.prompts_dir(self.root) / f"{self.pyconfig.translate.prompt}.txt"
        translator = self.session.models.get("translator")
        recognizer = self.session.models.get("recognizer")
        configuration = export.configuration(
            self.root, audio=self.file_path, config=self.config, pyconfig=self.pyconfig, options=self.options,
            recognizer=recognizer.name if recognizer else "", translator=translator.name if translator else "",
            prompt_file=prompt_file if translator else None,
            cli={"window": True, "realtime": self.realtime.isChecked()},
        )
        try:
            export.write(folder, self.collected.events, configuration)
            self.status.setText(f"Exported to {folder}")
        except OSError as e:
            self.session.last_error = f"cannot write the export to {folder}: {e}"
            self.refresh()

    # -------------------------------------------------------------- drawing

    def refresh(self) -> None:
        s = self.session
        running, file_mode = self.running(), self.source_file.isChecked()
        self.start_button.setText("Stop" if running else "Start")
        self.pause_button.setVisible(file_mode)
        self.pause_button.setEnabled(running and self.source is not None)
        self.pause_button.setText("Resume" if self.paused else "Pause")
        for widget in self.locked_while_running:
            widget.setEnabled(not running)
        self.revise.setEnabled(not running and self.use_context.isChecked())  # revision builds on context
        self.mode_box.setEnabled(not file_mode)
        self.style_box.setEnabled(self.mode_turn.isChecked())
        for widget in (self.open_button, self.realtime, self.fast, self.play_original, self.source_live, self.source_file):
            widget.setEnabled(not running)
        for widget in (self.realtime, self.fast, self.play_original, self.export_button, self.open_button, self.progress):
            widget.setVisible(file_mode)
        self.input_device.setEnabled(not file_mode and not running)
        self.half_duplex.setEnabled(not file_mode and not running)
        self.file_label.setVisible(file_mode)
        if self.file_path is not None:
            seconds = len(self.file_audio) / SAMPLE_RATE if self.file_audio is not None else 0
            self.file_label.setText(f"{self.file_path.name}  ({ses.clock(seconds)})")
            self.file_label.setToolTip(str(self.file_path))
            self.file_label.setStyleSheet("")

        text, colour = ses.indicator(s, self.config.mode.turn_key, self.config.mode.turn_style == "hold",
                                     self.paused, file_mode)
        self.indicator.setText(text)
        self.indicator.setStyleSheet(f"font-weight: bold; font-size: 15pt; padding: 6px 14px; "
                                     f"border-radius: 6px; background: {colour}; color: white")
        self.meter.setVisible(not file_mode)
        self.meter.setValue(int(max(METER_FLOOR_DB, s.level_db)) if s.level_db is not None else int(METER_FLOOR_DB))

        if s.progress:
            position, duration = s.progress
            self.progress.setRange(0, max(1, int(duration * 10)))
            self.progress.setValue(int(position * 10))
            self.progress.setFormat(f"{ses.clock(position)} / {ses.clock(duration)}")
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.progress.setFormat("")

        self._draw_lines()
        row = s.latest_timed()
        if row is not None:
            parts = [f"recognised {row.asr_ms} ms ({row.speech_ms} ms of speech)", f"translated {row.translate_ms} ms"]
            if row.first_audio_ms is not None:
                parts.append(f"first audio {row.first_audio_ms} ms")
            self.latency.setText("Latest: " + "  |  ".join(parts))
        else:
            self.latency.setText("")
        gpu, cpu = s.memory()
        if s.models:
            names = ", ".join(f"{m.role} {m.name} on the {'GPU' if m.device == 'cuda' else 'CPU'}"
                              for m in s.models.values())
            self.memory.setText(f"Loaded: GPU {gpu / 1e9:.1f} GB, system {cpu / 1e9:.1f} GB\n{names}")
        else:
            self.memory.setText("")
        # While a run is going the pipeline hasn't summed up yet: show what the rows say.
        stats = dict(s.stats) if s.stats else s.counts() | s.running_stats() | {"worst_stall_ms": s.worst_stall_ms}
        line = filerun.status_line(stats) if (s.lines or s.stats) else ""
        if s.comparing:
            line = "Comparing recognizers: nothing is translated or spoken.  " + line
        self.status.setText("\n".join(x for x in (line, self.scores) if x))
        self.error.setText(s.last_error or "")
        self.error.setVisible(bool(s.last_error))

    def _draw_lines(self) -> None:
        """Rows already drawn are updated in place; new lines are appended."""
        lines = self.session.lines
        if len(lines) < self._drawn:  # the bounded history dropped the oldest
            self.table.setRowCount(0)
            self._drawn = 0
        at_bottom = self.table.verticalScrollBar().value() >= self.table.verticalScrollBar().maximum() - 4
        provisional = self.session.provisional
        self.table.setRowCount(len(lines) + (1 if provisional else 0))
        self._changed = set()
        for index, line in enumerate(lines):
            if self.table.cellWidget(index, 1) is not None:  # was the provisional row
                self.table.removeCellWidget(index, 1)
            self._draw(index, line)
        if provisional:
            self._draw_provisional(len(lines), provisional)
        # A row grows when its translation arrives, not only when it is new.
        for index in self._changed:
            self.table.resizeRowToContents(index)
        if len(lines) > self._drawn and at_bottom:
            self.table.scrollToBottom()
        self._drawn = len(lines)

    def _draw_provisional(self, index: int, provisional) -> None:
        """The utterance being spoken: committed words in normal text, the
        current guess after them in a lighter style."""
        _utterance, pending, guess = provisional
        self._set(index, ["...", "", "", "listening"], muted=True)
        label = self.table.cellWidget(index, 1)
        if label is None:
            label = QLabel()
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setMargin(3)
            self.table.setCellWidget(index, 1, label)
        html = f"{escape(pending)} <span style='color:#909090'><i>{escape(guess)}</i></span>"
        if label.text() != html:
            label.setText(html)
            label.setLayoutDirection(Qt.LayoutDirection.RightToLeft if ses.has_rtl(pending + guess)
                                     else Qt.LayoutDirection.LeftToRight)
            self._changed.add(index)

    def _draw(self, index: int, line) -> None:
        if line.kind == "row":
            if line.held:
                self._set(index, ["...", line.source, "", "held: joining to what follows"], muted=True)
                return
            for column in range(self.table.columnCount()):  # no longer held or muted
                item = self.table.item(index, column)
                if item is not None:
                    item.setData(Qt.ItemDataRole.ForegroundRole, None)
            notes = []
            if line.problem:
                notes.append("not translated")
            if line.revised:
                notes.append(f"revised x{len(line.history)}")
            if line.translate_ms is not None:
                notes.append(f"{line.translate_ms} ms")
            cells = [ses.clock(line.start), line.source, line.target or (line.problem or ""), ", ".join(notes)]
            self._set(index, cells, span=(line.start, line.end))
            if line.problem and not line.target:
                self.table.item(index, 2).setForeground(QBrush(PROBLEM))
            if line.revised:
                # Highlighted briefly; the note and the earlier wording (tooltip) stay.
                item = self.table.item(index, 2)
                fresh = time.monotonic() - line.revised_at < REVISED_SECONDS
                item.setData(Qt.ItemDataRole.BackgroundRole, QBrush(REVISED) if fresh else None)
                if fresh:
                    item.setForeground(QBrush(REVISED_TEXT))
                item.setToolTip("Earlier: " + " | ".join(line.history))
        elif line.kind == "nothing":
            self._set(index, [ses.clock(line.start), f"(speech with no words, {line.speech_ms} ms)", "", ""], muted=True)
        elif line.kind == "dropped":
            self._set(index, [ses.clock(line.start), line.text, "(dropped: " + "; ".join(line.reasons) + ")",
                              "hallucination"], muted=True)
        elif line.kind == "comparison":
            c = line.comparison
            text = "\n".join(f"{'* ' if r.engine == self.config.asr.engine else ''}{r.engine}: "
                             f"{r.text or '(no text)'}  [{r.elapsed_ms} ms]" for r in c.runs)
            self._set(index, [ses.clock(c.start_ms / 1000), text, "", f"{c.segment_ms} ms of audio"])

    def _set(self, index: int, cells: list[str], span=None, muted: bool = False) -> None:
        for column, text in enumerate(cells):
            item = self.table.item(index, column)
            if item is None:
                item = QTableWidgetItem()
                self.table.setItem(index, column, item)
            if item.text() != text:
                item.setText(text)
                self._changed.add(index)
                # Direction is the delegate's job (DirectionDelegate); "leading"
                # is the left for English and the right for Arabic.
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeading | Qt.AlignmentFlag.AlignTop)
            if muted:
                item.setForeground(QBrush(MUTED))
        if span is not None:
            self.table.item(index, 0).setData(Qt.ItemDataRole.UserRole, span)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop()
        if self.pipeline is not None:
            self.pipeline.join(5.0)
        if self.player is not None:
            self.player.close()
        event.accept()


def _select(combo: QComboBox, data: str) -> bool:
    index = combo.findData(data)
    if index >= 0:
        combo.setCurrentIndex(index)
    return index >= 0
