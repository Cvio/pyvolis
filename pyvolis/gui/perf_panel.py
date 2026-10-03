"""The performance panel: View > Performance (Ctrl+Shift+P).

A dockable panel, closed when the app starts. It sits beside the window or is
pulled out as a window of its own. Three tabs:

* Now: GPU, memory and CPU, for pyvolis and for the machine, once a second,
  with a graph of the last five minutes (marks where sentences were
  translated), the models loaded and what each takes, and the last sentences'
  timings with the context each carried. Amber or red when memory is nearly
  full or speech falls behind.
* What if: another recognizer, translator and voice (and a second recognizer
  for shared machine mode), with the memory they would take, the speed
  measured for them here, and whether they fit, without loading anything.
* Benchmarks: every run measured on this machine; and a button that runs a
  recording through the combination chosen under What if.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QComboBox, QDockWidget, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .. import events as ev
from .. import perf

GB = 1e9
COLORS = {"ok": "", "amber": "#d08c00", "red": "#d03030", "fits": "#2e8b57", "tight": "#d08c00", "no": "#d03030"}
SERIES = [("GPU memory", "#4c8bf5"), ("GPU load", "#2e8b57"), ("CPU", "#d08c00")]


def gb(n: int) -> str:
    return f"{n / GB:.1f} GB"


class Graph(QWidget):
    """The last five minutes: GPU memory, GPU load and CPU as percentages, and
    a tick at the bottom for each translated sentence."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(120)
        self.samples: list[perf.Sample] = []
        self.marks: list[float] = []

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt's name
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, self.palette().base())
        p.setPen(QPen(self.palette().mid().color()))
        for share in (0.25, 0.5, 0.75):
            p.drawLine(0, int(h * share), w, int(h * share))
        if not self.samples:
            return
        now = time.monotonic()
        span = perf.HISTORY * perf.SAMPLE_S

        def x(t: float) -> int:
            return int(w - (now - t) / span * w)

        def values(s: perf.Sample) -> list[float]:
            g = s.gpu
            return [100 * g.used / g.total if g and g.total else 0.0, g.load if g else 0.0, s.cpu]

        for i, (_name, color) in enumerate(SERIES):
            p.setPen(QPen(QColor(color), 2))
            points = [(x(s.t), h - int(values(s)[i] / 100 * (h - 4)) - 2) for s in self.samples]
            for a, b in zip(points, points[1:]):
                p.drawLine(a[0], a[1], b[0], b[1])
        p.setPen(QPen(self.palette().text().color()))
        for t in self.marks:
            if now - t < span:
                p.drawLine(x(t), h - 8, x(t), h)


class PerfPanel(QDockWidget):
    def __init__(self, window) -> None:
        super().__init__("Performance", window)
        self.setObjectName("performance")
        self.window_ = window
        self.root = window.root
        self.measured = perf.Measurements.load(self.root)
        self.sampler = perf.Sampler()
        self.gpu = perf.gpu_available()
        self.loaded: dict[str, perf.Estimate] = {}  # "role:name" -> what it took when it loaded
        self.sentences: dict[str, dict] = {}  # id -> timings
        self.asr_ms: dict[int, tuple[int, int]] = {}  # utterance -> (speech ms, recognition ms)
        self.behind_until = 0.0
        self.run_began = 0.0
        self.run_peak = 0
        self.bench_events: queue.Queue | None = None
        self.bench_note = ""

        tabs = QTabWidget()
        tabs.addTab(self._now_tab(), "Now")
        tabs.addTab(self._whatif_tab(), "What if")
        tabs.addTab(self._bench_tab(), "Benchmarks")
        self.setWidget(tabs)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        self.visibilityChanged.connect(self._shown)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)

    # -------------------------------------------------------------- tabs

    def _now_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        grid = QGridLayout()
        self.now_labels: dict[str, QLabel] = {}
        rows = [("gpu", "GPU"), ("gpu_mem", "GPU memory"), ("gpu_split", "  pyvolis / others / free"),
                ("ram", "System memory"), ("ram_split", "  pyvolis / others / free"), ("cpu", "CPU"),
                ("speech", "Keeping up")]
        for i, (key, text) in enumerate(rows):
            grid.addWidget(QLabel(text), i, 0)
            self.now_labels[key] = QLabel("-")
            self.now_labels[key].setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(self.now_labels[key], i, 1)
        layout.addLayout(grid)
        legend = QHBoxLayout()
        for name, color in SERIES:
            legend.addWidget(QLabel(f'<span style="color:{color}">&#9632;</span> {name}'))
        legend.addWidget(QLabel("| a tick per sentence"))
        legend.addStretch()
        layout.addLayout(legend)
        self.graph = Graph()
        layout.addWidget(self.graph)
        self.models_table = QTableWidget(0, 4)
        self.models_table.setHorizontalHeaderLabels(["Loaded", "On", "GPU", "System"])
        layout.addWidget(self.models_table)
        self.sentence_table = QTableWidget(0, 6)
        self.sentence_table.setHorizontalHeaderLabels(["Sentence", "Speech ms", "Recognize ms", "Context",
                                                       "Translate ms", "To first sound ms"])
        layout.addWidget(self.sentence_table)
        return page

    def _whatif_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.w_asr, self.w_asr2, self.w_mt, self.w_voice = QComboBox(), QComboBox(), QComboBox(), QComboBox()
        form.addRow("Recognizer", self.w_asr)
        form.addRow("Second recognizer (shared machine)", self.w_asr2)
        form.addRow("Translator", self.w_mt)
        form.addRow("Voice", self.w_voice)
        fill = QPushButton("Use the window's current choice")
        fill.clicked.connect(self.fill_from_window)
        form.addRow(fill)
        self.verdict_label = QLabel()
        self.verdict_label.setWordWrap(True)
        form.addRow(self.verdict_label)
        self.estimate_table = QTableWidget(0, 5)
        self.estimate_table.setHorizontalHeaderLabels(["Model", "On", "GPU", "System", "From"])
        form.addRow(self.estimate_table)
        self.speed_label = QLabel()
        self.speed_label.setWordWrap(True)
        form.addRow(self.speed_label)
        for combo in (self.w_asr, self.w_asr2, self.w_mt, self.w_voice):
            combo.currentIndexChanged.connect(self.refresh_whatif)
        return page

    def _bench_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel("Every run is measured here and kept in logs\\performance.json. The button runs a "
                      "recording through the combination chosen under What if, translated, without speaking, "
                      "as fast as the models allow. Stop the window's own run first.")
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        self.bench_button = QPushButton("Benchmark the What if combination...")
        self.bench_button.clicked.connect(self.benchmark)
        row.addWidget(self.bench_button)
        self.bench_status = QLabel()
        row.addWidget(self.bench_status, 1)
        layout.addLayout(row)
        self.runs_table = QTableWidget(0, 8)
        self.runs_table.setHorizontalHeaderLabels(["When", "Recognizer", "Translator", "Mode", "ASR RTF",
                                                   "Translate ms", "First sound ms", "Peak GPU"])
        layout.addWidget(self.runs_table)
        return page

    # -------------------------------------------------------------- events

    def _shown(self, visible: bool) -> None:
        if visible:
            self.sampler.start()
            self._timer.start(1000)
            self.fill_from_window()
            self.refresh()
        else:
            self.sampler.stop()
            self._timer.stop()

    def observe(self, event) -> None:
        """Every pipeline event the window sees, and the benchmark's own."""
        now = time.monotonic()
        if isinstance(event, ev.Loading) and not self.run_began:
            self.run_began, self.run_peak = now, 0
            self.loaded.clear()
        elif isinstance(event, ev.ModelLoaded):
            self.loaded[f"{event.role}:{event.name}"] = perf.Estimate(event.role, event.name, event.device,
                                                                     event.gpu_bytes, event.cpu_bytes, True)
            self.measured.model(event.role, event.name, event.device, event.gpu_bytes, event.cpu_bytes)
        elif isinstance(event, ev.Final):
            self.asr_ms[event.index] = (event.speech_ms, event.asr_ms)
        elif isinstance(event, ev.Translated):
            row = self.sentences.setdefault(event.id, {})
            row.update(translate_ms=event.translate_ms, context=event.context_turns)
            utterance = int(event.id.split(".")[0]) if event.id.split(".")[0].isdigit() else -1
            if utterance in self.asr_ms:
                row["speech_ms"], row["asr_ms"] = self.asr_ms[utterance]
            self.graph.marks.append(now)
        elif isinstance(event, ev.SpeakingStarted) and event.id:
            self.sentences.setdefault(event.id, {})["first_audio_ms"] = event.first_audio_ms
        elif isinstance(event, (ev.NotTranslated, ev.Error)):
            text = getattr(event, "reason", "") or getattr(event, "message", "")
            if "fell behind" in text or "is behind" in text:
                self.behind_until = now + 30
        elif isinstance(event, ev.Summary):
            self._record_run(event.stats)
        elif isinstance(event, ev.Stopped):
            self.run_began = 0.0
            self.loaded.clear()

    def _record_run(self, stats: dict) -> None:
        role = {e.role: e.name for e in self.loaded.values()}
        firsts = sorted(r["first_audio_ms"] for r in self.sentences.values() if "first_audio_ms" in r)
        self.measured.run(perf.Run(perf.now_text(), role.get("recognizer", ""), role.get("translator", ""),
                                   self.window_.config.mode.kind, stats.get("asr_rtf"), stats.get("translate_ms_median"),
                                   firsts[len(firsts) // 2] if firsts else None, stats.get("sentences", 0),
                                   self.run_peak))

    # -------------------------------------------------------------- drawing

    def refresh(self) -> None:
        sample = self.sampler.latest()
        if sample is not None and sample.gpu is not None and self.run_began:
            self.run_peak = max(self.run_peak, sample.gpu.used)
        self._drain_benchmark()
        if not self.isVisible():
            return
        self.graph.samples = list(self.sampler.history)
        self.graph.update()
        ours_gpu = sum(e.gpu_bytes for e in self.loaded.values())
        L = self.now_labels
        if sample is None:
            return
        g = sample.gpu
        if g is None:
            L["gpu"].setText("no NVIDIA GPU found (nvidia-smi)")
            for key in ("gpu_mem", "gpu_split"):
                L[key].setText("-")
        else:
            L["gpu"].setText(f"{g.name}: {g.load:.0f}% load, {g.temperature:.0f} C, {g.clock_mhz:.0f} MHz, "
                             f"{g.power_w:.0f} W")
            self._set(L["gpu_mem"], f"{gb(g.used)} of {gb(g.total)}", perf.level(g.used, g.total))
            L["gpu_split"].setText(f"{gb(ours_gpu)} / {gb(max(g.used - ours_gpu, 0))} / {gb(g.total - g.used)}")
        self._set(L["ram"], f"{gb(sample.ram_used)} of {gb(sample.ram_total)}",
                  perf.level(sample.ram_used, sample.ram_total))
        L["ram_split"].setText(f"{gb(sample.ram_ours)} / {gb(max(sample.ram_used - sample.ram_ours, 0))} / "
                               f"{gb(sample.ram_total - sample.ram_used)}")
        L["cpu"].setText(f"{sample.cpu:.0f}% of the machine; pyvolis {sample.cpu_ours:.0f}% of one core")
        behind = time.monotonic() < self.behind_until
        self._set(L["speech"], "fell behind in the last 30 s" if behind else "yes", "red" if behind else "ok")
        self._fill(self.models_table, [[e.name, e.device.upper(), gb(e.gpu_bytes), gb(e.cpu_bytes)]
                                       for e in self.loaded.values()])
        rows = list(self.sentences.items())[-20:]
        self._fill(self.sentence_table, [[sid, r.get("speech_ms", ""), r.get("asr_ms", ""), r.get("context", ""),
                                          r.get("translate_ms", ""), r.get("first_audio_ms", "")]
                                         for sid, r in reversed(rows)])
        self.refresh_whatif()
        self._fill(self.runs_table, [[r["when"], r["recognizer"], r["translator"], r["mode"], r["asr_rtf"] or "",
                                      r["translate_ms"] or "", r["first_audio_ms"] or "",
                                      gb(r["peak_gpu_used"]) if r["peak_gpu_used"] else ""]
                                     for r in reversed(self.measured.runs[-100:])])

    def fill_from_window(self) -> None:
        w = self.window_
        self._choices(self.w_asr, [e.dir_name for e in w.engines if e.enabled()], w.recognizer.currentData())
        self._choices(self.w_asr2, ["(none)"] + [e.dir_name for e in w.engines if e.enabled()], "(none)")
        self._choices(self.w_mt, ["(none)"] + [t.id for t in w.translators if t.enabled()], w.translator.currentData())
        self._choices(self.w_voice, ["(none)"] + [v.dir_name for v in w.voices if v.enabled()], "(none)")
        self.refresh_whatif()

    def refresh_whatif(self) -> None:
        w = self.window_
        estimates = []
        for combo in (self.w_asr, self.w_asr2):
            engine = next((e for e in w.engines if e.dir_name == combo.currentText()), None)
            if engine is not None:
                estimates.append(perf.estimate_recognizer(engine, self.measured, self.gpu))
        entry = next((t for t in w.translators if t.id == self.w_mt.currentText()), None)
        if entry is not None:
            estimates.append(perf.estimate_translator(entry, self.measured, self.gpu))
        voice = next((v for v in w.voices if v.dir_name == self.w_voice.currentText()), None)
        if voice is not None:
            estimates.append(perf.estimate_voice(voice))
        sample = self.sampler.latest()
        ours_gpu = sum(e.gpu_bytes for e in self.loaded.values())
        ours_ram = sample.ram_ours if sample else 0
        verdict = perf.verdict(estimates, sample, ours_gpu, ours_ram)
        self._set(self.verdict_label, verdict.text if sample else "measuring...", verdict.level if sample else "ok")
        self._fill(self.estimate_table, [[e.name, e.device.upper(), gb(e.gpu_bytes), gb(e.cpu_bytes),
                                          "measured here" if e.measured else "its files"] for e in estimates])
        speed = self.measured.speed(self.w_asr.currentText(), self.w_mt.currentText())
        if speed["runs"]:
            self.speed_label.setText(
                f"Measured in {speed['runs']} run(s) here: recognition RTF {speed['asr_rtf']}, "
                f"translation {speed['translate_ms']} ms a sentence, first sound "
                f"{speed['first_audio_ms'] or '-'} ms after speech ends (medians).")
        else:
            self.speed_label.setText("Speed: not measured with this recognizer and translator yet "
                                     "(Benchmarks tab).")

    # -------------------------------------------------------------- benchmark

    def benchmark(self) -> None:
        w = self.window_
        if w.running() or self.bench_events is not None:
            self.bench_status.setText("Stop the current run first.")
            return
        start = str(self.root / "tests" / "fixtures" / "files")
        path, _ = QFileDialog.getOpenFileName(self, "A recording to benchmark with", start,
                                              "Audio (*.wav *.mp3 *.m4a *.flac *.ogg *.mp4);;All files (*)")
        if not path:
            return
        from ..config import PyvolisConfig
        from ..filesource import read_16k_mono
        from ..pipeline import ArraySource, Options, Pipeline

        asr, mt = self.w_asr.currentText(), self.w_mt.currentText()
        options = Options(translate=mt != "(none)", speak=False, mt="" if mt == "(none)" else mt, asr=asr,
                          streaming=False)
        config = dataclasses.replace(w.config)
        pyconfig = dataclasses.replace(w.pyconfig)
        self.bench_events = queue.Queue()
        self.bench_status.setText(f"running {Path(path).name}...")
        self.bench_button.setEnabled(False)
        events = self.bench_events

        def run() -> None:
            try:
                source = ArraySource(read_16k_mono(Path(path)))
                pipeline = Pipeline(self.root, config, options, events, source, pyconfig).start()
                pipeline.join()
            except Exception as e:  # shown in the tab, never raised into Qt
                events.put(ev.Error(f"benchmark failed: {e}"))
            events.put(None)

        threading.Thread(target=run, name="pyvolis-benchmark", daemon=True).start()

    def _drain_benchmark(self) -> None:
        if self.bench_events is None:
            return
        while True:
            try:
                event = self.bench_events.get_nowait()
            except queue.Empty:
                return
            if event is None:
                self.bench_events = None
                self.bench_button.setEnabled(True)
                self.bench_status.setText(self.bench_note or "done: see the table")
                self.bench_note = ""
                return
            if isinstance(event, ev.Error):
                self.bench_note = event.message
            self.observe(event)

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _set(label: QLabel, text: str, level: str) -> None:
        color = COLORS.get(level, "")
        label.setText(text)
        label.setStyleSheet(f"color: {color}; font-weight: bold" if color and level not in ("ok", "fits") else
                            f"color: {color}" if color else "")

    @staticmethod
    def _fill(table: QTableWidget, rows: list[list]) -> None:
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                item = table.item(r, c)
                if item is None:
                    table.setItem(r, c, QTableWidgetItem(str(value)))
                elif item.text() != str(value):
                    item.setText(str(value))
        table.resizeColumnsToContents()

    @staticmethod
    def _choices(combo: QComboBox, items: list[str], selected) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        if selected in items:
            combo.setCurrentIndex(items.index(selected))
        combo.blockSignals(False)
