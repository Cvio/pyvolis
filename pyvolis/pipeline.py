"""The pipeline: audio source -> VAD -> recognizer -> guards -> events.

Port of the continuous-listening core of Rust `pipeline.rs` (translation,
voice, turns, sharing and pairing join it at P3 to P10). The pipeline reports
only through `PipelineMsg` events on a queue; `listen.py` (`--listen`) and the
window are two consumers of it, as in Rust.

Threads, as Rust: the capture thread (audio.py) hands 16 kHz chunks to the
pipeline thread, which owns the detector and the recognizer. A file source
feeds the same queue, so a file goes through exactly what the microphone does.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import asr as asr_pkg
from . import compare, models, paths
from .asr import guards as guards_mod
from .audio import SAMPLE_RATE
from .config import Config, PyvolisConfig
from .ring import UtteranceRing
from .vad import Segment, Segmenter, VadSettings

log = logging.getLogger(__name__)

# Chunks queued between the source and the pipeline thread (about 6 s).
QUEUE_CHUNKS = 64


# ---------------------------------------------------------------- messages


@dataclass
class Listening:
    """The source is open; --seconds counts from here."""


@dataclass
class SpeechStarted:
    at: float


@dataclass
class Final:
    index: int
    text: str
    lang: str
    speech_ms: int
    asr_ms: int
    words: list | None = None
    start_ms: int = 0


@dataclass
class NothingRecognized:
    index: int
    speech_ms: int


@dataclass
class Dropped:
    """pyvolis: text the hallucination guards removed, with the reasons."""

    index: int
    text: str
    reasons: list[str]


@dataclass
class ComparisonMsg:
    comparison: compare.Comparison


@dataclass
class Error:
    message: str


@dataclass
class Stopped:
    pass


PipelineMsg = Listening | SpeechStarted | Final | NothingRecognized | Dropped | ComparisonMsg | Error | Stopped


@dataclass
class Options:
    write_wav: bool = False  # each utterance to logs/segments/
    compare: bool = False  # every usable recognizer on each utterance


# ---------------------------------------------------------------- sources


class MicSource:
    def __init__(self, device: str) -> None:
        self.device = device
        self._handle = None

    def start(self, out: queue.Queue) -> None:
        from . import audio

        self._handle = audio.spawn_capture(self.device, out)

    def done(self) -> bool:
        return False

    def stop(self) -> None:
        if self._handle is not None:
            self._handle.stop()
            self._handle = None


class ArraySource:
    """Audio already in memory (a decoded file), fed in 100 ms chunks. `realtime`
    paces it at playing speed; otherwise as fast as the pipeline takes it."""

    def __init__(self, audio: np.ndarray, realtime: bool = False, chunk: int = 1600) -> None:
        self.audio = audio
        self.realtime = realtime
        self.chunk = chunk
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, out: queue.Queue) -> None:
        def feed() -> None:
            began = time.perf_counter()
            for i, start in enumerate(range(0, len(self.audio), self.chunk)):
                if self._stop.is_set():
                    break
                if self.realtime:
                    wait = began + i * self.chunk / SAMPLE_RATE - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                out.put(self.audio[start : start + self.chunk])  # blocks: nothing is dropped
            self._finished.set()

        self._thread = threading.Thread(target=feed, name="pyvolis-file-source", daemon=True)
        self._thread.start()

    def done(self) -> bool:
        return self._finished.is_set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


# ---------------------------------------------------------------- the pipeline


class Pipeline:
    """Runs on its own thread until `stop()` or, for a file, the end of it."""

    def __init__(
        self,
        root: Path,
        config: Config,
        options: Options,
        events: queue.Queue,
        source=None,
        pyconfig: PyvolisConfig | None = None,
    ) -> None:
        self.root = root
        self.config = config
        self.options = options
        self.events = events
        self.source = source if source is not None else MicSource(config.audio.input_device)
        self.pyconfig = pyconfig if pyconfig is not None else PyvolisConfig.load(paths.pyvolis_config_file(root))[0]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pyvolis-pipeline", daemon=True)

    def start(self) -> Pipeline:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    # -------------------------------------------------------------- run

    def _run(self) -> None:
        try:
            self._loop()
        except Exception as e:  # anything unforeseen reaches the consumer, not just the log
            log.exception("the pipeline stopped")
            self.events.put(Error(str(e)))
        finally:
            self.events.put(Stopped())

    def _loop(self) -> None:
        root, config = self.root, self.config
        language = config.languages.source
        selected = config.asr.engine.strip()
        engines = [e for e in models.discover(paths.asr_dir(root), models.Role.ASR) if isinstance(e, models.Engine)]
        engine = next((e for e in engines if e.dir_name == selected), None)
        if not selected:
            raise asr_pkg.AsrError(f"[asr].engine is unset in {paths.config_file(root)}; run --report to see the models")
        if engine is None:
            # Never substitute a different one.
            raise asr_pkg.AsrError(
                f'[asr].engine = "{selected}" was not found in {paths.asr_dir(root)}; run --report'
            )
        recognizer = asr_pkg.load(engine)
        recognizer.prepare(language)

        guards = self._guards()
        comparison = compare.Engines(engines) if self.options.compare else None
        segmenter = Segmenter(
            VadSettings(
                model=paths.vad_model_file(root),
                threshold=config.vad.threshold,
                min_silence_ms=config.vad.min_silence_ms,
                min_speech_ms=config.vad.min_speech_ms,
                pre_roll_ms=self.pyconfig.vad.pre_roll_ms,
            )
        )
        segments_dir = paths.logs_dir(root) / "segments"
        if self.options.write_wav:
            segments_dir.mkdir(parents=True, exist_ok=True)
            log.info("writing utterances to %s", segments_dir)
        ring = UtteranceRing()
        chunks: queue.Queue = queue.Queue(QUEUE_CHUNKS)

        log.info('ready: listening continuously, "%s" hears "%s"', selected, language)
        self.source.start(chunks)
        self.events.put(Listening())
        speaking = False
        try:
            while not self._stop.is_set():
                try:
                    chunk = chunks.get(timeout=0.1)
                except queue.Empty:
                    if self.source.done():
                        break
                    continue
                for segment in segmenter.push(chunk):
                    self._handle(segment, recognizer, language, selected, ring, guards, comparison, segments_dir)
                now = segmenter.speech_in_progress()
                if now and not speaking:
                    self.events.put(SpeechStarted(time.monotonic()))
                speaking = now
            # End of input: whatever is still being said.
            while not chunks.empty():
                for segment in segmenter.push(chunks.get()):
                    self._handle(segment, recognizer, language, selected, ring, guards, comparison, segments_dir)
            for segment in segmenter.flush():
                self._handle(segment, recognizer, language, selected, ring, guards, comparison, segments_dir)
        finally:
            self.source.stop()
            recognizer.close()
            if comparison is not None:
                comparison.close()

    def _guards(self) -> guards_mod.Guards:
        g = self.pyconfig.guards
        scorer = guards_mod.SpeechScorer(paths.vad_model_file(self.root), g.min_peak_probability) if g.vad_probability else None
        phrases = guards_mod.load_phrases(paths.hallucinations_file(self.root)) if g.stock_phrases else {}
        return guards_mod.Guards(phrases, g.vad_probability, g.repeats, g.stock_phrases, scorer)

    def _handle(self, segment: Segment, recognizer, language, selected, ring, guards, comparison, segments_dir) -> None:
        utterance = ring.push(segment.start_ms(), segment.samples)
        log.info("utterance %d: %d ms .. %d ms (%d ms)", utterance.index, segment.start_ms(),
                 segment.end_ms(), segment.duration_ms())
        # Logged every time: if the language were lost, Whisper would guess,
        # be right most of the time, and hide the bug.
        log.info('  transcribing as "%s" with "%s"', language, selected)
        try:
            result = recognizer.transcribe(utterance.pcm, language)
        except asr_pkg.AsrError as e:
            log.warning("  transcription failed: %s", e)
            self.events.put(Error(f"utterance {utterance.index}: transcription failed: {e}"))
            result = None
        if result is not None:
            asr_ms = int(result.seconds * 1000)
            verdict = guards.check(result.text, language, utterance.pcm)
            if verdict.changed:
                for reason in verdict.reasons:
                    log.info('  guard: %s: "%s"', reason, result.text)
            if verdict.dropped:
                log.info("  dropped: %s", "; ".join(verdict.reasons))
                self.events.put(Dropped(utterance.index, result.text, verdict.reasons))
            elif not verdict.text:
                # No words: usually a cough, sometimes real speech the
                # recognizer failed on. Reported either way.
                log.info("  no words recognised in %d ms of audio (%d ms to decide)",
                         utterance.duration_ms(), asr_ms)
                self.events.put(NothingRecognized(utterance.index, utterance.duration_ms()))
            else:
                log.info("  [%s] %s\n  (%d ms to transcribe %d ms of audio)", language, verdict.text,
                         asr_ms, utterance.duration_ms())
                self.events.put(Final(utterance.index, verdict.text, language, utterance.duration_ms(),
                                      asr_ms, result.words, utterance.start_ms))
        if comparison is not None:
            table = compare.run_all(comparison, utterance, language, guards)
            compare.report(table, selected)
            self.events.put(ComparisonMsg(table))
        if self.options.write_wav:
            write_segment(utterance, segments_dir)


def write_segment(utterance, directory: Path) -> None:
    """As Rust: 16-bit PCM, named by index, start and duration."""
    path = directory / f"utterance-{utterance.index:04d}-at-{utterance.start_ms}ms-for-{utterance.duration_ms()}ms.wav"
    write_wav(path, utterance.pcm)
    log.info("  wrote %s", path)


def write_wav(path: Path, samples: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    """Rust's wav::write_any: clamp, scale by i16::MAX, round."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


@dataclass
class Collected:
    """Everything a run produced, for scripts and tests."""

    finals: list[Final] = field(default_factory=list)
    dropped: list[Dropped] = field(default_factory=list)
    nothing: list[NothingRecognized] = field(default_factory=list)
    comparisons: list[compare.Comparison] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run_to_end(pipeline: Pipeline, events: queue.Queue) -> Collected:
    """Start a pipeline on a finite source and collect its events until it stops."""
    out = Collected()
    pipeline.start()
    while True:
        msg = events.get()
        if isinstance(msg, Final):
            out.finals.append(msg)
        elif isinstance(msg, Dropped):
            out.dropped.append(msg)
        elif isinstance(msg, NothingRecognized):
            out.nothing.append(msg)
        elif isinstance(msg, ComparisonMsg):
            out.comparisons.append(msg.comparison)
        elif isinstance(msg, Error):
            out.errors.append(msg.message)
        elif isinstance(msg, Stopped):
            break
    pipeline.join()
    return out
