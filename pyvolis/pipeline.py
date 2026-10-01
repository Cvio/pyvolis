"""The pipeline: audio source -> VAD -> recognizer -> guards -> sentences ->
translator -> events.

Port of the continuous-listening core of Rust `pipeline.rs` (voice, turns,
sharing and pairing join it at P5 to P10). The pipeline reports only through
events (events.py) on a queue; `--listen`, file mode and the window are all
consumers of the same events, as in Rust.

Threads, as Rust, with the costs in mind:
  capture or file source   hands 16 kHz chunks on (audio.py, ArraySource)
  pipeline                 VAD and recognition, one utterance at a time
  translation              its own thread, so a slow model never stalls
                           recognition; a queue of TRANSLATION_QUEUE sentences
  stall probe              a 50 ms timer that reports when Python's threads
                           are held up (a library holding the GIL): the early
                           warning for the window's responsiveness (P5)
Live, a sentence the translator can't take in time is dropped and reported,
as Rust does; from a file, the pipeline waits instead, so a file run never
loses a sentence. Streaming (P7) and revision (P8) add threads of their own.
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
from . import compare, models, paths, sentences
from . import translate as tr
from .asr import guards as guards_mod
from .audio import SAMPLE_RATE
from .config import Config, PyvolisConfig
from .events import (  # noqa: F401 - re-exported for consumers
    ComparisonMsg, Dropped, Error, Event, Final, Listening, NothingRecognized, NotTranslated,
    Progress, SentenceMsg, SpeechStarted, Stall, Stopped, Summary, Translated,
)
from .ring import UtteranceRing
from .vad import Segment, Segmenter, VadSettings

log = logging.getLogger(__name__)

QUEUE_CHUNKS = 64  # source -> pipeline, about 6 s
TRANSLATION_QUEUE = 4  # Rust's TRANSLATION_QUEUE
STALL_TICK = 0.05  # the stall probe's timer
STALL_REPORT_MS = 150  # lateness worth an event


@dataclass
class Options:
    write_wav: bool = False  # each utterance to logs/segments/
    compare: bool = False  # every usable recognizer on each utterance
    translate: bool = True
    # Overrides for one run (file mode's --asr / --mt / --prompt); "" = settings.
    asr: str = ""
    mt: str = ""
    prompt: str = ""


# ---------------------------------------------------------------- sources


class MicSource:
    lossless = False  # live: never block the capture

    def __init__(self, device: str) -> None:
        self.device = device
        self._handle = None
        self.duration = None

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
    """Audio already in memory (a decoded file), fed in 100 ms chunks.
    `realtime` paces it at playing speed; otherwise as fast as the pipeline
    takes it. `pause()` / `resume()` hold the feed."""

    lossless = True  # a file run never drops a sentence

    def __init__(self, audio: np.ndarray, realtime: bool = False, chunk: int = 1600) -> None:
        self.audio = audio
        self.realtime = realtime
        self.chunk = chunk
        self.duration = len(audio) / SAMPLE_RATE
        self._stop = threading.Event()
        self._running = threading.Event()
        self._running.set()
        self._finished = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, out: queue.Queue) -> None:
        def feed() -> None:
            clock = time.perf_counter()
            for start in range(0, len(self.audio), self.chunk):
                if not self._running.is_set():
                    paused = time.perf_counter()
                    self._running.wait()
                    clock += time.perf_counter() - paused
                if self._stop.is_set():
                    break
                if self.realtime:
                    wait = clock + start / SAMPLE_RATE - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                out.put(self.audio[start : start + self.chunk])  # blocks: nothing is dropped
            self._finished.set()

        self._thread = threading.Thread(target=feed, name="pyvolis-file-source", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._running.clear()

    def resume(self) -> None:
        self._running.set()

    def done(self) -> bool:
        return self._finished.is_set()

    def stop(self) -> None:
        self._stop.set()
        self._running.set()
        if self._thread is not None:
            self._thread.join()


# ---------------------------------------------------------------- the pipeline


@dataclass
class Stats:
    """What the status line reports."""

    audio_seconds: float = 0.0  # speech the recognizer was given
    asr_seconds: float = 0.0  # time it took
    translate_ms: list[int] = field(default_factory=list)
    sentences: int = 0
    translated: int = 0
    not_translated: int = 0
    revisions: int = 0
    dropped: int = 0
    stalls_ms: list[int] = field(default_factory=list)

    def summary(self) -> dict:
        ms = sorted(self.translate_ms)

        def pct(q):
            return ms[min(len(ms) - 1, int(round(q * (len(ms) - 1))))] if ms else None

        return {
            "asr_rtf": round(self.asr_seconds / self.audio_seconds, 3) if self.audio_seconds else None,
            "translate_ms_median": pct(0.5),
            "translate_ms_p90": pct(0.9),
            "sentences": self.sentences,
            "translated": self.translated,
            "not_translated": self.not_translated,
            "revisions": self.revisions,
            "dropped": self.dropped,
            "worst_stall_ms": max(self.stalls_ms, default=0),
        }


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
        self.stats = Stats()
        self.recognizer_name = ""
        self.translator_name = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pyvolis-pipeline", daemon=True)

    def start(self) -> Pipeline:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def emit(self, event: Event) -> None:
        self.events.put(event)

    # -------------------------------------------------------------- run

    def _run(self) -> None:
        try:
            self._loop()
        except Exception as e:  # anything unforeseen reaches the consumer, not just the log
            log.exception("the pipeline stopped")
            self.emit(Error(str(e)))
        finally:
            self.emit(Summary(self.stats.summary()))
            self.emit(Stopped())

    def _recognizer(self, engines: list[models.Engine]):
        selected = (self.options.asr or self.config.asr.engine).strip()
        if not selected:
            raise asr_pkg.AsrError(f"[asr].engine is unset in {paths.config_file(self.root)}; run --report to see the models")
        engine = next((e for e in engines if e.dir_name == selected), None)
        if engine is None:  # never substitute a different one
            raise asr_pkg.AsrError(f'recognizer "{selected}" was not found in {paths.asr_dir(self.root)}; run --report')
        self.recognizer_name = selected
        return asr_pkg.load(engine)

    def _translator(self):
        from .translate import prompts

        entry = tr.choose(self.root, self.options.mt or self.pyconfig.translate.model)
        prompt = prompts.load(paths.prompts_dir(self.root), self.options.prompt or self.pyconfig.translate.prompt)
        self.translator_name = entry.id
        return tr.load(entry, prompt)

    def _loop(self) -> None:
        root, config = self.root, self.config
        language, target = config.languages.source, config.languages.target
        engines = [e for e in models.discover(paths.asr_dir(root), models.Role.ASR) if isinstance(e, models.Engine)]
        recognizer = self._recognizer(engines)
        recognizer.prepare(language)
        # Loaded here, on the pipeline thread before any audio, so a missing
        # or broken translator is an error now rather than a thread that dies later.
        translation = TranslationThread(self, self._translator(), target) if self.options.translate else None

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
        probe = StallProbe(self)

        def handle(segment: Segment) -> None:
            self._handle(segment, recognizer, language, ring, guards, comparison, segments_dir, translation)

        log.info('ready: listening continuously, "%s" hears "%s"', self.recognizer_name, language)
        probe.start()
        self.source.start(chunks)
        self.emit(Listening())
        speaking, consumed, reported = False, 0, 0.0
        try:
            while not self._stop.is_set():
                try:
                    chunk = chunks.get(timeout=0.1)
                except queue.Empty:
                    if self.source.done():
                        break
                    continue
                consumed += len(chunk)
                for segment in segmenter.push(chunk):
                    handle(segment)
                now = segmenter.speech_in_progress()
                if now and not speaking:
                    self.emit(SpeechStarted(start=consumed / SAMPLE_RATE))
                speaking = now
                if self.source.duration and consumed / SAMPLE_RATE - reported >= 1.0:
                    reported = consumed / SAMPLE_RATE
                    self.emit(Progress(reported, self.source.duration))
            # End of input: what's still queued, and whatever is still being said.
            while not self._stop.is_set() and not chunks.empty():
                for segment in segmenter.push(chunks.get()):
                    handle(segment)
            if not self._stop.is_set():
                for segment in segmenter.flush():
                    handle(segment)
            if self.source.duration:
                self.emit(Progress(self.source.duration, self.source.duration))
        finally:
            self.source.stop()
            if translation is not None:
                translation.finish(wait=not self._stop.is_set())
            probe.stop()
            recognizer.close()
            if comparison is not None:
                comparison.close()

    def _guards(self) -> guards_mod.Guards:
        g = self.pyconfig.guards
        scorer = guards_mod.SpeechScorer(paths.vad_model_file(self.root), g.min_peak_probability) if g.vad_probability else None
        phrases = guards_mod.load_phrases(paths.hallucinations_file(self.root)) if g.stock_phrases else {}
        return guards_mod.Guards(phrases, g.vad_probability, g.repeats, g.stock_phrases, scorer)

    def _handle(self, segment: Segment, recognizer, language, ring, guards, comparison, segments_dir, translation) -> None:
        utterance = ring.push(segment.start_ms(), segment.samples)
        start, end = segment.start_sample / SAMPLE_RATE, segment.start_sample / SAMPLE_RATE + len(segment.samples) / SAMPLE_RATE
        log.info("utterance %d: %d ms .. %d ms (%d ms)", utterance.index, segment.start_ms(),
                 segment.end_ms(), segment.duration_ms())
        # Logged every time: if the language were lost, Whisper would guess,
        # be right most of the time, and hide the bug.
        log.info('  transcribing as "%s" with "%s"', language, self.recognizer_name)
        try:
            result = recognizer.transcribe(utterance.pcm, language)
        except asr_pkg.AsrError as e:
            log.warning("  transcription failed: %s", e)
            self.emit(Error(f"utterance {utterance.index}: transcription failed: {e}"))
            result = None
        if result is not None:
            asr_ms = int(result.seconds * 1000)
            self.stats.audio_seconds += len(utterance.pcm) / SAMPLE_RATE
            self.stats.asr_seconds += result.seconds
            verdict = guards.check(result.text, language, utterance.pcm)
            for reason in verdict.reasons:
                log.info('  guard: %s: "%s"', reason, result.text)
            if verdict.dropped:
                log.info("  dropped: %s", "; ".join(verdict.reasons))
                self.stats.dropped += 1
                self.emit(Dropped(utterance.index, result.text, verdict.reasons, start))
            elif not verdict.text:
                log.info("  no words recognised in %d ms of audio (%d ms to decide)", utterance.duration_ms(), asr_ms)
                self.emit(NothingRecognized(utterance.index, utterance.duration_ms(), start))
            else:
                log.info("  [%s] %s\n  (%d ms to transcribe %d ms of audio)", language, verdict.text, asr_ms,
                         utterance.duration_ms())
                # Words were timed against the recognizer's own text; a guard
                # that changed the text makes them unreliable.
                words = result.words if not verdict.changed else None
                self.emit(Final(utterance.index, verdict.text, language, utterance.duration_ms(), asr_ms,
                                start, end, words))
                for sentence in sentences.split(verdict.text, utterance.index, start, end, words):
                    self.stats.sentences += 1
                    self.emit(SentenceMsg(sentence.id, sentence.utterance, sentence.text, language,
                                          sentence.start, sentence.end, sentence.approximate))
                    if translation is not None:
                        translation.submit(sentence, language)
        if comparison is not None:
            table = compare.run_all(comparison, utterance, language, guards)
            compare.report(table, self.recognizer_name)
            self.emit(ComparisonMsg(table))
        if self.options.write_wav:
            write_segment(utterance, segments_dir)


class TranslationThread:
    """Rust's spawn_translator: its own thread, a small queue."""

    def __init__(self, pipeline: Pipeline, translator, target: str) -> None:
        self.pipeline = pipeline
        self.translator = translator
        self.target = target
        self.queue: queue.Queue = queue.Queue(TRANSLATION_QUEUE)
        self._thread = threading.Thread(target=self._run, name="pyvolis-translate", daemon=True)
        self._thread.start()

    def submit(self, sentence: sentences.Sentence, source: str) -> None:
        job = (sentence, source)
        if self.pipeline.source.lossless:
            self.queue.put(job)  # a file waits rather than lose a sentence
            return
        try:
            self.queue.put_nowait(job)  # live: never block recognition
        except queue.Full:
            log.warning("translation is behind; sentence %s not translated", sentence.id)
            self.pipeline.stats.not_translated += 1
            self.pipeline.emit(NotTranslated(sentence.id, "translation fell behind the conversation"))

    def finish(self, wait: bool) -> None:
        """Stop after what's queued (wait) or at once."""
        if not wait:
            while not self.queue.empty():
                self.queue.get_nowait()
        self.queue.put(None)
        self._thread.join()
        self.translator.close()

    def _run(self) -> None:
        stats, emit = self.pipeline.stats, self.pipeline.emit
        while (job := self.queue.get()) is not None:
            sentence, source = job
            try:
                result = self.translator.translate(tr.TranslationRequest(sentence.text, source, self.target))
            except tr.Refused as e:
                log.warning("sentence %s: translation refused (%s): %s", sentence.id, e.guard, e)
                stats.not_translated += 1
                emit(NotTranslated(sentence.id, str(e), e.guard))
                continue
            except tr.TranslateError as e:
                log.warning("sentence %s: translation failed: %s", sentence.id, e)
                stats.not_translated += 1
                emit(NotTranslated(sentence.id, str(e)))
                continue
            if not result.text:
                stats.not_translated += 1
                emit(NotTranslated(sentence.id, "the translation came back empty"))
                continue
            ms = int(result.seconds * 1000)
            stats.translated += 1
            stats.translate_ms.append(ms)
            log.info("sentence %s\n  [%s] %s\n  [%s] %s\n  (%d ms to translate on the %s)", sentence.id, source,
                     sentence.text, self.target, result.text, ms, result.device.upper())
            emit(Translated(sentence.id, result.text, self.target, ms, result.device, self.pipeline.translator_name))
        log.info("translation stopped")


class StallProbe:
    """A 50 ms timer on its own thread. If it fires late, every Python thread
    was held up that long, which is what freezes a window."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pyvolis-stall-probe", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            began = time.perf_counter()
            time.sleep(STALL_TICK)
            late = int((time.perf_counter() - began - STALL_TICK) * 1000)
            if late >= STALL_REPORT_MS:
                self.pipeline.stats.stalls_ms.append(late)
                self.pipeline.emit(Stall(late))


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
    """Everything a run produced, for scripts, tests and the export."""

    events: list[Event] = field(default_factory=list)

    def of(self, kind) -> list:
        return [e for e in self.events if isinstance(e, kind)]

    @property
    def finals(self) -> list[Final]:
        return self.of(Final)

    @property
    def dropped(self) -> list[Dropped]:
        return self.of(Dropped)

    @property
    def nothing(self) -> list[NothingRecognized]:
        return self.of(NothingRecognized)

    @property
    def comparisons(self) -> list:
        return [e.comparison for e in self.of(ComparisonMsg)]

    @property
    def errors(self) -> list[str]:
        return [e.message for e in self.of(Error)]


def run_to_end(pipeline: Pipeline, events: queue.Queue, on_event=None) -> Collected:
    """Start a pipeline on a finite source and collect its events until it stops."""
    out = Collected()
    pipeline.start()
    while True:
        event = events.get()
        out.events.append(event)
        if on_event is not None:
            on_event(event)
        if isinstance(event, Stopped):
            break
    pipeline.join()
    return out
