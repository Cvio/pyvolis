"""The pipeline: audio source -> VAD -> recognizer -> guards -> sentences ->
translator -> events.

Port of the continuous-listening core of Rust `pipeline.rs` (voice, turns,
sharing and pairing join it at P5 to P10). The pipeline reports only through
events (events.py) on a queue; `--listen`, file mode and the window are all
consumers of the same events, as in Rust.

Threads, as Rust, with the costs in mind:
  capture or file source   hands 16 kHz chunks on (audio.py, ArraySource)
  pipeline                 the detector only: it must never fall behind the audio
  recognition              its own thread: the final pass over each utterance
                           and, when streaming, a provisional pass over the
                           growing utterance every interval (LocalAgreement).
                           Live, if a provisional pass is still running when
                           the next is due, only the newest is kept; a fast
                           file run does every pass, so it measures what live
                           use would produce
  translation              its own thread, so a slow model never stalls
                           recognition; a queue of TRANSLATION_QUEUE sentences
  speaking                 synthesis on its own thread; the sound card's
                           callback plays it (playback.py), and the half-duplex
                           gate keeps the microphone from hearing it
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
from .asr.streaming import LocalAgreement
from .events import (  # noqa: F401 - re-exported for consumers
    ComparisonMsg, Dropped, Error, Event, Final, Held, Level, Listening, Loading, Mode, ModelLoaded, Revised,
    NothingRecognized, NotTranslated, Partial, Progress, SentenceMsg, SpeakingEnded, SpeakingStarted,
    SpeechStarted, Stall, Stopped, Summary, Translated, TurnCancelled, TurnEnded, TurnStarted,
)
from .translate import context as ctx
from .translate import revision as rev
from .ring import UtteranceRing
from .vad import Segment, Segmenter, VadSettings

log = logging.getLogger(__name__)

QUEUE_CHUNKS = 64  # source -> pipeline, about 6 s
TRANSLATION_QUEUE = 4  # Rust's TRANSLATION_QUEUE
SPEECH_QUEUE = 8  # sentences waiting to be spoken
LEVEL_EVENT = 0.2  # seconds between Level events (Rust's LEVEL_EVENT)
LEVEL_LOG = 5.0  # seconds between level lines in the log (Rust's LEVEL_LOG)
# The longest stretch handed to a recognizer at once (Rust's MAX_PART). Whisper
# hears 30 seconds and no more; a long turn is split at its pauses into parts
# below that, transcribed in order and joined.
MAX_PART = 25 * SAMPLE_RATE

CONTINUOUS, TURN = "continuous", "turn"


@dataclass(frozen=True)
class Command:
    """What a consumer asks of a running pipeline (Rust's PipelineCmd)."""

    name: str  # "set_mode" | "begin_turn" | "end_turn" | "cancel"
    mode: str = ""


def describe_mode(mode: str) -> str:
    return "listening continuously" if mode == CONTINUOUS else "waiting for a turn, microphone closed"
STALL_TICK = 0.05  # the stall probe's timer
STALL_REPORT_MS = 150  # lateness worth an event


@dataclass
class Options:
    write_wav: bool = False  # each utterance to logs/segments/
    compare: bool = False  # every usable recognizer on each utterance
    translate: bool = True
    # Voice output. None = [tts].enabled for the microphone, off for a file.
    speak: bool | None = None
    # P7, each None = pyvolis.toml: streaming recognition, context ("off" |
    # "carry"), holding short fragments; and the session glossary.
    streaming: bool | None = None
    context: str | None = None
    hold: bool | None = None
    glossary: list[str] = field(default_factory=list)
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
    revise_ms: list[int] = field(default_factory=list)  # each time the translator was asked again
    dropped: int = 0
    stalls_ms: list[int] = field(default_factory=list)
    passes: int = 0  # recognition passes, provisional ones included
    held: int = 0  # fragments held to be joined
    first_text_ms: list[int] = field(default_factory=list)  # speech start to first text shown, per utterance

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
            "revision_passes": len(self.revise_ms),
            "revise_ms_median": sorted(self.revise_ms)[len(self.revise_ms) // 2] if self.revise_ms else None,
            "dropped": self.dropped,
            "worst_stall_ms": max(self.stalls_ms, default=0),
            "asr_passes": self.passes,
            "fragments_held": self.held,
            "first_text_ms_median": sorted(self.first_text_ms)[len(self.first_text_ms) // 2] if self.first_text_ms else None,
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
        self.glossary = list(options.glossary)
        self._stop = threading.Event()
        self._commands: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="pyvolis-pipeline", daemon=True)

    def send(self, command: Command) -> None:
        """Ask the running pipeline for something: a mode, a turn, a cancel."""
        self._commands.put(command)

    def set_glossary(self, terms: list[str]) -> None:
        """Names and terms to keep exactly, from the next sentence on."""
        self.glossary = list(terms)

    def set_mode(self, mode: str) -> None:
        self.send(Command("set_mode", mode))

    def begin_turn(self) -> None:
        self.send(Command("begin_turn"))

    def end_turn(self) -> None:
        self.send(Command("end_turn"))

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
        self.emit(Loading(f"recognizer {selected}"))
        began = time.perf_counter()
        recognizer = asr_pkg.load(engine)
        # A GPU model's first pass is several times slower than the rest;
        # spend it here rather than on the first thing the user says.
        warm_up = getattr(recognizer, "warm_up", None)
        if warm_up is not None:
            try:
                warm_up(self.config.languages.source)
            except Exception as e:  # a warm-up that fails changes nothing
                log.debug("warm-up pass failed: %s", e)
        memory = recognizer.memory()
        self.emit(ModelLoaded("recognizer", selected, memory.device, memory.gpu_bytes, memory.cpu_bytes,
                              time.perf_counter() - began))
        return recognizer

    def _translator(self):
        from .translate import prompts

        entry = tr.choose(self.root, self.options.mt or self.pyconfig.translate.model)
        prompt = prompts.load(paths.prompts_dir(self.root), self.options.prompt or self.pyconfig.translate.prompt)
        self.translator_name = entry.id
        self.emit(Loading(f"translator {entry.id}"))
        began = time.perf_counter()
        translator = tr.load(entry, prompt, self.pyconfig.translate.device)
        on_gpu = translator.device == "cuda"
        self.emit(ModelLoaded("translator", entry.id, translator.device, entry.size_bytes if on_gpu else 0,
                              0 if on_gpu else entry.size_bytes, time.perf_counter() - began))
        return translator

    def _speaker(self, live: bool):
        """The voice, when this run speaks: the player, its gate, and the
        speaking thread. A file has no microphone to protect, so no gate."""
        from . import playback

        wanted = self.options.speak if self.options.speak is not None else (live and self.config.tts.enabled)
        gate = playback.Gate(enabled=live and self.config.tts.half_duplex)
        if not wanted or not self.options.translate or self.options.compare:
            return gate, None
        player = playback.Player(self.config.audio.output_device, gate, on_ended=lambda: self.emit(SpeakingEnded()))
        return gate, SpeakThread(self, player)

    def _loop(self) -> None:
        root, config = self.root, self.config
        language, target = config.languages.source, config.languages.target
        engines = [e for e in models.discover(paths.asr_dir(root), models.Role.ASR) if isinstance(e, models.Engine)]
        recognizer = self._recognizer(engines)
        recognizer.prepare(language)
        # Loaded here, on the pipeline thread before any audio, so a missing
        # or broken translator is an error now rather than a thread that dies later.
        live = not self.source.lossless
        gate, speaker = self._speaker(live)
        translation = TranslationThread(self, self._translator(), target, speaker) if self.options.translate else None
        if speaker is not None:
            speaker.prepare(target)

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

        py = self.pyconfig
        streaming = self.options.streaming if self.options.streaming is not None else py.asr.streaming
        interval = max(1, int(py.asr.interval_s * SAMPLE_RATE))
        hold = self.options.hold if self.options.hold is not None else py.fragments.hold
        # A fast file run has no wall clock worth the name: the file's own
        # time decides when a held fragment has waited long enough, and every
        # provisional pass is done rather than only the newest.
        fast_file = self.source.lossless and not getattr(self.source, "realtime", False)
        worker = AsrWorker(
            self, recognizer, language, ring, guards, comparison, segments_dir, translation,
            ctx.FragmentHolder(py.fragments.min_words, py.fragments.hold_ms / 1000, enabled=hold and translation is not None),
            every_pass=fast_file, source_clock=fast_file,
        )

        def handle(segment: Segment, parts=None, speech_seconds=None) -> None:
            worker.final(segment, parts, speech_seconds)

        # Continuous mode keeps the microphone open. Turn mode keeps the device
        # closed until a turn is taken and closes it again when the turn ends:
        # between turns the microphone is released, not merely ignored. A file
        # has no turns.
        mode = CONTINUOUS if not live or config.mode.kind == CONTINUOUS else TURN
        if live and config.mode.kind not in (CONTINUOUS, TURN):
            self.emit(Error(f'mode "{config.mode.kind}" arrives at a later milestone; taking turns instead'))
        control = speaker.player.control() if speaker is not None else None
        mic_open = False
        turn: _Turn | None = None

        def open_mic() -> None:
            nonlocal mic_open
            self.source.start(chunks)
            mic_open = True
            # The meters measure the open microphone, not the time it was closed.
            event_level.restart()
            log_level.restart()

        def close_mic() -> None:
            nonlocal mic_open
            if mic_open:
                self.source.stop()
                mic_open = False
            while not chunks.empty():  # what the device delivered after the turn ended
                chunks.get_nowait()

        def finish_turn(active: _Turn) -> None:
            """Close the microphone, then treat everything said in the turn as
            one utterance, trimmed of silence at either end."""
            close_mic()
            log.info("turn ended: microphone closed")
            self.emit(TurnEnded())
            if control is not None:
                control.end_turn()  # anything that arrived for the speaker during the turn plays now
            segments = active.segments + segmenter.flush()
            audio = np.concatenate(active.audio) if active.audio else np.zeros(0, np.float32)
            turn_ms = len(audio) * 1000 // SAMPLE_RATE
            found = trim_and_split(active.origin, len(audio), segments, MAX_PART)
            if found is None:
                log.info("turn: %d ms captured, no speech in it", turn_ms)
                utterance = ring.push(active.origin * 1000 // SAMPLE_RATE, audio)
                self.emit(NothingRecognized(utterance.index, turn_ms, active.origin / SAMPLE_RATE))
                return
            (first, last), parts = found
            log.info("turn: %d ms captured, %d ms of speech in %d part(s)", turn_ms,
                     (last - first) * 1000 // SAMPLE_RATE, len(parts))
            # The speech the detector heard, pauses left out (for the guards).
            speech = sum(len(seg.samples) for seg in segments) / SAMPLE_RATE
            handle(Segment(active.origin + first, audio[first:last], active.origin + first), parts, speech)

        event_level, log_level = LevelMeter(LEVEL_EVENT), LevelMeter(LEVEL_LOG)
        probe.start()
        if mode == CONTINUOUS:
            open_mic()
        log.info("ready: %s", describe_mode(mode))
        self.emit(Listening())
        self.emit(Mode(mode))
        speaking, consumed, reported = False, 0, 0.0
        last_pass = 0  # where the last provisional pass was asked for
        try:
            while not self._stop.is_set():
                # Commands first. With the microphone closed there is no audio
                # to wait on, so wait on commands; a turn opens the moment it
                # is asked for.
                pending = []
                try:
                    pending.append(self._commands.get(timeout=0.1) if not mic_open else self._commands.get_nowait())
                    while True:
                        pending.append(self._commands.get_nowait())
                except queue.Empty:
                    pass
                for command in pending:
                    if not live:
                        continue  # a file has no turns and one mode
                    if command.name == "set_mode" and command.mode in (CONTINUOUS, TURN) and command.mode != mode:
                        # Finish whatever the old mode had in progress first.
                        if turn is not None:
                            finish_turn(turn)
                            turn = None
                        else:
                            for segment in segmenter.flush():
                                handle(segment)
                        close_mic()
                        mode = command.mode
                        segmenter.reset(consumed)
                        speaking = False
                        if mode == CONTINUOUS:
                            open_mic()
                        log.info("now %s", describe_mode(mode))
                        self.emit(Mode(mode))
                    elif command.name == "begin_turn" and mode == TURN and turn is None:
                        if control is not None:
                            control.begin_turn()  # stop any reply mid-word, hold what arrives
                        try:
                            open_mic()
                        except Exception as e:  # a device that won't open fails this turn, not the session
                            if control is not None:
                                control.end_turn()
                            log.warning("cannot start a turn: %s", e)
                            self.emit(Error(f"cannot start a turn: {e}"))
                            continue
                        log.info("turn started: microphone open")
                        segmenter.reset(consumed)
                        speaking = False
                        turn = _Turn(consumed)
                        self.emit(TurnStarted())
                    elif command.name == "end_turn" and turn is not None:
                        finish_turn(turn)
                        turn = None
                    elif command.name == "cancel":
                        if turn is not None:
                            close_mic()
                            segmenter.reset(consumed)
                            speaking = False
                            log.info("turn cancelled: %d ms of audio discarded, microphone closed",
                                     sum(map(len, turn.audio)) * 1000 // SAMPLE_RATE)
                            turn = None
                        if translation is not None:
                            translation.cancel()
                        if control is not None:
                            control.stop()
                            control.end_turn()
                        self.emit(TurnCancelled())
                if not mic_open:
                    continue

                try:
                    chunk = chunks.get(timeout=0.02 if live else 0.1)
                except queue.Empty:
                    if self.source.done():
                        break
                    continue
                consumed += len(chunk)
                # Half-duplex: while our own speech is playing, captured audio
                # is discarded and the detector held reset, so nothing of it
                # survives into the next utterance.
                if gate.is_closed():
                    # A turn holds playback, so this should not happen during
                    # one; if it does, silence keeps the turn's audio aligned
                    # with the timeline its segments are stamped on.
                    if turn is not None:
                        turn.audio.append(np.zeros(len(chunk), np.float32))
                    segmenter.reset(consumed)
                    speaking = False
                    continue
                if live:
                    event_level.observe(chunk)
                    log_level.observe(chunk)
                    if (due := event_level.take_if_due()) is not None:
                        self.emit(Level(due[0]))
                    if (due := log_level.take_if_due()) is not None:
                        if due[0] is None:
                            log.warning("input level: digital silence over the last %d s - is the microphone muted?", LEVEL_LOG)
                        else:
                            log.info("input level: peak %.1f dBFS over the last %d s", due[0], LEVEL_LOG)
                segments = segmenter.push(chunk)
                worker.source_time = consumed / SAMPLE_RATE
                if turn is not None:
                    # In a turn the user decides where the utterance ends; the
                    # detector's segments only mark where the speech is.
                    turn.audio.append(chunk)
                    turn.segments.extend(segments)
                else:
                    for segment in segments:
                        handle(segment)
                now = segmenter.speech_in_progress()
                if now and not speaking:
                    self.emit(SpeechStarted(start=consumed / SAMPLE_RATE))
                    # The first pass comes after half the interval, so text
                    # is on screen within about 1.5 s of speech; then every interval.
                    last_pass = consumed - interval // 2
                speaking = now
                # Streaming: every interval while the speech goes on, a pass
                # over the utterance so far. In a turn the user decides the
                # boundary, so there is nothing provisional to show.
                if streaming and now and turn is None and consumed - last_pass >= interval:
                    last_pass = consumed
                    snapshot = segmenter.current()
                    if snapshot is not None:
                        worker.partial(snapshot)
                if self.source.duration and consumed / SAMPLE_RATE - reported >= 1.0:
                    reported = consumed / SAMPLE_RATE
                    self.emit(Progress(reported, self.source.duration))
            # End of input: what's still queued, and whatever is still being said.
            while not self._stop.is_set() and mic_open and not chunks.empty():
                for segment in segmenter.push(chunks.get()):
                    handle(segment)
            if turn is not None:
                finish_turn(turn)
            elif not self._stop.is_set() or live:
                for segment in segmenter.flush():
                    handle(segment)
            if self.source.duration:
                self.emit(Progress(self.source.duration, self.source.duration))
        finally:
            self.source.stop()
            worker.finish(wait=not self._stop.is_set())
            if translation is not None:
                translation.finish(wait=not self._stop.is_set())
            if speaker is not None:
                speaker.finish(wait=not self._stop.is_set())
            probe.stop()
            recognizer.close()
            if comparison is not None:
                comparison.close()

    def _guards(self) -> guards_mod.Guards:
        g = self.pyconfig.guards
        scorer = guards_mod.SpeechScorer(paths.vad_model_file(self.root), g.min_peak_probability) if g.vad_probability else None
        phrases = guards_mod.load_phrases(paths.hallucinations_file(self.root)) if g.stock_phrases else {}
        return guards_mod.Guards(phrases, g.vad_probability, g.repeats, g.stock_phrases, scorer,
                                 g.sparse, g.min_words_per_second, g.sparse_min_seconds)


@dataclass
class _Turn:
    """A turn in progress: everything captured since it began, and where the
    detector heard speech in it."""

    origin: int  # capture position of the turn's first sample
    audio: list = field(default_factory=list)
    segments: list = field(default_factory=list)


def trim_and_split(origin: int, length: int, segments: list[Segment], max_part: int):
    """Where the speech is in a turn, and how to split it for the recognizer
    (Rust's trim_and_split). Returns ((first, last), parts): the span from the
    first speech to the last, relative to the start of the turn's audio, and
    that span divided into parts no longer than `max_part`, split at the pauses
    between segments; parts are relative to the span. None when the detector
    heard no speech at all."""
    spans = []
    for segment in segments:
        start = segment.start_sample - origin
        if start < 0:
            continue
        end = min(start + len(segment.samples), length)
        if start < end:
            spans.append((start, end))
    if not spans:
        return None
    first, last = spans[0][0], max(end for _, end in spans)
    parts, current = [], list(spans[0])
    for start, end in spans[1:]:
        if end - current[0] > max_part:
            parts.append(tuple(current))
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    parts.append(tuple(current))
    return (first, last), [(a - first, b - first) for a, b in parts]


def transcribe_parts(recognizer, pcm: np.ndarray, parts, language: str, timestamps: bool = True) -> asr_pkg.AsrResult:
    """Transcribe an utterance, part by part when it has been split, joining
    the texts and shifting each part's word timings onto the utterance."""
    if not parts or len(parts) <= 1:
        return recognizer.transcribe(pcm, language, timestamps)
    texts, words, seconds, timed = [], [], 0.0, True
    for start, end in parts:
        result = recognizer.transcribe(pcm[start:end], language, timestamps)
        seconds += result.seconds
        if not result.text.strip():
            continue
        texts.append(result.text.strip())
        if result.words is None:
            timed = False
        else:
            offset = start / SAMPLE_RATE
            words += [asr_pkg.Word(w.text, w.start + offset, w.end + offset) for w in result.words]
    return asr_pkg.AsrResult(" ".join(texts), words if timed and words else None, None,
                             language.split("-")[0], seconds)


@dataclass
class _Stream:
    """An utterance being recognised while it is still being spoken."""

    key: int  # its first sample on the capture timeline
    index: int  # the utterance index it will have
    agreement: LocalAgreement
    began: float  # wall time of its first pass
    pending: list = field(default_factory=list)  # committed words not yet in a sentence
    committed_text: str = ""
    sentences: int = 0
    last_end: float | None = None  # where the last emitted sentence ended
    seconds: float = 0.0
    shown: bool = False  # text has reached the screen


class AsrWorker:
    """Recognition on its own thread: the final pass over each utterance, and
    provisional passes while streaming. Everything recognised leaves here as
    sentences, through the fragment holder, to the translator."""

    def __init__(self, pipeline: Pipeline, recognizer, language: str, ring, guards, comparison, segments_dir,
                 translation, holder: ctx.FragmentHolder, every_pass: bool, source_clock: bool) -> None:
        self.pipeline, self.recognizer, self.language = pipeline, recognizer, language
        self.ring, self.guards, self.comparison, self.segments_dir = ring, guards, comparison, segments_dir
        self.translation, self.holder = translation, holder
        self.every_pass = every_pass
        self.source_clock = source_clock
        # Word timings are for a file's subtitles. Live, nothing uses them, and
        # for Whisper they cost about a second an utterance.
        self.words = pipeline.source.lossless
        self.source_time = 0.0  # set by the pipeline thread
        # Final passes queue up; a file waits rather than run ahead of them.
        self.queue: queue.Queue = queue.Queue(4 if pipeline.source.lossless else 0)
        self._latest = None  # the newest provisional snapshot not yet taken (live)
        self._lock = threading.Lock()
        self._stream: _Stream | None = None
        self._thread = threading.Thread(target=self._run, name="pyvolis-asr", daemon=True)
        self._thread.start()

    def clock(self) -> float:
        return self.source_time if self.source_clock else time.monotonic()

    # ---- called from the pipeline thread

    def final(self, segment: Segment, parts=None, speech_seconds=None) -> None:
        with self._lock:
            self._latest = None  # a provisional pass over an utterance that has ended is stale
        self.queue.put(("final", segment, parts, speech_seconds, time.monotonic()))

    def partial(self, snapshot: Segment) -> None:
        if self.every_pass:
            self.queue.put(("partial", snapshot))
        else:
            with self._lock:
                self._latest = snapshot  # only the newest is worth transcribing

    def finish(self, wait: bool) -> None:
        if not wait:
            with self._lock:
                self._latest = None
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
        self.queue.put(None)
        self._thread.join()

    # ---- the thread

    def _run(self) -> None:
        while True:
            try:
                job = self.queue.get(timeout=0.05)
            except queue.Empty:
                with self._lock:
                    snapshot, self._latest = self._latest, None
                job = ("partial", snapshot) if snapshot is not None else "idle"
            if job is None:
                break
            try:
                if job == "idle":
                    pass
                elif job[0] == "partial":
                    self._partial(job[1])
                else:
                    self._final(*job[1:])
                for ready in self.holder.due(self.clock()):
                    self._send(*ready)
            except Exception as e:  # one utterance failing must not end recognition
                log.exception("recognition failed")
                self.pipeline.emit(Error(f"recognition failed: {e}"))
        for ready in self.holder.flush():
            self._send(*ready)

    def _partial(self, snapshot: Segment) -> None:
        stream = self._stream
        if stream is None or stream.key != snapshot.start_sample:
            stream = self._stream = _Stream(snapshot.start_sample, self.ring.next_index(),
                                            LocalAgreement(self.recognizer, self.language, self.words),
                                            time.monotonic())
        try:
            update = stream.agreement.update(snapshot.samples)
        except asr_pkg.AsrError as e:
            log.warning("  a provisional pass failed: %s", e)
            return
        self._count(update.seconds)
        stream.seconds += update.seconds
        now_end = (snapshot.start_sample + len(snapshot.samples)) / SAMPLE_RATE
        self._commit(stream, update, snapshot.start_sample / SAMPLE_RATE, now_end, final=False)
        self._shown(stream, snapshot, update)
        self.pipeline.emit(Partial(stream.index, update.provisional, stream.committed_text,
                                   " ".join(w.text for w, _ in stream.pending)))

    def _shown(self, stream: _Stream, snapshot: Segment, update) -> None:
        """The first time an utterance has text on the screen: how long after
        it began (on a paced source, where that means something)."""
        if stream.shown or not (update.provisional or stream.committed_text):
            return
        stream.shown = True
        if not self.source_clock:
            heard = (snapshot.start_sample + len(snapshot.samples) - snapshot.detected_sample) / SAMPLE_RATE
            self.pipeline.stats.first_text_ms.append(int((heard + update.seconds) * 1000))

    def _count(self, seconds: float) -> None:
        self.pipeline.stats.asr_seconds += seconds
        self.pipeline.stats.passes += 1

    def _final(self, segment: Segment, parts, speech_seconds, cut_at: float) -> None:
        pipeline, language = self.pipeline, self.language
        stream, self._stream = self._stream, None
        if stream is not None and stream.key != segment.start_sample:
            stream = None  # a different utterance: its passes don't apply
        utterance = self.ring.push(segment.start_ms(), segment.samples)
        start = segment.start_sample / SAMPLE_RATE
        end = start + len(segment.samples) / SAMPLE_RATE
        log.info("utterance %d: %d ms .. %d ms (%d ms)", utterance.index, segment.start_ms(),
                 segment.end_ms(), segment.duration_ms())
        # Logged every time: if the language were lost, Whisper would guess,
        # be right most of the time, and hide the bug.
        log.info('  transcribing as "%s" with "%s"', language, pipeline.recognizer_name)
        pipeline.stats.audio_seconds += len(utterance.pcm) / SAMPLE_RATE
        if stream is not None:
            self._final_streamed(stream, utterance, start, end, cut_at)
        else:
            self._final_whole(utterance, parts, speech_seconds, start, end, cut_at)
        if self.comparison is not None:
            table = compare.run_all(self.comparison, utterance, language, self.guards)
            compare.report(table, pipeline.recognizer_name)
            pipeline.emit(ComparisonMsg(table))
        if pipeline.options.write_wav:
            write_segment(utterance, self.segments_dir)

    def _final_whole(self, utterance, parts, speech_seconds, start: float, end: float, cut_at: float) -> None:
        """One pass over a finished utterance: Rust's behaviour."""
        pipeline, language = self.pipeline, self.language
        try:
            result = transcribe_parts(self.recognizer, utterance.pcm, parts, language, self.words)
        except asr_pkg.AsrError as e:
            log.warning("  transcription failed: %s", e)
            pipeline.emit(Error(f"utterance {utterance.index}: transcription failed: {e}"))
            return
        asr_ms = int(result.seconds * 1000)
        self._count(result.seconds)
        verdict = self.guards.check(result.text, language, utterance.pcm, speech_seconds)
        for reason in verdict.reasons:
            log.info('  guard: %s: "%s"', reason, result.text)
        if verdict.dropped:
            log.info("  dropped: %s", "; ".join(verdict.reasons))
            pipeline.stats.dropped += 1
            pipeline.emit(Dropped(utterance.index, result.text, verdict.reasons, start))
        elif not verdict.text:
            log.info("  no words recognised in %d ms of audio (%d ms to decide)", utterance.duration_ms(), asr_ms)
            pipeline.emit(NothingRecognized(utterance.index, utterance.duration_ms(), start))
        else:
            log.info("  [%s] %s\n  (%d ms to transcribe %d ms of audio)", language, verdict.text, asr_ms,
                     utterance.duration_ms())
            # Words were timed against the recognizer's own text; a guard
            # that changed the text makes them unreliable.
            words = result.words if not verdict.changed else None
            pipeline.emit(Final(utterance.index, verdict.text, language, utterance.duration_ms(), asr_ms,
                                start, end, words))
            for sentence in sentences.split(verdict.text, utterance.index, start, end, words):
                self._release(sentence, cut_at)

    def _final_streamed(self, stream: _Stream, utterance, start: float, end: float, cut_at: float) -> None:
        """The last pass of an utterance that was recognised as it was spoken:
        whatever is not yet committed is committed now."""
        pipeline, language = self.pipeline, self.language
        try:
            update = stream.agreement.finish(utterance.pcm)
        except asr_pkg.AsrError as e:
            log.warning("  transcription failed: %s", e)
            pipeline.emit(Error(f"utterance {utterance.index}: transcription failed: {e}"))
            return
        self._count(update.seconds)
        stream.seconds += update.seconds
        self._commit(stream, update, start, end, final=True, cut_at=cut_at)
        text = stream.committed_text
        log.info("  [%s] %s\n  (%d passes, %d ms in all, to transcribe %d ms of audio)", language, text,
                 stream.agreement.passes, stream.seconds * 1000, utterance.duration_ms())
        if stream.sentences == 0:
            pipeline.emit(NothingRecognized(utterance.index, utterance.duration_ms(), start))
        else:
            pipeline.emit(Final(utterance.index, text, language, utterance.duration_ms(),
                                int(stream.seconds * 1000), start, end, None))

    def _commit(self, stream: _Stream, update, start: float, now_end: float, final: bool, cut_at: float = 0.0) -> None:
        """Newly committed words join the pending ones; every complete
        sentence among them goes on its way (all of them when `final`)."""
        if not update.committed and not final:
            return
        stream.pending.extend((w, update.timed) for w in update.committed)
        if update.committed:
            stream.committed_text = (stream.committed_text + " " + update.text).strip()
        text = " ".join(w.text for w, _ in stream.pending)
        pieces = sentences.split_text(text)
        if not final and pieces and not pieces[-1].rstrip("\"'»”’)]").endswith(sentences.SENTENCE_END):
            pieces = pieces[:-1]  # the last one isn't finished yet
        for piece in pieces:
            count = len(piece.split())
            taken, stream.pending = stream.pending[:count], stream.pending[count:]
            timed = all(t for _, t in taken)
            if timed:
                s_start, s_end = start + taken[0][0].start, start + taken[-1][0].end
            else:
                # No word timings: from where the last sentence ended to the
                # audio heard so far.
                s_start, s_end = (stream.last_end if stream.last_end is not None else start), now_end
            stream.last_end = s_end
            # The audio-based guards need the whole utterance; here, the text ones.
            verdict = self.guards.check(piece, self.language)
            if verdict.dropped or not verdict.text:
                log.info("  dropped: %s: %s", "; ".join(verdict.reasons), piece)
                self.pipeline.stats.dropped += 1
                self.pipeline.emit(Dropped(stream.index, piece, verdict.reasons, s_start))
                continue
            stream.sentences += 1
            sentence = sentences.Sentence(f"{stream.index}.{stream.sentences}", stream.index, verdict.text,
                                          round(s_start, 3), round(s_end, 3), not timed)
            self._release(sentence, cut_at or time.monotonic())

    def _release(self, sentence, cut_at: float) -> None:
        """A committed sentence: straight on, or held if it is a fragment."""
        ready, held = self.holder.offer(sentence, self.language, self.clock(), cut_at)
        if held is not None:
            self.pipeline.stats.held += 1
            log.info("  holding the fragment %r to join it to what follows", held.text)
            self.pipeline.emit(Held(held.id, held.text))
        for item in ready:
            self._send(*item)

    def _send(self, sentence, source: str, cut_at: float) -> None:
        self.pipeline.stats.sentences += 1
        self.pipeline.emit(SentenceMsg(sentence.id, sentence.utterance, sentence.text, source,
                                       sentence.start, sentence.end, sentence.approximate))
        if self.translation is not None:
            self.translation.submit(sentence, source, cut_at)


class TranslationThread:
    """Rust's spawn_translator: its own thread, a small queue."""

    def __init__(self, pipeline: Pipeline, translator, target: str, speaker=None) -> None:
        self.pipeline = pipeline
        self.translator = translator
        self.target = target
        self.speaker = speaker
        py = pipeline.pyconfig.context
        mode = pipeline.options.context if pipeline.options.context is not None else py.mode
        # Carry-forward: each sentence is translated knowing what came before.
        self.history = ctx.History(py.sentences if mode in ("carry", "revision") else 0, py.token_budget)
        # Revision: and the last few are translated again once a new one arrives.
        self.reviser = rev.Reviser(py.revise_sentences, py.revise_max_age_s, py.revise_max_words) if mode == "revision" else None
        self._count = getattr(translator, "count_tokens", lambda text: len(text) // 3)
        self.queue: queue.Queue = queue.Queue(TRANSLATION_QUEUE)
        self._thread = threading.Thread(target=self._run, name="pyvolis-translate", daemon=True)
        self._thread.start()

    def submit(self, sentence: sentences.Sentence, source: str, cut_at: float = 0.0) -> None:
        job = (sentence, source, cut_at)
        if self.pipeline.source.lossless:
            self.queue.put(job)  # a file waits rather than lose a sentence
            return
        try:
            self.queue.put_nowait(job)  # live: never block recognition
        except queue.Full:
            log.warning("translation is behind; sentence %s not translated", sentence.id)
            self.pipeline.stats.not_translated += 1
            self.pipeline.emit(NotTranslated(sentence.id, "translation fell behind the conversation"))

    def cancel(self) -> None:
        """Drop what is queued for translation (a cancelled turn)."""
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return

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
            sentence, source, cut_at = job
            request = tr.TranslationRequest(sentence.text, source, self.target,
                                            self.history.context(self._count), list(self.pipeline.glossary))
            try:
                result = self.translator.translate(request)
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
            self.history.add(sentence.text, result.text)
            stats.translated += 1
            stats.translate_ms.append(ms)
            log.info("sentence %s\n  [%s] %s\n  [%s] %s\n  (%d ms to translate on the %s)", sentence.id, source,
                     sentence.text, self.target, result.text, ms, result.device.upper())
            emit(Translated(sentence.id, result.text, self.target, ms, result.device, self.pipeline.translator_name))
            spoken = self.speaker is not None
            if spoken:
                self.speaker.submit(sentence.id, result.text, self.target, cut_at)
            if self.reviser is not None:
                self.reviser.add(rev.Done(sentence.id, sentence.text, source, result.text, sentence.end, spoken))
                self._revise()
        log.info("translation stopped")

    def _revise(self) -> None:
        """Translate the last few sentences again together and replace what
        changed. Skipped while a live conversation has sentences waiting:
        new speech comes before second thoughts."""
        if not self.pipeline.source.lossless and not self.queue.empty():
            return
        began = time.perf_counter()
        before = self.reviser.passes
        changes = self.reviser.revise(self.translator, self.history.context(self._count), self.target,
                                      list(self.pipeline.glossary))
        if self.reviser.passes > before:
            self.pipeline.stats.revise_ms.append(int((time.perf_counter() - began) * 1000))
        if not changes:
            return
        self.history.replace_last([d.translation for d in self.reviser.window()])
        for change in changes:
            self.pipeline.stats.revisions += 1
            log.info("sentence %s revised\n  was: %s\n  now: %s", change.id, change.old, change.new)
            self.pipeline.emit(Revised(change.id, change.old, change.new))


class SpeakThread:
    """Rust's spawn_speaker: synthesis and playback for everything this PC
    says. A voice is chosen per sentence by its language and loaded the first
    time it is needed."""

    def __init__(self, pipeline: Pipeline, player) -> None:
        self.pipeline = pipeline
        self.player = player
        self.voices = [e for e in models.discover(paths.tts_dir(pipeline.root), models.Role.TTS)
                       if isinstance(e, models.Engine)]
        self.loaded: dict = {}
        self.queue: queue.Queue = queue.Queue(SPEECH_QUEUE)
        self._thread = threading.Thread(target=self._run, name="pyvolis-speak", daemon=True)
        self._thread.start()

    def prepare(self, language: str) -> None:
        """Load the voice expected, at start, so the first sentence doesn't wait.
        A language with no voice is reported now, not at the first sentence."""
        try:
            self._voice(language)
        except Exception as e:
            log.warning("%s", e)
            self.pipeline.emit(Error(f"translations into {language} will not be spoken: {e}"))

    def _voice(self, language: str):
        from . import tts

        engine = tts.for_language(self.voices, language)
        if engine.dir_name not in self.loaded:
            self.pipeline.emit(Loading(f"voice {engine.dir_name}"))
            began = time.perf_counter()
            self.loaded[engine.dir_name] = tts.Voice(engine)
            size = sum(f.path.stat().st_size for f in engine.files if f.present)
            self.pipeline.emit(ModelLoaded("voice", engine.dir_name, "cpu", 0, size, time.perf_counter() - began))
        return self.loaded[engine.dir_name]

    def submit(self, sentence_id: str, text: str, language: str, cut_at: float) -> None:
        try:
            self.queue.put_nowait((sentence_id, text, language, cut_at))
        except queue.Full:
            log.warning("speech is behind; sentence %s not spoken", sentence_id)
            self.pipeline.emit(Error(f"speech fell behind the conversation; sentence {sentence_id} was not spoken"))

    def finish(self, wait: bool) -> None:
        if not wait:
            while not self.queue.empty():
                self.queue.get_nowait()
            self.player.control().stop()
        self.queue.put(None)
        self._thread.join()
        if wait:  # let what is queued for the sound card play out
            while self.player.queued() > 0:
                time.sleep(0.05)
        self.player.close()

    def _run(self) -> None:
        emit = self.pipeline.emit
        while (job := self.queue.get()) is not None:
            sentence_id, text, language, cut_at = job
            label = f"sentence {sentence_id}"
            try:
                voice = self._voice(language)
                began = time.perf_counter()
                speech = voice.speak(text)
            except Exception as e:
                log.warning("%s was not spoken: %s", label, e)
                emit(Error(f"{label} was not spoken: {e}"))
                continue
            if len(speech.samples) == 0:
                log.warning("%s: the voice produced no audio", label)
                emit(Error(f"{label}: the voice produced no audio"))
                continue
            synthesised_ms = int((time.perf_counter() - began) * 1000)
            self.player.play(speech.samples, speech.sample_rate)
            # From the moment the utterance was cut, through recognition,
            # translation and synthesis, to the sound card.
            first_audio_ms = int((time.monotonic() - cut_at) * 1000) if cut_at else synthesised_ms
            log.info("  speaking %s: %d ms (%d ms to synthesise, %d ms to first audio)", label,
                     speech.duration_ms(), synthesised_ms, first_audio_ms)
            emit(SpeakingStarted(sentence_id, first_audio_ms, voice.name))
        log.info("speaking stopped")


class LevelMeter:
    """Peak input level over a window (Rust's LevelMeter)."""

    def __init__(self, every: float) -> None:
        self.every = every
        self.peak = 0.0
        self.since = time.monotonic()

    def restart(self) -> None:
        self.peak, self.since = 0.0, time.monotonic()

    def observe(self, chunk: np.ndarray) -> None:
        if len(chunk):
            self.peak = max(self.peak, float(np.abs(chunk).max()))

    def take_if_due(self):
        """When the window has elapsed: a one-item tuple holding the peak in
        dBFS, or None for digital silence. Otherwise None."""
        if time.monotonic() - self.since < self.every:
            return None
        peak, self.peak, self.since = self.peak, 0.0, time.monotonic()
        return (20.0 * float(np.log10(peak)) if peak > 0 else None,)


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
