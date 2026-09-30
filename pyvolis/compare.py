"""The dual-run comparison harness. Port of Rust `compare.rs`.

Runs one utterance through every usable segment recognizer and reports, side
by side: the transcript, the wall-clock time, and the segment's duration
(Whisper pads to 30 s internally, so time means nothing without it).

New in pyvolis: engines run through the hallucination guards too, and GPU
memory is managed. Rust keeps every engine loaded; on an 8 GB GPU the
downloaded models don't all fit at once. So each engine is loaded if it fits
beside the others and kept; one that doesn't fit is loaded for its turn and
released straight after ("swapped"). Engines always run one after another,
never at once, so the timings measure the models, not contention.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from . import asr as asr_pkg
from .models import Engine
from .ring import Utterance

log = logging.getLogger(__name__)


@dataclass
class Run:
    engine: str
    text: str
    elapsed_ms: int
    ok: bool
    note: str = ""  # guard verdict, or "loaded for this run"


@dataclass
class Comparison:
    utterance_index: int
    start_ms: int
    segment_ms: int
    runs: list[Run] = field(default_factory=list)


class Engines:
    """The recognizers `--compare` runs, loaded as memory allows."""

    def __init__(self, engines: list[Engine]) -> None:
        self._engines = [e for e in engines if e.kind == "segment" and e.enabled()]
        self._loaded: dict[str, object] = {}
        self._swapped: set[str] = set()
        for engine in self._engines:
            if self._swapped and not engine.from_engine_toml:
                # Already taking turns on the GPU: don't load this one now.
                self._swapped.add(engine.dir_name)
                continue
            began = time.perf_counter()
            try:
                self._loaded[engine.dir_name] = asr_pkg.load(engine)
                log.info('loaded "%s" in %d ms', engine.dir_name, (time.perf_counter() - began) * 1000)
            except asr_pkg.AsrMemoryError as e:
                # The GPU models don't all fit at once. Rather than keep the
                # first few and starve the rest, every GPU model takes its turn:
                # loaded for its run, released after. CPU models stay loaded.
                log.info("GPU memory is short (%s); GPU models will be loaded one at a time", e)
                for name in [n for n, r in self._loaded.items() if r.memory().gpu_bytes]:
                    self._loaded.pop(name).close()
                    self._swapped.add(name)
                self._swapped.add(engine.dir_name)
            except asr_pkg.AsrError as e:
                log.warning('cannot load "%s": %s', engine.dir_name, e)

    def __len__(self) -> int:
        return len(self._loaded) + len(self._swapped)

    def names(self) -> list[str]:
        return [e.dir_name for e in self._engines if e.dir_name in self._loaded or e.dir_name in self._swapped]

    def each(self):
        """(name, recognizer, swapped) for every engine, in discovery order. A
        swapped engine is loaded here and released when the caller moves on."""
        for engine in self._engines:
            name = engine.dir_name
            if name in self._loaded:
                yield name, self._loaded[name], False
            elif name in self._swapped:
                try:
                    recognizer = asr_pkg.load(engine)
                except asr_pkg.AsrError as e:
                    yield name, e, True
                    continue
                try:
                    yield name, recognizer, True
                finally:
                    recognizer.close()

    def close(self) -> None:
        for recognizer in self._loaded.values():
            recognizer.close()
        self._loaded.clear()


def run_all(engines: Engines, utterance: Utterance, language: str, guards=None) -> Comparison:
    """Run `utterance` through each engine in turn."""
    comparison = Comparison(utterance.index, utterance.start_ms, utterance.duration_ms())
    for name, recognizer, swapped in engines.each():
        if isinstance(recognizer, Exception):
            comparison.runs.append(Run(name, str(recognizer), 0, False))
            continue
        began = time.perf_counter()
        try:
            result = recognizer.transcribe(utterance.pcm, language)
        except asr_pkg.AsrError as e:
            comparison.runs.append(Run(name, str(e), int((time.perf_counter() - began) * 1000), False))
            continue
        elapsed = int((time.perf_counter() - began) * 1000)
        run = Run(name, result.text, elapsed, True, "loaded for this run" if swapped else "")
        if guards is not None:
            verdict = guards.check(result.text, language, utterance.pcm)
            if verdict.changed:
                run.note = "; ".join(filter(None, [run.note, "guards: " + "; ".join(verdict.reasons)]))
        comparison.runs.append(run)
    return comparison


def format_table(comparison: Comparison, selected_engine: str) -> str:
    """Rust's table, line for line, plus a note line where pyvolis has one."""
    lines = [f"utterance {comparison.utterance_index} at {comparison.start_ms} ms - "
             f"{comparison.segment_ms} ms of audio"]
    width = max((len(r.engine) for r in comparison.runs), default=0)
    for run in comparison.runs:
        realtime = f" ({run.elapsed_ms / comparison.segment_ms:.2f}x realtime)" if comparison.segment_ms > 0 else ""
        mark = "*" if run.engine == selected_engine else " "
        lines.append(f"  {mark} {run.engine:<{width}}  {run.elapsed_ms:>6} ms{realtime}")
        text = run.text if run.text else "(no text)"
        lines.append(f"      {'' if run.ok else 'FAILED: '}{text}")
        if run.note:
            lines.append(f"      ({run.note})")
    return "\n".join(lines) + "\n"


def report(comparison: Comparison, selected_engine: str) -> None:
    """Through the log, so it reaches the terminal and logs/."""
    log.info("\n%s", format_table(comparison, selected_engine))
