"""`--file`: translate an audio file through the live pipeline, with no window.

    pyvolis --file talk.m4a --from es-MX --to en --asr <folder> --mt <id> --fast --export out\\

The file becomes the audio source in place of the microphone; everything
after it is the same pipeline as live use (VAD, pre-roll, recognizer, guards,
sentences, translator), so a file run measures what live use will do. It
writes the export folder, prints the scores if a reference sits next to the
file, and exits non-zero on failure. model-bench drives pyvolis through this.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import sys
from pathlib import Path

from . import export, paths, scoring
from .config import Config, ConfigError, PyvolisConfig
from .events import Error, Final, NotTranslated, Progress, SentenceMsg, Stall, Summary, Translated, restart_clock
from .filesource import FileSourceError, read_16k_mono
from .pipeline import ArraySource, Options, Pipeline, run_to_end

log = logging.getLogger(__name__)


@dataclasses.dataclass
class FileRun:
    path: Path
    source: str = ""
    target: str = ""
    asr: str = ""
    mt: str = ""
    prompt: str = ""
    fast: bool = False
    export: Path | None = None
    translate: bool = True


def run(root: Path, config: Config, job: FileRun) -> int:
    from . import varieties

    try:
        pyconfig, _ = PyvolisConfig.load(paths.pyvolis_config_file(root))
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    languages = dataclasses.replace(
        config.languages, source=job.source or config.languages.source, target=job.target or config.languages.target
    )
    for tag in (languages.source, languages.target):
        try:
            varieties.require(tag)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    config = dataclasses.replace(config, languages=languages)
    try:
        audio = read_16k_mono(job.path)
    except FileSourceError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    try:
        reference = scoring.find_reference(job.path)
    except scoring.ReferenceError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    options = Options(translate=job.translate, asr=job.asr, mt=job.mt, prompt=job.prompt)
    source = ArraySource(audio, realtime=not job.fast)
    print(f"{job.path.absolute()}: {source.duration:.1f} s, {languages.source} -> {languages.target}, "
          f"{'fast' if job.fast else 'real time'}")
    restart_clock()
    events: queue.Queue = queue.Queue()
    pipeline = Pipeline(root, config, options, events, source, pyconfig)
    run_out = run_to_end(pipeline, events, on_event=_show)
    errors = run_out.errors
    if errors and not run_out.of(Final):
        print(f"Error: {errors[0]}", file=sys.stderr)
        return 1

    prompt_file = paths.prompts_dir(root) / f"{options.prompt or pyconfig.translate.prompt}.txt"
    configuration = export.configuration(
        root, audio=job.path, config=config, pyconfig=pyconfig, options=options,
        recognizer=pipeline.recognizer_name, translator=pipeline.translator_name if job.translate else "",
        prompt_file=prompt_file if job.translate else None, cli=dataclasses.asdict(job) | {"path": str(job.path)},
    )
    folder = job.export or export.default_folder(root, job.path)
    try:
        written = export.write(folder, run_out.events, configuration)
    except OSError as e:
        print(f"Error: cannot write the export to {folder.absolute()}: {e}", file=sys.stderr)
        return 1

    summary = next((e.stats for e in reversed(run_out.events) if isinstance(e, Summary)), {})
    print()
    print(status_line(summary))
    if reference is not None:
        print(scores(reference, run_out, languages.source, languages.target))
    print(f"exported to {folder.absolute()}")
    for name in written:
        print(f"  {name}")
    return 1 if errors else 0


def _show(event) -> None:
    """The run as it happens, on the console."""
    if isinstance(event, SentenceMsg):
        print(f"[{event.start:7.1f}s] {event.id:>6} [{event.lang}] {event.text}", flush=True)
    elif isinstance(event, Translated):
        print(f"{'':10} {event.id:>6} [{event.lang}] {event.text}   ({event.translate_ms} ms, {event.device})", flush=True)
    elif isinstance(event, NotTranslated):
        print(f"{'':10} {event.id:>6} not translated: {event.reason}", flush=True)
    elif isinstance(event, Error):
        print(f"error: {event.message}", flush=True)
    elif isinstance(event, Stall):
        log.warning("the pipeline's threads were held up for %d ms", event.late_ms)
    elif isinstance(event, Progress) and event.position >= event.duration:
        print(f"{'':10} end of file ({event.duration:.1f} s)", flush=True)


def status_line(stats: dict) -> str:
    def ms(value):
        return f"{value} ms" if value is not None else "-"

    rtf = stats.get("asr_rtf")
    warn = "  (above 0.8: streaming couldn't keep up)" if rtf and rtf > 0.8 else ""
    return (
        f"ASR real-time factor {rtf if rtf is not None else '-'}{warn} | translation median "
        f"{ms(stats.get('translate_ms_median'))}, 90th percentile {ms(stats.get('translate_ms_p90'))} | "
        f"sentences {stats.get('sentences', 0)} | not translated {stats.get('not_translated', 0)} | "
        f"revisions {stats.get('revisions', 0)} | dropped hallucinations {stats.get('dropped', 0)} | "
        f"worst stall {ms(stats.get('worst_stall_ms'))}"
    )


def scores(reference: scoring.Reference, run_out, source: str, target: str) -> str:
    sentences = run_out.of(SentenceMsg)
    translations = {e.id: e.text for e in run_out.of(Translated)}
    lines = [f"scores against {reference.source.name} (quick check, model-bench's cleaning):"]
    if reference.transcript is not None:
        hyp = " ".join(s.text for s in sentences)
        lines.append(f"  transcript CER {scoring.cer(reference.transcript, hyp, source):.1f}%, "
                     f"WER {scoring.wer(reference.transcript, hyp, source):.1f}%")
    if reference.translation is not None:
        hyp = " ".join(translations.get(s.id, "") for s in sentences)
        lines.append(f"  translation chrF {scoring.chrf(reference.translation, hyp, target):.1f}")
    return "\n".join(lines)
