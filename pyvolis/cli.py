"""Command line. Port of Rust `cli.rs` and `main.rs`.

Deliberately small: the window is where the controls belong. These flags
exist so each milestone can be checked from a shell. Commands arrive with the
milestone that builds them (`--devices` at P1, `--listen` at P2).
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import __version__, models, paths, report
from .config import Config, ConfigError

HELP = """\
pyvolis - offline speech-to-speech translation

USAGE:
    pyvolis [COMMAND]

COMMANDS:
    (none)              Open the window
    --report            Print the discovered models and exit
    --report --load     Also load each usable model in turn and print where it
                        runs, its memory (GPU and system) and its load time
    --devices           List audio input and output devices and exit
    --listen            Capture from the microphone and transcribe
    --translate <TEXT>  Translate one sentence and print it
    --print-prompt <TEXT>  Print the exact prompt the translator would get
    --file <PATH>       Transcribe and translate an audio file, write an export
                        folder, and print scores if a reference is beside it

OPTIONS FOR --listen:
    --seconds <N>       Stop cleanly after N seconds (otherwise: Ctrl-C)
    --wav               Write each detected utterance to logs\\segments\\
    --compare           Run every usable recognizer on each utterance and print
                        transcripts, timings and segment durations side by side

OPTIONS FOR --translate AND --print-prompt:
    --from <TAG>        Source language or variety (default: [languages].source)
    --to <TAG>          Target (default: [languages].target)
    --mt <ID>           Translator, as --report lists it (default: pyvolis.toml)
    --prompt <NAME>     Prompt file in prompts\\ (default: pyvolis.toml)

OPTIONS FOR --file (and --from, --to, --mt, --prompt as above):
    --asr <FOLDER>      Recognizer, a folder in models\\asr\\ (default: [asr].engine)
    --fast              As fast as the models allow (default: real time)
    --export <DIR>      Where to write (default: exports\\<file>-<date>\\)
    --no-translate      Transcribe only
    --streaming         Recognise while the speech goes on (provisional text,
                        committed as two passes agree); --no-streaming forces it off
    --context <MODE>    "carry" (earlier sentences as context), "revision"
                        (carry, and earlier translations may be replaced) or "off"
    --no-hold           Don't hold short fragments to join them to what follows
    --glossary <TERMS>  Names and terms to keep exactly, separated by commas

    -h, --help          Show this message

pyvolis never accesses the internet. Models are read from models\\ beside the
application; see README.md for what to put there.
"""


class UsageError(Exception):
    pass


@dataclass(frozen=True)
class Command:
    name: str  # "gui" | "report" | "devices" | "listen" | "help"
    seconds: int | None = None  # --listen: stop after this many seconds
    write_wav: bool = False  # --listen --wav
    compare: bool = False  # --listen --compare
    text: str = ""  # --translate / --print-prompt
    source: str = ""
    target: str = ""
    mt: str = ""
    prompt: str = ""
    path: str = ""  # --file
    asr: str = ""
    fast: bool = False
    export: str = ""
    translate: bool = True
    load: bool = False  # --report --load
    streaming: bool | None = None  # --file: None = pyvolis.toml
    context: str = ""
    hold: bool | None = None
    glossary: str = ""


def parse(args: list[str]) -> Command:
    if not args:
        return Command("gui")  # a double-click from Explorer passes nothing
    first, rest = args[0], args[1:]
    if first in ("-h", "--help"):
        return Command("help")
    if first == "--report":
        if rest == ["--load"]:
            return Command("report", load=True)
        _reject_extra(rest)
        return Command("report")
    if first == "--devices":
        _reject_extra(rest)
        return Command("devices")
    if first == "--listen":
        seconds, write_wav, compare = None, False, False
        options = iter(rest)
        for arg in options:
            if arg == "--wav":
                write_wav = True
            elif arg == "--compare":
                compare = True
            elif arg == "--seconds":
                value = next(options, None)
                if value is None:
                    raise UsageError("--seconds needs a number of seconds")
                if not value.isdigit():
                    raise UsageError(f'--seconds "{value}" is not a number')
                seconds = int(value)
            else:
                raise _unknown(arg)
        return Command("listen", seconds, write_wav, compare)
    if first in ("--translate", "--print-prompt"):
        if not rest or rest[0].startswith("--"):
            raise UsageError(f"{first} needs the text to translate")
        values = {"--from": "", "--to": "", "--mt": "", "--prompt": ""}
        options = iter(rest[1:])
        for arg in options:
            if arg not in values:
                raise _unknown(arg)
            value = next(options, None)
            if value is None:
                raise UsageError(f"{arg} needs a value")
            values[arg] = value
        return Command(first[2:], text=rest[0], source=values["--from"], target=values["--to"],
                       mt=values["--mt"], prompt=values["--prompt"])
    if first == "--file":
        if not rest or rest[0].startswith("--"):
            raise UsageError("--file needs the path of an audio file")
        values = {"--from": "", "--to": "", "--mt": "", "--prompt": "", "--asr": "", "--export": "",
                  "--context": "", "--glossary": ""}
        flags = {"--fast": False, "--no-translate": False, "--streaming": False, "--no-streaming": False,
                 "--no-hold": False}
        options = iter(rest[1:])
        for arg in options:
            if arg in flags:
                flags[arg] = True
                continue
            if arg not in values:
                raise _unknown(arg)
            value = next(options, None)
            if value is None:
                raise UsageError(f"{arg} needs a value")
            values[arg] = value
        if values["--context"] not in ("", "off", "carry", "revision"):
            raise UsageError(f'--context "{values["--context"]}" is not "off", "carry" or "revision"')
        streaming = True if flags["--streaming"] else False if flags["--no-streaming"] else None
        return Command("file", path=rest[0], source=values["--from"], target=values["--to"], mt=values["--mt"],
                       prompt=values["--prompt"], asr=values["--asr"], export=values["--export"],
                       fast=flags["--fast"], translate=not flags["--no-translate"], streaming=streaming,
                       context=values["--context"], hold=False if flags["--no-hold"] else None,
                       glossary=values["--glossary"])
    raise _unknown(first)


def _reject_extra(rest: list[str]) -> None:
    if rest:
        raise _unknown(rest[0])


def _unknown(arg: str) -> UsageError:
    return UsageError(f'unknown argument "{arg}"\n\n{HELP}')


def run(args: list[str], root: Path) -> int:
    try:
        command = parse(args)
    except UsageError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    if command.name == "help":
        print(HELP, end="")
        return 0

    init_logging(root)
    print(f"pyvolis {__version__} - offline, no network required")
    print(f"app root: {root}")

    if command.name == "devices":
        return print_devices()

    config_path = paths.config_file(root)
    try:
        config, found = Config.load(config_path)
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if found:
        print(f"config:   {config_path}")
    else:
        print(f"config:   {config_path} (not present; using the defaults)")

    if command.name == "gui":
        from .gui import window

        return window.run(root, config)
    if command.name == "listen":
        from . import listen
        from .pipeline import Options

        return listen.run(root, config, command.seconds, Options(command.write_wav, command.compare))
    if command.name in ("translate", "print-prompt"):
        return run_translate(root, config, command)
    if command.name == "file":
        from .filerun import FileRun, run as run_file

        return run_file(root, config, FileRun(
            Path(command.path), command.source, command.target, command.asr, command.mt, command.prompt,
            command.fast, Path(command.export) if command.export else None, command.translate,
            command.streaming, command.context or None, command.hold, command.glossary,
        ))
    return run_report(root, config, command.load)


def run_translate(root: Path, config: Config, command: Command) -> int:
    """--translate and --print-prompt: one sentence, no audio."""
    from . import translate
    from .config import PyvolisConfig
    from .translate import prompts

    try:
        pyconfig, _ = PyvolisConfig.load(paths.pyvolis_config_file(root))
        entry = translate.choose(root, command.mt or pyconfig.translate.model)
        prompt = prompts.load(paths.prompts_dir(root), command.prompt or pyconfig.translate.prompt)
        translator = translate.load(entry, prompt, pyconfig.translate.device)
    except (ConfigError, translate.TranslateError, prompts.PromptError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    request = translate.TranslationRequest(
        command.text, command.source or config.languages.source, command.target or config.languages.target
    )
    try:
        if command.name == "print-prompt":
            # Bytes, so Windows doesn't turn LF into CRLF: the prompt is
            # compared byte for byte with Rust's.
            sys.stdout.flush()
            sys.stdout.buffer.write(translator.prompt(request).encode("utf-8"))
            sys.stdout.buffer.flush()
            return 0
        result = translator.translate(request)
    except translate.Refused as e:
        print(f"refused ({e.guard}): {e}", file=sys.stderr)
        return 1
    except translate.TranslateError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        translator.close()
    print(f"[{request.source}] {request.text}")
    print(f"[{request.target}] {result.text}")
    print(f"({result.seconds * 1000:.0f} ms to translate on the {result.device.upper()}, {entry.id}, "
          f"prompt {prompt.name})")
    return 0


def print_devices() -> int:
    """Audio devices, so the user can put exact names in [audio]."""
    from . import audio

    try:
        inputs, outputs = audio.list_input_devices(), audio.list_output_devices()
    except audio.AudioError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    _print_device_list("Audio input devices", "[audio].input_device", inputs)
    _print_device_list("Audio output devices", "[audio].output_device", outputs)
    print()
    print("  * = system default. Leave the setting empty to follow it.")
    return 0


def _print_device_list(title: str, setting: str, devices: list) -> None:
    print()
    print(f"{title}  ({setting})")
    if not devices:
        print("  (none)")
        return
    for device in devices:
        config = f"  ({device.default_config})" if device.default_config else ""
        print(f"  {'*' if device.is_default else ' '} {device.name}{config}")


def run_report(root: Path, config: Config, load: bool = False) -> int:
    models_root = paths.models_dir(root)
    if not models_root.is_dir():
        print(
            f"Error: model directory not found: {models_root}\n"
            "pyvolis never downloads models. Create that directory and place the model "
            "folders in it as described in README.md, then run again.",
            file=sys.stderr,
        )
        return 1
    asr_root, tts_root = paths.asr_dir(root), paths.tts_dir(root)
    asr = models.discover(asr_root, models.Role.ASR)
    tts = models.discover(tts_root, models.Role.TTS)
    translators = models.discover_translators(paths.mt_dir(root))

    report.print_table("ASR engines", asr_root, asr)
    report.print_table("TTS voices", tts_root, tts)
    report.print_single_files(root, translators)

    print()
    print("summary:")
    report.print_summary(models.Role.ASR, asr)
    report.print_summary(models.Role.TTS, tts)
    report.print_translator_summary(translators)
    report.report_selection(config, asr)
    if load:
        report.print_loaded(root, asr, tts, translators)
    return 0


class _Utc(logging.Formatter):
    converter = time.gmtime

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        t = self.converter(record.created)
        return time.strftime("%Y-%m-%dT%H:%M:%S", t) + f".{int(record.msecs):03d}Z"


def init_logging(root: Path) -> None:
    """stdout plus `<root>/logs/pyvolis.log.<date>`, as Rust does with
    `volis.log.<date>`. If the folder can't be created, stdout alone."""
    formatter = _Utc("%(asctime)s %(levelname)5s %(name)s: %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    logs = paths.logs_dir(root)
    problem = None
    try:
        logs.mkdir(parents=True, exist_ok=True)
        name = f"pyvolis.log.{time.strftime('%Y-%m-%d', time.gmtime())}"
        handlers.append(logging.FileHandler(logs / name, encoding="utf-8"))
    except OSError as e:
        problem = f"cannot create log directory {logs}: {e}"
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    if problem:
        logging.getLogger(__name__).warning(problem)
