"""Command line. Port of Rust `cli.rs` and `main.rs`.

Deliberately small: the window is where the controls belong. These flags
exist so each milestone can be checked from a shell. Commands arrive with the
milestone that builds them (`--devices` and `--listen` at P1 and P2).
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

    -h, --help          Show this message

pyvolis never accesses the internet. Models are read from models\\ beside the
application; see README.md for what to put there.
"""


class UsageError(Exception):
    pass


@dataclass(frozen=True)
class Command:
    name: str  # "gui" | "report" | "help"


def parse(args: list[str]) -> Command:
    if not args:
        return Command("gui")  # a double-click from Explorer passes nothing
    first, rest = args[0], args[1:]
    if first in ("-h", "--help"):
        return Command("help")
    if first == "--report":
        _reject_extra(rest)
        return Command("report")
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
        print("The window arrives at P5. For now: pyvolis --report", file=sys.stderr)
        return 2
    return run_report(root, config)


def run_report(root: Path, config: Config) -> int:
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
