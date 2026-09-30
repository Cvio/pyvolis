"""The report: a table of everything discovery found.

Port of Rust `report.rs` and the report half of `main.rs`. For folders with an
`engine.toml` the output is line-for-line Rust's, so the two reports can be
compared (`parity/report.py`). Downloaded folders and the translator list add
lines of their own. Printed to stdout: it is the program's output, not a log.
"""

from __future__ import annotations

from pathlib import Path

from . import paths
from .config import Config
from .models import Engine, Entry, Failed, Role, Translator

# Rust's column widths.
NAME_W = 34
KIND_W = 7
BACKEND_W = 15


def _row(a: str, b: str, c: str, d: str) -> str:
    return f"  {a:<{NAME_W}}  {b:<{KIND_W}}  {c:<{BACKEND_W}}  {d}"


def _more(text: str) -> str:
    return f"  {'':<{NAME_W}}  {text}"


def print_table(title: str, root: Path, entries: list[Entry]) -> None:
    print()
    print(f"{title}  ({root})")
    if not entries:
        print("  (none)")
        return
    print(_row("DIRECTORY", "KIND", "BACKEND", "STATUS"))
    for entry in entries:
        if isinstance(entry, Failed):
            print(_row(entry.dir_name, "-", "-", "ERROR"))
            print(_more(entry.error))
            continue
        print(_row(entry.dir_name, entry.kind, entry.backend, _status(entry)))
        print(_more(entry.name))
        print(_more(str(entry.dir)))
        if entry.detail:
            print(_more(f"model: {entry.detail}"))
        if entry.varieties:
            print(_more(f"tuned for: {', '.join(entry.varieties)}"))
        if entry.languages:
            print(_more(f"languages: {', '.join(entry.languages)}"))
        elif not entry.languages_known:
            print(
                _more(
                    "languages unknown: offered for every language; list them in "
                    f"{entry.dir / 'pyvolis.toml'}"
                )
            )
        for file in entry.files:
            print(_more(f"[{'+' if file.present else '!'}] {file.role} {file.path}"))


def _status(engine: Engine) -> str:
    missing = engine.missing_files()
    if missing:
        return f"DISABLED - missing: {', '.join(missing)}"
    if engine.unusable:
        return f"UNUSABLE - {engine.unusable}"
    return "ok"


def print_single_files(root: Path, translators: list[Translator | Failed]) -> None:
    """VAD and translation, which have no engine.toml. The `[+] <file>` lines
    for top-level translators are Rust's; the rest is pyvolis's."""
    vad = paths.vad_model_file(root)
    print()
    print(f"VAD      ({vad})")
    print(f"  [{'+' if vad.is_file() else '!'}] silero_vad.onnx")

    mt_root = paths.mt_dir(root)
    print()
    print(f"Translation  ({mt_root})")
    if not mt_root.is_dir():
        print(f"  cannot read {mt_root}: the folder does not exist")
        return
    top_level = [t for t in translators if isinstance(t, Translator) and t.top_level]
    if not translators:
        print("  (no .gguf files)")
        return
    for t in translators:
        if isinstance(t, Failed):
            print(f"  [!] {t.dir_name}")
            print(f"        ERROR: {t.error}")
            continue
        mark = "+" if t.enabled() else "!"
        print(f"  [{mark}] {t.id}")
        facts = [t.backend]
        if t.architecture:
            facts.append(t.architecture)
        if t.backend == "llamacpp":
            facts.append(f"chat template: {'yes' if t.has_chat_template else 'no'}")
        if t.size_bytes:
            facts.append(f"{t.size_bytes / 1e9:.1f} GB")
        print(f"        {t.name}  ({', '.join(facts)})")
        if t.missing:
            print(f"        DISABLED - missing: {', '.join(t.missing)}")
        elif t.unusable:
            print(f"        UNUSABLE - {t.unusable}")
    if len(top_level) > 1:
        print(
            f"  note: Rust volis refuses to start with {len(top_level)} .gguf files at the top of "
            f"{mt_root}. Give each extra translator its own folder there."
        )


def print_summary(role: Role, entries: list[Entry]) -> None:
    usable = sum(1 for e in entries if isinstance(e, Engine) and e.enabled())
    label = "ASR" if role is Role.ASR else "TTS"
    print(f"  {label}: {usable} of {len(entries)} entries usable")


def print_translator_summary(translators: list[Translator | Failed]) -> None:
    usable = sum(1 for t in translators if isinstance(t, Translator) and t.enabled())
    print(f"  Translators: {usable} of {len(translators)} usable")


def report_selection(config: Config, asr: list[Entry]) -> None:
    """Say plainly whether the configured engine exists and is usable. Never
    substitute a different one."""
    selected = config.asr.engine.strip()
    if not selected:
        print("  [asr].engine is unset; nothing selected")
        return
    entry = next((e for e in asr if e.dir_name == selected), None)
    if entry is None:
        print(f'  [asr].engine = "{selected}" was not found among the discovered engines')
    elif isinstance(entry, Failed):
        print(f'  [asr].engine = "{selected}" failed to load: {entry.error}')
    elif entry.enabled():
        print(f'  [asr].engine = "{selected}" -> {entry.name}')
        if not entry.from_engine_toml:
            print("    (a pyvolis-only model: Rust volis will report it as not found)")
    elif entry.missing_files():
        print(
            f'  [asr].engine = "{selected}" is DISABLED - missing: '
            f"{', '.join(entry.missing_files())}"
        )
    else:
        print(f'  [asr].engine = "{selected}" is UNUSABLE - {entry.unusable}')
