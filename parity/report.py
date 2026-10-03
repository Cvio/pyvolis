"""Discovery parity: Rust volis and pyvolis report the same Rust-format models,
and a volis.toml saved by pyvolis loads in Rust volis.

    .venv\\Scripts\\python.exe parity\\report.py

Needs machine.yaml (copy machine.example.yaml) naming the Rust volis.exe.
Rust volis finds models beside its own executable, so this makes
`parity\\.rust\\` holding a hard link to volis.exe, a hard-linked mirror of
pyvolis's models\\ and a copy of pyvolis's volis.toml, and runs both there.
Nothing in the Rust repository is written to.

Compared:
  * every ASR and TTS folder Rust lists: its block of lines, with the app root
    replaced by <root>, must be identical in both reports;
  * the VAD section, identical;
  * every translation file Rust lists (`[+] name`) must be in pyvolis's list;
  * the `[asr].engine` line, when the selected engine is a Rust-format folder;
  * pyvolis-only folders are listed, not compared.
Then a volis.toml with every selection changed by pyvolis is given to Rust.
Exit code 0 when everything agrees.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "parity"))

from link_models import mirror  # noqa: E402

from pyvolis.config import Config  # noqa: E402

WORK = REPO / "parity" / ".rust"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOG_LINE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}T\S+\s+(TRACE|DEBUG|INFO|WARN|ERROR)\b")


def volis_exe() -> Path:
    machine = REPO / "machine.yaml"
    if not machine.is_file():
        raise SystemExit(f"STOP: {machine} does not exist. Copy machine.example.yaml and set volis_exe.")
    exe = Path((yaml.safe_load(machine.read_text(encoding="utf-8")) or {}).get("volis_exe", ""))
    if not exe.is_file():
        raise SystemExit(f"STOP: volis_exe in {machine} is {str(exe)!r}, which is not a file")
    return exe


def run(cmd: list[str], cwd: Path) -> tuple[int, list[str]]:
    env = dict(os.environ, NO_COLOR="1", RUST_LOG="warn")
    done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env)
    lines = [ANSI.sub("", line) for line in (done.stdout + done.stderr).splitlines()]
    return done.returncode, [line for line in lines if not LOG_LINE.match(line)]


def sections(lines: list[str], root: Path) -> dict[str, list[str]]:
    """Split a report into its sections, with the app root made neutral."""
    root_text = str(root)
    out: dict[str, list[str]] = {}
    current = None
    for line in lines:
        line = line.replace(root_text, "<root>").rstrip()
        for title in ("ASR engines", "TTS voices", "VAD", "Translation", "summary:"):
            if line.startswith(title):
                current = title
                out[current] = []
                break
        else:
            if current is not None and line:
                out[current].append(line)
    return out


def blocks(section: list[str]) -> dict[str, list[str]]:
    """A model table's rows, keyed by directory: each row and the lines under it."""
    out: dict[str, list[str]] = {}
    current = None
    for line in section:
        if line.startswith("  DIRECTORY"):
            continue
        if line.startswith("  ") and not line.startswith("   "):
            current = line[2:].split()[0]
            out[current] = [line]
        elif current is not None:
            out[current].append(line)
    return out


def main() -> int:
    exe = volis_exe()
    if WORK.exists():
        shutil.rmtree(WORK)  # hard links only: the originals are untouched
    WORK.mkdir(parents=True)
    os.link(exe, WORK / exe.name)
    mirror(REPO / "models", WORK / "models")
    shutil.copyfile(REPO / "volis.toml", WORK / "volis.toml")

    problems: list[str] = []
    code_r, rust = run([str(WORK / exe.name), "--report"], WORK)
    code_p, py = run([sys.executable, "-m", "pyvolis", "--report"], REPO)
    if code_r != 0:
        problems.append(f"Rust --report exited {code_r}:\n" + "\n".join(rust[-20:]))
    if code_p != 0:
        problems.append(f"pyvolis --report exited {code_p}:\n" + "\n".join(py[-20:]))
    rs, ps = sections(rust, WORK), sections(py, REPO)

    print("Discovery")
    for title in ("ASR engines", "TTS voices"):
        rb, pb = blocks(rs.get(title, [])), blocks(ps.get(title, []))
        for name, lines in rb.items():
            if pb.get(name) == lines:
                print(f"  same   {title}: {name}")
            else:
                problems.append(f"{title}: {name} differs\n  Rust:\n    " + "\n    ".join(lines)
                                + "\n  pyvolis:\n    " + "\n    ".join(pb.get(name, ["(missing)"])))
                print(f"  DIFF   {title}: {name}")
        for name in pb.keys() - rb.keys():
            print(f"  pyvolis only   {title}: {name}  ({pb[name][0].split()[2]})")
    if rs.get("VAD") == ps.get("VAD"):
        print("  same   VAD")
    else:
        problems.append(f"VAD differs: Rust {rs.get('VAD')} pyvolis {ps.get('VAD')}")
    for line in rs.get("Translation", []):
        if line in ps.get("Translation", []):
            print(f"  same   Translation: {line.strip()}")
        else:
            problems.append(f"Translation: Rust lists {line.strip()!r}, pyvolis doesn't")
    rust_sel = [line for line in rs.get("summary:", []) if "[asr].engine" in line]
    py_sel = [line for line in ps.get("summary:", []) if "[asr].engine" in line]
    if rust_sel and "not found" not in rust_sel[0]:
        if rust_sel == py_sel:
            print(f"  same   selection: {rust_sel[0].strip()}")
        else:
            problems.append(f"selection differs: Rust {rust_sel} pyvolis {py_sel}")

    print("\nA volis.toml saved by pyvolis, loaded by Rust volis")
    config, _ = Config.load(WORK / "volis.toml")
    config.asr.engine = "parakeet-tdt-0.6b-v3-onnx-int8"
    config.languages.source, config.languages.target = "es-MX", "en-US"
    config.audio.input_device = "Microphone (parity test)"
    config.mode.kind, config.mode.turn_style = "continuous", "hold"
    config.tts.enabled, config.tts.half_duplex = False, False
    config.peer.enabled, config.peer.peer_addr = True, "192.168.50.2:47800"
    config.shared.left_language, config.shared.right_language = "ar-IQ", "en"
    config.shared.left_voice, config.shared.right_voice = "vits-piper-en_US-lessac-medium", ""
    config.shared.left_asr = "whisper-large-v3-turbo-arabic-dialectal-onnx-int8"
    config.save_selections(WORK / "volis.toml")
    code, out = run([str(WORK / exe.name), "--report"], WORK)
    loaded = any("config:" in line and "volis.toml" in line for line in out)
    if code == 0 and loaded and any('"parakeet-tdt-0.6b-v3-onnx-int8"' in line for line in out):
        print("  Rust loaded it, and reads [asr].engine as pyvolis wrote it")
    else:
        problems.append(f"Rust volis did not accept pyvolis's volis.toml (exit {code}):\n" + "\n".join(out[-15:]))
    # And from an empty file: every section added by pyvolis.
    (WORK / "volis.toml").write_text("", encoding="utf-8")
    Config().save_selections(WORK / "volis.toml")
    code, out = run([str(WORK / exe.name), "--report"], WORK)
    if code == 0:
        print("  Rust loaded one pyvolis wrote from nothing")
    else:
        problems.append(f"Rust volis refused a volis.toml pyvolis created (exit {code}):\n" + "\n".join(out[-15:]))

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"- {p}")
        return 1
    print("\nParity: everything agrees.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
