"""The P12 check, as far as one machine can do it: the built folder
(dist\\pyvolis\\) runs on its own and opens no connections.

    .venv\\Scripts\\python.exe scripts\\p12_check.py [folder]

Runs the built pyvolis.exe, never this repository's Python:

  1. --doctor: every library present and loaded from the folder;
  2. --report: every model found under the folder's own models\\;
  3. file mode on the Spanish and the Arabic fixture recordings, translated
     by Qwen3 1.7B (small enough to share the GPU), with the recognizers this
     machine's checks use; transcript CER and translation chrF against the
     references; and the same run from source, to compare the time (streaming
     and the other settings are pyvolis.toml's, the same for both);
  4. for each run, every network connection the process holds, sampled every
     0.2 s with psutil: there must be none.

Run from a copy of the folder elsewhere (another drive, a renamed folder) to
show nothing points back at the repository. Not covered, because they need a
second machine or a person: no Python and no internet on the machine, the
window from a double-click, the microphone, paired mode.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

from pyvolis import scoring  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "files"
RUNS = [
    ("es_419-to-en.wav", "es", "whisper-large-v3-turbo-es"),
    ("ar_eg-to-en.wav", "ar", "gemma-4-E4B-it-GGUF"),
]


MT = "qwen3-1.7b-q4_k_m.gguf"


def run(exe: Path | list[str], args: list[str], timeout: float = 1800) -> tuple[int, str, list[str]]:
    """Run the built program (or a command); return its exit code, output,
    and every connection it was seen holding."""
    command = [str(exe)] if isinstance(exe, Path) else exe
    cwd = exe.parent.parent if isinstance(exe, Path) else REPO
    process = subprocess.Popen([*command, *args], cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    seen: set[str] = set()
    out: list[str] = []
    import threading

    reader = threading.Thread(target=lambda: out.extend(process.stdout), daemon=True)
    reader.start()
    began = time.monotonic()
    watched = psutil.Process(process.pid)
    while process.poll() is None:
        if time.monotonic() - began > timeout:
            process.kill()
            break
        try:
            for c in watched.net_connections(kind="inet"):
                remote = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else "-"
                seen.add(f"{c.type.name} {c.laddr.ip}:{c.laddr.port} -> {remote} {c.status}")
        except psutil.Error:
            pass
        time.sleep(0.2)
    reader.join(5)
    return process.returncode, "".join(out), sorted(seen)


def main() -> int:
    folder = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO / "dist" / "pyvolis"
    exe = folder / "pyvolis.exe"
    if not exe.is_file():
        print(f"STOP: no {exe}. Run .\\build.ps1 first.")
        return 1
    ok = True
    print(f"built folder: {folder}")

    code, out, connections = run(exe, ["--doctor"])
    good = code == 0 and not connections
    ok &= good
    failed = [line for line in out.splitlines() if "FAIL" in line]
    print(f"  {'ok  ' if good else 'FAIL'} --doctor: exit {code}" + "".join(f"\n       {f}" for f in failed))

    code, out, connections = run(exe, ["--report"])
    usable = sum(1 for line in out.splitlines() if line.rstrip().endswith(" ok"))
    root_line = next((line for line in out.splitlines() if line.startswith("app root:")), "")
    good = code == 0 and str(folder) in root_line and usable > 0 and not connections
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} --report: {usable} usable models; {root_line}")

    for name, source, asr in RUNS:
        audio = FIXTURES / name
        if not audio.is_file():
            print(f"  (no {audio}; skipped)")
            continue
        export = folder / "exports" / f"p12-{audio.stem}"
        args = ["--file", str(audio), "--from", source, "--to", "en", "--asr", asr, "--mt", MT, "--fast"]
        began = time.monotonic()
        code, out, connections = run(exe, args + ["--export", str(export)])
        seconds = time.monotonic() - began
        began = time.monotonic()
        run([str(REPO / ".venv" / "Scripts" / "python.exe"), "-m", "pyvolis"],
            args + ["--export", str(REPO / "exports" / f"p12-source-{audio.stem}")])
        from_source = time.monotonic() - began
        reference = json.loads(audio.with_name(audio.name + ".ref.json").read_text(encoding="utf-8"))
        try:
            transcript = (export / "transcript.txt").read_text(encoding="utf-8")
            translation = (export / "translation.txt").read_text(encoding="utf-8")
        except OSError as e:
            ok = False
            print(f"  FAIL --file {name}: exit {code}, no export ({e})\n{out[-2000:]}")
            continue
        cer = scoring.cer(reference["transcript"], transcript, source)
        chrf = scoring.chrf(reference["translation"], translation, "en")
        good = code == 0 and cer < 10 and chrf > 40 and not connections
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} --file {name} ({asr}): CER {cer:.1f}%, chrF {chrf:.1f}, "
              f"{seconds:.0f} s built, {from_source:.0f} s from source, exit {code}")
        for c in connections:
            print(f"       connection: {c}")
    print("\nall passed" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
