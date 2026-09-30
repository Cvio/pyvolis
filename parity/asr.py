"""Recognition parity: pyvolis's sherpa backend gives Rust volis's text.

    .venv\\Scripts\\python.exe parity\\asr.py [es_419 ar_eg en_us ...]

Needs machine.yaml (volis_exe) and the VB-Audio Virtual Cable. Rust volis
reads audio only from a device, so:

  1. A parity copy of Rust volis (parity\\.rust\\, as parity\\report.py makes)
     runs `--listen --wav --compare`, listening to "CABLE Output", with voice
     output off.
  2. Each FLEURS fixture clip is played into "CABLE Input", one at a time,
     waiting for Rust's comparison table before the next (Rust's pipeline
     drops audio if it falls behind).
  3. Rust writes every utterance it heard to a WAV and logs every engine's
     transcript of it. pyvolis's sherpa backend transcribes those same WAVs
     with the same engines, and the texts are compared.

What this can and can't prove. The WAVs are 16-bit, and the audio also
passes through Windows and the cable driver, which scale it (measured gain
0.9896) and resample it internally even with both ends set to 16 kHz, so
pyvolis hears Rust's audio only to within about half a 16-bit step. That is
enough for Parakeet (identical on every Spanish utterance, 2026-09-30) and
not for int8 Whisper, whose text changes for differences smaller than one
16-bit step (shown by transcribing the same WAV with and without a sub-step
nudge). Exact Whisper parity needs Rust to transcribe a file itself.
Nothing in the Rust repository is written to.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "parity"))

from pyvolis import paths  # noqa: E402

paths.apply_offline_environment(paths.app_root())

from link_models import mirror  # noqa: E402
from report import volis_exe  # noqa: E402

from pyvolis import asr, audio, models  # noqa: E402
from pyvolis.config import Config  # noqa: E402
from pyvolis.filesource import read_16k_mono  # noqa: E402

WORK = REPO / "parity" / ".rust"
CABLE_IN = "CABLE Input (VB-Audio Virtual Cable)"
CABLE_OUT = "CABLE Output (VB-Audio Virtual Cable)"
LANGUAGE = {"es_419": ("es", "en"), "ar_eg": ("ar", "en"), "en_us": ("en", "es"), "fa_ir": ("fa", "en")}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TABLE = re.compile(r"^utterance (\d+) at (\d+) ms - (\d+) ms of audio$")
ROW = re.compile(r"^  [* ] (\S+)\s+(\d+) ms")
HEARD = re.compile(r"volis::pipeline: utterance (\d+): \d+ ms \.\.")
TRANSCRIPT = re.compile(r"volis::pipeline:   \[([A-Za-z-]+)\] (.*)$")
TRANSLATED = re.compile(r"volis::pipeline: utterance (\d+)$")
BRACKETED = re.compile(r"^  \[([A-Za-z-]+)\] (.*)$")
REFUSED = re.compile(r"utterance (\d+): translation failed: (.*)$")
WROTE = re.compile(r"wrote (.*utterance-(\d{4})-at-\d+ms-for-\d+ms\.wav)")


class Rust:
    """The parity copy of volis.exe, running --listen, its output parsed."""

    def __init__(self, source: str, target: str, compare: bool = True) -> None:
        exe = volis_exe()
        if WORK.exists():
            shutil.rmtree(WORK)
        WORK.mkdir(parents=True)
        os.link(exe, WORK / exe.name)
        mirror(REPO / "models", WORK / "models")
        shutil.copyfile(REPO / "volis.toml", WORK / "volis.toml")
        config, _ = Config.load(WORK / "volis.toml")
        config.languages.source, config.languages.target = source, target
        config.mode.kind = "continuous"
        config.audio.input_device = CABLE_OUT
        config.tts.enabled = False
        config.save_selections(WORK / "volis.toml")
        self.selected = config.asr.engine
        env = dict(os.environ, RUST_LOG="info", NO_COLOR="1")
        self.proc = subprocess.Popen(
            [str(WORK / exe.name), "--listen", "--wav"] + (["--compare"] if compare else []), cwd=WORK, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
        self.lines: list[str] = []
        self.tables: dict[int, dict[str, str]] = {}
        self.wavs: dict[int, Path] = {}
        # Translations: utterance -> (source text, translation or None if refused).
        self.translations: dict[int, tuple[str, str | None]] = {}
        self.refusals: dict[int, str] = {}
        self.transcripts: dict[int, str] = {}
        self.listening = threading.Event()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        current, engine = None, None
        pending: list = []  # a translation block being read: [index, source text]
        heard = 0  # the utterance being transcribed
        for raw in self.proc.stdout:
            line = ANSI.sub("", raw.rstrip("\n"))
            self.lines.append(line)
            if m := TRANSLATED.search(line):
                pending = [int(m.group(1))]
                continue
            if m := HEARD.search(line):
                heard = int(m.group(1))
            if not pending and heard and (m := TRANSCRIPT.search(line)):
                self.transcripts[heard] = m.group(2)
                continue
            if pending and (m := BRACKETED.match(line)):
                if len(pending) == 1:
                    pending.append(m.group(2))
                else:
                    self.translations[pending[0]] = (pending[1], m.group(2))
                    pending = []
                continue
            if m := REFUSED.search(line):
                self.refusals[int(m.group(1))] = m.group(2)
            if "Ctrl-C to stop" in line or "listening for" in line:
                self.listening.set()
            if m := TABLE.match(line):
                current = int(m.group(1))
                self.tables[current] = {}
                continue
            if current is not None and (m := ROW.match(line)):
                engine = m.group(1)
                continue
            if current is not None and engine and line.startswith("      "):
                text = line[6:]
                self.tables[current][engine] = "" if text == "(no text)" else text
                engine = None
                continue
            if m := WROTE.search(line):
                self.wavs[int(m.group(2))] = Path(m.group(1))

    def wait_for(self, count: int, timeout: float) -> bool:
        """Until `count` utterances have a full table and a WAV."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            done = [i for i in self.tables if i in self.wavs]
            if len(done) >= count and self.proc.poll() is None:
                return True
            if self.proc.poll() is not None:
                return False
            time.sleep(0.2)
        return False

    def stop(self) -> None:
        if self.proc.poll() is None:
            # Ctrl-C isn't deliverable to a child on Windows without a console
            # group; the table and WAVs are written as each utterance ends, so
            # ending the process loses nothing already reported.
            self.proc.terminate()
        self.proc.wait(timeout=30)


def play(clip: np.ndarray, device: int, rate: int = 16_000) -> None:
    """Play a 16 kHz clip into the cable at the cable's rate, then 1.5 s of silence."""
    import sounddevice as sd

    if rate != 16_000:
        import soxr

        clip = soxr.resample(clip, 16_000, rate).astype(np.float32)
    silence = np.zeros(int(rate * 1.5), np.float32)
    sd.play(np.concatenate([clip, silence]), samplerate=rate, device=device)
    sd.wait()


def read_rust_wav(path: Path) -> np.ndarray:
    """Undo Rust's write, round(x * i16::MAX): Rust's audio to within half a
    16-bit step."""
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return (data.astype(np.float32) / 32767.0).astype(np.float32)


def main(argv: list[str]) -> int:
    sets = argv or ["es_419", "ar_eg"]
    devices = {d.name: d for d in audio.list_output_devices()}
    outputs = {name: d.index for name, d in devices.items()}
    if CABLE_IN not in outputs:
        raise SystemExit(f"STOP: no output device named {CABLE_IN!r}; install the VB-Audio Virtual Cable")
    engines = [e for e in models.discover(paths.asr_dir(REPO), models.Role.ASR)
               if isinstance(e, models.Engine) and e.from_engine_toml and e.enabled()]
    total = same = 0
    diffs: list[str] = []
    for name in sets:
        source, target = LANGUAGE[name]
        folder = REPO / "tests" / "fixtures" / "fleurs" / name
        refs = [json.loads(x) for x in (folder / "refs.jsonl").read_text(encoding="utf-8").splitlines()]
        print(f"\n== {name}: {len(refs)} clips, Rust listening as \"{source}\"")
        rust = Rust(source, target)
        try:
            if not rust.listening.wait(180):
                raise SystemExit("STOP: Rust volis never started listening:\n" + "\n".join(rust.lines[-30:]))
            for i, ref in enumerate(refs):
                before = len(rust.tables)
                rate = int(devices[CABLE_IN].default_config.split(", ")[1].split()[0])
                play(read_16k_mono(folder / ref["file"]), outputs[CABLE_IN], rate)
                # Wait until Rust has finished every utterance of this clip.
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    time.sleep(1.0)
                    complete = [k for k in rust.tables if k in rust.wavs and len(rust.tables[k]) == len(engines)]
                    if len(rust.tables) > before and len(complete) == len(rust.tables):
                        break
                print(f"  clip {i + 1}: Rust has {len(rust.tables)} utterance(s) so far")
        finally:
            rust.stop()
        print(f"  pyvolis transcribing Rust's {len(rust.wavs)} utterances with {len(engines)} engines")
        for engine in engines:
            recognizer = asr.load(engine)
            for index in sorted(rust.tables):
                if index not in rust.wavs or engine.dir_name not in rust.tables[index]:
                    continue
                theirs = rust.tables[index][engine.dir_name]
                ours = recognizer.transcribe(read_rust_wav(rust.wavs[index]), source).text
                total += 1
                if ours == theirs:
                    same += 1
                else:
                    diffs.append(f"{name} utterance {index}, {engine.dir_name}:\n    Rust:    {theirs}\n    pyvolis: {ours}")
            recognizer.close()
            print(f"    {engine.dir_name}: done")
    print(f"\n{same} of {total} transcripts identical")
    for d in diffs:
        print(f"  DIFF {d}")
    return 0 if total and same == total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
