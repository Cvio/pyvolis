"""The P9 check against Rust volis: pyvolis and the real volis.exe pair, talk
in both directions, take the floor, and each notices when the other is killed.

    .venv\\Scripts\\python.exe scripts\\pair_check.py

Rust volis pairs only in its window (its --listen never does), so this opens
the Rust window from the parity folder and asks for two clicks in it: Start,
and later Connect. Everything else is driven from here. The pyvolis end is
the real peer thread (pyvolis/peer.py), without models: what crosses the wire
is text, and that is what is checked.

Needs machine.yaml (the path of volis.exe), the VB-Audio cable (Rust listens
to it; a Spanish clip is played into it) and tests\\fetch-fixtures.ps1.
pyvolis with pyvolis is checked by tests\\test_peer.py and tests\\test_paired.py.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "parity"))

from pyvolis import paths  # noqa: E402

paths.apply_offline_environment(paths.app_root())
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

import tomlkit  # noqa: E402
from asr import ANSI, CABLE_IN, CABLE_OUT, WORK, mirror, play, volis_exe  # noqa: E402

from pyvolis import audio, peer  # noqa: E402
from pyvolis import events as ev  # noqa: E402
from pyvolis.config import Config  # noqa: E402
from pyvolis.filesource import read_16k_mono  # noqa: E402

RUST_PORT, OUR_PORT = 47811, 47812
RESULTS: list[tuple[str, bool, str]] = []


def result(what: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((what, ok, detail))
    print(f"  {'ok  ' if ok else 'FAIL'} {what}" + (f": {detail}" if detail else ""), flush=True)


class RustWindow:
    """volis.exe's window, from the parity folder, set up to pair."""

    def __init__(self) -> None:
        exe = volis_exe()
        if WORK.exists():
            shutil.rmtree(WORK)
        WORK.mkdir(parents=True)
        os.link(exe, WORK / exe.name)
        mirror(REPO / "models", WORK / "models")
        shutil.copyfile(REPO / "volis.toml", WORK / "volis.toml")
        config, _ = Config.load(WORK / "volis.toml")
        config.languages.source, config.languages.target = "es", "en"
        config.mode.kind = "continuous"  # hears the cable without a turn key; the floor still answers
        config.audio.input_device = CABLE_OUT
        config.tts.enabled = False
        config.asr.engine = "parakeet-tdt-0.6b-v3-int8"
        config.peer.enabled = True
        config.peer.peer_addr = f"127.0.0.1:{OUR_PORT}"
        config.save_selections(WORK / "volis.toml")
        doc = tomlkit.parse((WORK / "volis.toml").read_text(encoding="utf-8"))
        doc["peer"]["listen_addr"] = f"127.0.0.1:{RUST_PORT}"
        doc["peer"]["display_name"] = "rust-volis"
        doc["peer"]["discovery"] = False
        (WORK / "volis.toml").write_text(tomlkit.dumps(doc), encoding="utf-8")
        self.proc = subprocess.Popen([str(WORK / exe.name)], cwd=WORK, env=dict(os.environ, RUST_LOG="info", NO_COLOR="1"),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                     errors="replace")
        self.lines: list[str] = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for raw in self.proc.stdout:
            self.lines.append(ANSI.sub("", raw.rstrip("\n")))

    def wait_for_line(self, pattern: str, seconds: float, since: int = 0) -> str | None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            for line in self.lines[since:]:
                if re.search(pattern, line):
                    return line
            if self.proc.poll() is not None:
                return None
            time.sleep(0.1)
        return None

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=30)


class Ours:
    """The pyvolis end: the peer thread, with its events and what it asked of the pipeline."""

    def __init__(self) -> None:
        config = Config()
        config.peer.enabled, config.peer.display_name = True, "pyvolis"
        config.peer.listen_addr, config.peer.discovery = f"127.0.0.1:{OUR_PORT}", False
        config.languages.source, config.languages.target = "en", "es"
        self.events: queue.Queue = queue.Queue()
        self.asked: queue.Queue = queue.Queue()
        self.peer = peer.start(config, "turn", peer.Wiring(
            begin_turn=lambda: self.asked.put("begin_turn"), end_turn=lambda: self.asked.put("end_turn"),
            speak=lambda lang, text: "off", emit=self.events.put))

    def wait_for(self, want, seconds: float = 15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                event = self.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if want(event):
                return event
        return None


def is_a(cls, **fields):
    return lambda e: isinstance(e, cls) and all(getattr(e, k) == v for k, v in fields.items())


def port_open(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), 0.3).close()
        return True
    except OSError:
        return False


def main() -> int:
    devices = {d.name: d for d in audio.list_output_devices()}
    if CABLE_IN not in devices:
        raise SystemExit(f"STOP: no output device named {CABLE_IN!r}; install the VB-Audio Virtual Cable")
    rate = int(devices[CABLE_IN].default_config.split(", ")[1].split()[0])
    clip = REPO / "tests" / "fixtures" / "fleurs" / "es_419"
    wavs = sorted(clip.glob("*.wav"))
    if not wavs:
        raise SystemExit("STOP: no fixtures; run tests\\fetch-fixtures.ps1")

    rust, ours = RustWindow(), Ours()
    try:
        print("\n>>> In the Rust volis window that just opened: press Start. (Waiting up to 5 minutes.)", flush=True)
        deadline = time.monotonic() + 300
        while not port_open(RUST_PORT):
            if time.monotonic() > deadline or rust.proc.poll() is not None:
                raise SystemExit("STOP: Rust volis never started listening on its pairing port")
            time.sleep(0.5)
        time.sleep(1.0)  # that probe was a connection too; let Rust drop it

        print("\npyvolis dials Rust volis")
        ours.peer.connect(peer.Address("127.0.0.1", RUST_PORT))
        hello = ours.wait_for(is_a(ev.PeerMsg, kind="connected"))
        result("pyvolis is paired, and knows who with", hello is not None and hello.name == "rust-volis",
               f"{hello.name}, speaks {hello.speaks}, sends {hello.sends}" if hello else "no Hello from Rust")
        line = rust.wait_for_line(r'paired with "pyvolis"', 10)
        result("Rust is paired, and knows who with", line is not None, (line or "").split("paired mode: ")[-1])

        print("\npyvolis takes a turn and speaks to Rust")
        ours.peer.want_turn()
        granted = ours.wait_for(is_a(ev.FloorChanged, holder="me"), 5)
        result("Rust grants the floor; only then does the microphone open",
               granted is not None and ours.asked.get(timeout=5) == "begin_turn")
        mark = len(rust.lines)
        ours.peer.deliver(peer.Outgoing("1.1", "es", "¿Dónde está la estación?", "en", "Where is the station?"))
        sent = ours.wait_for(is_a(ev.Sent), 5)
        result("the sentence is sent", sent is not None and sent.to == "rust-volis")
        rust.wait_for_line(r"paired mode: from pyvolis", 10, mark)
        time.sleep(0.3)
        got = "\n".join(rust.lines[mark:])
        result("Rust receives it: the translation and the original",
               "[es] ¿Dónde está la estación?" in got and "[en] Where is the station?" in got)
        ours.peer.release_floor()
        result("pyvolis hands the floor back", ours.wait_for(is_a(ev.FloorChanged, holder="free"), 5) is not None)

        print("\nRust volis hears Spanish (through the cable) and speaks to pyvolis")
        listening = rust.wait_for_line(r"capture started|listening continuously", 120)
        if listening is None:
            print("  (Rust's models are still loading...)")
        play(read_16k_mono(wavs[0]), devices[CABLE_IN].index, rate)
        remote = ours.wait_for(is_a(ev.Remote), 120)
        result("pyvolis receives Rust's translation, with the original",
               remote is not None and remote.lang == "en" and remote.source_lang == "es" and bool(remote.text),
               f"[{remote.source_lang}] {remote.source_text[:60]}... -> [{remote.lang}] {remote.text[:60]}..."
               if remote else "nothing arrived")

        print("\nthe pyvolis end is killed (its sockets close with no goodbye)")
        mark = len(rust.lines)
        for link in list(ours.peer._links):
            link.shut()
        line = rust.wait_for_line(r"paired mode: disconnected", 15, mark)
        result("Rust says so, by name", line is not None and "pyvolis" in line, (line or "").split("WARN ")[-1])
        gone = ours.wait_for(is_a(ev.PeerMsg, kind="disconnected"), 15)
        result("pyvolis is listening again", gone is not None)

        print("\n>>> In the Rust volis window: press Connect (the address is filled in). (Waiting up to 5 minutes.)",
              flush=True)
        hello = ours.wait_for(is_a(ev.PeerMsg, kind="connected"), 300)
        result("Rust dials pyvolis and they pair", hello is not None and hello.name == "rust-volis")

        print("\npyvolis holds the floor and Rust volis is killed")
        ours.peer.want_turn()
        ours.wait_for(is_a(ev.FloorChanged, holder="me"), 5)
        while not ours.asked.empty():
            ours.asked.get()
        rust.kill()
        gone = ours.wait_for(is_a(ev.PeerMsg, kind="disconnected"), 15)
        result("pyvolis says so, by name", gone is not None and "rust-volis" in gone.reason,
               gone.reason if gone else "")
        freed = ours.wait_for(is_a(ev.FloorChanged, holder="free"), 5)
        closed = None
        try:
            closed = ours.asked.get(timeout=5)
        except queue.Empty:
            pass
        result("the floor is released and the microphone closes", freed is not None and closed == "end_turn")
    finally:
        ours.peer.stop()
        rust.kill()
    failed = [what for what, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)} of {len(RESULTS)} passed" + (f"; failed: {'; '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
