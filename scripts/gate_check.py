"""Does pyvolis hear itself? The live pipeline with the voice on, run twice
through the VB-Audio cable: half-duplex on, then off (Rust's M4 check).

    .venv\\Scripts\\python.exe scripts\\gate_check.py [--clips 2]

The cable is both the microphone ("CABLE Output") and the speakers ("CABLE
Input"), so everything pyvolis says comes straight back into its microphone:
the worst case for feedback. Spanish fixture clips are played into the cable.
With the gate on, only that Spanish is transcribed. With it off, pyvolis
transcribes its own English voice as well, which is what the gate is for.
Nothing comes out of the real speakers.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pyvolis import paths  # noqa: E402

paths.apply_offline_environment(paths.app_root())
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from pyvolis import events as ev, playback  # noqa: E402
from pyvolis.config import Config, PyvolisConfig  # noqa: E402
from pyvolis.filesource import read_16k_mono  # noqa: E402
from pyvolis.pipeline import Options, Pipeline  # noqa: E402

CABLE_IN, CABLE_OUT = "CABLE Input (VB-Audio Virtual Cable)", "CABLE Output (VB-Audio Virtual Cable)"


def run(half_duplex: bool, clips: list[np.ndarray]) -> tuple[list[str], int]:
    root = paths.app_root()
    config = Config.parse(
        '[asr]\nengine = "parakeet-tdt-0.6b-v3-onnx-int8"\n[languages]\nsource = "es"\ntarget = "en"\n'
        f'[audio]\ninput_device = "{CABLE_OUT}"\noutput_device = "{CABLE_IN}"\n'
        f"[tts]\nenabled = true\nhalf_duplex = {'true' if half_duplex else 'false'}\n"
    )
    events: queue.Queue = queue.Queue()
    pipeline = Pipeline(root, config, Options(), events, None, PyvolisConfig()).start()
    heard, spoken, listening, stopped = [], 0, False, False
    # The "person talking": a second player on the same cable, with no gate of its own.
    talker = playback.Player(CABLE_IN, playback.Gate(False))
    deadline, pending = None, list(clips)

    def drain(seconds: float) -> None:
        nonlocal spoken, listening, stopped
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            try:
                event = events.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(event, ev.Listening):
                listening = True
            elif isinstance(event, ev.SentenceMsg):
                heard.append(event.text)
            elif isinstance(event, ev.SpeakingStarted):
                spoken += 1
            elif isinstance(event, ev.Error):
                print("  error:", event.message, flush=True)
            elif isinstance(event, ev.Stopped):
                stopped = True
                return

    while not listening and not stopped:
        drain(0.5)
    if stopped:
        talker.close()
        raise SystemExit("STOP: the pipeline stopped before it was listening (see the error above)")
    for clip in pending:
        talker.play(clip, 16_000)
        drain(len(clip) / 16_000 + 14.0)  # recognition, translation, the voice, and whatever comes back
    drain(6.0)
    pipeline.stop()
    drain(2.0)
    pipeline.join(10)
    talker.close()
    return heard, spoken


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=int, default=2)
    args = parser.parse_args()
    folder = REPO / "tests" / "fixtures" / "fleurs" / "es_419"
    refs = [json.loads(x) for x in (folder / "refs.jsonl").read_text(encoding="utf-8").splitlines()][: args.clips]
    clips = [read_16k_mono(folder / r["file"]) for r in refs]
    results = {}
    for half_duplex in (True, False):
        print(f"\n== half-duplex {'on' if half_duplex else 'off'}")
        heard, spoken = run(half_duplex, clips)
        results[half_duplex] = heard
        print(f"  pyvolis spoke {spoken} sentence(s); transcribed {len(heard)}:")
        for text in heard:
            print(f"    {text}")
    on, off = results[True], results[False]
    print(f"\nwith the gate: {len(on)} transcript(s); without it: {len(off)}")
    extra = len(off) - len(on)
    print("the gate keeps pyvolis from hearing itself: " +
          ("yes" if on and extra > 0 else "NOT SHOWN (no difference between the two runs)"))
    return 0 if on and extra > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
