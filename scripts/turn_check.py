"""Turn-based mode against real devices, through the VB-Audio cable (the
Rust M6 checks, repeated).

    .venv\\Scripts\\python.exe scripts\\turn_check.py

The cable is the microphone ("CABLE Output") and the speakers ("CABLE Input").
  1. Between turns the microphone is closed: speech played while idle produces
     nothing, no level reports, and the capture device isn't open.
  2. A turn captures everything said in it as one utterance, pauses included:
     two clips with a 2 s pause between them come out as one transcript.
  3. Taking a turn stops a reply mid-word.
  4. The mode changes while running: continuous then hears without the key.
Nothing comes out of the real speakers.
"""

from __future__ import annotations

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


class Run:
    def __init__(self) -> None:
        config = Config.parse(
            '[asr]\nengine = "parakeet-tdt-0.6b-v3-int8"\n[languages]\nsource = "es"\ntarget = "en"\n'
            '[mode]\nkind = "turn"\nturn_style = "toggle"\n'
            f'[audio]\ninput_device = "{CABLE_OUT}"\noutput_device = "{CABLE_IN}"\n'
            "[tts]\nenabled = true\nhalf_duplex = true\n"
        )
        self.events: queue.Queue = queue.Queue()
        self.pipeline = Pipeline(paths.app_root(), config, Options(), self.events, None, PyvolisConfig()).start()
        self.seen: list = []
        self.talker = playback.Player(CABLE_IN, playback.Gate(False))

    def drain(self, seconds: float, until=None) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            try:
                event = self.events.get(timeout=0.05)
            except queue.Empty:
                continue
            self.seen.append(event)
            if isinstance(event, ev.Error):
                print("  error:", event.message, flush=True)
            if until is not None and isinstance(event, until):
                return

    def since(self, mark: int, kind) -> list:
        return [e for e in self.seen[mark:] if isinstance(e, kind)]

    def say(self, clip: np.ndarray, wait: float = 0.3) -> None:
        self.talker.play(clip, 16_000)
        self.drain(len(clip) / 16_000 + wait)

    def close(self) -> None:
        self.pipeline.stop()
        self.drain(3.0, until=ev.Stopped)
        self.pipeline.join(10)
        self.talker.close()


def main() -> int:
    folder = REPO / "tests" / "fixtures" / "fleurs" / "es_419"
    refs = [json.loads(x) for x in (folder / "refs.jsonl").read_text(encoding="utf-8").splitlines()]
    clips = [read_16k_mono(folder / r["file"]) for r in refs[:4]]
    ok = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        ok.append(passed)
        print(f"  {'pass' if passed else 'FAIL'}  {name}{('  (' + detail + ')') if detail else ''}", flush=True)

    run = Run()
    try:
        run.drain(90, until=ev.Mode)
        mode = run.since(0, ev.Mode)
        check("starts in turn mode", bool(mode) and mode[-1].kind == "turn")

        print("\n1. between turns the microphone is closed", flush=True)
        mark = len(run.seen)
        run.say(clips[0], wait=4.0)
        check("speech while idle produces no transcript", not run.since(mark, (ev.Final, ev.SentenceMsg)))
        check("no level is reported: nothing is being captured", not run.since(mark, ev.Level))
        check("the capture device is not open", run.pipeline.source._handle is None)

        print("\n2. a turn is one utterance, pauses and all", flush=True)
        mark = len(run.seen)
        run.pipeline.begin_turn()
        run.drain(5, until=ev.TurnStarted)
        check("the turn opens the microphone", bool(run.since(mark, ev.TurnStarted)) and run.pipeline.source._handle is not None)
        run.say(clips[1], wait=2.0)  # a 2 s pause: the detector would cut here in continuous mode
        run.say(clips[2], wait=0.8)
        check("nothing is transcribed while the turn is open", not run.since(mark, ev.Final))
        run.pipeline.end_turn()
        run.drain(5, until=ev.TurnEnded)
        check("ending the turn closes the microphone", run.pipeline.source._handle is None)
        run.drain(40, until=ev.SpeakingStarted)
        finals = run.since(mark, ev.Final)
        text = finals[0].text if finals else ""
        both = refs[1]["raw_transcript"].split()[1].strip(",.").lower() in text.lower() and \
            refs[2]["raw_transcript"].split()[-2].strip(",.").lower() in text.lower()
        check("the whole turn is one transcript", len(finals) == 1 and both, f"{len(finals)} transcript(s), {len(text)} characters")
        sentences = run.since(mark, ev.SentenceMsg)
        check("its sentences are translated one by one", len(sentences) >= 2, f"{len(sentences)} sentences")
        print(f"       {text[:150]}...", flush=True)

        print("\n3. taking a turn stops the reply", flush=True)
        check("a reply is being spoken", run.pipeline_speaking())
        mark = len(run.seen)
        run.pipeline.begin_turn()
        run.drain(5, until=ev.TurnStarted)
        # The microphone is open now and hears the cable: if the reply were
        # still playing, the level meter would show it.
        level_mark = len(run.seen)
        run.drain(1.2)
        levels = [e.db for e in run.since(level_mark, ev.Level)]
        loudest = max((db for db in levels if db is not None), default=None)
        check("the reply is cut off at once", bool(levels) and (loudest is None or loudest < -50),
              f"loudest level in the next second: {'silence' if loudest is None else f'{loudest:.0f} dBFS'}")
        run.say(clips[3], wait=0.8)
        run.pipeline.end_turn()
        run.drain(40, until=ev.Translated)
        heard = run.since(mark, ev.Final)
        check("the new turn is heard from its first word",
              bool(heard) and refs[3]["raw_transcript"].split()[0].lower() in heard[0].text.lower(),
              heard[0].text[:60] if heard else "nothing")
        run.drain(25, until=ev.SpeakingEnded)

        print("\n4. the mode changes while running", flush=True)
        mark = len(run.seen)
        run.pipeline.set_mode("continuous")
        run.drain(5, until=ev.Mode)
        check("it says it is now continuous", bool(run.since(mark, ev.Mode)) and run.since(mark, ev.Mode)[-1].kind == "continuous")
        run.say(clips[0], wait=6.0)
        check("continuous mode hears without the key", bool(run.since(mark, ev.Final)))
        run.pipeline.set_mode("turn")
        run.drain(5, until=ev.Mode)
        check("and back: the microphone closes again", run.pipeline.source._handle is None)
    finally:
        run.close()
    print(f"\n{sum(ok)} of {len(ok)} checks passed")
    return 0 if all(ok) else 1


def _speaking(self) -> bool:
    """Whether the voice has audio queued for the sound card right now."""
    for thread_event in reversed(self.seen):
        if isinstance(thread_event, ev.SpeakingEnded):
            return False
        if isinstance(thread_event, ev.SpeakingStarted):
            return True
    return False


Run.pipeline_speaking = _speaking


if __name__ == "__main__":
    sys.exit(main())
