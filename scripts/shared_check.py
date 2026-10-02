"""The P10 check: the shared machine with real models.

    .venv\\Scripts\\python.exe scripts\\shared_check.py

Two people at one machine, English on the left and Spanish (Mexico) on the
right, each heard by their own recognizer. A recorded sentence stands in for
each person (the fixtures), fed in as a microphone would deliver it; the
recognizers, the translator and the voices are the real ones, and what is
spoken goes to the VB-Audio cable rather than the speakers.

Run once with the recognizers Rust volis has (Parakeet on the left, the
Spanish-tuned Whisper on the right: Rust's M7.6 check), then with a model
downloaded from Hugging Face on the right, then on the left.

For each turn it shows: which recognizer heard it and in which language, the
transcript and its error against the reference, the translation and its
language, and the voice that spoke it. The keys themselves, the two columns
and Escape are checked in tests\\test_shared_window.py and by hand.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pyvolis import paths  # noqa: E402

paths.apply_offline_environment(paths.app_root())
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

import logging  # noqa: E402

LOG_LINES: list[str] = []


class Keep(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        LOG_LINES.append(record.getMessage())


logging.getLogger("pyvolis").addHandler(Keep())
logging.getLogger("pyvolis").setLevel(logging.INFO)

from pyvolis import events as ev, models, scoring  # noqa: E402
from pyvolis import shared as sh  # noqa: E402
from pyvolis.config import Config, PyvolisConfig  # noqa: E402
from pyvolis.filesource import read_16k_mono  # noqa: E402
from pyvolis.pipeline import Options, Pipeline  # noqa: E402

CABLE_IN = "CABLE Input (VB-Audio Virtual Cable)"
RUNS = [
    ("Rust volis's models: Parakeet left, Spanish-tuned Whisper right",
     "parakeet-tdt-0.6b-v3-int8", "whisper-large-v3-turbo-es-adriszmar"),
    ("a downloaded model on the right", "parakeet-tdt-0.6b-v3-int8", "whisper-large-v3-turbo-es"),
    ("downloaded models on both sides", "whisper-small", "whisper-large-v3-turbo-es"),
]


class Clips:
    """A microphone that plays whichever recording was put in it last."""

    lossless = False
    duration = None

    def __init__(self) -> None:
        self.audio = None
        self.fed = threading.Event()
        self._stop = threading.Event()

    def start(self, out: queue.Queue) -> None:
        self._stop.clear()
        self.fed.clear()
        audio = self.audio

        def feed() -> None:
            for start in range(0, len(audio), 1600):
                if self._stop.is_set():
                    return
                out.put(audio[start:start + 1600])
                time.sleep(0.005)
            self.fed.set()

        threading.Thread(target=feed, daemon=True).start()

    def done(self) -> bool:
        return False

    def stop(self) -> None:
        self._stop.set()


def clips(name: str, count: int) -> list[tuple[Path, str]]:
    folder = REPO / "tests" / "fixtures" / "fleurs" / name
    refs = [json.loads(x) for x in (folder / "refs.jsonl").read_text(encoding="utf-8").splitlines()]
    return [(folder / r["file"], r["raw_transcript"]) for r in refs[:count]]


def run(root: Path, title: str, left_asr: str, right_asr: str, turns: int) -> bool:
    print(f"\n== {title}\n   left: English, {left_asr}; right: Spanish (Mexico), {right_asr}")
    asr_dir = paths.asr_dir(root)
    for folder in (left_asr, right_asr):
        if not (asr_dir / folder).is_dir():
            print(f"   (skipped: {asr_dir / folder} is not installed)")
            return True
    config = Config.parse(
        f'[asr]\nengine = "{left_asr}"\n[mode]\nkind = "shared"\n[tts]\nenabled = true\nhalf_duplex = true\n'
        f'[audio]\noutput_device = "{CABLE_IN}"\n'
        f'[shared]\nleft_language = "en"\nright_language = "es-MX"\nleft_asr = "{left_asr}"\nright_asr = "{right_asr}"\n')
    pyconfig, _ = PyvolisConfig.load(paths.pyvolis_config_file(root))
    events: queue.Queue = queue.Queue()
    mic = Clips()
    pipeline = Pipeline(root, config, Options(), events, mic, pyconfig).start()
    engines = [e for e in models.discover(asr_dir, models.Role.ASR) if isinstance(e, models.Engine)]
    voices = [e for e in models.discover(paths.tts_dir(root), models.Role.TTS) if isinstance(e, models.Engine)]
    seen: list = []

    def wait_for(want, seconds: float = 180):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                event = events.get(timeout=0.1)
            except queue.Empty:
                continue
            seen.append(event)
            if isinstance(event, ev.Error):
                print(f"   error: {event.message}")
            if want(event):
                return event
        return None

    ok = True
    try:
        if wait_for(lambda e: isinstance(e, ev.Listening)) is None:
            print("   FAIL: never started listening")
            return False
        problems = {e.side: e.problem for e in seen if isinstance(e, ev.SharedSide) and e.problem}
        if problems:
            print(f"   FAIL: {problems}")
            return False
        loaded = [e.name for e in seen if isinstance(e, ev.ModelLoaded) and e.role.startswith("recognizer")]
        print(f"   recognizers loaded: {', '.join(loaded)}")
        english, spanish = clips("en_us", turns), clips("es_419", turns)
        for i in range(turns):
            for side, (path, reference) in (("left", english[i]), ("right", spanish[i])):
                resolved = sh.direction(side, config.shared, engines, config.asr.engine, voices)
                direction = resolved.direction
                mic.audio = read_16k_mono(path)
                mark = len(LOG_LINES)
                pipeline.begin_shared_turn(direction)
                if wait_for(lambda e: isinstance(e, ev.TurnStarted) and e.side == side, 30) is None:
                    print(f"   FAIL: the {side} turn never started")
                    return False
                mic.fed.wait(60)
                time.sleep(0.3)
                pipeline.end_turn()
                finals, translated, spoken = [], [], []
                deadline = time.monotonic() + 120
                # Until everything the turn produced has been spoken.
                while time.monotonic() < deadline:
                    event = wait_for(lambda e: isinstance(e, (ev.Final, ev.Translated, ev.SpeakingStarted,
                                                               ev.NothingRecognized, ev.Dropped, ev.NotTranslated)), 5)
                    if isinstance(event, ev.Final):
                        finals.append(event)
                    elif isinstance(event, ev.Translated):
                        translated.append(event)
                    elif isinstance(event, ev.SpeakingStarted):
                        spoken.append(event)
                    elif event is not None:
                        break
                    sentences = [e for e in seen if isinstance(e, ev.SentenceMsg) and finals
                                 and e.utterance == finals[-1].index]
                    if finals and sentences and len(translated) >= len(sentences) and len(spoken) >= len(translated):
                        break
                wait_for(lambda e: isinstance(e, ev.SpeakingEnded), 60)
                told = next((line for line in LOG_LINES[mark:] if line.startswith("  transcribing as")), "")
                text = " ".join(f.text for f in finals)
                cer = scoring.cer(reference, text, direction.source) if text else 100.0
                langs = {t.lang for t in translated}
                voices_used = {s.voice for s in spoken}
                good = (bool(text) and cer < 35 and langs == {direction.target} and bool(spoken)
                        and f'"{direction.source}"' in told and f'"{direction.asr}"' in told)
                ok &= good
                print(f"   {'ok  ' if good else 'FAIL'} {side:5s} heard by {direction.asr} as {direction.source}"
                      f" (CER {cer:.1f}%)\n          [{direction.source}] {text[:110]}"
                      f"\n          [{'/'.join(sorted(langs)) or '-'}] {' '.join(t.text for t in translated)[:110]}"
                      f"\n          spoken by {', '.join(sorted(voices_used)) or 'nothing'}; log: {told.strip()}")
    finally:
        pipeline.stop()
        pipeline.join(30)
    return ok


def main() -> int:
    root = paths.app_root()
    turns = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    results = [(title, run(root, title, left, right, turns)) for title, left, right in RUNS]
    print()
    for title, ok in results:
        print(f"{'ok  ' if ok else 'FAIL'} {title}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
