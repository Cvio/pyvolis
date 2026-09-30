"""Cut an audio file into utterances with and without the pre-roll, and print
the cut points side by side (the P1 check).

    .venv\\Scripts\\python.exe scripts\\vad_cuts.py [audio file] [--wav]
    .venv\\Scripts\\python.exe scripts\\vad_cuts.py --mic 20 --wav     (from the microphone)

With no file, a test conversation is built from the Spanish FLEURS fixtures
(tests\\fetch-fixtures.ps1): each clip followed by 1.2 s of quiet room hiss, so
where every clip begins in the file is known. `--wav` writes every cut, both
ways, to logs\\segments\\ to listen to.

Columns: the detector's start (where Silero became confident), and each cut's
start and end. "lead-in" is how much audio the pre-roll restored in front of
the detector's start. A cut never reaches back into the previous one.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pyvolis import paths  # noqa: E402

paths.apply_offline_environment(paths.app_root())

from pyvolis.audio import SAMPLE_RATE  # noqa: E402
from pyvolis.config import Config, PyvolisConfig  # noqa: E402
from pyvolis.filesource import chunks, read_16k_mono  # noqa: E402
from pyvolis.vad import Segmenter, VadSettings  # noqa: E402

GAP_SECONDS = 1.2


def conversation(lang: str = "es_419") -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """FLEURS clips joined by quiet hiss. Returns the audio and, per clip,
    (text, start sample, end sample)."""
    folder = REPO / "tests" / "fixtures" / "fleurs" / lang
    refs = folder / "refs.jsonl"
    if not refs.is_file():
        raise SystemExit(f"STOP: no fixtures at {folder}. Run .\\tests\\fetch-fixtures.ps1 first.")
    rng = np.random.default_rng(0)
    parts, clips, position = [], [], 0
    for line in refs.read_text(encoding="utf-8").splitlines():
        ref = json.loads(line)
        audio = read_16k_mono(folder / ref["file"])
        gap = (rng.standard_normal(int(GAP_SECONDS * SAMPLE_RATE)) * 0.001).astype(np.float32)
        clips.append((ref["raw_transcript"], position, position + len(audio)))
        parts += [audio, gap]
        position += len(audio) + len(gap)
    return np.concatenate(parts), clips


def record(seconds: float, root: Path) -> np.ndarray:
    """Record from the configured microphone through the capture path."""
    import queue
    import time

    from pyvolis.audio import spawn_capture

    config, _ = Config.load(paths.config_file(root))
    q: queue.Queue = queue.Queue()
    print(f"recording {seconds:.0f} s: speak a few sentences with pauses between them...")
    handle = spawn_capture(config.audio.input_device, q)
    time.sleep(seconds)
    handle.stop()
    parts = []
    while not q.empty():
        parts.append(q.get())
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def cut(audio: np.ndarray, pre_roll_ms: int, root: Path):
    config, _ = Config.load(paths.config_file(root))
    settings = VadSettings(
        model=paths.vad_model_file(root),
        threshold=config.vad.threshold,
        min_silence_ms=config.vad.min_silence_ms,
        min_speech_ms=config.vad.min_speech_ms,
        pre_roll_ms=pre_roll_ms,
    )
    segmenter = Segmenter(settings)
    segments = []
    for chunk in chunks(audio, 700):  # doesn't divide the 512 window: exercises carry-over
        segments += segmenter.push(chunk)
    return segments + segmenter.flush()


def ms(samples: int) -> str:
    return f"{samples * 1000 // SAMPLE_RATE:>7}"


def write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", type=Path)
    parser.add_argument("--wav", action="store_true")
    parser.add_argument("--mic", type=float, metavar="SECONDS", help="record from [audio].input_device")
    args = parser.parse_args()
    root = paths.app_root()
    pre_roll_ms = PyvolisConfig.load(paths.pyvolis_config_file(root))[0].vad.pre_roll_ms or 600

    if args.mic:
        audio, clips = record(args.mic, root), []
    elif args.file:
        audio, clips = read_16k_mono(args.file), []
        print(f"{args.file.absolute()}: {len(audio) / SAMPLE_RATE:.1f} s")
    else:
        audio, clips = conversation()
        print(f"test conversation: {len(clips)} FLEURS es_419 clips, {GAP_SECONDS} s gaps, "
              f"{len(audio) / SAMPLE_RATE:.1f} s")

    off, on = cut(audio, 0, root), cut(audio, pre_roll_ms, root)
    print(f"{len(off)} cuts without pre-roll, {len(on)} with {pre_roll_ms} ms\n")
    print("   #  clip at |  detected | without: start     end |  with: start     end  lead-in")
    for i, (a, b) in enumerate(zip(off, on)):
        clip = next((c for c in clips if c[1] <= b.detected_sample < c[2] + SAMPLE_RATE), None)
        clip_at = ms(clip[1]) if clip else "      -"
        print(f"  {i + 1:>2}  {clip_at} |  {ms(a.detected_sample)}  |  {ms(a.start_sample)} "
              f"{ms(a.start_sample + len(a.samples))}  | {ms(b.start_sample)} "
              f"{ms(b.start_sample + len(b.samples))}  {b.lead_in_ms():>5} ms")
    if len(off) != len(on):
        print("  (the two runs cut a different number of utterances)")

    previous_end, ok = 0, True
    for s in on:
        if s.start_sample < previous_end:
            ok = False
            print(f"  cut at {s.start_ms()} ms reaches into the previous one")
        previous_end = s.start_sample + len(s.samples)
    exact = all(np.array_equal(s.samples, audio[s.start_sample : s.start_sample + len(s.samples)]) for s in on)
    print(f"\nwith pre-roll: every cut is the audio it claims: {'yes' if exact else 'NO'}; "
          f"none overlaps the previous: {'yes' if ok else 'NO'}")
    if clips:
        missed = [c for c in clips if not any(c[1] - SAMPLE_RATE < s.detected_sample < c[2] for s in on)]
        print(f"clips with no cut: {len(missed)}")

    if args.wav:
        out = paths.logs_dir(root) / "segments"
        for name, segments in (("no-preroll", off), (f"preroll-{pre_roll_ms}ms", on)):
            for i, s in enumerate(segments):
                write_wav(out / name / f"{i + 1:02d}-{s.start_ms()}ms.wav", s.samples)
        print(f"wrote the cuts to {out}")
    return 0 if exact and ok else 1


if __name__ == "__main__":
    sys.exit(main())
