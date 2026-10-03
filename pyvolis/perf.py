"""What pyvolis costs this machine, and what another choice of models would.

New in pyvolis (the performance panel, `gui/perf_panel.py`). Three parts:

* `Sampler`: once a second, the GPU (memory, load, temperature, clock,
  power), system memory and the CPU, for the machine and for this process.
  The GPU is read with `nvidia-smi`, which every NVIDIA driver installs: a
  short-lived program that reads counters, not a service and not inference.
  No new library is needed. Windows doesn't report a program's own GPU memory
  (`nvidia-smi` says N/A for every process), so pyvolis's share is the sum of
  what its loaded models take, and "other programs" is the rest.
* `estimate`: the memory a recognizer, translator or voice would take, without
  loading it: measured on this machine when it has been loaded before, else
  worked out from its files.
* `Measurements`: what each loaded model took and how fast each run went,
  kept in `logs/performance.json`, so the estimates and the speed figures get
  better with use. Nothing leaves the machine.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import models, paths

log = logging.getLogger(__name__)

SAMPLE_S = 1.0
HISTORY = 300  # samples kept: five minutes
KV_HEADROOM = 1.15  # a GGUF model on the GPU takes about its file size, plus its context cache
AMBER, RED = 0.90, 0.97  # share of memory in use


# ---------------------------------------------------------------- sampling


@dataclass
class Gpu:
    name: str
    used: int  # bytes, every program
    total: int
    load: float  # percent
    temperature: float  # degrees C
    clock_mhz: float
    power_w: float


@dataclass
class Sample:
    t: float  # time.monotonic()
    gpu: Gpu | None
    ram_total: int
    ram_used: int  # every program
    ram_ours: int  # this process
    cpu: float  # percent, the whole machine
    cpu_ours: float  # percent of one core, this process


def read_gpu() -> Gpu | None:
    """The first NVIDIA GPU, or None when there is none (or no driver)."""
    query = "name,memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm,power.draw"
    try:
        done = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=3,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0 or not done.stdout.strip():
        return None
    return parse_gpu(done.stdout.splitlines()[0])


def parse_gpu(line: str) -> Gpu | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7:
        return None

    def number(text: str) -> float:
        try:
            return float(text)
        except ValueError:  # "[N/A]" on some cards
            return 0.0

    mib = 1024 * 1024
    return Gpu(parts[0], int(number(parts[1]) * mib), int(number(parts[2]) * mib), number(parts[3]),
               number(parts[4]), number(parts[5]), number(parts[6]))


class Sampler:
    """Samples once a second on its own thread while the panel is open."""

    def __init__(self, read_gpu=read_gpu) -> None:
        import psutil

        self._psutil = psutil
        self._process = psutil.Process()
        self._read_gpu = read_gpu
        self.history: deque[Sample] = deque(maxlen=HISTORY)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process.cpu_percent(None)
        psutil.cpu_percent(None)

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="pyvolis-perf", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def sample(self) -> Sample:
        memory = self._psutil.virtual_memory()
        return Sample(time.monotonic(), self._read_gpu(), memory.total, memory.total - memory.available,
                      self._process.memory_info().rss, self._psutil.cpu_percent(None),
                      self._process.cpu_percent(None))

    def latest(self) -> Sample | None:
        return self.history[-1] if self.history else None

    def _run(self) -> None:
        while not self._stop.is_set():
            began = time.monotonic()
            try:
                self.history.append(self.sample())
            except Exception as e:  # a sampler must never take the window down
                log.debug("performance sample failed: %s", e)
            self._stop.wait(max(SAMPLE_S - (time.monotonic() - began), 0.1))


# ---------------------------------------------------------------- measurements


@dataclass
class Run:
    when: str
    recognizer: str
    translator: str
    mode: str
    asr_rtf: float | None
    translate_ms: int | None
    first_audio_ms: int | None
    sentences: int
    peak_gpu_used: int  # bytes, every program, the highest seen during the run


@dataclass
class Measurements:
    """`logs/performance.json`: model memory as loaded here, and runs."""

    path: Path
    models: dict[str, dict] = field(default_factory=dict)  # "role:name" -> {device, gpu_bytes, cpu_bytes}
    runs: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, root: Path) -> Measurements:
        path = paths.performance_file(root)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(path, data.get("models", {}), data.get("runs", []))
        except (OSError, ValueError):
            return cls(path)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"models": self.models, "runs": self.runs[-500:]}, indent=1),
                                 encoding="utf-8")
        except OSError as e:
            log.warning("could not write %s: %s", self.path.absolute(), e)

    def model(self, role: str, name: str, device: str, gpu_bytes: int, cpu_bytes: int) -> None:
        self.models[f"{role}:{name}"] = {"device": device, "gpu_bytes": gpu_bytes, "cpu_bytes": cpu_bytes}
        self.save()

    def run(self, run: Run) -> None:
        self.runs.append(asdict(run))
        self.save()

    def speed(self, recognizer: str, translator: str) -> dict:
        """Median figures of the runs with this recognizer and translator."""
        same = [r for r in self.runs if r["recognizer"] == recognizer and r["translator"] == translator]

        def median(key: str):
            values = sorted(r[key] for r in same if r.get(key) is not None)
            return values[len(values) // 2] if values else None

        return {"runs": len(same), "asr_rtf": median("asr_rtf"), "translate_ms": median("translate_ms"),
                "first_audio_ms": median("first_audio_ms")}


# ---------------------------------------------------------------- estimates


@dataclass
class Estimate:
    role: str
    name: str
    device: str  # "cuda" | "cpu"
    gpu_bytes: int
    cpu_bytes: int
    measured: bool  # loaded on this machine before, rather than worked out from files


def _size(files) -> int:
    return sum(f.stat().st_size for f in files if f.is_file())


def _weights(folder: Path) -> int:
    return _size(list(folder.glob("*.safetensors")) + list(folder.glob("*.bin")))


def _stored_as_float32(folder: Path) -> bool:
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return str(config.get("torch_dtype") or config.get("dtype") or "").endswith("float32")


def estimate_recognizer(engine: models.Engine, measured: Measurements | None, gpu: bool) -> Estimate:
    known = (measured.models if measured else {}).get(f"recognizer:{engine.dir_name}")
    if known:
        return Estimate("recognizer", engine.dir_name, known["device"], known["gpu_bytes"], known["cpu_bytes"], True)
    files = [f.path for f in engine.files]
    if engine.from_engine_toml:  # sherpa-onnx: always the CPU
        return Estimate("recognizer", engine.dir_name, "cpu", 0, _size(files), False)
    device = engine.settings.get("device") or ("cuda" if gpu else "cpu")
    if engine.backend == models.LLAMACPP_AUDIO:
        need = int(_size(files) * KV_HEADROOM)
    else:  # transformers: loaded in float16 on the GPU, so a float32 file takes half its size
        need = _weights(engine.dir) or _size(files)
        if (engine.dir / "adapter_config.json").is_file():  # a LoRA adapter: its base is what takes the room
            base = engine.settings.get("base_dir")
            need += _weights(Path(base)) if base else 0
        if device == "cuda" and _stored_as_float32(engine.dir):
            need //= 2
    return Estimate("recognizer", engine.dir_name, device, need if device == "cuda" else 0,
                    0 if device == "cuda" else need, False)


def estimate_translator(entry: models.Translator, measured: Measurements | None, gpu: bool) -> Estimate:
    known = (measured.models if measured else {}).get(f"translator:{entry.id}")
    if known:
        return Estimate("translator", entry.id, known["device"], known["gpu_bytes"], known["cpu_bytes"], True)
    size = entry.size_bytes or (_size([entry.path]) if entry.path.is_file() else _weights(entry.path))
    if entry.lora is not None:
        size += _size([entry.lora])
    device = entry.settings.get("device") or ("cuda" if gpu else "cpu")
    need = int(size * KV_HEADROOM)
    return Estimate("translator", entry.id, device, need if device == "cuda" else 0, 0 if device == "cuda" else need,
                    False)


def estimate_voice(engine: models.Engine) -> Estimate:
    return Estimate("voice", engine.dir_name, "cpu", 0, _size([f.path for f in engine.files]), False)


@dataclass
class Verdict:
    level: str  # "fits" | "tight" | "no"
    text: str
    gpu_need: int
    gpu_room: int  # what the GPU has for pyvolis, with other programs' use left in place
    ram_need: int
    ram_room: int


def verdict(estimates: list[Estimate], sample: Sample | None, ours_gpu_now: int = 0,
            ours_ram_now: int = 0) -> Verdict:
    """Whether a set of models fits this machine as it is now. pyvolis's own
    current use counts as room, since it would be replaced."""
    gpu_need = sum(e.gpu_bytes for e in estimates)
    ram_need = sum(e.cpu_bytes for e in estimates)
    gpu = sample.gpu if sample else None
    gpu_room = (gpu.total - max(gpu.used - ours_gpu_now, 0)) if gpu else 0
    ram_room = (sample.ram_total - max(sample.ram_used - ours_ram_now, 0)) if sample else 0
    if gpu_need and not gpu:
        return Verdict("no", "needs an NVIDIA GPU, and none was found", gpu_need, 0, ram_need, ram_room)
    worst = max(gpu_need / gpu_room if gpu_room else 0.0, ram_need / ram_room if ram_room else 0.0)
    gb = 1e9
    room = f"GPU {gpu_need / gb:.1f} of {gpu_room / gb:.1f} GB free for it, memory {ram_need / gb:.1f} of {ram_room / gb:.1f} GB"
    if worst > 1:
        return Verdict("no", f"doesn't fit: {room}. It would spill into system memory and run several times "
                       "slower.", gpu_need, gpu_room, ram_need, ram_room)
    if worst > AMBER:
        return Verdict("tight", f"fits, barely: {room}. Little room for another program.", gpu_need, gpu_room,
                       ram_need, ram_room)
    return Verdict("fits", f"fits: {room}.", gpu_need, gpu_room, ram_need, ram_room)


def level(used: int, total: int) -> str:
    """"ok", "amber" or "red" for a share of memory in use."""
    if not total:
        return "ok"
    share = used / total
    return "red" if share >= RED else "amber" if share >= AMBER else "ok"


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M")

