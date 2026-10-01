"""What the offline test runs under the guard: the app's own start-up, every
library it uses, and one model of each backend that exists so far.

Grows with the milestones: recognition with each ASR backend is exercised by
tests/test_asr.py; here, models' configurations load and a sentence is translated. Prints "OFFLINE-WORKLOAD-DONE" at the end.
"""

import sys
from pathlib import Path

# The same order as the entry point: the environment before any library.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyvolis import paths  # noqa: E402

root = paths.app_root()
paths.apply_offline_environment(root)

from pyvolis.translate.llamacpp import load_cuda_runtime  # noqa: E402

load_cuda_runtime()  # a GPU build of llama.cpp needs it before the import

import av  # noqa: E402, F401
import huggingface_hub  # noqa: E402, F401
import llama_cpp  # noqa: E402, F401
import numpy  # noqa: E402, F401
import peft  # noqa: E402, F401
import PySide6.QtCore  # noqa: E402, F401
import sherpa_onnx  # noqa: E402, F401
import sounddevice  # noqa: E402, F401
import soxr  # noqa: E402, F401
import torch  # noqa: E402
import transformers  # noqa: E402

from pyvolis import cli, models  # noqa: E402

print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}")

# The report, exactly as a user runs it.
code = cli.run(["--report"], root)
assert code == 0, f"--report exited {code}"

# Each downloaded transformers model: its configuration and processor, the
# files a load reads first. Offline mode must find them in the folder.
for entry in models.discover(paths.asr_dir(root), models.Role.ASR):
    if isinstance(entry, models.Engine) and entry.backend == "transformers" and entry.enabled():
        transformers.AutoConfig.from_pretrained(entry.dir, local_files_only=True)
        transformers.AutoProcessor.from_pretrained(entry.dir, local_files_only=True)
        print(f"loaded the configuration and processor of {entry.dir_name}")

# One recognizer of each backend, loaded and run: the smallest of each kind.
import numpy as np  # noqa: E402

from pyvolis import asr  # noqa: E402

engines = [e for e in models.discover(paths.asr_dir(root), models.Role.ASR) if isinstance(e, models.Engine) and e.enabled()]
for backend_is_sherpa in (True, False):
    kind = [e for e in engines if e.from_engine_toml == backend_is_sherpa]
    if kind:
        smallest = min(kind, key=lambda e: sum(f.path.stat().st_size for f in e.files if f.present and f.path.is_file()))
        recognizer = asr.load(smallest)
        recognizer.transcribe(np.zeros(16000, np.float32), (smallest.languages or ["en"])[0])
        recognizer.close()
        print(f"loaded and ran {smallest.dir_name} ({smallest.backend})")

# The translator (llama.cpp): one sentence.
from pyvolis import translate as tr  # noqa: E402
from pyvolis.translate import prompts  # noqa: E402

translator = tr.load(tr.choose(root, ""), prompts.load(paths.prompts_dir(root)))
print("translated:", translator.translate(tr.TranslationRequest("¿Dónde está la estación?", "es", "en")).text)
translator.close()

# The guard itself: a direct connection out must be refused.
import socket  # noqa: E402

try:
    socket.create_connection(("1.1.1.1", 443), timeout=2)
except OSError as e:
    assert "offline guard" in str(e), e
    print("guard refused 1.1.1.1 as it should")
else:
    raise AssertionError("the guard let a connection to 1.1.1.1 through")

print("OFFLINE-WORKLOAD-DONE")
