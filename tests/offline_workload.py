"""What the offline test runs under the guard: the app's own start-up, every
library it uses, and one model of each backend that exists so far.

Grows with the milestones: P2 adds recognition with each ASR backend, P3 a
translated sentence. Prints "OFFLINE-WORKLOAD-DONE" at the end.
"""

import sys
from pathlib import Path

# The same order as the entry point: the environment before any library.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyvolis import paths  # noqa: E402

root = paths.app_root()
paths.apply_offline_environment(root)

import av  # noqa: E402, F401
import huggingface_hub  # noqa: E402, F401
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
