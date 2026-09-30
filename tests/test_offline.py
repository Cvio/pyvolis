"""No library opens a connection to anywhere but the local network.

Runs `offline_workload.py` in a fresh process under `offline_guard.py`, so the
guard is in place before any library is imported, and fails naming the stack
of every refused attempt, including attempts a library caught and ignored.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_nothing_reaches_past_the_local_network():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_", "TRANSFORMERS_"))}
    done = subprocess.run(
        [sys.executable, str(HERE / "offline_guard.py"), str(HERE / "offline_workload.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=600,
    )
    attempts = [
        block.split("OFFLINE-GUARD-END")[0]
        for block in done.stderr.split("OFFLINE-GUARD: ")[1:]
        if "1.1.1.1" not in block.splitlines()[0]
    ]
    assert not attempts, "a library tried to reach the network:\n\n" + "\n".join(attempts)
    assert done.returncode == 0 and "OFFLINE-WORKLOAD-DONE" in done.stdout, (
        f"exit {done.returncode}\nstdout:\n{done.stdout[-3000:]}\nstderr:\n{done.stderr[-3000:]}"
    )
    assert "guard refused 1.1.1.1" in done.stdout


def test_the_guard_tells_local_from_internet():
    sys.path.insert(0, str(HERE))
    from offline_guard import is_local

    for host in ("127.0.0.1", "::1", "localhost", "192.168.50.2", "10.0.0.5", "169.254.49.46", "fe80::1"):
        assert is_local(host), host
    for host in ("1.1.1.1", "8.8.8.8", "huggingface.co", "2600:4040::1"):
        assert not is_local(host), host
