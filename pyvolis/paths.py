"""Every path in pyvolis derives from the application's own location.

Port of Rust `paths.rs`. Never the current working directory, never
`%APPDATA%`, `%LOCALAPPDATA%` or `~/.cache`. `app_root()` is the *only*
path-derivation function; every other location is a join off it, and those
joins live here too so there is one place to read.

This module imports only the standard library: the entry point uses it to set
the environment before any other library is imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def app_root() -> Path:
    """The folder the application lives in.

    Built (PyInstaller): the folder holding `pyvolis.exe`. From source: the
    repository root, the folder holding the `pyvolis` package. Either way,
    double-clicking and launching from a shell behave the same.
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent.parent
    return strip_verbatim(base)


def strip_verbatim(path: Path) -> Path:
    r"""Drop a `\\?\` prefix when what remains is an ordinary drive path.

    Every error message quotes an absolute path back to the user, and the
    verbatim form is correct but unreadable. UNC paths keep the prefix, where
    it is load-bearing.
    """
    text = str(path)
    if not text.startswith("\\\\?\\"):
        return path
    rest = text[4:]
    if len(rest) >= 3 and rest[0].isascii() and rest[0].isalpha() and rest[1:3] == ":\\":
        return Path(rest)
    return path


def config_file(root: Path) -> Path:
    """`<root>/volis.toml` - user selections, shared with Rust volis."""
    return root / "volis.toml"


def pyvolis_config_file(root: Path) -> Path:
    """`<root>/pyvolis.toml` - settings Rust volis doesn't have."""
    return root / "pyvolis.toml"


def hallucinations_file(root: Path) -> Path:
    """`<root>/config/hallucinations.toml` - stock phrases, user-editable."""
    return root / "config" / "hallucinations.toml"


def prompts_dir(root: Path) -> Path:
    """`<root>/prompts` - the translator's system text, one file per variant."""
    return root / "prompts"


def models_dir(root: Path) -> Path:
    return root / "models"


def vad_model_file(root: Path) -> Path:
    return models_dir(root) / "vad" / "silero_vad.onnx"


def asr_dir(root: Path) -> Path:
    return models_dir(root) / "asr"


def tts_dir(root: Path) -> Path:
    return models_dir(root) / "tts"


def mt_dir(root: Path) -> Path:
    return models_dir(root) / "mt"


def logs_dir(root: Path) -> Path:
    return root / "logs"


def cache_dir(root: Path) -> Path:
    """`<root>/cache` - where a library that insists on caching is sent.

    pyvolis itself caches nothing. This exists so that Hugging Face, PyTorch
    and the CUDA driver, which cache by default in the user profile, stay
    inside the application folder instead.
    """
    return root / "cache"


def offline_environment(root: Path) -> dict[str, str]:
    """The variables that keep every library offline and inside `root`.

    Values override whatever the user's environment holds: a personal
    `HF_HOME` must not pull pyvolis out of its own folder.
    """
    cache = cache_dir(root)
    return {
        # Hugging Face: no hub lookups, no telemetry, caches inside the app.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "DO_NOT_TRACK": "1",
        "HF_HOME": str(cache / "huggingface"),
        # PyTorch hub and compiled kernels.
        "TORCH_HOME": str(cache / "torch"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        # The CUDA driver's JIT cache defaults to %APPDATA%\NVIDIA\ComputeCache.
        "CUDA_CACHE_PATH": str(cache / "nvidia"),
        # numba (through librosa) caches compiled functions beside the code or
        # in the user profile; keep them here.
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        # Anything that follows the XDG convention.
        "XDG_CACHE_HOME": str(cache),
    }


def apply_offline_environment(root: Path) -> None:
    """Set `offline_environment(root)` in this process. Must run before any
    Hugging Face or PyTorch module is imported: they read it at import time."""
    os.environ.update(offline_environment(root))
