"""Check this machine's environment: `doctor.ps1`, or `python -m pyvolis.doctor`.

Modelled on model-converter's `0_doctor.py`. In order:

  1. Every native file (.dll, .pyd) the installed packages list is present and
     unaltered, against the SHA-256 in each package's RECORD. Security
     software on this machine has removed DLLs before; a package missing one
     imports fine until the moment it needs it.
  2. sherpa-onnx loads onnxruntime 1.28.2 (the version Rust volis links) from
     this environment, not another copy (Windows 11 has an old one in System32
     that crashes it).
  3. torch sees the GPU.
  4. llama-cpp-python loads, and whether it was built with GPU support.
  5. The CUDA runtime DLLs torch and llama.cpp each bring: which were loaded
     from where, and whether two copies of one library are loaded at once.

Informational: other onnxruntime.dll copies on PATH or in System32, and
McAfee/Trellix. Writes `logs/doctor.json` so two machines can be compared.
Exit code 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import sys

if __name__ == "__main__":
    # The environment before any library, exactly as the entry point does.
    from pyvolis import paths as _paths

    _paths.apply_offline_environment(_paths.app_root())

import base64
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from pathlib import Path

from pyvolis import paths

SHERPA_ORT_VERSION = "1.28.2"
NATIVE = {".dll", ".pyd", ".so"}
# CUDA runtime libraries whose duplicates matter: two copies of one of these in
# a process is how "works alone, crashes together" happens.
CUDA_DLL = re.compile(r"^(cudart|cublas|cublasLt|cudnn\w*|nvrtc|nvJitLink|cufft|curand|cusparse|cusolver)64_", re.I)
SECURITY_SERVICES = ["mfemms", "mfefire", "mfeesp", "mfetp", "masvc", "mfeatp"]
SECURITY_FOLDERS = ["McAfee", "Trellix"]


class Report:
    def __init__(self) -> None:
        self.results: dict[str, dict] = {}
        self.failed: list[str] = []

    def heading(self, text: str) -> None:
        print(f"\n== {text}")

    def ok(self, text: str) -> None:
        print(f"  ok    {text}")

    def info(self, text: str) -> None:
        print(f"  info  {text}")

    def fail(self, check: str, text: str) -> None:
        print(f"  FAIL  {text}")
        self.failed.append(f"{check}: {text}")


# ---------------------------------------------------------------- Windows DLL helpers


def loaded_modules() -> list[str]:
    """Full paths of every DLL loaded in this process (Windows only). From
    model-converter's engine_worker.py."""
    import ctypes
    from ctypes import wintypes

    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    process = kernel32.GetCurrentProcess()
    count = 1024
    while True:
        modules = (wintypes.HMODULE * count)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModules(process, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
            raise OSError(ctypes.get_last_error(), "EnumProcessModules failed")
        if needed.value <= ctypes.sizeof(modules):
            break
        count = needed.value // ctypes.sizeof(wintypes.HMODULE) + 16
    found = []
    buffer = ctypes.create_unicode_buffer(32768)
    for i in range(needed.value // ctypes.sizeof(wintypes.HMODULE)):
        if psapi.GetModuleFileNameExW(process, modules[i], buffer, len(buffer)):
            found.append(buffer.value)
    return found


def file_version(path: str) -> str:
    """The ProductVersion in a DLL's version resource, e.g. '1.28.2'."""
    import ctypes
    from ctypes import wintypes

    version = ctypes.WinDLL("version", use_last_error=True)
    size = version.GetFileVersionInfoSizeW(path, None)
    if not size:
        return "unknown"
    data = ctypes.create_string_buffer(size)
    if not version.GetFileVersionInfoW(path, 0, size, data):
        return "unknown"
    pointer = ctypes.c_void_p()
    length = wintypes.UINT()
    if (
        version.VerQueryValueW(data, "\\VarFileInfo\\Translation", ctypes.byref(pointer), ctypes.byref(length))
        and length.value >= 4
    ):
        lang, codepage = ctypes.cast(pointer, ctypes.POINTER(wintypes.WORD * 2)).contents
        key = f"\\StringFileInfo\\{lang:04x}{codepage:04x}\\ProductVersion"
        if version.VerQueryValueW(data, key, ctypes.byref(pointer), ctypes.byref(length)):
            return ctypes.wstring_at(pointer, length.value).rstrip("\x00").strip()
    return "unknown"


def _inside(path: str, folder: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(folder.resolve())
    except OSError:
        return False


# ---------------------------------------------------------------- checks


def check_native_files(r: Report) -> None:
    r.heading("Native files (.dll, .pyd) against each package's RECORD")
    missing, altered, unreadable, checked = [], [], [], 0
    for dist in importlib.metadata.distributions():
        # RECORD itself, not dist.files: since Python 3.12 dist.files leaves
        # out files that no longer exist, which are exactly the ones wanted.
        for row in csv.reader((dist.read_text("RECORD") or "").splitlines()):
            if not row or Path(row[0]).suffix.lower() not in NATIVE:
                continue
            name, recorded = row[0], row[1] if len(row) > 1 else ""
            path = Path(dist.locate_file(name))
            label = f"{dist.metadata['Name']}: {path}"
            if not path.is_file():
                missing.append(label)
                continue
            checked += 1
            if not recorded.startswith("sha256="):
                continue
            try:
                digest = hashlib.sha256()
                with open(path, "rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        digest.update(block)
            except OSError as e:  # locked or access denied, e.g. mid-scan
                unreadable.append(f"{label} ({e.strerror})")
                continue
            actual = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
            if actual != recorded.removeprefix("sha256="):
                altered.append(label)
    r.results["native_files"] = {
        "checked": checked, "missing": missing, "altered": altered, "unreadable": unreadable,
    }
    if missing or altered:
        for label in missing:
            r.fail("native_files", f"missing: {label}")
        for label in altered:
            r.fail("native_files", f"altered: {label}")
        r.info("If security software removed these, add an exclusion for "
               f"{paths.app_root()} and run .\\setup.ps1 -Reinstall")
    else:
        r.ok(f"{checked} native files present and unaltered")
    for label in unreadable:
        r.info(f"could not read (in use or blocked): {label}")


def check_sherpa(r: Report) -> None:
    r.heading(f"sherpa-onnx and its onnxruntime ({SHERPA_ORT_VERSION} expected)")
    if "onnxruntime" in sys.modules:
        r.fail("sherpa", "the onnxruntime Python package was imported first; it would claim onnxruntime.dll")
    try:
        import sherpa_onnx
    except Exception as e:  # noqa: BLE001 - any import failure is the finding
        r.fail("sherpa", f"import sherpa_onnx failed: {e}")
        return
    info = {"sherpa_onnx": getattr(sherpa_onnx, "__version__", "unknown")}
    r.results["sherpa"] = info
    r.ok(f"sherpa_onnx {info['sherpa_onnx']} imported")
    if os.name != "nt":
        r.info("the onnxruntime.dll check is Windows-only")
        return
    loaded = [p for p in loaded_modules() if Path(p).name.lower() == "onnxruntime.dll"]
    info["onnxruntime_dll"] = [f"{p} ({file_version(p)})" for p in loaded]
    if len(loaded) != 1:
        r.fail("sherpa", f"expected exactly one onnxruntime.dll loaded, found {loaded or 'none'}")
        return
    path, version = loaded[0], file_version(loaded[0])
    if not _inside(path, Path(sys.prefix)):
        r.fail("sherpa", f"onnxruntime.dll was loaded from {path} ({version}), outside this "
               f"environment ({sys.prefix}). The copy sherpa-onnx-core installs is missing: "
               "check native files above.")
    elif version != SHERPA_ORT_VERSION:
        r.fail("sherpa", f"{path} is onnxruntime {version}, not {SHERPA_ORT_VERSION}")
    else:
        r.ok(f"onnxruntime {version} from {path}")


def check_torch(r: Report) -> None:
    r.heading("PyTorch and the GPU")
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        r.fail("torch", f"import torch failed: {e}")
        return
    info = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "gpu": None}
    r.results["torch"] = info
    r.ok(f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        r.fail("torch", "torch does not see a CUDA GPU; models would run on the CPU")
        return
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    capability = ".".join(map(str, torch.cuda.get_device_capability(0)))
    info["gpu"] = {"name": name, "capability": capability, "free_gb": round(free / 1e9, 1),
                   "total_gb": round(total / 1e9, 1)}
    # A tiny kernel proves the build has code for this GPU (a 5090 on a
    # too-old CUDA build fails here, not at is_available()).
    try:
        (torch.ones(4, device="cuda") * 2).sum().item()
    except Exception as e:  # noqa: BLE001
        r.fail("torch", f"{name} is visible but a kernel failed to run: {e}")
        return
    r.ok(f"{name} (compute {capability}), {free / 1e9:.1f} of {total / 1e9:.1f} GB free, kernel ran")


def check_llama(r: Report) -> None:
    r.heading("llama-cpp-python")
    try:
        importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        r.info("not installed yet: it is added at P3, once its CUDA build is verified")
        r.results["llama_cpp"] = {"installed": False}
        return
    try:
        import llama_cpp
    except Exception as e:  # noqa: BLE001
        r.fail("llama_cpp", f"import llama_cpp failed: {e}")
        return
    gpu = bool(llama_cpp.llama_supports_gpu_offload())
    r.results["llama_cpp"] = {"installed": True, "version": llama_cpp.__version__, "gpu_offload": gpu}
    r.ok(f"llama_cpp {llama_cpp.__version__}, GPU offload: {'yes' if gpu else 'no'}")


def check_cuda_dlls(r: Report) -> None:
    r.heading("CUDA runtime DLLs loaded in this process (torch + llama.cpp)")
    if os.name != "nt":
        r.info("Windows-only check")
        return
    by_name: dict[str, set[str]] = {}
    for path in loaded_modules():
        name = Path(path).name
        if CUDA_DLL.match(name):
            by_name.setdefault(name.lower(), set()).add(os.path.normcase(path))
    listing = {name: sorted(f"{p} ({file_version(p)})" for p in where) for name, where in by_name.items()}
    r.results["cuda_dlls"] = listing
    if not listing:
        r.info("no CUDA runtime DLL loaded")
        return
    for name in sorted(listing):
        for entry in listing[name]:
            r.info(entry)
    conflicts = {n: w for n, w in by_name.items() if len(w) > 1}
    for name, where in conflicts.items():
        r.fail("cuda_dlls", f"{name} is loaded from {len(where)} places: {', '.join(sorted(where))}")
    families: dict[str, set[str]] = {}
    for name in by_name:
        family = name.split("64_")[0]
        families.setdefault(family, set()).add(name)
    for family, names in families.items():
        if len(names) > 1:
            r.info(f"{family}: {len(names)} different versions loaded side by side "
                   f"({', '.join(sorted(names))}); they have different names, so this is allowed")
    if not conflicts:
        r.ok("no library is loaded twice")


def other_onnxruntime_copies(r: Report) -> None:
    r.heading("Information")
    if os.name != "nt":
        return
    places = [Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"]
    places += [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    unique: dict[str, str] = {}
    for p in places:
        candidate = p / "onnxruntime.dll"
        if candidate.is_file():
            unique.setdefault(os.path.normcase(os.path.realpath(candidate)), str(candidate))
    found = sorted(unique.values())
    r.results["other_onnxruntime"] = found
    for path in found:
        r.info(f"another onnxruntime.dll on this machine: {path} ({file_version(path)})")
    running = []
    for service in SECURITY_SERVICES:
        done = subprocess.run(["sc.exe", "query", service], capture_output=True, text=True)
        if done.returncode == 0:
            running.append(service)
    folders = [
        str(Path(base) / name)
        for env in ("ProgramFiles", "ProgramFiles(x86)", "ProgramData")
        if (base := os.environ.get(env))
        for name in SECURITY_FOLDERS
        if (Path(base) / name).is_dir()
    ]
    r.results["security_software"] = {"services": running, "folders": folders}
    if running or folders:
        r.info(f"McAfee/Trellix present ({', '.join(running + folders)}). If DLLs go missing, ask "
               f"for an exclusion for {paths.app_root()}")
    else:
        r.info("no McAfee/Trellix found")


def main() -> int:
    r = Report()
    print(f"pyvolis doctor - Python {sys.version.split()[0]} at {sys.executable}")
    check_native_files(r)
    check_sherpa(r)  # first: sherpa-onnx must be the first to load onnxruntime.dll
    check_torch(r)
    check_llama(r)
    check_cuda_dlls(r)
    other_onnxruntime_copies(r)

    out = paths.logs_dir(paths.app_root()) / "doctor.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"failed": r.failed, **r.results}, indent=2), encoding="utf-8")
        print(f"\nwritten: {out}")
    except OSError as e:
        print(f"\ncould not write {out}: {e}")
    if r.failed:
        print(f"\n{len(r.failed)} check(s) failed.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
