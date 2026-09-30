"""Download a Hugging Face model into models/asr/ or models/mt/. Run through
fetch-model.ps1. A development tool: it uses the internet, and the app never
calls it.

The equivalent of `hf download <id> --local-dir models/<role>/<name>` with a
chosen file list, done through huggingface_hub's Python API so the file list
can be decided from the repo's contents first.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The token and download cache stay in the repo, never in the user profile,
# and never in the app's own cache/ (which is shipped).
os.environ["HF_HOME"] = str(REPO / ".uv" / "hf")
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from huggingface_hub import HfApi, login, snapshot_download  # noqa: E402
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError  # noqa: E402

# Never wanted: other frameworks' weights and exports.
ALWAYS_SKIP = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "*.onnx_data", "onnx/*", "flax_model*", "tf_model*",
               "*.mlmodel", "*.tflite", "coreml/*", "openvino/*",
               # training logs some repos publish
               "runs/*", "*.tfevents.*"]
DUPLICATE_WEIGHTS = ["*.bin", "*.pt", "*.pth", "*.ckpt"]


def choose(files: list[str], role: str, include: list[str]) -> list[str]:
    """The files to download from a repo's file list."""
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if ggufs:
        models = [f for f in ggufs if not os.path.basename(f).lower().startswith("mmproj")]
        if include:
            wanted = [f for f in ggufs if any(fnmatch.fnmatch(f, p) for p in include)]
        elif len(models) == 1:
            wanted = models
        else:
            names = "\n  ".join(models)
            raise SystemExit(
                f"STOP: this repo has {len(models)} .gguf files. Pick one with -Include, e.g. "
                f'-Include "*Q4_K_M.gguf":\n  {names}'
            )
        if role == "asr" and not any(os.path.basename(f).lower().startswith("mmproj") for f in wanted):
            mmproj = [f for f in ggufs if os.path.basename(f).lower().startswith("mmproj")]
            if mmproj:
                print(f"note: a speech model also needs its audio encoder; add -Include for one of: {mmproj}")
        if not wanted:
            raise SystemExit(f"STOP: nothing in the repo matches {include}")
        return wanted + [f for f in files if f in ("README.md", "LICENSE", "LICENSE.md")]

    skip = list(ALWAYS_SKIP)
    if any(f.endswith(".safetensors") for f in files):
        skip += DUPLICATE_WEIGHTS  # the same weights twice; take safetensors
    chosen = [f for f in files if not any(fnmatch.fnmatch(f, p) for p in skip)]
    if include:
        chosen = [f for f in chosen if any(fnmatch.fnmatch(f, p) for p in include)] + [
            f for f in chosen if f.endswith(".json") or f == "README.md"
        ]
    return sorted(set(chosen))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("id", nargs="?")
    parser.add_argument("--role", choices=["asr", "mt"])
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--name", help="folder name, if the model's own name is taken")
    parser.add_argument("--login", action="store_true")
    args = parser.parse_args()

    if args.login:
        login()
        print(f"token saved under {os.environ['HF_HOME']}")
        return 0

    target = REPO / "models" / args.role / (args.name or args.id.split("/")[-1])
    if (target / "engine.toml").is_file():
        # e.g. the Rust volis folder converted from this very model: same name.
        print(f"STOP: {target} is an engine.toml (sherpa-onnx) model folder. Give the download "
              "its own folder name with -Name.", file=sys.stderr)
        return 1
    api = HfApi()
    try:
        info = api.model_info(args.id)
    except GatedRepoError:
        print(f"STOP: {args.id} is gated. Accept its terms at https://huggingface.co/{args.id} "
              "in your browser, then run: .\\fetch-model.ps1 -Login", file=sys.stderr)
        return 1
    except RepositoryNotFoundError:
        print(f"STOP: no model {args.id} on Hugging Face (or it is private: run .\\fetch-model.ps1 -Login)",
              file=sys.stderr)
        return 1
    files = [s.rfilename for s in info.siblings or []]
    wanted = choose(files, args.role, args.include)
    print(f"{args.id} -> {target}")
    for f in wanted:
        print(f"  {f}")
    try:
        snapshot_download(args.id, local_dir=target, allow_patterns=wanted)
    except GatedRepoError:
        print(f"STOP: {args.id} is gated. Accept its terms at https://huggingface.co/{args.id} "
              "in your browser, then run: .\\fetch-model.ps1 -Login", file=sys.stderr)
        return 1
    print(f"\ndone. Check it with: .\\.venv\\Scripts\\python.exe -m pyvolis --report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
