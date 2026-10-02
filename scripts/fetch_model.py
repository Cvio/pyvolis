"""Download a Hugging Face model into models/asr/, models/mt/ or models/tts/. Run through
fetch-model.ps1. A development tool: it uses the internet, and the app never
calls it.

The equivalent of `hf download <id> --local-dir models/<role>/<name>` with a
chosen file list, done through huggingface_hub's Python API so the file list
can be decided from the repo's contents first.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
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
    if role == "tts":  # a voice: everything it was published with (the .onnx is the model)
        return [f for f in files if f != ".gitattributes"]
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


def piper_engine(name: str, language: str, model: str, variety: str = "", note: str = "") -> str:
    """The engine.toml of a Piper voice, as Rust volis's templates have it."""
    variety_line = f'varieties = ["{variety}"]   # the dialect this voice speaks\n' if variety else ""
    return (f'name = "{name}"\nkind = "segment"\nbackend = "vits"\nlanguages = ["{language}"]\n{variety_line}\n'
            "# espeak-ng does the pronunciation for Piper voices. It is a directory, so it\n"
            '# cannot be declared under [files].\ndata_dir = "espeak-ng-data"\n\n[files]\n'
            f'{note}model  = "{model}"\ntokens = "tokens.txt"\n')


RAW_PIPER = ("# The download is a raw Piper model. sherpa-onnx needs a few fields inside the model file,\n"
             "# so this is a copy with them added; the .onnx beside it is the original, untouched.\n")


def convert_piper(folder: Path, language: str = "") -> None:
    """A voice as Piper publishes it (X.onnx + X.onnx.json) lacks three things
    sherpa-onnx needs: tokens.txt, a few fields inside the model file, and
    espeak-ng-data. Writes tokens.txt and X.sherpa.onnx (the original is left
    untouched), and copies espeak-ng-data from another installed voice."""
    for config_path in folder.glob("*.onnx.json"):
        model = config_path.with_suffix("")  # X.onnx
        out = model.with_name(model.stem + ".sherpa.onnx")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not (folder / "tokens.txt").exists():
            with open(folder / "tokens.txt", "w", encoding="utf-8", newline="\n") as f:
                for symbol, ids in config["phoneme_id_map"].items():
                    f.write(f"{symbol} {ids[0]}\n")
        if not out.exists():
            # ONNX is protobuf: ModelProto.metadata_props is repeated field 14,
            # and appending entries to the file adds them to the list.
            meta = {"model_type": "vits", "comment": "piper",
                    # sherpa-onnx refuses a model with no language named; any name will do
                    "language": language or (config.get("language") or {}).get("name_english")
                    or config["espeak"]["voice"],
                    "voice": config["espeak"]["voice"], "has_espeak": 1, "n_speakers": config["num_speakers"],
                    "sample_rate": config["audio"]["sample_rate"]}
            out.write_bytes(model.read_bytes() + b"".join(_metadata_entry(k, v) for k, v in meta.items()))
            print(f"    wrote {out} and tokens.txt")
    if not (folder / "espeak-ng-data").is_dir():
        donors = [p for p in folder.parent.glob("*/espeak-ng-data") if (p / "phontab").is_file()]
        if not donors:
            raise SystemExit(f"STOP: {folder} needs espeak-ng-data, and no other voice in {folder.parent} has it to "
                             "copy. Fetch a sherpa-onnx Piper voice first (.\\fetch-models.ps1 -Only lessac).")
        shutil.copytree(donors[0], folder / "espeak-ng-data")
        print(f"    copied espeak-ng-data from {donors[0].parent.name}")


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _metadata_entry(key: str, value) -> bytes:
    k, v = key.encode(), str(value).encode()
    body = b"\x0a" + _varint(len(k)) + k + b"\x12" + _varint(len(v)) + v
    return b"\x72" + _varint(len(body)) + body


def finish_voice(folder: Path) -> None:
    """After a voice is downloaded: convert it if it is a raw Piper model, and
    write an engine.toml if it has none. The language is taken from the
    voice's own .onnx.json; no variety is declared."""
    raw = not (folder / "tokens.txt").is_file()
    if raw:
        convert_piper(folder)
    if (folder / "engine.toml").exists():
        return
    configs = sorted(folder.glob("*.onnx.json"))
    models = sorted(folder.glob("*.sherpa.onnx")) or sorted(folder.glob("*.onnx"))
    if not models:
        raise SystemExit(f"STOP: {folder} has no .onnx file, so it is not a Piper voice pyvolis can load")
    language = ""
    if configs:
        config = json.loads(configs[0].read_text(encoding="utf-8"))
        language = (config.get("language") or {}).get("family") or str(config.get("espeak", {}).get("voice", ""))[:2]
    (folder / "engine.toml").write_text(
        piper_engine(folder.name, language, models[0].name, note=RAW_PIPER if raw else ""), encoding="utf-8",
        newline="\n")
    print(f"wrote {folder / 'engine.toml'}: check its name and languages"
          + ("" if language else ' (languages is empty: set it, e.g. ["ar"])'))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("id", nargs="?")
    parser.add_argument("--role", choices=["asr", "mt", "tts"])
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--name", help="folder name, if the model's own name is taken")
    parser.add_argument("--login", action="store_true")
    args = parser.parse_args()

    if args.login:
        login()
        print(f"token saved under {os.environ['HF_HOME']}")
        return 0

    target = REPO / "models" / args.role / (args.name or args.id.split("/")[-1])
    if args.role != "tts" and (target / "engine.toml").is_file():
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
    if args.role == "tts":
        finish_voice(target)
    print(f"\ndone. Check it with: .\\.venv\\Scripts\\python.exe -m pyvolis --report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
