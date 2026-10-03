"""Put what the built program reads beside it: run by build.ps1 after
PyInstaller. dist\\pyvolis\\ then holds pyvolis.exe, _internal\\, and the same
config\\, prompts\\ and models\\ the repository has, where paths.app_root()
looks for them.

Models are hard-linked when dist\\ is on the same drive (the same bytes, no
second copy; a zip of the folder still holds real files) and copied when it
isn't. Hugging Face's download bookkeeping (.cache\\ in a model folder) is
left out.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def place(source: Path, target: Path) -> str:
    """Hard-link, or copy where a link can't be made. Returns which."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
        return "linked"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


def tree(source: Path, target: Path, skip=lambda p: False) -> dict[str, int]:
    counts = {"linked": 0, "copied": 0, "bytes": 0}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_dir() or skip(relative):
            continue
        counts[place(path, target / relative)] += 1
        counts["bytes"] += path.stat().st_size
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    parser.add_argument("--no-models", action="store_true", help="only the README files in models\\")
    args = parser.parse_args()
    dist = args.dist.resolve()
    if not (dist / "pyvolis.exe").is_file():
        print(f"STOP: no {dist / 'pyvolis.exe'}; PyInstaller has not built it", file=sys.stderr)
        return 1

    for folder in ("config", "prompts"):
        shutil.rmtree(dist / folder, ignore_errors=True)
        shutil.copytree(REPO / folder, dist / folder)
        print(f"  {folder}\\")
    # The settings as this machine has them, so the copy starts the same way.
    for name in ("volis.toml", "pyvolis.toml"):
        if (REPO / name).is_file():
            shutil.copy2(REPO / name, dist / name)
            print(f"  {name}")

    models = REPO / "models"
    shutil.rmtree(dist / "models", ignore_errors=True)
    if args.no_models:
        counts = tree(models, dist / "models", skip=lambda p: p.name != "README.txt")
    else:
        # .cache\: Hugging Face's download bookkeeping, never read by pyvolis.
        counts = tree(models, dist / "models", skip=lambda p: ".cache" in p.parts)
    print(f"  models\\: {counts['linked'] + counts['copied']} files, {counts['bytes'] / 1e9:.1f} GB "
          f"({counts['linked']} hard-linked, {counts['copied']} copied)")
    for name in ("README.md", "MODELS.md"):
        shutil.copy2(REPO / name, dist / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
