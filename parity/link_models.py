"""Mirror a models tree with hard links: no copies, and nothing in the source
changes.

    python parity/link_models.py <from models dir> <to models dir> [subpath ...]

Used to give pyvolis's development `models\\` the same files Rust volis uses,
and to give the parity copy of volis.exe the same `models\\` as pyvolis.
Hard links rather than junctions: deleting a hard link, even with a recursive
delete, never touches the other copy. Both folders must be on one drive.
Files that already exist are left alone.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def mirror(source: Path, target: Path) -> int:
    linked = 0
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(source, target)
            linked += 1
        return linked
    for dirpath, _dirnames, filenames in os.walk(source):
        here = Path(dirpath)
        there = target / here.relative_to(source)
        there.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            if not (there / name).exists():
                os.link(here / name, there / name)
                linked += 1
    return linked


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    source, target = Path(argv[0]).resolve(), Path(argv[1]).resolve()
    subpaths = argv[2:] or ["."]
    total = 0
    for sub in subpaths:
        s = source / sub
        if not s.exists():
            print(f"STOP: {s} does not exist", file=sys.stderr)
            return 1
        total += mirror(s, target / sub)
    print(f"linked {total} files from {source} into {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
