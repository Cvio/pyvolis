"""Read a GGUF file's metadata without loading the model.

Discovery needs a few facts about each `.gguf` (its architecture, its name,
whether it carries a chat template, whether it is a LoRA adapter or an audio
encoder) and must not pay for loading weights to learn them. The format is
llama.cpp's `docs/gguf.md`: a header, then typed key/value pairs, then the
tensor index. Only the key/value part is read.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GGUF"

# Value types, from gguf.md.
_FIXED = {
    0: "<B",  # uint8
    1: "<b",  # int8
    2: "<H",  # uint16
    3: "<h",  # int16
    4: "<I",  # uint32
    5: "<i",  # int32
    6: "<f",  # float32
    7: "<?",  # bool
    10: "<Q",  # uint64
    11: "<q",  # int64
    12: "<d",  # float64
}
_STRING = 8
_ARRAY = 9


class GgufError(Exception):
    pass


def read_metadata(path: Path) -> dict[str, Any]:
    """Every metadata key of `path`. Arrays are summarised as
    `("array", element_type, length)` rather than read, because a vocabulary is
    hundreds of thousands of strings and nothing in discovery needs them."""
    try:
        with open(path, "rb") as f:
            return _read(f, path)
    except OSError as e:
        raise GgufError(f"cannot read {path.absolute()}: {e}") from e
    except struct.error as e:
        raise GgufError(f"{path.absolute()} ends in the middle of its metadata") from e


def _read(f: BinaryIO, path: Path) -> dict[str, Any]:
    if f.read(4) != MAGIC:
        raise GgufError(f"{path.absolute()} is not a GGUF file (no GGUF magic)")
    (version,) = struct.unpack("<I", f.read(4))
    if version not in (2, 3):
        raise GgufError(f"{path.absolute()} is GGUF version {version}; only 2 and 3 are known")
    _tensors, kv_count = struct.unpack("<QQ", f.read(16))
    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = _string(f)
        (kind,) = struct.unpack("<I", f.read(4))
        metadata[key] = _value(f, kind, path)
    return metadata


def _string(f: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def _value(f: BinaryIO, kind: int, path: Path) -> Any:
    if kind in _FIXED:
        fmt = _FIXED[kind]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if kind == _STRING:
        return _string(f)
    if kind == _ARRAY:
        element, length = struct.unpack("<IQ", f.read(12))
        if element in _FIXED:
            f.seek(struct.calcsize(_FIXED[element]) * length, 1)
        elif element == _STRING:
            for _ in range(length):
                (size,) = struct.unpack("<Q", f.read(8))
                f.seek(size, 1)
        else:
            raise GgufError(f"{path.absolute()} has an array of unsupported type {element}")
        return ("array", element, length)
    raise GgufError(f"{path.absolute()} has a metadata value of unknown type {kind}")
