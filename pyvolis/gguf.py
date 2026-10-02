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


def read_tensor_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    """Every tensor's dimensions, by name, as GGUF lists them (innermost
    first). For checking that an adapter fits its base model."""
    try:
        with open(path, "rb") as f:
            _metadata, tensors = _read(f, path, with_tensors=True)
            return tensors
    except OSError as e:
        raise GgufError(f"cannot read {path.absolute()}: {e}") from e
    except struct.error as e:
        raise GgufError(f"{path.absolute()} ends in the middle of its tensor list") from e


def _read(f: BinaryIO, path: Path, with_tensors: bool = False):
    if f.read(4) != MAGIC:
        raise GgufError(f"{path.absolute()} is not a GGUF file (no GGUF magic)")
    (version,) = struct.unpack("<I", f.read(4))
    if version not in (2, 3):
        raise GgufError(f"{path.absolute()} is GGUF version {version}; only 2 and 3 are known")
    tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = _string(f)
        (kind,) = struct.unpack("<I", f.read(4))
        metadata[key] = _value(f, kind, path)
    if not with_tensors:
        return metadata
    tensors: dict[str, tuple[int, ...]] = {}
    for _ in range(tensor_count):
        name = _string(f)
        (dims,) = struct.unpack("<I", f.read(4))
        tensors[name] = struct.unpack(f"<{dims}Q", f.read(8 * dims))
        f.seek(12, 1)  # its type and where its data starts
    return metadata, tensors


def _string(f: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def _skip_strings(f: BinaryIO, count: int) -> None:
    """Pass over `count` length-prefixed strings. Walked in memory, a block
    at a time: a vocabulary is hundreds of thousands of them, and a read and
    a seek for each made listing five translators take over a second."""
    base, block, at = f.tell(), b"", 0
    for _ in range(count):
        if at + 8 > len(block):
            base += at
            f.seek(base)
            block, at = f.read(1 << 22), 0
        (size,) = struct.unpack_from("<Q", block, at)  # struct.error if the file ends here
        at += 8 + size
    f.seek(base + at)


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
            _skip_strings(f, length)
        else:
            raise GgufError(f"{path.absolute()} has an array of unsupported type {element}")
        return ("array", element, length)
    raise GgufError(f"{path.absolute()} has a metadata value of unknown type {kind}")
