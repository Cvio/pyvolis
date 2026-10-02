"""Write a small synthetic LoRA adapter, as a GGUF, for a GGUF translator.

    .venv\\Scripts\\python.exe scripts\\make_test_lora.py <base.gguf> <out.gguf> [--strength 0.05]

For testing that pyvolis loads an adapter onto its base, without training
one: the adapter changes the first block's attention query weights by a
random low-rank amount. With --strength 0 it changes nothing, so the output
must equal the base model's; with a strength above 0 it must differ. A real
adapter comes from llama.cpp's convert_lora_to_gguf.py and has the same shape:
general.type = "adapter", adapter.type = "lora", and a `.lora_a` / `.lora_b`
pair for each tensor it changes.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pyvolis import gguf  # noqa: E402

ALIGN = 32
UINT32, FLOAT32, STRING = 4, 6, 8


def _string(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def write_lora(path: Path, architecture: str, tensors: dict[str, np.ndarray], alpha: float, name: str) -> None:
    metadata = [("general.type", STRING, "adapter"), ("general.architecture", STRING, architecture),
                ("general.name", STRING, name), ("adapter.type", STRING, "lora"),
                ("adapter.lora.alpha", FLOAT32, alpha), ("general.alignment", UINT32, ALIGN)]
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, kind, value in metadata:
        out += _string(key) + struct.pack("<I", kind)
        out += _string(value) if kind == STRING else struct.pack("<f" if kind == FLOAT32 else "<I", value)
    offset = 0
    blobs = []
    for tensor_name, array in tensors.items():
        data = np.ascontiguousarray(array, dtype="<f4").tobytes()
        dims = array.shape[::-1]  # GGUF lists dimensions innermost first
        out += _string(tensor_name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        out += struct.pack("<IQ", 0, offset)  # 0 = F32
        blobs.append(data)
        offset += len(data) + (-len(data) % ALIGN)
    out += b"\0" * (-len(out) % ALIGN)
    for data in blobs:
        out += data + b"\0" * (-len(data) % ALIGN)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--strength", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=4)
    args = parser.parse_args()
    metadata = gguf.read_metadata(args.base)
    target = "blk.0.attn_q.weight"
    shapes = gguf.read_tensor_shapes(args.base)
    if target not in shapes:
        raise SystemExit(f"STOP: {args.base} has no tensor named {target}")
    columns, rows = shapes[target][:2]  # innermost first: inputs, then outputs
    rng = np.random.default_rng(7)
    a = rng.standard_normal((args.rank, columns)).astype(np.float32)
    b = (rng.standard_normal((rows, args.rank)) * args.strength).astype(np.float32)
    write_lora(args.out, metadata["general.architecture"], {f"{target}.lora_a": a, f"{target}.lora_b": b},
               alpha=float(args.rank), name=f"test adapter (strength {args.strength:g})")
    print(f"{args.out}: rank {args.rank} on {target} ({rows} x {columns}), strength {args.strength:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
