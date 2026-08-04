#!/usr/bin/env python3
"""Read selected scalar metadata keys from a GGUF file without extra packages."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import BinaryIO, Any


UINT8 = 0
INT8 = 1
UINT16 = 2
INT16 = 3
UINT32 = 4
INT32 = 5
FLOAT32 = 6
BOOL = 7
STRING = 8
ARRAY = 9
UINT64 = 10
INT64 = 11
FLOAT64 = 12

SCALARS: dict[int, tuple[str, int]] = {
    UINT8: ("<B", 1),
    INT8: ("<b", 1),
    UINT16: ("<H", 2),
    INT16: ("<h", 2),
    UINT32: ("<I", 4),
    INT32: ("<i", 4),
    FLOAT32: ("<f", 4),
    BOOL: ("<?", 1),
    UINT64: ("<Q", 8),
    INT64: ("<q", 8),
    FLOAT64: ("<d", 8),
}


def read_exact(source: BinaryIO, count: int) -> bytes:
    data = source.read(count)
    if len(data) != count:
        raise EOFError(f"Expected {count} bytes, received {len(data)}")
    return data


def read_u32(source: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(source, 4))[0]


def read_u64(source: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(source, 8))[0]


def read_string(source: BinaryIO, capture: bool) -> str | None:
    length = read_u64(source)
    if capture:
        return read_exact(source, length).decode("utf-8")
    source.seek(length, 1)
    return None


def read_value(source: BinaryIO, value_type: int, capture: bool) -> Any:
    if value_type in SCALARS:
        code, width = SCALARS[value_type]
        raw = read_exact(source, width)
        return struct.unpack(code, raw)[0] if capture else None
    if value_type == STRING:
        return read_string(source, capture)
    if value_type == ARRAY:
        item_type = read_u32(source)
        count = read_u64(source)
        if capture:
            return [read_value(source, item_type, True) for _ in range(count)]
        if item_type in SCALARS:
            source.seek(SCALARS[item_type][1] * count, 1)
        else:
            for _ in range(count):
                read_value(source, item_type, False)
        return None
    raise ValueError(f"Unsupported GGUF metadata value type: {value_type}")


def inspect(path: Path, wanted: set[str]) -> dict[str, Any]:
    with path.open("rb") as source:
        magic = read_exact(source, 4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")
        version = read_u32(source)
        tensor_count = read_u64(source)
        metadata_count = read_u64(source)
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = read_string(source, True)
            assert key is not None
            value_type = read_u32(source)
            capture = key in wanted
            value = read_value(source, value_type, capture)
            if capture:
                metadata[key] = value
    return {
        "path": str(path.resolve()),
        "gguf_version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "metadata": metadata,
        "missing_keys": sorted(wanted - metadata.keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument(
        "--key",
        action="append",
        default=[],
        help="Metadata key to read; repeat as needed.",
    )
    args = parser.parse_args()
    wanted = set(args.key or ["interleaved_n_text", "interleaved_n_audio"])
    print(json.dumps(inspect(args.gguf, wanted), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
