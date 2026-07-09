#!/usr/bin/env python3
"""Convert fvecs vectors to the streaming-build fbin matrix format.

Input fvecs format repeats a little-endian int32 dimension before every vector:
``[dim][float32 * dim]``.  The streaming SCORE builders consume fbin:
``[n:int32][dim:int32][reserved:int32=0][float32 * n * dim]``.
This converter streams fixed-size row chunks and validates every per-row fvecs
dimension header before writing the corresponding float payload.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


HEADER_STRUCT = struct.Struct("<iii")
DIM_STRUCT = struct.Struct("<i")


def infer_shape(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as handle:
        first_dim = handle.read(DIM_STRUCT.size)
    if len(first_dim) != DIM_STRUCT.size:
        raise ValueError(f"empty or truncated fvecs file: {path}")
    (dim,) = DIM_STRUCT.unpack(first_dim)
    if dim <= 0:
        raise ValueError(f"invalid fvecs dimension {dim}: {path}")
    record_bytes = DIM_STRUCT.size + dim * 4
    file_bytes = path.stat().st_size
    if file_bytes % record_bytes != 0:
        raise ValueError(
            f"file size {file_bytes} is not a multiple of inferred record size {record_bytes}: {path}"
        )
    rows = file_bytes // record_bytes
    if rows <= 0:
        raise ValueError(f"no vectors found in fvecs file: {path}")
    if rows > 2**31 - 1 or dim > 2**31 - 1:
        raise ValueError(f"fbin header uses int32; rows={rows}, dim={dim} exceed limit")
    return rows, dim, record_bytes


def convert_fvecs_to_fbin(src: Path, dst: Path, *, rows_per_chunk: int, overwrite: bool) -> None:
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output without --overwrite: {dst}")
    rows, dim, record_bytes = infer_shape(src)
    payload_bytes = dim * 4
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("rb") as source, dst.open("wb") as output:
        output.write(HEADER_STRUCT.pack(rows, dim, 0))
        remaining = rows
        processed = 0
        while remaining:
            chunk_rows = min(rows_per_chunk, remaining)
            chunk = source.read(record_bytes * chunk_rows)
            expected_bytes = record_bytes * chunk_rows
            if len(chunk) != expected_bytes:
                raise ValueError(
                    f"truncated fvecs chunk at row {processed}: expected {expected_bytes}, got {len(chunk)}"
                )
            for row in range(chunk_rows):
                offset = row * record_bytes
                (row_dim,) = DIM_STRUCT.unpack_from(chunk, offset)
                if row_dim != dim:
                    raise ValueError(f"dimension mismatch at row {processed + row}: expected {dim}, got {row_dim}")
                start = offset + DIM_STRUCT.size
                output.write(chunk[start : start + payload_bytes])
            processed += chunk_rows
            remaining -= chunk_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="input .fvecs file")
    parser.add_argument("dst", type=Path, help="output .fbin file")
    parser.add_argument("--rows-per-chunk", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    convert_fvecs_to_fbin(args.src, args.dst, rows_per_chunk=args.rows_per_chunk, overwrite=args.overwrite)
    rows, dim, _ = infer_shape(args.src)
    print(f"converted rows={rows} dim={dim} src={args.src} dst={args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
