"""Network weights as safetensors files, read and written with numpy alone.

safetensors (https://github.com/huggingface/safetensors) is a plain format:
an 8-byte little-endian header length, a JSON header naming each tensor's
dtype, shape and byte range, then the raw little-endian data.  Any framework
reads it, so the weights no longer tie the networks to the one that trained
them.  The header's ``__metadata__`` table carries the architecture name
(``arch``) and where the weights came from (``source``).
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

__all__ = ["load_safetensors", "save_safetensors"]

#: safetensors element types and their numpy equivalents.
_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U64": np.uint64,
    "U32": np.uint32,
    "U16": np.uint16,
    "U8": np.uint8,
    "BOOL": np.bool_,
}
_DTYPE_NAMES = {np.dtype(t): name for name, t in _DTYPES.items()}


def save_safetensors(path, tensors: dict, metadata: dict | None = None) -> None:
    """Write numpy arrays to *path* in the safetensors format.

    The file is written beside *path*, synced and renamed into place, so
    neither another process -- several T1Prep runs converting the weights at
    first use -- nor a crash leaves a partial file behind.

    Args:
        path: Destination file.
        tensors: Arrays by name; written widest element type first, then by
            name, as the reference writer orders them, so every tensor stays
            aligned.
        metadata: Optional string-to-string table stored in the header.
    """
    path = Path(path)
    header, blobs, offset = {}, [], 0
    for name in sorted(
        tensors, key=lambda n: (-np.asarray(tensors[n]).dtype.itemsize, n)
    ):
        array = np.ascontiguousarray(tensors[name])
        dtype = array.dtype.newbyteorder("=")
        if dtype not in _DTYPE_NAMES:
            raise TypeError(f"{name}: dtype {array.dtype} has no safetensors type")
        data = array.astype(dtype.newbyteorder("<"), copy=False).tobytes()
        header[name] = {
            "dtype": _DTYPE_NAMES[dtype],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # 8-byte aligned data, as the reference writer
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            # mkstemp creates the file 0600; a Docker image converts as root
            # and runs as another user, who still has to read the weights
            if hasattr(os, "fchmod"):
                os.fchmod(f.fileno(), 0o644)
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)
            for data in blobs:
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_safetensors(path) -> tuple[dict, dict]:
    """Read a safetensors file.

    Returns:
        ``(tensors, metadata)``: native-endian numpy arrays by name, and the
        header's metadata table (empty when there is none).
    """
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(size))
        # A writable buffer, so torch.from_numpy does not warn about it
        data = bytearray(f.read())
    metadata = header.pop("__metadata__", None) or {}
    tensors = {}
    for name, info in header.items():
        begin, end = info["data_offsets"]
        dtype = np.dtype(_DTYPES[info["dtype"]]).newbyteorder("<")
        count = int(np.prod(info["shape"], dtype=np.int64))
        if end - begin != count * dtype.itemsize or end > len(data):
            raise ValueError(f"{path}: tensor {name} does not match its offsets")
        array = np.frombuffer(data, dtype, count, begin).reshape(info["shape"])
        tensors[name] = array.astype(dtype.newbyteorder("="), copy=False)
    return tensors, metadata
