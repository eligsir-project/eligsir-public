"""Canonical, deterministic hashing of nested dict/dataclass content.

Used to compute ``source_artifact_hash``/``pose_hash``/``keyframe_hash``/etc.
The same logical content must always hash the same regardless of dict key
insertion order or incidental whitespace, so callers get stable hashes
across processes/runs (and across dataclasses.asdict() vs a hand-built dict
with the same keys).

For raw binary content (frozen RGB/depth artifacts, files) use the explicit
``hash_bytes``/``hash_file``/``hash_ndarray`` primitives directly, or route
them through ``content_hash`` -- it recognizes ``bytes`` and
``numpy.ndarray`` automatically and hashes their real byte content, never
a printable repr. Any other genuinely unsupported type raises ``TypeError``
rather than silently falling back to ``str(value)`` -- a reproducibility
hash must be byte-exact, and ``str()`` on an arbitrary object is neither
stable across versions nor guaranteed to reflect full content (e.g. numpy's
``repr`` truncates large arrays with ``...``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

_FILE_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def hash_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of raw ``bytes`` content."""
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file's content, streamed in chunks.

    Never loads the whole file into memory -- safe for large frozen RGB/depth
    artifacts.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_FILE_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_ndarray(arr: "np.ndarray") -> str:
    """Return a SHA-256 hex digest over a numpy array's exact content.

    Hashes dtype + shape + contiguous raw bytes, so two arrays are equal
    under this hash iff they have the same dtype, shape, and element
    values -- never the abbreviated/format-dependent ``repr()`` string.
    """
    contiguous = np.ascontiguousarray(arr)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(b"|shape=")
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(b"|data=")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _canonicalize(value: Any) -> Any:
    """Recursively convert ``value`` into a JSON-safe, order-stable form."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, set):
        return sorted(_canonicalize(v) for v in value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hash_bytes(value)}
    if isinstance(value, np.ndarray):
        return {"__ndarray_sha256__": hash_ndarray(value)}
    if isinstance(value, Path):
        # A filesystem path is an unambiguous, stable string identity (unlike
        # an arbitrary object's repr) -- safe to stringify directly.
        return str(value)
    raise TypeError(
        f"content_hash()/canonical_json(): unsupported type {type(value)!r} for value "
        f"{value!r}; add explicit handling (see hash_bytes/hash_file/hash_ndarray) instead "
        "of relying on str() -- a reproducibility hash must be byte-exact, not a printable repr."
    )


def canonical_json(content: Any) -> str:
    """Serialize ``content`` to a canonical JSON string.

    Sorted keys, no incidental whitespace, stable across dict key order and
    dataclass-vs-dict equivalents. Raises ``TypeError`` for unsupported leaf
    types -- see module docstring.
    """
    canonical = _canonicalize(content)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(content: Any) -> str:
    """Compute a stable SHA-256 hex digest of arbitrary nested content.

    ``bytes`` and ``numpy.ndarray`` values are routed through
    ``hash_bytes``/``hash_ndarray`` (their real byte content, not a repr).
    Any other unsupported leaf type raises ``TypeError`` rather than
    silently hashing ``str(value)``.
    """
    serialized = canonical_json(content)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
