#!/usr/bin/env python3
"""Explicitly download and verify external evaluation checkpoints.

This command is intentionally not invoked by any Dockerfile. The checkpoints
are not redistributed by EliGSiR because their upstream terms are unresolved;
users who are entitled to obtain them may run this command into a host
directory and mount that directory at ``ELIGSIR_EVAL_ASSET_DIR``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, record: dict) -> None:
    expected_size = int(record["size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"size mismatch for {path}: expected {expected_size}, got {actual_size}"
        )
    expected_sha = str(record["sha256"])
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected_sha}, got {actual_sha}"
        )


def materialize(manifest_path: Path, destination: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or not isinstance(manifest.get("assets"), dict):
        raise RuntimeError(f"unsupported evaluation asset manifest: {manifest_path}")
    destination.mkdir(parents=True, exist_ok=True)

    for asset_id, record in manifest["assets"].items():
        filename = str(record["filename"])
        if Path(filename).name != filename:
            raise RuntimeError(f"unsafe filename for {asset_id}: {filename}")
        target = destination / filename
        if target.is_file():
            try:
                _verify(target, record)
            except RuntimeError:
                pass
            else:
                print(f"verified existing evaluation asset {asset_id}: {target}")
                continue

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{filename}.", dir=destination, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                with urllib.request.urlopen(str(record["url"])) as response:
                    shutil.copyfileobj(response, temporary)
            _verify(temporary_path, record)
            os.replace(temporary_path, target)
            temporary_path = None
            print(f"installed verified evaluation asset {asset_id}: {target}")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download and verify external LPIPS/ImageNet assets into a user-"
            "owned directory; never run this implicitly at application startup."
        )
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="pinned asset manifest"
    )
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="external asset directory to populate and mount at runtime",
    )
    args = parser.parse_args()
    materialize(args.manifest, args.destination)


if __name__ == "__main__":
    main()
