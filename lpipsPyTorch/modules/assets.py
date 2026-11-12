"""Integrity-checked local model assets for scientific image evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import torch


DEFAULT_ASSET_DIR = Path("/opt/eligsir/eval-assets")
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "eval_assets.json"


class EvalAssetError(RuntimeError):
    """A required evaluation checkpoint is absent, corrupt, or malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalAssetError(f"cannot read evaluation asset manifest {path}: {exc}") from exc
    if payload.get("format_version") != 1 or not isinstance(payload.get("assets"), dict):
        raise EvalAssetError(f"unsupported evaluation asset manifest: {path}")
    return payload


def resolve_eval_asset(
    asset_id: str,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> Path:
    """Return a verified local checkpoint path; never access the network."""
    manifest_path = Path(manifest_path)
    assets = _manifest(manifest_path)["assets"]
    record = assets.get(asset_id)
    if not isinstance(record, dict):
        raise EvalAssetError(
            f"evaluation asset {asset_id!r} is not packaged by {manifest_path}"
        )

    asset_dir = Path(os.environ.get("ELIGSIR_EVAL_ASSET_DIR", DEFAULT_ASSET_DIR))
    path = asset_dir / str(record["filename"])
    if not path.is_file():
        raise EvalAssetError(
            f"required evaluation asset is missing: {path}. "
            "Materialize it explicitly with scripts/materialize_eval_assets.py "
            "into a host directory and mount that directory at runtime; "
            "runtime downloads are disabled."
        )

    expected_size = int(record["size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise EvalAssetError(
            f"evaluation asset size mismatch for {path}: "
            f"expected {expected_size}, got {actual_size}"
        )

    expected_sha = str(record["sha256"])
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise EvalAssetError(
            f"evaluation asset SHA-256 mismatch for {path}: "
            f"expected {expected_sha}, got {actual_sha}"
        )
    return path


def load_eval_state_dict(
    asset_id: str,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
):
    """Load one verified local PyTorch state dict without a URL fallback."""
    path = resolve_eval_asset(asset_id, manifest_path=manifest_path)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 compatibility for legacy dev environments
        state = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise EvalAssetError(f"cannot load evaluation asset {path}: {exc}") from exc
    if not isinstance(state, Mapping):
        raise EvalAssetError(f"evaluation asset is not a state dict: {path}")
    return state
