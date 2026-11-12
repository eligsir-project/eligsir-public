"""Offline, integrity-checked LPIPS/VGG evaluation asset contract."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from lpipsPyTorch import lpips as lpips_metric
from lpipsPyTorch.modules.assets import EvalAssetError, load_eval_state_dict
from lpipsPyTorch.modules.networks import BACKBONE_ASSET_IDS


ROOT = Path(__file__).resolve().parents[1]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(path: Path, *, source: Path, payload: bytes) -> None:
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "assets": {
                    "fixture": {
                        "filename": "fixture.pth",
                        "url": source.as_uri(),
                        "sha256": _sha256(payload),
                        "size_bytes": len(payload),
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_every_public_lpips_mode_has_pinned_external_assets():
    """Advertised LPIPS modes retain exact external asset provenance."""
    manifest = json.loads(
        (ROOT / "lpipsPyTorch" / "eval_assets.json").read_text(encoding="utf-8")
    )
    assets = manifest["assets"]

    assert set(BACKBONE_ASSET_IDS) == {"alex", "squeeze", "vgg"}
    default_mode = inspect.signature(lpips_metric).parameters["net_type"].default
    assert default_mode in BACKBONE_ASSET_IDS
    for mode, backbone_asset_id in BACKBONE_ASSET_IDS.items():
        for asset_id in (backbone_asset_id, f"lpips_v0.1_{mode}"):
            record = assets[asset_id]
            assert record["size_bytes"] > 0
            assert len(record["sha256"]) == 64


@pytest.mark.parametrize(
    "dockerfile_name", ["Dockerfile", "Dockerfile.cuda", "Dockerfile.cuda-noble"]
)
def test_scientific_runtime_images_do_not_bundle_or_download_eval_assets(dockerfile_name):
    """Release images require an explicit external asset mount."""
    dockerfile = (ROOT / dockerfile_name).read_text(encoding="utf-8")

    assert "ELIGSIR_EVAL_ASSET_DIR=/opt/eligsir/eval-assets" in dockerfile
    assert "COPY scripts/materialize_eval_assets.py" not in dockerfile
    assert "python /tmp/materialize_eval_assets.py" not in dockerfile
    assert "eval-assets" in dockerfile
    assert "Download LPIPS" not in dockerfile


def test_external_asset_materializer_is_explicit_and_never_a_runtime_import():
    """The downloader remains a user-run command, not application startup code."""
    materializer = (ROOT / "scripts" / "materialize_eval_assets.py").read_text(
        encoding="utf-8"
    )
    assert "--manifest" in materializer
    assert "--destination" in materializer
    assert "torch.hub" not in materializer


def test_materializer_rejects_corrupt_download_before_installing_asset(tmp_path):
    """A source whose bytes do not match the manifest must never be installed."""
    expected = b"expected checkpoint bytes"
    source = tmp_path / "source.pth"
    source.write_bytes(b"tampered checkpoint bytes")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, source=source, payload=expected)
    destination = tmp_path / "assets"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "materialize_eval_assets.py"),
            "--manifest",
            str(manifest),
            "--destination",
            str(destination),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr
    assert not (destination / "fixture.pth").exists()


def test_materializer_installs_exact_asset_and_is_offline_idempotent(tmp_path):
    """Once verified and installed, a rebuild must not need the source URL."""
    payload = b"exact checkpoint bytes"
    source = tmp_path / "source.pth"
    source.write_bytes(payload)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, source=source, payload=payload)
    destination = tmp_path / "assets"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "materialize_eval_assets.py"),
        "--manifest",
        str(manifest),
        "--destination",
        str(destination),
    ]

    subprocess.run(command, check=True, text=True, capture_output=True)
    source.unlink()
    subprocess.run(command, check=True, text=True, capture_output=True)

    assert (destination / "fixture.pth").read_bytes() == payload


def test_runtime_loader_reads_verified_local_state_without_torch_hub(
    tmp_path, monkeypatch
):
    """Metric initialization must load the baked file and never call torch.hub."""
    checkpoint = tmp_path / "fixture.pth"
    torch.save({"weight": torch.tensor([1.25])}, checkpoint)
    payload = checkpoint.read_bytes()
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, source=checkpoint, payload=payload)
    monkeypatch.setenv("ELIGSIR_EVAL_ASSET_DIR", str(tmp_path))

    def forbid_download(*_args, **_kwargs):
        raise AssertionError("runtime attempted a network-backed torch.hub load")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbid_download)

    state = load_eval_state_dict("fixture", manifest_path=manifest)

    assert torch.equal(state["weight"], torch.tensor([1.25]))


def test_runtime_loader_fails_closed_when_baked_asset_is_missing(tmp_path, monkeypatch):
    """A clean/mispackaged image must fail rather than populate a cache online."""
    source = tmp_path / "not-installed.pth"
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, source=source, payload=b"expected")
    asset_dir = tmp_path / "empty-assets"
    asset_dir.mkdir()
    monkeypatch.setenv("ELIGSIR_EVAL_ASSET_DIR", str(asset_dir))

    with pytest.raises(EvalAssetError, match="required evaluation asset is missing"):
        load_eval_state_dict("fixture", manifest_path=manifest)

    assert list(asset_dir.iterdir()) == []


def test_runtime_loader_fails_closed_when_baked_asset_is_corrupt(tmp_path, monkeypatch):
    """An altered image-layer checkpoint must not contribute an LPIPS score."""
    checkpoint = tmp_path / "fixture.pth"
    checkpoint.write_bytes(b"tampered")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, source=checkpoint, payload=b"expected")
    monkeypatch.setenv("ELIGSIR_EVAL_ASSET_DIR", str(tmp_path))

    with pytest.raises(EvalAssetError, match="SHA-256 mismatch"):
        load_eval_state_dict("fixture", manifest_path=manifest)
