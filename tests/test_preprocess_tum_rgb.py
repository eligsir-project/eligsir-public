"""Real CPU CLI contracts for split-safe, native-grid RGB conditioning."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/preprocess_tum_rgb.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def scene(tmp_path):
    src = tmp_path / "scene"
    (src / "rgb").mkdir(parents=True)
    (src / "depth").mkdir()
    rng = np.random.default_rng(47)
    image = np.clip(120 + rng.normal(0, 5, (32, 40, 3)), 0, 255).astype(np.uint8)
    image[:, 20:] += 35
    for name in ("train", "heldout", "guard", "unassociated"):
        assert cv2.imwrite(str(src / "rgb" / f"{name}.png"), image,
                           [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert cv2.imwrite(str(src / "depth/0.png"), np.full((32, 40), 1234, np.uint16))
    (src / "rgb.txt").write_text("# RGB\n0 rgb/train.png\n1 rgb/heldout.png\n2 rgb/guard.png\n")
    for name in ("depth.txt", "groundtruth.txt", "accelerometer.txt", "K.txt", "associations.txt"):
        (src / name).write_bytes(b"original calibration / timestamps / poses\n")
    manifest = {
        "schema": "eligsir.train-rgb-allowlist/v1",
        "source": {"scene_id": "tiny-fr3", "rgb_txt_sha256": sha(src / "rgb.txt")},
        "split": {"id": "production-split", "sha256": "a" * 64},
        "train_rgb_paths": ["rgb/train.png"],
    }
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps(manifest))
    return src, path, manifest


def run(src, allowlist, dst, operator="identity", *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--dst", str(dst),
                           "--allowlist", str(allowlist), "--operator", operator, *extra],
                          capture_output=True, text=True)


@pytest.mark.parametrize("operator", ["identity", "denoise-only", "sharpen-only"])
def test_native_operators_preserve_protected_bytes_and_bind_complete_receipt(scene, tmp_path, operator):
    src, allowlist, manifest = scene
    before = {p.relative_to(src).as_posix(): p.read_bytes() for p in src.rglob("*") if p.is_file()}
    dst = tmp_path / operator
    result = run(src, allowlist, dst, operator, "--jobs", "4")
    assert result.returncode == 0, result.stderr
    receipt = json.loads((dst / "preprocessing_manifest.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["allowlist_sha256"] == sha(allowlist)
    assert receipt["source"] == manifest["source"] and receipt["split"] == manifest["split"]
    assert receipt["elapsed_s"] >= 0
    assert receipt["opencv_version"] == cv2.__version__
    rows = {row["path"]: row for row in receipt["images"]}
    assert set(rows) == {f"rgb/{name}.png" for name in ("train", "heldout", "guard", "unassociated")}
    for rel, content in before.items():
        assert (src / rel).read_bytes() == content
        if rel != "rgb/train.png" or operator == "identity":
            assert (dst / rel).read_bytes() == content
        if rel in rows:
            row = rows[rel]
            transformed = rel == "rgb/train.png" and operator != "identity"
            assert row["transformed"] is transformed
            assert row["operator"] == (operator if transformed else "identity")
            assert row["source_sha256"] == sha(src / rel)
            assert row["output_sha256"] == sha(dst / rel)
    original = cv2.imread(str(src / "rgb/train.png"))
    actual = cv2.imread(str(dst / "rgb/train.png"))
    assert actual.shape == original.shape == (32, 40, 3)
    if operator == "denoise-only":
        expected = cv2.fastNlMeansDenoisingColored(original, None, 6, 6, 7, 21)
    elif operator == "sharpen-only":
        expected = cv2.addWeighted(original, 1.15, cv2.GaussianBlur(original, (0, 0), 1.2), -0.15, 0)
    else:
        expected = original
    np.testing.assert_array_equal(actual, expected)
    if operator != "identity":
        assert not np.array_equal(actual, original)
    repeat = tmp_path / f"{operator}-repeat"
    assert run(src, allowlist, repeat, operator, "--jobs", "1").returncode == 0
    assert sha(dst / "rgb/train.png") == sha(repeat / "rgb/train.png")


@pytest.mark.parametrize("paths", [["../outside.png"], ["/tmp/outside.png"],
    ["rgb/../rgb/train.png"], ["rgb/train.png", "rgb/train.png"],
    ["rgb/missing.png"], ["rgb/unassociated.png"], ["depth/0.png"], "rgb/train.png"])
def test_invalid_allowlist_never_creates_destination(scene, tmp_path, paths):
    src, allowlist, manifest = scene
    manifest["train_rgb_paths"] = paths
    allowlist.write_text(json.dumps(manifest))
    dst = tmp_path / "invalid"
    assert run(src, allowlist, dst).returncode != 0
    assert not dst.exists()


@pytest.mark.parametrize("field", ["schema", "source", "split"])
def test_invalid_provenance_is_rejected_before_copy(scene, tmp_path, field):
    src, allowlist, manifest = scene
    manifest[field] = "invalid"
    allowlist.write_text(json.dumps(manifest))
    dst = tmp_path / "invalid"
    assert run(src, allowlist, dst).returncode != 0
    assert not dst.exists()


def test_changed_source_index_is_not_accepted(scene, tmp_path):
    src, allowlist, _ = scene
    (src / "rgb.txt").write_text("0 rgb/train.png\n")
    dst = tmp_path / "invalid"
    assert run(src, allowlist, dst).returncode != 0
    assert not dst.exists()


@pytest.mark.parametrize("extra", [("--force",), ("--keep_upsampled",), ("--scale", "2"),
                                  ("--limit", "1"), ("--jobs", "5"), ("--jobs", "0")])
def test_unsafe_options_fail_closed(scene, tmp_path, extra):
    src, allowlist, _ = scene
    dst = tmp_path / "invalid"
    assert run(src, allowlist, dst, "identity", *extra).returncode != 0
    assert not dst.exists()


def test_existing_nested_and_symlink_destinations_are_rejected(scene, tmp_path):
    src, allowlist, _ = scene
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "sentinel").write_bytes(b"keep")
    link = tmp_path / "link"
    link.symlink_to(existing, target_is_directory=True)
    for dst in (src, src / "nested", existing, link):
        assert run(src, allowlist, dst).returncode != 0
    assert (existing / "sentinel").read_bytes() == b"keep"
    assert not (src / "nested").exists()


def test_outside_source_symlink_is_rejected_even_when_protected(scene, tmp_path):
    src, allowlist, _ = scene
    outside = tmp_path / "outside.png"
    outside.write_bytes((src / "rgb/heldout.png").read_bytes())
    (src / "rgb/heldout.png").unlink()
    (src / "rgb/heldout.png").symlink_to(outside)
    dst = tmp_path / "invalid"
    assert run(src, allowlist, dst).returncode != 0
    assert not dst.exists()


def test_failed_decode_has_no_success_manifest(scene, tmp_path):
    src, allowlist, _ = scene
    (src / "rgb/train.png").write_bytes(b"not a PNG")
    dst = tmp_path / "failed"
    assert run(src, allowlist, dst, "denoise-only").returncode != 0
    assert not (dst / "preprocessing_manifest.json").exists()


@pytest.mark.parametrize("name", ["preprocessing_manifest.json", "preprocessing_manifest.json.tmp"])
def test_source_receipt_names_cannot_leak_into_or_be_overwritten_in_output(scene, tmp_path, name):
    src, allowlist, _ = scene
    (src / name).write_text('{"status":"complete"}')
    dst = tmp_path / "invalid"
    assert run(src, allowlist, dst).returncode != 0
    assert not dst.exists()


@pytest.mark.parametrize("entry", ["rgb/missing.png", "../outside.png", "depth/0.png"])
def test_incomplete_or_unsafe_source_index_cannot_report_complete(scene, tmp_path, entry):
    src, allowlist, manifest = scene
    with (src / "rgb.txt").open("a") as file:
        file.write(f"3 {entry}\n")
    manifest["source"]["rgb_txt_sha256"] = sha(src / "rgb.txt")
    allowlist.write_text(json.dumps(manifest))
    dst = tmp_path / "invalid"
    assert run(src, allowlist, dst).returncode != 0
    assert not dst.exists()


def test_internal_file_alias_cannot_transform_protected_source_or_output(scene, tmp_path):
    src, allowlist, _ = scene
    (src / "rgb/train.png").unlink()
    (src / "rgb/train.png").symlink_to("heldout.png")
    original = (src / "rgb/heldout.png").read_bytes()
    dst = tmp_path / "internal-link"
    result = run(src, allowlist, dst, "sharpen-only")
    assert result.returncode == 0, result.stderr
    assert (src / "rgb/heldout.png").read_bytes() == original
    assert (dst / "rgb/heldout.png").read_bytes() == original
    assert (dst / "rgb/train.png").read_bytes() != original
    assert not (dst / "rgb/train.png").is_symlink()


def test_empty_allowlist_copies_whole_scene_even_for_filter_arm(scene, tmp_path):
    src, allowlist, manifest = scene
    manifest["train_rgb_paths"] = []
    allowlist.write_text(json.dumps(manifest))
    dst = tmp_path / "empty"
    result = run(src, allowlist, dst, "denoise-only")
    assert result.returncode == 0, result.stderr
    for path in src.rglob("*"):
        if path.is_file():
            assert (dst / path.relative_to(src)).read_bytes() == path.read_bytes()
    assert not any(row["transformed"] for row in json.loads(
        (dst / "preprocessing_manifest.json").read_text())["images"])
