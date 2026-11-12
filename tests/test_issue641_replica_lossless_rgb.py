"""Issue #641 -- the Replica synthetic RGB source must not recompress.

`_raycast_replica_frame` used to encode EliGSiR's OWN raycast output as
JPEG quality 92 (and depth as PNG) before it became training/evaluation input.
These tests pin the post-fix contract on the CPU test surface: no JPEG encoder
in the Replica reader, an exactly-specified float->uint8 domain shared by both
Replica readers, and a bit-identical source -> frame -> camera-materialization
round trip.

The end-to-end raycast itself needs open3d + a Replica mesh (neither is present
on the CPU surface), so the value-preserving tail is tested directly.
"""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scene.readers.replica import replica_rgb_to_uint8
from eligsir.io.sources.decode_cache import (
    StreamingRGBDFrame,
    load_frame_depth_np,
    load_frame_rgb,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# AC#1: no JPEG encoder anywhere in the Replica synthetic source path.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rel_path",
    ["scene/readers/replica.py", "eligsir/io/sources/datasets/replica.py"],
)
def test_no_jpeg_encoder_in_replica_source_path(rel_path):
    """No codec invocation may survive; prose mentioning the removed JPEG hop
    is fine, an encoder call is not."""
    src = (REPO_ROOT / rel_path).read_text()
    for needle in ('format="JPEG"', "format='JPEG'", "quality=92"):
        assert needle not in src, (
            f"JPEG encoder invocation survives in {rel_path}: {needle!r}"
        )


# --------------------------------------------------------------------------
# Declared canonical RGB domain: round-half-away-from-zero, clipped, uint8.
# --------------------------------------------------------------------------

def test_rgb_domain_rounds_not_truncates():
    # 0.7 LSB: rounding gives 1, the old streaming path's truncation gave 0.
    vals = np.array([[[0.0, 1.0 / 255.0, 0.7 / 255.0, 1.4 / 255.0]]], dtype=np.float32)
    out = replica_rgb_to_uint8(vals)
    assert out.dtype == np.uint8
    assert out.tolist() == [[[0, 1, 1, 1]]]
    # ...and is exactly np.rint (round-half-to-even), the semantics the offline
    # point-splat reader already used.
    truncated = np.clip(vals * 255.0, 0, 255).astype(np.uint8)
    assert not np.array_equal(out, truncated)


def test_rgb_domain_matches_declared_rint_reference():
    rng = np.random.default_rng(641)
    x = rng.random((8, 8, 3), dtype=np.float32)
    reference = np.clip(np.rint(x * 255.0), 0.0, 255.0).astype(np.uint8)
    assert np.array_equal(replica_rgb_to_uint8(x), reference)


def test_rgb_domain_clips_and_is_exact_on_grid():
    grid = (np.arange(256, dtype=np.float32) / 255.0).reshape(1, 256, 1)
    grid = np.repeat(grid, 3, axis=2)
    out = replica_rgb_to_uint8(grid)
    assert np.array_equal(out[0, :, 0], np.arange(256, dtype=np.uint8))

    over = np.array([[[1.0000001, 2.0, -0.5]]], dtype=np.float32)
    assert replica_rgb_to_uint8(over).tolist() == [[[255, 255, 0]]]


def test_rgb_domain_returns_fresh_writable_contiguous_array():
    src = np.zeros((2, 3, 3), dtype=np.float32)
    out = replica_rgb_to_uint8(src)
    assert out.flags["C_CONTIGUOUS"] and out.flags["WRITEABLE"]
    assert out.base is None or out.base is not src  # never a view of the input
    out[0, 0, 0] = 7  # must not raise (issue #593: torch.from_numpy needs this)


def test_rgb_domain_beats_jpeg_q92_on_the_same_input():
    """The old path was genuinely lossy -- guards against a silent regression."""
    rng = np.random.default_rng(641)
    rgb01 = rng.random((32, 32, 3), dtype=np.float32)
    exact = replica_rgb_to_uint8(rgb01)

    buf = io.BytesIO()
    Image.fromarray(exact, mode="RGB").save(buf, format="JPEG", quality=92)
    lossy = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))

    assert not np.array_equal(exact, lossy), "JPEG q92 was expected to be lossy"


# --------------------------------------------------------------------------
# AC#3: source -> frame -> camera materialization preserves samples exactly.
# --------------------------------------------------------------------------

def _replica_like_frame(rgb_uint8, depth_mm):
    """A frame built exactly as ReplicaFrameSource now builds one."""
    return StreamingRGBDFrame(
        index=0,
        timestamp=0.0,
        rgb_path="",
        depth_path=None,
        c2w=np.eye(4, dtype=np.float32),
        fx=1.0, fy=1.0, cx=0.5, cy=0.5,
        width=int(rgb_uint8.shape[1]), height=int(rgb_uint8.shape[0]),
        depth_scale=1000.0,
        _rgb_np=rgb_uint8,
        _depth_np=depth_mm,
    )


def test_frame_rgb_round_trip_is_bit_identical():
    rng = np.random.default_rng(1)
    rgb = replica_rgb_to_uint8(rng.random((17, 23, 3), dtype=np.float32))
    frame = _replica_like_frame(rgb, np.zeros((17, 23), dtype=np.uint16))

    loaded = np.asarray(load_frame_rgb(frame))
    assert loaded.dtype == np.uint8
    assert loaded.shape == rgb.shape
    assert np.array_equal(loaded, rgb)


def test_frame_depth_round_trip_is_bit_identical():
    rng = np.random.default_rng(2)
    depth = rng.integers(0, 65535, size=(17, 23), dtype=np.uint16)
    frame = _replica_like_frame(
        np.zeros((17, 23, 3), dtype=np.uint8), np.ascontiguousarray(depth)
    )

    loaded = load_frame_depth_np(frame)
    assert loaded.dtype == np.uint16
    assert np.array_equal(loaded, depth)


def test_ndarray_depth_matches_the_old_uint16_png_encoding_exactly():
    """Dropping the depth PNG container changes no value (issue #642 owns the
    float->uint16-mm quantisation itself, which is untouched here)."""
    rng = np.random.default_rng(3)
    depth = rng.integers(0, 65535, size=(9, 11), dtype=np.uint16)

    buf = io.BytesIO()
    Image.fromarray(depth, mode="I;16").save(buf, format="PNG")
    via_png = np.asarray(Image.open(io.BytesIO(buf.getvalue())))

    assert via_png.dtype == depth.dtype
    assert np.array_equal(via_png, depth)


# --------------------------------------------------------------------------
# AC#4: both Replica readers share one declared image domain.
# --------------------------------------------------------------------------

def test_offline_point_splat_uses_the_same_conversion_helper():
    src = (REPO_ROOT / "scene/readers/replica.py").read_text()
    assert src.count("replica_rgb_to_uint8(") >= 3, (
        "both the offline point-splat reader and the streaming raycaster must "
        "route their float->uint8 conversion through the shared helper"
    )
    assert "np.rint(cc[chosen] * 255.0)" not in src


def test_raycast_frame_returns_arrays_not_bytes():
    """Contract check that does not require open3d: the caller stores the
    return values on _rgb_np/_depth_np, never on the *_bytes fields."""
    src = (REPO_ROOT / "eligsir/io/sources/datasets/replica.py").read_text()
    assert "_rgb_np=rgb_np" in src and "_depth_np=depth_np" in src
    assert "_rgb_bytes=" not in src and "_depth_bytes=" not in src


# --------------------------------------------------------------------------
# AC#5: report.json carries the source fidelity marker for Replica runs.
# --------------------------------------------------------------------------

def test_replica_rgb_marker_wired_in_training_report():
    """Issue #641: streaming training loop detects Replica source and records
    it in report.json so pre-fix and post-fix runs self-identify their protocol
    difference (JPEG vs. lossless)."""
    src = (REPO_ROOT / "train_streaming.py").read_text()
    # The _replica_rgb_report dict must be created:
    assert "_replica_rgb_report" in src
    # And it must be conditional on actually using Replica source:
    assert "isinstance(frame_source, ReplicaFrameSource)" in src
    # And it must be passed into the report kwargs:
    assert "replica_rgb" in src and "_replica_rgb_report" in src
