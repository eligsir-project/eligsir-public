"""Tests for RealityScanFrameSource + its registry wiring.

The real `realityscan_importer` package is not installed on this host (it lives
in a sibling repo, see CLAUDE.md's "Related Repos"). Every test here injects a
fake module into sys.modules before constructing the source, per the brief's
intended test strategy — no test may require the real package to be importable.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Fake realityscan_importer scene model (mirrors the real package's API).
# ---------------------------------------------------------------------------

@dataclass
class FakeCamera:
    uuid: str
    image_path: Path
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_from_camera: np.ndarray
    prior_pose: Optional[np.ndarray] = None
    pose_source: str = "sfm"
    distortion_applied: bool = True


@dataclass
class FakePointCloud:
    xyz: np.ndarray
    rgb: Optional[np.ndarray] = None


@dataclass
class FakeRealityScanScene:
    cameras: list
    point_cloud: Optional[FakePointCloud] = None
    metadata: dict = field(default_factory=dict)


def _write_jpeg_with_exif(path: Path, *, datetime_original: Optional[str] = None):
    """Write a tiny real JPEG, optionally with EXIF DateTimeOriginal (tag 36867)."""
    img = Image.new("RGB", (8, 8), color=(128, 64, 32))
    if datetime_original is not None:
        exif = img.getexif()
        exif[36867] = datetime_original  # DateTimeOriginal
        img.save(path, format="JPEG", exif=exif)
    else:
        img.save(path, format="JPEG")


def _fake_c2w(tx: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx
    return m


def _make_scene_dir(tmp_path: Path) -> Path:
    (tmp_path / "sfm.txt").write_text('RSv2,["fake"]')
    (tmp_path / "dummy.xmp").write_text("<x/>")
    return tmp_path


def _install_fake_importer(monkeypatch, rs_scene):
    fake_module = types.ModuleType("realityscan_importer")

    def _load_project(project_dir, undistort=True, cache_dir=None, min_pcd_confidence=0.5):
        _load_project.called_with = dict(
            project_dir=project_dir,
            undistort=undistort,
            cache_dir=cache_dir,
            min_pcd_confidence=min_pcd_confidence,
        )
        return rs_scene

    fake_module.load_project = _load_project
    monkeypatch.setitem(sys.modules, "realityscan_importer", fake_module)
    return _load_project


def _build_three_camera_scene(tmp_path: Path):
    """3 cameras with UUID (random) filenames; 2 have EXIF DateTimeOriginal out of
    filename order, 1 has no EXIF at all (falls back to filename order, placed after
    the EXIF-ordered ones)."""
    # Filenames are intentionally NOT in capture order (UUID-like).
    p_zzz = tmp_path / "zzz-no-exif.jpg"
    p_bbb = tmp_path / "bbb-second.jpg"
    p_aaa = tmp_path / "aaa-first.jpg"

    # EXIF order deliberately DISAGREES with filename-alphabetical order, so a
    # test asserting the EXIF order cannot pass by accident from a pure
    # filename sort: alphabetically the filenames sort [aaa, bbb, zzz], but
    # bbb's EXIF DateTimeOriginal is EARLIER than aaa's, so the correct
    # EXIF-ordered result is [bbb, aaa, zzz].
    _write_jpeg_with_exif(p_bbb, datetime_original="2026:01:01 12:00:01")
    _write_jpeg_with_exif(p_aaa, datetime_original="2026:01:01 12:00:05")
    _write_jpeg_with_exif(p_zzz, datetime_original=None)  # no EXIF -> filename fallback, goes last

    cameras = [
        FakeCamera(
            uuid="cam-zzz", image_path=p_zzz, width=8, height=8,
            fx=10.0, fy=10.0, cx=4.0, cy=4.0, world_from_camera=_fake_c2w(99.0),
        ),
        FakeCamera(
            uuid="cam-bbb", image_path=p_bbb, width=8, height=8,
            fx=10.0, fy=10.0, cx=4.0, cy=4.0, world_from_camera=_fake_c2w(2.0),
        ),
        FakeCamera(
            uuid="cam-aaa", image_path=p_aaa, width=8, height=8,
            fx=10.0, fy=10.0, cx=4.0, cy=4.0, world_from_camera=_fake_c2w(1.0),
        ),
    ]
    xyz = np.random.rand(10, 3).astype(np.float32)
    rgb = (np.random.rand(10, 3) * 255).astype(np.uint8)
    pc = FakePointCloud(xyz=xyz, rgb=rgb)
    return FakeRealityScanScene(cameras=cameras, point_cloud=pc, metadata={})


def test_frames_ordered_by_exif_not_filename(tmp_path, monkeypatch):
    _make_scene_dir(tmp_path)
    rs_scene = _build_three_camera_scene(tmp_path)
    _install_fake_importer(monkeypatch, rs_scene)

    from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource

    src = RealityScanFrameSource(str(tmp_path))
    frames = src.get_all()
    assert len(frames) == 3
    # Expected capture order by EXIF: bbb (12:00:01), aaa (12:00:05), then zzz (no EXIF, filename fallback).
    # This ORDER DIFFERS from plain filename-alphabetical order ([aaa, bbb, zzz]),
    # so this assertion can only pass if EXIF timestamps actually drive ordering --
    # it would fail if the implementation silently fell back to filename sort.
    assert [f.c2w[0, 3] for f in frames] == [2.0, 1.0, 99.0]


def test_frames_have_no_depth_and_real_c2w(tmp_path, monkeypatch):
    _make_scene_dir(tmp_path)
    rs_scene = _build_three_camera_scene(tmp_path)
    _install_fake_importer(monkeypatch, rs_scene)

    from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource

    src = RealityScanFrameSource(str(tmp_path))
    for f in src.get_all():
        assert f.depth_path is None
        assert f.c2w is not None
        assert f.c2w.shape == (4, 4)


def test_seed_points_and_colors_from_point_cloud(tmp_path, monkeypatch):
    _make_scene_dir(tmp_path)
    rs_scene = _build_three_camera_scene(tmp_path)
    _install_fake_importer(monkeypatch, rs_scene)

    from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource

    src = RealityScanFrameSource(str(tmp_path))
    assert src.seed_points is not None
    assert src.seed_points.shape == (10, 3)
    assert src.seed_points.dtype == np.float32
    assert src.seed_colors is not None
    assert src.seed_colors.shape == (10, 3)
    # uint8 0-255 rgb must be normalized into [0, 1]
    assert src.seed_colors.min() >= 0.0
    assert src.seed_colors.max() <= 1.0


def test_seed_colors_none_when_no_point_cloud(tmp_path, monkeypatch):
    _make_scene_dir(tmp_path)
    rs_scene = _build_three_camera_scene(tmp_path)
    rs_scene.point_cloud = None
    _install_fake_importer(monkeypatch, rs_scene)

    from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource

    src = RealityScanFrameSource(str(tmp_path))
    assert src.seed_points is None
    assert src.seed_colors is None


def test_registry_dispatches_to_realityscan_frame_source(tmp_path, monkeypatch):
    _make_scene_dir(tmp_path)
    rs_scene = _build_three_camera_scene(tmp_path)
    _install_fake_importer(monkeypatch, rs_scene)

    from eligsir.io.sources.registry import make_frame_source
    from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource

    class FakeArgs:
        streaming_max_frames = 0
        streaming_frame_stride = 1

    src = make_frame_source(str(tmp_path), FakeArgs())
    assert isinstance(src, RealityScanFrameSource)


def test_detect_source_kind_realityscan(tmp_path, monkeypatch):
    _make_scene_dir(tmp_path)
    rs_scene = _build_three_camera_scene(tmp_path)
    _install_fake_importer(monkeypatch, rs_scene)

    from eligsir.io.sources.registry import detect_source_kind

    assert detect_source_kind(str(tmp_path)) == "realityscan"


def test_sfm_without_xmp_does_not_dispatch_to_realityscan(tmp_path):
    # sfm.txt present but no .xmp -> detection specificity: must NOT match realityscan,
    # and since nothing else matches either, make_frame_source must raise ValueError.
    (tmp_path / "sfm.txt").write_text('RSv2,["fake"]')

    from eligsir.io.sources.registry import make_frame_source

    class FakeArgs:
        streaming_max_frames = 0
        streaming_frame_stride = 1

    with pytest.raises(ValueError):
        make_frame_source(str(tmp_path), FakeArgs())


def test_max_frames_and_frame_stride_respected(tmp_path, monkeypatch):
    _make_scene_dir(tmp_path)
    rs_scene = _build_three_camera_scene(tmp_path)
    _install_fake_importer(monkeypatch, rs_scene)

    from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource

    # EXIF-ordered sequence is [bbb(2.0), aaa(1.0), zzz(99.0)] (see
    # _build_three_camera_scene) -- frame_stride=2 keeps indices 0, 2 of THAT
    # order, i.e. bbb then zzz.
    src = RealityScanFrameSource(str(tmp_path), frame_stride=2)
    frames = src.get_all()
    assert len(frames) == 2
    assert [f.c2w[0, 3] for f in frames] == [2.0, 99.0]

    src2 = RealityScanFrameSource(str(tmp_path), max_frames=1)
    frames2 = src2.get_all()
    assert len(frames2) == 1
    assert frames2[0].c2w[0, 3] == 2.0


def _build_no_exif_three_camera_scene(tmp_path: Path):
    """3 cameras, none with EXIF, filenames NOT in capture order, and
    `sfm.txt` registration order (the `cameras` list order) deliberately
    WRONG relative to true capture order -- reproducing the real 2026-08-12
    finding (docs/architecture/2026-08-12-realityscan-trajectory-distance-
    bug.md) that sfm.txt registration order can be uncorrelated with actual
    capture sequence (Spearman rho ~0.03 measured on a real hallway
    capture). File mtime is the only reliable signal available here."""
    p_c = tmp_path / "uuid-c.jpg"
    p_a = tmp_path / "uuid-a.jpg"
    p_b = tmp_path / "uuid-b.jpg"
    for p in (p_c, p_a, p_b):
        _write_jpeg_with_exif(p, datetime_original=None)

    cameras = [
        FakeCamera(
            uuid="cam-c", image_path=p_c, width=8, height=8,
            fx=10.0, fy=10.0, cx=4.0, cy=4.0, world_from_camera=_fake_c2w(3.0),
        ),
        FakeCamera(
            uuid="cam-a", image_path=p_a, width=8, height=8,
            fx=10.0, fy=10.0, cx=4.0, cy=4.0, world_from_camera=_fake_c2w(1.0),
        ),
        FakeCamera(
            uuid="cam-b", image_path=p_b, width=8, height=8,
            fx=10.0, fy=10.0, cx=4.0, cy=4.0, world_from_camera=_fake_c2w(2.0),
        ),
    ]
    xyz = np.random.rand(10, 3).astype(np.float32)
    rgb = (np.random.rand(10, 3) * 255).astype(np.uint8)
    pc = FakePointCloud(xyz=xyz, rgb=rgb)
    return FakeRealityScanScene(cameras=cameras, point_cloud=pc, metadata={}), (p_a, p_b, p_c)


def test_no_exif_falls_back_to_mtime_not_sfm_order(tmp_path, monkeypatch):
    rs_scene, (p_a, p_b, p_c) = _build_no_exif_three_camera_scene(tmp_path)
    _install_fake_importer(monkeypatch, rs_scene)

    import os
    import time
    now = time.time()
    # True capture order is a, b, c (ascending mtime) -- OPPOSITE of the
    # sfm.txt/`cameras` list order above (c, a, b). Only an mtime-based
    # fallback can recover [1.0, 2.0, 3.0] here.
    os.utime(p_a, (now, now))
    os.utime(p_b, (now + 5, now + 5))
    os.utime(p_c, (now + 10, now + 10))

    from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource

    src = RealityScanFrameSource(str(tmp_path))
    frames = src.get_all()
    assert [f.c2w[0, 3] for f in frames] == [1.0, 2.0, 3.0]


def test_no_exif_and_no_discriminating_mtime_falls_back_to_sfm_order(tmp_path, monkeypatch):
    """Gate check: when every file shares the same mtime (e.g. a bulk-copied
    export), the mtime signal carries no information and must NOT be trusted
    -- order must fall back to sfm.txt registration order rather than
    silently producing an arbitrary (e.g. filesystem-stat-order) result."""
    rs_scene, (p_a, p_b, p_c) = _build_no_exif_three_camera_scene(tmp_path)
    _install_fake_importer(monkeypatch, rs_scene)

    import os
    import time
    now = time.time()
    for p in (p_a, p_b, p_c):
        os.utime(p, (now, now))

    from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource

    src = RealityScanFrameSource(str(tmp_path))
    frames = src.get_all()
    # sfm.txt/`cameras` list order is [cam-c(3.0), cam-a(1.0), cam-b(2.0)].
    assert [f.c2w[0, 3] for f in frames] == [3.0, 1.0, 2.0]


def test_len_and_iter(tmp_path, monkeypatch):
    _make_scene_dir(tmp_path)
    rs_scene = _build_three_camera_scene(tmp_path)
    _install_fake_importer(monkeypatch, rs_scene)

    from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource

    src = RealityScanFrameSource(str(tmp_path))
    assert len(src) == 3
    assert len(list(iter(src))) == 3
    # indices assigned 0..N-1 in capture order
    assert [f.index for f in src] == [0, 1, 2]


def test_missing_realityscan_importer_raises_helpful_import_error(tmp_path, monkeypatch):
    _make_scene_dir(tmp_path)
    monkeypatch.setitem(sys.modules, "realityscan_importer", None)  # force ImportError on import

    from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource

    with pytest.raises(ImportError):
        RealityScanFrameSource(str(tmp_path))


# ---------------------------------------------------------------------------
# Resolution downscale — regression tests for the real GPU hang this session
# (2026-08-12): a RealityScan smoke run trained on native 3000x4000 (12MP)
# photos because train_streaming.py's --streaming_auto_downscale_max_pixels
# gate only checks for intrinsics.json, which RealityScan sources never have,
# and RealityScanFrameSource previously had no resolution parameter at all —
# every frame was built at full native size regardless of any flag.
# ---------------------------------------------------------------------------

def _build_large_camera_scene(tmp_path: Path, width: int = 4000, height: int = 3000):
    p = tmp_path / "big.jpg"
    _write_jpeg_with_exif(p, datetime_original="2026:01:01 12:00:01")
    cam = FakeCamera(
        uuid="cam-big", image_path=p, width=width, height=height,
        fx=3000.0, fy=3000.0, cx=width / 2.0, cy=height / 2.0,
        world_from_camera=_fake_c2w(0.0),
    )
    return FakeRealityScanScene(cameras=[cam], point_cloud=None, metadata={})


def test_explicit_streaming_resolution_downscales_frame(tmp_path, monkeypatch):
    _make_scene_dir(tmp_path)
    rs_scene = _build_large_camera_scene(tmp_path, width=4000, height=3000)
    _install_fake_importer(monkeypatch, rs_scene)

    from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource

    src = RealityScanFrameSource(str(tmp_path), streaming_resolution=2)
    frame = src.get_all()[0]
    assert (frame.width, frame.height) == (2000, 1500)
    # intrinsics scale with the resize (pixel-center-correct, per resize_rgbd_frame's convention)
    assert frame.fx == pytest.approx(1500.0)
    assert frame.fy == pytest.approx(1500.0)
    assert frame.cx == pytest.approx((2000.0 + 0.5) * 0.5 - 0.5)
    assert frame.cy == pytest.approx((1500.0 + 0.5) * 0.5 - 0.5)


def test_auto_downscale_max_pixels_computes_divisor_without_intrinsics_json(tmp_path, monkeypatch):
    _make_scene_dir(tmp_path)
    assert not (tmp_path / "intrinsics.json").exists()  # the exact condition that broke the old gate
    rs_scene = _build_large_camera_scene(tmp_path, width=4000, height=3000)  # 12,000,000 px
    _install_fake_importer(monkeypatch, rs_scene)

    from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource

    src = RealityScanFrameSource(str(tmp_path), auto_downscale_max_pixels=2_000_000)
    frame = src.get_all()[0]
    assert frame.width * frame.height <= 2_000_000
    # divisor must be the smallest integer that satisfies the cap (matches
    # train_streaming.py's own gate search: divisor=2 -> 2000x1500=3,000,000 px,
    # still over budget; divisor=3 -> 1333x1000=1,333,000 px, under budget).
    assert (frame.width, frame.height) == (4000 // 3, 3000 // 3)


def test_explicit_streaming_resolution_takes_precedence_over_auto_downscale(tmp_path, monkeypatch):
    _make_scene_dir(tmp_path)
    rs_scene = _build_large_camera_scene(tmp_path, width=4000, height=3000)
    _install_fake_importer(monkeypatch, rs_scene)

    from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource

    # explicit streaming_resolution=2 must win even though auto_downscale_max_pixels
    # alone would have picked a different (larger) divisor.
    src = RealityScanFrameSource(
        str(tmp_path), streaming_resolution=2, auto_downscale_max_pixels=2_000_000,
    )
    frame = src.get_all()[0]
    assert (frame.width, frame.height) == (2000, 1500)


def test_no_downscale_when_under_budget(tmp_path, monkeypatch):
    _make_scene_dir(tmp_path)
    rs_scene = _build_three_camera_scene(tmp_path)  # 8x8 cameras, tiny
    _install_fake_importer(monkeypatch, rs_scene)

    from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource

    src = RealityScanFrameSource(str(tmp_path), auto_downscale_max_pixels=2_000_000)
    frame = src.get_all()[0]
    assert (frame.width, frame.height) == (8, 8)  # unchanged: 64px is already under the cap


def test_default_construction_still_native_resolution(tmp_path, monkeypatch):
    """Behavior-neutral when neither new param is passed (registry callers before
    this fix, or any caller that omits them) — reproduces the exact pre-fix
    default so existing callers are unaffected."""
    _make_scene_dir(tmp_path)
    rs_scene = _build_large_camera_scene(tmp_path, width=4000, height=3000)
    _install_fake_importer(monkeypatch, rs_scene)

    from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource

    src = RealityScanFrameSource(str(tmp_path))
    frame = src.get_all()[0]
    assert (frame.width, frame.height) == (4000, 3000)
