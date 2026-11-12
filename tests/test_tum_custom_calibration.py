from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scene.readers.tum import (
    _load_tum_rgbd_pointcloud,
    _resolve_tum_camera_config,
)
from eligsir.io.sources.datasets.tum import TUMFrameSource


def _write_config(path, **overrides):
    values = {
        "H": 2,
        "W": 2,
        "fx": 10.0,
        "fy": 11.0,
        "cx": 1.0,
        "cy": 1.0,
        "depth_scale": 1000.0,
    }
    values.update(overrides)
    (path / "config.yaml").write_text(
        "\n".join(f"{key}: {value}" for key, value in values.items()) + "\n"
    )


def test_custom_config_overrides_intrinsics_dimensions_and_depth_scale(tmp_path):
    _write_config(tmp_path)

    config = _resolve_tum_camera_config(tmp_path, depth_scale=5000.0)

    assert (config.fx, config.fy, config.cx, config.cy) == (10.0, 11.0, 1.0, 1.0)
    assert (config.width, config.height) == (2, 2)
    assert config.depth_scale == 1000.0
    assert config.source == str(tmp_path / "config.yaml")


def test_public_tum_without_config_preserves_freiburg_defaults(tmp_path):
    path = tmp_path / "rgbd_dataset_freiburg3_long_office_household"
    path.mkdir()

    config = _resolve_tum_camera_config(path, depth_scale=5000.0)

    assert (config.fx, config.fy, config.cx, config.cy) == (535.4, 539.2, 320.1, 247.6)
    assert config.width is None and config.height is None
    assert config.depth_scale == 5000.0
    assert config.source == "tum_builtin"


def test_public_fr2_registered_pngs_use_openni_registered_geometry(tmp_path):
    """TUM's provided depth PNGs are already registered to the RGB image.

    The documented calibrated FR2 RGB model carries distortion and cannot be
    used as an undistorted pinhole model without jointly rectifying RGB and the
    registered depth.  The causal and legacy readers therefore need the
    documented registered OpenNI pinhole geometry instead.
    """

    path = tmp_path / "rgbd_dataset_freiburg2_desk"
    path.mkdir()

    config = _resolve_tum_camera_config(path, depth_scale=5000.0)

    assert (config.fx, config.fy, config.cx, config.cy) == (525.0, 525.0, 319.5, 239.5)
    assert config.depth_scale == 5000.0
    assert config.source == "tum_registered_openni"


def test_causal_tum_source_uses_the_same_fr2_registered_geometry(tmp_path):
    (tmp_path / "rgb").mkdir()
    (tmp_path / "depth").mkdir()
    Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(
        tmp_path / "rgb" / "0.png"
    )
    Image.fromarray(np.full((480, 640), 5000, dtype=np.uint16)).save(
        tmp_path / "depth" / "0.png"
    )
    (tmp_path / "rgb.txt").write_text("0.0 rgb/0.png\n")
    (tmp_path / "depth.txt").write_text("0.0 depth/0.png\n")
    (tmp_path / "groundtruth.txt").write_text("0.0 0 0 0 0 0 0 1\n")

    source = TUMFrameSource(str(tmp_path), sequence="freiburg2_desk")
    frame = source.get_all()[0]

    assert (frame.fx, frame.fy, frame.cx, frame.cy) == (525.0, 525.0, 319.5, 239.5)
    assert (frame.width, frame.height, frame.depth_scale) == (640, 480, 5000.0)


def test_present_but_partial_custom_config_fails_closed(tmp_path):
    _write_config(tmp_path)
    text = (tmp_path / "config.yaml").read_text().replace("fy: 11.0\n", "")
    (tmp_path / "config.yaml").write_text(text)

    with pytest.raises(ValueError, match="missing.*fy"):
        _resolve_tum_camera_config(tmp_path, depth_scale=5000.0)


def test_custom_depth_scale_is_used_for_rgbd_seed_geometry(tmp_path):
    (tmp_path / "depth").mkdir()
    Image.fromarray(np.full((2, 2), 1000, dtype=np.uint16)).save(
        tmp_path / "depth" / "000000.png"
    )
    _write_config(tmp_path)
    config = _resolve_tum_camera_config(tmp_path, depth_scale=5000.0)
    cam = SimpleNamespace(
        image_name="0_000000",
        image=Image.fromarray(np.full((2, 2, 3), 127, dtype=np.uint8)),
        R=np.eye(3, dtype=np.float32),
        T=np.zeros(3, dtype=np.float32),
    )

    pointcloud = _load_tum_rgbd_pointcloud(
        path=str(tmp_path),
        train_cam_infos=[cam],
        associations=[{"rgb_t": 0.0, "depth_rel": "depth/000000.png"}],
        depth_stride=1,
        init_frames=1,
        max_points=0,
        camera_config=config,
    )

    assert pointcloud.points.shape[0] == 4
    assert np.allclose(pointcloud.points[:, 2], 1.0)
