import json
import os

from eligsir.utils.streaming_frames import RGBDSequenceFrameSource


def _make_dataset(tmp_path, n_frames=3):
    (tmp_path / "rgb").mkdir()
    intrinsics = {
        "width": 1920, "height": 1440,
        "fx": 1425.3392, "fy": 1425.3392,
        "cx": 954.9099, "cy": 724.43195,
        "depth_scale": 1000.0,
    }
    with open(tmp_path / "intrinsics.json", "w") as f:
        json.dump(intrinsics, f)
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    with open(tmp_path / "frames.jsonl", "w") as f:
        for i in range(n_frames):
            rgb_rel = f"rgb/{i:06d}.png"
            open(tmp_path / rgb_rel, "wb").close()
            f.write(json.dumps({"id": i, "timestamp": float(i), "rgb": rgb_rel, "c2w": identity}) + "\n")
    return tmp_path


def test_no_scaling_by_default(tmp_path):
    _make_dataset(tmp_path)
    src = RGBDSequenceFrameSource(str(tmp_path))
    frames = src.get_all()
    assert len(frames) == 3
    assert frames[0].width == 1920 and frames[0].height == 1440
    assert frames[0].fx == 1425.3392


def test_streaming_resolution_scales_all_frames(tmp_path):
    _make_dataset(tmp_path)
    src = RGBDSequenceFrameSource(str(tmp_path), streaming_resolution=2)
    frames = src.get_all()
    assert all(f.width == 960 and f.height == 720 for f in frames)
    assert frames[0].fx == 1425.3392 * 0.5


def test_full_res_indices_override_scaling(tmp_path):
    _make_dataset(tmp_path)
    src = RGBDSequenceFrameSource(str(tmp_path), streaming_resolution=2, full_res_indices={1})
    frames = src.get_all()
    assert frames[0].width == 960 and frames[0].height == 720
    assert frames[1].width == 1920 and frames[1].height == 1440
    assert frames[2].width == 960 and frames[2].height == 720
