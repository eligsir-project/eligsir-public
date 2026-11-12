"""Histogram Laplacian-variance (sharpness) and mean brightness (exposure)
across a full ROS2 bag trajectory, sampled every Nth frame. No training run
needed -- use this to sanity-check whether a scene has real blur/exposure
variance before tuning --streaming_keyframe_quality_adaptive's blur_floor/
exposure_tolerance for it (see FrameQualityConfig.adaptive_quality). Run
inside the ros2-jazzy container: python3 scripts/bag_blur_exposure_histogram.py <bag_dir> [stride]
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/workspace/eligsir/scripts")
from bag_image_reader import _iter_bag_messages, _deserialize  # noqa: E402

TOPIC = "/camera/color/image_raw/compressed"


def main(bag_path, stride=10, max_frames=800):
    sharpness = []
    brightness = []
    n = 0
    for name, blob, type_map in _iter_bag_messages(bag_path):
        if name != TOPIC:
            continue
        n += 1
        if n % stride != 0:
            continue
        msg = _deserialize(type_map[TOPIC], blob)
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness.append(lap_var)
        brightness.append(float(gray.mean()))
        if len(sharpness) >= max_frames:
            break
    sharpness = np.array(sharpness)
    brightness = np.array(brightness)
    print(f"bag={bag_path} n_sampled={len(sharpness)} (stride={stride}, total_seen={n})")
    print(f"sharpness(laplacian var): mean={sharpness.mean():.1f} std={sharpness.std():.1f} "
          f"min={sharpness.min():.1f} p5={np.percentile(sharpness,5):.1f} "
          f"p50={np.percentile(sharpness,50):.1f} p95={np.percentile(sharpness,95):.1f} max={sharpness.max():.1f} "
          f"cv={sharpness.std()/sharpness.mean():.2f}")
    print(f"brightness(mean gray 0-255): mean={brightness.mean():.1f} std={brightness.std():.1f} "
          f"min={brightness.min():.1f} p5={np.percentile(brightness,5):.1f} "
          f"p50={np.percentile(brightness,50):.1f} p95={np.percentile(brightness,95):.1f} max={brightness.max():.1f} "
          f"cv={brightness.std()/brightness.mean():.2f}")


if __name__ == "__main__":
    main(sys.argv[1], stride=int(sys.argv[2]) if len(sys.argv) > 2 else 10)
