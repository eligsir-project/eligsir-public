#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-realsense_d435i_geom}"
echo "Record RealSense D435i with aligned depth, manual exposure, and manual white balance."
echo "Output bag prefix: ${OUT}"
ros2 bag record -o "${OUT}" \
  /camera/color/image_raw \
  /camera/aligned_depth_to_color/image_raw \
  /camera/color/camera_info
