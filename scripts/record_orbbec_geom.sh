#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-orbbec_femto_bolt_geom}"
echo "Record Orbbec Femto Bolt with aligned depth, manual exposure, and manual white balance."
echo "Output bag prefix: ${OUT}"
ros2 bag record -o "${OUT}" \
  /camera/color/image_raw \
  /camera/depth/image_raw \
  /camera/color/camera_info
