#!/usr/bin/env bash
set -euo pipefail

# Execution semantics come from a named profile.
PROFILE="${PROFILE:-causal_replay/tum_rgbd}"

# Run-specific paths and knobs live in the YAML overlay.
CONFIG="${CONFIG:-examples/runs/normal/config.yml}"

python train.py \
  --run_profile "${PROFILE}" \
  --config "${CONFIG}"
