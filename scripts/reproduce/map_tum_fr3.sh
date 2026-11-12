#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 SOURCE_DIR [OUTPUT_DIR]" >&2
  exit 2
fi
source_dir=$1
output_dir=${2:-output/paper_tum_fr3}
exec python train.py -s "$source_dir" -m "$output_dir" --config configs/paper/tum_fr3.yaml
