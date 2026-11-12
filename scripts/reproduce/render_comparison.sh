#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 OUTPUT_DIR [ITERATION]" >&2
  exit 2
fi
output_dir=$1
iteration=${2:--1}
exec python render.py -m "$output_dir" --iteration "$iteration"
