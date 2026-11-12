#!/usr/bin/env bash
set -euo pipefail
docker compose run --rm -e PYTHONPATH=/workspace/eligsir train python scripts/w3_34a_capture_probe.py
