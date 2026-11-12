#!/usr/bin/env bash
set -euo pipefail

ABLATOR_CONFIG="${ABLATOR_CONFIG:-${HOME}/.config/ablator/config.toml}"
SPEC="${SPEC:-examples/runs/ablation/plan.json}"

ablator --config "${ABLATOR_CONFIG}" plan "${SPEC}"
ablator --config "${ABLATOR_CONFIG}" status example_cap
