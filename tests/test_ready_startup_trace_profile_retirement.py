"""Retired READY-startup diagnostics must not revive their old launchers."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_PROFILE_PATH = _ROOT / "experiments/researchflow/profiles.ready_startup_trace.toml"
_RETIRED = {
    "ready-startup-trace-native": "V1 run-5acee188f5-d76d0339 INVALID",
    "ready-startup-trace-lifecycle-native": "V2 run-a78402e135-644d1ea9 FAILED",
    "ready-gpu-lifecycle-probe-native": "D run-56406892c1-b32491cd INVALID",
    "ready-startup-trace-session-native": "V4 run-eed834bf50-8e92f4e9 FAILED",
}
_HISTORICAL_STATES = (
    "V1 run-5acee188f5-d76d0339: INVALID",
    "V2 run-a78402e135-644d1ea9: FAILED",
    "V3: unexecuted and\n# unverified",
    "D run-56406892c1-b32491cd: INVALID",
    "V4 run-eed834bf50-8e92f4e9",
    "V5 run-3ab86dae00-4b05e60b: FAILED",
)


def test_retired_profiles_are_structurally_guarded_before_execution(tmp_path):
    raw = _PROFILE_PATH.read_text(encoding="utf-8")
    profiles = tomllib.loads(raw)["profiles"]
    assert set(profiles) == set(_RETIRED)
    for historical_state in _HISTORICAL_STATES:
        assert historical_state in raw

    for name, historical_id in _RETIRED.items():
        argv = profiles[name]["argv"]
        assert argv[:2] == ["bash", "-c"]
        assert "historical-retired:" in argv[2]
        assert historical_id in raw
        assert all(token not in argv[2] for token in ("docker", "mkdir", "nsys", "nvidia-smi"))

        before = tuple(tmp_path.iterdir())
        result = subprocess.run(
            argv, cwd=tmp_path, capture_output=True, text=True, check=False
        )
        assert result.returncode == 64
        assert "historical-retired:" in result.stderr
        assert tuple(tmp_path.iterdir()) == before
