from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path

import pytest


_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "experiments/researchflow/profiles.fr1_visibility_local.toml"
)


def _kfd_guard() -> str:
    profile = tomllib.loads(_PROFILE.read_text())["profiles"][
        "fr1-visibility-local-native-b8"
    ]
    launcher = profile["argv"][2]
    start = launcher.index("test -x /usr/bin/distrobox-host-exec")
    end = launcher.index("; docker image inspect", start)
    return launcher[start:end]


@pytest.mark.parametrize(
    ("mode", "expected_success"),
    [
        ("noholder", True),
        ("busy", False),
        ("queryerror1", False),
        ("queryerror", False),
        ("bridgeerror", False),
    ],
)
def test_local_visibility_kfd_guard_accepts_only_a_clean_host_query(
    tmp_path: Path, mode: str, expected_success: bool
) -> None:
    bridge = tmp_path / "distrobox-host-exec"
    bridge.write_text(
        "#!/bin/sh\n"
        'test "$1" = /usr/bin/fuser || exit 2\n'
        'test "$2" = /dev/kfd || exit 2\n'
        'case "$KFD_GUARD_MODE" in\n'
        "  noholder) exit 1 ;;\n"
        '  busy) printf "4242\\n"; exit 0 ;;\n'
        '  queryerror1) printf "fuser permission error\\n" >&2; exit 1 ;;\n'
        '  queryerror) printf "fuser failed\\n" >&2; exit 2 ;;\n'
        '  bridgeerror) printf "bridge failed\\n" >&2; exit 127 ;;\n'
        "esac\n"
    )
    bridge.chmod(0o755)
    env = {**os.environ, "KFD_GUARD_MODE": mode}
    result = subprocess.run(
        [
            "bash",
            "-euc",
            _kfd_guard().replace("/usr/bin/distrobox-host-exec", str(bridge)),
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    assert (result.returncode == 0) is expected_success, result.stderr
