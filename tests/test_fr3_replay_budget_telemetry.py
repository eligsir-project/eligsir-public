"""Exercise the recipe's collector lifetime and fail-closed outer wrapper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml


RECIPES = Path(__file__).resolve().parents[1] / "experiments/researchflow"


@pytest.fixture
def launch(tmp_path):
    # ResearchFlow is an external dependency here. Its real collector is
    # qualified separately; this double exposes the wrapper's call ordering.
    checkout = tmp_path / "telemetry-source"
    module = checkout / "src/researchflow/system_telemetry.py"
    module.parent.mkdir(parents=True)
    (module.parent / "__init__.py").touch()
    module.write_text('''import json
from pathlib import Path

class SystemTelemetry:
    def __init__(self, path, *, env, input_paths, expected_hostname):
        self.path = Path(path)
        self.env = env
        self.inputs = input_paths
        self.hostname = expected_hostname
    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"event": "start", "hostname": self.hostname,
                                        "input_paths": self.inputs}) + "\\n")
    def stop(self):
        with self.path.open("a") as out:
            out.write('{"event": "stop"}\\n')
        if self.env.get("COLLECTOR_RAISES") == "1":
            raise RuntimeError("collector failed")
        metadata = {"complete": self.env.get("COLLECTOR_COMPLETE", "1") == "1"}
        self.path.with_suffix(".svg").write_text("<svg/>")
        self.path.with_suffix(".manifest.json").write_text(json.dumps(metadata))
        return metadata
''', encoding="utf-8")
    for command in (
        ["git", "init", "-q", str(checkout)],
        ["git", "-C", str(checkout), "add", "src"],
        ["git", "-C", str(checkout), "-c", "user.name=Test", "-c",
         "user.email=test@example.invalid", "commit", "-qm", "collector fixture"],
    ):
        subprocess.run(command, check=True, capture_output=True)
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True,
    ).strip()
    profile = tomllib.loads((RECIPES / "profiles.fr3_replay_budget_20260907.toml").read_text())["profiles"]["fr3-replay-budget-r9700"]
    params = yaml.safe_load((RECIPES / "fr3_replay_budget_control_200_20260907.yaml").read_text())["run"]["parameters"]
    params.update(researchflow_telemetry_checkout=str(checkout),
                  researchflow_telemetry_commit=commit, model_path=str(tmp_path / "output"),
                  host_name="fixture-host", physical_source_path="/fixture/ram",
                  canonical_source_path="/fixture/canonical")
    argv = [part.format_map(params) for part in profile["argv"]]
    argv[0] = sys.executable  # The actual r9700 interpreter is a host qualification.
    telemetry = Path(params["model_path"] + ".host-system-telemetry.jsonl")
    return argv, telemetry, checkout


def _run(argv, **environment):
    return subprocess.run(argv, env={**os.environ, **environment},
                          capture_output=True, text=True, timeout=20)


@pytest.mark.parametrize(("child_status", "complete", "raises", "expected"), [
    (0, "1", "0", 0), (23, "1", "0", 23), (0, "0", "0", 66),
    (0, "1", "1", 66), (23, "1", "1", 23),
])
def test_outer_wrapper_brackets_child_and_preserves_failure(launch, child_status, complete, raises, expected):
    argv, telemetry, _ = launch
    # Replace only the external mapper command with a CPU sentinel. The whole
    # rendered Python wrapper, pin checks and positional arguments execute.
    child = '''import json, sys
from pathlib import Path
p = Path(sys.argv[1])
assert json.loads(p.read_text().splitlines()[0])["event"] == "start"
with p.open("a") as out:
    out.write('{"event": "child"}\\n')
raise SystemExit(int(sys.argv[2]))
'''
    result = _run(argv[:argv.index("env")] + [sys.executable, "-c", child, str(telemetry), str(child_status)],
                  COLLECTOR_COMPLETE=complete, COLLECTOR_RAISES=raises)
    assert result.returncode == expected, result.stderr
    rows = [json.loads(line) for line in telemetry.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["start", "child", "stop"]
    assert rows[0]["hostname"] == "fixture-host"
    assert rows[0]["input_paths"] == {"physical": "/fixture/ram", "canonical": "/fixture/canonical"}
    if raises == "0":
        assert telemetry.with_suffix(".svg").is_file()
        assert json.loads(telemetry.with_suffix(".manifest.json").read_text())["complete"] == (complete == "1")


@pytest.mark.parametrize("suffix", [".jsonl", ".svg", ".manifest.json"])
def test_outer_wrapper_rejects_existing_output_without_overwriting(launch, suffix):
    argv, telemetry, _ = launch
    existing = telemetry.with_suffix(suffix)
    existing.write_text("preserve me", encoding="utf-8")
    result = _run(argv)
    assert result.returncode != 0
    assert "host telemetry output paths must be fresh" in result.stderr
    assert existing.read_text() == "preserve me"


@pytest.mark.parametrize("invalid", ["commit", "dirty"])
def test_outer_wrapper_rejects_unattested_collector_before_start(launch, invalid):
    argv, telemetry, checkout = launch
    if invalid == "commit":
        argv[4] = "0" * 40
    else:
        (checkout / "untracked").write_text("dirty", encoding="utf-8")
    result = _run(argv)
    assert result.returncode != 0
    assert ("checkout commit mismatch" if invalid == "commit" else "checkout is not clean") in result.stderr
    assert not telemetry.exists()


def test_outer_wrapper_stops_after_actual_inner_preflight_rejection(launch, tmp_path):
    argv, telemetry, _ = launch
    # Keep the actual inner launcher, but point its mapping checkout at an
    # uncreated test-local path. Its Git preflight fails before any device or
    # Docker operation, but must still finish the collector.
    missing_checkout = tmp_path / "missing-mapping-checkout"
    assert not missing_checkout.exists()
    launcher_bash = argv.index("bash", argv.index("env") + 1)
    argv[launcher_bash + 4] = str(missing_checkout)
    result = _run(argv)
    assert result.returncode != 0
    assert f"cannot change to '{missing_checkout}'" in result.stderr
    assert [json.loads(line)["event"] for line in telemetry.read_text().splitlines()] == ["start", "stop"]
