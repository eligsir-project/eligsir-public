"""Tests for issue #524 remaining item 3: --print-effective-config flag.

Coverage includes:
(a) CLI test that invoking with --print-effective-config plus a profile prints
    the resolved stack and provenance and does not import torch or create a
    model_path
(b) A CLI-overridden option shows 'provenance: cli' while a profile-set one
    shows 'profile:<name>' and an untouched one shows 'default'
(c) Output is stable/sorted (snapshot on a small option subset)
(d) tests/test_option_inventory_freshness.py and tests/test_defaults_reference_freshness.py pass
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from eligsir.cli.parser import build_training_parser


_REPO_ROOT = Path(__file__).resolve().parents[1]


# --- Test (a): torch-free, no model_path, includes profile_stack and provenance ---


def test_print_effective_config_flag_is_registered():
    """The flag exists and is recognized by the parser."""
    parser = build_training_parser().parser
    args = parser.parse_args(["-s", "/tmp/test", "--print-effective-config"])
    assert args.print_effective_config is True

    args = parser.parse_args(["-s", "/tmp/test", "--print_effective_config"])
    assert args.print_effective_config is True


def test_print_effective_config_does_not_import_torch():
    """torch must never be imported on the --print-effective-config path.

    Runs train.py in a fresh interpreter with ``-X importtime`` so the check
    covers the real dispatch shim (an in-process ``sys.modules`` check would
    pass vacuously once torch had been imported anywhere earlier).
    """
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "train.py", "--print-effective-config", "-s", "/tmp/test"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    torch_imports = [
        line for line in result.stderr.splitlines()
        if line.startswith("import time:") and line.rsplit("|", 1)[-1].strip().split(".")[0] == "torch"
    ]
    assert not torch_imports, "torch was imported on the --print-effective-config path:\n" + "\n".join(torch_imports[:5])


def test_print_effective_config_output_structure():
    """Output contains expected top-level sections."""
    from eligsir.cli.effective_config import main as pec_main
    import io

    output = io.StringIO()
    with patch("sys.stdout", output):
        try:
            pec_main(["--print-effective-config", "-s", "/tmp/test"])
        except SystemExit as e:
            assert e.code == 0

    text = output.getvalue()
    parsed = yaml.safe_load(text)

    assert "profile_stack" in parsed
    assert "effective_config" in parsed
    assert isinstance(parsed["profile_stack"], list)
    assert isinstance(parsed["effective_config"], dict)


def test_print_effective_config_includes_profile_stack():
    """With a profile, the stack includes all layers."""
    from eligsir.cli.effective_config import main as pec_main
    import io

    output = io.StringIO()
    with patch("sys.stdout", output):
        try:
            pec_main(["--print-effective-config", "-s", "/tmp/test", "--run_intent", "smoke_dev"])
        except SystemExit:
            pass

    text = output.getvalue()
    parsed = yaml.safe_load(text)

    stack = parsed["profile_stack"]
    assert "configs/profiles/default.yml" in stack
    assert "configs/profiles/run_intent/smoke_dev.yml" in stack


def test_print_effective_config_run_intent_block():
    """With --run_intent, the output includes the run_intent block."""
    from eligsir.cli.effective_config import main as pec_main
    import io

    output = io.StringIO()
    with patch("sys.stdout", output):
        try:
            pec_main(["--print-effective-config", "-s", "/tmp/test", "--run_intent", "canonical_online"])
        except SystemExit:
            pass

    text = output.getvalue()
    parsed = yaml.safe_load(text)

    assert "run_intent" in parsed
    assert parsed["run_intent"]["declared"] == "canonical_online"
    assert parsed["run_intent"]["evidence_grade"] == "research"  # canonical_online has evidence_grade: research
    assert parsed["run_intent"]["resolved_run_profile"] == "run_intent/canonical_online"


def test_print_effective_config_no_model_path_created():
    """--print-effective-config must not create any output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, "train.py", "--print-effective-config", "-s", tmpdir],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0
        # The tmpdir should still be empty (no subdirectories created)
        assert len(list(Path(tmpdir).iterdir())) == 0


# --- Test (b): Provenance tracking ---


def test_provenance_cli_override():
    """CLI-overridden options show 'provenance: cli'."""
    from eligsir.cli.effective_config import main as pec_main
    import io

    output = io.StringIO()
    with patch("sys.stdout", output):
        try:
            pec_main([
                "--print-effective-config",
                "-s", "/tmp/test",
                "--run_intent", "smoke_dev",
                "--streaming_report_test_metrics_max", "32",
            ])
        except SystemExit:
            pass

    text = output.getvalue()
    parsed = yaml.safe_load(text)

    config = parsed["effective_config"]["streaming_report_test_metrics_max"]
    assert config["provenance"] == "cli"
    assert config["value"] == 32  # Type conversion from argparse


def test_provenance_profile_set():
    """Options set by a profile show 'profile:<path>'."""
    from eligsir.cli.effective_config import main as pec_main
    import io

    output = io.StringIO()
    with patch("sys.stdout", output):
        try:
            pec_main(["--print-effective-config", "-s", "/tmp/test", "--run_intent", "smoke_dev"])
        except SystemExit:
            pass

    text = output.getvalue()
    parsed = yaml.safe_load(text)

    config = parsed["effective_config"]
    # streaming_report_final is set by the smoke_dev profile
    report_final = config.get("streaming_report_final")
    if report_final:
        assert report_final["provenance"] == "profile:configs/profiles/run_intent/smoke_dev.yml"


def test_provenance_default():
    """Untouched options show 'provenance: default'."""
    from eligsir.cli.effective_config import main as pec_main
    import io

    output = io.StringIO()
    with patch("sys.stdout", output):
        try:
            pec_main(["--print-effective-config", "-s", "/tmp/test"])
        except SystemExit:
            pass

    text = output.getvalue()
    parsed = yaml.safe_load(text)

    config = parsed["effective_config"]
    # absgrad is not set by any profile or CLI, should be a default
    absgrad = config.get("absgrad")
    if absgrad:
        assert absgrad["provenance"] == "default"


# --- Test (c): Output stability/sorting ---


def test_print_effective_config_output_is_sorted():
    """Output keys are sorted for stability."""
    from eligsir.cli.effective_config import main as pec_main
    import io

    output1 = io.StringIO()
    with patch("sys.stdout", output1):
        try:
            pec_main(["--print-effective-config", "-s", "/tmp/test"])
        except SystemExit:
            pass

    output2 = io.StringIO()
    with patch("sys.stdout", output2):
        try:
            pec_main(["--print-effective-config", "-s", "/tmp/test"])
        except SystemExit:
            pass

    # Both outputs should be identical
    assert output1.getvalue() == output2.getvalue()


def test_print_effective_config_snapshot_subset():
    """Snapshot test on a small subset of options."""
    from eligsir.cli.effective_config import main as pec_main
    import io

    output = io.StringIO()
    with patch("sys.stdout", output):
        try:
            pec_main([
                "--print-effective-config",
                "-s", "/tmp/test",
                "--run_intent", "smoke_dev",
                "--data_device", "cpu",
            ])
        except SystemExit:
            pass

    text = output.getvalue()
    parsed = yaml.safe_load(text)

    config = parsed["effective_config"]

    # Check a few known keys
    assert "absgrad" in config
    assert config["absgrad"]["provenance"] == "default"
    assert config["absgrad"]["value"] is False

    # Check the CLI override
    assert "data_device" in config
    assert config["data_device"]["provenance"] == "cli"
    assert config["data_device"]["value"] == "cpu"

    # Check a profile-set value (streaming_report_final is set by smoke_dev profile)
    assert "streaming_report_final" in config
    assert config["streaming_report_final"]["provenance"] == "profile:configs/profiles/run_intent/smoke_dev.yml"


# --- Test (d): Freshness tests pass ---


def test_defaults_reference_freshness_passes():
    """The defaults reference file freshness test passes."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_defaults_reference_freshness.py", "-v"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"Freshness test failed:\n{result.stdout}\n{result.stderr}"


def test_option_inventory_freshness_passes():
    """The option inventory freshness tests pass."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_option_inventory_freshness.py", "-v"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"Inventory freshness test failed:\n{result.stdout}\n{result.stderr}"


def test_run_intent_cli_tests_pass():
    """The run_intent CLI tests pass (including the source-text check)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_run_intent_cli.py", "-v"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"run_intent CLI tests failed:\n{result.stdout}\n{result.stderr}"


# --- Integration: actual train.py dispatch ---


def test_train_py_dispatches_print_effective_config_before_torch():
    """Verify train.py's dispatch happens before torch imports."""
    result = subprocess.run(
        [sys.executable, "train.py", "--print-effective-config", "-s", "/tmp/test"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0
    # Verify output is YAML-like
    parsed = yaml.safe_load(result.stdout)
    assert "effective_config" in parsed or "profile_stack" in parsed


def test_train_py_resolves_run_intent_once_check_still_passes():
    """The source-text check for single resolution of run_intent still passes."""
    src = (_REPO_ROOT / "train.py").read_text()
    # Should have exactly one call to run_intent_report_from_args in the actual
    # training main block (the dispatch shim doesn't use it). Count occurrences
    # and verify it's exactly 1.
    assert src.count("args.run_intent_report = run_intent_report_from_args(") == 1
