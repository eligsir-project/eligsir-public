"""Tests for CLI config/profile loading."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from eligsir.cli import config as cfg
from eligsir.cli.config import (
    ConfigError,
    known_run_profiles,
    resolve_config_path,
)
from eligsir.cli.parser import parse_training_args


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return p


def test_validate_config_does_not_import_torch() -> None:
    """Config validation is a control-plane operation in the CPU image."""
    code = """
import sys
sys.modules["torch"] = None
from eligsir.cli.run_config import validate_config
assert validate_config({"cap_max": 5}) == {"cap_max": 5}
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# --- still-valid coverage carried over from the JSON-era loader ---------------


def test_known_run_profile_aliases_are_stable():
    # Removed by the #1076 ROS2 hard cut together with its YAML; the alias must
    # not outlive the file it points at (known_run_profiles() is a promise the
    # profile loads).
    assert "live/orbbec_ros2_rviz" not in known_run_profiles()
    assert "offline/tum_rgbd" not in known_run_profiles()
    assert "offline_reconstruction/tum_rgbd" in known_run_profiles()
    assert "causal_replay/tum_rgbd" in known_run_profiles()
    assert "live/canonical_mapping" in known_run_profiles()
    assert "bench/r9700_streaming" in known_run_profiles()


def test_removed_offline_profile_fails_closed_with_migration():
    with pytest.raises(
        ConfigError,
        match=r"offline/tum_rgbd.*removed.*causal_replay/tum_rgbd",
    ):
        resolve_config_path(run_profile="offline/tum_rgbd")


def test_removed_offline_profile_file_fails_closed_with_migration():
    with pytest.raises(
        ConfigError,
        match=r"offline/tum_rgbd.*removed.*causal_replay/tum_rgbd",
    ):
        cfg.load_cli_config(
            config_files=["configs/profiles/offline/tum_rgbd.yml"]
        )


def test_config_and_run_profile_are_mutually_exclusive():
    with pytest.raises(ConfigError, match="Use either --config or --run_profile"):
        resolve_config_path(config_file="configs/train.json", run_profile="offline/tum_rgbd")


def test_training_parser_exposes_run_profile_without_conflicting_with_profiler():
    args, _bundle = parse_training_args(
        [
            "-s",
            "ros2://live",
            "-m",
            "output/test",
            "--run-profile",
            "bench/r9700_streaming",
            "--profile",
            "/tmp/torch-profile.json",
        ]
    )

    assert args.run_profile == "bench/r9700_streaming"
    assert args.profile == "/tmp/torch-profile.json"


def test_training_parser_declares_the_canonical_execution_trainer_default():
    """A bare invocation is offline; causal mapping is an explicit choice."""
    args, bundle = parse_training_args(["-s", "/tmp/nonexistent"])

    assert args.execution_trainer == "offline_reconstruction"
    causal = bundle.parser.parse_args(
        ["-s", "/tmp/nonexistent", "--execution_trainer", "causal_mapping"]
    )
    assert causal.execution_trainer == "causal_mapping"


# --- Task 1: YAML loading -----------------------------------------------------


def test_load_yaml_reads_mapping(tmp_path):
    p = _write(tmp_path, "a.yml", """
        cap_max: 200000
        online_mode: true
    """)
    data = cfg.load_yaml(p)
    assert data == {"cap_max": 200000, "online_mode": True}


def test_load_yaml_rejects_non_mapping(tmp_path):
    p = _write(tmp_path, "bad.yml", "- 1\n- 2\n")
    with pytest.raises(cfg.ConfigError):
        cfg.load_yaml(p)


# --- Task 2: deep_merge -------------------------------------------------------


def test_deep_merge_later_wins_and_recurses():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    over = {"a": 2, "nested": {"y": 3, "z": 4}}
    out = cfg.deep_merge(base, over)
    assert out == {"a": 2, "nested": {"x": 1, "y": 3, "z": 4}}
    assert base == {"a": 1, "nested": {"x": 1, "y": 2}}  # inputs untouched


# --- Task 3: extends chain resolution -----------------------------------------


def test_resolve_layers_extends_chain(tmp_path):
    _write(tmp_path, "default.yml", "cap_max: 100\nonline_mode: false\n")
    _write(tmp_path, "mid.yml", "extends: [default.yml]\ncap_max: 200\n")
    leaf = _write(tmp_path, "leaf.yml", "extends: [mid.yml]\nonline_mode: true\n")
    merged = cfg.resolve_layers([leaf])
    # leaf < mid < default order, later (leaf) wins; extends key stripped
    assert merged == {"cap_max": 200, "online_mode": True}
    assert "extends" not in merged


def test_resolve_layers_detects_cycle(tmp_path):
    _write(tmp_path, "x.yml", "extends: [y.yml]\n")
    y = _write(tmp_path, "y.yml", "extends: [x.yml]\n")
    with pytest.raises(cfg.ConfigError, match="cycle"):
        cfg.resolve_layers([y])


# --- Task 4: multi --config + implicit default base ---------------------------


def test_load_cli_config_multi_and_default_base(tmp_path, monkeypatch):
    base = _write(tmp_path, "default.yml", "cap_max: 1\nopacity_reg: 0.01\n")
    a = _write(tmp_path, "a.yml", "cap_max: 2\n")
    b = _write(tmp_path, "b.yml", "cap_max: 3\n")
    monkeypatch.setattr(cfg, "_DEFAULT_BASE", base)  # implicit base
    merged = cfg.load_cli_config(config_files=[str(a), str(b)])
    assert merged["cap_max"] == 3       # last --config wins
    assert merged["opacity_reg"] == 0.01  # from implicit default base


# --- Task 5: pydantic RunConfig validation ------------------------------------


def test_validate_config_rejects_unknown_key():
    from eligsir.cli.run_config import validate_config
    with pytest.raises(Exception):  # ConfigError/ValidationError
        validate_config({"definitely_not_an_arg": 1})


def test_validate_config_passes_known_keys_only_set():
    from eligsir.cli.run_config import validate_config
    out = validate_config({"cap_max": 5})
    assert out == {"cap_max": 5}     # only the set key returned, types checked


# --- Task 6: explicit-CLI-flag detection --------------------------------------


def test_cli_explicit_dests_detects_passed_flags():
    from argparse import ArgumentParser, BooleanOptionalAction
    from eligsir.cli.run_config import cli_explicit_dests
    p = ArgumentParser()
    p.add_argument("--cap_max", type=int, default=1)
    p.add_argument("--online_mode", action=BooleanOptionalAction, default=False)
    p.add_argument("--opacity_reg", type=float, default=0.01)
    argv = ["--cap_max", "5", "--no-online_mode"]
    assert cli_explicit_dests(p, argv) == {"cap_max", "online_mode"}


# --- Task 7: integration contract used in train.py (CLI wins) -----------------


def test_apply_run_config_cli_wins(tmp_path, monkeypatch):
    # config sets cap_max=2; CLI passes --cap_max 9 -> CLI wins
    from argparse import ArgumentParser
    from eligsir.cli.run_config import cli_explicit_dests, validate_config
    from eligsir.cli import config as cfg
    base = _write(tmp_path, "default.yml", "cap_max: 1\n")
    over = _write(tmp_path, "o.yml", "cap_max: 2\n")
    monkeypatch.setattr(cfg, "_DEFAULT_BASE", base)
    p = ArgumentParser()
    p.add_argument("--cap_max", type=int, default=1)
    argv = ["--cap_max", "9"]
    args = p.parse_args(argv)
    merged = validate_config(cfg.load_cli_config(config_files=[str(over)]))
    explicit = cli_explicit_dests(p, argv)
    for k, v in merged.items():
        if k not in explicit:
            setattr(args, k, v)
    assert args.cap_max == 9       # CLI override beats config


# --- Task 8: profile migration round-trip -------------------------------------


def test_profile_roundtrip_causal_replay():
    merged = cfg.load_cli_config(run_profile="causal_replay/tum_rgbd")
    assert merged.get("execution_trainer") == "causal_mapping"
    assert merged.get("streaming_ingestion_mode") == "dataset_fps"
    assert merged.get("streaming_frame_admission") == "hybrid_keyframe"
    assert merged.get("gpu_frame_prep") is True
    assert merged.get("gpu_frame_resident") is True
    assert "streaming_replay" not in merged
    assert "extends" not in merged


def test_profile_roundtrip_offline_reconstruction():
    merged = cfg.load_cli_config(run_profile="offline_reconstruction/tum_rgbd")
    assert merged.get("streaming_replay") is False
    assert "extends" not in merged


@pytest.mark.parametrize(
    ("legacy", "trainer"),
    [(True, "causal_mapping"), (False, "offline_reconstruction")],
)
def test_historical_execution_migration_is_explicit_and_non_mutating(legacy, trainer):
    historical = {"streaming_replay": legacy, "cap_max": 100}

    migrated = cfg.migrate_historical_execution_config(historical)

    assert migrated == {"execution_trainer": trainer, "cap_max": 100}
    assert historical == {"streaming_replay": legacy, "cap_max": 100}
