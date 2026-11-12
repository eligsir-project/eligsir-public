"""Canonical trainer-selection contracts after the #1224 hard cut.

READ THIS BEFORE "FIXING" A FAILURE HERE.

``execution_trainer`` is the sole executable selection axis. Historical
``streaming_replay`` remains parser-compatible only long enough to fail with a
migration error; it cannot select a production execution route.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eligsir.cli import config as cfg  # noqa: E402


def _trainer_for(resolved: dict) -> str:
    return str(resolved.get("execution_trainer", "offline_reconstruction"))


def test_bare_invocation_defaults_to_offline_reconstruction():
    merged = cfg.load_cli_config()
    assert merged == {}
    assert _trainer_for(merged) == "offline_reconstruction"


def test_shared_default_profile_selects_causal_mapping():
    merged = cfg.load_cli_config(config_files=[str(_REPO_ROOT / "configs" / "profiles" / "default.yml")])
    assert merged.get("execution_trainer") == "causal_mapping"
    assert _trainer_for(merged) == "causal_mapping"


def test_removed_offline_named_profile_cannot_dispatch_streaming():
    with pytest.raises(cfg.ConfigError, match="causal_replay/tum_rgbd"):
        cfg.load_cli_config(run_profile="offline/tum_rgbd")


@pytest.mark.parametrize("profile", ["bench/r9700_streaming"])
def test_streaming_named_profiles_select_causal_mapping(profile):
    merged = cfg.load_cli_config(run_profile=profile)
    assert _trainer_for(merged) == "causal_mapping"


def test_arbitrary_config_file_inherits_canonical_trainer():
    merged = cfg.load_cli_config(config_files=[str(_REPO_ROOT / "configs" / "bicycle.json")])
    assert "execution_trainer" not in cfg.load_yaml(_REPO_ROOT / "configs" / "bicycle.json")
    assert _trainer_for(merged) == "causal_mapping"


def test_generated_defaults_reference_cannot_override_canonical_profile_trainer():
    snapshot = _REPO_ROOT / "configs" / "profiles" / "defaults.reference.yml"
    assert cfg.load_yaml(snapshot).get("streaming_replay") is False
    merged = cfg.load_cli_config(config_files=[str(snapshot)])
    assert merged.get("streaming_replay") is False
    assert _trainer_for(merged) == "causal_mapping"


def test_streaming_replay_remains_a_migration_input_not_a_trainer_axis():
    from eligsir.cli.parser import build_training_parser

    bundle = build_training_parser()
    args = bundle.parser.parse_args(["--streaming_replay"])
    assert args.streaming_replay is True


def test_train_py_dispatches_only_through_execution_trainer():
    src = (_REPO_ROOT / "train.py").read_text(encoding="utf-8")
    assert "if args.execution_trainer == \"causal_mapping\":" in src
    assert "from train_streaming import streaming_training" not in src


def test_online_mode_never_participates_in_trainer_dispatch():
    """CHARACTERIZATION: no dispatch condition in train.py reads ``online_mode``.

    ``online_mode`` is a streaming-internal stop/budget axis misnamed as a trainer axis
    (issue #168 section 1: "make online_mode precise or deprecate it"). Deliberately
    scoped to *dispatch conditions* rather than asserting the substring is absent from
    the file -- a mere comment mentioning online_mode is not a semantic regression, and
    the audit's conclusion only needs the dispatch-scoped claim. The stronger
    "unreachable on the offline path" finding rests on call-graph evidence recorded in
    section 1.5 of the audit doc, not on this test.
    """
    src = (_REPO_ROOT / "train.py").read_text(encoding="utf-8")
    conditions = [
        line for line in src.splitlines()
        if "online_mode" in line and line.lstrip().startswith(("if ", "elif "))
    ]
    assert conditions == [], conditions
