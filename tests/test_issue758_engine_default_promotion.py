"""Regression guards for #758's GaussianOptimizationEngine default promotion."""
from argparse import ArgumentParser
import json
from pathlib import Path

from arguments import StreamingParams
from eligsir.utils.execution_semantics import (
    CORE_GAUSSIAN_ENGINE,
    CORE_TRAINER_INLINE_CONTROL,
    resolve_execution_semantics,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _parsed_streaming_args(*argv: str):
    parser = ArgumentParser()
    StreamingParams(parser)
    return parser.parse_args(list(argv))


def test_real_argparse_default_is_gaussian_engine():
    args = _parsed_streaming_args()
    assert args.streaming_optimization_engine is True


def test_boolean_optional_action_keeps_explicit_inline_rollback():
    args = _parsed_streaming_args("--no-streaming_optimization_engine")
    assert args.streaming_optimization_engine is False


def test_legacy_execution_semantics_remains_compatibility_only():
    streaming = _parsed_streaming_args("--streaming_replay")
    streaming.source_path = "/data/tum/fr3"
    offline = _parsed_streaming_args()
    offline.source_path = "/data/tum/fr3"

    assert resolve_execution_semantics(streaming)["optimization_core"] == CORE_TRAINER_INLINE_CONTROL
    assert resolve_execution_semantics(offline)["optimization_core"] == CORE_TRAINER_INLINE_CONTROL


def test_curated_profile_selects_canonical_execution_not_legacy_engine():
    text = (_REPO_ROOT / "configs/profiles/default.yml").read_text(encoding="utf-8")
    assert "execution_trainer: causal_mapping" in text
    assert "streaming_optimization_engine" not in text


def test_issue934_pending_control_pins_its_historical_inline_core():
    spec = json.loads(
        (_REPO_ROOT / "specs/issue934_stage1b_fixed_stride_control_fr3.json").read_text(
            encoding="utf-8"
        )
    )
    base_args = spec["base"]["base_args"]
    assert "--no-streaming_optimization_engine" in base_args
