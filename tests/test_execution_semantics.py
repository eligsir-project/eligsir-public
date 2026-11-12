"""Tests for the resolved execution-semantics contract (issue #168).

Deliberately torch-free so they run in the lean CPU CI image without
``tests/conftest.py`` registration -- same fence as
``tests/test_experiment_validity.py`` (issues #423/#462/#478).

Three groups:
  1. axis resolution (the vocabulary), table-driven over the scenarios #168's
     "Config-resolution matrix" section asks for;
  2. the validator's refusals and warnings;
  3. ADDITIVITY: the validator produces zero hard errors for every
     ``specs/*.json`` job and every ``configs/profiles/**`` leaf that exists
     today. This is the load-bearing evidence that turning refusal on cannot
     brick the live ablator queue.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eligsir.utils import execution_semantics as es  # noqa: E402


def _args(**over):
    """A namespace carrying the real argparse defaults for every axis flag."""
    base = dict(
        source_path="/data/tum/fr3",
        execution_trainer="offline_reconstruction",
        streaming_replay=False,          # arguments/__init__.py
        online_mode=False,
        streaming_ingestion_mode="auto",
        streaming_input_fps_cap=30.0,
        streaming_frame_admission="all",
        streaming_insert_from_depth=True,
        streaming_training_mode="normal",
        # #755/#758: shared engine is the normal streaming core; the False
        # value is now an explicit compatibility/historical-control request.
        streaming_optimization_engine=True,
        streaming_max_iterations=0,
        streaming_refinement_iterations=50_000,
        streaming_camera_profile="default",
        pose_source="dataset_pose",
        pose_provenance="auto",
        iterations=30_000,
        ros2_live_bag_replay=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# 1. Axis resolution
# --------------------------------------------------------------------------- #

# (label, arg overrides, expected axis subset) -- #168's config-resolution matrix.
_MATRIX = [
    (
        "true offline dataset reconstruction",
        {},
        {
            "trainer": "offline",
            "source_mode": "materialized",
            "release_policy": "all_at_start",
            "integration_policy": "bootstrap_only",
            "budget_policy": "fixed_iterations",
            # The streaming engine flag is inert on the offline trainer until #757.
            "optimization_core": "trainer_inline_control",
        },
    ),
    (
        "causal TUM replay through the streaming mapper",
        {"execution_trainer": "causal_mapping"},
        {
            "trainer": "streaming",
            "source_mode": "causal_replay",
            "release_policy": "dataset_fps",
            "integration_policy": "incremental_rgbd",
            "budget_policy": "fixed_iterations",
            "optimization_core": "gaussian_engine",
        },
    ),
    (
        "genuine ROS2 live mapping",
        {"execution_trainer": "causal_mapping", "online_mode": True, "source_path": "ros2://live"},
        {
            "trainer": "streaming",
            "source_mode": "live",
            "release_policy": "dataset_fps",
            "budget_policy": "rolling_idle",
            "optimization_core": "gaussian_engine",
        },
    ),
    (
        "rosbag causal replay (recorded, but algorithmically causal)",
        {
            "execution_trainer": "causal_mapping",
            "online_mode": True,
            "source_path": "ros2://live",
            "ros2_live_bag_replay": True,
        },
        {"trainer": "streaming", "source_mode": "causal_replay"},
    ),
    (
        "streaming with a fixed iteration cap for benchmark reproducibility",
        {
            "execution_trainer": "causal_mapping",
            "online_mode": True,
            "streaming_max_iterations": 60_000,
            "streaming_ingestion_mode": "iter_based",
        },
        {"budget_policy": "rolling_idle_capped", "release_policy": "iter_based"},
    ),
    (
        "streaming with source exhaustion + refinement tail",
        {"execution_trainer": "causal_mapping", "online_mode": True,
         "streaming_ingestion_mode": "iter_based"},
        {"budget_policy": "rolling_idle"},
    ),
    (
        "rolling-seed experimental mode",
        {"execution_trainer": "causal_mapping", "streaming_training_mode": "rolling_seed"},
        {"integration_policy": "rolling_seed",
         "integration_policy_is_separate_trainer": True},
    ),
    (
        "submap-stitch experimental mode",
        {"execution_trainer": "causal_mapping", "streaming_training_mode": "submap_stitch"},
        {"integration_policy": "submap_stitch",
         "integration_policy_is_separate_trainer": True},
    ),
    (
        "diagnostic placement_only is NOT a separate trainer",
        {"execution_trainer": "causal_mapping", "streaming_training_mode": "placement_only"},
        {"integration_policy": "incremental_rgbd",
         "integration_policy_is_separate_trainer": False},
    ),
    (
        "explicit trainer-inline compatibility selector (#758 migration window)",
        {"execution_trainer": "causal_mapping", "streaming_optimization_engine": False},
        {"optimization_core": "trainer_inline_control"},
    ),
    (
        "gated admission policy",
        {"execution_trainer": "causal_mapping", "streaming_frame_admission": "hybrid_keyframe"},
        {"admission_policy": "hybrid_keyframe"},
    ),
]


@pytest.mark.parametrize(
    "label,over,expected", _MATRIX, ids=[m[0] for m in _MATRIX]
)
def test_axis_resolution_matrix(label, over, expected):
    resolved = es.resolve_execution_semantics(_args(**over))
    for key, want in expected.items():
        assert resolved[key] == want, f"{label}: {key}={resolved[key]!r} != {want!r}"


def test_retired_streaming_replay_does_not_select_the_trainer():
    resolved = es.resolve_execution_semantics(
        _args(execution_trainer="offline_reconstruction", streaming_replay=True)
    )
    assert resolved["trainer"] == "offline"
    assert resolved["raw_flags"]["streaming_replay"] is True


def test_root_entrypoint_is_derived_from_the_resolved_trainer():
    assert es.resolve_execution_semantics(_args(execution_trainer="offline_reconstruction"))[
        "root_entrypoint"
    ] == "train.training"
    assert es.resolve_execution_semantics(_args(execution_trainer="causal_mapping"))[
        "root_entrypoint"
    ] == "eligsir.runtime.causal_entry.run_causal_mapping"


# --------------------------------------------------------------------------- #
# 1b. Issue #701: supervision / renderer execution semantics
# --------------------------------------------------------------------------- #

def test_offline_trainer_is_always_single_view_regardless_of_the_flag():
    """train.py never reads streaming_optimization_batch_size at all -- a
    value inherited from a profile must not leak into the offline reading."""
    sem = es.resolve_execution_semantics(
        _args(streaming_replay=False, streaming_optimization_batch_size=4)
    )
    assert sem["trainer"] == "offline"
    assert sem["supervision_mode"] == es.SUPERVISION_SINGLE_VIEW
    assert sem["requested_supervision_batch_size"] == 1


def test_streaming_b1_is_single_view():
    sem = es.resolve_execution_semantics(
        _args(execution_trainer="causal_mapping", streaming_optimization_batch_size=1)
    )
    assert sem["supervision_mode"] == es.SUPERVISION_SINGLE_VIEW
    assert sem["requested_supervision_batch_size"] == 1


def test_streaming_default_batch_size_is_single_view():
    """Default namespace omits streaming_optimization_batch_size entirely --
    must resolve exactly as if it were the real argparse default of 1."""
    sem = es.resolve_execution_semantics(_args(execution_trainer="causal_mapping"))
    assert sem["supervision_mode"] == es.SUPERVISION_SINGLE_VIEW
    assert sem["requested_supervision_batch_size"] == 1


@pytest.mark.parametrize("b", [2, 3, 4])
def test_causal_b_gt_1_is_fail_closed_true_camera_batch(b):
    sem = es.resolve_execution_semantics(
        _args(execution_trainer="causal_mapping", streaming_optimization_batch_size=b)
    )
    assert sem["supervision_mode"] == es.SUPERVISION_TRUE_CAMERA_BATCH
    assert sem["requested_supervision_batch_size"] == b
    assert sem["renderer_camera_batching"] is True
    assert sem["rasterization"] == es.RASTERIZATION_VECTORIZED_CAMERA_BATCH
    assert sem["optimizer_updates_per_supervision_batch"] == 1


def test_causal_matrix_reports_true_camera_batching_for_b_gt_one():
    for _label, over, _expected in _MATRIX:
        for b in (1, 2, 4):
            merged = dict(over)
            merged["streaming_optimization_batch_size"] = b
            sem = es.resolve_execution_semantics(_args(**merged))
            expected = (
                merged.get("execution_trainer") == "causal_mapping"
                and merged.get("streaming_training_mode") != "placement_only"
                and b > 1
            )
            assert sem["renderer_camera_batching"] is expected
            assert sem["rasterization"] == (
                es.RASTERIZATION_VECTORIZED_CAMERA_BATCH
                if expected
                else es.RASTERIZATION_SEQUENTIAL_PER_VIEW
            )
            assert sem["optimizer_updates_per_supervision_batch"] == 1


def test_placement_only_gates_supervision_mode_back_to_single_view():
    """Placement-only performs no optimizer batch and therefore cannot claim C4."""
    sem = es.resolve_execution_semantics(
        _args(
            execution_trainer="causal_mapping",
            streaming_training_mode="placement_only",
            streaming_optimization_batch_size=4,
        )
    )
    assert sem["supervision_mode"] == es.SUPERVISION_SINGLE_VIEW
    assert sem["requested_supervision_batch_size"] == 4


def test_colors_only_does_not_gate_supervision_mode():
    """Colors-only freezes geometry but retains canonical true-C supervision."""
    sem = es.resolve_execution_semantics(
        _args(
            execution_trainer="causal_mapping",
            streaming_training_mode="colors_only",
            streaming_optimization_batch_size=4,
        )
    )
    assert sem["supervision_mode"] == es.SUPERVISION_TRUE_CAMERA_BATCH
    assert sem["renderer_camera_batching"] is True
    assert sem["rasterization"] == es.RASTERIZATION_VECTORIZED_CAMERA_BATCH


def test_streaming_optimization_batch_size_recorded_in_raw_flags():
    sem = es.resolve_execution_semantics(
        _args(execution_trainer="causal_mapping", streaming_optimization_batch_size=4)
    )
    assert sem["raw_flags"]["streaming_optimization_batch_size"] == 4


def test_supervision_fields_are_json_serializable_and_never_none():
    """The resolved supervision contract is always concrete and serializable."""
    block = es.build_execution_semantics_block(_args(execution_trainer="causal_mapping"))
    json.dumps(block)
    for key in (
        "supervision_mode",
        "requested_supervision_batch_size",
        "renderer_camera_batching",
        "rasterization",
        "optimizer_updates_per_supervision_batch",
    ):
        assert block[key] is not None, key


def test_resolution_is_pure_and_never_raises_on_a_bare_namespace():
    """A namespace with only source_path must still resolve to the documented
    argparse defaults -- the startup banner runs before validation."""
    resolved = es.resolve_execution_semantics(SimpleNamespace(source_path="/x"))
    assert resolved["trainer"] == "offline"
    assert resolved["optimization_core"] == es.CORE_TRAINER_INLINE_CONTROL
    assert resolved["contract_version"] == es.CONTRACT_VERSION
    assert set(es.AXIS_NAMES).issubset(resolved)


def test_evaluation_semantics_axis_is_explicitly_unrepresented():
    """Issue #168's axis 9 has no authoritative field in code. The contract must
    say so rather than invent a value (the 'intent as implementation' failure)."""
    resolved = es.resolve_execution_semantics(_args())
    assert resolved["evaluation_semantics"] is None
    assert "evaluation_semantics" in resolved["unrepresented_axes"]


def test_raw_flags_are_recorded_for_auditability():
    resolved = es.resolve_execution_semantics(_args(execution_trainer="causal_mapping"))
    raw = resolved["raw_flags"]
    assert raw["execution_trainer"] == "causal_mapping"
    assert raw["streaming_replay"] is False
    assert raw["online_mode"] is False
    assert raw["streaming_optimization_engine"] is True
    assert raw["source_is_ros2"] is False


def test_release_policy_reason_attributes_the_online_mode_default():
    """Issue #302: dataset_fps arrived via the online_mode default, not a request."""
    resolved = es.resolve_execution_semantics(
        _args(execution_trainer="causal_mapping", online_mode=True)
    )
    assert resolved["release_policy"] == "dataset_fps"
    assert resolved["release_policy_reason"] == "online_mode_default"


def test_causal_fps_cap_is_the_canonical_default_release_policy():
    """The causal runner and its report must describe the same 30 Hz clock."""
    resolved = es.resolve_execution_semantics(
        _args(
            execution_trainer="causal_mapping",
            online_mode=False,
            streaming_ingestion_mode="auto",
            streaming_input_fps_cap=30.0,
        )
    )
    assert resolved["release_policy"] == "dataset_fps"
    assert resolved["release_policy_reason"] == "causal_fps_cap_default"


def test_explicit_iter_based_overrides_causal_fps_cap():
    resolved = es.resolve_execution_semantics(
        _args(
            execution_trainer="causal_mapping",
            streaming_ingestion_mode="iter_based",
            streaming_input_fps_cap=30.0,
        )
    )
    assert resolved["release_policy"] == "iter_based"
    assert resolved["release_policy_reason"] == "explicit_flag"


def test_release_policy_reuses_the_production_resolver():
    """The contract must not fork the ingestion decision table."""
    from eligsir.mapping.scheduler.frame_scheduler import (
        resolve_ingestion_mode_with_reason,
    )

    for online in (False, True):
        for src in ("/data/tum/fr3", "ros2://live"):
            want = resolve_ingestion_mode_with_reason(src, "auto", online_mode=online)
            got = es._resolve_release_policy(src, "auto", online)
            assert got == want, (src, online, got, want)


# --------------------------------------------------------------------------- #
# 2. Validator
# --------------------------------------------------------------------------- #

def test_refuses_offline_trainer_with_a_ros2_source():
    sem = es.resolve_execution_semantics(
        _args(streaming_replay=False, source_path="ros2://live")
    )
    errors, _ = es.validate_execution_semantics(sem)
    assert any("ros2://" in e for e in errors), errors


def test_refuses_live_source_with_all_at_start_release():
    sem = {
        "trainer": "streaming",
        "source_mode": es.SOURCE_LIVE,
        "release_policy": es.RELEASE_ALL_AT_START,
        "raw_flags": {},
    }
    errors, _ = es.validate_execution_semantics(sem)
    assert any("all_at_start" in e for e in errors), errors


def test_refuses_separate_trainer_mode_on_the_offline_path():
    sem = es.resolve_execution_semantics(
        _args(streaming_replay=False, streaming_training_mode="submap_stitch")
    )
    errors, _ = es.validate_execution_semantics(sem)
    assert any("submap_stitch" in e for e in errors), errors


def test_warns_when_streaming_only_flags_are_inert_on_an_offline_run():
    """The GENERALIZED form of the offline/tum_rgbd finding: measured 2026-08-14,
    train.py reads none of these, so a non-default value silently does nothing."""
    args = _args(
        streaming_replay=False,
        streaming_ingestion_mode="iter_based",
        streaming_camera_profile="tum_kinect_v1",
    )
    sem = es.resolve_execution_semantics(args)
    errors, warnings = es.validate_execution_semantics(sem, args)
    assert errors == []
    joined = " ".join(warnings)
    assert "inert" in joined
    assert "no read on any module the offline trainer reaches" in joined
    assert "streaming_ingestion_mode" in joined
    assert "streaming_camera_profile" in joined


def test_warns_that_online_mode_is_inert_on_the_offline_trainer():
    args = _args(streaming_replay=False, online_mode=True)
    sem = es.resolve_execution_semantics(args)
    _, warnings = es.validate_execution_semantics(sem, args)
    assert any("NO effect on the offline trainer" in w for w in warnings), warnings


def test_warns_on_gated_admission_without_online_mode():
    args = _args(execution_trainer="causal_mapping", streaming_frame_admission="hybrid_keyframe")
    sem = es.resolve_execution_semantics(args)
    _, warnings = es.validate_execution_semantics(sem, args)
    assert any("GATED admission" in w for w in warnings), warnings


def test_does_not_warn_on_gated_admission_for_admission_all():
    """admission=all guarantees coverage trivially -- must NOT be flagged
    (the standing floor3 champion-stack convention)."""
    args = _args(execution_trainer="causal_mapping", streaming_frame_admission="all")
    sem = es.resolve_execution_semantics(args)
    _, warnings = es.validate_execution_semantics(sem, args)
    assert not any("GATED admission" in w for w in warnings), warnings


def test_warns_on_explicit_trainer_inline_compatibility_path():
    args = _args(execution_trainer="causal_mapping", streaming_optimization_engine=False)
    sem = es.resolve_execution_semantics(args)
    _, warnings = es.validate_execution_semantics(sem, args)
    joined = " ".join(warnings)
    assert "compatibility/control path" in joined
    assert "#755 promoted gaussian_engine" in joined


def test_offline_does_not_warn_about_the_streaming_compatibility_path():
    args = _args(streaming_replay=False)
    sem = es.resolve_execution_semantics(args)
    assert sem["optimization_core"] == es.CORE_TRAINER_INLINE_CONTROL
    _, warnings = es.validate_execution_semantics(sem, args)
    assert not any("compatibility/control path" in w for w in warnings), warnings


def test_warns_that_separate_trainer_modes_are_not_comparable_policies():
    args = _args(execution_trainer="causal_mapping", streaming_training_mode="rolling_seed")
    sem = es.resolve_execution_semantics(args)
    _, warnings = es.validate_execution_semantics(sem, args)
    assert any("separate\ntrainer" in w or "separate " in w for w in warnings), warnings


def test_default_streaming_and_default_offline_runs_are_clean():
    for over in ({}, {"execution_trainer": "causal_mapping"}):
        args = _args(**over)
        sem = es.resolve_execution_semantics(args)
        errors, warnings = es.validate_execution_semantics(sem, args)
        assert errors == [], (over, errors)
        assert warnings == [], (over, warnings)


# --------------------------------------------------------------------------- #
# 2b. Issue #753/#758: Optimization core terminology/default and compatibility
# --------------------------------------------------------------------------- #

def test_contract_version_is_4_since_issue600():
    """CONTRACT_VERSION marks every axis-vocabulary change.

    3 (#753): optimization_core ``inline_legacy`` -> ``trainer_inline_control``.
    4 (#600): axis 4's ``admission_policy`` token ``all`` changed meaning from
    unconditional hard-all to capacity-aware soft-all. A consumer must be able to
    tell the two apart from the report alone, which is exactly what this version
    number is for.
    """
    assert es.CONTRACT_VERSION == 4


def test_canonical_optimization_core_resolves_legacy_inline_legacy():
    """The old serialized value 'inline_legacy' resolves to the new canonical name."""
    assert es.canonical_optimization_core("inline_legacy") == es.CORE_TRAINER_INLINE_CONTROL


def test_canonical_optimization_core_is_idempotent():
    """Calling canonical_optimization_core() multiple times gives the same result."""
    canonical = es.canonical_optimization_core("inline_legacy")
    assert es.canonical_optimization_core(canonical) == canonical
    assert es.canonical_optimization_core(es.canonical_optimization_core(canonical)) == canonical


def test_canonical_optimization_core_passthrough_new_value():
    """New canonical values pass through canonical_optimization_core() unchanged."""
    assert es.canonical_optimization_core(es.CORE_TRAINER_INLINE_CONTROL) == es.CORE_TRAINER_INLINE_CONTROL
    assert es.canonical_optimization_core(es.CORE_GAUSSIAN_ENGINE) == es.CORE_GAUSSIAN_ENGINE


def test_new_streaming_runs_use_promoted_gaussian_engine():
    """Fresh streaming runs use #755's promoted core without an opt-in flag."""
    sem = es.resolve_execution_semantics(_args(execution_trainer="causal_mapping"))
    assert sem["optimization_core"] == es.CORE_GAUSSIAN_ENGINE
    assert "legacy" not in sem["optimization_core"]


def test_explicit_compatibility_request_uses_canonical_inline_name():
    sem = es.resolve_execution_semantics(
        _args(execution_trainer="causal_mapping", streaming_optimization_engine=False)
    )
    assert sem["optimization_core"] == es.CORE_TRAINER_INLINE_CONTROL
    assert "legacy" not in sem["optimization_core"]
    assert sem["optimization_core"] != "inline_legacy"


def test_canonical_name_no_legacy_word_anywhere_in_matrix():
    """The word 'legacy' must not appear in any resolved optimization_core across the matrix."""
    for _label, over, _expected in _MATRIX:
        sem = es.resolve_execution_semantics(_args(**over))
        assert "legacy" not in sem["optimization_core"], (
            f"{_label}: {sem['optimization_core']!r} contains 'legacy'"
        )


# --------------------------------------------------------------------------- #
# 3. Block + banner emission
# --------------------------------------------------------------------------- #

def test_block_is_json_serializable_and_carries_the_verdict():
    block = es.build_execution_semantics_block(_args(execution_trainer="causal_mapping"))
    json.dumps(block)  # must not raise
    assert block["valid"] is True
    assert block["errors"] == []
    assert block["contract_version"] == es.CONTRACT_VERSION


def test_block_marks_itself_invalid_on_an_impossible_combination():
    block = es.build_execution_semantics_block(
        _args(streaming_replay=False, source_path="ros2://live")
    )
    assert block["valid"] is False
    assert block["errors"]


def test_banner_prints_every_represented_axis():
    block = es.build_execution_semantics_block(_args(execution_trainer="causal_mapping"))
    text = es.format_execution_semantics(block)
    assert "RESOLVED EXECUTION SEMANTICS" in text
    for axis in es.AXIS_NAMES:
        assert axis in text, axis
    assert "trainer              = streaming" in text
    assert "optimization_core    = gaussian_engine" in text


def test_banner_prints_issue701_supervision_fields():
    block = es.build_execution_semantics_block(
        _args(execution_trainer="causal_mapping", streaming_optimization_batch_size=4)
    )
    text = es.format_execution_semantics(block)
    assert "supervision_mode" in text
    assert es.SUPERVISION_TRUE_CAMERA_BATCH in text
    assert "configured_max_batch_size=4; realized_B/C_in_tick_telemetry" in text
    assert "renderer_batching" in text
    assert "True" in text
    assert es.RASTERIZATION_VECTORIZED_CAMERA_BATCH in text
    assert "optimizer_updates_per_batch=1" in text


def test_report_writer_signature_accepts_the_block_kwarg():
    """The block must land in report.json via the SAME mechanism as
    git_provenance -- an explicit kwarg on write_post_training_report.

    Signature-level (not source-grep) so a comment mentioning the name cannot
    make this pass.
    """
    import inspect

    # utils/comparison_report.py does a module-scope `import torch`, so this
    # test cannot run in the torch-free CPU CI image. It is NOT caught by
    # tests/conftest.py's #518 AST scan, which only excludes files whose OWN
    # module scope imports torch -- here the dependency is transitive and
    # inside the test body, so the file collects fine and then dies with
    # ModuleNotFoundError at call time. importorskip is the form #518's
    # scanner explicitly recognizes; it skips visibly with a reason in the
    # CPU image and runs for real in the training image.
    pytest.importorskip("torch")
    from eligsir.utils import comparison_report

    params = inspect.signature(comparison_report.write_post_training_report).parameters
    assert "execution_semantics" in params
    assert params["execution_semantics"].default is None


def test_every_git_provenance_call_site_also_passes_execution_semantics():
    """Follow the git_provenance grep rather than hand-enumerating writers.

    Each of the four report writers passes ``git_provenance=`` exactly where it
    must also pass ``execution_semantics=``. The streaming path routes both
    through the shared ``_streaming_report_kwargs()`` dict, which is splatted
    into every one of its write_post_training_report calls.
    """
    for rel in (
        "train.py",
        "eligsir/mapping/integration/rolling_seed_application.py",
        "eligsir/mapping/integration/submap_stitch_application.py",
    ):
        text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        assert text.count("git_provenance=") == text.count("execution_semantics="), rel

    # #1075 task P3: the shared builder's body moved out of
    # train_streaming.py's closure into eligsir/runtime/report_kwargs.py,
    # where the captured names are explicit parameters. The two keys are still
    # emitted side by side by the same dict; only the file and the parameter
    # spelling changed.
    builder = (_REPO_ROOT / "eligsir" / "runtime" / "report_kwargs.py").read_text(
        encoding="utf-8")
    assert '"git_provenance": git_provenance_from_args(args),' in builder
    assert '"execution_semantics": execution_semantics_from_args(args),' in builder
    # ...and that dict really is splatted into the report calls.
    streaming = (_REPO_ROOT / "train_streaming.py").read_text(encoding="utf-8")
    assert streaming.count("**_streaming_report_kwargs(") >= 3


def test_report_writer_actually_persists_the_block(tmp_path):
    """End-to-end on the persistence step itself, with no torch/render needed.

    Exercises the same assignment write_post_training_report performs, proving
    the block survives json round-trip into a report.json on disk.
    """
    block = es.build_execution_semantics_block(_args(execution_trainer="causal_mapping"))
    summary = {"n_test": 1, "execution_semantics": block}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    read_back = json.loads(path.read_text(encoding="utf-8"))
    assert read_back["execution_semantics"]["trainer"] == "streaming"
    assert read_back["execution_semantics"]["optimization_core"] == "gaussian_engine"
    assert read_back["execution_semantics"]["contract_version"] == es.CONTRACT_VERSION


@pytest.mark.parametrize("removed_pose_source", ["precomputed", "ros_slam"])
def test_removed_pose_source_aliases_fail_closed(removed_pose_source):
    """Axis 8 must not resurrect removed aliases when recording semantics."""
    args = _args(
        execution_trainer="causal_mapping",
        source_path="ros2://live",
        pose_source=removed_pose_source,
    )
    args.ros2_live_pose_source = "odom"
    args.ros2_live_pose_topic = "/slam/odom"
    with pytest.raises(ValueError, match=removed_pose_source):
        es.resolve_execution_semantics(args)


def test_external_pose_provider_is_absent_for_non_ros2_sources():
    sem = es.resolve_execution_semantics(_args(execution_trainer="causal_mapping"))
    assert sem["external_pose_provider"] is None
    assert sem["pose_source"] == "dataset_pose"


def test_startup_banner_is_wired_before_trainer_dispatch():
    """train.py's __main__ must print the contract and refuse impossible combos
    BEFORE calling training(), so all four trainer paths self-identify."""
    src = (_REPO_ROOT / "train.py").read_text(encoding="utf-8")
    banner_at = src.index("format_execution_semantics(_exec_sem)")
    dispatch_at = src.rindex("training(lp.extract(args)")
    assert banner_at < dispatch_at
    assert "allow_invalid_execution_semantics" in src


def test_streaming_banner_is_wired_after_run_start_before_camera_profile():
    """Issue #701: streaming_training() must also print the banner itself,
    not rely solely on train.py's __main__ print -- the streaming-scoped
    print exists so a reader anchored on the [RUN START] line of a rerun's
    concatenated train.log (CLAUDE.md's rerun/requeue note) finds the banner
    from that same anchor, without having to search earlier in the log for
    an attempt that predates the current [RUN START]. Source-grep, not an
    import, since train_streaming.py pulls in torch at module scope."""
    src = (_REPO_ROOT / "train_streaming.py").read_text(encoding="utf-8")
    run_start_at = src.index('f"[RUN START] {_datetime.datetime.now()')
    banner_call_at = src.index(
        "format_execution_semantics(build_execution_semantics_block(args))"
    )
    # rindex, not index: "_apply_camera_profile(args)" also matches
    # `def _apply_camera_profile(args) -> None:`'s own definition, earlier in
    # the file -- the call site we care about is the LAST occurrence.
    camera_profile_at = src.rindex("    _apply_camera_profile(args)\n")
    assert run_start_at < banner_call_at < camera_profile_at
    # And the plain (unaliased) import must actually be present -- the
    # existing _execution_semantics_from_args alias is a different name bound
    # to the same function and does not prove this import exists.
    assert "    build_execution_semantics_block,\n" in src
    assert "    format_execution_semantics,\n" in src


# --------------------------------------------------------------------------- #
# 4. ADDITIVITY -- the load-bearing evidence that refusal is safe
# --------------------------------------------------------------------------- #

def _args_from_extra_args(extra: str):
    """Best-effort: apply an ablator spec's extra_args string onto the defaults.

    Only the flags this contract reads are interpreted; everything else is
    ignored. Deliberately crude -- the point is to prove no realistic spec trips
    a HARD error, not to reimplement argparse.
    """
    # Specs use both `--flag value` and `--flag=value`; normalize to the former.
    tokens: list = []
    for raw in str(extra or "").split():
        if raw.startswith("--") and "=" in raw:
            key, _, val = raw.partition("=")
            tokens.extend([key, val])
        else:
            tokens.append(raw)
    over: dict = {}
    bool_flags = {
        "--streaming_replay": ("streaming_replay", True),
        "--no-streaming_replay": ("streaming_replay", False),
        "--online_mode": ("online_mode", True),
        "--no-online_mode": ("online_mode", False),
        "--streaming_optimization_engine": ("streaming_optimization_engine", True),
        "--no-streaming_optimization_engine": ("streaming_optimization_engine", False),
        "--streaming_insert_from_depth": ("streaming_insert_from_depth", True),
        "--no-streaming_insert_from_depth": ("streaming_insert_from_depth", False),
    }
    value_flags = {
        "--execution_trainer": "execution_trainer",
        "--streaming_ingestion_mode": "streaming_ingestion_mode",
        "--streaming_frame_admission": "streaming_frame_admission",
        "--streaming_training_mode": "streaming_training_mode",
        "--streaming_camera_profile": "streaming_camera_profile",
        "--pose_source": "pose_source",
        "-s": "source_path",
        "--source_path": "source_path",
    }
    int_flags = {
        "--streaming_max_iterations": "streaming_max_iterations",
        "--iterations": "iterations",
        "--streaming_refinement_iterations": "streaming_refinement_iterations",
    }
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in bool_flags:
            key, val = bool_flags[tok]
            over[key] = val
        elif tok in value_flags and i + 1 < len(tokens):
            over[value_flags[tok]] = tokens[i + 1]
            i += 1
        elif tok in int_flags and i + 1 < len(tokens):
            try:
                over[int_flags[tok]] = int(tokens[i + 1])
            except ValueError:
                pass
            i += 1
        i += 1
    return _args(**over)


def _as_str(value) -> str:
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    return str(value or "")


def _iter_spec_jobs():
    """Yield (spec_file, arm_id, scene, effective_args) for every ablator arm.

    Real spec layout (verified 2026-08-14): a top-level ``base`` (or flat
    ``base_args``/``frozen_shared_args``) carrying ``scene`` + ``base_args``,
    plus an ``arms`` list whose entries add ``extra_args``.
    """
    for path in sorted((_REPO_ROOT / "specs").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        base = data.get("base") if isinstance(data.get("base"), dict) else {}
        job_type = str(base.get("type") or data.get("type") or "")
        shared = " ".join(
            _as_str(x)
            for x in (
                base.get("base_args"),
                data.get("base_args"),
                data.get("frozen_shared_args"),
            )
        )
        scene = _as_str(base.get("scene") or data.get("scene") or data.get("source_path"))
        arms = data.get("arms")
        if not isinstance(arms, list):
            continue
        for arm in arms:
            if not isinstance(arm, dict):
                continue
            arm_scene = _as_str(arm.get("scene")) or scene
            args = " ".join(
                x for x in (shared, _as_str(arm.get("extra_args"))) if x.strip()
            )
            yield (
                path.name,
                str(arm.get("id") or arm.get("name") or "?"),
                arm_scene,
                args,
                job_type,
            )


def test_no_existing_spec_job_trips_a_hard_error():
    """ADDITIVITY PROOF: every arm of every specs/*.json resolves without a hard
    error, so switching refusal on cannot brick the live ablator queue.

    SCOPE, stated honestly: most specs never write ``--execution_trainer`` into
    their own args -- ablator's job TYPE supplies it (a ``bag`` job selects the
    causal-mapping trainer). ``bag``-type arms are therefore
    simulated the way that script really launches them; every other arm is
    simulated at its declared flag surface plus the argparse defaults. So this
    proves "no spec's DECLARED axis combination is refused", not "every possible
    runner-injected combination is safe". The
    ``--allow_invalid_execution_semantics`` escape hatch covers the residue.
    """
    jobs = list(_iter_spec_jobs())
    assert len(jobs) > 100, f"only {len(jobs)} arms found -- proof looks vacuous"
    offenders = []
    bag_arms = 0
    for spec, name, scene, extra, job_type in jobs:
        args = _args_from_extra_args(extra)
        if scene:
            args.source_path = scene
        if job_type == "bag":
            # The canonical causal-mapping dispatch for bag jobs.
            args.source_path = "ros2://live"
            args.execution_trainer = "causal_mapping"
            bag_arms += 1
        sem = es.resolve_execution_semantics(args)
        errors, _ = es.validate_execution_semantics(sem, args)
        if errors:
            offenders.append((spec, name, job_type, errors))
    assert offenders == [], offenders
    # Guard: if the bag-type simulation silently stopped matching any arm, the
    # ros2://-source branch of this proof would go untested without saying so.
    assert bag_arms > 0, "no bag-type arms exercised the ros2:// source branch"


def test_no_profile_leaf_trips_a_hard_error():
    """Same proof for every configs/profiles/** leaf, after extends resolution."""
    from eligsir.cli import config as cfg

    leaves = [
        p for p in sorted((_REPO_ROOT / "configs" / "profiles").rglob("*.yml"))
        if p.name not in ("default.yml", "defaults.reference.yml")
    ]
    assert leaves, "no profile leaves found"
    offenders = []
    for leaf in leaves:
        merged = cfg.load_cli_config(config_files=[str(leaf)])
        args = _args(**{k: v for k, v in merged.items() if hasattr(_args(), k)})
        # Profiles carry no -s/--source_path; use a plain dataset path unless the
        # profile itself names a ros2:// source.
        args.source_path = str(merged.get("source_path", "/data/tum/fr3"))
        sem = es.resolve_execution_semantics(args)
        errors, _ = es.validate_execution_semantics(sem, args)
        if errors:
            offenders.append((leaf.name, errors))
    assert offenders == [], offenders


# --------------------------------------------------------------------------- #
# 5. The demonstrated contradiction (issue #168's own opening example)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "profile,source_path,expected",
    [
        (
            "offline_reconstruction/tum_rgbd",
            "/data/tum/fr3",
            {
                "trainer": "offline",
                "source_mode": "materialized",
                "release_policy": "all_at_start",
                "admission_policy": "all",
                "integration_policy": "bootstrap_only",
                "optimization_core": "trainer_inline_control",
                "budget_policy": "fixed_iterations",
                "pose_source": "dataset_pose",
                "root_entrypoint": "train.training",
            },
        ),
        (
            "causal_replay/tum_rgbd",
            "/data/tum/fr3",
            {
                "trainer": "streaming",
                "source_mode": "causal_replay",
                "release_policy": "dataset_fps",
                "admission_policy": "preselected",
                "integration_policy": "incremental_rgbd",
                "optimization_core": "gaussian_engine",
                "budget_policy": "fixed_iterations",
                "pose_source": "dataset_pose",
                "root_entrypoint": "eligsir.runtime.causal_entry.run_causal_mapping",
            },
        ),
        (
            "bench/r9700_streaming",
            "/data/tum/fr3",
            {
                "trainer": "streaming",
                "source_mode": "causal_replay",
                "release_policy": "dataset_fps",
                "admission_policy": "hybrid_keyframe",
                "integration_policy": "incremental_rgbd",
                "optimization_core": "gaussian_engine",
                "budget_policy": "fixed_iterations",
                "pose_source": "dataset_pose",
                "root_entrypoint": "eligsir.runtime.causal_entry.run_causal_mapping",
            },
        ),
        (
            "live/canonical_mapping",
            "ros2://live",
            {
                "trainer": "streaming",
                "source_mode": "live",
                "release_policy": "dataset_fps",
                "admission_policy": "preselected",
                "integration_policy": "incremental_rgbd",
                "optimization_core": "gaussian_engine",
                "budget_policy": "fixed_iterations",
                "pose_source": "dataset_pose",
                "root_entrypoint": "eligsir.runtime.causal_entry.run_causal_mapping",
            },
        ),
    ],
)
def test_canonical_execution_profile_resolution_matrix(profile, source_path, expected):
    """Every executable profile leaf resolves all axes and the selected loop."""
    from eligsir.cli import config as cfg

    merged = cfg.load_cli_config(run_profile=profile)
    defaults = vars(_args())
    args = _args(**{k: v for k, v in merged.items() if k in defaults})
    args.source_path = source_path
    sem = es.resolve_execution_semantics(args)
    assert {key: sem[key] for key in expected} == expected
    errors, _ = es.validate_execution_semantics(sem, args)
    assert errors == []


def test_execution_profile_matrix_covers_all_leaves_and_excludes_intent_mixins():
    from eligsir.cli import config as cfg

    assert set(cfg.known_execution_profiles()) == {
        "bench/r9700_streaming",
        "causal_replay/tum_rgbd",
        "live/canonical_mapping",
        "offline_reconstruction/tum_rgbd",
    }
    assert not any(name.startswith("run_intent/") for name in cfg.known_execution_profiles())


@pytest.mark.parametrize(
    "args,needle",
    [
        (_args(streaming_replay=False, source_path="ros2://live"), "ros2://"),
        (
            _args(streaming_replay=False, streaming_training_mode="submap_stitch"),
            "submap_stitch",
        ),
    ],
)
def test_representable_contradictions_build_an_invalid_startup_block(args, needle):
    block = es.build_execution_semantics_block(args)
    assert block["valid"] is False
    assert any(needle in error for error in block["errors"])
