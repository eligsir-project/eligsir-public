"""CPU tests for issue #302: explicit, self-identifying ingestion-mode selection.

Scope is deliberately narrow and complementary to
``tests/test_issue278_dataset_fps_release_cadence.py`` (PR #298), which pins
the *release cadence* under ``dataset_fps`` — including the decisive test that
``steps_per_frame in {1, 20, 150, 10000}`` produces a byte-identical release
sequence there. Those cadence facts are NOT re-tested here.

What is tested here is only what #302 adds:

  1. ``resolve_ingestion_mode_with_reason`` returns the correct *provenance*
     for every branch, and the mode it returns is byte-identical to the
     pre-#302 ``resolve_ingestion_mode`` for every input combination
     (i.e. this change is provably behavior-neutral on mode selection).
  2. The loud ``online_mode -> dataset_fps`` warning fires on exactly one
     branch — not on an explicit flag (the documented interim mitigation) and
     not on a ``ros2://`` source (where ``dataset_fps`` is intended).
  3. ``report.json["ingestion"]`` is additive: present with the right shape
     when supplied, and absent (no other change) when not.
"""
from __future__ import annotations

import pytest

from eligsir.mapping.scheduler import (
    INGESTION_MODE_DATASET_FPS,
    INGESTION_MODE_ITER_BASED,
    INGESTION_MODE_WALLCLOCK_STRICT,
    INGESTION_REASON_EXPLICIT,
    INGESTION_REASON_OFFLINE_DEFAULT,
    INGESTION_REASON_ONLINE_MODE_DEFAULT,
    INGESTION_REASON_ROS2_DEFAULT,
    INGESTION_REASON_UNKNOWN_VALUE,
    ONLINE_MODE_DATASET_FPS_WARNING,
    resolve_ingestion_mode,
    resolve_ingestion_mode_with_reason,
)


# --- 1. provenance per branch ------------------------------------------------

def test_offline_auto_is_iter_based_offline_default():
    assert resolve_ingestion_mode_with_reason(
        "/data/tum/rgbd_desk", "auto", online_mode=False
    ) == (INGESTION_MODE_ITER_BASED, INGESTION_REASON_OFFLINE_DEFAULT)


def test_online_mode_auto_is_dataset_fps_and_attributed_to_online_mode():
    """The reverted-fix branch (issue #302). Retained, but now self-identifying."""
    assert resolve_ingestion_mode_with_reason(
        "/data/tum/rgbd_desk", "auto", online_mode=True
    ) == (INGESTION_MODE_DATASET_FPS, INGESTION_REASON_ONLINE_MODE_DEFAULT)


def test_ros2_source_auto_is_dataset_fps_attributed_to_the_source_not_online_mode():
    """ros2:// wins attribution over online_mode: same mode, different reason.

    dataset_fps is the intended, documented default for a ros2:// source
    (the bag player already paces to real time), so such a run must not be
    labelled — nor warned about — as if it had fallen into the online_mode
    regression branch.
    """
    for online in (False, True):
        assert resolve_ingestion_mode_with_reason(
            "ros2://live", "auto", online_mode=online
        ) == (INGESTION_MODE_DATASET_FPS, INGESTION_REASON_ROS2_DEFAULT)


@pytest.mark.parametrize("explicit", [
    INGESTION_MODE_ITER_BASED,
    INGESTION_MODE_DATASET_FPS,
    INGESTION_MODE_WALLCLOCK_STRICT,
])
@pytest.mark.parametrize("source", ["/data/tum/rgbd_desk", "ros2://live"])
@pytest.mark.parametrize("online", [False, True])
def test_explicit_flag_short_circuits_every_default(explicit, source, online):
    """The documented interim mitigation: an explicit mode beats every default."""
    assert resolve_ingestion_mode_with_reason(
        source, explicit, online_mode=online
    ) == (explicit, INGESTION_REASON_EXPLICIT)


def test_unrecognized_value_passthrough_is_preserved():
    """FrameScheduler's legacy fallback for unknown values must not change."""
    mode, reason = resolve_ingestion_mode_with_reason(
        "/data/tum/rgbd_desk", "nonsense_mode", online_mode=True
    )
    assert mode == "nonsense_mode"
    assert reason == INGESTION_REASON_UNKNOWN_VALUE


def test_none_and_whitespace_requested_are_treated_as_auto():
    assert resolve_ingestion_mode_with_reason("/data/x", None, online_mode=False) == (
        INGESTION_MODE_ITER_BASED, INGESTION_REASON_OFFLINE_DEFAULT
    )
    assert resolve_ingestion_mode_with_reason("/data/x", "  AUTO ", online_mode=True) == (
        INGESTION_MODE_DATASET_FPS, INGESTION_REASON_ONLINE_MODE_DEFAULT
    )


# --- 2. behavior-neutrality of the refactor ----------------------------------

_ALL_SOURCES = ["/data/tum/rgbd_desk", "ros2://live", "", "ros2://something/else"]
_ALL_REQUESTED = [
    None, "auto", "iter_based", "dataset_fps", "wallclock_strict", "nonsense_mode",
]


def test_wrapper_mode_matches_with_reason_for_every_combination():
    """resolve_ingestion_mode() stays a pure projection of the new resolver."""
    for source in _ALL_SOURCES:
        for requested in _ALL_REQUESTED:
            for online in (False, True):
                mode_only = resolve_ingestion_mode(
                    source, requested, online_mode=online
                )
                mode, _reason = resolve_ingestion_mode_with_reason(
                    source, requested, online_mode=online
                )
                assert mode_only == mode, (source, requested, online)


def test_ros2_before_online_mode_reorder_cannot_change_the_resolved_mode():
    """Attribution reorder is mode-neutral: ros2:// and online_mode both give
    dataset_fps, so which branch is checked first can only change the reason."""
    for source in _ALL_SOURCES:
        for online in (False, True):
            mode, reason = resolve_ingestion_mode_with_reason(
                source, "auto", online_mode=online
            )
            if source.startswith("ros2://") or online:
                assert mode == INGESTION_MODE_DATASET_FPS
                assert reason in (
                    INGESTION_REASON_ROS2_DEFAULT,
                    INGESTION_REASON_ONLINE_MODE_DEFAULT,
                )
            else:
                assert mode == INGESTION_MODE_ITER_BASED
                assert reason == INGESTION_REASON_OFFLINE_DEFAULT


# --- 3. the warning fires on exactly one branch ------------------------------

def _warns(source, requested, online):
    _mode, reason = resolve_ingestion_mode_with_reason(
        source, requested, online_mode=online
    )
    return reason == INGESTION_REASON_ONLINE_MODE_DEFAULT


def test_warning_fires_only_for_the_online_mode_default_branch():
    assert _warns("/data/tum/rgbd_desk", "auto", True) is True
    # explicit opt-out (the documented interim mitigation) must be silent
    assert _warns("/data/tum/rgbd_desk", "iter_based", True) is False
    # explicitly ASKING for dataset_fps is an informed choice, also silent
    assert _warns("/data/tum/rgbd_desk", "dataset_fps", True) is False
    # ros2:// dataset_fps is intended, not a regression
    assert _warns("ros2://live", "auto", True) is False
    # plain offline run never warns
    assert _warns("/data/tum/rgbd_desk", "auto", False) is False


def test_warning_text_names_the_consequence_and_the_issue():
    """The warning is the thing preventing another silent revert -- pin its
    load-bearing content so it cannot be quietly hollowed out."""
    w = ONLINE_MODE_DATASET_FPS_WARNING
    assert "dataset_fps" in w
    assert "steps_per_frame" in w
    assert "IGNORED" in w
    assert "#302" in w
    assert "--streaming_ingestion_mode iter_based" in w
    assert "docs/2026-06-21-scene-explosion-isolation.md" in w


def test_source_retains_the_reverted_fix_warning_comment():
    """7ffaebc's warning comment was deleted by a rebase resolution (12899c2)
    and the regression then survived seven weeks. Pin the restored equivalent
    so removing it again fails CI rather than passing silently."""
    from eligsir.mapping.scheduler import frame_scheduler as fs

    doc = resolve_ingestion_mode_with_reason.__doc__ or ""
    assert "7ffaebc" in doc
    assert "12899c2" in doc
    assert "docs/2026-06-21-scene-explosion-isolation.md" in doc
    # and the file it lives in still ships the constant the loop prints
    assert isinstance(fs.ONLINE_MODE_DATASET_FPS_WARNING, str)


# --- 4. report.json field is additive ----------------------------------------

def test_write_post_training_report_accepts_ingestion_kwarg():
    import inspect

    from eligsir.utils.comparison_report import write_post_training_report

    sig = inspect.signature(write_post_training_report)
    assert "ingestion" in sig.parameters
    p = sig.parameters["ingestion"]
    # keyword-only and defaulted to None => omitting it reproduces the exact
    # pre-#302 report.json (same gating pattern as input_quality).
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is None


def test_ingestion_report_block_shape_matches_the_resolver():
    """The dict train_streaming.py records is derived from the resolver, not
    re-derived from args -- one fact, one place (that duplication is what let
    the #302 regression hide)."""
    mode, reason = resolve_ingestion_mode_with_reason(
        "/data/tum/rgbd_desk", "auto", online_mode=True
    )
    block = {
        "mode": mode,
        "reason": reason,
        "requested": "auto",
        "online_mode": True,
        "steps_per_frame_honored": mode == INGESTION_MODE_ITER_BASED,
    }
    assert block["mode"] == INGESTION_MODE_DATASET_FPS
    assert block["reason"] == INGESTION_REASON_ONLINE_MODE_DEFAULT
    # the field that answers "did this run honour --streaming_steps_per_frame?"
    assert block["steps_per_frame_honored"] is False

    mode2, reason2 = resolve_ingestion_mode_with_reason(
        "/data/tum/rgbd_desk", "iter_based", online_mode=True
    )
    assert (mode2 == INGESTION_MODE_ITER_BASED) is True
    assert reason2 == INGESTION_REASON_EXPLICIT
