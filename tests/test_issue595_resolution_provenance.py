"""Resolution provenance is recorded, and the two mechanisms stay distinct (#595).

A real 2026-08-14 run trained at 960x720 from a 1920x1440 source because the
hard ``--streaming_auto_downscale_max_pixels`` startup gate fired. Nothing
machine-readable said so, so two runs from one spec on machines with different
memory limits could differ 4x in pixel budget and look identical in every
artifact a comparison reads.

These tests pin #595's four required cases -- native under cap, native over
cap, an explicitly requested resolution, and a dynamic tier change AFTER a hard
cap -- plus the properties that make the record trustworthy: the hard gate and
the ResolutionController never share a field, the history is bounded in a way
that keeps the early transitions, the trained size is observed rather than
re-derived, and the record's arithmetic matches the gate's own divisor loop.

TEST SURFACE (#552): torch-free by construction -- it imports only
``eligsir.mapping.resolution_provenance`` and
``eligsir.mapping.resolution_controller``, neither of which imports torch
(``eligsir/mapping/__init__.py`` is an empty namespace). So this file is
COLLECTED AND RUN on ``cpu-ci``, the only surface any CI job runs, not merely
on ``training-nogpu``. Keep it that way: importing ``train_streaming`` or
``utils.comparison_report`` here would silently drop it out of CI. The
call-chain guard that does need those two modules is AST-only, in
``tests/test_issue595_report_call_chain.py``, for exactly this reason.
"""

import json

import pytest

from eligsir.mapping.resolution_controller import ResolutionController
from eligsir.mapping.resolution_provenance import (
    DEFAULT_TIER_HISTORY_LIMIT,
    NATIVE_SOURCE_INTRINSICS_JSON,
    NATIVE_SOURCE_UNKNOWN,
    OVERRIDE_AUTO_DOWNSCALE_MAX_PIXELS,
    OVERRIDE_NONE,
    OVERRIDE_REQUESTED_SCALE,
    ResolutionTierLog,
    build_resolution_provenance,
    read_native_resolution,
    resolution_report_block,
)

# The motivating run's numbers (from the issue's quoted train.log line).
NATIVE_W, NATIVE_H = 1920, 1440
CAP_PX = 1228800


def _gate_divisor(w: int, h: int, max_px: int) -> int:
    """The hard gate's divisor loop, verbatim from ``train_streaming.py``.

    Duplicated ONLY here, in a test, so the record's ``hard_resolved_*``
    arithmetic can be checked against the real rule without this module
    (which must never decide a resolution) growing a second copy of it.
    """
    if not (w > 0 and h > 0 and w * h > max_px):
        return 1
    d = 1
    while (w // (d + 1)) * (h // (d + 1)) > max_px:
        d += 1
    return d + 1


def _provenance(**overrides):
    kw = dict(
        native_width=NATIVE_W,
        native_height=NATIVE_H,
        native_source=NATIVE_SOURCE_INTRINSICS_JSON,
        native_read_error=None,
        requested_scale=1,
        hard_max_pixels=CAP_PX,
        hard_gate_disabled=False,
        hard_gate_evaluated=True,
        hard_gate_fired=False,
        resolved_scale=1,
        camera_resolution_arg=1,
    )
    kw.update(overrides)
    return build_resolution_provenance(**kw)


# --- #595's four required cases --------------------------------------------


def test_native_under_cap_records_no_override():
    """A 640x480 source under a 1.2MP cap: gate evaluated, nothing overridden."""
    p = _provenance(native_width=640, native_height=480)
    assert p.startup_override_reason == OVERRIDE_NONE
    assert p.hard_gate_evaluated is True
    assert p.hard_gate_fired is False
    assert p.hard_resolved_scale == 1
    # The report still states the size explicitly: acceptance criterion 1 is
    # "obvious what it trained at", which must hold for the un-capped run too.
    assert (p.hard_resolved_width, p.hard_resolved_height) == (640, 480)
    assert p.native_pixels == 640 * 480


def test_native_over_cap_records_the_hard_memory_override():
    """The motivating run: 1920x1440 -> 960x720, attributed to the hard cap."""
    divisor = _gate_divisor(NATIVE_W, NATIVE_H, CAP_PX)
    assert divisor == 2, "the issue's own run resolved to divisor 2"
    p = _provenance(hard_gate_fired=True, resolved_scale=divisor)

    assert p.startup_override_reason == OVERRIDE_AUTO_DOWNSCALE_MAX_PIXELS
    assert (p.native_width, p.native_height) == (1920, 1440)
    assert (p.hard_resolved_width, p.hard_resolved_height) == (960, 720)
    assert p.hard_resolved_pixels == 960 * 720 <= CAP_PX
    assert p.hard_max_pixels == CAP_PX
    # And it says so in words, naming the mechanism, not just the number.
    log = p.format_log()
    assert "HARD MEMORY/SAFETY cap" in log
    assert "did NOT train at its source's native resolution" in log


def test_explicitly_requested_resolution_is_not_attributed_to_the_gate():
    """``--streaming_resolution 4`` (or a --config/profile supplying it).

    The gate never evaluates in this case -- and crucially the reason must NOT
    be the memory cap, or a comparison would blame the wrong mechanism.
    """
    p = _provenance(requested_scale=4, resolved_scale=4, hard_gate_evaluated=False)
    assert p.startup_override_reason == OVERRIDE_REQUESTED_SCALE
    assert p.hard_gate_evaluated is False
    assert p.hard_gate_fired is False
    assert p.hard_resolved_scale == 4
    assert (p.hard_resolved_width, p.hard_resolved_height) == (480, 360)


def test_dynamic_tier_change_after_a_hard_cap_stays_distinguishable():
    """#595's fourth case, driven through the REAL ResolutionController.

    Floors must be non-zero here: CLAUDE.md's standing guidance is to pass
    ``--streaming_multires_it_v_floor 0 --streaming_multires_map_fps_floor 0``,
    under which ``floor_breached`` can never be true and this test would pass
    vacuously.
    """
    p = _provenance(hard_gate_fired=True, resolved_scale=2)
    ctrl = ResolutionController(it_v_floor=30.0, map_fps_floor=3.0, window=5, recovery_streak=2)
    log = ResolutionTierLog(it_v_floor=30.0, map_fps_floor=3.0, low_divisor=4)
    assert log.controller_enabled is True

    # Fast admissions (50 iters apart, 0.01s apart) -> both floors satisfied.
    for k in range(6):
        ctrl.record_admission(k * 50, k * 0.01)
        log.observe(k * 50, ctrl.decide(mapping_done=False), width=960, height=720)
    assert log.current_tier == "high"
    assert log.tier_change_count == 0

    # Then a throughput collapse (2 iters apart) -> the it_v floor breaks.
    for k in range(6, 14):
        ctrl.record_admission(300 + (k - 6) * 2, 0.06 + (k - 6) * 0.5)
        log.observe(300 + (k - 6) * 2, ctrl.decide(mapping_done=False), width=240, height=180)

    block = resolution_report_block(p, log)
    dyn = block["dynamic"]
    assert dyn["current_tier"] == "low"
    assert dyn["tier_change_count"] == 1
    assert dyn["initial_tier"] == "high"
    assert dyn["tier_history"][0]["from_tier"] == "high"
    assert dyn["tier_history"][0]["to_tier"] == "low"

    # The whole point of #595: the two mechanisms remain separable. The hard
    # cap's fields are untouched by the tier change, and the tier change is not
    # reported as a resolution override.
    assert block["startup_override_reason"] == OVERRIDE_AUTO_DOWNSCALE_MAX_PIXELS
    assert block["hard_resolved_scale"] == 2
    assert (block["hard_resolved_width"], block["hard_resolved_height"]) == (960, 720)
    # Non-overlapping by construction: no dynamic key leaks into the startup
    # half, so neither can be silently read as the other.
    assert set(block) & set(dyn) == set()


# --- the second, independent camera-level downscale -------------------------


def test_camera_level_resolution_arg_is_recorded_and_flagged():
    """``-r 2`` composes with the divisor; ``hard_resolved_*`` is not the truth.

    Without this the report would claim 960x720 for a run that really trained
    at 480x360 -- the exact class of silent mismatch #595 exists to end.
    """
    p = _provenance(hard_gate_fired=True, resolved_scale=2, camera_resolution_arg=2)
    assert p.camera_downscale_active is True
    assert p.camera_resolution_arg == 2
    assert "SECOND, independent camera-level downscale" in p.format_log()

    default = _provenance(hard_gate_fired=True, resolved_scale=2)
    assert default.camera_downscale_active is False
    assert "SECOND, independent" not in default.format_log()


def test_trained_size_is_observed_not_derived():
    """The dynamic block reports the size the loop measured on real cameras."""
    log = ResolutionTierLog()
    log.observe(10, "high", width=480, height=360)
    log.observe(20, "high", width=480, height=360)
    d = log.as_dict()
    assert d["observed_train_size"] == [480, 360]
    assert d["first_observed_train_size"] == [480, 360]


# --- properties of the record ----------------------------------------------


def test_controller_disabled_when_both_floors_are_zero():
    """The repo's own standing flags make the controller structurally inert;
    calling that "enabled" would be the same kind of lie #595 is about."""
    assert ResolutionTierLog(it_v_floor=0.0, map_fps_floor=0.0).controller_enabled is False
    assert ResolutionTierLog(it_v_floor=30.0, map_fps_floor=0.0).controller_enabled is True
    assert ResolutionTierLog(it_v_floor=0.0, map_fps_floor=3.0).controller_enabled is True


def test_explicit_bootstrap_schedule_is_reported_as_an_active_controller():
    """Dropping schedule provenance must not make a curriculum look native-only."""
    dynamic = ResolutionTierLog(
        it_v_floor=0.0,
        map_fps_floor=0.0,
        scheduled_low_until_iteration=350,
    ).as_dict()

    assert dynamic["controller_enabled"] is True
    assert dynamic["scheduled_low_until_iteration"] == 350


def test_tier_history_is_bounded_and_keeps_the_EARLIEST_transitions():
    """A tail-truncating buffer would discard exactly what #595 asks about
    ("did a tier change follow the hard cap"), so the first N are kept."""
    log = ResolutionTierLog(history_limit=3)
    for i in range(1, 21):
        log.observe(i, "low" if i % 2 else "high")
    d = log.as_dict()
    assert len(d["tier_history"]) == 3
    assert [e["iteration"] for e in d["tier_history"]] == [2, 3, 4]
    assert d["tier_history_truncated"] is True
    # The COUNT stays unbounded/honest even though the list is capped.
    assert d["tier_change_count"] == 19
    assert d["tier_history_limit"] == 3
    assert DEFAULT_TIER_HISTORY_LIMIT == 256


def test_first_observed_tier_is_not_counted_as_a_change():
    log = ResolutionTierLog()
    log.observe(1, "high")
    assert log.initial_tier == "high"
    assert log.as_dict()["tier_change_count"] == 0
    assert log.as_dict()["tier_history"] == []


def test_report_block_is_json_serializable():
    block = resolution_report_block(_provenance(hard_gate_fired=True, resolved_scale=2),
                                    ResolutionTierLog())
    assert json.loads(json.dumps(block)) == block


# --- reading the native resolution -----------------------------------------


def test_read_native_resolution_reads_intrinsics_json(tmp_path):
    (tmp_path / "intrinsics.json").write_text(json.dumps({"width": 1920, "height": 1440}))
    assert read_native_resolution(str(tmp_path)) == (1920, 1440, None)


def test_read_native_resolution_absent_file_is_not_an_error(tmp_path):
    """The gate prints nothing when intrinsics.json is absent; absence must
    therefore be distinguishable from a read failure."""
    assert read_native_resolution(str(tmp_path)) == (None, None, None)


def test_read_native_resolution_reports_a_parse_failure(tmp_path):
    (tmp_path / "intrinsics.json").write_text("{not json")
    w, h, err = read_native_resolution(str(tmp_path))
    assert (w, h) == (None, None)
    assert err  # the caller owns the message text; only its presence is pinned


def test_unknown_native_resolution_is_recorded_as_unknown_never_guessed():
    p = _provenance(native_width=None, native_height=None,
                    native_source=NATIVE_SOURCE_UNKNOWN, hard_gate_evaluated=False)
    assert p.native_pixels is None
    assert p.hard_resolved_width is None
    assert p.native_source == NATIVE_SOURCE_UNKNOWN
    assert "unknown" in p.format_log()


@pytest.mark.parametrize(
    "w,h,cap,expect",
    [
        (1920, 1440, 1228800, 2),   # the motivating run
        (4032, 3024, 1228800, 4),   # 12MP phone photo
        (640, 480, 1228800, 1),     # under cap: no divisor
    ],
)
def test_hard_resolved_size_matches_the_gates_own_divisor_rule(w, h, cap, expect):
    d = _gate_divisor(w, h, cap)
    assert d == expect
    p = _provenance(native_width=w, native_height=h, hard_max_pixels=cap,
                    hard_gate_fired=d > 1, resolved_scale=d)
    assert p.hard_resolved_width == w // d
    assert p.hard_resolved_height == h // d
    assert p.hard_resolved_pixels <= cap
