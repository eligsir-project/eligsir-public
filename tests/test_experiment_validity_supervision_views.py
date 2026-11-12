"""Tests for the ``supervision_views`` field of
``utils.experiment_validity.build_experiment_validity_block`` (issue #203,
wiring the realized-cadence ``supervision_view`` clock from issue #289/#514
into this previously-always-null field).

Torch-free by construction -- imports ONLY ``utils.experiment_validity``
(itself importing only ``utils.partition_eval``, numpy), never
``utils.comparison_report`` (which pulls in torch) -- so this file needs no
``tests/conftest.py`` registration and is collectable in the lean CPU CI
image (issues #423/#462/#478), matching the sibling
``tests/test_experiment_validity.py``.

Scope note: this only exercises ``build_experiment_validity_block`` directly
with already-extracted scalars, mirroring how a caller (currently only
``utils.comparison_report.write_post_training_report``) is expected to pass
them. The extraction logic itself -- reading
``realized_cadence["clocks"]["supervision_view"]`` and
``realized_cadence["coverage"]["partial_coverage"]`` /
``resumed_from_iteration`` out of a live report call -- lives in
``utils/comparison_report.py`` (a torch-importing module) and is therefore
NOT exercised on this CPU-only surface. See the PR description for how that
gap was verified instead.

This module does NOT decide, and must not be read as deciding, #426's open
Q6 (whether the canonical work counter should be views, pixels, or optimizer
updates) -- it only wires an existing measured counter into an existing
report field.
"""
from __future__ import annotations

from eligsir.utils.experiment_validity import build_experiment_validity_block


def test_supervision_views_present_when_measured():
    """A caller with a live, non-resumed realized-cadence tracker passes a
    real int through untouched, with no field_provenance entry (the field is
    not null, so it needs no reason)."""
    block = build_experiment_validity_block(
        declared=None,
        supervision_views=4231,
    )
    assert block["supervision_views"] == 4231
    assert "supervision_views" not in block["field_provenance"]


def test_supervision_views_zero_is_not_collapsed_to_none():
    """0 supervision views is a real, meaningful measurement (tracker
    existed, run had not rendered a supervised view yet) and must be
    distinguishable from 'not measured' -- an `or None` extraction bug would
    silently collapse this to null."""
    block = build_experiment_validity_block(
        declared=None,
        supervision_views=0,
    )
    assert block["supervision_views"] == 0
    assert "supervision_views" not in block["field_provenance"]


def test_supervision_views_default_null_reason_unchanged_shape():
    """No value passed at all (offline train.py path, or any caller
    predating issue #289) -- stays null, with a reason present, exactly as
    for every other #203 mandatory field. This is the default a caller
    that predates this change still gets."""
    block = build_experiment_validity_block(declared=None)
    assert block["supervision_views"] is None
    assert "supervision_views" in block["field_provenance"]
    reason = block["field_provenance"]["supervision_views"]
    assert isinstance(reason, str) and reason


def test_supervision_views_resumed_run_withholds_number_not_reports_it():
    """A resumed run's realized-cadence tracker is in-memory only and
    restarts its counter from the resume point (see
    utils/realized_cadence.py's 'Honesty about partial coverage' docstring
    and issue #290's resume-gap class). The caller is expected to detect
    this and pass supervision_views=None with an explicit reason -- this
    field must NEVER silently report a partial, understated count as if it
    were the whole-run total."""
    partial_reason = (
        "resumed run: the realized-cadence tracker is in-memory only, so "
        "its observed count (17 supervision views) describes only the "
        "post-resume segment, not the whole run (#290's resume-gap class) "
        "-- withheld rather than reported as if it were the full-run total"
    )
    block = build_experiment_validity_block(
        declared=None,
        supervision_views=None,
        supervision_views_unavailable_reason=partial_reason,
    )
    assert block["supervision_views"] is None
    assert block["field_provenance"]["supervision_views"] == partial_reason
    assert "resumed" in block["field_provenance"]["supervision_views"]


def test_resumed_reason_distinguishable_from_default_reason():
    """The resumed-run reason and the 'never measured at all' default
    reason must not read as the same provenance story -- a corpus scan
    needs to tell 'streaming run, but resumed mid-run' apart from 'offline
    run, no clock exists' from the reason text alone."""
    default_block = build_experiment_validity_block(declared=None)
    resumed_block = build_experiment_validity_block(
        declared=None,
        supervision_views=None,
        supervision_views_unavailable_reason="resumed run: withheld (17 seen)",
    )
    default_reason = default_block["field_provenance"]["supervision_views"]
    resumed_reason = resumed_block["field_provenance"]["supervision_views"]
    assert default_reason != resumed_reason


def test_supervision_views_does_not_affect_class_derivation():
    """This field is purely reported alongside the declared/achieved/
    effective classification -- it must not change any of #203's
    permission fields, which are derived solely from source-frame coverage
    (a separate, already-tested code path in test_experiment_validity.py)."""
    with_value = build_experiment_validity_block(
        declared="SMOKE_ONLY", supervision_views=999)
    without_value = build_experiment_validity_block(declared="SMOKE_ONLY")
    for key in (
        "declared", "achieved", "effective", "mismatch",
        "negative_conclusions_permitted",
        "strong_negative_conclusions_permitted",
        "negative_result_status_ceiling",
    ):
        assert with_value[key] == without_value[key], key
