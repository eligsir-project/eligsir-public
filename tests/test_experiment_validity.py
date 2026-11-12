"""Tests for utils/experiment_validity.py (issue #203, DAG lane P0-E of #202).

Torch-free by construction -- ``utils.experiment_validity`` imports only
``utils.partition_eval`` (numpy), so this file needs NO ``tests/conftest.py``
registration and runs in the lean CPU CI image (issues #423/#462/#478).
"""
from __future__ import annotations

import pytest

from eligsir.utils.experiment_validity import (
    CLASS_DISTRIBUTED_TRAJECTORY_PROXY,
    CLASS_FULL_CONTINUAL_VALIDATION,
    CLASS_SMOKE_ONLY,
    CLASS_UNDECLARED,
    CLASS_UNKNOWN,
    build_experiment_validity_block,
    class_rank,
    derive_achieved_class,
    more_conservative,
    normalize_declared_class,
)

FR3_N = 2525


# --------------------------------------------------------------------------
# declaration normalization
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("SMOKE_ONLY", CLASS_SMOKE_ONLY),
    ("smoke_only", CLASS_SMOKE_ONLY),
    ("smoke-only", CLASS_SMOKE_ONLY),
    ("DISTRIBUTED_TRAJECTORY_PROXY", CLASS_DISTRIBUTED_TRAJECTORY_PROXY),
    ("full_continual_validation", CLASS_FULL_CONTINUAL_VALIDATION),
])
def test_normalize_accepts_declarable_classes(raw, expected):
    assert normalize_declared_class(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "typo", "FULL", "UNKNOWN"])
def test_normalize_never_upgrades_unknown_input(raw):
    """A typo, an absent value, or an attempt to declare the internal
    placeholder must all land on UNDECLARED -- never on a permissive class."""
    assert normalize_declared_class(raw) == CLASS_UNDECLARED


def test_unknown_and_undeclared_are_bottom_of_the_lattice():
    assert class_rank(CLASS_UNKNOWN) == 0
    assert class_rank(CLASS_UNDECLARED) == 0
    assert class_rank(CLASS_SMOKE_ONLY) == 1
    assert class_rank(CLASS_DISTRIBUTED_TRAJECTORY_PROXY) == 2
    assert class_rank(CLASS_FULL_CONTINUAL_VALIDATION) == 3
    assert more_conservative(
        CLASS_FULL_CONTINUAL_VALIDATION, CLASS_UNKNOWN) == CLASS_UNKNOWN
    assert more_conservative(
        CLASS_SMOKE_ONLY, CLASS_FULL_CONTINUAL_VALIDATION) == CLASS_SMOKE_ONLY


# --------------------------------------------------------------------------
# derivation
# --------------------------------------------------------------------------

def test_derive_full_continual_validation():
    d = derive_achieved_class(range(0, FR3_N), FR3_N, is_streaming_run=True)
    assert d["achieved"] == CLASS_FULL_CONTINUAL_VALIDATION
    assert d["source_progress_fraction"] == 1.0
    assert d["leading_prefix_only"] is False


def test_derive_full_survives_every_8th_holdout_reservation():
    """A genuinely complete run still never TRAINS the holdout frames, so
    its realized coverage is ~87.5% at --streaming_eval_hold 8. That must
    still grade FULL (this is why FULL_COVERAGE_FRACTION_MIN < 1.0)."""
    idx = [i for i in range(FR3_N) if i % 8 != 0]
    d = derive_achieved_class(idx, FR3_N, is_streaming_run=True)
    assert d["achieved"] == CLASS_FULL_CONTINUAL_VALIDATION


def test_derive_distributed_proxy():
    """Every 20th frame across the WHOLE route: 5% of the frames, but the
    whole span and every partition occupied."""
    d = derive_achieved_class(range(0, FR3_N, 20), FR3_N, is_streaming_run=True)
    assert d["achieved"] == CLASS_DISTRIBUTED_TRAJECTORY_PROXY
    assert d["source_progress_span_fraction"] > 0.99
    assert d["occupied_region_fraction"] == 1.0
    assert d["leading_prefix_only"] is False


def test_derive_reaching_the_end_but_skipping_the_middle_is_not_a_proxy():
    """Same frame count and same span as a real proxy, but only the two
    ends of the route -- must NOT be graded a distributed proxy."""
    idx = list(range(0, 60)) + list(range(FR3_N - 60, FR3_N))
    d = derive_achieved_class(idx, FR3_N, is_streaming_run=True)
    assert d["achieved"] == CLASS_SMOKE_ONLY
    assert d["source_progress_span_fraction"] > 0.9
    assert d["occupied_region_fraction"] < 0.6


@pytest.mark.parametrize("streaming", [None, False])
def test_registry_hit_alone_is_not_enough_non_streaming_stays_unknown(streaming):
    """A registry hit does NOT prove causal source-frame ordering:
    ``scene_source`` comes from the SCENE_SOURCE env var, which launch
    scripts set for streaming and non-streaming jobs alike. A legacy
    ``train.py --eval -s /data/tum/rgbd_desk`` run loads every camera at
    once with sequential uids and would otherwise grade
    FULL_CONTINUAL_VALIDATION."""
    d = derive_achieved_class(range(0, FR3_N), FR3_N, is_streaming_run=streaming)
    assert d["achieved"] == CLASS_UNKNOWN
    assert "streaming" in d["derivation_reason"]

    block = build_experiment_validity_block(
        declared="FULL_CONTINUAL_VALIDATION",
        source_indices=range(0, FR3_N),
        source_frame_count_total=FR3_N,
        is_streaming_run=streaming,
    )
    assert block["effective"] == CLASS_UNKNOWN
    assert block["negative_conclusions_permitted"] is False
    assert block["strong_negative_conclusions_permitted"] is False


def test_derive_unknown_without_registered_total():
    d = derive_achieved_class(range(0, 200), None, is_streaming_run=True)
    assert d["achieved"] == CLASS_UNKNOWN
    assert "source_frame_count_total" in d["derivation_reason"]


def test_derive_unknown_without_indices():
    d = derive_achieved_class([], FR3_N, is_streaming_run=True)
    assert d["achieved"] == CLASS_UNKNOWN
    assert d["source_frames_selected"] == 0


def test_derive_ignores_duplicates_and_negatives():
    d = derive_achieved_class([5, 5, 5, -1, None, "7"], FR3_N, is_streaming_run=True)
    assert d["source_frames_selected"] == 2
    assert d["last_source_frame_reached"] == 7


# --------------------------------------------------------------------------
# #203's named regression guard: the historical 126/2525 ~= 5% fr3 prefix
# --------------------------------------------------------------------------

def test_fr3_126_of_2525_prefix_declared_full_is_caught():
    """#203 "Specific regression guard", verbatim: a run labelled as a
    continual-mapping/forgetting/replay experiment that covers only a small
    chronological prefix must be impossible to miss."""
    block = build_experiment_validity_block(
        declared="FULL_CONTINUAL_VALIDATION",
        source_indices=range(126),
        source_frame_count_total=FR3_N,
        is_streaming_run=True,
        source_index_provenance="train_cameras",
    )
    assert block["declared"] == CLASS_FULL_CONTINUAL_VALIDATION
    assert block["achieved"] == CLASS_SMOKE_ONLY
    assert block["effective"] == CLASS_SMOKE_ONLY
    assert block["mismatch"] is True
    assert block["prefix_guard_triggered"] is True
    assert block["leading_prefix_only"] is True
    assert block["negative_conclusions_permitted"] is False
    assert block["strong_negative_conclusions_permitted"] is False
    assert block["negative_result_status_ceiling"] == "needs_better_proxy"
    assert block["source_frames_selected"] == 126
    assert block["source_frame_count_total"] == FR3_N
    assert round(block["source_progress_fraction"], 4) == 0.0499
    assert any("VALIDITY CLASS MISMATCH" in w for w in block["warnings"])
    assert any("PREFIX GUARD" in w for w in block["warnings"])


def test_honest_smoke_declaration_is_not_a_mismatch():
    block = build_experiment_validity_block(
        declared="SMOKE_ONLY",
        source_indices=range(126),
        source_frame_count_total=FR3_N,
        is_streaming_run=True,
    )
    assert block["mismatch"] is False
    assert block["prefix_guard_triggered"] is False
    # Still forbidden from producing a negative research conclusion.
    assert block["negative_conclusions_permitted"] is False
    assert "forgetting" in block["forbidden_smoke_conclusion_topics"]


# --------------------------------------------------------------------------
# permissions never come from the declaration alone
# --------------------------------------------------------------------------

def test_declared_full_with_underivable_achieved_gets_no_permission():
    """Requirement D: absent must stay distinguishable from verified. A
    declared-FULL run on an unregistered scene must NOT inherit permission
    from its own declaration."""
    block = build_experiment_validity_block(
        declared="FULL_CONTINUAL_VALIDATION",
        source_indices=range(5000),
        source_frame_count_total=None,
        is_streaming_run=True,
    )
    assert block["achieved"] == CLASS_UNKNOWN
    assert block["effective"] == CLASS_UNKNOWN
    assert block["negative_conclusions_permitted"] is False
    assert block["negative_result_status_ceiling"] == "needs_better_proxy"
    assert block["source_frame_count_total"] is None
    assert "source_frame_count_total" in block["field_provenance"]


def test_undeclared_run_is_recorded_as_undeclared_not_permissive():
    block = build_experiment_validity_block(
        declared=None,
        source_indices=range(0, FR3_N),
        source_frame_count_total=FR3_N,
        is_streaming_run=True,
    )
    assert block["declared"] == CLASS_UNDECLARED
    assert block["achieved"] == CLASS_FULL_CONTINUAL_VALIDATION
    # Effective is the MORE CONSERVATIVE of the two -> no free promotion.
    assert block["effective"] == CLASS_UNDECLARED
    assert block["negative_conclusions_permitted"] is False
    assert block["mismatch"] is False  # under-claiming is not a mismatch
    assert any("no experiment_validity_class declared" in w
               for w in block["warnings"])


def test_typo_declaration_is_visible_not_silently_dropped():
    block = build_experiment_validity_block(
        declared="FULL_VALIDATION",
        source_indices=range(0, FR3_N),
        source_frame_count_total=FR3_N,
        is_streaming_run=True,
    )
    assert block["declared"] == CLASS_UNDECLARED
    assert block["declared_raw"] == "FULL_VALIDATION"
    assert block["declaration_rejected"] is True
    assert any("unrecognized experiment_validity_class" in w
               for w in block["warnings"])


def test_matched_full_run_permits_a_strong_negative():
    block = build_experiment_validity_block(
        declared="FULL_CONTINUAL_VALIDATION",
        source_indices=[i for i in range(FR3_N) if i % 8 != 0],
        source_frame_count_total=FR3_N,
        is_streaming_run=True,
        realized_release_count=FR3_N,
        realized_admission_count=2209,
        source_insert_count=123456,
        mapping_end_iter=40000,
        final_iteration=90000,
        optimizer_updates=90000,
        is_final_report=True,
    )
    assert block["effective"] == CLASS_FULL_CONTINUAL_VALIDATION
    assert block["negative_conclusions_permitted"] is True
    assert block["strong_negative_conclusions_permitted"] is True
    assert block["negative_result_status_ceiling"] == "full_negative"
    assert block["post_mapping_span"] == 50000
    assert block["warnings"] == []


def test_proxy_permits_only_a_provisional_negative():
    block = build_experiment_validity_block(
        declared="DISTRIBUTED_TRAJECTORY_PROXY",
        source_indices=range(0, FR3_N, 20),
        source_frame_count_total=FR3_N,
        is_streaming_run=True,
    )
    assert block["effective"] == CLASS_DISTRIBUTED_TRAJECTORY_PROXY
    assert block["negative_conclusions_permitted"] is True
    assert block["strong_negative_conclusions_permitted"] is False
    assert block["negative_result_status_ceiling"] == "provisional_negative"


# --------------------------------------------------------------------------
# mandatory-field completeness
# --------------------------------------------------------------------------

MANDATORY_FIELDS = [
    "source_frame_count_total",
    "source_frames_selected",
    "source_progress_min",
    "source_progress_max",
    "source_progress_fraction",
    "last_source_frame_reached",
    "trajectory_distance_or_proxy",
    "world_spatial_extent",
    "occupied_spatial_cells_or_regions",
    "mapping_end_iter",
    "post_mapping_span",
    "realized_release_count",
    "realized_admission_count",
    "source_insert_count",
    "optimizer_updates",
    "supervision_views",
]


def test_every_mandatory_field_is_present_even_when_unknown():
    """#203's mandatory-report-fields block. An uncomputable field is
    present-but-null WITH a reason, never omitted."""
    block = build_experiment_validity_block(declared=None)
    for name in MANDATORY_FIELDS:
        assert name in block, name
    for name in MANDATORY_FIELDS:
        if block[name] is None:
            assert name in block["field_provenance"], name


def test_block_is_json_serializable_and_records_thresholds():
    import json
    block = build_experiment_validity_block(
        declared="SMOKE_ONLY",
        source_indices=range(126),
        source_frame_count_total=FR3_N,
        is_streaming_run=True,
    )
    json.dumps(block)
    assert block["thresholds"]["proxy_span_fraction_min"] == 0.80
    assert block["issue_refs"] == ["#203", "#202"]
