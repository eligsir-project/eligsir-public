import math

import pytest

from eligsir.mapping.evidence.evidence_demand import (
    EvidenceDemand,
    EvidenceDemandBatch,
    EvidenceDemandKind,
    EvidenceDemandModality,
)


def _demand(**overrides):
    values = dict(
        kind=EvidenceDemandKind.RGB_DETAIL,
        requested_modality=EvidenceDemandModality.RGB,
        target_ids=("gaussian-7",),
        map_version=1,
        topology_version=2,
        coordinate_version=3,
        schema_version=4,
        optimizer_version=5,
        surface_version=6,
        policy_version=7,
        priority=1,
        expiry_epoch=100,
    )
    values.update(overrides)
    return EvidenceDemand(**values)


def test_valid_demand_is_immutable_and_describes_scheduler_need():
    demand = _demand()

    assert demand.kind is EvidenceDemandKind.RGB_DETAIL
    assert demand.snapshot_versions == {
        "map": 1,
        "topology": 2,
        "coordinate": 3,
        "schema": 4,
        "optimizer": 5,
        "surface": 6,
        "policy": 7,
    }
    description = demand.describe()
    assert description["kind"] == "rgb_detail"
    assert description["requested_modality"] == "rgb"
    assert description["target_ids"] == ("gaussian-7",)
    with pytest.raises((AttributeError, TypeError)):
        demand.priority = 2  # type: ignore[misc]


def test_demand_requires_a_target_or_region_or_cell():
    with pytest.raises(ValueError, match="target"):
        _demand(target_ids=(), region_id=None, cell_id=None)


def test_region_only_and_cell_only_demands_are_valid():
    assert _demand(target_ids=(), region_id="region-1").region_id == "region-1"
    assert _demand(target_ids=(), cell_id="cell-1").cell_id == "cell-1"


def test_demand_rejects_unknown_modality():
    with pytest.raises((TypeError, ValueError), match="modality"):
        _demand(requested_modality="infrared")


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf, True])
def test_demand_rejects_non_positive_or_non_finite_priority(value):
    with pytest.raises(ValueError, match="priority"):
        _demand(priority=value)


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf, True])
def test_demand_rejects_non_positive_or_non_finite_expiry(value):
    with pytest.raises(ValueError, match="expiry"):
        _demand(expiry_epoch=value)


@pytest.mark.parametrize("target_ids", ["gaussian-7", ["gaussian-7"], ("",), (" ",)])
def test_demand_rejects_non_tuple_or_blank_target_identifiers(target_ids):
    with pytest.raises((TypeError, ValueError), match="target"):
        _demand(target_ids=target_ids)


@pytest.mark.parametrize("field", ["region_id", "cell_id"])
def test_demand_rejects_blank_region_or_cell_identifier(field):
    with pytest.raises(ValueError, match=field):
        _demand(target_ids=(), **{field: " "})


@pytest.mark.parametrize(
    "field",
    [
        "map_version",
        "topology_version",
        "coordinate_version",
        "schema_version",
        "optimizer_version",
        "surface_version",
        "policy_version",
    ],
)
def test_demand_rejects_invalid_snapshot_versions(field):
    with pytest.raises(ValueError, match="version"):
        _demand(**{field: -1})


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_demand_rejects_non_integer_snapshot_versions(value):
    with pytest.raises(ValueError, match="version"):
        _demand(map_version=value)


def test_batch_rejects_duplicate_demand_key():
    demand = _demand()
    with pytest.raises(ValueError, match="duplicate"):
        EvidenceDemandBatch((demand, demand), max_size=2)


def test_batch_rejects_overflow():
    with pytest.raises(ValueError, match="bound"):
        EvidenceDemandBatch((_demand(), _demand(target_ids=("gaussian-8",))), max_size=1)


def test_demand_has_no_scheduler_action_or_frame_selection_api():
    demand = _demand()
    assert not hasattr(demand, "select_frame")
    assert not hasattr(demand, "select_view")
    assert not hasattr(demand, "scheduler_action")


@pytest.mark.parametrize(
    "kind",
    list(EvidenceDemandKind),
)
def test_all_demand_kinds_are_constructible(kind):
    assert _demand(kind=kind).kind is kind
