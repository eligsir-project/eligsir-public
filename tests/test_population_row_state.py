"""Pure contracts for population-wide Gaussian row-state mutations."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from eligsir.map.row_state import (  # noqa: E402
    GaussianRowStateRegistry,
    PopulationRowCreation,
    PopulationRowRelocation,
    PopulationRowTransform,
    RowStateDescriptor,
)


def _field(name: str) -> RowStateDescriptor:
    return RowStateDescriptor(
        name=name,
        family="lineage",
        owner="population_arbiter",
        shape_suffix=(),
        dtype=torch.int64,
        checkpoint=True,
        default=0,
        create="fresh",
        grow="fresh_with_parent",
        relocate="fresh",
        retire="gather_survivors",
    )


def test_registry_rejects_duplicate_row_state_names():
    registry = GaussianRowStateRegistry()
    registry.register(_field("hypothesis_uid"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_field("hypothesis_uid"))


def test_registry_accepts_exact_create_and_retire_row_coverage():
    registry = GaussianRowStateRegistry([_field("hypothesis_uid"), _field("visibility_ema")])
    transform = PopulationRowTransform(
        old_population=3,
        new_population=3,
        survivor_old_indices=torch.tensor([0, 2]),
        retired_old_indices=torch.tensor([1]),
        created_rows=(PopulationRowCreation(row_index=2, hypothesis_uid="new-3"),),
        relocated_rows=(),
        covered_fields=frozenset({"hypothesis_uid", "visibility_ema"}),
    )

    registry.validate_transform(
        transform,
        survivor_hypothesis_uids={"live-0", "live-2"},
    )


def test_relocation_requires_a_fresh_hypothesis_identity():
    with pytest.raises(ValueError, match="fresh hypothesis_uid"):
        PopulationRowTransform(
            old_population=2,
            new_population=2,
            survivor_old_indices=torch.tensor([0, 1]),
            retired_old_indices=torch.tensor([], dtype=torch.int64),
            created_rows=(),
            relocated_rows=(
                PopulationRowRelocation(
                    row_index=1,
                    old_hypothesis_uid="old-1",
                    new_hypothesis_uid="old-1",
                ),
            ),
            covered_fields=frozenset(),
        )


def test_registry_rejects_invalid_row_mapping_and_inexact_field_coverage():
    registry = GaussianRowStateRegistry([_field("hypothesis_uid"), _field("visibility_ema")])
    invalid = PopulationRowTransform(
        old_population=2,
        new_population=2,
        survivor_old_indices=torch.tensor([0, 0]),
        retired_old_indices=torch.tensor([], dtype=torch.int64),
        created_rows=(),
        relocated_rows=(),
        covered_fields=frozenset({"hypothesis_uid"}),
    )

    with pytest.raises(ValueError, match="unique"):
        registry.validate_transform(invalid, survivor_hypothesis_uids=())


@pytest.mark.parametrize(
    "covered_fields",
    [
        frozenset({"hypothesis_uid"}),
        frozenset({"hypothesis_uid", "visibility_ema", "unknown_sidecar"}),
    ],
)
def test_registry_rejects_transform_with_inexact_field_coverage(covered_fields):
    registry = GaussianRowStateRegistry([_field("hypothesis_uid"), _field("visibility_ema")])
    transform = PopulationRowTransform(
        old_population=1,
        new_population=1,
        survivor_old_indices=torch.tensor([0]),
        retired_old_indices=torch.tensor([], dtype=torch.int64),
        created_rows=(),
        relocated_rows=(),
        covered_fields=covered_fields,
    )

    with pytest.raises(ValueError, match="exactly the registered fields"):
        registry.validate_transform(transform, survivor_hypothesis_uids=())


def test_registry_requires_survivor_hypothesis_inventory():
    transform = PopulationRowTransform(
        old_population=1,
        new_population=1,
        survivor_old_indices=torch.tensor([0]),
        retired_old_indices=torch.tensor([], dtype=torch.int64),
        created_rows=(),
        relocated_rows=(),
        covered_fields=frozenset(),
    )

    with pytest.raises(TypeError, match="survivor_hypothesis_uids"):
        transform.validate()
    with pytest.raises(TypeError, match="survivor_hypothesis_uids"):
        GaussianRowStateRegistry().validate_transform(transform)


def test_transform_rejects_duplicate_created_hypothesis_uids():
    transform = PopulationRowTransform(
        old_population=1,
        new_population=3,
        survivor_old_indices=torch.tensor([0]),
        retired_old_indices=torch.tensor([], dtype=torch.int64),
        created_rows=(
            PopulationRowCreation(row_index=1, hypothesis_uid="minted"),
            PopulationRowCreation(row_index=2, hypothesis_uid="minted"),
        ),
        relocated_rows=(),
        covered_fields=frozenset(),
    )

    with pytest.raises(ValueError, match="globally unique"):
        GaussianRowStateRegistry().validate_transform(
            transform,
            survivor_hypothesis_uids={"live-0"},
        )


def test_transform_rejects_create_relocation_uid_collision_and_survivor_collision():
    transform = PopulationRowTransform(
        old_population=2,
        new_population=3,
        survivor_old_indices=torch.tensor([0]),
        retired_old_indices=torch.tensor([1]),
        created_rows=(PopulationRowCreation(row_index=2, hypothesis_uid="minted"),),
        relocated_rows=(
            PopulationRowRelocation(
                row_index=1,
                old_hypothesis_uid="retired-1",
                new_hypothesis_uid="minted",
            ),
        ),
        covered_fields=frozenset(),
    )

    with pytest.raises(ValueError, match="globally unique"):
        GaussianRowStateRegistry().validate_transform(
            transform,
            survivor_hypothesis_uids={"survivor-0"},
        )


def test_transform_rejects_minted_uid_that_collides_with_survivor_inventory():
    transform = PopulationRowTransform(
        old_population=2,
        new_population=2,
        survivor_old_indices=torch.tensor([0]),
        retired_old_indices=torch.tensor([1]),
        created_rows=(),
        relocated_rows=(
            PopulationRowRelocation(
                row_index=1,
                old_hypothesis_uid="retired-1",
                new_hypothesis_uid="survivor-0",
            ),
        ),
        covered_fields=frozenset(),
    )

    with pytest.raises(ValueError, match="surviving hypothesis"):
        GaussianRowStateRegistry().validate_transform(
            transform,
            survivor_hypothesis_uids={"survivor-0"},
        )


def test_transform_rejects_duplicate_relocated_hypothesis_uids():
    transform = PopulationRowTransform(
        old_population=3,
        new_population=3,
        survivor_old_indices=torch.tensor([0]),
        retired_old_indices=torch.tensor([1, 2]),
        created_rows=(),
        relocated_rows=(
            PopulationRowRelocation(1, "retired-1", "minted"),
            PopulationRowRelocation(2, "retired-2", "minted"),
        ),
        covered_fields=frozenset(),
    )

    with pytest.raises(ValueError, match="globally unique"):
        GaussianRowStateRegistry().validate_transform(
            transform,
            survivor_hypothesis_uids={"live-0"},
        )


def test_transform_rejects_relocation_into_a_survivor_output_row():
    transform = PopulationRowTransform(
        old_population=2,
        new_population=2,
        survivor_old_indices=torch.tensor([0]),
        retired_old_indices=torch.tensor([1]),
        created_rows=(),
        relocated_rows=(
            PopulationRowRelocation(
                row_index=0,
                old_hypothesis_uid="retired-1",
                new_hypothesis_uid="minted-1",
            ),
        ),
        covered_fields=frozenset(),
    )

    with pytest.raises(ValueError, match="survivor output rows"):
        GaussianRowStateRegistry().validate_transform(
            transform,
            survivor_hypothesis_uids={"live-0"},
        )


def test_transform_rejects_relocation_create_output_overlap():
    transform = PopulationRowTransform(
        old_population=2,
        new_population=3,
        survivor_old_indices=torch.tensor([0]),
        retired_old_indices=torch.tensor([1]),
        created_rows=(PopulationRowCreation(row_index=2, hypothesis_uid="created"),),
        relocated_rows=(PopulationRowRelocation(2, "retired-1", "relocated"),),
        covered_fields=frozenset(),
    )

    with pytest.raises(ValueError, match="both created and relocated"):
        GaussianRowStateRegistry().validate_transform(
            transform,
            survivor_hypothesis_uids={"live-0"},
        )
