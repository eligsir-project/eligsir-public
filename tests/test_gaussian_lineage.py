import torch
from eligsir.utils.gaussian_lineage import (
    BIRTH_METHOD_BOOTSTRAP,
    BIRTH_METHOD_SOURCE_DEPTH,
    BIRTH_METHOD_MCMC_GROW,
    BIRTH_METHOD_MCMC_RESEED,
    new_root_lineage_rows,
    growth_lineage_rows,
    reseed_lineage_update,
    assert_lineage_consistent,
)


def test_new_root_lineage_rows_source_depth():
    uid_slice = torch.tensor([10, 11, 12], dtype=torch.int64)
    rows = new_root_lineage_rows(uid_slice, BIRTH_METHOD_SOURCE_DEPTH)
    assert torch.equal(rows["parent_uid"], torch.tensor([-1, -1, -1], dtype=torch.int64))
    assert torch.equal(rows["root_uid"], uid_slice)
    assert torch.equal(rows["generation"], torch.zeros(3, dtype=torch.int32))
    assert torch.equal(rows["birth_method"], torch.full((3,), BIRTH_METHOD_SOURCE_DEPTH, dtype=torch.int8))
    assert torch.equal(rows["last_reseed_iter"], torch.full((3,), -1, dtype=torch.int32))


def test_new_root_lineage_rows_bootstrap_distinct_method():
    uid_slice = torch.tensor([0, 1], dtype=torch.int64)
    rows = new_root_lineage_rows(uid_slice, BIRTH_METHOD_BOOTSTRAP)
    assert torch.equal(rows["birth_method"], torch.full((2,), BIRTH_METHOD_BOOTSTRAP, dtype=torch.int8))


def test_growth_lineage_rows_inherits_root_and_increments_generation():
    # existing_gaussian_uid: each of the 3 CURRENT rows' own identity (NOT their
    # parent_uid) — parent_idxs selects which of these rows the new clones are
    # sampled from, so the clone's parent_uid must be THIS array indexed, not
    # the sampled rows' own parent_uid (that would make clones siblings, not children).
    existing_gaussian_uid = torch.tensor([100, 101, 102], dtype=torch.int64)
    existing_root_uid = torch.tensor([100, 100, 102], dtype=torch.int64)  # 101 is itself a non-root descendant
    existing_generation = torch.tensor([0, 1, 0], dtype=torch.int32)
    parent_idxs = torch.tensor([1, 2], dtype=torch.int64)  # two clones, sampled from rows at index 1 and 2

    rows = growth_lineage_rows(existing_gaussian_uid, existing_root_uid, existing_generation, parent_idxs)

    assert torch.equal(rows["parent_uid"], torch.tensor([101, 102], dtype=torch.int64))
    assert torch.equal(rows["root_uid"], torch.tensor([100, 102], dtype=torch.int64))
    assert torch.equal(rows["generation"], torch.tensor([2, 1], dtype=torch.int32))
    assert torch.equal(rows["birth_method"], torch.full((2,), BIRTH_METHOD_MCMC_GROW, dtype=torch.int8))
    assert torch.equal(rows["last_reseed_iter"], torch.full((2,), -1, dtype=torch.int32))


def test_reseed_lineage_update_sets_donor_lineage_and_iteration():
    gaussian_uid = torch.tensor([5, 6, 7, 8], dtype=torch.int64)
    root_uid = torch.tensor([5, 5, 7, 7], dtype=torch.int64)
    generation = torch.tensor([0, 3, 0, 1], dtype=torch.int32)
    dead_indices = torch.tensor([0, 3], dtype=torch.int64)   # slots 0 and 3 died
    sampled_idxs = torch.tensor([1, 2], dtype=torch.int64)   # revived from donors at 1 and 2

    rows = reseed_lineage_update(gaussian_uid, root_uid, generation, dead_indices, sampled_idxs, iteration=12300)

    assert torch.equal(rows["parent_uid"], torch.tensor([6, 7], dtype=torch.int64))
    assert torch.equal(rows["root_uid"], torch.tensor([5, 7], dtype=torch.int64))
    assert torch.equal(rows["generation"], torch.tensor([4, 1], dtype=torch.int32))
    assert torch.equal(rows["birth_method"], torch.full((2,), BIRTH_METHOD_MCMC_RESEED, dtype=torch.int8))
    assert torch.equal(rows["last_reseed_iter"], torch.full((2,), 12300, dtype=torch.int32))


def test_lineage_rows_are_device_agnostic():
    uid_slice = torch.tensor([1, 2], dtype=torch.int64)  # CPU tensor
    rows = new_root_lineage_rows(uid_slice, BIRTH_METHOD_BOOTSTRAP)
    for v in rows.values():
        assert v.device == uid_slice.device


class _FakeGaussians:
    def __init__(self, n, mismatched=None):
        self.get_xyz = torch.zeros(n, 3)
        for name in ("gaussian_uid", "parent_uid", "root_uid", "generation", "birth_method", "last_reseed_iter", "birth_iter"):
            size = mismatched if (mismatched is not None and name == "birth_iter") else n
            setattr(self, name, torch.zeros(size, dtype=torch.int64))


def test_assert_lineage_consistent_passes_when_aligned():
    g = _FakeGaussians(5)
    assert_lineage_consistent(g, context="test")  # must not raise


def test_assert_lineage_consistent_raises_on_mismatch():
    g = _FakeGaussians(5, mismatched=3)
    try:
        assert_lineage_consistent(g, context="after_growth")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "after_growth" in str(e)
        assert "birth_iter" in str(e)


def test_assert_lineage_consistent_skips_missing_buffers():
    class _Minimal:
        def __init__(self):
            self.get_xyz = torch.zeros(4, 3)
            self.gaussian_uid = torch.zeros(4, dtype=torch.int64)
            # every other lineage buffer intentionally absent
    assert_lineage_consistent(_Minimal(), context="minimal")  # must not raise
