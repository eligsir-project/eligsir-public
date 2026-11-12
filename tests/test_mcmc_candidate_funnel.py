"""Unit tests for the purely-observational MCMC candidate-survival funnel
(issue #1365, formerly issue #1372 Phase 7 -- see
utils/mcmc_candidate_funnel.py's module docstring). All synthetic/mocked --
no GPU or real training loop required.
"""

import pytest
import torch

from eligsir.utils import mcmc_candidate_funnel as funnel


class _FakeArgs:
    def __init__(self, **overrides):
        self.streaming_mcmc_candidate_funnel_telemetry = False
        self.streaming_mcmc_candidate_funnel_interval = 200
        self.streaming_young_prot_iters = 0
        self.streaming_lifecycle_enabled = False
        self.streaming_reloc_local_only = False
        self.streaming_spatial_bubble_enabled = False
        self.streaming_local_stats = False
        self.streaming_growth_parents_visible_only = False
        self.streaming_bubble_radius = 0.0
        self.streaming_bubble_guard_m = 0.0
        self.streaming_resolution_exposure_veto = False
        self.streaming_depth_lock_centers = False
        for key, value in overrides.items():
            setattr(self, key, value)


class _FakeGaussians:
    """Minimal stand-in exposing exactly the attributes the funnel's
    read-only gate predicates look at."""

    def __init__(self, n):
        self.get_xyz = torch.randn(n, 3)
        self.get_scaling = torch.full((n, 3), 0.02)


def _make_population(n=50):
    return _FakeGaussians(n)


# ---------------------------------------------------------------------------
# should_sample / funnel_enabled
# ---------------------------------------------------------------------------


def test_funnel_disabled_by_default():
    args = _FakeArgs()
    assert funnel.funnel_enabled(args) is False
    assert funnel.should_sample(args, 100) is False


def test_should_sample_respects_interval():
    args = _FakeArgs(streaming_mcmc_candidate_funnel_telemetry=True,
                      streaming_mcmc_candidate_funnel_interval=100)
    assert funnel.should_sample(args, 0) is True
    assert funnel.should_sample(args, 100) is True
    assert funnel.should_sample(args, 150) is False
    assert funnel.should_sample(args, 200) is True


# ---------------------------------------------------------------------------
# Read-only gate predicates
# ---------------------------------------------------------------------------


def test_young_protected_mask_age_only():
    n = 10
    g = _make_population(n)
    g.birth_iter = torch.zeros(n, dtype=torch.int64)
    g.birth_iter[:5] = 990  # young: born at iter 990
    args = _FakeArgs(streaming_young_prot_iters=400)
    mask = funnel.young_protected_mask(g, args, iteration=1000)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == 5
    assert bool(mask[:5].all())
    assert not bool(mask[5:].any())


def test_young_protected_mask_disabled_when_flag_zero():
    n = 10
    g = _make_population(n)
    g.birth_iter = torch.zeros(n, dtype=torch.int64)
    args = _FakeArgs(streaming_young_prot_iters=0)
    mask = funnel.young_protected_mask(g, args, iteration=1000)
    assert mask.sum().item() == 0


def test_lifecycle_mature_mask_none_when_disabled():
    g = _make_population(10)
    args = _FakeArgs(streaming_lifecycle_enabled=False)
    assert funnel.lifecycle_mature_mask(g, args) is None


def test_lifecycle_mature_mask_thresholds_at_mature():
    n = 8
    g = _make_population(n)
    g.lifecycle_state = torch.tensor([0, 1, 1, 2, 2, 3, 0, 1])
    args = _FakeArgs(streaming_lifecycle_enabled=True)
    mask = funnel.lifecycle_mature_mask(g, args)
    assert mask.tolist() == [False, False, False, True, True, True, False, False]


def test_depth_conflicted_mask_none_when_missing():
    g = _make_population(5)
    assert funnel.depth_conflicted_mask(g) is None


def test_depth_conflicted_mask_reads_buffer():
    n = 5
    g = _make_population(n)
    g.depth_conflict_count = torch.tensor([0, 1, 0, 2, 0])
    mask = funnel.depth_conflicted_mask(g)
    assert mask.tolist() == [False, True, False, True, False]


def test_underexposed_mask_none_when_feature_off():
    g = _make_population(5)
    args = _FakeArgs(streaming_resolution_exposure_veto=False)
    assert funnel.underexposed_mask(g, args) is None


def test_depth_locked_mask_none_when_disabled():
    g = _make_population(5)
    args = _FakeArgs(streaming_depth_lock_centers=False)
    assert funnel.depth_locked_mask_readonly(g, args) is None


def test_bubble_mask_none_when_disabled():
    g = _make_population(5)
    args = _FakeArgs(streaming_spatial_bubble_enabled=False, streaming_local_stats=False)
    assert funnel.bubble_survive_mask_readonly(g, args) is None


def test_seed_freeze_mask_none_when_no_controller():
    g = _make_population(5)
    args = _FakeArgs()
    assert funnel.seed_freeze_mask_readonly(g, args, iteration=0) is None


# ---------------------------------------------------------------------------
# Funnel assembly: internal consistency (monotonic survivor counts, blocked
# counts add up, intersections bounded by both operands)
# ---------------------------------------------------------------------------


def test_relocation_funnel_monotonic_and_consistent():
    n = 200
    g = _make_population(n)
    g.birth_iter = torch.randint(0, 1000, (n,), dtype=torch.int64)
    g.lifecycle_state = torch.randint(0, 4, (n,))
    g.depth_conflict_count = torch.randint(0, 2, (n,))
    args = _FakeArgs(
        streaming_mcmc_candidate_funnel_telemetry=True,
        streaming_young_prot_iters=400,
        streaming_lifecycle_enabled=True,
    )
    raw = torch.rand(n) < 0.4  # ~40% raw death candidates
    result = funnel.compute_relocation_funnel(g, args, iteration=1000, raw_candidate_mask=raw)

    counts = result["survivor_counts"]
    ordered = ["raw_candidates", "lifecycle", "young", "exposure", "depth_lock",
               "seed_freeze", "bubble", "visibility"]
    values = [counts[k] for k in ordered]
    # Each stage's survivor count must never exceed the previous stage's.
    for prev, cur in zip(values, values[1:]):
        assert cur <= prev

    assert result["final_eligible"] <= counts["raw_candidates"]
    assert 0.0 <= result["mcmc_relocation_candidate_survival_fraction"] <= 1.0

    # Gate-intersection counts must never exceed either individual gate's
    # own count.
    lifecycle_count = int(funnel.lifecycle_mature_mask(g, args).sum().item())
    young_count = int(funnel.young_protected_mask(g, args, 1000).sum().item())
    depth_conflict_count = int(funnel.depth_conflicted_mask(g).sum().item())
    inter = result["intersections"]
    assert inter["mature_and_young_protected"] <= min(lifecycle_count, young_count)
    assert inter["established_and_depth_conflicted"] <= min(lifecycle_count, depth_conflict_count)


def test_relocation_funnel_blocked_counts_partition_raw_candidates():
    n = 100
    g = _make_population(n)
    g.birth_iter = torch.zeros(n, dtype=torch.int64)
    g.birth_iter[:30] = 990  # 30 young-protected
    args = _FakeArgs(
        streaming_mcmc_candidate_funnel_telemetry=True,
        streaming_young_prot_iters=400,
    )
    raw = torch.ones(n, dtype=torch.bool)  # everyone is a raw candidate
    result = funnel.compute_relocation_funnel(g, args, iteration=1000, raw_candidate_mask=raw)
    assert result["survivor_counts"]["raw_candidates"] == n
    assert result["blocked_by_young"] == 30
    assert result["survivor_counts"]["young"] == n - 30
    # No other gate is active in this args config -> no further reduction.
    assert result["final_eligible"] == n - 30


def test_relocation_funnel_never_mutates_input_mask():
    n = 20
    g = _make_population(n)
    args = _FakeArgs(streaming_mcmc_candidate_funnel_telemetry=True)
    raw = torch.ones(n, dtype=torch.bool)
    raw_copy = raw.clone()
    funnel.compute_relocation_funnel(g, args, iteration=0, raw_candidate_mask=raw)
    assert torch.equal(raw, raw_copy)


def test_growth_funnel_monotonic_and_consistent():
    n = 150
    g = _make_population(n)
    g.visibility_ema = torch.rand(n)
    args = _FakeArgs(
        streaming_mcmc_candidate_funnel_telemetry=True,
        streaming_growth_parents_visible_only=True,
    )
    result = funnel.compute_growth_funnel(g, args, iteration=500, current_n=n)
    counts = result["survivor_counts"]
    ordered = ["n_total", "depth_lock", "bubble", "visibility", "seed_freeze", "exposure"]
    values = [counts[k] for k in ordered]
    for prev, cur in zip(values, values[1:]):
        assert cur <= prev
    assert result["final_eligible"] == counts["exposure"]
    assert 0.0 <= result["mcmc_growth_parent_survival_fraction"] <= 1.0


def test_growth_funnel_visibility_gate_blocks_low_visibility():
    n = 40
    g = _make_population(n)
    g.visibility_ema = torch.zeros(n)
    g.visibility_ema[:10] = 0.5  # only first 10 are "recently visible"
    args = _FakeArgs(
        streaming_mcmc_candidate_funnel_telemetry=True,
        streaming_growth_parents_visible_only=True,
    )
    result = funnel.compute_growth_funnel(g, args, iteration=0, current_n=n)
    assert result["survivor_counts"]["visibility"] == 10
    assert result["blocked_by_visibility"] == n - 10
    assert result["final_eligible"] == 10


def test_growth_funnel_visibility_gate_noop_when_flag_off():
    n = 40
    g = _make_population(n)
    g.visibility_ema = torch.zeros(n)
    args = _FakeArgs(streaming_growth_parents_visible_only=False)
    result = funnel.compute_growth_funnel(g, args, iteration=0, current_n=n)
    assert result["blocked_by_visibility"] == 0
    assert result["survivor_counts"]["visibility"] == n


# ---------------------------------------------------------------------------
# History / TensorBoard logging
# ---------------------------------------------------------------------------


def test_record_and_summarize_history():
    funnel.reset_funnel_history()
    assert funnel.summarize_history() is None
    n = 30
    g = _make_population(n)
    args = _FakeArgs(streaming_mcmc_candidate_funnel_telemetry=True)
    raw = torch.ones(n, dtype=torch.bool)
    reloc = funnel.compute_relocation_funnel(g, args, iteration=0, raw_candidate_mask=raw)
    funnel.record_sample(reloc, None)
    growth = funnel.compute_growth_funnel(g, args, iteration=0, current_n=n)
    funnel.record_sample(None, growth)

    summary = funnel.summarize_history()
    assert summary["n_samples"] == 2
    assert summary["mean_relocation_candidate_survival_fraction"] == (
        reloc["mcmc_relocation_candidate_survival_fraction"]
    )
    assert summary["mean_growth_parent_survival_fraction"] == (
        growth["mcmc_growth_parent_survival_fraction"]
    )
    funnel.reset_funnel_history()
    assert funnel.get_funnel_history() == []


class _FakeTBWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, step):
        self.scalars[name] = value


def test_log_to_tensorboard_emits_expected_scalars():
    n = 20
    g = _make_population(n)
    args = _FakeArgs(streaming_mcmc_candidate_funnel_telemetry=True)
    raw = torch.ones(n, dtype=torch.bool)
    reloc = funnel.compute_relocation_funnel(g, args, iteration=0, raw_candidate_mask=raw)
    growth = funnel.compute_growth_funnel(g, args, iteration=0, current_n=n)
    tb = _FakeTBWriter()
    funnel.log_to_tensorboard(tb, 42, reloc, growth)
    assert "streaming/mcmc_funnel/relocation_candidate_survival_fraction" in tb.scalars
    assert "streaming/mcmc_funnel/growth_parent_survival_fraction" in tb.scalars
    assert "streaming/mcmc_funnel/reloc_blocked_by_lifecycle" in tb.scalars
    assert "streaming/mcmc_funnel/growth_blocked_by_depth_lock" in tb.scalars


def test_log_to_tensorboard_noop_with_none_writer():
    # Must not raise.
    funnel.log_to_tensorboard(None, 0, None, None)


def test_compact_snapshot_ingestion_accepts_async_device_summary_without_tensors():
    funnel.reset_funnel_history()
    relocation = {
        "iteration": 12,
        "n_total": 100,
        "survivor_counts": {
            "raw_candidates": 40,
            "lifecycle": 30,
            "young": 25,
            "exposure": 20,
            "depth_lock": 20,
            "seed_freeze": 18,
            "bubble": 18,
            "visibility": 16,
        },
        "final_eligible": 16,
        "intersections": {
            "mature_and_young_protected": 5,
            "mature_and_underexposed": 4,
            "young_protected_and_underexposed": 3,
            "established_and_depth_conflicted": 2,
        },
    }
    growth = {
        "iteration": 12,
        "n_total": 100,
        "survivor_counts": {
            "n_total": 100,
            "depth_lock": 90,
            "bubble": 90,
            "visibility": 70,
            "seed_freeze": 65,
            "exposure": 60,
        },
        "final_eligible": 60,
    }

    funnel.record_compact_sample(relocation, growth)
    sample = funnel.get_funnel_history()[0]
    assert sample["relocation"]["mcmc_relocation_candidate_survival_fraction"] == 0.4
    assert sample["growth"]["mcmc_growth_parent_survival_fraction"] == 0.6
    assert sample["relocation"]["intersections"]["mature_and_young_protected"] == 5


def test_compact_snapshot_rejects_non_monotonic_or_impossible_counts():
    bad = {
        "iteration": 1,
        "n_total": 10,
        "survivor_counts": {"raw_candidates": 2, "lifecycle": 3},
        "final_eligible": 3,
    }
    with pytest.raises(ValueError, match="monotonic"):
        funnel.record_compact_sample(bad, None)

    impossible = {
        "iteration": 1,
        "n_total": 10,
        "survivor_counts": {"raw_candidates": 2, "lifecycle": 1},
        "final_eligible": 3,
    }
    with pytest.raises(ValueError, match="final_eligible"):
        funnel.record_compact_sample(impossible, None)


def test_compact_snapshot_rejects_tensor_and_proxy_scalar_coercion():
    base = {
        "iteration": 1,
        "n_total": 10,
        "survivor_counts": {"raw_candidates": 2, "lifecycle": 1},
        "final_eligible": 1,
    }
    tensor_value = dict(base, n_total=torch.tensor(10))
    with pytest.raises(TypeError, match="native host integer"):
        funnel.record_compact_sample(tensor_value, None)

    class IntProxy:
        def __int__(self):
            return 10

    proxy_value = dict(base, n_total=IntProxy())
    with pytest.raises(TypeError, match="native host integer"):
        funnel.record_compact_sample(proxy_value, None)


def test_compact_snapshot_rejects_stage_outside_population():
    too_many_raw = {
        "iteration": 1,
        "n_total": 10,
        "survivor_counts": {"raw_candidates": 11, "lifecycle": 11},
        "final_eligible": 11,
    }
    with pytest.raises(ValueError, match="population"):
        funnel.record_compact_sample(too_many_raw, None)

    bad_growth = {
        "iteration": 1,
        "n_total": 10,
        "survivor_counts": {"n_total": 9, "depth_lock": 8},
        "final_eligible": 8,
    }
    with pytest.raises(ValueError, match="n_total"):
        funnel.record_compact_sample(None, bad_growth)

    late_stage_overflow = {
        "iteration": 1,
        "n_total": 10,
        "survivor_counts": {"raw_candidates": 10, "lifecycle": 10, "young": 11},
        "final_eligible": 10,
    }
    with pytest.raises(ValueError, match="population"):
        funnel.record_compact_sample(late_stage_overflow, None)

    oversized_intersection = {
        "iteration": 1,
        "n_total": 10,
        "survivor_counts": {"raw_candidates": 2, "lifecycle": 1},
        "final_eligible": 1,
        "intersections": {"mature_and_young_protected": 11},
    }
    with pytest.raises(ValueError, match="intersection"):
        funnel.record_compact_sample(oversized_intersection, None)


def test_compact_history_retention_is_bounded():
    funnel.reset_funnel_history()
    sample = {
        "iteration": 1,
        "n_total": 1,
        "survivor_counts": {"n_total": 1},
        "final_eligible": 1,
    }
    for iteration in range(funnel.MAX_FUNNEL_HISTORY + 7):
        funnel.record_compact_sample(dict(sample, iteration=iteration), None)
    history = funnel.get_funnel_history()
    assert len(history) == funnel.MAX_FUNNEL_HISTORY
    assert history[0]["growth"] is None
    assert history[0]["relocation"]["iteration"] == 7
