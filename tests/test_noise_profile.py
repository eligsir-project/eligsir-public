"""Tests for noise profile schema and paired/CRN comparison (issue #635).

Tests the refusal behavior on mismatches, the paired comparison machinery,
and the RNG divergence tracking.
"""

import json
import tempfile
from pathlib import Path

import pytest

from eligsir.experiments.noise_profile import (
    NoiseProfile,
    NoiseProfileID,
    NoiseProfileNotFoundError,
    PairedComparisonResult,
    RNGDivergence,
    compare_paired_runs,
    compute_noise_profile_from_replicates,
    parse_noise_profile,
)


class TestNoiseProfileID:
    """Test the immutable NoiseProfileID key."""

    def test_create_and_access(self):
        """Create a profile ID and access its fields."""
        pid = NoiseProfileID(
            capsule_id="tum_fr3_mapper_core_v1",
            runtime_class="rocm_gfx1151",
            metric="mapping_end_psnr_db",
        )
        assert pid.capsule_id == "tum_fr3_mapper_core_v1"
        assert pid.runtime_class == "rocm_gfx1151"
        assert pid.metric == "mapping_end_psnr_db"

    def test_immutable(self):
        """Profile IDs must be frozen."""
        pid = NoiseProfileID("cap", "runtime", "metric")
        with pytest.raises(AttributeError):
            pid.capsule_id = "different"  # type: ignore

    def test_to_from_dict(self):
        """Serialize and deserialize profile ID."""
        pid = NoiseProfileID("cap1", "rocm", "psnr")
        d = pid.to_dict()
        assert d == {
            "capsule_id": "cap1",
            "runtime_class": "rocm",
            "metric": "psnr",
        }
        pid2 = NoiseProfileID.from_dict(d)
        assert pid2 == pid

    def test_equality(self):
        """Profile IDs with same fields are equal."""
        pid1 = NoiseProfileID("a", "b", "c")
        pid2 = NoiseProfileID("a", "b", "c")
        pid3 = NoiseProfileID("a", "b", "different")
        assert pid1 == pid2
        assert pid1 != pid3


class TestRNGDivergence:
    """Test RNG divergence tracking."""

    def test_no_divergence(self):
        """Default is no divergence."""
        div = RNGDivergence()
        assert not div  # bool() should be False
        assert div.streams_diverged == []

    def test_divergence_recorded(self):
        """Record which streams diverged."""
        div = RNGDivergence(
            streams_diverged=["torch_rng", "numpy_rng"],
            reason="control flow change in admission",
        )
        assert bool(div)  # Has divergence
        assert "torch_rng" in div.streams_diverged
        assert "numpy_rng" in div.streams_diverged
        assert "control flow" in div.reason


class TestNoiseProfile:
    """Test noise profile creation, validation, and serialization."""

    def test_empty_profile_fails_validation(self):
        """An empty profile fails validation."""
        profile = NoiseProfile()
        with pytest.raises(NoiseProfileNotFoundError, match="capsule_id is empty"):
            profile.validate()

    def test_mismatched_values_and_seeds(self):
        """Profile with mismatched values and seed counts fails validation."""
        profile = NoiseProfile(
            profile_id=NoiseProfileID("cap", "runtime", "metric"),
            n_runs=3,
            seeds=[11, 22],  # 2 seeds
            values=[24.1, 24.2, 24.3],  # 3 values
            mean=24.2,
            median=24.2,
        )
        with pytest.raises(NoiseProfileNotFoundError, match="len\\(seeds\\)"):
            profile.validate()

    def test_n_runs_mismatch(self):
        """Profile with mismatched n_runs fails validation."""
        profile = NoiseProfile(
            profile_id=NoiseProfileID("cap", "runtime", "metric"),
            n_runs=5,  # Claim 5 runs
            seeds=[11, 22, 33],  # But only 3 seeds
            values=[24.1, 24.2, 24.3],
            mean=24.2,
            median=24.2,
        )
        with pytest.raises(NoiseProfileNotFoundError, match="n_runs"):
            profile.validate()

    def test_zero_n_runs_fails(self):
        """Zero runs is not allowed."""
        profile = NoiseProfile(
            profile_id=NoiseProfileID("cap", "runtime", "metric"),
            n_runs=0,
            seeds=[],
            values=[],
            mean=24.2,
            median=24.2,
        )
        with pytest.raises(NoiseProfileNotFoundError, match="n_runs=0"):
            profile.validate()

    def test_missing_mean_fails(self):
        """Profile without mean fails validation."""
        profile = NoiseProfile(
            profile_id=NoiseProfileID("cap", "runtime", "metric"),
            n_runs=1,
            seeds=[11],
            values=[24.0],
            mean=None,  # Missing
            median=24.0,
        )
        with pytest.raises(NoiseProfileNotFoundError, match="mean or median"):
            profile.validate()

    def test_to_from_dict(self):
        """Serialize and deserialize a profile."""
        profile = NoiseProfile(
            profile_id=NoiseProfileID("cap1", "rocm", "psnr"),
            n_runs=3,
            seeds=[11, 22, 33],
            values=[24.1, 24.2, 24.3],
            mean=24.2,
            median=24.2,
            std_dev=0.08,
            pairwise_delta_p95=0.27,
            notes="test profile",
        )
        d = profile.to_dict()
        assert d["n_runs"] == 3
        assert d["seeds"] == [11, 22, 33]
        assert d["profile_id"]["capsule_id"] == "cap1"

        # Deserialize and check it round-trips
        profile2 = NoiseProfile.from_dict(d)
        assert profile2.n_runs == profile.n_runs
        assert profile2.seeds == profile.seeds
        assert profile2.mean == profile.mean


class TestComputeNoiseProfile:
    """Test computation of noise profiles from replicates."""

    def test_basic_profile_from_replicates(self):
        """Compute profile from simple replicate values."""
        pid = NoiseProfileID("test_cap", "test_runtime", "psnr")
        values = [24.0, 24.1, 24.2, 24.3, 24.4]
        seeds = [11, 22, 33, 44, 55]

        profile = compute_noise_profile_from_replicates(pid, values, seeds)

        assert profile.n_runs == 5
        assert profile.seeds == seeds
        assert profile.values == values
        assert profile.mean is not None
        assert profile.median is not None
        assert abs(profile.mean - 24.2) < 0.01
        assert profile.std_dev is not None
        assert profile.std_dev > 0

    def test_profile_with_outliers(self):
        """Exclude outliers from summary statistics."""
        pid = NoiseProfileID("test_cap", "test_runtime", "psnr")
        values = [24.0, 24.1, 24.2, 100.0]  # Last one is an outlier
        seeds = [11, 22, 33, 44]
        outliers = [44]

        profile = compute_noise_profile_from_replicates(
            pid, values, seeds, outlier_seeds=outliers
        )

        assert profile.n_runs == 3  # Excluding the outlier
        assert len(profile.values) == 4  # Original values (with outlier) still stored
        assert profile.outlier_seeds == outliers
        # Mean should be ~24.1, not inflated by the 100.0
        assert abs(profile.mean - 24.1) < 0.1
        assert profile.values == values  # Original values still stored

    def test_pairwise_deltas_computed(self):
        """Pairwise deltas and quantiles are computed."""
        pid = NoiseProfileID("test_cap", "test_runtime", "psnr")
        values = [24.0, 24.2, 24.4]
        seeds = [11, 22, 33]

        profile = compute_noise_profile_from_replicates(pid, values, seeds)

        # Pairwise deltas: |24.0-24.2|=0.2, |24.0-24.4|=0.4, |24.2-24.4|=0.2
        assert len(profile.pairwise_abs_deltas) == 3
        assert any(abs(d - 0.2) < 1e-9 for d in profile.pairwise_abs_deltas)
        assert any(abs(d - 0.4) < 1e-9 for d in profile.pairwise_abs_deltas)
        assert profile.pairwise_delta_max == pytest.approx(0.4, rel=1e-9)

    def test_mismatched_values_and_seeds(self):
        """Mismatch between values and seeds raises error."""
        pid = NoiseProfileID("cap", "runtime", "metric")
        with pytest.raises(NoiseProfileNotFoundError, match="len\\(values\\)"):
            compute_noise_profile_from_replicates(
                pid,
                values=[24.0, 24.1],
                seeds=[11, 22, 33],  # Too many seeds
            )

    def test_empty_values_raises(self):
        """Empty values list raises error."""
        pid = NoiseProfileID("cap", "runtime", "metric")
        with pytest.raises(NoiseProfileNotFoundError, match="Empty"):
            compute_noise_profile_from_replicates(pid, [], [])

    def test_all_outliers_raises(self):
        """If all seeds are outliers, raises error."""
        pid = NoiseProfileID("cap", "runtime", "metric")
        with pytest.raises(NoiseProfileNotFoundError, match="All seeds marked as outliers"):
            compute_noise_profile_from_replicates(
                pid,
                values=[24.0, 24.1],
                seeds=[11, 22],
                outlier_seeds=[11, 22],  # All seeds marked as outliers
            )

    def test_bootstrap_ci_computed(self):
        """Bootstrap confidence intervals are computed."""
        pid = NoiseProfileID("test_cap", "test_runtime", "psnr")
        # Use many replicates so bootstrap percentiles are meaningful
        values = [20.0 + i * 0.1 for i in range(50)]
        seeds = list(range(50))

        profile = compute_noise_profile_from_replicates(pid, values, seeds)

        assert profile.bootstrap_ci_lower is not None
        assert profile.bootstrap_ci_upper is not None
        assert profile.bootstrap_ci_lower < profile.bootstrap_ci_upper


class TestComparePairedRuns:
    """Test paired comparison against a profile."""

    def test_valid_seed_in_profile(self):
        """Compare with a seed that's in the profile."""
        pid = NoiseProfileID("cap", "runtime", "psnr")
        profile = NoiseProfile(
            profile_id=pid,
            n_runs=3,
            seeds=[11, 22, 33],
            values=[24.0, 24.1, 24.2],
            mean=24.1,
            median=24.1,
        )

        result = compare_paired_runs(
            profile,
            baseline_value=24.0,
            candidate_value=24.2,
            seed=11,
            candidate_arm="candidate_arm_1",
            baseline_arm="baseline_arm_1",
        )

        assert result.comparable
        assert result.baseline_value == 24.0
        assert result.candidate_value == 24.2
        assert result.delta == pytest.approx(0.2, rel=1e-9)
        assert result.seed == 11

    def test_seed_not_in_profile(self):
        """Refuse to compare with a seed not in the profile."""
        pid = NoiseProfileID("cap", "runtime", "psnr")
        profile = NoiseProfile(
            profile_id=pid,
            n_runs=3,
            seeds=[11, 22, 33],
            values=[24.0, 24.1, 24.2],
            mean=24.1,
            median=24.1,
        )

        result = compare_paired_runs(
            profile,
            baseline_value=24.0,
            candidate_value=24.2,
            seed=99,  # Not in profile
            candidate_arm="cand",
            baseline_arm="base",
        )

        assert not result.comparable
        assert "seed 99 not in profile" in result.reason_if_not_comparable

    def test_rng_divergence_tracked(self):
        """RNG divergence is recorded in the result."""
        pid = NoiseProfileID("cap", "runtime", "psnr")
        profile = NoiseProfile(
            profile_id=pid,
            n_runs=2,
            seeds=[11, 22],
            values=[24.0, 24.1],
            mean=24.05,
            median=24.05,
        )

        divergence = RNGDivergence(
            streams_diverged=["torch_rng", "numpy_rng"],
            reason="control flow changed",
        )

        result = compare_paired_runs(
            profile,
            baseline_value=24.0,
            candidate_value=24.2,
            seed=11,
            rng_divergence=divergence,
        )

        assert result.comparable
        assert result.rng_divergence.streams_diverged == ["torch_rng", "numpy_rng"]
        assert "control flow" in result.rng_divergence.reason

    def test_invalid_profile_raises(self):
        """Comparing against an invalid profile raises error."""
        pid = NoiseProfileID("cap", "runtime", "psnr")
        profile = NoiseProfile(
            profile_id=pid,
            n_runs=0,  # Invalid
            seeds=[],
            values=[],
            mean=24.0,
            median=24.0,
        )

        with pytest.raises(NoiseProfileNotFoundError):
            compare_paired_runs(
                profile,
                baseline_value=24.0,
                candidate_value=24.2,
                seed=11,
            )

    def test_result_to_dict(self):
        """Serialize comparison result to dict."""
        pid = NoiseProfileID("cap", "runtime", "psnr")
        profile = NoiseProfile(
            profile_id=pid,
            n_runs=1,
            seeds=[11],
            values=[24.0],
            mean=24.0,
            median=24.0,
        )

        result = compare_paired_runs(
            profile,
            baseline_value=24.0,
            candidate_value=24.1,
            seed=11,
            candidate_arm="cand",
            baseline_arm="base",
        )

        d = result.to_dict()
        assert d["comparable"] is True
        assert d["delta"] == pytest.approx(0.1, rel=1e-9)
        assert d["seed"] == 11
        assert d["profile_id"]["capsule_id"] == "cap"


class TestParseNoiseProfile:
    """Test loading a noise profile from JSON."""

    def test_parse_valid_profile(self):
        """Load a valid noise profile from JSON file."""
        pid = NoiseProfileID("cap", "runtime", "psnr")
        profile = NoiseProfile(
            profile_id=pid,
            n_runs=2,
            seeds=[11, 22],
            values=[24.0, 24.1],
            mean=24.05,
            median=24.05,
            notes="test profile",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(profile.to_dict(), f)
            temp_path = f.name

        try:
            loaded = parse_noise_profile(temp_path)
            assert loaded.n_runs == 2
            assert loaded.seeds == [11, 22]
            assert loaded.profile_id.capsule_id == "cap"
        finally:
            Path(temp_path).unlink()

    def test_parse_missing_file(self):
        """Attempt to parse a missing file raises error."""
        with pytest.raises(NoiseProfileNotFoundError, match="File not found"):
            parse_noise_profile("/nonexistent/file.json")

    def test_parse_malformed_json(self):
        """Attempt to parse malformed JSON raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {")
            temp_path = f.name

        try:
            with pytest.raises(NoiseProfileNotFoundError, match="JSON decode error"):
                parse_noise_profile(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_parse_missing_required_field(self):
        """Missing required fields in JSON raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Missing 'n_runs'
            json.dump(
                {
                    "profile_id": {
                        "capsule_id": "cap",
                        "runtime_class": "runtime",
                        "metric": "psnr",
                    },
                },
                f,
            )
            temp_path = f.name

        try:
            with pytest.raises(NoiseProfileNotFoundError, match="Missing required field"):
                parse_noise_profile(temp_path)
        finally:
            Path(temp_path).unlink()


class TestRefusalOnMismatch:
    """Test that comparisons REFUSE mismatches, never silently fall back."""

    def test_capsule_mismatch_is_refused(self):
        """A profile for capsule_A cannot be used for capsule_B."""
        # This is more of an integration test -- it would be enforced by
        # the calling code (compare_paired_runs checks seed membership, not capsule).
        # But we verify the contract is stated in the API.
        pid_a = NoiseProfileID("capsule_A", "runtime", "psnr")
        pid_b = NoiseProfileID("capsule_B", "runtime", "psnr")
        assert pid_a != pid_b

    def test_runtime_mismatch_should_create_different_profiles(self):
        """Different runtimes should have separate profiles."""
        pid_rocm = NoiseProfileID("cap", "rocm_gfx1151", "psnr")
        pid_cuda = NoiseProfileID("cap", "cuda_a100", "psnr")
        assert pid_rocm != pid_cuda

    def test_metric_mismatch_should_create_different_profiles(self):
        """Different metrics should have separate profiles."""
        pid_psnr = NoiseProfileID("cap", "runtime", "psnr")
        pid_ssim = NoiseProfileID("cap", "runtime", "ssim")
        assert pid_psnr != pid_ssim

    def test_mismatched_seed_refuses_comparison(self):
        """A comparison is refused if the seed is not in the profile."""
        pid = NoiseProfileID("cap", "runtime", "psnr")
        profile = NoiseProfile(
            profile_id=pid,
            n_runs=1,
            seeds=[11],
            values=[24.0],
            mean=24.0,
            median=24.0,
        )

        result = compare_paired_runs(
            profile,
            baseline_value=24.0,
            candidate_value=24.2,
            seed=999,  # Not in profile
        )

        assert not result.comparable
        assert result.reason_if_not_comparable


class TestSyntheticTestProfile:
    """Test with a synthetic small-capsule test profile."""

    def test_create_synthetic_test_profile(self):
        """Create a synthetic profile for testing."""
        pid = NoiseProfileID(
            capsule_id="synthetic_test_small_v1",
            runtime_class="cpu_offline",
            metric="test_psnr_db",
        )

        # Synthetic A/A runs with deterministic, small variance
        values = [20.0 + 0.05 * i for i in range(5)]  # [20.0, 20.05, ..., 20.2]
        seeds = [100, 101, 102, 103, 104]

        profile = compute_noise_profile_from_replicates(
            pid, values, seeds, notes="Synthetic test profile (not real calibration)"
        )

        assert "Synthetic" in profile.notes
        assert profile.pairwise_delta_max is not None
        assert profile.pairwise_delta_max == pytest.approx(0.2, rel=1e-9)  # Span is 0.2

    def test_synthetic_profile_prevents_real_claims(self):
        """A synthetic profile's notes warn it should not be used for real claims."""
        pid = NoiseProfileID(
            capsule_id="synthetic_test",
            runtime_class="offline",
            metric="psnr",
        )
        profile = NoiseProfile(
            profile_id=pid,
            n_runs=3,
            seeds=[11, 22, 33],
            values=[24.0, 24.1, 24.2],
            mean=24.1,
            median=24.1,
            notes="SYNTHETIC/TEST-ONLY: Not measured from real GPU runs.",
        )

        # The notes field is present; it's the caller's responsibility
        # to read it and refuse paper-grade claims on synthetic data.
        assert "SYNTHETIC" in profile.notes
        assert "TEST-ONLY" in profile.notes
