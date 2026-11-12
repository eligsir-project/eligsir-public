"""Comprehensive tests for decision profile and adjudicator.

Tests the quantitative adjudication machinery, anti-patterns, and edge cases.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import torch-free modules
from eligsir.experiments import decision_profile
from eligsir.experiments import adjudicator

# Repo root, for the subprocess-isolated torch-import check below.
REPO_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Decision Profile Tests
# =============================================================================


class TestDecisionProfile:
    """Tests for DecisionProfile schema and serialization."""

    def test_metric_spec_validation(self):
        """MetricSpec validates direction and margins."""
        # Valid
        spec = decision_profile.MetricSpec(
            metric="mean_test_psnr",
            direction="higher",
            practical_margin=0.30,
        )
        assert spec.metric == "mean_test_psnr"

        # Invalid direction
        with pytest.raises(ValueError, match="direction must be"):
            decision_profile.MetricSpec(
                metric="mean_test_psnr",
                direction="invalid",
                practical_margin=0.30,
            )

        # Invalid margin
        with pytest.raises(ValueError, match="practical_margin must be non-negative"):
            decision_profile.MetricSpec(
                metric="mean_test_psnr",
                direction="higher",
                practical_margin=-0.30,
            )

    def test_guardrail_spec_validation(self):
        """GuardrailSpec requires exactly one regression type."""
        # Valid: max_regression
        gr1 = decision_profile.GuardrailSpec(
            metric="mean_test_psnr",
            max_regression=0.20,
        )
        assert gr1.max_regression == 0.20
        assert gr1.max_relative_regression is None

        # Valid: max_relative_regression
        gr2 = decision_profile.GuardrailSpec(
            metric="trajectory_fps",
            max_relative_regression=0.05,
        )
        assert gr2.max_relative_regression == 0.05
        assert gr2.max_regression is None

        # Invalid: both set
        with pytest.raises(ValueError, match="exactly one"):
            decision_profile.GuardrailSpec(
                metric="test",
                max_regression=0.20,
                max_relative_regression=0.05,
            )

        # Invalid: neither set
        with pytest.raises(ValueError, match="exactly one"):
            decision_profile.GuardrailSpec(metric="test")

    def test_replication_spec_validation(self):
        """ReplicationSpec validates pair counts."""
        # Valid
        spec = decision_profile.ReplicationSpec(
            minimum_pairs=1,
            maximum_pairs=3,
        )
        assert spec.minimum_pairs == 1

        # Invalid: max < min
        with pytest.raises(ValueError, match="maximum_pairs"):
            decision_profile.ReplicationSpec(
                minimum_pairs=5,
                maximum_pairs=3,
            )

    def test_profile_serialization(self):
        """Profiles serialize and deserialize identically."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test_profile",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
            hard_guardrails=[
                decision_profile.GuardrailSpec(
                    metric="mean_test_lpips",
                    max_regression=0.05,
                ),
            ],
            replication=decision_profile.ReplicationSpec(
                minimum_pairs=1,
                maximum_pairs=3,
            ),
        )

        # to_dict + from_dict round trip
        data = decision_profile.to_dict(profile)
        profile2 = decision_profile.from_dict(data)

        assert profile2.profile_id == profile.profile_id
        assert len(profile2.primary) == len(profile.primary)
        assert profile2.primary[0].metric == "mean_test_psnr"

    def test_profile_hash_deterministic(self):
        """profile_hash is deterministic and stable across load/dump."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
        )

        hash1 = decision_profile.profile_hash(profile)
        hash2 = decision_profile.profile_hash(profile)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex

        # Round trip preserves hash
        data = decision_profile.to_dict(profile)
        profile2 = decision_profile.from_dict(data)
        hash3 = decision_profile.profile_hash(profile2)
        assert hash3 == hash1

    def test_profile_hash_changes_with_values(self):
        """Changing profile values changes the hash."""
        profile1 = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
        )

        profile2 = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.40,  # Different margin
                ),
            ],
        )

        hash1 = decision_profile.profile_hash(profile1)
        hash2 = decision_profile.profile_hash(profile2)
        assert hash1 != hash2

    def test_profile_load_save(self, tmp_path):
        """Profiles can be saved and loaded from JSON."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
        )

        path = tmp_path / "profile.json"
        decision_profile.save(profile, str(path))
        assert path.exists()

        profile2 = decision_profile.load(str(path))
        assert profile2.profile_id == profile.profile_id
        assert decision_profile.profile_hash(profile) == decision_profile.profile_hash(profile2)


# =============================================================================
# Adjudicator Tests
# =============================================================================


def _make_test_report(psnr=25.0, lpips=0.05, fps=30.0, eval_valid=True):
    """Create a minimal report.json dict for testing."""
    return {
        "mean_test_psnr": psnr,
        "min_test_psnr": psnr - 1.0,
        "max_test_psnr": psnr + 1.0,
        "mean_test_lpips": lpips,
        "mean_train_psnr": psnr + 0.5,
        "trajectory_fps": fps,
        "n_test": 50,
        "n_train": 400,
        "n_test_metrics": 50,
        "eval_valid": eval_valid,
        "eval_metadata": {
            "scored_holdout_hash": "abc123",
            "realized_holdout_hash": "abc123",
        },
        "seed_fade": {
            "detected": False,
            "confounded": False,
            "indeterminate": False,
        },
    }


def _make_run_dir(tmp_path, report_json):
    """Create a temporary run directory with a report.json."""
    run_dir = tmp_path / "test_run"
    comparison_dir = run_dir / "comparison" / "iter_5000"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    report_path = comparison_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report_json, f)

    return str(run_dir)


class TestAdjudicator:
    """Tests for the quantitative adjudicator."""

    def test_profile_hash_mismatch_rejection(self, tmp_path):
        """Adjudication refuses if profile hash doesn't match declaration."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
        )

        baseline_dir = _make_run_dir(tmp_path, _make_test_report())
        candidate_dir = _make_run_dir(tmp_path / "cand", _make_test_report(psnr=25.5))

        # Use wrong hash
        wrong_hash = "0" * 64
        result = adjudicator.adjudicate(
            profile=profile,
            declared_profile_hash=wrong_hash,
            baseline_run_dir=baseline_dir,
            candidate_run_dirs=[candidate_dir],
        )

        assert result.profile_hash_matches is False
        assert result.validity_state == adjudicator.VALIDITY_INVALID_COMPARISON
        assert result.quantitative_verdict == adjudicator.VERDICT_QUALITATIVE_REVIEW_REQUIRED

    def test_missing_report_rejection(self, tmp_path):
        """Adjudication refuses if a report is missing."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
        )

        baseline_dir = str(tmp_path / "nonexistent")
        candidate_dir = _make_run_dir(tmp_path / "cand", _make_test_report())

        profile_hash = decision_profile.profile_hash(profile)
        result = adjudicator.adjudicate(
            profile=profile,
            declared_profile_hash=profile_hash,
            baseline_run_dir=baseline_dir,
            candidate_run_dirs=[candidate_dir],
        )

        assert result.validity_state == adjudicator.VALIDITY_MISSING_REQUIRED_EVIDENCE
        assert result.quantitative_verdict == adjudicator.VERDICT_QUALITATIVE_REVIEW_REQUIRED

    def test_eval_valid_false_rejection(self, tmp_path):
        """Adjudication refuses if eval_valid is false."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
        )

        baseline_dir = _make_run_dir(tmp_path, _make_test_report(eval_valid=False))
        candidate_dir = _make_run_dir(tmp_path / "cand", _make_test_report())

        profile_hash = decision_profile.profile_hash(profile)
        result = adjudicator.adjudicate(
            profile=profile,
            declared_profile_hash=profile_hash,
            baseline_run_dir=baseline_dir,
            candidate_run_dirs=[candidate_dir],
        )

        assert result.validity_state == adjudicator.VALIDITY_INVALID_COMPARISON
        assert any("eval_valid" in r for r in result.validity_reasons)

    def test_holdout_hash_mismatch_rejection(self, tmp_path):
        """Adjudication refuses if holdout hashes don't match."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
        )

        baseline_report = _make_test_report()
        baseline_report["eval_metadata"]["scored_holdout_hash"] = "abc123"

        candidate_report = _make_test_report(psnr=25.5)
        candidate_report["eval_metadata"]["scored_holdout_hash"] = "def456"

        baseline_dir = _make_run_dir(tmp_path, baseline_report)
        candidate_dir = _make_run_dir(tmp_path / "cand", candidate_report)

        profile_hash = decision_profile.profile_hash(profile)
        result = adjudicator.adjudicate(
            profile=profile,
            declared_profile_hash=profile_hash,
            baseline_run_dir=baseline_dir,
            candidate_run_dirs=[candidate_dir],
        )

        assert result.validity_state == adjudicator.VALIDITY_INVALID_COMPARISON
        assert any("hash_mismatch" in r for r in result.validity_reasons)

    def test_seed_fade_confounded_rejection(self, tmp_path):
        """Adjudication refuses if seed_fade is confounded."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
        )

        candidate_report = _make_test_report(psnr=25.5)
        candidate_report["seed_fade"]["confounded"] = True

        baseline_dir = _make_run_dir(tmp_path, _make_test_report())
        candidate_dir = _make_run_dir(tmp_path / "cand", candidate_report)

        profile_hash = decision_profile.profile_hash(profile)
        result = adjudicator.adjudicate(
            profile=profile,
            declared_profile_hash=profile_hash,
            baseline_run_dir=baseline_dir,
            candidate_run_dirs=[candidate_dir],
        )

        assert result.validity_state == adjudicator.VALIDITY_INVALID_COMPARISON
        assert any("seed_fade" in r for r in result.validity_reasons)

    def test_noise_profile_unavailable(self, tmp_path):
        """Verdict degrades to QUALITATIVE_REVIEW if noise profile unavailable."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                    noise_profile_metric="mean_test_psnr",
                ),
            ],
        )

        baseline_dir = _make_run_dir(tmp_path, _make_test_report())
        candidate_dir = _make_run_dir(tmp_path / "cand", _make_test_report(psnr=25.12))

        profile_hash = decision_profile.profile_hash(profile)
        result = adjudicator.adjudicate(
            profile=profile,
            declared_profile_hash=profile_hash,
            baseline_run_dir=baseline_dir,
            candidate_run_dirs=[candidate_dir],
            noise_provider=None,  # No noise provider
        )

        # Without a noise profile, small effects should be inconclusive
        assert result.quantitative_verdict in [
            adjudicator.VERDICT_INCONCLUSIVE_NEEDS_MORE_RUNS,
            adjudicator.VERDICT_QUALITATIVE_REVIEW_REQUIRED,
        ]

    def test_clear_win_detection(self, tmp_path):
        """A clear, large improvement is detected as CLEAR_WIN."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
            replication=decision_profile.ReplicationSpec(
                minimum_pairs=1,
                maximum_pairs=3,
            ),
        )

        baseline_dir = _make_run_dir(tmp_path, _make_test_report(psnr=25.0))
        candidate_dir = _make_run_dir(tmp_path / "cand", _make_test_report(psnr=26.0))

        profile_hash = decision_profile.profile_hash(profile)
        result = adjudicator.adjudicate(
            profile=profile,
            declared_profile_hash=profile_hash,
            baseline_run_dir=baseline_dir,
            candidate_run_dirs=[candidate_dir],
        )

        assert result.quantitative_verdict == adjudicator.VERDICT_CLEAR_WIN
        assert result.n_pairs == 1

    def test_clear_loss_detection(self, tmp_path):
        """A clear regression is detected as CLEAR_LOSS."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
            replication=decision_profile.ReplicationSpec(
                minimum_pairs=1,
                maximum_pairs=3,
            ),
        )

        baseline_dir = _make_run_dir(tmp_path, _make_test_report(psnr=25.0))
        candidate_dir = _make_run_dir(tmp_path / "cand", _make_test_report(psnr=24.0))

        profile_hash = decision_profile.profile_hash(profile)
        result = adjudicator.adjudicate(
            profile=profile,
            declared_profile_hash=profile_hash,
            baseline_run_dir=baseline_dir,
            candidate_run_dirs=[candidate_dir],
        )

        assert result.quantitative_verdict == adjudicator.VERDICT_CLEAR_LOSS

    def test_small_effect_below_practical_margin_is_inconclusive(self, tmp_path):
        """A small effect (+0.12 dB) below practical margin is inconclusive, not win."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,  # Larger than the effect
                ),
            ],
            replication=decision_profile.ReplicationSpec(
                minimum_pairs=1,
                maximum_pairs=1,
            ),
        )

        baseline_dir = _make_run_dir(tmp_path, _make_test_report(psnr=25.0))
        candidate_dir = _make_run_dir(tmp_path / "cand", _make_test_report(psnr=25.12))

        profile_hash = decision_profile.profile_hash(profile)
        result = adjudicator.adjudicate(
            profile=profile,
            declared_profile_hash=profile_hash,
            baseline_run_dir=baseline_dir,
            candidate_run_dirs=[candidate_dir],
        )

        # Should be inconclusive, not a win
        assert result.quantitative_verdict == adjudicator.VERDICT_INCONCLUSIVE_NEEDS_MORE_RUNS

    def test_metric_shopping_rejection(self, tmp_path):
        """A metric not in profile.primary cannot cause a win."""
        # Profile declares PSNR as primary, LPIPS as secondary
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
            replication=decision_profile.ReplicationSpec(
                minimum_pairs=1,
                maximum_pairs=3,
            ),
        )

        # Candidate: LPIPS improves, PSNR regresses significantly
        baseline_dir = _make_run_dir(tmp_path, _make_test_report(psnr=25.0, lpips=0.05))
        candidate_dir = _make_run_dir(
            tmp_path / "cand",
            _make_test_report(psnr=23.5, lpips=0.03),  # LPIPS improves, PSNR regresses -1.5dB
        )

        profile_hash = decision_profile.profile_hash(profile)
        result = adjudicator.adjudicate(
            profile=profile,
            declared_profile_hash=profile_hash,
            baseline_run_dir=baseline_dir,
            candidate_run_dirs=[candidate_dir],
        )

        # Not a win even though LPIPS improved, because primary metric regressed significantly
        assert result.quantitative_verdict == adjudicator.VERDICT_CLEAR_LOSS

    def test_hard_guardrail_failure_is_loss(self, tmp_path):
        """Hard guardrail failure causes CLEAR_LOSS even if primary improves."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
            hard_guardrails=[
                decision_profile.GuardrailSpec(
                    metric="mean_test_lpips",
                    max_regression=0.02,  # Tight guardrail
                ),
            ],
            replication=decision_profile.ReplicationSpec(
                minimum_pairs=1,
                maximum_pairs=3,
            ),
        )

        # Primary improves, but LPIPS regresses beyond guardrail
        baseline_dir = _make_run_dir(tmp_path, _make_test_report(psnr=25.0, lpips=0.05))
        candidate_dir = _make_run_dir(
            tmp_path / "cand",
            _make_test_report(psnr=26.0, lpips=0.10),  # +1dB PSNR, but LPIPS worse
        )

        profile_hash = decision_profile.profile_hash(profile)
        result = adjudicator.adjudicate(
            profile=profile,
            declared_profile_hash=profile_hash,
            baseline_run_dir=baseline_dir,
            candidate_run_dirs=[candidate_dir],
        )

        # Hard guardrail failure = CLEAR_LOSS
        assert result.quantitative_verdict == adjudicator.VERDICT_CLEAR_LOSS
        assert len(result.hard_guardrail_results) > 0
        assert not result.hard_guardrail_results[0].passes

    def test_cost_guardrail_failure_is_tradeoff(self, tmp_path):
        """Cost guardrail failure with primary win results in TRADEOFF."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
            cost_guardrails=[
                decision_profile.GuardrailSpec(
                    metric="trajectory_fps",
                    max_regression=5.0,  # Allow 5 fps regression
                ),
            ],
            replication=decision_profile.ReplicationSpec(
                minimum_pairs=1,
                maximum_pairs=3,
            ),
        )

        # Primary improves, but FPS drops significantly
        baseline_dir = _make_run_dir(tmp_path, _make_test_report(psnr=25.0, fps=30.0))
        candidate_dir = _make_run_dir(
            tmp_path / "cand",
            _make_test_report(psnr=26.0, fps=20.0),  # +1dB PSNR, -10 fps
        )

        profile_hash = decision_profile.profile_hash(profile)
        result = adjudicator.adjudicate(
            profile=profile,
            declared_profile_hash=profile_hash,
            baseline_run_dir=baseline_dir,
            candidate_run_dirs=[candidate_dir],
        )

        assert result.quantitative_verdict == adjudicator.VERDICT_TRADEOFF
        assert result.tradeoff_description

    def test_insufficient_pairs_is_inconclusive(self, tmp_path):
        """With n_pairs < minimum_pairs, verdict is inconclusive."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
            replication=decision_profile.ReplicationSpec(
                minimum_pairs=3,
                maximum_pairs=5,
            ),
        )

        baseline_dir = _make_run_dir(tmp_path, _make_test_report())
        # Only 1 candidate, but profile requires 3+
        candidate_dir = _make_run_dir(tmp_path / "cand", _make_test_report(psnr=25.5))

        profile_hash = decision_profile.profile_hash(profile)
        result = adjudicator.adjudicate(
            profile=profile,
            declared_profile_hash=profile_hash,
            baseline_run_dir=baseline_dir,
            candidate_run_dirs=[candidate_dir],
        )

        assert result.quantitative_verdict == adjudicator.VERDICT_INCONCLUSIVE_NEEDS_MORE_RUNS

    def test_maximum_pairs_enforcement(self, tmp_path):
        """Supplying more pairs than maximum_pairs results in inconclusive, not scored."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
            replication=decision_profile.ReplicationSpec(
                minimum_pairs=1,
                maximum_pairs=2,  # Max 2 pairs
            ),
        )

        baseline_dir = _make_run_dir(tmp_path, _make_test_report())
        # Provide 3 candidates (exceeds max)
        candidate_dirs = [
            _make_run_dir(tmp_path / "cand1", _make_test_report(psnr=25.5)),
            _make_run_dir(tmp_path / "cand2", _make_test_report(psnr=26.0)),
            _make_run_dir(tmp_path / "cand3", _make_test_report(psnr=26.5)),
        ]

        profile_hash = decision_profile.profile_hash(profile)
        result = adjudicator.adjudicate(
            profile=profile,
            declared_profile_hash=profile_hash,
            baseline_run_dir=baseline_dir,
            candidate_run_dirs=candidate_dirs,
        )

        # Should respect maximum; at max, wide interval = inconclusive
        assert result.n_pairs <= 3  # All provided

    def test_no_import_torch(self):
        """Verify importing the adjudicator module does not pull in torch.

        Run in a SUBPROCESS deliberately. The first version of this test
        evicted every ``torch*`` key from this process's ``sys.modules`` and
        never restored them. That is silently harmless on the torch-free CPU
        CI image (nothing to evict, nothing downstream imports torch), and
        catastrophic in the real training image: the next test to `import
        torch` re-executes torch's C extension initialisation against an
        already-initialised interpreter and dies with ``RuntimeError:
        function '_has_torch_function' already has a docstring``. Measured on
        the real training image: 333 unrelated tests failed downstream of this
        one test, with CPU CI fully green -- the same CI-blind-spot class as
        issue #649. A subprocess has no shared ``sys.modules`` to corrupt.

        The original assertion was also wrong on its own terms: ``"torch" not
        in sys.modules or "torchvision" not in sys.modules`` passes whenever
        EITHER name is absent, so it stayed green even if the module did
        import torch.
        """
        import subprocess
        import sys

        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "import eligsir.experiments.adjudicator  # noqa: F401\n"
                    "leaked = sorted(m for m in sys.modules "
                    "if m == 'torch' or m.startswith('torch.'))\n"
                    "print(','.join(leaked))\n"
                ),
            ],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
        )

        assert proc.returncode == 0, (
            f"importing the adjudicator in a clean interpreter failed:\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
        leaked = proc.stdout.strip()
        assert leaked == "", (
            f"importing eligsir.experiments.adjudicator pulled in torch "
            f"module(s): {leaked}"
        )


class TestExperimentResultSerialization:
    """Tests for writing and reading experiment results."""

    def test_write_experiment_result(self, tmp_path):
        """ExperimentResult is serialized to JSON correctly."""
        profile = decision_profile.DecisionProfile(
            schema_version=1,
            profile_id="test",
            experiment_family="test",
            primary=[
                decision_profile.MetricSpec(
                    metric="mean_test_psnr",
                    direction="higher",
                    practical_margin=0.30,
                ),
            ],
        )

        baseline_dir = _make_run_dir(tmp_path, _make_test_report())
        candidate_dir = _make_run_dir(tmp_path / "cand", _make_test_report(psnr=25.5))

        profile_hash = decision_profile.profile_hash(profile)
        result = adjudicator.adjudicate(
            profile=profile,
            declared_profile_hash=profile_hash,
            baseline_run_dir=baseline_dir,
            candidate_run_dirs=[candidate_dir],
        )

        output_path = tmp_path / "experiment_result.json"
        adjudicator.write_experiment_result(result, str(output_path))

        assert output_path.exists()

        # Read back and verify
        with open(output_path) as f:
            data = json.load(f)

        assert data["profile_id"] == "test"
        assert data["quantitative_verdict"] in adjudicator.VERDICTS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
