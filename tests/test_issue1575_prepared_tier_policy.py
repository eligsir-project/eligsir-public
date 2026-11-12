"""Test prepared tier policy immutability and declaration enforcement (issue #1575 C.1).

Tests verify that:
- The canonical profile's declared tier set equals today's materialized set;
- The producer materializes exactly the declared tiers (no more, no less);
- Undeclared tiers after READY raise with a clear error;
- The policy is immutable and resolved once per application.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from eligsir.contracts import SourceIdentity, ModalityEligibility, BufferOwnership, BufferRef, BufferSpec
from eligsir.contracts.control import ResolutionTier, PreparedTierPolicy
from eligsir.runtime.prepared_source_producer import PreparedSourceProducer
from eligsir.runtime.ready_source import ReadySourceOwner


class TestPreparedTierPolicy(unittest.TestCase):
    """Test PreparedTierPolicy immutability and validation."""

    def test_policy_creation_from_supported_tiers(self) -> None:
        """Test creating a policy from supported tiers."""
        tiers = (ResolutionTier.NATIVE, ResolutionTier.HALF, ResolutionTier.QUARTER)
        policy = PreparedTierPolicy.from_supported_tiers(tiers)
        self.assertIsInstance(policy, PreparedTierPolicy)
        self.assertEqual(policy.declared_tiers, frozenset(tiers))

    def test_policy_creation_with_single_tier(self) -> None:
        """Test creating a policy with a single tier (fixed policy case)."""
        tiers = (ResolutionTier.NATIVE,)
        policy = PreparedTierPolicy.from_supported_tiers(tiers)
        self.assertEqual(policy.declared_tiers, frozenset(tiers))

    def test_policy_creation_with_frozenset(self) -> None:
        """Test creating a policy from a frozenset directly."""
        tiers = frozenset((ResolutionTier.NATIVE, ResolutionTier.QUARTER))
        policy = PreparedTierPolicy.from_supported_tiers(tiers)
        self.assertEqual(policy.declared_tiers, tiers)

    def test_policy_is_frozen(self) -> None:
        """Test that policy instances are immutable."""
        policy = PreparedTierPolicy.from_supported_tiers(
            (ResolutionTier.NATIVE, ResolutionTier.HALF)
        )
        with self.assertRaises(Exception):
            # FrozenInstanceError is raised when trying to modify a frozen dataclass
            policy.declared_tiers = frozenset((ResolutionTier.QUARTER,))

    def test_policy_same_identity_across_creations(self) -> None:
        """Test that the same tiers yield the same policy identity (test d)."""
        tiers = (ResolutionTier.NATIVE, ResolutionTier.HALF, ResolutionTier.QUARTER)
        policy_a = PreparedTierPolicy.from_supported_tiers(tiers)
        policy_b = PreparedTierPolicy.from_supported_tiers(tiers)
        # Same content, but different objects (dataclass equality vs identity)
        self.assertEqual(policy_a, policy_b)
        # The policy itself is the resolved/immutable artifact;
        # in production, the application creates one policy shared across all producers.

    def test_policy_rejects_empty_tiers(self) -> None:
        """Test that policy rejects empty tier sets."""
        with self.assertRaises(ValueError):
            PreparedTierPolicy(declared_tiers=frozenset())

    def test_policy_rejects_non_frozenset(self) -> None:
        """Test that policy rejects non-frozenset tier containers."""
        with self.assertRaises(TypeError):
            PreparedTierPolicy(declared_tiers=[ResolutionTier.NATIVE])  # type: ignore

    def test_policy_rejects_invalid_tier_types(self) -> None:
        """Test that policy rejects non-ResolutionTier values."""
        with self.assertRaises(TypeError):
            PreparedTierPolicy(declared_tiers=frozenset(("native", "half")))  # type: ignore

    def test_policy_highest_adaptive(self) -> None:
        """Test that highest() returns native for adaptive (default) policy."""
        policy = PreparedTierPolicy.from_supported_tiers(
            (ResolutionTier.NATIVE, ResolutionTier.HALF, ResolutionTier.QUARTER)
        )
        self.assertEqual(policy.highest(), ResolutionTier.NATIVE)

    def test_policy_highest_fixed_quarter(self) -> None:
        """Test that highest() returns quarter for fixed-quarter policy."""
        policy = PreparedTierPolicy.from_supported_tiers((ResolutionTier.QUARTER,))
        self.assertEqual(policy.highest(), ResolutionTier.QUARTER)

    def test_policy_highest_fixed_half(self) -> None:
        """Test that highest() returns half for fixed-half policy."""
        policy = PreparedTierPolicy.from_supported_tiers((ResolutionTier.HALF,))
        self.assertEqual(policy.highest(), ResolutionTier.HALF)

    def test_policy_highest_mixed_set(self) -> None:
        """Test that highest() correctly selects from mixed tier set."""
        # Order in frozenset is undefined, so verify determinism
        policy = PreparedTierPolicy.from_supported_tiers(
            (ResolutionTier.QUARTER, ResolutionTier.HALF)
        )
        self.assertEqual(policy.highest(), ResolutionTier.HALF)


class TestPreparedSourceProducerTierPolicy(unittest.TestCase):
    """Test that the producer respects the tier policy."""

    def test_producer_default_policy_includes_eighth_tier(self) -> None:
        """The fallback producer must prepare every default adaptive tier."""
        owner = ReadySourceOwner(max_fresh=2, max_replay=2)
        transfer_owner = mock.MagicMock()
        transfer_owner.prepare_source = mock.MagicMock(return_value=(None, None))
        transfer_owner.prepare_source_tiers = None  # Use CPU path

        producer = PreparedSourceProducer(owner, transfer_owner=transfer_owner)
        self.assertEqual(
            producer._prepared_tier_policy.declared_tiers,
            frozenset(ResolutionTier),
        )

    def test_producer_accepts_custom_policy(self) -> None:
        """Test that producer accepts a custom tier policy."""
        owner = ReadySourceOwner(max_fresh=2, max_replay=2)
        transfer_owner = mock.MagicMock()
        transfer_owner.prepare_source = mock.MagicMock(return_value=(None, None))

        policy = PreparedTierPolicy.from_supported_tiers((ResolutionTier.NATIVE,))
        producer = PreparedSourceProducer(
            owner, transfer_owner=transfer_owner, prepared_tier_policy=policy
        )
        self.assertIs(producer._prepared_tier_policy, policy)

    def test_producer_rejects_invalid_policy_type(self) -> None:
        """Test that producer rejects non-PreparedTierPolicy objects."""
        owner = ReadySourceOwner(max_fresh=2, max_replay=2)
        transfer_owner = mock.MagicMock()
        transfer_owner.prepare_source = mock.MagicMock(return_value=(None, None))

        with self.assertRaises(TypeError):
            PreparedSourceProducer(
                owner,
                transfer_owner=transfer_owner,
                prepared_tier_policy="native",  # type: ignore
            )

    def test_prepare_tier_payloads_uses_declared_tiers(self) -> None:
        """Test that _prepare_tier_payloads materializes only declared tiers (test b)."""
        owner = ReadySourceOwner(max_fresh=2, max_replay=2)
        transfer_owner = mock.MagicMock()
        transfer_owner.prepare_source = mock.MagicMock(return_value=(None, None))
        # Force prepare_source_tiers to None so CPU path is used
        del transfer_owner.prepare_source_tiers

        # CPU double path: we can spy on materialize_resolution_tensors
        with mock.patch(
            "eligsir.mapping.resolution_materialization.materialize_resolution_tensors"
        ) as mock_materialize:
            mock_materialize.return_value = (
                torch.zeros(1, 1, 3),  # rgb_hwc
                torch.zeros(1, 1),    # depth_hw
                torch.eye(3),         # intrinsics
                None,                 # confidence
            )

            policy = PreparedTierPolicy.from_supported_tiers((ResolutionTier.NATIVE,))
            producer = PreparedSourceProducer(
                owner,
                transfer_owner=transfer_owner,
                prepared_tier_policy=policy,
            )

            # Call _prepare_tier_payloads
            rgb = torch.zeros(2, 2, 3)
            depth = torch.zeros(2, 2)
            intrinsics = torch.eye(3)
            fallback_event = mock.MagicMock()
            fallback_event.query = mock.MagicMock(return_value=True)

            payloads, event = producer._prepare_tier_payloads(
                rgb, depth, intrinsics, fallback_event=fallback_event
            )

            # Should materialize only native tier
            self.assertEqual(set(payloads), {"native"})
            # materialize_resolution_tensors should be called only once (for native)
            self.assertEqual(mock_materialize.call_count, 1)
            mock_materialize.assert_called_with(
                rgb,
                depth,
                intrinsics,
                ResolutionTier.NATIVE,
                resampling_mode="legacy",
                depth_downsample="nearest",
            )

    def test_prepare_tier_payloads_fails_on_wrong_set(self) -> None:
        """Test that _prepare_tier_payloads fails if transfer owner returns wrong set."""
        owner = ReadySourceOwner(max_fresh=2, max_replay=2)
        transfer_owner = mock.MagicMock()

        # Mock transfer owner that returns wrong tiers
        transfer_owner.prepare_source_tiers = mock.MagicMock(
            return_value=(
                {
                    "native": (torch.zeros(2, 2, 3), torch.zeros(2, 2), torch.eye(3), None),
                    "half": (torch.zeros(1, 1, 3), torch.zeros(1, 1), torch.eye(3), None),
                },
                mock.MagicMock(query=mock.MagicMock(return_value=True)),
            )
        )

        policy = PreparedTierPolicy.from_supported_tiers((ResolutionTier.NATIVE,))
        producer = PreparedSourceProducer(
            owner,
            transfer_owner=transfer_owner,
            prepared_tier_policy=policy,
        )

        rgb = torch.zeros(2, 2, 3)
        depth = torch.zeros(2, 2)
        intrinsics = torch.eye(3)
        fallback_event = mock.MagicMock()

        # Should raise because returned tiers don't match declared
        with self.assertRaises(TypeError) as cm:
            producer._prepare_tier_payloads(
                rgb, depth, intrinsics, fallback_event=fallback_event
            )
        self.assertIn("native", str(cm.exception))
        self.assertIn("half", str(cm.exception))


class TestCanonicalProfileDeclaredTierSet(unittest.TestCase):
    """Active adaptive profiles must expose every legal live tier."""

    def test_active_adaptive_profiles_and_mcap_alias_allow_eighth(self) -> None:
        """TUM, transport-neutral MCAP, and high-resolution overlays share the ladder."""
        from eligsir.cli.config import load_cli_config
        from eligsir.cli.parser import parse_training_args
        from eligsir.cli.run_config import validate_config
        from eligsir.runtime.config_boundary import mapping_runtime_config_from_args

        scan_overlay = "configs/profiles/evaluation/scannetpp_iphone_adaptive.yml"
        cases = (
            ("bare-default", None, (), "/tmp/source"),
            ("canonical-tum", "live/canonical_mapping", (), "/tmp/tum"),
            ("canonical-mcap", "live/canonical_mapping", (), "/tmp/kitchen.mcap"),
            ("tum-replay", "causal_replay/tum_rgbd", (), "/tmp/tum"),
            ("r9700-bench", "bench/r9700_streaming", (), "/tmp/tum"),
            ("scannetpp", "live/canonical_mapping", (scan_overlay,), "/tmp/scannet"),
            ("live-aa-classic", "experiments/live_aa/8060s/classic", (), "/tmp/tum"),
            ("live-aa-antialiased", "experiments/live_aa/8060s/antialiased", (), "/tmp/tum"),
            ("8060s-high-frequency", "hardware/8060s/high_frequency", (), "/tmp/tum"),
            ("8060s-live-throughput", "hardware/8060s/live_throughput", (), "/tmp/tum"),
            ("8060s-refinement", "hardware/8060s/refinement", (), "/tmp/tum"),
            ("r9700-high-frequency", "hardware/r9700/high_frequency", (), "/tmp/tum"),
            ("r9700-live-throughput", "hardware/r9700/live_throughput", (), "/tmp/tum"),
            ("r9700-refinement", "hardware/r9700/refinement", (), "/tmp/tum"),
            ("rtx3090-high-frequency", "hardware/rtx3090/high_frequency", (), "/tmp/tum"),
            ("rtx3090-live-throughput", "hardware/rtx3090/live_throughput", (), "/tmp/tum"),
            ("rtx3090-refinement", "hardware/rtx3090/refinement", (), "/tmp/tum"),
            ("classic-rendering", "rendering/live_classic", (), "/tmp/tum"),
            ("antialiased-rendering", "rendering/live_antialiased", (), "/tmp/tum"),
        )
        for label, run_profile, config_files, source_path in cases:
            with self.subTest(profile=label):
                args, _bundle = parse_training_args(["-s", source_path])
                for key, value in validate_config(
                    load_cli_config(
                        run_profile=run_profile, config_files=list(config_files)
                    )
                ).items():
                    setattr(args, key, value)
                args.model_path = f"/tmp/{label}"
                runtime = mapping_runtime_config_from_args(args)
                self.assertEqual(args.streaming_multires_low_divisor, 8)
                self.assertEqual(
                    runtime.static_options["mapping_resolution_tier"], "adaptive"
                )
                self.assertEqual(
                    runtime.static_options["mapping_resolution_min_tier"], "eighth"
                )
                supported_tiers = tuple(
                    tier
                    for tier in ResolutionTier
                    if tier.downsample_factor
                    <= ResolutionTier(
                        runtime.static_options["mapping_resolution_min_tier"]
                    ).downsample_factor
                )
                self.assertEqual(supported_tiers, tuple(ResolutionTier))


if __name__ == "__main__":
    unittest.main()


def _bootstrap_bind_tier_spy(monkeypatch, policy):
    """Drive the real `_bind_ready_bootstrap_cameras` with a stub runtime and
    record the `tier=` each `bind_ready_camera` call receives."""
    from types import SimpleNamespace
    from eligsir.runtime import production_application as pa
    from eligsir.runtime import ready_camera_binder

    recorded = []
    monkeypatch.setattr(
        ready_camera_binder, "bind_ready_camera",
        lambda binding, tier="native", **kw: recorded.append(tier) or ("camera", tier),
    )
    runtime = object.__new__(pa.CanonicalMappingRuntime)
    bindings = [SimpleNamespace(handoff_to_compute=lambda owner: None) for _ in range(3)]
    monkeypatch.setattr(runtime, "_ensure_ready_transfer_owner", lambda: object(), raising=False)
    monkeypatch.setattr(runtime, "bind_ready_only_sources", lambda selection: bindings, raising=False)
    # Store the policy on the runtime so _bind_ready_bootstrap_cameras can read it
    runtime._prepared_tier_policy = policy
    cameras = pa.CanonicalMappingRuntime._bind_ready_bootstrap_cameras(
        runtime, selection=object()
    )
    assert len(cameras) == 3
    return recorded


def test_bootstrap_bind_uses_quarter_under_fixed_quarter_policy(monkeypatch):



    policy = PreparedTierPolicy.from_supported_tiers((ResolutionTier.QUARTER,))
    assert _bootstrap_bind_tier_spy(monkeypatch, policy) == ["quarter"] * 3


def test_bootstrap_bind_uses_native_under_adaptive_policy_and_without_policy(monkeypatch):



    adaptive = PreparedTierPolicy.from_supported_tiers(
        (ResolutionTier.NATIVE, ResolutionTier.HALF, ResolutionTier.QUARTER)
    )
    assert _bootstrap_bind_tier_spy(monkeypatch, adaptive) == ["native"] * 3
    assert _bootstrap_bind_tier_spy(monkeypatch, None) == ["native"] * 3
