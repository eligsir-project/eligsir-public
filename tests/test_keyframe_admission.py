"""Tests for KeyframeAdmitter (Milestone 4)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from eligsir.mapping.scheduler import (
    FrameInfo,
    KeyframeAdmitter,
    KeyframeAdmitterConfig,
    KeyframeDecision,
)


def test_scheduler_admitter_import_does_not_require_torch() -> None:
    """The dependency-light scheduler API must remain usable in CPU CI."""
    code = """
import sys
sys.modules["torch"] = None
from eligsir.mapping.scheduler import KeyframeAdmitter
assert KeyframeAdmitter.__name__ == "BudgetedCohortKeyframeAdmitter"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


class _FakeFrame:
    def __init__(self, index: int, c2w: np.ndarray) -> None:
        self.index = index
        self.c2w = c2w


def _identity_c2w() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def _translated_c2w(tx: float, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


class TestCheapGate:
    def test_utils_keyframe_admission_reexports_package_admitter(self) -> None:
        from eligsir.utils.keyframe_admission import KeyframeAdmitter as CompatKeyframeAdmitter

        assert CompatKeyframeAdmitter is KeyframeAdmitter

    def test_first_frame_with_no_render_falls_back(self) -> None:
        """First frame with no reference: the cheap gate returns 0,0,0;
        with no render_fn the admitter falls back to the cheap path and
        rejects (not moved, not forced)."""
        a = KeyframeAdmitter(KeyframeAdmitterConfig())
        decision, render_pkg = a.admit(cam=None, frame=_FakeFrame(0, _identity_c2w()))
        assert isinstance(decision, KeyframeDecision)
        assert decision.admitted is False
        assert decision.reason == "render_failed"
        assert render_pkg is None

    def test_motion_pregate_skips_render_for_non_moved_frame(self) -> None:
        """With motion_pregate on, a frame that did not move past the motion
        thresholds is rejected WITHOUT invoking the (expensive) render_fn."""
        calls = []

        def _render_spy(*args, **kwargs):
            calls.append((args, kwargs))
            return {"alpha": None}

        cfg = KeyframeAdmitterConfig(min_translation_m=0.05, min_rotation_deg=5.0,
                                     max_gap=100, motion_pregate=True)
        a = KeyframeAdmitter(cfg, render_fn=_render_spy)
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        # Frame 1: tiny translation (0.001 m << 0.05), gap=1 << max_gap → not moved.
        decision, render_pkg = a.admit(cam=None, frame=_FakeFrame(1, _translated_c2w(0.001)))
        assert decision.admitted is False
        assert decision.reason == "no_motion"
        assert render_pkg is None
        assert calls == []  # render_fn never invoked

    def test_motion_pregate_still_renders_moved_frame(self) -> None:
        """A frame that DID move past the thresholds still goes through the
        render gate even with motion_pregate on."""
        calls = []

        def _render_spy(*args, **kwargs):
            calls.append(1)
            return {"alpha": None}

        cfg = KeyframeAdmitterConfig(min_translation_m=0.05, min_rotation_deg=5.0,
                                     max_gap=100, motion_pregate=True)
        a = KeyframeAdmitter(cfg, render_fn=_render_spy)
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        # Frame 1: 0.5 m translation >> 0.05 → moved → render gate runs.
        a.admit(cam=None, frame=_FakeFrame(1, _translated_c2w(0.5)))
        assert len(calls) == 1  # render_fn invoked for the moved frame

    def test_motion_pregate_admits_first_frame_with_no_baseline(self) -> None:
        """With no baseline keyframe yet, the pregate must NOT short-circuit the
        first frame as no_motion (delta is 0 only because there's no reference).
        It must fall through to the render gate so the baseline initializes —
        otherwise every frame deadlocks at no_motion and the map collapses."""
        calls = []

        def _render_spy(*args, **kwargs):
            calls.append(1)
            return {"alpha": None}

        cfg = KeyframeAdmitterConfig(motion_pregate=True, max_gap=1000)
        a = KeyframeAdmitter(cfg, render_fn=_render_spy)
        # NO note_train_frame: baseline is None.
        d, _ = a.admit(cam=None, frame=_FakeFrame(0, _identity_c2w()))
        assert d.reason != "no_motion", "first frame must not be pregate-rejected"
        assert len(calls) == 1, "first frame must reach the render gate to set baseline"

    def test_motion_pregate_accumulates_motion_across_rejections(self) -> None:
        """Rejected (no_motion) frames must NOT advance the pose baseline.
        Sub-threshold per-frame motion that cumulatively exceeds the threshold
        must eventually be admitted (delta measured vs last ADMITTED keyframe,
        not vs the previous rejected frame). Regression for the collapse-to-
        bootstrap bug where _record_admission ran on rejected frames."""
        calls = []

        def _render_spy(*args, **kwargs):
            calls.append(1)
            return {"alpha": None}

        cfg = KeyframeAdmitterConfig(min_translation_m=0.05, min_rotation_deg=5.0,
                                     max_gap=1000, motion_pregate=True)
        a = KeyframeAdmitter(cfg, render_fn=_render_spy)
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))  # baseline at origin
        # Frames each +0.02 m (below 0.05). Cumulative: 0.02, 0.04, 0.06, 0.08.
        reasons = []
        for i, tx in enumerate([0.02, 0.04, 0.06, 0.08], start=1):
            d, _ = a.admit(cam=None, frame=_FakeFrame(i, _translated_c2w(tx)))
            reasons.append((tx, d.reason))
        assert reasons[0][1] == "no_motion"
        assert reasons[1][1] == "no_motion"
        # Once cumulative motion >= 0.05 the render gate must run (not no_motion).
        assert len(calls) >= 1, f"render never ran despite accumulated motion: {reasons}"

    def test_max_gap_forces_admission(self) -> None:
        a = KeyframeAdmitter(KeyframeAdmitterConfig(max_gap=3))
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        # Frame 5 is gap=5 >= max_gap=3 → forced admit.
        decision, _ = a.admit(cam=None, frame=_FakeFrame(5, _translated_c2w(0.01)))
        assert decision.admitted is True
        assert decision.reason == "max_gap"
        assert decision.forced is True
        # Pose-delta fields are populated for downstream telemetry.
        assert decision.pose_delta_m == pytest.approx(0.01, rel=1e-6)
        assert decision.gap == 5

    def test_pose_delta_admits_without_render(self) -> None:
        """If the render fails, the cheap gate still admits a frame
        with sufficient pose delta."""
        a = KeyframeAdmitter(
            KeyframeAdmitterConfig(min_translation_m=0.05, min_rotation_deg=5.0, max_gap=1000),
        )
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        # Translation 0.1 m > 0.05 m threshold → admit (cheap path).
        decision, _ = a.admit(cam=None, frame=_FakeFrame(1, _translated_c2w(0.1)))
        assert decision.admitted is True
        assert decision.reason == "pose_delta"
        assert decision.moved is True
        assert decision.pose_delta_m == pytest.approx(0.1, rel=1e-6)
        assert decision.pose_delta_deg == pytest.approx(0.0, abs=1e-6)
        assert decision.gap == 1

    def test_redundant_no_pose_delta_no_render_rejected(self) -> None:
        a = KeyframeAdmitter(
            KeyframeAdmitterConfig(min_translation_m=0.05, min_rotation_deg=5.0, max_gap=1000,
                                   motion_pregate=False),
        )
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        # Identity translation, no render → cheap path: not moved, not forced.
        decision, render_pkg = a.admit(cam=None, frame=_FakeFrame(1, _identity_c2w()))
        assert decision.admitted is False
        assert decision.reason == "render_failed"
        assert render_pkg is None

    def test_cheap_score_scaled_by_pose_delta(self) -> None:
        a = KeyframeAdmitter(
            KeyframeAdmitterConfig(min_translation_m=0.05, min_rotation_deg=5.0, max_gap=1000),
        )
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        decision, _ = a.admit(cam=None, frame=_FakeFrame(1, _translated_c2w(0.1)))
        # 0.1 / 0.05 = 2.0 → cheap_score = 0.4 * 2.0 = 0.8
        assert decision.cheap_score == pytest.approx(0.8, rel=0.1)


class TestExpensiveGate:
    def _stub_render_fn(self, overlap_value: float):
        """Return a real torch tensor (not a mock) for alpha so the
        ``(alpha.detach() > thr).float().mean().item()`` chain in the
        admitter works without surprises.

        Uses a 100×1 tensor filled with the requested overlap so the
        mean approximates the requested fraction.
        """
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        def _render(cam, g, pipe, bg, render_depth=True):
            n = 100
            n_above = int(round(overlap_value * n))
            data = [1.0] * n_above + [0.0] * (n - n_above)
            return {"alpha": torch.tensor(data).reshape(n, 1), "_stub": True}

        return _render

    def test_low_overlap_rejected(self) -> None:
        a = KeyframeAdmitter(
            KeyframeAdmitterConfig(min_overlap=0.25, min_translation_m=10.0, min_rotation_deg=1000.0, max_gap=1000,
                                   motion_pregate=False),
        )
        a._render_fn = self._stub_render_fn(0.1)  # 10 % overlap < 25 %
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        decision, render_pkg = a.admit(cam=None, frame=_FakeFrame(1, _identity_c2w()))
        assert decision.admitted is False
        assert decision.reason == "low_overlap"
        # Render pkg is propagated so the caller can cache it.
        assert render_pkg is not None and render_pkg.get("_stub") is True

    def test_low_overlap_force_admit_disabled_by_default(self) -> None:
        """Default low_overlap_force_admit_iters=0 must reproduce the exact
        prior behaviour (byte-identical): repeated low_overlap candidates
        stay rejected forever, no matter how long the admitter has starved."""
        a = KeyframeAdmitter(
            KeyframeAdmitterConfig(min_overlap=0.25, min_translation_m=10.0,
                                    min_rotation_deg=1000.0, max_gap=100000,
                                    motion_pregate=False),
        )
        a._render_fn = self._stub_render_fn(0.1)  # always 10 % overlap < 25 %
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        for i in range(1, 500):
            decision, _ = a.admit(cam=None, frame=_FakeFrame(i, _identity_c2w()))
            assert decision.admitted is False
            assert decision.reason == "low_overlap"

    def test_low_overlap_force_admit_relieves_starvation(self) -> None:
        """With low_overlap_force_admit_iters set, a persistent low_overlap
        candidate is admitted once starved long enough — regression test for
        the fr1_desk frontier-starvation trap (596-frame desk-orbit scene,
        56% of candidates rejected low_overlap, 7.0% total admission rate)."""
        a = KeyframeAdmitter(
            KeyframeAdmitterConfig(min_overlap=0.25, min_translation_m=10.0,
                                    min_rotation_deg=1000.0, max_gap=100000,
                                    low_overlap_force_admit_iters=5,
                                    motion_pregate=False),
        )
        a._render_fn = self._stub_render_fn(0.1)  # always 10 % overlap < 25 %
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        reasons = []
        for i in range(1, 8):
            decision, _ = a.admit(cam=None, frame=_FakeFrame(i, _identity_c2w()))
            reasons.append(decision.reason)
            if decision.admitted:
                break
        assert "low_overlap_force_admit" in reasons, reasons
        assert reasons[:-1] == ["low_overlap"] * (len(reasons) - 1)

    def test_low_overlap_force_admit_does_not_affect_max_gap_or_moved(self) -> None:
        """Starvation relief must only intervene on the pure low_overlap
        rejection path; forced (max_gap) and moved admissions are unaffected."""
        a = KeyframeAdmitter(
            KeyframeAdmitterConfig(min_overlap=0.25, min_translation_m=0.05,
                                    min_rotation_deg=5.0, max_gap=3,
                                    low_overlap_force_admit_iters=5),
        )
        a._render_fn = self._stub_render_fn(0.1)
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        # gap=5 >= max_gap=3 → forced admit regardless of overlap or starvation state.
        decision, _ = a.admit(cam=None, frame=_FakeFrame(5, _translated_c2w(0.01)))
        assert decision.admitted is True
        assert decision.reason == "max_gap"

    def test_novel_view_admitted(self) -> None:
        a = KeyframeAdmitter(
            KeyframeAdmitterConfig(
                min_overlap=0.25,
                max_overlap=0.90,
                min_translation_m=10.0,  # disable pose gate
                min_rotation_deg=1000.0,
                max_gap=1000,
                motion_pregate=False,
            ),
        )
        a._render_fn = self._stub_render_fn(0.5)  # 50 % overlap → novel
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        decision, render_pkg = a.admit(cam=None, frame=_FakeFrame(1, _identity_c2w()))
        assert decision.admitted is True
        assert decision.reason == "new_content"
        assert decision.novel is True
        assert render_pkg is not None

    def test_full_overlap_rejected_redundant(self) -> None:
        a = KeyframeAdmitter(
            KeyframeAdmitterConfig(
                min_overlap=0.25,
                max_overlap=0.90,
                min_translation_m=10.0,
                min_rotation_deg=1000.0,
                max_gap=1000,
                motion_pregate=False,
            ),
        )
        a._render_fn = self._stub_render_fn(0.95)  # 95 % overlap > 90 % max
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        decision, _ = a.admit(cam=None, frame=_FakeFrame(1, _identity_c2w()))
        assert decision.admitted is False
        # high overlap + no pose delta + no forced → "redundant"
        assert decision.reason in ("redundant", "low_overlap")


class TestIndexTracking:
    def test_keyframe_indices_appended(self) -> None:
        a = KeyframeAdmitter(
            KeyframeAdmitterConfig(min_translation_m=0.05, min_rotation_deg=5.0, max_gap=1000),
        )
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        d1, _ = a.admit(cam=None, frame=_FakeFrame(1, _translated_c2w(0.1)))
        d2, _ = a.admit(cam=None, frame=_FakeFrame(2, _translated_c2w(0.2)))
        d3, _ = a.admit(cam=None, frame=_FakeFrame(3, _translated_c2w(0.3)))
        assert d1.admitted and d2.admitted and d3.admitted
        assert a.keyframe_indices == [1, 2, 3]
        assert a.keyframes_admitted == 3

    def test_reset_clears_indices(self) -> None:
        a = KeyframeAdmitter(KeyframeAdmitterConfig())
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        a.admit(cam=None, frame=_FakeFrame(1, _translated_c2w(0.1)))
        assert a.keyframes_admitted == 1
        a.reset()
        assert a.keyframe_indices == []
        assert a.keyframes_admitted == 0

    def test_max_history_caps(self) -> None:
        a = KeyframeAdmitter(
            KeyframeAdmitterConfig(
                min_translation_m=0.05,
                min_rotation_deg=5.0,
                max_gap=1000,
                max_history=4,
            ),
        )
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        for i in range(1, 8):
            a.admit(cam=None, frame=_FakeFrame(i, _translated_c2w(0.1 * i)))
        # 7 admissions, but max_history=4 → after the cap the list is
        # half-trimmed; latest 4 indices should be present.
        idxs = a.keyframe_indices
        assert len(idxs) <= 4
        assert idxs[-1] == 7


class TestConfigClamping:
    """``KeyframeAdmitterConfig`` must clamp degenerate operator inputs."""

    def test_max_gap_clamped_to_at_least_one(self) -> None:
        cfg = KeyframeAdmitterConfig(max_gap=0)
        assert cfg.max_gap == 1
        cfg = KeyframeAdmitterConfig(max_gap=-7)
        assert cfg.max_gap == 1
        # Sane values are preserved.
        cfg = KeyframeAdmitterConfig(max_gap=12)
        assert cfg.max_gap == 12

    def test_max_history_clamped_to_at_least_one(self) -> None:
        cfg = KeyframeAdmitterConfig(max_history=0)
        assert cfg.max_history == 1
        cfg = KeyframeAdmitterConfig(max_history=-3)
        assert cfg.max_history == 1


class TestQualityGate:
    """Coverage for the optional quality-scoring path."""

    def _stub_render_fn(self, overlap_value: float):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        def _render(cam, g, pipe, bg, render_depth=True):
            n = 100
            n_above = int(round(overlap_value * n))
            data = [1.0] * n_above + [0.0] * (n - n_above)
            return {"alpha": torch.tensor(data).reshape(n, 1), "_stub": True}

        return _render

    def test_quality_enabled_can_reject(self) -> None:
        """When ``quality_enabled=True`` and the scorer returns a
        reject, the admitter must honour the rejection and report
        ``reason="quality_reject"``."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")

        cfg = KeyframeAdmitterConfig(
            min_overlap=0.25,
            max_overlap=0.90,
            # Disable pose gate so the only admit signal is the novel
            # overlap. quality_enabled=True then funnels the decision
            # through ``score_keyframe`` which we patch to reject.
            min_translation_m=10.0,
            min_rotation_deg=1000.0,
            max_gap=1000,
            quality_enabled=True,
            motion_pregate=False,
        )
        a = KeyframeAdmitter(config=cfg)
        a._render_fn = self._stub_render_fn(0.5)

        # Stub the quality evaluator and loaders, and patch the
        # module-level ``score_keyframe`` to always reject — this
        # exercises the quality-reject branch end-to-end.
        def _eval_quality(rgb, depth, frame=None):
            # Provide both attributes ``quality_weight`` and
            # ``accepted`` so score_keyframe's pre-check passes.
            return type("Q", (), {"quality_weight": 0.0, "accepted": True, "reject_reason": ""})()

        import eligsir.mapping.scheduler.keyframes as ka_mod
        from eligsir.mapping.scheduler import KeyframeScore
        original_score = ka_mod.score_keyframe

        def _force_reject(*, translation_m, rotation_deg, gap, overlap,
                          quality=None, render_residual=0.0,
                          depth_disagreement=0.0, uncertainty=0.0,
                          new_surface_ratio=0.0, config=None):
            res = original_score(
                translation_m=translation_m,
                rotation_deg=rotation_deg,
                gap=gap,
                overlap=overlap,
                quality=quality,
                render_residual=render_residual,
                depth_disagreement=depth_disagreement,
                uncertainty=uncertainty,
                new_surface_ratio=new_surface_ratio,
                config=config,
            )
            return KeyframeScore(
                pose_delta_score=res.pose_delta_score,
                rotation_delta_score=res.rotation_delta_score,
                overlap_score=res.overlap_score,
                new_surface_score=res.new_surface_score,
                render_residual_score=res.render_residual_score,
                depth_disagreement_score=res.depth_disagreement_score,
                quality_score=0.0,
                uncertainty_score=res.uncertainty_score,
                final_score=res.final_score,
                decision=False,
                reason="quality_reject",
                terms=dict(res.terms),
            )

        a._evaluate_quality = _eval_quality
        a._load_rgb = lambda f: None
        a._load_depth_m = lambda f: None
        ka_mod.score_keyframe = _force_reject
        try:
            a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
            decision, _ = a.admit(cam=None, frame=_FakeFrame(1, _identity_c2w()))
            assert decision.admitted is False
            assert decision.reason == "quality_reject"
        finally:
            ka_mod.score_keyframe = original_score

    def test_quality_disabled_uses_cheap_gate(self) -> None:
        """``quality_enabled=False`` (the default) must NOT invoke
        ``score_keyframe`` even when the evaluate/load helpers are
        bound. This guarantees we don't change behaviour for the
        default path."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")

        cfg = KeyframeAdmitterConfig(
            quality_enabled=False,
            min_translation_m=10.0,
            min_rotation_deg=1000.0,
            max_gap=1000,
            motion_pregate=False,
        )
        a = KeyframeAdmitter(config=cfg)
        a._render_fn = self._stub_render_fn(0.5)
        a._evaluate_quality = lambda rgb, depth, frame=None: None
        a._load_rgb = lambda f: None
        a._load_depth_m = lambda f: None

        import eligsir.mapping.scheduler.keyframes as ka_mod
        calls = {"count": 0}
        original_score = ka_mod.score_keyframe

        def _spy_score(*args, **kwargs):
            calls["count"] += 1
            return original_score(*args, **kwargs)

        ka_mod.score_keyframe = _spy_score
        try:
            a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
            decision, _ = a.admit(cam=None, frame=_FakeFrame(1, _identity_c2w()))
            assert decision.admitted is True
            assert decision.reason == "new_content"
            assert calls["count"] == 0
        finally:
            ka_mod.score_keyframe = original_score

    def test_quality_enabled_does_not_veto_bootstrap_admission(self) -> None:
        """Regression test (2026-07-31): before a baseline keyframe exists
        (``_last_train_c2w is None``), quality scoring must NOT run --
        ``score_keyframe``'s composite score sees t_m=r_deg=0 for the very
        first candidate (no prior pose to diff against) and can reject it
        even with a perfect quality_weight=1.0, which means
        ``_record_admission`` never fires, the baseline never establishes,
        and every subsequent frame reads (0, 0, 0) forever -- a permanent
        admission deadlock. Reproduced live on floor2 bag replay: 0/1991
        admitted with --streaming_keyframe_quality_enabled vs ~186/1991
        with it off, identical everything else. Cold-start admission (no
        ``note_train_frame`` call) must behave identically whether
        quality_enabled is True or False."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")

        def _eval_quality_perfect(rgb, depth, frame=None):
            return type("Q", (), {"quality_weight": 1.0, "accepted": True, "reject_reason": ""})()

        results = {}
        for quality_enabled in (False, True):
            cfg = KeyframeAdmitterConfig(
                min_translation_m=0.05,
                min_rotation_deg=5.0,
                min_overlap=0.25,
                max_overlap=0.90,
                max_gap=10,
                motion_pregate=True,
                quality_enabled=quality_enabled,
            )
            a = KeyframeAdmitter(config=cfg)
            a._render_fn = self._stub_render_fn(0.5)
            if quality_enabled:
                a._evaluate_quality = _eval_quality_perfect
                a._load_rgb = lambda f: None
                a._load_depth_m = lambda f: None
            # No note_train_frame() call -- genuine cold start, mirrors the
            # very first frame of a real run.
            decision, _ = a.admit(cam=None, frame=_FakeFrame(0, _identity_c2w()))
            results[quality_enabled] = decision

        assert results[False].admitted is True
        assert results[True].admitted is True
        assert results[False].reason == results[True].reason == "new_content"


class TestIntegration:
    def test_decision_dataclass_fields(self) -> None:
        a = KeyframeAdmitter(
            KeyframeAdmitterConfig(min_translation_m=0.05, min_rotation_deg=5.0, max_gap=1000),
        )
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        decision, _ = a.admit(cam=None, frame=_FakeFrame(1, _translated_c2w(0.2)))
        # Verify the dataclass fields are populated.
        assert decision.admitted is True
        assert decision.keyframe_index == 1
        assert decision.moved is True
        assert decision.cheap_score > 0.0
        # Pose-delta fields are populated.
        assert decision.pose_delta_m == pytest.approx(0.2, rel=1e-6)
        assert decision.gap == 1


class TestPregate:
    """KeyframeAdmitter.pregate() -- cheap check usable before cam is built."""

    def test_disabled_returns_none(self) -> None:
        cfg = KeyframeAdmitterConfig(motion_pregate=False)
        a = KeyframeAdmitter(cfg)
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        assert a.pregate(_FakeFrame(1, _translated_c2w(0.001))) is None

    def test_no_baseline_returns_none(self) -> None:
        cfg = KeyframeAdmitterConfig(motion_pregate=True)
        a = KeyframeAdmitter(cfg)
        # No note_train_frame call: baseline is None.
        assert a.pregate(_FakeFrame(0, _identity_c2w())) is None

    def test_velocity_adaptive_returns_none(self) -> None:
        cfg = KeyframeAdmitterConfig(motion_pregate=True, velocity_adaptive=True)
        a = KeyframeAdmitter(cfg)
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        assert a.pregate(_FakeFrame(1, _translated_c2w(0.001))) is None

    def test_coverage_gain_gate_returns_none_even_when_not_moved(self) -> None:
        """When coverage_gain_gate is on, pregate() must ALWAYS defer to
        admit(), even for a frame that would otherwise be pregate-rejected as
        no_motion -- because a caller that skips record_fn for this frame
        would also skip its record_frame_observation() call, and
        admit()'s coverage_gain_gate branch queries self._observation_db
        (via _db_diversity_score) to decide OTHER, later frames' admission.
        See the pregate() docstring's coverage_gain_gate section."""
        cfg = KeyframeAdmitterConfig(min_translation_m=0.05, min_rotation_deg=5.0,
                                     max_gap=100, motion_pregate=True,
                                     coverage_gain_gate=True)
        a = KeyframeAdmitter(cfg)
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        assert a.pregate(_FakeFrame(1, _translated_c2w(0.001))) is None

    def test_forced_frame_returns_none(self) -> None:
        cfg = KeyframeAdmitterConfig(motion_pregate=True, max_gap=1)
        a = KeyframeAdmitter(cfg)
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        # gap (1) >= max_gap (1) -> forced; must defer to admit(), not pre-reject.
        assert a.pregate(_FakeFrame(1, _translated_c2w(0.001))) is None

    def test_moved_frame_returns_none(self) -> None:
        cfg = KeyframeAdmitterConfig(min_translation_m=0.05, min_rotation_deg=5.0,
                                     max_gap=100, motion_pregate=True)
        a = KeyframeAdmitter(cfg)
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        assert a.pregate(_FakeFrame(1, _translated_c2w(0.5))) is None

    def test_non_moved_non_forced_rejects(self) -> None:
        cfg = KeyframeAdmitterConfig(min_translation_m=0.05, min_rotation_deg=5.0,
                                     max_gap=100, motion_pregate=True)
        a = KeyframeAdmitter(cfg)
        a.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        decision = a.pregate(_FakeFrame(1, _translated_c2w(0.001)))
        assert decision is not None
        assert decision.admitted is False
        assert decision.reason == "no_motion"

    def test_agrees_with_admit_no_motion_case(self) -> None:
        """Where pregate() decides (non-forced, has-baseline, not velocity_adaptive,
        not moved), it must match what admit() would have decided, without
        invoking the render_fn admit() would otherwise call."""
        calls = []

        def _render_spy(*args, **kwargs):
            calls.append(1)
            return {"alpha": None}

        cfg = KeyframeAdmitterConfig(min_translation_m=0.05, min_rotation_deg=5.0,
                                     max_gap=100, motion_pregate=True)
        a_pre = KeyframeAdmitter(cfg, render_fn=_render_spy)
        a_pre.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        pre_decision = a_pre.pregate(_FakeFrame(1, _translated_c2w(0.001)))

        a_full = KeyframeAdmitter(cfg, render_fn=_render_spy)
        a_full.note_train_frame(FrameInfo(c2w=_identity_c2w(), index=0))
        full_decision, _ = a_full.admit(cam=None, frame=_FakeFrame(1, _translated_c2w(0.001)))

        assert pre_decision is not None
        assert pre_decision.admitted == full_decision.admitted
        assert pre_decision.reason == full_decision.reason
        assert pre_decision.pose_delta_m == pytest.approx(full_decision.pose_delta_m)
        assert calls == []  # neither path invoked the render gate


class TestAdmissionQualityWeightForStats:
    """Regression tests for the train_streaming.py stats-dict assembly gap
    (2026-07-31) -- the bootstrap-skip fix in keyframes.py's admit() means
    KeyframeDecision.quality can legitimately be None, and the closure that
    turns a decision into the row persisted to keyframe_admission_log.jsonl
    must handle that without crashing. This is the exact gap a
    KeyframeAdmitter-only unit test cannot catch: it lives in the
    train_streaming.py glue code, not in keyframes.py."""

    def test_none_quality_with_quality_enabled_does_not_crash(self) -> None:
        import train_streaming

        decision = KeyframeDecision(
            admitted=True,
            reason="new_content",
            score=1.0,
            cheap_score=1.0,
            quality=None,
        )
        result = train_streaming._admission_quality_weight_for_stats(
            decision, quality_enabled=True
        )
        assert result is None

    def test_real_quality_with_quality_enabled_passes_through(self) -> None:
        import train_streaming

        decision = KeyframeDecision(
            admitted=True,
            reason="new_content",
            score=1.0,
            cheap_score=1.0,
            quality=0.73,
        )
        result = train_streaming._admission_quality_weight_for_stats(
            decision, quality_enabled=True
        )
        assert result == pytest.approx(0.73)

    def test_quality_disabled_returns_none_regardless_of_decision(self) -> None:
        import train_streaming

        decision = KeyframeDecision(
            admitted=True,
            reason="new_content",
            score=1.0,
            cheap_score=1.0,
            quality=0.5,
        )
        result = train_streaming._admission_quality_weight_for_stats(
            decision, quality_enabled=False
        )
        assert result is None


class TestIssue933StarvationClock:
    """Issue #933: the hard starvation ceiling must stay reachable even when
    no keyframe has ever been admitted.

    Before the fix, ``_pose_delta_from_last_train`` returned ``gap = 0``
    whenever ``_last_train_c2w is None``. If an *information* gate (coverage
    here; parallax / render-overlap can starve the same way) rejected the very
    first candidate, no baseline was ever recorded, so every later candidate
    also read ``gap == 0`` and neither the soft ``max_gap`` force nor the hard
    ``max_gap * max_gap_hard_mult`` ceiling could ever fire. Measured live on
    ``i792novthresh_fr2desk``: 2005 candidates, 0 admitted, all gaps 0, 9.84 dB.
    """

    def _stub_render_fn(self, overlap_value: float):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        def _render(cam, g, pipe, bg, render_depth=True):
            n = 100
            n_above = int(round(overlap_value * n))
            data = [1.0] * n_above + [0.0] * (n - n_above)
            return {"alpha": torch.tensor(data).reshape(n, 1), "_stub": True}

        return _render

    def _starved_admitter(self) -> KeyframeAdmitter:
        cfg = KeyframeAdmitterConfig(
            # Motion can never satisfy the gate, so `moved` is False for every
            # candidate -- exactly as it is pre-baseline in the real failure.
            min_translation_m=10.0,
            min_rotation_deg=1000.0,
            min_overlap=0.25,
            max_overlap=0.90,
            max_gap=5,
            max_gap_hard_mult=2.0,
            motion_pregate=True,
            coverage_gain_gate=True,
            max_gap_require_novelty=True,
            min_new_voxel_frac=0.5,
        )
        a = KeyframeAdmitter(config=cfg)
        # 99 % overlap: above max_overlap, so `novel` is False too. Merely
        # falling through to the render gate therefore CANNOT rescue this
        # frame -- only the hard starvation ceiling can, which is the point.
        a._render_fn = self._stub_render_fn(0.99)
        a._occupancy_fn = lambda pts: np.zeros(len(pts), dtype=bool)
        a._load_depth_m = lambda f: np.ones((4, 4), dtype=np.float32)
        # Every candidate is fully redundant, so the coverage gate always
        # rejects and the novelty gate always withdraws a soft force.
        a._coverage_new_voxel_frac = lambda cam, frame: 0.0
        return a

    def test_hard_ceiling_reachable_with_no_baseline(self) -> None:
        a = self._starved_admitter()
        assert a.starvation_clock_source == "starvation_origin"

        gaps: list = []
        first_admit = None
        for i in range(20):
            decision, _ = a.admit(cam=None, frame=_FakeFrame(i, _identity_c2w()))
            gaps.append(decision.gap)
            if decision.admitted and first_admit is None:
                first_admit = (i, decision)

        # The clock advanced instead of freezing at 0. The pre-fix failure was
        # exactly `set(gaps) == {0}` with `first_admit is None`.
        assert set(gaps) != {0}
        assert first_admit is not None, "hard starvation ceiling never fired"

        idx, decision = first_admit
        # max_gap(5) * max_gap_hard_mult(2.0) == 10, and the starvation origin
        # is candidate 0, so the ceiling is crossed at candidate 10.
        assert idx == 10
        assert decision.gap == 10
        assert decision.reason == "max_gap"
        assert decision.forced is True
        # Strictly increasing up to (and including) the first admission.
        assert gaps[:11] == list(range(11))

    def test_pregate_not_weakened_pre_baseline(self) -> None:
        """The clock is source-frame based, so pose deltas stay 0.0 with no
        baseline: no synthetic motion is invented and `moved` stays False."""
        a = self._starved_admitter()
        for i in range(10):
            decision, _ = a.admit(
                cam=None, frame=_FakeFrame(i, _translated_c2w(i * 3.0))
            )
            assert decision.pose_delta_m == 0.0
            assert decision.pose_delta_deg == 0.0
            assert decision.moved is False
            assert decision.admitted is False

    def test_clock_resets_on_first_admission(self) -> None:
        a = self._starved_admitter()
        decision = None
        for i in range(11):
            decision, _ = a.admit(cam=None, frame=_FakeFrame(i, _identity_c2w()))
        assert decision is not None and decision.admitted is True
        assert a.starvation_clock_source == "last_keyframe"

        # With a baseline the gap is measured from it again, so the next
        # candidate restarts at 1 rather than continuing from 10.
        nxt, _ = a.admit(cam=None, frame=_FakeFrame(11, _identity_c2w()))
        assert nxt.gap == 1
        assert nxt.admitted is False

    def test_inert_when_first_candidate_is_admitted(self) -> None:
        """Healthy case: nothing about this fix is observable. The coverage
        gate passes, the first frame is admitted normally, and the starvation
        origin never governs any decision."""
        a = self._starved_admitter()
        a._coverage_new_voxel_frac = lambda cam, frame: 1.0
        a._render_fn = self._stub_render_fn(0.5)  # inside [min_overlap, max_overlap]
        decision, _ = a.admit(cam=None, frame=_FakeFrame(0, _identity_c2w()))
        assert decision.admitted is True
        assert decision.gap == 0
        assert a.starvation_clock_source == "last_keyframe"
