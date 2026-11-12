"""Issue #601 -- holdouts must never be subject to backpressure.

The Run Contracts require the holdout selection to be **identical across
admission modes**, and a holdout's ``is_holdout`` status is determined
*inside* ``StreamingScene.ingest_next_frame``. A pre-materialization gate
that calls ``skip_next_frame()`` therefore prevents the frame from ever
being realized as a test camera at all -- silently shrinking ``n_test``,
changing ``realized_holdout_hash``, and making any ``--baseline_run``
comparison (including #601's own performance-acceptance run) refuse or
lie. At aggressive bounds it can drive ``n_test`` to 0, i.e.
``eval_valid == false``.

These tests pin the fix on both sides:

1. the holdout rule is ONE implementation, exposed as a cheap public
   predicate (``StreamingScene.holdout_status``) that ``ingest_next_frame``
   itself calls -- no second, divergent copy;
2. the drain-loop gate hard-BYPASSES holdouts/guards rather than routing
   them through the owner (``METADATA_ONLY`` would be the same bug, since
   its ``materializes`` is also False).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

STREAMING_SCENE = pathlib.Path("scene/streaming_scene.py")
TRAIN_STREAMING = pathlib.Path("train_streaming.py")


def _strip_comments(text: str) -> str:
    """Drop ``#`` comment tails so an assertion about CODE is not satisfied
    (or defeated) by prose in a comment."""
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


class _Frame:
    def __init__(self, index, source_ordinal=None):
        self.index = index
        self.source_ordinal = source_ordinal


class _Args:
    streaming_eval_hold = 8


class _Scene:
    """Minimal stand-in exercising the real predicate's logic paths."""

    holdout_status = None  # bound below

    def __init__(self, holdout_indices=None, guard_indices=None):
        self.args = _Args()
        self._holdout_indices = holdout_indices
        self._guard_indices = guard_indices or set()


def _bind_real_predicate():
    """Extract and exec the real ``holdout_status`` body so the test runs
    without importing ``streaming_scene`` (which pulls in torch).

    issue #1302: the method's live branch now delegates its arithmetic to
    ``utils.holdout_rule.online_every_nth_holdout`` (itself deliberately
    torch-free -- see that module's docstring) instead of inlining it, so
    that predicate is injected into the exec namespace here too, keeping
    this test's "run the body without importing streaming_scene" property
    intact.
    """
    from eligsir.utils.holdout_rule import online_every_nth_holdout

    tree = ast.parse(STREAMING_SCENE.read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "holdout_status"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Tuple": tuple, "online_every_nth_holdout": online_every_nth_holdout}
    exec(compile(module, "<holdout_status>", "exec"), namespace)
    return namespace["holdout_status"]


@pytest.fixture(scope="module")
def holdout_status():
    return _bind_real_predicate()


class TestPredicateIsTheSingleImplementation:
    def test_ingest_next_frame_calls_the_predicate(self):
        src = STREAMING_SCENE.read_text()
        assert "def holdout_status(self, frame)" in src, (
            "the cheap public holdout predicate is missing"
        )
        assert "is_holdout, is_guard = self.holdout_status(frame)" in src, (
            "ingest_next_frame no longer calls holdout_status() -- the rule has "
            "forked into a second, divergent copy"
        )

    def test_ingest_next_frame_does_not_recompute_the_rule(self):
        """``ingest_next_frame`` must delegate, not carry its own copy.

        (The up-front reservation that BUILDS ``_holdout_indices`` is a
        legitimately separate site -- this pins that the per-frame path
        is not a second implementation of the same predicate.)"""
        tree = ast.parse(STREAMING_SCENE.read_text())
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "ingest_next_frame"
        )
        body = ast.unparse(fn)
        assert "self.holdout_status(frame)" in body
        assert "eval_hold == 0" not in body, (
            "ingest_next_frame recomputes the every-Nth holdout rule instead of "
            "delegating to holdout_status()"
        )
        assert "_holdout_indices" not in body, (
            "ingest_next_frame reads _holdout_indices directly instead of "
            "delegating to holdout_status()"
        )

    def test_explicit_holdout_indices(self, holdout_status):
        scene = _Scene(holdout_indices={3, 9}, guard_indices={4})
        assert holdout_status(scene, _Frame(3)) == (True, False)
        assert holdout_status(scene, _Frame(4)) == (False, True)
        assert holdout_status(scene, _Frame(5)) == (False, False)

    def test_live_fallback_uses_the_drop_invariant_ordinal(self, holdout_status):
        scene = _Scene(holdout_indices=None)
        # source_ordinal wins over index when present (issue #745).
        assert holdout_status(scene, _Frame(index=1, source_ordinal=8)) == (
            True,
            False,
        )
        assert holdout_status(scene, _Frame(index=8, source_ordinal=1)) == (
            False,
            False,
        )
        # ...and index is the fallback when it is absent.
        assert holdout_status(scene, _Frame(index=16)) == (True, False)

    def test_predicate_is_side_effect_free(self, holdout_status):
        scene = _Scene(holdout_indices={2})
        before = dict(scene.__dict__)
        for i in range(20):
            holdout_status(scene, _Frame(i))
        assert scene.__dict__ == before, (
            "holdout_status mutated the scene -- it must be safe to call at "
            "peek time, before the source cursor advances"
        )


class TestDrainLoopHardBypass:
    """Static assertions on the wiring: the gate is ~5900 lines into a
    12k-line function whose surrounding machinery cannot be constructed in
    a unit test, so the contract is pinned on the source text."""

    def test_gate_consults_holdout_status_before_deciding(self):
        src = TRAIN_STREAMING.read_text()
        assert "streaming_scene.holdout_status(_bp_peeked)" in src, (
            "the backpressure gate does not consult holdout_status -- a holdout "
            "can be skipped before it is ever realized as a test camera"
        )

    def test_holdouts_bypass_rather_than_being_decided(self):
        src = TRAIN_STREAMING.read_text()
        assert (
            "if _bp_peeked is not None and not (_bp_is_holdout or _bp_is_guard):"
            in src
        ), (
            "holdouts/guards are not hard-bypassed around the owner's gate"
        )
        # And they are NOT routed through the owner as non-candidates, which
        # would yield METADATA_ONLY -> materializes False -> the same bug.
        gate = _strip_comments(src[src.index("HARD BYPASS -- holdouts") :][:2000])
        assert "mapping_candidate=False" not in gate

    def test_skip_path_keeps_the_release_clock_honest(self):
        src = TRAIN_STREAMING.read_text()
        # Scope to the gate block itself rather than a fixed character
        # window -- the window silently truncated once the #602 disposition
        # bridge was added inside the gate.
        start = src.index("HARD BYPASS -- holdouts")
        gate = src[start : src.index("W3-15 deferred-admission gate", start)]
        for call in (
            "streaming_scene.skip_next_frame()",
            "scheduler.mark_released()",
            "_cadence.record_source_released()",
        ):
            assert call in gate, (
                f"the backpressure skip path is missing {call} -- a skipped "
                "observation is still RELEASED on the #289 release clock"
            )

    def test_force_all_stress_does_not_also_disable_burst_budget(self):
        """force-all exists to reproduce historical hard-all exactly; also
        neutering BurstBudget would make it differ from its own baseline."""
        src = TRAIN_STREAMING.read_text()
        assert (
            "if _burst_budget is not None and not _backpressure_cfg.force_all_stress:"
            in src
        )


class TestBypassBehaviourNotJustItsSourceText:
    """The static assertions above pin the gate's *shape*; this pins its
    *behaviour*.

    A regression that keeps the ``holdout_status`` call but inverts the
    bypass condition -- or moves the gate -- would satisfy every string
    assertion above and still lose every holdout. This test fails in that
    case, because it replays the gate's real predicate against a real
    owner under real overload.
    """

    def _drive(self, eval_hold=8, n_frames=240):
        from eligsir.mapping.scheduler.backpressure import (
            BackpressureAction,
            BackpressureConfig,
            BackpressureOwner,
            PendingObservation,
        )

        holdout_status = _bind_real_predicate()
        scene = _Scene(holdout_indices=None)
        scene.args.streaming_eval_hold = eval_hold

        # Bounds aggressive enough that essentially every non-protected
        # frame is compressed away.
        owner = BackpressureOwner(
            BackpressureConfig(
                enabled=True,
                pending_max_count=64,
                pending_max_age_control_ticks=100_000,
                max_admissions_per_window=1,
                admission_window_control_ticks=100_000,
                novelty_protect=1.1,   # protection can never fire
                quality_protect=1.1,
                keyframe_quota=0,
            )
        )

        materialized, skipped, bypassed = [], [], []
        for i in range(n_frames):
            frame = _Frame(index=i)
            # --- this is exactly what the drain-loop gate does ---
            is_holdout, is_guard = holdout_status(scene, frame)
            if is_holdout or is_guard:
                bypassed.append(i)
                materialized.append(i)   # falls through to ingest_next_frame
                continue
            for d in owner.submit(
                PendingObservation(
                    frame_id=i,
                    release_control_tick=i,
                    novelty=0.01,
                    quality=0.0,
                    motion=0.0,
                    translation_m=0.0005,
                    rotation_deg=0.01,
                ),
                control_tick=i,
            ):
                if d.action is BackpressureAction.ADMIT:
                    materialized.append(d.frame_id)
                else:
                    skipped.append(d.frame_id)
        return materialized, skipped, bypassed

    def test_every_holdout_reaches_materialization_under_overload(self):
        materialized, skipped, bypassed = self._drive()
        expected = [i for i in range(240) if i % 8 == 0]
        assert bypassed == expected
        for frame_id in expected:
            assert frame_id in materialized, (
                f"holdout frame {frame_id} never reached materialization -- it "
                "would never become a test camera, silently shrinking n_test"
            )
            assert frame_id not in skipped, (
                f"holdout frame {frame_id} was decided by the backpressure "
                "owner instead of being bypassed"
            )

    def test_the_test_actually_exercises_overload(self):
        """Guard against a vacuous pass: the surrounding non-holdout frames
        must really be getting compressed away."""
        materialized, skipped, _ = self._drive()
        assert len(skipped) > 150, (
            f"only {len(skipped)} frames were skipped -- the bounds were not "
            "aggressive enough for this to be a real overload test"
        )
        non_holdout_materialized = [i for i in materialized if i % 8 != 0]
        assert len(non_holdout_materialized) < 20, (
            "too many ordinary frames survived; the holdouts' survival is not "
            "evidence of the bypass"
        )

    def test_holdout_survival_is_independent_of_the_bound(self):
        """Tightening the bound must not cost a single holdout."""
        for eval_hold in (4, 8, 16):
            materialized, _, _ = self._drive(eval_hold=eval_hold)
            expected = {i for i in range(240) if i % eval_hold == 0}
            assert expected <= set(materialized), (
                f"holdouts lost at eval_hold={eval_hold}: "
                f"{sorted(expected - set(materialized))}"
            )
