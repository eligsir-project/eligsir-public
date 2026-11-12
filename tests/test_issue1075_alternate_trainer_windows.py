"""Characterization tests for the alternate trainers' shared frame windowing (#1075).

``_run_rolling_seed`` and ``_run_submap_stitch`` each used to inline a
byte-identical frame-collection/pairing/windowing prologue. That block now lives
once in ``eligsir/mapping/integration/frame_windows.py``.

These tests exist to prove the extraction changed nothing observable. The
central test is :func:`test_matches_pre_extraction_reference` -- an independent
transcription of the pre-extraction source (``_reference_prologue`` below,
copied from the two call sites as they stood at commit ``e4892a6c``) is run
against randomized scenes and asserted equal to the extracted helper. The
remaining tests pin the individual contract clauses that equivalence rests on,
so a future change that breaks one gets a named failure rather than only an
opaque mismatch.

They are deliberately fake-driven (no torch, no GPU, no real ``StreamingScene``):
the block under test is pure list arithmetic over frames and cameras, and
running it against a real scene would test the scene, not this.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional, Tuple

import pytest

from _source_inspection import function_source, module_level_imports
from eligsir.mapping.integration.frame_windows import collect_frame_windows


# --- fakes -----------------------------------------------------------------


@dataclass
class _Frame:
    """Stand-in for ``StreamingRGBDFrame``; only ``.index`` is read."""

    index: int


@dataclass
class _Cam:
    """Stand-in for a ``Camera``; identity-compared only."""

    label: str


@dataclass
class _FakeScene:
    """Minimal ``StreamingScene`` surface the prologue actually touches.

    ``ingest_plan`` is the list of results ``ingest_next_frame()`` returns in
    order -- ``None`` models a frame the source dropped, ``is_train=False``
    models a holdout/rejected frame that yields no train camera.
    """

    frames: List[_Frame]
    train_cameras: List[_Cam]
    ingest_plan: List[Optional[Tuple[_Cam, _Frame, bool]]] = field(default_factory=list)
    ingest_calls: int = 0

    def iter_all_frames(self):
        return iter(self.frames)

    def has_next_frame(self) -> bool:
        return self.ingest_calls < len(self.ingest_plan)

    def ingest_next_frame(self):
        result = self.ingest_plan[self.ingest_calls]
        self.ingest_calls += 1
        return result


def _args(eval_hold: int) -> SimpleNamespace:
    return SimpleNamespace(streaming_eval_hold=eval_hold)


# --- the pre-extraction reference ------------------------------------------


def _reference_prologue(streaming_scene, args, submap_frames):
    """Verbatim transcription of the inline prologue prior to the extraction.

    Copied from ``rolling_seed_application.py`` (and identical, modulo comments
    and two ``print`` calls, in ``submap_stitch_application.py``) as of commit
    ``e4892a6c``. Kept as a literal duplicate on purpose: a reference that
    imported the helper would prove nothing.
    """
    all_frames = list(streaming_scene.iter_all_frames())
    all_train_cameras = list(streaming_scene.train_cameras)
    while streaming_scene.has_next_frame():
        result = streaming_scene.ingest_next_frame()
        if result is not None:
            cam, frame, is_train = result
            if is_train:
                all_train_cameras.append(cam)

    eval_hold = args.streaming_eval_hold
    train_frames = [f for f in all_frames if eval_hold <= 0 or f.index % eval_hold != 0]
    n_paired = min(len(train_frames), len(all_train_cameras))
    frame_cam_pairs = list(zip(train_frames[:n_paired], all_train_cameras[:n_paired]))

    windows = [
        frame_cam_pairs[i: i + submap_frames]
        for i in range(0, len(frame_cam_pairs), submap_frames)
    ]
    return all_train_cameras, windows


def _random_scene(rng: random.Random) -> _FakeScene:
    n_frames = rng.randint(0, 40)
    frames = [_Frame(index=i) for i in range(n_frames)]
    n_bootstrap = rng.randint(0, min(5, n_frames) if n_frames else 0)
    bootstrap = [_Cam(f"boot{i}") for i in range(n_bootstrap)]

    plan: List[Optional[Tuple[_Cam, _Frame, bool]]] = []
    for i in range(rng.randint(0, n_frames)):
        roll = rng.random()
        if roll < 0.15:
            plan.append(None)                      # source dropped the frame
        elif roll < 0.35:
            plan.append((_Cam(f"hold{i}"), frames[i], False))   # holdout
        else:
            plan.append((_Cam(f"train{i}"), frames[i], True))
    return _FakeScene(frames=frames, train_cameras=bootstrap, ingest_plan=plan)


# --- the equivalence proof --------------------------------------------------


def test_matches_pre_extraction_reference():
    """Extracted helper == inline prologue, over 300 randomized scenes.

    This is the test that makes the refactor safe: identical train-camera lists
    and identical window partitions mean both trainers see exactly the same
    geometry per window, and therefore produce the same insertions and the same
    growth events.
    """
    rng = random.Random(20260823)
    for trial in range(300):
        submap_frames = rng.choice([2, 3, 5, 20])
        eval_hold = rng.choice([-1, 0, 1, 2, 5, 8])

        scene_a = _random_scene(rng)
        # Two scenes with identical contents so each consumer drains its own.
        scene_b = _FakeScene(
            frames=list(scene_a.frames),
            train_cameras=list(scene_a.train_cameras),
            ingest_plan=list(scene_a.ingest_plan),
        )

        got_cams, got_windows = collect_frame_windows(
            scene_a, _args(eval_hold), submap_frames
        )
        ref_cams, ref_windows = _reference_prologue(
            scene_b, _args(eval_hold), submap_frames
        )

        assert got_cams == ref_cams, f"train cameras diverged on trial {trial}"
        assert got_windows == ref_windows, f"windows diverged on trial {trial}"


# --- the individual contract clauses ---------------------------------------


def test_drains_the_source_completely():
    """Both trainers rely on the source being exhausted before they proceed."""
    scene = _FakeScene(
        frames=[_Frame(i) for i in range(6)],
        train_cameras=[],
        ingest_plan=[(_Cam(f"c{i}"), _Frame(i), True) for i in range(6)],
    )
    collect_frame_windows(scene, _args(0), 3)
    assert scene.ingest_calls == 6
    assert not scene.has_next_frame()


def test_bootstrap_cameras_come_first_and_are_not_aliased():
    """Bootstrap cameras lead the list, and the scene's own list is untouched."""
    bootstrap = [_Cam("boot0"), _Cam("boot1")]
    scene = _FakeScene(
        frames=[_Frame(i) for i in range(4)],
        train_cameras=bootstrap,
        ingest_plan=[(_Cam(f"c{i}"), _Frame(i), True) for i in range(2)],
    )
    cams, _ = collect_frame_windows(scene, _args(0), 2)
    assert cams[:2] == bootstrap
    assert [c.label for c in cams] == ["boot0", "boot1", "c0", "c1"]
    # The helper must not mutate the scene's own list.
    assert scene.train_cameras == bootstrap


def test_non_train_and_dropped_ingests_yield_no_camera():
    scene = _FakeScene(
        frames=[_Frame(i) for i in range(4)],
        train_cameras=[],
        ingest_plan=[
            (_Cam("a"), _Frame(0), True),
            None,
            (_Cam("b"), _Frame(2), False),
            (_Cam("c"), _Frame(3), True),
        ],
    )
    cams, _ = collect_frame_windows(scene, _args(0), 10)
    assert [c.label for c in cams] == ["a", "c"]


@pytest.mark.parametrize("eval_hold", [0, -1, -5])
def test_non_positive_eval_hold_keeps_every_frame(eval_hold):
    """``eval_hold <= 0`` disables holdout filtering entirely."""
    scene = _FakeScene(
        frames=[_Frame(i) for i in range(5)],
        train_cameras=[_Cam(f"c{i}") for i in range(5)],
    )
    _, windows = collect_frame_windows(scene, _args(eval_hold), 10)
    assert [f.index for f, _ in windows[0]] == [0, 1, 2, 3, 4]


def test_positive_eval_hold_drops_every_nth_frame():
    """Frames whose index is a multiple of ``eval_hold`` are holdouts."""
    scene = _FakeScene(
        frames=[_Frame(i) for i in range(10)],
        train_cameras=[_Cam(f"c{i}") for i in range(10)],
    )
    _, windows = collect_frame_windows(scene, _args(3), 100)
    kept = [f.index for f, _ in windows[0]]
    assert kept == [1, 2, 4, 5, 7, 8]
    assert all(i % 3 != 0 for i in kept)


def test_pairing_truncates_to_the_shorter_list():
    """Fewer cameras than frames (admission rejected some) truncates the pairs."""
    scene = _FakeScene(
        frames=[_Frame(i) for i in range(10)],
        train_cameras=[_Cam(f"c{i}") for i in range(3)],
    )
    cams, windows = collect_frame_windows(scene, _args(0), 100)
    assert len(cams) == 3
    pairs = windows[0]
    assert len(pairs) == 3
    assert [f.index for f, _ in pairs] == [0, 1, 2]


def test_pairing_is_positional_not_by_index():
    """The i-th kept frame pairs with the i-th camera, holdout gaps included.

    This is the pre-existing (and slightly surprising) contract: with a holdout
    stride the frame indices skip, but the camera list does not, so frame 1
    pairs with the first camera rather than the second.
    """
    scene = _FakeScene(
        frames=[_Frame(i) for i in range(6)],
        train_cameras=[_Cam(f"c{i}") for i in range(6)],
    )
    _, windows = collect_frame_windows(scene, _args(2), 100)
    pairs = windows[0]
    assert [(f.index, c.label) for f, c in pairs] == [
        (1, "c0"), (3, "c1"), (5, "c2"),
    ]


def test_windows_chunk_at_submap_frames_with_a_short_tail():
    scene = _FakeScene(
        frames=[_Frame(i) for i in range(7)],
        train_cameras=[_Cam(f"c{i}") for i in range(7)],
    )
    _, windows = collect_frame_windows(scene, _args(0), 3)
    assert [len(w) for w in windows] == [3, 3, 1]
    assert [f.index for w in windows for f, _ in w] == [0, 1, 2, 3, 4, 5, 6]


def test_empty_scene_yields_no_windows():
    """Both trainers guard on empty windows; an empty source must not raise."""
    scene = _FakeScene(frames=[], train_cameras=[])
    cams, windows = collect_frame_windows(scene, _args(2), 5)
    assert cams == []
    assert windows == []


def test_frames_but_no_cameras_yields_no_windows():
    scene = _FakeScene(frames=[_Frame(i) for i in range(5)], train_cameras=[])
    cams, windows = collect_frame_windows(scene, _args(0), 5)
    assert cams == []
    assert windows == []


# --- the call sites actually use it ----------------------------------------


def test_both_trainers_call_the_shared_helper():
    """Guard against one trainer silently drifting back to an inline copy.

    Read off disk via AST rather than ``import ... ; inspect.getsource(...)``:
    both trainer modules import torch at module scope (they are real training
    code and genuinely need it), so importing them made this guard die with
    ``ModuleNotFoundError`` -- guarding nothing -- on the torch-free CPU image.
    See ``tests/_source_inspection`` and issue #1112.
    """
    for rel_path, fn_name in (
        (
            "eligsir/mapping/integration/rolling_seed_application.py",
            "_run_rolling_seed",
        ),
        (
            "eligsir/mapping/integration/submap_stitch_application.py",
            "_run_submap_stitch",
        ),
    ):
        source = function_source(rel_path, fn_name)
        assert "collect_frame_windows(" in source, (
            f"{fn_name} no longer calls the shared windowing helper"
        )
        assert "iter_all_frames()" not in source, (
            f"{fn_name} appears to have re-inlined the frame-collection block"
        )


def test_submap_stitch_imports_os():
    """Regression guard for the missing-``os`` NameError (WS-E5 split fallout).

    ``_run_submap_stitch`` calls ``os.path.join`` to save the merged and refined
    PLYs; the module lost its ``import os`` when it was split out of
    ``rgbd_integrator_impl.py``, so every submap_stitch run raised ``NameError``
    immediately after the merge.

    Asserted against the module's own module-scope import statements rather
    than ``hasattr`` on an imported module object (#1112): the module imports
    torch and so could not be imported at all on the CPU image, and this form
    is tighter anyway -- ``hasattr(mod, "os")`` would also be satisfied by an
    ``os`` attribute that a star-import or a monkeypatch happened to supply,
    neither of which fixes the ``NameError`` this guard is about.
    """
    assert "os" in module_level_imports(
        "eligsir/mapping/integration/submap_stitch_application.py"
    ), "submap_stitch_application lost its module-scope `import os` again"
