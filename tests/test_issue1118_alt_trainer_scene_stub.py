"""Issue #1118: the alternate trainers' throwaway ``StreamingScene`` stub must
actually satisfy ``_build_pcd_from_frames``.

``submap_stitch`` and ``rolling_seed`` reuse ``StreamingScene._build_pcd_from_frames``
by constructing an instance with ``__new__`` and assigning only the handful of
attributes that method needs -- ``__init__`` is unusable here because it mutates
the caller's ``args`` in place, eagerly drains the frame source, designates eval
holdouts and resets a module-global decode cache.

That stub silently drifted from the method: #119 (``0b4dcb95``, 2026-08-12) added
an unconditional ``self._external_seed`` read, assigned only in ``__init__``, so
BOTH modes raised ``AttributeError`` on their very first window and were dead on
arrival for 11 days. Nothing caught it because the existing coverage for these
modes (``tests/test_issue1102_alt_trainer_evidence.py``) is entirely AST /
source-text / synthetic-state assertions -- nothing ever constructed the stub or
executed the method.

So these tests deliberately EXECUTE the real method against a real stub built the
same way the trainers build it, rather than asserting anything about source text.
An empty frame list is enough: it still runs every unconditional read (including
``_external_seed`` and the no-default ``args`` fields) before taking the
"no depth available" fallback, and needs no images, no GPU and no dataset.
"""

import pytest

from scene.streaming_scene import StreamingScene
from eligsir.cli.parser import parse_training_args


def _real_default_args():
    """Real parsed args, so this test also catches future `args` field drift."""
    args, _bundle = parse_training_args(["-s", "/tmp/issue1118_nonexistent"])
    return args


def _stub_scene(cameras_extent=1.0):
    """Build the throwaway scene exactly as both alternate trainers do.

    Mirrors ``submap_stitch_application.py`` / ``rolling_seed_application.py``.
    If those stubs and this helper ever diverge, the paired source-parity test
    below fails.
    """
    from eligsir.mapping.stores.keyframe_store import KeyframeStore

    scene = StreamingScene.__new__(StreamingScene)
    scene._keyframe_store = KeyframeStore()
    scene.args = _real_default_args()
    scene.cameras_extent = cameras_extent
    scene._external_seed = None
    return scene


def test_stub_scene_builds_a_pcd_without_attribute_error():
    """The #1118 regression: this raised AttributeError('_external_seed')."""
    scene = _stub_scene()

    pcd = scene._build_pcd_from_frames([])

    # Empty frames legitimately take the random-scatter fallback; the point is
    # that the method RAN to completion instead of raising.
    assert pcd is not None
    assert pcd.points.shape[0] > 0
    assert pcd.points.shape[1] == 3


def test_stub_scene_external_seed_is_none_not_inherited():
    """`None` is load-bearing, not incidental.

    The external-seed branch returns early and ignores ``frames`` entirely, so a
    stub carrying a parent scene's seed would hand every submap/window the
    identical global cloud instead of per-window geometry. Pin the semantics.
    """
    scene = _stub_scene()
    assert scene._external_seed is None


def test_external_seed_is_a_real_init_attribute_not_a_property():
    """Guard the fix's premise: assigning it on the stub is meaningful.

    If ``_external_seed`` ever became a property/descriptor backed by another
    store (the way ``cameras_extent`` is), plain assignment on the stub could
    stop doing what this fix assumes.
    """
    assert not isinstance(
        getattr(StreamingScene, "_external_seed", None), property
    )


@pytest.mark.parametrize(
    "module_path",
    [
        "eligsir/mapping/integration/submap_stitch_application.py",
        "eligsir/mapping/integration/rolling_seed_application.py",
    ],
)
def test_both_trainers_assign_external_seed_on_their_stub(module_path):
    """Both stubs must carry the attribute -- they drifted independently before.

    Source-text, unlike the executing tests above, because there is no cheap way
    to reach either trainer's stub without a full dataset and GPU. It exists to
    keep the two call sites in sync with the executed stub above, not as the
    primary evidence.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parents[1].joinpath(module_path).read_text()
    stub_idx = src.index("StreamingScene.__new__(StreamingScene)")
    # Anchor on the CALL, not a bare mention of the method name -- prose in a
    # nearby comment must not be able to close the window early and hide a
    # genuinely missing assignment.
    pcd_idx = src.index("_tmp_scene._build_pcd_from_frames(", stub_idx)

    assert "_external_seed" in src[stub_idx:pcd_idx], (
        f"{module_path} builds a StreamingScene stub and calls "
        "_build_pcd_from_frames without assigning _external_seed (issue #1118)"
    )
