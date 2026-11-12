"""Wiring test for issue #834 review defect #1: the pure resolver
(``resolve_source_sampling_policy`` / ``resolve_source_sampling_application``)
must actually sit on the real frame-source construction path, not merely be
reachable from ``report.json`` construction.

This exercises ``eligsir.io.sources.registry.make_frame_source`` --
the exact factory ``train_streaming.py`` calls to build its frame source --
against a tiny synthetic ``frames.jsonl`` + ``intrinsics.json`` dataset,
using the same override-then-restore pattern ``train_streaming.py`` applies:
resolve the policy, temporarily override ``args.streaming_frame_stride`` to
the resolved native-owner stride, construct the source, and (if the generic
owner was assigned) wrap it with ``StridedFrameSource`` afterward.

No torch import is required (``RGBDSequenceFrameSource`` only touches numpy
and PIL), so this runs on the CPU-only test surface.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from PIL import Image

from eligsir.io.sources.registry import make_frame_source
from eligsir.mapping.scheduler.frame_scheduler import (
    resolve_source_sampling_application,
)
from train_streaming import StridedFrameSource


def _make_rgbd_sequence_dir(tmp_path, n_frames: int) -> str:
    root = tmp_path / "seq"
    (root / "rgb").mkdir(parents=True)

    intrinsics = {
        "width": 8,
        "height": 8,
        "fx": 10.0,
        "fy": 10.0,
        "cx": 4.0,
        "cy": 4.0,
        "depth_scale": 1000.0,
    }
    (root / "intrinsics.json").write_text(json.dumps(intrinsics))

    with open(root / "frames.jsonl", "w") as f:
        for i in range(n_frames):
            rgb_rel = f"rgb/{i:04d}.png"
            Image.fromarray(
                (np.ones((8, 8, 3), dtype=np.uint8) * (i % 255))
            ).save(root / rgb_rel)
            c2w = np.eye(4, dtype=np.float32).tolist()
            record = {
                "id": i,
                "timestamp": float(i),
                "rgb": rgb_rel,
                "c2w": c2w,
            }
            f.write(json.dumps(record) + "\n")

    return str(root)


class _FakeArgs:
    """Minimal args stand-in: only the attributes registry.make_frame_source's
    RGBDSequence branch actually reads via getattr."""

    def __init__(self, streaming_frame_stride=1, streaming_max_frames=0):
        self.streaming_frame_stride = streaming_frame_stride
        self.streaming_max_frames = streaming_max_frames
        self.streaming_resolution = 1
        self.streaming_full_res_frame_indices = ""


def _build_frame_source_like_train_streaming(source_path: str, args: _FakeArgs, sampling_mode: str):
    """Reproduces the exact override-then-restore + StridedFrameSource-wrap
    sequence train_streaming.py's streaming_training() applies around its
    make_frame_source() call, so this test is pinned to the real production
    code path (resolve_source_sampling_application), not a re-implementation
    of its logic."""
    requested_frame_stride = args.streaming_frame_stride
    requested_source_stride = getattr(args, "streaming_source_stride", 1)

    application = resolve_source_sampling_application(
        requested_frame_stride,
        requested_source_stride,
        sampling_mode=sampling_mode,
    )
    args.streaming_frame_stride = application["native_stride_to_apply"]
    apply_source_stride = application["generic_stride_to_apply"]

    frame_source = make_frame_source(source_path, args)
    args.streaming_frame_stride = requested_frame_stride  # restore, as train_streaming.py does

    if apply_source_stride > 1:
        frame_source = StridedFrameSource(frame_source, apply_source_stride)

    return frame_source, application


class TestSingleOwnerWiringEndToEnd:
    """Defect #1: prove the resolver actually gates what make_frame_source /
    StridedFrameSource apply, not just what report.json claims."""

    def test_only_frame_stride_set(self, tmp_path):
        path = _make_rgbd_sequence_dir(tmp_path, 20)
        args = _FakeArgs(streaming_frame_stride=2)
        args.streaming_source_stride = 1
        frame_source, application = _build_frame_source_like_train_streaming(
            path, args, "single_owner"
        )
        assert len(frame_source) == 10  # 20 frames strided by 2, once
        assert application["policy"]["owner"] == "native"

    def test_only_source_stride_set(self, tmp_path):
        path = _make_rgbd_sequence_dir(tmp_path, 20)
        args = _FakeArgs(streaming_frame_stride=1)
        args.streaming_source_stride = 2
        frame_source, application = _build_frame_source_like_train_streaming(
            path, args, "single_owner"
        )
        assert len(frame_source) == 10  # 20 frames strided by 2, once
        assert application["policy"]["owner"] == "generic"

    def test_equal_nonone_strides_apply_once_not_squared(self, tmp_path):
        """The defect-1 regression case: frame_stride == source_stride == 2
        used to (pre-fix) apply BOTH, decimating by 2*2=4. Post-fix, exactly
        one owner (native, per the resolver's Rule 3) applies stride 2."""
        path = _make_rgbd_sequence_dir(tmp_path, 20)
        args = _FakeArgs(streaming_frame_stride=2)
        args.streaming_source_stride = 2
        frame_source, application = _build_frame_source_like_train_streaming(
            path, args, "single_owner"
        )
        assert len(frame_source) == 10  # NOT 5 (which 2*2 would give)
        assert application["policy"]["owner"] == "native"
        assert application["policy"]["effective_stride"] == 2

    def test_conflicting_strides_refuse_before_construction(self, tmp_path):
        """A both-non-1-and-unequal request must fail BEFORE any frame is
        read (defect #4: fail early, never after training completes)."""
        path = _make_rgbd_sequence_dir(tmp_path, 20)
        args = _FakeArgs(streaming_frame_stride=2)
        args.streaming_source_stride = 3
        from eligsir.mapping.scheduler.frame_scheduler import ConflictingStrideError

        with pytest.raises(ConflictingStrideError):
            _build_frame_source_like_train_streaming(path, args, "single_owner")

    def test_legacy_compose_reproduces_historical_multiplication(self, tmp_path):
        """The explicit escape hatch still composes both strides verbatim."""
        path = _make_rgbd_sequence_dir(tmp_path, 60)
        args = _FakeArgs(streaming_frame_stride=2)
        args.streaming_source_stride = 3
        frame_source, application = _build_frame_source_like_train_streaming(
            path, args, "legacy_compose"
        )
        assert len(frame_source) == 10  # 60 / 2 (native) / 3 (generic) = 10
        assert application["policy"]["owner"] == "both"

    def test_args_streaming_frame_stride_restored_after_construction(self, tmp_path):
        """The temporary override must not leak into the caller's args
        object once construction is done."""
        path = _make_rgbd_sequence_dir(tmp_path, 20)
        args = _FakeArgs(streaming_frame_stride=1)
        args.streaming_source_stride = 2
        _build_frame_source_like_train_streaming(path, args, "single_owner")
        assert args.streaming_frame_stride == 1  # restored, not left at the internal override


class TestResolverActuallyCalledInProductionPath:
    """AST pin (mirrors tests/test_issue805_source_decimation_provenance.py's
    style): train_streaming.py's real frame-source construction code must
    CALL resolve_source_sampling_application, not merely have it importable.

    This is the discriminator for review defect #1 -- the pre-#834-completion
    file (PR #841 commit 99dee2fc) has resolve_source_decimation wired into
    report.json construction only; resolve_source_sampling_application (and
    the args.streaming_frame_stride override this test also checks for) did
    not exist in that file at all.
    """

    def test_resolve_source_sampling_application_is_called(self):
        import ast

        tree = ast.parse(open("train_streaming.py").read())
        called_names = {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "resolve_source_sampling_application" in called_names, (
            "resolve_source_sampling_application is never CALLED in "
            "train_streaming.py -- the resolver is not wired into the real "
            "sampling path (it may only be reachable from report.json "
            "construction, e.g. via resolve_source_decimation)."
        )

    def test_native_stride_override_is_applied_to_args(self):
        """The resolved native-owner stride must actually be written onto
        `args.streaming_frame_stride` before the frame source is built --
        otherwise the native readers (registry.make_frame_source's getattr
        call sites) would still see the raw requested stride and the
        resolver's decision would be inert."""
        import ast

        tree = ast.parse(open("train_streaming.py").read())
        found = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "streaming_frame_stride"
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "args"
            ):
                found = True
                break
        assert found, (
            "No assignment to args.streaming_frame_stride found in "
            "train_streaming.py -- the resolved native-owner stride is "
            "never applied to the object the native readers actually read."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
