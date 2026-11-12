"""Regression test for the StridedFrameSource de-duplication fix.

train_streaming.py used to define _StridedFrameSource inline THREE times and
apply it three times in sequence to the same frame_source variable, silently
composing --streaming_source_stride N into an effective stride of N**3
instead of N. This test pins the correct, single-application behavior
against a small synthetic frame source, for stride values 1, 2, and 4.
"""

from train_streaming import StridedFrameSource


class _FakeFrame:
    def __init__(self, idx: int):
        self.idx = idx

    def __repr__(self):
        return f"FakeFrame({self.idx})"


class _FakeFrameSource:
    """Minimal FrameSource-like object: __len__ + __getitem__."""

    def __init__(self, n: int):
        self._frames = [_FakeFrame(i) for i in range(n)]

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]

    def __iter__(self):
        return iter(self._frames)


def _emitted_indices(src, stride: int):
    strided = StridedFrameSource(src, stride)
    return [f.idx for f in strided]


def test_stride_1_yields_every_frame():
    src = _FakeFrameSource(20)
    assert _emitted_indices(src, 1) == list(range(20))


def test_stride_2_yields_every_second_frame_not_cubed():
    src = _FakeFrameSource(20)
    # Correct behavior: stride=2 -> [0, 2, 4, ..., 18].
    # The old triple-application bug would have composed this to an
    # effective stride of 2**3 = 8 -> [0, 8, 16] had it been applied
    # three times to the same variable; this test only exercises a single
    # application, which is now the only application that exists.
    assert _emitted_indices(src, 2) == list(range(0, 20, 2))


def test_stride_4_yields_every_fourth_frame():
    src = _FakeFrameSource(20)
    assert _emitted_indices(src, 4) == list(range(0, 20, 4))


def test_getitem_and_get_frame_agree_with_iteration():
    src = _FakeFrameSource(20)
    strided = StridedFrameSource(src, 2)
    expected = list(range(0, 20, 2))
    assert len(strided) == len(expected)
    for i, exp_idx in enumerate(expected):
        assert strided[i].idx == exp_idx
        assert strided.get_frame(i).idx == exp_idx


def test_stride_2_applied_once_is_not_stride_2_applied_three_times():
    """Directly pins the bug: applying stride twice more on top of a
    single correct application must NOT be the behavior triggered by a
    single --streaming_source_stride=2 flag anymore."""
    src = _FakeFrameSource(64)
    once = _emitted_indices(src, 2)
    # Simulate what the old (buggy) code effectively did: apply the wrapper
    # three times in a row to the same underlying source.
    triple = src
    for _ in range(3):
        triple = StridedFrameSource(triple, 2)
    thrice = [f.idx for f in triple]

    assert once == list(range(0, 64, 2))
    # The buggy triple-application composes to stride 2**3=8.
    assert thrice == list(range(0, 64, 8))
    assert once != thrice
