"""Tests for the inert-``--streaming_replay_buffer`` detector (issue #360).

The ring buffer (`StreamingScene._replay_buffer`) is only sampled when the
never-evicted global reservoir is empty, so with the default
``--streaming_global_reservoir_stride 1`` the buffer's *size* can never bind.
These tests pin both the inert and the non-inert configurations, plus the
default-value mirror the check relies on.
"""

import argparse
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import StreamingParams  # noqa: E402
from train_streaming import (  # noqa: E402
    STREAMING_REPLAY_BUFFER_DEFAULT,
    check_replay_buffer_reachability,
)


def _args(**kw):
    base = dict(
        streaming_replay=True,
        streaming_global_reservoir_stride=1,
        streaming_replay_buffer=STREAMING_REPLAY_BUFFER_DEFAULT,
    )
    base.update(kw)
    return SimpleNamespace(**base)


# Attributes a *partial* namespace must still declare, because
# check_replay_buffer_reachability reads them WITHOUT a getattr fallback. Kept in
# sync with the real source by the torch-free
# tests/test_reachability_fixture_args_completeness.py (see the NOTE at the
# bottom of this file). Issue #455.
_PARTIAL_NS_KW = dict(
    streaming_replay=True,
    streaming_global_reservoir_stride=1,
)


def test_default_constant_matches_declared_default():
    """The mirrored default must track arguments/__init__.py."""
    parser = argparse.ArgumentParser()
    group = StreamingParams(parser)
    assert group.streaming_replay_buffer == STREAMING_REPLAY_BUFFER_DEFAULT


def test_inert_config_warns():
    """Non-default buffer size + reservoir enabled => the size is inert."""
    warnings = check_replay_buffer_reachability(
        _args(streaming_replay_buffer=4000, streaming_global_reservoir_stride=1)
    )
    assert len(warnings) == 1
    msg = warnings[0]
    assert "--streaming_replay_buffer=4000" in msg
    assert "NO EFFECT" in msg
    assert "--streaming_global_reservoir_stride" in msg


def test_inert_config_warns_for_any_positive_stride():
    """Stride > 1 still fills the reservoir permanently -> still inert."""
    warnings = check_replay_buffer_reachability(
        _args(streaming_replay_buffer=100, streaming_global_reservoir_stride=4)
    )
    assert len(warnings) == 1


def test_non_inert_config_is_silent():
    """stride == 0 disables the reservoir; the ring buffer is the real pool."""
    assert (
        check_replay_buffer_reachability(
            _args(streaming_replay_buffer=4000, streaming_global_reservoir_stride=0)
        )
        == []
    )


def test_default_buffer_size_is_silent():
    """A buffer left at its default is not a misleading lever -- stay quiet."""
    assert check_replay_buffer_reachability(_args()) == []


def test_non_streaming_run_is_silent():
    """Offline train.py runs touch neither flag."""
    assert (
        check_replay_buffer_reachability(
            _args(streaming_replay=False, streaming_replay_buffer=4000)
        )
        == []
    )


def test_missing_attributes_do_not_raise():
    """A partial namespace must not crash -- but it has to declare the flags the
    check reads bare.

    Issue #455 decision (see that issue and this PR): this is NOT a "constructors
    tolerate arbitrary partial args" contract. ``check_replay_buffer_reachability``
    has exactly ONE production caller (``train_streaming.py``'s
    ``streaming_training()``), which passes the full, live-parsed trainer args;
    no script, diagnostic tool or restored-``cfg_args`` path calls it. Its own
    docstring says it "takes the merged ``run_args`` namespace". So when #394
    (commit ``4e139d2``) replaced ``getattr(args,
    "streaming_global_reservoir_stride", 0)`` with a bare read, the bare read was
    right and *this fixture* was the stale half -- the same call the repo already
    made for ``insert_gaussians_from_frame`` in #408/PR #453.

    What remains genuinely tolerated (and is still asserted here) is the
    ``streaming_replay`` probe itself: a namespace that does not even claim to be
    a streaming run returns early, before any bare read.
    """
    assert check_replay_buffer_reachability(SimpleNamespace()) == []
    assert check_replay_buffer_reachability(SimpleNamespace(**_PARTIAL_NS_KW)) == []


# NOTE: the durable guard that keeps _PARTIAL_NS_KW in sync with the function's
# real bare ``args.<flag>`` reads deliberately does NOT live here -- this module
# imports train_streaming (hence torch), so tests/conftest.py's collect_ignore
# skips the whole file on a plain `pytest tests/` run and a guard placed here
# would never actually run. It lives in the torch-free
# tests/test_reachability_fixture_args_completeness.py, which reads both this
# file and train_streaming.py as source. Issue #455.
