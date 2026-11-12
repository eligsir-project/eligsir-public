"""PR3 (2026-08-12 cadence/batching plan): flag-gated multi-keyframe gradient
accumulation -- CLI flag default + startup-validation contract.

The validation entrypoint takes the RAW, merged argparse namespace (the same
``run_args`` object ``train_streaming.py`` assigns to ``args``), not a
per-``ParamGroup`` ``.extract()`` result: the three flags it checks are declared
across two different groups (``OptimizationParams`` owns ``optimizer_type`` /
``gsplat_sparse_grad``; ``StreamingParams`` owns
``streaming_spatial_bubble_enabled`` and the new batch-size flag), and
``ParamGroup.extract()`` only copies attributes declared on that one group.
"""

from argparse import ArgumentParser

import pytest

from arguments import OptimizationParams, StreamingParams
from train_streaming import validate_streaming_optimization_batch_size


def _make_args(**overrides):
    parser = ArgumentParser()
    OptimizationParams(parser)
    StreamingParams(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_default_batch_size_is_one():
    args = _make_args()
    assert args.streaming_optimization_batch_size == 1


def test_valid_batch_sizes_accepted():
    for b in (1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256):
        args = _make_args(
            streaming_optimization_batch_size=b,
            optimizer_type="adam",
            gsplat_sparse_grad=False,
        )
        validate_streaming_optimization_batch_size(args)  # must not raise


@pytest.mark.parametrize("batch_size", (0, 5, 17, 48, 512))
def test_invalid_batch_size_rejected(batch_size):
    args = _make_args(streaming_optimization_batch_size=batch_size)
    with pytest.raises(ValueError, match="1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256"):
        validate_streaming_optimization_batch_size(args)


def test_batch_size_rejected_with_selective_adam():
    args = _make_args(
        streaming_optimization_batch_size=2,
        optimizer_type="selective_adam",
        streaming_optimization_engine=False,
    )
    with pytest.raises(ValueError, match="selective_adam"):
        validate_streaming_optimization_batch_size(args)


def test_batch_size_rejected_with_spatial_bubble():
    args = _make_args(
        streaming_optimization_batch_size=2,
        optimizer_type="adam",
        gsplat_sparse_grad=False,
        streaming_spatial_bubble_enabled=True,
    )
    with pytest.raises(ValueError, match="spatial_bubble"):
        validate_streaming_optimization_batch_size(args)


def test_conflict_message_matches_post_fix_profile_semantics():
    """Issue #228, revised 2026-08-13.

    Pre-fix, `--parallelism_profile off` did NOT reset optimizer_type /
    gsplat_sparse_grad, so this message had to warn that 'off' alone was not a
    remedy. With #228 fixed, 'off' really does select dense Adam, so the
    message now offers it as the primary remedy again -- while still naming
    the explicit pair, still warning that --no-gsplat_sparse_grad ALONE is
    insufficient, and still pointing at #228 so anyone reading a pre-fix run's
    log knows which semantics applied."""
    args = _make_args(
        streaming_optimization_batch_size=2,
        optimizer_type="selective_adam",
        gsplat_sparse_grad=True,
        streaming_optimization_engine=False,
    )
    with pytest.raises(ValueError) as excinfo:
        validate_streaming_optimization_batch_size(args)
    msg = str(excinfo.value)
    assert "--optimizer_type adam" in msg
    assert "--no-gsplat_sparse_grad" in msg
    assert "--parallelism_profile off" in msg
    assert "on its own is" in msg  # ...still not enough (selective_adam case)
    assert "228" in msg
