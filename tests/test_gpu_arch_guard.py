"""Regression tests for issue #1392's GPU-arch-mismatch guard.

Issue #1392 was filed as "latest-gfx1151 image contains gfx1201 binaries",
based on an agent pulling the gfx1151-tagged image onto R9700 -- which is
gfx1201 hardware, not gfx1151 (docker-compose.yml's ELIGSIR_IMAGE_TAG and
PYTORCH_ROCM_ARCH both default to gfx1151, so a bare `docker compose run --rm
train` on a different-arch host silently selects the wrong image). Both
`latest-gfx1151` and `latest-gfx1201` were independently verified (two
digests each, including the exact digest cited in the issue) to contain only
their own architecture's Tensile binaries -- there was no image defect.

These tests exercise `utils/gpu_arch_guard.check_rocm_arch_match`, the guard
added so this exact confusion fails fast and legibly instead of surfacing as
an opaque hipBLASLt error that reads like a build defect.
"""

import os
from unittest import mock

import pytest

# The CPU smoke image (Dockerfile.ci-cpu) deliberately does not install torch
# (see tests/conftest.py's module docstring). Every test below patches
# "torch.cuda.*" by dotted string, which requires torch to be importable even
# to construct the mock -- so importorskip the whole file rather than only the
# individual tests that technically don't touch torch, matching this repo's
# existing convention (e.g. test_psnr_history.py).
torch = pytest.importorskip("torch")

from eligsir.utils.gpu_arch_guard import GPUArchMismatchError, check_rocm_arch_match  # noqa: E402


def test_no_baked_arch_env_is_a_noop():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PYTORCH_ROCM_ARCH", None)
        check_rocm_arch_match()  # must not raise


def test_no_cuda_device_is_a_noop():
    with mock.patch.dict(os.environ, {"PYTORCH_ROCM_ARCH": "gfx1151"}):
        with mock.patch("torch.cuda.is_available", return_value=False):
            check_rocm_arch_match()  # must not raise


def test_matching_arch_is_a_noop():
    class _Props:
        gcnArchName = "gfx1151"

    with mock.patch.dict(os.environ, {"PYTORCH_ROCM_ARCH": "gfx1151"}):
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.get_device_properties", return_value=_Props()
        ):
            check_rocm_arch_match()  # must not raise


def test_matching_arch_ignores_feature_suffix():
    class _Props:
        gcnArchName = "gfx1151:xnack+"

    with mock.patch.dict(os.environ, {"PYTORCH_ROCM_ARCH": "gfx1151"}):
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.get_device_properties", return_value=_Props()
        ):
            check_rocm_arch_match()  # must not raise


def test_mismatched_arch_raises_actionable_error():
    class _Props:
        gcnArchName = "gfx1201"

    with mock.patch.dict(os.environ, {"PYTORCH_ROCM_ARCH": "gfx1151"}):
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.get_device_properties", return_value=_Props()
        ):
            with pytest.raises(GPUArchMismatchError) as excinfo:
                check_rocm_arch_match()

    message = str(excinfo.value)
    assert "gfx1151" in message
    assert "gfx1201" in message
    assert "latest-gfx1201" in message


def test_device_properties_failure_is_a_noop():
    with mock.patch.dict(os.environ, {"PYTORCH_ROCM_ARCH": "gfx1151"}):
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.get_device_properties", side_effect=RuntimeError("no device")
        ):
            check_rocm_arch_match()  # must not raise
