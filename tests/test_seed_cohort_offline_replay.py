"""Issue #416: the offline replay shim must report the REAL unavailability reason.

``utils.gaussian_lineage.seed_uid_range`` raises ``SeedCohortUnavailable`` with
``gaussians._seed_uid_range_reason`` when that attribute is set, and otherwise
falls back to a default text blaming the LEGACY (non-gsplat) model layout.

``scripts/seed_cohort_offline_replay.py``'s ``_CheckpointShim`` used to set only
``_seed_uid_range``, so every pre-#373 gsplat checkpoint -- the exact population
this script exists to inspect, and #407's sole substrate among them -- raised
with that legacy-layout text despite carrying all seven lineage buffers. The
diagnosis pointed at the model layout when the real cause was the checkpoint's
code vintage.
"""
import importlib.util
import os
import sys

import pytest
import torch

from eligsir.utils.gaussian_lineage import SeedCohortUnavailable, seed_uid_range

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "scripts", "seed_cohort_offline_replay.py")

_spec = importlib.util.spec_from_file_location("_seed_cohort_offline_replay", _SCRIPT)
_replay = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _replay
_spec.loader.exec_module(_replay)


def _blob(*, seed_uid_range_value, with_streaming_state=True, n=4):
    """A gsplat-layout checkpoint blob with the full lineage buffer set."""
    streaming_state = {}
    if with_streaming_state:
        streaming_state = {
            "gaussian_uid": torch.arange(n, dtype=torch.long),
            "parent_uid": torch.full((n,), -1, dtype=torch.long),
            "root_uid": torch.arange(n, dtype=torch.long),
            "generation": torch.zeros(n, dtype=torch.long),
            "birth_method": torch.zeros(n, dtype=torch.long),
            "last_reseed_iter": torch.zeros(n, dtype=torch.long),
            "birth_iter": torch.zeros(n, dtype=torch.long),
        }
    return {
        "layout": "gsplat",
        "seed_uid_range": seed_uid_range_value,
        "streaming_state": streaming_state,
        "params": {
            "means": torch.zeros(n, 3),
            "opacities": torch.zeros(n),
        },
    }


def test_pre_373_gsplat_checkpoint_blames_the_missing_key_not_the_layout():
    """The regression test: lineage buffers present, ``seed_uid_range`` absent."""
    shim = _replay._CheckpointShim(_blob(seed_uid_range_value=None))

    # Precondition: this checkpoint is NOT the legacy layout -- it has lineage.
    assert shim.gaussian_uid is not None
    assert shim.birth_method is not None

    with pytest.raises(SeedCohortUnavailable) as excinfo:
        seed_uid_range(shim)
    message = str(excinfo.value)
    assert "predates the SEED uid range" in message
    assert "no 'seed_uid_range' key" in message
    # The bug this test pins: never blame the legacy layout for a gsplat
    # checkpoint that plainly carries per-Gaussian lineage buffers.
    assert "legacy GaussianModel layout" not in message


def test_checkpoint_without_streaming_state_reports_that_instead():
    shim = _replay._CheckpointShim(
        _blob(seed_uid_range_value=None, with_streaming_state=False)
    )
    with pytest.raises(SeedCohortUnavailable) as excinfo:
        seed_uid_range(shim)
    message = str(excinfo.value)
    assert "carries no streaming_state" in message
    assert "legacy GaussianModel layout" not in message


def test_available_range_still_resolves_and_carries_no_reason():
    shim = _replay._CheckpointShim(_blob(seed_uid_range_value=(0, 4)))
    assert shim._seed_uid_range_reason == ""
    assert seed_uid_range(shim) == (0, 4)
