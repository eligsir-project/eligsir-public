import os

import pytest

from scripts.sh_clamp_diagnostic.check_ply_degree_serialization import check_degree_mismatch

_CKPT_FR3 = "/data/ckpt_fr3"


@pytest.mark.skipif(
    not os.path.isdir(_CKPT_FR3),
    reason=f"needs a real, in-progress training checkpoint mounted at {_CKPT_FR3}",
)
def test_reports_mismatch_shape():
    # Uses a real checkpoint fixture path on shared scratch storage
    # (/data/ckpt_fr3, a live/training-in-progress output directory, NOT a
    # frozen fixture -- Scene(load_iteration=-1) always resolves to whatever
    # iteration_<N> is latest at test time). Its PLYs predate the PLY
    # active_sh_degree serialization fix (docs/superpowers/plans/2026-07-16-
    # ply-active-sh-degree-fix.md), so they have no embedded
    # active_sh_degree comment and always exercise the fix's
    # backward-compatible fallback branch (reloaded = max_sh_degree). That
    # fallback branch is confirmed correct as of this task (Task 3): for
    # this fixture it currently resolves to iteration_80000, whose genuine
    # chkpnt80000.pth active_sh_degree already happens to equal
    # max_sh_degree (3), so mismatch is currently False -- a coincidence of
    # how far along training the latest checkpoint is, not a change in
    # fallback behavior. Earlier in this same run's history (iteration_2000,
    # genuine degree 2) the identical fallback branch produced mismatch=True
    # (trained=2, reloaded=3); see this plan's Task 3 report for a directly
    # forced worked example confirming the fix round-trips the genuine
    # degree once a PLY is re-saved by the fixed save_ply. This test
    # intentionally does not hardcode either outcome, since which value the
    # fallback happens to coincide with depends on which iteration is
    # latest and how far its genuine active_sh_degree had progressed.
    result = check_degree_mismatch(
        "/data/ckpt_fr3", "gsplat"
    )
    assert set(result.keys()) == {"trained_active_sh_degree", "reloaded_active_sh_degree", "mismatch", "model_path"}
    assert result["mismatch"] == (result["trained_active_sh_degree"] != result["reloaded_active_sh_degree"])
