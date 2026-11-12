"""Slow integration guard: train a short streaming run, assert the comparison
harness detects (or clears) back-half degradation on a REAL run.

Opt-in only — needs the container (torch/CUDA + dataset). Enable with:
    SPLAT_RUN_SLOW=1 SPLAT_DEGRADE_SCENE=/data/tum/rgbd_dataset_freiburg1_desk \
        python -m pytest tests/regression/test_backhalf_degradation.py -q

Default behaviour (no flag): skipped, so the suite stays fast/host-friendly.

Once Part 3 identifies a noise-schedule that holds quality, set
SPLAT_DEGRADE_EXTRA_ARGS to that config and flip EXPECT_DEGRADED=0 to turn this
into a true no-regression gate.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

from eligsir.eval.compare import checkpoint_curve, degradation_report

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _enabled():
    if os.environ.get("SPLAT_RUN_SLOW") != "1":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _enabled(), reason="set SPLAT_RUN_SLOW=1 (needs container+dataset)")
def test_short_run_degradation_harness():
    scene = os.environ.get("SPLAT_DEGRADE_SCENE", "/data/tum/rgbd_dataset_freiburg1_desk")
    iters = int(os.environ.get("SPLAT_DEGRADE_ITERS", "6000"))
    extra = os.environ.get("SPLAT_DEGRADE_EXTRA_ARGS", "").split()
    out = tempfile.mkdtemp(prefix="degrade_test_", dir=os.path.join(REPO, "output"))
    try:
        cmd = [
            sys.executable, "train.py", "-s", scene, "-m", out,
            "--streaming_replay", "--cap_max", "50000", "--iterations", str(iters),
            "--tum_eval_hold", "8", "--test_iterations", str(iters // 2), str(iters),
        ] + extra
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=3600)
        assert r.returncode == 0, f"train failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"

        curve = checkpoint_curve(out)
        assert len(curve) >= 2, f"expected >=2 checkpoints, got {sorted(curve)}"
        dr = degradation_report(out)
        # The harness must produce a well-formed verdict on a real run.
        assert dr.peak_iter is not None and dr.final_iter is not None
        expect = os.environ.get("EXPECT_DEGRADED")
        if expect is not None:
            assert dr.degraded is bool(int(expect)), (
                f"degraded={dr.degraded}, expected {bool(int(expect))}; reasons={dr.reasons}"
            )
    finally:
        shutil.rmtree(out, ignore_errors=True)
