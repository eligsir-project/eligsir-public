"""Unit tests for scripts/run_error.py failure classification, plus the
pause-flag lifecycle and not_before backoff gating wired into queue_runner.py."""
import importlib.util
import json
import os
import time

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_error",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "run_error.py"),
)
run_error = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_error)

_QR_SPEC = importlib.util.spec_from_file_location(
    "queue_runner",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "queue_runner.py"),
)
qr = importlib.util.module_from_spec(_QR_SPEC)
_QR_SPEC.loader.exec_module(qr)


@pytest.fixture
def tmp_queue(tmp_path, monkeypatch):
    q = str(tmp_path / "queue.jsonl")
    monkeypatch.setattr(qr, "QUEUE", q)
    return q


def _job(**kw):
    j = {"id": "j1", "scene": "/data/kitchen1", "model_path": "output/scratch/j1"}
    j.update(kw)
    return j


def test_disk_full_from_log_marker():
    r = run_error.classify_failure(_job(), "some log\nNo space left on device\nmore", 1, {})
    assert r["category"] == "disk_full"
    assert r["suggested_action"] == "pause_queue_alert"


def test_disk_full_from_machine_context():
    r = run_error.classify_failure(_job(), "normal training log", 1,
                                    {"disk_free_bytes": 500 * 1024 * 1024})
    assert r["category"] == "disk_full"


def test_image_missing():
    r = run_error.classify_failure(_job(), "Error: pull access denied for ghcr.io/x", 125, {})
    assert r["category"] == "image_missing"
    assert r["suggested_action"] == "skip_permanently_this_machine"


def test_image_missing_unable_to_find():
    r = run_error.classify_failure(_job(), "Unable to find image 'foo:latest' locally", 125, {})
    assert r["category"] == "image_missing"


def test_gpu_busy_conflict_requires_claim_time_flag():
    log = "RuntimeError: HIP out of memory"
    # without the claim-time flag it should NOT be gpu_busy_conflict (falls back)
    r = run_error.classify_failure(_job(), log, 1, {})
    assert r["category"] == "gpu_busy_conflict"
    assert r["confidence"] < 0.9  # lower confidence path
    r2 = run_error.classify_failure(_job(gpu_busy_at_claim=True), log, 1, {})
    assert r2["category"] == "gpu_busy_conflict"
    assert r2["confidence"] >= 0.9
    assert r2["suggested_action"] == "requeue_backoff_5min"


def test_oom_killed_137_no_cuda_signature():
    r = run_error.classify_failure(_job(), "process killed\n...", 137, {})
    assert r["category"] == "oom_killed"
    assert r["suggested_action"] == "requeue_once_needs_review"


def test_oom_killed_with_dmesg_confirmation():
    r = run_error.classify_failure(_job(), "process killed", 137,
                                    {"dmesg_tail": "Out of memory: Killed process 123"})
    assert r["category"] == "oom_killed"
    assert r["confidence"] >= 0.9


def test_scene_missing():
    job = _job(scene="/data/kitchen1")
    log = "FileNotFoundError: [Errno 2] No such file or directory: '/data/kitchen1/rgb.txt'"
    r = run_error.classify_failure(job, log, 1, {})
    assert r["category"] == "scene_missing"
    assert r["suggested_action"] == "quarantine_no_retry"


def test_network_transient_dns():
    r = run_error.classify_failure(_job(), "Temporary failure in name resolution", 1, {})
    assert r["category"] == "network_transient"
    assert r["suggested_action"] == "requeue_backoff_2min"


def test_network_transient_stale_file_handle():
    r = run_error.classify_failure(_job(), "OSError: [Errno 116] Stale file handle", 1, {})
    assert r["category"] == "network_transient"


def test_code_error_traceback():
    log = "Traceback (most recent call last):\n  File x.py\nValueError: bad shape"
    r = run_error.classify_failure(_job(), log, 1, {})
    assert r["category"] == "code_error"
    assert r["suggested_action"] == "quarantine_code_fix_needed"


def test_unknown_no_signature():
    r = run_error.classify_failure(_job(), "training ended for unclear reasons", 1, {})
    assert r["category"] == "unknown"
    assert r["suggested_action"] == "retry_once_then_quarantine"


def test_priority_disk_full_over_traceback():
    log = "Traceback (most recent call last):\nOSError: No space left on device"
    r = run_error.classify_failure(_job(), log, 1, {})
    assert r["category"] == "disk_full"


# --- pause-flag lifecycle (queue_runner.py) ----------------------------------

def test_pause_flag_create_and_clear(tmp_queue, monkeypatch):
    monkeypatch.setattr(qr, "machine_name", lambda: "testmachine")
    assert not qr.is_paused("testmachine")
    path = qr.write_pause_flag("disk_full", "evidence text", "testmachine")
    assert os.path.exists(path)
    assert qr.is_paused("testmachine")
    content = open(path).read()
    assert "category=disk_full" in content
    assert "evidence text" in content
    cleared = qr.clear_pause_flag("testmachine")
    assert cleared is True
    assert not qr.is_paused("testmachine")


def test_clear_pause_flag_missing_returns_false(tmp_queue, monkeypatch):
    monkeypatch.setattr(qr, "machine_name", lambda: "testmachine")
    assert qr.clear_pause_flag("testmachine") is False


def test_claim_next_refuses_when_paused(tmp_queue, monkeypatch, tmp_path):
    monkeypatch.setattr(qr, "machine_name", lambda: "testmachine")
    with open(qr.QUEUE, "w") as f:
        f.write('{"id": "a", "status": "pending", "machine": "any", '
                '"scene": "x", "model_path": "output/scratch/a"}\n')
    qr.write_pause_flag("disk_full", "evidence", "testmachine")
    assert qr.claim_next() is None
    qr.clear_pause_flag("testmachine")


# --- not_before backoff gating -----------------------------------------------

def test_not_before_future_not_claimable():
    job = {"not_before": time.time() + 300}
    assert qr.not_before_ok(job) is False


def test_not_before_past_claimable():
    job = {"not_before": time.time() - 300}
    assert qr.not_before_ok(job) is True


def test_not_before_absent_claimable():
    assert qr.not_before_ok({}) is True


def test_claim_next_skips_future_not_before(tmp_queue, monkeypatch):
    monkeypatch.setattr(qr, "machine_name", lambda: "testmachine")
    monkeypatch.setattr(qr, "gpu_busy", lambda: False)
    import json
    jobs = [
        {"id": "future", "status": "pending", "machine": "any",
         "scene": "x", "model_path": "output/scratch/future",
         "not_before": time.time() + 300},
        {"id": "ready", "status": "pending", "machine": "any",
         "scene": "x", "model_path": "output/scratch/ready"},
    ]
    with open(qr.QUEUE, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    claimed = qr.claim_next()
    assert claimed["id"] == "ready"

# The status/errors command-output tests that used to live here exercised
# scripts/ablation_runner.py, which was deliberately removed (commit
# 73294cc, "docs: migrate ablator docs off ablation_runner, remove legacy
# runner script") in favor of the `ablator` package/submodule (see
# CLAUDE.md's "Related Repos" section) -- this test file was never updated
# after that removal, so its module-level importlib.util.spec_from_file_location
# for a now-nonexistent script broke collection for the ENTIRE file (the
# still-valid queue/pause-flag tests above included). Removed the dead
# block; the equivalent status/errors-formatting behavior now belongs to
# the ablator package's own test suite, not this repo.
