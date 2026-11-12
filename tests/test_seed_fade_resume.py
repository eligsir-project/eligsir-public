"""Issue #290: a resumed run's seed-fade verdict must either be FAITHFUL
(detector state carried across the checkpoint) or INDETERMINATE -- never
silently wrong.

The regression these guard: ``SeedFadeMonitor._running_max`` was in-memory
only, so any run resumed from ``--start_checkpoint`` (ablator requeue,
preemption, the GPU-memory-danger SIGTERM that hit
``issue290seedfadecross_s8b_ctrl`` at iter 7712) graded itself against a
running max that restarted from ``-inf`` mid-run.
"""

import json
import os

import pytest

from eligsir.utils.seed_fade_monitor import STATE_VERSION, SeedFadeMonitor
from eligsir.utils.seed_fade_checker import check_seed_fade_block, check_run_dir, detect_resume

# utils.comparison_report and eligsir.mapping.integration.checkpoint_io both
# pull in torchvision, which is broken outside the training container on some
# hosts (`operator torchvision::nms does not exist`) -- the same cause as the
# ~38 pre-existing collection errors in this suite. Guarded rather than
# unconditionally imported so the pure-monitor/checker tests below still run
# everywhere; the guarded ones run in the container.
try:  # pragma: no cover - environment probe
    import torchvision  # noqa: F401
    from torchvision.utils import save_image  # noqa: F401
    _TORCHVISION_OK = True
except Exception:  # pragma: no cover
    _TORCHVISION_OK = False

requires_torchvision = pytest.mark.skipif(
    not _TORCHVISION_OK, reason="torchvision unavailable outside the container"
)


# A series with a real, deep fade BEFORE the resume boundary and a healthy
# stretch after it: the exact shape whose verdict flips if _running_max resets.
PRE_RESUME = [(i, v) for i, v in zip(range(0, 2000, 100), [
    10.0, 14.0, 18.0, 22.0, 25.0, 25.5, 26.0,   # climb to a real peak
    18.0, 17.0, 16.5, 16.0, 15.8,               # sustained ~10dB fade
    16.2, 17.0, 18.0, 19.0, 20.0, 21.0, 21.5, 21.8,
])]
POST_RESUME = [(2000 + i * 100, v) for i, v in enumerate(
    [22.0, 22.2, 22.4, 22.3, 22.5, 22.6, 22.4, 22.7, 22.8, 22.6]
)]

MONITOR_KWARGS = dict(
    tolerance_db=2.0, sustained_checks=3, warmup_iters=0, sample_interval=100
)


def _run(samples, monitor=None):
    m = monitor or SeedFadeMonitor(**MONITOR_KWARGS)
    for it, v in samples:
        m.update(it, v)
    return m


# ---------------------------------------------------------------------------
# persistence: a restored resume must equal the uninterrupted run exactly
# ---------------------------------------------------------------------------

def test_restored_resume_verdict_identical_to_uninterrupted():
    uninterrupted = _run(PRE_RESUME + POST_RESUME)

    first = _run(PRE_RESUME)
    payload = first.state_dict()
    second = SeedFadeMonitor(**MONITOR_KWARGS)
    assert second.mark_resumed(2000, payload) is True
    _run(POST_RESUME, second)

    a = uninterrupted.report_block()
    b = second.report_block()
    # Everything the verdict is made of must match, field for field.
    for key in ("detected", "max_dip_db", "deepest_dip_db", "fade_start_iter",
                "first_fade_start_iter", "last_fade_start_iter", "recovered",
                "recovered_final", "fade_active_at_end", "status", "n_samples",
                "n_samples_in_fade", "episode_count", "episodes",
                "running_max_db", "trough_iter", "trough_psnr",
                "trough_reference_psnr", "min_psnr", "final_psnr", "final_iter"):
        assert b[key] == a[key], f"{key}: resumed={b[key]!r} uninterrupted={a[key]!r}"
    assert b["indeterminate"] is False
    assert b["resume_state_restored"] is True
    assert b["resumed_from_iteration"] == 2000


def test_unrestored_resume_would_have_reported_clean():
    """The bug, demonstrated: without persistence the post-resume segment alone
    reads CLEAN even though the run really faded ~10dB before the boundary."""
    uninterrupted = _run(PRE_RESUME + POST_RESUME)
    assert uninterrupted.report_block()["detected"] is True

    naive = _run(POST_RESUME)  # what the old code did on resume
    assert naive.report_block()["detected"] is False

    # With the fix, the same naive monitor is forced to declare itself unusable.
    marked = SeedFadeMonitor(**MONITOR_KWARGS)
    assert marked.mark_resumed(2000, None) is False
    _run(POST_RESUME, marked)
    blk = marked.report_block()
    assert blk["detected"] is False           # detection semantics unchanged
    assert blk["indeterminate"] is True       # ...but not quotable
    assert blk["resume_state_restored"] is False
    assert blk["indeterminate_reasons"]


# ---------------------------------------------------------------------------
# every refusal path falls back to INDETERMINATE, never a wrong number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload,why", [
    (None, "no payload at all (pre-#290 checkpoint)"),
    ({}, "empty payload"),
    ("garbage", "wrong type"),
    ({"version": STATE_VERSION - 1, "state": {}, "config": {}}, "old version"),
    ({"version": STATE_VERSION, "config": {}}, "no state mapping"),
])
def test_bad_or_old_payload_is_indeterminate(payload, why):
    m = SeedFadeMonitor(**MONITOR_KWARGS)
    assert m.mark_resumed(7712, payload) is False, why
    assert m.indeterminate is True
    assert m.indeterminate_reasons
    # Nothing was applied: a refused payload must not half-restore state.
    assert m._running_max == float("-inf")
    assert m.n_samples == 0


def test_truncated_payload_is_refused_wholesale():
    good = _run(PRE_RESUME).state_dict()
    good["state"].pop("_running_max")
    m = SeedFadeMonitor(**MONITOR_KWARGS)
    assert m.mark_resumed(2000, good) is False
    assert m._running_max == float("-inf")
    assert any("missing field" in r for r in m.resume_notes)


def test_config_change_across_resume_is_indeterminate():
    payload = _run(PRE_RESUME).state_dict()
    changed = SeedFadeMonitor(**{**MONITOR_KWARGS, "tolerance_db": 4.0})
    assert changed.mark_resumed(2000, payload) is False
    assert any("tolerance_db" in r for r in changed.resume_notes)
    assert changed._running_max == float("-inf")


def test_state_dict_is_json_serializable():
    # It rides inside a checkpoint written in a SIGTERM grace window: plain
    # scalars only, no tensors, and -inf must round-trip.
    fresh = SeedFadeMonitor(**MONITOR_KWARGS)
    payload = json.loads(json.dumps(fresh.state_dict()))
    other = SeedFadeMonitor(**MONITOR_KWARGS)
    assert other.mark_resumed(1, payload) is True
    assert other._running_max == float("-inf")

    used = json.loads(json.dumps(_run(PRE_RESUME).state_dict()))
    m = SeedFadeMonitor(**MONITOR_KWARGS)
    assert m.mark_resumed(2000, used) is True


def test_no_resume_leaves_block_unchanged_in_meaning():
    blk = _run(PRE_RESUME + POST_RESUME).report_block()
    assert blk["resumed_from_iteration"] is None
    assert blk["resume_state_restored"] is None
    assert blk["indeterminate"] is False
    assert blk["indeterminate_reasons"] == []


# ---------------------------------------------------------------------------
# checker / audit path
# ---------------------------------------------------------------------------

def test_checker_refuses_indeterminate_block():
    m = SeedFadeMonitor(**MONITOR_KWARGS)
    m.mark_resumed(7712, None)
    _run(POST_RESUME, m)
    v = check_seed_fade_block(m.report_block())
    assert v.certified is False
    assert any("#290" in r for r in v.refusals)


def test_checker_certifies_restored_resume():
    first = _run(PRE_RESUME)
    m = SeedFadeMonitor(**MONITOR_KWARGS)
    m.mark_resumed(2000, first.state_dict())
    _run(POST_RESUME, m)
    v = check_seed_fade_block(m.report_block())
    assert not v.refusals, v.refusals
    assert not v.errors, v.errors
    assert v.certified is True


def test_top_level_mirror_mismatch_is_an_error():
    m = SeedFadeMonitor(**MONITOR_KWARGS)
    m.mark_resumed(7712, None)
    _run(POST_RESUME, m)
    v = check_seed_fade_block(m.report_block(), report={"seed_fade_indeterminate": False})
    assert any("seed_fade_indeterminate" in e for e in v.errors)


def _write_run(tmp_path, name, *, report=None, run_args=None, train_log=None):
    run = tmp_path / name
    (run / "comparison" / "iter_4000").mkdir(parents=True)
    if report is not None:
        (run / "comparison" / "iter_4000" / "report.json").write_text(json.dumps(report))
    if run_args is not None:
        (run / "run_args").write_text(json.dumps(run_args))
    if train_log is not None:
        (run / "train.log").write_text(train_log)
    return str(run)


def test_detect_resume_from_train_log_alone(tmp_path):
    """The pre-#290 corpus case: nothing in the report says 'resumed', but the
    log does. This is what retroactively invalidates already-finished runs."""
    run = _write_run(
        tmp_path, "old_run",
        report={"seed_fade": {"detected": False}},
        run_args={"start_checkpoint": None},
        train_log="[streaming] resume: restored frame-source cursor _source_idx=16\n",
    )
    with open(os.path.join(run, "comparison", "iter_4000", "report.json")) as f:
        report = json.load(f)
    res = detect_resume(run, report)
    assert res["resumed"] is True
    assert res["indeterminate"] is True
    assert res["state_restored"] is None
    assert any("train.log" in e for e in res["evidence"])


def test_detect_resume_from_run_args(tmp_path):
    run = _write_run(tmp_path, "requeued",
                     run_args={"start_checkpoint": "output/x/chkpnt7712.pth"})
    res = detect_resume(run, None)
    assert res["resumed"] is True
    assert any("start_checkpoint" in e for e in res["evidence"])


def test_detect_resume_negative(tmp_path):
    run = _write_run(tmp_path, "clean", report={"resumed_from_iteration": None},
                     run_args={"start_checkpoint": None},
                     train_log="[streaming] nothing to see here\n")
    res = detect_resume(run, {"resumed_from_iteration": None})
    assert res["resumed"] is False
    assert res["indeterminate"] is False
    assert res["evidence"] == []


def test_check_run_dir_refuses_resumed_run(tmp_path):
    m = SeedFadeMonitor(**MONITOR_KWARGS)
    _run(PRE_RESUME + POST_RESUME, m)
    blk = m.report_block()   # a fully-valid, post-#282, non-resumed block...
    run = _write_run(
        tmp_path, "resumed_run",
        report={"seed_fade": blk, "resumed_from_iteration": 2000},
        train_log="[streaming] resume: restored frame-source cursor\n",
    )
    v = check_run_dir(run)
    # ...that is nonetheless refused, purely on on-disk resume evidence.
    assert v.certified is False
    assert any("RESUMED" in r for r in v.refusals)


def test_check_run_dir_certifies_restored_resume(tmp_path):
    first = _run(PRE_RESUME)
    m = SeedFadeMonitor(**MONITOR_KWARGS)
    m.mark_resumed(2000, first.state_dict())
    _run(POST_RESUME, m)
    run = _write_run(
        tmp_path, "faithful_resume",
        report={"seed_fade": m.report_block(), "resumed_from_iteration": 2000},
        train_log="[streaming] resume: restored seed-fade detector state\n",
    )
    v = check_run_dir(run)
    assert not v.refusals, v.refusals
    assert v.certified is True


# ---------------------------------------------------------------------------
# report + promotion gate
# ---------------------------------------------------------------------------

def _write_report(tmp_path, name, *, seed_fade, resumed_from_iteration):
    """Drive the real write_post_training_report with the smallest inputs it
    accepts (mirrors tests/test_realized_holdout_hash.py's fixtures)."""
    import torch
    from eligsir.utils.comparison_report import write_post_training_report

    class _Pipe:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False

    class _G:
        def get_xyz(self):
            return torch.zeros(0, 3)

    out = tmp_path / name
    out.mkdir()
    return write_post_training_report(
        model_path=str(out),
        iteration=5000,
        gaussians=_G(),
        train_cams=[],
        test_cams=[],
        render_fn=lambda cam, g, pipe, bg: {"render": torch.zeros(3, 8, 8)},
        pipe=_Pipe(),
        background=None,
        seed_fade=seed_fade,
        resumed_from_iteration=resumed_from_iteration,
    )


@requires_torchvision
def test_report_derives_indeterminate_without_monitor_cooperation(tmp_path):
    """`resumed_from_iteration` is plumbed independently of the monitor, so a
    call path that resumes but forgets mark_resumed() still cannot report a
    clean seed-fade verdict by omission."""
    summary = _write_report(
        tmp_path, "forgot_mark_resumed",
        # A block from a monitor that never learned it was resumed: no
        # resume_state_restored assertion anywhere.
        seed_fade={"detected": False, "max_dip_db": 0.0},
        resumed_from_iteration=2000,
    )
    assert summary["seed_fade_indeterminate"] is True
    assert summary["seed_fade"]["indeterminate"] is True
    assert summary["seed_fade"]["indeterminate_reasons"]
    # ...and alongside, never replacing, the #346 severity flag.
    assert "seed_fade_confounded" in summary


@requires_torchvision
def test_report_clean_for_restored_resume(tmp_path):
    first = _run(PRE_RESUME)
    m = SeedFadeMonitor(**MONITOR_KWARGS)
    m.mark_resumed(2000, first.state_dict())
    _run(POST_RESUME, m)
    summary = _write_report(tmp_path, "restored", seed_fade=m.report_block(),
                            resumed_from_iteration=2000)
    assert summary["seed_fade_indeterminate"] is False


@requires_torchvision
def test_report_clean_for_non_resumed_run(tmp_path):
    summary = _write_report(tmp_path, "scratch",
                            seed_fade=_run(PRE_RESUME).report_block(),
                            resumed_from_iteration=None)
    assert summary["seed_fade_indeterminate"] is False


# ---------------------------------------------------------------------------
# checkpoint plumbing
# ---------------------------------------------------------------------------

def test_state_dict_round_trips_through_torch_save(tmp_path):
    import torch

    payload = _run(PRE_RESUME).state_dict()
    p = tmp_path / "chkpnt2000.pth"
    torch.save(({"fake": "model"}, 2000, {"source_idx": 16, "seed_fade": payload}), p)
    _state, it, meta = torch.load(p, weights_only=False)
    m = SeedFadeMonitor(**MONITOR_KWARGS)
    assert m.mark_resumed(it, meta.get("seed_fade")) is True
    assert m.n_samples == len(PRE_RESUME)


@requires_torchvision
def test_checkpoint_writer_embeds_seed_fade_state(tmp_path):
    from eligsir.mapping.integration import checkpoint_io

    captured = {}

    class _Worker:
        def enqueue(self, fn):
            captured["fn"] = fn

    class _G:
        def capture(self):
            return {"fake": "model"}

    monitor = _run(PRE_RESUME)
    checkpoint_io._save_streaming_checkpoint(
        _Worker(), _G(), str(tmp_path), 2000,
        n_frames_ingested=16, last_ingest_iter=1900, source_idx=16,
        seed_fade_monitor=monitor,
    )
    captured["fn"]()   # run the deferred write

    import torch
    _state, it, meta = torch.load(tmp_path / "chkpnt2000.pth", weights_only=False)
    assert "seed_fade" in meta
    restored = SeedFadeMonitor(**MONITOR_KWARGS)
    assert restored.mark_resumed(it, meta["seed_fade"]) is True
    assert restored.report_block()["running_max_db"] == \
        monitor.report_block()["running_max_db"]


@requires_torchvision
def test_checkpoint_writer_without_monitor_is_unchanged(tmp_path):
    """Omitting the monitor must reproduce the pre-#290 payload exactly."""
    from eligsir.mapping.integration import checkpoint_io

    captured = {}

    class _Worker:
        def enqueue(self, fn):
            captured["fn"] = fn

    class _G:
        def capture(self):
            return {"fake": "model"}

    checkpoint_io._save_streaming_checkpoint(
        _Worker(), _G(), str(tmp_path), 2000,
        n_frames_ingested=16, last_ingest_iter=1900, source_idx=16,
    )
    captured["fn"]()
    import torch
    _state, _it, meta = torch.load(tmp_path / "chkpnt2000.pth", weights_only=False)
    assert "seed_fade" not in meta


# ---------------------------------------------------------------------------
# audit-path precision
# ---------------------------------------------------------------------------

def test_only_the_last_run_start_segment_counts(tmp_path):
    """ablator's requeue reuses model_path, so train.log concatenates attempts.
    An abandoned earlier attempt that resumed must not condemn a final attempt
    that ran clean from scratch."""
    run = _write_run(
        tmp_path, "requeued_clean",
        train_log=(
            "[RUN START] attempt 1\n"
            "[streaming] resume: restored frame-source cursor\n"
            "[RUN START] attempt 2\n"
            "[streaming] bootstrap from scratch\n"
        ),
    )
    assert detect_resume(run, None)["resumed"] is False


def test_last_segment_resume_is_still_caught(tmp_path):
    run = _write_run(
        tmp_path, "requeued_resumed",
        train_log=(
            "[RUN START] attempt 1\n"
            "[streaming] bootstrap from scratch\n"
            "[RUN START] attempt 2\n"
            "[streaming] resume: restored frame-source cursor\n"
        ),
    )
    assert detect_resume(run, None)["resumed"] is True


def test_resume_below_warmup_is_annotated_but_still_indeterminate(tmp_path):
    """The real local-corpus shape: resumed at iter 100 with warmup_iters=2000,
    so the detector had taken NO samples yet and nothing was actually lost.
    Still flagged (the warmup value is itself read from artifacts), but the
    note stops it reading as a false alarm."""
    res = detect_resume(
        _write_run(tmp_path, "warm_start"),
        {"resumed_from_iteration": 100,
         "seed_fade": {"warmup_iters": 2000, "detected": False}},
    )
    assert res["resumed_below_warmup"] is True
    assert res["indeterminate"] is True
    assert any("no decision state was actually lost" in e for e in res["evidence"])


def test_resume_above_warmup_has_no_such_note(tmp_path):
    res = detect_resume(
        _write_run(tmp_path, "mid_run"),
        {"resumed_from_iteration": 7712,
         "seed_fade": {"warmup_iters": 2000, "detected": False}},
    )
    assert res["resumed_below_warmup"] is False
    assert not any("actually lost" in e for e in res["evidence"])


def test_promotion_gate_blocks_indeterminate():
    from eligsir.utils.champion_promotion import validate_report_contract

    report = {
        "eval_valid": True,
        "n_test": 10,
        "mean_test_psnr": 27.0,
        "seed_fade": {"detected": False, "indeterminate": True,
                      "indeterminate_reasons": ["state not restored"]},
        "seed_fade_indeterminate": True,
    }
    res = validate_report_contract(report)
    assert any("INDETERMINATE" in b for b in res.blockers)


def test_promotion_gate_allows_faithful_resume():
    from eligsir.utils.champion_promotion import validate_report_contract

    report = {
        "eval_valid": True,
        "n_test": 10,
        "mean_test_psnr": 27.0,
        "seed_fade": {"detected": False, "indeterminate": False},
        "seed_fade_indeterminate": False,
    }
    res = validate_report_contract(report)
    assert not any("INDETERMINATE" in b for b in res.blockers)
