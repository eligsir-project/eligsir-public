"""Issue #278 — frame-release cadence under ``--online_mode``.

These tests pin the *actual* release behavior that produced #278's headline
symptom on the fr3 champion recipe: the entire 2525-frame TUM stream was
released within ~1250 training iterations despite the run's own startup banner
advertising ``effective_spf=150`` / ``steps_per_frame=150``.

Root cause (see
``docs/architecture/2026-08-13-issue278-online-mode-dataset-fps-regression.md``):
``resolve_ingestion_mode(..., online_mode=True)`` returns ``dataset_fps``, whose
``should_release()`` branch never reads ``steps_per_frame`` at all. The frame
stream is instead paced against a *simulated wall clock* compared to
``frame_idx / fps_cap`` — so a 2525-frame/30fps sequence (84.2 s of dataset
time) is entirely "due" as soon as ~84 s of real training time has elapsed, and
the only remaining limiter is ``--streaming_max_drain_per_iter``.

Deliberately in its own file: ``tests/test_stream_scheduler.py`` is modified by
open PR #269, which must not be conflicted with.
"""
from __future__ import annotations

import pytest

from eligsir.mapping.scheduler import FrameScheduler, resolve_ingestion_mode
from eligsir.mapping.scheduler.frame_scheduler import (
    INGESTION_MODE_DATASET_FPS,
    INGESTION_MODE_ITER_BASED,
)


TUM_FR3_N_FRAMES = 2525          # eval/scenes.json n_source_frames for fr3_long_office
DEFAULT_FPS_CAP = 30.0           # arguments/__init__.py streaming_input_fps_cap
DEFAULT_DRAIN_CAP = 2            # arguments/__init__.py streaming_max_drain_per_iter
DEFAULT_SPF = 150                # arguments/__init__.py streaming_steps_per_frame


# ---------------------------------------------------------------------------
# 1. Mode resolution: the regression itself
# ---------------------------------------------------------------------------

def test_online_mode_on_a_local_dataset_resolves_to_dataset_fps():
    """CHARACTERIZATION of the #278 regression, not an endorsement.

    Commit 7ffaebc (2026-06-21) deliberately REMOVED the
    ``if online_mode: return INGESTION_MODE_DATASET_FPS`` branch after
    root-causing scene self-deconstruction to exactly the runaway this file
    measures below (``docs/2026-06-21-scene-explosion-isolation.md``). Commit
    12899c2 (PR #20, 2026-06-29, a ROS2 transport/package-restructure PR)
    reinstated the branch verbatim and deleted 7ffaebc's explanatory docstring
    warning along with it.

    When #278's fix lands, this assertion flips to ``INGESTION_MODE_ITER_BASED``
    — that flip is the intended, visible signal that the revert was undone.
    """
    assert resolve_ingestion_mode(
        "/data/tum/rgbd_dataset_freiburg3_long_office_household",
        "auto",
        online_mode=True,
    ) == INGESTION_MODE_DATASET_FPS


def test_offline_replay_on_the_same_dataset_resolves_to_iter_based():
    """Only the ``online_mode`` flag differs — this is the one-factor contrast."""
    assert resolve_ingestion_mode(
        "/data/tum/rgbd_dataset_freiburg3_long_office_household",
        "auto",
        online_mode=False,
    ) == INGESTION_MODE_ITER_BASED


def test_explicit_iter_based_survives_online_mode():
    """``--streaming_ingestion_mode iter_based`` is the available user-side
    workaround until the revert is undone: an explicit (non-``auto``) request
    short-circuits before the ``online_mode`` branch."""
    assert resolve_ingestion_mode(
        "/data/tum/rgbd_dataset_freiburg3_long_office_household",
        "iter_based",
        online_mode=True,
    ) == INGESTION_MODE_ITER_BASED


# ---------------------------------------------------------------------------
# 2. dataset_fps structurally ignores steps_per_frame
# ---------------------------------------------------------------------------

def _release_sequence(scheduler: FrameScheduler, n_iters: int, dt: float,
                      drain_cap: int = DEFAULT_DRAIN_CAP,
                      n_source_frames: int = TUM_FR3_N_FRAMES) -> list[int]:
    """Replay ``train_streaming.py``'s per-iteration drain loop (lines ~4049-4120).

    Mirrors the real loop's three governing facts:
      * at most ``drain_cap`` frames per iteration,
      * ``dt`` is charged to the simulated clock only on the FIRST
        ``should_release`` call of an iteration (``_drain_dt = 0.0`` after),
      * the source is finite.

    Returns cumulative frames released after each iteration.
    """
    released = 0
    out: list[int] = []
    for iteration in range(1, n_iters + 1):
        drain_dt: float | None = dt
        drained = 0
        while (
            drained < drain_cap
            and released < n_source_frames
            and scheduler.should_release(iteration, dt=drain_dt)
        ):
            drained += 1
            drain_dt = 0.0
            scheduler.mark_released()
            released += 1
        out.append(released)
    return out


@pytest.mark.parametrize("spf", [1, 20, DEFAULT_SPF, 10_000])
def test_dataset_fps_release_is_independent_of_steps_per_frame(spf):
    """The proof that ``effective_spf=150`` in the fr3 log is cosmetic.

    A 10000x spread in ``steps_per_frame`` produces a byte-identical release
    sequence, because ``should_release``'s ``dataset_fps`` branch never reads
    the attribute.
    """
    sched = FrameScheduler(
        steps_per_frame=spf,
        ingestion_mode=INGESTION_MODE_DATASET_FPS,
        fps_cap=DEFAULT_FPS_CAP,
    )
    # 0.09 s/iter ~= the 11 it/s the real fr3 run averaged during ingestion.
    seq = _release_sequence(sched, n_iters=200, dt=0.09)

    reference = FrameScheduler(
        steps_per_frame=1,
        ingestion_mode=INGESTION_MODE_DATASET_FPS,
        fps_cap=DEFAULT_FPS_CAP,
    )
    assert seq == _release_sequence(reference, n_iters=200, dt=0.09)


def test_iter_based_by_contrast_does_respect_steps_per_frame():
    """Same scheduler, same drain loop, ``iter_based``: ~1 frame per spf iters.

    Pins the *release iterations* themselves, not just the running total —
    a total-only assertion would also pass at ``drain_cap=1`` and so would not
    isolate what this test claims. ``should_release``'s iter_based branch is
    ``expected = 1 + iteration // steps_per_frame``, so one bootstrap release
    fires immediately and the next fires at each multiple of ``spf``.
    """
    sched = FrameScheduler(
        steps_per_frame=DEFAULT_SPF,
        ingestion_mode=INGESTION_MODE_ITER_BASED,
    )
    seq = _release_sequence(sched, n_iters=1500, dt=0.09)

    # seq[i] is the cumulative count after iteration i+1. A release lands on
    # exactly these iterations and nowhere else.
    release_iters = [i for i in range(1, 1501) if seq[i - 1] != (seq[i - 2] if i > 1 else 0)]
    assert release_iters == [1] + [k * DEFAULT_SPF for k in range(1, 1500 // DEFAULT_SPF + 1)]

    # Exactly one frame per release — never a batch, unlike dataset_fps.
    assert all(
        seq[i - 1] - (seq[i - 2] if i > 1 else 0) == 1 for i in release_iters
    )

    # Nowhere near draining fr3 in 1500 iterations — the whole point of #278.
    assert seq[-1] == 1 + 1500 // DEFAULT_SPF == 11


# ---------------------------------------------------------------------------
# 3. The realized fr3 cadence: drain-cap-bound, exactly 2 frames/iteration
# ---------------------------------------------------------------------------

def test_dataset_fps_saturates_and_is_drain_cap_bound_on_fr3():
    """Reproduces the exact figure read off the real fr3 champion run's tqdm bar.

    ``/output/fr3postmerge_champ_asis/train.log``
    (single ``[RUN START]``, ``[FINAL STEP 51261]``) shows F advancing as
    ``24 + 2*iteration`` — F=424@210, F=1224@610, F=2024@1010, F=2424@1210 —
    i.e. pinned at the ``streaming_max_drain_per_iter=2`` cap on EVERY
    iteration, until the 2525-frame source is exhausted at iteration ~1251.

    Mechanism: at ~11 it/s the simulated clock accrues 0.09 s per iteration
    while ``fps_cap=30`` makes each frame only 0.033 s of dataset time apart,
    so the clock outruns the frame index ~2.7x per iteration and never stops
    saying "due".
    """
    sched = FrameScheduler(
        steps_per_frame=DEFAULT_SPF,
        ingestion_mode=INGESTION_MODE_DATASET_FPS,
        fps_cap=DEFAULT_FPS_CAP,
    )
    seq = _release_sequence(sched, n_iters=1400, dt=0.09)

    # Exactly cap-bound for the whole ingestion phase.
    for iteration in (100, 210, 610, 1010, 1210):
        assert seq[iteration - 1] == DEFAULT_DRAIN_CAP * iteration, (
            f"iter {iteration}: expected drain-cap-bound "
            f"{DEFAULT_DRAIN_CAP * iteration}, got {seq[iteration - 1]}"
        )

    # Whole 2525-frame stream gone in ~1263 iterations, vs the 378_750
    # (2525 x 150) that the advertised steps_per_frame implies.
    drained_at = next(i for i, n in enumerate(seq, start=1) if n >= TUM_FR3_N_FRAMES)
    assert drained_at < 1300
    assert TUM_FR3_N_FRAMES * DEFAULT_SPF / drained_at > 250  # >250x faster


def test_the_drain_cap_binds_only_below_the_clocks_demanded_rate():
    """Locates the exact crossover between the two possible limiters.

    ``dataset_fps`` demands ``dt * fps_cap`` frames per iteration — at the fr3
    run's measured 0.09 s/iter that is 0.09*30 = **2.7 frames/iter**. So:

      * ``--streaming_max_drain_per_iter`` <= 2 is DRAIN-CAP-bound (release is
        exactly ``cap`` per iteration — what the real run did at the default 2),
      * >= 3 is CLOCK-bound and saturates at 2.7/iter regardless of the cap.

    The default cap of 2 therefore sits just barely under the demanded rate:
    it is the only thing holding fr3's release to 2/iter instead of 2.7/iter,
    and it is *not* a meaningful brake — both regimes drain the full 2525-frame
    stream in well under 1400 iterations. Raising the cap does not make #278
    worse in any interesting way, and lowering it is not a fix.
    """
    rates = {}
    for cap in (1, 2, 4, 8):
        sched = FrameScheduler(
            steps_per_frame=DEFAULT_SPF,
            ingestion_mode=INGESTION_MODE_DATASET_FPS,
            fps_cap=DEFAULT_FPS_CAP,
        )
        rates[cap] = _release_sequence(sched, n_iters=300, dt=0.09, drain_cap=cap)[-1]

    assert rates[1] == 300      # cap-bound at 1.0/iter
    assert rates[2] == 600      # cap-bound at 2.0/iter  <- the real fr3 run
    clock_bound = 300 * 0.09 * DEFAULT_FPS_CAP  # 810
    assert rates[4] == pytest.approx(clock_bound, abs=2)
    assert rates[8] == rates[4], "beyond the crossover the cap stops mattering"


def test_a_realtime_capable_run_is_not_saturated():
    """Not a blanket indictment of dataset_fps: when training actually keeps up
    with sensor rate (dt < 1/fps_cap), the clock does NOT run away and release
    tracks the dataset's own 30fps timeline as designed."""
    sched = FrameScheduler(
        steps_per_frame=DEFAULT_SPF,
        ingestion_mode=INGESTION_MODE_DATASET_FPS,
        fps_cap=DEFAULT_FPS_CAP,
    )
    # 0.01 s/iter = 100 it/s, comfortably faster than 30 fps.
    seq = _release_sequence(sched, n_iters=1000, dt=0.01)
    assert seq[-1] == pytest.approx(1000 * 0.01 * DEFAULT_FPS_CAP, rel=0.05)
    assert seq[-1] < TUM_FR3_N_FRAMES  # nowhere near drained


# ---------------------------------------------------------------------------
# 4. PR #269's adaptive-cadence mechanism cannot bear on #278
# ---------------------------------------------------------------------------

def test_set_next_interval_is_inert_under_dataset_fps():
    """``set_next_interval`` short-circuits inside the ``iter_based`` branch of
    ``should_release``, so issue #109's E3 adaptive-exposure lever (open PR
    #269) has no effect whatsoever on an ``--online_mode`` run. #278 and #269
    do not interact."""
    sched = FrameScheduler(
        steps_per_frame=DEFAULT_SPF,
        ingestion_mode=INGESTION_MODE_DATASET_FPS,
        fps_cap=DEFAULT_FPS_CAP,
    )
    baseline = _release_sequence(sched, n_iters=200, dt=0.09)

    sched2 = FrameScheduler(
        steps_per_frame=DEFAULT_SPF,
        ingestion_mode=INGESTION_MODE_DATASET_FPS,
        fps_cap=DEFAULT_FPS_CAP,
    )
    sched2.set_next_interval(0, 100_000)  # would stall release entirely if honoured
    assert _release_sequence(sched2, n_iters=200, dt=0.09) == baseline
