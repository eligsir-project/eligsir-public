"""Seed-view resolution provenance in SeedFadeMonitor (issue #346 sub-finding).

Motivating measurement-validity defect: ``optimization/seed_psnr`` renders
``streaming_scene.train_cameras[0]``, a TRAINING camera, whose
``original_image``/``image_height``/``image_width`` are rebound to the active
multires tier by ``bind_resolution_tier`` on the iterations where it is the
sampled viewpoint. PSNR is not comparable across image sizes, so
``max_dip_db`` (running max minus a strictly later sample) could silently span
two measurement regimes with nothing in ``report.json`` recording it.

These tests pin: (a) detection semantics are byte-identical whether or not a
resolution is supplied, (b) the new keys are None/False-safe for every call
site that does not supply one, (c) a real cross-regime dip is flagged, and
(d) the state survives a checkpoint round-trip without making an old
checkpoint's resume indeterminate.
"""

import pytest

from eligsir.utils.seed_fade_monitor import SeedFadeMonitor


def _mk(**kw):
    kw.setdefault("tolerance_db", 2.0)
    kw.setdefault("sustained_checks", 3)
    kw.setdefault("sample_interval", 100)
    return SeedFadeMonitor(**kw)


# --- (a) detection is unchanged --------------------------------------------

def test_detection_identical_with_and_without_resolution():
    """The resolution argument must not touch a single detection number."""
    series = [25.0, 26.0, 27.0, 24.0, 20.0, 19.0, 18.0, 26.5, 27.2, 12.0, 11.0, 10.5]
    plain, tagged = _mk(), _mk()
    for i, v in enumerate(series):
        it = (i + 1) * 100
        plain.update(it, v)
        tagged.update(it, v, resolution=(480, 640))

    detection_keys = [
        "detected", "max_dip_db", "fade_start_iter", "recovered",
        "status", "n_samples", "n_samples_in_fade", "episode_count",
        "first_fade_start_iter", "last_fade_start_iter", "deepest_dip_db",
        "trough_iter", "trough_psnr", "trough_reference_psnr",
        "running_max_db", "final_psnr", "final_iter", "min_psnr", "episodes",
    ]
    a, b = plain.report_block(), tagged.report_block()
    assert {k: a[k] for k in detection_keys} == {k: b[k] for k in detection_keys}
    assert a["detected"] is True, "series must actually fade or this proves nothing"


# --- (b) unrecorded == None, never a silent 'stable' -----------------------

def test_no_resolution_reports_none_not_false():
    m = _mk()
    for i, v in enumerate([25.0, 24.0, 20.0, 19.0, 18.0]):
        m.update((i + 1) * 100, v)
    b = m.report_block()
    assert b["seed_view_resolutions"] is None
    assert b["seed_view_resolution_counts"] is None
    assert b["resolution_changed"] is None
    assert b["max_dip_cross_resolution"] is None
    assert b["resolution_confounded"] is None, (
        "unknown must not be reported as 'not confounded' -- the whole point "
        "is that the question is unanswerable for runs predating this code"
    )
    assert b["resolution_confounded_reasons"] == []
    assert b["n_samples_with_resolution"] == 0


def test_single_resolution_is_not_confounded():
    m = _mk()
    for i, v in enumerate([25.0, 24.0, 20.0, 19.0, 18.0]):
        m.update((i + 1) * 100, v, resolution=(480, 640))
    b = m.report_block()
    assert b["seed_view_resolutions"] == ["480x640"]
    assert b["seed_view_resolution_counts"] == {"480x640": 5}
    assert b["resolution_changed"] is False
    assert b["max_dip_cross_resolution"] is False
    assert b["resolution_confounded"] is False
    assert b["resolution_confounded_reasons"] == []
    assert b["n_samples_with_resolution"] == 5


# --- (c) the real defect: a cross-regime dip -------------------------------

def test_cross_resolution_dip_is_flagged():
    """Peak measured at 120x160, deepest dip at 480x640 -- the M0 shape."""
    m = _mk()
    it = 0
    for v in [22.0, 25.0, 27.0]:          # low tier, sets the running max
        it += 100
        m.update(it, v, resolution=(120, 160))
    for v in [26.0, 20.0, 14.0, 11.0]:     # restored tier, dip lands here
        it += 100
        m.update(it, v, resolution=(480, 640))

    b = m.report_block()
    assert b["max_dip_reference_resolution"] == "120x160"
    assert b["trough_resolution"] == "480x640"
    assert b["max_dip_cross_resolution"] is True
    assert b["resolution_changed"] is True
    assert b["resolution_confounded"] is True
    assert len(b["resolution_confounded_reasons"]) == 2
    assert any("not comparable across image sizes" in r
               for r in b["resolution_confounded_reasons"])
    assert any("120x160" in r and "480x640" in r
               for r in b["resolution_confounded_reasons"])
    # Most-sampled-first ordering.
    assert b["seed_view_resolutions"] == ["480x640", "120x160"]
    # And the dip it is warning about is really the reported one.
    assert b["deepest_dip_db"] == pytest.approx(27.0 - 11.0, abs=0.01)


def test_resolution_change_without_cross_regime_dip_still_confounded():
    """Changed units are enough on their own -- both halves of the dip can
    sit in the same regime while earlier samples were in another."""
    m = _mk()
    it = 0
    for v in [20.0, 21.0]:
        it += 100
        m.update(it, v, resolution=(120, 160))
    for v in [27.0, 26.0, 20.0, 12.0]:
        it += 100
        m.update(it, v, resolution=(480, 640))
    b = m.report_block()
    assert b["max_dip_cross_resolution"] is False
    assert b["resolution_changed"] is True
    assert b["resolution_confounded"] is True
    assert len(b["resolution_confounded_reasons"]) == 1


# --- (d) hostile input must never kill a training run ----------------------

@pytest.mark.parametrize("bad", [None, (0, 640), (480, -1), ("a", "b"), (480,), 480])
def test_bad_resolution_is_treated_as_unrecorded(bad):
    m = _mk()
    m.update(100, 25.0, resolution=bad)
    assert m.n_samples == 1, "the sample itself must still count"
    assert m.report_block()["resolution_confounded"] is None


def test_warmup_samples_do_not_contribute_resolutions():
    m = _mk(warmup_iters=500)
    m.update(100, 30.0, resolution=(120, 160))
    m.update(600, 25.0, resolution=(480, 640))
    b = m.report_block()
    assert b["seed_view_resolutions"] == ["480x640"]
    assert b["resolution_changed"] is False


# --- (e) checkpoint round-trip ---------------------------------------------

def test_state_dict_round_trip_preserves_resolution_provenance():
    src = _mk()
    it = 0
    for v in [22.0, 27.0]:
        it += 100
        src.update(it, v, resolution=(120, 160))
    for v in [20.0, 14.0, 11.0]:
        it += 100
        src.update(it, v, resolution=(480, 640))

    dst = _mk()
    assert dst.load_state_dict(src.state_dict()) == []
    assert dst.report_block()["resolution_confounded"] is True
    assert dst.report_block()["max_dip_reference_resolution"] == "120x160"
    assert dst.report_block()["trough_resolution"] == "480x640"
    assert (dst.report_block()["seed_view_resolution_counts"]
            == {"120x160": 2, "480x640": 3})


def test_old_checkpoint_without_resolution_section_still_restores_cleanly():
    """The load must NOT become indeterminate just because a checkpoint
    predates this field -- that would be a regression on issue #290."""
    src = _mk()
    for i, v in enumerate([25.0, 24.0, 20.0, 18.0]):
        src.update((i + 1) * 100, v)
    payload = src.state_dict()
    del payload["resolution"]          # simulate a pre-#346 checkpoint

    dst = _mk()
    assert dst.load_state_dict(payload) == []
    assert dst.mark_resumed(400, payload) is True
    assert dst.report_block()["indeterminate"] is False
    assert dst.report_block()["resolution_confounded"] is None
