"""Tests for #428 Phase 1 -- the canonical causal reference/eligibility builder.

Torch-free by construction (numpy only), so this file collects and runs in
the lean CPU CI image as well as in the training image.

Fixtures are synthetic: hand-placed world points and camera centres, with a
cell size of 1.0 m so every expected ``cell_key`` is derivable by hand. No
real scene, no real depth, no GPU.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from eligsir.utils.causal_reference import (
    CausalReference,
    CausalReferenceBuilder,
    CausalReferenceError,
    CausalReferenceParams,
    MicrocellRegionIdentity,
    PinholeIntrinsics,
    SourceFrame,
    backproject_depth_image,
    build_causal_reference,
    build_holdout_membership,
    compute_eligibility_hash,
    read_causal_reference,
    source_frame_from_depth_image,
    spec_version_minor,
    write_causal_reference,
)
from eligsir.utils.local_quality import cell_ijk, cell_key

CELL = 1.0
CELL_A = np.array([0.5, 0.5, 0.5])          # -> ijk (0,0,0)
CELL_B = np.array([5.5, 0.5, 0.5])          # -> ijk (5,0,0)


def key_of(point, cell_size=CELL) -> int:
    return int(cell_key(cell_ijk(np.asarray(point, dtype=float).reshape(1, 3), cell_size))[0])


def make_frame(index, center, cells, n_pix=16, depth=2.0, incidence=0.0, frame_id=None):
    """A SourceFrame that puts ``n_pix`` points into each cell centre listed."""
    pts = np.repeat(np.asarray(cells, dtype=float).reshape(-1, 3), n_pix, axis=0)
    return SourceFrame(
        index=index,
        frame_id=frame_id or f"f{index:05d}",
        camera_center=np.asarray(center, dtype=float),
        points_world=pts,
        point_depth=np.full(pts.shape[0], float(depth)),
        point_incidence_deg=np.full(pts.shape[0], float(incidence)),
    )


def base_params(**kw) -> CausalReferenceParams:
    d = dict(cell_size_m=CELL, m_pix=8, d_min=0.3, d_max=5.0, theta_max_deg=75.0,
             K_vis=2, b_min_m=0.10, alpha_min_deg=5.0)
    d.update(kw)
    return CausalReferenceParams(**d)


def build(frames, *, declared=100, params=None, **kw):
    return build_causal_reference(
        frames,
        scene_id="synthetic",
        n_source_frames_declared=declared,
        params=params or base_params(),
        reference_pose_source_id="fixture_gt",
        **kw,
    )


# ---------------------------------------------------------------------------
# t_visible / eligibility core
# ---------------------------------------------------------------------------
def test_t_visible_is_the_frame_that_reaches_K_vis():
    frames = [
        make_frame(1, [0.0, 0.0, -3.0], [CELL_A]),
        make_frame(2, [2.0, 0.0, -3.0], [CELL_A]),   # distinct by baseline -> K_vis=2 reached
        make_frame(3, [4.0, 0.0, -3.0], [CELL_A]),
    ]
    ref = build(frames)
    k = key_of(CELL_A)
    assert ref.status == "ok"
    assert ref.t_visible_of(k) == 2
    assert ref.n_src_of(k) == 3
    assert ref.p_visible_of(k) == pytest.approx(2 / 100.0)


def test_p_visible_uses_the_DECLARED_denominator_not_the_realized_stream():
    """Spec §3.1 / S5: renormalizing against a short realized stream is the
    incomplete-run gaming vector, so the declared N is required and used."""
    frames = [make_frame(i, [i * 2.0, 0.0, -3.0], [CELL_A]) for i in (1, 2)]
    ref = build(frames, declared=1000)
    assert ref.n_source_frames_declared == 1000
    assert ref.n_source_frames_realized == 2
    assert ref.progress_denominator_ok is False
    assert "progress_denominator_mismatch" in ref.notes
    assert ref.p_visible_of(key_of(CELL_A)) == pytest.approx(2 / 1000.0)


def test_declared_denominator_is_required_and_never_inferred():
    with pytest.raises(ValueError):
        CausalReferenceBuilder(scene_id="s", n_source_frames_declared=0)
    sig = inspect.signature(CausalReferenceBuilder.__init__)
    assert sig.parameters["n_source_frames_declared"].default is inspect.Parameter.empty


def test_eligibility_is_monotone_in_progress():
    frames = [make_frame(i, [i * 2.0, 0.0, -3.0], [CELL_A]) for i in (1, 2, 3)]
    ref = build(frames, declared=10)
    k = key_of(CELL_A)
    seen_true = False
    for p in np.linspace(0.0, 1.0, 51):
        e = ref.eligible(k, float(p))
        if seen_true:
            assert e, "eligibility retracted -- spec §4.4 forbids this"
        seen_true = seen_true or e
    assert seen_true
    assert ref.t_visible_of(k) == 2
    assert not ref.eligible(k, 0.19) and ref.eligible(k, 0.2)


def test_truncating_the_source_stream_does_not_change_earlier_eligibility():
    """Spec §4.1 consequence 1 / S1's fourth REQUIRE: no future leakage."""
    # CELL_C deliberately SPANS the cut (seen at 1,2 and again at 7,8): a cell
    # observed only on one side of the truncation cannot catch a future change
    # where later frames retroactively alter an earlier t_visible.
    cell_c = np.array([0.5, 5.5, 0.5])
    frames = [
        make_frame(i, [i * 2.0, 0.0, -3.0], [CELL_A, cell_c] if i in (1, 2) else [CELL_A])
        for i in range(1, 6)
    ]
    frames += [
        make_frame(i, [i * 2.0, 0.0, -3.0], [CELL_B, cell_c] if i in (7, 8) else [CELL_B])
        for i in range(6, 11)
    ]
    full = build(frames, declared=10)
    trunc = build(frames[:5], declared=10)

    for k in trunc.cell_keys.tolist():
        assert trunc.t_visible_of(int(k)) == full.t_visible_of(int(k))
    assert full.t_visible_of(key_of(cell_c)) == 2
    assert trunc.t_visible_of(key_of(cell_c)) == 2
    assert full.n_src_of(key_of(cell_c)) == 4      # later views add support ...
    assert trunc.n_src_of(key_of(cell_c)) == 2     # ... without moving t_visible
    # CELL_B only becomes visible after the truncation point and must be
    # absent from the truncated reference entirely -- never eligible early.
    assert key_of(CELL_B) not in trunc.cell_keys.tolist()
    assert full.t_visible_of(key_of(CELL_B)) == 7
    for p in np.linspace(0.0, 0.5, 11):
        assert not full.eligible(key_of(CELL_B), float(p))


# ---------------------------------------------------------------------------
# Support rule: greedy chronological, and the distinct-view disjunction
# ---------------------------------------------------------------------------
def test_greedy_chronological_differs_from_maximum_clique():
    """Spec §4.3 makes the GREEDY count part of the metric's meaning.

    f1 is not distinct from either f2 or f3, but f2 and f3 are distinct from
    each other. A maximum-pairwise-distinct subset is {f2, f3} (size 2);
    the mandated greedy chronological accumulator accepts f1 first and then
    rejects both, giving 1. A test that passed under both rules would prove
    nothing.
    """
    params = base_params(b_min_m=1.0, alpha_min_deg=90.0, K_vis=2)
    frames = [
        make_frame(1, [0.0, 0.0, -3.0], [CELL_A]),
        make_frame(2, [-0.6, 0.0, -3.0], [CELL_A]),
        make_frame(3, [0.6, 0.0, -3.0], [CELL_A]),
    ]
    ref = build(frames, params=params)
    k = key_of(CELL_A)
    assert np.linalg.norm(np.array([-0.6, 0, -3.0]) - np.array([0.6, 0, -3.0])) >= 1.0
    assert ref.n_src_of(k) == 1                      # greedy, not the clique's 2
    assert ref.t_visible_of(k) is None
    assert ref.accounting.cell_frame_pairs_not_distinct == 2
    assert ref.params.support_rule == "greedy_chronological"


def test_distinct_by_baseline_alone():
    params = base_params(b_min_m=0.5, alpha_min_deg=90.0)
    frames = [
        make_frame(1, [0.0, 0.0, -100.0], [CELL_A]),
        make_frame(2, [1.0, 0.0, -100.0], [CELL_A]),   # 1.0 m apart, ~0.6 deg apart
    ]
    ref = build(frames, params=params)
    assert ref.n_src_of(key_of(CELL_A)) == 2


def test_distinct_by_angle_alone():
    params = base_params(b_min_m=100.0, alpha_min_deg=5.0)
    frames = [
        make_frame(1, [0.5, 0.5, -3.0], [CELL_A]),
        make_frame(2, [3.5, 0.5, 0.5], [CELL_A]),      # ~90 deg apart, 4.2 m < b_min
    ]
    ref = build(frames, params=params)
    assert ref.n_src_of(key_of(CELL_A)) == 2


def test_support_rule_cannot_be_overridden():
    with pytest.raises(ValueError):
        CausalReferenceParams(support_rule="max_clique")


# ---------------------------------------------------------------------------
# Guards: m_pix, K_vis, depth range, incidence
# ---------------------------------------------------------------------------
def test_m_pix_guard_rejects_a_thin_observation():
    params = base_params(m_pix=8)
    frames = [
        make_frame(1, [0.0, 0.0, -3.0], [CELL_A], n_pix=7),
        make_frame(2, [2.0, 0.0, -3.0], [CELL_A], n_pix=7),
    ]
    ref = build(frames, params=params)
    assert ref.n_cells == 0
    assert ref.accounting.cell_frame_pairs_below_m_pix == 2
    assert ref.status == "not_computed"
    assert ref.reason == "no_qualifying_source_observations"


def test_K_vis_guard_leaves_a_single_view_cell_never_visible_but_counted():
    frames = [
        make_frame(1, [0.0, 0.0, -3.0], [CELL_A, CELL_B]),
        make_frame(2, [2.0, 0.0, -3.0], [CELL_A]),
    ]
    ref = build(frames)
    assert ref.t_visible_of(key_of(CELL_A)) == 2
    assert ref.t_visible_of(key_of(CELL_B)) is None      # counted, never dropped
    assert ref.n_cells == 2
    assert ref.n_cells_never_visible == 1
    assert ref.accounting.cells_never_visible == 1
    assert not ref.eligible(key_of(CELL_B), 1.0)


def test_depth_range_rejections_are_counted():
    params = base_params(d_min=0.5, d_max=3.0)
    frames = [make_frame(1, [0.0, 0.0, -3.0], [CELL_A], depth=9.0)]
    ref = build(frames, params=params)
    assert ref.accounting.points_rejected_depth_range == 16
    assert ref.accounting.points_accepted == 0
    assert ref.status == "not_computed"


def test_non_finite_geometry_has_its_own_counter():
    """Out-of-depth-range and non-finite are two reasons, not one."""
    pts = np.concatenate([
        np.repeat(CELL_A.reshape(1, 3), 16, axis=0),
        np.full((3, 3), np.nan),
    ])
    f = SourceFrame(
        index=1, frame_id="f1", camera_center=[0.0, 0.0, -3.0],
        points_world=pts, point_depth=np.full(pts.shape[0], 2.0),
        point_incidence_deg=np.zeros(pts.shape[0]),
    )
    ref = build([f])
    assert ref.accounting.points_rejected_non_finite == 3
    assert ref.accounting.points_rejected_depth_range == 0
    assert ref.accounting.points_accepted == 16


def test_incidence_filter_rejections_are_counted():
    params = base_params(theta_max_deg=75.0)
    frames = [
        make_frame(1, [0.0, 0.0, -3.0], [CELL_A], incidence=80.0),
        make_frame(2, [2.0, 0.0, -3.0], [CELL_A], incidence=10.0),
    ]
    ref = build(frames, params=params)
    assert ref.accounting.points_rejected_incidence == 16
    assert ref.n_src_of(key_of(CELL_A)) == 1


def test_incidence_filter_cannot_be_silently_skipped():
    """Spec §4.2: absent normals must be DECLARED, not silently ignored."""
    b = CausalReferenceBuilder(scene_id="s", n_source_frames_declared=10, params=base_params())
    f = make_frame(1, [0.0, 0.0, -3.0], [CELL_A])
    f.point_incidence_deg = None
    with pytest.raises(CausalReferenceError):
        b.observe_frame(f)

    b2 = CausalReferenceBuilder(
        scene_id="s", n_source_frames_declared=10,
        params=base_params(incidence_filter="unavailable"),
    )
    b2.observe_frame(f)
    ref = b2.finalize()
    assert "incidence_filter_unavailable" in ref.notes


def test_out_of_representable_range_points_are_counted_not_swallowed():
    """`cell_key` refuses to alias an absurdly distant point; the builder must
    account for the drop under a named counter, never a bare except/continue."""
    pts = np.concatenate([
        np.repeat(CELL_A.reshape(1, 3), 16, axis=0),
        np.full((4, 3), 1e9),
    ])
    f = SourceFrame(
        index=1, frame_id="f1", camera_center=[0.0, 0.0, -3.0],
        points_world=pts, point_depth=np.full(pts.shape[0], 2.0),
        point_incidence_deg=np.zeros(pts.shape[0]),
    )
    ref = build([f])
    assert ref.accounting.points_skipped_cell_range == 4
    assert ref.accounting.points_accepted == 16
    assert "points_skipped_cell_range" in ref.notes


def test_frames_must_arrive_in_increasing_source_order():
    b = CausalReferenceBuilder(scene_id="s", n_source_frames_declared=10, params=base_params())
    b.observe_frame(make_frame(3, [0.0, 0.0, -3.0], [CELL_A]))
    with pytest.raises(CausalReferenceError):
        b.observe_frame(make_frame(2, [2.0, 0.0, -3.0], [CELL_A]))


# ---------------------------------------------------------------------------
# Architecture principle: no method-private state can reach the reference
# ---------------------------------------------------------------------------
_ADMISSION_TOKENS = (
    "admit", "admission", "keyframe", "replay", "mcmc", "gaussian", "lineage",
    "render", "opacity", "psnr", "confidence",
)
# NOTE: "alpha" is deliberately NOT a token -- rendered alpha is already
# covered by "render", and `alpha_min_deg` is a legitimate source-side
# view-angle parameter (spec §4.3).


def test_no_method_private_input_surface_exists():
    """#428's architecture principle, as an executable regression net."""
    names = set(inspect.signature(CausalReferenceBuilder.__init__).parameters)
    names |= set(inspect.signature(CausalReferenceBuilder.observe_frame).parameters)
    names |= set(inspect.signature(build_causal_reference).parameters)
    names |= set(SourceFrame.__dataclass_fields__)
    names |= set(CausalReferenceParams.__dataclass_fields__)
    for n in names:
        low = n.lower()
        assert not any(tok in low for tok in _ADMISSION_TOKENS), (
            f"parameter {n!r} looks like method-private state; the causal "
            f"reference must depend only on the frozen source stream"
        )


def test_build_is_deterministic():
    """Same inputs -> same hash and same mask.

    Deliberately NOT labelled an S7 (admission-independence) test: admission
    is not an input to this module at all, so there is no admission rate to
    vary here. The real structural evidence for S7 is
    ``test_no_method_private_input_surface_exists`` above -- this one only
    pins determinism.
    """
    frames = [make_frame(i, [i * 0.4, 0.0, -3.0], [CELL_A]) for i in range(1, 11)]
    ref_a = build(frames, declared=10)
    ref_b = build(frames, declared=10)
    assert ref_a.eligibility_hash == ref_b.eligibility_hash
    assert np.array_equal(ref_a.cell_keys, ref_b.cell_keys)
    assert np.array_equal(ref_a.t_visible, ref_b.t_visible)


# ---------------------------------------------------------------------------
# eligibility_hash
# ---------------------------------------------------------------------------
def _hash(**over):
    kw = dict(
        scene_id="scene", source_frame_ids=["a", "b"], reference_pose_source_id="gt",
        holdout_camera_ids=["h1"], params=base_params(),
        region_identity_name="level_a_microcell", region_identity_params={"cell_size_m": CELL},
    )
    kw.update(over)
    return compute_eligibility_hash(**kw)


def test_eligibility_hash_is_stable():
    assert _hash() == _hash()


@pytest.mark.parametrize("field,value", [
    ("cell_size_m", 0.25), ("m_pix", 9), ("d_min", 0.4), ("d_max", 6.0),
    ("theta_max_deg", 60.0), ("K_vis", 3), ("b_min_m", 0.2), ("alpha_min_deg", 10.0),
])
def test_eligibility_hash_is_sensitive_to_every_support_parameter(field, value):
    assert _hash(params=base_params(**{field: value})) != _hash()


def test_eligibility_hash_is_sensitive_to_incidence_filter_status():
    assert _hash(params=base_params(incidence_filter="unavailable")) != _hash()


@pytest.mark.parametrize("over", [
    {"scene_id": "other"}, {"source_frame_ids": ["b", "a"]},
    {"reference_pose_source_id": "orbslam"}, {"holdout_camera_ids": ["h2"]},
    {"region_identity_name": "issue358_adaptive"},
])
def test_eligibility_hash_is_sensitive_to_identity_fields(over):
    assert _hash(**over) != _hash()


def test_patch_level_spec_bump_does_not_invalidate_the_hash():
    """§4.5: major.minor only participates, so a prose fix is not a rebuild."""
    assert spec_version_minor("1.1.0-draft") == "1.1"
    assert _hash(version="1.1.0-draft") == _hash(version="1.1.7")
    assert _hash(version="1.2.0") != _hash(version="1.1.0")


# ---------------------------------------------------------------------------
# Holdout membership -> unresolvable
# ---------------------------------------------------------------------------
def test_unresolvable_cells_are_flagged_and_counted():
    frames = [
        make_frame(1, [0.0, 0.0, -3.0], [CELL_A, CELL_B]),
        make_frame(2, [2.0, 0.0, -3.0], [CELL_A, CELL_B]),
    ]
    membership = build_holdout_membership(
        [("h0", np.repeat(CELL_A.reshape(1, 3), 5, axis=0))], cell_size_m=CELL
    )
    ref = build(frames, holdout_membership=membership)
    rows = {int(k): i for i, k in enumerate(ref.cell_keys.tolist())}
    assert bool(ref.holdout_resolvable[rows[key_of(CELL_A)]]) is True
    assert bool(ref.holdout_resolvable[rows[key_of(CELL_B)]]) is False
    assert ref.accounting.cells_unresolvable == 1
    assert ref.holdout_camera_ids == ["h0"]


def test_absent_holdout_membership_is_declared_not_assumed_clean():
    frames = [make_frame(i, [i * 2.0, 0.0, -3.0], [CELL_A]) for i in (1, 2)]
    ref = build(frames)
    assert ref.holdout_resolvable is None
    assert "holdout_membership_absent_unresolvable_undetermined" in ref.notes
    assert ref.accounting.cells_unresolvable == 0


# ---------------------------------------------------------------------------
# Degenerate inputs must refuse, not report a healthy zero (spec §5)
# ---------------------------------------------------------------------------
def test_zero_frames_refuses():
    ref = build([])
    assert ref.status == "not_computed"
    assert ref.reason == "no_source_frames_observed"
    with pytest.raises(CausalReferenceError):
        ref.require_ok()


def test_all_invalid_depth_refuses():
    f = SourceFrame(
        index=1, frame_id="f1", camera_center=[0.0, 0.0, -3.0],
        points_world=np.repeat(CELL_A.reshape(1, 3), 16, axis=0),
        point_depth=np.full(16, 99.0), point_incidence_deg=np.zeros(16),
    )
    ref = build([f])
    assert ref.status == "not_computed"
    assert ref.reason == "no_qualifying_source_observations"


def test_no_cell_reaching_K_vis_refuses():
    ref = build([make_frame(1, [0.0, 0.0, -3.0], [CELL_A])])
    assert ref.status == "not_computed"
    assert ref.reason == "no_cell_reached_K_vis"
    assert ref.n_cells == 1                     # the cell is still COUNTED


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_artifact_round_trip(tmp_path):
    frames = [
        make_frame(1, [0.0, 0.0, -3.0], [CELL_A, CELL_B]),
        make_frame(2, [2.0, 0.0, -3.0], [CELL_A]),
        make_frame(3, [4.0, 0.0, -3.0], [CELL_B]),
    ]
    membership = build_holdout_membership(
        [("h0", np.repeat(CELL_A.reshape(1, 3), 5, axis=0))], cell_size_m=CELL
    )
    ref = build(frames, holdout_membership=membership)
    path = write_causal_reference(ref, str(tmp_path))
    assert path.endswith(f"eligibility.{ref.eligibility_hash}.json")

    back = read_causal_reference(path)
    assert isinstance(back, CausalReference)
    assert back.eligibility_hash == ref.eligibility_hash
    assert back.spec_version == ref.spec_version
    assert back.params == ref.params
    assert back.status == ref.status and back.notes == ref.notes
    assert back.n_source_frames_declared == ref.n_source_frames_declared
    assert np.array_equal(back.cell_keys, ref.cell_keys)
    assert np.array_equal(back.t_visible, ref.t_visible)
    assert np.array_equal(back.n_src, ref.n_src)
    assert np.array_equal(back.holdout_resolvable, ref.holdout_resolvable)
    assert back.accounting.as_dict() == ref.accounting.as_dict()
    for p in (0.0, 0.02, 0.5, 1.0):
        assert np.array_equal(back.eligible_mask(p), ref.eligible_mask(p))


def test_artifact_records_the_spec_version_and_refuses_issue358_claim(tmp_path):
    ref = build([make_frame(i, [i * 2.0, 0.0, -3.0], [CELL_A]) for i in (1, 2)])
    d = ref.to_dict()
    assert d["spec_id"] == "CAUSAL_ONLINE_MAPPING_METRICS"
    assert d["spec_version"] == "1.1.0-draft"
    assert d["spec_doc_sha256"] is None or len(d["spec_doc_sha256"]) == 64
    assert d["region_identity"]["name"] == "level_a_microcell"
    assert d["region_identity"]["issue358_adaptive_regions_implemented"] is False
    assert d["phase"] == "issue428_phase1"


# ---------------------------------------------------------------------------
# Region identity is pluggable
# ---------------------------------------------------------------------------
class _ShiftedIdentity(MicrocellRegionIdentity):
    """A second identity scheme, only to prove the seam is real."""

    name = "fixture_shifted_microcell"

    def keys_for_points(self, xyz):
        return super().keys_for_points(np.asarray(xyz, dtype=float).reshape(-1, 3) + 100.0)


def test_a_different_region_identity_yields_a_different_hash_and_keys():
    frames = [make_frame(i, [i * 2.0, 0.0, -3.0], [CELL_A]) for i in (1, 2)]
    default = build(frames)
    shifted = build(frames, region_identity=_ShiftedIdentity(CELL))
    assert shifted.region_identity_name == "fixture_shifted_microcell"
    assert shifted.eligibility_hash != default.eligibility_hash
    assert set(shifted.cell_keys.tolist()) != set(default.cell_keys.tolist())


def test_microcell_identity_matches_local_quality_keys():
    ident = MicrocellRegionIdentity(0.5)
    pts = np.array([[0.1, 0.2, 0.3], [-2.4, 7.7, 0.0]])
    assert np.array_equal(ident.keys_for_points(pts), cell_key(cell_ijk(pts, 0.5)))


# ---------------------------------------------------------------------------
# Backprojection helper
# ---------------------------------------------------------------------------
def test_backprojection_is_identity_under_an_identity_pose():
    depth = np.array([[2.0, 0.0], [np.nan, 3.0]])
    intr = PinholeIntrinsics(fx=100.0, fy=100.0, cx=0.5, cy=0.5)
    pts, z, inc = backproject_depth_image(depth, intr, np.eye(4))
    assert pts.shape == (2, 3) and inc is None
    assert z.tolist() == [2.0, 3.0]
    assert pts[0] == pytest.approx([-0.01, -0.01, 2.0])

    frame = source_frame_from_depth_image(1, "f1", depth, intr, np.eye(4))
    assert frame.camera_center.tolist() == [0.0, 0.0, 0.0]
    assert frame.points_world.shape[0] == 2


def test_backprojection_incidence_is_zero_for_a_head_on_surface():
    depth = np.full((2, 2), 2.0)
    intr = PinholeIntrinsics(fx=1000.0, fy=1000.0, cx=0.5, cy=0.5)
    normals = np.tile(np.array([0.0, 0.0, -1.0]), (2, 2, 1))
    _, _, inc = backproject_depth_image(depth, intr, np.eye(4), normals_world=normals)
    assert inc is not None
    assert np.allclose(inc, 0.0, atol=0.2)
    assert math.isfinite(float(inc.max()))
