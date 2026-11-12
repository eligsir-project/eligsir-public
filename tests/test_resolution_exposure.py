"""Tests for the resolution-exposure eligibility mechanism (issue #1262).

Covers, per the design doc's own acceptance shape:

* the exposure predicate (synthetic footprints vs. a synthetic ray lattice),
  including the two traps that would ship it silently inert;
* the veto's composition with the other four per-Gaussian vetoes;
* the exposure-currency young gate (arm A2);
* the §6.2.1 opacity-survival telemetry and its three-row outcome table.

All CPU-only — no CUDA, no model class, no renderer.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from eligsir.utils.exposure_opacity_survival import (  # noqa: E402
    OUTCOME_HARMLESS_UNDER_DETAIL,
    OUTCOME_INDETERMINATE,
    OUTCOME_OPACITY_DEATH,
    OUTCOME_TOPOLOGY_DESTRUCTION,
    classify_outcome,
    exposure_groups,
    exposure_opacity_survival_block,
    group_stats,
    per_checkpoint_row,
    snapshot,
    zombie_recovery_frac,
)
from eligsir.utils.gaussian_lineage import (  # noqa: E402
    BIRTH_METHOD_MCMC_GROW,
    BIRTH_METHOD_SOURCE_DEPTH,
)
from eligsir.utils.resolution_exposure import (  # noqa: E402
    PREDICATE_LATTICE,
    PREDICATE_SIGMA_RATIO,
    RASTERIZER_ALPHA_THRESHOLD,
    TIER_HIGH_WATER_UNSET,
    ExposureConfig,
    ResolutionExposureVeto,
    accumulate_exposure,
    compute_exposure,
    ensure_exposure_buffers,
    exclude_unexposed_mask,
    exposure_config_from_args,
    exposure_young_mask,
    filter_unexposed_indices,
    native_camera_geometry,
    publish_exposure_young_min,
    reset_exposure_rows,
    unexposed_statistic_pool_mask,
)


# ---------------------------------------------------------------------------
# duck-typed stand-ins
# ---------------------------------------------------------------------------


class _Tier:
    def __init__(self, factor):
        self.downsample_factor = int(factor)


class _NativeState:
    def __init__(self, fx, width, height):
        self.fx = self.fy = float(fx)
        self.width = int(width)
        self.height = int(height)
        self.cx = width / 2.0
        self.cy = height / 2.0


class FakeCamera:
    """Camera at the world origin looking down +Z, with a tier binding."""

    def __init__(self, *, fx=500.0, width=640, height=480, factor=1, uid=0):
        self.uid = uid
        native = _NativeState(fx, width, height)
        self._canonical_resolution_binding_native_state = native
        self._canonical_resolution_tier = _Tier(factor)
        # The TIER-BOUND intrinsics, exactly as materialize_resolution_tier
        # writes them: fx already divided by s. Reading THESE would cancel the
        # tier out of the predicate (the §3.3 trap).
        self.fx = self.fy = fx / factor
        self.image_width = width // factor
        self.image_height = height // factor
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0
        self.world_view_transform = torch.eye(4, dtype=torch.float32)


class FakeGaussians:
    def __init__(self, means, scales, *, opacities=None, birth_iter=None):
        self._means = torch.as_tensor(means, dtype=torch.float32)
        n = self._means.shape[0]
        self._scales = torch.as_tensor(scales, dtype=torch.float32)
        if self._scales.ndim == 1:
            self._scales = self._scales.unsqueeze(-1).repeat(1, 3)
        self._quats = torch.zeros(n, 4)
        self._quats[:, 0] = 1.0
        self._opacities = (
            torch.full((n,), 0.5) if opacities is None else torch.as_tensor(opacities)
        )
        self.birth_iter = (
            torch.zeros(n, dtype=torch.int32)
            if birth_iter is None
            else torch.as_tensor(birth_iter, dtype=torch.int32)
        )

    @property
    def get_xyz(self):
        return self._means

    @property
    def get_scaling(self):
        return self._scales

    @property
    def get_rotation(self):
        return self._quats

    @property
    def get_opacity(self):
        return self._opacities.unsqueeze(-1)


class Args:
    def __init__(self, **kw):
        self.streaming_resolution_exposure_veto = False
        self.streaming_resolution_exposure_telemetry = False
        self.streaming_resolution_exposure_predicate = PREDICATE_LATTICE
        self.streaming_resolution_exposure_tau = 0.5
        self.streaming_resolution_exposure_sigma_c = 3.0
        self.streaming_resolution_exposure_min_count = 3
        self.streaming_resolution_exposure_max_protect_iters = 4000
        for k, v in kw.items():
            setattr(self, k, v)


def _gauss_at(depth, world_radius, *, x=0.0, y=0.0, n=1):
    means = [[x, y, depth]] * n
    return FakeGaussians(means, [[world_radius] * 3] * n)


# ---------------------------------------------------------------------------
# the predicate
# ---------------------------------------------------------------------------


def test_native_geometry_uses_native_focal_not_tier_bound():
    """The §3.3 trap: camera.fx has ALREADY been divided by s."""
    cam = FakeCamera(fx=500.0, factor=4)
    geom = native_camera_geometry(cam)
    assert geom.fx == pytest.approx(500.0)
    assert cam.fx == pytest.approx(125.0)  # the value that must NOT be used
    assert geom.downsample_factor == 4


def test_native_geometry_defaults_to_factor_one_without_binding():
    class Bare:
        fx = fy = 400.0
        image_width = 64
        image_height = 48
        cx = 32.0
        cy = 24.0

    geom = native_camera_geometry(Bare())
    assert geom.downsample_factor == 1
    assert geom.fx == pytest.approx(400.0)


@pytest.mark.parametrize(
    "predicate,world_radius,offset_px",
    [
        # lattice: a 0.2px-sigma Gaussian sitting 2px off the origin is ON a
        # native pixel and BETWEEN quarter-tier rays.
        (PREDICATE_LATTICE, 0.0004, 2.0),
        # sigma_ratio is phase-blind, so size alone must carry it: 1.0px sigma
        # clears tau*1 at native and misses tau*4 at quarter.
        (PREDICATE_SIGMA_RATIO, 0.002, 0.0),
    ],
)
def test_predicate_is_tier_sensitive(predicate, world_radius, offset_px):
    """The inert-mechanism regression guard.

    A Gaussian resolvable at native must become unexposed at quarter tier. If
    this ever stops holding, the predicate is reading a tier-invariant quantity
    (``radii``, or the tier-bound ``fx``) and the whole mechanism is inert.
    """
    # sigma_proj = f_native * s2 / z, in NATIVE pixels.
    g = _gauss_at(1.0, world_radius, x=offset_px / 500.0)
    cfg = ExposureConfig(enabled=True, predicate=predicate)
    native = compute_exposure(g, FakeCamera(factor=1), cfg)
    quarter = compute_exposure(g, FakeCamera(factor=4), cfg)
    assert bool(native.exposed[0]) is True
    assert bool(quarter.exposed[0]) is False


def test_exposure_is_monotone_in_tier():
    """Zero-phase nesting L(4) subset L(2) subset L(1) => monotone exposure."""
    torch.manual_seed(0)
    n = 200
    means = torch.stack(
        [
            torch.empty(n).uniform_(-0.4, 0.4),
            torch.empty(n).uniform_(-0.3, 0.3),
            torch.empty(n).uniform_(0.8, 4.0),
        ],
        dim=-1,
    )
    scales = torch.empty(n, 3).uniform_(0.001, 0.05)
    g = FakeGaussians(means, scales)
    cfg = ExposureConfig(enabled=True, predicate=PREDICATE_LATTICE)
    e1 = compute_exposure(g, FakeCamera(factor=1), cfg).exposed
    e2 = compute_exposure(g, FakeCamera(factor=2), cfg).exposed
    e4 = compute_exposure(g, FakeCamera(factor=4), cfg).exposed
    assert bool((e1 | ~e2).all()), "exposed at half must imply exposed at native"
    assert bool((e2 | ~e4).all()), "exposed at quarter must imply exposed at half"


def test_lattice_predicate_is_near_noop_at_native():
    torch.manual_seed(1)
    n = 100
    means = torch.stack(
        [
            torch.empty(n).uniform_(-0.3, 0.3),
            torch.empty(n).uniform_(-0.2, 0.2),
            torch.empty(n).uniform_(1.0, 3.0),
        ],
        dim=-1,
    )
    g = FakeGaussians(means, torch.full((n, 3), 0.01))
    cfg = ExposureConfig(enabled=True, predicate=PREDICATE_LATTICE)
    result = compute_exposure(g, FakeCamera(factor=1), cfg)
    on = result.on_screen
    assert bool(on.any())
    assert float(result.exposed[on].float().mean()) > 0.95


def test_lattice_predicate_is_phase_aware():
    """A tiny Gaussian ON a retained ray is exposed; the same one BETWEEN rays is not.

    Identical ``sigma``, opposite truth — the exact case the sigma-ratio proxy
    provably cannot represent (§3.2).
    """
    fx, width, s = 500.0, 640, 4
    cx = width / 2.0  # 320.0, itself a lattice point (320 % 4 == 0)
    z = 1.0
    cfg = ExposureConfig(enabled=True, predicate=PREDICATE_LATTICE, sigma_c=3.0)

    # sigma_proj = 500 * 0.0004 / 1.0 = 0.2 px  => 3-sigma radius 0.6 px << 4.
    tiny = 0.0004
    # Centred exactly on a retained lattice point (u = cx = 320).
    on_ray = FakeGaussians([[0.0, 0.0, z]], [[tiny] * 3])
    # Shifted by half a lattice cell: u = cx + 2 px, maximally between rays.
    dx = 2.0 * z / fx
    off_ray = FakeGaussians([[dx, 0.0, z]], [[tiny] * 3])

    cam = FakeCamera(fx=fx, width=width, factor=s)
    assert bool(compute_exposure(on_ray, cam, cfg).exposed[0]) is True
    assert bool(compute_exposure(off_ray, cam, cfg).exposed[0]) is False

    # The sigma-ratio proxy gives the SAME answer for both — that is precisely
    # its documented limitation, and why the ambiguity band is reported.
    proxy = ExposureConfig(enabled=True, predicate=PREDICATE_SIGMA_RATIO, tau=0.5)
    assert bool(compute_exposure(on_ray, cam, proxy).exposed[0]) is bool(
        compute_exposure(off_ray, cam, proxy).exposed[0]
    )


def test_sigma_ratio_proxy_reports_its_phase_ambiguous_band():
    fx, s, tau = 500.0, 4, 0.5
    z = 1.0
    # tau*s = 2.0 px; the ambiguous band is [1.0, 4.0] px.
    in_band = 2.0 * z / fx  # sigma_proj = 2.0 px
    out_band = 20.0 * z / fx  # sigma_proj = 20.0 px
    g = FakeGaussians([[0.0, 0.0, z], [0.0, 0.0, z]], [[in_band] * 3, [out_band] * 3])
    cfg = ExposureConfig(enabled=True, predicate=PREDICATE_SIGMA_RATIO, tau=tau)
    result = compute_exposure(g, FakeCamera(fx=fx, factor=s), cfg)
    assert result.n_ambiguous == 1
    assert result.ambiguous_frac == pytest.approx(0.5)


def test_offscreen_and_behind_camera_are_never_exposed():
    g = FakeGaussians(
        [[0.0, 0.0, -1.0], [100.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        torch.full((3, 3), 0.05),
    )
    cfg = ExposureConfig(enabled=True)
    result = compute_exposure(g, FakeCamera(factor=1), cfg)
    assert bool(result.exposed[0]) is False  # behind
    assert bool(result.exposed[1]) is False  # off-screen
    assert bool(result.exposed[2]) is True


def test_compute_exposure_returns_none_without_resolvable_geometry():
    class NoIntrinsics:
        pass

    g = _gauss_at(1.0, 0.01)
    assert compute_exposure(g, NoIntrinsics(), ExposureConfig(enabled=True)) is None


def test_empty_population_is_handled():
    g = FakeGaussians(torch.zeros(0, 3), torch.zeros(0, 3))
    result = compute_exposure(g, FakeCamera(), ExposureConfig(enabled=True))
    assert result.exposed.numel() == 0


def test_unsupported_predicate_is_refused():
    with pytest.raises(ValueError):
        ExposureConfig(predicate="radii")  # the trap, spelled out


# ---------------------------------------------------------------------------
# buffers
# ---------------------------------------------------------------------------


def test_buffers_allocate_resize_and_preserve_history():
    g = _gauss_at(1.0, 0.01, n=3)
    ensure_exposure_buffers(g)
    g.exposure_count += torch.tensor([1, 2, 3], dtype=torch.int32)
    # simulate MCMC growth: population grows, buffers must follow WITHOUT
    # discarding history (the GAP-4 defect).
    g._means = torch.cat([g._means, g._means[:2]], dim=0)
    ensure_exposure_buffers(g)
    assert g.exposure_count.shape[0] == 5
    assert g.exposure_count[:3].tolist() == [1, 2, 3]
    assert g.exposure_count[3:].tolist() == [0, 0]
    assert g.supervised_tier_high_water[3:].tolist() == [
        TIER_HIGH_WATER_UNSET,
        TIER_HIGH_WATER_UNSET,
    ]


def test_reset_exposure_rows_does_not_inherit_donor_history():
    g = _gauss_at(1.0, 0.01, n=4)
    ensure_exposure_buffers(g)
    g.exposure_count += torch.tensor([5, 6, 7, 8], dtype=torch.int32)
    g.supervised_tier_high_water[:] = 7
    reset_exposure_rows(g, torch.tensor([1, 2]))
    assert g.exposure_count.tolist() == [5, 0, 0, 8]
    assert g.supervised_tier_high_water.tolist() == [7, -1, -1, 7]


def test_accumulate_dedups_by_view_uid():
    g = _gauss_at(1.0, 0.01)
    cfg = ExposureConfig(enabled=True)
    cam = FakeCamera(factor=1, uid=11)
    accumulate_exposure(g, cam, cfg)
    accumulate_exposure(g, cam, cfg)  # same view again — must not double-count
    assert int(g.exposure_count[0]) == 1
    accumulate_exposure(g, FakeCamera(factor=1, uid=12), cfg)
    assert int(g.exposure_count[0]) == 2
    # high-water is monotone and finer tiers rank higher
    assert int(g.supervised_tier_high_water[0]) == 8


def test_accumulate_is_a_noop_when_disabled():
    g = _gauss_at(1.0, 0.01)
    assert accumulate_exposure(g, FakeCamera(), ExposureConfig(enabled=False)) is None
    assert getattr(g, "exposure_count", None) is None


# ---------------------------------------------------------------------------
# the veto
# ---------------------------------------------------------------------------


def _veto_gaussians(counts, *, birth_iter=None):
    g = _gauss_at(1.0, 0.01, n=len(counts))
    ensure_exposure_buffers(g)
    g.exposure_count = torch.tensor(counts, dtype=torch.int32)
    if birth_iter is not None:
        g.birth_iter = torch.tensor(birth_iter, dtype=torch.int32)
    return g


def test_veto_is_a_total_noop_when_flag_off():
    g = _veto_gaussians([0, 0, 0])
    args = Args()
    mask = torch.tensor([True, True, False])
    out = exclude_unexposed_mask(g, args, mask, iteration=10, role="relocation_target")
    assert out is mask
    idx = torch.tensor([0, 1, 2])
    assert filter_unexposed_indices(g, args, idx, iteration=10, role="donor") is idx
    assert unexposed_statistic_pool_mask(g, args, iteration=10, base=None) is None


def test_veto_removes_unexposed_rows_from_the_dead_mask():
    g = _veto_gaussians([0, 5, 1])
    args = Args(streaming_resolution_exposure_veto=True)
    dead = torch.tensor([True, True, True])
    out = exclude_unexposed_mask(g, args, dead, iteration=10, role="relocation_target")
    assert out.tolist() == [False, True, False]


def test_veto_removes_unexposed_rows_from_the_statistic_pool():
    """§4 item 2 — the half that is easy to miss."""
    g = _veto_gaussians([0, 5, 1])
    args = Args(streaming_resolution_exposure_veto=True)
    pool = unexposed_statistic_pool_mask(g, args, iteration=10, base=None)
    assert pool.tolist() == [False, True, False]
    base = torch.tensor([True, False, True])
    combined = unexposed_statistic_pool_mask(g, args, iteration=10, base=base)
    assert combined.tolist() == [False, False, False]


def test_veto_filters_donor_and_parent_index_arrays():
    g = _veto_gaussians([0, 5, 1, 9])
    args = Args(streaming_resolution_exposure_veto=True)
    idx = torch.tensor([0, 1, 2, 3])
    kept = filter_unexposed_indices(g, args, idx, iteration=10, role="growth_parent")
    assert kept.tolist() == [1, 3]


def test_veto_bounded_immunity_lapses(capsys):
    """W3-45's lesson: immunity must be bounded by construction."""
    g = _veto_gaussians([0, 0], birth_iter=[0, 0])
    args = Args(
        streaming_resolution_exposure_veto=True,
        streaming_resolution_exposure_max_protect_iters=100,
    )
    dead = torch.tensor([True, True])
    early = exclude_unexposed_mask(g, args, dead, iteration=50, role="relocation_target")
    assert early.tolist() == [False, False]
    late = exclude_unexposed_mask(g, args, dead, iteration=500, role="relocation_target")
    assert late.tolist() == [True, True]
    assert "bounded-immunity ceiling" in capsys.readouterr().out


def test_veto_zero_position_noise_is_implemented_but_unwired():
    g = _veto_gaussians([0, 5])
    veto = ResolutionExposureVeto(g, ExposureConfig(enabled=True))
    noise = torch.ones(2, 3)
    out = veto.zero_position_noise(noise, iteration=10)
    assert out[0].tolist() == [0.0, 0.0, 0.0]
    assert out[1].tolist() == [1.0, 1.0, 1.0]
    # Category (c) is an OPEN follow-up (§2.2.2): the method exists so the veto
    # has SeedLineageFreeze's surface, but no production call site invokes it and
    # the policy (zeroing vs. exposure-proportional attenuation) is unspecified.
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "tests" not in path.parts
        and path.name
        not in ("resolution_exposure.py", "seed_lineage_freeze.py")
        and "zero_position_noise" in path.read_text(errors="ignore")
    ]
    assert offenders == []


def test_veto_composes_with_the_other_vetoes_without_double_counting():
    """Order-independent intersection with the depth lock / lifecycle / bubble.

    Every one of the five vetoes only ever REMOVES rows from a kill set, so any
    two of them commute. This is the property that lets the new veto be dropped
    into the existing pipeline at any point before the KNN augment.
    """
    g = _veto_gaussians([0, 5, 0, 5])
    args = Args(streaming_resolution_exposure_veto=True)
    dead = torch.tensor([True, True, True, True])
    lifecycle_shield = torch.tensor([True, True, False, False])  # stand-in for lc < 2
    bubble = torch.tensor([True, False, True, True])

    a = exclude_unexposed_mask(
        g, args, dead & lifecycle_shield & bubble, iteration=10, role="r"
    )
    b = (
        exclude_unexposed_mask(g, args, dead, iteration=10, role="r")
        & lifecycle_shield
        & bubble
    )
    assert a.tolist() == b.tolist() == [False, False, False, False]

    dead2 = torch.tensor([True, True, True, True])
    assert (
        exclude_unexposed_mask(g, args, dead2, iteration=10, role="r") & bubble
    ).tolist() == [False, False, False, True]


def test_veto_exclusion_counters_are_per_role():
    g = _veto_gaussians([0, 0, 5])
    args = Args(streaming_resolution_exposure_veto=True)
    dead = torch.tensor([True, True, True])
    exclude_unexposed_mask(g, args, dead, iteration=1, role="relocation_target")
    filter_unexposed_indices(
        g, args, torch.tensor([0, 1, 2]), iteration=1, role="growth_parent"
    )
    counts = g._resolution_exposure_veto.excluded_counts
    assert counts["relocation_target"] == 2
    assert counts["growth_parent"] == 2


def test_telemetry_only_mode_accumulates_but_never_vetoes():
    g = _veto_gaussians([0, 0])
    args = Args(streaming_resolution_exposure_telemetry=True)
    cfg = exposure_config_from_args(args)
    assert cfg.enabled and cfg.telemetry_only
    dead = torch.tensor([True, True])
    assert exclude_unexposed_mask(
        g, args, dead, iteration=1, role="relocation_target"
    ).tolist() == [True, True]


# ---------------------------------------------------------------------------
# the exposure-currency young gate (A2)
# ---------------------------------------------------------------------------


def test_exposure_young_mask_is_none_until_published():
    g = _veto_gaussians([0, 5])
    assert exposure_young_mask(g, 2) is None


def test_exposure_young_mask_flags_the_undermeasured():
    g = _veto_gaussians([0, 2, 5])
    publish_exposure_young_min(g, Args(streaming_resolution_exposure_veto=True))
    mask = exposure_young_mask(g, 3)
    assert mask.tolist() == [True, True, False]


def test_exposure_young_gate_composes_with_the_age_gate_as_a_floor():
    from eligsir.utils.energy_mcmc import _or_exposure_young

    g = _veto_gaussians([0, 9])
    publish_exposure_young_min(g, Args(streaming_resolution_exposure_veto=True))
    age_young = torch.tensor([False, True])
    combined = _or_exposure_young(g, age_young)
    # union, never a replacement: the age gate's own protection survives
    assert combined.tolist() == [True, True]


def test_exposure_young_gate_is_a_noop_when_off():
    from eligsir.utils.energy_mcmc import _or_exposure_young

    g = _veto_gaussians([0, 0])
    age_young = torch.tensor([False, True])
    assert _or_exposure_young(g, age_young).tolist() == [False, True]


# ---------------------------------------------------------------------------
# telemetry (§6.2.1)
# ---------------------------------------------------------------------------


def test_group_stats_uses_the_rasterizer_threshold_not_alpha_dead():
    alpha = torch.tensor([0.001, 0.003, 0.5, 0.9])  # two below 1/255
    stats = group_stats(alpha, alpha_dead=0.0005)
    assert stats["n"] == 4
    assert stats["frac_below_rasterizer_cull"] == pytest.approx(0.5)
    assert stats["frac_below_alpha_dead"] == pytest.approx(0.0)
    assert stats["alpha_p50"] == pytest.approx(0.2515, abs=1e-3)


def test_group_stats_on_empty_group_is_nan_not_zero():
    stats = group_stats(torch.zeros(0), alpha_dead=0.0005)
    assert stats["n"] == 0
    assert math.isnan(stats["frac_below_rasterizer_cull"])


def test_groups_are_restricted_to_depth_inserted_population():
    alpha = torch.tensor([0.5, 0.5, 0.5, 0.5])
    birth = torch.tensor(
        [
            BIRTH_METHOD_SOURCE_DEPTH,
            BIRTH_METHOD_SOURCE_DEPTH,
            BIRTH_METHOD_MCMC_GROW,
            BIRTH_METHOD_MCMC_GROW,
        ],
        dtype=torch.int8,
    )
    exposed = torch.tensor([True, False, True, False])
    e_plus, e_minus = exposure_groups(alpha=alpha, birth_method=birth, exposed=exposed)
    assert e_plus.tolist() == [True, False, False, False]
    assert e_minus.tolist() == [False, True, False, False]


def test_snapshot_partitions_a_synthetic_population():
    alpha = torch.tensor([0.001, 0.5, 0.002, 0.8])
    birth = torch.full((4,), BIRTH_METHOD_SOURCE_DEPTH, dtype=torch.int8)
    exposed = torch.tensor([True, True, False, False])
    uids = torch.tensor([10, 11, 12, 13])
    snap = snapshot(
        iteration=1000, alpha=alpha, exposed=exposed, birth_method=birth,
        gaussian_uid=uids, alpha_dead=0.0005,
    )
    assert snap.e_plus["n"] == 2 and snap.e_minus["n"] == 2
    assert snap.e_plus["frac_below_rasterizer_cull"] == pytest.approx(0.5)
    assert snap.e_minus["frac_below_rasterizer_cull"] == pytest.approx(0.5)
    assert snap.e_minus_culled_uids.tolist() == [12]


def test_zombie_recovery_frac_records_the_falsifiable_sub_test():
    alpha = torch.tensor([0.001, 0.001])
    birth = torch.full((2,), BIRTH_METHOD_SOURCE_DEPTH, dtype=torch.int8)
    uids = torch.tensor([1, 2])
    at_switch = snapshot(
        iteration=100, alpha=alpha, exposed=torch.tensor([False, False]),
        birth_method=birth, gaussian_uid=uids,
    )
    # neither recovers — the mechanism's own prediction
    assert zombie_recovery_frac(
        at_switch, alpha=torch.tensor([0.001, 0.0005]), gaussian_uid=uids
    ) == pytest.approx(0.0)
    # one climbs back above 1/255 — a measured falsification, recorded either way
    assert zombie_recovery_frac(
        at_switch, alpha=torch.tensor([0.5, 0.001]), gaussian_uid=uids
    ) == pytest.approx(0.5)
    # a row that disappeared entirely counts as NOT recovered
    assert zombie_recovery_frac(
        at_switch, alpha=torch.tensor([0.5]), gaussian_uid=torch.tensor([1])
    ) == pytest.approx(0.5)


def test_outcome_table_separates_the_three_rows():
    # Row 1: topology destruction — E- relocated far more often.
    assert (
        classify_outcome(
            relocation_rate_e_minus=0.4,
            relocation_rate_e_plus=0.1,
            holeishness_series=[0.5, 0.5],
            frac_below_cull_e_minus_at_switch=0.0,
            frac_below_cull_e_minus_at_end=0.0,
        )
        == OUTCOME_TOPOLOGY_DESTRUCTION
    )
    # Row 2: zombie — relocation flat, H plateaus, cull fraction does not fall.
    assert (
        classify_outcome(
            relocation_rate_e_minus=0.1,
            relocation_rate_e_plus=0.1,
            holeishness_series=[0.5, 0.5],
            frac_below_cull_e_minus_at_switch=0.4,
            frac_below_cull_e_minus_at_end=0.4,
        )
        == OUTCOME_OPACITY_DEATH
    )
    # Row 3: the desired outcome — H decays, cull fraction low throughout.
    assert (
        classify_outcome(
            relocation_rate_e_minus=0.1,
            relocation_rate_e_plus=0.1,
            holeishness_series=[0.5, 0.1],
            frac_below_cull_e_minus_at_switch=0.01,
            frac_below_cull_e_minus_at_end=0.01,
        )
        == OUTCOME_HARMLESS_UNDER_DETAIL
    )


def test_rows_one_and_two_share_an_H_signature_and_are_split_only_by_opacity():
    """The reason §6.2.1's measurement is REQUIRED, not optional."""
    common = dict(
        relocation_rate_e_plus=0.1,
        holeishness_series=[0.5, 0.5],  # identical plateau in both
    )
    zombie = classify_outcome(
        relocation_rate_e_minus=0.1,
        frac_below_cull_e_minus_at_switch=0.4,
        frac_below_cull_e_minus_at_end=0.4,
        **common,
    )
    topology = classify_outcome(
        relocation_rate_e_minus=0.4,
        frac_below_cull_e_minus_at_switch=0.0,
        frac_below_cull_e_minus_at_end=0.0,
        **common,
    )
    assert zombie != topology


def test_outcome_is_indeterminate_without_evidence():
    assert (
        classify_outcome(
            relocation_rate_e_minus=None,
            relocation_rate_e_plus=None,
            holeishness_series=None,
            frac_below_cull_e_minus_at_switch=None,
            frac_below_cull_e_minus_at_end=None,
        )
        == OUTCOME_INDETERMINATE
    )


def test_report_block_shape_and_null_control_provenance():
    alpha = torch.tensor([0.001, 0.5])
    birth = torch.full((2,), BIRTH_METHOD_SOURCE_DEPTH, dtype=torch.int8)
    snap = snapshot(
        iteration=10, alpha=alpha, exposed=torch.tensor([True, False]),
        birth_method=birth, gaussian_uid=torch.tensor([1, 2]),
    )
    block = exposure_opacity_survival_block(
        at_tier_switch=None,
        at_run_end=snap,
        predicate=PREDICATE_LATTICE,
        alpha_dead_effective=0.005,
        proxy_ambiguous_frac=None,
    )
    assert block["rasterizer_alpha_threshold"] == pytest.approx(
        RASTERIZER_ALPHA_THRESHOLD
    )
    assert block["exposure_predicate"] == PREDICATE_LATTICE
    # A single run cannot produce the NATIVE-tier control arm — recorded as null
    # with provenance rather than fabricated.
    assert block["groups"]["control"] is None
    assert "control requires a separate NATIVE-tier run" in block["control_provenance"]


def test_per_checkpoint_row_carries_the_three_separating_fields():
    alpha = torch.tensor([0.001, 0.5])
    birth = torch.full((2,), BIRTH_METHOD_SOURCE_DEPTH, dtype=torch.int8)
    snap = snapshot(
        iteration=42, alpha=alpha, exposed=torch.tensor([False, True]),
        birth_method=birth,
    )
    row = per_checkpoint_row(snap, resolution_tier="QUARTER")
    assert set(row) == {
        "iteration",
        "resolution_tier",
        "frac_below_cull_e_minus",
        "frac_below_cull_e_plus",
        "n_e_minus",
    }
    assert row["n_e_minus"] == 1
