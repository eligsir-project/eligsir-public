"""Resolution-exposure eligibility for the energy-guided MCMC (issue #1262).

Implements category **(a) topology protection** of
``docs/architecture/2026-08-29-mcmc-resolution-exposure-eligibility.md``:
distinguish *"observed at this tier and found low-utility"* from *"the current
tier's ray lattice could not resolve this Gaussian, so there is no evidence
either way"*, and remove the latter rows from death/relocation selection **and
from the quantile statistic pool**.

Two quantities the design doc (§1.4) insists must not be conflated:

``V_i^renderer``
    Did gsplat's ``eps2d``-dilated rasterizer include Gaussian ``i``?  Answered
    by ``render_pkg["visibility_filter"]``.  **Tier-safe** — ``eps2d`` is added
    to the 2D covariance *before* the radius is taken, so the radius is floored
    independent of resolution.  Reading it here would ship this module silently
    inert (the W3-45 ``phase_c_reseeding`` failure mode).

``E_i^measurement``
    Did this tier's actual sensor-ray sampling contain enough evidence to
    *judge* Gaussian ``i``?  That is what this module computes, from the
    **pre-eps2d** projected ellipse expressed in **native** pixel units.

Both traps named in §3.3 are avoided by construction here: this module never
reads ``render_pkg["radii"]``, and it resolves the **native** focal length from
the camera's canonical-resolution binding (``camera.fx`` has already been
divided by the downsample factor, so using it would cancel the tier out of the
predicate entirely).

Everything is pure torch and device-agnostic — unit-testable on CPU against
duck-typed stand-ins, with no ``GsplatGaussianModel``/CUDA dependency.

**Scope.**  Only category (a) is implemented.  Category (b) (population-loss /
opacity protection) and category (c) (position-noise *policy*) are open
follow-ups per §2.2.2 — :meth:`ResolutionExposureVeto.zero_position_noise`
exists as part of the veto surface but is deliberately not wired into the noise
path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

#: gsplat's own rasterizer cull threshold (``cuda/include/Common.h:72``).  A
#: Gaussian below this opacity is dropped from the rasterizer entirely: zero
#: radii, absent from ``meta["gaussian_ids"]``, therefore receiving no gradient
#: and unable to climb back — the "zombie Gaussian" of §2.2.1.
RASTERIZER_ALPHA_THRESHOLD = 1.0 / 255.0

PREDICATE_LATTICE = "lattice_intersection"
PREDICATE_SIGMA_RATIO = "sigma_ratio"
SUPPORTED_PREDICATES = frozenset({PREDICATE_LATTICE, PREDICATE_SIGMA_RATIO})

#: Buffer names this module owns.  Registered at every site listed in §7.3 so
#: they cannot desync on MCMC growth (GAP-4).
EXPOSURE_BUFFER_NAMES = (
    "exposure_count",
    "exposure_last_view_uid",
    "supervised_tier_high_water",
)

#: Sentinel for ``supervised_tier_high_water`` meaning "never exposed at any
#: tier".  Deliberately below every real ``ResolutionTier`` ordinal.
TIER_HIGH_WATER_UNSET = -1

#: Max distance from an arbitrary point to the nearest point of a square
#: lattice of spacing ``s`` is ``s * sqrt(2) / 2``.  A disk of at least that
#: radius therefore always contains a lattice point, whatever its phase.
_LATTICE_COVER_RADIUS_FACTOR = 0.7071067811865476

#: Neighbourhood half-width (in lattice cells) searched around the projected
#: mean before falling back to the guaranteed-cover test above.
_LATTICE_SEARCH_K = 2


@dataclass(frozen=True)
class ExposureConfig:
    """Resolved knobs for the exposure predicate and veto."""

    enabled: bool = False
    predicate: str = PREDICATE_LATTICE
    #: ``sigma_ratio`` proxy threshold: exposed iff ``sigma_proj >= tau * s``.
    tau: float = 0.5
    #: Mahalanobis radius of the ellipse tested for lattice intersection.
    sigma_c: float = 3.0
    #: ``n_min`` of §3.4 — below this cumulative exposure count a Gaussian is
    #: "not yet measured enough" and is protected.
    min_exposures: int = 3
    #: W3-45 bounded-immunity ceiling (§4).  Protection lapses beyond this many
    #: iterations of age regardless of exposure, so a Gaussian that is never
    #: exposed at any tier the run reaches cannot be immune forever.
    #: ``0`` disables the ceiling (NOT recommended).
    max_protect_iters: int = 4000
    #: Accumulate exposure counters / telemetry without vetoing anything.
    telemetry_only: bool = False

    def __post_init__(self) -> None:
        if self.predicate not in SUPPORTED_PREDICATES:
            raise ValueError(
                "exposure predicate must be one of: "
                + ", ".join(sorted(SUPPORTED_PREDICATES))
            )


def exposure_config_from_args(args) -> ExposureConfig:
    """Build an :class:`ExposureConfig` from parsed CLI args (default: OFF)."""
    enabled = bool(getattr(args, "streaming_resolution_exposure_veto", False))
    telemetry = bool(getattr(args, "streaming_resolution_exposure_telemetry", False))
    return ExposureConfig(
        enabled=enabled or telemetry,
        predicate=str(
            getattr(args, "streaming_resolution_exposure_predicate", PREDICATE_LATTICE)
        ),
        tau=float(getattr(args, "streaming_resolution_exposure_tau", 0.5)),
        sigma_c=float(getattr(args, "streaming_resolution_exposure_sigma_c", 3.0)),
        min_exposures=int(
            getattr(args, "streaming_resolution_exposure_min_count", 3)
        ),
        max_protect_iters=int(
            getattr(args, "streaming_resolution_exposure_max_protect_iters", 4000)
        ),
        telemetry_only=(telemetry and not enabled),
    )


# --------------------------------------------------------------------------
# Native camera geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeCameraGeometry:
    """Native-resolution intrinsics plus the tier's decimation factor.

    ``fx``/``fy``/``width``/``height`` are **native**; ``downsample_factor`` is
    the tier's ``s``.  Resolving these from the canonical binding rather than
    from ``camera.fx`` is what keeps the predicate tier-sensitive (§3.3).
    """

    fx: float
    fy: float
    width: int
    height: int
    cx: float
    cy: float
    downsample_factor: int
    tier_ordinal: int = TIER_HIGH_WATER_UNSET


def native_camera_geometry(camera) -> Optional[NativeCameraGeometry]:
    """Resolve native intrinsics + downsample factor for ``camera``.

    Reads ``camera._canonical_resolution_binding_native_state`` (written by the
    resolution-materialization boundary) when present; otherwise the camera is
    at its own native resolution and ``s = 1``.  Returns ``None`` when no focal
    length can be resolved at all.
    """
    native = getattr(camera, "_canonical_resolution_binding_native_state", None)
    tier = getattr(camera, "_canonical_resolution_tier", None)
    factor = 1
    tier_ordinal = TIER_HIGH_WATER_UNSET
    if tier is not None:
        factor = int(getattr(tier, "downsample_factor", 1) or 1)
        tier_ordinal = _tier_ordinal(tier)
    source = native if native is not None else camera
    fx = getattr(source, "fx", None)
    fy = getattr(source, "fy", None)
    if fx is None:
        return None
    if fy is None:
        fy = fx
    width = int(
        getattr(source, "width", None)
        or getattr(source, "image_width", 0)
        or 0
    )
    height = int(
        getattr(source, "height", None)
        or getattr(source, "image_height", 0)
        or 0
    )
    cx = float(getattr(source, "cx", width / 2.0))
    cy = float(getattr(source, "cy", height / 2.0))
    return NativeCameraGeometry(
        fx=float(fx),
        fy=float(fy),
        width=width,
        height=height,
        cx=cx,
        cy=cy,
        downsample_factor=max(1, factor),
        tier_ordinal=tier_ordinal,
    )


def _tier_ordinal(tier) -> int:
    """Best-effort monotone ordinal for a resolution tier (finer = larger).

    Derived from the tier's own ``downsample_factor`` so this module never has
    to import — let alone depend on the internals of — the resolution
    scheduler.
    """
    factor = int(getattr(tier, "downsample_factor", 1) or 1)
    ordinal = 0
    while factor > 1:
        factor //= 2
        ordinal += 1
    # NATIVE (s=1) is the finest tier and must sort highest.
    return 8 - ordinal


# --------------------------------------------------------------------------
# The exposure predicate  E_i(R, v)
# --------------------------------------------------------------------------


@dataclass
class ExposureResult:
    """Per-Gaussian exposure for one supervision view."""

    #: ``[N]`` bool, True = this tier's ray lattice could carry attributable
    #: evidence about the Gaussian.
    exposed: torch.Tensor
    #: ``[N]`` bool, True = the Gaussian's projected mean is in front of the
    #: camera and inside the native image.
    on_screen: torch.Tensor
    #: ``[N]`` float, pre-``eps2d`` projected sigma in NATIVE pixel units.
    sigma_proj: torch.Tensor
    #: ``sigma_ratio`` proxy only: fraction of on-screen Gaussians in the band
    #: where the lattice PHASE, not the size, decides the truth (§3.2).
    ambiguous_frac: float = 0.0
    n_ambiguous: int = 0
    predicate: str = PREDICATE_LATTICE
    downsample_factor: int = 1
    tier_ordinal: int = TIER_HIGH_WATER_UNSET


def _world_to_camera(gaussians, camera):
    """Return ``(R_wc [3,3], t_wc [3])`` mapping world -> camera, or ``None``."""
    means = gaussians.get_xyz
    device, dtype = means.device, means.dtype
    viewmat = getattr(camera, "world_view_transform", None)
    if viewmat is not None:
        m = torch.as_tensor(viewmat, device=device, dtype=dtype)
        if m.shape == (4, 4):
            # This repo's Camera stores the TRANSPOSED view matrix (row-vector
            # convention: x_cam = x_world @ world_view_transform).
            return m[:3, :3].transpose(0, 1).contiguous(), m[3, :3].contiguous()
    R = getattr(camera, "R", None)
    T = getattr(camera, "T", None)
    if R is None or T is None:
        return None
    R = torch.as_tensor(R, device=device, dtype=dtype)
    T = torch.as_tensor(T, device=device, dtype=dtype)
    # Camera.R is world->camera transposed (colmap convention in this repo).
    return R.transpose(0, 1).contiguous(), T.contiguous()


def _covariance_world(gaussians) -> Optional[torch.Tensor]:
    """``[N,3,3]`` world-space covariance from scales + quats, or ``None``."""
    scales = getattr(gaussians, "get_scaling", None)
    quats = getattr(gaussians, "get_rotation", None)
    if scales is None or quats is None:
        return None
    q = torch.nn.functional.normalize(quats, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    S = scales
    M = R * S.unsqueeze(-2)  # R @ diag(S)
    return M @ M.transpose(1, 2)


def compute_exposure(
    gaussians,
    camera,
    config: ExposureConfig,
) -> Optional[ExposureResult]:
    """Evaluate ``E_i(R, v)`` for every Gaussian against one supervision view.

    Returns ``None`` when the camera geometry cannot be resolved (in which case
    callers must treat exposure as unknown and change nothing).
    """
    geom = native_camera_geometry(camera)
    if geom is None:
        return None
    means = gaussians.get_xyz.detach()
    n = int(means.shape[0])
    device, dtype = means.device, means.dtype
    if n == 0:
        empty_b = torch.zeros(0, dtype=torch.bool, device=device)
        return ExposureResult(
            exposed=empty_b,
            on_screen=empty_b,
            sigma_proj=torch.zeros(0, dtype=dtype, device=device),
            predicate=config.predicate,
            downsample_factor=geom.downsample_factor,
            tier_ordinal=geom.tier_ordinal,
        )
    wc = _world_to_camera(gaussians, camera)
    if wc is None:
        return None
    R_wc, t_wc = wc
    cam_pts = means @ R_wc.transpose(0, 1) + t_wc
    z = cam_pts[:, 2]
    in_front = z > 1e-6
    z_safe = z.clamp_min(1e-6)

    fx = torch.as_tensor(geom.fx, device=device, dtype=dtype)
    fy = torch.as_tensor(geom.fy, device=device, dtype=dtype)
    mu_u = fx * cam_pts[:, 0] / z_safe + geom.cx
    mu_v = fy * cam_pts[:, 1] / z_safe + geom.cy
    if geom.width > 0 and geom.height > 0:
        inside = (
            (mu_u >= 0) & (mu_u <= geom.width - 1)
            & (mu_v >= 0) & (mu_v <= geom.height - 1)
        )
    else:
        inside = torch.ones_like(in_front)
    on_screen = in_front & inside

    cov_w = _covariance_world(gaussians)
    s = int(geom.downsample_factor)
    if cov_w is None:
        # No shape information: fall back to a scalar sigma of zero, which the
        # predicates below read as "unresolvable at any coarse tier".
        sigma_uu = torch.zeros(n, device=device, dtype=dtype)
        sigma_vv = sigma_uu
        sigma_uv = sigma_uu
    else:
        cov_c = R_wc @ cov_w @ R_wc.transpose(0, 1)
        # EWA Jacobian in NATIVE pixel units (pre-eps2d, per §3.3).
        inv_z = 1.0 / z_safe
        j00 = fx * inv_z
        j02 = -fx * cam_pts[:, 0] * inv_z * inv_z
        j11 = fy * inv_z
        j12 = -fy * cam_pts[:, 1] * inv_z * inv_z
        J = torch.zeros(n, 2, 3, device=device, dtype=dtype)
        J[:, 0, 0] = j00
        J[:, 0, 2] = j02
        J[:, 1, 1] = j11
        J[:, 1, 2] = j12
        cov2d = J @ cov_c @ J.transpose(1, 2)
        sigma_uu = cov2d[:, 0, 0].clamp_min(0.0)
        sigma_vv = cov2d[:, 1, 1].clamp_min(0.0)
        sigma_uv = cov2d[:, 0, 1]

    # sigma_proj: the SECOND-largest projected axis (§3.3) — the axis that must
    # be resolvable for the Gaussian to be distinguishable across the lattice.
    tr = sigma_uu + sigma_vv
    det = (sigma_uu * sigma_vv - sigma_uv * sigma_uv).clamp_min(0.0)
    disc = (tr * tr / 4.0 - det).clamp_min(0.0).sqrt()
    lam_min = (tr / 2.0 - disc).clamp_min(0.0)
    lam_max = (tr / 2.0 + disc).clamp_min(0.0)
    sigma_min = lam_min.sqrt()
    sigma_max = lam_max.sqrt()
    sigma_proj = sigma_min  # second-largest axis of a 2D ellipse

    n_ambiguous = 0
    ambiguous_frac = 0.0
    if config.predicate == PREDICATE_SIGMA_RATIO:
        exposed = on_screen & (sigma_proj >= config.tau * s)
        if s > 1:
            lo = 0.5 * config.tau * s
            hi = 2.0 * config.tau * s
            band = on_screen & (sigma_proj >= lo) & (sigma_proj <= hi)
            n_ambiguous = int(band.sum())
            denom = int(on_screen.sum())
            ambiguous_frac = (n_ambiguous / denom) if denom else 0.0
    else:
        exposed = _lattice_intersection(
            mu_u, mu_v, sigma_uu, sigma_vv, sigma_uv, sigma_min,
            factor=s, c=config.sigma_c,
        ) & on_screen

    return ExposureResult(
        exposed=exposed,
        on_screen=on_screen,
        sigma_proj=sigma_proj,
        ambiguous_frac=ambiguous_frac,
        n_ambiguous=n_ambiguous,
        predicate=config.predicate,
        downsample_factor=s,
        tier_ordinal=geom.tier_ordinal,
    )


def _lattice_intersection(
    mu_u, mu_v, sigma_uu, sigma_vv, sigma_uv, sigma_min, *, factor: int, c: float
) -> torch.Tensor:
    """Does the pre-dilation ``c``-sigma ellipse cover a retained lattice point?

    Retained lattice (PR #1261's zero-phase decimation):
    ``L(s) = {(k*s, l*s)}``.  At ``s = 1`` every pixel is retained, so this is a
    near-no-op by construction — as §4 requires.

    **Documented approximation.**  Lattice points are searched only within
    ``±_LATTICE_SEARCH_K`` cells of the projected mean, so a needle whose long
    axis reaches a retained ray further away than that is classified
    *unexposed*.  That errs toward **over-protection**, the benign direction of
    §3.2's two misclassifications (it dilutes a measured effect; it never leaves
    a genuinely unexposed Gaussian eligible).
    """
    s = max(1, int(factor))
    # Guaranteed-cover shortcut: a disk of radius >= s*sqrt(2)/2 always contains
    # a lattice point of spacing s regardless of phase.
    covers = (c * sigma_min) >= (_LATTICE_COVER_RADIUS_FACTOR * s)

    det = (sigma_uu * sigma_vv - sigma_uv * sigma_uv).clamp_min(1e-12)
    inv00 = sigma_vv / det
    inv11 = sigma_uu / det
    inv01 = -sigma_uv / det
    degenerate = det <= 1e-12

    base_k = torch.round(mu_u / s)
    base_l = torch.round(mu_v / s)
    hit = covers.clone()
    k_span = range(-_LATTICE_SEARCH_K, _LATTICE_SEARCH_K + 1)
    for dk in k_span:
        for dl in k_span:
            pu = (base_k + dk) * s
            pv = (base_l + dl) * s
            du = pu - mu_u
            dv = pv - mu_v
            maha = inv00 * du * du + 2.0 * inv01 * du * dv + inv11 * dv * dv
            hit = hit | ((maha <= c * c) & ~degenerate)
    return hit


# --------------------------------------------------------------------------
# Per-Gaussian exposure buffers  (§7.3)
# --------------------------------------------------------------------------


def exposure_buffers_present(gaussians) -> bool:
    buf = getattr(gaussians, "exposure_count", None)
    return buf is not None and buf.shape[0] == int(gaussians.get_xyz.shape[0])


def ensure_exposure_buffers(gaussians) -> None:
    """Allocate/resize the exposure buffers (flag-gated by the caller).

    Resizing preserves history for the rows that survive — never a blanket
    re-allocation, which is exactly the ``last_supervised_iter`` defect
    (GAP-4) this design must not repeat.
    """
    n = int(gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device
    spec = (
        ("exposure_count", 0, torch.int32),
        ("exposure_last_view_uid", -1, torch.int64),
        ("supervised_tier_high_water", TIER_HIGH_WATER_UNSET, torch.int8),
    )
    for name, fill, dtype in spec:
        buf = getattr(gaussians, name, None)
        if buf is None:
            setattr(
                gaussians,
                name,
                torch.full((n,), fill, dtype=dtype, device=device),
            )
            continue
        if buf.shape[0] == n:
            continue
        if buf.shape[0] > n:
            setattr(gaussians, name, buf[:n].contiguous())
        else:
            pad = torch.full(
                (n - buf.shape[0],), fill, dtype=buf.dtype, device=buf.device
            )
            setattr(gaussians, name, torch.cat([buf, pad], dim=0))


def reset_exposure_rows(gaussians, indices) -> None:
    """Reset exposure state for relocated (reseeded) slots.

    A reseeded slot is *new geometry at a new position* and has had zero
    exposure there — it must NOT inherit the donor's counters (§7.3 site 7,
    mirroring ``reset_consensus_counts``).
    """
    n = int(gaussians.get_xyz.shape[0])
    for name, fill in (
        ("exposure_count", 0),
        ("exposure_last_view_uid", -1),
        ("supervised_tier_high_water", TIER_HIGH_WATER_UNSET),
    ):
        buf = getattr(gaussians, name, None)
        if buf is not None and buf.shape[0] == n:
            buf[indices] = fill


def _view_uid(camera) -> Optional[int]:
    for attr in ("uid", "colmap_id", "image_id"):
        value = getattr(camera, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def accumulate_exposure(gaussians, camera, config: ExposureConfig):
    """Fold one supervision view's exposure into the per-Gaussian counters.

    Deduplicated by view uid (the ``seen_count`` precedent) so the counter
    measures **distinct views**, not training steps — the degeneracy Wave 3's
    ``support_count`` already fell into.  Returns the :class:`ExposureResult`
    (or ``None`` if exposure could not be evaluated).
    """
    if not config.enabled:
        return None
    result = compute_exposure(gaussians, camera, config)
    if result is None:
        return None
    ensure_exposure_buffers(gaussians)
    exposed = result.exposed
    uid = _view_uid(camera)
    with torch.no_grad():
        if uid is None:
            fresh = exposed
        else:
            last = gaussians.exposure_last_view_uid
            fresh = exposed & (last != int(uid))
            gaussians.exposure_last_view_uid = torch.where(
                fresh, torch.full_like(last, int(uid)), last
            )
        counts = gaussians.exposure_count
        gaussians.exposure_count = counts + fresh.to(counts.dtype)
        if result.tier_ordinal != TIER_HIGH_WATER_UNSET:
            hw = gaussians.supervised_tier_high_water
            gaussians.supervised_tier_high_water = torch.where(
                exposed,
                torch.maximum(hw, torch.full_like(hw, int(result.tier_ordinal))),
                hw,
            )
    return result


# --------------------------------------------------------------------------
# The veto  (§3.6)
# --------------------------------------------------------------------------


class ResolutionExposureVeto:
    """Fifth member of this repo's composable per-Gaussian veto pipeline.

    Same four-method surface as :class:`utils.seed_lineage_freeze.SeedLineageFreeze`
    (``current_mask`` / ``exclude_mask`` / ``filter_indices`` /
    ``zero_position_noise``) with per-role exclusion counters.

    Composition contract: every method is a pure intersection/filter that only
    ever *removes* rows from a kill set.  It therefore composes with the depth
    lock, the lifecycle shield, the spatial bubble and the seed-lineage freeze
    without double-counting — applying any two in either order gives the same
    mask.
    """

    def __init__(self, gaussians, config: ExposureConfig):
        self.gaussians = gaussians
        self.config = config
        self.excluded_counts: dict[str, int] = {}
        self.bound_ceiling_lapses = 0
        self._ceiling_warned = False

    # -- protection predicate ------------------------------------------------

    def active(self) -> bool:
        return bool(self.config.enabled) and not self.config.telemetry_only

    def current_mask(self, *, iteration: int) -> Optional[torch.Tensor]:
        """``[N]`` bool, True = **protected** (insufficiently exposed).

        ``None`` when the veto is inactive or the buffers are unavailable, so
        every call site is a byte-identical no-op with the flag off.
        """
        if not self.active():
            return None
        gaussians = self.gaussians
        n = int(gaussians.get_xyz.shape[0])
        counts = getattr(gaussians, "exposure_count", None)
        if counts is None or counts.shape[0] != n:
            return None
        protected = counts.to(torch.int64) < int(self.config.min_exposures)
        # W3-45 bounded immunity: protection must lapse, or a Gaussian never
        # exposed at any tier the run reaches would be immune forever.
        ceiling = int(self.config.max_protect_iters)
        if ceiling > 0:
            birth_iter = getattr(gaussians, "birth_iter", None)
            if birth_iter is not None and birth_iter.shape[0] == n:
                age = int(iteration) - birth_iter.to(dtype=torch.int64)
                lapsed = protected & (age >= ceiling)
                n_lapsed = int(lapsed.sum())
                if n_lapsed:
                    self.bound_ceiling_lapses += n_lapsed
                    if not self._ceiling_warned:
                        self._ceiling_warned = True
                        print(
                            "[resolution-exposure] WARNING: bounded-immunity ceiling "
                            f"({ceiling} iters) lapsed protection for {n_lapsed} "
                            "never-sufficiently-exposed Gaussians; they rejoin normal "
                            "selection."
                        )
                protected = protected & ~lapsed
        return protected

    # -- veto surface --------------------------------------------------------

    def _count(self, role: str, amount) -> None:
        self.excluded_counts[role] = self.excluded_counts.get(role, 0) + int(amount)

    def exclude_mask(self, mask, *, iteration: int, role: str):
        protected = self.current_mask(iteration=iteration)
        if protected is None or mask is None or mask.shape != protected.shape:
            return mask
        mask = mask.to(dtype=torch.bool)
        self._count(role, (mask & protected).sum())
        return mask & ~protected

    def filter_indices(self, indices, *, iteration: int, role: str):
        protected = self.current_mask(iteration=iteration)
        if protected is None or indices is None or indices.numel() == 0:
            return indices
        valid = (indices >= 0) & (indices < protected.shape[0])
        safe = indices.clamp(min=0, max=protected.shape[0] - 1)
        blocked = protected[safe] & valid
        self._count(role, blocked.sum())
        return indices[valid & ~blocked]

    def zero_position_noise(self, noise, *, iteration: int):
        """Category (c) surface — implemented, deliberately NOT wired.

        §2.2.2 lists position-noise protection as an open follow-up whose
        *policy* (full zeroing vs. exposure-proportional attenuation) is
        unspecified.  The method exists so the veto has the same surface as
        ``SeedLineageFreeze``; no production call site invokes it.
        """
        protected = self.current_mask(iteration=iteration)
        if protected is None or noise is None or noise.shape[0] != protected.shape[0]:
            return noise
        result = noise.clone()
        result[protected] = 0
        self._count("noise", protected.sum())
        return result

    def statistic_pool_mask(self, *, iteration: int, base=None):
        """``active_mask`` for the quantile/mean statistic pool (§4 item 2).

        The half that is easy to miss: unexposed rows must leave the statistic
        pool too, or they drag the threshold and change which *exposed*
        Gaussians die.  Same precedent as the bubble's ``active_mask``.
        """
        protected = self.current_mask(iteration=iteration)
        if protected is None:
            return base
        active = ~protected
        if base is not None and base.shape == active.shape:
            return base & active.to(base.device)
        return active


# --------------------------------------------------------------------------
# Free-function adapters (mirroring utils/seed_lineage_freeze.py)
# --------------------------------------------------------------------------


def get_resolution_exposure_veto(gaussians, args) -> Optional[ResolutionExposureVeto]:
    """Cached per-model veto, or ``None`` when the feature is off."""
    if gaussians is None or args is None:
        return None
    try:
        config = exposure_config_from_args(args)
    except ValueError:
        return None
    if not config.enabled:
        return None
    veto = getattr(gaussians, "_resolution_exposure_veto", None)
    if veto is None or veto.config != config:
        veto = ResolutionExposureVeto(gaussians, config)
        gaussians._resolution_exposure_veto = veto
    return veto


def exclude_unexposed_mask(gaussians, args, mask, *, iteration: int, role: str):
    veto = get_resolution_exposure_veto(gaussians, args)
    if veto is None:
        return mask
    return veto.exclude_mask(mask, iteration=iteration, role=role)


def filter_unexposed_indices(gaussians, args, indices, *, iteration: int, role: str):
    veto = get_resolution_exposure_veto(gaussians, args)
    if veto is None:
        return indices
    return veto.filter_indices(indices, iteration=iteration, role=role)


def unexposed_statistic_pool_mask(gaussians, args, *, iteration: int, base=None):
    veto = get_resolution_exposure_veto(gaussians, args)
    if veto is None:
        return base
    return veto.statistic_pool_mask(iteration=iteration, base=base)


def exposure_veto_telemetry(gaussians) -> dict:
    veto = getattr(gaussians, "_resolution_exposure_veto", None)
    if veto is None:
        return {}
    return {
        "excluded_counts": dict(veto.excluded_counts),
        "bounded_immunity_lapses": int(veto.bound_ceiling_lapses),
        "predicate": veto.config.predicate,
        "min_exposures": int(veto.config.min_exposures),
        "max_protect_iters": int(veto.config.max_protect_iters),
    }


# --------------------------------------------------------------------------
# The exposure-currency young gate  (§3.4, arm A2)
# --------------------------------------------------------------------------


def publish_exposure_young_min(gaussians, args) -> None:
    """Stash ``n_min`` on the model so args-free dead-mask sites can read it."""
    veto = get_resolution_exposure_veto(gaussians, args)
    gaussians._exposure_young_min = (
        int(veto.config.min_exposures) if (veto is not None and veto.active()) else 0
    )


def exposure_young_mask(gaussians, n_rows: int):
    """``[N]`` bool "not yet measured enough", or ``None`` when unavailable.

    Composed with the existing age gate as ``young = age_young | exposure_young``
    (§3.4: the two protect against different things — the age gate protects a
    Gaussian that *is* being measured but has not converged; this one protects a
    Gaussian that is not being measured at all).  The age gate is kept as a
    floor, never replaced.
    """
    n_min = int(getattr(gaussians, "_exposure_young_min", 0) or 0)
    if n_min <= 0:
        return None
    counts = getattr(gaussians, "exposure_count", None)
    if counts is None or counts.shape[0] != int(n_rows):
        return None
    return counts.to(torch.int64) < n_min


__all__ = [
    "RASTERIZER_ALPHA_THRESHOLD",
    "PREDICATE_LATTICE",
    "PREDICATE_SIGMA_RATIO",
    "EXPOSURE_BUFFER_NAMES",
    "TIER_HIGH_WATER_UNSET",
    "ExposureConfig",
    "ExposureResult",
    "NativeCameraGeometry",
    "ResolutionExposureVeto",
    "accumulate_exposure",
    "compute_exposure",
    "ensure_exposure_buffers",
    "exclude_unexposed_mask",
    "exposure_buffers_present",
    "exposure_config_from_args",
    "exposure_veto_telemetry",
    "exposure_young_mask",
    "filter_unexposed_indices",
    "get_resolution_exposure_veto",
    "native_camera_geometry",
    "publish_exposure_young_min",
    "reset_exposure_rows",
    "unexposed_statistic_pool_mask",
]
