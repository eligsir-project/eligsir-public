"""Borrowed, device-resident read view over live Gaussian map state.

Wave 1 consolidation plan, step 2 (see
the read-only view boundary.

``GaussianMapReadView`` is deliberately distinct from
:class:`eligsir.map.snapshots.MapSnapshot`: this type stays on-GPU and
is only valid while the map's topology is unchanged, so it is meant for the
hot path (renderer, optimization engine, scoring, correspondence search) —
not for viewers, persistence, or any other consumer that needs an owned,
stable payload across mutations. See ``docs/architecture/gaussian_ownership_audit.md``
for the read-only-view-candidate call sites this type is meant to replace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class GaussianMapReadView:
    """A read-only, device-resident view of the live Gaussian map.

    All tensor fields are borrowed references into the backend's live
    storage — they are not copies. The view is only valid for the
    ``map_version``/``topology_version`` it was produced under; a consumer
    that holds a view across a mutation must request a fresh one.

    ``*_activated`` fields mirror the model's existing ``get_xyz`` /
    ``get_scaling`` / ``get_rotation`` / ``get_opacity`` accessors (values
    after exp/sigmoid/normalize). The raw (pre-activation) fields are
    exposed alongside them because some existing call sites — e.g. anchor
    maturity bookkeeping in ``train_streaming.py`` — specifically need the
    raw log/logit values, not the activated ones; giving both here means
    that code no longer needs to reach into ``.params[...]`` directly (see
    the ownership audit's ``read-only-view-candidate`` entries).
    """

    # Activated (post exp/sigmoid/normalize) values — the renderer's usual
    # read path, mirrors get_xyz/get_scaling/get_rotation/get_opacity.
    means: "torch.Tensor"
    scales: "torch.Tensor"
    quaternions: "torch.Tensor"
    opacities: "torch.Tensor"
    sh0: "torch.Tensor"
    shN: "torch.Tensor"

    # Raw (pre-activation) values — for callers that specifically need the
    # log-scale/logit-opacity parameterization (e.g. anchor snapshotting).
    raw_scales: "torch.Tensor"
    raw_opacities: "torch.Tensor"

    # Identity/versioning.
    active_count: int
    map_version: int
    topology_version: int
    active_sh_degree: int = 0


@dataclass(frozen=True)
class GaussianMapOutputSample:
    """Owned O(cap) device sample captured at one backend version boundary."""

    positions: "torch.Tensor"
    colors: "torch.Tensor"
    scales: "torch.Tensor"
    quats: "torch.Tensor"
    opacities: "torch.Tensor"
    n_total_gaussians: int
    map_content_version: int
    topology_version: int
    parameter_version: int


def sample_output_from_model(
    gaussians,
    *,
    max_gaussians: int,
    map_content_version: int,
    topology_version: int,
    parameter_version: int,
) -> GaussianMapOutputSample:
    """Activate and own only deterministic selected rows, never full tensors."""

    import torch

    cap = int(max_gaussians)
    if cap <= 0:
        raise ValueError("max_gaussians must be positive")
    means = gaussians.params["means"] if hasattr(gaussians, "params") else gaussians._xyz
    total = int(means.shape[0])
    if total <= 0:
        raise ValueError("cannot publish an empty Gaussian map")
    count = min(total, cap)
    indices = torch.div(
        torch.arange(count, device=means.device, dtype=torch.int64) * total,
        count,
        rounding_mode="floor",
    )

    def selected(value):
        return value.detach().index_select(0, indices).clone().contiguous()

    if hasattr(gaussians, "params"):
        positions = selected(gaussians.params["means"])
        scales = torch.exp(selected(gaussians.params["scales"]))
        quats = torch.nn.functional.normalize(
            selected(gaussians.params["quats"]), dim=1, eps=1e-8
        )
        opacities = torch.sigmoid(selected(gaussians.params["opacities"])).unsqueeze(-1)
        sh0 = selected(gaussians.params["sh0"])
    else:
        positions = selected(gaussians._xyz)
        scales = gaussians.scaling_activation(selected(gaussians._scaling))
        quats = gaussians.rotation_activation(selected(gaussians._rotation))
        opacities = gaussians.opacity_activation(selected(gaussians._opacity))
        sh0 = selected(gaussians._features_dc)
    if sh0.ndim == 3 and int(sh0.shape[1]) == 1:
        sh0 = sh0[:, 0, :]
    colors = (0.5 + 0.282095 * sh0).clamp(0.0, 1.0)
    return GaussianMapOutputSample(
        positions=positions,
        colors=colors,
        scales=scales,
        quats=quats,
        opacities=opacities,
        n_total_gaussians=total,
        map_content_version=int(map_content_version),
        topology_version=int(topology_version),
        parameter_version=int(parameter_version),
    )


def read_view_from_model(
    gaussians, map_version: int = 0, topology_version: int = 0
) -> GaussianMapReadView:
    """Build a :class:`GaussianMapReadView` directly from a raw
    ``GsplatGaussianModel``/``GaussianModel`` instance, without requiring a
    full :class:`eligsir.map.contracts.GaussianMapBackend` to already be
    wired in at the call site.

    This is a bridge for call sites (viewers, reports) that only need a
    one-shot read view and predate ``MappingCoordinator`` owning a
    long-lived backend instance (Wave 1 step 10) — once that lands, those
    call sites should hold a real backend and call ``backend.read_view()``
    instead. :meth:`eligsir.map.adapters.gsplat.GsplatMapBackend.read_view`
    and its legacy counterpart both delegate to this same function, so there
    is exactly one place that maps model attributes onto this dataclass.
    """
    # `params` (an nn.ParameterDict) only exists on the gsplat layout; the
    # legacy layout stores `_features_dc`/`_features_rest` directly. Gate on
    # the actual storage shape rather than `uses_gsplat_layout`, which the
    # legacy model doesn't define at all (so a naive attribute default here
    # would silently mis-route legacy models through the gsplat branch).
    if hasattr(gaussians, "params"):
        sh0 = gaussians.params["sh0"]
        shN = gaussians.params["shN"]
    else:
        sh0 = gaussians._features_dc
        shN = gaussians._features_rest
    return GaussianMapReadView(
        means=gaussians.get_xyz,
        scales=gaussians.get_scaling,
        quaternions=gaussians.get_rotation,
        opacities=gaussians.get_opacity,
        sh0=sh0,
        shN=shN,
        raw_scales=gaussians.raw_scales,
        raw_opacities=gaussians.raw_opacities,
        active_count=int(gaussians.get_xyz.shape[0]),
        map_version=map_version,
        topology_version=topology_version,
        active_sh_degree=int(getattr(gaussians, "active_sh_degree", 0)),
    )
