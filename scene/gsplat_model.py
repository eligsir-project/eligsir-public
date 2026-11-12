import math
import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn

from eligsir.map.row_state import GaussianRowStateRegistry, RowStateDescriptor
from eligsir.utils.gaussian_belief import (
    NEW_BUFFER_NAMES as BELIEF_BUFFER_NAMES,
    ensure_belief_buffers,
    init_belief_buffers,
    new_row_belief_buffers,
    prune_belief_buffers,
    reset_belief_buffers,
)
from eligsir.utils.gaussian_lineage import BIRTH_METHOD_BOOTSTRAP, BIRTH_METHOD_SOURCE_DEPTH, new_root_lineage_rows
from eligsir.utils.general_utils import get_expon_lr_func, inverse_sigmoid
from eligsir.utils.rocm_knn_fallback import distCUDA2
from eligsir.utils.sh_utils import RGB2SH
from eligsir.utils.system_utils import mkdir_p


def _immutable_tensor_snapshot(value):
    """Recursively copy materialized checkpoint tensors into CPU storage."""
    if isinstance(value, torch.Tensor):
        if value.device.type == "meta":
            raise ValueError("cannot checkpoint a non-materialized meta tensor")
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _immutable_tensor_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_immutable_tensor_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_immutable_tensor_snapshot(item) for item in value)
    return value


class OptimizerDictProxy:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        groups = []
        for optimizer in self.optimizers.values():
            groups.extend(optimizer.param_groups)
        return groups

    @property
    def state(self):
        merged = {}
        for optimizer in self.optimizers.values():
            merged.update(optimizer.state)
        return merged

    def step(self, visibility=None):
        for optimizer in self.optimizers.values():
            if visibility is not None:
                optimizer.step(visibility)
            else:
                optimizer.step()

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}

    def load_state_dict(self, state_dict):
        for name, state in state_dict.items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state)


class GsplatGaussianModel:
    uses_gsplat_layout = True

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.params = nn.ParameterDict()
        self.optimizers = {}
        self.optimizer = OptimizerDictProxy(self.optimizers)
        self.optimizer_type = "adam"
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.xyz_scheduler_args = None
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.visibility_ema = torch.empty(0)
        # Streaming state
        self.birth_frame = torch.empty(0, dtype=torch.int32, device="cuda")
        self.birth_iter = torch.empty(0, dtype=torch.int32, device="cuda")
        self.support_count = torch.empty(0, dtype=torch.int16, device="cuda")
        self.provisional = torch.empty(0, dtype=torch.bool, device="cuda")
        self.anchor_xyz = torch.empty((0, 3), dtype=torch.float32, device="cuda")
        self.anchor_iter = torch.empty(0, dtype=torch.int32, device="cuda")
        self.depth_conflict_count = torch.empty(0, dtype=torch.int16, device="cuda")
        self.depth_last_support_uid = torch.empty(0, dtype=torch.int32, device="cuda")
        self.depth_last_conflict_uid = torch.empty(0, dtype=torch.int32, device="cuda")
        # Four-state lifecycle: 0=PROVISIONAL, 1=YOUNG, 2=MATURE, 3=FROZEN
        self.lifecycle_state = torch.empty(0, dtype=torch.int8, device="cuda")
        # Per-Gaussian utility EMA (rolling average of energy_mcmc utility score)
        self.utility_ema = torch.empty(0, dtype=torch.float32, device="cuda")
        # Anchors snapshotted at YOUNG->MATURE transition for anti-fade losses
        self.anchor_scale_log = torch.empty((0, 3), dtype=torch.float32, device="cuda")
        # Phase 12: submap metadata
        self.submap_id = torch.empty(0, dtype=torch.int32, device="cuda")
        self.anchor_keyframe_id = torch.empty(0, dtype=torch.int32, device="cuda")
        self.anchor_opacity_logit = torch.empty(0, dtype=torch.float32, device="cuda")
        # Persistent Gaussian identity (Frame/Gaussian observation DB)
        self.gaussian_uid = torch.empty(0, dtype=torch.int64, device="cuda")
        self._next_gaussian_uid = 0
        # Immutable SEED-cohort marker (issue #283). The contiguous
        # [lo, hi) gaussian_uid block assigned to the BOOTSTRAP population, or
        # None when it cannot be determined exactly (PLY load, pre-#283
        # checkpoint, mid-run re-bootstrap). Deliberately a scalar pair rather
        # than a per-Gaussian buffer: uids are only ever created, mask-selected
        # or concatenated -- never remapped -- so a range over them is
        # index-shift-safe for free and needs no new handling at any mutation
        # site. `_invalidated` makes the "undetermined" state sticky.
        self._seed_uid_range = None
        self._seed_uid_range_reason = "no bootstrap has run on this model yet"
        self._seed_uid_range_invalidated = False
        # Per-Gaussian lineage (see utils/gaussian_lineage.py). Allocated here,
        # not only in _reset_running_state, so the class invariant "these five
        # buffers always exist" holds from construction like gaussian_uid's does
        # -- _commit_new_gaussians reads getattr(self, name).shape[0] with no
        # default and would raise AttributeError on a model that never reset.
        self.parent_uid = torch.empty(0, dtype=torch.int64, device="cuda")
        self.root_uid = torch.empty(0, dtype=torch.int64, device="cuda")
        self.generation = torch.empty(0, dtype=torch.int32, device="cuda")
        self.birth_method = torch.empty(0, dtype=torch.int8, device="cuda")
        self.last_reseed_iter = torch.empty(0, dtype=torch.int32, device="cuda")
        # Geometry-prior buffers (parity with legacy GaussianModel; populated by
        # _reset_running_state / add_points_as_gaussians). anchor_normal is the
        # depth-derived surface normal (world frame), surface_state uses
        # geometry.priors.SurfaceState semantics.
        self.anchor_normal = torch.empty((0, 3), dtype=torch.float32, device="cuda")
        self.anchor_confidence = torch.empty(0, dtype=torch.float32, device="cuda")
        self.geometry_source = torch.empty(0, dtype=torch.int16, device="cuda")
        self.surface_state = torch.empty(0, dtype=torch.int8, device="cuda")
        # GaussianBelief (issue #1365 Phase 1 -- see utils/gaussian_belief.py). Additive
        # only this phase: nothing reads these for any decision yet.
        init_belief_buffers(self, device="cuda")
        # P3 amortized growth (staged insertion). Off unless training_setup enables it
        # from args.staged_gaussian_insert AND a staging session is open. See
        # begin_staged_inserts / flush_staged_inserts.
        self.staged_gaussian_insert = False
        self._staging_active = False
        self._staged_tensors: list = []
        self._staged_buffers: list = []
        self._staged_count: int = 0
        # Set only by the canonical gsplat map adapter once live population
        # ownership crosses its complete plan-snapshot fence.  Legacy and
        # compatibility callers intentionally retain their existing writers.
        self._population_plan_ownership_enabled = False

    @property
    def _xyz(self):
        return self.params["means"]

    @property
    def _scaling(self):
        return self.params["scales"]

    @property
    def _rotation(self):
        return self.params["quats"]

    @property
    def _opacity(self):
        return self.params["opacities"].unsqueeze(-1)

    @property
    def _features_dc(self):
        return self.params["sh0"]

    @property
    def _features_rest(self):
        return self.params["shN"]

    # ------------------------------------------------------------------
    # Layout-agnostic adapter properties — streaming code uses these so
    # it doesn't need to branch on model_layout.
    # ------------------------------------------------------------------

    @property
    def raw_opacities(self):
        """Raw (pre-sigmoid) opacity logits, shape (N,)."""
        return self.params["opacities"]

    @property
    def raw_scales(self):
        """Raw (log) scale parameters, shape (N, 3)."""
        return self.params["scales"]

    @property
    def means_param(self):
        """The means / xyz position parameter."""
        return self.params["means"]

    @property
    def geometry_params(self):
        """Geometric learnable parameters [means, scales, quats]."""
        return [self.params["means"], self.params["scales"], self.params["quats"]]

    @property
    def all_learnable_params(self):
        """All learnable parameters as a list."""
        return list(self.params.values())

    @property
    def get_xyz(self):
        return self.params["means"]

    @property
    def get_scaling(self):
        return torch.exp(self.params["scales"])

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self.params["quats"], dim=1, eps=1e-8)

    @property
    def get_opacity(self):
        return torch.sigmoid(self.params["opacities"]).unsqueeze(-1)

    @property
    def get_features(self):
        return torch.cat((self.params["sh0"], self.params["shN"]), dim=1)

    def as_gsplat_params(self, activated=True):
        if activated:
            return {
                "means": self.params["means"],
                "scales": self.get_scaling,
                "quats": self.get_rotation,
                "opacities": torch.sigmoid(self.params["opacities"]),
                "sh0": self.params["sh0"],
                "shN": self.params["shN"],
            }
        return self.params

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd, spatial_lr_scale: float, init_scale_mode="fixed", init_scale=0.01, voxel_size=0.02):
        import os as _os, time as _t
        _prof = _os.environ.get("SPLAT_PROFILE_BOOTSTRAP", "0") == "1"
        def _mark(_lbl, _t0):
            if _prof:
                torch.cuda.synchronize(); print(f"[prof:create_from_pcd] {_lbl}: {(_t.perf_counter()-_t0)*1e3:.1f}ms", flush=True)
            return _t.perf_counter()
        _tp = _t.perf_counter()
        self.spatial_lr_scale = spatial_lr_scale
        points_np = np.asarray(pcd.points)
        colors_np = np.asarray(pcd.colors)
        _tp = _mark("np_asarray", _tp)
        fused_point_cloud = torch.tensor(points_np, dtype=torch.float32, device="cuda")
        fused_color = RGB2SH(torch.tensor(colors_np, dtype=torch.float32, device="cuda"))
        _tp = _mark("to_cuda (1st GPU op / warmup)", _tp)
        num_sh = (self.max_sh_degree + 1) ** 2
        features = torch.zeros((fused_color.shape[0], num_sh, 3), dtype=torch.float32, device="cuda")
        features[:, 0, :] = fused_color

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        N = fused_point_cloud.shape[0]
        if init_scale_mode == "fixed":
            dist2 = torch.full((N,), init_scale ** 2, device="cuda")
        elif init_scale_mode == "knn":
            # distCUDA2 (utils/rocm_knn_fallback.py) already handles both devices --
            # it only uses the compiled kernel when the input is CUDA-resident and
            # otherwise runs its pure-torch fallback, so the input should follow
            # fused_point_cloud's existing device rather than force .cuda()
            # unconditionally here.
            dist2 = torch.clamp_min(
                distCUDA2(torch.from_numpy(points_np).float().to(fused_point_cloud.device)),
                0.0000001,
            )
        elif init_scale_mode == "voxel":
            dist2 = torch.full((N,), (voxel_size * 0.5) ** 2, device="cuda")
        else:
            raise ValueError(f"Unknown init_scale_mode: {init_scale_mode}")
        scales = torch.log(torch.sqrt(dist2) * 0.1)[..., None].repeat(1, 3)
        quats = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        quats[:, 0] = 1
        opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0],), dtype=torch.float32, device="cuda"))

        _tp = _mark(f"scale_init mode={init_scale_mode}", _tp)
        self.params = nn.ParameterDict({
            "means": nn.Parameter(fused_point_cloud.requires_grad_(True)),
            "scales": nn.Parameter(scales.requires_grad_(True)),
            "quats": nn.Parameter(quats.requires_grad_(True)),
            "opacities": nn.Parameter(opacities.requires_grad_(True)),
            "sh0": nn.Parameter(features[:, :1, :].contiguous().requires_grad_(True)),
            "shN": nn.Parameter(features[:, 1:, :].contiguous().requires_grad_(True)),
        })
        _tp = _mark("ParameterDict", _tp)
        self._reset_running_state()
        _tp = _mark("_reset_running_state", _tp)

    def initialize_from_prepared_device_seed(
        self,
        seed,
        spatial_lr_scale: float,
        init_scale_mode="fixed",
        init_scale=0.01,
        voxel_size=0.02,
    ):
        """Initialize directly from device-native RGB-D backprojection tensors.

        Unlike :meth:`create_from_pcd`, this path never accepts a CPU
        ``BasicPointCloud`` and never reconstructs host arrays before upload.
        It is used only by the explicit prepared-camera bootstrap seam.
        """

        points = getattr(seed, "points", None)
        colors = getattr(seed, "colors", None)
        if not isinstance(points, torch.Tensor) or not isinstance(colors, torch.Tensor):
            raise TypeError("prepared device seed requires tensor points and colors")
        if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
            raise ValueError("prepared device seed points and colors must be Nx3")
        if points.shape[0] == 0 or points.dtype != torch.float32 or colors.dtype != torch.float32:
            raise ValueError("prepared device seed must be non-empty float32 tensors")
        if points.device != colors.device:
            raise ValueError("prepared device seed tensors must share one device")
        self.spatial_lr_scale = float(spatial_lr_scale)
        fused_point_cloud = points
        fused_color = RGB2SH(colors)
        num_sh = (self.max_sh_degree + 1) ** 2
        features = torch.zeros(
            (fused_color.shape[0], num_sh, 3), dtype=torch.float32, device=points.device
        )
        features[:, 0, :] = fused_color
        count = fused_point_cloud.shape[0]
        if init_scale_mode in {"fixed", "constant"}:
            dist2 = torch.full((count,), float(init_scale) ** 2, device=points.device)
        elif init_scale_mode == "knn":
            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        elif init_scale_mode == "voxel":
            dist2 = torch.full((count,), (float(voxel_size) * 0.5) ** 2, device=points.device)
        else:
            raise ValueError(f"Unknown init_scale_mode: {init_scale_mode}")
        scales = torch.log(torch.sqrt(dist2) * 0.1)[..., None].repeat(1, 3)
        quats = torch.zeros((count, 4), device=points.device)
        quats[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5 * torch.ones((count,), dtype=torch.float32, device=points.device)
        )
        self.params = nn.ParameterDict({
            "means": nn.Parameter(fused_point_cloud.requires_grad_(True)),
            "scales": nn.Parameter(scales.requires_grad_(True)),
            "quats": nn.Parameter(quats.requires_grad_(True)),
            "opacities": nn.Parameter(opacities.requires_grad_(True)),
            "sh0": nn.Parameter(features[:, :1, :].contiguous().requires_grad_(True)),
            "shN": nn.Parameter(features[:, 1:, :].contiguous().requires_grad_(True)),
        })
        self._reset_running_state()

    def training_setup(
        self,
        training_args,
        preserve_lineage: bool = False,
        *,
        lineage_site: str = "",
        iteration: "int | None" = None,
    ):
        """Build the optimizers/LR schedule and reset per-Gaussian running state.

        ``preserve_lineage=True`` (issue #374) keeps the CURRENT population's
        birth/identity provenance across the ``_reset_running_state()`` call
        below, for the callers that invoke ``training_setup()`` mid-run on an
        already-populated model (``rolling_seed_application`` /
        ``submap_stitch_application``, both of which re-enter here purely to
        rebuild optimizers after inserting points). Without it the reset
        re-stamps every existing Gaussian as a freshly-born BOOTSTRAP root and
        hands out a brand-new uid block, silently destroying the lineage the
        cohort machinery (#283/#288) is defined in terms of.

        Default is ``False``: the fresh-bootstrap path (``train.py`` /
        ``train_streaming.py`` / ``restore()``) is bit-identical to before this
        flag existed -- ``_reset_running_state()`` itself is untouched.

        ``lineage_site``/``iteration`` (issue #374 item 3, follow-up to #377):
        caller-supplied identity for the event record appended to
        ``_lineage_preserving_setup_events`` -- ``training_setup()`` itself has
        no notion of a training iteration (it is called before either
        ``rolling_seed``/``submap_stitch``'s global-refinement loop starts), so
        the call site is the one that knows it. Both current call sites pass
        ``iteration=0``, matching the file-level convention already used for
        ``birth_iter=0`` on the same pre-refinement inserts ("literal, not a
        placeholder"). Ignored when ``preserve_lineage=False``.
        """
        self.percent_dense = training_args.percent_dense
        self.staged_gaussian_insert = bool(getattr(training_args, "staged_gaussian_insert", False))
        self.optimizer_type = getattr(training_args, "optimizer_type", "adam").lower()
        if self.optimizer_type == "default":
            self.optimizer_type = "adam"

        lr_by_name = {
            "means": training_args.position_lr_init * self.spatial_lr_scale,
            "scales": training_args.scaling_lr,
            "quats": training_args.rotation_lr,
            "opacities": training_args.opacity_lr,
            "sh0": training_args.feature_lr,
            "shN": training_args.feature_lr / 20.0,
        }

        if self.optimizer_type == "selective_adam":
            try:
                from gsplat.optimizers import SelectiveAdam
            except Exception as exc:
                raise RuntimeError(
                    "--optimizer_type selective_adam requires gsplat.optimizers.SelectiveAdam "
                    "inside the project container."
                ) from exc
            optimizer_cls = SelectiveAdam
            kwargs = {"eps": 1e-15, "betas": (0.9, 0.999)}
        elif self.optimizer_type == "adam":
            optimizer_cls = torch.optim.Adam
            kwargs = {"lr": 0.0, "eps": 1e-15}
        else:
            raise ValueError(
                f"Unsupported optimizer_type '{self.optimizer_type}'. "
                "Expected 'adam' or 'selective_adam'."
            )

        self.optimizers = {}
        for name, lr in lr_by_name.items():
            param_group = {"params": [self.params[name]], "lr": lr, "name": name}
            self.optimizers[name] = optimizer_cls([param_group], **kwargs)
        self.optimizer = OptimizerDictProxy(self.optimizers)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        _snapshot = self._snapshot_lineage() if preserve_lineage else None
        self._reset_running_state()
        if _snapshot is not None:
            self._restore_lineage_snapshot(_snapshot, site=lineage_site, iteration=iteration)

    # Per-Gaussian provenance preserved across a mid-run training_setup()
    # (issue #374). gaussian_uid is the identity every lineage field points at;
    # birth_iter/birth_frame are the birth timestamps the cohort machinery reads;
    # submap_id is the per-row submap provenance apply_submap_correction consumes
    # (the reset flattens it to _current_submap_id for the whole population).
    # Deliberately NOT preserved: everything else _reset_running_state() zeroes
    # (visibility_ema, lifecycle_state, utility_ema, support_count, anchors, ...)
    # -- those are running/adaptive state, not provenance, and preserving them
    # would be a much larger behavioral change than this fix is scoped to.
    _LINEAGE_PRESERVE_NAMES: tuple = (
        "gaussian_uid", "birth_iter", "birth_frame", "submap_id",
        "parent_uid", "root_uid", "generation", "birth_method", "last_reseed_iter",
    )

    def _snapshot_lineage(self) -> dict:
        """Clone the lineage buffers that currently match the population size."""
        n = self.get_xyz.shape[0] if len(self.params) else 0
        snap = {"_n": n, "_next_gaussian_uid": int(self._next_gaussian_uid), "buffers": {}}
        for name in self._LINEAGE_PRESERVE_NAMES:
            buf = getattr(self, name, None)
            if buf is not None and buf.shape[0] == n:
                snap["buffers"][name] = buf.clone()
        return snap

    def _restore_lineage_snapshot(
        self, snap: dict, *, site: str = "", iteration: "int | None" = None,
    ) -> None:
        """Re-apply a ``_snapshot_lineage()`` result after ``_reset_running_state()``.

        ``_reset_running_state()`` never changes the population size, so a
        size mismatch here means the snapshot itself was stale -- in that case
        the freshly-reset default is kept for that buffer and the skip is
        reported, rather than restoring a buffer that doesn't line up with the
        live rows.
        """
        n = self.get_xyz.shape[0] if len(self.params) else 0
        # gaussian_uid + the five lineage fields are ATOMIC: the lineage fields
        # point into gaussian_uid's space, so restoring them while keeping the
        # reset's throwaway `arange` uids would leave root_uid/parent_uid
        # referencing uids present nowhere in the model -- shapes match, nothing
        # raises, exactly the silent corruption _ensure_lineage_after_restore()
        # refuses for the same reason. If gaussian_uid isn't restorable, keep the
        # reset's freshly-seeded (coherent) lineage for the whole group.
        uid_group = ("gaussian_uid",) + self._LINEAGE_BUFFER_NAMES
        uid_snap = snap["buffers"].get("gaussian_uid")
        uid_group_ok = uid_snap is not None and uid_snap.shape[0] == n
        restored, skipped = [], []
        for name, buf in snap["buffers"].items():
            if name in uid_group and not uid_group_ok:
                skipped.append((name, int(buf.shape[0]), "uid-group withheld"))
                continue
            if buf.shape[0] == n:
                setattr(self, name, buf)
                restored.append(name)
            else:
                skipped.append((name, int(buf.shape[0]), "stale length"))
        if "gaussian_uid" in restored:
            # The reset handed out a throwaway uid block that no live row now
            # holds; rewinding the counter to its pre-reset value is safe (and
            # keeps uids dense) precisely because that block was discarded.
            self._next_gaussian_uid = int(snap["_next_gaussian_uid"])
            uid = self.gaussian_uid
            assert uid.numel() == 0 or int(uid.max().item()) < self._next_gaussian_uid, (
                "preserve_lineage left a live gaussian_uid >= _next_gaussian_uid "
                "(future inserts would collide)"
            )
        self._lineage_preserving_setup_count = int(
            getattr(self, "_lineage_preserving_setup_count", 0)
        ) + 1
        # Per-event record (issue #374 item 3, follow-up to #377): the plain
        # counter alone doesn't say WHEN a lineage-preserving setup happened.
        # ``iteration`` is caller-supplied (see training_setup's docstring for
        # why it can't be known here); ``None`` means the caller didn't pass
        # one. Consumed by write_post_training_report's
        # ``lineage_preserving_setup`` report block when a run's normal report
        # path is reached -- today neither real call site (rolling_seed /
        # submap_stitch) is, so this list is the durable record for them via
        # this same train.log line.
        if not hasattr(self, "_lineage_preserving_setup_events"):
            self._lineage_preserving_setup_events: list = []
        self._lineage_preserving_setup_events.append({
            "setup_index": self._lineage_preserving_setup_count,
            "iteration": iteration,
            "site": site or None,
            "n_gaussians": n,
            "restored": list(restored),
            "skipped": [name for name, _len, _reason in skipped],
            "next_gaussian_uid": int(self._next_gaussian_uid),
        })
        print(
            f"[lineage] training_setup(preserve_lineage=True) #"
            f"{self._lineage_preserving_setup_count}"
            + (f" iteration={iteration}" if iteration is not None else "")
            + (f" site={site}" if site else "")
            + f": preserved {len(restored)} "
            f"provenance buffer(s) over {n} Gaussians "
            f"(next_gaussian_uid={self._next_gaussian_uid})"
            + (f"; SKIPPED stale-length {skipped}" if skipped else ""),
            flush=True,
        )

    def update_learning_rate(self, iteration):
        lr = self.xyz_scheduler_args(iteration)
        self.optimizers["means"].param_groups[0]["lr"] = lr
        return lr

    def normalize_rotation_params(self, mask=None):
        with torch.no_grad():
            quats = self.params["quats"]
            normalized = torch.nn.functional.normalize(quats, dim=1, eps=1e-8)
            if mask is None:
                quats.copy_(normalized)
            elif getattr(mask, "dtype", None) is torch.bool:
                # PERF (issue #1259): `quats[mask] = ...` is boolean advanced
                # indexing, which lowers to nonzero() -- a data-dependent output
                # shape, so the host must learn the selected count and the call
                # becomes a DEVICE->HOST SYNC that drains the whole kernel queue.
                # This runs once per optimizer step on the sparse-grad path
                # (eligsir/optimization/gaussian/engine.py's run_optimizer_step
                # passes the boolean `step_mask`), so it forced a full pipeline
                # flush every iteration and destroyed CPU/GPU overlap.
                #
                # torch.where over the full tensor is numerically identical
                # (rows outside the mask keep their exact prior values, and
                # normalize is elementwise per row so masked rows get the same
                # value they would from the gather/scatter form) but has a
                # statically-known shape and therefore issues no sync.
                quats.copy_(torch.where(mask.unsqueeze(1), normalized, quats))
            else:
                # Index-tensor masks already have a host-known shape; keep the
                # gather/scatter form so only the selected rows are written.
                quats[mask] = normalized[mask]

    def prepare_selective_adam_step(self, allow_dense_grads=False):
        for optimizer in self.optimizers.values():
            for group in optimizer.param_groups:
                param = group["params"][0]
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
                if param.grad is not None:
                    if getattr(param.grad, "layout", torch.strided) != torch.strided:
                        param.grad = param.grad.to_dense().contiguous()
                    elif not param.grad.is_contiguous():
                        param.grad = param.grad.contiguous()
                stored_state = optimizer.state.get(param, None)
                if stored_state is not None:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in stored_state and not stored_state[key].is_contiguous():
                            stored_state[key] = stored_state[key].contiguous()

    # Names of streaming lifecycle buffers saved in capture() and restored in restore().
    # Kept as a class constant so capture/restore and _sync_streaming_state_lengths all
    # reference the same list.
    _STREAMING_BUFFER_NAMES: tuple = (
        "birth_frame", "birth_iter", "support_count", "provisional", "anchor_iter",
        "depth_conflict_count", "depth_last_support_uid", "depth_last_conflict_uid",
        "lifecycle_state", "utility_ema", "anchor_scale_log", "anchor_opacity_logit",
        "anchor_xyz",
        "anchor_normal", "anchor_confidence", "geometry_source", "surface_state",
        "submap_id", "anchor_keyframe_id", "gaussian_uid",  # Phase 12 + observation DB
        "parent_uid", "root_uid", "generation", "birth_method", "last_reseed_iter",  # Lineage
        # Part C: floater consensus (only allocated when flag is on; capture() skips
        # missing attrs so flag-off checkpoints are unchanged).
        "seen_count", "consensus_support_count", "consensus_conflict_count",
        "consensus_last_seen_uid",
        # Part D1: per-Gaussian last parallax voter center ([N,3], only allocated
        # when the baseline gate is on; capture() skips missing attrs).
        "consensus_last_voter_center",
        # Recency-of-supervision opacity-reg relaxation: iteration this Gaussian
        # was last visible in a training batch. Only allocated when
        # --streaming_recency_reg_horizon > 0 (capture() skips missing attrs).
        "last_supervised_iter",
        # Resolution-exposure eligibility (issue #1262). Only allocated when
        # --streaming_resolution_exposure_veto/_telemetry is on; capture() skips
        # missing attrs so flag-off checkpoints are unchanged.
        "exposure_count", "exposure_last_view_uid", "supervised_tier_high_water",
        # Live population-policy E0 evidence.  Allocated only by the live
        # profile; these counters never alias resolution-veto/lifecycle state.
        "population_policy_exposure_count", "population_policy_support_count",
        "population_policy_source_seen",
    ) + BELIEF_BUFFER_NAMES  # GaussianBelief (issue #1365 Phase 1), always allocated.

    _ROW_STATE_PARAM_NAMES: tuple = (
        "means", "scales", "quats", "opacities", "sh0", "shN",
    )
    _ROW_STATE_RENDER_BUFFER_NAMES: tuple = (
        "max_radii2D", "xyz_gradient_accum", "denom", "visibility_ema",
    )
    _ROW_STATE_SURFACE_BUFFER_NAMES: tuple = (
        "anchor_normal", "anchor_confidence", "geometry_source", "surface_state",
    )
    _ROW_STATE_COMPATIBILITY_BUFFER_NAMES: tuple = (
        "submap_id", "anchor_keyframe_id", "seen_count", "consensus_support_count",
        "consensus_conflict_count", "consensus_last_seen_uid",
        "consensus_last_voter_center", "last_supervised_iter", "exposure_count",
        "exposure_last_view_uid", "supervised_tier_high_water",
        "population_policy_exposure_count", "population_policy_support_count",
        "population_policy_source_seen",
    )
    _ROW_STATE_OPTIONAL_COMPATIBILITY_BUFFER_NAMES: frozenset = frozenset(
        _ROW_STATE_COMPATIBILITY_BUFFER_NAMES[2:]
    )
    _ROW_STATE_REQUIRED_BUFFER_NAMES: tuple = (
        _ROW_STATE_RENDER_BUFFER_NAMES
        + _STREAMING_BUFFER_NAMES[:_STREAMING_BUFFER_NAMES.index("seen_count")]
        + BELIEF_BUFFER_NAMES
    )
    # family, checkpoint, default, create, grow, relocate, retire. This table is
    # deliberately exhaustive: a new row-aligned buffer must declare its own
    # lifecycle before a topology transform can claim complete coverage.
    _ROW_STATE_BUFFER_DESCRIPTOR_TABLE: dict = {
        "max_radii2D": ("render", False, "zero", "zero", "zero", "reset", "select_survivors"),
        "xyz_gradient_accum": ("render", False, "zero", "zero", "zero", "reset", "select_survivors"),
        "denom": ("render", False, "zero", "zero", "zero", "reset", "select_survivors"),
        "visibility_ema": ("render", True, "zero", "zero", "zero", "reset", "select_survivors"),
        "birth_frame": ("provenance", True, "zero", "creation_frame", "creation_frame", "relocation_frame", "select_survivors"),
        "birth_iter": ("provenance", True, "zero", "creation_iteration", "creation_iteration", "current_iteration", "select_survivors"),
        "support_count": ("provenance", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "provisional": ("provenance", True, "false", "initial_provisional", "initial_provisional", "initial_provisional", "select_survivors"),
        "anchor_xyz": ("provenance", True, "birth_position", "birth_position", "donor_anchor", "donor_anchor", "select_survivors"),
        "anchor_iter": ("provenance", True, "zero", "creation_iteration", "creation_iteration", "current_iteration", "select_survivors"),
        "depth_conflict_count": ("provenance", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "depth_last_support_uid": ("provenance", True, "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "select_survivors"),
        "depth_last_conflict_uid": ("provenance", True, "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "select_survivors"),
        "lifecycle_state": ("provenance", True, "young", "initial_lifecycle_state", "initial_lifecycle_state", "relocation_lifecycle_state", "select_survivors"),
        "utility_ema": ("provenance", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "anchor_scale_log": ("provenance", True, "birth_scale", "birth_scale", "donor_anchor", "donor_anchor", "select_survivors"),
        "anchor_opacity_logit": ("provenance", True, "birth_opacity", "birth_opacity", "donor_anchor", "donor_anchor", "select_survivors"),
        "anchor_normal": ("surface", True, "unit_z", "surface_normal_or_unit_z", "donor_normal", "donor_normal", "select_survivors"),
        "anchor_confidence": ("surface", True, "zero", "input_confidence_or_zero", "zero", "zero", "select_survivors"),
        "geometry_source": ("surface", True, "zero", "input_source_or_zero", "zero", "zero", "select_survivors"),
        "surface_state": ("surface", True, "unclassified", "input_state_or_lifecycle_state", "zero", "zero", "select_survivors"),
        "submap_id": ("compatibility", True, "current_submap_id", "current_submap_id", "current_submap_id", "strategy_defined", "select_survivors"),
        "anchor_keyframe_id": ("compatibility", True, "birth_frame", "birth_frame", "birth_frame", "strategy_defined", "select_survivors"),
        "gaussian_uid": ("provenance", True, "fresh_monotonic_uid", "fresh_uid", "fresh_uid", "mint_fresh_hypothesis_uid", "select_survivors"),
        "parent_uid": ("provenance", True, "no_parent_sentinel", "root_parent_sentinel", "donor_gaussian_uid", "donor_gaussian_uid", "select_survivors"),
        "root_uid": ("provenance", True, "own_gaussian_uid", "own_gaussian_uid", "donor_root_uid", "donor_root_uid", "select_survivors"),
        "generation": ("provenance", True, "zero", "zero", "donor_generation_plus_one", "donor_generation_plus_one", "select_survivors"),
        "birth_method": ("provenance", True, "bootstrap_method", "birth_method", "mcmc_grow_method", "mcmc_reseed_method", "select_survivors"),
        "last_reseed_iter": ("provenance", True, "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "current_iteration", "select_survivors"),
        "seen_count": ("compatibility", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "consensus_support_count": ("compatibility", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "consensus_conflict_count": ("compatibility", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "consensus_last_seen_uid": ("compatibility", True, "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "select_survivors"),
        "consensus_last_voter_center": ("compatibility", True, "nan_vector", "nan_vector", "nan_vector", "nan_vector", "select_survivors"),
        "last_supervised_iter": ("compatibility", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "exposure_count": ("compatibility", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "exposure_last_view_uid": ("compatibility", True, "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "select_survivors"),
        "supervised_tier_high_water": ("compatibility", True, "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "select_survivors"),
        "population_policy_exposure_count": ("belief", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "population_policy_support_count": ("belief", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "population_policy_source_seen": ("belief", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "birth_anchor_xyz": ("belief", True, "birth_position", "birth_position", "donor_anchor", "donor_anchor", "select_survivors"),
        "baseline_diversity": ("belief", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "last_exposed_update": ("belief", True, "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "select_survivors"),
        "support_evidence": ("belief", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "contradiction_evidence": ("belief", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "p_exist": ("belief", True, "one", "one", "one", "one", "select_survivors"),
        "surface_anchor_id": ("belief", True, "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "select_survivors"),
        "surface_position_uncertainty": ("belief", True, "zero", "zero", "donor_value", "donor_value", "select_survivors"),
        "surface_normal_uncertainty": ("belief", True, "zero", "zero", "donor_value", "donor_value", "select_survivors"),
        "rgb_residual_ema": ("belief", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "depth_residual_ema": ("belief", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "redundancy": ("belief", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "expected_marginal_gain": ("belief", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "change_probability": ("belief", True, "zero", "zero", "zero", "zero", "select_survivors"),
        "evidence_recency": ("belief", True, "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "sentinel_minus_one", "select_survivors"),
        "confidence_decay": ("belief", True, "zero", "zero", "zero", "zero", "select_survivors"),
    }

    def build_row_state_registry(self) -> GaussianRowStateRegistry:
        """Describe and validate every live population-aligned gsplat field.

        This is deliberately read-only. Existing topology operations continue to
        own their sidecar synchronization until they are migrated to consume the
        registry's transform contract.
        """
        params = getattr(self, "params", None)
        if params is None:
            raise ValueError("row-state params are missing")
        missing_params = [name for name in GsplatGaussianModel._ROW_STATE_PARAM_NAMES if name not in params]
        if missing_params:
            raise ValueError(f"row-state params missing required fields: {missing_params}")
        extra_params = sorted(set(params) - set(GsplatGaussianModel._ROW_STATE_PARAM_NAMES))
        if extra_params:
            raise ValueError(f"row-state params have unregistered fields: {extra_params}")
        population = int(params["means"].shape[0])
        descriptors = []

        def register_tensor(name, family, tensor, *, checkpoint, default, create, grow, relocate, retire="select_survivors"):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"row-state field {name} is missing or not a tensor")
            if tensor.ndim == 0 or tensor.shape[0] != population:
                rows = 0 if tensor.ndim == 0 else tensor.shape[0]
                raise ValueError(
                    f"row-state field {name} has rows={rows}; population={population}"
                )
            descriptors.append(RowStateDescriptor(
                name=name,
                family=family,
                owner="GsplatGaussianModel",
                shape_suffix=tuple(tensor.shape[1:]),
                dtype=tensor.dtype,
                checkpoint=checkpoint,
                default=default,
                create=create,
                grow=grow,
                relocate=relocate,
                retire=retire,
            ))

        for name in GsplatGaussianModel._ROW_STATE_PARAM_NAMES:
            register_tensor(
                f"params.{name}", "params", params[name], checkpoint=True,
                default="initializer", create="initialize", grow="initialize",
                relocate="strategy_defined",
            )

        optimizers = getattr(self, "optimizers", None)
        if not isinstance(optimizers, dict):
            raise ValueError("row-state optimizers are missing")
        extra_optimizers = sorted(set(optimizers) - set(GsplatGaussianModel._ROW_STATE_PARAM_NAMES))
        if extra_optimizers:
            raise ValueError(f"row-state optimizers have unregistered fields: {extra_optimizers}")
        for name in GsplatGaussianModel._ROW_STATE_PARAM_NAMES:
            optimizer = optimizers.get(name)
            if optimizer is None:
                raise ValueError(f"row-state field optimizer.{name} is missing")
            groups = getattr(optimizer, "param_groups", ())
            group_params = groups[0].get("params", ()) if len(groups) == 1 else ()
            if len(groups) != 1 or len(group_params) != 1 or group_params[0] is not params[name]:
                raise ValueError(f"row-state field optimizer.{name} is not attached to params.{name}")
            # Optimizer state is lazy: before the first step there are no
            # row-aligned moment tensors to transform. Scalar entries (such as
            # Adam's ``step``) are global bookkeeping, not population state.
            for state_name, value in getattr(optimizer, "state", {}).get(params[name], {}).items():
                if not isinstance(value, torch.Tensor) or value.ndim == 0:
                    continue
                if value.shape[0] != population:
                    raise ValueError(
                        f"row-state field optimizer.{name}.{state_name} has rows={value.shape[0]}; "
                        f"population={population}"
                    )
                register_tensor(
                    f"optimizer.{name}.{state_name}", "optimizer", value, checkpoint=True,
                    default="zero_moments", create="zero_moments", grow="zero_moments",
                    relocate="strategy_defined",
                )

        required = set(GsplatGaussianModel._ROW_STATE_REQUIRED_BUFFER_NAMES)
        live_buffers = list(GsplatGaussianModel._ROW_STATE_REQUIRED_BUFFER_NAMES)
        live_buffers.extend(
            name for name in GsplatGaussianModel._ROW_STATE_OPTIONAL_COMPATIBILITY_BUFFER_NAMES
            if getattr(self, name, None) is not None
        )
        for name in live_buffers:
            metadata = GsplatGaussianModel._ROW_STATE_BUFFER_DESCRIPTOR_TABLE.get(name)
            if metadata is None:
                raise ValueError(f"canonical descriptor table missing required field: {name}")
            family, checkpoint, default, create, grow, relocate, retire = metadata
            value = getattr(self, name, None)
            if value is None and name in required:
                raise ValueError(f"row-state field {family}.{name} is missing")
            register_tensor(
                f"{family}.{name}", family, value,
                checkpoint=checkpoint,
                default=default, create=create, grow=grow, relocate=relocate,
                retire=retire,
            )
        registered_buffer_names = set(live_buffers)
        for name, value in vars(self).items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == population
                and name not in registered_buffer_names
            ):
                raise ValueError(f"unregistered live row-aligned tensor: {name}")
        return GaussianRowStateRegistry(descriptors)

    def attest_live_population_rows(self) -> GaussianRowStateRegistry:
        """Verify that every registered live row family has one population.

        This is the post-plan safety check.  ``build_row_state_registry``
        validates parameters, optimizer moments, UID/lineage, belief, and
        anchor buffers against ``params.means`` without relying on a copied
        proposal inventory.
        """

        try:
            registry = self.build_row_state_registry()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"live population row attestation failed: {exc}") from exc
        families = {descriptor.family for descriptor in registry.descriptors}
        required_families = {"params", "optimizer", "belief", "provenance", "surface"}
        if not required_families.issubset(families):
            missing = sorted(required_families - families)
            raise ValueError(
                "live population row attestation failed: missing registry families "
                f"{missing}"
            )
        return registry

    def _reject_legacy_population_writer(self, writer: str) -> None:
        """Fail closed once the live plan transaction owns population rows."""

        if getattr(self, "_population_plan_ownership_enabled", False):
            raise RuntimeError(
                "live population ownership requires PopulationMutationPlan; "
                f"legacy writer {writer} is unavailable"
            )

    def capture(self):
        streaming_state = {}
        for name in self._STREAMING_BUFFER_NAMES:
            buf = getattr(self, name, None)
            if buf is not None:
                streaming_state[name] = buf
        return _immutable_tensor_snapshot({
            "layout": "gsplat",
            "active_sh_degree": self.active_sh_degree,
            "next_gaussian_uid": self._next_gaussian_uid,
            # Issue #283: None when undetermined -- restore() treats that as
            # "unavailable", exactly like a checkpoint predating this key.
            "seed_uid_range": self._seed_uid_range,
            "params": {name: param.detach() for name, param in self.params.items()},
            "optimizers": self.optimizer.state_dict(),
            "spatial_lr_scale": self.spatial_lr_scale,
            "visibility_ema": self.visibility_ema,
            "streaming_state": streaming_state,
        })

    def restore(self, model_args, training_args):
        if not isinstance(model_args, dict) or model_args.get("layout") != "gsplat":
            raise ValueError("GsplatGaussianModel can only restore gsplat-layout checkpoints.")
        self.active_sh_degree = model_args["active_sh_degree"]
        self.spatial_lr_scale = model_args["spatial_lr_scale"]
        live_params = getattr(self, "params", {})
        restore_target = (
            live_params["means"].device
            if "means" in live_params
            else model_args["params"]["means"].device
        )
        self.params = nn.ParameterDict({
            name: nn.Parameter(
                tensor.detach().to(device=restore_target).requires_grad_(True)
            )
            for name, tensor in model_args["params"].items()
        })
        self.training_setup(training_args)
        self.optimizer.load_state_dict(model_args.get("optimizers", {}))
        saved_visibility_ema = model_args.get("visibility_ema")
        self.visibility_ema = (
            saved_visibility_ema.to(device=restore_target)
            if saved_visibility_ema is not None
            else torch.zeros((self.get_xyz.shape[0], 1), device=restore_target)
        )
        streaming_state = model_args.get("streaming_state", {})
        _uid_skipped = False
        _lineage_reseeded = False
        if streaming_state:
            expected_n = self.get_xyz.shape[0]
            skipped = []
            for name, buf in streaming_state.items():
                if buf.shape[0] != expected_n:
                    # Checkpoint's own N grew via a strategy that mutates the
                    # Gaussian population without extending our per-Gaussian
                    # tracking buffers (e.g. gsplat's own DefaultStrategy
                    # clone/split ADC). Applying this stale-sized buffer would
                    # silently desync it from every other buffer restored/reset
                    # to `expected_n` -- keep _reset_running_state()'s freshly
                    # allocated default instead (already sized to expected_n).
                    skipped.append((name, buf.shape[0]))
                    continue
                setattr(self, name, buf.to(device=restore_target))
            if skipped:
                print(
                    "[checkpoint] Warning: skipped restoring "
                    f"{len(skipped)} streaming buffer(s) whose saved size doesn't "
                    f"match the current Gaussian count ({expected_n}) -- likely grown "
                    "by an external strategy that doesn't extend these buffers. "
                    "Kept freshly-reset defaults instead: "
                    f"{[(n, s) for n, s in skipped]}",
                    flush=True,
                )
            _uid_skipped = any(n == "gaussian_uid" for n, _ in skipped)
            self._ensure_geometry_streaming_buffers()
            _lineage_reseeded = bool(
                self._ensure_lineage_after_restore(
                    streaming_state, skipped={n for n, _ in skipped}
                )
            )
        elif any(hasattr(self, n) for n in self._STREAMING_BUFFER_NAMES):
            print(
                "[checkpoint] Warning: checkpoint has no streaming_state; "
                "lifecycle/anchor/support buffers reset to defaults.",
                flush=True,
            )
        # The row bitmap is checkpointed, but its bounded SourceIdentity
        # fingerprint-to-slot table is process-local metadata.  Restoring a
        # bitmap without that exact mapping could alias a new source to an old
        # slot and fabricate (or erase) physical-view support.  Fail closed by
        # discarding the complete E0 ledger; normal evidence rebuilds it.
        if any(
            name in streaming_state
            for name in (
                "population_policy_exposure_count",
                "population_policy_support_count",
                "population_policy_source_seen",
            )
        ):
            self._reset_population_policy_e0_state(
                self.get_xyz.shape[0], device=restore_target
            )
        # Restore the persistent Gaussian UID counter. For checkpoints saved before
        # this feature, derive it from the max UID in the restored gaussian_uid buffer
        # (if present), or default to 0 if the buffer is empty.
        self._next_gaussian_uid = model_args.get(
            "next_gaussian_uid",
            int(self.gaussian_uid.max().item()) + 1 if getattr(self, "gaussian_uid", None) is not None and self.gaussian_uid.shape[0] > 0 else 0,
        )
        # Issue #283: authoritative -- overrides the throwaway range that
        # training_setup()'s _reset_running_state recorded a few lines above.
        self._restore_seed_uid_range(
            model_args.get("seed_uid_range"),
            has_streaming_state=bool(streaming_state),
            uid_skipped=_uid_skipped,
            lineage_reseeded=_lineage_reseeded,
        )

    _LINEAGE_BUFFER_NAMES: tuple = (
        "parent_uid", "root_uid", "generation", "birth_method", "last_reseed_iter",
    )

    def _ensure_lineage_after_restore(self, streaming_state: dict, skipped: frozenset = frozenset()) -> bool:
        """Re-seed lineage when restoring a PRE-lineage-fix checkpoint.

        ``restore()`` -> ``training_setup()`` -> ``_reset_running_state()`` seeds
        ``root_uid = gaussian_uid.clone()`` off a throwaway fresh ``arange``
        block, and the ``streaming_state`` loop above then overwrites
        ``gaussian_uid`` from the checkpoint WITHOUT re-deriving ``root_uid``.
        On an old checkpoint (no lineage keys saved) that leaves every
        ``root_uid`` pointing at a uid present nowhere in the model -- shapes
        match, so nothing crashes and the corruption is silent.

        Treats a PARTIALLY-present lineage set as pre-fix too: a mix of restored
        and stale fields is never coherent, so all five are re-seeded together.
        A checkpoint carrying the complete set is left untouched.

        ``skipped`` is the set of buffer names ``restore()`` refused to apply
        because their saved size didn't match the current Gaussian count (see
        the ``streaming_state`` loop above). If ``gaussian_uid`` itself was
        skipped, any restored lineage fields (even if individually
        correctly-sized) reference the OLD uid space, not the freshly
        bootstrapped ``gaussian_uid`` now in place -- force a full reseed.
        """
        # Present-but-stale-length lineage is treated as pre-fix too (same
        # standing as _ensure_geometry_streaming_buffers' own shape checks):
        # matching shapes are what keep this corruption class silent.
        if "gaussian_uid" not in skipped and skipped.isdisjoint(self._LINEAGE_BUFFER_NAMES) and all(
            name in streaming_state for name in self._LINEAGE_BUFFER_NAMES
        ) and all(
            getattr(self, name, None) is not None
            and getattr(self, name).shape[0] == self.get_xyz.shape[0]
            for name in self._LINEAGE_BUFFER_NAMES
        ):
            return False
        from eligsir.utils.gaussian_lineage import BIRTH_METHOD_BOOTSTRAP, new_root_lineage_rows

        uid = getattr(self, "gaussian_uid", None)
        if uid is None or uid.shape[0] != self.get_xyz.shape[0]:
            return False
        print(
            "[checkpoint] Warning: checkpoint predates per-Gaussian lineage "
            f"({sorted(set(self._LINEAGE_BUFFER_NAMES) - set(streaming_state))} missing); "
            "re-seeding lineage from the restored gaussian_uid as fresh roots.",
            flush=True,
        )
        for name, value in new_root_lineage_rows(uid, BIRTH_METHOD_BOOTSTRAP).items():
            setattr(self, name, value)
        return True

    def _ensure_geometry_streaming_buffers(self):
        count = self.get_xyz.shape[0]
        device = self.get_xyz.device
        if not hasattr(self, "anchor_normal") or self.anchor_normal.shape[0] != count:
            normal = torch.zeros((count, 3), dtype=torch.float32, device=device)
            if count > 0:
                normal[:, 2] = 1.0
            self.anchor_normal = normal
        if not hasattr(self, "anchor_confidence") or self.anchor_confidence.shape[0] != count:
            self.anchor_confidence = torch.zeros(count, dtype=torch.float32, device=device)
        if not hasattr(self, "geometry_source") or self.geometry_source.shape[0] != count:
            self.geometry_source = torch.zeros(count, dtype=torch.int16, device=device)
        if not hasattr(self, "surface_state") or self.surface_state.shape[0] != count:
            self.surface_state = torch.zeros(count, dtype=torch.int8, device=device)
        ensure_belief_buffers(self)

    def construct_list_of_attributes(self):
        names = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self.params["sh0"].shape[1] * self.params["sh0"].shape[2]):
            names.append(f"f_dc_{i}")
        n_active_bands = (self.active_sh_degree + 1) ** 2 - 1
        for i in range(n_active_bands * self.params["shN"].shape[2]):
            names.append(f"f_rest_{i}")
        names.append("opacity")
        for i in range(self.params["scales"].shape[1]):
            names.append(f"scale_{i}")
        for i in range(self.params["quats"].shape[1]):
            names.append(f"rot_{i}")
        return names

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        n_active_bands = (self.active_sh_degree + 1) ** 2 - 1

        xyz = self.params["means"].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.params["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = (
            self.params["shN"][:, :n_active_bands, :]
            .detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        )
        opacities = self.params["opacities"].detach().unsqueeze(-1).cpu().numpy()
        scale = self.params["scales"].detach().cpu().numpy()
        rotation = self.params["quats"].detach().cpu().numpy()
        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        PlyData(
            [PlyElement.describe(elements, "vertex")],
            comments=[f"active_sh_degree {self.active_sh_degree}"],
        ).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        saved_degree = None
        for comment in plydata.comments:
            parts = comment.strip().split()
            if len(parts) == 2 and parts[0] == "active_sh_degree":
                saved_degree = int(parts[1])
                break

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))

        if saved_degree is not None:
            expected_len = 3 * ((saved_degree + 1) ** 2 - 1)
            assert len(extra_f_names) == expected_len, (
                f"PLY declares active_sh_degree={saved_degree} (expects {expected_len} "
                f"f_rest_* properties) but has {len(extra_f_names)}"
            )
            n_active_bands = (saved_degree + 1) ** 2 - 1
        else:
            assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            n_active_bands = (self.max_sh_degree + 1) ** 2 - 1

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, n_active_bands))

        max_bands = (self.max_sh_degree + 1) ** 2 - 1
        features_rest_full = np.zeros((xyz.shape[0], 3, max_bands))
        features_rest_full[:, :, :n_active_bands] = features_extra

        scale_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")],
            key=lambda x: int(x.split("_")[-1]),
        )
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self.params = nn.ParameterDict({
            "means": nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device="cuda").requires_grad_(True)),
            "scales": nn.Parameter(torch.tensor(scales, dtype=torch.float32, device="cuda").requires_grad_(True)),
            "quats": nn.Parameter(torch.tensor(rots, dtype=torch.float32, device="cuda").requires_grad_(True)),
            "opacities": nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device="cuda").requires_grad_(True)),
            "sh0": nn.Parameter(torch.tensor(features_dc, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)),
            "shN": nn.Parameter(torch.tensor(features_rest_full, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)),
        })
        self.active_sh_degree = saved_degree if saved_degree is not None else self.max_sh_degree
        self._reset_running_state()
        # Issue #283: a PLY carries no lineage buffers at all, so the reset above
        # just stamped the ENTIRE loaded population as bootstrap roots with a
        # fresh uid block. That is indistinguishable from a real seed cohort and
        # is exactly the contamination this marker exists to prevent.
        self._invalidate_seed_uid_range(
            "model was loaded from a PLY, which carries no lineage buffers; every "
            "row was stamped BOOTSTRAP by the reset regardless of how it was born"
        )

    def _reset_running_state(self):
        count = self.get_xyz.shape[0] if len(self.params) else 0
        self.xyz_gradient_accum = torch.zeros((count, 1), device="cuda")
        self.denom = torch.zeros((count, 1), device="cuda")
        self.max_radii2D = torch.zeros((count,), device="cuda")
        self.visibility_ema = torch.zeros((count, 1), device="cuda")
        self.birth_frame = torch.zeros(count, dtype=torch.int32, device="cuda")
        self.birth_iter = torch.zeros(count, dtype=torch.int32, device="cuda")
        self.support_count = torch.zeros(count, dtype=torch.int16, device="cuda")
        self.provisional = torch.zeros(count, dtype=torch.bool, device="cuda")
        xyz = self.get_xyz.detach() if count > 0 else torch.zeros((0, 3), device="cuda")
        self.anchor_xyz = xyz.clone()
        self.anchor_iter = torch.zeros(count, dtype=torch.int32, device="cuda")
        self.depth_conflict_count = torch.zeros(count, dtype=torch.int16, device="cuda")
        self.depth_last_support_uid = torch.full((count,), -1, dtype=torch.int32, device="cuda")
        self.depth_last_conflict_uid = torch.full((count,), -1, dtype=torch.int32, device="cuda")
        self.lifecycle_state = torch.ones(count, dtype=torch.int8, device="cuda")
        self.utility_ema = torch.zeros(count, dtype=torch.float32, device="cuda")
        scales = self.params["scales"].detach() if count > 0 else torch.zeros((0, 3), device="cuda")
        opacities = self.params["opacities"].detach() if count > 0 else torch.zeros(0, device="cuda")
        self.anchor_scale_log = scales.clone()
        self.anchor_opacity_logit = opacities.clone()
        self.anchor_normal = torch.zeros((count, 3), dtype=torch.float32, device="cuda")
        if count > 0:
            self.anchor_normal[:, 2] = 1.0
        self.anchor_confidence = torch.zeros(count, dtype=torch.float32, device="cuda")
        self.geometry_source = torch.zeros(count, dtype=torch.int16, device="cuda")
        self.surface_state = torch.zeros(count, dtype=torch.int8, device="cuda")
        # GaussianBelief (issue #1365 Phase 1). birth_anchor_xyz is seeded to the
        # current position below, matching how anchor_xyz/anchor_scale_log are
        # already seeded at reset ("anchor = current value at birth").
        reset_belief_buffers(self, count, device="cuda")
        self._reset_population_policy_e0_state(count, device=self.get_xyz.device)
        # Bootstrap-created Gaussians never went through add_points_as_gaussians
        # (which is the only place submap_id rows get created, at line ~840), so
        # without this they permanently desync submap_id's length from get_xyz's
        # -- apply_submap_correction's boolean-mask indexing against the shorter
        # buffer then raises, silently swallowed by pose_corrections.py's bare
        # except, so gaussians_corrected stayed exactly 0 in every run regardless
        # of match_radius/max_rmse/min_coherence tuning. Stamp bootstrap points
        # with the current submap (0 at true bootstrap; _current_submap_id may
        # already be nonzero on a checkpoint-restore that predates this fix).
        self.submap_id = torch.full(
            (count,), int(getattr(self, "_current_submap_id", 0)),
            dtype=torch.int32, device="cuda",
        )
        # Bootstrap Gaussians likewise predate add_points_as_gaussians(), the
        # only growth path that stamps anchor_keyframe_id.  Keep this
        # checkpointed per-Gaussian buffer aligned from birth so an exact
        # mapping-endpoint resume never has to discard it as stale-sized.
        self.anchor_keyframe_id = torch.zeros(
            count, dtype=torch.int32, device="cuda"
        )
        # Persistent Gaussian identity (Frame/Gaussian observation DB). Assign a
        # contiguous block of UIDs at bootstrap, advancing the global counter --
        # every OTHER buffer in this method is 0/default-initialized per-run, but
        # this one must never repeat a value across the model's lifetime, matching
        # every add_points_as_gaussians call below.
        _seed_lo = int(self._next_gaussian_uid)
        self.gaussian_uid = torch.arange(
            self._next_gaussian_uid, self._next_gaussian_uid + count,
            dtype=torch.int64, device="cuda",
        )
        self._next_gaussian_uid += count
        lineage_rows = new_root_lineage_rows(self.gaussian_uid, BIRTH_METHOD_BOOTSTRAP)
        for _name, _val in lineage_rows.items():
            setattr(self, _name, _val)
        # Issue #283: this reset just declared the WHOLE population to be
        # bootstrap roots, so the block it allocated is the SEED cohort -- as
        # long as this really is a bootstrap and not a re-setup of an already
        # evolved population (see _record_seed_uid_range).
        self._record_seed_uid_range(_seed_lo, int(self._next_gaussian_uid))

    def _reset_population_policy_e0_state(self, count: int, *, device: torch.device) -> None:
        """Clear optional live-policy evidence without enabling the policy.

        Legacy shadow-only reset must not preserve an old physical-source
        ledger for newly bootstrapped hypotheses.  Buffers remain optional so
        non-live profiles do not allocate this feature merely by resetting.
        """

        for name, dtype in (
            ("population_policy_exposure_count", torch.int32),
            ("population_policy_support_count", torch.int16),
        ):
            if getattr(self, name, None) is not None:
                setattr(self, name, torch.zeros(count, dtype=dtype, device=device))
        seen = getattr(self, "population_policy_source_seen", None)
        if seen is not None:
            if not isinstance(seen, torch.Tensor) or seen.ndim != 2:
                raise ValueError("invalid population policy source-seen buffer")
            self.population_policy_source_seen = torch.zeros(
                (count, seen.shape[1]), dtype=torch.bool, device=device
            )
        for name in (
            "_population_policy_source_slot_by_fingerprint",
            "_population_policy_source_identity_by_fingerprint",
        ):
            table = getattr(self, name, None)
            if table is not None:
                if not isinstance(table, dict):
                    raise ValueError("invalid population policy source ledger")
                table.clear()

    # ---- SEED-cohort uid range (issue #283) --------------------------------

    def _record_seed_uid_range(self, lo: int, hi: int) -> None:
        """Record ``[lo, hi)`` as the bootstrap uid block, or invalidate loudly.

        A fresh run legitimately resets twice (``create_from_pcd`` then
        ``training_setup``) with NO uid allocated in between, so the second
        block starts exactly where the first ended and simply supersedes it.
        If uids WERE allocated in between, this reset is a mid-run
        re-bootstrap of an already-grown population (``training_setup()`` is
        called again mid-run by ``rolling_seed_application`` and
        ``submap_stitch_application`` after inserting points) -- the original
        seed identity is destroyed by that reset and cannot be recovered, so
        the cohort becomes UNDETERMINED rather than silently redefined to mean
        "everything alive at re-setup time".
        """
        if getattr(self, "_seed_uid_range_invalidated", False):
            return
        prev = getattr(self, "_seed_uid_range", None)
        if prev is not None and int(prev[1]) != int(lo):
            self._invalidate_seed_uid_range(
                f"_reset_running_state re-bootstrapped an already-grown population "
                f"(previous seed block ended at uid {int(prev[1])}, this reset starts "
                f"at uid {int(lo)}); the original bootstrap rows' lineage was "
                "overwritten and cannot be recovered"
            )
            return
        self._seed_uid_range = (int(lo), int(hi))
        self._seed_uid_range_reason = ""

    def _invalidate_seed_uid_range(self, reason: str) -> None:
        """Mark the SEED cohort undetermined (sticky) and say so once, loudly."""
        already = (
            getattr(self, "_seed_uid_range", None) is None
            and getattr(self, "_seed_uid_range_invalidated", False)
        )
        self._seed_uid_range = None
        self._seed_uid_range_reason = reason
        self._seed_uid_range_invalidated = True
        if not already:
            print(
                f"[seed_cohort] WARNING: SEED cohort is UNDETERMINED -- {reason}. "
                "utils.gaussian_lineage.seed_cohort_mask() will raise "
                "SeedCohortUnavailable rather than return a contaminated mask "
                "(issue #283).",
                flush=True,
            )

    def _restore_seed_uid_range(
        self, saved_range, *, has_streaming_state: bool,
        uid_skipped: bool, lineage_reseeded: bool,
    ) -> None:
        """Apply the checkpoint's SEED uid range, or invalidate with the reason.

        ``restore()`` -> ``training_setup()`` -> ``_reset_running_state()`` has
        already recorded a THROWAWAY range over a throwaway ``arange`` block by
        the time this runs, so this method is authoritative and always either
        overwrites or invalidates -- never leaves that value in place.
        """
        # Authoritative: clear any range recorded by the throwaway reset above.
        self._seed_uid_range = None
        self._seed_uid_range_invalidated = False
        if not has_streaming_state:
            self._invalidate_seed_uid_range(
                "checkpoint carries no streaming_state, so no lineage was restored"
            )
            return
        if saved_range is None:
            self._invalidate_seed_uid_range(
                "checkpoint predates the SEED uid range (no 'seed_uid_range' key); "
                "which uids were bootstrap rows is not recorded anywhere in it"
            )
            return
        if uid_skipped:
            self._invalidate_seed_uid_range(
                "checkpoint's gaussian_uid buffer was skipped on a size mismatch, so "
                "the live uids are a throwaway arange unrelated to the saved range"
            )
            return
        if lineage_reseeded:
            self._invalidate_seed_uid_range(
                "checkpoint predates per-Gaussian lineage and every row was re-seeded "
                "as a fresh BOOTSTRAP root, which is indistinguishable from a real "
                "seed cohort"
            )
            return
        self._seed_uid_range = (int(saved_range[0]), int(saved_range[1]))
        self._seed_uid_range_reason = ""

    def prune_points(self, mask):
        self._reject_legacy_population_writer("prune_points")
        valid_points_mask = ~mask

        # Prune optimizable parameters
        for name, optimizer in self.optimizers.items():
            group = optimizer.param_groups[0]
            old_param = group["params"][0]
            stored_state = optimizer.state.get(old_param, None)

            new_param = nn.Parameter(old_param[valid_points_mask].detach().requires_grad_(True))
            if stored_state is not None:
                new_state = {}
                for k, v in stored_state.items():
                    if isinstance(v, torch.Tensor) and v.dim() > 0:
                        new_state[k] = v[valid_points_mask]
                    else:
                        new_state[k] = v  # scalar 'step' or Python int/float
                del optimizer.state[old_param]
                optimizer.state[new_param] = new_state
            group["params"][0] = new_param
            self.params[name] = new_param

        # Prune running state buffers
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.visibility_ema = self.visibility_ema[valid_points_mask]
        self.birth_frame = self.birth_frame[valid_points_mask]
        if self.birth_iter.shape[0] == mask.shape[0]:
            self.birth_iter = self.birth_iter[valid_points_mask]
        self.support_count = self.support_count[valid_points_mask]
        self.provisional = self.provisional[valid_points_mask]
        self.anchor_xyz = self.anchor_xyz[valid_points_mask]
        self.anchor_iter = self.anchor_iter[valid_points_mask]
        if self.depth_conflict_count.shape[0] == mask.shape[0]:
            self.depth_conflict_count = self.depth_conflict_count[valid_points_mask]
        if self.depth_last_support_uid.shape[0] == mask.shape[0]:
            self.depth_last_support_uid = self.depth_last_support_uid[valid_points_mask]
        if self.depth_last_conflict_uid.shape[0] == mask.shape[0]:
            self.depth_last_conflict_uid = self.depth_last_conflict_uid[valid_points_mask]
        if self.lifecycle_state.shape[0] == mask.shape[0]:
            self.lifecycle_state = self.lifecycle_state[valid_points_mask]
        if self.utility_ema.shape[0] == mask.shape[0]:
            self.utility_ema = self.utility_ema[valid_points_mask]
        if self.anchor_scale_log.shape[0] == mask.shape[0]:
            self.anchor_scale_log = self.anchor_scale_log[valid_points_mask]
        if self.anchor_opacity_logit.shape[0] == mask.shape[0]:
            self.anchor_opacity_logit = self.anchor_opacity_logit[valid_points_mask]
        if hasattr(self, "anchor_normal") and self.anchor_normal.shape[0] == mask.shape[0]:
            self.anchor_normal = self.anchor_normal[valid_points_mask]
        if hasattr(self, "anchor_confidence") and self.anchor_confidence.shape[0] == mask.shape[0]:
            self.anchor_confidence = self.anchor_confidence[valid_points_mask]
        if hasattr(self, "geometry_source") and self.geometry_source.shape[0] == mask.shape[0]:
            self.geometry_source = self.geometry_source[valid_points_mask]
        if hasattr(self, "surface_state") and self.surface_state.shape[0] == mask.shape[0]:
            self.surface_state = self.surface_state[valid_points_mask]
        # Phase 12: submap metadata
        if self.submap_id.shape[0] == mask.shape[0]:
            self.submap_id = self.submap_id[valid_points_mask]
        if self.anchor_keyframe_id.shape[0] == mask.shape[0]:
            self.anchor_keyframe_id = self.anchor_keyframe_id[valid_points_mask]
        if self.gaussian_uid.shape[0] == mask.shape[0]:
            self.gaussian_uid = self.gaussian_uid[valid_points_mask]
        for _name in ("parent_uid", "root_uid", "generation", "birth_method", "last_reseed_iter"):
            _t = getattr(self, _name, None)
            if _t is not None and _t.shape[0] == mask.shape[0]:
                setattr(self, _name, _t[valid_points_mask])
        # Part C floater-consensus buffers (only present when the flag is on).
        for _name in ("seen_count", "consensus_support_count",
                      "consensus_conflict_count", "consensus_last_seen_uid",
                      "last_supervised_iter",
                      # issue #1262 exposure buffers (only present when the flag is on)
                      "exposure_count", "exposure_last_view_uid",
                      "supervised_tier_high_water",
                      "population_policy_exposure_count",
                      "population_policy_support_count",
                      "population_policy_source_seen"):
            _t = getattr(self, _name, None)
            if _t is not None and _t.shape[0] == mask.shape[0]:
                setattr(self, _name, _t[valid_points_mask])
        # Part D1 voter buffer ([N,3]); bool-mask indexing handles the 2-D shape.
        _voter = getattr(self, "consensus_last_voter_center", None)
        if _voter is not None and _voter.shape[0] == mask.shape[0]:
            self.consensus_last_voter_center = _voter[valid_points_mask]
        # GaussianBelief (issue #1365 Phase 1).
        prune_belief_buffers(self, mask, valid_points_mask)

    def add_points_as_gaussians(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        init_scale: float = 0.01,
        init_opacity: float = 0.5,
        use_knn_scale: bool = False,
        normals: torch.Tensor = None,
        scales: torch.Tensor = None,
        rotations: torch.Tensor = None,
        is_provisional: bool = False,
        birth_frame: int = 0,
        birth_iter: int = 0,
        opacities_raw: torch.Tensor = None,
        sh_rest: torch.Tensor = None,
        confidence: torch.Tensor = None,
        source: torch.Tensor = None,
        surface_state: torch.Tensor = None,
        birth_method: int = BIRTH_METHOD_SOURCE_DEPTH,
    ) -> int:
        """
        Append new Gaussians initialised from 3-D world-space points and RGB
        colours (float32, range [0, 1]).  Extends each per-parameter optimizer
        with zero-initialised momentum state for the new entries.

        use_knn_scale: derive per-Gaussian scale from k-NN distance (matches
        bootstrap quality); falls back to fixed init_scale when False.

        Returns the number of Gaussians actually added.
        """
        self._reject_legacy_population_writer("add_points_as_gaussians")
        N = points.shape[0]
        if N == 0:
            return 0

        from eligsir.utils.sh_utils import RGB2SH
        from eligsir.utils.general_utils import inverse_sigmoid

        num_sh = (self.max_sh_degree + 1) ** 2
        fused_color = RGB2SH(colors.to(device="cuda", dtype=torch.float32))  # (N, 3)

        # Scale initialisation
        if scales is not None:
            log_scales = scales.to(device="cuda", dtype=torch.float32)
        elif use_knn_scale and N > 1:
            from eligsir.utils.rocm_knn_fallback import distCUDA2
            pts_cuda = points.to(device="cuda", dtype=torch.float32)
            dist_sq = distCUDA2(pts_cuda)  # (N,) squared dist to nearest neighbour
            scales_1d = torch.clamp(dist_sq.sqrt() * 0.5, 1e-4, 0.1)
            log_scales = torch.log(scales_1d).unsqueeze(-1).expand(-1, 3).contiguous()
        else:
            log_scales = torch.full((N, 3), math.log(init_scale * 0.1), device="cuda")

        if rotations is not None:
            new_quats = rotations.to(device="cuda", dtype=torch.float32)
        else:
            new_quats = torch.cat([                                                    # wxyz identity
                torch.ones(N, 1, device="cuda"),
                torch.zeros(N, 3, device="cuda"),
            ], dim=1)

        if sh_rest is not None:
            sh_rest_t = sh_rest.to(device="cuda", dtype=torch.float32)
            if sh_rest_t.shape != (N, num_sh - 1, 3):
                # Pad or truncate to match current max_sh_degree
                padded = torch.zeros((N, num_sh - 1, 3), device="cuda")
                copy_len = min(sh_rest_t.shape[1], num_sh - 1)
                padded[:, :copy_len, :] = sh_rest_t[:, :copy_len, :]
                sh_rest_t = padded
        else:
            sh_rest_t = torch.zeros((N, num_sh - 1, 3), device="cuda")

        if opacities_raw is not None:
            opacities_t = opacities_raw.to(device="cuda", dtype=torch.float32).reshape(N)
        else:
            opacities_t = inverse_sigmoid(torch.full((N,), float(init_opacity), device="cuda"))

        new_tensors = {
            "means": points.to(device="cuda", dtype=torch.float32).contiguous(),
            "sh0": fused_color[:, None, :].contiguous(),                          # (N, 1, 3)
            "shN": sh_rest_t,                                                     # (N, R-1, 3)
            "scales": log_scales,
            "quats": new_quats,
            "opacities": opacities_t,
        }

        # Build the full set of new-row buffer tensors up front (a single dict keyed
        # by the model attribute name). Buffers that are conditionally allocated
        # (consensus / voter) are only included when the live attribute is present,
        # so the commit path extends exactly the buffers the legacy code did.
        new_buffers = self._build_new_buffer_rows(
            N=N,
            new_means=new_tensors["means"],
            new_scales=new_tensors["scales"],
            new_opacities=new_tensors["opacities"],
            normals=normals,
            confidence=confidence,
            source=source,
            surface_state=surface_state,
            is_provisional=is_provisional,
            birth_frame=birth_frame,
            birth_iter=birth_iter,
            birth_method=birth_method,
        )

        if self._staging_active and self.staged_gaussian_insert:
            # P3: defer the O(N_total) cat-storm. Accumulate the new rows in a side
            # list; flush_staged_inserts() materializes everything with one cat each.
            self._staged_tensors.append(new_tensors)
            self._staged_buffers.append(new_buffers)
            self._staged_count += N
            return N

        self._commit_new_gaussians(new_tensors, new_buffers, N)
        return N

    # ------------------------------------------------------------------
    # P3 amortized growth: staged insertion machinery
    # ------------------------------------------------------------------

    def begin_staged_inserts(self):
        """Open a staging session.

        Inside a session (and only when ``self.staged_gaussian_insert`` is True),
        ``add_points_as_gaussians`` appends new rows to a side buffer instead of
        materializing them into ``self.params`` / optimizers. The caller MUST call
        :meth:`flush_staged_inserts` before any code reads live model state
        (``self.params[...]``, a strategy step, ``capture()``, or a render). Use only
        across back-to-back inserts with no intervening live-state reads (e.g. the
        submap-merge loop). Idempotent if already open.
        """
        self._staging_active = True

    def flush_staged_inserts(self) -> int:
        """Materialize all staged inserts into params/optimizers with one cat each.

        Concatenates the staged rows together first (cheap — only over staged rows),
        then performs a single ``torch.cat`` against each live param / optimizer-state
        / buffer tensor. Closes the session. Returns the number of Gaussians
        materialized. Safe to call when no session is open or nothing is staged.
        """
        self._reject_legacy_population_writer("flush_staged_inserts")
        self._staging_active = False
        if self._staged_count == 0:
            self._staged_tensors = []
            self._staged_buffers = []
            return 0

        total = self._staged_count
        # Merge staged param rows: concat across the staged chunks per key.
        merged_tensors = {}
        for name in self._staged_tensors[0].keys():
            merged_tensors[name] = torch.cat(
                [chunk[name] for chunk in self._staged_tensors], dim=0
            )
        # Merge staged buffer rows. Every staged chunk extends the same set of
        # buffers (built by _build_new_buffer_rows against the live attrs), so the
        # key sets are identical across chunks.
        merged_buffers = {}
        for name in self._staged_buffers[0].keys():
            merged_buffers[name] = torch.cat(
                [chunk[name] for chunk in self._staged_buffers], dim=0
            )

        self._staged_tensors = []
        self._staged_buffers = []
        self._staged_count = 0

        self._commit_new_gaussians(merged_tensors, merged_buffers, total)
        return total

    def _build_new_buffer_rows(
        self, *, N, new_means, new_scales, new_opacities, normals, confidence,
        source, surface_state, is_provisional, birth_frame, birth_iter, birth_method,
    ) -> dict:
        """Construct all new-row buffer tensors for an insert of ``N`` Gaussians.

        Returns a dict mapping model-attribute name -> (N, ...) tensor. Conditionally
        allocated buffers (Part C consensus / Part D1 voter) are only included when
        the corresponding live attribute already exists, matching the legacy commit.
        """
        rows = {}
        # Running-state buffers
        rows["max_radii2D"] = torch.zeros(N, device="cuda")
        rows["xyz_gradient_accum"] = torch.zeros(N, 1, device="cuda")
        rows["denom"] = torch.zeros(N, 1, device="cuda")
        rows["visibility_ema"] = torch.zeros(N, 1, device="cuda")
        # Streaming buffers
        rows["birth_frame"] = torch.full((N,), int(birth_frame), dtype=torch.int32, device="cuda")
        rows["birth_iter"] = torch.full((N,), int(birth_iter), dtype=torch.int32, device="cuda")
        rows["support_count"] = torch.zeros(N, dtype=torch.int16, device="cuda")
        rows["provisional"] = torch.full((N,), bool(is_provisional), dtype=torch.bool, device="cuda")
        rows["anchor_xyz"] = new_means.detach().clone()
        # Anchor age is row-aligned mutation state. Initialize it atomically
        # with birth_iter so no caller has to patch the model after commit.
        rows["anchor_iter"] = torch.full(
            (N,), int(birth_iter), dtype=torch.int32, device="cuda"
        )
        rows["depth_conflict_count"] = torch.zeros(N, dtype=torch.int16, device="cuda")
        rows["depth_last_support_uid"] = torch.full((N,), -1, dtype=torch.int32, device="cuda")
        rows["depth_last_conflict_uid"] = torch.full((N,), -1, dtype=torch.int32, device="cuda")
        # lifecycle_state enum: 0=PROVISIONAL, 1=YOUNG, 2=MATURE, 3=FROZEN. Must
        # agree with the `provisional` bool buffer above -- a zero-fill here
        # regardless of `is_provisional` disagreed with `provisional=False`
        # insertions until a periodic repair pass fixed it up (issue #908).
        rows["lifecycle_state"] = torch.full(
            (N,), 0 if is_provisional else 1, dtype=torch.int8, device="cuda"
        )
        rows["utility_ema"] = torch.zeros(N, dtype=torch.float32, device="cuda")
        rows["anchor_scale_log"] = new_scales.detach().clone()
        rows["anchor_opacity_logit"] = new_opacities.detach().clone()

        if normals is not None:
            new_anchor_normal = torch.nn.functional.normalize(
                normals.to(device="cuda", dtype=torch.float32), dim=-1
            )
        else:
            new_anchor_normal = torch.zeros((N, 3), dtype=torch.float32, device="cuda")
            new_anchor_normal[:, 2] = 1.0
        rows["anchor_normal"] = new_anchor_normal
        rows["anchor_confidence"] = (
            confidence.to(device="cuda", dtype=torch.float32).reshape(N)
            if confidence is not None
            else torch.zeros(N, dtype=torch.float32, device="cuda")
        )
        rows["geometry_source"] = (
            source.to(device="cuda", dtype=torch.int16).reshape(N)
            if source is not None
            else torch.zeros(N, dtype=torch.int16, device="cuda")
        )
        rows["surface_state"] = (
            surface_state.to(device="cuda", dtype=torch.int8).reshape(N)
            if surface_state is not None
            else torch.full((N,), 1 if is_provisional else 2, dtype=torch.int8, device="cuda")
        )
        # Phase 12: submap metadata
        _cur_submap = int(getattr(self, "_current_submap_id", 0))
        rows["submap_id"] = torch.full((N,), _cur_submap, dtype=torch.int32, device="cuda")
        rows["anchor_keyframe_id"] = torch.full((N,), int(birth_frame), dtype=torch.int32, device="cuda")
        rows["gaussian_uid"] = torch.arange(
            self._next_gaussian_uid, self._next_gaussian_uid + N,
            dtype=torch.int64, device="cuda",
        )
        self._next_gaussian_uid += N
        lineage_rows = new_root_lineage_rows(rows["gaussian_uid"], birth_method)
        rows.update(lineage_rows)

        # Part C floater-consensus buffers: only when already allocated.
        for _name, _fill, _dtype in (
            ("seen_count", 0, torch.int32),
            ("consensus_support_count", 0, torch.int32),
            ("consensus_conflict_count", 0, torch.int32),
            ("consensus_last_seen_uid", -1, torch.int32),
            # Recency-reg buffer: fill 0 (not birth_iter) so a newly-inserted
            # Gaussian starts with a large recency gap until its first render
            # pass marks it supervised -- gets the ramp-min opacity-reg weight
            # immediately rather than full pressure, mirroring the young/
            # maturity exemptions' intent of protecting fresh mass.
            ("last_supervised_iter", 0, torch.int32),
            # issue #1262: a freshly inserted Gaussian has had ZERO exposure at
            # any tier, and no view uid recorded yet.
            ("exposure_count", 0, torch.int32),
            ("exposure_last_view_uid", -1, torch.int64),
            ("supervised_tier_high_water", -1, torch.int8),
            ("population_policy_exposure_count", 0, torch.int32),
            ("population_policy_support_count", 0, torch.int16),
        ):
            if getattr(self, _name, None) is not None:
                rows[_name] = torch.full((N,), _fill, dtype=_dtype, device="cuda")
        source_seen = getattr(self, "population_policy_source_seen", None)
        if source_seen is not None:
            if not isinstance(source_seen, torch.Tensor) or source_seen.ndim != 2:
                raise ValueError("invalid population policy source-seen buffer")
            rows["population_policy_source_seen"] = torch.zeros(
                (N, source_seen.shape[1]), dtype=torch.bool, device="cuda"
            )
        # Part D1 voter buffer ([N,3]): only when already allocated.
        if getattr(self, "consensus_last_voter_center", None) is not None:
            rows["consensus_last_voter_center"] = torch.full(
                (N, 3), float("nan"), dtype=torch.float32, device="cuda"
            )
        # GaussianBelief (issue #1365 Phase 1). birth_anchor_xyz = this insertion's
        # own position, matching "anchor = position at birth".
        rows.update(new_row_belief_buffers(N, new_means))
        return rows

    def cat_tensors_to_optimizer(self, tensors_dict: dict) -> dict:
        """Append ``tensors_dict[name]`` rows to each param + its Adam momentum state.

        Mutates ``self.optimizers`` and ``self.params`` in lockstep so that, for every
        ``name``, ``self.params[name]`` and ``self.optimizers[name].param_groups[0]
        ["params"][0]`` remain the SAME nn.Parameter object (the strategy contract).
        New momentum rows (exp_avg / exp_avg_sq) are zero-initialised. Returns the
        dict of new params for convenience. Extracted so it is unit-testable.
        """
        self._reject_legacy_population_writer("cat_tensors_to_optimizer")
        new_params = {}
        for name, ext in tensors_dict.items():
            optimizer = self.optimizers[name]
            group = optimizer.param_groups[0]
            old_param = group["params"][0]
            stored_state = optimizer.state.get(old_param, None)
            n_new = ext.shape[0]

            new_param = nn.Parameter(
                torch.cat([old_param.detach(), ext.detach()], dim=0).requires_grad_(True)
            )
            if stored_state is not None:
                new_state = {}
                for k, v in stored_state.items():
                    if isinstance(v, torch.Tensor) and v.dim() == ext.dim():
                        zeros = torch.zeros(n_new, *ext.shape[1:], dtype=v.dtype, device=v.device)
                        new_state[k] = torch.cat([v, zeros], dim=0)
                    else:
                        new_state[k] = v  # scalar 'step' or other non-extensible state
                del optimizer.state[old_param]
                optimizer.state[new_param] = new_state
            group["params"][0] = new_param
            self.params[name] = new_param
            new_params[name] = new_param
        return new_params

    def _commit_new_gaussians(self, new_tensors: dict, new_buffers: dict, N: int):
        """Materialize ``N`` new Gaussians: extend params/optimizers + every buffer.

        Whether the rows came from a single insert (legacy path) or a flushed staging
        batch, this does exactly one cat per param / momentum tensor / buffer, leaving
        ``self.params`` truthful and all buffers length-consistent on return.
        """
        self._reject_legacy_population_writer("_commit_new_gaussians")
        old_lengths = {name: getattr(self, name).shape[0] for name in new_buffers}
        self.cat_tensors_to_optimizer(new_tensors)
        for name, rows in new_buffers.items():
            old = getattr(self, name)
            setattr(self, name, torch.cat([old, rows], dim=0))
        self._assert_live_count_consistent(new_buffers, old_lengths, N)

    def _assert_live_count_consistent(self, new_buffers, old_lengths, N):
        """Debug invariant: params and every length-tracked buffer agree on N.

        Strategy landmine guard — if a buffer that *was* in lockstep with the model
        falls out of step, the MCMC strategy (which reads ``params["means"].shape[0]``
        as the live count and guards buffers with ``shape[0] == N``) would silently
        skip or, worse, noise/relocate/prune garbage rows.

        ``submap_id`` / ``anchor_keyframe_id`` are advisory in this codebase: they are
        NOT reset by ``_reset_running_state`` and can legitimately start desynced (see
        the ``shape[0] == mask.shape[0]`` guards in ``prune_points``). We therefore
        only require that each buffer GREW by exactly ``N`` and that buffers which were
        already in lockstep with the model remain so.
        """
        if not __debug__:
            return
        live = self.params["means"].shape[0]
        for name, rows in new_buffers.items():
            buf = getattr(self, name, None)
            if buf is None:
                continue
            grew_by = buf.shape[0] - old_lengths[name]
            assert grew_by == N, (
                f"staged-insert invariant violated: {name} grew by {grew_by}, "
                f"expected {N}"
            )
            # If it was tracking the model before, it must still track it now.
            if old_lengths[name] == live - N:
                assert buf.shape[0] == live, (
                    f"staged-insert invariant violated: {name}.shape[0]="
                    f"{buf.shape[0]} != live count {live}"
                )
