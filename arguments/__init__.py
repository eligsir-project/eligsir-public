#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace, BooleanOptionalAction
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action=BooleanOptionalAction)
                elif t == list:
                    item_type = type(value[0]) if value else str
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, nargs="+", type=item_type)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action=BooleanOptionalAction)
                elif t == list:
                    item_type = type(value[0]) if value else str
                    group.add_argument("--" + key, default=value, nargs="+", type=item_type)
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        # Optional Scene Replay Capsule directory path (issue #632).
        # When set, the capsule is validated at startup and metadata recorded
        # into <model_path>/replay_capsule.json. Default: None (flag absent).
        self.replay_capsule = ""
        # RNG seed for safe_state (init + stochastic training path). Override with --seed.
        self.seed = 1
        # Strict determinism mode (audit §14): threads through to
        # utils.general_utils.safe_state's strict_deterministic parameter, which
        # forces torch.use_deterministic_algorithms(True) and disables cudnn
        # benchmarking/enables cudnn determinism. Default False preserves the
        # existing "normal research mode" behavior exactly (source/order/RNG
        # reproducibility only, kernel-level numerical variation still allowed).
        # Deterministic kernels may be slower or unavailable for some ops, so
        # only set --strict_deterministic for qualification/debug runs. The
        # resolved value is also recorded into report.json["determinism"] so a
        # run's own artifacts can answer which mode it actually ran under.
        # Negate with --no-strict_deterministic.
        self.strict_deterministic = False
        # Eval is on by default — pass --no-eval to disable the test holdout.
        self.eval = True
        # Scientific mode (issue #1013): when set, enforces mandatory registered
        # ResearchFlow origin metadata (RESEARCHFLOW_RUN_ID, RESEARCHFLOW_PLAN_SHA256,
        # or experiment_origin.json) and emits evidence_class="MANAGED". Direct developer
        # runs without this flag remain unmanaged (evidence_class="UNMANAGED").
        self.scientific_mode = False
        # Source attestation (eligsir#1302): a narrow, explicit escape hatch
        # for a small, individually-reviewed set of historical RGB-D bags (e.g.
        # kitchen1/floor2) that predate eligsir-rgbd-sync's capture-provenance
        # gate and can never be re-recorded through it. Default "unknown" (NOT
        # "validated" -- unknown is the correct bottom-of-lattice default, mirroring
        # experiment_validity.py's UNKNOWN/UNDECLARED convention) so every run that
        # doesn't pass this flag is completely unaffected. "legacy_unattested" caps
        # evidence_class at UNMANAGED regardless of ResearchFlow registration -- see
        # utils/source_attestation.py and docs/architecture/2026-08-30-issue1302-legacy-attestation-design.md.
        # "validated" is reserved for a future ROS2-side producer that has actually
        # confirmed CaptureProvenance attestation; nothing sets it today.
        self.source_attestation_status = "unknown"
        self.source_attestation_bag_id = ""
        self.source_attestation_reason = ""
        self.cap_max = 500000
        self.init_type = "sfm"
        self.model_layout = "gsplat"
        # Gaussian scale init at bootstrap. "knn" is the canonical 3DGS init (scale
        # from the mean distance to the 3 nearest neighbours) and is now the default
        # again: the HIPified simple_knn distCUDA2 makes it ~30-290x cheaper than the
        # old O(N^2) torch fallback that forced the move to "fixed". Override with
        # --init_scale_mode {fixed,knn,voxel}. One-time bootstrap cost only.
        self.init_scale_mode = "knn"
        self.init_scale = 0.01
        self.scannet_frame_stride = 10
        self.scannet_max_frames = 0
        self.scannet_eval_hold = 20
        self.scannet_init = "rgbd"
        self.scannet_depth_stride = 8
        self.scannet_init_frames = 200
        self.scannet_max_init_points = 250000
        self.scannet_depth_scale = 1000.0
        self.scannet_random_num_pts = 250000
        self.tum_frame_stride = 1
        self.tum_max_frames = 0
        self.tum_eval_hold = 8
        self.tum_init = "rgbd"
        self.tum_depth_stride = 4
        self.tum_init_frames = 300
        self.tum_max_init_points = 250000
        self.tum_depth_scale = 5000.0
        self.tum_association_max_dt = 0.03
        self.tum_capture_time_pose_interpolation = False
        self.tum_sequence = ""
        self.tum_gt_path = ""
        self.tum_random_num_pts = 250000
        self.rgbd_eval_hold = 8
        self.rgbd_depth_stride = 4
        self.rgbd_init_frames = 300
        self.rgbd_max_init_points = 250000
        self.rgbd_min_depth = 0.1
        self.rgbd_max_depth = 8.0
        self.rgbd_random_num_pts = 250000
        self.pointcloud_preprocess = "open3d"
        self.pcd_voxel_size = 0.02
        self.pcd_outlier_filter = "none"
        self.pcd_stat_nb_neighbors = 20
        self.pcd_stat_std_ratio = 2.0
        self.pcd_radius = 0.05
        self.pcd_min_neighbors = 4
        self.pcd_estimate_normals = False
        self.pcd_force_regenerate = False
        self.replica_init = "mesh"
        self.replica_num_views = 120
        self.replica_width = 640
        self.replica_height = 480
        self.replica_eval_hold = 8
        self.replica_fov = 70.0
        self.replica_max_init_points = 250000
        self.replica_render_points = 300000
        self.replica_splat_radius = 1
        self.replica_random_num_pts = 250000
        self.hypersim_cam_id = "cam_00"
        self.hypersim_frame_stride = 1
        self.hypersim_eval_hold = 8
        self.orbbec_color_topic = "/camera/color/image_raw/compressed"
        self.orbbec_depth_topic = "/camera/depth/image_raw/compressed"
        self.orbbec_pose_topic = "/camera_pose"
        self.orbbec_pose_source = "camera_pose"
        self.orbbec_camera_info_topic = "/camera/color/camera_info"
        self.orbbec_sync_threshold_ms = 5.0
        self.orbbec_pose_sync_threshold_ms = 10.0
        self.orbbec_sync_estimate_offset = False
        self.orbbec_sync_offset_ms = "auto"
        self.orbbec_sync_strict = True
        self.orbbec_sync_report_json = ""
        self.orbbec_open3d_odom_max_failure_ratio = 0.25
        self.orbbec_open3d_odom_cache = True
        self.orbbec_open3d_odom_cache_dir = ""
        self.orbbec_open3d_odom_stride = 1
        self.orbbec_open3d_odom_downscale = 1
        self.orbbec_open3d_odom_async = True
        self.orbbec_open3d_odom_async_queue_size = 32
        self.orbbec_open3d_odom_motion_prior = False
        self.orbbec_open3d_odom_motion_gate = False
        self.orbbec_open3d_odom_max_trans_per_edge = 0.15
        self.orbbec_open3d_odom_max_rot_deg_per_edge = 8.0
        self.orbbec_open3d_odom_method = "hybrid"
        self.orbbec_open3d_odom_depth_diff_max = 0.07
        self.orbbec_vbg_voxel_size = 0.0058   # m, hash-map voxel resolution
        self.orbbec_vbg_block_count = 40000    # hash-map maximum blocks
        self.orbbec_vbg_depth_max = 3.0        # m, integration truncation distance
        self.orbbec_open3d_icp_max_distance = 0.07
        self.orbbec_open3d_icp_robust_kernel = "huber"
        self.orbbec_open3d_icp_sigma = 0.05
        # Generic ROS1 RGB-D bag input. "auto" keeps the legacy RealSense
        # defaults unless TUM-style topics are present in the bag.
        self.rosbag_profile = "auto"  # auto | realsense | tum
        self.rosbag_color_topic = ""
        self.rosbag_depth_topic = ""
        self.rosbag_color_info_topic = ""
        self.rosbag_depth_info_topic = ""
        self.rosbag_sync_threshold_ms = 33.0
        self.rosbag_sync_estimate_offset = False
        self.rosbag_sync_offset_ms = "0"
        self.rosbag_sync_report_json = ""
        self.rosbag_rgb_encoding_override = ""
        self.rosbag_depth_scale = 0.0
        self.rosbag_alignment_debug_dir = ""
        self.rosbag_alignment_debug_frames = 3
        self.rosbag_realsense_depth_filters = True
        # Transport-neutral provenance fields accepted from the ROS-owned
        # application. Core does not subscribe to or configure ROS topics.
        self.ros2_live_pose_source = "odom"  # tf | pose | odom
        self.ros2_live_pose_topic = "/slam/odom"
        self.ros2_live_pose_convention = "ros"
        # Set by bag-replay launch scripts (05_run_offline_rosbag.sh) that
        # publish /eligsir/bag_playback_complete once `ros2 bag play`
        # genuinely exits -- lets ros2_live's idle-timeout branch distinguish
        # a real bag end from a silently-stalled upstream pipeline (rgbd-sync/
        # decompress/ORB-SLAM3 dying mid-bag), recorded as
        # report.json's trajectory_coverage.stream_end_reason. Must stay False
        # for a genuine live-camera session (default) -- nothing ever
        # publishes that topic there, so leaving this on would mislabel every
        # normal live-session end as a suspected stall.
        self.ros2_live_bag_replay = False
        # Backward-compatible aliases for older RealSense commands.
        self.realsense_color_topic = "/device_0/sensor_1/Color_0/image/data"
        self.realsense_depth_topic = "/device_0/sensor_0/Depth_0/image/data"
        self.realsense_color_info_topic = "/device_0/sensor_1/Color_0/info/camera_info"
        self.realsense_depth_info_topic = "/device_0/sensor_0/Depth_0/info/camera_info"
        self.realsense_sync_threshold_ms = 33.0
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        if str(g.source_path).lower().startswith("ros2://"):
            g.source_path = str(g.source_path)
        else:
            g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser, fill_none=False):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.tile_size = 16
        self.sh_impl = "gsplat"  # {"python", "gsplat"} -- default flipped 2026-07-17, see
        # docs/superpowers/reports/2026-07-17-phase-c-sh-default-flip-findings.md
        self.near_plane = 0.01
        self.far_plane = 1e10
        self.radius_clip = 0.0
        self.eps2d = 0.3
        self.sh_color_clamp_mode = "clamped"  # {"clamped", "lower_only"}; diagnostic-only, do not change default
        self.render_mode = "RGB"
        self.absgrad = False
        self.rasterize_mode = "classic"
        super().__init__(parser, "Pipeline Parameters", fill_none)


RENDERER_OPTIMIZATION_DEFAULTS = {
    "gsplat_sparse_grad": True,
    "sparse_mode_detach_sh_dir": True,
    "compile_sh": False,
}


def add_renderer_optimization_args(parser):
    """Register renderer-only overrides without replacing saved cfg values."""
    group = parser.add_argument_group("Renderer Optimization Overrides")
    for name in RENDERER_OPTIMIZATION_DEFAULTS:
        group.add_argument(
            f"--{name}", default=None, action=BooleanOptionalAction
        )


def apply_renderer_optimization_params(pipeline, source):
    """Apply optimization-owned settings that change renderer semantics."""
    for name, fallback in RENDERER_OPTIMIZATION_DEFAULTS.items():
        value = getattr(source, name, None)
        setattr(pipeline, name, fallback if value is None else bool(value))
    return pipeline


def get_default_pipeline_params():
    """Build a PipelineParams-extracted GroupParams object with every real
    default field set. For ad-hoc render call sites (eval/debug tooling)
    that need a pipe object without a full training-argument CLI context --
    guarantees these defaults can never silently drift out of sync with
    PipelineParams' real values, since they come from the same class.
    """
    parser = ArgumentParser()
    pp = PipelineParams(parser)
    args = parser.parse_args([])
    return pp.extract(args)

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_lpips = 0.0
        self.lpips_interval = 10
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 25_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        self.optimizer_type = "selective_adam"
        self.gsplat_sparse_grad = True
        self.sparse_policy = "active_set"
        self.sh_update_interval = 16
        self.sh_degree_schedule_rgb_pixels = [
            307_200_000,
            614_400_000,
            921_600_000,
        ]
        self.parallelism_profile = "safe"
        self.selective_adam_allow_dense_grads = False
        self.sparse_mode_detach_sh_dir = True
        self.selective_adam_zero_invisible_grads = True
        self.sparse_energy_global_interval = 500
        self.benchmark_dir = ""
        self.noise_lr = 5e5
        self.noise_max_step = 0.01    # max per-iter MCMC L2 displacement (metres); 0 = unclamped
        self.noise_max_scale = 0.05   # per-axis scale cap for noise covariance; 0 = unclamped
        # gsplat_energy_mcmc (project-default densification strategy)'s own
        # noise clamp -- issue #1039. Deliberately SEPARATE from
        # noise_max_step/noise_max_scale above (which remain the legacy/mcmc
        # strategy's already-tuned values, unchanged by this addition):
        # this strategy's default noise path used to call gsplat's stock,
        # UNCLAMPED inject_noise_to_position with no cap at all, which flung
        # newly-grown/relocated Gaussians tens-to-hundreds of metres outside
        # a real Replica room0 scene (scene_bbox_volume_ratio 173x the
        # camera trajectory's own bbox). A real GPU A/B (Replica room0,
        # matched 6000-iteration recipe) found noise_max_step=0.01 (borrowed
        # verbatim from the legacy strategy) over-constrains this strategy's
        # much larger noise_lr=5e5, costing ~2.0dB test PSNR vs. fully
        # unclamped (22.32 vs 24.37dB) while also reducing
        # scene_bbox_volume_ratio to 13.9x; 0.1m recovers most of that
        # quality (22.92dB) while still cutting the ratio to 25.5x (vs.
        # unclamped's 1338x) -- a small, disclosed, real tradeoff in
        # exchange for eliminating an unbounded, catastrophic escape-plume
        # failure mode. Not yet tuned beyond this one A/B; revisit if a
        # future run finds either the residual PSNR cost or the residual
        # bbox ratio unacceptable for a given scene.
        self.gsplat_mcmc_noise_max_step = 0.1     # metres; 0 = unclamped
        self.gsplat_mcmc_noise_max_scale = 0.05   # metres; 0 = unclamped
        self.scale_kill_threshold     = 0.20  # dead_mask: relocate when max_scale > this AND support low
        self.scale_moderate_threshold = 0.05  # dead_mask: relax opacity gate for splats above this scale
        self.anisotropy_kill_ratio    = 20.0  # dead_mask: kill needles with s_max/s_min > this AND low support; 0 = off
        # issue #385: the remaining GaussianEnergyConfig dead-mask family members.
        # These siblings of the three flags above were declared on the dataclass
        # but had NO flag at all, so compute_legacy_dead_mask() -- which reads all
        # of them and drives real Gaussian birth/death topology on the default
        # gsplat_energy_mcmc strategy -- could only ever see the dataclass default.
        # Each default below is that same dataclass default, so declaring them is
        # behaviour-neutral; they are now merely reachable. (They also match
        # utils/energy_mcmc.compute_dead_mask()'s own signature defaults, the
        # non-policy implementation of the identical rule.)
        self.energy_support_threshold = 0.01   # dead_mask: support-EMA death threshold
        self.energy_scale_moderate_opacity_mult = 4.0  # dead_mask: opacity multiplier for mid-scale kills
        self.energy_scale_dead_support_mult = 1.0      # dead_mask: support multiplier for scale/aniso clauses
        self.energy_utility_quantile = 0.05    # dead_mask: bottom utility quantile killed
        self.energy_use_utility_quantile = True  # dead_mask: enable the utility-quantile clause
        self.anisotropy_reg = 0.01   # weight for max/min scale ratio penalty; 0 = disabled
        self.anisotropy_soft_limit = 10.0  # ratio above which the squared-hinge anisotropy penalty (energy_loss path) activates
        # Needle penalty (campaign C1/C2): targets second-axis collapse (needle <=> s2/s1 small),
        # i.e. (needle_reg_threshold - s2/s1)_+^2. Unlike anisotropy_reg (s_max/s_min) it does
        # NOT penalise healthy flat disks. 0 = off (baseline). energy_loss path only.
        self.needle_reg_weight = 0.0
        self.needle_reg_threshold = 0.3
        # Scale-reg form (campaign C4): "l1" (default/baseline) or "l2" (Frobenius) — L1 is
        # sparsity-inducing and drives minor axes toward zero (anisotropy at the source).
        self.scale_reg_mode = "l1"
        # Dead-mask/utility anisotropy target (campaign C3): "s_max_min" (default) or "s2_s1".
        self.aniso_kill_mode = "s_max_min"
        self.utility_aniso_weight     = 0.1   # anisotropy penalty weight in compute_gaussian_utility
        self.utility_aniso_soft_limit = 10.0  # anisotropy ratio below which no utility penalty
        self.mcmc_noise_stop_iter = 30_000
        self.scale_reg = 0.01
        self.opacity_reg = 0.01
        self.densification_strategy = "gsplat_energy_mcmc"
        self.mcmc_stop_growth_iter = 12000
        self.mcmc_growth_factor_start = 1.05
        # Streaming only: once all source frames are admitted, replace
        # mcmc_growth_factor_start with this value for the rest of the run.
        # 0.0 (default) disables this -- growth_factor_start applies
        # unconditionally throughout, unchanged from today's behavior.
        self.streaming_mcmc_post_ingestion_growth_factor_start = 0.0
        # Optional source-complete override for the canonical RGB-pixel growth
        # cadence. Zero preserves mcmc_grow_interval_pixels in both phases.
        self.streaming_mcmc_post_ingestion_grow_interval_pixels = 0
        # Optional source-complete override for relocation exposure cadence.
        # Zero preserves mcmc_relocate_interval_pixels in both phases.
        self.streaming_mcmc_post_ingestion_relocate_interval_pixels = 0
        self.mcmc_growth_factor_min = 1.002
        self.mcmc_growth_factor_tau = 0.35
        # Canonical live MCMC cadence is measured in realized RGB supervision
        # pixels, not optimizer transactions. These defaults preserve the
        # historical low-end 100/50 cadence as explicit exposure at native
        # 640x480 resolution; the legacy iteration fields below remain for the
        # offline compatibility trainer only.
        self.mcmc_grow_interval_pixels = 30_720_000
        self.mcmc_relocate_interval_pixels = 15_360_000
        self.mcmc_min_optimizer_updates_after_structural_mutation = 2
        self.mcmc_noise_stop_optimizer_updates = 30_000
        self.mcmc_temperature_horizon_optimizer_updates = 30_000
        self.mcmc_grow_interval_min = 100
        self.mcmc_grow_interval_max = 2000
        self.mcmc_grow_tau = 0.35
        self.energy_mcmc = True
        # 10x-damped energy_loss weights (was 0.1 / 0.01). The original values
        # produced "starry-spike" / needle artifacts in novel views under the
        # combined Phase-15 config; an A6-regime sweep (200f/10k/cap100k, see
        # docs/BASELINE.md) found 10x reduction best on PSNR+LPIPS and visibly
        # fewer spikes. Both terms must drop together (zeroing entropy alone did
        # not help). Costs ~0.3 dB on full-coverage runs where spikes are absent.
        self.lambda_eff_count = 0.01
        self.lambda_opacity_entropy = 0.001
        self.energy_w_alpha = 1.0
        self.energy_w_vis = 2.0
        self.energy_w_grad = 3.0
        self.energy_w_scale = 0.5
        self.energy_w_dead = 1.0
        self.energy_temp_start = 2.0
        self.energy_temp_min = 0.3
        self.energy_temp_tau = 0.4
        self.energy_beta_opacity = 1.0
        self.energy_beta_scale = 0.5
        # Raised 0.005 -> 0.05 (2026-07-06): this is a SEPARATE threshold
        # from mcmc_dead_opacity_start/end (MCMCScheduleConfig) -- that
        # config only feeds the MCMC relocation schedule via
        # utils/strategies/mcmc_strategy.py's sched["dead_opacity_threshold"].
        # train_streaming.py's direct GaussianEnergyConfig construction
        # sites (alpha_dead=getattr(args, "energy_alpha_dead", 0.005)) read
        # THIS flag instead, so compute_dead_mask()'s actual per-Gaussian
        # kill test used the old 0.005 threshold regardless of the earlier
        # schedule fix -- confirmed live: relocated=N went nonzero after
        # that fix, but geometry.pruning.total_pruned stayed exactly 0.
        # Aligned to the same 0.05 value used by opacity_near0_frac
        # (the floater-reporting definition elsewhere in the codebase).
        self.energy_alpha_dead = 0.05
        self.energy_w_support = 2.0
        # MCMC schedule parameters (exposed for ablation studies)
        self.mcmc_relocate_interval_min = 50
        self.mcmc_relocate_interval_max = 500
        self.mcmc_relocate_tau = 0.65
        self.mcmc_cap_growth_power = 2.0
        self.mcmc_cap_interval_strength = 4.0
        self.mcmc_cap_interval_power = 2.0
        self.mcmc_cap_stop_ratio = 0.98
        # 2026-07-06: raised from 0.003/0.010 -- live log evidence on
        # fr3par_hybrid_ctrl showed "relocated=0" at every logged interval
        # for the entire run while opacity_near0_frac (the geometry-report
        # floater metric, thresholded at 0.05 in scene_geometry_metrics.py)
        # climbed monotonically 35%->52% with zero cleanup. Root cause: the
        # actual dead-mask kill threshold (0.010 max) was 5x stricter than
        # the 0.05 threshold this codebase itself uses to call a Gaussian
        # "dead-looking" for reporting -- so real floater mass in the
        # [0.01, 0.05) opacity range was never eligible for pruning at all.
        # Aligned end-threshold with the reporting definition; start kept
        # at the same ~3.3x ratio to end.
        self.mcmc_dead_opacity_start = 0.015
        self.mcmc_dead_opacity_end = 0.050
        self.mcmc_dead_opacity_power = 1.5
        self.mcmc_use_target_deficit = False
        self.mcmc_target_splat_end = 150000
        # Merge A (CapacityPolicy): soft TARGET-N the count-loss / growth deficit
        # aim for, decoupled from cap_max (the hard relocation/insertion ceiling).
        # Default 0 = "use existing behaviour" (byte-identical to before).
        self.streaming_target_capacity = 0
        # Merge B (AnnealClock): finite anneal horizon (iters) for online mode.
        # Decouples the count-loss/schedule anneal from the unbounded loop bound
        # so the target curve can actually ramp to streaming_target_capacity.
        # 0 = neutral (anneal horizon == loop bound, u~=0, no ramp = today's
        # behaviour).
        self.streaming_anneal_horizon = 0
        # Decoupled noise-stop clock for online mode (2026-07-06): like
        # streaming_anneal_horizon, this is a narrow clock separate from
        # streaming_online_schedule_horizon -- it stops MCMC xyz-noise
        # injection at a real, finite iteration WITHOUT also annealing
        # growth-stop/dead-opacity-ramp/densify-until (the over-prune risk
        # streaming_online_schedule_horizon's docstring warns about). Motivation:
        # online_mode's unbounded loop bound otherwise also pins mcmc_noise_stop_iter
        # to 2^31 (apply_online_schedule_overrides), so noise perturbation never
        # stops -- fine details/text never get a clean, noise-free convergence
        # window late in training. 0 = legacy unbounded (noise never stops,
        # today's behaviour, opt-in only). Set to a real iteration count (e.g.
        # matching streaming_online_schedule_horizon or a bit later) to give the
        # run a quiet tail once real coverage is established.
        self.streaming_noise_stop_horizon = 0
        # Growth-stop clock, decoupled from streaming_noise_stop_horizon
        # (2026-07-18, split after real-GPU testing found the coupled behavior
        # starves capacity -- see apply_online_schedule_overrides's docstring
        # for the full floor3 finding). 0/unset falls back to
        # streaming_noise_stop_horizon (the prior coupled behavior, fully
        # behaviour-neutral if that's also unset). MCMC growth/relocation is an
        # ongoing densification mechanism independent of new-frame arrival, so
        # this should generally be set MUCH later than streaming_noise_stop_horizon
        # (close to the run's actual end, leaving only a short final polish
        # tail) rather than at the point mapping/ingestion completes.
        self.streaming_growth_stop_horizon = 0
        # Adaptive schedule horizons (2026-07-18): instead of hand-picking a
        # fixed --streaming_noise_stop_horizon / --position_lr_max_steps /
        # anneal horizon per scene (error-prone -- e.g. floor3's real
        # mapping-end point is ~iter 5700 out of a 60000-iteration budget,
        # nothing about that is knowable in advance without a prior run),
        # detect the real mapping-end point live: once
        # streaming_adaptive_mapping_end_threshold iterations pass with no
        # new frame ingested (_last_ingest_iter stale), treat mapping as
        # ended and ONCE snap mcmc_noise_stop_iter / mcmc_stop_growth_iter /
        # the count-loss anneal clock / the position-LR schedule's max_steps
        # to that iteration -- giving every scene a clean, noise-free
        # refinement tail regardless of its size, with no manual tuning.
        # DEFAULT OFF (0 = disabled via streaming_adaptive_schedule_horizons
        # below) pending real-GPU verification.
        self.streaming_adaptive_schedule_horizons = False
        # Small threshold (iterations of no new frame) confirming mapping
        # has actually ended, not just a momentary admission-gate gap.
        # Deliberately much smaller than streaming_refinement_iterations
        # (default 50000, which governs when the WHOLE RUN stops) -- this
        # only governs the schedule transition, not termination.
        self.streaming_adaptive_mapping_end_threshold = 1000
        self.mcmc_target_q_start = 0.05
        self.mcmc_target_q_end = 0.85
        self.mcmc_target_tau = 0.45
        # Taming-3DGS strategy parameters. These are inert unless
        # --densification_strategy is set to "taming" or "hybrid".
        self.taming_budget = -1.0
        self.taming_budget_mode = "final_count"
        self.taming_cams = 3
        self.taming_score_interval = 100
        self.taming_min_opacity = 0.005
        self.taming_prune_stop_iter = 3200
        self.taming_view_importance = 50.0
        self.taming_edge_importance = 50.0
        self.taming_mse_importance = 50.0
        self.taming_grad_importance = 25.0
        self.taming_dist_importance = 50.0
        self.taming_opacity_importance = 100.0
        self.taming_depth_importance = 5.0
        self.taming_loss_importance = 10.0
        self.taming_radii_importance = 10.0
        self.taming_scale_importance = 25.0
        self.taming_count_importance = 0.1
        self.taming_blend_importance = 50.0
        # torch.compile configuration (off by default — only opt-in for ablation)
        self.compile_mode = "off"
        # dynamic=True: the streaming/MCMC Gaussian count N changes every step, so
        # a static compile recompiles on every shape change (multi-second GPU-idle
        # stalls). Keep dynamic shapes on by default; configure_torch_compile also
        # force-enables it for reduce-overhead.
        self.compile_dynamic = True
        self.compile_after_iter = 12000
        self.compile_growth_margin = 500
        self.compile_sh = False
        self.compile_energy = False
        self.compile_utility = False
        # SSIM backend (per-iteration launch-bound critical path).
        #   --fused_ssim   : fused_ssim HIP kernel (~16x faster SSIM fwd+bwd;
        #                    +29% it/s gfx1201, +81% gfx1151, value bit-identical).
        #                    DEFAULT ON; set_ssim_mode("fused") falls back to the
        #                    reference conv SSIM if the package is unavailable.
        #                    Disable with --no-fused_ssim.
        #   --compile_loss : torch.compile(dynamic=True) the conv SSIM (no win on
        #                    ROCm; off by default, mutually exclusive with fused).
        self.fused_ssim = True
        # GPU/torch voxel-downsample + occupancy hash (bit-exact numpy parity).
        # Fully wired (scene/streaming_scene.py's _gpu_occupancy_enabled path)
        # but benchmarked slower than the CPU/numpy path (10.4->11.8ms per
        # docs/perf-r9700-gpu-frame-prep.md) -- off by default for that
        # reason, not because it's unimplemented.
        self.gpu_occupancy = False
        self.compile_loss = False
        # GPU-resident fast path for per-frame depth insertion (Phase 1 perf).
        #   --gpu_frame_prep : keep render_alpha/render_depth occupancy masking,
        #                      backprojection, finite-diff normals and grazing/edge
        #                      masks on the GPU instead of copying full-res render
        #                      tensors to CPU + numpy every ingested frame. The CPU
        #                      numpy path remains the byte-for-byte default fallback
        #                      when this is off. Only the tiny strided stable/temporal
        #                      depth masks still run on CPU (deliberate follow-up).
        self.gpu_frame_prep = False
        # Normal-estimation backend for the --gpu_frame_prep insertion path.
        #   finite_diff : pixel-space central-difference normals (default, matches
        #                 the numpy CPU path within float32 tolerance).
        #   kornia      : kornia.geometry.depth.depth_to_normals on the full depth
        #                 map, sampled at the strided grid. Pure-torch / ROCm-safe.
        #                 NOT bit-identical to finite_diff (full-res Sobel gradient
        #                 vs strided central diff) — the grazing threshold
        #                 (--streaming_max_view_angle) may need re-tuning. Only
        #                 active when --gpu_frame_prep is on.
        self.gpu_frame_prep_normals = "finite_diff"
        # WS2 M1 (docs/2026-06-30-gpu-resident-ingestion-plan.md): requires
        # --gpu_frame_prep. Ports the temporal depth-consistency check
        # (_sample_temporal_depth_mask) to a dense torch grid computation
        # (_sample_temporal_depth_mask_gpu) instead of a numpy flat-array
        # gather, and fuses the stable/render-occupancy/temporal/edge masks
        # into ONE final .cpu().numpy() download instead of the ~4 separate,
        # interleaved ones. Default off: byte-for-byte identical to
        # --gpu_frame_prep alone when this stays False. See
        # tests/test_temporal_depth_mask_gpu_parity.py and
        # tests/test_gpu_frame_resident_fused_mask_parity.py.
        self.gpu_frame_resident = False
        # P3 amortized growth: stage streaming depth-insertions in a side buffer and
        # materialize them into params/optimizers with a single torch.cat per flush,
        # instead of one O(N_total) cat-storm per insert call. The per-insert cat path
        # is the fallback. Only the model-internal staging session
        # (begin_staged_inserts / flush_staged_inserts) honours this flag — outside a
        # session add_points_as_gaussians is bit-identical to the legacy path.
        # Default ON: -11% iter_ms on gfx1151 at 30k iters.
        self.staged_gaussian_insert = True
        # P5 async prefetch (offline replay): a daemon thread pre-decodes upcoming
        # frames' RGB+depth into a bounded LRU cache so CPU decode overlaps GPU
        # mapping (offline replay is otherwise single-thread synchronous). Also
        # removes the redundant 2-3x decode of the same frame within an iteration.
        # Deterministic (keyed by frame.index); only applies to non-lazy offline
        # sources (TUM/ScanNet/RGBD/Replica), never live ROS2.
        # Default ON: combined with staged_insert gives -13% on gfx1151.
        self.streaming_prefetch = True
        self.streaming_prefetch_depth = 3
        # Number of concurrent decode-ahead worker threads (2026-07-27). Pillow's
        # JPEG decode (libjpeg-turbo, the codec used by most RGB sources here)
        # releases the GIL during the actual decode loop, so multiple threads
        # racing over the look-ahead window give real parallel decode, not just
        # GIL-serialized ones. 1 preserves the original single-thread behavior.
        # Clamped to streaming_prefetch_depth (no point having more workers than
        # positions in the look-ahead window).
        self.streaming_prefetch_workers = 4
        # Floor on spf after coverage-enforcement arithmetic (0=disable).
        # LOGGING-ONLY as of 2026-07-04 — the coverage-enforcement branch that
        # used to consult this floor to cap admitted frame count and re-stride
        # the trajectory no longer restricts anything; the unified rolling
        # idle-based termination rule (main training loop) governs coverage
        # instead. Retained only so the informational log line at the
        # MappingScheduler call site can still report what the naive
        # steps_per_frame×n_frames≤iterations arithmetic would have required.
        self.streaming_min_steps_per_frame = 50
        # Adaptive optimizer-exposure-per-insertion (issue #109 rescoped
        # Phase 2, arm E3). Default OFF — every prior run's cadence is
        # unaffected. When enabled, the FIXED --streaming_steps_per_frame
        # cadence (or its coverage-clamped effective_spf, see the
        # min_steps_per_frame docstring above and CLAUDE.md's W3-38
        # correction — that clamp is NOT logging-only for a non-online_mode
        # registry-hit scene) governs only the FIRST frame release; every
        # subsequent release interval is instead sized from the PREVIOUS
        # frame's real Phase-2 depth-insertion count via
        # FrameScheduler.set_next_interval() (eligsir/mapping/scheduler/
        # frame_scheduler.py): next_steps = clamp(gain * n_inserted,
        # streaming_adaptive_steps_min, streaming_adaptive_steps_max). A
        # frame that inserts nothing gets the floor (never an unboundedly
        # short interval); a frame with heavy new geometry gets more
        # optimizer exposure before the next frame arrives. This is the
        # literal "steps scaled by per-frame insertion count" mechanism
        # named in issue #109's 2026-08-12 rescope comment, distinct from
        # the fixed-multiplier E1/E2 arms (2x/5x --streaming_steps_per_frame
        # + matching --iterations, the same mechanism Phase 0's arm B used —
        # see docs/architecture/2026-08-10-w3-66-insertion-onoff-2x2.md).
        self.streaming_adaptive_steps_per_insertion = False
        # Steps granted per inserted Gaussian, before min/max clamping.
        self.streaming_adaptive_steps_gain = 1.0
        # Floor on the adaptive interval (also the effective baseline for a
        # zero-insertion frame). Matches --streaming_steps_per_frame's E0
        # value by convention, not enforced.
        self.streaming_adaptive_steps_min = 20
        # Ceiling on the adaptive interval, so one enormous insertion cannot
        # stall frame admission for an unbounded number of iterations.
        self.streaming_adaptive_steps_max = 400
        # KNN-density floater kill (Phase A): mark a large, low-support Gaussian dead
        # if it is spatially isolated — its self-KNN nearest-neighbour distance
        # (simple_knn distCUDA2) exceeds this multiple of the scene-median spacing.
        # Catches white-streak/blue-dot floaters before the visibility-EMA decays.
        # 0 = disabled (default); typical enabled value ~3.0. Applied additively in
        # the MCMC maintenance prune, outside the parity-tested dead-mask clauses.
        self.streaming_knn_isolation_kill_mult = 0.0
        # Support (visibility-EMA) gate for the KNN-isolation kill above: an isolated,
        # large Gaussian is only marked dead when its visibility_ema is BELOW this.
        # Private to the KNN-isolation site — deliberately NOT the same knob as
        # `streaming_shadow_utility_support_threshold` (which feeds
        # EvidenceBuilder.build_summary's under-supported-mass term, a different
        # consumer), and not the same as GaussianEnergyConfig.support_threshold
        # (the parity-tested dead-mask clauses). 0.01 matches those siblings, so
        # the default is behaviour-identical to the previously-hardcoded literal.
        self.streaming_knn_isolation_support_threshold = 0.01
        # Phase B: per-frame insertion scale source. "knn" (default) = local-density
        # spacing from the simple_knn distCUDA2 self-KNN over the frame's candidate
        # batch (principled: splats sized to actual neighbour spacing); "footprint" =
        # analytic depth-pixel footprint (z/focal*stride); "hybrid" = geometric mean.
        # Feeds the same iso/aniso/parallax scale logic. Default is knn: on TUM fr3 it
        # gave +0.135 dB / better depth-RMSE / ~15% fewer Gaussians vs footprint —
        # within the run-to-run noise floor (gsplat atomic backward; ~0.1-0.3 dB,
        # ~11% n_live) so not provably significant, but the local-density sizing is
        # the principled choice and trends the right way. Override with
        # --streaming_insert_scale_mode footprint.
        self.streaming_insert_scale_mode = "knn"
        # Persistent web viewer. Disabled by default; use --web-viewer to enable.
        self.web_viewer_enabled = False
        self.web_viewer_port = 6010
        self.web_viewer_host = "0.0.0.0"
        self.web_viewer_backend = "process"
        self.web_viewer_cache_dir = ""
        self.web_viewer_image_interval = 100
        self.web_viewer_fixed_camera = False
        self.web_viewer_keep_alive = True
        self.web_viewer_scene_cache_interval = 0
        self.web_viewer_scene_cache_keep = 3
        self.record_video = False
        self.record_video_cameras = ""
        self.record_video_interval = 100
        self.record_video_fps = 30
        self.record_video_crf = 23
        self.record_video_preset = "veryfast"
        # Training-progress video: render a fixed camera every N iters,
        # write training_progress.mp4 at end of training.
        self.progress_video_interval = 0     # 0 = disabled (default off; enable with --progress_video_interval 200)
        self.progress_video_fps = 10         # output FPS
        self.scalar_log_interval = 100
        # Supervision-tile evidence probe (#310 items 4-5 / #611 Phase 0).
        # NxN lattice over the supervision image; per-tile RGB L1 residual and
        # dispersion are logged to TensorBoard under `supervision_tile/*`.
        # 0 = off (default). DIAGNOSTIC ONLY: it never enters the loss and no
        # scheduler may read it. Reuses the residual the step already computed
        # -- no extra render -- and transfers once, only on iterations that
        # already sync for scalar logging. Max 16 (MAX_LATTICE_DIM).
        self.supervision_tile_probe_grid = 0
        self.geometry_log_interval = 500
        self.sfm_anchor_interval = 2000
        self.strategy_log_interval = 500
        self.mcmc_control_log_interval = 500
        super().__init__(parser, "Optimization Parameters")


class StreamingParams(ParamGroup):
    def __init__(self, parser):
        # Enable pure online mapping mode (overrides global iteration limits)
        self.online_mode = False
        # Enable streaming replay mode
        self.streaming_replay = False
        # Offline TUM-only sensor-agreement treatment.  It remains off unless
        # an approved experiment explicitly selects an identity or motion arm.
        self.streaming_motion_depth_confidence_mode = "off"
        self.streaming_motion_depth_confidence_native_tolerance_px = 1.0
        self.streaming_motion_depth_confidence_min_weight = 0.05
        self.streaming_input_fps = 30.0
        # Opt-in canonical control-plane debug trace. Relative paths resolve
        # below model_path. Empty keeps the live hot path free of file I/O.
        self.causal_decision_trace = ""
        # Opt-in exact-B4 readiness diagnostics sampled by the existing
        # low-rate progress sink. Unlike decision tracing, this writes no
        # per-tick JSONL records and is suitable for feed qualification.
        self.causal_ready_diagnostics = False
        # Opt-in exact pre-update PSNR for the primary supervision view. This
        # adds one score reduction/readback per scalar_log_interval and is
        # intended for diagnostic/evaluation runs, not the default live path.
        self.streaming_track_primary_psnr = False
        # Low-rate scalar status only; decision/mutation JSONL remains opt-in.
        self.causal_progress_interval_s = 10.0
        # Dataset/rosbag transport queue used by the independent causal source
        # clock. Transport overflow stays distinct from core semantic skipping.
        self.causal_source_buffer_frames = 256
        # ThreadPoolExecutor workers for decoding source frames. On 1920x1440
        # ScanNet++ RGB-D sources at 30 fps, 2-thread decode is bottleneck
        # causing mapper idle 84% and frame drops. Increase for starvation.
        self.causal_decode_workers = 2
        # --causal_decode_workers_policy (eligsir/cli/parser.py) is the
        # opt-in dynamic sizing for the above; not declared as a `self.`
        # attribute here (same reason as --streaming_payload_budget_policy
        # above it in this file) because ParamGroup auto-registers a plain
        # `--<attr>` flag with no `choices=` restriction for every attribute
        # set in this class, and this flag needs {fixed, cpu_scaled}
        # validation. "fixed" (default) leaves causal_decode_workers exactly
        # as set/parsed -- byte-identical to pre-existing behavior.
        # "cpu_scaled" derives the worker count from the source's native
        # resolution and available CPU cores (see
        # eligsir.runtime.payload_budget.resolve_decode_workers), floored
        # at causal_decode_workers and capped at half the CPU core count
        # (minimum 2), to address the starvation documented above without
        # requiring every high-resolution recipe to hand-tune the flag.
        # --causal_thread_pinning_policy (eligsir/cli/parser.py) is
        # likewise not declared as a `self.` attribute here (same
        # ParamGroup auto-registration conflict as --causal_decode_workers_
        # policy and --streaming_payload_budget_policy above): it needs
        # {unbound, pinned} `choices=` validation, which a bare
        # auto-registered flag would not enforce. "unbound" (default) is
        # byte-identical to pre-existing behavior; "pinned" pins cv2's
        # internal thread pool and the process's BLAS thread-pool env vars
        # to 1 once at startup (eligsir.runtime.thread_pinning.
        # pin_decode_threads), addressing the CPU oversubscription measured
        # on scn5_decode8 (see docs/superpowers/specs/2026-09-08-residency-
        # budget-design.md, "Addendum 2": GPU util 23.4% mean, CPU process
        # 1012% mean/1886% peak of 3200% possible, views/s unchanged 60.3
        # vs 61.8 at decode_workers=2).
        # Issue #1273 P0: bounded value-scored thinning of the causal source
        # buffer's ARRIVAL boundary. "fifo_drop_newest" (the default) is the
        # pre-existing, unchanged drop-newest behavior; "value_ranked"
        # preemptively refuses low-novelty arrivals so headroom survives for
        # genuinely novel viewpoints under sustained throughput deficit.
        # Opt-in only: value-based refusal is REAL_TIME_DROP semantics and
        # must never be silently applied to a deterministic replay.
        # NOTE (design doc §9): this phase provably CANNOT bound mapper lag --
        # at saturation the only available decision is admit-or-refuse the
        # newest arrival, with no way to evict an old resident. Read
        # report.json's ingress_max_observed_buffer_lag_s for the unfixed half.
        self.causal_ingestion_policy = "fifo_drop_newest"
        # Bounded ring of ADMITTED records the novelty score is measured
        # against (never the in-flight decode set, which oscillates 0<->1 at
        # saturation and would make the mechanism inert by construction).
        self.causal_ingestion_history_frames = 64
        # Metres of translation considered equivalent to one radian of
        # rotation when combining the two into one pose-novelty distance.
        self.causal_ingestion_rotation_weight = 0.25
        # Threshold = threshold_scale * pressure**pressure_exponent * EMA(novelty).
        # At zero buffer depth the threshold is 0 (admit everything).
        self.causal_ingestion_pressure_exponent = 2.0
        self.causal_ingestion_threshold_scale = 1.0
        # Specified, recorded, and deliberately INERT at this phase so P1's
        # real eviction inherits the config surface instead of re-litigating it.
        self.causal_ingestion_lag_budget_s = 0.0
        # LEGACY, confirmed dead (2026-07-04): only read by FrameScheduler's
        # "unknown ingestion_mode string" fallback branch
        # (eligsir/mapping/scheduler/frame_scheduler.py), which is
        # unreachable in every real call path — train_streaming.py always
        # pre-resolves ingestion_mode to one of the three canonical values
        # via resolve_ingestion_mode() before constructing FrameScheduler, so
        # this kwarg can never actually change behavior. Not referenced by
        # any spec/launch script. Kept only for CLI back-compat (does not
        # error if passed); use --streaming_ingestion_mode instead.
        self.streaming_wallclock = False
        # Release one frame every N training iterations — genuinely still
        # controls iter_based ingestion pacing for non-online_mode runs. What
        # is LOGGING-ONLY as of 2026-07-04 is specifically the coverage-
        # enforcement auto-adjustment of this value (see streaming_min_steps_per_frame):
        # MappingScheduler no longer mutates this at runtime based on
        # steps_per_frame×n_frames≤iterations arithmetic; only the informational
        # log line remains. Globally, "steps per frame" as a CONTROLLING
        # coverage/pacing lever is incoherent for a source whose length may
        # not be fixed at all (live camera) — see the unified rolling
        # idle-based termination rule in train_streaming.py's main loop,
        # which replaces it for that purpose.
        self.streaming_steps_per_frame = 150
        # Default ON. As of 2026-07-04, train_streaming.py resolves the scene
        # identity (SCENE_SOURCE env var, falling back to source_path) against
        # the eval/scenes.json registry BEFORE honoring this flag: if the
        # scene resolves to a real, known trajectory length (any bag-replay
        # job — bag replay is implemented via ros2://live, so this includes
        # every "live-looking" job whose scene is actually a finite, known
        # bag), coverage is genuinely enforceable and this flag's intent is
        # honored. If the scene does NOT resolve (unregistered, or a truly
        # unbounded live camera with no fixed length), enforcement is forced
        # OFF regardless of this flag's value — n_frames has no real meaning
        # there, so "enforcing" it would just be confidently wrong (this was
        # a real, confirmed bug before the 2026-07-04 fix: an aspirational
        # comment here previously claimed this classification already
        # existed, but no such guard was ever implemented).
        # NOTE: what "full coverage" actually means at the mechanism level
        # also changed 2026-07-04 — see streaming_refinement_iterations. This
        # flag's steps_per_frame-shrinking arithmetic is now informational
        # only (logged); the unified rolling idle-based termination rule is
        # what actually guarantees coverage.
        self.streaming_enforce_full_coverage = True
        # Ingestion pacing mode: auto | iter_based | dataset_fps | wallclock_strict
        # auto            — offline replay uses iter_based, ros2://live uses wallclock_strict,
        #                   generic online mode uses dataset_fps
        # iter_based      — deterministic, release every streaming_steps_per_frame iters
        # dataset_fps     — simulated clock; train as many iters as possible per real-time second;
        #                   cap arrivals at streaming_input_fps_cap (never drops frames)
        # wallclock_strict — true wall-clock pacing; drops frames when training is slow
        self.streaming_ingestion_mode = "auto"
        self.streaming_input_fps_cap = 30.0       # max frame rate for dataset_fps / wallclock_strict
        self.streaming_max_frames = 0             # 0 = all frames in dataset
        self.streaming_frame_stride = 1           # Release every Nth frame from the source
        # NOTE (2026-07-04): NOT redundant with streaming_source_stride below — they
        # act at different pipeline stages and compose (both applied = strided twice).
        # streaming_frame_stride is consumed INSIDE dataset-reader construction
        # (utils/streaming_frames.py, 6 call sites: TUM/ScanNet/RGBD/etc constructors
        # each take frame_stride= and subsample their internal frame/association list
        # before the source object exists). streaming_source_stride (below) wraps the
        # ALREADY-BUILT frame_source generically post-hoc via _StridedFrameSource
        # (train_streaming.py, applied after construction), so it works even for
        # source types with no native stride support (e.g. ros2://live). Prefer
        # streaming_source_stride for ad-hoc/uniform striding across any source type;
        # use streaming_frame_stride only when you need the reader itself to never
        # even load the skipped frames (e.g. to save I/O on huge datasets).
        self.streaming_initial_frames = 5         # frames used for bootstrap point cloud + init
        # Intelligent seed clustering: voxel-centroid-cluster the bootstrap point
        # cloud so overlapping multi-frame backprojections collapse to one averaged
        # Gaussian per cell. Critical for dense sensors (e.g. Orbbec ToF at
        # depth_stride=4) where the raw seed can exceed cap_max before training even
        # starts -> MCMC churns at-cap -> scene implodes. Default 0.04 m (one
        # Gaussian per 4 cm cell at bootstrap); set 0.0 to disable (raw seed).
        # NOTE: changes the initial point cloud (hence absolute PSNR) for ALL
        # streaming sources; holdout_hash is unaffected (test selection is
        # admission-independent), but re-baseline cross-run PSNR comparisons.
        self.streaming_seed_voxel_size = 0.04     # metres; 0.0 = off
        # Warmup: until this many train cameras have arrived, suppress depth
        # insertion + MCMC growth (plain photometric refinement only) so the model
        # doesn't grow/overfit on too few early views. 0 = disabled (no warmup gate).
        self.streaming_warmup_min_frames = 0
        # Page each admitted keyframe's full-res GPU image tensor back to CPU once it
        # falls outside the recency window (train_cameras/replay_buffer/global_reservoir
        # never evict the Camera object itself, only its image -- see
        # scene/cameras.offload_camera_image docstring). Without this, GPU memory for
        # camera images grows unboundedly with (admitted-frame-count x resolution),
        # independent of --cap_max -- catastrophic on high-resolution sources over long
        # trajectories (e.g. ScanNet++ iPhone-LiDAR 1920x1440 x thousands of frames hit
        # 100+GB GTT while N sat comfortably under cap_max). Default on; byte-identical
        # training math, only affects which device an image tensor lives on between
        # uses (paged back via prepare_camera_for_render before every training render).
        self.streaming_offload_camera_images = True
        # Issue #279/#1259/#1342: decode a just-admitted frame's native RGB/depth
        # on a background thread ahead of the tick that first draws it for
        # supervision (`eligsir/runtime/production_application.py`'s
        # `_maybe_schedule_admission_prefetch`), closing the admission-to-first-
        # draw gap `record_candidate` previously left as a guaranteed cold-decode
        # cache miss. Only touches WHEN the native pixels are decoded, never
        # WHAT is decoded/rendered -- disabling reproduces the exact pre-existing
        # cold-decode-on-first-draw path. `causal_mapping`/`CanonicalMappingRuntime`
        # only; legacy `train.py`/`train_streaming.py` paths are unaffected.
        # Default OFF (supervisor decision, issue #279/#1259 throughput cluster
        # wrap-up): a real fr3 causal_replay/tum_rgbd GPU A/B measured this
        # mechanism as a correct no-op on that recipe (m_tick_camera_materialize
        # 22.1ms -> 21.9ms, cache hit/miss counts identical) -- PixelReloadToken
        # is only ever constructed by `eligsir/runtime/replay.py` (a
        # deterministic non-ROS qualification helper reachable only via
        # `eligsir/cli/session.py`), never by the causal_replay/tum_rgbd
        # adapter this session's live-camera/bag-replay/TUM goal scenarios
        # actually exercise (`CausalReplayDriver`'s own `decode_executor`,
        # #1339, already overlaps the only decode on that path). Flip back to
        # True only for a workload that is confirmed to build PixelReloadToken
        # handles at admission time -- see
        # docs/architecture/2026-08-31-issue279-admission-prefetch-attempts.md.
        self.streaming_admission_prefetch = False
        # Issue #279/#1342 follow-up (2026-09-04): materialize a just-admitted
        # train frame's supervision tier AT ADMISSION and write it through to
        # the HOT cache, so the fresh view's first supervision draw is a HOT
        # hit instead of the guaranteed miss it is today (`record_candidate`
        # builds no tier; `activate_supervision` only runs after the first
        # materialize). A per-draw trace on rtx4090/fr3 measured that first
        # draws are ~54% of all camera-materialize misses -- a ~25% floor no
        # cache size or COLD promotion can remove (see
        # docs/superpowers/specs/2026-09-03-throughput-recovery-post-p0-gates-design.md).
        # Only meaningful under a CONCRETE --streaming_mapping_resolution_tier:
        # with `adaptive` the tier is not known at admission and a guess can
        # only miss (the 2026-08-31 attempt-2 regression), so the runtime
        # refuses the combination. Default False: opt-in, behavior-neutral.
        self.streaming_admission_prematerialize = False
        # Issue #1670 Phase A (2026-09-06): evaluate the per-Gaussian
        # regularizers (active_reg[+young]) ONCE per optimizer step instead
        # of once per view. Measured on the same rtx4090c/fr3 prepared/
        # READY-B4 recipe as the flag above: `m_loss_regs_gaussian` alone is
        # 241 launch-class events/tick (60/view) and 7.8s of a 17.1s
        # `m_loss_assembly` window -- quantities that do not depend on the
        # view except through the visibility mask, since every term is a
        # masked MEAN of a view-independent per-Gaussian quantity. When ON
        # (true_batch_executor.py, B4 groups only, sparse_active_set,
        # energy_loss_enabled off), the executor computes per-Gaussian
        # weights `W_i = sum_v mask_v[i] / n_v` ONCE (`W` sums to the
        # non-empty-view count) and calls
        # `losses.assemble_step_regularizers` once for the whole group
        # instead of `assemble_loss` doing it per view (each view's
        # `LossAssemblyInputs.skip_gaussian_regularizers=True`). Exact
        # parity with the per-view loop is pinned by
        # tests/test_step_regularizers.py -- see
        # .superpowers/sdd/2026-09-03-throughput-recovery-tracks-1-2/
        # step-regularizers-brief.md. Default True since 2026-09-06 (PR
        # #1741): the rewrite is exact (value and gradients pinned against
        # the per-view loop for B in 1..8, multi-group and skipped-view
        # cases), and it measured +16% views/s at B4 on rtx4090c; the
        # 2026-09-04 ROCm validation (r9700, Strix Halo) ran with it on.
        # `--no-streaming_objective_step_regularizers` restores the per-view
        # loop (assemble_loss's regularizer branch is untouched either way).
        self.streaming_objective_step_regularizers = True
        # Issue #1670 Phase B (2026-09-06): evaluate the per-pixel RGB
        # (L1+SSIM) and masked-depth supervision terms ONCE over the whole
        # true-camera group's `[B,3,H,W]`/`[B,1,H,W]` stacks instead of once
        # per view. Same rtx4090c/fr3 prepared/READY-B4 recipe as the two
        # flags above: `m_loss_rgb` is 64 launch-class events/tick (16/view:
        # L1 + fused SSIM + bookkeeping), `m_loss_depth` is 72 (18/view) --
        # evaluating both once over the stack turns ~34 launches x B into
        # ~34 launches per step (plus ~2 stack copies), and shrinks the
        # backward mirror the same way. When ON (true_batch_executor.py,
        # `len(group) > 1`, every view `rgb_loss_enabled`, no free-space/
        # appearance-only/tile-probe flags active, TB scalar logging not due
        # this iteration), the executor builds `renders`/`gts` stacks once
        # (reusing the memoized gt-finite validator) and calls
        # `batched_pixel_terms.assemble_batched_pixel_terms` once for the
        # whole group, passing each view's `PixelTermsView` into
        # `LossAssemblyInputs.precomputed_pixel_terms`. Depth batching is a
        # further, independent condition (uniform `depth_loss_enabled`
        # across the group, `surface_confidence_depth_loss` off, sensor
        # depth available for every view) -- when it does not hold, RGB
        # still batches and each view's own `PixelTermsView.depth` is simply
        # `None`, so `assemble_loss` falls through to its normal per-view
        # depth computation for that view exactly as before. The depth term
        # is normalised PER VIEW by that view's own valid-pixel count, never
        # over the whole stack (see `batched_pixel_terms`'s module
        # docstring and its CPU parity tests,
        # tests/test_batched_pixel_terms.py). Exact parity with the
        # per-view loop is pinned by tests/test_batched_pixel_terms.py (the
        # function) and tests/test_objective_batched_pixel_terms_wiring.py
        # (the wiring) -- see
        # .superpowers/sdd/2026-09-03-throughput-recovery-tracks-1-2/
        # batched-pixel-terms-brief.md. Default True since 2026-09-06 (PR
        # #1741): parity with the per-view loop within the documented
        # tolerances, +14% views/s at B16 and +22% at B32 on rtx4090c, on in
        # every 2026-09-04 ROCm validation arm;
        # `--no-streaming_objective_batched_pixel_terms` restores the
        # per-view terms. Remaining caveat of the ON state, measured 2026-09-04:
        # behavior-neutral when off (`assemble_loss`'s `m_loss_rgb`/
        # `m_loss_depth` segments are untouched; the executor never builds
        # the stacks or calls `assemble_batched_pixel_terms`).
        #
        # Non-finite granularity (measured on the device, 2026-09-04): with this
        # flag ON, SSIM is one batch-mean scalar shared by every view of the
        # group (`PixelTermsView.ssim`). A NaN pixel in one render therefore
        # COULD poison the whole group -- but the fused SSIM kernel production
        # uses does not propagate a single NaN pixel into the batch mean: on
        # rtx4090c a corrupted secondary render left the shared ssim finite
        # (0.00298) while that view's own Ll1 was NaN, so exactly that view was
        # skipped and the step proceeded, identical to flag OFF. The group-wide
        # abort would only occur on the reference conv SSIM path (fused kernel
        # unavailable). Pinned end to end by `eligsir/optimization/gaussian/
        # tests/test_batched_pixel_terms_gpu.py`; no host sync is added for it.
        self.streaming_objective_batched_pixel_terms = True
        # Issue #399 (P0-C root leaf): per-subsystem GPU memory residency
        # telemetry. Walks the live Gaussian model / optimizer state / camera
        # lists / multires resolution cache each logging interval and attributes
        # torch.cuda.memory_allocated() to named buckets, with an explicit
        # `unattributed` residual -- see utils/gpu_memory_telemetry.py. Pure
        # observability: it never moves, frees, or offloads a tensor and changes
        # no training math. It also introduces NO host/device synchronization
        # (metadata-only walk; the torch.cuda.memory_* readers query the caching
        # allocator's bookkeeping and do not sync), per the #480 invariant that
        # observability must never backpressure training. Default ON because the
        # entire #396/#398/#333 lane is blocked on having this measurement from
        # real long runs; pass --no-streaming_memory_telemetry to disable, in
        # which case the tracker is never constructed and the cost is one
        # `is None` test per logging interval.
        self.streaming_memory_telemetry = True
        # Iterations between residency snapshots. 0 = use --scalar_log_interval
        # (the existing scalar cadence).
        self.streaming_memory_telemetry_interval = 0
        # Hard cap on retained snapshots in report.json["gpu_memory"]["series"].
        # Once exceeded the series is decimated in place (every other sample
        # dropped, stride doubled) so the telemetry's own memory stays bounded
        # as frame history grows -- it must not reproduce the unbounded-growth
        # bug class it exists to measure.
        self.streaming_memory_telemetry_max_samples = 512
        # PR #1783 follow-up: default-on per-run system-performance trace.
        # #1783 made timestamped host CPU/RAM + GPU util/VRAM/GTT/power/temp
        # curves a documented evaluation requirement
        # (docs/EVALUATION_PROTOCOL.md#system-performance-and-input-storage)
        # without shipping an implementation -- only ResearchFlow-wrapped runs
        # got a trace (researchflow/execution.py's TelemetrySampler). A plain
        # causal_mapping or offline train.py run had none. See
        # eligsir/telemetry/system_sampler.py. Default ON (unlike the
        # opt-in duty-cycle telemetry below) because every run, not just
        # specifically-instrumented ones, must carry this evidence; pass
        # --no-system_telemetry to disable, in which case the sampler thread
        # is never started and the cost is zero.
        self.system_telemetry = True
        # Sampling interval in seconds for the background system-performance
        # thread (host/process CPU%, RSS, host RAM, GPU util/mem/GTT/power/
        # temp). 1.0s matches the ResearchFlow TelemetrySampler default.
        self.system_telemetry_interval_s = 1.0
        # --- Issue #562: CPU/GPU duty-cycle timeline + stall taxonomy --------
        # Partitions host wall-clock across a documented taxonomy (see
        # utils/duty_cycle.py) by reusing the EXISTING `_timed()` region
        # vocabulary -- no new measurement path, and NO torch.cuda.synchronize()
        # anywhere, per the #480 observability-never-backpressures invariant.
        # Default OFF (unlike #399's memory telemetry) because #561's lane needs
        # it on specific long runs, not on every run: with it off, the timeline
        # is never constructed and `_timed` pays exactly one module-global load
        # plus one `is not None` branch per region -- no clock read, no dict
        # write, no call.
        self.streaming_duty_cycle_telemetry = False
        # Iterations per rolling window (one TB sample + one JSONL row each).
        self.streaming_duty_cycle_window = 100
        # Hard cap on retained window rows (bounded memory on an hours-long run;
        # oldest are dropped and the drop count reported).
        self.streaming_duty_cycle_max_windows = 4096
        # A single contiguous interval billed to one category at or above this
        # many milliseconds is recorded as a stall event (top-N kept).
        self.streaming_duty_cycle_stall_ms = 250.0
        self.streaming_duty_cycle_max_stalls = 64
        self.streaming_keyframe_window = 120       # minimum keyframe window (floor)
        self.streaming_keyframe_coverage = 0.6    # adaptive: keep >= this fraction of ingested frames in window (0 = disabled)
        self.streaming_replay_buffer = 512        # size of older-frame replay ring buffer
        # Weighted full-cache replay always retains a non-zero chance for every
        # admitted identity. Controllers may increase per-view weights to focus
        # struggling regions without starving the rest of the trajectory.
        self.streaming_replay_global_weight_floor = 0.05
        # Dynamic replay-resolution controller target. C-camera transactions
        # receive C/target seconds so batching is measured as supervised views/s.
        self.streaming_replay_target_views_per_s = 100.0
        # Closed-loop relaxation of that open-loop budget (2026-09-04,
        # tier-budget floor-lock): when even the cheapest *measured* tier costs
        # more than C/target, no tier is ever feasible and the scheduler holds
        # the cheapest tier forever, however well the pipeline keeps up. While
        # freshness reads HEALTHY the effective budget is relaxed x1.25 per
        # decision, capped at this factor; any non-HEALTHY reading resets it.
        # 1.0 disables the relaxation (the pre-2026-09-04 behaviour).
        self.streaming_resolution_budget_relax_max = 4.0
        # Canonical work-plan resolution authority. ``adaptive`` retains the
        # freshness/throughput ladder; a concrete tier is an explicit
        # experiment/debug constraint and applies to mapping as well as replay.
        self.streaming_mapping_resolution_tier = "adaptive"
        # Minimum legal tier for the adaptive mapping ladder. Payload
        # residency remains bounded independently. Eighth is available by
        # default; the controller still selects a tier from measured work.
        self.streaming_mapping_resolution_min_tier = "eighth"
        # issue #1346: the canonical (causal_mapping) work-budget owner's
        # post-refinement transaction class has always forced NATIVE tier
        # with an unbounded (None) transaction budget -- "post-refinement has
        # no realtime constraint; supervision quality is maximal". That is
        # the correct call for a long, fixed-step refinement budget (e.g. the
        # ICRA27 --streaming_post_mapping_refinement_steps recipe), but a
        # real GPU measurement (docs/architecture/2026-08-31-issue1259-*)
        # showed it produces a ~2.9x views/s cliff (~76 -> ~26 views/s on
        # fr3) the instant refinement starts, directly fighting the standing
        # goal of a short (<=2 minute), throughput-bounded refinement tail.
        # "native" (default) preserves this exact historical behavior.
        # "bounded" keeps ResolutionPolicyProvider/WorkBudgetOwner's ordinary
        # freshness/throughput-ladder active during refinement instead of
        # forcing native, so a short refinement tail keeps sustaining
        # replay_target_views_per_s rather than falling off a cliff.
        # "progressive" (follow-up to #1346): behaves like "bounded" for the
        # first `streaming_refinement_progressive_switch_frac` fraction of
        # the refinement step budget (cheap-tier, high-throughput steps
        # while the population still has the most to gain per step), then
        # switches to "native"'s forced-native/unbounded behavior for the
        # remainder (spend the final stretch on full-quality supervision
        # once cheap gains plateau). Only meaningful with
        # --streaming_post_mapping_refinement_steps > 0 (a finite refinement
        # step budget to compute a progress fraction against) -- with a
        # rolling-idle-stop refinement tail (no fixed budget) the progress
        # fraction is undefined and this policy behaves exactly like
        # "bounded" (never crosses the switch point).
        self.streaming_refinement_resolution_policy = "native"
        # See streaming_refinement_resolution_policy's "progressive" case
        # above. Fraction of the refinement step budget (completed/requested)
        # at which the policy switches from bounded-like to native-like
        # behavior. Ignored unless the policy above is "progressive".
        self.streaming_refinement_progressive_switch_frac = 0.5
        # Post-ingest refinement may add a small amount of settled replay
        # depth through the canonical mutation barrier. Zero keeps it disabled.
        self.streaming_replay_depth_max_per_event = 0
        self.streaming_replay_depth_interval = 50
        # issue #1259 far-region-starvation follow-up: the live/replay depth
        # insertion occupancy gate (alpha_threshold, shared with the live
        # per-frame path) treats a pixel as "already covered" once rendered
        # alpha reaches this bar -- a low bar a sparse, coarse-scale far-room
        # region can clear with very few Gaussians (KNN-init scale grows with
        # neighbor spacing), after which that region is permanently excluded
        # from further depth-based densification for the rest of the run,
        # even though the bootstrap seed (which has no occupancy gate at all,
        # since there is no existing render at iteration 0) admitted the same
        # far geometry unconditionally. This is the root cause of the
        # bootstrap-vs-ongoing-insertion asymmetry (see
        # docs/architecture/2026-08-30-issue1259-far-region-insertion-asymmetry.md).
        # These two knobs let the post-source-complete replay-depth-insertion
        # pass above (`streaming_replay_depth_max_per_event`) use a RELAXED
        # occupancy/range policy distinct from the live path's, so under-
        # covered far regions get one more chance right as mapping ends --
        # deliberately scoped to that already-bounded, already-fast pass
        # (interval is in supervision views, not iterations) rather than a
        # long refinement tail. Sentinels: -1.0 (alpha) / 0.0 (max_depth)
        # mean "inherit the live insertion policy's value unchanged" -- the
        # default is fully behavior-neutral even when
        # streaming_replay_depth_max_per_event > 0.
        self.streaming_replay_depth_relaxed_alpha_threshold = -1.0
        self.streaming_replay_depth_relaxed_max_depth = 0.0
        # Replay-tail depth may refine settled geometry more densely than live
        # admission.  Zero inherits the live sparse voxel and is neutral.
        self.streaming_replay_depth_voxel_size = 0.0
        # Tail-only same-holdout PSNR feedback. The weighted replay provider
        # keeps every cached identity reachable at its global floor.
        self.streaming_partition_replay_focus = False
        self.streaming_partition_replay_focus_min_history = 3
        self.streaming_partition_replay_focus_min_views = 2
        self.streaming_partition_replay_focus_low_gain = 0.5
        self.streaming_partition_replay_focus_regression_gain = 1.0
        self.streaming_partition_replay_focus_regression_tolerance_db = 0.25
        self.streaming_partition_replay_focus_max_weight = 8.0
        # Training-only spatial replay focus. Fixed world microcells come from
        # sensor depth/pose; the existing RGB render supplies residuals.
        self.streaming_spatial_replay_focus = False
        self.streaming_spatial_replay_focus_cell_size = 0.5
        self.streaming_spatial_replay_focus_interval = 25
        # Off by default (issue #1259). When enabled, replay-weight
        # actuation switches from frame-averaged debt (default: a frame's
        # weight is derived from the sample-weighted AVERAGE PSNR across its
        # visible cells) to cell-debt weighted (each visible cell's own debt
        # is computed first, then sample-weighted-averaged into the frame
        # weight), so one catastrophically bad cell co-visible with good
        # cells is no longer averaged away before it can raise a frame's
        # replay priority.
        self.streaming_spatial_replay_focus_cell_debt_actuation = False
        # Canonical live MappingApplication payload-retirement bound. Pose
        # records remain in train_cameras while old RGB-D owners are released.
        # When --streaming_payload_budget_policy is "resolution_scaled" (default),
        # None here means the value is derived from source native resolution;
        # legacy default 512 is retained when policy is "legacy".
        self.streaming_live_admitted_payload_max_frames = None
        # Non-reloadable live DEFER observations may retain a finite CPU RGB-D
        # archive. This is separate from active/HOT admitted camera residency.
        self.streaming_live_cold_payload_max_frames = 128
        self.streaming_live_cold_payload_max_bytes = 2_147_483_648
        # 2026-09-07 (`--streaming_post_mapping_refinement_sampling all`
        # catalog collapse, floor3 r9700 `psnr26v12h_orb12_floor3_plain_
        # sampleall_h`): the 128-frame default above is sized for the LIVE
        # mapper's own bounded backpressure/replay needs, not for "all"
        # mode's explicit goal of covering the WHOLE trajectory -- on a
        # non-reloadable in-memory bag source (Orbbec/RealSense ROS2, no
        # `rgb_path`/reload token) a frame evicted past that count is
        # PERMANENTLY unreplayable, and `--streaming_live_cold_payload_
        # max_frames` defaulting to 128 was exactly why that run's "all"
        # catalog collapsed to 128 unique frames and matched the bounded
        # "pool" baseline. Off by default (byte-identical to before this
        # flag existed); `eligsir.runtime.causal_entry` auto-enables it
        # whenever `--streaming_post_mapping_refinement_sampling` is not
        # `"pool"`. When active, removes ONLY the frame-COUNT cap; the
        # frame-BYTES cap (`--streaming_live_cold_payload_max_bytes`,
        # raised to at least `--streaming_observation_db_retain_all_max_
        # bytes`) remains a real, finite ceiling -- see
        # `eligsir.runtime.config_boundary.resolve_cold_payload_bounds`.
        self.streaming_observation_db_retain_all = False
        self.streaming_observation_db_retain_all_max_bytes = 8_589_934_592  # 8 GiB
        # Headroom fractions for residency budget resolver (when policy is "residency").
        # See docs/superpowers/specs/2026-09-08-residency-budget-design.md.
        self.streaming_payload_budget_device_headroom_frac = 0.5
        self.streaming_payload_budget_host_headroom_frac = 0.25
        # Shared pinned-memory ceiling for canonical transfer-owner pools.
        # Large true-camera batches may need a larger explicitly bounded pool.
        # Raised 2026-08-29 (issue #1259): the prior 256MB default crashed a
        # Replica room0 run (larger frames than TUM) by exceeding the pool by
        # ~1.2%. Host RAM headroom is not a real constraint on this project's
        # machines, so the default is now a generous 2GB; per-scene profiles
        # (e.g. configs/profiles/live/canonical_mapping.yml's explicit 640MB)
        # remain the way to raise it further for a specific measured need.
        self.streaming_max_transfer_bytes = 2_147_483_648
        # Canonical live sensor-depth LRU bound; 0 keeps legacy unbounded
        # behavior for non-canonical trainer profiles.
        self.streaming_sensor_depth_cache_max_frames = 0
        self.streaming_keyframe_sampling_weight = 2.0  # bootstrap/keyframe frames over-sampled by this factor
        self.streaming_bootstrap_in_window = True  # keep bootstrap frames in local window until streaming catches up
        self.streaming_global_replay_ratio = 0.3  # fraction of steps drawn from replay buffer
        # Partition-focus sampler probability; wired by StreamingScene and
        # train_streaming's partition-control refresh path.
        self.streaming_partition_focus_weight = 0.0
        # #1723: drive the invasive promote/demote actuation above from the
        # spatial weight-only controller's world-microcell coverage/debt
        # instead of ordinal-partition means. Requires BOTH
        # streaming_partition_focus_weight > 0 (this only changes which
        # signal feeds actuation, not whether actuation runs) and
        # streaming_spatial_replay_focus (the source of that signal). Off by
        # default -- when off, causal_evaluation._measure_partition_control
        # takes its original ordinal-partition path unchanged.
        self.streaming_partition_focus_spatial_substrate = False
        # #1723 v2: unified deficit-round-robin debt allocator, replacing
        # decide_spatial_partition_focus's separate promote/demote logic
        # with one ranked selection. Requires
        # streaming_partition_focus_spatial_substrate (enforced by
        # CausalEvaluationObserver's constructor guard). "Independent" in
        # the sense that the OLD (decide_spatial_partition_focus) and NEW
        # (this debt allocator) actuation shapes are two different
        # implementations that can be A/B'd against each other by toggling
        # this flag while spatial_substrate stays on -- not that this flag
        # works standalone. Off by default.
        self.streaming_partition_focus_debt_allocator = False
        # Legacy #1308 export gate for #116 TARGETED_HISTORY ranking. Replay
        # feedback now records modality-specific evidence independently.
        self.streaming_historical_usefulness_tracking = False
        # Divides the raw per-view loss before usefulness_weight_from_loss's
        # tanh saturation -- calibrate to this run's typical total-loss
        # magnitude (photometric + depth + regularizers) so the transform's
        # saturation point is meaningful rather than saturating immediately
        # or never engaging.
        self.streaming_historical_usefulness_loss_scale = 1.0
        # Opt-in COLD replay trials and physical READY replay turnover. All
        # defaults preserve the existing scorer-only behavior.
        self.streaming_replay_challenger_fraction = 0.0
        self.streaming_replay_challenger_demotion_cadence = 0
        self.streaming_replay_challenger_demotion_limit = 0
        self.streaming_replay_challenger_lease_ticks = 100
        # #802 bounded cohort admission. Off by default until its registered
        # ResearchFlow comparison promotes the policy.
        self.streaming_budgeted_admission = False
        self.streaming_admission_cohort_max_candidates = 4
        self.streaming_admission_cohort_max_source_age = 8
        self.streaming_admission_rank_policy = "combined"
        # Spatial replay window: when enabled, replay-buffer frames are selected
        # by spatial proximity to the *current* camera (pose-sphere that follows
        # the camera) instead of pure recency. Lets revisits re-include old nearby
        # frames. Default False => byte-identical temporal behaviour.
        self.streaming_spatial_window_enabled = False
        self.streaming_spatial_window_radius = 1.5    # meters: cameras within this radius are candidates
        self.streaming_spatial_window_sigma = 0.75    # meters: falloff for exp(-dist/sigma) weighting
        # Debt-weighted global replay: sample replay frames proportional to the
        # training debt of Gaussians in their frustum (sum of max(0, target -
        # support_count)). Requires lifecycle/support counts; falls back to uniform.
        self.streaming_debt_replay = False
        self.streaming_debt_replay_target_support = 10  # support count considered "mature enough"
        # Residual-weighted active-ingestion resampling (2026-07-09): periodically
        # renders each admitted training camera, computes PSNR vs its GT, and biases
        # sampling toward high-residual (struggling) views -- targets seed-fade
        # (the origin/bootstrap view's quality crashing mid-training as new frames
        # keep getting admitted) DURING active ingestion, unlike streaming_debt_replay
        # which only applies once ingestion goes idle. 0 = off (no behavior change).
        self.streaming_residual_replay_weight = 0.0
        self.streaming_residual_replay_interval = 200
        # Depth-based incremental Gaussian insertion (Phase 2)
        self.streaming_insert_from_depth = True
        # Manual insertion-stop horizon, independent of --online_mode (2026-08-10,
        # W3-66 insertion-ON/OFF-vs-optimization-budget 2x2 experiment). The
        # existing --streaming_growth_stop_horizon mechanism only takes effect
        # via apply_online_schedule_overrides(), which requires --online_mode --
        # this flag lets a run stop Phase 2 depth insertion after a fixed
        # iteration WITHOUT online_mode's other behavior changes (unbounded
        # opt.iterations, admission-schedule overrides, etc.), so admission=all
        # runs can isolate "does insertion itself need to stop" from "does the
        # whole online-mode schedule need to change". Frames still admit/track/
        # pose/supervise normally past this point; only new-Gaussian insertion
        # stops. 0 = off (default, no behavior change).
        self.streaming_insert_stop_iter = 0
        self.streaming_depth_stride = 8
        # Sparse-to-dense depth-insertion density schedule (issue #1279 design
        # follow-up, docs/architecture/2026-08-29-staged-coarse-to-fine-depth-insertion.md).
        # "fixed" (default): streaming_depth_stride above is used unconditionally
        # for the whole run -- byte-for-byte identical to every run before this
        # flag existed. "coarse_to_fine": the EFFECTIVE per-call depth_stride
        # ramps geometrically from streaming_depth_sparsity_start_stride (sparse,
        # early mapping) down to streaming_depth_stride itself (the run's normal,
        # dense target -- unless streaming_depth_sparsity_end_stride overrides it)
        # over streaming_depth_sparsity_ramp_frames admitted frames, keyed on
        # frames-ingested-so-far (current_frame_idx), NOT wall-clock or a fraction
        # of a total frame count -- required per the design doc's §5 constraint 3,
        # since the total is unknown for a live/unregistered camera. This local
        # variable is threaded through insert_gaussians_from_frame's existing
        # depth_stride usage (mgrid subsampling, edge/normal finite-difference
        # span, footprint-mode seeded scale) -- never by mutating this attribute
        # itself, so scene/streaming_scene.py's two other bare readers of
        # streaming_depth_stride (bootstrap PCD construction, scene-complexity
        # telemetry) are unaffected, per the design doc's §5 constraint 1.
        self.streaming_depth_sparsity_schedule = "fixed"
        # Coarse starting stride for "coarse_to_fine" (ignored under "fixed").
        self.streaming_depth_sparsity_start_stride = 16
        # Dense ending stride for "coarse_to_fine". 0 (default, sentinel) means
        # "use streaming_depth_stride" -- the ordinary convention that the
        # schedule converges to the run's own configured density rather than a
        # second, independently-tunable target. Ignored under "fixed".
        self.streaming_depth_sparsity_end_stride = 0
        # Admitted frames over which the geometric ramp completes; the schedule
        # holds at streaming_depth_sparsity_end_stride once
        # current_frame_idx >= this value. Ignored under "fixed".
        self.streaming_depth_sparsity_ramp_frames = 500
        self.streaming_max_new_gaussians_per_frame = 1000
        self.streaming_insert_voxel_size = 0.02
        # #109/#192 deterministic sparse reliable source-depth seed policy.
        self.streaming_sparse_seed_min_confidence = 0.0
        self.streaming_sparse_seed_voxel_size = 0.02
        self.streaming_sparse_seed_max_per_frame = 1000
        self.streaming_min_depth = 0.1
        self.streaming_max_depth = 8.0
        # Compatibility mode: make depth insertion obey cap_max. Default False
        # because SLAM map size is unknown and depth insertion is sensor-driven.
        self.streaming_depth_respects_cap = False
        # Out-of-bounds Gaussian pruning at global maintenance cadence.
        # Prunes Gaussians outside the camera-trajectory AABB expanded by 3×max_depth.
        self.streaming_bbox_prune = True
        # Local MCMC: restrict noise/reloc to visible/active Gaussians only
        self.streaming_mcmc_local_only = True
        # Canonical fixed-topology streaming optimization core. #755's
        # representative-scale and commit-matched confirmation gates ended in
        # PROMOTE_STREAMING_DEFAULT=YES. BooleanOptionalAction intentionally
        # keeps --no-streaming_optimization_engine as a bounded compatibility/
        # historical-control switch while the duplicate inline path is retired.
        # Topology mutation remains outside the engine in either mode.
        self.streaming_optimization_engine = True
        # A1 (ablation): restrict MCMC RELOCATION to the visible set — intersect the
        # dead_mask with the current frame's visibility_filter so out-of-view dead
        # Gaussians are never teleported away. Targets the traveling-scene seed fade
        # (out-of-view bootstrap geometry getting relocated). Default off.
        self.streaming_reloc_local_only = False
        # Detach the global effective-count / opacity-entropy loss gradient on
        # inactive (out-of-view, non-provisional) opacities, so MCMC count-control
        # only pressures supervised Gaussians. ROOT-CAUSE fix for the traveling-scene
        # seed fade: the global count loss otherwise crushes unsupervised opacity to 0
        # (worse in online_mode where N_target is pinned ~q_start*cap_max). Default off.
        self.streaming_count_loss_local = False
        # A6: finite MCMC-annealing horizon for online_mode (iterations). The
        # progress-normalized MCMC schedule (growth_factor, grow/relocate intervals,
        # dead-opacity ramp 0.003->0.01) and the effective-count N_target curve are
        # computed as iter/horizon. online_mode otherwise pins this to 2^31, freezing
        # the whole schedule at u≈0 (growth at MAX, intervals at MIN, dead-opacity at
        # its 0.003 floor) -> perpetual low-opacity "fog" that is never pruned (the
        # traveling-scene seed fade). A finite horizon lets the schedule anneal to
        # quiescence (offline back-half behavior) while ingestion continues, so the
        # dead-opacity threshold rises and the fog is pruned/relocated. The training
        # loop itself stays unbounded (governed by the online stop policy), not this.
        # 0 = legacy unbounded schedule (no annealing). Try ~30000 to match offline.
        self.streaming_online_schedule_horizon = 0
        # OPTIONAL hard safety ceiling for bounded/ablation runs (0 = none, rely
        # purely on the rolling idle-stop below). Prior to 2026-07-04 this was
        # the ONLY termination mechanism for online_mode runs and silently
        # truncated real, incomplete ingestion (coverage-enforcement math was
        # separately broken — see streaming_refinement_iterations); it is now
        # a genuine opt-in ceiling: if it fires before the rolling rule would
        # have stopped naturally, that's logged as a visible truncation, not
        # a silent one.
        self.streaming_max_iterations = 0
        # Unified rolling-budget termination rule (2026-07-04): for online_mode
        # runs (bag-replay-via-ros2 or genuinely-live camera — both flow
        # through the same path), the training loop keeps running as long as
        # a new frame has been ingested within the last N iterations, re-arming
        # every time one arrives. Only once N iterations have passed with
        # nothing new does the run end. For a finite bag this produces a
        # clean, predictable tail of exactly N iterations after the last
        # source frame (no special-casing needed — nothing ever re-arms it
        # again); for a live camera this is literally "keep going while
        # frames arrive." Has no effect on non-online_mode runs, where
        # --iterations is already a deliberate, finite, real budget.
        # 2026-07-04: raised 10_000 -> 50_000 after the first coverage-fixed
        # re-runs (rfmech_loopclosure, fr3par_hybrid_{ctrl,gate}) confirmed
        # full real coverage in only a few thousand iterations, leaving the
        # rest of the (much larger, real) budget for genuine post-coverage
        # refinement — a longer guaranteed tail lets us actually chart the
        # test-PSNR-vs-iterations curve during and after mapping to find the
        # refinement sweet spot (see streaming_refinement_curve_interval).
        self.streaming_refinement_iterations = 50_000
        # Explicit, scientifically separable post-mapping optimizer tail.
        # Unlike streaming_refinement_iterations (the online idle timeout), this
        # starts only after every source frame, including holdouts, has received
        # a disposition and the mapping endpoint has been persisted. 0 keeps the
        # historical single-phase loop unchanged.
        self.streaming_post_mapping_refinement_steps = 0
        # Optional accepted-map quality guard. Each positive-size chunk is
        # measured on the reserved control probe and retained only when every
        # temporal partition improves without widening the quality gap.
        self.streaming_refinement_control_probe_chunk_steps = 0
        # Fail closed instead of retrying non-improving map candidates forever.
        self.streaming_refinement_control_probe_max_rejections = 3
        # Wall-clock companion to streaming_post_mapping_refinement_steps
        # (added 2026-08-31, standing-goal enforcement): every fr3/floor3/
        # kitchen1 comparison this session used a fixed optimizer-step budget
        # for the post-mapping refinement tail, never an actual wall-clock
        # cap -- so no run to date has honestly measured PSNR under the
        # stated ">25dB within a 2-minute wall-clock refinement" goal. This
        # is a REAL elapsed-time timer (time.perf_counter() measured from the
        # mapping-endpoint boundary to now), not an iteration-count proxy --
        # GPU speed varies across hosts (r9700/rtx3090/rtx4090/local), so a
        # step count alone cannot stand in for a wall-clock promise. 0 (the
        # default) disables the cap entirely and preserves the historical
        # step-count-only behavior byte-for-byte. When both this and
        # streaming_post_mapping_refinement_steps are set, refinement stops
        # at WHICHEVER limit is reached first; the run's
        # streaming_phase_metadata.json/report.json record which one
        # (`refinement_stop_reason`: "step_cap" / "time_cap" /
        # "requested_steps_zero").
        self.streaming_post_mapping_refinement_seconds = 0.0
        # Issue #1306: the post-mapping refinement loop can run for tens of
        # thousands of optimizer steps with NOTHING durable saved until the
        # very last one succeeds -- a crash anywhere in that window (a real
        # CUDA OOM during backward has been observed in production, see
        # #1306) previously lost 100% of the refinement work with only the
        # pre-refinement mapping_endpoint/ surviving. Every N completed
        # refinement steps, a `refinement_checkpoint/` boundary
        # (chkpnt<iter>.pth + point_cloud.ply) is written, overwriting the
        # previous one -- so at most one durable snapshot of RECOVERABLE
        # STATE (not a resume path: nothing in the causal_mapping trainer
        # loads a refinement_checkpoint/ and continues from it, mirroring
        # the same caveat on mapping_endpoint/'s checkpoint) is ever kept
        # under `refinement_checkpoint/` itself. This does NOT bound total
        # disk growth end to end: each boundary's underlying flush() also
        # leaves its own `point_cloud/optimizer_updates_<iter>_.../` PLY
        # behind under the run's `point_cloud/` directory (the same
        # mechanism the one-time mapping_endpoint boundary already used),
        # and those per-interval directories are not pruned. 0 disables
        # periodic checkpointing (final-only, the historical behavior) --
        # never set this to 0 for a real evaluation/ablation refinement run
        # of meaningful length.
        self.streaming_refinement_checkpoint_interval = 2000
        # Early stall warning (2026-07-26): a loud, one-shot log line the
        # moment idle time crosses this MUCH smaller threshold, so a stalled
        # ingestion pipeline (e.g. a live SLAM frontend that silently lost
        # tracking / stopped publishing poses) is visible in train.log within
        # minutes -- not only after burning the full streaming_refinement_iterations
        # idle-stop wait (default 50000, which can take 30+ minutes of wasted
        # GPU time before anyone notices). 0 = disabled.
        self.streaming_stall_warn_iters = 1000
        # Multi-resolution frame cache (2026-07-25 design): low-res tier's
        # downscale divisor, and the ResolutionController's dual throughput
        # floors (see eligsir/mapping/resolution_controller.py).
        self.streaming_multires_low_divisor = 8
        self.streaming_multires_it_v_floor = 30.0
        self.streaming_multires_map_fps_floor = 3.0
        # Explicit coarse-to-fine bootstrap. Use LOW through this optimizer
        # iteration, then HIGH; 0 preserves the reactive dual-floor policy.
        self.streaming_multires_low_until_iteration = 0
        # Issue #1259: ResolutionController.decide() has always permanently
        # latched to "high" (native) resolution the moment mapping_done is
        # observed, regardless of throughput -- measured on fr3 as a ~2.9x
        # views/s cliff (quarter-tier mapping ~76 views/s vs native-tier
        # refinement ~26 views/s), which eats most of a short, fixed
        # wall-clock refinement budget's supervised-view count (see
        # docs/architecture/2026-08-31-issue1259-refinement-tier-cliff.md).
        # "native" (default) preserves that historical latch unconditionally.
        # "mapping_tier" disables it: the reactive floor logic keeps running
        # through refinement, which in practice (no more admission events
        # arriving to move the rolling window) just carries forward whatever
        # tier mapping ended at. REFUSED at startup when combined with
        # `--streaming_post_mapping_refinement_steps > 0` ("exact
        # refinement"), which hard-requires native resolution for owner-
        # transfer/resume-envelope reproducibility (train_streaming.py
        # raises RuntimeError on any other tier there) -- this flag is only
        # for the general rolling-idle refinement tail.
        self.streaming_refinement_resolution_policy = "native"
        # 2026-08-12 fix: for depth-less streaming sources (e.g. RealityScan
        # Mobile -- RGB + poses + sparse SfM cloud, no per-frame depth), the
        # multires resolution_cache was previously ALWAYS empty (depth_m is
        # None -> record_frame_observation skipped cache building entirely),
        # so bind_resolution_tier failed on literally 100% of calls for a
        # whole run -- confirmed via a real run's own end-of-run summary
        # (`bind stats: {'low_ok': 0, 'low_fail': 0, 'high_ok': 0,
        # 'high_fail': 30000}`). The low/high tier DECISION
        # (ResolutionController.decide) is driven purely by ingestion
        # throughput, not depth content, so a depth-less source can still
        # meaningfully use rgb-only tiers -- see
        # eligsir/mapping/resolution_cache.py's build_resolution_cache
        # depth=None path. Default True since this only ever changes
        # behavior for sources that were previously 100% broken on this
        # mechanism (never for a depth-having source); set False to fall
        # back to the old always-empty-cache behavior for a depth-less
        # source if this ever needs to be isolated as a regression suspect.
        self.streaming_multires_depthless_rgb_cache = True
        # Global maintenance: run full MCMC pass every N iterations (0 = never)
        self.streaming_global_maintenance_interval = 200
        # Issue #1044 fix: MCMC growth (new Gaussian births) is otherwise driven
        # purely by the time-based get_mcmc_schedule() curve (iteration/cap_max),
        # completely decoupled from whether the map has actually received any new
        # observation recently. Confirmed on a real fr3 (TUM) --online_mode
        # --streaming_frame_admission hybrid_keyframe run: admission stalled to
        # ~20 unique views over 22,661 iterations (a slow-panning scene against
        # max_gap's soft/hard starvation ceiling -- see issue #794/#933/#946's
        # admission-precedence chain, which is NOT what this flag addresses),
        # yet Gaussian count still ballooned to 253,564+ with
        # frac_mass_outside_bbox=0.66 -- pure MCMC-internal inflation against a
        # scene that stopped receiving new photometric/geometric grounding.
        # This brake suppresses growth once more than this many iterations have
        # elapsed since the last real map commit (a depth-insertion cycle that
        # actually added Gaussians, tracked via
        # MappingCoordinator.four_clocks.record_map_commit -- see the #1044 fix
        # note at its non-batch-path call site in train_streaming.py for why that
        # clock was itself silently broken on the default insertion path before
        # this same fix). Iteration-based (not wall-clock) deliberately: this
        # project's own hardware-speed variance (R9700 vs Strix Halo vs rtx3090)
        # makes a fixed-seconds threshold unreliable, per the
        # r9700-perf-gpu-starvation finding. Growth resumes the very next
        # iteration a fresh commit lands. 0 disables the brake (pre-#1044
        # behavior, unconditional time-based growth). Deliberately does NOT
        # suppress relocation/noise -- those redistribute the EXISTING budget
        # rather than growing N, so they are not the mechanism this brake exists
        # to stop.
        # DEFAULT 0 (OFF) -- deliberately, and this must not be flipped on
        # without new real-GPU evidence. This brake suppresses growth once
        # `streaming_growth_staleness_max_iters` iterations pass with no map
        # commit, but a streaming run's mapping phase ENDS long before growth
        # should stop: `streaming_adaptive_mapping_end_threshold` (default
        # 1000) already declares mapping over after 1000 idle iterations, and
        # the refinement tail then runs for up to
        # `streaming_refinement_iterations` (default 50,000) more with, by
        # construction, zero further insertions. A default-on staleness brake
        # would therefore suppress MCMC growth for essentially the ENTIRE
        # refinement tail of every streaming run -- which is precisely the
        # growth-stop-coupled-to-mapping-end behavior that the 2026-07-18
        # correction (see the [ADAPTIVE-HORIZON] block in train_streaming.py)
        # removed after real-GPU testing measured it starving floor3 of
        # capacity: N frozen at ~13k vs a healthy ~230k, test PSNR 19dB vs
        # 22dB. MCMC growth is an ongoing densification mechanism independent
        # of new-frame arrival; "mapping ended" is NOT "growth has saturated".
        # Set >0 only for a genuine-stall investigation (e.g. depth
        # unavailable / occupancy machinery unwired), where suppressing growth
        # against a map that is receiving no observations at all is the point.
        # Note this brake also did NOT fire in, and is not what fixed, the
        # #1044 repro -- see streaming_growth_max_gaussians_per_view below,
        # which is the mechanism that actually caught that pathology.
        self.streaming_growth_staleness_max_iters = 0
        # Issue #1044 fix, part 2: caps live Gaussian count against the number
        # of DISTINCT admitted training views (len(streaming_scene.train_cameras)),
        # suppressing MCMC growth once current_n >= n_admitted_views * this
        # value. See the sched-block comment in train_streaming.py (search
        # "Admitted-view-density growth brake") for the full rationale,
        # including why this -- not streaming_growth_staleness_max_iters above
        # -- is the mechanism that actually catches the #1044 fr3 pathology
        # (N ballooned to 253,564+ while admitted views stayed pinned at 20;
        # depth insertion itself never stalled, so the staleness brake never
        # fired). 0 disables (pre-#1044 behavior, unconditional time-based
        # growth). Deliberately generous default -- a safety ceiling against
        # unbounded runaway growth, not a validated per-scene optimum.
        #
        # Default-on safety measured against this repo's own run corpus
        # (2026-08-23): 472 real-holdout report.json's under output/
        # (degenerate *_bootstrap_views debug reports excluded -- their
        # n_train=5 is the bootstrap block, not the admitted-camera count),
        # ratio = geometry.shape.n_total / n_train. n_total, NOT n_live,
        # because this brake compares against gaussians.get_xyz.shape[0], the
        # full population; median live_frac in that corpus is 0.696, so an
        # n_live basis understates what the brake actually sees. Result:
        # p50 196, p90 1,424, p99 24,806, max 25,700 Gaussians per admitted
        # view. Only 23 of 472 reports exceed 8000/view, and all 23 are
        # mid-run checkpoints of a SINGLE run (w3_r9_first5pct_fr3_20260810,
        # 15 admitted views / ~281k Gaussians) -- the canonical
        # first-5%-prefix degenerate case, i.e. exactly what this brake
        # targets. Per-scene maxima: replica 2,606, kitchen 581, floor3 521,
        # fr1 118, so the floor3/kitchen champion recipes sit well over an
        # order of magnitude under the ceiling and are unaffected. Re-run that
        # measurement before changing this default.
        self.streaming_growth_max_gaussians_per_view = 8000
        # Save PLY every N frames ingested (0 = iteration-based only)
        self.streaming_save_frame_interval = 50
        # Initial opacity for depth-inserted Gaussians — high enough for gradient signal
        self.streaming_insert_opacity = 0.05
        # Track B: birth-local opacity_reg exemption — exempt Gaussians younger than
        # this many iters from the opacity regularizer (0 = disabled). Local version
        # of global dead_opacity_end relaxation; avoids the E2c failure mode (global → N fell).
        # DEFAULT-ON (2026-06-30): the live haze fix. Live young Gaussians were crushed
        # by opacity_reg (no exemption on the live train_streaming.py path) -> opacity
        # stuck ~0.004. Exempting young (<400 iter) opacity-reg validated on kitchen1:
        # inside-scene opacity 0.004 -> ~0.8, train_psnr 15->22. Pairs with young_prot=400.
        self.streaming_young_reg_exempt_iters = 400
        # Track B (maturity ramp): scale per-Gaussian opacity-reg pressure by
        # clamp(support_count / target_support, ramp_min, 1.0) so newborns get
        # ~ramp_min pressure and well-supervised Gaussians get full pressure.
        # Continuous refinement of the binary young_reg_exempt mask; composes
        # with it. Default False = behavior-neutral (off). target_support reuses
        # streaming_debt_replay_target_support (default 10).
        # DEFAULT-ON (2026-06-30): part of the validated haze fix (composes with
        # young_reg_exempt). Its commit also targets the offline back-half regression.
        self.streaming_maturity_reg_ramp = True
        self.streaming_maturity_reg_ramp_min = 0.2
        # Recency-of-supervision opacity-reg relaxation (2026-07-18): the young/
        # maturity exemptions above only protect NEWBORN Gaussians -- a mature,
        # well-supported Gaussian (e.g. the streaming seed/bootstrap region)
        # gets zero exemption and full opacity_reg L1 pressure forever, even
        # during windows where it isn't being trained (camera moved away,
        # keyframe-window/replay sampling temporarily deprioritizing it). Root-
        # caused via the floor3 seed-fade residual-dip investigation: the
        # count-loss fix (--streaming_target_capacity/--streaming_anneal_horizon)
        # closed most of the seed-PSNR dip, but a ~6.4dB residual remained,
        # traced to this constant-pressure gap rather than the count loss.
        # DEFAULT OFF (0 = disabled, behavior-neutral) pending real-GPU
        # verification -- see utils.compiled_kernels.young_opacity_reg_weights's
        # recency_gap/recency_horizon/recency_ramp_min params.
        self.streaming_recency_reg_horizon = 0
        self.streaming_recency_reg_ramp_min = 0.2
        # Post-admission consolidation burst: after each keyframe admission,
        # substitute up to streaming_consolidation_burst_steps normal training
        # steps with steps drawn from the spatially nearest cameras to the newly
        # admitted frame (StreamingScene.get_nearest_cameras), giving newborns
        # immediate multi-view supervision. Drawn from the existing budget — no
        # extra iterations added. DEFAULT-ON (2026-06-30): part of the validated haze
        # fix — gives newborns immediate multi-view supervision so they mature faster.
        self.streaming_consolidation_burst = True
        self.streaming_consolidation_burst_steps = 4
        # #21: depth-quality-aware insertion. Seed each depth-inserted Gaussian's
        # opacity proportional to a per-point depth confidence (low at far range /
        # grazing angles where Orbbec depth is noisiest) so noisy far-range geometry
        # starts faint and must earn opacity photometrically instead of starting as a
        # solid floater. The floor bounds how much the opacity can be reduced
        # (1.0 = full init_opacity; 0.3 = at most a 70% reduction). Default OFF.
        self.streaming_insert_confidence_opacity = False
        self.streaming_insert_confidence_opacity_floor = 0.3
        # Batch N frames of depth-insertion candidates into a single add_points_as_gaussians
        # call, reducing O(N²) optimizer-state rebuild to O(N/batch) calls. 0 = disable batching.
        self.streaming_insertion_batch_frames = 1
        # Flush the insertion batch early if it exceeds this many pending points (0 = no early flush).
        self.streaming_insertion_batch_max_points = 8000
        # Evidence-driven-mutation-cadence experiment (Task 1): step-based (optimizer-
        # iteration-count) alternative to the frame-count-based flush trigger above.
        # 0 (default) = disabled, existing streaming_insertion_batch_frames-only behavior
        # is completely unchanged. When >0, flush fires on (iteration - last_flush_iteration)
        # >= this value instead of frame count -- the two triggers are mutually exclusive
        # alternatives (step-based takes precedence when set), not combined. See
        # _should_flush_insertion_batch in eligsir/mapping/integration/insertion_cadence.py.
        self.streaming_insertion_batch_steps = 0
        # When True, disables the streaming_insertion_batch_max_points early-flush trigger
        # entirely (regardless of its value), leaving only the frame/step-count trigger
        # active. Default False = existing point-cap behavior unchanged.
        self.streaming_insertion_batch_disable_point_cap = False
        # Evidence-driven-mutation-cadence experiment (Task 3): when True, records
        # per-proposal / per-cohort / per-mutation-batch telemetry (JSONL, see
        # eligsir/optimization/gaussian/cohort_trace.py) alongside every real
        # insertion-batch flush. Pure telemetry -- zero effect on training when off
        # (default). Written to <model_path>/cohort_trace.jsonl at run end.
        self.streaming_cohort_trace_enabled = False
        # Evidence-driven-mutation-cadence experiment (Task 5 prep, C4 arm): when >0,
        # suppresses MCMC growth/relocation (step_post_backward) for this many
        # iterations immediately following the most recent insertion-batch commit --
        # staggers MCMC mutation away from the insertion-commit iteration, avoiding a
        # same-step collision between the two mutation sources. 0 (default) = no
        # suppression, existing behavior unchanged.
        self.streaming_mcmc_offset_from_insertion = 0
        # PR3 (2026-08-12 cadence/batching plan, feat/realityscan-streaming): number
        # of views to accumulate gradients over before a single optimizer step
        # (gradient accumulation, no renderer batching). 1 (default) is today's
        # behavior -- one view, one optimizer step, byte-identical to pre-PR3
        # output. Bounded candidate set: 1, 2, 4, 6, 8, 16. Values above 4
        # remain experimental until their real-GPU qualification is recorded.
        # Distinct from the Task 1/3/5 flags above
        # (independently developed on a different branch, no overlap in what they
        # control) -- kept from feat/realityscan-streaming's side of this merge.
        self.streaming_optimization_batch_size = 1
        # Explicit transitional causal-mapping profile: accepted source
        # packets use prepared ingress and the ready B4 streaming adapter.
        # Default false preserves the canonical legacy causal composition.
        self.causal_prepared_ready_b4 = False
        # Opt-in one source-depth insertion from the prepared READY render.
        # READY never retains its borrowed lease depth for a later tick, so
        # this does not enable cross-tick temporal insertion masking.
        self.causal_prepared_ready_source_insertion = False
        # Hardware execution cap for true-camera renderer concurrency (C).
        # Zero resolves to the largest qualified C not exceeding method batch B.
        # Unlike streaming_optimization_batch_size, this does not change loss
        # weighting or the number of views supervised by one optimizer update.
        self.streaming_optimization_camera_count_max = 0
        # Optional end-to-end optimization transaction freshness budget in
        # milliseconds. Zero keeps the mode-derived default (33.333333 ms for
        # live/online, no deadline for offline/refinement-only execution).
        self.streaming_optimization_transaction_budget_ms = 0
        # Runtime constellation owner. ``adaptive`` is the live production
        # policy; ``fixed`` is reserved for controlled B/C ablations where the
        # requested method batch and renderer camera cap must not be shrunk by
        # freshness or latency evidence.
        self.streaming_optimization_constellation_policy = "adaptive"
        # PR2 (2026-08-12 cadence/batching plan, feat/realityscan-streaming): emit
        # explicit optimizer/view/mutation-event counters to TensorBoard
        # (streaming/clocks/*), telemetry only, zero training behavior change.
        # Default False. Distinct from streaming_cohort_trace_enabled above (that
        # one writes a JSONL per-cohort/per-mutation-batch record to disk; this
        # one writes TB scalars for the running clock counters) -- both may be on
        # independently.
        self.streaming_cadence_telemetry_enabled = False
        # Issue #289: window length, in OPTIMIZER STEPS, for the realized-cadence
        # windowed series persisted to report.json["realized_cadence"]["windows"].
        # The block itself is always written (no flag gates it -- no streaming
        # experiment may cite nominal steps_per_frame any more, so realized
        # cadence has to be readable from every run); this only controls the
        # resolution of the windowed series. Retained windows are hard-capped
        # (RealizedCadenceTracker.max_windows) by pairwise merging + window
        # doubling, so a long run costs bounded memory and report size at a
        # coarser resolution rather than growing without limit.
        self.streaming_cadence_window_steps = 500
        # LEGACY, confirmed dead (2026-07-04): no consumer anywhere in the
        # codebase (occupancy hash enforces a single voxel size instead).
        # Kept only for CLI back-compat.
        self.streaming_cover_voxel_size = 0.0
        self.streaming_cover_voxel_multiplier = 1.0
        # Rebuild occupancy hash from current Gaussian positions every N iters (0 = only on stale prune)
        self.streaming_occupancy_rebuild_interval = 200
        # Depth discontinuity threshold: reject pixels where |dz/dx|+|dz/dy| > this (metres)
        self.streaming_depth_edge_threshold = 0.02
        # Early depth-candidate stability filters before insertion.
        self.streaming_depth_filter_enabled = False
        self.streaming_depth_filter_erode_px = 2
        self.streaming_depth_filter_window = 5
        self.streaming_depth_filter_min_valid_ratio = 0.75
        self.streaming_depth_filter_max_range = 0.08
        self.streaming_depth_filter_median_thresh = 0.04
        self.streaming_depth_filter_bottom_margin = 0.0
        self.streaming_depth_temporal_check = False
        self.streaming_depth_temporal_thresh = 0.07
        self.streaming_depth_temporal_require_for_sensor_closer = False
        self.streaming_depth_temporal_require_for_insert = False
        self.streaming_depth_temporal_require_for_seed = False
        # Phase 1.5 experiment #1 (D-vs-ED depth-loss fix, 2026-08-11): which
        # rendered-depth quantity the depth-supervision loss reads. "d"
        # (default) = today's accumulated/alpha-composited depth
        # (render_pkg["rendered_depth"]), unchanged behavior. "ed" = the
        # alpha-normalised "expected" depth (render_pkg["rendered_depth_ed"]),
        # metrically correct per-pixel but numerically unstable at low alpha
        # -- pair with streaming_depth_loss_alpha_gate below when using "ed".
        # "ed_alpha_weighted" (issue #1833) also reads the alpha-normalised
        # ED, but instead of a hard alpha_gate cutoff it multiplies each
        # valid pixel's residual by its rendered alpha (detached, so the
        # loss cannot be reduced merely by inflating opacity) and normalises
        # by the sum of those weights -- this avoids both "d"'s low-alpha
        # bias-toward-zero (which rewards opacity inflation, see #1833) and
        # "ed"'s low-alpha numerical instability, without needing a hand-
        # tuned hard gate threshold. Kept OFF by default (this default is
        # deliberately NOT changed to "ed_alpha_weighted" yet: the
        # validation arms in #1833 were queued, not completed, at the time
        # this flag was added -- see docs/paper/ELIGSIR_DECISION_HISTORY.md
        # for this repo's evidence-before-default-change convention. Opt in
        # explicitly with --streaming_depth_loss_quantity ed_alpha_weighted).
        # See docs/architecture/2026-08-10-debt-controlled-sparse-assimilation-plan.md
        # section "Phase 1.5" for the full rationale.
        self.streaming_depth_loss_quantity = "d"
        # Alpha-coverage gate on the depth-loss validity mask: pixels with
        # render_pkg["alpha"] <= this threshold are excluded from the depth
        # loss regardless of which quantity (d/ed) is selected above. Default
        # 0.0 = no gate, today's exact behavior (only sensor_d>0 and
        # rend_d>0 gate validity, as before this flag existed). Deliberately
        # independent of streaming_depth_loss_quantity so "d + gate" (arm B)
        # and "ed + gate" (arm C) can be compared at the SAME gate value --
        # see Task 1's frozen-checkpoint diagnostic for a data-driven
        # recommendation on what to set this to for a real comparison arm
        # (this default of 0.0 is deliberately NOT that recommended value --
        # changing the default would be a behavior change, which this flag
        # must not cause when unset).
        self.streaming_depth_loss_alpha_gate = 0.0
        # Issue #1833: sensor-depth downsampling strategy consulted only by
        # the coarse-tier sensor depth that materialize_resolution_tier()
        # caches on the camera (independent of
        # --streaming_resolution_resampling_mode's RGB/depth resampling
        # strategy, which selects legacy/masked/ray_preserving overall).
        # Applies to the "legacy"/"masked" area/nearest resize branch (not
        # "ray_preserving", which never resizes). "nearest" (default) =
        # today's exact behavior (F.interpolate(mode="nearest")), zero
        # behavior change when unset. "valid_area" replaces it with a
        # validity-aware area average: each output pixel is
        # sum(sensor_depth * valid) / sum(valid) over its native footprint,
        # counting only pixels with sensor_depth > 0, and is exactly 0 (not
        # NaN/inf) when the footprint has no valid sensor depth at all -- so
        # a coarse-tier depth edge is a true blend of the valid depths under
        # it instead of one nearest-neighbor pick that may disagree with the
        # area-averaged RGB pixel it is paired with.
        self.streaming_depth_loss_downsample = "nearest"
        # Issue #1833: depth-supervision loss weight used during the
        # post-mapping refinement loop (train_streaming.py), independent of
        # the mapping-phase --streaming_depth_loss_weight. -1.0 (default) =
        # reuse the mapping weight unchanged, zero behavior change when
        # unset. Any value >= 0.0 overrides the refinement-phase weight
        # (e.g. 0.0 to zero the depth-loss objective during refinement while
        # retaining any independently authorized insertion/evidence render).
        self.streaming_refinement_depth_loss_weight = -1.0
        # Resolution-tier RGB/depth resampling strategy for
        # materialize_resolution_tier() (eligsir/mapping/
        # resolution_materialization.py). "legacy" (default) is today's exact
        # behavior: RGB via F.interpolate(mode="area"), depth via
        # F.interpolate(mode="nearest"), independently -- zero behavior
        # change when this flag is left unset. "masked" keeps that same
        # independent area/nearest resampling but additionally computes a
        # per-output-pixel depth-discontinuity confidence weight (see
        # streaming_resolution_discontinuity_* below) and downweights the
        # depth-supervision loss (never the RGB loss) at pixels whose native
        # footprint straddles a depth discontinuity -- an area-averaged RGB
        # blend of two surfaces paired with a nearest-neighbor depth pick
        # from only one of them is a physically inconsistent supervision
        # pair. "ray_preserving" replaces the resampling itself with
        # zero-phase strided decimation (tier[u',v'] = native[factor*u',
        # factor*v'] for both RGB and depth identically), so every
        # low-resolution supervision pixel is an exact single native camera
        # ray with no mixed-surface risk in the first place -- see that
        # function's own docstring for the intrinsics formula and the
        # native-registration prerequisite this mode assumes. All three
        # values are initial, not yet ablated against each other -- see
        # materialize_resolution_tier()'s own docstring in
        # eligsir/mapping/resolution_materialization.py for the full
        # design, thresholds, and this mode's registration prerequisite.
        self.streaming_resolution_resampling_mode = "legacy"
        # "masked" mode's discontinuity threshold: threshold(z) =
        # max(streaming_resolution_discontinuity_abs_floor_m,
        # streaming_resolution_discontinuity_rel_frac * z), where z is the
        # nearest (minimum) valid depth in the output pixel's native
        # footprint. A footprint's delta_z (max valid depth - min valid
        # depth) above this threshold is treated as a mixed/boundary pixel.
        # Both defaults are directionally reasonable initial guesses, not
        # validated final values -- subject to future ablation.
        self.streaming_resolution_discontinuity_abs_floor_m = 0.05
        self.streaming_resolution_discontinuity_rel_frac = 0.02
        # Training-time D/ED-vs-alpha instrumentation (2026-08-11): logs
        # per-alpha-bin pixel-fraction population plus mean-absolute D and ED
        # residuals against sensor depth, on the ACTUAL training cameras at
        # the same cadence as streaming/depth_loss_raw_magnitude
        # (scalar_log_interval). Answers whether the frozen-checkpoint
        # diagnostic's near-empty low-alpha population also holds during
        # active map expansion, or whether that's specific to converged
        # final-checkpoint test-holdout renders. Default False = zero
        # behavior/logging change; this is purely additive TB scalars under
        # streaming/depth_alpha_bin/* when enabled -- see
        # utils/depth_alpha_diagnostics.py.
        self.streaming_depth_alpha_bin_diagnostic = False
        # Grazing-angle rejection: reject surface normals > this angle from view direction (degrees)
        self.streaming_max_view_angle = 70.0
        # Use KNN-based initial scale for inserted Gaussians (matches bootstrap quality)
        self.streaming_insert_knn_scale = True
        # Two-frame depth consistency threshold (metres; 0 = disabled)
        self.streaming_depth_consistency_thresh = 0.05
        # Depth-insertion scoring over rendered alpha/depth vs sensor depth.
        # boolean_or preserves the original gate:
        # low alpha OR no rendered depth OR sensor closer than rendered depth.
        self.streaming_insert_score_mode = "boolean_or"  # boolean_or | alpha_only | depth_gap_only | sensor_closer_only | weighted_topk
        self.streaming_insert_alpha_threshold = 0.3
        self.streaming_insert_score_alpha_weight = 1.0
        self.streaming_insert_score_no_depth_weight = 1.0
        self.streaming_insert_score_sensor_closer_weight = 2.0
        # Depth-first center locking is experimental and opt-in until validated.
        self.streaming_depth_lock_centers = False
        self.streaming_depth_lock_bootstrap = False
        self.streaming_depth_lock_insertions = False
        self.streaming_depth_multiview_conflict_enabled = False
        self.streaming_depth_lock_prune_conflicts = False
        self.streaming_depth_unlock_conflict_views = 2
        self.streaming_depth_lock_conflict_thresh = self.streaming_depth_consistency_thresh
        self.streaming_depth_lock_report_interval = 100
        # Multi-view floater consensus (utils/streaming_depth_lock.py): detect
        # mid-air "floater" Gaussians by multi-view depth disagreement and prune
        # them even if MATURE. Fully gated; all default to a no-op. Streaming/
        # online only — the offline train.py path never reaches this code.
        self.streaming_floater_consensus_enabled = False  # master gate (off = identical behaviour)
        self.streaming_floater_consensus_min_views = 3    # require >= K conflicting views
        self.streaming_floater_consensus_conflict_ratio = 0.5  # conflict/seen fraction threshold
        self.streaming_floater_consensus_depth_tol_m = 0.05    # free-space depth tolerance (m)
        # Part D1: parallax / baseline gate. A view only votes on a Gaussian if the
        # camera moved >= min_baseline_m from the previous accepted voter for that
        # Gaussian (mm-apart cameras can't disambiguate depth). Default off.
        self.streaming_floater_consensus_baseline_gate = False
        self.streaming_floater_consensus_min_baseline_m = 0.03  # min parallax baseline (m)
        # Hold-out every Nth arriving frame for test evaluation (0 = disabled).
        # Default 8 ≈ 12.5% test split, evenly spread along the trajectory.
        # Test PSNR + comparison renders are produced post-training whenever
        # this is > 0 (mandatory unless explicitly turned off).
        self.streaming_eval_hold = 8
        # Real-holdout policy (every-Nth + minimum pose gap). Holdouts are
        # selected deterministically from the *source* trajectory (independent of
        # frame-admission), so the test set is identical across runs on the same
        # scene. A candidate every-Nth frame is kept as test only if its viewpoint
        # differs from the nearest trained neighbour by at least one of these
        # thresholds; otherwise it is demoted back to train (a near-duplicate is a
        # trivial "interpolation" view, not a real novel view). An optional guard
        # band withholds neighbours from training to widen the parallax gap.
        self.streaming_eval_min_baseline_m = 0.03   # translation gap (m)
        self.streaming_eval_min_angle_deg = 2.0      # rotation gap (deg)
        self.streaming_eval_guard = 0                # frames withheld each side of a holdout
        # Development/evaluation-only causal mapping curve.  When enabled,
        # the frozen original-frame holdout is rendered synchronously while
        # source ingestion is still live, at streaming_psnr_chart_interval
        # cumulative optimizer updates.  This intentionally pauses mapping and
        # therefore invalidates throughput/timing claims for the whole run.
        # Default False preserves endpoint-only evaluation and live timing.
        self.streaming_mapping_holdout_curve = False
        # Async mode for the mapping-phase holdout curve (issue #1314). Instead
        # of pausing the mapping thread, renders/scores the holdout off a
        # dedicated snapshot CUDA/HIP stream + background thread, from a
        # device-to-device clone of the rendering-relevant Gaussian params
        # (means/scales/quats/opacities/sh) taken at the eval-cadence tick.
        # The mapping thread only pays the (measured, reported) cost of
        # enqueuing that clone -- never a full render+score pause. Requires
        # streaming_mapping_holdout_curve=True. throughput_evidence_valid is
        # derived from the ACTUAL measured snapshot-copy stall, not hardcoded.
        self.streaming_mapping_holdout_curve_async = False
        # Snapshot-copy stall (the time the mapping thread spends enqueuing
        # the async snapshot clone, per streaming_mapping_holdout_curve_async
        # eval tick) above which throughput_evidence_valid is marked False
        # for that run. Sub-millisecond to low-millisecond stalls are
        # expected and considered negligible by default.
        self.streaming_mapping_holdout_curve_async_stall_threshold_ms = 5.0
        # Accumulate observed-surface cells during ingestion for the scene-expressiveness
        # metric SV (n_live/SV efficiency). Off => SV unavailable (e.g. to save ingest time).
        self.streaming_track_scene_complexity = True
        # Accumulate the input-quality metric (sensor-noise vs tracking-error split).
        self.streaming_track_input_quality = True
        # SLAM lifecycle: provisional -> persistent (Step 8)
        self.streaming_min_support_views = 2      # required multi-view confirmations
        self.streaming_support_window = 8         # check support against recent frames
        self.streaming_provisional_max_age = 20   # frames before pruning low-support points
        self.streaming_promote_opacity = 0.3      # opacity boost on promotion
        # Save renders at frame-PLY save milestones (opt-in to avoid overhead)
        self.streaming_render_at_saves = False
        # Pre-training bootstrap snapshot: render bootstrap views before training starts
        # and write to iter_0_bootstrap_views/ (opt-in; adds overhead on startup)
        self.streaming_report_pre_training = False
        # Final report is useful, but can dominate profiling if left unbounded.
        self.streaming_report_final = True
        # Representative metric caps for streaming reports (0 = skip that split).
        self.streaming_report_train_metrics_max = 128
        self.streaming_report_test_metrics_max = 128
        # Representative trajectory-video cap for streaming reports.
        self.streaming_report_trajectory_max_frames = 120
        # Issue #397: trajectory.mp4/train_depth_timeseries.mp4 resolution
        # policy -- aspect-preserving post-render CPU upscale to at least
        # this width (0 disables the upscale side; see
        # utils/trajectory_video.py's scale_dims_for_video). Also read by
        # the offline (train.py) report call site, since StreamingParams
        # is registered for every trainer invocation regardless of
        # --streaming_replay.
        self.streaming_report_video_min_width = 1920
        # Safety cap: downscale to fit this width if the rendered camera is
        # wider (0 disables the cap side). Prevents a very high-resolution
        # source from silently ballooning report-step encode time/output size.
        self.streaming_report_video_max_width = 3840
        # Issue #397: conventional presentation fps for trajectory.mp4's
        # timestamped-source pacing path (utils/trajectory_video.py's
        # distribute_frame_repeats) -- also the playback rate for the
        # spatial (non-timestamped) fallback's default_fps and the encode
        # fps for train_depth_timeseries.mp4. NOT derived from the source
        # capture's own frame rate (see compute_trajectory_timing's
        # docstring for why that degenerates to a slideshow on a
        # subsampled camera list).
        self.streaming_report_video_fps = 30
        # Issue #1861/task 8: CLI-flag equivalent of the SPLAT_REPORT_VIDEOS
        # env var (utils/comparison_report.py's write_post_training_report,
        # `report_videos` kwarg) -- confirmed live (docker exec <container>
        # env, twice) that ablator does not reliably pass SPLAT_REPORT_VIDEOS
        # through to the container for this job type even when set in the
        # ablator spec's per-arm "env" field. --extra_args CLI flags have
        # worked reliably every time this session, so this flag gives future
        # diagnostic runs a way to skip the two post-mapping videos
        # (trajectory.mp4/train_depth_timeseries.mp4) without depending on
        # that broken env passthrough. Default True reproduces today's
        # behavior byte-for-byte; the env var (when it DOES reach the
        # container) remains a secondary override on top of this flag.
        self.streaming_report_videos = True
        # Issue #1323: presentation-only SSAA (supersampling antialiasing) +
        # unsharp-mask sharpening for human-facing report artifacts. Applies ONLY to
        # trajectory.mp4 and train_contact_sheet.png -- the two report artifacts whose
        # render pass is fully independent of the scored mean_test_psnr/mean_train_psnr/
        # SSIM/LPIPS computation (utils/comparison_report.py's _compute_metrics). Never
        # applied to the TEST-split contact sheets (they share their render tensor with
        # those scored metrics) or to anything in the live streaming/mapping hot loop --
        # see utils/presentation_ssaa.py's module docstring. Safe to leave on by default:
        # it cannot move a scored number, only report-generation wall-clock cost.
        self.presentation_supersample_enabled = True
        # Linear supersampling factor (8 = render at 8x width/height = 64x pixel count,
        # then area-downsample back to native resolution). 1 disables SSAA outright even
        # if presentation_supersample_enabled is True. Raised from 2 to 8 (2026-08-31)
        # once the demo campaign showed 8x is needed to actually look clean; this only
        # affects presentation-only report artifacts (trajectory.mp4, train_contact_sheet.png),
        # never the scored metrics render path -- see the comment above.
        self.presentation_supersample_factor = 8
        # Unsharp-mask strength applied after downsampling, to counteract the softening
        # the area-downsample filter introduces. Keep small -- over-sharpening produces
        # visible halos/artifacts, which look worse than doing nothing. 0 disables
        # sharpening. Applied regardless of presentation_supersample_enabled (a nonzero
        # value with supersampling off just sharpens the native-resolution render).
        self.presentation_sharpen_strength = 0.15
        # Issue #397 follow-up ("trajectory.mp4 looks chonky/stuttery"): presentation-only
        # motion-smoothness fix for the real-timestamped pacing path only (TUM/Orbbec/
        # rosbag replay -- see utils/trajectory_video.py's module docstring, section 3).
        # Each repeated-hold output frame within a real camera's time slice is instead a
        # unique SLERP/lerp-interpolated pose swept toward the next real camera, with the
        # exact same total encoded-frame count/duration either way. Default on: bounded,
        # presentation-generation-only cost (see the two caps below), and this cannot
        # reach _infer_realtime_fps or any scored metric (utils/comparison_report.py's
        # _expand_cams_with_interpolation runs strictly after duration/fps are derived
        # from the original, un-interpolated timestamps).
        self.presentation_trajectory_interpolate = True
        # Hard ceiling on ADDITIONAL unique renders this feature may add across one
        # trajectory.mp4, beyond the real camera renders already budgeted by
        # streaming_report_trajectory_max_frames. Guards a very long/high-fps real
        # capture from making the report step render an unboundedly large number of
        # extra frames -- see utils/trajectory_video.py's
        # DEFAULT_MAX_INTERPOLATION_RENDERS docstring for the worst-case cost estimate.
        # This is the UNSCALED value at SSAA off; utils.trajectory_video's
        # presentation_trajectory_interpolate_kwargs_from_args (the helper both
        # train.py and the streaming report-kwargs builder call) further divides
        # this down by --presentation_supersample_factor at report time, since
        # trajectory.mp4 is also the SSAA video and an extra interpolated render
        # costs exactly as much as a real (SSAA-inflated) one.
        self.presentation_trajectory_interpolate_max_extra_renders = 300
        # Per-time-slice cap (i.e. per gap between two consecutive real cameras),
        # independent of the global cap above -- stops one unusually long real gap from
        # alone consuming the whole render budget.
        self.presentation_trajectory_interpolate_max_per_slice = 8
        # Freeze bootstrap Gaussians (birth_frame==0) from MCMC noise displacement.
        # After each step_post_backward, their positions are restored to the pre-noise
        # values. Tests whether position displacement of early Gaussians causes forgetting.
        self.streaming_anchor_bootstrap = False
        # Diagnostic training-mode gate — controls how much optimisation happens:
        #   "normal"         — standard streaming (default)
        #   "placement_only" — skip backward, optimizer step, and all MCMC;
        #                       inserted points stay exactly where placed (H1 ablation)
        #   "colors_only"    — zero LR on means/scales/quats/opacities, SH trains;
        #                       geometry frozen, colors converge to GT (H2 ablation)
        self.streaming_training_mode = "normal"
        # H3 ablation: replace anisotropic surfel scales (tx, ty, 0.2·min)
        # with isotropic scales (geometric mean of tx·ty for all three axes).
        # Prevents edge-on streaking from flat-disc insertions.
        self.streaming_insert_isotropic_scale = False
        # Scale clamping for depth-inserted Gaussians:
        #   scale_mult:         multiplier on the raw depth-derived footprint
        #   scale_max:          hard upper bound per axis (metres)
        #   normal_scale_ratio: z-axis fraction of in-plane scale (anisotropic mode)
        self.streaming_insert_scale_mult = 0.5
        self.streaming_insert_scale_max = 0.05
        self.streaming_insert_normal_scale_ratio = 0.15
        # D2: parallax-based insertion uncertainty. A depth point from a single
        # view is uncertain ALONG the ray; first-order stereo depth error grows as
        # δz ∝ z²/(f·b) where b is the recent trajectory baseline. When the camera
        # has little parallax (stalled / low-baseline views), seed an elongated
        # along-ray cloud so later multi-view consensus can compress it onto the
        # surface; with ample baseline the cloud stays thin. Default OFF.
        self.streaming_insert_parallax_uncertainty = False
        self.streaming_insert_parallax_baseline_frames = 5   # look-back for the baseline estimate
        self.streaming_insert_parallax_coeff = 1.0           # scales the δz ∝ z²/(f·b) cloud
        # LEVER 1 (depth-geometry quality, default OFF): explicit distance-aware
        # in-plane scale. A pixel at depth z subtends world size ~= z/f, so far
        # Gaussians are enlarged to tile the surface hole-free and near ones kept
        # small/sharp. Composes via element-wise MAX with the active footprint/knn/
        # hybrid scale (never shrinks below local density; restores far-tiling when
        # scale_mode=knn discarded the depth footprint). OFF => byte-identical.
        self.streaming_insert_distance_scale = False
        self.streaming_insert_distance_scale_coeff = 1.0
        # LEVER 2 (depth-geometry quality, default OFF): KNN surface-fusion of the
        # voxel-dedup representatives. Instead of merely keeping one point per voxel,
        # average each representative with its K nearest neighbours (denoise depth
        # jitter) and fit a PCA-plane normal (oriented to the finite-diff normal).
        # Fail-open: any error falls back to the plain subsample. OFF => byte-identical.
        self.streaming_insert_surface_fusion = False
        self.streaming_insert_surface_fusion_k = 8
        # Edge-aware radius (× insert voxel size): only fuse neighbours within this
        # distance so fusion denoises within-surface jitter without smearing across
        # depth edges (the cause of the first surface_fusion PSNR regression). 0 = off.
        self.streaming_insert_surface_fusion_max_radius_mult = 2.0
        # Free-space / floater loss weight (penalises opacity rendered in front of surface)
        self.streaming_free_space_loss_weight = 0.0
        # Depth supervision loss weight (0 = disabled)
        self.streaming_depth_loss_weight = 0.05
        # Experimental, train-only display-referred EV + two-coordinate
        # zero-sum log-WB correction. Off preserves the legacy RGB objective.
        self.streaming_train_evwb_enabled = False
        self.streaming_train_evwb_reference_source = ""
        self.streaming_train_evwb_learning_rate = 0.01
        self.streaming_train_evwb_prior_weight = 0.0
        self.streaming_train_evwb_max_ev = 2.0
        self.streaming_train_evwb_max_log_wb = 1.0
        self.streaming_train_evwb_display_margin = 1.0 / 255.0
        # Depth-filtering profile. "default" preserves the generic settings.
        self.streaming_camera_profile = "default"
        # Admission policy. Recognized values:
        #   "all"            -- production SOFT-ALL (issue #600). Retains every
        #                       useful observation the mapper can keep up with;
        #                       may skip mapping SUPERVISION for a frame when the
        #                       backlog bound below is exceeded. A skipped frame
        #                       still advances pose/source/history and is never
        #                       depth-inserted. Default.
        #   "force_all"      -- EXPERIMENTAL/STRESS (issue #600). The historical
        #                       hard-all: unconditional admission, no capacity
        #                       bound, unbounded mapping backlog permitted.
        #                       Retained only because assimilation-overload is a
        #                       scientifically useful condition (#603/#290). Not
        #                       a production setting; every run using it emits a
        #                       loud provenance warning.
        #   "hybrid_keyframe" / "fixed_stride" -- gated selectors, unchanged.
        # Read the RESOLVED semantics (report.json's admission_target), never
        # this raw token: since #600 the token "all" alone no longer tells you
        # whether hard-all or soft-all ran.
        self.streaming_frame_admission = "all"
        # Issue #600 soft-all bounds. Both are a flagged judgment call; see
        # SoftAllAdmitter's docstring for why these defaults are the weakest
        # bound that is still a real bound (inert on every healthy run, refuses
        # only the already-root-caused dataset_fps backlog runaway).
        # min_steps_per_admit: optimizer steps the previously-admitted frame must
        # have received before a new frame may be admitted. 0 disables the
        # capacity bound (soft-all in name only; NOT the same as force_all,
        # which is a reported semantics change).
        self.streaming_soft_all_min_steps_per_admit = 1
        # Age/frontier bound: force-admit after this many consecutive skips, so
        # no observation is starved indefinitely.
        self.streaming_soft_all_max_consecutive_skips = 4
        # Issue #934: explicit fixed-stride admission -- the principal
        # experimental CONTROL arm for comparing hybrid_keyframe's smart
        # selector against a budget-matched, information-free baseline.
        # Active only when --streaming_frame_admission fixed_stride is set;
        # admits every Nth candidate reaching the admission_fn call site
        # (see train_streaming.py's _fixed_stride_admission_closure), i.e.
        # the SAME candidate population hybrid_keyframe sees, unlike
        # --streaming_source_stride/--streaming_frame_stride which decimate
        # the frame source before that population even exists and are
        # therefore NOT a valid control for this comparison. Default 10
        # matches the historical realized max_gap=10 stride #792/#796 showed
        # hybrid_keyframe degenerates to -- the natural budget-matched
        # starting point.
        self.streaming_fixed_stride_n = 10
        self.streaming_keyframe_min_translation = 0.05
        self.streaming_keyframe_min_rotation_deg = 5.0
        self.streaming_keyframe_min_overlap = 0.25
        self.streaming_keyframe_max_overlap = 0.90
        self.streaming_keyframe_max_gap = 10
        # When True, hybrid_keyframe admission skips the expensive render-overlap
        # gate for frames that have not moved past the motion thresholds (and are
        # not force-admitted by max_gap). Removes the ~2.6 ms admission render from
        # every slow-motion frame — the dominant during-ingestion GPU cost — and is
        # claimed "admission-preserving" (a non-moved frame has ~1.0 overlap and the
        # render gate would reject it anyway) -- validated on kitchen1 live
        # (iter_ms 16.9→15.0 at matched N, admission render 2.75→0.10 ms, coverage
        # preserved N 20645→21688). See docs/2026-06-30-live-ingestion-gil-contention.md.
        #
        # Default ON (reverted 2026-07-24, see below): under a sparse GT pose source
        # (pose_normalizer republishing the last known pose, unchanged, until a new
        # GT message arrives), this pregate has a SECOND, accidental effect beyond
        # its intended perf optimization: since every frame between two real GT ticks
        # holds the exact same (stale, unchanged) pose value, only the first frame
        # after each tick shows a real delta and gets admitted -- every other frame in
        # between is correctly rejected as "no_motion," which in this specific
        # situation means "no NEW pose data," not "camera didn't move." This is a
        # data-quality safety net we rely on, not a bug to route around: it keeps
        # stale/wrong-pose frames OUT of the training buffer. Initially misdiagnosed
        # as a correctness bug and flipped to False (commit 2356057) -- that was
        # WRONG: disabling it would have let every stale-pose frame flood into
        # training with an incorrect (held-over, non-instantaneous) pose instead of
        # being filtered out, actively corrupting reconstruction rather than merely
        # under-covering it. Reverted. The real, still-open problem this surfaced is
        # that there is no EXPLICIT, purpose-built pose-staleness check (comparing the
        # associated pose message's own timestamp against the frame's timestamp,
        # honoring --max-association-dt) -- the motion pregate's side effect is
        # covering for that gap, not a substitute for it. See
        # [[motion-pregate-silent-admission-corruption]] memory for the full
        # misdiagnosis/correction history.
        self.streaming_keyframe_motion_pregate = True
        self.streaming_keyframe_coverage_alpha = 0.3
        # Velocity-adaptive admission: target 2fps admit rate, adapts threshold to scene speed.
        self.streaming_keyframe_velocity_adaptive = True
        self.streaming_keyframe_target_admit_fps = 2.0
        self.streaming_keyframe_input_fps = 30.0
        self.streaming_keyframe_velocity_ema_alpha = 0.1
        # DEPRECATED (no-op): holdouts are now reserved deterministically before
        # admission, so they are always kept as test regardless of this flag.
        # Retained only so existing configs/CLI do not error.
        self.streaming_keyframe_admit_eval_holdouts = False
        self.streaming_keyframe_quality_enabled = False
        self.streaming_keyframe_quality_min_blur = 8.0
        self.streaming_keyframe_quality_max_clip_ratio = 0.25
        self.streaming_keyframe_quality_min_depth_valid = 0.10
        # Adaptive (scene-relative, resolution-normalized) blur/exposure
        # scoring -- only takes effect when quality_enabled is also True.
        # Replaces the hard min_blur reject with a soft, causal-running-
        # stats-relative weight that self-disables on already-sharp scenes
        # (see FrameQualityConfig.adaptive_quality's docstring).
        self.streaming_keyframe_quality_adaptive = False
        self.streaming_keyframe_quality_blur_floor = 20.0
        self.streaming_keyframe_quality_exposure_tolerance = 0.35
        self.streaming_keyframe_quality_ema_alpha = 0.05
        self.streaming_keyframe_quality_warmup = 20
        # Admission-rate knob for the scored path (quality_enabled=True): a
        # single scalar on the weighted score (pose/rotation/overlap/
        # new_surface/render_residual/depth_disagreement/uncertainty). Higher
        # = fewer, higher-information-content frames admitted.
        self.streaming_keyframe_final_threshold = 0.45
        # Track A: depth-coverage gain gate — rejects frames with ≤ min_new_voxel_frac
        # unoccupied depth backprojections. Cheaper than the render-overlap gate.
        # Enabled by default as part of the AB2-v6 validated config.
        self.streaming_keyframe_coverage_gain_gate = True
        self.streaming_keyframe_min_new_voxel_frac = 0.05
        self.streaming_keyframe_coverage_gain_depth_stride = 8
        # Component 2: novelty-gated max_gap. When True, a would-be max_gap-forced
        # frame must also pass the coverage novelty check (new-voxel-frac >=
        # min_new_voxel_frac) to be admitted, so long forward walks over already-
        # mapped space stop inflating the training-view count (the dominant driver
        # of floor3 oscillation — Phase-0: fewer views 41%->22% imploded). A hard
        # ceiling gap >= max_gap*hard_mult still force-admits (anti-starvation).
        # DEFAULT FLIPPED TO True 2026-08-15 (issue #792/#794): with this off,
        # max_gap force-admits every Nth candidate UNCONDITIONALLY and, at the
        # default max_gap=10 vs the velocity-adaptive controller's nominal
        # ~15-frame target interval (input_fps/target_admit_rate_fps), max_gap
        # fires first almost every time -- measured on the real corpus audit
        # (issue #792/PR #814) at 485/486 (99.8%) of hybrid_keyframe admission
        # arms, and independently on a real ScanNet++ ablation battery this
        # session (iphone_lidar camera profile, see that profile's own comment
        # a few lines above train_streaming.py's _CAMERA_PROFILES) where
        # max_gap-dominated admission sat right on the 1/max_gap floor across
        # scenes of very different frame counts. Flipping this on is a fast,
        # narrowly-scoped fix for #794's core complaint (max_gap silently
        # bypassing the novelty/coverage gate, not just the motion gate) using
        # a mechanism that already existed and was already validated in that
        # profile -- it does NOT touch #793 (the velocity-adaptive controller's
        # separate, deeper "speed EMA is actually cumulative displacement, not
        # velocity" correctness bug) or #794's full candidate-escalation-vs-
        # hard-force precedence redesign (exposure/blur/quality gates are still
        # bypassed by a forced admission). Those remain open follow-up work.
        self.streaming_keyframe_max_gap_require_novelty = True
        self.streaming_keyframe_max_gap_hard_mult = 4.0
        # Issue #794 admission precedence contract. What a SOFT max_gap force
        # (max_gap <= gap < max_gap*max_gap_hard_mult) is permitted to bypass.
        # The full, single-place table lives in
        # eligsir/mapping/scheduler/precedence.py.
        #   "legacy_all_gates" (DEFAULT): pre-#794 behaviour -- a soft force
        #       short-circuits straight to admission, skipping the parallax /
        #       exposure / blur / render-overlap / quality gates. Unchanged
        #       admission on every existing run; the only difference vs. before
        #       #794 is that each bypass is now RECORDED (KeyframeDecision.trace
        #       + report.json's admission precedence counters).
        #   "motion_only": the #794 contract -- a soft force relaxes the motion
        #       criterion only (motion counts as satisfied, so the backstop still
        #       works) and every information/quality gate still binds. CHANGES
        #       which frames are admitted; opt in deliberately.
        # The hard ceiling is unaffected by this knob, and NO setting can bypass
        # the ORB-SLAM3 map-reinit integrity veto (#806/#817), which is enforced
        # upstream in StreamingScene.ingest_next_frame.
        self.streaming_keyframe_max_gap_force_scope = "legacy_all_gates"
        # Issue #946: DECOUPLE the soft escalation threshold from the hard
        # starvation ceiling, and resolve the soft one in source time.
        #
        # Before #946 there was ONE resolved quantity -- max_gap frames -- and
        # the hard ceiling was DERIVED from it (max_gap * max_gap_hard_mult).
        # Raising max_gap from 10 to 15/20 therefore also moved the
        # last-resort unconditional safety bound from 40 to 60/80 frames,
        # making any experiment on the soft backstop inseparable from a change
        # in the safety bound it is supposed to hold fixed.
        #
        #   "legacy_frames" (DEFAULT): soft = max_gap frames,
        #       hard = max_gap * max_gap_hard_mult frames. Verbatim historical
        #       semantics -- bit-identical admit/reject decisions on every
        #       existing config, spec and CLI invocation in this repo. The
        #       multiplicative coupling survives ONLY here, as this mode's
        #       explicit definition.
        #   "target_time": soft = soft_backstop_multiplier x the velocity
        #       controller's own target admit interval (1/target_admit_fps),
        #       expressed in SOURCE SECONDS via the canonical #793/#933
        #       timebase; hard is resolved independently from
        #       streaming_keyframe_hard_backstop_s /
        #       streaming_keyframe_hard_backstop_frames and reads NO soft
        #       input. OPT-IN and NOT GPU-validated -- issue #947 owns the
        #       first experiment. An unrecognised value falls back to
        #       "legacy_frames" with a loud startup warning; it never silently
        #       enables production semantics.
        self.streaming_keyframe_soft_backstop_mode = "legacy_frames"
        # target_time only. MUST be >= 1: the soft backstop exists to catch a
        # controller that has gone quiet, so it must be structurally incapable
        # of firing BEFORE the cadence it backstops. A value below 1 is
        # normalised up to 1.0 with a loud warning, never silently accepted.
        self.streaming_keyframe_soft_backstop_multiplier = 1.25
        # target_time only. Independent hard starvation ceiling, in SOURCE
        # SECONDS. 0.0 => resolve from streaming_keyframe_hard_backstop_frames
        # instead (the default).
        self.streaming_keyframe_hard_backstop_s = 0.0
        # target_time only. Independent hard starvation ceiling, in SOURCE
        # FRAMES. The default is the CURRENT EFFECTIVE production ceiling made
        # explicit: streaming_keyframe_max_gap (10) *
        # streaming_keyframe_max_gap_hard_mult (4.0) = 40 source frames.
        # Stated as a literal precisely so a future soft-timing change of
        # 15/18/22 cannot move it. A test asserts this stays equal to the
        # product of those two CLI defaults, so the "explicit" ceiling cannot
        # silently desynchronise from the historical one.
        self.streaming_keyframe_hard_backstop_frames = 40.0
        # Issue #1044. `target_time` mode enforces a non-preemption invariant
        # (#946 criterion 2): the SOFT backstop may never fire earlier than the
        # velocity controller's target interval. `legacy_frames` -- the default
        # -- never had that floor, and the production profile violates it:
        # max_gap=10 frames vs a controller interval of
        # input_fps/target_admit_rate_fps = 30/2 = 15 frames. Both the motion
        # threshold and the backstop scale linearly with scene speed, so the
        # 1.5x shortfall is scene-INDEPENDENT and the TRANSLATION criterion can
        # never own an admission, not merely rarely. `moved` is an OR of two
        # terms and only translation is velocity-scaled -- the static
        # min_rotation_deg (5 deg) survives and admits on rotation-heavy
        # segments, which is why the backstop owns ~78% rather than 100%.
        # Measured on a real fr3 run over the full 2525-frame trajectory
        # (i1044_A_prefix, 2026-08-22): 197/251 admissions max_gap-owned, and all
        # 54 motion-owned ones ROTATION-driven -- translation owned exactly 0,
        # despite 111 candidates satisfying its threshold (max_gap pre-empted
        # every one). With the floor: 134 motion-owned, 28 translation-driven.
        # admission_preflight.py has warned about this since #792 without
        # resolving it. True (default) raises only the soft threshold to the
        # controller interval; the hard ceiling stays at
        # max_gap * max_gap_hard_mult per #946's independence rule. Set False
        # to restore pre-#1044 semantics.
        self.streaming_keyframe_backstop_nonpreemption_floor = True
        # Issue #794 (gap closure, 2026-08-23). `legacy_frames` -- the default
        # backstop mode above -- only WARNS on the two structurally
        # contradictory configurations covered by the non-preemption floor:
        # (a) the floor declining to apply because
        # streaming_keyframe_max_gap * streaming_keyframe_max_gap_hard_mult
        # is not strictly beyond the controller's target interval, and (b)
        # the pre-existing hard_frames <= soft_frames case. `target_time` mode
        # raises BackstopConfigError unconditionally on the equivalent
        # contradiction; `legacy_frames` never had a raising path, only a
        # warning. When True, `legacy_frames` raises too, in exactly those two
        # cases, refusing to start training on a config where the soft
        # starvation escalation can never own an admission (max_gap pre-empts
        # the motion criterion before it is ever consulted -- the ownership
        # failure #794 is about). Default False: must not change behaviour for
        # ANY existing spec -- this is an explicit opt-in, not a new default.
        self.streaming_keyframe_require_reachable_motion_gate = False
        # Issue #934: novelty-DRIVEN admission. Until this flag existed the
        # coverage/novelty gate was architecturally VETO-ONLY -- it could
        # suppress an admission but never cause one, and the motion pregate
        # short-circuited before it was even consulted for a non-moving
        # candidate. Two fr3 GPU runs measured ~96% of admissions owned by the
        # motion gate with only 4-5 ever owned by novelty, at BOTH ends of the
        # threshold sweep. When True, a candidate whose measured new-voxel
        # fraction reaches novelty_force_frac has the MOTION criterion treated
        # as satisfied (ForcePath.NOVELTY_FORCE, bypassing Gate.MOTION and
        # NOTHING else) and continues through every remaining gate. Default
        # False => behaviour-identical to before; note ARTIFACTS still gain
        # additive keys either way (4 new `precedence` trace keys, a
        # `novelty_forced` sidecar column, and report.json's
        # admission_fingerprint["novelty_force"] block), so a diff against a
        # pre-#934 run's artifacts is not empty even with the flag off. See
        # docs/architecture/2026-08-16-issue934-novelty-driven-admission.md.
        self.streaming_keyframe_novelty_force_admit = False
        # Trigger threshold. MUST exceed streaming_keyframe_min_new_voxel_frac
        # (clamped upward with a warning if a spec inverts them). Pin it from a
        # realized run's own novelty_gate_fraction/novelty_force_fraction
        # distribution, and note the density-feedback caveat on #934: admitting
        # more frames densifies the occupancy grid and shifts the whole
        # distribution DOWN, so a value pinned at one run's median can land
        # above the next run's p90.
        self.streaming_keyframe_novelty_force_frac = 0.6
        # Stale-pose guard (default ON). This trigger bypasses the motion
        # pregate, which is a standing safety net against stale-pose frames
        # under a sparse GT-pose stream -- and a stale pose backprojects depth
        # at the WRONG pose, landing in unobserved voxels, which reads as HIGH
        # novelty. Requires a strictly-positive instantaneous translation
        # velocity sample for the candidate; fail-closed when none exists
        # (including velocity_adaptive=False).
        self.streaming_keyframe_novelty_force_require_fresh_pose = True
        # Component 1: spatial active bubble. When enabled, Gaussians farther than
        # streaming_bubble_radius (m) from the robot's current pose are FROZEN this
        # temporal window — excluded from MCMC relocation, position noise, and
        # gradient updates — so travel-away ("forgetting after ~iter 6000") geometry
        # cannot drift/vanish while the robot works elsewhere. The mask is recomputed
        # from CURRENT positions every step (no stored state → no grow/prune reindex),
        # so a region reactivates automatically when the robot returns. This is the
        # CORRECTNESS step (no forgetting); a frustum-overlap view sampler (perf) is a
        # separate follow-on. Default off => byte-identical.
        self.streaming_spatial_bubble_enabled = False
        self.streaming_bubble_radius = 5.0
        # Guard band (m) beyond the radius: radius..radius+guard is a trainable seam
        # that softens the frozen boundary (avoids alpha-composite seam artifacts).
        self.streaming_bubble_guard_m = 1.0
        # Forward offset (m) that pushes the bubble center off the (empty-space) camera
        # position onto the observed geometry ahead. 0 = auto (median forward-distance
        # of geometry ahead of the robot); >0 = fixed offset.
        self.streaming_bubble_forward_offset = 0.0
        # --- Local-capacity telemetry (instrumentation only; default OFF) ---------
        # Measures the marginal utility of Gaussians in the moving local region so we
        # can see whether MORE Gaussians would explain MORE detail (dPSNR/dN) — the
        # signal for a future per-region adaptive capacity controller. When ON, every
        # global-sentinel cadence it records, for the region around the robot's latest
        # pose: local PSNR (over nearby keyframes vs their GT), N_gaussians in-region,
        # #views, region cell id, and the finite-difference dPSNR/dN vs the previous
        # sample. Rows are appended to <model_path>/local_capacity.jsonl and a few
        # scalars go to TB. No behavior change — pure observation.
        self.streaming_local_capacity_telemetry = False
        # Region radius (m) around the robot's latest camera center. Defaults to the
        # bubble radius so "local region" == "the active bubble".
        self.streaming_local_cap_radius = 5.0
        # Region-binning cell size (m) for the per-region PSNR-vs-N curves. Coarse
        # (~bubble sized) so each cell is roughly one dwell region.
        self.streaming_local_cap_region_m = 5.0
        # --- Adaptive capacity controller v1 (brake-only; default OFF) ------------
        # Keys on the RELIABLE global-PSNR sentinel vs N (whole-scene, not train-view).
        # Growth stays permissive up to cap_max; when marginal utility dPSNR/dN goes
        # sustainedly NEGATIVE (overgrowth — the validated brake signal), the growth
        # cap is frozen at the current N so N stops ballooning. Releases back to cap_max
        # when utility recovers positive. kitchen1 (u>0) never brakes; floor3 (u<0) does.
        # OOB floater prune: margin beyond the camera-trajectory bbox = mult × max_depth.
        # Legacy 3.0 (=24 m at 8 m range) is far too lenient for small scenes; lower culls
        # far floaters. Default 3.0 keeps legacy behaviour; validating 1.0.
        self.streaming_bbox_prune_margin_mult = 3.0
        self.streaming_adaptive_capacity = False
        self.streaming_adcap_u_brake = -1.0e-5   # dPSNR/dN below this (sustained) => brake
        self.streaming_adcap_u_release = 1.0e-5  # above this => release cap back to cap_max
        self.streaming_adcap_hysteresis = 1      # samples before a state flip; 1 = prompt brake
                                                 # (validated: floor3 brakes at N~21k, the ideal;
                                                 # the release path self-corrects any false brake)
        self.streaming_adcap_min_dN = 200        # ignore samples with |ΔN| below this (noise)
        self.streaming_adcap_ema = 0.5           # EMA smoothing of the utility estimate
        self.streaming_keyframe_min_admit_frames = 0  # 0=disabled; offline only (requires total_frames)
        # Rolling-window floor (live-SLAM-compatible): force-admit at least
        # rolling_window_min_admits frames in every rolling_window_size-frame window.
        # Prevents starvation collapse on slow/hovering cameras where the coverage-gain
        # gate would reject everything (near-zero voxel novelty on revisit passes).
        # Default: at least 1 admit per 30-frame window → ~1fps floor at 30fps input.
        self.streaming_keyframe_rolling_window_size = 30
        self.streaming_keyframe_rolling_window_min_admits = 1
        # Exposure gate: reject keyframe candidates whose mean luminance has
        # shifted too far from the last admitted frame (proxy for AE jumps).
        self.streaming_keyframe_exposure_gate = False
        self.streaming_keyframe_exposure_max_delta = 0.15
        self.streaming_keyframe_exposure_force_admit_iters = 200
        # Blur gate: hard-reject frames below Laplacian-variance threshold.
        self.streaming_keyframe_blur_gate = False
        self.streaming_keyframe_blur_min_laplacian = 20.0
        self.streaming_keyframe_blur_downsample = 4
        self.streaming_keyframe_blur_force_admit_iters = 200
        # Low-overlap starvation relief (default 0 = disabled, byte-identical).
        # The render-overlap gate rejects "low_overlap" candidates that view
        # mostly unreconstructed map space — exactly the frontier-expansion
        # frames a keyframe selector should welcome. On small/fast-panning
        # scenes this starves admission near-permanently (confirmed on TUM
        # fr1_desk: hybrid_keyframe+coverage_gain_gate landed at 7.0% admission,
        # 56% of candidates rejected low_overlap). Set > 0 to force-admit a
        # would-be low_overlap rejection after this many starved iters, mirroring
        # the exposure/blur force-admit pattern. See keyframes.py
        # KeyframeAdmitterConfig.low_overlap_force_admit_iters docstring.
        self.streaming_keyframe_low_overlap_force_admit_iters = 0
        self.streaming_source_stride = 1  # pre-stride frame source before StreamingScene
        # NOTE (2026-07-04): distinct from streaming_frame_stride above, not redundant
        # with it — see that comment for the precedence/composition rule. This one
        # wraps the already-constructed frame_source post-hoc (_StridedFrameSource,
        # train_streaming.py) so it applies uniformly regardless of source type,
        # including sources that don't accept frame_stride natively (ros2://live).
        #
        # Issue #834: streaming_frame_stride and streaming_source_stride are two
        # independently-active source-rate knobs that used to silently MULTIPLY
        # when both were set to different non-1 values (stride applied twice —
        # once inside the native reader, once again by the generic
        # StridedFrameSource wrapper). Source sampling must be requested once,
        # resolved once, applied once — see
        # eligsir/mapping/scheduler/frame_scheduler.py's
        # resolve_source_sampling_policy() for the resolution rules and
        # train_streaming.py's frame-source construction for where the resolved
        # policy is actually enforced (exactly one owner applies the stride).
        #
        # "single_owner" (default, canonical): exactly one owner (native reader
        # if only streaming_frame_stride is set or both are set equal; the
        # generic wrapper if only streaming_source_stride is set) applies the
        # requested stride. Setting BOTH to different non-1 values is a
        # configuration error — training refuses to start
        # (ConflictingStrideError) rather than silently composing the two,
        # letting the operator fix the spec instead of discovering a mystery
        # 2x/3x/... over-decimated run after the fact.
        # "legacy_compose" (opt-in escape hatch): restores the old
        # multiplication behavior (frame_stride * source_stride) for a spec
        # that genuinely wants that composition. Never the silent default.
        self.streaming_source_sampling_mode = "single_owner"
        # Depth loss type: "l1" or "huber"
        self.streaming_depth_loss_type = "huber"
        # H7: Freeze confirmed geometry gradients during streaming training.
        # When enabled, xyz/scales/quats gradients are zeroed for old confirmed
        # Gaussians so the optimizer cannot drag existing good splats to explain
        # new views — depth insertion must handle new geometry instead.
        #
        # EXPERIMENTAL / EXPERIMENTALLY CONFIRMED BROKEN — leave this False.
        # A 4-way Replica room0 ablation + a real kitchen1 Orbbec run both show
        # this flag causes a catastrophic, non-recovering, global optimization
        # collapse (e.g. 27.95dB -> 14.54dB on Replica; 16-20dB -> 5-7dB on
        # kitchen1, train PSNR == test PSNR throughout, loss flat for the whole
        # run) — not selective old-region forgetting. Root cause: the flag ages
        # Gaussians in *frames* (streaming_young_age_frames /
        # streaming_freeze_new_frame_steps), but CLAUDE.md's mandatory
        # full-trajectory-coverage contract drives steps_per_frame very low for
        # any real iteration budget (Replica: 20, kitchen1: 5), so confirmed
        # Gaussians get locked long before they've converged. See
        # docs/streaming-anti-forgetting-defaults.md for the full writeup
        # before re-enabling this for an evaluation run.
        self.streaming_freeze_old_geometry = False
        # A Gaussian is "young" (gradients allowed) for this many frames after birth.
        self.streaming_young_age_frames = 5
        # Protect newly inserted Gaussians from dead-mask pruning for this many iters
        # after birth. Prevents under-supervised splats from being killed before opacity
        # builds. 0 = disabled (legacy behavior). Validated sweet spot: 400 (prot=800 is
        # -0.48 dB worse; fr3 ablation 2026-06-22; see docs/streaming_levers.md L3).
        self.streaming_young_prot_iters = 400
        # --- Candidate-survival funnel telemetry (issue #1365, formerly #1372 Phase 7) ---
        # Purely observational: reports which existing MCMC eligibility gates
        # (lifecycle, young-protection, resolution-exposure, depth-lock,
        # seed-freeze, spatial bubble, visibility) a relocation-target/
        # growth-parent candidate would pass or fail, and their pairwise
        # intersections (e.g. MATURE age-only vs. exposure-based
        # young-protection). Never changes any gating decision. Default OFF:
        # byte-identical when unset. See utils/mcmc_candidate_funnel.py and
        # docs/architecture/2026-08-31-issue1372-candidate-survival-funnel.md.
        self.streaming_mcmc_candidate_funnel_telemetry = False
        # Sample interval (in optimizer updates) for the funnel above -- it is
        # cheap (a handful of boolean tensor ops over N) but still skipped on
        # most calls by default so it never materially affects throughput even
        # when MCMC epochs fire every few tens of iterations.
        self.streaming_mcmc_candidate_funnel_interval = 200
        # --- Resolution-exposure eligibility (issue #1262, category (a) only) ---
        # Distinguish "observed at this tier and found low-utility" from "the tier's
        # ray lattice could not resolve it, so there is no evidence either way", and
        # keep the latter out of MCMC death/relocation selection AND out of the
        # quantile statistic pool. Default OFF: byte-identical when unset.
        # See docs/architecture/2026-08-29-mcmc-resolution-exposure-eligibility.md.
        self.streaming_resolution_exposure_veto = False
        # Accumulate exposure counters + opacity-survival telemetry WITHOUT vetoing
        # anything (the measurement-only arm; §6.2.1 requires this data to tell
        # topology destruction apart from zombie opacity death).
        self.streaming_resolution_exposure_telemetry = False
        # "lattice_intersection" (preferred, phase-aware) | "sigma_ratio" (first-pass
        # proxy; its phase-ambiguous band is reported as its own uncertainty estimate).
        self.streaming_resolution_exposure_predicate = "lattice_intersection"
        # sigma_ratio proxy threshold: exposed iff sigma_proj >= tau * downsample_factor.
        self.streaming_resolution_exposure_tau = 0.5
        # Mahalanobis radius of the ellipse tested for lattice intersection.
        self.streaming_resolution_exposure_sigma_c = 3.0
        # n_min of §3.4 — cumulative distinct-view exposures below which a Gaussian
        # counts as "not yet measured enough" (composed with, never replacing, the
        # iteration-currency young gate above).
        self.streaming_resolution_exposure_min_count = 3
        # W3-45 bounded immunity: protection lapses past this age regardless of
        # exposure, so a never-exposed Gaussian cannot be immune forever. 0 = no
        # ceiling (not recommended).
        self.streaming_resolution_exposure_max_protect_iters = 4000
        # Extra-strict freeze for this many steps immediately after a new frame arrives.
        # During this window, only provisional splats get geometry gradients.
        self.streaming_freeze_new_frame_steps = 50
        # H8: New-frame warmup — train exclusively on the just-arrived frame for this
        # many steps before mixing in the local window / replay.
        self.streaming_new_frame_warmup_steps = 0
        # H9: Anchor loss for young provisional splats.
        # Penalises drift from the depth-insertion position while the splat is young.
        self.streaming_anchor_loss_weight = 0.0
        # Anchor penalty decays linearly to zero over this many training iterations.
        self.streaming_anchor_decay_steps = 500
        # Bootstrap-specific position anchor (applied to birth_frame==0 Gaussians).
        # Separate from H9 provisional anchor; covers the multi-frame seed geometry
        # that the optimizer would otherwise slide under photometric gradient.
        # Weight decays linearly to zero over bootstrap_anchor_decay_steps iterations.
        self.streaming_bootstrap_anchor_weight = 1.0
        self.streaming_bootstrap_anchor_decay_steps = 2000
        # Per-frame depth-insertion anchor (applied to birth_frame>0 Gaussians).
        # Each inserted Gaussian's means gradient is stochastically suppressed for
        # insertion_anchor_decay_steps iterations after its own anchor_iter, giving
        # newly added geometry time to settle before the optimizer is free to move it.
        # Uses the same proven gradient-masking mechanism as the bootstrap anchor.
        self.streaming_insertion_anchor_weight = 1.0
        self.streaming_insertion_anchor_decay_steps = 200
        # Sync-free anchor masking: draw the stochastic bootstrap/insertion anchor
        # keep-masks over ALL N Gaussians (torch.rand(N) < retain) instead of over a
        # boolean-indexed/.sum()-sized subset. Statistically identical (i.i.d. keep
        # with the same per-splat prob) but removes the device→host syncs (the
        # torch.rand(mask.sum()) 0-d-tensor-as-size sync + the .any() guards) that
        # serialize this launch-bound loop. --no-streaming_anchor_fullN_draw restores
        # the exact-RNG masked-count path for A/B.
        self.streaming_anchor_fullN_draw = True
        # Bootstrap motion diagnostics (off by default; useful for early geometry drift).
        self.streaming_debug_bootstrap_motion = False
        self.streaming_debug_bootstrap_ply_interval = 0
        self.streaming_debug_bootstrap_motion_threshold = 0.05
        self.streaming_debug_bootstrap_first_iters = 200
        # Fine-grained parameter freeze for isolation ablations.
        # Comma-separated list of param-group names to hard-zero-LR at training start.
        # Valid names (gsplat layout): means, scales, quats, opacities, sh0, shN
        # Valid names (legacy layout): xyz, scaling, rotation, opacity, f_dc, f_rest
        # Example: "means" → isolate center drift; "scales,quats" → isolate shape drift.
        # Empty string (default) → no freeze applied (normal / colors_only still apply).
        self.streaming_freeze_params = ""
        # H10: Global keyframe reservoir.
        # Every Nth ingested train frame is kept in a permanent reservoir for
        # trajectory-wide replay. 0 = disabled.
        # Default 1: keep every frame. When the replay buffer falls back to this
        # reservoir the 30% replay steps sample uniformly over ALL historical frames,
        # preventing early-frame forgetting once they exit the sliding window.
        self.streaming_global_reservoir_stride = 1
        # H11: Submap-stitching mode parameters.
        # streaming_training_mode = "submap_stitch" enables the submap path.
        self.streaming_submap_frames = 20       # frames per independent submap
        self.streaming_submap_iters = 3000      # optimisation iterations per submap
        self.streaming_global_refine_iters = 5000  # final global refinement iters
        # Issue #110 shadow mode: log what each of the four assimilation-pressure
        # levers (insertion suppression, consolidation-budget, replay weighting,
        # growth suppression) WOULD decide at the current pressure tier, without
        # applying any of them. Read-only wrt training behavior when this is set --
        # MappingDecision's actual allow_insertion/steps/etc are unaffected.
        # See docs/architecture/2026-08-12-w3-issue110-assimilation-shadow-mode-design.md.
        self.streaming_assimilation_shadow_mode = False
        # -----------------------------------------------------------------------
        # Four-state Gaussian lifecycle (PROVISIONAL=0, YOUNG=1, MATURE=2, FROZEN=3)
        # Requires --streaming_lifecycle_enabled to activate.
        # DEFAULT-ON (2026-06-30): the ISOLATED QUIESCENCE gate (mature out-of-view
        # Gaussians stop getting MCMC-perturbed). The doc-flagged -1.6 dB was the FULL
        # BUNDLE (quiescence + mature-anchor losses + stratified sampling); with the
        # anchor weights at 0.0 and sampling_mode="legacy" (both still default below),
        # only quiescence is active. Validated on kitchen1: train_psnr 20.8, opacity
        # tempered 0.96->0.66, no PSNR collapse. Do NOT raise mature_anchor_*_weight or
        # set sampling_mode=stratified without re-ablating (that revives the harmful bundle).
        # -----------------------------------------------------------------------
        self.streaming_lifecycle_enabled = True
        # Full-map lifecycle classification is O(N). Run it on an explicit
        # optimizer-update clock instead of rescanning every transaction.
        # 1 preserves the historical cadence; promoted live profiles choose a
        # larger bounded interval. Overdue work arms one future scan only.
        self.streaming_lifecycle_interval_updates = 1
        # Frame age (since birth) at which a YOUNG Gaussian becomes MATURE
        self.streaming_mature_age_frames = 15
        # Decaying isotropy regularizer on immature (provisional+young) Gaussians:
        # penalises (max_scale/min_scale - 1)^2, ramped from full strength at birth
        # to zero at maturity, to discourage newborn needles before support accrues.
        # 0.0 = off (no-op).
        self.streaming_young_isotropy_weight = 0.0
        self.streaming_young_isotropy_ramp_frames = 15
        # Minimum utility EMA for YOUNG→MATURE promotion (0 = age-only)
        self.streaming_mature_min_utility = 0.1
        # issue #1259 C1 arm: decouples lifecycle CLASSIFICATION (unchanged)
        # from MCMC MOBILITY policy for the position-noise-reversion gate in
        # utils/strategies/mcmc/energy.py. That gate is independent of
        # `allow_position_noise`/source_complete today -- a Gaussian that
        # matured during mapping (the resolved `live/canonical_mapping`
        # profile, configs/profiles/live/canonical_mapping.yml, uses
        # streaming_lifecycle_interval_updates=8 and
        # streaming_mature_min_utility=0.0, i.e. essentially age-only
        # maturity) stays immobile through the whole refinement tail
        # regardless of the "unconstrained refinement after mapping" framing.
        # When this flag is set AND the schedule reports source_complete
        # (mapping has ended) AND at least
        # streaming_post_mapping_mature_settle_updates successful optimizer
        # updates have elapsed since (an exposure-based settling fence so
        # MCMC does not immediately collide with PR #1313's post-mapping
        # depth-insertion pass), MATURE/FROZEN Gaussians receive real MCMC
        # position noise again instead of having it reverted. Off by
        # default: byte-identical behavior when unset. Deliberately does NOT
        # touch MCMC relocation-target eligibility (MATURE/FROZEN remain
        # non-targets always -- that anti-forgetting mechanism is untouched),
        # anchor gradient masking, MCMC proposal covariance, or any
        # regularizer weight -- see issue #1259.
        self.streaming_post_mapping_mature_mobility = False
        # Exposure-based settling fence for the flag above: number of
        # successful optimizer updates that must elapse after source_complete
        # first becomes true before C1 activates. Within the specified 8-32
        # range; unused when the flag above is off.
        self.streaming_post_mapping_mature_settle_updates = 16
        # Frame age at which a MATURE Gaussian becomes FROZEN (-1 = never)
        self.streaming_freeze_age_frames = -1
        # EMA decay for per-Gaussian utility tracking
        self.streaming_utility_ema_beta = 0.95
        # P0.5: require multi-view depth consensus (conflict <= support) for
        # YOUNG->MATURE promotion, so bad-depth floaters can't mature/hide.
        self.streaming_maturity_consensus = False
        # Phase 3 lifecycle-v2 bundle (default off): (a) utility_ema decays for
        # INVISIBLE Gaussians (recency requirement — no maturing on stale
        # opacity-proxy evidence), (b) MATURE/FROZEN demote back to YOUNG on
        # net multi-view depth conflict (maturity no longer monotonic).
        self.streaming_lifecycle_v2 = False
        self.streaming_lifecycle_v2_invis_decay = 0.98
        # Localized opacity reset for lifecycle-v2 demotion (default off).
        # Demotion alone re-exposes a suspect Gaussian to noise/relocation
        # but leaves its opacity untouched — it keeps rendering as if
        # trustworthy while it re-earns evidence. Clamp opacity to at most
        # this value on demotion so it must re-accumulate support. 0=off.
        self.streaming_suspect_opacity_reset = 0.0
        # Per-ingested-frame TensorBoard logging (tracking-confidence dict +
        # keyframe admission stats) fires once per frame ingested by the
        # drain loop, not once per iteration. Each call costs real
        # synchronous protobuf-serialization time (~15-20ms observed even
        # after enlarging the writer queue), so at high admission rates it
        # directly couples training it/s to frame arrival rate — exactly
        # the opposite of the non-blocking-drain design intent. Log every
        # Nth ingested frame instead; 1 = log every frame (old behavior).
        self.streaming_tb_frame_log_interval = 5
        # P0.4: idle/consolidation-regime sampler. When no frame has been
        # admitted for streaming_consolidation_idle_steps optimizer steps
        # (stream idle or bag ended), switch from the recency-biased ingest
        # sampler to a global sampler over ALL train cameras:
        #   off (default) | uniform | debt (debt-weighted, cached)
        self.streaming_consolidation_sampler = "off"
        self.streaming_consolidation_idle_steps = 500
        # 2.2 (bubble-as-local-bounds): compute MCMC population statistics
        # (energy dead quantiles, schedule rho=N/cap) over the ACTIVE bubble
        # population instead of the whole map, so every bubble runs at the same
        # MCMC operating point regardless of world size. Requires
        # streaming_spatial_bubble_enabled. Global stats remain in telemetry.
        self.streaming_local_stats = False
        # 2.4 (bubble-as-local-bounds): distance-weighted soft anchor loss —
        # loss-native replacement for the hard out-of-bubble freeze. lambda_max
        # for L = lam * mean(w_i * ||theta - anchor||^2); w_i ~ sigmoid distance
        # outside the bubble. 0 = off. Requires lifecycle + bubble center.
        self.streaming_soft_anchor_weight = 0.0
        # Multiplier applied to the soft-anchor weight in the idle regime
        # (no admission for streaming_consolidation_idle_steps) so global
        # refinement can move the outer world again.
        self.streaming_soft_anchor_idle_mult = 0.3
        # 5A: parallax-aware covisibility admission gate (golden rule). Rejects
        # candidates that mostly re-see occupied voxels (covis >= thresh) from
        # nearly the same rays (median parallax < deg) — the forward-walk flood
        # that defeats pose-delta + coverage gates. Never rejects forced
        # (max_gap) frames. Requires the coverage-gate machinery to be wired.
        self.streaming_kf_parallax_gate = False
        self.streaming_kf_parallax_deg = 3.0
        self.streaming_kf_covis_thresh = 0.6
        # Escape-plume fixes (kitchen1 forensic 2026-07-02: 55% of live mass
        # escaped into the depth-invalid frustum zone — RGB-only, never carved,
        # amplified by MCMC growth):
        # (1) carve alpha on rays with INVALID sensor depth up to max_depth
        self.streaming_carve_invalid_depth = False
        # carve only where a (2k+1)^2 neighborhood is ENTIRELY invalid —
        # spares IR-dark/glossy near objects (invalid speckle inside valid
        # surroundings) while still carving doorways/windows.
        self.streaming_carve_erode_px = 15
        # GPU-native depth fast path (issue #1348, docs/architecture/
        # 2026-08-31-gpu-native-decode-and-compressed-ingress.md, priority 1).
        # Default OFF. When True, the canonical materialization boundary
        # (eligsir/runtime/runtime_binding.py's runtime_binding_from_
        # runtime -> MappingSession.submit -> materialize_mapping_frame)
        # performs the raw-depth -> metric-float32 depth_scale division via a
        # pinned-H2D + on-device torch op instead of NumPy on the CPU. Output
        # is numerically identical (float32, meters, same values within
        # float precision) and the MappingFrame remains CPU-owned afterward,
        # so every existing downstream contract (canonical insertion's
        # CPU-metric-depth requirement, DeviceTransferOwner's later H2D of
        # cam._sensor_depth_cache) is unchanged. RGB/JPEG decode is untouched
        # (priority 2, gated on rocJPEG/nvJPEG availability, not implemented
        # here). Only affects the causal_mapping/CanonicalMappingRuntime
        # production path (eligsir/runtime/runtime_binding.py); the
        # legacy train_streaming.py loader (utils/streaming_frames.py /
        # eligsir/mapping/integration/depth_helpers.py) is untouched.
        self.streaming_gpu_native_depth = False
        # Tri-state ray semantics (Wave 3, PR W3-08, plan section 13): the
        # principled replacement for streaming_carve_invalid_depth's "any
        # invalid-depth pixel is free space out to max_depth" assumption,
        # which CLAUDE.md's "streaming_carve_invalid_depth vs. long-corridor
        # scenes" section documents collapsing a real floor2 run from
        # 22.7-33.4dB to 6-13dB by erasing legitimate, cross-frame-
        # corroborated geometry down a corridor. When enabled, an
        # invalid-depth pixel is only ever treated as free space
        # (RayEvidenceKind.CERTIFIED_FREE_SPACE) once >= streaming_ray_
        # semantics_min_views distinct, baseline-gated frames have already
        # directly measured that same 3-D region as free (via a DIFFERENT
        # frame's valid sensor depth) -- a single frame's own invalid-depth
        # mask alone can never certify anything (deliverable 1). Default OFF
        # (plan section 2.4.2: new behaviour defaults off); when off, the
        # free-space loss block below is byte-identical to pre-W3-08
        # behaviour, INCLUDING when streaming_carve_invalid_depth is also
        # True (deliverable 3: legacy carving stays available as an
        # explicit, unmodified legacy ablation). streaming_carve_invalid_depth
        # and streaming_ray_semantics_enabled should not both be True in the
        # same run -- ray semantics takes precedence and a one-time warning
        # is logged (see eligsir/optimization/gaussian/objective.py).
        self.streaming_ray_semantics_enabled = False
        self.streaming_ray_semantics_min_views = 3
        self.streaming_ray_semantics_min_baseline_m = 0.03
        self.streaming_ray_semantics_voxel_size_m = 0.05
        self.streaming_ray_semantics_min_pose_confidence = 0.0
        # Geometry (depth) vs. appearance supervision are independently
        # weighted (deliverable 6): streaming_depth_loss_weight already
        # controls geometry; the appearance-only auxiliary update below has
        # its own, unrelated weight.
        self.streaming_appearance_only_enabled = False
        self.streaming_appearance_only_weight = 0.0
        self.streaming_appearance_only_interval = 20
        self.streaming_appearance_only_reliability_thresh = 0.35
        # W3-48 (2026-08-10): update_cameras_extent()'s monotonic, outlier-blind
        # max() over all camera positions ever seen was root-caused
        # (docs/architecture/2026-08-10-w3-30-init-seed-bootstrap-confidence-gate.md)
        # as the real floor2 scale-explosion amplifier -- a single poorly-
        # converged pose (a covariance-poor tracking moment, not necessarily a
        # full relocalization failure) permanently ratchets cameras_extent up
        # by 5-10x, which then feeds MCMC noise/scale regularization for the
        # WHOLE map for the rest of the run, with no way to correct itself.
        # A jump beyond this multiple of the current cameras_extent in a single
        # recompute window is treated as suspect and withheld from the extent
        # (the camera itself is still used for training -- this only gates the
        # extent bookkeeping) until corroborated by streaming_extent_outlier_corroboration
        # further consecutive recompute windows that keep seeing a similarly
        # large extent -- distinguishing a one-off pose glitch (self-corrects,
        # never corroborated) from genuine sustained scene growth (e.g. walking
        # further down a long corridor, which keeps producing a large extent on
        # every subsequent window). 1.5 = a >50% jump in one window is suspect;
        # tune higher for scenes with genuinely fast legitimate expansion.
        self.streaming_extent_growth_ceiling_multiplier = 1.5
        self.streaming_extent_outlier_corroboration = 2
        # (2) prune splats farther than max_depth + this margin from EVERY
        #     camera (never depth-verifiable). 0 = off.
        self.streaming_reach_prune_margin_m = 0.0
        # (3) recycle instead of delete: unreachable splats get opacity crushed
        #     so MCMC dead-relocation teleports them back onto in-scene parents.
        self.streaming_reach_recycle = False
        # (3b) make recycled (crushed) splats relocatable: reset lifecycle_state
        #     to PROVISIONAL so the mature-gate doesn't shield them from MCMC
        #     dead-relocation. Past densify_until_iter (relocation no longer
        #     runs) unreachable splats are hard-pruned instead of crushed.
        self.streaming_recycle_relocatable = False
        # Export dead-mass prune: when > 0, PLY snapshots exclude splats with
        # sigmoid(opacity) <= this threshold. 0 = export everything (off).
        self.streaming_export_min_opacity = 0.0
        # Bootstrap seed decay: one-shot when n_frames_ingested crosses this
        # value, clamp opacity logits of birth_frame==0 splats to
        # inverse_sigmoid(0.1) and demote them to YOUNG so they must re-earn
        # maturity. 0 = off.
        self.streaming_bootstrap_seed_decay_frames = 0
        # Issue #346 causal manipulation: freeze the exact living bootstrap
        # lineage from iteration K onward. -1 is off and behavior-neutral.
        self.streaming_seed_lineage_freeze_from_iter = -1
        # Write one exact mid-mapping causal-fork checkpoint after iteration K.
        # -1 is fully disabled and preserves the ordinary streaming path.
        self.streaming_exact_fork_iteration = -1
        # (4) growth/relocation parents must be recently visible (vis_ema>0.01)
        self.streaming_growth_parents_visible_only = False
        # Local capacity density (Gaussians per m^3) for cap_local =
        # density * (4/3)*pi*R^3. 0 = keep global cap in the schedule even
        # when local_stats is on (quantiles only).
        self.streaming_local_cap_density = 0.0
        # -----------------------------------------------------------------------
        # Soft-anchor anti-fade losses for MATURE Gaussians (Component C)
        # All three weights default to 0.0 (opt-in). Recommended: ~0.05 each.
        # -----------------------------------------------------------------------
        self.streaming_mature_anchor_xyz_weight = 0.0
        self.streaming_mature_anchor_scale_weight = 0.0
        self.streaming_mature_anchor_opacity_weight = 0.0
        # Fix (2026-07-07): by default the YOUNG->MATURE transition re-snapshots
        # anchor_xyz at the CURRENT (possibly already-drifted) position, which
        # defeats the point of the xyz anchor loss. Set True to instead keep the
        # true birth/depth-insertion position (already tracked in anchor_xyz and
        # never otherwise overwritten) as the anchor target throughout life.
        self.streaming_mature_anchor_use_birth_xyz = False
        # Absolute (not relative/percentile) hard cap on Gaussian scale, applied
        # unconditionally -- including to MATURE/FROZEN Gaussians, which the
        # existing relative scale_kill_threshold dead-mask explicitly excludes
        # (utils/strategies/mcmc_strategy.py: `dead_mask & (_lc < 2)`). Safety
        # net for the >5m-radius floater population found 2026-07-07 that a
        # relative/support-gated scale kill can never reach once matured.
        # 0 = off (opt-in; no known legitimate scene geometry this large).
        self.streaming_max_gaussian_radius_m = 0.0
        # -----------------------------------------------------------------------
        # Stratified replay sampling (Component D)
        # -----------------------------------------------------------------------
        # sampling_mode: "legacy" (existing ring-buffer logic) or "stratified"
        self.streaming_sampling_mode = "legacy"
        # Comma-separated ratios for [recent, covisible, global_reservoir, hard_frames]
        # Must sum to ~1.0. Only used when streaming_sampling_mode = "stratified".
        self.streaming_sampling_ratios = "0.70,0.15,0.10,0.05"
        # Depth comparison export: save rendered_depth vs sensor_depth side-by-side PNGs
        # for every ingested training frame. Output goes to <model_path>/depth_comparison/.
        # Set to True to enable; requires streaming_depth_loss_weight > 0 (depth must be rendered).
        self.streaming_export_depth_comparison = False
        # Minimum shared Gaussians to consider two frames covisible
        self.streaming_covisible_min_shared = 200
        # Number of recent high-loss frames to keep as "hard frames"
        self.streaming_hard_frame_history = 8
        # -----------------------------------------------------------------------
        # KNN insertion dedup (Component E)
        # -----------------------------------------------------------------------
        self.streaming_insert_knn_dedup = False
        # Radius factor: candidate rejected if nearest existing Gaussian is within
        # depth * streaming_insert_knn_radius_factor metres
        self.streaming_insert_knn_radius_factor = 0.005
        # Max existing Gaussians to query (beyond this, subsample via voxel grid)
        self.streaming_insert_knn_max_existing = 200000
        # Normal-aware dedup: only reject a near candidate if the nearest existing
        # splat faces the SAME way (|dot(normals)| > cos), so layered/opposite-facing
        # surfaces (thin walls, front/back) aren't false-merged. Needs the knnIndices
        # kernel (gathers the neighbour's normal via the returned index). Off by default.
        self.streaming_insert_knn_dedup_normal_aware = False
        self.streaming_insert_knn_dedup_normal_cos = 0.85
        # Hot-path KNN scale-smoothness regularizer (amortized cached neighbour
        # graph). weight 0 = off (default). The k-NN search is rebuilt only every
        # --knn_reg_interval iters (cached on the model; also on count change); the
        # per-iteration cost is just a gather on detached indices + elementwise loss
        # (~0.4 ms @ ~22k Gaussians). Penalizes a Gaussian's scale deviating from
        # its k nearest neighbours' mean (local scale coherence). See
        # docs/perf / knn-hotpath-profiling.
        self.knn_reg_weight = 0.0
        self.knn_reg_k = 8
        self.knn_reg_interval = 100
        # -----------------------------------------------------------------------
        # Insertion telemetry (Component F)
        # -----------------------------------------------------------------------
        self.streaming_insertion_debug = False
        self.streaming_insertion_debug_ply = False
        # Per-frame keyframe-ADMISSION telemetry (2026-07-30): durable JSONL
        # sidecar (<model_path>/keyframe_admission_log.jsonl) recording, for
        # every ingested frame that produced admission stats (any admission
        # mode, not just hybrid_keyframe), the same dict already computed for
        # in-run TB scalars/aggregate counters -- frame index, iteration,
        # admitted?, reason, and (when --streaming_keyframe_quality_enabled)
        # the blur/exposure quality_weight. Default ON and cheap (one small
        # JSON line per ingested frame, append-only, best-effort/never crashes
        # training) because the lack of this was the exact gap that made the
        # floor3 adaptive-quality-mechanism result unverifiable this session --
        # PSNR alone can't distinguish "the mechanism engaged and helped",
        # "engaged and didn't matter", or "never engaged" without a per-frame
        # record of what admission actually did. Infrastructure for comparing
        # future keyframe-selection methods (parallax/coverage,
        # tracking-confidence, blur/exposure) against each other.
        self.streaming_admission_log_enabled = True
        # -----------------------------------------------------------------------
        # Tracking-confidence keyframe-admission signal (2026-07-30; see
        # eligsir/mapping/scheduler/tracking_confidence.py's module
        # docstring for the full design + the real ORB-SLAM3/build/topic
        # verification done for it). The upstream signal now exists (a
        # patched rgbd_inertial_node_cpp publishing /slam/tracking_status,
        # eligsir-orbslam3 commit b45a57b) and the Python consumer is
        # wired (eligsir/io/sources/ros2_live.py). Default OFF regardless
        # -- no GPU validation run queued, blocked on DATA (no genuinely
        # tracking-degraded scene in inventory; floor3/kitchen1 are
        # well-tracked by construction and this mechanism self-disables
        # there by design, so a run there would be an uninformative null,
        # not a review-gated question). Flip this on deliberately once a
        # suitable scene exists and a real validation run has been done.
        self.streaming_tracking_confidence_enabled = False
        self.streaming_tracking_confidence_warmup = 20
        self.streaming_tracking_confidence_ema_alpha = 0.05
        self.streaming_tracking_confidence_floor = 0.4
        # W3-30 (2026-08-10): online-SLAM bootstrap (initialize_from_frames'
        # frame_filter caller in train_streaming.py) previously accepted the
        # first N *valid* tracking results unconditionally, ignoring
        # TrackingResult.confidence entirely -- unlike every other consumer
        # of this signal in the file (streaming_tracking_confidence_enabled
        # above, MappingScheduler's tracking-confidence gate). Closes that
        # real gap as defense-in-depth. NOT the floor2 init-seed
        # scale-explosion fix, despite motivating this flag's addition --
        # checking real floor2 output disproved the bootstrap hypothesis
        # (input_seed.ply bbox is ~6.2m and consistent across 6 independent
        # runs; the actual explosion is a post-bootstrap, run-to-run
        # nondeterministic event under byte-identical configs, pointing at
        # scene/streaming_scene.py's update_cameras_extent() instead -- see
        # docs/architecture/2026-08-10-w3-30-init-seed-bootstrap-confidence-gate.md
        # for the full evidence and the real next step).
        # 0.0 = off (default; matches every non-ROS adapter's fixed
        # confidence=1.0, so this is a no-op for them regardless of value).
        # NOTE: setting this > 0.0 can hard-fail bootstrap (RuntimeError) on
        # a scene where SLAM covariance is uniformly mediocre -- see the doc.
        self.streaming_bootstrap_min_confidence = 0.0
        # Map-reinitialization hard veto (Phase 2c, 2026-08-06) -- distinct
        # from and independent of streaming_tracking_confidence_enabled
        # above (a data-integrity issue, not a quality-tuning knob; applies
        # unconditionally whenever a frame carries _tracking_status). How
        # many consecutive OK-tracking frames must be seen after a
        # post-bootstrap map re-init before admission resumes -- see
        # train_streaming.py's _check_tracking_reinit.
        self.streaming_tracking_reinit_settle_frames = 30
        # Second, independent consumer of the same /slam/tracking_status_stamped
        # signal (2026-08-04): feeds TrackingResult.confidence on the
        # dataset_pose/ros2-live ingestion path (train_streaming.py, where
        # confidence was previously hardcoded to 1.0 for every frame), making
        # MappingScheduler's three-tier confidence gate (mapping_scheduler.py)
        # reachable on real degraded ORB-SLAM3 tracking. Independent of
        # --streaming_tracking_confidence_enabled above (KeyframeAdmitter
        # quality-weight downweighting) -- that mechanism is untouched; this
        # is a second consumer of the same upstream data, feeding a different
        # downstream gate. Default OFF for the same reason as above: no
        # validation run yet confirms the gate firing on real degraded
        # tracking is a net-positive behavior change on any real scene.
        self.mapping_scheduler_tracking_confidence_enabled = False
        self.mapping_scheduler_tracking_confidence_warmup = 20
        self.mapping_scheduler_tracking_confidence_ema_alpha = 0.05
        self.mapping_scheduler_tracking_confidence_floor = 0.4
        # Geometry-prior abstraction and soft surface constraints.
        self.geometry_prior_mode = "sensor_only"  # sensor_only | sensor_vggt_opportunistic | sensor_vggt_offline
        self.surface_normal_mode = "finite_difference"  # finite_difference | pca_plane
        self.surface_normal_window = 5
        self.surface_normal_max_depth_jump = 0.05
        self.surface_normal_min_valid_ratio = 0.6
        self.surface_loss_enabled = False
        self.surface_anchor_normal_weight = 0.0
        self.surface_anchor_tangent_weight = 0.0
        self.surface_normal_align_weight = 0.0
        self.surface_thinness_weight = 0.0
        self.surface_loss_min_confidence = 0.1
        self.surface_confidence_depth_loss = False
        self.streaming_surface_cell_dedup = False
        self.surface_mcmc_normal_scale = 0.15
        self.surface_mcmc_tangent_scale = 1.0
        self.surface_mcmc_provisional_normal_scale = 0.75
        self.energy_w_geometry_support = 0.0
        self.energy_w_duplicate = 0.0
        self.energy_w_surface_drift = 0.0
        self.energy_w_free_space = 0.0
        self.energy_w_budget = 0.0
        # Save estimated trajectory as TUM .txt + PNG plot + ATE/RPE metrics
        # to <model_path>/trajectory_eval/. Default on.
        self.streaming_trajectory_eval = True
        self.streaming_resolution = 1  # integer divisor: scale input frames before odometry and splatting
        # Benchmark mode: disable final report, progress video, LPIPS, and train metrics.
        # Use for core-training throughput profiling to avoid report VRAM overhead.
        self.streaming_benchmark_mode = False
        # -----------------------------------------------------------------------
        # Tracking frontend selection (see eligsir/tracking/factory.py)
        # -----------------------------------------------------------------------
        # Selects the TrackingProvider/frontend used for pose estimation.
        # "dataset_pose" consumes the pose the frame source already established
        #   -- a dataset's baked trajectory, or a ros2:// source's own
        #   pose/odom/TF join (default).
        # "precomputed" is kept as a deprecated alias of dataset_pose.
        # "open3d_odometry_live" runs frame-to-frame Open3D RGBD odometry.
        # "open3d_icp_live" runs the Open3D ICP tracker.
        # REMOVED by #1075: "ros_slam" (ROS pose/odom/TF subscription and
        #   conversion now live in the eligsir-ros2 package) and
        #   "gs_rendered_rgbd_odometry" (a parked stub that was never
        #   selectable). Passing either fails with actionable guidance.
        self.pose_source = "dataset_pose"
        # Trust tier for the poses driving this run (see
        # eligsir/slam/pose_provenance.py). "auto" detects from the bag
        # pose topic / dataset directory format; pass one of dataset_gt,
        # colmap_pseudo_gt, slam_estimated, unknown to override auto-detection
        # (e.g. a renamed topic auto-detection can't resolve).
        self.pose_provenance = "auto"
        # REMOVED by #1075 together with the ros_slam backend they configured:
        # slam_pose_topic / slam_pose_msg_type / slam_pose_timeout_sec /
        # slam_pose_staleness_sec / slam_covariance_decay_k. Their sole reader
        # was ROSSLAMBackend's constructor; ROS pose subscription, staleness
        # and covariance->confidence handling now belong to eligsir-ros2,
        # which stamps an already-converted TrackingObservation onto each
        # SensorPacket. Keeping them declared would have left five CLI flags
        # that silently do nothing.
        # Bootstrap configuration.
        self.bootstrap_random_fallback = False     # True = allow random PCD if SLAM never provides pose
        self.bootstrap_max_attempts_multiplier = 10  # try up to n_init * multiplier frames to find posed frames
        # Confidence thresholds for mapping gates (Phase 8 — pass-through in Phase 7).
        self.tracking_min_conf_persistent = 0.7
        self.tracking_min_conf_provisional = 0.4
        # -----------------------------------------------------------------------
        # Live viewer (Phase 1.5+)
        # -----------------------------------------------------------------------
        # "none" (default), "rerun", "ros2_rviz", or "viser" (issue #570/#572;
        # experimental, opt-in only -- routes through CanonicalViewer, unlike
        # the other three presets which are unchanged legacy backends).
        self.viewer = "none"
        # issue #570/#574/#583: explicit, comma-separated canonical sink list
        # (e.g. "viser" or "viser,recorder"), independent of --viewer. Empty
        # (default) means "no override" -- --viewer alone still resolves to
        # its usual preset/legacy backend, so this flag is fully behavior-
        # neutral unset. When set, it takes over sink selection: it is
        # resolved via eligsir.viewers.canonical.resolve_preset_sinks and
        # built through the same CanonicalViewerAdapter/PreviewSink machinery
        # --viewer viser already uses, so e.g. --viewer_sinks viser reaches
        # the identical constructed object as --viewer viser without having
        # to change --viewer. "--viewer none --viewer_sinks ..." is refused
        # (an explicitly-disabled viewer must not be silently re-enabled).
        # Only sink names core itself can build are accepted here today
        # ("viser"); other canonical sink names recognized by the parser
        # (e.g. "rviz", implemented in the separate eligsir-ros2 package
        # per issue #1073's core/ROS hard cut) raise a clear, actionable
        # error rather than silently doing nothing -- see
        # eligsir/viewers/registry.py's _SINK_FACTORIES.
        self.viewer_sinks = ""
        self.viewer_rerun_port = 9876
        self.viewer_rerun_max_gaussians = 50000
        self.viewer_viser_port = 8080
        self.viewer_viser_host = "0.0.0.0"
        # issue #1402: caps the live map preview sample forwarded to the
        # canonical viewer (CanonicalViewerConfig.map_max_gaussians,
        # currently only consumed by the viser sink's VersionedMapOutputLane
        # -- other viewers ignore this flag). Diagnosed as a likely cause of
        # a "large blurry discs" artifact: sample_output_from_model's
        # min(total, cap) uniform-stride selection is not a valid
        # Gaussian-splatting LOD, so a live map well above this count is
        # shown at a fraction of its real density. 0 = no cap, clamped to
        # VIEWER_MAP_MAX_GAUSSIANS_HARD_CEILING (2,000,000) rather than truly
        # unbounded -- see eligsir/viewers/canonical.py's
        # resolve_viewer_map_max_gaussians. The 100_000 default here matches
        # CanonicalViewerConfig's own pre-existing default exactly, so an
        # unset flag changes nothing about any run predating this option.
        self.viewer_map_max_gaussians = 100_000
        self.viewer_rviz_topic = "/eligsir/gaussian_splats"
        # Which RViz splat transport(s) to publish: "all" (default; every
        # available transport — legacy+blob+tile+snapshot_ref, ~4x snapshot
        # serialization + a per-cycle PLY write) or a single transport
        # (legacy_splatarray / blob_snapshot / tile_stream / snapshot_ref) to
        # publish only that one. Set to the transport your RViz config subscribes
        # to for the cheapest publish path. (Previously this arg was ignored.)
        self.viewer_rviz_transport = "all"
        self.viewer_rviz_blob_topic = "/eligsir/splat_map/blob_chunks"
        self.viewer_rviz_tile_topic = "/eligsir/splat_map/tile_chunks"
        self.viewer_rviz_session_id = "eligsir"
        self.viewer_rviz_chunk_max_bytes = 1048576
        self.viewer_rviz_tile_size_m = 1.0
        self.viewer_rviz_publish_interval = 100
        self.viewer_rviz_min_publish_period_s = 2.0
        self.viewer_rviz_camera_interval = 1
        self.viewer_rviz_publish_clock = True
        self.viewer_rviz_max_gaussians = 0
        self.viewer_rviz_only_when_map_version_changed = False
        self.viewer_rviz_world_frame = "map"
        # -----------------------------------------------------------------------
        # GaussianEnergyPolicy modes (Phases 3-6, Phase 10, Phase 15)
        # All default to "legacy_*" to preserve prior behaviour as ablation baseline.
        # Phase 15 flips these to the new defaults; see docs/ROADMAP.md Phase 15.
        # -----------------------------------------------------------------------
        self.energy_mode = "legacy_utility"          # "population_energy" in Phase 15
        self.energy_dead_mode = "legacy_clauses"      # "energy_quantile" in Phase 15
        self.mcmc_proposal_mode = "legacy_clamped"   # "trust_region" in Phase 15
        self.soft_lifecycle_enabled = False            # True in Phase 15
        # issue #385: soft-lifecycle sharpness, read by compute_maturity_logit().
        # Declared on GaussianEnergyConfig but previously unreachable -- turning
        # soft_lifecycle_enabled on gave you a fixed, untunable maturity curve.
        # Defaults equal the dataclass defaults, so declaring them is
        # behaviour-neutral.
        self.soft_lifecycle_support_k = 10.0
        self.soft_lifecycle_age_k = 5.0
        # Phase 5: differentiable energy loss flag (replaces active_reg+entropy+eff_count call sites)
        # NOTE: Default False; the parity assertion in tests/test_energy_wiring_parity.py documents
        # that compute_differentiable_loss uses raw opacities and the live path uses activated
        # opacities — parity holds within float32 tolerance for the entropy term but NOT for
        # active_reg (different calling convention). Enable only after validating on your scene.
        self.energy_loss_enabled = False               # True in Phase 15 after parity validation
        # Population energy extra weights not in OptimizationParams (Phase 3+)
        self.energy_w_opacity = 1.0       # was energy_w_alpha in OptimizationParams
        self.energy_w_anisotropy = 0.3
        self.energy_w_conflict = 0.5
        self.energy_w_age = 0.2
        # Simplified per-frame-scalar depth-residual gate: temperature multiplier
        # on the growth-utility gradient term (see compute_gaussian_utility). 0.0 = off.
        self.energy_w_depth_residual = 0.0
        # Energy death thresholds (Phase 4+)
        self.energy_dead_quantile = 0.05
        self.energy_dead_temperature = 10.0
        # Trust-region MCMC (Phase 6+)
        self.mcmc_trust_radius = 0.05
        self.mcmc_trust_support_power = 0.5
        self.mcmc_trust_opacity_power = 0.5
        # Submap (Phase 12+)
        self.submap_frames_per_submap = 50
        # Loop closure (Phase 11+)
        self.loop_force_field_interval = 0   # 0 = disabled
        self.loop_force_field_match_radius = 0.1
        # RMSE (m) above which score_force_field's coherence collapses to 0 (see
        # loop_closure/gaussian_force_field.py). Must be widened alongside
        # loop_force_field_match_radius: a looser match radius finds matches over
        # a larger drift range, which naturally increases the fitted SE3's
        # residual RMSE, so the two knobs need to move together or coherence
        # never clears loop_closure_min_coherence no matter how many inliers
        # are found. Found live on floor3 (2026-07-25): widening match_radius
        # from 0.1m to 0.3-0.5m did find real inliers (86-288) with rmse
        # 0.19-0.24m, but coherence stayed exactly 0.0 the entire run against
        # this stale default, so no loop edge was ever added.
        self.loop_force_field_max_rmse = 0.05
        # Phase 13: PoseGraph loop closure
        self.loop_closure_enabled = False           # master switch; off by default
        self.loop_closure_min_coherence = 0.6       # min force-field coherence to add a loop edge
        self.loop_closure_min_inliers = 10          # min inliers to add a loop edge
        self.loop_closure_optimize_interval = 500   # iterations between pose-graph optimization
        # Phase 14: apply submap correction
        self.loop_closure_apply_correction = False  # apply corrections to Gaussian means/rotations
        # Phase 2b diagnostic (2026-08-06): dump PoseGraph.debug_snapshot()
        # before/after every optimize() call to <model_path>/loop_closure_debug.jsonl
        # -- investigating why a corrected submap's pose can end up far
        # outside the scene's real extent. Off by default; not a permanent
        # verbose log path.
        self.loop_closure_debug = False
        # -----------------------------------------------------------------------
        # OnlineController (Milestone 2 — map-age-aware scheduling)
        # Async ingestion: run per-frame CPU prep (frame->camera, candidate
        # collection) on a background worker thread, overlapped with the optimizer
        # GPU step, so the GPU stops idling (~60%->~86%) while frames arrive. The
        # optimizer stays the SOLE writer of params/optimizer-state/occupancy
        # (single-writer handoff, no lock); the worker only produces read-only
        # camera + candidate batches. Default off => byte-identical to today.
        self.streaming_async_ingest = False
        # Max frames the async worker prepares ahead of the optimizer (bounded
        # queue; back-pressure defers, never drops -> coverage preserved).
        self.streaming_async_ingest_depth = 4
        # Variant C (batched ingestion): when True, the drain loop ingests up to
        # streaming_batch_ingest_max frames per training iteration and defers the
        # per-frame cameras_extent readback to a single call at the end of the
        # batch. Coverage-preserving (every drained frame is still ingested in
        # cursor order; admission renders stay per-frame and sequential because
        # each reads the model the previous add_points mutated). Default off =>
        # byte-identical. NOTE: A's profiling shows the extent readback is already
        # every-8 gated and per-frame syncs are not a measurable cost, so this is
        # expected to land ~flat — it exists to confirm that empirically.
        self.streaming_batch_ingest = False
        self.streaming_batch_ingest_max = 4

        # -----------------------------------------------------------------------
        # Bounded-queue prepared-candidate pipeline (PR W3-03, plan section 8).
        # Default OFF => byte-identical to today: the coordinator never
        # constructs BoundedCandidatePipeline, the FramePreparationWorker
        # thread never starts, and the existing inline
        # decode/quality-score/admission path (_frame_to_camera +
        # KeyframeAdmitter, run synchronously on the optimizer thread) is
        # unchanged. When True, MappingCoordinator.enable_bounded_queue_pipeline()
        # is called and raw/tracked frames are additionally routed through
        # the bounded input/output/commit queues (see
        # eligsir/mapping/coordinator.py's BoundedCandidatePipeline and
        # eligsir/runtime/workers/frame_preparation_worker.py) so a
        # rejected candidate's decode/quality-scoring cost moves off the
        # optimizer thread and never allocates a full-resolution GPU image.
        # This is forward infrastructure: it is NOT yet the frame source
        # for the live admission decision (that migration, and the decision
        # to flip this flag's default, is deferred to a later PR once
        # cross-scene parity evidence exists -- see plan section 8's
        # non-negotiable deliverable #6 and PR W3-13).
        self.streaming_bounded_queue_pipeline = False
        self.streaming_bounded_queue_input_capacity = 4
        self.streaming_bounded_queue_output_capacity = 4
        self.streaming_bounded_queue_commit_capacity = 8

        # -----------------------------------------------------------------------
        # Deferred-admission SLAM worker integration (W3-15, completing W3-04's
        # unmet "no SLAM backend executes synchronously in the training loop"
        # promise). Default OFF -- the existing synchronous _slam_backend.track()
        # call stays the default until this has its own promotion gate (per
        # §2.4.2, "new behavior defaults off/shadow"). ONLY applies to the
        # branch that already calls _slam_backend.track() in-loop, i.e. an
        # in-process SLAM backend computing its own odometry
        # (open3d_odometry_live) -- see eligsir/slam/pose_sources.py.
        # A dataset-baked-pose run never reaches this branch at all, so
        # this flag is a documented no-op for those runs.
        #
        # When enabled, the next source frame is submitted to a real
        # ProcessSLAMWorker subprocess (non-blocking) and is only admitted
        # into the mapping pipeline once its real, asynchronously-computed
        # pose has actually arrived -- never a stale or predicted pose (see
        # eligsir/mapping/scheduler/deferred_slam_admission.py's module
        # docstring for the full rationale; this mirrors the exact bug class
        # documented as motion-pregate-silent-admission-corruption /
        # 3dgs-slam-fragile-to-coordinate-discontinuity in project memory).
        # Backends with requires_gaussian_map=True (e.g.
        # gs_rendered_rgbd_odometry) are refused at construction time -- the
        # worker's child process has no access to the live Gaussian model.
        self.streaming_slam_worker_deferred_admission = False
        self.streaming_slam_worker_capacity = 4
        self.streaming_slam_worker_pending_timeout_s = 5.0

        # -----------------------------------------------------------------------
        # Shadow frame-utility scoring (PR W3-06, plan section 11). Default OFF
        # => byte-identical to today: ShadowUtilityPolicy is constructed
        # disabled (near-zero-cost no-op, same pattern as EvidenceRequirements
        # .none() in W3-05), the extra evidence flags below are never set, and
        # _keyframe_admission_closure's return value (admitted/stats/render_pkg)
        # is completely unaffected -- this only SCORES AND LOGS candidate
        # frames after the real KeyframeAdmitter decision has already been
        # made. See eligsir/mapping/admission/shadow_policy.py's docstring
        # for the deliverable-1 ("does not alter keyframe decisions, replay,
        # optimization, or mutation") argument in full. W3-07 is the (separate,
        # future) PR that would act on these scores.
        self.streaming_shadow_utility_enabled = False
        # Issue #1365 M5: observational-only Gaussian population-policy
        # bridge.  Disabled by default; enabling it never changes MCMC,
        # lifecycle, depth insertion, scheduler selection, or map mutation.
        self.streaming_population_shadow_enabled = False
        # Issue #1365 M5/M6: the opt-in live successor to the shadow bridge.
        # It is deliberately independent from shadow telemetry so the runtime
        # can reject an incomplete composition rather than accidentally
        # activating a policy when only observation was requested.
        self.streaming_population_live_enabled = False
        # Iterations of training a just-admitted camera must accumulate before
        # its "realized gain" (loss_before - loss_after) is recorded --
        # see eligsir/mapping/admission/realized_gain.py.
        self.streaming_shadow_utility_realized_gain_horizon = 200
        # support_threshold passed to EvidenceBuilder.build_summary()'s
        # under_supported_mass reduction -- mirrors the LSOM-style threshold
        # convention already used in utils/mcmc_schedule.py.
        self.streaming_shadow_utility_support_threshold = 0.01
        # CLI-overridable UtilityWeights (utility_contracts.py) -- lets the
        # W3-07 frozen-Gaussian-policy ablation (deliverable 4) isolate one
        # utility formula term at a time (e.g. zero everything except
        # w_new_surface) without editing code. Defaults exactly match
        # UtilityWeights()'s dataclass defaults, so leaving all of these
        # alone reproduces the same score every prior run already computed.
        self.streaming_shadow_utility_w_new_surface = 1.0
        self.streaming_shadow_utility_w_residual = 1.0
        self.streaming_shadow_utility_w_debt = 1.0
        self.streaming_shadow_utility_w_gradient = 1.0
        self.streaming_shadow_utility_w_parallax = 0.5
        self.streaming_shadow_utility_w_redundancy = 1.0
        self.streaming_shadow_utility_w_specular = 1.0
        self.streaming_shadow_utility_w_cost = 0.25
        # Issue #118: `under_supported_mass` (evidence/reducers.py's
        # opacity-sum-over-the-entire-live-map term) is unbounded (observed
        # range 0-6708 in the frozen W3-29 fr3 data) while every other
        # utility component is a bounded ~[0,1] fraction/mean. Summed at
        # w_debt=1.0 alongside those, the unbounded term swamps the rest by
        # 3-4 orders of magnitude -- the W3-29 diagnosis found
        # corr(predicted_utility, under_supported_mass) = 0.9999999999997,
        # i.e. the aggregate score is in practice just this one term. Default
        # "raw" reproduces that exact (broken) pre-existing behavior
        # byte-for-byte -- this flag changes nothing unless explicitly set.
        # "per_gaussian" divides under_supported_mass by n_gaussians before
        # it enters the utility formula (only the formula input -- the
        # realized_gain.jsonl diagnostic fields `under_supported_mass_before`
        # / `_after` still log the raw, un-normalized value, since those are
        # a separate LSOM-style diagnostic, not a scoring input). Dividing by
        # live-Gaussian count turns the term into "mean per-Gaussian
        # under-supported opacity mass", which is bounded to ~[0,1] the same
        # way an opacity value itself is (each Gaussian contributes at most
        # 1.0 of opacity), putting it on the same footing as the other
        # components instead of a raw map-wide sum. Gated behind this flag
        # (default off) rather than changed unconditionally because
        # `under_supported_mass`/`predicted_utility` is a live, real
        # consumer of admission decisions in
        # `--streaming_admission_rollout_mode` values other than the
        # (default) "shadow" -- see `AdmissionRolloutPolicy.decide`'s
        # `utility_admit_threshold` comparison in rollout_policy.py -- so a
        # silent semantics change here would retroactively alter what an
        # `active`/`advisory`/`active_with_budget` run does, which is not
        # acceptable while other machines may have jobs queued against
        # unmodified `main`. See
        # docs/architecture/2026-08-12-issue118-utility-debt-rescale.md.
        self.streaming_shadow_utility_debt_normalization = "raw"

        # -----------------------------------------------------------------------
        # Active, evidence-driven admission rollout policy (PR W3-07, plan
        # section 12). Consumes the W3-06 utility score above (this flag
        # force-enables that scorer if it wasn't already on -- see
        # train_streaming.py's `_shadow_utility_enabled` computation) to
        # optionally act on, rather than just log, keyframe admission
        # decisions.
        #
        # "shadow" (default): identical behavior to W3-06 -- no decision
        # changes, matches every historical run. **Do not change this
        # default without cross-scene evidence justifying the promotion**
        # (deliverable 6) -- see docs/architecture/ completion doc for
        # W3-07 for what evidence exists so far (as of this PR: none; the
        # frozen-Gaussian-policy ablation this PR ran is a real-but-narrow
        # smoke-scale signal, not the cross-scene evidence bar this
        # constraint requires).
        # "advisory": may reject a baseline admission, may never admit a
        # baseline rejection (can only shrink the admitted set).
        # "active": owns the admit/reject decision in both directions,
        # still strictly subordinate to the hard safety gates (tracking
        # re-init veto, etc. -- see rollout_policy.py's module docstring).
        # "active_with_budget": same as "active" plus an explicit
        # frame-count/compute budget (see the three budget flags below);
        # requires at least one of the two budget flags to be > 0 to have
        # any real effect (both default to 0 = unlimited, same as "active").
        self.streaming_admission_rollout_mode = "shadow"  # shadow | advisory | active | active_with_budget
        # Utility threshold a candidate must clear (>=) for
        # advisory/active/active_with_budget to want to admit it. The raw
        # utility formula (see utility_contracts.py's module docstring) is
        # not bounded to [0, 1] -- 0.0 is a reasonable "net positive value"
        # default, not a calibrated cutoff.
        self.streaming_admission_utility_threshold = 0.0
        # active_with_budget-only: 0 = unlimited (matches plain "active").
        # NOTE: train_streaming.py never calls MappingBudgetController.
        # reset_window() today, so in practice this is a WHOLE-RUN budget,
        # not a periodically-refreshing one, despite the "per window"
        # language in budget_controller.py's docstrings (that class
        # supports periodic reset for a future caller; this PR does not
        # yet wire one in). Size these flags accordingly (e.g. relative to
        # the dataset's total expected frame count), not as a per-minute
        # rate.
        self.streaming_admission_budget_max_frames = 0
        self.streaming_admission_budget_max_compute = 0.0
        self.streaming_admission_budget_max_burst_steps = 200

        # -----------------------------------------------------------------------
        # Master switch: when True, the OnlineController wraps the
        # MappingScheduler and overrides steps/insertion/growth per mode.
        self.streaming_online_controller = False
        # Map-age thresholds (seconds). Hysteresis: CATCH_UP enters above
        # catch_up_enter and exits below catch_up_exit; TRACK_ONLY enters
        # above track_only_enter (hard ceiling) and exits below track_only_exit.
        self.streaming_map_age_target_s = 0.5
        self.streaming_map_age_catch_up_enter_s = 0.8
        self.streaming_map_age_catch_up_exit_s = 0.3
        self.streaming_map_age_track_only_enter_s = 1.5
        self.streaming_map_age_track_only_exit_s = 0.5
        # Train-step clamps applied by the controller in BALANCED / CATCH_UP.
        self.streaming_online_min_steps = 1
        self.streaming_online_max_steps = 50
        # EMA smoothing for measured iteration cost (Phase B adaptive steps).
        self.streaming_online_iter_dt_ema_beta = 0.5
        # Wall-clock budget per training iteration (seconds). Default 50 ms
        # ≈ 20 iters/s headroom at 30 Hz sensor rate.
        self.streaming_online_budget_per_iter_s = 0.05
        # Cap on depth-inserted Gaussians per ingestion event.
        self.streaming_online_insertion_cap = 1500
        # Publish RViz diagnostics every N iterations (overrides viewer_rviz_publish_interval
        # for the scheduler_stats topic only).
        self.streaming_online_rviz_publish_interval = 10
        # Publish /eligsir/scheduler_stats (F2) when viewer is enabled.
        self.streaming_online_publish_stats = True
        # -----------------------------------------------------------------------
        # BurstBudget (Milestone 3 — ingestion cadence + drop policy)
        # -----------------------------------------------------------------------
        # Maximum frames ingested per training iteration. Defends against
        # input bursts that would otherwise starve the optimiser.
        self.streaming_ingestion_max_per_iter = 2
        # Max frames drained from the input queue per training iteration (>=1).
        # Default 2 matches the reader fallback in train_streaming.py.
        self.streaming_max_drain_per_iter = 2
        # Max keyframe history retained by the keyframe tracker (frames).
        # Default 1024 matches the reader fallback in train_streaming.py.
        self.streaming_keyframe_max_history = 1024
        # Drop non-keyframes when the controller backlog reaches this many
        # frames. Keyframes are always protected.
        self.streaming_drop_nonkf_backlog = 4
        # Also drop non-keyframes in CATCH_UP mode (default True).
        # NOTE (#601): when --streaming_backpressure is on, this drop
        # policy is neutered (BurstBudget becomes a signal provider and
        # its max_per_iter cadence cap is all that stays in force) so
        # backpressure has exactly one execution owner.
        self.streaming_drop_nonkf_in_catch_up = True
        # -----------------------------------------------------------------------
        # Bounded backpressure owner (issue #601, design parent #599)
        # -----------------------------------------------------------------------
        # Master switch. Default OFF => the owner is never constructed and
        # the drain loop is byte-identical to today. When ON, the owner is
        # the SINGLE execution authority for skip/coalesce decisions: it
        # bounds pending mapping work by skipping/coalescing redundant
        # observations BEFORE expensive RGB/depth materialization, instead
        # of deferring an ever-staler queue to preserve raw-frame coverage.
        #
        # COVERAGE INTERACTION: a skipped observation still advances
        # pose/source/history metadata (observe=True) but does NOT become
        # expensive mapping work. Coverage accounting therefore counts
        # OBSERVED source frames; report.json["backpressure"]["n_admitted"]
        # is the expensive-mapping subset, not the coverage numerator. A
        # startup warning fires when this is combined with
        # --streaming_enforce_full_coverage.
        self.streaming_backpressure = False
        # #600's explicitly-named stress mode: bypass the owner entirely
        # (historical hard-all piling behaviour) for intentional
        # assimilation-overload experiments. Raises a loud stress flag in
        # report.json. Never a production setting.
        self.streaming_backpressure_force_all_stress = False
        # "causal" (default, live-compatible: decide each observation at
        # the moment it is observed, representatives chosen by
        # accumulated-motion threshold crossing) or "lookahead" (buffered:
        # argmax representative over a pending segment; needs a source that
        # can materialize a frame after its cursor has advanced).
        self.streaming_backpressure_resolution_mode = "causal"
        # HARD BOUNDS (all echoed into report.json["backpressure"]["config"]).
        self.streaming_backpressure_pending_max_count = 8
        # Canonical control epochs, independent of optimizer update rate.
        self.streaming_backpressure_pending_max_age_control_ticks = 600
        # Maximum EXPENSIVE admissions per canonical control-tick window.
        self.streaming_backpressure_max_admissions_per_window = 4
        self.streaming_backpressure_admission_window_control_ticks = 100
        # Protected frontier/keyframe quotas.
        self.streaming_backpressure_frontier_quota = 1
        self.streaming_backpressure_keyframe_quota = 1
        self.streaming_backpressure_representatives = 1
        # Cheap explicit protection thresholds (no learned/opaque score).
        self.streaming_backpressure_novelty_protect = 0.85
        self.streaming_backpressure_quality_protect = 0.95
        # Below these accumulated deltas an observation is "clearly
        # redundant" and is coalesced into the last representative.
        self.streaming_backpressure_redundant_translation_m = 0.05
        self.streaming_backpressure_redundant_rotation_deg = 5.0
        # At or above these accumulated deltas a new representative is
        # admitted (causal mode's threshold-crossing selection rule).
        self.streaming_backpressure_admit_translation_m = 0.10
        self.streaming_backpressure_admit_rotation_deg = 10.0
        # How many recent frame_id -> decision entries the owner remembers,
        # so a defer/retry path can never resurrect a skipped frame. Bounded
        # so nothing in the owner scales with source length; a defer path
        # only ever re-offers a recent frame.
        self.streaming_backpressure_decision_ledger_size = 4096
        # Idle-time threshold (seconds) to flag a camera pause. When the
        # most recent input arrival is older than this, the controller
        # emits pause=True. 0 disables the detector.
        self.streaming_pause_idle_threshold_s = 0.5
        # Part D3 (restored 2026-07-26, see docs/architecture/2026-06-30-lost-on-
        # merge-inventory.md; original commit 41082ac): velocity-adaptive
        # refinement (online-controller only). Stalled camera -> more steps;
        # fast camera -> fewer. Off by default.
        self.streaming_velocity_adaptive_steps = False
        self.streaming_velocity_stall_mps = 0.02
        self.streaming_velocity_fast_mps = 0.30
        self.streaming_velocity_step_gain = 1.5
        # -----------------------------------------------------------------------
        # PoseCorrector (Milestone 5 — loop-closure propagation)
        # -----------------------------------------------------------------------
        # Correction mode: cheap | medium | expensive
        # cheap    — update train_cameras c2w and submap transforms only.
        # medium   — also transform Gaussians via apply_submap_correction.
        # expensive — also reproject local depth (placeholder; future work).
        self.streaming_correction_mode = "medium"
        # Cap on submaps corrected per event (0 = unlimited).
        self.streaming_correction_max_submaps_per_event = 0
        # Minimum wall-clock seconds between corrections (rate limiter).
        self.streaming_correction_min_dt_s = 0.0
        # When True, --pose_source ros_slam implies --loop_closure_apply_correction
        # (online maps need to follow frontend corrections in real time).
        # Disable for offline / debug runs.
        self.streaming_correction_default_on_for_live = True
        # -----------------------------------------------------------------------
        # W3-17: --densification_strategy gsplat_live_strategy activation phases
        # (Wave 3, W3-10's ActivationPhaseConfig,
        # eligsir/optimization/gaussian/strategies/live/backend.py). Only
        # meaningful with --densification_strategy gsplat_live_strategy; ignored
        # by every other strategy. Every flag defaults OFF (plan section
        # 2.4.2's opt-in convention) -- with all five off, gsplat_live_strategy
        # is behavior-identical to gsplat_energy_mcmc (see
        # utils/strategies/mcmc/live_wrapped.py's module docstring). Phases
        # beyond A are wired to backend.py's AlignedBufferRegistry but are NOT
        # verified against a real training run by this PR (task #22, gated on
        # this PR merging first) -- see backend.py's own topology_version
        # integration caveat before enabling phase_c_reseeding.
        self.streaming_live_strategy_phase_a_lifecycle = False
        self.streaming_live_strategy_phase_b_retirement = False
        self.streaming_live_strategy_phase_c_reseeding = False
        self.streaming_live_strategy_phase_d_cell_budgets = False
        self.streaming_live_strategy_phase_e_constrained_movement = False
        super().__init__(parser, "Streaming Parameters")


class OutputStagingParams(ParamGroup):
    def __init__(self, parser):
        # Local-disk staging for a network-filesystem -m path -- root cause:
        # NFS/CIFS/etc RPC latency on hot writes (train.log, checkpoints)
        # collapses training from 60-98 it/s to 5-6 it/s. See
        # eligsir/runtime/output_staging.py (OutputStagingManager,
        # PersistenceWorker). "auto" detects a network filesystem under -m and
        # redirects hot writes to local disk, mirroring back in the
        # background; "on"/"off" force the behavior regardless of detection.
        self.output_staging = "auto"  # {"auto", "on", "off"}
        self.output_staging_root = ""  # "" = let resolve_staging_root() pick
        self.output_sync_interval = 300.0
        self.output_log_sync_interval = 2.0
        self.output_keep_local = False
        super().__init__(parser, "Output Staging")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
