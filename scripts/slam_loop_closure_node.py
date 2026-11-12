#!/usr/bin/env python3
"""Shared loop-closure backend for the odometry-only EliGSiR SLAM frontends.

The gradSLAM and Open3D-odometry frontends are pure visual odometry (frame-to-
frame / frame-to-model) with **no loop closure**, so global drift accumulates
(gradSLAM odometry: ~0.85 m ATE over the full workshop1_slam bag). ORB-SLAM3 and
RTAB-Map already have their own loop closing + pose-graph backends; this node adds
the same capability to the two that lack it, *without* touching the frontends.

It is frontend-agnostic: it subscribes the frontend's `/slam/pose` plus the
compressed RGB-D, maintains a keyframe pose graph (odometry edges between
consecutive keyframes + appearance-based loop edges), runs Open3D's
`global_optimization` (Levenberg-Marquardt) whenever a loop is accepted, and
republishes the drift-corrected trajectory on `/slam/path` and the latest
corrected pose on `/slam/pose_corrected`.

CPU-only (ORB + Open3D pose graph) — no GPU contention with 3DGS training.

Loop detection: ORB descriptors (cv2) matched against past keyframes (BFMatcher +
Lowe ratio), geometrically verified by backprojecting matched keypoints to 3D via
depth and running Open3D correspondence-RANSAC; a loop edge is accepted only with
enough inliers (guards against perceptual aliasing).

Pose-graph convention (matches Open3D's multiway-registration tutorial):
  node.pose = T_world_cam (Twc, as published on /slam/pose)
  edge(source, target).transformation = inv(pose_target) @ pose_source
                                       = T_target_source  (aligns source -> target)

Params (--ros-args -p ...):
  pose_topic (/slam/pose), color_topic, depth_topic, camera_info_topic,
  world_frame (map), depth_scale (1000.0),
  kf_trans (0.10 m), kf_rot_deg (10) — keyframe spacing,
  loop_min_gap (20) — skip this many recent keyframes when searching,
  orb_features (1000), ratio (0.75), min_matches (40), min_inliers (25),
  ransac_dist (0.05 m), icp_refine (1), downscale (2),
  loop_min_fitness (0.05) — min RANSAC fitness to accept a loop,
  loop_max_resid (10.0 m) — max |loop_trans| in world space (room-size sanity).
"""
import os
import queue
import threading
from collections import deque
import numpy as np
import cv2
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from message_filters import Subscriber, ApproximateTimeSynchronizer

_reg = o3d.pipelines.registration

# This is a CPU SIDE-process meant to leave the GPU frontend's CPU-side work (DDS
# delivery, image decode, the gradslam orchestration) un-starved. By default
# OpenCV (ORB/BFMatcher), Open3D (RANSAC/ICP/global_optimization LM) and OpenBLAS
# each open a thread pool sized to the CORE COUNT (32 here) -> ~150 threads that
# saturate every core and steal CPU from the frontend. Cap them. (OMP/OpenBLAS
# pools form at import, so the run scripts also export *_NUM_THREADS; these
# runtime calls bound the two big consumers regardless.)
_LC_THREADS = int(os.environ.get("LC_NUM_THREADS", "4"))
cv2.setNumThreads(_LC_THREADS)
try:
    o3d.utility.set_global_num_threads(_LC_THREADS)
except Exception:
    pass


def _quat_to_R(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w) + 1e-12
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return x, y, z, w


class Keyframe:
    __slots__ = ("Twc", "Twc_odom", "kp_xyz", "desc", "cloud", "stamp")

    def __init__(self, Twc, kp_xyz, desc, cloud, stamp):
        self.Twc = Twc            # 4x4 cam->world; OVERWRITTEN by pose-graph optimize
        self.Twc_odom = Twc.copy()  # raw frontend odometry pose, never optimized --
                                    # used by the loop sanity gate (optimized Twc drifts)
        self.kp_xyz = kp_xyz      # (M,3) ORB keypoints backprojected, cam frame
        self.desc = desc          # (M,32) ORB descriptors
        self.cloud = cloud        # o3d PointCloud (downsampled), cam frame
        self.stamp = stamp


class LoopClosureNode(Node):
    def __init__(self):
        super().__init__("slam_loop_closure_node")
        gs = lambda n, d: self.declare_parameter(n, d).get_parameter_value().string_value
        gi = lambda n, d: self.declare_parameter(n, d).get_parameter_value().integer_value
        gf = lambda n, d: self.declare_parameter(n, d).get_parameter_value().double_value
        self.pose_topic = gs("pose_topic", "/slam/pose")
        self.color_topic = gs("color_topic", "/camera/color/image_raw/compressed")
        self.depth_topic = gs("depth_topic", "/camera/depth/image_raw/compressed")
        self.info_topic = gs("camera_info_topic", "/camera/color/camera_info")
        self.world_frame = gs("world_frame", "map")
        self.depth_scale = gf("depth_scale", 1000.0)
        # compressed=1: Orbbec CompressedImage; compressed=0: raw sensor_msgs/Image
        # (TUM: rgb8 + 32FC1-metres depth -> pass depth_scale:=1.0).
        self.compressed = bool(gi("compressed", 1))
        self.kf_trans = gf("kf_trans", 0.10)
        self.kf_rot = np.deg2rad(gf("kf_rot_deg", 10.0))
        self.loop_min_gap = gi("loop_min_gap", 20)
        self.loop_max_candidates = gi("loop_max_candidates", 60)  # bound O(KF^2)
        self.opt_every = gi("opt_every", 8)        # batch pose-graph optimization
        self.publish_every = gi("publish_every", 10)  # full /slam/path cadence
        self._pending_opt = 0
        self.ratio = gf("ratio", 0.75)
        self.min_matches = gi("min_matches", 40)
        self.min_inliers = gi("min_inliers", 25)
        self.ransac_dist = gf("ransac_dist", 0.05)
        self.icp_refine = gi("icp_refine", 1)
        # Pose-graph weighting. Loop edges were eye(6)*fitness (fit~0.1-0.5), i.e.
        # WEAKER than the eye(6) odometry backbone, so the LM optimizer (with the
        # line process) left them as no-ops. Weight loops by their actual evidence
        # (fitness * scale, well above the odometry stiffness) and prefer loop
        # closures in the global optimization so they can bend the odometry chain.
        self.odom_info = gf("odom_info", 1.0)
        self.loop_info_scale = gf("loop_info_scale", 30.0)
        self.pref_loop_closure = gf("pref_loop_closure", 2.0)
        self.edge_prune = gf("edge_prune", 0.25)
        # Verified loops are HARD constraints by default (see _add_keyframe);
        # set loop_uncertain=1 to re-enable Open3D's line-process rejection.
        self.loop_uncertain = bool(gi("loop_uncertain", 0))
        # Loop sanity gates.
        # (1) RANSAC fitness: fraction of source points with a valid correspondence.
        # A degenerate match (texture-less / planar region) gives fit≈0 even with
        # many inliers; a real loop on a structured scene gives fit≥0.05.
        self.loop_min_fitness = gf("loop_min_fitness", 0.05)
        # (2) Absolute-translation cap: the RANSAC loop transform says the two KFs
        # are loop_max_resid apart in world space.  Anything larger than a room (~10m)
        # is a grossly wrong RANSAC result and should be rejected regardless of inlier
        # count.  NOTE: do NOT compare to the odometry-relative — the whole point of
        # loop closure is to correct accumulated drift, so disagreement with odometry
        # is expected and desired.
        self.loop_max_resid = gf("loop_max_resid", 10.0)
        # Temporal/group consistency (ORB-SLAM3 / pyslam LoopGroupConsistencyChecker):
        # a candidate must recur across `loop_consistency` consecutive keyframes
        # (matching a past KF within `loop_consistency_window` of the last match)
        # before it's trusted -- rejects one-off spurious appearance matches that the
        # geometry gate alone can't (a wrong place can still yield a low residual).
        self.loop_consistency = gi("loop_consistency", 3)
        self.loop_consistency_window = gi("loop_consistency_window", 10)
        # delta-err guard: roll back an optimize that moves any node further than this
        # (m) — catches true blow-ups (inconsistent false loops → nodes fly to
        # infinity) while allowing the large initial correction step the LM optimizer
        # needs to reconcile accumulated drift.  5 m is above any legitimate room
        # traversal correction but well below a true blow-up.
        self.opt_move_cap = gf("opt_move_cap", 5.0)
        self._consistency = 0
        self._last_cand = None
        self.downscale = gi("downscale", 2)
        self.orb = cv2.ORB_create(gi("orb_features", 1000))
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.K = None
        self.kfs = []                 # list[Keyframe]
        self.pg = _reg.PoseGraph()
        self.n_loops = 0
        self.n_rejected = 0       # loops killed by the sanity gate
        self.n_seen = 0

        # /slam/pose lags the images (frontend worker thread), so a 3-way sync of
        # pose+colour+depth mostly misses. Instead: sync colour+depth into a
        # rolling stamp-indexed buffer, and pair each /slam/pose with the nearest
        # buffered image by timestamp (the frontend stamps the pose with the
        # colour stamp, so the match is exact up to buffering delay).
        self._imgbuf = deque(maxlen=400)
        self._buflock = threading.Lock()

        self.path_pub = self.create_publisher(Path, "/slam/path", 10)
        self.corr_pub = self.create_publisher(PoseStamped, "/slam/pose_corrected", 10)
        self.create_subscription(CameraInfo, self.info_topic, self._info, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, self.pose_topic, self._pose_cb, qos_profile_sensor_data)
        ImgT = CompressedImage if self.compressed else Image
        cs = Subscriber(self, ImgT, self.color_topic, qos_profile=qos_profile_sensor_data)
        ds = Subscriber(self, ImgT, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self._ats = ApproximateTimeSynchronizer([cs, ds], queue_size=200, slop=0.15)
        self._ats.registerCallback(self._img_cb)
        # All heavy work (ORB, O(KF) loop search, pose-graph optimization) runs in
        # this worker, NOT the sync callback — otherwise the executor is blocked
        # and DDS starves (the node would fall behind and effectively hang).
        self._q = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self.get_logger().info(
            f"loop_closure: kf_trans={self.kf_trans} kf_rot={np.rad2deg(self.kf_rot):.0f}deg "
            f"min_matches={self.min_matches} min_inliers={self.min_inliers} "
            f"loop_min_gap={self.loop_min_gap}")

    def _info(self, msg: CameraInfo):
        if self.K is None:
            s = 1.0 / self.downscale
            k = msg.k
            self.fx, self.fy = k[0] * s, k[4] * s
            self.cx, self.cy = k[2] * s, k[5] * s
            self.K = True
            self.get_logger().info(f"intrinsics fx={self.fx:.1f} downscale={self.downscale}")

    def _backproject(self, color_gray, depth_m):
        """ORB keypoints + descriptors with their backprojected 3D (cam frame),
        and a downsampled point cloud of the whole frame (cam frame)."""
        kps, desc = self.orb.detectAndCompute(color_gray, None)
        kp_xyz, keep = [], []
        H, W = depth_m.shape
        if desc is not None:
            for i, kp in enumerate(kps):
                u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
                if 0 <= u < W and 0 <= v < H:
                    z = depth_m[v, u]
                    if 0.1 < z < 8.0:
                        kp_xyz.append([(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z])
                        keep.append(i)
        kp_xyz = np.asarray(kp_xyz, np.float64).reshape(-1, 3)
        desc = desc[keep] if (desc is not None and keep) else np.empty((0, 32), np.uint8)
        # full-frame cloud (downsampled) for ICP refine
        vv, uu = np.where((depth_m > 0.1) & (depth_m < 8.0))
        zz = depth_m[vv, uu]
        pts = np.stack([(uu - self.cx) / self.fx * zz, (vv - self.cy) / self.fy * zz, zz], 1)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
        cloud = cloud.voxel_down_sample(0.03)
        return kp_xyz, desc, cloud

    def _img_cb(self, cmsg, dmsg):
        """Buffer synced colour+depth by timestamp for pairing with poses. Keep the
        full messages (raw Image needs encoding+dims to decode); decode is deferred
        to the worker so this callback stays light."""
        t = cmsg.header.stamp.sec + cmsg.header.stamp.nanosec * 1e-9
        with self._buflock:
            self._imgbuf.append((t, cmsg, dmsg))

    def _pose_cb(self, pmsg: PoseStamped):
        """Pose-driven keyframe gate; pair with the nearest buffered image."""
        if self.K is None:
            return
        p = pmsg.pose
        R = _quat_to_R(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
        Twc = np.eye(4); Twc[:3, :3] = R
        Twc[:3, 3] = [p.position.x, p.position.y, p.position.z]
        if self.kfs and not self._is_keyframe(Twc):
            return
        pt = pmsg.header.stamp.sec + pmsg.header.stamp.nanosec * 1e-9
        with self._buflock:
            if not self._imgbuf:
                return
            best = min(self._imgbuf, key=lambda e: abs(e[0] - pt))
        if abs(best[0] - pt) > 0.05:        # no image within 50 ms
            return
        self._q.put((Twc, best[1], best[2], pmsg.header.stamp))

    def _decode(self, cmsg, dmsg):
        """Decode buffered colour+depth messages to (gray uint8, depth raw). Handles
        Orbbec CompressedImage (jpeg + png-16UC1) and raw Image (rgb8/bgr8 +
        32FC1-metres/16UC1-mm)."""
        if self.compressed:
            color = cv2.imdecode(np.frombuffer(cmsg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
            # compressedDepth transport (issue #303) prepends a ConfigHeader
            # ahead of the PNG, so a bare imdecode returns None here too --
            # same defect class as the live source's, same shared decoder.
            from eligsir.io.sources.ros2_live import (
                _decode_compressed_depth_image,
                _is_compressed_depth_format,
            )
            if _is_compressed_depth_format(getattr(dmsg, "format", "")):
                depth, _enc = _decode_compressed_depth_image(dmsg)
            else:
                depth = cv2.imdecode(np.frombuffer(dmsg.data, np.uint8), cv2.IMREAD_UNCHANGED)
            return color, depth
        c = np.frombuffer(cmsg.data, np.uint8).reshape(cmsg.height, cmsg.width, -1)[:, :, :3]
        color = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY if "rgb" in cmsg.encoding.lower()
                             else cv2.COLOR_BGR2GRAY)
        de = dmsg.encoding.lower()
        if "32f" in de:
            depth = np.frombuffer(dmsg.data, np.float32).reshape(dmsg.height, dmsg.width)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            depth = np.frombuffer(dmsg.data, np.uint16).reshape(dmsg.height, dmsg.width)
        return np.ascontiguousarray(color), np.ascontiguousarray(depth)

    def _worker_loop(self):
        while True:
            Twc, cmsg, dmsg, stamp = self._q.get()
            color, depth = self._decode(cmsg, dmsg)
            if color is None or depth is None:
                continue
            if self.downscale > 1:
                w, h = color.shape[1] // self.downscale, color.shape[0] // self.downscale
                color = cv2.resize(color, (w, h), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            depth_m = depth.astype(np.float32) / self.depth_scale
            kp_xyz, desc, cloud = self._backproject(color, depth_m)
            self.n_seen += 1
            self._add_keyframe(Twc, kp_xyz, desc, cloud, stamp)
            if self.n_seen % 20 == 0:
                self.get_logger().info(
                    f"keyframes={len(self.kfs)} loops={self.n_loops} backlog={self._q.qsize()}")

    def _is_keyframe(self, Twc):
        last = self.kfs[-1].Twc
        dt = np.linalg.norm(Twc[:3, 3] - last[:3, 3])
        dR = last[:3, :3].T @ Twc[:3, :3]
        ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        return dt > self.kf_trans or ang > self.kf_rot

    def _add_keyframe(self, Twc, kp_xyz, desc, cloud, stamp):
        i = len(self.kfs)
        self.kfs.append(Keyframe(Twc, kp_xyz, desc, cloud, stamp))
        self.pg.nodes.append(_reg.PoseGraphNode(Twc.copy()))
        if i > 0:                                              # odometry edge
            X = np.linalg.inv(Twc) @ self.kfs[i - 1].Twc       # T_target_source
            self.pg.edges.append(_reg.PoseGraphEdge(
                i - 1, i, X, np.eye(6) * self.odom_info, uncertain=False))
        loop = self._detect_loop(i)
        accepted = False
        if loop is not None:
            j, X, info, ninl, fit = loop
            # (A) RANSAC fitness gate: reject matches where RANSAC found fewer than
            # loop_min_fitness of source points within ransac_dist.  Degenerate /
            # planar scenes can produce high inlier counts but near-zero fitness,
            # indicating the correspondence set is too small to constrain the
            # transform.  Real loops on structured scenes give fit > 0.05.
            loop_trans = float(np.linalg.norm(X[:3, 3]))
            if fit < self.loop_min_fitness:
                self.n_rejected += 1
                self._consistency = 0; self._last_cand = None
                self.get_logger().info(
                    f"LOOP {j}<->{i} REJECTED fit={fit:.3f} < {self.loop_min_fitness} "
                    f"(inliers={ninl}) [rejected {self.n_rejected}]")
            # (B) Absolute-translation sanity: a loop that claims two KFs are
            # further than loop_max_resid apart in world space is geometrically
            # implausible (larger than the room).  This is NOT a drift check —
            # drift residuals disagree with odometry BY DESIGN and must not be
            # rejected here.
            elif loop_trans > self.loop_max_resid:
                self.n_rejected += 1
                self._consistency = 0; self._last_cand = None
                self.get_logger().info(
                    f"LOOP {j}<->{i} REJECTED |trans|={loop_trans:.1f}m > "
                    f"{self.loop_max_resid}m (inliers={ninl} fit={fit:.2f}) "
                    f"[rejected {self.n_rejected}]")
            else:
                # (C) TEMPORAL CONSISTENCY (pyslam/ORB-SLAM3): only trust the loop once
                # the same past region matches over several consecutive keyframes.
                if (self._last_cand is not None
                        and abs(j - self._last_cand) <= self.loop_consistency_window):
                    self._consistency += 1
                else:
                    self._consistency = 1
                self._last_cand = j
                if self._consistency < self.loop_consistency:
                    self.get_logger().info(
                        f"LOOP {j}<->{i} HELD consistency {self._consistency}/"
                        f"{self.loop_consistency} (|trans|={loop_trans:.2f}m)")
                else:
                    self.pg.edges.append(_reg.PoseGraphEdge(  # hard constraint
                        j, i, X, info, uncertain=self.loop_uncertain))
                    self.n_loops += 1
                    self._pending_opt += 1
                    accepted = True
                    self.get_logger().info(
                        f"LOOP {j}<->{i} accepted inliers={ninl} fit={fit:.2f} "
                        f"|trans|={loop_trans:.2f}m (total {self.n_loops})")
                    if self._pending_opt >= self.opt_every:
                        self._optimize(); self._pending_opt = 0
        if loop is None and not accepted:          # no candidate this KF -> reset streak
            self._consistency = 0; self._last_cand = None
        self._publish(stamp)

    def finalize(self):
        """Final optimization + publish (call on shutdown).
        Always runs a final optimization regardless of pending_opt — intermediate
        optimizations may have rolled back (cap too tight for the initial step), and
        we want the best solution using all accumulated constraints at shutdown."""
        if self.kfs and self.n_loops > 0:
            self._optimize()
            self._pending_opt = 0
        self._publish(self.kfs[-1].stamp, force=True) if self.kfs else None

    def _detect_loop(self, i):
        cur = self.kfs[i]
        if cur.desc.shape[0] < self.min_matches:
            return None
        # Bound the search: cap the number of past keyframes compared per new KF
        # (evenly subsampled) so total work stays ~O(KF * cap), not O(KF^2).
        hi = i - self.loop_min_gap
        if hi <= 0:
            return None
        cand = range(0, hi)
        if hi > self.loop_max_candidates:
            cand = np.linspace(0, hi - 1, self.loop_max_candidates).astype(int)
        best = None
        for j in cand:
            j = int(j)
            ref = self.kfs[j]
            if ref.desc.shape[0] < self.min_matches:
                continue
            matches = self.bf.knnMatch(cur.desc, ref.desc, k=2)
            good = [(m.queryIdx, m.trainIdx) for m, n in matches
                    if m.distance < self.ratio * n.distance]
            if len(good) < self.min_matches:
                continue
            X, ninl, fit = self._loop_transform(ref, cur, good)   # source=ref(j), target=cur(i)
            if ninl >= self.min_inliers and (best is None or ninl > best[3]):
                best = (j, X, np.eye(6) * fit * self.loop_info_scale, ninl, fit)
        if best is None:
            return None
        return best[0], best[1], best[2], best[3], best[4]

    def _loop_transform(self, ref, cur, good):
        """RANSAC correspondence registration aligning ref->cur (source=ref,
        target=cur) so the edge transformation = T_cur_ref. Optionally ICP-refine
        on the dense clouds. Returns (X, n_inliers, fitness)."""
        ref_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref.kp_xyz))
        cur_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cur.kp_xyz))
        corr = np.array([[c, q] for q, c in good], dtype=np.int32)  # [ref_idx(src), cur_idx(tgt)]
        corr = o3d.utility.Vector2iVector(corr)
        res = _reg.registration_ransac_based_on_correspondence(
            ref_pc, cur_pc, corr, self.ransac_dist,
            _reg.TransformationEstimationPointToPoint(False), 3,
            [_reg.CorrespondenceCheckerBasedOnDistance(self.ransac_dist)],
            _reg.RANSACConvergenceCriteria(100000, 0.999))
        X = res.transformation.copy()
        ninl = int(len(res.correspondence_set))
        fit = res.fitness
        if self.icp_refine and ninl >= self.min_inliers:
            icp = _reg.registration_icp(
                ref.cloud, cur.cloud, self.ransac_dist, X,
                _reg.TransformationEstimationPointToPoint())
            if icp.fitness > 0:
                X = icp.transformation.copy(); fit = icp.fitness
        return X, ninl, max(fit, 1e-3)

    def _optimize(self):
        opt = _reg.GlobalOptimizationLevenbergMarquardt()
        crit = _reg.GlobalOptimizationConvergenceCriteria()
        option = _reg.GlobalOptimizationOption(
            max_correspondence_distance=self.ransac_dist,
            edge_prune_threshold=self.edge_prune,
            preference_loop_closure=self.pref_loop_closure,
            reference_node=0)
        before_full = [np.asarray(n.pose).copy() for n in self.pg.nodes]
        before = np.stack([p[:3, 3] for p in before_full])
        _reg.global_optimization(self.pg, opt, crit, option)
        after = np.stack([np.asarray(n.pose)[:3, 3] for n in self.pg.nodes])
        moved = float(np.linalg.norm(after - before, axis=1).max() * 1000)
        # delta-err guard: a single optimize that flings a node further than the cap
        # is a blow-up (inconsistent edges) -> roll the whole graph back, drop the
        # offending pending edges' effect by leaving Twc untouched.
        if moved > self.opt_move_cap * 1000:
            for k, p in enumerate(before_full):
                self.pg.nodes[k].pose = p
            self.get_logger().warn(
                f"OPTIMIZE ROLLED BACK: max node moved {moved:.0f}mm > "
                f"{self.opt_move_cap*1000:.0f}mm cap ({len(self.pg.edges)} edges)")
            return
        self.get_logger().info(
            f"OPTIMIZE {len(self.pg.nodes)} nodes, {len(self.pg.edges)} edges, "
            f"{self.n_loops} loops -> max node moved {moved:.0f}mm")
        for k, node in enumerate(self.pg.nodes):
            self.kfs[k].Twc = np.asarray(node.pose).copy()

    def _publish(self, stamp, force=False):
        # Always publish the cheap latest corrected pose; publish the FULL
        # /slam/path only periodically (and at finalize) — republishing the whole
        # trajectory every keyframe bloats the recorded bag and makes eval crawl.
        if self.kfs:
            kf = self.kfs[-1]
            last = PoseStamped()
            last.header.frame_id = self.world_frame
            last.header.stamp = stamp
            last.pose.position.x, last.pose.position.y, last.pose.position.z = kf.Twc[:3, 3]
            qx, qy, qz, qw = _R_to_quat(kf.Twc[:3, :3])
            last.pose.orientation.x, last.pose.orientation.y = qx, qy
            last.pose.orientation.z, last.pose.orientation.w = qz, qw
            self.corr_pub.publish(last)
        if not (force or len(self.kfs) % self.publish_every == 0):
            return
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self.world_frame
        for kf in self.kfs:
            ps = PoseStamped()
            ps.header.frame_id = self.world_frame
            ps.header.stamp = kf.stamp          # KF color stamp — for eval assoc
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = kf.Twc[:3, 3]
            qx, qy, qz, qw = _R_to_quat(kf.Twc[:3, :3])
            ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
            ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
            path.poses.append(ps)
        self.path_pub.publish(path)


def main():
    rclpy.init()
    node = LoopClosureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.finalize()
        except Exception:
            pass
        node.get_logger().info(
            f"loop_closure done: {len(node.kfs)} keyframes, {node.n_loops} loops")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
