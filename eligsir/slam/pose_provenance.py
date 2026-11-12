"""Pose provenance/confidence tiering.

Pose quality is a large, currently-invisible confound on every quality claim
(PSNR, geometry metrics) made about a run: a run trained on hand-wavy live
SLAM poses is not comparable to one trained on dataset ground truth, even if
both produce the same PSNR number. This module tags every run with a single
provenance tier so that distinction is visible everywhere a run's quality is
reported (banner, report.json, TensorBoard, plots) instead of silently
assumed.

Precedence (issue #594)
-----------------------
A run once reported ``pose_source = dataset_pose`` (the authoritative resolved
value, from ``eligsir/slam/pose_sources.py``) alongside
``tier=slam_estimated method=auto_detected_topic (topic=/camera_pose)``. Both
were produced from the same namespace: ``--orbbec_pose_topic`` defaults to
``/camera_pose`` (``arguments/__init__.py:122``) and is read by exactly ONE
call site, ``eligsir/io/sources/registry.py:310``, which is reached only
for a rosbag2/MCAP source. For a TUM/ScanNet/Replica *directory* the flag is
inert -- yet the classifier ranked it above the dataset-layout evidence and
relabelled a ground-truth-pose run as SLAM-estimated.

The resolution order is therefore:

1. ``--pose_provenance`` explicit CLI override -- always wins.
2. ``orbbec_pose_source`` odometry selection (a real, in-process pose
   *estimator*, not a naming heuristic).
3. Pose-topic detection -- **admissible only when the topic is really the
   channel this run's poses arrive on** (see :func:`pose_topic_is_live_channel`).
   A topic name alone can never outrank an authoritative resolved pose source.
4. Dataset-directory-layout detection.
5. :class:`PoseProvenance.UNKNOWN` -- never a silent trusted default.

The tier vocabulary is deliberately unchanged: #594 asks to distinguish
prerecorded-external from live-SLAM estimates, but ``PoseProvenance`` feeds
``POSE_PROVENANCE_CHOICES`` (the CLI ``choices=``) and every persisted report,
so that distinction is recorded as :attr:`ProvenanceResult.provider` /
:attr:`ProvenanceResult.live` instead of as a new tier.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class PoseProvenance(str, Enum):
    """Trust tier for the camera poses driving a run, highest trust first."""

    DATASET_GT = "dataset_gt"
    COLMAP_PSEUDO_GT = "colmap_pseudo_gt"
    SLAM_ESTIMATED = "slam_estimated"
    UNKNOWN = "unknown"


class ProvenanceMethod(str, Enum):
    """How a run's :class:`PoseProvenance` tier was determined."""

    AUTO_DETECTED_TOPIC = "auto_detected_topic"
    AUTO_DETECTED_DATASET_FORMAT = "auto_detected_dataset_format"
    EXPLICIT_CLI_OVERRIDE = "explicit_cli_override"
    UNKNOWN = "unknown"


POSE_PROVENANCE_CHOICES = tuple(t.value for t in PoseProvenance) + ("auto",)

# Bag topic name -> tier. `/camera_pose` is the ORB-SLAM3-style live/bag
# frontend topic (see arguments/__init__.py orbbec_pose_topic default);
# `/gt/colmap_pose` is written by the sibling rosbag-colmap-pipeline (gttool)
# repo's COLMAP/GLOMAP + depth-metric-scale pseudo-GT export.
_TOPIC_PROVENANCE = {
    "/camera_pose": PoseProvenance.SLAM_ESTIMATED,
    "/slam/pose": PoseProvenance.SLAM_ESTIMATED,
    "/gt/colmap_pose": PoseProvenance.COLMAP_PSEUDO_GT,
}


# Pose-source names (``eligsir/slam/pose_sources.py``) that mean "the frame
# source itself supplies the pose" -- i.e. no in-process SLAM estimator runs.
DATASET_POSE_SOURCES = frozenset({"dataset_pose", "gt_pose", "none"})


def pose_topic_is_live_channel(
    source_path: str, *, explicit: Optional[bool] = None
) -> bool:
    """Is a pose *topic* name real evidence about this run's poses?

    ``explicit`` short-circuits for a caller that already knows (e.g. the
    ros2:// live path). Otherwise the answer follows the ONE place a pose
    topic is actually consumed for a file-backed source:
    ``eligsir/io/sources/registry.py`` selects a rosbag source when the
    path holds a ``metadata.yaml`` (rosbag2) or is a single ``.mcap``/``.bag``
    file, and only that source reads ``--orbbec_pose_topic``. Anywhere else the
    flag is inert and its default value (``/camera_pose``) is not evidence of
    anything -- which is exactly the #594 mislabelling.

    An EMPTY ``source_path`` answers True: with no source to inspect there is
    no evidence the topic is inert either, and a caller that hands in a topic
    and nothing else is asserting it matters. The gate deliberately only
    demotes topic evidence where the source is positively known NOT to consume
    it.
    """

    if explicit is not None:
        return bool(explicit)
    path = str(source_path or "")
    if not path:
        return True
    if path.startswith("ros2://"):
        return True
    lowered = path.lower()
    if os.path.isfile(path) and (lowered.endswith(".mcap") or lowered.endswith(".bag")):
        return True
    return os.path.exists(os.path.join(path, "metadata.yaml"))


@dataclass(frozen=True)
class ProvenanceResult:
    """Result of pose provenance detection for a run.

    Records the determined trust tier (ground truth, pseudo-GT, SLAM-estimated, unknown),
    the detection method used, and metadata about the pose source.

    Attributes:
        tier: Trust tier for the camera poses (DATASET_GT, COLMAP_PSEUDO_GT,
            SLAM_ESTIMATED, or UNKNOWN).
        method: How the tier was determined (explicit flag, topic auto-detect,
            dataset-format detection, or unknown).
        detail: Human-readable detail string (e.g., specific topic name or
            dataset format found).
        provider: The actual pose PROVIDER source (e.g., "live_ros_topic:/slam/pose",
            "bag_topic:/gt/colmap_pose", "dataset_files").
        live: True when poses were estimated online during this run, False when
            prerecorded, None when not determinable.
        topic_evidence_admissible: Whether pose-topic evidence was valid for
            this run's source type.
    """

    tier: PoseProvenance
    method: ProvenanceMethod
    detail: str = ""
    #: The actual pose PROVIDER, not merely the channel poses arrived through
    #: (#594 required-work bullet 5). e.g. ``"live_ros_topic:/slam/pose"``,
    #: ``"bag_topic:/gt/colmap_pose"``, ``"dataset_files"``,
    #: ``"open3d_odometry_live"``, ``"declared"``, ``""`` when unknown.
    provider: str = ""
    #: True when poses were estimated online during this run, False when they
    #: were read from something prerecorded, None when not determinable. This
    #: is how the prerecorded-external vs live-SLAM distinction is recorded
    #: without adding a tier (see the module docstring).
    live: Optional[bool] = None
    #: Whether pose-topic evidence was admissible at all for this run.
    topic_evidence_admissible: Optional[bool] = None

    def as_dict(self) -> dict:
        """Convert this result to a dictionary for JSON serialization.

        Returns:
            Dictionary with keys: tier, method, detail, provider, live,
            topic_evidence_admissible.
        """
        return {
            "tier": self.tier.value,
            "method": self.method.value,
            "detail": self.detail,
            "provider": self.provider,
            "live": self.live,
            "topic_evidence_admissible": self.topic_evidence_admissible,
        }

    def banner_line(self) -> str:
        """Generate a formatted log banner line for this provenance result.

        Returns:
            A string suitable for logging, starting with [pose-provenance],
            including tier, method, and optional provider/detail info.
            Uses "!!!" for UNKNOWN tier, "---" otherwise.
        """
        marker = "!!!" if self.tier is PoseProvenance.UNKNOWN else "---"
        return (
            f"[pose-provenance] {marker} tier={self.tier.value} "
            f"method={self.method.value}"
            + (f" provider={self.provider}" if self.provider else "")
            + (f" ({self.detail})" if self.detail else "")
        )


def detect_from_bag_topic(
    pose_topic: str | None, *, live_source: bool = False
) -> ProvenanceResult:
    """Detect provenance from a ROS bag/topic pose-source name.

    Callers must gate this on :func:`pose_topic_is_live_channel` -- it answers
    "what would this topic mean", not "is this topic relevant to this run".
    """

    topic = str(pose_topic or "")
    tier = _TOPIC_PROVENANCE.get(topic)
    if tier is not None:
        live = tier is PoseProvenance.SLAM_ESTIMATED and bool(live_source)
        return ProvenanceResult(
            tier,
            ProvenanceMethod.AUTO_DETECTED_TOPIC,
            detail=f"topic={topic}",
            provider=("live_ros_topic:" if live_source else "bag_topic:") + topic,
            live=live,
            topic_evidence_admissible=True,
        )
    return ProvenanceResult(
        PoseProvenance.UNKNOWN,
        ProvenanceMethod.UNKNOWN,
        detail=f"unrecognized pose topic {topic!r}",
        topic_evidence_admissible=True,
    )


def detect_from_dataset_format(source_path: str) -> ProvenanceResult:
    """Detect provenance from a benchmark dataset's own directory layout."""

    path = str(source_path or "")
    if os.path.exists(os.path.join(path, "groundtruth.txt")):
        return ProvenanceResult(
            PoseProvenance.DATASET_GT,
            ProvenanceMethod.AUTO_DETECTED_DATASET_FORMAT,
            detail="TUM groundtruth.txt",
            provider="dataset_files",
            live=False,
        )
    if os.path.exists(os.path.join(path, "traj.txt")) or os.path.exists(os.path.join(path, "mesh.ply")):
        return ProvenanceResult(
            PoseProvenance.DATASET_GT,
            ProvenanceMethod.AUTO_DETECTED_DATASET_FORMAT,
            detail="Replica baked trajectory",
            provider="dataset_files",
            live=False,
        )
    if os.path.isdir(os.path.join(path, "pose")) and os.path.isdir(os.path.join(path, "color")):
        return ProvenanceResult(
            PoseProvenance.DATASET_GT,
            ProvenanceMethod.AUTO_DETECTED_DATASET_FORMAT,
            detail="ScanNet-style pose/ directory",
            provider="dataset_files",
            live=False,
        )
    return ProvenanceResult(
        PoseProvenance.UNKNOWN,
        ProvenanceMethod.UNKNOWN,
        detail=f"unrecognized dataset layout at {path!r}",
    )


def resolve_pose_provenance(
    *,
    cli_override: str | None = None,
    source_path: str = "",
    pose_topic: str | None = None,
    orbbec_pose_source: str | None = None,
    pose_topic_active: bool | None = None,
    resolved_pose_source: str | None = None,
) -> ProvenanceResult:
    """Resolve a single run's pose provenance tier.

    See the module docstring for the full precedence order. The two arguments
    added by #594 are both optional and both default to the historical
    behaviour *only where that behaviour was already correct*:

    ``pose_topic_active``
        Explicitly declare whether ``pose_topic`` is the channel this run's
        poses arrive on. When omitted it is derived from ``source_path`` via
        :func:`pose_topic_is_live_channel`, so a caller that (like
        ``train_streaming.py``) hands in an inert ``--orbbec_pose_topic``
        default for a dataset-directory run no longer gets a topic-name tier.

    ``resolved_pose_source``
        The authoritative resolved pose source
        (``eligsir/slam/pose_sources.normalize_pose_source``).
        Recorded and used for the contradiction check; it never *upgrades* a
        tier on its own, because ``dataset_pose`` only says "the frame source
        supplies the pose", which for a bag still means a SLAM topic.
    """

    if cli_override and cli_override not in ("auto", ""):
        if cli_override not in {t.value for t in PoseProvenance}:
            raise ValueError(
                f"--pose_provenance must be one of {POSE_PROVENANCE_CHOICES} (got {cli_override!r})"
            )
        return ProvenanceResult(
            PoseProvenance(cli_override),
            ProvenanceMethod.EXPLICIT_CLI_OVERRIDE,
            detail="--pose_provenance",
            provider="declared",
        )

    topic_admissible = pose_topic_is_live_channel(
        source_path, explicit=pose_topic_active
    )
    live_source = str(source_path or "").startswith("ros2://")

    if orbbec_pose_source in ("open3d_odometry", "open3d_odometry_live"):
        return ProvenanceResult(
            PoseProvenance.SLAM_ESTIMATED,
            ProvenanceMethod.AUTO_DETECTED_TOPIC,
            detail=f"orbbec_pose_source={orbbec_pose_source}",
            provider=str(orbbec_pose_source),
            live=orbbec_pose_source == "open3d_odometry_live",
            topic_evidence_admissible=topic_admissible,
        )

    if pose_topic and topic_admissible:
        by_topic = detect_from_bag_topic(pose_topic, live_source=live_source)
        if by_topic.tier is not PoseProvenance.UNKNOWN:
            return by_topic

    if source_path:
        by_format = detect_from_dataset_format(source_path)
        if by_format.tier is not PoseProvenance.UNKNOWN:
            return ProvenanceResult(
                by_format.tier,
                by_format.method,
                detail=by_format.detail,
                provider=by_format.provider,
                live=by_format.live,
                topic_evidence_admissible=topic_admissible,
            )

    detail_bits = []
    if pose_topic:
        detail_bits.append(
            f"pose_topic={pose_topic!r}"
            + ("" if topic_admissible else " [INERT for this source, not used as evidence]")
        )
    if source_path:
        detail_bits.append(f"source_path={source_path!r}")
    if resolved_pose_source:
        detail_bits.append(f"pose_source={resolved_pose_source!r}")
    return ProvenanceResult(
        PoseProvenance.UNKNOWN,
        ProvenanceMethod.UNKNOWN,
        detail="could not auto-detect (" + ", ".join(detail_bits) + ")" if detail_bits else "no signal available",
        topic_evidence_admissible=topic_admissible,
    )


def pose_provenance_contradictions(
    *,
    result: ProvenanceResult,
    resolved_pose_source: str | None,
    source_path: str = "",
) -> List[str]:
    """Warnings for a run whose provenance tier fights its resolved pose source.

    Deliberately warnings, not errors: ``validate_execution_semantics``'s
    contract reserves errors for *provably impossible* combinations, and this
    one is reachable.

    A ``ros2://`` source is EXEMPT. ``utils/execution_semantics.py``'s
    ``_external_pose_provider`` docstring already settled that question: the
    live resolver normalizes ``ros_slam`` to ``dataset_pose`` because the live
    frame source owns pose sync, so "a live profile carrying ``pose_source:
    dataset_pose`` alongside ``ros2_live_pose_topic: /slam/odom`` is NOT a
    contradiction". Flagging it would fire on every live run.
    """

    warnings_out: List[str] = []
    if str(source_path or "").startswith("ros2://"):
        return warnings_out
    src = str(resolved_pose_source or "")
    if not src:
        return warnings_out
    if src in DATASET_POSE_SOURCES and result.tier is PoseProvenance.SLAM_ESTIMATED:
        warnings_out.append(
            f"pose_source={src!r} (authoritative, resolved) but pose provenance "
            f"tier={result.tier.value} via method={result.method.value} "
            f"({result.detail}). Dataset/GT poses and SLAM-estimated poses are "
            "different benchmark conditions; one of the two is wrong for this "
            "run. Pass --pose_provenance explicitly to declare the truth."
        )
    if src not in DATASET_POSE_SOURCES and result.tier is PoseProvenance.DATASET_GT:
        warnings_out.append(
            f"pose_source={src!r} runs an in-process pose estimator, but pose "
            f"provenance tier=dataset_gt ({result.detail}). The dataset's baked "
            "ground truth is NOT what drove this run's cameras unless it was "
            "explicitly declared."
        )
    return warnings_out


def warn_provenance_tier_change(
    *,
    previous: PoseProvenance,
    new: PoseProvenance,
    reason: str,
) -> None:
    """Loudly flag a mid-run provenance-tier change (e.g. camera_pose topic
    missing -> falling back to open3d_odometry_live). This is a real
    tier-mixing event, not a detail to swallow: it silently changes the trust
    level of poses feeding the rest of the run.
    """

    if previous == new:
        return
    msg = (
        f"[pose-provenance] !!! TIER CHANGE MID-RUN: {previous.value} -> {new.value} "
        f"({reason}). Downstream quality claims (PSNR, geometry) for this run "
        "are no longer uniformly attributable to a single provenance tier."
    )
    print(msg, flush=True)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def flag_orbbec_auto_pose_fallback(pose_topic: str) -> None:
    """Flag the ``orbbec_pose_source=auto`` mid-run fallback from the
    ``camera_pose`` topic to ``open3d_odometry_live`` when the pose topic is
    missing from the bag. Both resolve to :class:`PoseProvenance.SLAM_ESTIMATED`
    (so it is not a cross-tier jump), but it swaps a real live-SLAM pose
    source for a much weaker frame-to-frame odometry estimator with no
    loop-closure/relocalization -- a confidence change within the tier that
    must not be silently absorbed into a uniform "slam_estimated" label.
    """

    msg = (
        f"[pose-provenance] !!! AUTO POSE-SOURCE FALLBACK: '{pose_topic}' topic missing "
        "from bag; falling back camera_pose -> open3d_odometry_live. Both are "
        f"tier={PoseProvenance.SLAM_ESTIMATED.value}, but open3d_odometry_live has no "
        "loop closure/relocalization and is materially lower-confidence. Pass "
        "--pose_provenance explicitly if this run's poses should not be treated "
        "as uniform slam_estimated confidence."
    )
    print(msg, flush=True)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
