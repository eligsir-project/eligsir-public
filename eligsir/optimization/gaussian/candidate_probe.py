"""Side-effect-free batched candidate probing (issue #803, parent #802).

A *candidate probe* renders a bounded cohort of prefilter-surviving cameras
against exactly ONE fixed Gaussian/topology snapshot and returns per-view
render/loss evidence, **before** any durable keyframe admission decision is
made. It is an evidence API, not an admission policy: nothing in this module
(or in :meth:`GaussianOptimizationEngine.probe_candidates`) ranks, thresholds,
selects, or admits anything, and no combined ``candidate_quality`` scalar is
produced here -- that is issue #500's job, deliberately downstream.

## The invariant this module exists to guarantee

A probe performs real GPU work against live training state, which is exactly
why its non-interference has to be structural rather than aspirational::

    NO optimizer step
    NO mutation / topology change
    NO persistent parameter update
    NO hidden .grad accumulation
    NO replay / tier / admission state mutation

How each is enforced, concretely:

* **No optimizer step / no LR-scheduler advance.** The probe does not call
  ``session.update_learning_rate``, ``optimizer.step``, ``zero_grad``, or any
  ``run_*_step`` path. It calls :func:`invoke_render` and computes an objective;
  that is the whole transaction.
* **No hidden ``.grad``.** Forward-only mode runs under ``torch.no_grad()``.
  Gradient mode (Phase 3) runs under ``torch.enable_grad()`` but reaches the
  gradient through :func:`torch.autograd.grad`, which *returns* gradients and
  never accumulates into ``.grad`` -- ``loss.backward()`` is never called here
  and must never be.
* **No RNG-stream consumption.** The whole probe runs inside
  :meth:`RandomStateToken.scoped`, so any randomness a render path consumes is
  rolled back; the caller's MCMC/noise stream continues exactly where it was.
* **Fail loud, do not hope.** Every probe verifies a state fingerprint before
  and after (see :func:`capture_state_fingerprint`) and raises
  :class:`ProbeStateMutationError` if anything moved. The fingerprint uses
  ``torch.Tensor._version`` and ``data_ptr()``, which detect in-place mutation
  and tensor replacement respectively **without** a host sync, so the check is
  affordable enough to leave permanently on rather than behind a debug flag.
* **Camera-tier restoration.** ``materialize_resolution_tier`` mutates the
  camera object in place (RGB, depth cache, intrinsics, renderer tensor cache).
  Probe cohort cameras are frequently the same objects a later supervision
  render will use, so :func:`probe_resolution_binding` restores the exact
  previous binding -- including deleting the binding markers on a camera that
  had never been bound. This leak is invisible to any Gaussian/optimizer state
  hash, which is why it has its own dedicated test.
* **Not during an optimization transaction.** The engine sets a re-entrancy
  flag around its optimizer transaction; probing inside one raises
  :class:`ProbeUnsafeContextError` rather than silently observing a half-stepped
  map.

## Evidence vocabulary

Per-view scalars are returned in the canonical
:class:`eligsir.mapping.evidence.canonical.FrameEvidence` envelope, whose
:class:`EvidenceIdentity` already carries map/topology/coordinate-frame version,
optimization step, objective version, resolution bucket and schema version --
i.e. issue #803's entire "snapshot/version semantics" requirement, already
typed. Issue #292 (post-admission repeated-exposure evidence) reuses the same
:class:`GaussianProbeEvidence` fields; the difference between the two consumers
is time and history, never metric definitions.

Evidence that is not a scalar (visible-Gaussian masks/IDs, downsampled residual
sketches, per-parameter-family gradient energies) cannot live inside a
``ScalarEvidence`` envelope and is carried on :class:`CandidateProbeResult`
alongside the ``FrameEvidence``.

## Relationship to #106

The reference implementation below renders the cohort **sequentially**, reusing
the same :func:`invoke_render` machinery ``run_batched_step`` uses. True
``C > 1`` camera batching is a future consumer, not a prerequisite. Every batch
result reports ``candidate_probe_rasterizer_call_count``, so a future claim
that a probe was executed as one true camera-batch render is verifiable from
the accounting rather than taken on faith.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterator, Mapping, Sequence

import torch

from eligsir.contracts.control import ResolutionTier
from eligsir.mapping.evidence.canonical import (
    EvidenceAvailability,
    EvidenceIdentity,
    FrameEvidence,
    GaussianProbeEvidence,
    ScalarEvidence,
)
from eligsir.mapping.resolution_materialization import (
    RESAMPLING_MODES,
    materialize_resolution_tier,
)
from eligsir.optimization.gaussian.render_port import invoke_render

#: Version of the objective the probe evaluates. Recorded in every result's
#: ``EvidenceIdentity.objective_version``. Bump whenever the numerical
#: definition of any probe objective component changes, so stored evidence
#: from two different definitions can never be silently compared.
CANDIDATE_PROBE_OBJECTIVE_VERSION = "candidate_probe_objective_v1"

#: Version of the evidence STRUCTURE this module emits (field set, pairwise
#: shape, accounting names). Independent of the objective version above: the
#: same objective can be reported in a richer envelope, and the same envelope
#: can carry a redefined objective.
CANDIDATE_PROBE_EVIDENCE_SCHEMA_VERSION = 1

#: Default absolute RGB residual below which a pixel counts as "already well
#: explained" by the current map. Chosen at 2/255 of the [0, 1] image range:
#: roughly 8-bit quantization noise, i.e. the point past which a pixel cannot
#: usefully be improved by more supervision.
DEFAULT_WELL_EXPLAINED_RESIDUAL = 2.0 / 255.0

_EPS = 1e-12


class ProbeStateMutationError(RuntimeError):
    """A probe was observed to have changed optimization state.

    This is always a bug in the probe path (or in something it called), never a
    condition a caller should catch and continue past: by the time it is raised,
    live training state has already been perturbed by a read-only operation.
    """


class ProbeUnsafeContextError(RuntimeError):
    """A probe was requested from inside an open optimizer transaction."""


class ProbeEvidenceChannel(str, Enum):
    """Opt-in evidence families. Every channel costs something."""

    #: RGB objective components, normalized residual, residual spatial structure.
    RGB = "rgb"
    #: Sensor-depth residual where the frame carries valid depth.
    DEPTH = "depth"
    #: Alpha/visibility coverage and visible-Gaussian counts.
    VISIBILITY = "visibility"
    #: Retain the per-Gaussian visibility mask/IDs (memory: one bool row per
    #: Gaussian per view) -- required for pairwise visible-set overlap.
    VISIBLE_GAUSSIAN_IDS = "visible_gaussian_ids"
    #: Retain a small downsampled residual sketch -- required for pairwise
    #: residual-map similarity.
    RESIDUAL_SKETCH = "residual_sketch"
    #: Phase 3: exact per-view gradient of the probe objective. Forward AND
    #: backward per view; never enabled implicitly.
    GRADIENT = "gradient"


#: The forward-only default cohort: everything issue #803 Phase 1 asks for,
#: plus the two retention channels Phase 2's pairwise evidence needs. The
#: gradient channel is deliberately NOT included.
DEFAULT_PROBE_CHANNELS: frozenset[ProbeEvidenceChannel] = frozenset(
    {
        ProbeEvidenceChannel.RGB,
        ProbeEvidenceChannel.DEPTH,
        ProbeEvidenceChannel.VISIBILITY,
        ProbeEvidenceChannel.VISIBLE_GAUSSIAN_IDS,
        ProbeEvidenceChannel.RESIDUAL_SKETCH,
    }
)


# ---------------------------------------------------------------------------
# Request / result contracts (naming follows contracts.py's style)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateProbeView:
    """One candidate camera to probe, with its stable source identity.

    ``source_id``/``source_ordinal``/``timestamp_ns`` are carried through to the
    result verbatim so a scheduler can associate every result with the exact
    candidate it asked about -- never by list position alone, which a skipped
    candidate would silently shift.
    """

    camera: Any
    background: Any = None
    source_id: str = ""
    source_ordinal: int = -1
    timestamp_ns: int = 0
    render_depth: bool = True
    #: Probes default to NOT updating the SH-rest band: it is a cost the
    #: evidence does not need, and leaving it False keeps the probe further
    #: away from anything resembling a training render.
    update_sh_rest: bool = False


@dataclass(frozen=True)
class CandidateProbeRequest:
    """One probe batch: B candidate views against one map/topology snapshot."""

    views: tuple[CandidateProbeView, ...]
    probe_batch_id: str = "probe"
    optimization_step: int = 0
    #: Probe resolution. **Not every signal survives a reduced tier** -- measured
    #: (issue #803, ``scripts/issue803_probe_resolution_benchmark.py``, 50k
    #: Gaussians / 16 cameras / 640px, forward-only, Radeon 8060S):
    #:
    #: * Magnitude/mean signals rank essentially identically down to quarter
    #:   resolution -- ``rgb_l1_mean``, ``rgb_mse_mean``, ``rgb_objective``,
    #:   ``rgb_residual_normalized``, ``residual_top_decile_mass_fraction``,
    #:   ``alpha_mean``, ``visible_gaussian_fraction`` all hold Spearman >= 0.97
    #:   against the native probe.
    #: * **Threshold-crossing fractions do NOT.** ``supported_pixel_fraction``
    #:   (alpha > 0.5) measured Spearman -0.88 at half and -0.99 at quarter --
    #:   an inverted ranking, not merely a noisy one -- and
    #:   ``well_explained_pixel_fraction`` measured ~-0.07, i.e. no relationship
    #:   at all. The reason is that a threshold-crossing statistic is not
    #:   resolution-invariant whenever the per-pixel value distribution it
    #:   thresholds is itself resolution-dependent -- and here it is, by two
    #:   INDEPENDENT routes: alpha rises with pixel footprint because a coarser
    #:   pixel composites more Gaussians (a rendering property, which
    #:   ``resampling_mode`` cannot change), while the RGB residual shifts
    #:   because both the ground truth is resampled and the render is coarser.
    #:   Note the alpha case measured on a synthetic near-saturated-coverage
    #:   scene, the regime where a ceiling effect is strongest; the sign of the
    #:   inversion is not claimed to be scene-independent, only its instability.
    #: * ``residual_entropy`` is unstable across tiers (-0.65 at half, +0.37 at
    #:   quarter): it is normalized by pixel count, and pooling changes the
    #:   distribution's shape, not just its scale.
    #:
    #: Consequence: a consumer that reduces the probe tier for speed must
    #: restrict itself to the magnitude/mean signals. Comparing a reduced-tier
    #: threshold-fraction against a native-tier one is not a comparison.
    resolution_tier: ResolutionTier = ResolutionTier.NATIVE
    resampling_mode: str = "legacy"
    evidence_channels: frozenset[ProbeEvidenceChannel] = DEFAULT_PROBE_CHANNELS
    objective_version: str = CANDIDATE_PROBE_OBJECTIVE_VERSION
    #: Photometric SSIM weight. Mirrors ``opt.lambda_dssim``; the engine fills
    #: it from the runtime context when the caller leaves it at ``None``.
    lambda_dssim: float | None = None
    well_explained_residual_threshold: float = DEFAULT_WELL_EXPLAINED_RESIDUAL
    #: Linear size of the square residual sketch used for pairwise similarity.
    residual_sketch_size: int = 16
    #: Compute pairwise (Phase 2) evidence for this cohort.
    compute_pairwise: bool = True
    #: Synchronize the device around each view's timing. Off by default: a
    #: synchronize inside live mapping serializes the queue. Turn it on for
    #: benchmarking, where an unsynchronized "GPU ms" is not a measurement at
    #: all. ``probe_gpu_ms`` is reported ONLY when this is enabled, rather than
    #: quietly aliasing wall time and calling it device time.
    synchronize_device: bool = False

    def __post_init__(self) -> None:
        if not self.views:
            raise ValueError("a candidate probe batch requires at least one view")
        if not isinstance(self.resolution_tier, ResolutionTier):
            raise TypeError("resolution_tier must be a ResolutionTier")
        if self.resampling_mode not in RESAMPLING_MODES:
            raise ValueError(f"resampling_mode must be one of {RESAMPLING_MODES!r}")
        if int(self.residual_sketch_size) < 1:
            raise ValueError("residual_sketch_size must be >= 1")
        object.__setattr__(
            self, "evidence_channels", frozenset(self.evidence_channels)
        )
        object.__setattr__(self, "views", tuple(self.views))

    def wants(self, channel: ProbeEvidenceChannel) -> bool:
        """Return whether this request asked for ``channel``."""

        return channel in self.evidence_channels


@dataclass(frozen=True)
class ProbeGradientEvidence:
    """Exact per-view gradient evidence for one candidate (Phase 3).

    Produced by an isolated :func:`torch.autograd.grad` on THIS view's own
    objective. It is never a share of a reduced batch gradient: a per-view
    gradient signal derived from an aggregate backward has no per-view
    mathematical meaning, and issue #803 rules it out explicitly.
    """

    #: L2 norm over every parameter family, concatenated.
    total_norm: float
    #: Squared L2 norm (energy) over every parameter family.
    total_energy: float
    #: Per-family squared L2 norm, keyed by canonical family name
    #: (``means``/``opacity``/``scale``/``rotation``/``color``).
    energy_by_family: Mapping[str, float]
    #: Fraction of Gaussian rows receiving any nonzero positional gradient.
    touched_gaussian_fraction: float | None
    #: Number of Gaussian rows receiving any nonzero positional gradient.
    touched_gaussian_count: int | None
    #: Flattened, L2-normalized low-dimensional sketch of the positional
    #: gradient, for cheap pairwise alignment/conflict comparison. ``None``
    #: when positional gradients were unavailable.
    gradient_sketch: Any = None


@dataclass(frozen=True)
class CandidateProbeResult:
    """Per-view probe evidence for exactly one candidate."""

    view_index: int
    source_id: str
    source_ordinal: int
    timestamp_ns: int
    evidence: FrameEvidence
    #: True when this candidate could not be probed at all (see ``skip_reason``).
    #: A skipped candidate still occupies its position in ``per_view`` and still
    #: carries its own identity -- it is never silently dropped, because a
    #: dropped element would shift every later result's index.
    skipped: bool = False
    skip_reason: str | None = None
    probe_width: int = 0
    probe_height: int = 0
    probe_pixel_count: int = 0
    rasterizer_call_count: int = 0
    #: Bool tensor over Gaussian rows, retained only for the
    #: ``VISIBLE_GAUSSIAN_IDS`` channel.
    visible_gaussian_mask: Any = None
    #: Int64 tensor of visible Gaussian row indices (same channel).
    visible_gaussian_ids: Any = None
    visible_gaussian_count: int | None = None
    #: ``[S, S]`` float tensor of mean absolute residual per cell.
    residual_sketch: Any = None
    gradient: ProbeGradientEvidence | None = None
    #: Camera-to-world translation, retained for pose-baseline pairwise
    #: evidence when the candidate camera exposes one.
    camera_center: Any = None
    #: Camera forward axis in world space, for pairwise viewing-angle evidence.
    camera_forward: Any = None


@dataclass(frozen=True)
class CandidateProbePairEvidence:
    """Cheap complementarity/redundancy evidence for one ordered pair.

    Every field is deliberately raw geometry or raw set/vector similarity. What
    counts as "complementary enough to be worth admitting both" is a scheduler
    policy question and is not answered here.
    """

    view_index_a: int
    view_index_b: int
    source_id_a: str
    source_id_b: str
    #: |A n B| / |A u B| over visible-Gaussian sets.
    visible_gaussian_jaccard: float | None = None
    #: |A n B| / min(|A|, |B|) -- containment rather than symmetric overlap.
    visible_gaussian_containment: float | None = None
    visible_gaussian_intersection: int | None = None
    #: Cosine similarity of the two downsampled residual sketches.
    residual_sketch_cosine: float | None = None
    #: Euclidean distance between camera centres, metres.
    pose_baseline_m: float | None = None
    #: Angle between camera forward axes, degrees.
    pose_angle_deg: float | None = None
    #: Cosine similarity of positional-gradient sketches (gradient channel only).
    gradient_sketch_cosine: float | None = None


@dataclass(frozen=True)
class CandidateProbeAccounting:
    """Probe work accounting (issue #803's performance contract).

    Deliberately named and typed separately from issue #703's supervision
    accounting on ``OptimizationStepResult``. Probe views render real pixels but
    participate in NO optimizer update, so folding them into supervision counts
    would corrupt the very number #703 exists to make trustworthy. These fields
    must never be summed into ``supervision_view_count`` /
    ``supervised_pixel_count`` / ``rasterizer_call_count``.
    """

    candidate_probe_batch_count: int = 0
    candidate_probe_view_count: int = 0
    candidate_probe_rasterizer_call_count: int = 0
    candidate_probe_pixel_count: int = 0
    candidate_probe_wall_ms: float = 0.0
    candidate_probe_gpu_ms: float = 0.0
    candidate_probe_gradient_view_count: int = 0
    #: Candidates that could not be probed (invalid source, render failure).
    candidate_probe_skipped_view_count: int = 0

    def merged(self, other: "CandidateProbeAccounting") -> "CandidateProbeAccounting":
        """Return the element-wise sum of two accounting records."""

        return CandidateProbeAccounting(
            candidate_probe_batch_count=self.candidate_probe_batch_count
            + other.candidate_probe_batch_count,
            candidate_probe_view_count=self.candidate_probe_view_count
            + other.candidate_probe_view_count,
            candidate_probe_rasterizer_call_count=(
                self.candidate_probe_rasterizer_call_count
                + other.candidate_probe_rasterizer_call_count
            ),
            candidate_probe_pixel_count=self.candidate_probe_pixel_count
            + other.candidate_probe_pixel_count,
            candidate_probe_wall_ms=self.candidate_probe_wall_ms
            + other.candidate_probe_wall_ms,
            candidate_probe_gpu_ms=self.candidate_probe_gpu_ms
            + other.candidate_probe_gpu_ms,
            candidate_probe_gradient_view_count=(
                self.candidate_probe_gradient_view_count
                + other.candidate_probe_gradient_view_count
            ),
            candidate_probe_skipped_view_count=(
                self.candidate_probe_skipped_view_count
                + other.candidate_probe_skipped_view_count
            ),
        )


@dataclass(frozen=True)
class CandidateProbeBatchResult:
    """The whole cohort's evidence against one map/topology snapshot.

    ``per_view`` is index-aligned with ``request.views``: element *i* always
    describes view *i*, skipped or not.
    """

    probe_batch_id: str
    per_view: tuple[CandidateProbeResult, ...]
    pairwise: tuple[CandidateProbePairEvidence, ...]
    map_version: int
    topology_version: int
    coordinate_frame_version: int
    optimization_step: int
    resolution_tier: ResolutionTier
    resolution_bucket: str
    objective_version: str
    evidence_schema_version: int
    accounting: CandidateProbeAccounting
    gaussian_count: int = 0
    evidence_channels: frozenset[ProbeEvidenceChannel] = frozenset()

    def result_for(self, source_id: str) -> CandidateProbeResult:
        """Return the result carrying ``source_id``, raising if ambiguous."""

        matches = [r for r in self.per_view if r.source_id == source_id]
        if len(matches) != 1:
            raise KeyError(
                f"expected exactly one probe result for source_id={source_id!r}, "
                f"found {len(matches)}"
            )
        return matches[0]


# ---------------------------------------------------------------------------
# State fingerprinting -- the mechanical proof of non-interference
# ---------------------------------------------------------------------------


def _parameter_families(gaussians: Any) -> dict[str, Any]:
    """Return ``{canonical_family_name: Parameter}`` for either model layout.

    Both layouts are handled here so the probe (and its side-effect proof)
    works on the gsplat ``ParameterDict`` layout and the legacy attribute
    layout without either call site branching.
    """

    out: dict[str, Any] = {}
    params = getattr(gaussians, "params", None)
    if params is not None:
        try:
            items = list(params.items())
        except AttributeError:
            items = []
        for name, tensor in items:
            if isinstance(tensor, torch.Tensor):
                out[str(name)] = tensor
    if out:
        return out
    for attr in (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_opacity",
        "_scaling",
        "_rotation",
    ):
        tensor = getattr(gaussians, attr, None)
        if isinstance(tensor, torch.Tensor):
            out[attr] = tensor
    return out


#: Canonical parameter-family names, so gradient energies are reported under
#: one vocabulary regardless of which model layout produced them.
_FAMILY_ALIASES = {
    "means": "means",
    "_xyz": "means",
    "opacities": "opacity",
    "_opacity": "opacity",
    "scales": "scale",
    "_scaling": "scale",
    "quats": "rotation",
    "_rotation": "rotation",
    "sh0": "color",
    "shN": "color",
    "_features_dc": "color",
    "_features_rest": "color",
}

_POSITION_PARAM_NAMES = ("means", "_xyz")


def probe_gaussian_row_count(gaussians: Any) -> int:
    """Number of Gaussian rows in the snapshot, for either model layout."""

    xyz = getattr(gaussians, "get_xyz", None)
    if isinstance(xyz, torch.Tensor) and xyz.ndim >= 1:
        return int(xyz.shape[0])
    for name in _POSITION_PARAM_NAMES:
        tensor = _parameter_families(gaussians).get(name)
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
            return int(tensor.shape[0])
    return 0


def canonical_family(param_name: str) -> str:
    """Map a layout-specific parameter name to its canonical family name."""

    return _FAMILY_ALIASES.get(param_name, param_name)


def _optimizers(session: Any, gaussians: Any) -> list[tuple[str, Any]]:
    """Every optimizer reachable from the session/model, labelled.

    The gsplat layout keeps one Adam per parameter in ``gaussians.optimizers``;
    the legacy layout keeps a single ``optimizer``. Both are returned so the
    state fingerprint covers whichever layout is live.
    """

    found: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for holder, label in ((session, "session"), (gaussians, "model")):
        if holder is None:
            continue
        single = getattr(holder, "optimizer", None)
        if single is not None and id(single) not in seen:
            seen.add(id(single))
            found.append((label, single))
        group = getattr(holder, "optimizers", None)
        if isinstance(group, Mapping):
            for name, optimizer in sorted(group.items(), key=lambda kv: str(kv[0])):
                if optimizer is not None and id(optimizer) not in seen:
                    seen.add(id(optimizer))
                    found.append((f"{label}.{name}", optimizer))
    return found


def capture_state_fingerprint(
    *,
    backend: Any = None,
    session: Any = None,
    gaussians: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Snapshot everything a probe is forbidden to change.

    Uses ``Tensor._version`` (bumped by ANY in-place mutation, including one
    that leaves the values numerically identical) and ``data_ptr()`` (changes
    when a tensor is replaced) rather than value hashes. That makes the check
    exact for the mutation classes that matter AND free of a host
    synchronization, which is what lets it run on every probe instead of behind
    a debug flag -- a correctness gate nobody can afford to leave on is a
    correctness gate that will be off when it matters.
    """

    fingerprint: dict[str, Any] = {}
    for name in (
        "map_content_version",
        "topology_version",
        "coordinate_frame_version",
        "cumulative_inserted",
        "cumulative_pruned",
    ):
        if backend is not None and hasattr(backend, name):
            fingerprint[f"backend.{name}"] = int(getattr(backend, name))
    if session is not None and hasattr(session, "parameter_version"):
        fingerprint["session.parameter_version"] = int(
            getattr(session, "parameter_version") or 0
        )

    if gaussians is not None:
        for name, tensor in sorted(_parameter_families(gaussians).items()):
            fingerprint[f"param.{name}.ptr"] = tensor.data_ptr()
            fingerprint[f"param.{name}.version"] = tensor._version
            fingerprint[f"param.{name}.shape"] = tuple(tensor.shape)
            fingerprint[f"param.{name}.requires_grad"] = bool(tensor.requires_grad)
            grad = tensor.grad
            fingerprint[f"grad.{name}.present"] = grad is not None
            if grad is not None:
                fingerprint[f"grad.{name}.ptr"] = grad.data_ptr()
                fingerprint[f"grad.{name}.version"] = grad._version

    # Both optimizer shapes this codebase uses are covered: the legacy layout's
    # single ``optimizer`` and the gsplat layout's per-parameter ``optimizers``
    # dict. Missing one of them would silently make the strongest half of this
    # check (Adam moment/step state) a no-op on the production layout.
    for label, optimizer in _optimizers(session, gaussians):
        groups = getattr(optimizer, "param_groups", None)
        if groups is not None:
            fingerprint[f"optimizer.{label}.lrs"] = tuple(
                float(group.get("lr", 0.0)) for group in groups
            )
        state = getattr(optimizer, "state", None)
        if state is None:
            continue
        step_values: list[Any] = []
        tensor_marks: list[Any] = []
        for entry in state.values():
            if not isinstance(entry, dict):
                continue
            for key, value in sorted(entry.items()):
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1 and key == "step":
                        step_values.append(float(value.detach().cpu()))
                    else:
                        tensor_marks.append((key, value.data_ptr(), value._version))
                elif isinstance(value, (int, float)):
                    step_values.append(float(value))
        fingerprint[f"optimizer.{label}.state_scalars"] = tuple(step_values)
        fingerprint[f"optimizer.{label}.state_tensors"] = tuple(tensor_marks)

    if extra:
        for key, value in sorted(extra.items()):
            fingerprint[f"extra.{key}"] = value
    return fingerprint


def diff_state_fingerprints(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> list[str]:
    """Return human-readable descriptions of every fingerprint difference."""

    problems: list[str] = []
    for key in sorted(set(before) | set(after)):
        if key not in before:
            problems.append(f"{key}: appeared (now {after[key]!r})")
        elif key not in after:
            problems.append(f"{key}: disappeared (was {before[key]!r})")
        elif before[key] != after[key]:
            problems.append(f"{key}: {before[key]!r} -> {after[key]!r}")
    return problems


# ---------------------------------------------------------------------------
# Resolution binding (reuses the canonical multires materialization)
# ---------------------------------------------------------------------------

_TIER_MARKER_ATTRS = (
    "_canonical_resolution_tier",
    "_canonical_resolution_binding_native_state",
    "_canonical_resolution_resampling_mode",
    "_canonical_resolution_depth_downsample",
)


@contextmanager
def probe_resolution_binding(
    camera: Any,
    tier: ResolutionTier,
    *,
    resampling_mode: str = "legacy",
    depth_downsample: str = "nearest",
) -> Iterator[bool]:
    """Bind ``camera`` to ``tier`` for the probe, then restore it exactly.

    ``materialize_resolution_tier`` is the project's ONE resize/cache
    subsystem for supervision resolution and is reused verbatim -- issue #803
    is explicit that a probe must not grow a second one. What this wrapper adds
    is the part the probe needs and supervision does not: putting the camera
    back. Cohort cameras are routinely the same objects the training loop will
    later render, and a probe that leaves one bound at quarter resolution has
    silently changed a training run while every Gaussian and optimizer hash
    still matches.

    Yields ``True`` when a real rebinding happened, ``False`` when the camera
    was already in the requested state and nothing was touched.
    """

    had_tier = hasattr(camera, "_canonical_resolution_tier")
    had_native = hasattr(camera, "_canonical_native_resolution_state")
    previous_tier = getattr(camera, "_canonical_resolution_tier", None)
    previous_mode = getattr(camera, "_canonical_resolution_resampling_mode", None)
    previous_depth_downsample = getattr(
        camera, "_canonical_resolution_depth_downsample", None
    )
    previous_weight = getattr(camera, "_resolution_depth_confidence_weight", None)
    had_weight = hasattr(camera, "_resolution_depth_confidence_weight")
    previous_gsplat = getattr(camera, "gsplat", None)

    if not had_tier and tier is ResolutionTier.NATIVE:
        # An unbound camera IS its native binding. Materializing would only add
        # marker attributes and drop the renderer tensor cache for nothing.
        yield False
        return
    if (
        had_tier
        and previous_tier is tier
        and previous_mode == resampling_mode
        and previous_depth_downsample == depth_downsample
    ):
        yield False
        return

    materialize_resolution_tier(
        camera, tier, resampling_mode=resampling_mode, depth_downsample=depth_downsample
    )
    try:
        yield True
    finally:
        materialize_resolution_tier(
            camera,
            previous_tier if had_tier else ResolutionTier.NATIVE,
            resampling_mode=previous_mode if previous_mode is not None else "legacy",
            depth_downsample=(
                previous_depth_downsample if previous_depth_downsample is not None else "nearest"
            ),
        )
        if not had_tier:
            for attr in _TIER_MARKER_ATTRS:
                if hasattr(camera, attr):
                    delattr(camera, attr)
            if not had_native and hasattr(
                camera, "_canonical_native_resolution_state"
            ):
                delattr(camera, "_canonical_native_resolution_state")
        if had_weight:
            camera._resolution_depth_confidence_weight = previous_weight
        elif hasattr(camera, "_resolution_depth_confidence_weight"):
            delattr(camera, "_resolution_depth_confidence_weight")
        # The restored geometry is identical to the pre-probe geometry, so the
        # renderer's cached camera tensors are valid again; putting the cache
        # back avoids charging the next training render for the probe.
        camera.gsplat = previous_gsplat


# ---------------------------------------------------------------------------
# Per-view evidence computation
# ---------------------------------------------------------------------------


def _available(value: float, unit: str = "unitless") -> ScalarEvidence:
    if not math.isfinite(float(value)):
        return ScalarEvidence.missing(
            EvidenceAvailability.INVALID_SOURCE,
            unit=unit,
            reason="non-finite probe measurement",
        )
    return ScalarEvidence(float(value), unit=unit)


def _unsupported(reason: str, unit: str = "unitless") -> ScalarEvidence:
    return ScalarEvidence.missing(
        EvidenceAvailability.UNEVALUABLE_NO_MAP_SUPPORT, unit=unit, reason=reason
    )


def _not_requested(unit: str = "unitless") -> ScalarEvidence:
    return ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED, unit=unit)


def residual_sketch(residual_hw: torch.Tensor, size: int) -> torch.Tensor:
    """Downsample a ``[H, W]`` non-negative residual map to ``[size, size]``.

    Average pooling (not sampling): every native residual pixel contributes to
    exactly one cell, and each cell reports its footprint's MEAN residual.
    Mean- rather than sum-preserving on purpose -- two probes of the same view
    at different resolution tiers then produce sketches on the same numerical
    scale, so a pairwise residual-similarity comparison stays meaningful across
    tiers instead of being dominated by how many pixels each tier had.
    """

    tensor = residual_hw.to(torch.float32)
    if tensor.ndim != 2:
        raise ValueError("residual sketch input must be [H, W]")
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        tensor[None, None], (int(size), int(size))
    )
    return pooled[0, 0].contiguous()


def _cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    flat_a = a.reshape(-1).to(torch.float32)
    flat_b = b.reshape(-1).to(torch.float32)
    if flat_a.numel() != flat_b.numel() or flat_a.numel() == 0:
        return None
    denominator = float(flat_a.norm()) * float(flat_b.norm())
    if denominator <= _EPS:
        return None
    return float(torch.dot(flat_a, flat_b)) / denominator


def _camera_geometry(camera: Any) -> tuple[Any, Any]:
    """Return ``(camera_centre, forward_axis)`` as CPU tensors, or ``(None, None)``.

    Read from whatever the camera already exposes; a candidate camera that
    carries no usable pose simply yields no pose-derived pairwise evidence,
    rather than the probe inventing one.
    """

    centre = getattr(camera, "camera_center", None)
    if isinstance(centre, torch.Tensor) and centre.numel() >= 3:
        centre = centre.detach().reshape(-1)[:3].to(torch.float32).cpu()
    else:
        centre = None
    forward = None
    world_view = getattr(camera, "world_view_transform", None)
    if isinstance(world_view, torch.Tensor) and world_view.shape[-2:] == (4, 4):
        # This codebase's convention stores the transposed world->view matrix;
        # its third COLUMN of the rotation block is the camera's world-space
        # forward axis.
        forward = world_view.detach().to(torch.float32).cpu()[:3, 2].reshape(-1)
        norm = float(forward.norm())
        forward = forward / norm if norm > _EPS else None
    return centre, forward


def _rgb_evidence(
    render_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    *,
    lambda_dssim: float,
    well_explained_threshold: float,
) -> tuple[dict[str, ScalarEvidence], torch.Tensor]:
    """Compute the RGB evidence block and return the per-pixel residual map."""

    from eligsir.utils.loss_utils import ssim as _ssim_fn

    residual = (render_rgb - gt_rgb).abs()
    residual_hw = residual.mean(dim=0)
    l1_mean = float(residual.mean())
    mse_mean = float((render_rgb - gt_rgb).square().mean())
    gt_energy = float(gt_rgb.abs().mean())
    ssim_value = float(_ssim_fn(render_rgb, gt_rgb))
    objective = (1.0 - lambda_dssim) * l1_mean + lambda_dssim * (1.0 - ssim_value)

    flat = residual_hw.reshape(-1)
    total = float(flat.sum())
    if flat.numel() > 1 and total > _EPS:
        probabilities = flat / total
        entropy = float(
            -(probabilities * probabilities.clamp_min(_EPS).log()).sum()
        ) / math.log(float(flat.numel()))
        decile = max(1, flat.numel() // 10)
        top_mass = float(torch.topk(flat, decile).values.sum()) / total
    else:
        entropy = 0.0
        top_mass = 0.0
    well_explained = float(
        (residual_hw <= float(well_explained_threshold)).to(torch.float32).mean()
    )

    return (
        {
            "rgb_l1_mean": _available(l1_mean),
            "rgb_mse_mean": _available(mse_mean),
            "rgb_ssim": _available(ssim_value),
            "rgb_objective": _available(objective),
            "rgb_residual_normalized": _available(l1_mean / (gt_energy + _EPS)),
            "residual_entropy": _available(entropy),
            "residual_top_decile_mass_fraction": _available(top_mass),
            "well_explained_pixel_fraction": _available(well_explained),
        },
        residual_hw,
    )


def _depth_evidence(
    payload: Mapping[str, Any], camera: Any, height: int, width: int
) -> dict[str, ScalarEvidence]:
    """Sensor-depth residual over valid pixels, or reason-coded absence."""

    rendered = payload.get("rendered_depth", payload.get("depth"))
    if rendered is None:
        reason = "renderer produced no depth channel"
        return {
            "depth_l1_mean_m": _unsupported(reason, "m"),
            "depth_rmse_m": _unsupported(reason, "m"),
            "depth_valid_pixel_fraction": _unsupported(reason),
            "depth_valid_pixel_count": _unsupported(reason, "pixels"),
        }
    # REFUSE rather than populate. `_get_sensor_depth`'s FIRST-access branch
    # decodes and WRITES `camera._sensor_depth_cache` at whatever target size it
    # was asked for -- here, the probe's (possibly reduced) resolution. That
    # write survives this probe's tier restore, because the restore replays
    # `_canonical_native_resolution_state`, whose `depth` was captured as None
    # precisely because the cache was empty when the tier was bound. A later
    # supervision render at native resolution would then read a quarter-
    # resolution depth cache -- a silent training-state corruption in a
    # different attribute than the ones the tier restore snapshots, i.e. exactly
    # the leak class this probe exists to rule out, wearing a different hat.
    # With a cache already present, `_get_sensor_depth` resizes and returns
    # WITHOUT writing back, which is safe and is the path production takes.
    if not isinstance(getattr(camera, "_sensor_depth_cache", None), torch.Tensor):
        reason = (
            "sensor depth is not materialized for this camera; a probe must not "
            "populate the depth cache at probe resolution"
        )
        return {
            "depth_l1_mean_m": _unsupported(reason, "m"),
            "depth_rmse_m": _unsupported(reason, "m"),
            "depth_valid_pixel_fraction": _unsupported(reason),
            "depth_valid_pixel_count": _unsupported(reason, "pixels"),
        }
    from eligsir.mapping.integration.depth_helpers import _get_sensor_depth

    sensor = _get_sensor_depth(camera, int(height), int(width))
    if sensor is None:
        reason = "candidate frame carries no sensor depth"
        return {
            "depth_l1_mean_m": _unsupported(reason, "m"),
            "depth_rmse_m": _unsupported(reason, "m"),
            "depth_valid_pixel_fraction": _unsupported(reason),
            "depth_valid_pixel_count": _unsupported(reason, "pixels"),
        }
    rendered_hw = rendered.detach().to(torch.float32).reshape(int(height), int(width))
    sensor_hw = (
        sensor.detach()
        .to(device=rendered_hw.device, dtype=torch.float32)
        .reshape(int(height), int(width))
    )
    valid = torch.isfinite(rendered_hw) & torch.isfinite(sensor_hw) & (sensor_hw > 0)
    valid_count = int(valid.sum())
    fraction = valid_count / float(max(1, valid.numel()))
    if valid_count == 0:
        reason = "no valid sensor-depth pixels in this view"
        return {
            "depth_l1_mean_m": _unsupported(reason, "m"),
            "depth_rmse_m": _unsupported(reason, "m"),
            "depth_valid_pixel_fraction": _available(fraction),
            "depth_valid_pixel_count": _available(valid_count, "pixels"),
        }
    error = rendered_hw[valid] - sensor_hw[valid]
    return {
        "depth_l1_mean_m": _available(float(error.abs().mean()), "m"),
        "depth_rmse_m": _available(float(error.square().mean().sqrt()), "m"),
        "depth_valid_pixel_fraction": _available(fraction),
        "depth_valid_pixel_count": _available(valid_count, "pixels"),
    }


def _visibility_evidence(
    payload: Mapping[str, Any],
) -> tuple[dict[str, ScalarEvidence], Any, int | None]:
    """Alpha coverage plus visible-Gaussian support for one view."""

    alpha = payload.get("alpha", payload.get("accumulation"))
    if alpha is None:
        reason = "renderer produced no alpha channel"
        block = {
            "alpha_mean": _unsupported(reason),
            "supported_pixel_fraction": _unsupported(reason),
            "unexplained_pixel_fraction": _unsupported(reason),
        }
    else:
        alpha_f = alpha.detach().to(torch.float32)
        block = {
            "alpha_mean": _available(float(alpha_f.mean())),
            "supported_pixel_fraction": _available(
                float((alpha_f > 0.5).to(torch.float32).mean())
            ),
            "unexplained_pixel_fraction": _available(
                float((alpha_f < 0.1).to(torch.float32).mean())
            ),
        }

    visible = payload.get("visibility_filter")
    if isinstance(visible, torch.Tensor) and visible.numel() > 0:
        mask = visible.detach().to(torch.bool)
        count = int(mask.sum())
        block["visible_gaussian_count"] = _available(count, "gaussians")
        block["visible_gaussian_fraction"] = _available(
            count / float(max(1, mask.numel()))
        )
        return block, mask, count
    reason = "renderer produced no visibility filter"
    block["visible_gaussian_count"] = _unsupported(reason, "gaussians")
    block["visible_gaussian_fraction"] = _unsupported(reason)
    return block, None, None


def _gradient_evidence(
    objective: torch.Tensor, gaussians: Any, *, sketch_size: int
) -> ProbeGradientEvidence:
    """Exact per-view gradient of ``objective``, via isolated ``autograd.grad``.

    ``torch.autograd.grad`` RETURNS the gradients instead of accumulating them
    into ``.grad``. That is not an incidental convenience: it is the reason a
    gradient probe can satisfy the "no hidden ``.grad`` accumulation" invariant
    at all. ``loss.backward()`` must never appear on this path.
    """

    families = _parameter_families(gaussians)
    names = [n for n, t in families.items() if t.requires_grad]
    if not names:
        return ProbeGradientEvidence(
            total_norm=0.0,
            total_energy=0.0,
            energy_by_family={},
            touched_gaussian_fraction=None,
            touched_gaussian_count=None,
        )
    tensors = [families[name] for name in names]
    grads = torch.autograd.grad(
        objective,
        tensors,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    energy_by_family: dict[str, float] = {}
    total_energy = 0.0
    position_grad = None
    for name, grad in zip(names, grads):
        if grad is None:
            continue
        energy = float(grad.detach().square().sum())
        family = canonical_family(name)
        energy_by_family[family] = energy_by_family.get(family, 0.0) + energy
        total_energy += energy
        if name in _POSITION_PARAM_NAMES:
            position_grad = grad.detach()

    touched_count = None
    touched_fraction = None
    sketch = None
    if position_grad is not None and position_grad.ndim >= 2:
        rows = position_grad.reshape(position_grad.shape[0], -1)
        row_energy = rows.square().sum(dim=1)
        touched = row_energy > 0
        touched_count = int(touched.sum())
        touched_fraction = touched_count / float(max(1, rows.shape[0]))
        # Low-dimensional sketch: bucket rows by index into `sketch_size`
        # contiguous blocks and record each block's gradient energy. Cheap,
        # deterministic, and comparable across views of the SAME snapshot
        # (row order is fixed for a fixed topology, which is exactly the
        # condition a probe batch guarantees).
        buckets = max(1, int(sketch_size))
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            row_energy.reshape(1, 1, -1).to(torch.float32), buckets
        )[0, 0]
        norm = float(pooled.norm())
        sketch = (pooled / norm).cpu() if norm > _EPS else pooled.cpu()

    return ProbeGradientEvidence(
        total_norm=math.sqrt(max(0.0, total_energy)),
        total_energy=total_energy,
        energy_by_family=dict(sorted(energy_by_family.items())),
        touched_gaussian_fraction=touched_fraction,
        touched_gaussian_count=touched_count,
        gradient_sketch=sketch,
    )


def build_pairwise_evidence(
    results: Sequence[CandidateProbeResult],
) -> tuple[CandidateProbePairEvidence, ...]:
    """Phase 2: cheap complementarity/redundancy evidence, no rerendering.

    Every pair ``(i, j)`` with ``i < j`` over the non-skipped results, in
    ascending index order -- deterministic by construction, so two probes of
    the same cohort produce byte-identical pairwise output.
    """

    usable = [r for r in results if not r.skipped]
    pairs: list[CandidateProbePairEvidence] = []
    for position, left in enumerate(usable):
        for right in usable[position + 1 :]:
            jaccard = containment = None
            intersection = None
            if (
                isinstance(left.visible_gaussian_mask, torch.Tensor)
                and isinstance(right.visible_gaussian_mask, torch.Tensor)
                and left.visible_gaussian_mask.shape
                == right.visible_gaussian_mask.shape
            ):
                a = left.visible_gaussian_mask
                b = right.visible_gaussian_mask
                intersection = int((a & b).sum())
                union = int((a | b).sum())
                smaller = min(int(a.sum()), int(b.sum()))
                jaccard = intersection / float(union) if union > 0 else None
                containment = (
                    intersection / float(smaller) if smaller > 0 else None
                )
            baseline = angle = None
            if left.camera_center is not None and right.camera_center is not None:
                baseline = float((left.camera_center - right.camera_center).norm())
            if left.camera_forward is not None and right.camera_forward is not None:
                cosine = float(
                    torch.dot(left.camera_forward, right.camera_forward)
                )
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            pairs.append(
                CandidateProbePairEvidence(
                    view_index_a=left.view_index,
                    view_index_b=right.view_index,
                    source_id_a=left.source_id,
                    source_id_b=right.source_id,
                    visible_gaussian_jaccard=jaccard,
                    visible_gaussian_containment=containment,
                    visible_gaussian_intersection=intersection,
                    residual_sketch_cosine=_cosine(
                        left.residual_sketch, right.residual_sketch
                    ),
                    pose_baseline_m=baseline,
                    pose_angle_deg=angle,
                    gradient_sketch_cosine=_cosine(
                        None if left.gradient is None else left.gradient.gradient_sketch,
                        None
                        if right.gradient is None
                        else right.gradient.gradient_sketch,
                    ),
                )
            )
    return tuple(pairs)


def probe_single_view(
    *,
    gaussians: Any,
    view: CandidateProbeView,
    request: CandidateProbeRequest,
    pipe: Any,
    identity: EvidenceIdentity,
    view_index: int,
    gaussian_count: int,
    lambda_dssim: float,
    render_fn: Any = None,
) -> CandidateProbeResult:
    """Render one candidate and assemble its evidence. Never mutates state.

    Raising is reserved for genuine programming errors; an unusable candidate
    (missing intrinsics, undecodable image, a renderer that refuses the camera)
    comes back as a ``skipped`` result carrying its reason, because a cohort
    must not be destroyed by one bad member.
    """

    # Resolved at CALL time, not bound as a default: this codebase has no CPU
    # rendering path at all, so every CPU-lane test of this function must
    # substitute a fake renderer, and a default-bound reference would make the
    # module-level name unpatchable.
    render = render_fn if render_fn is not None else invoke_render
    identity = replace(
        identity,
        source_id=str(view.source_id),
        source_ordinal=int(view.source_ordinal),
        timestamp_ns=int(view.timestamp_ns),
    )
    synchronize = bool(request.synchronize_device) and torch.cuda.is_available()
    if synchronize:
        torch.cuda.synchronize()
    wall_start = time.perf_counter()
    want_gradient = request.wants(ProbeEvidenceChannel.GRADIENT)
    grad_context = torch.enable_grad() if want_gradient else torch.no_grad()

    try:
        with probe_resolution_binding(
            view.camera,
            request.resolution_tier,
            resampling_mode=request.resampling_mode,
        ):
            with grad_context:
                outputs = render(
                    gaussians,
                    view.camera,
                    pipe,
                    view.background,
                    update_sh_rest=bool(view.update_sh_rest),
                    render_depth=bool(view.render_depth),
                )
                payload = outputs.payload
                render_rgb = payload["render"]
                gt_rgb = view.camera.original_image[:3].to(
                    device=render_rgb.device, dtype=render_rgb.dtype
                )
                height, width = int(render_rgb.shape[-2]), int(render_rgb.shape[-1])

                blocks: dict[str, ScalarEvidence] = {}
                sketch = None
                residual_hw = None
                if request.wants(ProbeEvidenceChannel.RGB) or want_gradient:
                    rgb_block, residual_hw = _rgb_evidence(
                        render_rgb.detach().to(torch.float32),
                        gt_rgb.detach().to(torch.float32),
                        lambda_dssim=lambda_dssim,
                        well_explained_threshold=(
                            request.well_explained_residual_threshold
                        ),
                    )
                    if request.wants(ProbeEvidenceChannel.RGB):
                        blocks.update(rgb_block)
                if request.wants(ProbeEvidenceChannel.DEPTH) and view.render_depth:
                    blocks.update(_depth_evidence(payload, view.camera, height, width))
                visible_mask = None
                visible_count = None
                if request.wants(ProbeEvidenceChannel.VISIBILITY) or request.wants(
                    ProbeEvidenceChannel.VISIBLE_GAUSSIAN_IDS
                ):
                    visibility_block, visible_mask, visible_count = (
                        _visibility_evidence(payload)
                    )
                    if request.wants(ProbeEvidenceChannel.VISIBILITY):
                        blocks.update(visibility_block)
                if (
                    request.wants(ProbeEvidenceChannel.RESIDUAL_SKETCH)
                    and residual_hw is not None
                ):
                    sketch = residual_sketch(
                        residual_hw, int(request.residual_sketch_size)
                    ).cpu()

                gradient = None
                if want_gradient:
                    from eligsir.utils.loss_utils import l1_loss as _l1_fn
                    from eligsir.utils.loss_utils import ssim as _ssim_fn

                    objective = (1.0 - lambda_dssim) * _l1_fn(
                        render_rgb, gt_rgb
                    ) + lambda_dssim * (1.0 - _ssim_fn(render_rgb, gt_rgb))
                    gradient = _gradient_evidence(
                        objective,
                        gaussians,
                        sketch_size=int(request.residual_sketch_size),
                    )
                    blocks["gradient_norm"] = _available(gradient.total_norm)
                    blocks["gradient_energy"] = _available(gradient.total_energy)
                    if gradient.touched_gaussian_fraction is not None:
                        blocks["gradient_touched_gaussian_fraction"] = _available(
                            gradient.touched_gaussian_fraction
                        )
                    for family, key in (
                        ("means", "gradient_energy_means"),
                        ("opacity", "gradient_energy_opacity"),
                        ("scale", "gradient_energy_scale"),
                        ("rotation", "gradient_energy_rotation"),
                        ("color", "gradient_energy_color"),
                    ):
                        if family in gradient.energy_by_family:
                            blocks[key] = _available(
                                gradient.energy_by_family[family]
                            )

                if not request.wants(ProbeEvidenceChannel.VISIBLE_GAUSSIAN_IDS):
                    visible_mask = None
                visible_ids = (
                    visible_mask.nonzero(as_tuple=False).reshape(-1).cpu()
                    if isinstance(visible_mask, torch.Tensor)
                    else None
                )
                if isinstance(visible_mask, torch.Tensor):
                    visible_mask = visible_mask.cpu()
                centre, forward = _camera_geometry(view.camera)
    except Exception as error:  # noqa: BLE001 -- one bad candidate, not a dead cohort
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        reason = f"{type(error).__name__}: {error}"
        return CandidateProbeResult(
            view_index=view_index,
            source_id=view.source_id,
            source_ordinal=view.source_ordinal,
            timestamp_ns=view.timestamp_ns,
            evidence=FrameEvidence(
                identity=identity,
                probe=GaussianProbeEvidence(
                    probe_wall_ms=_available(wall_ms, "ms"),
                ),
                rgb_eligible=False,
                depth_eligible=False,
                hard_reject_reasons=("candidate_probe_failed",),
            ),
            skipped=True,
            skip_reason=reason,
        )

    if synchronize:
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    blocks["probe_pixel_count"] = _available(height * width, "pixels")
    blocks["probe_wall_ms"] = _available(wall_ms, "ms")
    blocks["probe_gpu_ms"] = (
        _available(wall_ms, "ms")
        if synchronize
        else ScalarEvidence.missing(
            EvidenceAvailability.NOT_SUPPORTED,
            unit="ms",
            reason="device timing requires CandidateProbeRequest.synchronize_device",
        )
    )
    return CandidateProbeResult(
        view_index=view_index,
        source_id=view.source_id,
        source_ordinal=view.source_ordinal,
        timestamp_ns=view.timestamp_ns,
        evidence=FrameEvidence(
            identity=identity, probe=GaussianProbeEvidence(**blocks)
        ),
        probe_width=width,
        probe_height=height,
        probe_pixel_count=height * width,
        rasterizer_call_count=1,
        visible_gaussian_mask=visible_mask,
        visible_gaussian_ids=visible_ids,
        visible_gaussian_count=visible_count,
        residual_sketch=sketch,
        gradient=gradient,
        camera_center=centre,
        camera_forward=forward,
    )


__all__ = [
    "CANDIDATE_PROBE_EVIDENCE_SCHEMA_VERSION",
    "CANDIDATE_PROBE_OBJECTIVE_VERSION",
    "DEFAULT_PROBE_CHANNELS",
    "DEFAULT_WELL_EXPLAINED_RESIDUAL",
    "CandidateProbeAccounting",
    "CandidateProbeBatchResult",
    "CandidateProbePairEvidence",
    "CandidateProbeRequest",
    "CandidateProbeResult",
    "CandidateProbeView",
    "ProbeEvidenceChannel",
    "ProbeGradientEvidence",
    "ProbeStateMutationError",
    "ProbeUnsafeContextError",
    "build_pairwise_evidence",
    "canonical_family",
    "capture_state_fingerprint",
    "diff_state_fingerprints",
    "probe_gaussian_row_count",
    "probe_resolution_binding",
    "probe_single_view",
    "residual_sketch",
]
