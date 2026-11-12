# Wave 3 implementation plan: live mapping runtime, evidence-driven admission, and native Gaussian population strategy

**Status:** implementation plan
**Created:** 2026-08-06
**Base:** `main` after merge of PR #37 (`0bc7cf4fa92dd88282fc41fc62b7d90c079b2e2e`)
**Plan branch:** `agent/wave3-live-mapping-pipeline-plan`
**Intended users:** coding agents, subagents, reviewers, experiment owners, and maintainers

---

## 1. Executive decision

Wave 3 shall not begin with another Gaussian filter, another scene-specific configuration, or a large all-in-one live-mapping strategy class.

The implementation order is:

1. establish performance truth and experiment governance;
2. make one Gaussian optimization step a complete typed subsystem;
3. decouple source, SLAM, frame preparation, admission, map commits, and optimization;
4. expose shared mapping evidence without changing behavior;
5. use that evidence for shadow-mode frame utility and realized-gain replay;
6. correct depth semantics with `SURFACE_HIT`, `CERTIFIED_FREE_SPACE`, and `UNKNOWN`;
7. implement a dedicated live Gaussian strategy that proposes typed mutation plans;
8. move to fixed-capacity storage and explicit topology epochs;
9. capture stable optimization epochs with CUDA/HIP graphs;
10. run controlled frozen-variable ablations and retire filters that merely compensate for upstream or data-path defects.

The central runtime invariant is:

> **Source rate and SLAM latency may change the age and availability of candidate observations, but they must never block or pace Gaussian optimization.**

The central research invariant is:

> **Only one causal variable may change in a claimed ablation. Everything else must be frozen, hashed, and reported.**

---

## 2. Program-wide non-negotiable deliverables

These are mandatory across every PR in this plan. A PR is not complete if its local implementation works but violates one of these program rules.

### 2.1 Ownership and dependency rules

1. `GaussianMapBackend` remains the sole owner of live Gaussian parameters, aligned per-Gaussian buffers, optimizer state, map version, and topology version.
2. Fixed-shape numerical updates use `begin_optimization()`.
3. Insert, grow, relocate, reseed, retire, and prune operations use `begin_mutation()` or the eventual backend-owned mutation-session replacement.
4. Mapping policy must not directly write `gaussians.params`, `gaussians.optimizers`, private upstream tensors, or optimizer state.
5. Canonical `eligsir.*` packages must not import root entry points such as `train_streaming.py`.
6. Canonical packages must not depend backward on compatibility modules under `utils.*` or other legacy roots, except through temporary, explicitly allowlisted migration shims.
7. Upstream-derived gsplat rendering and core strategy primitives remain intact behind adapters. Do not split upstream code merely to match EliGSiR package boundaries.

### 2.2 Runtime rules

1. The optimization loop never performs a blocking wait for a sensor frame, SLAM result, decode result, telemetry result, viewer result, or checkpoint write.
2. Every queue is bounded and exposes queue depth, oldest-item age, drops, and drop reason.
3. Bag replay is lossless or invalid. A bag run with source loss must never produce `mapping_result_valid=true`.
4. A live camera may use an explicit freshness policy, but every dropped item and reason must be counted.
5. SLAM failure cannot deadlock Gaussian optimization.
6. Source and SLAM processes must have explicit shutdown and health states.
7. Accepted-keyframe commits may consume a bounded portion of trainer resources; rejected source frames must be almost free from the trainer GPU's perspective.

### 2.3 Experiment and verification rules

1. Every behavior-changing PR ships with a frozen-variable experiment manifest.
2. Every claimed improvement includes a baseline arm and at least one negative control.
3. Full-scene comparisons use matching source artifact hash, pose hash, holdout hash, seed policy, renderer policy, compute budget, and map initialization.
4. A result is not quotable unless `mapping_result_valid=true`.
5. Performance claims use GPU events for GPU work and monotonic wall time for queues and end-to-end latency.
6. A feature is not promoted to a default from one scene alone.
7. Scene-specific switches are considered experimental until cross-scene evidence shows they correspond to a real scene property rather than a hidden implementation defect.
8. Safety fuses remain separately labeled from normal mapping mechanisms and must report trigger counts.

### 2.4 Compatibility rules

1. Current champion configurations and existing CLI flags remain usable throughout Wave 3.
2. New behavior defaults to `off`, `shadow`, or `observe_only` until its promotion gate passes.
3. Old names may remain as deprecated aliases, but live application metrics use mapping terminology:
   - `optimization_steps_hz`, not generic `it/s`;
   - `admitted_keyframes_hz`;
   - `map_commits_hz`;
   - `map_age_ms`;
   - `optimization_step_gpu_ms`.
4. The root streaming entry point may remain as a compatibility composition root while subsystem ownership moves inward.

---

## 3. Golden rule: frozen-variable ablations

This rule shall be added to repository experiment documentation and referenced from all future research plans.

### 3.1 Rule

> When testing a new function, policy, filter, loss, strategy, scheduler, data-path change, or kernel, freeze every variable that is not the object of the experiment.

A comparison that changes frontend poses, keyframes, iteration count, resolution, random seed, map capacity, loss schedule, and filter settings simultaneously is a configuration comparison, not an ablation. It may be operationally useful, but it cannot support a causal claim.

### 3.2 Freeze levels

#### Level A: frozen source

Required for all algorithmic comparisons.

Freeze:

- RGB payloads;
- depth payloads;
- intrinsics;
- timestamps;
- depth scale and unit conversion;
- source frame IDs;
- tracking status;
- SLAM map ID;
- exact poses;
- pose confidence or covariance when available;
- stream-end reason;
- source-loss counters.

Output one deterministic artifact manifest with content hashes. Large payloads do not need to live in Git, but the manifest, creation script, and hashes do.

#### Level B: frozen frame policy

Required when comparing Gaussian population or optimization mechanisms.

Additionally freeze:

- bootstrap frames;
- admitted keyframe IDs;
- keyframe order;
- replay selections, or the replay random stream;
- holdout IDs and guard frames;
- per-frame integration decision;
- integration candidate tensors when the experiment is strictly about post-integration optimization.

#### Level C: frozen map and compute budget

Required for loss, optimizer, strategy, and performance comparisons.

Additionally freeze:

- initial Gaussian checkpoint;
- Gaussian layout and capacity;
- renderer mode;
- resolution schedule;
- SH degree schedule;
- optimizer configuration;
- number of optimization steps;
- wall-time budget when that is the independent variable;
- mutation schedule;
- random seeds and generator states;
- hardware target and software versions.

#### Level D: frozen mutation trace

Required when testing only the numerical optimization step or graph capture.

Additionally freeze:

- topology version sequence;
- mutation plans and exact indices;
- inserted Gaussian values;
- pruned indices;
- reseeded slots;
- optimizer-state resize events.

### 3.3 Required experiment manifest

Every experiment family shall produce a machine-readable manifest containing at least:

```yaml
experiment_id: string
hypothesis: string
independent_variable: string
frozen_level: A | B | C | D
repository_commit: sha
source_artifact_hash: sha256
pose_hash: sha256
holdout_hash: sha256
keyframe_hash: sha256 | null
initial_checkpoint_hash: sha256 | null
mutation_trace_hash: sha256 | null
config_hash: sha256
seed_policy: string
hardware:
  gpu: string
  driver: string
  torch: string
  backend: cuda | rocm
validity:
  mapping_result_valid: bool
  source_frames_dropped: int
  map_reinitializations: int
arms:
  - name: baseline
    changed_fields: []
  - name: treatment
    changed_fields:
      - exact.path.to.changed.setting
```

### 3.4 Enforcement

PR W3-01 shall add:

- a manifest schema;
- a manifest writer;
- a validator;
- a report section listing frozen and changed fields;
- a CI test that rejects an ablation report with missing hashes or more changed fields than declared.

### 3.5 Promotion rule

A feature can move from experimental to default only when:

1. its causal ablation is valid;
2. it improves or preserves the primary metric on at least three materially different scenes;
3. it does not produce a statistically meaningful regression on any required guard scene;
4. performance and memory remain within the declared budget;
5. safety-fuse trigger counts do not indicate routine dependence;
6. its behavior has a typed configuration and an explicit disable path.

---

## 4. Target architecture

```text
SOURCE PROCESS / REPLAY READER
  raw RGB-D + intrinsics + timestamps
                |
                v
SLAM PROCESS
  pose + tracking status + map ID + confidence
                |
                v
FRAME PREPARATION WORKER
  decode / resize / depth semantics / image quality / cheap novelty
                |
                v
BOUNDED PREPARED-CANDIDATE QUEUE
                |
                v
MAPPING COORDINATOR
  non-blocking poll
  cheap policy
  budgeted map-aware admission
  pending commit queue
                |
                +------------------------------+
                |                              |
                v                              v
MUTATION COORDINATOR                 GAUSSIAN OPTIMIZATION ENGINE
  explicit mutation barrier           fixed-topology optimization steps
  typed mutation plans                render -> loss -> backward -> masks -> step
                |                              |
                +---------------+--------------+
                                v
                    GAUSSIAN MAP BACKEND
                sole GPU state and topology owner
                                |
                                v
                    MAPPING EVIDENCE STREAM
             frame utility + per-Gaussian population evidence
                                |
                    +-----------+-----------+
                    |                       |
                    v                       v
          LIVE FRAME UTILITY POLICY   LIVE GAUSSIAN STRATEGY
          admission and replay        population mutation proposals
```

### 4.1 Core scheduling invariant

```python
while runtime.running:
    candidate = prepared_queue.try_pop()  # never blocks
    if candidate is not None:
        admission_controller.consider(candidate)

    if mutation_barrier.due():
        mutation_controller.commit_pending()

    optimization_engine.run_step()
```

If no candidate exists, optimization continues. If SLAM is late, optimization continues. If the source stops, optimization continues until the configured refinement stop policy is reached. Only explicit mutation barriers interrupt stable-topology optimization.

---

## 5. PR map

| PR | Title | Behavior change | Depends on |
|---|---|---:|---|
| W3-01 | Experiment contract and live-pipeline benchmark harness | No | Wave 2 merge |
| W3-02 | Complete Gaussian optimization-step engine | Intended parity only | W3-01 |
| W3-03 | Prepared-candidate pipeline and bounded runtime queues | Yes, behind flag | W3-01, W3-02 |
| W3-04 | SLAM worker isolation and source-mode policies | Yes, behind flag | W3-03 |
| W3-05 | Mapping evidence contracts and observation-only collection | No decision changes | W3-02 |
| W3-06 | Shadow frame utility and realized-gain replay evidence | Shadow only | W3-05 |
| W3-07 | Active evidence-driven admission and mapping budget control | Yes, opt-in | W3-03, W3-06 |
| W3-08 | Tri-state ray semantics and geometry reliability | Yes, opt-in | W3-05 |
| W3-09 | Live Gaussian strategy in observe-only and dry-run mode | No mutations initially | W3-05, W3-08 |
| W3-10 | Active lifecycle, retirement, reseeding, and cell budgets | Yes, opt-in | W3-09 |
| W3-11 | Fixed-capacity map storage and topology epochs | Yes, backend-internal | W3-02, W3-10 |
| W3-12 | CUDA/HIP graph runtime for stable optimization epochs | Performance only | W3-02, W3-11 |
| W3-13 | Frozen ablation campaign, filter retirement, and default promotion | Config/default changes | all relevant PRs |

PRs shall remain separate. Do not merge W3-03 through W3-10 into one branch. The purpose of the sequence is to preserve attribution, reviewability, and rollback.

---

# 6. PR W3-01: experiment contract and live-pipeline benchmark harness

## 6.1 Objective

Create the measurement and reproducibility foundation before changing the hot path.

## 6.2 Why this edit is necessary

Current metrics mix source processing, admission, map mutation, and numerical optimization. Without separate clocks and GPU-event measurements, a refactor can appear faster by doing less work, admitting fewer frames, changing the map size, or moving cost outside the measured interval.

This PR prevents later architecture and strategy work from being evaluated through ambiguous `it/s` or unmatched scene runs.

## 6.3 Proposed files

```text
eligsir/experiments/
  manifest.py
  hashing.py
  validation.py
  frozen_artifact.py
  report.py

eligsir/runtime/profiling/
  events.py
  pipeline_trace.py
  queue_metrics.py
  gpu_timing.py
  benchmark_scenarios.py

scripts/
  create_frozen_input_artifact.py
  benchmark_live_pipeline.py
  validate_experiment_manifest.py

docs/experiments/
  FROZEN_ABLATION_RULE.md
  LIVE_PIPELINE_BENCHMARK.md

tests/
  test_experiment_manifest.py
  test_frozen_ablation_validation.py
  test_pipeline_trace.py
  test_queue_metrics.py
```

## 6.4 Subagent work packages

### Subagent A: experiment schema and hashing

Deliver:

- typed manifest model;
- canonical serialization;
- stable SHA-256 computation;
- config-diff calculation;
- changed-field declaration validation.

Must not touch the optimization loop.

### Subagent B: runtime telemetry

Deliver:

- GPU event wrappers;
- CPU monotonic spans;
- queue metrics;
- per-stage counters;
- JSON export.

Must keep profiling disabled or low-overhead by default.

### Subagent C: synthetic benchmark source

Deliver:

- deterministic 30 FPS source;
- configurable rejection ratio;
- configurable accepted-keyframe rate;
- artificial SLAM delay and failure injection;
- burst mode;
- finite and long-running modes.

### Subagent D: tests and docs

Deliver:

- schema validation tests;
- negative tests for undeclared changed variables;
- benchmark instructions;
- terminology documentation.

### Integration owner

Integrates A-D, confirms no mapping behavior change, and produces the first baseline report on the R9700.

## 6.5 Non-negotiable deliverables

1. One authoritative experiment manifest format.
2. GPU-event timing for optimization, admission, and mutation.
3. Queue depth and oldest-item age metrics.
4. Baseline benchmark JSON for:
   - no source activity;
   - 30 FPS, all rejected;
   - 30 FPS, approximately 2 accepted keyframes/s;
   - 500 ms SLAM delay;
   - SLAM failure;
   - burst input.
5. Golden-rule documentation merged and referenced by later PR templates.
6. No default mapping behavior changes.

## 6.6 Acceptance gates

- profiler-off path shows less than 1% overhead;
- profiler-on baseline is repeatable within the expected stochastic band;
- all benchmark arms record the same frozen source artifact hash where applicable;
- invalid manifests fail loudly;
- run validity is included in every benchmark output.

## 6.7 Out of scope

- moving source processing to workers;
- changing admission;
- changing Gaussian strategy;
- graph capture.

---

# 7. PR W3-02: complete Gaussian optimization-step engine

## 7.1 Objective

Make `GaussianOptimizationEngine` own one complete fixed-topology numerical update.

## 7.2 Why this edit is necessary

The current engine owns only the pre-backward strategy hook and `loss.backward()`. Rendering, loss assembly, gradient masking, optimizer sequencing, learning-rate update, gradient snapshots, and zeroing remain distributed through the root loop. This prevents:

- clean GPU timing;
- graph capture;
- reuse in live, replay, and offline refinement;
- typed evidence production;
- independent unit and parity testing;
- strict separation between optimization and topology mutation.

## 7.3 Target API

```python
@dataclass(frozen=True)
class OptimizationStepRequest:
    view: OptimizationView
    iteration: int
    map_version: int
    topology_version: int
    resolution_bucket: str
    evidence_requirements: EvidenceRequirements
    random_state_token: RandomStateToken


@dataclass
class OptimizationStepResult:
    loss_breakdown: LossBreakdown
    render_outputs: RenderOutputs
    evidence: OptimizationEvidence | None
    optimization_gpu_ms: float | None
    map_version_before: int
    map_version_after: int
    topology_version: int
```

## 7.4 Major edits and rationale

### Move render invocation behind an injected renderer port

Necessary because the engine must own the full step while the upstream gsplat renderer remains intact. The renderer is called through a narrow adapter; it is not internally split.

### Move loss assembly verbatim first

Necessary to preserve numerical parity. Do not redesign the loss stack during extraction.

### Move gradient masks into typed policies

Necessary because the current long inline cascade is part of the numerical step and must be captured, tested, and eventually graphed. Suggested contracts:

```text
GradientPolicy
  apply(context) -> GradientPolicyResult
```

Policies may include stable-geometry freezing, bootstrap anchors, insertion anchors, depth locks, and active masks.

### Centralize optimizer sequencing

Necessary to guarantee one ordering for:

```text
pre-backward hook
backward
gradient policies
energy-gradient snapshot
optimizer step
rotation normalization
LR update
zero_grad
```

### Explicitly exclude topology mutation

Necessary to keep static-shape epochs possible. Insert, prune, grow, relocate, and reseed remain mutation operations.

## 7.5 Proposed files

```text
eligsir/optimization/gaussian/
  engine.py
  contracts.py
  render_port.py
  objective.py
  loss_breakdown.py
  gradient_policies/
    base.py
    stable_geometry.py
    anchors.py
    depth_lock.py
    active_mask.py
  random_state.py

eligsir/optimization/gaussian/tests/
  test_step_parity.py
  test_gradient_policy_order.py
  test_no_topology_mutation.py
```

## 7.6 Subagent work packages

### Subagent A: contracts and result types

Owns new dataclasses and interfaces only.

### Subagent B: render and objective extraction

Moves existing code verbatim. No simplification until parity passes.

### Subagent C: gradient-policy extraction

Moves masks one at a time with exact order tests.

### Subagent D: optimizer-session integration

Routes step, LR update, normalization, and zeroing through the optimization session.

### Subagent E: parity harness

Builds deterministic miniature comparisons and full-GPU checkpoint comparisons.

### Integration owner

Performs the final root-loop replacement and deletes the duplicated inline path only after parity is proven.

## 7.7 Non-negotiable deliverables

1. `run_step()` owns the complete fixed-topology numerical update.
2. No structural mutation is reachable from the engine.
3. Exact ordering is documented and tested.
4. Deterministic miniature parity covers tensors, gradients, optimizer state, and parameters after the step.
5. At least one full-GPU parity run uses the same frozen source, keyframes, checkpoint, seed, and mutation trace.
6. No new host synchronization in the hot step.

## 7.8 Acceptance gates

- deterministic parity at tight numerical tolerance;
- matched full-run PSNR within the known stochastic envelope;
- matched Gaussian-count trajectory when mutation trace is frozen;
- no optimization-step throughput regression above 2%;
- root loop no longer contains a second full optimization implementation.

---

# 8. PR W3-03: prepared-candidate pipeline and bounded runtime queues

## 8.1 Objective

Separate frame arrival and CPU preparation from trainer-owned map-aware admission and optimization.

## 8.2 Why this edit is necessary

The current loop does not wait for a frame, but a ready frame still interrupts optimization with camera construction, visibility projection, surface analysis, admission bookkeeping, and insertion preparation. A 30 FPS source therefore causes periodic trainer stalls even if only a few frames become keyframes.

The target is not zero cost for accepted frames. The target is near-zero trainer cost for rejected frames and bounded, explicit cost for accepted-keyframe commits.

## 8.3 New data contracts

```python
@dataclass(frozen=True)
class RawFramePacket:
    source_index: int
    timestamp_ns: int
    rgb_ref: BufferRef
    depth_ref: BufferRef
    intrinsics: CameraIntrinsics


@dataclass(frozen=True)
class TrackedFramePacket:
    frame: RawFramePacket
    world_T_camera: np.ndarray
    tracking_state: TrackingState
    tracking_confidence: float
    slam_map_id: int


@dataclass(frozen=True)
class PreparedFrameCandidate:
    source_index: int
    timestamp_ns: int
    pose: Pose
    rgb_ref: PreparedImageRef
    depth_ref: PreparedDepthRef
    input_evidence: InputEvidence
    coarse_novelty: float
```

## 8.4 Major edits and rationale

### Add bounded candidate and commit queues

Necessary to prevent source rate from turning into unbounded memory growth and to expose real backlog rather than inferred counters.

### Create a CPU/pinned-memory preparation worker

Necessary to move decode, resize, blur, exposure, valid-depth fraction, and cheap pose novelty off the optimizer thread.

### Split candidate camera geometry from training image materialization

Necessary because admission often needs pose, intrinsics, width, height, and perhaps a low-resolution image, but rejected frames should not allocate a full GPU supervision image.

### Introduce non-blocking queue polling in the coordinator

Necessary to preserve maximum optimization throughput when no candidate is ready.

### Introduce a pending commit queue

Necessary to batch accepted-frame map mutations at explicit barriers rather than interleaving arbitrary insertions with numerical steps.

## 8.5 Proposed files

```text
eligsir/runtime/queues/
  bounded_queue.py
  metrics.py
  policies.py

eligsir/runtime/workers/
  frame_preparation_worker.py

eligsir/mapping/candidates/
  contracts.py
  preparation.py
  prefilter.py

eligsir/mapping/commit_queue.py
```

## 8.6 Subagent work packages

### Subagent A: queue contracts and metrics

Implements bounded queues, close/fail semantics, oldest age, and drop accounting.

### Subagent B: preparation worker

Moves CPU-safe preparation from the trainer thread.

### Subagent C: camera/input split

Extracts direct image decode/preparation so lazy materialization does not instantiate a temporary Camera.

### Subagent D: coordinator integration

Adds non-blocking polling and pending commits behind a feature flag.

### Subagent E: performance acceptance

Runs W3-01 benchmark arms and compares optimization-exclusive GPU time and wall throughput.

## 8.7 Non-negotiable deliverables

1. Every runtime queue is bounded.
2. No queue operation blocks the optimization thread.
3. Rejected candidates do not upload full-resolution training images to the trainer GPU.
4. No temporary Camera is constructed solely to decode an image.
5. Source backlog is measured from actual queues.
6. Feature flag allows immediate fallback to the current path.
7. Synthetic 30 FPS all-rejected arm is within 3% of no-source baseline.

## 8.8 Acceptance gates

- zero unbounded queues in the new path;
- no deadlock under producer failure or consumer shutdown;
- long-duration RSS reaches a plateau;
- accepted-frame path preserves reconstruction parity;
- all-rejected path performs no full-map admission render unless explicitly configured.

---

# 9. PR W3-04: SLAM worker isolation and source-mode policies

## 9.1 Objective

Ensure every SLAM backend is logically asynchronous and define different integrity policies for bag replay and live sensing.

## 9.2 Why this edit is necessary

External ORB-SLAM is already process-separated, but other backends can still execute `track()` inside the mapping loop. Any such backend pauses optimization. In addition, replay and live cameras have different correctness requirements: replay should preserve every frame, while live sensing may prefer freshness.

## 9.3 Major edits and rationale

### Add `SLAMWorker` protocol

Necessary to provide one asynchronous interface for external and in-process backends.

```python
class SLAMWorker(Protocol):
    def submit(self, frame: RawFramePacket) -> SubmitResult: ...
    def try_get(self) -> TrackedFramePacket | None: ...
    def health(self) -> WorkerHealth: ...
    def close(self) -> None: ...
```

### Wrap in-process backends in worker processes

Necessary so backend latency cannot block optimization.

### Add bag-replay source policy

Necessary to guarantee lossless replay or an invalid run. Backpressure, durable bag references, or spill are acceptable; silent drop is not.

### Add live freshness policy

Necessary to allow bounded operation under overload while recording exactly what was dropped and why.

### Add resource-isolation documentation and launch support

Necessary because process separation alone does not prevent CPU or GPU contention. Add cpuset/quota configuration and clearly state that hard isolation requires dedicated resources.

## 9.4 Proposed files

```text
eligsir/slam/runtime/
  worker.py
  process_worker.py
  external_worker.py
  health.py

eligsir/io/policies/
  bag_replay.py
  live_freshness.py

ros2_ws/... launch changes
scripts/benchmark_slam_isolation.py
```

## 9.5 Subagent work packages

### Subagent A: worker protocol and process implementation

### Subagent B: backend adapters

Wraps current SLAM backends without changing their algorithms.

### Subagent C: replay/live source policies

Implements lossless versus freshness semantics.

### Subagent D: failure and overload tests

Covers delay, crash, restart policy, queue saturation, and shutdown.

### Subagent E: launch/resource isolation

Adds cpusets, quotas, and documentation.

## 9.6 Non-negotiable deliverables

1. No production SLAM backend executes synchronously in the optimization thread.
2. A 500 ms artificial SLAM delay does not change optimization-exclusive GPU time.
3. SLAM failure cannot deadlock the mapper.
4. Bag frame loss forces `mapping_result_valid=false`.
5. Live drops are reason-coded and counted.
6. Resource contention assumptions are explicit; no false claim of absolute isolation on a shared GPU.

---

# 10. PR W3-05: mapping evidence contracts and observation-only collection

## 10.1 Objective

Create one shared evidence substrate for admission, replay, population policy, diagnostics, and future learning without changing decisions.

## 10.2 Why this edit is necessary

The mapper already computes visibility, residuals, gradients, support, depth conflicts, opacity, scale, age, and contribution-like signals in several places. Independent recomputation creates overhead and causes admission, replay, lifecycle, and MCMC to disagree about what constitutes useful information.

An observation-only first step prevents behavior changes from being mixed with the evidence refactor.

## 10.3 Evidence types

```python
@dataclass(frozen=True)
class InputEvidence:
    pose_confidence: float
    image_quality: float
    depth_valid_fraction: float
    geometry_reliability_mean: float


@dataclass
class RenderEvidence:
    rgb_residual: torch.Tensor
    depth_residual: torch.Tensor | None
    alpha: torch.Tensor
    rendered_depth: torch.Tensor | None
    transmittance: torch.Tensor | None


@dataclass
class GaussianEvidence:
    visibility: torch.Tensor
    projected_radius: torch.Tensor
    contribution: torch.Tensor
    xyz_grad_ema: torch.Tensor
    opacity_grad_ema: torch.Tensor
    support: torch.Tensor
    geometry_confidence: torch.Tensor
    free_space_conflict: torch.Tensor
    age: torch.Tensor


@dataclass(frozen=True)
class EvidenceSummary:
    new_surface_ratio: float
    persistent_residual: float
    under_supported_mass: float
    expected_gradient_value: float
    predicted_cost_ms: float
```

All evidence carries frame ID, map version, topology version, and resolution bucket.

## 10.4 Major edits and rationale

### Add `EvidenceRequirements`

Necessary to avoid computing expensive diagnostic tensors when no active policy needs them.

### Add reducers and EMAs

Necessary to turn transient per-pixel and per-Gaussian signals into stable online evidence.

### Add topology-version validity

Necessary because per-Gaussian evidence indices become stale after structural mutation.

### Add serialization and shadow logs

Necessary for offline analysis and frozen replay of policy decisions.

## 10.5 Proposed files

```text
eligsir/mapping/evidence/
  contracts.py
  requirements.py
  builder.py
  reducers.py
  gaussian_history.py
  frame_history.py
  serialization.py
```

## 10.6 Subagent work packages

### Subagent A: evidence contracts and versioning

### Subagent B: render/gradient evidence extraction

### Subagent C: history and reducers

### Subagent D: serialization and analysis tooling

### Subagent E: overhead benchmarking

## 10.7 Non-negotiable deliverables

1. No mapping decision changes.
2. Evidence generation is demand-driven.
3. Stale topology-indexed evidence is rejected.
4. Evidence overhead is separately measured.
5. Existing duplicate computations are documented for later removal.
6. Frozen evidence logs can be replayed into policy code without rerunning rendering.

---

# 11. PR W3-06: shadow frame utility and realized-gain replay evidence

## 11.1 Objective

Evaluate map-aware frame utility without allowing it to change admission yet.

## 11.2 Why this edit is necessary

Motion thresholds and overlap are weak proxies for mapping value. A frame can be geometrically novel but blurred, redundant, specular, badly tracked, or expensive. Conversely, a small viewpoint change may expose an under-supported region. The mapper should exploit residuals, support debt, historical gradients, new surface, and cost estimates already available from the pipeline.

Shadow mode is mandatory before active use so score quality can be evaluated independently of the map trajectory it would create.

## 11.3 Utility model

Initial deterministic score:

```text
U = pose_quality * image_quality *
    (w_new_surface * new_surface
     + w_residual * persistent_depth_supported_residual
     + w_debt * under_supported_mass
     + w_gradient * expected_gradient_value
     + w_parallax * useful_parallax)
    - w_redundancy * redundancy
    - w_specular * specular_risk
    - w_cost * predicted_gpu_cost
```

Do not use raw RGB residual alone.

Expected gradient value shall initially use historical per-Gaussian gradient EMA and visibility. It shall not run a backward pass for every candidate.

## 11.4 Realized gain

For admitted frames, record:

- loss before future local optimization;
- loss after a defined number of optimization opportunities;
- useful gradient mass generated;
- under-supported cells improved;
- depth consistency improvement;
- compute consumed.

The realized-gain label is used to evaluate and later calibrate the predicted utility score.

## 11.5 Proposed files

```text
eligsir/mapping/admission/
  utility_contracts.py
  shadow_policy.py
  scoring.py
  cost_model.py
  realized_gain.py

scripts/analyze_frame_utility.py
```

## 11.6 Subagent work packages

### Subagent A: utility feature extraction

### Subagent B: deterministic scorer

### Subagent C: realized-gain tracker

### Subagent D: analysis and calibration reports

### Subagent E: shadow integration

## 11.7 Non-negotiable deliverables

1. Shadow mode does not alter keyframe decisions, replay, optimization, or mutation.
2. Every score component is logged separately.
3. Predictions are compared against realized gain.
4. Scores are evaluated on at least floor2, floor3, kitchen1, one TUM scene, and one Replica scene.
5. Policy cost is measured.
6. A score dominated by blur, pose errors, or glass residuals must fail promotion.

---

# 12. PR W3-07: active evidence-driven admission and mapping budget control

## 12.1 Objective

Allow validated frame utility to control candidate admission and optimization budget under an opt-in policy.

## 12.2 Why this edit is necessary

The system needs to allocate limited map commits and optimization opportunities to observations that yield the greatest map improvement. Admission and optimization budget should be related but distinct: a frame may deserve storage yet not immediate heavy refinement, or it may deserve a short local burst because it exposes a weak region.

## 12.3 Rollout modes

1. `shadow`: no decision changes.
2. `advisory`: new policy may reject current-policy admissions but may not add extra frames.
3. `active`: utility policy owns admission within hard safety constraints.
4. `active_with_budget`: policy assigns local optimization and replay budget.

## 12.4 Major edits and rationale

### Add `AdmissionDecision`

Necessary to replace boolean and dictionary results with typed reasons, score components, and predicted cost.

### Add `MappingBudgetRequest`

Necessary to separate frame admission from how much optimization it receives.

### Add budget controller

Necessary to keep map age and queue backlog bounded while preserving optimization throughput.

### Preserve hard safety gates

Tracking validity, map ID, timestamp integrity, and source validity remain non-negotiable and are not overridden by utility.

## 12.5 Subagent work packages

### Subagent A: decision and budget contracts

### Subagent B: advisory integration

### Subagent C: active policy and fallback

### Subagent D: budget controller

### Subagent E: frozen frame-policy ablation

## 12.6 Non-negotiable deliverables

1. Current policy remains available as fallback.
2. All decisions include machine-readable reasons.
3. Active policy operates under explicit frame-count and compute budgets.
4. Frozen-Gaussian-policy ablation compares current, motion/overlap, new-surface, persistent-residual, gradient/debt, and combined utility.
5. No performance regression beyond the declared budget.
6. No active promotion without cross-scene evidence.

---

# 13. PR W3-08: tri-state ray semantics and geometry reliability

## 13.1 Objective

Replace the unsafe assumption that invalid depth implies free space.

## 13.2 Why this edit is necessary

Invalid RGB-D measurements may mean glass, specular failure, low return, occlusion boundary, beyond range, or missing data. Treating all invalid pixels as free space can remove correct geometry and make scene-specific carving filters appear necessary.

The Gaussian strategy must receive physically meaningful geometry evidence before it can make principled retire, prune, or reseed decisions.

## 13.3 Ray evidence model

```python
class RayEvidenceKind(Enum):
    SURFACE_HIT = auto()
    CERTIFIED_FREE_SPACE = auto()
    UNKNOWN = auto()
```

### `SURFACE_HIT`

Valid depth with sufficient sensor and pose confidence.

Allows:

- surface insertion;
- depth supervision;
- geometry support;
- free-space evidence only along the ray segment before the hit.

### `CERTIFIED_FREE_SPACE`

Requires explicit evidence such as:

- valid ray traversal before a measured hit;
- documented sensor max-range return semantics;
- repeated multi-view agreement;
- adequate baseline and pose confidence.

### `UNKNOWN`

Includes ordinary invalid depth and uncertain transmissive/specular regions.

Rules:

- no geometry insertion;
- no free-space carving;
- no depth-geometry gradients;
- optional cautious appearance supervision.

## 13.4 Geometry reliability

Add a `[0,1]` reliability map based initially on:

- depth validity;
- local depth discontinuity;
- temporal RGB consistency;
- temporal depth consistency;
- saturation and extreme intensity;
- view-dependent appearance change;
- multi-view support;
- pose confidence.

Use it for depth loss weighting, insertion confidence, support updates, and free-space evidence.

## 13.5 Appearance-only handling

For uncertain/specular regions, optionally run an auxiliary low-frequency, reduced-resolution appearance update with geometry tensors detached. Do not add a second full-resolution render to every step.

## 13.6 Subagent work packages

### Subagent A: contracts and sensor semantics

### Subagent B: classification implementation

### Subagent C: loss/insertion integration

### Subagent D: material-risk heuristics and appearance-only path

### Subagent E: pixel-level evaluation masks and reports

## 13.7 Non-negotiable deliverables

1. Invalid depth is never automatically certified free space.
2. Classification reason is inspectable and counted.
3. Current carving remains available only as an explicit legacy ablation.
4. Frozen-keyframe Gaussian-policy ablation includes tri-state semantics.
5. Glass/specular evaluation uses pixel masks, not only trajectory index ranges.
6. Geometry and appearance supervision can be weighted independently.

---

# 14. PR W3-09: live Gaussian strategy in observe-only and dry-run mode

## 14.1 Objective

Introduce `EliGSiRLiveGaussianStrategy` as a gsplat-compatible population policy that initially proposes actions without mutating the map.

## 14.2 Why this edit is necessary

The current mapper surrounds an offline-oriented MCMC strategy with reach recycling, floater consensus, KNN isolation, scale kills, anisotropy kills, depth locks, lifecycle freezing, and other corrections. Some are valid safety mechanisms; others compensate for population dynamics that were not designed for an expanding, partially observed live map.

The new strategy must consolidate evidence into a coherent policy instead of adding another independent gate.

## 14.3 Package layout

```text
eligsir/optimization/gaussian/strategies/
  contracts.py
  registry.py
  gsplat_mcmc_adapter.py
  live/
    strategy.py
    state.py
    scoring.py
    lifecycle.py
    cell_budget.py
    mutation_plan.py
```

## 14.4 Strategy boundary

The strategy proposes a typed plan:

```python
@dataclass
class PopulationMutationPlan:
    retire_indices: torch.Tensor
    prune_indices: torch.Tensor
    reseed_requests: tuple[SurfaceSeedRequest, ...]
    growth_requests: tuple[SurfaceSeedRequest, ...]
    lifecycle_updates: LifecycleUpdateBatch
    reasons: MutationReasonSummary
```

It does not directly resize or replace model tensors.

## 14.5 Per-Gaussian state

At minimum:

- geometry lifecycle state;
- appearance lifecycle state;
- support count;
- contribution EMA;
- RGB residual EMA;
- depth residual EMA;
- position-gradient EMA;
- opacity-gradient EMA;
- free-space-conflict EMA;
- geometry confidence;
- last visible step;
- birth frame;
- source surface cell.

## 14.6 Lifecycle model

```text
PROVISIONAL
  -> ACTIVE after sufficient multi-view support
  -> SUSPECT after persistent conflict or non-contribution
ACTIVE
  -> STABLE after geometry convergence
  -> SUSPECT after persistent conflict
STABLE
  -> SUSPECT only under strong evidence
SUSPECT
  -> ACTIVE if evidence recovers
  -> RETIRED if confirmed unhelpful
RETIRED
  -> RESEEDED when a high-utility surface request exists
```

Geometry and appearance maturity are separate.

## 14.7 Observe-only output

The strategy logs:

- proposed retire/prune/reseed/grow counts;
- reasons;
- overlap with current MCMC actions;
- predicted affected cells;
- expected contribution change;
- safety-fuse trigger counts.

No map mutation occurs.

## 14.8 Subagent work packages

### Subagent A: strategy contracts and registry migration

### Subagent B: lifecycle state and aligned buffers

### Subagent C: scoring and cell budgets

### Subagent D: mutation-plan generation

### Subagent E: dry-run comparison and reports

## 14.9 Non-negotiable deliverables

1. Observe-only mode is behavior-neutral.
2. Strategy code lives in canonical packages; `utils.strategies` becomes compatibility construction only.
3. Mutation plans are typed and versioned.
4. Every proposed action has a reason.
5. Stable Gaussians are not proposed for global teleportation under ordinary conditions.
6. Safety fuses remain separate from normal strategy scoring.

---

# 15. PR W3-10: active lifecycle, retirement, reseeding, and cell budgets

## 15.1 Objective

Activate the validated live strategy incrementally.

## 15.2 Why this edit is necessary

The current indirect recycling sequence lowers opacity, waits for dead-mask selection, and relocates based on MCMC parent sampling. This obscures intent and can repeatedly misallocate population capacity. Explicit retirement and reseeding make the population transition testable and controllable.

## 15.3 Activation phases inside the PR

These may be separate commits or temporary feature modes, but must be evaluated independently.

### Phase A: lifecycle updates only

No retire/prune/reseed. Verify state transitions and recovery.

### Phase B: retirement only

Retired rows become inactive but are not reused. Verify quality and counts.

### Phase C: explicit reseeding

Reuse retired slots for high-confidence surface requests.

### Phase D: cell budgets

Allocate minimum, normal, temporary, and reserve budgets per surface/spatial cell.

### Phase E: constrained movement

Limit supported geometry motion according to depth uncertainty and tangent-plane confidence. Unrestricted MCMC remains limited to provisional geometry.

## 15.4 Major edits and rationale

### Backend-applied mutation plans

Necessary to preserve GPU ownership and centralized aligned-buffer updates.

### Explicit slot inheritance/reset rules

Necessary so reseeded Gaussians do not inherit stale optimizer moments, support, or lifecycle fields accidentally.

### Per-cell population accounting

Necessary to prevent saturated regions from consuming the same growth budget as newly observed or under-reconstructed cells.

## 15.5 Subagent work packages

### Subagent A: lifecycle application

### Subagent B: retire/reseed backend operations

### Subagent C: cell-budget controller

### Subagent D: constrained-motion policy

### Subagent E: frozen-keyframe population-policy ablations

## 15.6 Non-negotiable deliverables

1. Every aligned per-Gaussian buffer is updated through one backend-owned registry.
2. Reseeded slots have explicit optimizer-state reset semantics.
3. Topology/map versions update correctly.
4. Current strategy remains available.
5. Frozen-keyframe ablations compare minimal MCMC, current champion, reach recycle, conflict consensus, safety guards, tri-state semantics, and live strategy.
6. Hard safety kills report trigger counts and are not counted as normal strategy wins.

---

# 16. PR W3-11: fixed-capacity map storage and topology epochs

## 16.1 Objective

Replace frequent `torch.cat` growth and parameter replacement with reusable capacity and explicit mutation epochs.

## 16.2 Why this edit is necessary

Dynamic concatenation and optimizer-state resizing cause allocator churn, transient memory spikes, pointer instability, and graph invalidation. The live strategy's explicit retired slots make fixed-capacity reuse practical.

## 16.3 Storage model

```text
capacity: allocated rows
active_count or active_mask: live rows
free_slots: reusable retired rows
logical_ids: stable identity separate from physical row
```

The initial implementation may use an active mask rather than requiring a contiguous live prefix if that better matches gsplat constraints.

## 16.4 Topology epochs

```text
mutation barrier
  -> apply batched insert/reseed/prune/retire
  -> rebuild required indices
  -> increment topology version if structural meaning changed
  -> run K stable optimization steps
  -> next mutation barrier
```

During a stable epoch:

- no parameter object replacement;
- no optimizer-state resize;
- no storage reallocation;
- fixed resolution bucket;
- fixed loss topology;
- stable control flow.

## 16.5 Major edits and rationale

### Capacity allocator

Necessary to centralize free-slot reuse and growth policy.

### Aligned-buffer registry

Necessary to eliminate manual lists of buffers in every mutation path.

### Logical-to-physical identity

Necessary so evidence and lifecycle history survive slot movement or compaction safely.

### Mutation barrier API

Necessary for deterministic batching, graph invalidation, and performance accounting.

## 16.6 Subagent work packages

### Subagent A: allocator and slot model

### Subagent B: aligned-buffer registry

### Subagent C: optimizer-state slot operations

### Subagent D: mutation barrier and versioning

### Subagent E: parity, memory, and fragmentation benchmarks

## 16.7 Non-negotiable deliverables

1. Normal insertion and reseeding do not call `torch.cat` when free capacity exists.
2. Per-Gaussian buffers cannot silently become length-misaligned.
3. Optimizer-state semantics are tested for insert, reuse, retire, prune, and restore.
4. Checkpoints round-trip capacity, active state, logical IDs, and versions.
5. Peak memory and mutation latency are reported against the dynamic baseline.
6. A compatibility mode remains until full scene parity passes.

---

# 17. PR W3-12: CUDA/HIP graph runtime for stable optimization epochs

## 17.1 Objective

Capture and replay the fixed-topology optimization step on CUDA and ROCm after the engine and storage model are stable.

## 17.2 Why this edit is necessary

Existing profiles show significant launch/Python overhead relative to render and backward time. Graph replay can remove repeated host launch overhead, but only after shapes, pointers, and control flow are stable.

Graph capture before W3-02 and W3-11 would either fail or require fragile recapture on every topology change.

## 17.3 Graph scope

Capture:

- copy selected view tensors into static input buffers;
- render when the selected backend path is graph-safe;
- loss assembly;
- backward;
- tensor-only gradient policies;
- optimizer step;
- zeroing.

Do not capture:

- source handling;
- SLAM;
- queue operations;
- admission;
- insertion;
- growth;
- retirement;
- reseeding;
- pruning;
- checkpointing;
- viewer export.

## 17.4 Runtime abstraction

```text
GraphRuntime
  EagerRuntime
  TorchAcceleratorGraphRuntime
  NativeHipGraphRuntime (later, only if required)
  NativeCudaGraphRuntime (later, only if required)
```

Start with PyTorch's graph interfaces and runtime capability probing. Do not assume identical support merely because ROCm uses `torch.cuda` APIs.

## 17.5 Graph signature

At minimum:

```text
device backend and architecture
capacity bucket
active-count or active-mask shape
resolution bucket
SH degree
renderer mode
packed/unpacked mode
sparse-gradient mode
optimizer mode
loss-policy ID
gradient-policy ID
```

## 17.6 Subagent work packages

### Subagent A: graph capability probe and runtime interface

### Subagent B: static input/output buffers

### Subagent C: graph-safe optimizer step

### Subagent D: renderer capture investigation

### Subagent E: benchmark and fallback testing

## 17.7 Non-negotiable deliverables

1. Eager fallback is always available.
2. Graph failure cannot corrupt the map; it falls back before applying a partial step.
3. Randomness is explicit and reproducible.
4. Capture invalidation rules are documented and tested.
5. CUDA and ROCm are benchmarked separately.
6. Graph mode must produce parity within the declared numerical tolerance.
7. The PR reports launch count, GPU duty cycle, wall optimization steps/s, capture cost, and amortization point.

---

# 18. PR W3-13: frozen ablation campaign, filter retirement, and default promotion

## 18.1 Objective

Use the completed infrastructure to decide which current filters are real safety mechanisms, which are useful mapping mechanisms, and which compensate for incorrect data semantics or population behavior.

## 18.2 Why this edit is necessary

The repository currently contains invalid-depth carving, reach pruning/recycling, KNN isolation kills, scale/anisotropy kills, depth locking, floater consensus, lifecycle freezing, and several count/noise schedules. Keeping all of them enabled obscures causality and increases configuration fragility.

This PR is primarily experiment manifests, configs, reports, and carefully justified default changes. It must not become another algorithm-development PR.

## 18.3 Frozen input campaign

Create deterministic artifacts for at least:

- floor2;
- floor3;
- kitchen1;
- one TUM RGB-D sequence;
- one Replica scene.

Each artifact includes exact RGB-D, poses, status, map IDs, timestamps, and validity metadata.

## 18.4 Population-policy ablation

Hold keyframes, compute budget, renderer, initialization, and source fixed.

| Arm | Population policy |
|---|---|
| G0 | Minimal energy MCMC; corrective filters off |
| G1 | Current scene champion policy |
| G2 | G0 + reach recycling only |
| G3 | G0 + multi-view conflict/consensus only |
| G4 | G0 + scale/anisotropy/KNN safety controls |
| G5 | G0 + tri-state ray semantics |
| G6 | Live strategy lifecycle/retire/reseed |
| G7 | G6 + geometry-reliability/specular gating |

## 18.5 Frame-policy ablation

Hold Gaussian policy and compute budget fixed.

| Arm | Frame policy |
|---|---|
| F0 | Current hybrid policy |
| F1 | Motion and overlap only |
| F2 | New-surface utility |
| F3 | Persistent depth-supported residual |
| F4 | Gradient/debt utility |
| F5 | Combined expected map improvement |
| F6 | F5 + realized-gain replay |

## 18.6 Metrics

Required:

- mean, minimum, and partition PSNR;
- SSIM and LPIPS;
- valid-depth RMSE;
- off-surface opacity-weighted mass;
- certified-free-space violation mass;
- large-scale fraction;
- needle fraction;
- Gaussian count over time;
- lifecycle counts;
- retired/reseeded counts;
- trigger counts for every safety fuse;
- optimization steps/s;
- optimization GPU time;
- admission and commit cost;
- RSS and VRAM plateau;
- queue depths and drops;
- map age and commit latency.

## 18.7 Filter classification

### Safety fuse

A mechanism may remain as a safety fuse when:

- it triggers rarely;
- it prevents catastrophic values or NaNs;
- ordinary quality does not depend on routine activation;
- its trigger is reported.

### Mapping mechanism

A mechanism may remain active when:

- its causal ablation shows repeatable benefit across scenes;
- it corresponds to a well-defined physical or optimization property;
- it does not conceal invalid data semantics;
- it has bounded cost.

### Duct-tape candidate

A mechanism should be deprecated when:

- it triggers continually in normal data;
- it helps one scene and harms another without a clear scene property;
- tri-state semantics or the live strategy make it unnecessary;
- it improves appearance while degrading geometry;
- it compensates for a known source, pose, or synchronization defect.

## 18.8 Subagent work packages

### Subagent A: artifact generation and validation

### Subagent B: population-policy experiment matrix

### Subagent C: frame-policy experiment matrix

### Subagent D: geometry/material-specific metrics

### Subagent E: statistical analysis and promotion report

### Integration owner

Makes only default/config changes justified by the report and records removal milestones for deprecated filters.

## 18.9 Non-negotiable deliverables

1. Every result uses the golden frozen-variable rule.
2. Invalid runs are excluded automatically.
3. No default promotion from one scene.
4. No claim based solely on mean PSNR.
5. Each retained filter is classified as safety fuse or mapping mechanism.
6. Each deprecated filter has a replacement, deprecation notice, and removal condition.
7. Champion configs remain archived and reproducible.

---

## 19. Subagent execution protocol

Each implementation PR should use parallel subagents only where file ownership is disjoint.

### 19.1 Recommended roles

- **Contracts agent:** types, protocols, schemas, no hot-path implementation.
- **Implementation agent:** one subsystem only.
- **Test agent:** parity, failures, property tests, and fixtures.
- **Benchmark agent:** frozen manifests, performance runs, and reports.
- **Integration agent:** composition root, migration shims, final deletion of old path.
- **Documentation agent:** architecture, operational guide, and deprecation updates.

### 19.2 Parallelism rule

Subagents must not concurrently edit the same high-conflict files, especially:

- `train_streaming.py`;
- `scene/streaming_scene.py`;
- `arguments/__init__.py`;
- map backend adapters;
- strategy construction code.

The integration agent owns those files after subsystem agents finish.

### 19.3 Required handoff from every subagent

Each subagent reports:

1. files changed;
2. invariants preserved;
3. tests added and run;
4. behavior flags added;
5. known limitations;
6. exact integration steps;
7. benchmark impact if relevant.

### 19.4 Commit discipline

Prefer commits in this order:

1. contracts and tests;
2. behavior-neutral extraction;
3. integration behind flag;
4. parity evidence;
5. deletion of duplicate path;
6. docs and reports.

Do not combine a refactor, a pose fix, a new mapping policy, and a scene-specific configuration change in one commit.

---

## 20. Cross-PR acceptance matrix

| Capability | W3-01 | W3-02 | W3-03 | W3-04 | W3-05 | W3-06 | W3-07 | W3-08 | W3-09 | W3-10 | W3-11 | W3-12 | W3-13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen experiment manifests | required | required | required | required | required | required | required | required | required | required | required | required | required |
| Optimization step parity | baseline | pass | guard | guard | guard | guard | guard | guard | guard | guard | guard | strict | guard |
| Source/SLAM non-blocking | measure | unchanged | implement | complete | guard | guard | guard | guard | guard | guard | guard | guard | validate |
| Mapping decision unchanged | yes | yes | flag off | flag off | yes | yes | advisory/shadow modes | flag off | yes | flag off | backend parity | yes | justified changes |
| Map ownership enforced | guard | improve | guard | guard | guard | guard | guard | guard | plan only | active plans | complete storage | complete | guard |
| Full-scene regression | baseline | required | required | required | required | required | required | required | required | required | required | required | campaign |
| Performance budget | establish | <=2% regression | all-reject <=3% | delay independent | evidence cost reported | score cost reported | <= declared budget | cost reported | observe cost | cost/quality tradeoff | memory win | speed win | final |

---

## 21. Definition of done for Wave 3

Wave 3 is complete only when all of the following hold:

1. Gaussian optimization continues at maximum available rate on already buffered keyframes independently of source and SLAM arrival.
2. Rejected 30 FPS input remains within approximately 3% of no-source optimization throughput.
3. Accepted-keyframe costs are explicit, bounded, and separately measured.
4. Every SLAM backend is asynchronous relative to optimization.
5. `GaussianOptimizationEngine` owns the complete fixed-topology numerical step.
6. Admission, replay, and Gaussian population policy consume shared versioned evidence.
7. Invalid depth is not treated as free space without certification.
8. The live strategy proposes and applies explicit lifecycle and mutation plans through the map backend.
9. Normal population updates reuse capacity rather than repeatedly concatenating tensors.
10. Stable optimization epochs can run under CUDA/HIP graph replay with eager fallback.
11. Current floater/plume/glass filters have been causally ablated under frozen inputs and classified.
12. Defaults reflect cross-scene evidence, not scene-specific rescue configurations.
13. All promoted results are reproducible from committed manifests and scripts.

---

## 22. Immediate starting point

The first implementation branch should be:

```text
agent/w3-01-live-pipeline-performance-contract
```

Its scope is only W3-01. It must not change admission, Gaussian strategy, depth semantics, or source scheduling.

After W3-01 is merged, begin W3-02. W3-03 and W3-05 may then proceed partly in parallel because one targets runtime preparation/queues and the other targets evidence contracts, but both must integrate against the complete optimization-step boundary from W3-02.

The new live Gaussian strategy begins only after W3-05 and W3-08 provide trustworthy evidence and sensor semantics.
