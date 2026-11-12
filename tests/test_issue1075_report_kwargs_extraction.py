"""Issue #1075 (task P3): the streaming report-kwargs builder, lifted out of
``train_streaming.streaming_training()``'s closure into
``eligsir/runtime/report_kwargs.py``.

Before the lift, ``_streaming_report_kwargs()`` was a ~280-line closure that
reached ~35 of the training loop's locals implicitly, through its closure
cell. That is the exact shape issue #729 hit in production: one of those free
variables (``_git_provenance_from_args``) was bound only inside a conditional,
so every mid-training report raised ``NameError`` and ``_run_mid_report``
swallowed it -- reports silently vanished for 26 run directories.

The lift makes each of those captures a named keyword parameter. What was an
invisible free variable is now a required keyword argument, so the #729 shape
fails loudly at the call site (``TypeError``) instead of quietly at report
time. These tests pin that property and the behavior the lift had to preserve.

Two halves:

* **Behavioral** (needs torch, because the builder module transitively imports
  it): call the extracted function against fully faked collaborators and assert
  the emitted key set, the ``final`` depth-insertion gate (#303), the
  three tri-state "telemetry disabled" branches (#399 gpu_memory, #562
  duty_cycle, #601 backpressure), and both documented side effects
  (``register_supervision_source`` per train camera, and #951's
  ``set_partition_control_roles``).
* **Static** (torch-free, so it is reachable on the CPU-only CI image -- see
  ``test_issue646_report_kwargs_signature.py``'s docstring for why that
  matters): the ``train_streaming.py`` wrapper must pass EVERY parameter the
  extracted function declares. A parameter silently left to its default would
  reintroduce the #729 class from the other direction -- the builder would run
  with stale/absent state and emit a wrong-but-present report block, which is
  worse than a crash.

Equivalence with the pre-lift closure was verified once, out-of-band, by
exec'ing the original closure body (git ``e4892a6c``) against the same fakes
used below and diffing the two returned dicts: identical for all five
scenarios exercised here. That harness is deliberately NOT committed -- it
would pin a copy of the old body that bit-rots on the first legitimate report
change. These tests pin the CONTRACT, which does not.
"""

import ast
import dataclasses
import pathlib
from types import SimpleNamespace

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_STREAMING = REPO_ROOT / "train_streaming.py"
REPORT_KWARGS = REPO_ROOT / "eligsir" / "runtime" / "report_kwargs.py"
BUILDER_NAME = "build_streaming_report_kwargs"
WRAPPER_NAME = "_streaming_report_kwargs"


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _FourClocks:
    optimization_step_count: int = 1
    mapping_step_count: float = 2.5


class _ReportBlockOwner:
    """Stands in for every collaborator whose only use here is
    ``.report_block()`` (seed-fade monitor, memory tracker, duty-cycle
    timeline, backpressure owner)."""

    def __init__(self, tag):
        self.tag = tag

    def report_block(self):
        return {"tag": self.tag}


class _Roles:
    def __init__(self):
        self.value = "roles-obj"


class _PartitionControlRuntime:
    def __init__(self):
        self.prepared = []
        self.recorder = SimpleNamespace(
            report_block=lambda roles: {"roles": roles.value}
        )

    def prepare(self, train_cams, test_cams):
        self.prepared.append((list(train_cams), list(test_cams)))
        return _Roles(), ["unused-control-cams"]


def _stub_module_helpers(monkeypatch, module):
    """The builder calls six helpers it imports at module level. Replace them
    with identity-ish stubs so this test measures the BUILDER's assembly, not
    those helpers' own (separately tested) internals."""
    monkeypatch.setattr(
        module, "realized_cadence_report_block",
        lambda cadence, **kw: {"rc": cadence.last_iteration, **kw})
    monkeypatch.setattr(
        module, "admission_cadence_report_block",
        lambda ac, **kw: {"ac": ac.name, "nested": sorted(kw)})
    monkeypatch.setattr(
        module, "resolution_report_block", lambda prov, tier: {"res": [prov, tier]})
    monkeypatch.setattr(
        module, "supervision_work_report_block",
        lambda *a, **kw: {"sw": list(a), **kw})
    monkeypatch.setattr(
        module, "pose_provenance_report_dict", lambda args: {"pose": args.model_path})
    monkeypatch.setattr(
        module, "execution_semantics_from_args", lambda args: {"es": args.model_path})
    # Imported inside the function body, so patch it at its source.
    import eligsir.utils.experiment_origin as eo

    monkeypatch.setattr(eo, "experiment_origin_from_args",
                        lambda args: {"origin": args.model_path})


def _make_kwargs(*, mem_tracker_on=True, duty_on=True, backpressure_on=True,
                 lineage=None, replica=None):
    args = SimpleNamespace(
        streaming_frame_admission="all",
        streaming_memory_telemetry=True,
        streaming_report_video_min_width=1920,
        streaming_report_video_max_width=3840,
        streaming_report_video_fps=30,
        reference_map="/ref.ply",
        map_comparison_align=False,
        baseline_run="/base",
        experiment_validity_class="SMOKE_ONLY",
        run_intent_report={"intent": "x"},
        experiment_context={"ctx": "y"},
        model_path="/mp",
    )
    scene = SimpleNamespace(
        train_cameras=["camA", "camB"],
        getTestCameras=lambda: ["camT"],
        _keyframes_admitted=7,
        _frame_source=SimpleNamespace(bag_replay_report=None, ingest_counters=None),
    )
    frame_source = SimpleNamespace(
        stream_end_reason="bag_complete",
        bag_replay_report={"br": 1},
        ingest_counters={"ic": 2},
    )
    return dict(
        args=args,
        streaming_scene=scene,
        frame_source=frame_source,
        first_iter=10,
        seed_fade_monitor=_ReportBlockOwner("seed_fade"),
        cadence=SimpleNamespace(last_iteration=110),
        admission_cadence=SimpleNamespace(name="adm"),
        coordinator=SimpleNamespace(
            four_clocks=SimpleNamespace(summary=lambda: _FourClocks())),
        spf_resolution=SimpleNamespace(as_dict=lambda: {"spf": 50}),
        resolution_provenance="RESPROV",
        resolution_tier_log="TIERLOG",
        ingestion_report={"mode": "iter_based"},
        replica_rgb_report=replica,
        soft_all_admitter=SimpleNamespace(summary=lambda: {"skipped": 0}),
        soft_all_invariant={"inserted_from_skipped_frame": 0},
        reinit_tracker=SimpleNamespace(events=["e1"]),
        partition_control_runtime=_PartitionControlRuntime(),
        mem_tracker=_ReportBlockOwner("mem") if mem_tracker_on else None,
        duty=_ReportBlockOwner("duty") if duty_on else None,
        backpressure_owner=_ReportBlockOwner("bp") if backpressure_on else None,
        clk_optimizer_step_id=11,
        clk_supervision_view_count=22,
        clk_rasterizer_call_count=33,
        clk_supervised_pixel_count=44,
        per_source_supervision_counts={"src": 5},
        report_train_metrics_max=8,
        report_test_metrics_max=9,
        report_trajectory_max_frames=10,
        depth_insertion_report_block=lambda: {"total_inserted": 3},
        lineage_preserving_setup_report_block=lambda: lineage,
        seed_cohort_report_block=lambda: {"available": True},
        ingress_ownership_report_block=lambda: {"owner": "z"},
        register_supervision_source=lambda cam: None,
        git_provenance_from_args=lambda args: {"gp": args.model_path},
        multires_cache_report_block=lambda: {"mc": 1},
        load_bag_source_manifest=lambda mp: {"mp": mp},
        set_partition_control_roles=lambda roles: None,
        cadence_reconcile_kwargs=lambda: {"reconciled": True},
    )


@pytest.fixture()
def builder(monkeypatch):
    pytest.importorskip("torch")
    import eligsir.runtime.report_kwargs as module

    _stub_module_helpers(monkeypatch, module)
    return module.build_streaming_report_kwargs


# --------------------------------------------------------------------------
# Behavioral
# --------------------------------------------------------------------------


def test_emits_the_expected_report_keys(builder):
    out = builder(final=True, **_make_kwargs())
    # Anchors spanning every report generation represented in the builder --
    # if the assembly silently stops emitting one of these, report.json loses
    # a block and nothing else in the suite notices.
    for key in (
        "depth_insertion", "seed_cohort", "ingress_ownership", "partition_control",
        "backpressure", "multires_cache", "gpu_memory", "duty_cycle",
        "pose_provenance", "ingestion", "spf_resolution", "resolution",
        "realized_cadence", "admission_cadence", "supervision_work",
        "mapping_clocks", "git_provenance", "experiment_origin",
        "execution_semantics", "run_intent", "experiment_context",
        "seed_fade", "frame_admission", "soft_all_admission",
        "frames_admitted_override", "stream_end_reason",
        "tracking_reinit_events", "bag_replay_report", "ingest_counters",
        "bag_source_manifest", "train_metrics_max", "test_metrics_max",
        "trajectory_max_frames", "report_video_min_width",
        "report_video_max_width", "mp4_fps", "reference_map_path",
        "map_comparison_align", "baseline_run", "experiment_validity_class",
        "experiment_validity_is_final",
    ):
        assert key in out, f"builder stopped emitting report kwarg {key!r}"


def test_final_gates_the_depth_insertion_block(builder):
    """#303: a pre-run/mid-run report legitimately has total_inserted == 0, so
    the depth-insertion gate must not run there."""
    assert "depth_insertion" not in builder(final=False, **_make_kwargs())
    assert builder(final=True, **_make_kwargs())["depth_insertion"] == {
        "total_inserted": 3}


def test_experiment_validity_is_final_tracks_final(builder):
    assert builder(final=False, **_make_kwargs())["experiment_validity_is_final"] is False
    assert builder(final=True, **_make_kwargs())["experiment_validity_is_final"] is True


def test_lineage_block_is_omitted_when_none_and_present_otherwise(builder):
    assert "lineage_preserving_setup" not in builder(final=False, **_make_kwargs())
    out = builder(final=False, **_make_kwargs(lineage={"lin": 1}))
    assert out["lineage_preserving_setup"] == {"lin": 1}


def test_replica_rgb_key_is_absent_for_non_replica_runs(builder):
    """#641: omitted (not None) for non-Replica runs -- an absent key is the
    documented signal, so a caller cannot read a null as 'measured nothing'."""
    assert "replica_rgb" not in builder(final=False, **_make_kwargs())
    out = builder(final=False, **_make_kwargs(replica={"rep": 2}))
    assert out["replica_rgb"] == {"rep": 2}


@pytest.mark.parametrize(
    "flag,key",
    [("mem_tracker_on", "gpu_memory"), ("duty_on", "duty_cycle")],
)
def test_disabled_telemetry_still_emits_an_explicit_block(builder, flag, key):
    """#399/#562: an ABSENT key must mean 'the report predates this
    mechanism', never 'the telemetry was off' -- so the off case emits a real,
    explicitly-disabled block rather than nothing."""
    out = builder(final=True, **_make_kwargs(**{flag: False}))
    assert key in out
    assert out[key] != {"tag": key.replace("_cycle", "").replace("gpu_", "")}
    assert isinstance(out[key], dict) and out[key]


def test_backpressure_key_is_absent_when_the_owner_is_inactive(builder):
    """#601 is the opposite convention to gpu_memory/duty_cycle above, and
    that asymmetry is deliberate -- pin it so a future tidy-up does not
    'harmonize' the three and destroy #601's own distinction."""
    assert "backpressure" not in builder(final=True,
                                         **_make_kwargs(backpressure_on=False))
    assert builder(final=True, **_make_kwargs())["backpressure"] == {"tag": "bp"}


def test_registers_every_train_camera_as_a_supervision_source(builder):
    seen = []
    kwargs = _make_kwargs()
    kwargs["register_supervision_source"] = seen.append
    builder(final=True, **kwargs)
    assert seen == ["camA", "camB"]


def test_partition_control_roles_are_refreshed_through_the_callback(builder):
    """#951's roles refresh was a ``nonlocal`` write from inside the closure.
    It is a real side effect of building the report and had to survive the
    lift -- now as an explicit callback rather than an invisible cell write."""
    captured = []
    kwargs = _make_kwargs()
    kwargs["set_partition_control_roles"] = captured.append
    runtime = kwargs["partition_control_runtime"]
    out = builder(final=True, **kwargs)

    assert len(captured) == 1
    assert captured[0].value == "roles-obj"
    # The refresh uses the CURRENT train/test pools, not a frozen snapshot.
    assert runtime.prepared == [(["camA", "camB"], ["camT"])]
    assert out["partition_control"] == {"roles": "roles-obj"}


def test_soft_all_active_flag_follows_the_admission_mode(builder):
    """#600: False for force_all / hybrid_keyframe / fixed_stride, where the
    admitter is constructed but never wired -- so a reader cannot mistake its
    all-zero counters for 'soft-all ran and never skipped'."""
    kwargs = _make_kwargs()
    assert builder(final=True, **kwargs)["soft_all_admission"]["active"] is True
    for mode in ("force_all", "hybrid_keyframe", "fixed_stride"):
        kwargs = _make_kwargs()
        kwargs["args"].streaming_frame_admission = mode
        assert builder(final=True, **kwargs)["soft_all_admission"]["active"] is False


def test_frames_admitted_override_uses_the_cumulative_counter(builder):
    """Not len(train_cameras) -- that undercounts whenever partition-focus has
    evicted admitted cameras out of the active buffer."""
    kwargs = _make_kwargs()
    assert builder(final=True, **kwargs)["frames_admitted_override"] == 7
    kwargs = _make_kwargs()
    kwargs["streaming_scene"]._keyframes_admitted = 0
    assert builder(final=True, **kwargs)["frames_admitted_override"] == 0

    # A real zero cannot fall through to resident-camera cardinality.
    kwargs["coverage_counters"] = {"frames_admitted": 0}
    kwargs["streaming_scene"]._keyframes_admitted = 99
    assert builder(final=True, **kwargs)["frames_admitted_override"] == 0


def test_coverage_counter_ledger_is_forwarded_without_recomputation(builder):
    """Coverage-v2 must receive the loop's source dispositions, not camera counts.

    Replacing this with ``len(train_cameras)`` or rebuilding it from a
    partition report would make correctly reserved holdouts look missing.
    """
    counters = {
        "source_frames_ingested_total": 10,
        "bootstrap_frames_consumed": 2,
        "reserved_holdout_frames": 1,
        "reserved_guard_frames": 0,
        "rejected_invalid_frames": 0,
        "eligible_mapping_frames": 7,
        "frames_admitted": 7,
        "mapping_skipped_frames": 0,
        "unprocessed_frames": 0,
    }
    kwargs = _make_kwargs()
    kwargs["coverage_counters"] = counters
    assert builder(final=True, **kwargs)["coverage_counters"] == counters


def test_realized_cadence_block_is_built_once_and_shared(builder):
    """#602: the same block object is the top-level ``realized_cadence`` AND
    the copy embedded in ``admission_cadence`` -- two independently-built
    copies is the shape that silently desyncs."""
    out = builder(final=True, **_make_kwargs())
    assert out["realized_cadence"]["rc"] == 110
    assert out["admission_cadence"]["nested"] == ["realized_cadence"]


def test_every_parameter_is_required_or_meaningfully_defaulted(builder):
    """The point of the lift: a formerly-implicit capture that goes missing is
    a loud TypeError, not a silently-wrong report."""
    import inspect
    import eligsir.runtime.report_kwargs as module

    params = inspect.signature(module.build_streaming_report_kwargs).parameters
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values()), (
        "every parameter must be keyword-only -- the wrapper passes ~39 of "
        "them and positional passing would silently mis-bind on any reorder."
    )
    kwargs = _make_kwargs()
    kwargs.pop("seed_fade_monitor")
    with pytest.raises(TypeError):
        builder(final=True, **kwargs)


# --------------------------------------------------------------------------
# Static (torch-free)
# --------------------------------------------------------------------------


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found -- retarget this test, do not delete it")


def test_wrapper_passes_every_declared_parameter():
    """No parameter may be left to its default by the production call site.

    A default that quietly stands in for a real loop local is the #729 bug
    class wearing a different hat: the report would be built from absent state
    and still look complete on disk.
    """
    builder = _find_function(ast.parse(REPORT_KWARGS.read_text()), BUILDER_NAME)
    declared = {a.arg for a in builder.args.kwonlyargs}

    wrapper = _find_function(ast.parse(TRAIN_STREAMING.read_text()), WRAPPER_NAME)
    calls = [
        n for n in ast.walk(wrapper)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id.endswith("build_streaming_report_kwargs")
    ]
    assert len(calls) == 1, (
        f"expected exactly one delegation from {WRAPPER_NAME}() to the "
        f"extracted builder, found {len(calls)}"
    )
    passed = {kw.arg for kw in calls[0].keywords if kw.arg is not None}
    assert not any(kw.arg is None for kw in calls[0].keywords), (
        "the wrapper splats **kwargs into the builder -- that re-hides exactly "
        "the dependencies this extraction exists to make explicit."
    )
    missing = sorted(declared - passed)
    assert not missing, (
        f"{WRAPPER_NAME}() does not pass {missing} -- the builder would fall "
        "back to a default instead of this run's real state."
    )
    extra = sorted(passed - declared)
    assert not extra, f"wrapper passes unknown kwargs {extra}"

    # ...and each is bound to the LOOP LOCAL OF THE SAME NAME. Checking only
    # *which* kwargs are passed would let a transposition through
    # (`report_test_metrics_max=_report_train_metrics_max`), which is the
    # wrong-but-present report block this module's docstring argues is worse
    # than a crash. The lift renamed each capture by stripping its leading
    # underscore and nothing else, so the binding is mechanically checkable.
    for kw in calls[0].keywords:
        assert isinstance(kw.value, ast.Name), (
            f"{kw.arg}= is bound to an expression, not a plain name. The "
            "wrapper must only bind loop locals -- logic belongs in the "
            "builder, where it is testable."
        )
        assert kw.value.id.lstrip("_") == kw.arg, (
            f"{kw.arg}= is bound to {kw.value.id}, not the loop local of the "
            "same name. If this is a deliberate rename, rename the builder's "
            "parameter to match -- a silently transposed binding ships a "
            "wrong-but-present report block."
        )


def test_module_level_helpers_are_the_same_objects_train_streaming_binds():
    """The six helpers the builder no longer receives as parameters must
    resolve to the SAME objects the training loop uses.

    This is not hypothetical: while lifting the body, one of these
    (`build_execution_semantics_block`) was first re-imported from
    `eligsir.mapping.optimization_runtime` -- a module train_streaming.py
    imports several *other* names from -- rather than from its real home
    `utils.execution_semantics`. That is an ImportError, so it was caught; a
    near-miss that resolved to a DIFFERENT but importable object of the same
    name would not have been, and the report would have silently changed
    shape. Identity, not importability, is the property that matters.
    """
    pytest.importorskip("torch")
    import train_streaming as loop
    import eligsir.runtime.report_kwargs as module

    for loop_name, builder_name in (
        ("realized_cadence_report_block", "realized_cadence_report_block"),
        ("admission_cadence_report_block", "admission_cadence_report_block"),
        ("resolution_report_block", "resolution_report_block"),
        ("supervision_work_report_block", "supervision_work_report_block"),
        ("_pose_provenance_report_dict", "pose_provenance_report_dict"),
        ("_execution_semantics_from_args", "execution_semantics_from_args"),
    ):
        assert getattr(loop, loop_name) is getattr(module, builder_name), (
            f"train_streaming.{loop_name} and report_kwargs.{builder_name} are "
            "different objects -- the lifted body is calling a different "
            "implementation than the loop it came from."
        )


def test_builder_is_a_module_level_function_not_a_closure():
    """The regression guard for the lift itself: if a future edit moves the
    body back inside streaming_training(), every ~35-way dependency becomes
    invisible again and the #729 class reopens."""
    tree = ast.parse(REPORT_KWARGS.read_text())
    assert any(
        isinstance(n, ast.FunctionDef) and n.name == BUILDER_NAME for n in tree.body
    ), f"{BUILDER_NAME}() is no longer a module-level function in {REPORT_KWARGS.name}"

    wrapper = _find_function(ast.parse(TRAIN_STREAMING.read_text()), WRAPPER_NAME)
    dict_keys = {
        k.value
        for n in ast.walk(wrapper)
        if isinstance(n, ast.Dict)
        for k in n.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    assert not dict_keys, (
        f"{WRAPPER_NAME}() grew report-key dict literals again ({sorted(dict_keys)}). "
        "Report keys belong in eligsir/runtime/report_kwargs.py, which is "
        "where test_issue646_report_kwargs_signature.py checks them against the "
        "writer's signature -- keys added here are invisible to that guard."
    )
