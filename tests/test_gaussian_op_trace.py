"""Issue #326 -- CPU-side proof that the operation tracer records what it claims.

TORCH-FREE AT MODULE LEVEL BY CONSTRUCTION. The tracer core
(``eligsir.diagnostics.gaussian_op_trace``) and the metric derivation
(``...gaussian_op_conflicts``) take plain Python values, so these tests collect
and really execute in the lean CPU CI image (``Dockerfile.ci-cpu``; see
``tests/conftest.py`` and issue #423) rather than being silently excluded --
the same discipline #543's registry tests used.

The probe (``...gaussian_op_probe``) is the only torch-touching half; its tests
below call ``pytest.importorskip("torch")`` INSIDE the test function so this
file stays collectable everywhere.

WHAT THIS FILE DOES AND DOES NOT PROVE
--------------------------------------
It proves the tracer + derivation record and compute what they claim, over a
fully-specified synthetic mutation sequence -- the same substrate #396's
residency invariant was proven on CPU-side in PR #531. It proves NOTHING about
what the real fr3/floor3 mechanisms actually do at runtime; that needs the GPU
run documented in ``docs/architecture/2026-08-15-issue326-runtime-conflict-trace.md``
and is #326's remaining blocker.
"""

import json
import os

import pytest

from eligsir.diagnostics import gaussian_op_conflicts as C
from eligsir.diagnostics.gaussian_op_trace import (
    MECHANISMS,
    OPERATIONS,
    PHASES,
    UNATTRIBUTABLE,
    GaussianOpEvent,
    GaussianOpTracer,
    VocabularyError,
    get_tracer,
    install_from_env,
    load_events,
    set_tracer,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_JSON = os.path.join(
    REPO_ROOT, "docs", "architecture", "gaussian_writer_vocabulary.json"
)

# MEASURED 2026-08-15 on the `cpu-ci` surface (localhost/eligsir-ci-cpu, built
# at HEAD from Dockerfile.ci-cpu): that image COPYs source trees but NOT `docs/`,
# so #543's checked-in registry snapshot is genuinely absent there. The two
# vocabulary-join tests below are the only ones that need it. They are skipped
# with this explicit reason rather than silently passing on an empty set -- and
# they DO run on `host` and on `training-nogpu`/`training-gpu`, which is where
# the join is actually verified. (Adding `docs/` to Dockerfile.ci-cpu would be
# the better fix; that file was owned by another agent this cycle.)
requires_registry = pytest.mark.skipif(
    not os.path.exists(REGISTRY_JSON),
    reason="docs/architecture/gaussian_writer_vocabulary.json is not present on this "
           "test surface (the cpu-ci image does not COPY docs/); the #543 "
           "vocabulary join is verified on host / training-nogpu instead",
)


# ---------------------------------------------------------------------------
# 1. Vocabulary joins back to #543's static registry
# ---------------------------------------------------------------------------

def _registry_vocabulary():
    with open(REGISTRY_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


@requires_registry
def test_mechanism_vocabulary_is_the_registry_declared_owner_vocabulary():
    """The standalone vocabulary contract and runtime tracer must be identical."""
    owners = set(_registry_vocabulary()["declared_owners"])
    assert owners == MECHANISMS


@requires_registry
def test_operation_vocabulary_is_the_registry_semantic_operation_vocabulary():
    ops = set(_registry_vocabulary()["semantic_operations"])
    assert ops == OPERATIONS


def test_strict_vocabulary_refuses_an_invented_mechanism():
    t = GaussianOpTracer()
    with pytest.raises(VocabularyError):
        t.record(iteration=1, op="grow", mechanism="made_up_owner", phase="applied")
    with pytest.raises(VocabularyError):
        t.record(iteration=1, op="made_up_op", mechanism="mcmc", phase="applied")
    with pytest.raises(VocabularyError):
        t.record(iteration=1, op="grow", mechanism="mcmc", phase="made_up_phase")


def test_unattributable_classes_are_named_and_auto_annotated():
    """#326 asks for `proposed delta`. For these classes it is NOT capturable, and
    the tracer says so on the event instead of fabricating a value."""
    assert "optimizer/move_xyz" in UNATTRIBUTABLE
    assert "mcmc/relocate" in UNATTRIBUTABLE
    t = GaussianOpTracer()
    ev = t.record(
        iteration=7, op="move_xyz", mechanism="optimizer", phase="bracket",
        scope="uid", uid=0, applied_delta={"xyz": [1.0, 0.0, 0.0]},
    )
    assert ev.proposed_delta is None
    assert ev.unattributed_reason and "fused native kernel" in ev.unattributed_reason


# ---------------------------------------------------------------------------
# 2. Zero-overhead-when-off
# ---------------------------------------------------------------------------

def test_tracer_is_off_by_default_and_import_installs_nothing():
    assert get_tracer() is None, "importing the module must not install a tracer"


def test_install_from_env_returns_none_without_the_env_var():
    assert install_from_env(model_path="/tmp/nope", env={}) is None
    assert install_from_env(model_path="/tmp/nope", env={"ELIGSIR_GAUSSIAN_OP_TRACE": "0"}) is None
    assert get_tracer() is None


def test_install_from_env_installs_when_enabled(tmp_path):
    out = str(tmp_path / "t.jsonl")
    try:
        t = install_from_env(env={
            "ELIGSIR_GAUSSIAN_OP_TRACE": "1",
            "ELIGSIR_GAUSSIAN_OP_TRACE_OUT": out,
            "ELIGSIR_GAUSSIAN_OP_TRACE_UID_STRIDE": "512",
            "ELIGSIR_GAUSSIAN_OP_TRACE_EVERY": "10",
        })
        assert t is not None and get_tracer() is t
        assert t.uid_stride == 512 and t.trace_every == 10 and t.out_path == out
    finally:
        tr = get_tracer()
        if tr is not None:
            tr.close()
        set_tracer(None)
    assert get_tracer() is None


def test_train_streaming_hook_sites_keep_every_tensor_touch_inside_the_guard():
    """The real zero-overhead risk is a GPU sync on the OFF path, not the bool test.

    Every ``_optrace``-related line in ``train_streaming.py`` that performs a
    device->host sync (``.cpu()`` / ``.item()`` / ``.clone()`` / a probe call)
    must sit lexically inside an ``if _optrace is not None:`` block. This test
    reads the source and asserts it, so the invariant cannot silently rot.
    """
    src_path = os.path.join(REPO_ROOT, "train_streaming.py")
    with open(src_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    guard_depth = None      # indentation of the active `if _optrace is not None` body
    offenders = []
    for i, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        if guard_depth is not None and indent <= guard_depth:
            guard_depth = None
        if "_optrace is not None" in stripped and stripped.startswith("if "):
            guard_depth = indent
            continue
        if "_gop." in stripped or "gaussian_op_probe" in stripped:
            if guard_depth is None:
                offenders.append((i, stripped))
    assert not offenders, (
        "probe/tensor work outside an `if _optrace is not None:` guard -- this "
        f"would cost a GPU sync on every disabled run: {offenders}"
    )


def test_train_streaming_installs_the_tracer_exactly_once_from_env():
    src_path = os.path.join(REPO_ROOT, "train_streaming.py")
    with open(src_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    assert src.count("_install_optrace(") == 1, (
        "the tracer must be resolved ONCE at startup and held in a local"
    )
    code = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    )
    assert "ELIGSIR_GAUSSIAN_OP_TRACE" not in code, (
        "train_streaming must not read the env var itself; install_from_env owns it"
    )


# ---------------------------------------------------------------------------
# 3. The synthetic mutation sequence
# ---------------------------------------------------------------------------

def _scenario():
    """A fully-specified fake mutation sequence with KNOWN conflict content.

    Deliberately constructed so every derived metric has a hand-computable
    expected value (asserted below), which is what makes this a proof that the
    derivation is correct rather than merely that it runs.

    Content:
      it=100  uid 0     optimizer moves +0.010 x ; depth-lock restores -0.010 x
                        -> full cancellation, opposing pair (optimizer, anchors_depth_locks)
      it=100  uid 1024  mcmc noise moves +0.020 x, nothing cancels it
      it=100  source_depth grow 40 rows, topology 3 -> 4      (advance: healthy)
      it=100  mcmc grow 12 rows, topology 4 -> 4              (SILENT: #543 finding 1)
      it=200  source_depth grow 7 rows while allow_insertion=False (gate bypass)
      it=300  lifecycle nominates {0,1024,2048}; freespace nominates {1024,2048,3072}
                        -> 2 shared death candidates
      it=150  uid 4096 born via source_depth ; it=400 retired by freespace
                        -> cross-mechanism grow-then-prune
    """
    t = GaussianOpTracer(uid_stride=1024, strict_vocabulary=True)

    t.set_gate_state(allow_insertion=True, allow_growth=True)
    t.record(iteration=100, op="move_xyz", mechanism="optimizer", phase="bracket",
             scope="uid", uid=0, topology_version=3,
             state_before={"xyz": [0.0, 0.0, 0.0]},
             state_after={"xyz": [0.010, 0.0, 0.0]},
             applied_delta={"xyz": [0.010, 0.0, 0.0]})
    t.record(iteration=100, op="move_xyz", mechanism="anchors_depth_locks",
             phase="cancelled", scope="uid", uid=0, topology_version=3,
             state_before={"xyz": [0.010, 0.0, 0.0]},
             state_after={"xyz": [0.0, 0.0, 0.0]},
             applied_delta={"xyz": [-0.010, 0.0, 0.0]},
             evidence={"reason": "restore_depth_locked_centers(post_optimizer)"})
    t.record(iteration=100, op="move_xyz", mechanism="mcmc", phase="applied",
             scope="uid", uid=1024, topology_version=3,
             state_before={"xyz": [1.0, 0.0, 0.0]},
             state_after={"xyz": [1.020, 0.0, 0.0]},
             applied_delta={"xyz": [0.020, 0.0, 0.0]},
             evidence={"support": 5.0})

    t.record(iteration=100, op="grow", mechanism="source_depth", phase="applied",
             scope="aggregate", topology_version=4, n_rows=40,
             evidence={"topology_version_before": 3, "topology_version_after": 4})
    t.record(iteration=100, op="grow", mechanism="mcmc", phase="applied",
             scope="aggregate", topology_version=4, n_rows=12,
             evidence={"topology_version_before": 4, "topology_version_after": 4})

    t.record(iteration=150, op="grow", mechanism="source_depth", phase="applied",
             scope="uid", uid=4096, topology_version=5, n_rows=1,
             state_after={"xyz": [2.0, 0.0, 0.0]})

    t.set_gate_state(allow_insertion=False, allow_growth=False)
    t.record(iteration=200, op="grow", mechanism="source_depth", phase="applied",
             scope="aggregate", topology_version=6, n_rows=7,
             evidence={"topology_version_before": 5, "topology_version_after": 6})

    t.set_gate_state(allow_insertion=True, allow_growth=True)
    t.record(iteration=300, op="retire", mechanism="lifecycle", phase="candidate",
             scope="aggregate", topology_version=6, n_rows=3,
             candidate_uids=[0, 1024, 2048], evidence={"reason": "stale"})
    t.record(iteration=300, op="retire", mechanism="freespace_contradiction",
             phase="candidate", scope="aggregate", topology_version=6, n_rows=3,
             candidate_uids=[1024, 2048, 3072], evidence={"reason": "oob"})

    t.record(iteration=400, op="retire", mechanism="freespace_contradiction",
             phase="applied", scope="uid", uid=4096, topology_version=7,
             evidence={"reason": "oob"})
    return t


def test_scenario_records_every_event():
    t = _scenario()
    assert len(t.events) == 10
    assert {e.mechanism for e in t.events} == {
        "optimizer", "anchors_depth_locks", "mcmc", "source_depth",
        "lifecycle", "freespace_contradiction",
    }
    assert all(e.phase in PHASES for e in t.events)


def test_gate_state_is_stamped_on_every_event_as_of_that_iteration():
    t = _scenario()
    gated = [e for e in t.events if e.iteration == 200]
    assert gated and all(e.gate["allow_insertion"] is False for e in gated)
    early = [e for e in t.events if e.iteration == 100]
    assert all(e.gate["allow_insertion"] is True for e in early)


# ---------------------------------------------------------------------------
# 4. Every derived metric, against hand-computed expectations
# ---------------------------------------------------------------------------

def test_movement_cancellation_ratio():
    m = C.movement_cancellation_ratio(_scenario().events)
    # applied: 0.010 (optimizer, uid 0) + 0.010 (depth-lock, uid 0) + 0.020 (mcmc)
    assert m["total_applied_magnitude"] == pytest.approx(0.040)
    # only the optimizer's 0.010 is reversed
    assert m["total_cancelled_magnitude"] == pytest.approx(0.010)
    assert m["ratio"] == pytest.approx(0.25)
    top = m["by_pair"][0]
    assert (top["mover"], top["canceller"]) == ("optimizer", "anchors_depth_locks")
    assert top["ratio"] == pytest.approx(1.0), "100% of that mover's motion was undone"


def test_same_uid_opposing_action_rate():
    m = C.same_uid_opposing_action_rate(_scenario().events)
    assert m["n_multi_mechanism_cells"] == 1       # only uid 0 @ it=100 has two mechanisms
    assert m["n_opposing_cells"] == 1
    assert m["rate"] == pytest.approx(1.0)
    assert m["by_pair"][0]["mechanism_a"] == "anchors_depth_locks"
    assert m["by_pair"][0]["mechanism_b"] == "optimizer"
    assert m["by_pair"][0]["state"] == "xyz"


def test_grow_then_prune_is_cross_mechanism():
    m = C.grow_then_prune_rate(_scenario().events, window_iters=500)
    assert m["n_births_observed"] == 1
    assert m["n_short_lived"] == 1
    assert m["n_cross_mechanism"] == 1
    hit = m["examples"][0]
    assert hit["uid"] == 4096 and hit["lifetime_iters"] == 250
    assert (hit["creator"], hit["killer"]) == ("source_depth", "freespace_contradiction")


def test_grow_then_prune_window_is_respected():
    m = C.grow_then_prune_rate(_scenario().events, window_iters=100)
    assert m["n_short_lived"] == 0, "a 250-iteration lifetime is outside a 100-iteration window"


def test_relocate_then_restore_rate():
    m = C.relocate_then_restore_rate(_scenario().events, window_iters=1)
    assert m["n_moves"] == 3
    assert m["n_restored"] == 1
    assert m["by_pair"][0] == {"mover": "optimizer", "restorer": "anchors_depth_locks", "n": 1}


def test_death_candidate_overlap_finds_the_two_shared_uids():
    m = C.death_candidate_overlap(_scenario().events)
    assert m["n_iterations_with_candidates"] == 1
    row = m["by_pair"][0]
    assert {row["mechanism_a"], row["mechanism_b"]} == {"lifecycle", "freespace_contradiction"}
    assert row["n_shared_uids"] == 2            # {1024, 2048}
    assert row["jaccard"] == pytest.approx(2 / 4)


def test_topology_mutation_while_gate_off():
    m = C.topology_mutation_while_gate_off(_scenario().events)
    assert m["n_population_changing_events"] == 5
    assert m["n_while_gate_off"] == 1
    assert m["examples"][0]["iteration"] == 200
    assert m["examples"][0]["mechanism"] == "source_depth"


def test_topology_mutation_while_gate_off_catches_mcmc_growth_too():
    """Issue #330: before train_streaming.py's "hook I" (the
    op-trace emission added alongside step_post_backward's existing
    population-count/topology_version delta check), no production MCMC path
    ever called ``tracer.record()`` -- so this metric, despite being exactly
    the "runtime invariant" #330 asks for, could structurally never catch an
    MCMC growth committed while the scheduler said ``allow_growth=False``,
    no matter how real the violation was. This test proves the METRIC itself
    has no mechanism-specific blind spot: an ``mcmc``-owned aggregate ``grow``
    event recorded while ``allow_growth=False`` is caught exactly like a
    ``source_depth`` one is in `_scenario()` above. It does not (and cannot,
    being CPU-only / torch-free) prove that the real GPU training loop's new
    hook fires correctly -- that needs a real run, per the #330 audit doc.
    """
    t = GaussianOpTracer(uid_stride=1024, strict_vocabulary=True)
    t.set_gate_state(allow_insertion=True, allow_growth=False)
    t.record(iteration=500, op="grow", mechanism="mcmc", phase="applied",
             scope="aggregate", topology_version=9, n_rows=3,
             evidence={"topology_version_before": 9, "topology_version_after": 9,
                       "n_before": 100, "n_after": 103})
    m = C.topology_mutation_while_gate_off(t.events)
    assert m["n_population_changing_events"] == 1
    assert m["n_while_gate_off"] == 1
    assert m["by_mechanism"] == [{"mechanism": "mcmc", "op": "grow", "n": 1}]


def test_population_change_without_topology_advance_reproduces_543_finding_1():
    """#543 secondary finding #1, statically derived, made runtime-checkable."""
    m = C.population_change_without_topology_advance(_scenario().events)
    assert m["n_population_events_with_topology_evidence"] == 3
    assert m["n_silent_population_changes"] == 1
    assert m["by_mechanism"] == [{"mechanism": "mcmc", "n": 1}]
    assert m["examples"][0]["n_rows"] == 12


def test_mutations_after_support_starvation_needs_support_evidence():
    ev = _scenario().events
    m = C.mutations_after_support_starvation(ev, support_threshold=0.0)
    assert m["n_mutations_with_support_evidence"] == 1   # only the mcmc move carries support
    assert m["n_on_starved"] == 0
    m2 = C.mutations_after_support_starvation(ev, support_threshold=10.0)
    assert m2["n_on_starved"] == 1


def test_growth_near_shape_bound_hit_needs_shape_evidence():
    """Issue #329 class 2 ("shape constraints -> compensatory growth"): mirrors
    #326's support-starvation honesty contract -- no production hook populates
    ``evidence["shape_bound_hit"]`` yet, so on the standard scenario (none of
    whose grow events carry it) the metric must report zero-with-evidence
    rather than a fabricated rate."""
    m = C.growth_near_shape_bound_hit_rate(_scenario().events)
    assert m["n_growth_events_with_shape_evidence"] == 0
    assert m["n_near_shape_bound"] == 0
    assert m["rate"] == 0.0
    assert m["by_mechanism"] == []


def test_growth_near_shape_bound_hit_rate_when_a_hook_does_populate_evidence():
    t = GaussianOpTracer(uid_stride=1024, strict_vocabulary=True)
    t.set_gate_state(allow_insertion=True, allow_growth=True)
    # A grow event near a Gaussian that just hit its scale-prior/anisotropy clamp.
    t.record(iteration=500, op="grow", mechanism="mcmc", phase="applied",
             scope="aggregate", topology_version=1, n_rows=3,
             evidence={"shape_bound_hit": True})
    # A grow event with no nearby bound-hit.
    t.record(iteration=500, op="grow", mechanism="mcmc", phase="applied",
             scope="aggregate", topology_version=1, n_rows=5,
             evidence={"shape_bound_hit": False})
    # A grow event carrying no shape evidence at all -- must not count either way.
    t.record(iteration=500, op="grow", mechanism="source_depth", phase="applied",
             scope="aggregate", topology_version=1, n_rows=2)

    m = C.growth_near_shape_bound_hit_rate(t.events)
    assert m["n_growth_events_with_shape_evidence"] == 2
    assert m["n_near_shape_bound"] == 1
    assert m["rate"] == pytest.approx(0.5)
    assert m["by_mechanism"] == [{"mechanism": "mcmc", "n": 1}]


def test_annotate_modified_again_names_the_later_mechanism():
    annotated = C.annotate_modified_again(_scenario().events)
    by_seq = {e.seq: e for e in annotated}
    opt_ev = next(e for e in annotated if e.mechanism == "optimizer")
    assert by_seq[opt_ev.seq].modified_again_by == "anchors_depth_locks"
    mcmc_move = next(e for e in annotated
                     if e.mechanism == "mcmc" and e.scope == "uid")
    assert mcmc_move.modified_again_by is None


# ---------------------------------------------------------------------------
# 5. The acceptance artifact: the ranked conflict table
# ---------------------------------------------------------------------------

def test_ranked_conflict_table_confirms_only_real_fights():
    table = C.ranked_conflict_table(_scenario().events)
    kinds = {r["kind"] for r in table}
    assert {"movement_cancellation", "opposing_same_uid", "grow_then_prune",
            "death_candidate_overlap"} <= kinds

    fighting = {(r["mechanism_a"], r["mechanism_b"]) for r in table if r["confirmed_fighting"]}
    assert ("optimizer", "anchors_depth_locks") in fighting
    assert ("source_depth", "freespace_contradiction") in fighting

    # Candidate-set overlap alone is redundant authority, NOT an applied fight,
    # and #326's acceptance wording forbids calling it one.
    overlap_rows = [r for r in table if r["kind"] == "death_candidate_overlap"]
    assert overlap_rows and all(not r["confirmed_fighting"] for r in overlap_rows)


def test_ranked_table_is_sorted_by_frequency_then_magnitude():
    table = C.ranked_conflict_table(_scenario().events)
    keys = [(-r["n_events"], -r["magnitude"]) for r in table]
    assert keys == sorted(keys)


def test_derive_all_is_complete_and_json_serialisable():
    m = C.derive_all(_scenario().events)
    for key in (
        "movement_cancellation", "same_uid_opposing_actions", "grow_then_prune",
        "relocate_then_restore", "death_candidate_overlap",
        "topology_mutation_while_gate_off",
        "population_change_without_topology_advance",
        "mutations_after_support_starvation", "growth_near_shape_bound_hit",
        "ranked_conflict_table",
    ):
        assert key in m, key
    assert m["iterations_covered"] == [100, 400]
    assert "POSITIVE evidence only" in m["evidence_class_caveat"]
    json.dumps(m)  # must not raise


def test_empty_trace_yields_zeroed_metrics_not_a_crash():
    m = C.derive_all([])
    assert m["n_events"] == 0
    assert m["ranked_conflict_table"] == []
    assert m["movement_cancellation"]["ratio"] == 0.0


# ---------------------------------------------------------------------------
# 6. JSONL round-trip (what a real run actually writes)
# ---------------------------------------------------------------------------

def test_jsonl_round_trip_preserves_every_metric(tmp_path):
    out = str(tmp_path / "gaussian_op_trace.jsonl")
    src = _scenario()
    sink = GaussianOpTracer(out_path=out, uid_stride=1024)
    for e in src.events:
        d = {k: v for k, v in e.__dict__.items()}
        d.pop("seq"); d.pop("modified_again_by")
        sink.record(**d)
    sink.close()

    reloaded = load_events(out)
    assert len(reloaded) == len(src.events)
    assert C.derive_all(reloaded) == C.derive_all(src.events)


def test_max_events_drops_rather_than_growing_unbounded():
    t = GaussianOpTracer(max_events=3)
    for i in range(10):
        t.record(iteration=i, op="grow", mechanism="mcmc", phase="applied")
    assert len(t.events) == 3
    assert t.dropped == 7


def test_cli_end_to_end(tmp_path):
    import subprocess
    import sys as _sys
    out = tmp_path / "gaussian_op_trace.jsonl"
    src = _scenario()
    with open(out, "w", encoding="utf-8") as fh:
        for e in src.events:
            fh.write(e.to_json() + "\n")
    script = os.path.join(REPO_ROOT, "scripts", "gaussian_op_conflicts.py")
    res = subprocess.run(
        [_sys.executable, script, str(tmp_path), "--md"],
        capture_output=True, text=True, check=True,
    )
    assert "Ranked conflict table" in res.stdout
    assert "optimizer" in res.stdout
    with open(tmp_path / "gaussian_op_conflicts.json", encoding="utf-8") as fh:
        got = json.load(fh)
    assert got["population_change_without_topology_advance"]["n_silent_population_changes"] == 1


# ---------------------------------------------------------------------------
# 7. Watch set + the torch-touching probe
# ---------------------------------------------------------------------------

def test_watch_set_is_stateless_and_deterministic():
    t = GaussianOpTracer(uid_stride=1024)
    assert t.watches(0) and t.watches(1024) and t.watches(2048)
    assert not t.watches(1) and not t.watches(1023)
    # Stable as the population grows: membership depends only on the uid.
    assert [u for u in range(5000) if t.watches(u)] == [0, 1024, 2048, 3072, 4096]


def test_trace_every_gates_iterations():
    t = GaussianOpTracer(trace_every=50)
    assert t.traces_iteration(0) and t.traces_iteration(100)
    assert not t.traces_iteration(1) and not t.traces_iteration(99)


class _FakeGaussians:
    """Duck-typed stand-in exposing only what the probe touches."""

    def __init__(self, torch, xyz, uids=None):
        self._t = torch
        self._xyz = torch.tensor(xyz, dtype=torch.float32)
        n = self._xyz.shape[0]
        self.gaussian_uid = (
            torch.arange(n, dtype=torch.int64) if uids is None
            else torch.tensor(uids, dtype=torch.int64)
        )
        self._op = torch.full((n, 1), 0.5)
        self._sc = torch.full((n, 3), 0.1)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self._op

    @property
    def get_scaling(self):
        return self._sc


def test_probe_matches_rows_by_uid_not_by_index_across_a_prune():
    torch = pytest.importorskip("torch")
    from eligsir.diagnostics import gaussian_op_probe as P

    g = _FakeGaussians(torch, [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]], uids=[0, 2, 4])
    before, real = P.snapshot(g, uid_stride=2)
    assert real is True
    assert before.uids == [0, 2, 4]

    # Row 1 (uid 2) is pruned; uid 4 shifts from index 2 to index 1 and moves.
    g2 = _FakeGaussians(torch, [[0.0, 0, 0], [2.5, 0, 0]], uids=[0, 4])
    after, _ = P.snapshot(g2, uid_stride=2)

    t = GaussianOpTracer(uid_stride=2)
    n = P.emit_uid_deltas(t, before=before, after=after, iteration=5,
                          op="move_xyz", mechanism="mcmc", phase="applied",
                          topology_version=1)
    assert n == 1, "only uid 4 changed; uid 2 vanished and must not be diffed"
    ev = t.events[0]
    assert ev.uid == 4
    assert ev.applied_delta["xyz"][0] == pytest.approx(0.5), (
        "an index-matched diff would have reported 1.5 here -- the wrong Gaussian"
    )


def test_probe_falls_back_to_row_indices_and_says_so_without_lineage_buffers():
    torch = pytest.importorskip("torch")
    from eligsir.diagnostics import gaussian_op_probe as P

    g = _FakeGaussians(torch, [[0.0, 0, 0]] * 6)
    del g.gaussian_uid
    snap, real = P.snapshot(g, uid_stride=2)
    assert real is False, "must report that 'uid' is really a row index"
    assert snap.uids == [0, 2, 4]


def test_probe_candidate_set_records_only_the_watched_subset():
    torch = pytest.importorskip("torch")
    from eligsir.diagnostics import gaussian_op_probe as P

    g = _FakeGaussians(torch, [[float(i), 0, 0] for i in range(8)])
    mask = torch.tensor([True, True, True, True, False, False, True, True])
    t = GaussianOpTracer(uid_stride=2)
    P.emit_candidate_set(t, g, mask, iteration=9, op="retire",
                         mechanism="lifecycle", topology_version=2, reason="stale")
    ev = t.events[0]
    assert ev.phase == "candidate"
    assert ev.n_rows == 6, "the full candidate COUNT is recorded even though uids are sampled"
    assert ev.candidate_uids == [0, 2, 6]


def test_probe_emits_nothing_when_no_state_changed():
    torch = pytest.importorskip("torch")
    from eligsir.diagnostics import gaussian_op_probe as P

    g = _FakeGaussians(torch, [[1.0, 2.0, 3.0]] * 4)
    snap, _ = P.snapshot(g, uid_stride=2)
    t = GaussianOpTracer(uid_stride=2)
    assert P.emit_uid_deltas(t, before=snap, after=snap, iteration=1,
                             op="move_xyz", mechanism="mcmc", phase="applied",
                             topology_version=0) == 0
    assert t.events == []
