"""Regression coverage for issue #879.

#399 wired a GPU memory residency snapshot (``_mem_tracker.snapshot(...)``)
into ``train_streaming.py``'s per-iteration body, but the call site ended up
living entirely inside the inline (``else:``) arm of the engine/inline
two-arm branch that #809 documented -- so any run using
``--streaming_optimization_engine`` (the ``if _streaming_optimization_runtime
is not None:`` arm) took ZERO snapshots. Because ``report_block()`` still
unconditionally emitted every attribution field at its ``__init__``-time
default of ``0``, the resulting ``report.json["gpu_memory"]`` looked like a
fully-populated, real measurement of zero rather than an absent one -- see
the four completed champion-scale fr3 arms in #879's issue body.

This file covers the two CPU-verifiable acceptance items:

(a) **Structural regression guard** -- a full ``streaming_training()`` call
    is not runnable on CPU (it needs a live GPU-backed render loop), so this
    is a static/AST assertion, not an execution one. It parses
    ``train_streaming.py`` itself and asserts:
      - the two-arm branch has exactly one snapshot call site;
      - that call site is NOT nested inside either arm's body/orelse;
      - that call site IS the immediate next sibling statement after the
        two-arm ``if``/``else`` in the same statement list, i.e. it runs
        unconditionally on the control-flow path both arms rejoin into.
    This is a precise, narrow claim: it proves the *code shape* that made
    the engine arm miss the snapshot cannot recur without this test
    failing. It does NOT prove the snapshot fires correctly at runtime on
    either arm -- that needs a real (GPU) run; see the still-open GPU
    acceptance item in issue #879.

(b) **Zero-sample self-reporting** -- ``GPUMemoryResidencyTracker.report_block()``
    must not emit a zero-filled block indistinguishable from a genuine
    measurement of zero. Exercised directly against the tracker (CPU-only,
    per the existing ``tests/test_gpu_memory_telemetry.py`` convention of
    constructing it with ``device="cpu"`` and injected allocator readers).
"""

import ast
from pathlib import Path

import pytest

from eligsir.utils.gpu_memory_telemetry import GPUMemoryResidencyTracker


TRAIN_STREAMING_PATH = Path(__file__).resolve().parent.parent / "train_streaming.py"


# ------------------------------------------------- structural (a) ------------


def _is_engine_arm_guard(test: ast.expr) -> bool:
    """True for the exact `_streaming_optimization_runtime is not None` test
    that splits train_streaming.py's per-iteration body into the engine arm
    (if) and the inline arm (else) -- see issue #809's description of this
    split and #879's issue body, which names this branch verbatim."""
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "_streaming_optimization_runtime"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.IsNot)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


def _calls_mapping_application_tick(nodes) -> bool:
    return any(
        isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "tick"
        and isinstance(sub.func.value, ast.Name)
        and sub.func.value.id == "_mapping_application"
        for node in nodes
        for sub in ast.walk(node)
    )


def _contains_mem_tracker_snapshot_call(nodes) -> bool:
    """Whether any statement in `nodes` contains a `_mem_tracker.snapshot(...)`
    call anywhere in its subtree.

    Deliberately narrower than "any `.snapshot(` attribute access": the
    inline arm also calls an unrelated `_gop.snapshot(...)` (gaussian_op_probe
    diagnostic helper, a completely different function) a few lines away from
    where #879's `_mem_tracker.snapshot(...)` used to live -- matching on the
    bare attribute name alone produces a false positive against that call.
    """
    for n in nodes:
        for sub in ast.walk(n):
            if (
                isinstance(sub, ast.Attribute)
                and sub.attr == "snapshot"
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "_mem_tracker"
            ):
                return True
    return False


def _find_two_arm_branch():
    """Locate the engine/inline `if`/`else` statement (the one whose `if`
    body calls the canonical mapping application's tick and whose test is the
    `_streaming_optimization_runtime is not None` guard), plus the
    statement list it lives in and its index within that list.

    Returns (containing_body, if_node, index).
    """
    src = TRAIN_STREAMING_PATH.read_text()
    tree = ast.parse(src, filename=str(TRAIN_STREAMING_PATH))

    func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "streaming_training":
            func = node
            break
    assert func is not None, "streaming_training() not found in train_streaming.py"

    matches = []

    def walk_bodies(node):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                for item in value:
                    if (
                        isinstance(item, ast.If)
                        and _is_engine_arm_guard(item.test)
                        and item.orelse
                        and _calls_mapping_application_tick(item.body)
                    ):
                        matches.append((value, item))
                    walk_bodies(item)
            elif isinstance(value, ast.AST):
                walk_bodies(value)

    walk_bodies(func)
    assert len(matches) == 1, (
        f"expected exactly one engine/inline two-arm branch inside "
        f"streaming_training(), found {len(matches)} -- the branch this "
        f"test targets may have been restructured; update the matcher"
    )
    body, if_node = matches[0]
    return body, if_node, body.index(if_node)


def test_two_arm_branch_is_where_issue_809_and_879_say_it_is():
    """Sanity check on the matcher itself: the branch must actually be the
    one #809/#879 describe (engine `if`, inline `else:`), with only the engine
    arm calling the canonical mapping application's tick."""
    body, if_node, idx = _find_two_arm_branch()
    assert _calls_mapping_application_tick(if_node.body)
    assert not _calls_mapping_application_tick(if_node.orelse)
    # Inline arm sets _backward_ran = False as its first statement (see the
    # `else:` at what was train_streaming.py:6939 when issue #879 was filed).
    assert isinstance(if_node.orelse[0], ast.Assign)


def test_memory_snapshot_call_site_is_not_nested_in_either_arm():
    """The #879 defect: `_mem_tracker.snapshot(...)` living inside the
    inline `else:` arm only. Neither arm's body may contain a `.snapshot(`
    call after the fix -- it must live outside the branch entirely."""
    body, if_node, idx = _find_two_arm_branch()
    assert not _contains_mem_tracker_snapshot_call(if_node.body), (
        "a `.snapshot(` call was found inside the engine arm's body -- "
        "this would mean the snapshot logic got duplicated into the engine "
        "arm instead of hoisted to a shared location, which issue #879 "
        "explicitly says NOT to do (two divergent copies)"
    )
    assert not _contains_mem_tracker_snapshot_call(if_node.orelse), (
        "a `.snapshot(` call was found inside the inline arm's body -- this "
        "is the original #879 defect location; the call site must be "
        "hoisted to the shared code both arms rejoin into"
    )


def test_memory_snapshot_call_site_is_the_immediate_next_sibling_statement():
    """The regression guard proper: the snapshot call site must be reachable
    unconditionally from BOTH arms, which this test pins structurally by
    requiring it to be the very next statement in the same statement list
    as the two-arm `if`/`else` -- i.e. on the control-flow path both arms
    always rejoin into, with no further per-arm conditional in between.

    This is deliberately strict (immediate next sibling, not "somewhere
    later in the function") so that a future refactor which re-nests this
    call inside a NEW per-arm guard, or moves it behind another branch
    point, fails this test rather than silently reintroducing #879's bug
    class.
    """
    body, if_node, idx = _find_two_arm_branch()
    assert idx + 1 < len(body), "two-arm branch has no following sibling statement"
    next_stmt = body[idx + 1]
    assert isinstance(next_stmt, ast.If), (
        f"expected the next statement after the engine/inline branch to be "
        f"the memory-snapshot `if _mem_tracker is not None ...:` guard, got "
        f"{type(next_stmt).__name__} at line {next_stmt.lineno}"
    )
    assert _contains_mem_tracker_snapshot_call([next_stmt]), (
        "the statement immediately following the engine/inline branch does "
        "not contain a `.snapshot(` call -- the hoisted memory-telemetry "
        "block may have been moved or removed"
    )


def test_exactly_one_mem_tracker_snapshot_call_site_in_the_whole_file():
    """Belt-and-suspenders textual check: guards against a future edit that
    hoists the call correctly but ALSO leaves (or reintroduces) a second,
    divergent copy inside one of the arms -- issue #879 is explicit that a
    duplicated-not-hoisted fix is unacceptable."""
    src = TRAIN_STREAMING_PATH.read_text()
    assert src.count("_mem_tracker.snapshot(") == 1


# ------------------------------------------------ zero-sample block (b) ------


def _tracker(allocated=0):
    return GPUMemoryResidencyTracker(
        device="cpu",
        total_bytes_fn=lambda: allocated,
        reserved_bytes_fn=lambda: allocated,
        peak_bytes_fn=lambda: allocated,
    )


def test_report_block_zero_samples_is_explicitly_unavailable():
    """A tracker that was constructed but never had snapshot() called on it
    (exactly what happened on every engine-arm run before this fix) must
    self-report as unavailable, not as a real zero measurement."""
    trk = _tracker()
    block = trk.report_block()

    assert block["n_snapshots"] == 0
    assert block["available"] is False
    assert isinstance(block.get("reason"), str) and block["reason"], (
        "a zero-sample report_block() must carry a non-empty, verbatim "
        "reason string, mirroring the seed_cohort / measurement_incomplete "
        "precedent used elsewhere in the reporting layer"
    )
    # The attribution fields are still present (backward compatible: the key
    # never disappears), but they are unmeasured __init__ defaults -- the
    # `available` flag above is what a consumer must check before trusting
    # them, exactly as with seed_cohort["available"].
    assert block["peak_allocated_bytes"] == 0
    assert block["peak_reserved_bytes"] == 0
    assert all(v == 0 for v in block["peak_bucket_bytes"].values())


def test_report_block_with_samples_is_explicitly_available():
    """The positive case: once at least one snapshot was taken, `available`
    must be True and no misleading `reason` should be attached to the
    top-level block (the pre-existing nested `final`/`reason` pair is
    unaffected either way)."""
    trk = _tracker(allocated=10**6)
    trk.snapshot(iteration=0, frames_ingested=1)
    block = trk.report_block()

    assert block["n_snapshots"] == 1
    assert block["available"] is True
    assert block["final"] is not None


def test_report_block_available_key_does_not_remove_existing_keys():
    """Backward compatibility: every key report_block() emitted before #879
    must still be present, for runs that DID sample."""
    trk = _tracker(allocated=10**6)
    trk.snapshot(iteration=0, frames_ingested=1)
    block = trk.report_block()

    for key in (
        "enabled",
        "device",
        "n_snapshots",
        "bucket_order",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "peak_bucket_bytes",
        "history_len",
        "history_stride",
        "history_max",
        "final",
        "growth",
        "series",
    ):
        assert key in block, f"pre-#879 key {key!r} missing from report_block()"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
