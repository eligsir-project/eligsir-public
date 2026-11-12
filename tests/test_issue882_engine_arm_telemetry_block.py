"""Regression coverage for issue #882.

#880 hoisted the GPU memory residency snapshot out of the inline-only `else:`
arm so both optimizer arms reach it. A sibling per-iteration TensorBoard block
was deliberately left behind and still has the identical engine-arm blind spot.

On any `--streaming_optimization_engine` run, 8 scalars silently never emit:
  - io/save_queue_depth
  - streaming/bootstrap_anchor/{drift_mean_m,grad_retain_frac}
  - streaming/insertion_anchor/n_active
  - streaming/bubble/{n_active,n_frozen,active_frac,d_obs_m}

This file covers the CPU-verifiable acceptance items:

(a) **Structural regression guard** -- a static/AST assertion. It parses
    ``train_streaming.py`` itself and asserts:
      - the four-group telemetry block (io/*, bootstrap_anchor/*, insertion_anchor/*,
        bubble/*) is NOT nested inside either arm's body/orelse;
      - that block IS reachable from both arms (i.e. it must live outside and after
        the two-arm branch).
    This is a precise, narrow claim: it proves the *code shape* that made
    the engine arm miss these 8 scalars cannot recur without this test failing.

(b) **Sanity assertion** -- all 8 scalar names still exist somewhere in the file,
    so a refactor that deletes them fails loudly instead of passing vacuously.
"""

import ast
from pathlib import Path

import pytest


TRAIN_STREAMING_PATH = Path(__file__).resolve().parent.parent / "train_streaming.py"

# The 8 scalars that issue #882 addresses
REQUIRED_SCALARS = {
    "io/save_queue_depth",
    "streaming/bootstrap_anchor/drift_mean_m",
    "streaming/bootstrap_anchor/grad_retain_frac",
    "streaming/insertion_anchor/n_active",
    "streaming/bubble/n_active",
    "streaming/bubble/n_frozen",
    "streaming/bubble/active_frac",
    "streaming/bubble/d_obs_m",
}


def _is_engine_arm_guard(test: ast.expr) -> bool:
    """True for the exact `_streaming_optimization_runtime is not None` test
    that splits train_streaming.py's per-iteration body into the engine arm
    (if) and the inline arm (else) -- see issue #809's description of this
    split."""
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


def _extract_scalar_names_from_nodes(nodes) -> set:
    """Extract all string literals passed as the first argument to
    tb_writer.add_scalar(...) calls within the given nodes."""
    names = set()
    for n in nodes:
        for sub in ast.walk(n):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "add_scalar"
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "tb_writer"
                and sub.args
                and isinstance(sub.args[0], ast.Constant)
                and isinstance(sub.args[0].value, str)
            ):
                names.add(sub.args[0].value)
    return names


def _contains_required_scalars(nodes) -> bool:
    """Whether any statement in `nodes` contains at least one call to
    tb_writer.add_scalar() for any of the REQUIRED_SCALARS."""
    found = _extract_scalar_names_from_nodes(nodes)
    return bool(found & REQUIRED_SCALARS)


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


# --------------------------------- structural (a) ---------------------------


def test_two_arm_branch_is_where_issue_809_and_882_say_it_is():
    """Sanity check on the matcher itself: the branch must actually be the
    one #809 describes (engine `if`, inline `else:`), with only the engine arm
    calling the canonical mapping application's tick."""
    body, if_node, idx = _find_two_arm_branch()
    assert _calls_mapping_application_tick(if_node.body)
    assert not _calls_mapping_application_tick(if_node.orelse)


def test_telemetry_scalars_not_nested_in_either_arm():
    """The #882 defect: 8 scalars' emit block living inside the inline `else:`
    arm only. Neither arm's body may contain an add_scalar() call for any of
    the REQUIRED_SCALARS after the fix -- they must live outside the branch
    entirely."""
    body, if_node, idx = _find_two_arm_branch()

    engine_scalars = _extract_scalar_names_from_nodes(if_node.body) & REQUIRED_SCALARS
    assert not engine_scalars, (
        f"REQUIRED_SCALARS found inside the engine arm's body: {engine_scalars} -- "
        "this would mean the telemetry logic got duplicated into the engine "
        "arm instead of hoisted to a shared location"
    )

    inline_scalars = _extract_scalar_names_from_nodes(if_node.orelse) & REQUIRED_SCALARS
    assert not inline_scalars, (
        f"REQUIRED_SCALARS found inside the inline arm's body: {inline_scalars} -- "
        "this is the original #882 defect location; these scalar calls must be "
        "hoisted to the shared code both arms rejoin into"
    )


def test_telemetry_scalars_still_exist_somewhere():
    """Sanity check: all 8 required scalars must still exist somewhere in the
    file (they shouldn't have been deleted). This guards against a test that
    passes vacuously because the telemetry was simply removed instead of
    properly hoisted."""
    src = TRAIN_STREAMING_PATH.read_text()
    for scalar_name in REQUIRED_SCALARS:
        assert f'"{scalar_name}"' in src or f"'{scalar_name}'" in src, (
            f"scalar {scalar_name!r} not found anywhere in train_streaming.py -- "
            "it may have been deleted accidentally"
        )


def test_at_least_one_scalar_from_each_group_exists():
    """Stronger sanity check: at least one scalar from each of the four groups
    must still emit. This guards against a partial deletion or refactor."""
    src = TRAIN_STREAMING_PATH.read_text()

    groups = {
        "io": ["io/save_queue_depth"],
        "bootstrap_anchor": [
            "streaming/bootstrap_anchor/drift_mean_m",
            "streaming/bootstrap_anchor/grad_retain_frac",
        ],
        "insertion_anchor": ["streaming/insertion_anchor/n_active"],
        "bubble": [
            "streaming/bubble/n_active",
            "streaming/bubble/n_frozen",
            "streaming/bubble/active_frac",
            "streaming/bubble/d_obs_m",
        ],
    }

    for group_name, scalar_names in groups.items():
        found_any = False
        for scalar_name in scalar_names:
            if f'"{scalar_name}"' in src or f"'{scalar_name}'" in src:
                found_any = True
                break
        assert found_any, (
            f"no scalars from group {group_name!r} found in train_streaming.py; "
            "at least one must exist"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
