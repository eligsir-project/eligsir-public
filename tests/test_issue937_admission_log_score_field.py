"""Issue #937 regression guard: the keyframe_admission_log.jsonl sidecar must
persist every scalar field the admission-stats builders actually produce.

Background. `train_streaming.py`'s admission closure builds a per-decision
`stats` dict from a `KeyframeDecision`, and a separate block later writes a
row of that dict to `<model_path>/keyframe_admission_log.jsonl`. The writer
does NOT dump the dict -- it copies a hardcoded whitelist of keys. Any key
the builder produces but the whitelist omits is computed, carried on the
frame object, and then silently discarded at persist time.

That has now happened twice:

  * `precedence` (#928/#927) -- the admission precedence reason chain was
    computed for every decision but dropped, so the two byte-identical-looking
    `forced=True, reason="max_gap"` paths stayed indistinguishable in the
    artifacts for a full release cycle.
  * `score` / `cheap_score` / `forced` / `moved` / `novel` (#937) -- dropped
    since the sidecar was introduced.

The `score` case has a concrete, blocking cost. On the `db_diverse_revisit`
and `rej_redundant_angle` exits of the coverage-gain gate, `decision.score`
IS the observation-DB overlap fraction that the gate thresholds on -- and it
is the only record of that quantity anywhere. An audit of 100 historical
admission logs (2026-08-16) found 12 runs carrying 238 `rej_redundant_angle`
rows each and ZERO `db_diverse_revisit` rows in the entire corpus, but could
not determine whether the overlap term or the 20-degree angle term was the
binding constraint -- because the only overlap-shaped field that WAS
persisted, `overlap`, is `decision.render_overlap`, a different quantity that
is 0.0 on those exits. #937 (that branch compares an overlap fraction against
`min_new_voxel_frac` rather than `min_overlap`) therefore cannot be settled
from any existing artifact.

Consumer note, restated here because it is easy to get wrong: `score` is
EXIT-DEPENDENT. It is the DB overlap fraction on the two reasons above and a
render/heuristic score elsewhere. Always filter rows by `reason` before
aggregating `score`.

This test is a source-level guard rather than a behavioural one because the
whitelist lives inline in a ~10k-line training loop with no injectable seam.
It asserts the invariant that actually matters: the persisted row's key set
covers the stats builders' key set.
"""

from __future__ import annotations

import ast
import os
from typing import List, Set

import pytest

_TRAIN_STREAMING = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "train_streaming.py",
)

# Keys #937 specifically restored. Named explicitly so the test fails loudly
# on the exact regression, not just on a generic set-difference.
_ISSUE_937_KEYS = ("score", "cheap_score", "forced", "moved", "novel")


def _module_ast() -> ast.Module:
    with open(_TRAIN_STREAMING, "r", encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=_TRAIN_STREAMING)


def _sidecar_row_keys(tree: ast.Module) -> Set[str]:
    """Key set of the row actually written to keyframe_admission_log.jsonl.

    That is: the base `_admit_log_row = {...}` literal, plus the whitelist
    tuple of the `for _k in (...)` copy loop that follows it. The copy loop is
    identified by containing "quality_weight", which distinguishes it from the
    unrelated TensorBoard scalar loop that also iterates over `_k`.
    """
    base: Set[str] = set()
    whitelist: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            if (
                targets
                and targets[0].id == "_admit_log_row"
                and isinstance(node.value, ast.Dict)
            ):
                for k in node.value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        base.add(k.value)

        if isinstance(node, ast.For):
            target = node.target
            if not (isinstance(target, ast.Name) and target.id == "_k"):
                continue
            if not isinstance(node.iter, ast.Tuple):
                continue
            keys = {
                e.value
                for e in node.iter.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
            if "quality_weight" in keys:
                whitelist |= keys

    assert base, "could not locate the _admit_log_row base dict literal"
    assert whitelist, "could not locate the sidecar whitelist copy loop"
    return base | whitelist


def _stats_builder_key_sets(tree: ast.Module) -> List[Set[str]]:
    """Key sets of the admission-stats dict literals.

    Identified by carrying BOTH "cheap_score" and "reason" -- the signature of
    the per-decision stats dicts assembled in the admission closure.
    """
    found: List[Set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {
            k.value
            for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        if {"cheap_score", "reason"} <= keys:
            found.append(keys)
    return found


def test_sidecar_persists_the_issue937_fields() -> None:
    persisted = _sidecar_row_keys(_module_ast())
    missing = [k for k in _ISSUE_937_KEYS if k not in persisted]
    assert not missing, (
        f"keyframe_admission_log.jsonl would silently drop {missing}. "
        "These are produced by the admission-stats builders; `score` in "
        "particular is the ONLY record of the observation-DB overlap fraction "
        "on the db_diverse_revisit / rej_redundant_angle exits (see #937). "
        "Add them to the whitelist copy loop in train_streaming.py."
    )


def test_sidecar_persists_the_issue928_precedence_chain() -> None:
    """#928's field, pinned here so the same class of drop cannot recur."""
    assert "precedence" in _sidecar_row_keys(_module_ast())


def test_no_stats_builder_key_is_silently_dropped() -> None:
    """The general invariant, of which #928 and #937 were both instances."""
    tree = _module_ast()
    persisted = _sidecar_row_keys(tree)
    builders = _stats_builder_key_sets(tree)
    if not builders:
        pytest.skip("admission-stats dict literals not locatable by signature")

    produced: Set[str] = set()
    for keys in builders:
        produced |= keys

    # "mode"/"admitted"/"reason" are in the base row under their own names;
    # everything else must be whitelisted.
    dropped = sorted(produced - persisted)
    assert not dropped, (
        "admission-stats fields computed but never persisted to "
        f"keyframe_admission_log.jsonl: {dropped}. Either whitelist them in "
        "train_streaming.py's sidecar writer, or stop computing them."
    )
