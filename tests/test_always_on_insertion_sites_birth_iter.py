"""I2: every always-on insertion site must state its ``birth_iter`` explicitly.

``rolling_seed_application._run_rolling_seed`` and
``submap_stitch_application._run_submap_stitch`` both insert Gaussians before
any refinement iteration exists, so ``birth_iter=0`` is *correct* there rather
than a missing value -- but relying on the parameter default makes "0 because
it's genuinely iteration 0" indistinguishable from "0 because nobody threaded
the iteration through", which is exactly the bug class this plan fixed
elsewhere. These tests pin the value as an explicit, documented choice.

CPU-only: pure source inspection, no model instantiation.
"""

from __future__ import annotations

import inspect
import re


def _insertion_call_args(source: str) -> list[str]:
    """Extract the argument text of every add_points_as_gaussians(...) /
    GaussianInsertionBatch(...) call in ``source``.

    Scoped to real call sites (matching parens from the call) rather than a
    raw substring search over the whole function body, so unrelated prose in
    a comment (e.g. one that mentions ``birth_iter=0`` while explaining a
    *different* call's ``iteration=0`` argument) cannot be miscounted as an
    insertion site. Mirrors the same brace-matching approach already used by
    ``test_no_always_on_insertion_site_omits_birth_iter`` below.
    """
    calls = []
    for match in re.finditer(r"(add_points_as_gaussians|GaussianInsertionBatch)\(", source):
        start = match.end()
        depth, i = 1, start
        while i < len(source) and depth:
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
            i += 1
        calls.append(source[start:max(i - 1, start)])
    return calls


def test_rolling_seed_insertions_state_birth_iter_explicitly():
    from eligsir.mapping.integration import rolling_seed_application

    source = inspect.getsource(rolling_seed_application._run_rolling_seed)
    calls = _insertion_call_args(source)
    # One map_backend/GaussianInsertionBatch branch + one direct
    # add_points_as_gaussians branch. (2026-08-13, issue #470 item B.2: a
    # later commit added a comment mentioning "birth_iter=0" in prose while
    # documenting an unrelated training_setup(iteration=0) call -- that is
    # not a third insertion site, so this asserts over actual call sites
    # rather than a raw text count that comment would inflate.)
    assert len(calls) == 2, (
        f"expected exactly 2 Gaussian-insertion call sites in _run_rolling_seed, found {len(calls)}"
    )
    for call_args in calls:
        assert "birth_frame=0" in call_args, (
            "both rolling-seed insertion branches must pass birth_frame explicitly"
        )
        assert "birth_iter=0" in call_args, (
            "both rolling-seed insertion branches must pass birth_iter explicitly"
        )


def test_submap_stitch_insertion_states_birth_iter_explicitly():
    from eligsir.mapping.integration import submap_stitch_application

    source = inspect.getsource(submap_stitch_application._run_submap_stitch)
    assert "birth_iter=0" in source, (
        "the submap merge insertion must pass birth_iter explicitly"
    )


def test_no_always_on_insertion_site_omits_birth_iter():
    """Guard against a NEW insertion site landing without birth_iter.

    Scans the always-on integration modules for add_points_as_gaussians /
    GaussianInsertionBatch call sites and requires a birth_iter= in each call's
    argument text.
    """
    import re
    from pathlib import Path

    import eligsir.mapping.integration as pkg

    root = Path(pkg.__file__).parent
    offenders = []
    for path in sorted(root.glob("*.py")):
        text = path.read_text()
        for match in re.finditer(r"(add_points_as_gaussians|GaussianInsertionBatch)\(", text):
            start = match.end()
            depth, i = 1, start
            while i < len(text) and depth:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                i += 1
            call_args = text[start:max(i - 1, start)]  # drop the closing paren
            if call_args.strip() in ("", "..."):
                continue  # docstring prose, e.g. ``add_points_as_gaussians(...)``
            if "birth_iter" not in call_args:
                line = text[:match.start()].count("\n") + 1
                offenders.append(f"{path.name}:{line}")
    assert not offenders, (
        "insertion call sites without an explicit birth_iter: " + ", ".join(offenders)
    )
