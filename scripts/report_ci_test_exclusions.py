#!/usr/bin/env python3
"""CI test-exclusion visibility report (issue #536).

**The gap this closes:** issue #478/#518's ``tests/_torch_collect_scan.py`` +
``tests/conftest.py`` correctly derive which ``tests/test_*.py`` files cannot
be collected in the torch-free CPU smoke image (``.github/workflows/
ci-cpu.yml``'s ``smoke`` job) and silently ``collect_ignore`` them. That
derivation is correct and stays untouched here (see #536's "not in scope").
What was missing is any SIGNAL to a PR author that their new test file
landed in that excluded set -- PR #528 added 17 tests via a file with a
module-scope ``import torch``, the CPU suite's pass count did not move
(``1552 passed, 116 skipped`` on both `main` and the PR branch), and CI went
green. This script makes that exclusion visible on every run (option 1 of
#536) and calls it out loudly when a PR's own new test file is the cause
(option 2), without weakening or duplicating #518's derivation.

**How the excluded set is computed, and a real trap this script's first
draft fell into:** the obvious approach -- diff ``tests/test_*.py`` files
that appear in ``pytest --collect-only -q tests/`` output against every
such file on disk -- looks right but is WRONG. It conflates two very
different things:

1. **Truly invisible (collect_ignore, issue #536's actual problem):** the
   whole file is in ``tests/conftest.py``'s ``collect_ignore`` list, so
   pytest never even attempts to import it. It produces zero output
   anywhere -- not in ``--collect-only``, not in a real run's summary, not
   even a skip reason. This is the "17 tests, zero signal" failure mode.
2. **Visible module-level skip (self-guarded, NOT #536's problem):** the
   file self-guards with a module-scope ``pytest.importorskip("torch")``.
   Confirmed empirically against this image:
   ``pytest --collect-only -q tests/test_issue384_depth_loss_live_guard.py``
   prints ``no tests collected`` (indistinguishable from category 1 in
   ``--collect-only`` output!), but the real run
   (``pytest -q -rs tests/test_issue384_depth_loss_live_guard.py``) shows
   ``1 skipped`` with an explicit reason line
   (``SKIPPED [1] tests/test_issue384_depth_loss_live_guard.py:N: could not
   import 'torch': No module named 'torch'``). This is exactly the pattern
   #518's scanner is designed to recognize and NOT flag -- the file already
   has real, visible signal; #536 does not need to add any.

A collect-only diff cannot tell these apart (both show zero collected node
ids for the file). This script instead reads ``tests/conftest.py``'s own
``collect_ignore`` list directly -- READ, never modified, matching this
task's ownership boundary -- which is pytest's actual, ground-truth input
for category 1. Verified against this tree: ``len(collect_ignore) == 152``,
matching ``_torch_collect_scan.discover_torch_needing_test_files()``'s 151
torch-flagged files plus the single tensorboard-only entry, and does NOT
include ``test_issue384_depth_loss_live_guard.py`` (confirmed correctly
absent). ``_torch_collect_scan.file_requires_torch()`` (already-owned,
unmodified) is then used only to attribute a *reason* to each excluded
file (torch vs. the one non-torch entry), not to decide membership.

**Option 2 (new-test-file check):** when invoked with ``--pr-repo``/
``--pr-number``, hits the GitHub REST API
(``GET /repos/{repo}/pulls/{number}/files``) to list files this PR *adds*
(status ``added``), intersects that with the excluded set above, and emits
a ``::warning file=...::`` annotation per match plus a dedicated job-summary
section -- this is a WARNING, not a failure. See the module-level docstring
of this repo's CLAUDE.md (Run Contracts) for the general philosophy of
loud-but-non-blocking visibility; the specific reasoning for warn-over-fail
here is in this PR's description: a hard fail would block legitimate,
deliberate exclusions (a test file that must import torch for a real
reason), and the whole point of #536 is to inform the choice, not remove it.
Network/API failures here are caught and degrade to a warning annotation
of their own -- this script must never be the reason CI goes red, since it
runs with ``if: always()`` after the real test step.

**Option 3 (executed-test count, "falls out cheaply"):** parses the final
summary line pytest already printed for the real
``python -m pytest -q tests/`` run (passed to this script via
``--pytest-output``) and echoes it into the job summary next to the
exclusion count, so a reviewer sees "N executed, M excluded" side by side
without hunting through the raw log.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"

sys.path.insert(0, str(TESTS_DIR))


def _all_top_level_test_files() -> set[str]:
    return {f"tests/{p.name}" for p in TESTS_DIR.glob("test_*.py")}


def compute_excluded_files() -> set[str]:
    """The set of `tests/test_*.py` files pytest will NEVER EVEN IMPORT in
    this environment -- read directly from `tests/conftest.py`'s own
    `collect_ignore` list (ground truth for pytest's `collect_ignore`
    mechanism), not re-derived. See the module docstring's "trap" section
    for why a `--collect-only` diff is NOT equivalent to this and would
    silently misclassify self-guarded (`pytest.importorskip("torch")`)
    files as excluded when they are not.

    Imports `tests/conftest.py` as a plain module -- READ-ONLY, this
    process never writes to it and this script does not own it.
    """
    sys.path.insert(0, str(TESTS_DIR))
    import conftest  # noqa: PLC0415

    return {f"tests/{name}" for name in conftest.collect_ignore}


def reason_for(path_str: str) -> str:
    """Best-effort human-readable reason, using the already-owned #518
    scanner ONLY to attribute a reason -- membership itself comes from
    `conftest.collect_ignore` (see `compute_excluded_files`), not from this.
    """
    try:
        import _torch_collect_scan  # noqa: PLC0415

        full_path = REPO_ROOT / path_str
        if _torch_collect_scan.file_requires_torch(full_path):
            return "torch import (direct or transitive) -- see tests/_torch_collect_scan.py"
    except Exception as exc:  # pragma: no cover - best-effort attribution only
        return f"torch-scan attribution failed ({exc!r}); reason unknown"
    return "excluded for a non-torch reason (see tests/conftest.py, e.g. a missing tensorboard/matplotlib dependency)"


def parse_executed_summary(pytest_output_path: Path | None) -> str | None:
    if pytest_output_path is None or not pytest_output_path.is_file():
        return None
    text = pytest_output_path.read_text(errors="replace")
    # Last non-empty line of a `pytest -q` run is the summary, e.g.
    # "1552 passed, 116 skipped, 2 warnings in 28.16s".
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line and re.search(r"\bin [0-9.]+s\b", line):
            return line
    return None


def _is_added_top_level_test_file(entry: dict) -> bool:
    """Pure predicate, factored out of `fetch_added_test_files` so it is
    testable without any network access: does this single entry from
    `GET /pulls/{n}/files`'s JSON array represent a top-level
    `tests/test_*.py` file that this PR ADDS (not modifies, not one under a
    subdirectory like `tests/regression/`)?
    """
    filename = entry.get("filename", "")
    return (
        entry.get("status") == "added"
        and filename.startswith("tests/test_")
        and filename.endswith(".py")
        and "/" not in filename[len("tests/") :]
    )


def fetch_added_test_files(repo: str, pr_number: str, token: str | None) -> list[str] | None:
    """List of `tests/test_*.py` files this PR ADDS (status == "added"),
    via the GitHub REST API. Returns None (not []) on any failure so the
    caller can tell "checked, found none" apart from "could not check" --
    the latter must not be silently read as the former.
    """
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files?per_page=100"
    added: list[str] = []
    page = 1
    try:
        while True:
            req = urllib.request.Request(
                f"{url}&page={page}",
                headers={
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    **({"Authorization": f"Bearer {token}"} if token else {}),
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                batch = json.loads(resp.read())
            if not batch:
                break
            for entry in batch:
                if _is_added_top_level_test_file(entry):
                    added.append(entry["filename"])
            if len(batch) < 100:
                break
            page += 1
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        print(f"::warning::report_ci_test_exclusions: could not fetch PR file list ({exc!r}); "
              "skipping the new-test-exclusion check for this run.")
        return None
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytest-output", type=Path, default=None,
                         help="Path to the captured stdout of the real `pytest -q tests/` run.")
    parser.add_argument("--pr-repo", default=None, help="owner/repo, only for pull_request events.")
    parser.add_argument("--pr-number", default=None, help="PR number, only for pull_request events.")
    parser.add_argument("--summary-file", type=Path, default=Path(os.environ.get("GITHUB_STEP_SUMMARY", "")) if os.environ.get("GITHUB_STEP_SUMMARY") else None)
    args = parser.parse_args()

    excluded = compute_excluded_files()
    excluded_sorted = sorted(excluded)

    lines: list[str] = []
    lines.append("## CI test-exclusion visibility (issue #536)")
    lines.append("")
    lines.append(
        f"**{len(excluded_sorted)} of {len(_all_top_level_test_files())}** top-level "
        f"`tests/test_*.py` files are excluded from this CPU smoke run "
        f"(cannot be collected in this torch-free image)."
    )

    executed_summary = parse_executed_summary(args.pytest_output)
    if executed_summary:
        lines.append("")
        lines.append(f"**Executed this run:** `{executed_summary}`")

    if excluded_sorted:
        lines.append("")
        lines.append("<details><summary>Excluded files (click to expand)</summary>")
        lines.append("")
        lines.append("| file | likely reason |")
        lines.append("|---|---|")
        for f in excluded_sorted:
            lines.append(f"| `{f}` | {reason_for(f)} |")
        lines.append("")
        lines.append("</details>")

    print(f"::notice::CI test-exclusion visibility (#536): {len(excluded_sorted)} of "
          f"{len(_all_top_level_test_files())} tests/test_*.py files excluded from this "
          f"CPU smoke run (torch-free image). See the job summary for the full list.")

    # Option 2: warn (never fail) when a PR's own NEW test file lands in the
    # excluded set.
    if args.pr_repo and args.pr_number:
        added = fetch_added_test_files(args.pr_repo, args.pr_number, os.environ.get("GITHUB_TOKEN"))
        if added is not None:
            newly_excluded = sorted(set(added) & excluded)
            lines.append("")
            lines.append("### New test files added by this PR")
            if not newly_excluded:
                lines.append("")
                lines.append(f"None of this PR's {len(added)} new `tests/test_*.py` file(s) "
                              "are excluded from the CPU image. ✅" if added else
                              "This PR adds no new `tests/test_*.py` files.")
            else:
                lines.append("")
                lines.append(
                    f"⚠️ **{len(newly_excluded)} of this PR's {len(added)} new test file(s) "
                    "will NOT run in CPU CI:**"
                )
                lines.append("")
                for f in newly_excluded:
                    reason = reason_for(f)
                    lines.append(f"- `{f}` — {reason}")
                    print(
                        f"::warning file={f}::This new test file is excluded from CPU smoke CI "
                        f"({reason}). Pytest never even IMPORTS it here -- it produces zero output "
                        "anywhere (no pass, no skip, no error), and a green check on this job says "
                        "nothing about whether these tests pass. Two remedies, pick deliberately: "
                        "(1) make the file genuinely torch-free so it actually EXECUTES on this "
                        "image (moves the 'passed' count); (2) self-guard the torch-needing parts "
                        'with a module- or test-scope `pytest.importorskip("torch")` -- the #518 '
                        "scanner recognizes that form and stops excluding the file, so it will be "
                        "collected and show as a real, visible SKIPPED with a reason instead of "
                        "silence, but note this does NOT make it execute here (it moves the "
                        "'skipped' count, not 'passed' -- the training/ROCm image is what actually "
                        "runs it). Or (3) accept the exclusion deliberately (no action needed)."
                    )
                lines.append("")
                lines.append(
                    "Two remedies, pick deliberately: **(1)** make the file genuinely torch-free "
                    "so it actually executes here (moves the `passed` count on this job); **(2)** "
                    'self-guard with `pytest.importorskip("torch")` -- the #518 scanner recognizes '
                    "that form and stops excluding the file, so it becomes a real, visible "
                    "`SKIPPED` with a reason instead of silence, but this does **not** make it "
                    "execute on the CPU image (it moves the `skipped` count, not `passed`; only "
                    "the training/ROCm image actually runs it). Or **(3)** accept the exclusion "
                    "deliberately."
                )

    report = "\n".join(lines) + "\n"
    print(report)

    if args.summary_file is not None:
        try:
            with open(args.summary_file, "a", encoding="utf-8") as fh:
                fh.write(report)
        except OSError as exc:  # pragma: no cover - best-effort, never fatal
            print(f"::warning::report_ci_test_exclusions: could not write job summary ({exc!r})")

    # Deliberately always returns 0 -- this is a visibility-only report and
    # must never fail the job on its own (see module docstring: warn, not
    # fail-closed).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
