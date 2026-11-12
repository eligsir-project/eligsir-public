#!/usr/bin/env python3
"""Report the JUnit results of the manual training-nogpu qualification surface.

Required PR tests now run in CPU-Torch (see tests/README.md). A torch-import scan
cannot establish which tests another CI run executed, so this report makes no
cross-surface coverage claims. It reads existing results; it never scans source
or imports conftest. Pytest's exit status, not this best-effort report, is the gate.
"""
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def classname_to_file(classname: str) -> str:
    """Map a JUnit classname to its test module, trimming any test-class suffix."""
    parts = [p for p in classname.split(".") if p]
    while parts and not parts[-1].startswith("test_"):
        parts.pop()
    return "/".join(parts) + ".py" if parts else ""


def parse_junit(path: Path) -> list[dict]:
    """Flatten JUnit testcases, preserving failures, errors and visible skips."""
    root = ET.parse(path).getroot()
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError(f"Not a JUnit testsuite(s) document: {root.tag}")
    cases = []
    for case in root.iter("testcase"):
        outcome = "passed"
        for tag, label in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
            if case.find(tag) is not None:
                outcome = label
                break
        classname = case.get("classname", "")
        file = classname_to_file(classname)
        name = case.get("name", "")
        cases.append({
            "classname": classname, "name": name, "file": file,
            "outcome": outcome, "nodeid": f"{file}::{name}",
        })
    return cases


def build_report(cases: list[dict], image: str, summary_line: str | None) -> tuple[str, dict]:
    """Summarize only recorded outcomes; no executed cases is not green coverage."""
    counts = Counter(c["outcome"] for c in cases)
    stats = {
        "surface": "training-nogpu", "image": image, "total": len(cases),
        "passed": counts["passed"], "failed": counts["failed"],
        "errors": counts["error"], "skipped": counts["skipped"],
        "executed": len(cases) - counts["skipped"], "summary_line": summary_line,
    }
    bad = [c for c in cases if c["outcome"] in {"failed", "error"}]
    lines = [
        "## Training-image test surface (`training-nogpu`)", "",
        "| field | value |", "|---|---|",
        "| surface | `training-nogpu` |", f"| image | `{image}` |",
        f"| pytest summary | `{summary_line or '<unparsed>'}` |",
        f"| executed (passed+failed+errors) | **{stats['executed']}** |",
        f"| passed / failed / errors / skipped | {stats['passed']} / **{stats['failed']}** / "
        f"**{stats['errors']}** / {stats['skipped']} |", "",
        "These are recorded outcomes from this run, not evidence of coverage unique "
        "to this surface or native GPU qualification. Required PR coverage runs in "
        "CPU-Torch; see `tests/README.md`.", "",
    ]
    if bad:
        lines += [f"### {len(bad)} failing / erroring test(s)", "",
                  "No allowlist filters these results.", "",
                  "| test | outcome |", "|---|---|"]
        for case in sorted(bad, key=lambda c: c["nodeid"])[:100]:
            lines.append(f"| `{case['nodeid']}` | {case['outcome']} |")
        if len(bad) > 100:
            lines.append(f"| … {len(bad) - 100} more | |")
    elif not stats["executed"]:
        lines.append("### No executed tests recorded; this is not passing coverage.")
    else:
        lines.append("### No failures or errors on this surface.")
    return "\n".join(lines) + "\n", stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit-xml", type=Path, required=True)
    parser.add_argument("--pytest-output", type=Path, default=None)
    parser.add_argument("--image", default="(unknown)")
    parser.add_argument("--summary-file", type=Path,
                        default=Path(os.environ["GITHUB_STEP_SUMMARY"])
                        if os.environ.get("GITHUB_STEP_SUMMARY") else None)
    args = parser.parse_args()
    summary_line = None
    if args.pytest_output is not None:
        try:
            from report_ci_test_exclusions import parse_executed_summary

            summary_line = parse_executed_summary(args.pytest_output)
        except Exception as exc:  # Reporting must not replace pytest's verdict.
            print(f"::warning::report_training_surface: could not parse pytest summary ({exc!r})")
    try:
        cases = parse_junit(args.junit_xml)
        markdown, stats = build_report(cases, args.image, summary_line)
    except Exception as exc:
        print(f"::warning::report_training_surface: no usable JUnit results ({exc!r}). "
              "No coverage figures for this run.")
        return 0
    print(markdown)
    if args.summary_file is not None:
        try:
            with args.summary_file.open("a", encoding="utf-8") as stream:
                stream.write(markdown + "\n")
        except OSError as exc:
            print(f"::warning::report_training_surface: could not write job summary ({exc!r})")
    print(
        f"::notice::training-nogpu surface: {stats['executed']} tests executed "
        f"({stats['passed']} passed, {stats['failed']} failed, {stats['errors']} errors, "
        f"{stats['skipped']} skipped)."
    )
    if stats["failed"] or stats["errors"]:
        print(f"::error::training-nogpu surface is RED: {stats['failed']} failed, "
              f"{stats['errors']} errors. See the job summary for the node IDs.")
    elif not stats["executed"]:
        print("::warning::training-nogpu surface: no executed tests recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
