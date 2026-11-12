#!/usr/bin/env python3
"""Dump ALL current argparse defaults to a single grouped YAML — the run-config
"at a glance" reference.

argparse (arguments/__init__.py) is the source of truth for defaults; this snapshot
makes the full set visible in one file. Regenerate after changing any default:

    python3 scripts/dump_default_config.py > configs/profiles/defaults.reference.yml
"""
from __future__ import annotations
import sys
from argparse import ArgumentParser
from pathlib import Path

import yaml

# Running this file as a script puts scripts/ (not the repo root) on sys.path, so
# `import arguments` fails with the documented invocation above unless we add the
# repo root ourselves. (Confirmed 2026-08-12: the docstring's command raised
# ModuleNotFoundError until this was added -- see issue #182.)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arguments import ModelParams, PipelineParams, OptimizationParams, StreamingParams

#: Path of the committed snapshot this script generates. The freshness test
#: (tests/test_defaults_reference_freshness.py) compares it against a live render.
REFERENCE_PATH = _REPO_ROOT / "configs" / "profiles" / "defaults.reference.yml"

REGEN_COMMAND = (
    "python3 scripts/dump_default_config.py > configs/profiles/defaults.reference.yml"
)

_GROUPS = [
    ("ModelParams", ModelParams),
    ("PipelineParams", PipelineParams),
    ("OptimizationParams", OptimizationParams),
    ("StreamingParams", StreamingParams),
]


def _group_keys(group_cls) -> list[str]:
    """The arg names a ParamGroup declares (strip leading-underscore shorthand)."""
    probe = ArgumentParser()
    inst = group_cls(probe)
    keys = []
    for k in vars(inst):
        keys.append(k[1:] if k.startswith("_") else k)
    return keys


def collect_defaults() -> dict:
    """The flat {arg_name: default} mapping argparse currently declares."""
    parser = ArgumentParser()
    for _, cls in _GROUPS:
        cls(parser)
    return vars(parser.parse_args([]))


def render_defaults_yaml() -> str:
    """The exact text `main()` prints, as a string (one trailing newline).

    Factored out of main() so the freshness test can render the current
    argparse state without shelling out.
    """
    defaults = collect_defaults()

    out_lines = [
        "# GENERATED — full argparse default snapshot (run-config at a glance).",
        "# Source of truth: arguments/__init__.py. Do NOT hand-edit; regenerate with",
        "#   python3 scripts/dump_default_config.py > configs/profiles/defaults.reference.yml",
        "",
    ]
    seen = set()
    for name, cls in _GROUPS:
        keys = [k for k in _group_keys(cls) if k in defaults and k not in seen]
        if not keys:
            continue
        out_lines.append(f"# ==== {name} " + "=" * (60 - len(name)))
        block = {k: defaults[k] for k in sorted(keys)}
        out_lines.append(yaml.safe_dump(block, default_flow_style=False, sort_keys=True).rstrip())
        out_lines.append("")
        seen.update(keys)
    # out_lines deliberately ends in an empty element, so join() itself emits
    # exactly one terminal newline. Do not append a second one: the freshness
    # contract and this function's docstring both promise one trailing newline.
    return "\n".join(out_lines)


def main() -> int:
    sys.stdout.write(render_defaults_yaml())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
