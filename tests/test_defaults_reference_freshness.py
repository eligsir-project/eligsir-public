"""configs/profiles/defaults.reference.yml must stay in sync with argparse.

The file claims to be generated from `arguments/__init__.py` via
`scripts/dump_default_config.py`, but nothing enforced that. It silently drifted
for ~6 weeks (last regenerated 2026-06-30; argparse changed as recently as
2026-08-12): 3 changed values, 1 removed key, and 142 options missing entirely
by the time issue #182 caught it. This test is the enforcement the issue asks
for ("add an automated freshness/parity check in CI/tests so this cannot drift
silently again").

Scope note: this checks ONLY `defaults.reference.yml`, which is unambiguously a
generated snapshot. It deliberately does NOT check `configs/profiles/default.yml`
-- issue #182 leaves that file's role undecided (curated base policy profile vs.
generated full snapshot), and asserting parity here would prejudge that decision.
"""
from __future__ import annotations

import yaml

from scripts.dump_default_config import (
    REFERENCE_PATH,
    REGEN_COMMAND,
    render_defaults_yaml,
)


def _semantic_delta(committed: dict, current: dict) -> list[str]:
    """Human-readable key-level differences, empty when the two agree."""
    lines: list[str] = []
    for key in sorted(set(current) - set(committed)):
        lines.append(f"  MISSING from reference: {key} = {current[key]!r}")
    for key in sorted(set(committed) - set(current)):
        lines.append(f"  STALE in reference (no longer an argparse arg): {key}")
    for key in sorted(set(committed) & set(current)):
        if committed[key] != current[key]:
            lines.append(
                f"  CHANGED: {key}: reference={committed[key]!r} argparse={current[key]!r}"
            )
    return lines


def test_defaults_reference_matches_argparse():
    """Generate once; exact equality also proves semantic default equality."""
    current = render_defaults_yaml()
    committed = REFERENCE_PATH.read_text()
    assert committed == current, (
        f"{REFERENCE_PATH.name} differs from generated defaults. "
        f"Regenerate with: {REGEN_COMMAND}\n"
        + ("\n".join(_semantic_delta(yaml.safe_load(committed), yaml.safe_load(current)))
           or "Values agree; formatting or YAML serialization differs.")
    )
