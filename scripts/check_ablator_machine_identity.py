#!/usr/bin/env python3
"""Read-only checker for ablator machine-identity resolution (issue #341).

Motivation: rtx3090's ablator machine identity (`[machines.rtx3090]`
hostname_patterns) and its `repo_cwd` (the real local eligsir checkout
path on that host) were found to exist only in a hand-edited, untracked
`~/.config/ablator/config.toml` on that machine -- not reviewable, not
diffable, and silently reverted by any future reimage or config
regeneration. This script closes the *detection* gap (not the deployment
gap -- this repo cannot SSH into or modify another machine's state, see
issue #341's own constraints): it reimplements ablator's hostname-pattern
resolution algorithm locally (so it works even where the `ablator` package
isn't pip-installed) and can:

  1. Resolve which `[machines.*]` block a given (or the current) hostname
     would match, using the tracked `configs/ablator.toml` -- and loudly
     flag when that resolution went through the FALLBACK path (no/`"*"`
     hostname_patterns) rather than an explicit pattern match, which is
     exactly the masquerade class that hit rtx3090 (see
     `docs/architecture/2026-08-13-issue341-...md`).
  2. Flag when MORE THAN ONE `[machines.*]` block is fallback-eligible --
     resolution then silently depends on TOML table order, a second latent
     instance of the same bug class found investigating this issue
     (`[machines.a100cluster]` before this fix).
  3. Report the resolved machine's `repo_cwd` annotation (if present).
  4. Diff a deployed config.toml (e.g. rtx3090's real
     ~/.config/ablator/config.toml, read locally on that host) against the
     tracked `configs/ablator.toml`, reporting any machine-identity or
     repo_cwd divergence -- the direct, read-only detector for the actual
     failure mode issue #341 describes.

This script deliberately duplicates (rather than imports) ablator's
`machine_name()` algorithm from `ablator/src/ablator/config.py`: the
`ablator` submodule is not guaranteed to be checked out/importable in
every environment this runs in (e.g. CI, or a worktree that hasn't run
`git submodule update --init`), and this checker's whole point is to be
usable without that dependency. `tests/test_check_ablator_machine_identity.py`
pins this duplication against the real submodule source so drift between
the two is caught, not silent.

Usage:
    python scripts/check_ablator_machine_identity.py [--hostname HOST]
        [--config configs/ablator.toml] [--deployed PATH]
"""
from __future__ import annotations

import argparse
import fnmatch
import socket
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "ablator.toml"


def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_machine_name(cfg: dict, hostname: str) -> tuple[str, str, list[str]]:
    """Mirrors ablator's config.machine_name() exactly (see module docstring
    for why this is duplicated rather than imported).

    Returns (resolved_name, mechanism, fallback_candidates) where mechanism
    is "explicit_match" or "fallback", and fallback_candidates lists every
    machine name that is fallback-eligible (no hostname_patterns, or
    hostname_patterns == ["*"]), in dict/file order -- ANY entry after the
    first here is a latent ordering hazard, per the module docstring's
    item 2.
    """
    host = hostname.lower()
    machines = cfg.get("machines", {})
    fallback_candidates: list[str] = []
    fallback: str | None = None
    for name, m in machines.items():
        pats = m.get("hostname_patterns", [])
        if not pats or pats == ["*"]:
            fallback_candidates.append(name)
            fallback = fallback or name
            continue
        if any(fnmatch.fnmatch(host, p.lower()) for p in pats):
            return name, "explicit_match", fallback_candidates
    return (fallback or "unknown"), "fallback", fallback_candidates


def check_resolution(cfg: dict, hostname: str, label: str) -> list[str]:
    """Returns a list of warning strings (empty if all clear)."""
    warnings: list[str] = []
    name, mechanism, fallback_candidates = resolve_machine_name(cfg, hostname)
    print(f"[{label}] hostname {hostname!r} -> machine {name!r} "
          f"(via {mechanism})")

    if len(fallback_candidates) > 1:
        warnings.append(
            f"[{label}] AMBIGUOUS FALLBACK: {len(fallback_candidates)} "
            f"machine blocks are fallback-eligible (no hostname_patterns, "
            f"or [\"*\"]): {fallback_candidates}. Resolution for any "
            f"hostname that doesn't explicitly match silently depends on "
            f"TOML table order -- only the first ({fallback_candidates[0]!r}) "
            f"is ever actually chosen today, but reordering the file (or "
            f"adding/removing an explicit pattern elsewhere) would change "
            f"this with no warning. Give every non-primary machine block "
            f"an explicit, real (or permanently non-matching, for "
            f"non-SSH-reachable backends like k8s) hostname_patterns entry."
        )
    if mechanism == "fallback" and name != "unknown":
        mcfg = cfg.get("machines", {}).get(name, {})
        if mcfg.get("hostname_patterns") not in (None, [], ["*"]):
            # Shouldn't happen given resolve_machine_name's own logic, but
            # defensive: a machine resolved as fallback must actually be
            # fallback-eligible.
            warnings.append(
                f"[{label}] INTERNAL: {name!r} resolved via fallback but "
                f"has real hostname_patterns {mcfg.get('hostname_patterns')!r}"
            )

    machine_cfg = cfg.get("machines", {}).get(name, {})
    repo_cwd = machine_cfg.get("repo_cwd")
    if repo_cwd:
        print(f"[{label}] machine {name!r} repo_cwd = {repo_cwd!r}")
        p = Path(repo_cwd)
        if p.is_dir():
            if not (p / ".git").exists():
                warnings.append(
                    f"[{label}] repo_cwd {repo_cwd!r} exists locally but "
                    f"has no .git -- not a git checkout"
                )
        # else: repo_cwd for a REMOTE machine won't exist on this host;
        # that's expected and not itself a warning.
    else:
        print(f"[{label}] machine {name!r} has no tracked repo_cwd annotation")

    return warnings


def diff_configs(tracked: dict, deployed: dict) -> list[str]:
    """Structural diff of the parts of the config that determine machine
    identity and repo_cwd -- the exact mechanism issue #341 found silently
    drifting. Returns a list of human-readable divergence descriptions."""
    diffs: list[str] = []
    t_machines = tracked.get("machines", {})
    d_machines = deployed.get("machines", {})

    all_names = sorted(set(t_machines) | set(d_machines))
    for name in all_names:
        if name not in d_machines:
            diffs.append(f"machine {name!r} present in tracked config, "
                         f"MISSING from deployed config")
            continue
        if name not in t_machines:
            diffs.append(f"machine {name!r} present in DEPLOYED config, "
                         f"missing from tracked configs/ablator.toml "
                         f"(untracked drift)")
            continue
        t_m, d_m = t_machines[name], d_machines[name]
        for key in ("hostname_patterns", "repo_cwd", "ssh", "backend"):
            t_v, d_v = t_m.get(key), d_m.get(key)
            if t_v != d_v:
                diffs.append(f"machine {name!r} key {key!r}: "
                             f"tracked={t_v!r} deployed={d_v!r}")

    t_uf, d_uf = tracked.get("urgent_fixes"), deployed.get("urgent_fixes")
    if t_uf != d_uf:
        diffs.append(f"[urgent_fixes] differs: tracked={t_uf!r} "
                     f"deployed={d_uf!r}")

    return diffs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hostname", default=socket.gethostname(),
                    help="Hostname to resolve identity for (default: this host's)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="Tracked config (default: configs/ablator.toml)")
    ap.add_argument("--deployed", type=Path, default=None,
                    help="Deployed config to diff against --config, e.g. "
                         "~/.config/ablator/config.toml")
    args = ap.parse_args(argv)

    tracked = load_toml(args.config)
    warnings = check_resolution(tracked, args.hostname, "tracked")

    exit_code = 0
    if args.deployed is not None:
        if not args.deployed.exists():
            print(f"[deployed] {args.deployed} does not exist", file=sys.stderr)
            return 2
        deployed = load_toml(args.deployed)
        warnings += check_resolution(deployed, args.hostname, "deployed")
        diffs = diff_configs(tracked, deployed)
        if diffs:
            print(f"\n[DIVERGENCE] deployed config differs from tracked "
                  f"configs/ablator.toml in {len(diffs)} way(s):")
            for d in diffs:
                print(f"  - {d}")
            exit_code = 1
        else:
            print("\n[OK] deployed config matches tracked configs/ablator.toml "
                  "for machine identity + repo_cwd")

    if warnings:
        print(f"\n[WARNINGS] {len(warnings)}:")
        for w in warnings:
            print(f"  - {w}")
        exit_code = exit_code or 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
