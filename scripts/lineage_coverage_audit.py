#!/usr/bin/env python3
"""Audit which run directories carry ANALYZABLE per-Gaussian lineage (issue #346).

Why this exists
---------------
Every #346 sub-issue that measures the seed cohort (#283 cohort tracking, #286,
#287's intrinsic-drift analysis, #288) needs the EXACT bootstrap cohort, which
``utils.gaussian_lineage.seed_cohort_mask`` will only produce from a checkpoint
carrying (a) the full per-Gaussian lineage buffer set and (b) the
``seed_uid_range`` marker. Neither is discoverable from any run artifact other
than the ``chkpnt*.pth`` archive itself, so sessions kept re-discovering by hand
that "only N of M runs are measurable" -- and disagreeing on N, because the
criterion was never written down.

This script writes the criterion down, executably, and emits a manifest so a
future session can read the answer instead of rediscovering it. The bundled
manifest is ``docs/architecture/lineage_coverage_manifest.json``.

RECURSIVE DISCOVERY (fixed 2026-08-13, issue #432) -- read this before quoting
a coverage number
-------------------------------------------------------------------------------
Earlier versions of this script scanned ``output_root`` exactly one level
deep, so a run directory nested in a SUBdirectory of it was invisible: it
collapsed into a single ``no_checkpoint`` row for the subdirectory itself.
That is why #411's manifest read "0 of 174 trustworthy" over ``output/``
while 9 ``trustworthy`` runs already existed under ``output/scratch/``
(issue #416) -- PR #429 caught and hand-corrected that ONE invocation but
left the script itself unfixed, so any future single-root sweep would
silently reproduce the exact same wrong answer (issue #432, the third
recurrence of this defect class in three cycles).

The script now discovers run directories **recursively, at any depth**,
keyed on a real run-dir marker (``cfg_args``, ``train.log``, or at least one
``chkpnt*.pth`` -- see ``_is_run_dir``), not on an assumed nesting depth. A
directory is classified as a run the moment it carries a marker, and the
walk does NOT recurse further into it -- a run's own artifact subdirectories
(``comparison/iter_N/``, ``point_cloud/iteration_N/``) can themselves contain
a ``report.json``/etc. but never a run-root marker, so they cannot spuriously
become extra rows. A directory with no marker is a container and gets walked
into; a container with no subdirectories at all is a dead leaf, recorded
separately (not silently dropped) so a reconciliation can see it.

One invocation over ``output`` now finds everything ``output/`` and
``output/scratch/`` used to need two separate invocations (and two separate
manifest files) for:

    python scripts/lineage_coverage_audit.py output \
        -o docs/architecture/lineage_coverage_manifest.json

Each row's ``run`` key is now a root-relative path (e.g. ``scratch/foo``),
not a bare basename -- recursion means basenames can collide across
subdirectories, and issue #432 explicitly asked for a path-qualified row
instead of two same-shaped manifests.

**Loud failure, not a silent clean bill of health.** If discovery finds
ZERO run directories under the given root -- almost always a sign the root
itself is wrong, not that there really are no runs -- the script refuses to
emit a manifest and exits nonzero with an explicit error, rather than
printing an empty-but-plausible-looking summary. The manifest's top-level
``discovery`` block also records every container directory traversed and
every empty leaf directory skipped, so a reconciliation against a prior
manifest's row count doesn't require re-running the scan.

Note also that ``seed_uid_range``, ``n_gaussians`` and the per-buffer lists live
under each row's ``detail`` key, not at the row's top level; a ``trustworthy``
row does record exactly what made it trustworthy.

Classification (per run directory, judged on its NUMERICALLY-largest checkpoint)
-------------------------------------------------------------------------------
``no_checkpoint``        no ``chkpnt*.pth`` at all -- not a training run dir, or
                         checkpoints were never written/were deleted.
``no_lineage``           checkpoint predates per-Gaussian lineage (PR #145,
                         2026-08-12). Not analyzable, not recoverable.
``lineage_no_seed_range``  full lineage buffers, but no ``seed_uid_range`` key
                         (predates #283/PR #373, 2026-08-13). ``restore()``
                         invalidates the cohort, so ``seed_cohort_mask`` raises.
                         Partially recoverable in principle -- see the caveat in
                         ``--help``'s epilog -- but NOT by the shipped code, and
                         this script does not bless any such derivation.
``trustworthy``          lineage buffers all present at the right length AND a
                         non-None ``seed_uid_range``. Analyzable.

Tier 1 is a cheap byte scan of the archive's pickle (no tensors deserialized);
Tier 2 loads only the Tier-1 hits and checks the same four conditions the real
code enforces, with the buffer names taken from ``utils.gaussian_lineage`` so
this file cannot drift from the checker.

Usage
-----
    python scripts/lineage_coverage_audit.py <output_root> [-o manifest.json]
    python scripts/lineage_coverage_audit.py <output_root> --no-load   # tier 1 only

Read-only: never writes into any run directory.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import zipfile

_CHKPNT_RE = re.compile(r"^chkpnt(\d+)\.pth$")

# Imported lazily in _tier2 so tier 1 works without torch installed.
_LINEAGE_BUFFERS = ("gaussian_uid", "parent_uid", "root_uid", "generation",
                    "birth_method", "last_reseed_iter", "birth_iter")

# Markers that identify a directory as a training run's OWN root (never as
# one of its artifact subdirectories -- comparison/iter_N/ carries
# report.json but never these). Keep in sync with the "Output Structure"
# section of CLAUDE.md.
_RUN_DIR_MARKER_FILES = ("cfg_args", "train.log")

# Hard backstop against pathological/very deep trees (independent of the
# visited-inode symlink-cycle guard in discover_run_dirs -- see its
# docstring). Output trees in this repo are never more than a handful of
# levels deep in practice.
_MAX_WALK_DEPTH = 64


def _is_run_dir(path: str) -> bool:
    """True iff ``path`` carries a run-root marker: ``cfg_args``,
    ``train.log``, or at least one ``chkpnt*.pth``. Discovery treats this as
    a leaf -- it never recurses below a directory once this returns True, so
    a run's own artifact subdirectories cannot spuriously become extra rows.
    """
    try:
        names = os.listdir(path)
    except OSError:
        return False
    if any(n in names for n in _RUN_DIR_MARKER_FILES):
        return True
    return any(_CHKPNT_RE.match(n) for n in names)


def discover_run_dirs(root: str) -> tuple:
    """Recursively find run directories under ``root`` at ANY depth.

    Returns ``(run_rel_paths, containers, skipped_empty_leaves)``:

    - ``run_rel_paths``: root-relative paths (``""`` if ``root`` itself is a
      run dir, POSIX-style separators otherwise) of every discovered run dir.
    - ``containers``: root-relative paths of every non-run directory that was
      walked into (i.e. it held at least one subdirectory).
    - ``skipped_empty_leaves``: root-relative paths of non-run directories
      with no subdirectories at all -- a dead leaf, recorded rather than
      silently dropped so a reconciliation can see it explicitly.

    A directory is classified as a run the moment ``_is_run_dir`` is True;
    recursion stops there. Anything else is walked into.

    Guards against symlink cycles: ``os.listdir``/``os.path.isdir`` follow
    symlinks blindly, and this tree is known to contain real symlinks (e.g.
    ``output/scratch`` into a shared mount) -- a symlink pointing back at an
    ancestor directory would otherwise recurse until the process dies. Every
    directory actually descended into is tracked by its ``(st_dev, st_ino)``
    pair; a repeat is skipped rather than re-walked. A hard depth cap
    (``_MAX_WALK_DEPTH``) is a second, independent backstop in case some
    filesystem/mount combination ever makes distinct paths report identical
    inode pairs (e.g. certain network filesystems).
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"not a directory: {root!r}")

    run_dirs: list = []
    containers: list = []
    skipped: list = []
    visited: set = set()

    def _walk(path: str, rel: str, depth: int) -> None:
        if _is_run_dir(path):
            run_dirs.append(rel)
            return
        try:
            st = os.stat(path)
            key = (st.st_dev, st.st_ino)
        except OSError:
            skipped.append(rel)
            return
        if key in visited:
            skipped.append(rel)
            return
        visited.add(key)
        if depth >= _MAX_WALK_DEPTH:
            skipped.append(rel)
            return
        try:
            subdirs = sorted(
                e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))
            )
        except OSError:
            subdirs = []
        if not subdirs:
            skipped.append(rel)
            return
        containers.append(rel)
        for e in subdirs:
            child_rel = f"{rel}/{e}" if rel else e
            _walk(os.path.join(path, e), child_rel, depth + 1)

    _walk(root, "", 0)
    return run_dirs, containers, skipped


def latest_checkpoint(run_dir: str):
    """NUMERICALLY-largest ``chkpnt<N>.pth`` in ``run_dir`` (never lexical)."""
    best = None
    try:
        names = os.listdir(run_dir)
    except OSError:
        return None
    for name in names:
        m = _CHKPNT_RE.match(name)
        if not m:
            continue
        n = int(m.group(1))
        if best is None or n > best[0]:
            best = (n, os.path.join(run_dir, name))
    return best


def _tier1(path: str) -> dict:
    """Byte-scan the checkpoint pickle for lineage key NAMES. No tensors read."""
    with zipfile.ZipFile(path) as z:
        pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
        blob = z.read(pkl)
    keys = {name: (name.encode() in blob) for name in _LINEAGE_BUFFERS}
    keys["seed_uid_range"] = b"seed_uid_range" in blob
    keys["streaming_state"] = b"streaming_state" in blob
    return keys


def _tier2(path: str) -> dict:
    """Load the checkpoint (CPU) and apply the REAL trustworthiness conditions.

    Mirrors ``GsplatGaussianModel.restore`` -> ``_restore_seed_uid_range`` and
    ``utils.gaussian_lineage.seed_cohort_mask``: a determinable cohort needs
    streaming_state, a non-None saved ``seed_uid_range``, and every lineage
    buffer present at exactly the population length.
    """
    import torch  # noqa: PLC0415 -- optional for tier 1

    obj = torch.load(path, map_location="cpu", weights_only=False)
    model_args = obj[0] if isinstance(obj, (tuple, list)) else obj
    if not isinstance(model_args, dict):
        return {"loadable": False, "why": f"unexpected checkpoint payload type {type(model_args)!r}"}
    params = model_args.get("params") or {}
    means = params.get("means")
    n = int(means.shape[0]) if means is not None else 0
    streaming = model_args.get("streaming_state") or {}
    saved_range = model_args.get("seed_uid_range", None)
    present, wrong_len = [], []
    for name in _LINEAGE_BUFFERS:
        buf = streaming.get(name)
        if buf is None:
            continue
        present.append(name)
        if int(buf.shape[0]) != n:
            wrong_len.append(f"{name}({int(buf.shape[0])}!={n})")
    return {
        "loadable": True,
        "n_gaussians": n,
        "has_streaming_state": bool(streaming),
        "seed_uid_range": list(saved_range) if saved_range is not None else None,
        "lineage_buffers_present": present,
        "lineage_buffers_missing": [b for b in _LINEAGE_BUFFERS if b not in present],
        "lineage_buffers_wrong_length": wrong_len,
    }


def classify(tier1: dict, tier2) -> tuple:
    """-> (status, reason). ``tier2`` may be None when --no-load was passed."""
    if not tier1.get("birth_method"):
        return "no_lineage", (
            "checkpoint carries no per-Gaussian lineage buffers (predates PR #145, "
            "2026-08-12); the bootstrap cohort is not recorded anywhere in it"
        )
    if not tier1.get("seed_uid_range"):
        return "lineage_no_seed_range", (
            "lineage buffers present but no 'seed_uid_range' key (predates #283/PR "
            "#373, 2026-08-13); restore() invalidates the cohort, so "
            "seed_cohort_mask() raises SeedCohortUnavailable"
        )
    if tier2 is None:
        return "trustworthy_unverified", "tier-1 keys look complete; not loaded (--no-load)"
    if not tier2.get("loadable"):
        return "unreadable", tier2.get("why", "could not load")
    if not tier2["has_streaming_state"]:
        return "lineage_no_seed_range", "no streaming_state, so no lineage would be restored"
    if tier2["seed_uid_range"] is None:
        return "lineage_no_seed_range", "saved seed_uid_range is None (recorded as UNDETERMINED at capture time)"
    if tier2["lineage_buffers_missing"]:
        return "lineage_incomplete", f"missing buffers: {tier2['lineage_buffers_missing']}"
    if tier2["lineage_buffers_wrong_length"]:
        return "lineage_incomplete", f"buffers out of lockstep with N: {tier2['lineage_buffers_wrong_length']}"
    return "trustworthy", ""


def audit(output_root: str, load: bool = True) -> dict:
    run_rel_paths, containers, skipped_leaves = discover_run_dirs(output_root)

    if not run_rel_paths:
        raise RuntimeError(
            f"lineage_coverage_audit: found ZERO run directories under {output_root!r} "
            f"(walked {len(containers)} container dir(s), skipped {len(skipped_leaves)} "
            "empty leaf dir(s) with no subdirectories and no run marker). This almost "
            "certainly means the root is wrong or the tree is unexpectedly shaped, not "
            "that there genuinely are no runs -- refusing to emit a manifest that would "
            "silently read as a clean/empty result. See issue #432."
        )

    runs = []
    for rel in sorted(run_rel_paths):
        name = rel if rel else os.path.basename(os.path.abspath(output_root))
        run_dir = os.path.join(output_root, rel) if rel else output_root
        best = latest_checkpoint(run_dir)
        if best is None:
            runs.append({"run": name, "status": "no_checkpoint", "reason": "no chkpnt*.pth"})
            continue
        iters, ckpt = best
        row = {
            "run": name,
            "checkpoint": os.path.basename(ckpt),
            "checkpoint_iteration": iters,
            # UTC, explicit suffix (issue #443 part 3): datetime.fromtimestamp()
            # with no tz arg converts the (tz-agnostic) epoch mtime to whatever
            # local timezone the EXECUTING process/OS happens to be in -- e.g.
            # a container running UTC vs. a host running CEST/CET -- so the
            # exact same on-disk file's mtime can print two different wall-clock
            # strings depending only on where the audit was run, not on any real
            # difference in the artifact. Confirmed on #438's own regenerated
            # manifest: every row's checkpoint_mtime shifted by a uniform 2
            # hours between the old and new manifest, which is diagnostic of a
            # tz-conversion difference between the two generating environments,
            # not real mtime changes. Pinning to UTC with an explicit '+00:00'
            # suffix makes future manifests unambiguous regardless of where they
            # are generated; it does not retroactively correct any
            # already-recorded naive-local-time value.
            "checkpoint_mtime": _dt.datetime.fromtimestamp(
                os.path.getmtime(ckpt), tz=_dt.timezone.utc
            ).isoformat(timespec="seconds"),
        }
        try:
            t1 = _tier1(ckpt)
        except Exception as exc:
            row.update(status="unreadable", reason=f"tier-1 scan failed: {exc!r}")
            runs.append(row)
            continue
        t2 = None
        if load and t1.get("birth_method") and t1.get("seed_uid_range"):
            try:
                t2 = _tier2(ckpt)
            except Exception as exc:
                t2 = {"loadable": False, "why": f"torch.load failed: {exc!r}"}
        status, reason = classify(t1, t2)
        row.update(status=status, reason=reason)
        if t2 is not None:
            row["detail"] = t2
        runs.append(row)

    counts: dict = {}
    for r in runs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {
        # Same UTC-pinning rationale as checkpoint_mtime above.
        "generated": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
        "output_root": os.path.abspath(output_root),
        "criterion": (
            "A run is 'trustworthy' iff its numerically-largest chkpnt*.pth carries "
            "streaming_state, a non-None seed_uid_range, and all of "
            f"{list(_LINEAGE_BUFFERS)} at length == N. This is exactly what "
            "GsplatGaussianModel.restore() -> utils.gaussian_lineage.seed_cohort_mask() "
            "requires; anything else raises SeedCohortUnavailable."
        ),
        "discovery": {
            "note": (
                "Run dirs are discovered recursively at any depth (see discover_run_dirs); "
                "'run' keys below are root-relative paths, not bare basenames."
            ),
            "containers_traversed": containers,
            "n_containers_traversed": len(containers),
            "empty_leaf_dirs_skipped": skipped_leaves,
            "n_empty_leaf_dirs_skipped": len(skipped_leaves),
        },
        "n_run_dirs": len(runs),
        "n_with_checkpoint": sum(1 for r in runs if r["status"] != "no_checkpoint"),
        "counts": counts,
        "runs": runs,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Caveat on 'lineage_no_seed_range': the seed block is NOT reliably "
            "recoverable post hoc. Deriving it as 'the lowest contiguous uid block with "
            "birth_method == BOOTSTRAP' undercounts whenever a top-of-block seed slot was "
            "reused by MCMC relocation (reset_relocated_state keeps the slot's uid but "
            "rewrites its birth_method to the donor's), so the derived hi is a lower bound "
            "of unknown tightness. Any analysis built on such a derivation must say so."
        ),
    )
    ap.add_argument("output_root", help="directory containing run directories (e.g. output/)")
    ap.add_argument("-o", "--out", default=None, help="write the manifest JSON here")
    ap.add_argument("--no-load", action="store_true",
                    help="tier-1 byte scan only; do not torch.load anything")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.output_root):
        print(f"lineage_coverage_audit: not a directory: {args.output_root!r}", file=sys.stderr)
        return 1

    try:
        manifest = audit(args.output_root, load=not args.no_load)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    disc = manifest["discovery"]
    print(
        f"scanned {disc['n_containers_traversed']} container dir(s), "
        f"skipped {disc['n_empty_leaf_dirs_skipped']} empty leaf dir(s), "
        f"found {manifest['n_run_dirs']} run dir(s) under {manifest['output_root']}"
    )
    for status, n in sorted(manifest["counts"].items(), key=lambda kv: -kv[1]):
        print(f"{n:5d}  {status}")
    print(f"{manifest['n_with_checkpoint']} of {manifest['n_run_dirs']} run dirs have a checkpoint")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
