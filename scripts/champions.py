#!/usr/bin/env python3
"""Champion registry CLI (issue #332): list / fetch / verify.

    python scripts/champions.py list
    python scripts/champions.py fetch fr3:<champion_id> -o /path/to/dest
    python scripts/champions.py verify fr3:<champion_id>

Registry lives at artifacts/champions/index.json (small, plain-Git file)
plus one artifacts/champions/<dataset>/<scene>/<champion_id>/manifest.json
per promoted champion. Binaries themselves live as GitHub Release assets
(see docs/architecture/2026-08-13-champion-artifact-retention-decision.md)
and are never part of a normal `git clone`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eligsir.utils.champion_promotion import sha256_file  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_ROOT = os.path.join(REPO_ROOT, "artifacts", "champions")
INDEX_PATH = os.path.join(REGISTRY_ROOT, "index.json")


def load_index() -> dict:
    if os.path.isfile(INDEX_PATH):
        with open(INDEX_PATH) as f:
            return json.load(f)
    return {"schema_version": 1, "champions": []}


def load_manifest(entry: dict) -> dict:
    path = os.path.join(REPO_ROOT, entry["manifest_path"])
    with open(path) as f:
        return json.load(f)


def find_entry(index: dict, key: str) -> dict:
    """key is "<scene>:<champion_id>"."""
    if ":" not in key:
        raise SystemExit(f"expected <scene>:<champion_id>, got {key!r}")
    scene, champion_id = key.split(":", 1)
    for entry in index.get("champions", []):
        if entry["scene"] == scene and entry["champion_id"] == champion_id:
            return entry
    raise SystemExit(f"no champion found matching {key!r}")


def cmd_list(args) -> int:
    index = load_index()
    champions = index.get("champions", [])
    if not champions:
        print("(no champions promoted yet -- registry is empty)")
        return 0
    print(f"{'scene':<12} {'champion_id':<32} {'status':<12} {'promoted':<22} manifest")
    for c in champions:
        print(f"{c['scene']:<12} {c['champion_id']:<32} {c['status']:<12} "
              f"{c.get('promoted_date', '?'):<22} {c['manifest_path']}")
    return 0


def cmd_fetch(args) -> int:
    index = load_index()
    entry = find_entry(index, args.champion)
    manifest = load_manifest(entry)
    dest = args.output or os.path.join(".", "champion_downloads", manifest["champion_id"])
    os.makedirs(dest, exist_ok=True)

    if manifest["storage_backend"] != "release":
        raise SystemExit(f"fetch not implemented for storage_backend={manifest['storage_backend']!r}")
    tag = manifest["storage_locator"]
    if not tag:
        raise SystemExit(f"manifest for {args.champion} has no storage_locator (not yet uploaded?)")

    filenames = [
        rec["filename"] for rec in (manifest.get("final_ply"), manifest.get("final_checkpoint"))
        if rec
    ]
    if not filenames:
        raise SystemExit(f"manifest for {args.champion} lists no binary assets to fetch")

    print(f"Fetching {len(filenames)} asset(s) from release {tag} into {dest} ...")
    cmd = ["gh", "release", "download", tag, "-D", dest, "--clobber"]
    for name in filenames:
        cmd += ["-p", name]
    subprocess.run(cmd, check=True)

    _verify_local(manifest, dest)
    print(f"Fetched and verified {args.champion} into {dest}")
    return 0


def _verify_local(manifest: dict, dest: str) -> None:
    for key in ("final_ply", "final_checkpoint"):
        rec = manifest.get(key)
        if not rec:
            continue
        local_path = os.path.join(dest, rec["filename"])
        if not os.path.isfile(local_path):
            raise SystemExit(f"expected downloaded file missing: {local_path}")
        actual_size = os.path.getsize(local_path)
        if actual_size != rec["size_bytes"]:
            raise SystemExit(
                f"size mismatch for {rec['filename']}: expected "
                f"{rec['size_bytes']}, got {actual_size}"
            )
        actual_sha = sha256_file(local_path)
        if actual_sha != rec["sha256"]:
            raise SystemExit(
                f"SHA256 mismatch for {rec['filename']}: expected "
                f"{rec['sha256']}, got {actual_sha} -- DO NOT TRUST this file"
            )
        print(f"  verified {rec['filename']}: size + sha256 OK")


def cmd_verify(args) -> int:
    index = load_index()
    entry = find_entry(index, args.champion)
    manifest = load_manifest(entry)

    if args.local_dir:
        _verify_local(manifest, args.local_dir)
        return 0

    if manifest["storage_backend"] != "release":
        raise SystemExit(f"remote verify not implemented for storage_backend={manifest['storage_backend']!r}")
    tag = manifest["storage_locator"]
    if not tag:
        raise SystemExit(f"manifest for {args.champion} has no storage_locator (not yet uploaded?)")

    out = subprocess.run(
        ["gh", "release", "view", tag, "--json", "assets"],
        capture_output=True, text=True, check=True,
    )
    remote_assets = {a["name"]: a for a in json.loads(out.stdout).get("assets", [])}
    ok = True
    for key in ("final_ply", "final_checkpoint"):
        rec = manifest.get(key)
        if not rec:
            continue
        remote = remote_assets.get(rec["filename"])
        if remote is None:
            print(f"  MISSING remote asset: {rec['filename']}")
            ok = False
            continue
        if remote.get("size") != rec["size_bytes"]:
            print(f"  SIZE MISMATCH for {rec['filename']}: manifest={rec['size_bytes']} remote={remote.get('size')}")
            ok = False
        else:
            print(f"  {rec['filename']}: remote size matches manifest ({rec['size_bytes']} bytes). "
                  "(sha256 not checked without downloading -- pass a local dir with --local-dir "
                  "after `fetch` for a full-content check)")
    if not ok:
        return 1
    print(f"{args.champion}: remote asset sizes match manifest.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all registered champions")

    fetch_p = sub.add_parser("fetch", help="Download a champion's binary assets")
    fetch_p.add_argument("champion", help="<scene>:<champion_id>")
    fetch_p.add_argument("-o", "--output", default=None, help="Destination directory")

    verify_p = sub.add_parser("verify", help="Verify checksums/metadata for a champion")
    verify_p.add_argument("champion", help="<scene>:<champion_id>")
    verify_p.add_argument("--local-dir", default=None,
                           help="Verify already-downloaded files in this dir (full sha256 check) "
                                "instead of a remote size-only check")

    args = p.parse_args(argv)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "fetch":
        return cmd_fetch(args)
    if args.command == "verify":
        return cmd_verify(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
