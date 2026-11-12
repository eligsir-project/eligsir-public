"""Output staging: detect network filesystems, map hot-write paths to local
disk, and track which artifacts still need syncing to the persistent path.

Root cause this exists for: training runs writing directly to an NFS-backed
`-m` path (for example, a network-mounted output directory with no
SLOG/L2ARC, ~200-290ms per sync RPC) collapse from 60-98 it/s to 5-6 it/s.
This module is detection + path-mapping only; the background sync worker
that actually copies local_path -> persist_path is a separate component.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from typing import Dict, Optional, Tuple

_MANIFEST_NAME = ".staging_manifest.json"
_LOG_NAME = "train.log"

_NETWORK_FSTYPES = {"nfs", "nfs4", "cifs", "ceph", "lustre", "glusterfs"}


def _default_read_mounts() -> str:
    with open("/proc/mounts", "r") as f:
        return f.read()


def _parse_mounts(mounts_text: str):
    """Yield (mountpoint, fstype) pairs in file order.

    /proc/mounts can list the same mountpoint twice when a filesystem is
    stacked on top of another (observed on this project's own host: autofs
    triggers an entry at a network-mounted output directory, then the real nfs4
    mount appears as a second row at the identical path once automount
    fires). Later rows reflect the current, effective mount -- callers
    should prefer the last match for a given mountpoint.
    """
    for line in mounts_text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        _device, mountpoint, fstype = parts[0], parts[1], parts[2]
        yield mountpoint, fstype


def _resolve_mount(path: str, mounts_text: str) -> Optional[str]:
    """Return the fstype of the longest mount prefix matching `path`."""
    best_mountpoint = None
    best_fstype = None
    for mountpoint, fstype in _parse_mounts(mounts_text):
        if path == mountpoint or path.startswith(mountpoint.rstrip("/") + "/") or mountpoint == "/":
            if best_mountpoint is None or (
                len(mountpoint) >= len(best_mountpoint)
            ):
                best_mountpoint = mountpoint
                best_fstype = fstype
    return best_fstype


def is_network_filesystem(
    path: str,
    mounts_text: Optional[str] = None,
    _read_mounts=_default_read_mounts,
) -> bool:
    """True iff `path`'s containing mount is a network filesystem type.

    Fails safe to False on any read error or if nothing matches -- staging
    must never crash a training run over a detection quirk.
    """
    try:
        text = mounts_text if mounts_text is not None else _read_mounts()
    except OSError:
        return False

    try:
        fstype = _resolve_mount(os.path.abspath(path), text)
    except Exception:
        return False

    return fstype in _NETWORK_FSTYPES


def _is_tmpfs_mount(path: str) -> bool:
    try:
        text = _default_read_mounts()
    except OSError:
        return False
    try:
        fstype = _resolve_mount(os.path.abspath(path), text)
    except Exception:
        return False
    return fstype == "tmpfs"


def _nearest_existing_ancestor(path: str) -> str:
    current = os.path.abspath(path)
    while not os.path.exists(current):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return current


def _candidate_writable(path: str) -> bool:
    ancestor = _nearest_existing_ancestor(path)
    return os.access(ancestor, os.W_OK)


def resolve_staging_root(explicit_root: Optional[str]) -> str:
    """Pick the local staging root, in priority order, skipping any
    candidate that is unwritable or lives on a tmpfs mount.
    """
    candidates = []
    if explicit_root:
        candidates.append(explicit_root)

    env_root = os.environ.get("ELIGSIR_OUTPUT_STAGING_ROOT")
    if env_root:
        candidates.append(env_root)

    user = os.environ.get("USER", "unknown")
    candidates.append(os.path.join("/scratch", user, "eligsir"))

    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        candidates.append(os.path.join(tmpdir, "eligsir"))

    candidates.append("/tmp/eligsir")

    for candidate in candidates:
        if not _candidate_writable(candidate):
            continue
        if is_network_filesystem(candidate) or _is_tmpfs_mount(candidate):
            continue
        return candidate

    raise RuntimeError(
        "resolve_staging_root: no writable, non-tmpfs candidate found among "
        f"{candidates!r} -- refusing to silently train to a doomed location"
    )


class UnmountedPersistentPathError(RuntimeError):
    """Raised when a run's persistent output path (`-m`/`--model_path`)
    resolves to the SAME filesystem device as the container's own root
    ("/") despite living under a known shared-mount prefix -- i.e. the
    external bind mount that was supposed to back this path is silently
    absent, so anything written there lands in the container's ephemeral
    writable layer and is destroyed by `--rm` on exit, exit code 0, no
    error anywhere in between.

    Root cause this guards against (eligsir issue #293, 2026-08-13):
    rtx3090's ablator runner config lost its `-v
    ${OUTPUT_ROOT:-./output}:/output` (read-write)
    bind mount, leaving only two unrelated `:ro` subpath mounts. A job
    with `-m /output/<name>` trained a
    full 30000 iterations (464s), wrote every artifact, printed "Training
    complete." -- and every one of those artifacts vanished on container
    exit, because that path was never actually a separate mount inside
    the container; it was just an ordinary directory on the overlay root.
    Only a post-hoc `require_result_artifact` ledger check in ablator
    caught it, after the run had already burned its full budget.

    This check runs at trainer startup, before any training work, so the
    same class of misconfiguration fails loudly in well under a second
    instead of silently discarding a completed run.
    """


def _default_mount_prefixes() -> tuple:
    env = os.environ.get("ELIGSIR_PERSISTENT_MOUNT_PREFIXES")
    if env:
        return tuple(p for p in (s.strip() for s in env.split(",")) if p)
    return ("/data/", "/output/")


def assert_persistent_mount(
    path: str,
    mount_prefixes: Optional[Tuple[str, ...]] = None,
    probe_write: bool = True,
    root_dev: Optional[int] = None,
) -> None:
    """Mandatory pre-training preflight: assert `path` is genuinely backed
    by a distinct, persistent mount -- not merely writable.

    Deliberately scoped to paths under a "known shared-mount prefix" (by
    default data and output mount prefixes, suitable for common containerized NFS
    persistent/scratch share -- see CLAUDE.md's Related Repos / Run
    Contracts) to avoid false positives on ordinary relative `output/...`
    paths or paths under the repo's own `/workspace` bind mount, which are
    legitimate even though they may share the container's root device.
    Override via the ``ELIGSIR_PERSISTENT_MOUNT_PREFIXES`` env var
    (comma-separated) or the `mount_prefixes` kwarg.

    Two independent checks, both best-effort but loud on failure:

    1. **Writability probe** (``probe_write=True``): create the directory,
       write a small unique marker file, fsync, read it back, unlink it.
       Catches a genuinely read-only or broken mount outright.
    2. **Distinct-device check**: compare the `st_dev` of the path's
       nearest existing ancestor against the `st_dev` of the container
       root ("/"). If they're equal, the path is NOT a separately mounted
       volume -- it's just a directory on the container's own (ephemeral,
       `--rm`-destroyed) writable layer, which is exactly issue #293's
       failure mode. This is the check a write-success probe alone cannot
       catch, since writes to the ephemeral layer succeed just fine.

    A relative path, or an absolute path that doesn't start with any of
    `mount_prefixes`, is left untouched (returns immediately) -- this
    function is intentionally narrow, not a general-purpose "is this path
    real" check.
    """
    if not os.path.isabs(path):
        return
    prefixes = mount_prefixes if mount_prefixes is not None else _default_mount_prefixes()
    if not any(path.startswith(p) for p in prefixes):
        return

    ancestor = _nearest_existing_ancestor(path)

    if probe_write:
        probe_path = os.path.join(path, f".eligsir_mount_probe_{os.getpid()}.tmp")
        token = uuid.uuid4().hex
        try:
            os.makedirs(path, exist_ok=True)
            with open(probe_path, "w") as f:
                f.write(token)
                f.flush()
                os.fsync(f.fileno())
            with open(probe_path) as f:
                readback = f.read()
            if readback != token:
                raise UnmountedPersistentPathError(
                    f"persistent-mount preflight FAILED for {path!r}: wrote a "
                    f"probe marker but read back a different value "
                    f"({readback!r} != {token!r}) -- refusing to train into "
                    "an unreliable output path."
                )
        except OSError as e:
            raise UnmountedPersistentPathError(
                f"persistent-mount preflight FAILED for {path!r}: not "
                f"writable ({e!r}). Check the container's bind mounts for "
                "this path's expected shared-mount prefix."
            ) from e
        finally:
            try:
                os.remove(probe_path)
            except OSError:
                pass

    try:
        ancestor_dev = os.stat(ancestor).st_dev
        actual_root_dev = root_dev if root_dev is not None else os.stat("/").st_dev
    except OSError:
        # Can't determine devices -- don't block a run over an unrelated
        # stat failure; the writability probe above is the primary guard.
        return

    if ancestor_dev == actual_root_dev:
        raise UnmountedPersistentPathError(
            f"persistent-mount preflight FAILED for {path!r}: its nearest "
            f"existing ancestor {ancestor!r} lives on the SAME filesystem "
            "device as the container root (\"/\") -- this path is under a "
            f"known shared-mount prefix ({prefixes!r}) but is NOT actually "
            "a separate mounted volume inside this container. Writes here "
            "will land in the container's ephemeral writable layer and be "
            "destroyed by `--rm` on exit, exit code 0, with no error in "
            "between (this is exactly eligsir issue #293). Check the "
            "container launch's `-v` bind mounts for the missing "
            f"read-write mount of {prefixes!r} on this machine."
        )


class OutputStagingManager:
    """Maps a persistent `-m` output path to a local staging path when the
    persistent path is (or is forced to be treated as) network-backed.
    """

    def __init__(
        self,
        persist_model_path: str,
        mode: str = "auto",
        staging_root: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        if mode not in ("auto", "on", "off"):
            raise ValueError(f"mode must be one of auto/on/off, got {mode!r}")

        self.persist_path = persist_model_path
        self.run_id = run_id or (
            f"{os.path.basename(persist_model_path)}-{uuid.uuid4().hex[:8]}-{os.getpid()}"
        )

        if mode == "off":
            self.active = False
        elif mode == "on":
            self.active = True
        else:
            ancestor = _nearest_existing_ancestor(persist_model_path)
            self.active = is_network_filesystem(ancestor)

        if self.active:
            self.local_path = os.path.join(resolve_staging_root(staging_root), self.run_id)
        else:
            self.local_path = persist_model_path

    def setup(self) -> None:
        os.makedirs(self.local_path, exist_ok=True)
        if self.active:
            os.makedirs(self.persist_path, exist_ok=True)


class ArtifactManifest:
    """In-memory record of which staged artifacts have already been synced,
    keyed by (size, mtime_ns) so the sync worker can skip unchanged files
    without re-stating the whole remote tree.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, Tuple[int, int]] = {}

    @classmethod
    def load(cls, path: str) -> "ArtifactManifest":
        manifest = cls()
        if os.path.isfile(path):
            with open(path, "r") as f:
                data = json.load(f)
            manifest._entries = {k: (int(v[0]), int(v[1])) for k, v in data.items()}
        return manifest

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(os.path.abspath(path)) or ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._entries, f)
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def changed(self, rel_path: str, size: int, mtime_ns: int) -> bool:
        existing = self._entries.get(rel_path)
        return existing is None or existing != (size, mtime_ns)

    def update(self, rel_path: str, size: int, mtime_ns: int) -> None:
        self._entries[rel_path] = (size, mtime_ns)


def _sync_log_tail(local_path: str, persist_path: str, offset: int) -> int:
    """Append bytes written to local_path/train.log since `offset` onto the
    persist_path copy. Append-mode, not temp+replace -- this file must
    support `tail -F` following it live at the persistent path.
    """
    src = os.path.join(local_path, _LOG_NAME)
    if not os.path.isfile(src):
        return offset
    size = os.path.getsize(src)
    if size <= offset:
        return offset
    with open(src, "rb") as f:
        f.seek(offset)
        data = f.read()
    dest = os.path.join(persist_path, _LOG_NAME)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "ab") as f:
        f.write(data)
    return offset + len(data)


def _sync_artifacts(local_path: str, persist_path: str, manifest: "ArtifactManifest") -> None:
    """Stat-before/copy/stat-after sync of every changed artifact under
    local_path onto persist_path, skipping train.log (handled by
    _sync_log_tail) and the manifest/temp files.
    """
    for root, _dirs, files in os.walk(local_path):
        for fname in files:
            if fname.endswith(".tmp") or fname.endswith(".partial"):
                continue
            if fname == _LOG_NAME or fname == _MANIFEST_NAME:
                continue
            src = os.path.join(root, fname)
            rel = os.path.relpath(src, local_path)
            try:
                stat_before = os.stat(src)
            except OSError:
                continue
            if not manifest.changed(rel, stat_before.st_size, stat_before.st_mtime_ns):
                continue

            dest = os.path.join(persist_path, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            partial = dest + ".partial"
            try:
                shutil.copyfile(src, partial)
                stat_after = os.stat(src)
            except OSError:
                if os.path.exists(partial):
                    os.remove(partial)
                continue

            if (stat_after.st_size, stat_after.st_mtime_ns) != (
                stat_before.st_size,
                stat_before.st_mtime_ns,
            ):
                os.remove(partial)
                continue

            os.replace(partial, dest)
            manifest.update(rel, stat_after.st_size, stat_after.st_mtime_ns)

    manifest.save(os.path.join(persist_path, _MANIFEST_NAME))


def _persistence_loop(
    local_path: str,
    persist_path: str,
    log_sync_interval: float,
    artifact_sync_interval: float,
    shutdown_event,
    status_queue,
    finalize_event,
    sync_request_queue,
    sync_result_queue,
    sync_wakeup_event,
) -> None:
    """Child process entry point: tail train.log every `log_sync_interval`
    seconds, mirror changed artifacts every `artifact_sync_interval`
    seconds, until told to shut down or finalize.
    """
    try:
        os.makedirs(persist_path, exist_ok=True)
        manifest = ArtifactManifest.load(os.path.join(persist_path, _MANIFEST_NAME))
        log_offset = 0
        last_artifact_sync = 0.0

        while True:
            if finalize_event.is_set():
                log_offset = _sync_log_tail(local_path, persist_path, log_offset)
                _sync_artifacts(local_path, persist_path, manifest)
                status_queue.put(("done", None))
                return
            if shutdown_event.is_set():
                return

            try:
                sync_request_id = sync_request_queue.get_nowait()
            except queue.Empty:
                sync_request_id = None
            if sync_request_id is not None:
                sync_wakeup_event.clear()
                try:
                    log_offset = _sync_log_tail(local_path, persist_path, log_offset)
                    _sync_artifacts(local_path, persist_path, manifest)
                except Exception as exc:  # noqa: BLE001 - exact request result
                    sync_result_queue.put((sync_request_id, "error", str(exc)))
                else:
                    sync_result_queue.put((sync_request_id, "done", None))
                continue

            log_offset = _sync_log_tail(local_path, persist_path, log_offset)

            now = time.monotonic()
            if now - last_artifact_sync >= artifact_sync_interval:
                _sync_artifacts(local_path, persist_path, manifest)
                last_artifact_sync = now

            slept = 0.0
            while slept < log_sync_interval:
                if (
                    shutdown_event.is_set()
                    or finalize_event.is_set()
                    or sync_wakeup_event.is_set()
                ):
                    break
                step = min(0.1, log_sync_interval - slept)
                time.sleep(step)
                slept += step
    except Exception as exc:  # noqa: BLE001 - report to parent, never crash silently
        try:
            status_queue.put(("error", str(exc)))
        except Exception:
            pass


class PersistenceWorker:
    """Owns a background OS process that mirrors `manager.local_path` onto
    `manager.persist_path` -- a separate process, not a thread, since the
    per-file stat/copy work is CPU/IO-bound and this project has already
    measured GIL contention stalling a training thread from background
    numpy work (rvizsplat's live-viewer publish path).
    """

    def __init__(
        self,
        manager: OutputStagingManager,
        log_sync_interval: float = 2.0,
        artifact_sync_interval: float = 300.0,
    ) -> None:
        self._manager = manager
        self._log_sync_interval = log_sync_interval
        self._artifact_sync_interval = artifact_sync_interval
        self._process: Optional[multiprocessing.Process] = None
        self._shutdown_event = None
        self._finalize_event = None
        self._status_queue = None
        self._sync_request_queue = None
        self._sync_result_queue = None
        self._sync_wakeup_event = None
        self._sync_lock = threading.Lock()
        self._next_sync_request_id = 0

    def start(self) -> None:
        if not self._manager.active:
            return

        self._shutdown_event = multiprocessing.Event()
        self._finalize_event = multiprocessing.Event()
        self._status_queue = multiprocessing.Queue()
        self._sync_request_queue = multiprocessing.Queue()
        self._sync_result_queue = multiprocessing.Queue()
        self._sync_wakeup_event = multiprocessing.Event()
        self._process = multiprocessing.Process(
            target=_persistence_loop,
            args=(
                self._manager.local_path,
                self._manager.persist_path,
                self._log_sync_interval,
                self._artifact_sync_interval,
                self._shutdown_event,
                self._status_queue,
                self._finalize_event,
                self._sync_request_queue,
                self._sync_result_queue,
                self._sync_wakeup_event,
            ),
            daemon=True,
        )
        self._process.start()

    def sync_now(self, timeout: float = 120.0) -> None:
        """Synchronously request one child-owned artifact sync."""
        if self._process is None:
            return
        with self._sync_lock:
            if not self._process.is_alive():
                raise RuntimeError("persistence worker is not alive")
            self._next_sync_request_id += 1
            request_id = self._next_sync_request_id
            self._sync_request_queue.put(request_id)
            self._sync_wakeup_event.set()
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"persistence sync request {request_id} timed out"
                    )
                try:
                    result_id, status, detail = self._sync_result_queue.get(
                        timeout=min(remaining, 0.25)
                    )
                except queue.Empty:
                    if not self._process.is_alive():
                        raise RuntimeError(
                            f"persistence worker died during sync request {request_id}"
                        )
                    continue
                if result_id != request_id:
                    raise RuntimeError(
                        "persistence sync response id mismatch: "
                        f"expected {request_id}, got {result_id}"
                    )
                if status != "done":
                    raise RuntimeError(
                        f"persistence sync request {request_id} failed: {detail}"
                    )
                return

    def finalize(self, timeout: float = 120.0) -> bool:
        if self._process is None:
            return True

        self._finalize_event.set()
        try:
            status, _detail = self._status_queue.get(timeout=timeout)
        except Exception:
            return False
        return status == "done"

    def shutdown(self) -> None:
        if self._process is None:
            return
        self._shutdown_event.set()
        self._process.join(timeout=5.0)
