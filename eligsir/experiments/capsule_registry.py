"""Canonical scene replay-capsule registry with enforced immutability (issue #634).

This module owns the versioned registry of all published Scene Replay Capsules.
Each capsule is addressable by a stable, immutable ID (e.g., `tum_fr3_mapper_core_v1`)
and content-hashed for deterministic identity.

A capsule in the registry has a lifecycle status:

- **DRAFT**: capsule manifest is registered but not yet validated. May have
  `pending_artifacts` (event logs from #633 that don't yet exist) or
  `blocked_on` dependencies.
- **VALIDATED**: all artifacts exist and `validate_capsule()` passes.
- **CANONICAL**: validated AND a MAPPER_CORE or POLICY pairing has been
  launched and verified reproducible against it.
- **RETIRED**: replaced by a newer version. A retired capsule cannot back
  new experiments.

Registry stores only `CapsuleManifest` objects (what substrate exists), not
`FreezeAxisManifest` objects (experiment-specific pairings). Lane rules and
experiment-pairing validation are #633/#636's responsibility.

This module records no traces, reads no rosbag, and is imported by no training
path. It is pure Python, CPU-only, torch-free, and import-side-effect-free.
Data files live in `experiments/capsules/` as JSON, not in-module constants,
to keep the registry lazily-loadable and testable without importing the
entire codebase.

## Registry lifecycle rules

1. **Immutability by content hash**: once a capsule ID is published at a given
   content hash, that ID forever refers to exactly that manifest. Any scientific
   change (source, artifacts, evaluation identity) produces a new version and
   new ID.

2. **Explicit retirement**: never rewrite or delete a published capsule. Mark
   it RETIRED with a replacement ID and reason. Historical results remain
   interpretable.

3. **Validation gates promotion**: a capsule stays DRAFT until all its artifact
   dependencies are resolved and content-hashable. VALIDATED means
   `validate_capsule()` passed. CANONICAL requires reproducible experiment
   evidence.

4. **Deferred entries are explicit**: a DRAFT entry names its `pending_artifacts`
   and `blocked_on` issue list so future work can locate what unblocks it.

## File format: JSON per capsule

Each capsule is one JSON file under `experiments/capsules/<capsule_id>/`.
Path structure:

```
experiments/capsules/
  tum_fr3_mapper_core_v1/
    manifest.json          # serialized CapsuleManifest.to_dict()
    registry_entry.json    # registry-level metadata (status, hash, etc.)
  replica_small_mapper_core_v1/
    manifest.json
    registry_entry.json
  ...
```

This layout allows:
- Per-ID atomicity (all data for one capsule in one directory)
- Parallel registration of unrelated capsules
- Capsule migration (move the directory to relocate)
- Independent review of manifest vs registry metadata

## What the registry validates

- Schema version and format (JSON parseable, required fields present)
- Immutability: if an ID is already registered, reregistration must have
  identical content hash
- Lifecycle coherence: DRAFT may have pending artifacts, VALIDATED/CANONICAL
  must have none, RETIRED must name successor and reason
- `validate_capsule()` against the CapsuleManifest for non-DRAFT entries
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from eligsir.experiments.hashing import content_hash
from eligsir.experiments.replay_capsule import (
    CAPSULE_SCHEMA_VERSION,
    CapsuleManifest,
    CapsuleValidationError,
    validate_capsule,
)


class RegistryError(Exception):
    """Raised when a registry operation fails."""


RegistryStatus = Mapping[str, str]  # "DRAFT" | "VALIDATED" | "CANONICAL" | "RETIRED"
VALID_REGISTRY_STATUSES: tuple[str, ...] = ("DRAFT", "VALIDATED", "CANONICAL", "RETIRED")


@dataclass
class RegistryEntry:
    """Registry metadata for one capsule (distinct from CapsuleManifest).

    This object pairs with a CapsuleManifest to form the complete immutability
    contract. It records:
    - capsule_hash: the content hash of the manifest at registration time
    - status: lifecycle (DRAFT/VALIDATED/CANONICAL/RETIRED)
    - pending_artifacts: list of artifact keys still to be produced (DRAFT only)
    - blocked_on: list of issue numbers/branches blocking completion (DRAFT only)
    - replaced_by: new version ID when RETIRED
    - retirement_reason: free-text explanation when RETIRED

    Args:
        capsule_id: Immutable identity (e.g., "tum_fr3_mapper_core_v1").
        capsule_hash: SHA-256 of CapsuleManifest.identity_payload() at
            registration time.
        status: One of DRAFT/VALIDATED/CANONICAL/RETIRED.
        pending_artifacts: Artifact keys that don't yet exist (DRAFT only).
        blocked_on: Issue numbers as strings (DRAFT only).
        replaced_by: Replacement capsule ID (RETIRED only).
        retirement_reason: Why it was retired (RETIRED only).
        notes: Free-text.
    """

    capsule_id: str
    capsule_hash: str
    status: str = "DRAFT"
    pending_artifacts: tuple[str, ...] = field(default_factory=tuple)
    blocked_on: tuple[str, ...] = field(default_factory=tuple)
    replaced_by: str | None = None
    retirement_reason: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "capsule_hash": self.capsule_hash,
            "status": self.status,
            "pending_artifacts": list(self.pending_artifacts),
            "blocked_on": list(self.blocked_on),
            "replaced_by": self.replaced_by,
            "retirement_reason": self.retirement_reason,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegistryEntry":
        return cls(
            capsule_id=data["capsule_id"],
            capsule_hash=data["capsule_hash"],
            status=data.get("status", "DRAFT"),
            pending_artifacts=tuple(data.get("pending_artifacts", ())),
            blocked_on=tuple(data.get("blocked_on", ())),
            replaced_by=data.get("replaced_by"),
            retirement_reason=data.get("retirement_reason", ""),
            notes=data.get("notes", ""),
        )


def validate_registry_entry(
    entry: RegistryEntry,
    manifest: CapsuleManifest,
) -> None:
    """Validate a registry entry against its paired manifest.

    Raises:
        RegistryError: on validation failure.
    """
    if not entry.capsule_id:
        raise RegistryError("capsule_id must be non-empty")
    if entry.capsule_id != manifest.capsule_id:
        raise RegistryError(
            f"registry entry capsule_id {entry.capsule_id!r} does not match "
            f"manifest capsule_id {manifest.capsule_id!r}"
        )

    if entry.status not in VALID_REGISTRY_STATUSES:
        raise RegistryError(
            f"status must be one of {VALID_REGISTRY_STATUSES}, got {entry.status!r}"
        )

    # Immutability: compute the real hash and match it to what's recorded.
    actual_hash = manifest.capsule_hash()
    if entry.capsule_hash != actual_hash:
        raise RegistryError(
            f"capsule_hash mismatch for {entry.capsule_id!r}: registry entry declares "
            f"{entry.capsule_hash!r}, manifest content hashes to {actual_hash!r}. "
            "A registered capsule's manifest must not be modified in place."
        )

    # Status coherence.
    if entry.status == "RETIRED":
        if not entry.replaced_by or not entry.retirement_reason:
            raise RegistryError(
                f"retired capsule {entry.capsule_id!r} must record both replaced_by "
                "and retirement_reason so historical results stay interpretable"
            )
    elif entry.status in ("VALIDATED", "CANONICAL"):
        if entry.pending_artifacts:
            raise RegistryError(
                f"capsule {entry.capsule_id!r} status is {entry.status} but has "
                f"pending_artifacts {entry.pending_artifacts!r}; pending artifacts "
                "are only allowed for DRAFT"
            )

    if entry.status in ("VALIDATED", "CANONICAL"):
        # Validate the manifest itself for non-DRAFT entries.
        try:
            validate_capsule(manifest)
        except CapsuleValidationError as e:
            raise RegistryError(
                f"capsule {entry.capsule_id!r} status is {entry.status} but "
                f"validate_capsule() failed: {e}"
            ) from e


class CapsuleRegistry:
    """In-memory registry of all published Scene Replay Capsules.

    Loads capsule manifests and registry entries from disk, validates them,
    and enforces immutability.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._capsules: dict[str, CapsuleManifest] = {}
        self._entries: dict[str, RegistryEntry] = {}

    def load_directory(self, root: Path) -> None:
        """Load all capsules from a directory tree.

        Expects structure:
        ```
        <root>/
          <capsule_id>/
            manifest.json
            registry_entry.json
          ...
        ```

        Each capsule_id subdirectory should have both files. Missing files are
        an error; all registered entries are validated.

        Args:
            root: Root directory containing capsule subdirectories.

        Raises:
            RegistryError: on any load or validation failure.
        """
        root = Path(root)
        if not root.is_dir():
            raise RegistryError(f"registry root {root} does not exist or is not a directory")

        for capsule_dir in sorted(root.iterdir()):
            if not capsule_dir.is_dir():
                continue
            capsule_id = capsule_dir.name
            try:
                self._load_capsule(capsule_dir, capsule_id)
            except Exception as e:
                raise RegistryError(
                    f"failed to load capsule {capsule_id} from {capsule_dir}: {e}"
                ) from e

    def _load_capsule(self, capsule_dir: Path, capsule_id: str) -> None:
        """Load and validate one capsule from its directory."""
        manifest_path = capsule_dir / "manifest.json"
        entry_path = capsule_dir / "registry_entry.json"

        if not manifest_path.exists():
            raise RegistryError(f"missing manifest.json in {capsule_dir}")
        if not entry_path.exists():
            raise RegistryError(f"missing registry_entry.json in {capsule_dir}")

        # Load manifest.
        with open(manifest_path) as f:
            manifest_data = json.load(f)
        manifest = CapsuleManifest.from_dict(manifest_data)

        # Load entry.
        with open(entry_path) as f:
            entry_data = json.load(f)
        entry = RegistryEntry.from_dict(entry_data)

        # Validate pairing.
        validate_registry_entry(entry, manifest)

        # Check immutability: if already registered, must be identical.
        if capsule_id in self._capsules:
            existing_entry = self._entries[capsule_id]
            if existing_entry.capsule_hash != entry.capsule_hash:
                raise RegistryError(
                    f"immutability violation: capsule_id {capsule_id!r} is already "
                    f"registered with hash {existing_entry.capsule_hash!r}, but this "
                    f"registration declares hash {entry.capsule_hash!r}. A capsule "
                    "ID forever refers to exactly one manifest. Create a new version/ID."
                )
        else:
            self._capsules[capsule_id] = manifest
            self._entries[capsule_id] = entry

    def get_capsule(self, capsule_id: str) -> CapsuleManifest | None:
        """Retrieve a capsule by ID, or None if not registered."""
        return self._capsules.get(capsule_id)

    def get_entry(self, capsule_id: str) -> RegistryEntry | None:
        """Retrieve a registry entry by capsule ID, or None if not registered."""
        return self._entries.get(capsule_id)

    def list_capsules(self, status: str | None = None) -> tuple[str, ...]:
        """List all registered capsule IDs, optionally filtered by status.

        Args:
            status: Filter to entries with this status (DRAFT/VALIDATED/CANONICAL/RETIRED).
                None returns all.

        Returns:
            Tuple of capsule IDs, sorted.
        """
        if status is None:
            return tuple(sorted(self._capsules.keys()))
        return tuple(
            sorted(
                capsule_id
                for capsule_id, entry in self._entries.items()
                if entry.status == status
            )
        )

    def validate_pairing_against_lane(
        self,
        capsule_id: str,
        lane: str,
        *,
        checksums: Mapping[str, str] | None = None,
    ) -> None:
        """Validate a capsule against a specific lane's freeze requirements.

        This is a convenience that combines registry lookup with
        `replay_capsule.validate_pairing()` to check a registered capsule
        against a lane before launching an experiment.

        Args:
            capsule_id: Registered capsule ID.
            lane: Lane name (MAPPER_CORE/POLICY/END_TO_END).
            checksums: Optional checksums.sha256 sidecar for artifact verification.

        Raises:
            RegistryError: if the capsule is not registered.
            CapsuleValidationError: if the capsule cannot be validated for the lane.
        """
        from eligsir.experiments.replay_capsule import (
            LANE_RULES,
            AxisDeclaration,
            FreezeAxisManifest,
            make_axis_map,
            validate_pairing,
        )

        capsule = self.get_capsule(capsule_id)
        if capsule is None:
            raise RegistryError(
                f"capsule {capsule_id!r} is not registered in this registry"
            )

        entry = self.get_entry(capsule_id)
        if entry.status == "RETIRED":
            raise RegistryError(
                f"capsule {capsule_id!r} is RETIRED (replaced by {entry.replaced_by!r}); "
                "cannot back new experiments"
            )

        # Build a freeze-axis manifest for the lane.
        # Use NOT_APPLICABLE with a reason for axes this lane doesn't use.
        axes = make_axis_map(
            "FROZEN",
            default_reason="all axes required frozen for this lane",
        )

        # Validate the lane and pairing.
        freeze = FreezeAxisManifest(
            experiment_id=f"{capsule_id}_lane_check",
            capsule_id=capsule_id,
            capsule_hash=entry.capsule_hash,
            lane=lane,
            axes=axes,
            oracle_declared=(capsule.frontend_trace_mode == "FROZEN_FINAL_POSES"),
        )
        validate_pairing(freeze, capsule, checksums=checksums)
