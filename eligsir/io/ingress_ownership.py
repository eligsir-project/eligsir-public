"""Issue #837: machine-readable ingress ownership contract.

Makes the single-owner ingress architecture from
``docs/architecture/2026-08-15-single-owner-rgbd-ingress-consolidation-plan.md``
(the "canonicalization plan", parent issue #805) enforceable rather than
relying on launch-script convention alone.

Scope of this module (acceptance criteria 1-7 of #837 only -- see the
docstring on :func:`resolve_ingress_ownership` for the explicit exclusion of
criterion 8):

1. an explicit enumeration of the 10 named ingress responsibilities;
2. explicit role semantics (``MUTATOR``/``DECISION_OWNER``/``ASSERTOR``/
   ``CONSUMER``/``COMPATIBILITY_ADAPTER``);
3. a resolver that detects ``DUPLICATE_INGRESS_AUTHORITY`` when two active
   mutating authorities (``MUTATOR`` or ``DECISION_OWNER``) claim the same
   responsibility;
4. source-guarantee-driven resolution (a source's declared upstream
   guarantees select which owners are actually *required*, rather than
   assuming every source needs every correction stage);
5. an explicit, labeled legacy-topology representation, so a legacy
   multi-owner configuration is never silently mistaken for canonical;
6. optional owner-handoff recording (e.g. a validated hardware source
   handing correction responsibility to a software fallback owner), with
   the invariant that a handoff retires the old owner rather than letting
   both run concurrently;
7. a round-trippable ``report.json["ingress_ownership"]`` block.

Criterion 8 -- "#832/#833/#834/#836 feed their realized owner state into
this contract" -- is explicitly OUT OF SCOPE here (those PRs are in flight
concurrently). This module instead exposes :func:`register_ingress_owner`,
a small declarative registration API: once those sibling components land,
they can call it at import/construction time to declare their real,
realized ownership, and the resolver picks their declarations up via
:func:`registered_declarations` automatically -- no further change to this
module or its call site should be required. Until then the registry is
empty and only the responsibilities this repo can already resolve from
existing, real runtime state (currently: ``source_sampling``, reusing PR
#818's ``resolve_source_decimation``/``resolve_source_sampling_policy``
already wired in ``eligsir/mapping/scheduler/frame_scheduler.py`` and
``train_streaming.py``) are populated in a live run's report.

This module is deliberately **behavior-neutral**: it only computes and
persists a report/provenance block. Nothing here raises, aborts a run,
gates admission, or changes any loss/mutation/default. Per the #837 spec
and CLAUDE.md's Run Contracts convention (``eval_valid``,
``SUSPECTED_TRUNCATED``, ``map_reinit_invalidated``, etc.), the
"fail-closed" requirement for canonical/paper-grade runs is expressed as a
loud, machine-readable verdict (``result.valid``, ``result.failure_codes``,
a printed warning) for a *human or a downstream validity gate* to act on --
never as an in-process abort. See ``resolve_ingress_ownership``'s docstring
for the exact semantics.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set

# Schema/contract version for report.json["ingress_ownership"]["version"].
# Bump only on a breaking change to the persisted shape (not on adding new
# optional fields with a safe default).
CONTRACT_VERSION = 1

DUPLICATE_INGRESS_AUTHORITY = "DUPLICATE_INGRESS_AUTHORITY"


class Responsibility(str, Enum):
    """The 10 named ingress semantic responsibilities from issue #837."""

    RGBD_TEMPORAL_PAIRING = "rgbd_temporal_pairing"
    RGBD_TIMESTAMP_CORRECTION = "rgbd_timestamp_correction"
    RGBD_REGISTRATION = "rgbd_registration"
    TRANSPORT_DECODE = "transport_decode"
    POSE_ASSOCIATION = "pose_association"
    SOURCE_SAMPLING = "source_sampling"
    OVERFLOW_DROP_POLICY = "overflow_drop_policy"
    COORDINATE_INTEGRITY = "coordinate_integrity"
    CANDIDATE_PREFILTER = "candidate_prefilter"
    KEYFRAME_COMMIT = "keyframe_commit"


class OwnershipRole(str, Enum):
    """Role semantics for a component declaring involvement in a
    responsibility (issue #837 "Roles" section)."""

    MUTATOR = "MUTATOR"
    DECISION_OWNER = "DECISION_OWNER"
    ASSERTOR = "ASSERTOR"
    CONSUMER = "CONSUMER"
    COMPATIBILITY_ADAPTER = "COMPATIBILITY_ADAPTER"


# At most one of these may be active per responsibility in canonical mode.
# A COMPATIBILITY_ADAPTER (e.g. a splitter that republishes an already-paired
# authoritative observation as two topics for a downstream consumer) is
# explicitly NOT a mutating authority -- issue #837: "A compatibility
# splitter is not a second RGB-D pairing owner merely because it republishes
# two halves of an authoritative pair."
MUTATING_ROLES = frozenset({OwnershipRole.MUTATOR, OwnershipRole.DECISION_OWNER})


class TopologyMode(str, Enum):
    """Whether a resolved ownership block represents the single-owner
    architecture (``CANONICAL``) or an explicitly-permitted historical
    multi-owner configuration (``LEGACY``, issue #837 test 7). Every
    consumer of ``report.json["ingress_ownership"]`` must branch on this
    field before trusting ``valid`` -- a legacy topology is never
    mistaken for canonical because it carries its own label."""

    CANONICAL = "canonical"
    LEGACY = "legacy"


@dataclasses.dataclass(frozen=True)
class OwnerDeclaration:
    """One component's claimed involvement in one responsibility."""

    responsibility: Responsibility
    owner: str
    role: OwnershipRole
    mode: Optional[str] = None
    attributes: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this declaration to a plain, JSON-safe dict -- the
        same shape :meth:`from_dict` consumes, so a declaration always
        round-trips through ``report.json`` without loss."""
        return {
            "responsibility": Responsibility(self.responsibility).value,
            "owner": self.owner,
            "role": OwnershipRole(self.role).value,
            "mode": self.mode,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OwnerDeclaration":
        """Reconstruct a declaration from a dict produced by :meth:`to_dict`,
        the inverse operation used when reloading a persisted
        ``ingress_ownership`` block."""
        return cls(
            responsibility=Responsibility(d["responsibility"]),
            owner=d["owner"],
            role=OwnershipRole(d["role"]),
            mode=d.get("mode"),
            attributes=dict(d.get("attributes") or {}),
        )


@dataclasses.dataclass(frozen=True)
class OwnerHandoff:
    """Records an explicit owner-handoff event (issue #837 "Owner handoff").

    ``from_owner`` is retired for ``responsibility`` once this handoff is
    present: the resolver excludes any mutating declaration whose
    ``(responsibility, owner)`` matches ``(responsibility, from_owner)`` from
    the duplicate-authority count, so a *declared* handoff never itself
    trips ``DUPLICATE_INGRESS_AUTHORITY`` -- two owners are never treated as
    concurrently active once a handoff between them is on record.
    """

    responsibility: Responsibility
    from_owner: str
    to_owner: str
    reason: str
    at: Optional[str] = None  # free-form provenance (e.g. an iteration or wall-time string)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this handoff event to a plain, JSON-safe dict for the
        ``report.json["ingress_ownership"]["handoffs"]`` list."""
        return {
            "responsibility": Responsibility(self.responsibility).value,
            "from_owner": self.from_owner,
            "to_owner": self.to_owner,
            "reason": self.reason,
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OwnerHandoff":
        """Reconstruct a handoff event from a dict produced by
        :meth:`to_dict`, used when reloading a persisted
        ``ingress_ownership`` block's handoff history."""
        return cls(
            responsibility=Responsibility(d["responsibility"]),
            from_owner=d["from_owner"],
            to_owner=d["to_owner"],
            reason=d["reason"],
            at=d.get("at"),
        )


@dataclasses.dataclass(frozen=True)
class SourceGuaranteeProfile:
    """Declares, for one canonical source mode, which responsibilities need
    an actively-resolved owner at all (issue #837 "Source guarantee
    declaration"). The resolver uses this to avoid demanding e.g. a
    software RGB-D sync owner on a source whose driver already guarantees
    hardware-validated pairing.
    """

    name: str
    required_responsibilities: Set[Responsibility] = dataclasses.field(default_factory=set)
    notes: Dict[str, str] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this guarantee profile (its name, the responsibilities
        it requires a resolved owner for, and any free-form notes) to a
        plain, JSON-safe dict."""
        return {
            "name": self.name,
            "required_responsibilities": sorted(
                Responsibility(r).value for r in self.required_responsibilities
            ),
            "notes": dict(self.notes),
        }


# Canonical path B (plan doc section 7): validated live Orbbec. The driver
# hardware-syncs and registers; nothing downstream needs to re-derive pairing
# or registration.
LIVE_ORBBEC_GUARANTEES = SourceGuaranteeProfile(
    name="live_orbbec",
    required_responsibilities={
        Responsibility.RGBD_TEMPORAL_PAIRING,
        Responsibility.RGBD_REGISTRATION,
        Responsibility.POSE_ASSOCIATION,
        Responsibility.SOURCE_SAMPLING,
        Responsibility.OVERFLOW_DROP_POLICY,
        Responsibility.COORDINATE_INTEGRITY,
        Responsibility.CANDIDATE_PREFILTER,
        Responsibility.KEYFRAME_COMMIT,
    },
    notes={
        "rgbd_sync": "hardware_validated (align_target_stream + enable_frame_sync)",
        "registration": "aligned_to_color at the driver",
    },
)

# Canonical path A (plan doc section 6): unvalidated rosbag. Pairing and
# timestamp correction are NOT guaranteed by the recording -- an explicit
# software owner (eligsir-rgbd-sync) is required.
UNVALIDATED_ROSBAG_GUARANTEES = SourceGuaranteeProfile(
    name="unvalidated_rosbag",
    required_responsibilities={
        Responsibility.RGBD_TEMPORAL_PAIRING,
        Responsibility.RGBD_TIMESTAMP_CORRECTION,
        Responsibility.TRANSPORT_DECODE,
        Responsibility.POSE_ASSOCIATION,
        Responsibility.SOURCE_SAMPLING,
        Responsibility.OVERFLOW_DROP_POLICY,
        Responsibility.COORDINATE_INTEGRITY,
        Responsibility.CANDIDATE_PREFILTER,
        Responsibility.KEYFRAME_COMMIT,
    },
    notes={
        "rgbd_sync": "unknown/unvalidated until eligsir_rgbd_sync corrects it",
        "registration": "declared/verified separately",
    },
)


@dataclasses.dataclass
class ResponsibilityResolution:
    """The resolved state of one responsibility after applying declarations,
    retired-by-handoff filtering, and duplicate detection."""

    responsibility: Responsibility
    topology: TopologyMode
    resolved_owner: Optional[OwnerDeclaration]
    mutating_declarations: List[OwnerDeclaration]
    assertor_declarations: List[OwnerDeclaration]
    compatibility_adapter_declarations: List[OwnerDeclaration]
    consumer_declarations: List[OwnerDeclaration]
    duplicate: bool
    failure_code: Optional[str]
    required: Optional[bool] = None  # None = no guarantee profile supplied

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this responsibility's resolved state -- the winning
        mutating owner (if any) plus every declared assertor/adapter/
        consumer and the duplicate/failure verdict -- to the plain dict
        stored under ``report.json["ingress_ownership"]["responsibilities"]
        [<name>]``. Round-trips via :meth:`from_dict`."""
        d: Dict[str, Any] = {
            "topology": self.topology.value,
            "duplicate": self.duplicate,
            "failure_code": self.failure_code,
            "required": self.required,
            "mutating_owners": [x.to_dict() for x in self.mutating_declarations],
            "assertors": [x.to_dict() for x in self.assertor_declarations],
            "compatibility_adapters": [x.to_dict() for x in self.compatibility_adapter_declarations],
            "consumers": [x.to_dict() for x in self.consumer_declarations],
        }
        if self.resolved_owner is not None:
            d["owner"] = self.resolved_owner.owner
            d["role"] = OwnershipRole(self.resolved_owner.role).value
            d["mode"] = self.resolved_owner.mode
            d["attributes"] = dict(self.resolved_owner.attributes)
        else:
            d["owner"] = None
            d["role"] = None
            d["mode"] = None
            d["attributes"] = {}
        return d

    @classmethod
    def from_dict(cls, responsibility: Responsibility, d: Dict[str, Any]) -> "ResponsibilityResolution":
        """Reconstruct one responsibility's resolved state from the dict
        produced by :meth:`to_dict`, given the :class:`Responsibility` it
        belongs under (the report dict itself only carries that identity
        as the enclosing key, not inside the block)."""
        resolved_owner = None
        if d.get("owner") is not None and d.get("role") is not None:
            resolved_owner = OwnerDeclaration(
                responsibility=responsibility,
                owner=d["owner"],
                role=OwnershipRole(d["role"]),
                mode=d.get("mode"),
                attributes=dict(d.get("attributes") or {}),
            )
        return cls(
            responsibility=responsibility,
            topology=TopologyMode(d.get("topology", TopologyMode.CANONICAL.value)),
            resolved_owner=resolved_owner,
            mutating_declarations=[OwnerDeclaration.from_dict({**x, "responsibility": responsibility.value})
                                    for x in d.get("mutating_owners", [])],
            assertor_declarations=[OwnerDeclaration.from_dict({**x, "responsibility": responsibility.value})
                                    for x in d.get("assertors", [])],
            compatibility_adapter_declarations=[
                OwnerDeclaration.from_dict({**x, "responsibility": responsibility.value})
                for x in d.get("compatibility_adapters", [])
            ],
            consumer_declarations=[OwnerDeclaration.from_dict({**x, "responsibility": responsibility.value})
                                    for x in d.get("consumers", [])],
            duplicate=bool(d.get("duplicate", False)),
            failure_code=d.get("failure_code"),
            required=d.get("required"),
        )


@dataclasses.dataclass
class IngressOwnershipResult:
    """The full outcome of resolving one run's ingress ownership: one
    :class:`ResponsibilityResolution` per responsibility, the recorded
    :class:`OwnerHandoff` events, and the overall structural verdict
    (``valid``/``failure_codes``) this project's report/provenance
    consumers read the same way as ``eval_valid``/``SUSPECTED_TRUNCATED``.
    Returned by :func:`resolve_ingress_ownership`; persisted via
    :meth:`to_report_dict`."""

    version: int
    topology: TopologyMode
    responsibilities: Dict[Responsibility, ResponsibilityResolution]
    handoffs: List[OwnerHandoff]
    valid: bool
    failure_codes: List[str]
    guarantee_profile: Optional[str]
    missing_required_owners: List[str]

    def to_report_dict(self) -> Dict[str, Any]:
        """The exact ``report.json["ingress_ownership"]`` shape. Round-trips
        via :func:`ingress_ownership_from_report_dict` (issue #837 test 8)."""
        return {
            "version": self.version,
            "topology": self.topology.value,
            "valid": self.valid,
            "failure_codes": list(self.failure_codes),
            "guarantee_profile": self.guarantee_profile,
            "missing_required_owners": list(self.missing_required_owners),
            "responsibilities": {
                Responsibility(k).value: v.to_dict() for k, v in self.responsibilities.items()
            },
            "handoffs": [h.to_dict() for h in self.handoffs],
        }


def ingress_ownership_from_report_dict(d: Dict[str, Any]) -> IngressOwnershipResult:
    """Reconstruct an :class:`IngressOwnershipResult` from a
    ``report.json["ingress_ownership"]`` block, preserving owner/role/mode
    /version (issue #837 test 8: "report round-trip preserves owner/mode
    /version")."""
    responsibilities: Dict[Responsibility, ResponsibilityResolution] = {}
    for name, block in (d.get("responsibilities") or {}).items():
        resp = Responsibility(name)
        responsibilities[resp] = ResponsibilityResolution.from_dict(resp, block)
    return IngressOwnershipResult(
        version=int(d.get("version", CONTRACT_VERSION)),
        topology=TopologyMode(d.get("topology", TopologyMode.CANONICAL.value)),
        responsibilities=responsibilities,
        handoffs=[OwnerHandoff.from_dict(h) for h in d.get("handoffs", [])],
        valid=bool(d.get("valid", True)),
        failure_codes=list(d.get("failure_codes", [])),
        guarantee_profile=d.get("guarantee_profile"),
        missing_required_owners=list(d.get("missing_required_owners", [])),
    )


def resolve_ingress_ownership(
    declarations: Sequence[OwnerDeclaration],
    *,
    legacy_topology: bool = False,
    guarantees: Optional[SourceGuaranteeProfile] = None,
    handoffs: Optional[Sequence[OwnerHandoff]] = None,
) -> IngressOwnershipResult:
    """Resolve a set of declared ingress-component ownership claims into
    one block per responsibility, and detect duplicate mutating authority.

    ``legacy_topology=True`` represents an EXPLICIT legacy multi-owner
    configuration (issue #837 test 7): duplicate mutating declarations are
    still recorded (``duplicate=True`` on the affected responsibility) but
    are labeled ``topology="legacy"`` rather than raising
    ``DUPLICATE_INGRESS_AUTHORITY`` -- a legacy topology can be represented
    without ever being mistaken for a canonical one, since every consumer of
    this block can check ``topology`` before trusting ``valid``.

    ``legacy_topology=False`` (canonical, the default) sets
    ``result.valid = False`` and records ``DUPLICATE_INGRESS_AUTHORITY`` in
    ``result.failure_codes`` whenever two or more active ``MUTATOR``/
    ``DECISION_OWNER`` declarations claim the same responsibility. This is a
    **fail-closed structural verdict**, not an in-process abort: per the
    #837 spec and the architecture plan's section 12 ("a structural warning,
    not inferred from final reconstruction metrics"), this function never
    raises and never mutates training state -- it is the caller's
    responsibility to treat ``result.valid is False`` the same way this
    project already treats ``eval_valid is False`` / ``SUSPECTED_TRUNCATED``
    / ``map_reinit_invalidated``: never quote such a run as canonical/valid,
    but do not alter its actual training behavior because of it.

    A component participating via :class:`OwnerHandoff` as ``from_owner`` is
    excluded from the duplicate-authority count for that responsibility --
    an explicit, recorded handoff is definitionally not "two owners active
    concurrently".

    ``guarantees``, if supplied, restricts which responsibilities are
    expected to have a resolved mutating owner at all (issue #837 "Source
    guarantee declaration"): a responsibility outside
    ``guarantees.required_responsibilities`` with no declared owner is not
    reported as missing.
    """
    topology = TopologyMode.LEGACY if legacy_topology else TopologyMode.CANONICAL
    handoffs = list(handoffs or [])

    retired: Set[tuple] = {(h.responsibility, h.from_owner) for h in handoffs}

    by_responsibility: Dict[Responsibility, List[OwnerDeclaration]] = {r: [] for r in Responsibility}
    for decl in declarations:
        by_responsibility.setdefault(decl.responsibility, []).append(decl)

    resolved: Dict[Responsibility, ResponsibilityResolution] = {}
    failure_codes: List[str] = []

    for resp in Responsibility:
        group = by_responsibility.get(resp, [])
        mutating = [
            d for d in group
            if d.role in MUTATING_ROLES and (resp, d.owner) not in retired
        ]
        assertors = [d for d in group if d.role == OwnershipRole.ASSERTOR]
        adapters = [d for d in group if d.role == OwnershipRole.COMPATIBILITY_ADAPTER]
        consumers = [d for d in group if d.role == OwnershipRole.CONSUMER]

        duplicate = len(mutating) > 1
        failure_code = None
        resolved_owner = mutating[0] if len(mutating) == 1 else None

        if duplicate:
            if topology is TopologyMode.CANONICAL:
                failure_code = DUPLICATE_INGRESS_AUTHORITY
                if failure_code not in failure_codes:
                    failure_codes.append(failure_code)
            # legacy topology: duplicate recorded, no failure code, no
            # resolved_owner (ambiguous by construction of the legacy mode).

        required = None
        if guarantees is not None:
            required = resp in guarantees.required_responsibilities

        resolved[resp] = ResponsibilityResolution(
            responsibility=resp,
            topology=topology,
            resolved_owner=resolved_owner,
            mutating_declarations=mutating,
            assertor_declarations=assertors,
            compatibility_adapter_declarations=adapters,
            consumer_declarations=consumers,
            duplicate=duplicate,
            failure_code=failure_code,
            required=required,
        )

    missing_required_owners: List[str] = []
    if guarantees is not None:
        for resp in guarantees.required_responsibilities:
            res = resolved.get(resp)
            if res is None:
                missing_required_owners.append(Responsibility(resp).value)
                continue
            # A required responsibility is satisfied by a resolved mutating
            # owner OR, for assert-only-by-design responsibilities (e.g.
            # coordinate_integrity's mandatory integrity observer -- see the
            # plan doc's authority matrix), by at least one ASSERTOR. Only
            # flag it missing when NOTHING was declared for it at all.
            has_any_declaration = bool(
                res.mutating_declarations
                or res.assertor_declarations
                or res.compatibility_adapter_declarations
                or res.consumer_declarations
            )
            if not has_any_declaration:
                missing_required_owners.append(Responsibility(resp).value)

    valid = topology is TopologyMode.LEGACY or not failure_codes

    return IngressOwnershipResult(
        version=CONTRACT_VERSION,
        topology=topology,
        responsibilities=resolved,
        handoffs=handoffs,
        valid=valid,
        failure_codes=failure_codes,
        guarantee_profile=guarantees.name if guarantees is not None else None,
        missing_required_owners=missing_required_owners,
    )


# --------------------------------------------------------------------------
# Declarative registration API for future sibling components (issue #837
# criterion 8, deliberately NOT wired to any real owner state by this PR --
# see the module docstring). #832/#833/#834/#836 (or any later ingress
# component) can call `register_ingress_owner` once at import/construction
# time to declare its real, realized ownership; `registered_declarations()`
# is what a report-writing call site should merge with any locally-known
# declarations before calling `resolve_ingress_ownership`.
# --------------------------------------------------------------------------

_REGISTRY: List[OwnerDeclaration] = []


def register_ingress_owner(declaration: OwnerDeclaration) -> None:
    """Declare one component's real, realized ingress ownership.

    Intentionally process-global and additive (not keyed/deduplicated) --
    resolution-time duplicate detection is exactly what
    :func:`resolve_ingress_ownership` is for; this registry is just the
    collection point.
    """
    _REGISTRY.append(declaration)


def registered_declarations() -> List[OwnerDeclaration]:
    """All declarations registered so far via :func:`register_ingress_owner`."""
    return list(_REGISTRY)


def clear_registry() -> None:
    """Test-only: reset the process-global registry between test cases."""
    _REGISTRY.clear()
