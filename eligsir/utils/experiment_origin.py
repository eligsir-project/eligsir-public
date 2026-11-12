"""Task/issue provenance for experiment output directories (issue #511,
DAG lane P0-A of #202).

This module is deliberately **torch-free and side-effect-free** (apart from
the two explicit file helpers at the bottom) so it can be unit-tested in the
lean CPU CI image without any GPU or ``tests/conftest.py`` registration --
same constraint as ``utils/experiment_validity.py`` (issues #423/#462/#478).

Why this exists
---------------
``report.json["git_provenance"]`` (issue #259/#491) records **what
executed**: the commit, branch and dirty state of the checkout that ran.
It cannot answer **why** the run exists or **which tracked task owns the
result**. Historically that link lived only in the output directory *name*
(``output/issue494_E2``) plus whatever the launching agent happened to
remember -- neither survives the directory being copied to another machine,
detached from the ablator queue entry that launched it, or read six months
later. A directory name is also not machine-readable in any load-bearing
sense: nothing validates it, nothing rejects a typo, and a rename silently
rewrites history.

So: every run emits one small root-level ``experiment_origin.json``, and the
**same resolved object** is embedded as ``report.json["experiment_origin"]``.

Non-divergence
--------------
The file and the report key must never be able to disagree. The mechanism is
structural, not conventional:

1. ``resolve_experiment_origin_from_args(args)`` is called EXACTLY ONCE, at
   trainer startup (``train.py``'s ``__main__``), and stores the resolved
   object on ``args.experiment_origin``.
2. ``experiment_origin_from_args(args)`` -- what the report call sites use --
   is a **pure accessor**. It performs no resolution, no parsing and no
   defaulting; it returns a deep copy of the already-resolved object (a copy,
   so a later mutation of the report cannot drift the file's source object,
   or vice versa).

Consequently a change to the raw declaration fields *after* startup cannot
make the two artifacts differ: there is only one resolution event.
``tests/test_experiment_origin.py`` proves this by mutating the raw args
between writing the file and building the report.

Schema (``schema_version = 1``)
-------------------------------
``schema_version``     int, this module's ``SCHEMA_VERSION``.
``repository``         ``owner/name``. Defaults to this repo.
``issue``              int or None -- the owning GitHub issue.
``issue_url``          DERIVED from ``repository`` + ``issue``, never an input.
``pull_request``       int or None.
``pull_request_url``   DERIVED, never an input.
``task_kind``          DERIVED, one of ``github_issue`` / ``github_pull_request``
                       / ``ad_hoc`` / ``origin_unknown``.
``experiment_id``      str or None -- the sweep/experiment identifier.
``arm``                str or None -- which arm of that sweep this run is.
``purpose``            str or None -- what claim the run was meant to test.
``status``             ``complete`` / ``ad_hoc`` / ``incomplete_provenance`` /
                       ``origin_unknown``.
``research_experiment``        bool -- did this run look like tracked research?
``research_experiment_reasons`` list[str] -- why the classifier said so.
``origin_source``      dict[field -> layer name] -- which declaration layer
                       won each field (``cli`` / ``spec_file`` / ``env`` /
                       ``default``). Field-level precedence, so a spec file
                       can supply ``experiment_id``/``arm`` while the CLI
                       overrides only ``arm``.
``warnings``           list[str] -- surfaced loudly by the caller.

Deliberately NOT fabricated
---------------------------
A run with no declaration resolves to ``task_kind: origin_unknown`` and says
so. A hand-run exploratory job declares ``ad_hoc`` plus a ``purpose``. The
output directory name is never consulted -- not as a fallback, not as a hint.
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = 1
DEFAULT_REPOSITORY = "eligsir-project/eligsir-public"
ORIGIN_FILENAME = "experiment_origin.json"

#: JSON-encoded declaration layer, for launchers (ablator, agents, scripts)
#: that cannot easily add CLI flags. Lowest precedence.
ORIGIN_ENV_VAR = "ELIGSIR_EXPERIMENT_ORIGIN"

TASK_KIND_GITHUB_ISSUE = "github_issue"
TASK_KIND_GITHUB_PR = "github_pull_request"
TASK_KIND_AD_HOC = "ad_hoc"
TASK_KIND_UNKNOWN = "origin_unknown"

TASK_KINDS = (
    TASK_KIND_GITHUB_ISSUE,
    TASK_KIND_GITHUB_PR,
    TASK_KIND_AD_HOC,
    TASK_KIND_UNKNOWN,
)

STATUS_COMPLETE = "complete"
STATUS_AD_HOC = "ad_hoc"
STATUS_INCOMPLETE = "incomplete_provenance"
STATUS_UNKNOWN = "origin_unknown"

EVIDENCE_CLASS_MANAGED = "MANAGED"
EVIDENCE_CLASS_UNMANAGED = "UNMANAGED"
RESEARCHFLOW_RUN_ID_ENV = "RESEARCHFLOW_RUN_ID"
RESEARCHFLOW_PLAN_SHA_ENV = "RESEARCHFLOW_PLAN_SHA256"

#: The raw declaration fields a layer may set. ``issue_url``/``pull_request_url``
#: are absent on purpose: they are derived, never declared, so the object can
#: never be internally inconsistent.
DECLARED_FIELDS: Tuple[str, ...] = (
    "repository",
    "issue",
    "pull_request",
    "task_kind",
    "experiment_id",
    "arm",
    "purpose",
    "researchflow_run_id",
    "researchflow_plan_sha256",
    "evidence_class",
)

_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_HASH_NUM_RE = re.compile(r"^#?(\d+)$")
_URL_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/"
    r"(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/"
    r"(?P<kind>issues|pull)/"
    r"(?P<num>\d+)/?$"
)

_KIND_URL_SEGMENT = {
    "issue": "issues",
    "pull_request": "pull",
}


class ExperimentOriginError(ValueError):
    """A declared experiment origin is malformed or self-contradictory.

    Raised loudly at trainer startup rather than degraded to a warning: a
    typo'd issue number produces provenance that points at the WRONG tracked
    task, which is worse than no provenance at all.
    """


# --------------------------------------------------------------------------
# parsing / validation
# --------------------------------------------------------------------------

def parse_repository(value: Any) -> str:
    """Validate an ``owner/name`` repository slug."""
    if not isinstance(value, str) or not _REPO_RE.match(value.strip()):
        raise ExperimentOriginError(
            f"malformed repository {value!r} -- expected 'owner/name' "
            f"(e.g. {DEFAULT_REPOSITORY!r})"
        )
    return value.strip()


def parse_reference(value: Any, *, field: str, repository: str) -> int:
    """Parse an issue/PR reference into a positive int.

    Accepts ``494``, ``"494"``, ``"#494"`` and the full GitHub URL form.
    Rejects ``0``, negatives, bools, floats, junk strings, a URL whose
    ``kind`` segment contradicts ``field``, and a URL whose repository
    contradicts ``repository``.
    """
    if field not in _KIND_URL_SEGMENT:
        raise ExperimentOriginError(f"unknown reference field {field!r}")

    if isinstance(value, bool):
        raise ExperimentOriginError(
            f"malformed {field} reference {value!r} -- expected a positive integer")
    if isinstance(value, int):
        num = value
    elif isinstance(value, float):
        raise ExperimentOriginError(
            f"malformed {field} reference {value!r} -- expected an integer, not a float")
    elif isinstance(value, str):
        text = value.strip()
        m = _HASH_NUM_RE.match(text)
        if m:
            num = int(m.group(1))
        else:
            u = _URL_RE.match(text)
            if not u:
                raise ExperimentOriginError(
                    f"malformed {field} reference {value!r} -- expected 494, '494', "
                    f"'#494' or a https://github.com/owner/repo/"
                    f"{_KIND_URL_SEGMENT[field]}/494 URL"
                )
            if u.group("kind") != _KIND_URL_SEGMENT[field]:
                raise ExperimentOriginError(
                    f"{field} reference {value!r} is a "
                    f"{'pull request' if u.group('kind') == 'pull' else 'issue'} URL "
                    f"-- declare it as the matching field instead"
                )
            if u.group("repo") != repository:
                raise ExperimentOriginError(
                    f"{field} reference {value!r} points at repository "
                    f"{u.group('repo')!r} but this run declares {repository!r}"
                )
            num = int(u.group("num"))
    else:
        raise ExperimentOriginError(
            f"malformed {field} reference {value!r} of type {type(value).__name__}")

    if num <= 0:
        raise ExperimentOriginError(
            f"malformed {field} reference {value!r} -- must be a positive integer")
    return num


def _clean_text(value: Any, *, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ExperimentOriginError(
            f"{field} must be a string, got {type(value).__name__}: {value!r}")
    text = value.strip()
    return text or None


def reference_url(repository: str, number: Optional[int], *, field: str) -> Optional[str]:
    """Derive the canonical GitHub URL for a reference. Never an input."""
    if number is None:
        return None
    return f"https://github.com/{repository}/{_KIND_URL_SEGMENT[field]}/{number}"


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

def _merge_layers(
    layers: Sequence[Tuple[str, Mapping[str, Any]]],
) -> Tuple[dict, dict]:
    """Field-level precedence merge. Earlier layers win per FIELD, not
    wholesale -- so a spec file may supply ``experiment_id``/``arm`` while the
    CLI overrides only ``arm``. Returns ``(values, source_by_field)``.
    """
    values: dict = {}
    sources: dict = {}
    for layer_name, mapping in layers:
        if not mapping:
            continue
        unknown = set(mapping) - set(DECLARED_FIELDS)
        if unknown:
            raise ExperimentOriginError(
                f"unknown experiment_origin field(s) {sorted(unknown)!r} in "
                f"{layer_name} layer -- allowed: {list(DECLARED_FIELDS)}"
            )
        for field in DECLARED_FIELDS:
            if field in values:
                continue
            v = mapping.get(field)
            if v is None:
                continue
            values[field] = v
            sources[field] = layer_name
    return values, sources


def resolve_experiment_origin(
    layers: Sequence[Tuple[str, Mapping[str, Any]]] = (),
    *,
    research_experiment: bool = False,
    research_experiment_reasons: Optional[Iterable[str]] = None,
) -> dict:
    """Resolve declaration layers into the one canonical origin object.

    ``layers`` is an ordered sequence of ``(layer_name, mapping)`` pairs,
    HIGHEST precedence first (the trainer passes ``cli``, ``spec_file``,
    ``env``). Missing/None fields fall through to the next layer, then to a
    documented default (``repository``) or ``None``.

    Raises ``ExperimentOriginError`` for malformed or self-contradictory
    declarations. Never fabricates a task reference.
    """
    values, sources = _merge_layers(layers)

    repository = (
        parse_repository(values["repository"]) if "repository" in values
        else DEFAULT_REPOSITORY
    )
    sources.setdefault("repository", "default")

    issue = (
        parse_reference(values["issue"], field="issue", repository=repository)
        if "issue" in values else None
    )
    pull_request = (
        parse_reference(values["pull_request"], field="pull_request",
                        repository=repository)
        if "pull_request" in values else None
    )
    experiment_id = _clean_text(values.get("experiment_id"), field="experiment_id")
    arm = _clean_text(values.get("arm"), field="arm")
    purpose = _clean_text(values.get("purpose"), field="purpose")

    declared_kind = _clean_text(values.get("task_kind"), field="task_kind")
    if declared_kind is not None and declared_kind not in TASK_KINDS:
        raise ExperimentOriginError(
            f"unknown task_kind {declared_kind!r} -- expected one of {list(TASK_KINDS)}")

    # task_kind is DERIVED, never free text. An issue reference outranks a PR
    # reference: when a run carries both, the issue is the owning task and the
    # PR is the change under test.
    if issue is not None:
        derived_kind = TASK_KIND_GITHUB_ISSUE
    elif pull_request is not None:
        derived_kind = TASK_KIND_GITHUB_PR
    elif declared_kind == TASK_KIND_AD_HOC or (
        declared_kind is None and purpose is not None
    ):
        derived_kind = TASK_KIND_AD_HOC
    else:
        derived_kind = TASK_KIND_UNKNOWN

    if declared_kind is not None and declared_kind != derived_kind:
        raise ExperimentOriginError(
            f"declared task_kind {declared_kind!r} contradicts the declared "
            f"references (issue={issue!r}, pull_request={pull_request!r}, "
            f"purpose={'set' if purpose else 'unset'}) which imply "
            f"{derived_kind!r} -- refusing to silently prefer one"
        )
    task_kind = derived_kind

    if task_kind == TASK_KIND_AD_HOC and not purpose:
        raise ExperimentOriginError(
            "task_kind 'ad_hoc' requires a non-empty purpose describing what "
            "the run was for (an untracked run with no stated purpose is "
            "indistinguishable from origin_unknown)"
        )

    warnings: list = []
    reasons = list(research_experiment_reasons or [])
    if task_kind in (TASK_KIND_GITHUB_ISSUE, TASK_KIND_GITHUB_PR):
        status = STATUS_COMPLETE
    elif task_kind == TASK_KIND_AD_HOC:
        status = STATUS_AD_HOC
        if research_experiment:
            warnings.append(
                "run looks like a tracked research experiment "
                f"({', '.join(reasons) or 'classifier'}) but declares only "
                "task_kind 'ad_hoc' -- if it is evidence for an issue/PR, "
                "declare --experiment_origin_issue / --experiment_origin_pr"
            )
    elif research_experiment:
        status = STATUS_INCOMPLETE
        warnings.append(
            "INCOMPLETE PROVENANCE: this run looks like a tracked research "
            f"experiment ({', '.join(reasons) or 'classifier'}) but declares no "
            "owning issue/PR. Declare --experiment_origin_issue N (or "
            "--experiment_origin_pr N, or --experiment_origin_ad_hoc with "
            "--experiment_purpose). The output directory NAME is never used as "
            "task identity."
        )
    else:
        status = STATUS_UNKNOWN
    researchflow_run_id = _clean_text(values.get("researchflow_run_id"), field="researchflow_run_id")
    researchflow_plan_sha256 = _clean_text(values.get("researchflow_plan_sha256"), field="researchflow_plan_sha256")
    evidence_class = _clean_text(values.get("evidence_class"), field="evidence_class")

    return {
        "schema_version": SCHEMA_VERSION,
        "repository": repository,
        "issue": issue,
        "issue_url": reference_url(repository, issue, field="issue"),
        "pull_request": pull_request,
        "pull_request_url": reference_url(
            repository, pull_request, field="pull_request"),
        "task_kind": task_kind,
        "experiment_id": experiment_id,
        "arm": arm,
        "purpose": purpose,
        "researchflow_run_id": researchflow_run_id,
        "researchflow_plan_sha256": researchflow_plan_sha256,
        "evidence_class": evidence_class,
        "status": status,
        "research_experiment": bool(research_experiment),
        "research_experiment_reasons": reasons,
        "origin_source": {k: sources[k] for k in sorted(sources)},
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# declaration layers
# --------------------------------------------------------------------------

def _layer_from_spec_file(path: Optional[str]) -> Mapping[str, Any]:
    """Load ``--experiment_origin_file``: a small JSON document holding the
    declaration either at the top level or nested under ``experiment_origin``
    (so an existing spec file can carry it as one key).
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise ExperimentOriginError(f"--experiment_origin_file not found: {path}")
    try:
        with open(path, "r") as f:
            doc = json.load(f)
    except Exception as e:
        raise ExperimentOriginError(
            f"--experiment_origin_file {path} is not readable JSON: {e}") from e
    if not isinstance(doc, dict):
        raise ExperimentOriginError(
            f"--experiment_origin_file {path} must contain a JSON object")
    if "experiment_origin" in doc:
        doc = doc["experiment_origin"]
        if not isinstance(doc, dict):
            raise ExperimentOriginError(
                f"--experiment_origin_file {path}: 'experiment_origin' must be "
                "a JSON object")
    # A spec file may legitimately carry schema_version echoed back from a
    # previous run's artifact; ignore it rather than failing on it.
    return {k: v for k, v in doc.items() if k != "schema_version"}


def _layer_from_env(env: Optional[Mapping[str, str]] = None) -> Mapping[str, Any]:
    env = os.environ if env is None else env
    raw = env.get(ORIGIN_ENV_VAR, "")
    if not raw.strip():
        return {}
    try:
        doc = json.loads(raw)
    except Exception as e:
        raise ExperimentOriginError(
            f"${ORIGIN_ENV_VAR} is not valid JSON: {e}") from e
    if not isinstance(doc, dict):
        raise ExperimentOriginError(f"${ORIGIN_ENV_VAR} must be a JSON object")
    if "experiment_origin" in doc and isinstance(doc["experiment_origin"], dict):
        doc = doc["experiment_origin"]
    return {k: v for k, v in doc.items() if k != "schema_version"}


def classify_research_experiment(args) -> Tuple[bool, list]:
    """Best-effort "is this a tracked research experiment?" classifier.

    Deliberately conservative and NAMED (rather than implicit) so its gaps are
    reviewable: it fires on a declared ``--experiment_validity_class`` (issue
    #203) or a ``--baseline_run`` comparison (issue-driven A/B). A run that
    forgot BOTH of those and also forgot its origin is NOT caught -- that gap
    is stated in the PR rather than papered over.
    """
    reasons: list = []
    if args is None:
        return False, reasons
    if getattr(args, "experiment_validity_class", None):
        reasons.append(
            f"--experiment_validity_class "
            f"{getattr(args, 'experiment_validity_class')}")
    if getattr(args, "baseline_run", None):
        reasons.append("--baseline_run set")
    return bool(reasons), reasons


def is_valid_researchflow_origin(
    origin: Optional[Mapping[str, Any]] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[bool, Optional[str]]:
    """Check if valid ResearchFlow origin metadata is present via environment or origin dict."""
    env = os.environ if env is None else env
    rf_run_id = env.get(RESEARCHFLOW_RUN_ID_ENV, "").strip()
    rf_plan_sha = env.get(RESEARCHFLOW_PLAN_SHA_ENV, "").strip()

    if rf_run_id and rf_plan_sha:
        return True, None
    if (rf_run_id and not rf_plan_sha) or (rf_plan_sha and not rf_run_id):
        return False, (
            f"incomplete ResearchFlow environment: "
            f"{RESEARCHFLOW_RUN_ID_ENV}={'set' if rf_run_id else 'missing'}, "
            f"{RESEARCHFLOW_PLAN_SHA_ENV}={'set' if rf_plan_sha else 'missing'}"
        )

    if isinstance(origin, Mapping):
        r_id = origin.get("researchflow_run_id") or origin.get("run_id")
        p_sha = origin.get("researchflow_plan_sha256") or origin.get("plan_sha256")
        if r_id and p_sha and str(r_id).strip() and str(p_sha).strip():
            return True, None

        nested_rf = origin.get("researchflow")
        if isinstance(nested_rf, Mapping):
            n_rid = nested_rf.get("run_id") or nested_rf.get("researchflow_run_id")
            n_psha = nested_rf.get("plan_sha256") or nested_rf.get("researchflow_plan_sha256")
            if n_rid and n_psha and str(n_rid).strip() and str(n_psha).strip():
                return True, None

        # NOTE (issue #1013 follow-up, 2026-08-22): this used to also accept
        # origin.get("evidence_class") == EVIDENCE_CLASS_MANAGED, and
        # separately any origin with task_kind in
        # (github_issue, github_pull_request) + status=="complete" +
        # a truthy experiment_id -- i.e. just naming a GitHub issue and an
        # experiment_id on the CLI (`--experiment_origin_issue N
        # --experiment_id foo`), with NO real ResearchFlow run_id/plan_sha
        # anywhere. That self-certified evidence_class=MANAGED without ever
        # checking a real ResearchFlow registration, so every arm of
        # specs/champion_reproduction_mesh.json passed this check and got
        # stamped MANAGED despite being entirely absent from the actual
        # ResearchFlow Experiment Registry (confirmed: 0 matching entries).
        # This is exactly the "convenience flag ... to silently promote
        # unmanaged runs to canonical evidence" the mandatory-ResearchFlow
        # HARD CONSTRAINT in CLAUDE.md says must not exist -- removed rather
        # than tightened, since naming an issue/experiment_id is something
        # any CLI invocation can do and proves nothing about registration.
        pass

    return False, "Missing registered ResearchFlow origin metadata (neither RESEARCHFLOW_RUN_ID/RESEARCHFLOW_PLAN_SHA256 nor a real researchflow_run_id/plan_sha256 in experiment_origin.json)"


def validate_researchflow_origin(
    args: Any = None,
    origin: Optional[Mapping[str, Any]] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Fail closed immediately if scientific_mode is set and ResearchFlow origin metadata is missing or invalid."""
    scientific_mode = bool(getattr(args, "scientific_mode", False)) if args is not None else False
    if not scientific_mode:
        return

    if origin is None and args is not None:
        origin = getattr(args, "experiment_origin", None)

    valid, reason = is_valid_researchflow_origin(origin, env=env)
    if not valid:
        raise ValueError(
            f"Canonical scientific experiment requires registered ResearchFlow origin metadata: {reason}"
        )


def resolve_experiment_origin_from_args(args, *, env: Optional[Mapping[str, str]] = None) -> dict:
    """Resolve the origin ONCE from an argparse namespace.

    Call site: ``train.py``'s ``__main__``, which stores the result on
    ``args.experiment_origin``. Everything downstream reads that attribute
    through the pure accessor ``experiment_origin_from_args()``.
    """
    env_map = os.environ if env is None else env
    rf_run_id = env_map.get(RESEARCHFLOW_RUN_ID_ENV, "").strip() or None
    rf_plan_sha = env_map.get(RESEARCHFLOW_PLAN_SHA_ENV, "").strip() or None

    cli = {
        "repository": getattr(args, "experiment_origin_repo", None),
        "issue": getattr(args, "experiment_origin_issue", None),
        "pull_request": getattr(args, "experiment_origin_pr", None),
        "task_kind": (
            TASK_KIND_AD_HOC
            if getattr(args, "experiment_origin_ad_hoc", False) else None
        ),
        "experiment_id": getattr(args, "experiment_id", None),
        "arm": getattr(args, "experiment_arm", None),
        "purpose": getattr(args, "experiment_purpose", None),
        "researchflow_run_id": rf_run_id,
        "researchflow_plan_sha256": rf_plan_sha,
    }
    layers = [
        ("cli", {k: v for k, v in cli.items() if v is not None}),
        ("spec_file", _layer_from_spec_file(
            getattr(args, "experiment_origin_file", None))),
        ("env", _layer_from_env(env)),
    ]
    research, reasons = classify_research_experiment(args)
    origin = resolve_experiment_origin(
        layers,
        research_experiment=research,
        research_experiment_reasons=reasons,
    )

    if rf_run_id and not origin.get("researchflow_run_id"):
        origin["researchflow_run_id"] = rf_run_id
    if rf_plan_sha and not origin.get("researchflow_plan_sha256"):
        origin["researchflow_plan_sha256"] = rf_plan_sha

    scientific_mode = bool(getattr(args, "scientific_mode", False)) if args is not None else False
    if scientific_mode:
        validate_researchflow_origin(args, origin=origin, env=env)
        origin["evidence_class"] = EVIDENCE_CLASS_MANAGED
    else:
        origin["evidence_class"] = origin.get("evidence_class") or EVIDENCE_CLASS_UNMANAGED

    return origin


def experiment_origin_from_args(args) -> Optional[dict]:
    """PURE ACCESSOR for the already-resolved origin object.

    Performs no resolution, parsing or defaulting -- that happens exactly once,
    in ``resolve_experiment_origin_from_args()`` at startup. This is what makes
    ``experiment_origin.json`` and ``report.json["experiment_origin"]``
    structurally unable to diverge. Returns a deep copy so a downstream mutation
    of one artifact cannot drift the other.

    Returns ``None`` when the attribute is absent (a call site that never went
    through the trainer startup path), so the report writer degrades to its
    pre-#511 behavior instead of writing a misleading empty block.
    """
    if args is None:
        return None
    origin = getattr(args, "experiment_origin", None)
    if not isinstance(origin, dict):
        return None
    return copy.deepcopy(origin)


# --------------------------------------------------------------------------
# file I/O + presentation
# --------------------------------------------------------------------------

def format_origin_line(origin: Mapping[str, Any]) -> str:
    """One-line human-readable summary, printed at startup into train.log."""
    kind = origin.get("task_kind")
    if kind == TASK_KIND_GITHUB_ISSUE:
        ref = f"{origin.get('repository')} issue #{origin.get('issue')}"
    elif kind == TASK_KIND_GITHUB_PR:
        ref = f"{origin.get('repository')} PR #{origin.get('pull_request')}"
    elif kind == TASK_KIND_AD_HOC:
        ref = "ad-hoc (no tracked task)"
    else:
        ref = "UNKNOWN (no task declared)"
    bits = [f"[origin] {ref}"]
    if origin.get("experiment_id"):
        bits.append(f"experiment={origin['experiment_id']}")
    if origin.get("arm"):
        bits.append(f"arm={origin['arm']}")
    bits.append(f"status={origin.get('status')}")
    if origin.get("purpose"):
        bits.append(f"purpose={origin['purpose']!r}")
    return " | ".join(bits)


def write_experiment_origin_file(model_path: str, origin: Mapping[str, Any],
                                 persist_path: Optional[str] = None) -> list:
    """Write ``experiment_origin.json`` at the run output ROOT.

    Written to ``model_path`` and, when output staging is active and differs,
    also directly to ``persist_path`` -- mirroring what ``cfg_args`` already
    does, so the reference exists at the canonical path immediately instead of
    only after the staging mirror completes.

    Returns the list of paths written.
    """
    written = []
    seen = set()
    for root in [model_path, persist_path]:
        if not root:
            continue
        # realpath, not string equality: `/x` vs `/x/` vs a relative form all
        # name one directory and must not be written twice.
        os.makedirs(root, exist_ok=True)
        key = os.path.realpath(root)
        if key in seen:
            continue
        seen.add(key)
        path = os.path.join(root, ORIGIN_FILENAME)
        with open(path, "w") as f:
            json.dump(dict(origin), f, indent=2)
            f.write("\n")
        written.append(path)
    return written


def read_experiment_origin_file(model_path: str) -> Optional[dict]:
    """Read back an ``experiment_origin.json`` from a run root (or None)."""
    path = os.path.join(model_path, ORIGIN_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)
