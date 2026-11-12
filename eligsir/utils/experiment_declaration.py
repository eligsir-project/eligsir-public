"""EliGSiR-side consumer for Ablator ExperimentDeclaration (#695/#705).

This module deliberately does *not* import the Ablator Python package. The
cross-repository contract is a versioned, canonical JSON declaration plus a
SHA-256 supplied by the execution manager. EliGSiR verifies that contract,
classifies gradeability, persists it at run start, and renders the human banner.

Generic Ablator queue/submission provenance is a separate, best-effort contract
(issue #999). An Ablator-managed job may therefore be explicitly undeclared and
non-gradeable while still carrying trusted queue identity for reproduction.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


DECLARATION_ENV = "ABLATOR_EXPERIMENT_DECLARATION_JSON"
DECLARATION_SHA_ENV = "ABLATOR_EXPERIMENT_DECLARATION_SHA256"
JOB_ID_ENV = "ABLATOR_JOB_ID"
JOB_JSON_ENV = "ABLATOR_JOB_JSON"
SUBMISSION_ENV = "ABLATOR_SUBMISSION_JSON"

SUPPORTED_DECLARATION_SCHEMA_VERSION = 1
CONTEXT_SCHEMA_VERSION = 1
DECLARATION_RECEIPT_SCHEMA_VERSION = 1

GRADEABLE_RUN_CLASSES = frozenset({"experiment", "benchmark", "verification"})
NON_GRADEABLE_RUN_CLASSES = {
    "developer_smoke": "NON_GRADEABLE_DEVELOPER_SMOKE",
    "debug": "NON_GRADEABLE_DEBUG",
}


class ExperimentDeclarationError(RuntimeError):
    """Raised when a purported managed declaration is malformed/ambiguous."""


def canonical_declaration_json(declaration: Mapping[str, Any]) -> str:
    """Return the cross-repo canonical JSON representation."""
    return json.dumps(
        dict(declaration),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def declaration_sha256(declaration: Mapping[str, Any]) -> str:
    canonical = canonical_declaration_json(declaration).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def validate_declaration(declaration: Mapping[str, Any]) -> None:
    """Defense-in-depth validation of the generic cross-repo contract."""
    if not isinstance(declaration, Mapping):
        raise ExperimentDeclarationError("experiment declaration must be a JSON object")

    errors: list[str] = []
    schema_version = declaration.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != SUPPORTED_DECLARATION_SCHEMA_VERSION
    ):
        errors.append(
            "unsupported schema_version "
            f"{schema_version!r}; expected {SUPPORTED_DECLARATION_SCHEMA_VERSION}"
        )

    run_class = declaration.get("run_class")
    known_classes = GRADEABLE_RUN_CLASSES | set(NON_GRADEABLE_RUN_CLASSES)
    if run_class not in known_classes:
        errors.append(f"unsupported run_class {run_class!r}")

    for key in ("title", "purpose"):
        if not _present(declaration.get(key)):
            errors.append(f"missing non-empty {key}")

    if run_class in GRADEABLE_RUN_CLASSES:
        for key in ("experiment_id", "expected_evidence"):
            if not _present(declaration.get(key)):
                errors.append(f"missing non-empty {key}")

        if not (
            _present(declaration.get("owner_refs"))
            or _present(declaration.get("standalone_reason"))
        ):
            errors.append("gradeable declaration requires owner_refs or standalone_reason")

        arm = declaration.get("arm")
        if not isinstance(arm, Mapping):
            errors.append("gradeable declaration requires arm object")
        else:
            for key in ("id", "comparison_role", "manipulation"):
                if not _present(arm.get(key)):
                    errors.append(f"gradeable declaration arm missing non-empty {key}")

    if errors:
        raise ExperimentDeclarationError("; ".join(errors))


def _gradeability_for_run_class(run_class: str) -> str:
    if run_class in GRADEABLE_RUN_CLASSES:
        return "GRADEABLE_DECLARED"
    return NON_GRADEABLE_RUN_CLASSES[run_class]


def _direct_context() -> dict[str, Any]:
    return {
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "execution_management": "DIRECT",
        "gradeability": "NON_GRADEABLE_DIRECT",
        "ablator_job_id": None,
        "experiment_declaration_sha256": None,
        "experiment_declaration_json": None,
        "experiment_declaration": None,
    }


def _undeclared_ablator_context(job_id: str) -> dict[str, Any]:
    return {
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "execution_management": "ABLATOR",
        "gradeability": "NON_GRADEABLE_ABLATOR_UNDECLARED",
        "ablator_job_id": job_id,
        "experiment_declaration_sha256": None,
        "experiment_declaration_json": None,
        "experiment_declaration": None,
    }


def load_experiment_context(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Load and verify the execution-manager declaration from environment.

    A direct invocation has no Ablator environment and is non-gradeable. Generic
    ``ABLATOR_JOB_ID`` is scheduler identity, not declaration state: a job id may
    therefore be present without an immutable experiment declaration. Such a run
    is classified as *Ablator-managed but undeclared* and remains non-gradeable.

    Declaration ambiguity still fails closed: declaration JSON without its hash
    or job id, a hash without declaration JSON, malformed JSON, non-canonical
    bytes, invalid required fields, and hash mismatch all raise. Generic job or
    submission JSON without a job id also fails because the scheduler identity
    would be ambiguous.
    """
    env = os.environ if env is None else env
    raw = env.get(DECLARATION_ENV)
    claimed_sha = env.get(DECLARATION_SHA_ENV)
    job_id = env.get(JOB_ID_ENV)
    generic_provenance = bool(env.get(JOB_JSON_ENV) or env.get(SUBMISSION_ENV))

    if not raw:
        if claimed_sha:
            raise ExperimentDeclarationError(
                f"{DECLARATION_SHA_ENV} is set but {DECLARATION_ENV} is missing"
            )
        if generic_provenance and (not job_id or not job_id.strip()):
            raise ExperimentDeclarationError(
                f"Ablator launch provenance is present but {JOB_ID_ENV} is missing"
            )
        if job_id is not None and job_id.strip():
            return _undeclared_ablator_context(job_id.strip())
        # An EMPTY (or whitespace-only) job id with no declaration and no
        # generic Ablator provenance carries no scheduler identity at all,
        # so there is nothing ambiguous to fail closed on: it is
        # indistinguishable from the variable being unset. This is exactly
        # what a manual `docker compose run train` produces, because
        # docker-compose.yml passes `ABLATOR_JOB_ID: "${ABLATOR_JOB_ID:-}"`
        # and compose materializes that as a set-but-empty variable rather
        # than omitting it -- so raising here hard-failed EVERY manual
        # compose run at startup, contradicting that file's own "empty
        # defaults keep direct/manual compose runs explicitly non-gradeable
        # and unchanged" contract. Fail-closed still applies wherever the
        # identity really is ambiguous: an empty job id alongside generic
        # provenance (JOB_JSON_ENV/SUBMISSION_ENV) still raises above, and
        # an empty job id alongside a real declaration still raises below.
        return _direct_context()

    if not claimed_sha:
        raise ExperimentDeclarationError(
            f"managed experiment declaration is missing {DECLARATION_SHA_ENV}"
        )
    if not job_id or not job_id.strip():
        raise ExperimentDeclarationError(
            f"managed experiment declaration is missing {JOB_ID_ENV}"
        )

    try:
        declaration = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExperimentDeclarationError(
            f"{DECLARATION_ENV} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(declaration, dict):
        raise ExperimentDeclarationError(
            f"{DECLARATION_ENV} must decode to a JSON object"
        )

    validate_declaration(declaration)
    canonical_json = canonical_declaration_json(declaration)
    if raw != canonical_json:
        raise ExperimentDeclarationError(
            f"{DECLARATION_ENV} must be canonical JSON (sorted keys, compact "
            "separators, UTF-8 Unicode preserved, no trailing newline)"
        )
    computed_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if claimed_sha != computed_sha:
        raise ExperimentDeclarationError(
            "experiment declaration SHA-256 mismatch: "
            f"claimed={claimed_sha} computed={computed_sha}"
        )

    return {
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "execution_management": "ABLATOR",
        "gradeability": _gradeability_for_run_class(str(declaration["run_class"])),
        "ablator_job_id": job_id.strip(),
        "experiment_declaration_sha256": computed_sha,
        "experiment_declaration_json": raw,
        "experiment_declaration": declaration,
    }


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _multiline_field(lines: list[str], label: str, value: Any) -> None:
    if not _present(value):
        return
    if isinstance(value, (list, tuple)):
        lines.append(f"{label}:")
        for item in value:
            lines.append(f"  - {_text(item)}")
        return
    text = _text(value)
    chunks = text.splitlines() or [text]
    lines.append(f"{label}: {chunks[0]}")
    lines.extend(f"  {chunk}" for chunk in chunks[1:])


def render_experiment_banner(context: Mapping[str, Any]) -> str:
    """Render the deterministic human-readable #695 startup banner."""
    rule = "=" * 80
    management = context.get("execution_management")
    gradeability = context.get("gradeability", "UNKNOWN")
    declaration = context.get("experiment_declaration")

    if management == "ABLATOR" and not isinstance(declaration, Mapping):
        return "\n".join(
            [
                rule,
                "ELIGSIR RUN — NON-GRADEABLE ABLATOR-MANAGED UNDECLARED RUN",
                rule,
                "Execution management: ABLATOR",
                f"Ablator job ID: {context.get('ablator_job_id')}",
                f"Gradeability: {gradeability}",
                "Experiment declaration: MISSING",
                "The queue/scheduler identity is preserved, but no gradeable experiment declaration exists.",
                "Do not use it as gradeable baseline, champion, or paper evidence.",
                rule,
            ]
        )

    if management != "ABLATOR" or not isinstance(declaration, Mapping):
        return "\n".join(
            [
                rule,
                "ELIGSIR RUN — NON-GRADEABLE DIRECT RUN",
                rule,
                "Execution management: DIRECT",
                f"Gradeability: {gradeability}",
                "Experiment declaration: MISSING",
                "This run was not launched through the canonical experiment runner.",
                "Do not use it as baseline, experiment evidence, champion evidence, or paper evidence.",
                rule,
            ]
        )

    arm = declaration.get("arm") if isinstance(declaration.get("arm"), Mapping) else {}
    lines = [
        rule,
        "ELIGSIR RUN",
        rule,
        f"Title: {declaration.get('title')}",
        f"Ablator job ID: {context.get('ablator_job_id')}",
        f"Experiment ID: {declaration.get('experiment_id', '-')}",
        f"Run class: {declaration.get('run_class')}",
        f"Gradeability: {gradeability}",
        f"Declaration SHA256: {context.get('experiment_declaration_sha256')}",
        f"Declaration JSON: {context.get('experiment_declaration_json')}",
        f"Arm: {arm.get('id', '-')}",
        f"Comparison role: {arm.get('comparison_role', '-')}",
    ]
    if _present(arm.get("title")):
        lines.append(f"Arm title: {_text(arm.get('title'))}")

    _multiline_field(lines, "Purpose", declaration.get("purpose"))
    _multiline_field(lines, "Manipulation", arm.get("manipulation"))
    _multiline_field(lines, "Expected evidence / discriminator", declaration.get("expected_evidence"))
    _multiline_field(lines, "Hypothesis", declaration.get("hypothesis"))
    _multiline_field(lines, "Supports the hypothesis if", declaration.get("supports_if"))
    _multiline_field(lines, "Refutes or weakens the hypothesis if", declaration.get("refutes_if"))
    _multiline_field(lines, "Owner refs", declaration.get("owner_refs"))
    _multiline_field(lines, "Standalone reason", declaration.get("standalone_reason"))
    _multiline_field(lines, "Comparison group", declaration.get("comparison_group_id"))
    _multiline_field(lines, "Decision profile", declaration.get("decision_profile_id"))
    lines.append(rule)
    return "\n".join(lines)


def persist_experiment_context(
    model_path: str | os.PathLike[str],
    context: Mapping[str, Any],
    *,
    filename: str = "experiment_declaration.json",
) -> Path:
    """Immutably persist the declared-before-run context."""
    return _write_immutable_json(Path(model_path) / filename, context)


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Create deterministic JSON once, allowing only byte-identical replay."""
    encoded = (
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as fh:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ExperimentDeclarationError(
                f"refusing to replace immutable experiment declaration artifact: {path}"
            )
    return path


def persist_experiment_declaration_receipt(
    model_path: str | os.PathLike[str],
    context: Mapping[str, Any],
    *,
    filename: str = "experiment_declaration_receipt.json",
) -> Path:
    """Persist #705's declaration-only receipt without claiming #655 scope."""
    receipt = {
        "artifact_type": "experiment_declaration_receipt",
        "receipt_schema_version": DECLARATION_RECEIPT_SCHEMA_VERSION,
        **dict(context),
    }
    return _write_immutable_json(Path(model_path) / filename, receipt)


def initialize_experiment_context(
    model_path: str | os.PathLike[str],
    *,
    env: Mapping[str, str] | None = None,
    emit=print,
) -> dict[str, Any]:
    """Load, verify, persist, and emit the public experiment declaration."""
    source_env = os.environ if env is None else env
    context = load_experiment_context(source_env)
    persist_experiment_context(model_path, context)
    persist_experiment_declaration_receipt(model_path, context)

    # Queue-specific reproduction hooks are intentionally outside the public
    # runtime closure. Preserve an explicit diagnostic for managed callers,
    # without importing campaign-only code into map startup.
    if source_env.get(JOB_ID_ENV) or source_env.get(JOB_JSON_ENV) or source_env.get(SUBMISSION_ENV):
        emit(
            "[eligsir][managed-provenance] queue reproduction hooks are outside "
            "the publication runtime closure",
            flush=True,
        )

    emit(render_experiment_banner(context), flush=True)
    return context
