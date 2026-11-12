"""Launch exactly one pre-registered #1365 population-policy arm.

ResearchFlow v1 expands a Cartesian matrix and deliberately has no conditional
argv templating.  This small trusted-profile entrypoint keeps the arm selector
semantic while making the two mutually exclusive Boolean controls explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from collections.abc import Mapping

# When invoked as ``python scripts/<launcher>.py``, Python places only the
# scripts directory on sys.path.  Resolve the checkout root explicitly so the
# trusted entrypoint works from the container-mounted source tree without a
# caller-provided PYTHONPATH.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eligsir.population_control_deprecation import (
    LEGACY_POPULATION_CONTROL_ARG_PREFIXES,
)


_POLICY_ARGS = {
    "legacy_compatibility": (
        "--no-streaming_population_live_enabled",
        "--no-streaming_population_shadow_enabled",
    ),
    "live_probabilistic": (
        "--streaming_population_live_enabled",
        "--streaming_population_shadow_enabled",
        "--no-streaming_lifecycle_enabled",
        "--no-streaming_post_mapping_mature_mobility",
        "--no-streaming_depth_lock_centers",
        "--no-streaming_insert_from_depth",
    ),
}
_OWNED_POLICY_PREFIXES = ("--streaming_population_",)
_REQUIRED_PROVENANCE_ENV = (
    "RESEARCHFLOW_RUN_ID",
    "RESEARCHFLOW_PLAN_SHA256",
    "RESEARCHFLOW_SOURCE_MANIFEST_SHA256",
    "RESEARCHFLOW_EVALUATION_MANIFEST_SHA256",
    "RESEARCHFLOW_CONTAINER_DIGEST",
    "RESEARCHFLOW_HOST_IDENTITY_SHA256",
    "RESEARCHFLOW_ACTIVATION_MANIFEST_SHA256",
    "RESEARCHFLOW_METRICS_CONTRACT_SHA256",
    "RESEARCHFLOW_METRICS_PATH",
    "ELIGSIR_GIT_COMMIT",
)


def training_args_for_arm(arm: str, training_args: Sequence[str]) -> list[str]:
    """Return immutable policy flags appended to otherwise shared train args."""
    if arm not in _POLICY_ARGS:
        raise ValueError(f"unsupported population-policy arm: {arm}")
    if any(argument.startswith(_OWNED_POLICY_PREFIXES) for argument in training_args):
        raise ValueError("population policy flags are owned by the launcher")
    if arm == "live_probabilistic" and any(
        argument.startswith(LEGACY_POPULATION_CONTROL_ARG_PREFIXES)
        for argument in training_args
    ):
        raise ValueError(
            "population policy flags are owned by the launcher; legacy population "
            "controls are incompatible with the live "
            "probabilistic arm"
        )
    return [*training_args, *_POLICY_ARGS[arm]]


def require_immutable_provenance(environment: Mapping[str, str]) -> None:
    """Reject absent and template values before a GPU process can start."""
    missing = [
        name
        for name in _REQUIRED_PROVENANCE_ENV
        if not str(environment.get(name, "")).strip()
    ]
    unresolved = [
        name
        for name in _REQUIRED_PROVENANCE_ENV
        if str(environment.get(name, "")).strip().startswith("UNRESOLVED_")
    ]
    if missing:
        raise ValueError("missing immutable provenance: " + ", ".join(missing))
    if unresolved:
        raise ValueError("unresolved immutable provenance: " + ", ".join(unresolved))


def _sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_pre_dispatch_contract(args: argparse.Namespace, environment: Mapping[str, str]) -> None:
    """Verify plan-bound provenance and refuse an unimplemented metric producer."""
    require_immutable_provenance(environment)
    expected = {
        args.source_manifest: environment["RESEARCHFLOW_SOURCE_MANIFEST_SHA256"],
        args.evaluation_manifest: environment["RESEARCHFLOW_EVALUATION_MANIFEST_SHA256"],
        args.host_manifest: environment["RESEARCHFLOW_HOST_IDENTITY_SHA256"],
        args.activation_manifest: environment["RESEARCHFLOW_ACTIVATION_MANIFEST_SHA256"],
        args.metrics_contract: environment["RESEARCHFLOW_METRICS_CONTRACT_SHA256"],
    }
    for path, digest in expected.items():
        if _sha256(path) != digest:
            raise ValueError(f"plan-bound provenance digest mismatch: {path}")
    with open(args.container_manifest, encoding="utf-8") as handle:
        container = json.load(handle)
    if container.get("container_digest") != environment["RESEARCHFLOW_CONTAINER_DIGEST"]:
        raise ValueError("plan-bound container digest mismatch")
    with open(args.metrics_contract, encoding="utf-8") as handle:
        contract = json.load(handle)
    if contract.get("producer_status") != "READY":
        raise ValueError("metrics producer is not ready; launch remains blocked")
    if contract.get("metrics_path") != "$RESEARCHFLOW_METRICS_PATH":
        raise ValueError("plan-bound metrics path mismatch")


def model_path_from_training_args(training_args: Sequence[str]) -> str:
    """Resolve the one immutable model root for post-run metrics assembly."""
    model_paths: list[str] = []
    index = 0
    while index < len(training_args):
        argument = training_args[index]
        if argument in {"-m", "--model_path"}:
            if index + 1 >= len(training_args):
                raise ValueError("model path option requires a value")
            model_paths.append(training_args[index + 1])
            index += 2
            continue
        if argument.startswith("--model_path="):
            model_paths.append(argument.split("=", 1)[1])
        index += 1
    if len(model_paths) != 1 or not model_paths[0]:
        raise ValueError("exactly one model path is required for metric assembly")
    return model_paths[0]


def assemble_metrics_after_success(
    *, model_path: str, args: argparse.Namespace, environment: Mapping[str, str]
) -> int:
    """Run the fail-closed producer only after a successful training process."""
    command = [
        sys.executable,
        "scripts/assemble_issue1365_population_researchflow_metrics.py",
        "--model-path",
        model_path,
        "--output",
        environment["RESEARCHFLOW_METRICS_PATH"],
        "--contract",
        args.metrics_contract,
    ]
    # The production launcher always supplies this path; retaining the small
    # compatibility branch keeps callers of this helper focused on command
    # construction and does not weaken the CLI path.
    if getattr(args, "evaluation_manifest", None):
        command.extend(["--evaluation-manifest", args.evaluation_manifest])
    return subprocess.call(command, env=dict(environment))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-arm", choices=sorted(_POLICY_ARGS), required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--container-manifest", required=True)
    parser.add_argument("--host-manifest", required=True)
    parser.add_argument("--activation-manifest", required=True)
    parser.add_argument("--metrics-contract", required=True)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    training_args = list(parsed.training_args)
    if training_args[:1] == ["--"]:
        training_args = training_args[1:]
    if not training_args:
        parser.error("training arguments after -- are required")
    try:
        require_pre_dispatch_contract(parsed, os.environ)
    except ValueError as error:
        parser.error(str(error))
    try:
        model_path = model_path_from_training_args(training_args)
    except ValueError as error:
        parser.error(str(error))
    training_result = subprocess.call(
        [
            sys.executable,
            "train.py",
            *training_args_for_arm(parsed.population_arm, training_args),
        ]
    )
    if training_result != 0:
        return training_result
    return assemble_metrics_after_success(
        model_path=model_path,
        args=parsed,
        environment=os.environ,
    )


if __name__ == "__main__":
    raise SystemExit(main())
