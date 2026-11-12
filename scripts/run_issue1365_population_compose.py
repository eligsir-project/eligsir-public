"""Inject execute-time ResearchFlow values into the trusted Compose argv."""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence


_RUNTIME_ENV = ("RESEARCHFLOW_RUN_ID", "RESEARCHFLOW_PLAN_SHA256")


def inject_runtime_environment(
    argv: Sequence[str], environment: Mapping[str, str]
) -> list[str]:
    """Return Compose argv with generated ResearchFlow values as direct args."""
    command = list(argv)
    try:
        run_index = command.index("run")
    except ValueError as exc:
        raise ValueError("Compose command must contain `run`") from exc
    values: list[str] = []
    for name in _RUNTIME_ENV:
        value = str(environment.get(name, "")).strip()
        if not value:
            raise ValueError(f"missing runtime ResearchFlow environment: {name}")
        values.extend(("-e", f"{name}={value}"))
    return [*command[: run_index + 1], *values, *command[run_index + 1 :]]


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_issue1365_population_compose.py COMPOSE_ARG ...")
    command = inject_runtime_environment(sys.argv[1:], os.environ)
    return subprocess.call(command, env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
