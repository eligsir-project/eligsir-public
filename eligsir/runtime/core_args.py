"""Translate trainer CLI tokens into a validated mapping-runtime override dict.

Why this exists
---------------
``create_production_mapping_application`` has always accepted an ``overrides``
dict that is merged over the resolved run profile
(:func:`eligsir.runtime.production_application._resolve_parameters`), but
nothing outside the trainer could *produce* one: the only textual, tuned,
copy-pasteable form a EliGSiR recipe has ever had is a ``train.py`` command
line. The canonical ROS2 lifecycle node therefore had no way to express a tuned
scene recipe at all and silently ran whatever the profile defaulted to -- found
2026-08-29 during the floor2 GPU campaign (eligsir#1259).

This module is that missing translation, and it is deliberately the *only* one:
it reuses :func:`eligsir.cli.parser.build_training_parser` -- the exact
parser ``train.py`` itself parses with -- rather than reimplementing flag
parsing, so a token list here means precisely what it means on a trainer command
line, including types, ``store_true`` semantics and alias spellings.

Explicitness, not a defaults diff
---------------------------------
Only flags a caller *named* become overrides. Diffing the parsed namespace
against argparse defaults would be wrong, not merely imprecise: ``overrides`` is
merged over the run **profile**, not over argparse defaults, so
``--cap_max 500000`` (which happens to equal the argparse default) intending to
override a profile's ``600000`` would silently vanish. The explicit set is
derived from argparse's own ``_option_string_actions`` table, so option spelling
and ``dest`` mapping stay owned by the parser.

Fail-closed rules
-----------------
* Unknown flags are rejected loudly (a typo is never silently dropped).
* Abbreviated flags are rejected: only exact option strings are accepted, so a
  prefix that argparse would happily expand today cannot change meaning when a
  new flag is added later.
* ``-h``/``--help`` is rejected rather than being allowed to ``SystemExit`` a
  long-running host process.
* Flags naming state a transport owns (see :data:`RESERVED_DESTS` /
  :data:`RESERVED_DEST_PREFIXES`) are rejected, never silently applied: the
  transport's own declared parameters always win, and a conflict is an error
  rather than a last-writer-wins race.
* Flags that parse but are not valid run-config keys (e.g. trainer-process-only
  switches like ``--test_iterations``) are rejected here, where the message can
  say so, instead of failing later inside pydantic ``extra="forbid"``.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

#: Dests naming state the transport/deployment owns, not the mapping recipe.
#: ``model_path`` is owned by the node's ``mapping_model_path`` parameter;
#: ``source_path`` (and the ``ros2_live_*``/``orbbec_*`` families) describe an
#: ingestion path that the ROS2 boundary replaced entirely. ``config`` and
#: ``run_profile`` would introduce a second, competing config root.
RESERVED_DESTS = frozenset(
    {"model_path", "source_path", "config", "run_profile", "start_checkpoint"}
)

#: Dest prefixes reserved for the same reason as :data:`RESERVED_DESTS`.
RESERVED_DEST_PREFIXES = ("ros2_live_", "orbbec_")

_NEGATIVE_NUMBER = re.compile(r"^-(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


class CoreArgsError(ValueError):
    """Raised when core CLI tokens cannot become a valid override dict."""


def _reserved_reason(dest: str) -> str | None:
    if dest in RESERVED_DESTS:
        return f"{dest!r} is owned by the transport/deployment, not the mapping recipe"
    for prefix in RESERVED_DEST_PREFIXES:
        if dest.startswith(prefix):
            return (
                f"{dest!r} configures a retired in-core ingestion path "
                f"(reserved prefix {prefix!r}); ingress is owned by the transport"
            )
    return None


def _explicit_dests(parser: Any, tokens: Sequence[str]) -> list[str]:
    """Return the dests explicitly named by ``tokens``, in first-seen order."""

    table = parser._option_string_actions  # argparse's own option table
    dests: list[str] = []
    for token in tokens:
        if not token.startswith("-") or token == "-" or token == "--":
            continue  # a positional/option value, not a flag
        head = token.split("=", 1)[0]
        action = table.get(head)
        if action is not None:
            if action.dest not in dests:
                dests.append(action.dest)
            continue
        if _NEGATIVE_NUMBER.match(token):
            continue  # a negative numeric value such as `--debug_from -1`
        raise CoreArgsError(
            f"unknown core argument {head!r}: it is not an exact option string of "
            "the canonical trainer parser (abbreviations are refused on purpose). "
            "Check the spelling against arguments/__init__.py."
        )
    return dests


def core_args_to_overrides(tokens: Sequence[str]) -> dict[str, Any]:
    """Parse trainer CLI ``tokens`` into an explicit run-config override dict.

    The returned mapping is exactly ``{dest: value}`` for the flags named in
    ``tokens``, with values produced by the canonical trainer parser, and is
    suitable as ``create_production_mapping_application(overrides=...)``.
    """

    tokens = list(tokens)
    for token in tokens:
        if not isinstance(token, str):
            raise CoreArgsError(f"core arguments must be strings, got {token!r}")
    if not tokens:
        return {}
    for helpish in ("-h", "--help"):
        if helpish in tokens:
            raise CoreArgsError(
                "refusing to handle --help in core arguments: it would terminate "
                "the host process instead of configuring a run"
            )

    from eligsir.cli.parser import build_training_parser
    from eligsir.cli.run_config import build_run_config_model

    bundle = build_training_parser(description="core argument override translation")
    parser = bundle.parser
    dests = _explicit_dests(parser, tokens)

    def _refuse(message: str = "", *_args: Any, **_kwargs: Any):
        raise CoreArgsError(f"core arguments rejected by the trainer parser: {message}")

    parser.error = _refuse  # type: ignore[method-assign]
    parser.exit = _refuse  # type: ignore[method-assign]
    try:
        args = parser.parse_args(tokens)
    except CoreArgsError:
        raise
    except SystemExit as exc:  # pragma: no cover - defensive
        raise CoreArgsError(f"core arguments rejected by the trainer parser ({exc})") from exc

    reserved = [
        f"--{dest} ({reason})"
        for dest, reason in ((d, _reserved_reason(d)) for d in dests)
        if reason is not None
    ]
    if reserved:
        raise CoreArgsError(
            "refusing core arguments that collide with transport-owned "
            "configuration: " + "; ".join(reserved)
        )

    allowed = set(build_run_config_model().model_fields)
    not_config = [d for d in dests if d not in allowed]
    if not_config:
        raise CoreArgsError(
            "these flags parse but are not mapping run-config keys (they are "
            "trainer-process-only switches and cannot be applied to a running "
            "mapping application): " + ", ".join(f"--{d}" for d in not_config)
        )

    return {dest: getattr(args, dest) for dest in dests}


__all__ = [
    "RESERVED_DESTS",
    "RESERVED_DEST_PREFIXES",
    "CoreArgsError",
    "core_args_to_overrides",
]
