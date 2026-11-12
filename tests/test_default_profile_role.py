"""`configs/profiles/default.yml` is a curated base POLICY profile, not a snapshot.

Issue #182 acceptance criterion 2 asked for this file's role to be decided and
documented, because it sat in an "ambiguous middle state": its header claimed its
keys were "kept IN SYNC with the argparse defaults", which reads as a parity
promise it does not actually make.

The decision recorded here (#182 part 2, 2026-08-14): **canonical base policy
profile.** It carries a small, curated set of run-policy keys that profile-based
runs should inherit; it is explicitly NOT a mirror of argparse. The generated
full snapshot already exists separately as
`configs/profiles/defaults.reference.yml` (enforced by
`tests/test_defaults_reference_freshness.py`), so making this file a second
generated snapshot would duplicate it for no gain.

That decision is enforceable without changing any runtime value, via the two
checks below:

1. every key in `default.yml` is a real argparse dest -- catches a typo'd or
   since-removed key silently becoming an inert no-op (a genuine drift class
   that nothing else in the tree checks);
2. every key whose VALUE differs from the argparse default is declared in
   `INTENTIONAL_DIVERGENCES` with a reason -- so a divergence is always a
   deliberate, reviewed policy statement rather than accidental drift.

Deliberately NOT asserted: that `default.yml`'s values equal the argparse
defaults. Such a test would fail today on `streaming_replay`, and the only way to
make it pass is to flip a default that changes the behaviour of every
config-driven run -- exactly the behaviour-changing promotion #182 gates behind
matched-seed GPU A/Bs. The declared-divergence registry is the honest form of the
same protection.
"""
from __future__ import annotations

import yaml

from scripts.dump_default_config import _REPO_ROOT, collect_defaults
from eligsir.cli.parser import build_training_parser

DEFAULT_PROFILE_PATH = _REPO_ROOT / "configs" / "profiles" / "default.yml"

#: Keys `default.yml` sets to something OTHER than the argparse default, each
#: with the reason it is a deliberate policy choice. Adding an entry here is a
#: review checkpoint, not a formality: it means bare-CLI runs and profile-based
#: runs will behave differently on that key.
INTENTIONAL_DIVERGENCES: dict[str, str] = {
    "execution_trainer": (
        "the parser defaults to offline_reconstruction, while the curated "
        "canonical policy profile explicitly selects causal_mapping. Legacy "
        "streaming_replay is a rejected migration input, never a profile route."
    ),
}


def _profile_keys() -> dict:
    data = yaml.safe_load(DEFAULT_PROFILE_PATH.read_text())
    assert isinstance(data, dict), f"{DEFAULT_PROFILE_PATH.name} must be a YAML mapping"
    # `extends` is consumed by the loader (eligsir/cli/config.py pops it) and
    # is not an argparse option.
    data.pop("extends", None)
    return data


def test_every_key_is_a_real_argparse_dest():
    """A key that is not an argparse dest is an inert no-op, silently."""
    defaults = _all_parser_defaults()
    unknown = sorted(k for k in _profile_keys() if k not in defaults)
    assert not unknown, (
        f"{DEFAULT_PROFILE_PATH.name} sets keys that are not argparse options, so "
        f"they do nothing at runtime: {unknown}. Remove them, or fix the spelling "
        f"against arguments/__init__.py."
    )


def test_value_divergences_from_argparse_are_declared():
    """Any value differing from argparse must be a declared policy choice."""
    defaults = _all_parser_defaults()
    undeclared = {
        k: (v, defaults[k])
        for k, v in _profile_keys().items()
        if k in defaults and v != defaults[k] and k not in INTENTIONAL_DIVERGENCES
    }
    assert not undeclared, (
        f"{DEFAULT_PROFILE_PATH.name} diverges from the argparse default on keys "
        f"that are not declared in INTENTIONAL_DIVERGENCES (profile, argparse): "
        f"{undeclared}.\nEither restore parity, or add the key to "
        f"INTENTIONAL_DIVERGENCES in this file with the reason it is deliberate."
    )


def test_declared_divergences_are_not_stale():
    """A declared divergence that no longer diverges must be retired."""
    defaults = _all_parser_defaults()
    profile = _profile_keys()
    stale = [
        k
        for k in INTENTIONAL_DIVERGENCES
        if k not in profile or (k in defaults and profile[k] == defaults[k])
    ]
    assert not stale, (
        f"INTENTIONAL_DIVERGENCES lists keys that no longer diverge from argparse "
        f"(or are no longer set by {DEFAULT_PROFILE_PATH.name}): {stale}. Remove "
        f"them so the registry keeps meaning something."
    )


def test_default_yml_stays_a_curated_subset_not_a_full_snapshot():
    """The role decision itself: a policy profile, not a generated mirror.

    If someone regenerates this file as a full argparse dump it stops being a
    reviewable policy statement and silently duplicates defaults.reference.yml.
    """
    profile = _profile_keys()
    defaults = _all_parser_defaults()
    assert len(profile) < len(defaults) / 2, (
        f"{DEFAULT_PROFILE_PATH.name} now carries {len(profile)} of "
        f"{len(defaults)} argparse options. It is defined as a CURATED BASE "
        f"POLICY profile (see this module's docstring and the file's own header); "
        f"the generated full snapshot is configs/profiles/defaults.reference.yml. "
        f"If the role is being deliberately changed, change it in both places."
    )


def _all_parser_defaults() -> dict:
    """Combine legacy ParamGroup and canonical execution parser defaults."""
    defaults = dict(collect_defaults())
    parser = build_training_parser().parser
    defaults.update(
        {
            action.dest: action.default
            for action in parser._actions
            if action.dest != "help"
        }
    )
    return defaults
