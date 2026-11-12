"""Pins the CLI -> ``GaussianEnergyConfig`` wiring (issue #385).

Before #385, every production construction site of ``GaussianEnergyConfig``
hand-listed its own subset of fields, so any field a site forgot was pinned at
the dataclass default no matter what the CLI said. The worst case was
``utils/strategies/mcmc/energy.py``'s ``_pcfg`` (and its legacy-layout twin),
which drives ``compute_legacy_dead_mask`` -- real Gaussian birth/death topology
on the default ``gsplat_energy_mcmc`` strategy -- while pinning five of the
nine fields that method reads.

The fix routes every site through ``energy_config_from_args``. These tests pin
the four properties that keep it honest:

1. Every tunable dataclass field is mapped to a flag, and every mapped flag is
   really declared by the live trainer parser (no name drift -- the exact
   failure mode issue #382 fixed for ``support_threshold``).
2. Every mapped flag's parser default equals the dataclass default. This is
   the behaviour-neutrality proof for #385: making a previously-pinned field
   reachable cannot change results at default settings, and it also means the
   factory's dataclass-sourced fallback can never disagree with the parser.
3. Each flag actually reaches its field through the factory (set a distinct
   sentinel per flag; read it back off the built config).
4. ``alpha_dead`` stays per-call-site explicit -- the schedule-driven sites
   must keep taking ``sched["dead_opacity_threshold"]`` and must NOT be
   silently switched to ``args.energy_alpha_dead``, whose parser default (0.05)
   is 10x the dataclass default (0.005). That swap would be a real topology
   change; a per-flag "flag reaches field" test would not catch it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eligsir.utils.strategies.mcmc.energy_policy import (  # noqa: E402
    _ARGS_FIELD_MAP,
    _ARGS_FIELD_MAP_EXEMPT,
    GaussianEnergyConfig,
    energy_config_from_args,
)


def _parser_defaults() -> dict:
    """{flag: default} from the live, full trainer parser.

    Reuses issue #368 / PR #380's audit helper so this test and the
    getattr-fallback CI guard share one definition of "the real default".
    """
    from scripts.audit_getattr_fallbacks import real_defaults

    return real_defaults()


# --------------------------------------------------------------------------
# 1. Coverage: every tunable field mapped, every mapped flag really exists
# --------------------------------------------------------------------------


def test_every_dataclass_field_is_mapped_or_explicitly_exempt():
    fields = set(GaussianEnergyConfig.__dataclass_fields__)
    covered = set(_ARGS_FIELD_MAP) | set(_ARGS_FIELD_MAP_EXEMPT)
    missing = fields - covered
    assert not missing, (
        "GaussianEnergyConfig gained field(s) that are neither CLI-mapped nor "
        f"explicitly exempt: {sorted(missing)}. Add them to _ARGS_FIELD_MAP "
        "(with a flag in arguments/__init__.py) or to _ARGS_FIELD_MAP_EXEMPT "
        "with a reason -- otherwise they are silently unreachable (issue #385)."
    )
    stale = set(_ARGS_FIELD_MAP) - fields
    assert not stale, f"_ARGS_FIELD_MAP names non-existent field(s): {sorted(stale)}"


def test_every_mapped_flag_is_declared_by_the_real_parser():
    declared = _parser_defaults()
    unknown = {
        field: flag for field, flag in _ARGS_FIELD_MAP.items() if flag not in declared
    }
    assert not unknown, (
        "Mapped flag(s) are not declared by the live trainer parser -- this is "
        f"exactly the name drift issue #382 fixed: {unknown}"
    )


# --------------------------------------------------------------------------
# 2. Behaviour neutrality: parser default == dataclass default
# --------------------------------------------------------------------------


def test_parser_defaults_match_dataclass_defaults():
    """No mapped field may change behaviour at default settings.

    ``alpha_dead`` is deliberately NOT in ``_ARGS_FIELD_MAP`` precisely because
    it is the one field where this does not hold: the parser's
    ``energy_alpha_dead`` default is 0.05 against a dataclass default of 0.005.
    """
    declared = _parser_defaults()
    dataclass_defaults = GaussianEnergyConfig()
    mismatches = {}
    for field, flag in _ARGS_FIELD_MAP.items():
        parser_value = declared[flag]
        dataclass_value = getattr(dataclass_defaults, field)
        if parser_value != dataclass_value:
            mismatches[field] = (flag, parser_value, dataclass_value)
    assert not mismatches, (
        "parser default != dataclass default for: "
        f"{mismatches}. Routing these through energy_config_from_args would "
        "CHANGE TRAINING BEHAVIOUR at default settings -- do not merge without "
        "a measured, matched-operating-point comparison (issue #385 req. 3)."
    )


def test_alpha_dead_is_the_documented_exception():
    """Guard the premise of the test above, so it can't silently go vacuous."""
    declared = _parser_defaults()
    assert "alpha_dead" not in _ARGS_FIELD_MAP
    assert "alpha_dead" in _ARGS_FIELD_MAP_EXEMPT
    assert GaussianEnergyConfig().alpha_dead == 0.005
    assert declared["energy_alpha_dead"] == 0.05


# --------------------------------------------------------------------------
# 3. Each flag really reaches its field
# --------------------------------------------------------------------------


def _sentinel(default, i: int):
    """A value distinguishable from ``default`` and from other fields'."""
    if isinstance(default, bool):
        return not default
    if isinstance(default, float):
        return 100.0 + i
    if isinstance(default, str):
        return f"__sentinel_{i}__"
    raise AssertionError(f"unhandled field type {type(default)!r}")


@pytest.mark.parametrize("field,flag", sorted(_ARGS_FIELD_MAP.items()))
def test_flag_reaches_config_field(field, flag):
    defaults = GaussianEnergyConfig()
    sentinel = _sentinel(getattr(defaults, field), 1)
    # Only this one flag is present on args; everything else falls back.
    args = SimpleNamespace(**{flag: sentinel})
    cfg = energy_config_from_args(args, alpha_dead=0.5)
    assert getattr(cfg, field) == sentinel, (
        f"--{flag} does not reach GaussianEnergyConfig.{field}"
    )


def test_all_flags_reach_their_fields_simultaneously():
    """Catches a map whose entries collide on one flag/field pair."""
    defaults = GaussianEnergyConfig()
    args_kwargs, expected = {}, {}
    for i, (field, flag) in enumerate(sorted(_ARGS_FIELD_MAP.items())):
        if flag in args_kwargs:
            # Two fields legitimately share a flag (e.g. w_grad and
            # energy_w_grad both read energy_w_grad); reuse the same value.
            expected[field] = args_kwargs[flag]
            continue
        value = _sentinel(getattr(defaults, field), i)
        args_kwargs[flag] = value
        expected[field] = value
    cfg = energy_config_from_args(SimpleNamespace(**args_kwargs), alpha_dead=0.5)
    for field, value in expected.items():
        assert getattr(cfg, field) == value, f"{field} did not receive its flag value"


def test_missing_attribute_falls_back_to_dataclass_default():
    cfg = energy_config_from_args(SimpleNamespace(), alpha_dead=0.5)
    defaults = GaussianEnergyConfig()
    for field in _ARGS_FIELD_MAP:
        assert getattr(cfg, field) == getattr(defaults, field)
    assert cfg.alpha_dead == 0.5


def test_alpha_dead_is_required_keyword_only():
    with pytest.raises(TypeError):
        energy_config_from_args(SimpleNamespace())  # type: ignore[call-arg]


def test_overrides_win_over_args():
    args = SimpleNamespace(energy_mode="legacy_utility")
    cfg = energy_config_from_args(args, alpha_dead=0.5, energy_mode="population_energy")
    assert cfg.energy_mode == "population_energy"


# --------------------------------------------------------------------------
# 4. alpha_dead source per call site (the regression that would actually hurt)
# --------------------------------------------------------------------------

_SCHEDULE_DRIVEN_SITES = [
    ("utils/strategies/mcmc/energy.py", 3),
    ("utils/strategies/mcmc/legacy.py", 2),
]
_ARGS_DRIVEN_SITES = [
    ("utils/energy_mcmc.py", 1),
    ("train.py", 1),
    ("train_streaming.py", 1),
    # The two trust-region sites: alpha_dead is unread by
    # sample_trust_region_noise, but stated explicitly by contract.
    ("utils/strategies/mcmc/energy.py", 1),
    ("utils/strategies/mcmc/legacy.py", 1),
]

_CALL_RE = re.compile(
    r"energy_config_from_args\(\s*args\s*,\s*alpha_dead\s*=\s*([^,\n]+)", re.S
)


def _alpha_dead_sources(rel_path: str) -> list[str]:
    text = (_REPO_ROOT / rel_path).read_text()
    return [m.group(1).strip() for m in _CALL_RE.finditer(text)]


def test_no_construction_site_bypasses_the_factory():
    """Production code must not build GaussianEnergyConfig(...) by hand again."""
    offenders = []
    for path in _REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel.startswith("tests/") or "/tests/" in rel or rel.startswith(".claude"):
            continue
        if rel == "utils/strategies/mcmc/energy_policy.py":
            continue  # the definition site + the factory's own construction
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if "GaussianEnergyConfig(" in line:
                offenders.append(f"{rel}:{lineno}")
    assert not offenders, (
        "Direct GaussianEnergyConfig(...) construction outside the factory "
        f"re-introduces the issue #385 pinning defect at: {offenders}. Use "
        "energy_config_from_args(args, alpha_dead=...) instead."
    )


@pytest.mark.parametrize("rel_path,expected_count", _SCHEDULE_DRIVEN_SITES)
def test_schedule_driven_sites_keep_schedule_alpha_dead(rel_path, expected_count):
    """MCMC strategy sites must keep the iteration-varying schedule threshold.

    Switching these to ``args.energy_alpha_dead`` would replace a
    cap_max/schedule-derived, iteration-varying kill threshold with a static one
    10x larger -- a real, silent topology change on the default strategy.
    """
    sources = _alpha_dead_sources(rel_path)
    schedule = [s for s in sources if 'sched["dead_opacity_threshold"]' in s]
    assert len(schedule) == expected_count, (
        f"{rel_path}: expected {expected_count} schedule-driven alpha_dead "
        f"call site(s), found {len(schedule)} in {sources}"
    )


def test_args_driven_sites_use_energy_alpha_dead():
    counts: dict[str, int] = {}
    for rel_path, n in _ARGS_DRIVEN_SITES:
        counts[rel_path] = counts.get(rel_path, 0) + n
    for rel_path, expected in counts.items():
        sources = _alpha_dead_sources(rel_path)
        from_args = [s for s in sources if "args.energy_alpha_dead" in s]
        assert len(from_args) == expected, (
            f"{rel_path}: expected {expected} args-driven alpha_dead call "
            f"site(s), found {len(from_args)} in {sources}"
        )


def test_no_args_driven_site_reintroduces_a_getattr_fallback_for_alpha_dead():
    """Issue #368: energy_alpha_dead must be read bare, never via getattr."""
    pattern = re.compile(r"getattr\(\s*args\s*,\s*[\"']energy_alpha_dead[\"']\s*,")
    for rel_path in {p for p, _ in _ARGS_DRIVEN_SITES}:
        text = (_REPO_ROOT / rel_path).read_text()
        assert not pattern.search(text), (
            f"{rel_path} re-introduced a getattr fallback for energy_alpha_dead"
        )
