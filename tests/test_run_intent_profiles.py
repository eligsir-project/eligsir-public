"""The three canonical run intents -- issue #524.

SMOKE_DEV / CANONICAL_ONLINE / PAPER_REGISTERED exist as resolvable profiles
under `configs/profiles/run_intent/`, with their non-argparse metadata in
`docs/inventory/run_intents.yml`.

WHAT THESE TESTS PROTECT

  * an intent RESOLVES -- through the same layering a real run gets, including
    the unconditional `default.yml` base and `paper_registered`'s `extends`
    chain. Asserting on an intent file read in isolation would check something
    no run ever sees;
  * every key an intent sets is a real argparse dest AND is declared
    semantics-neutral in `docs/inventory/promotion_policy.yml`. An intent that
    could change admission, replay, losses, MCMC or insertion would make "which
    intent did this run use" a confound in every comparison this project makes;
  * resolution is DETERMINISTIC (#524's "effective config snapshot is
    deterministic");
  * no intent trips the execution-semantics contract (#168/#530);
  * SMOKE_DEV is machine-readably non-paper-grade, and PAPER_REGISTERED really
    does score all eligible test cameras;
  * a pending declaration names its blocking issue and is DECLARED, NOT ENABLED
    -- i.e. the intent does not set the option it says is still gated.

WHAT THEY DELIBERATELY DO NOT CHECK: that any of this is wired to a
`--run_intent` flag or persisted in `report.json`. That is #524's deferred half
(acceptance criteria 3/6/7) and needs edits to `arguments/__init__.py` and the
report writer -- outside this PR's behaviour-neutral scope. Note also that no
assertion here touches a `report.json` key, so the host-vs-CI key-divergence
class from #544/PR #547 does not apply to this file.

Torch-free by construction: `eligsir.cli.config` was measured to pull in no
torch module (`[m for m in sys.modules if m.startswith('torch')] == []` inside
the real `Dockerfile.ci-cpu` image), and `arguments/` imports none either. So
this executes in the lean CPU CI image rather than being silently excluded by
`tests/conftest.py` -- see issue #536.
"""
from __future__ import annotations

import yaml

from scripts.build_option_inventory import RUN_INTENT_DIR
from scripts.dump_default_config import _REPO_ROOT, collect_defaults

INTENT_METADATA_PATH = _REPO_ROOT / "docs" / "inventory" / "run_intents.yml"
PROMOTION_POLICY_PATH = _REPO_ROOT / "docs" / "inventory" / "promotion_policy.yml"

CANONICAL_INTENTS = ("smoke_dev", "canonical_online", "paper_registered")


def _metadata() -> dict:
    return yaml.safe_load(INTENT_METADATA_PATH.read_text())


def _semantics_neutral_options() -> set[str]:
    policy = yaml.safe_load(PROMOTION_POLICY_PATH.read_text())
    return {
        e["feature_or_option"]
        for e in policy["entries"]
        if e["kind"] == "option" and e["training_semantics_changed"] is False
    }


def _resolve(intent: str) -> dict:
    """Resolve an intent exactly as a run would: `--run_profile run_intent/<name>`."""
    from eligsir.cli import config as cfg

    return cfg.load_cli_config(run_profile=f"run_intent/{intent}")


# --------------------------------------------------------------------------- #
# Existence and resolution
# --------------------------------------------------------------------------- #

def test_the_three_canonical_intents_exist_as_profiles():
    """#524 acceptance criterion 1: three canonical run intents exist."""
    on_disk = sorted(p.stem for p in RUN_INTENT_DIR.glob("*.yml"))
    assert on_disk == sorted(CANONICAL_INTENTS), (
        f"configs/profiles/run_intent/ holds {on_disk}; #524 requires exactly "
        f"{sorted(CANONICAL_INTENTS)}"
    )
    assert sorted(_metadata()["intents"]) == sorted(CANONICAL_INTENTS)


def test_each_intent_is_selectable_by_run_profile_alias():
    """No code change was needed for this: `resolve_config_path()` falls back to
    `configs/profiles/<name>.yml` for an alias it does not know. Verified rather
    than assumed, because the whole usability claim of the intent system rests
    on it.
    """
    from eligsir.cli import config as cfg

    for intent in CANONICAL_INTENTS:
        alias = _metadata()["intents"][intent]["run_profile_alias"]
        assert alias == f"run_intent/{intent}"
        path = cfg.resolve_config_path(run_profile=alias)
        assert path.exists(), f"{alias} resolves to a nonexistent file {path}"
        assert path == RUN_INTENT_DIR / f"{intent}.yml"


def test_each_intent_resolves_and_validates_as_a_run_config():
    """`run_config.validate_config` uses `extra="forbid"`, so a single non-dest
    key turns a profile into a hard ConfigError at load time.
    """
    from eligsir.cli.run_config import validate_config

    for intent in CANONICAL_INTENTS:
        merged = _resolve(intent)
        assert merged, f"{intent} resolved to an empty config"
        validate_config(merged)


def test_resolution_is_deterministic():
    """#524's Tests: "Effective config snapshot is deterministic"."""
    for intent in CANONICAL_INTENTS:
        first = _resolve(intent)
        second = _resolve(intent)
        assert first == second, f"{intent} resolved differently on two reads"


def test_paper_registered_extends_canonical_online():
    """The stack is default.yml -> canonical_online -> paper_registered."""
    text = (RUN_INTENT_DIR / "paper_registered.yml").read_text()
    data = yaml.safe_load(text)
    assert data["extends"] == ["canonical_online.yml"]
    paper = _resolve("paper_registered")
    online = _resolve("canonical_online")
    # Everything canonical_online sets is present in paper, modulo the deltas
    # paper deliberately overrides.
    for key in online:
        assert key in paper, f"paper_registered lost canonical_online's {key!r}"


def test_every_intent_inherits_the_default_profile_base():
    """`load_cli_config()` prepends `configs/profiles/default.yml` unconditionally.

    Stated as a test rather than a comment because it is the reason an intent
    file may stay small: the base policy profile is inherited, while retired
    legacy execution flags are not silently re-declared per intent.
    """
    for intent in CANONICAL_INTENTS:
        merged = _resolve(intent)
        assert "streaming_replay" not in merged
        assert merged.get("model_layout") == "gsplat"


def test_an_intent_composes_with_a_scene_profile():
    """`--run_profile <intent> --config <scene>` resolves BOTH.

    Worth pinning because it is genuinely non-obvious: `resolve_config_path()`
    raises "Use either --config or --run_profile, not both", but
    `load_cli_config()` calls it once per argument, so that guard never fires
    from the path train.py uses. Reading the guard alone would suggest an intent
    cannot be combined with a scene profile at all, which would make the intent
    system nearly useless.

    ALSO pins the sharp edge: `--config` layers AFTER the run profile, so the
    scene profile WINS on any key both set. That is arguably the wrong
    precedence for a policy layer, and is harmless today only because no scene
    profile sets a reporting key. #524's composable-layer architecture (with a
    defined intent precedence) is part of its deferred half.
    """
    from eligsir.cli import config as cfg

    merged = cfg.load_cli_config(
        config_files=["configs/profiles/causal_replay/tum_rgbd.yml"],
        run_profile="run_intent/smoke_dev",
    )
    # the intent's contribution survived
    assert merged["streaming_report_test_metrics_max"] == 8
    # and so did the scene profile's
    assert "streaming_camera_profile" in merged

    scene_only = cfg.load_cli_config(
        config_files=["configs/profiles/causal_replay/tum_rgbd.yml"]
    )
    for key, value in scene_only.items():
        assert merged[key] == value, (
            f"composing an intent under a scene profile changed the scene "
            f"profile's own {key!r} -- the layering order documented in "
            "docs/inventory/run_intents.yml no longer holds"
        )


# --------------------------------------------------------------------------- #
# What an intent is allowed to contain
# --------------------------------------------------------------------------- #

def test_intent_keys_are_real_argparse_dests():
    """#524's Tests: "CI verifies profile keys exist and are wired"."""
    dests = set(collect_defaults())
    for intent in CANONICAL_INTENTS:
        data = yaml.safe_load((RUN_INTENT_DIR / f"{intent}.yml").read_text())
        keys = {k for k in data if k != "extends"}
        bogus = sorted(keys - dests)
        assert not bogus, (
            f"run_intent/{intent}.yml sets {bogus}, which are not argparse dests. "
            "A typo'd key is a silent no-op, and run_config's extra=\"forbid\" "
            "would reject it at load time anyway."
        )


def test_no_intent_sets_a_training_semantics_changing_key():
    """The load-bearing safety property of the whole intent system.

    Every key an intent sets must be declared in promotion_policy.yml with
    `training_semantics_changed: false`. Absence from the registry fails too:
    an undeclared key is an unreviewed key, and defaulting the permissive answer
    for unreviewed keys is precisely how a method change would slip in wearing a
    reporting profile's clothes.
    """
    allowed = _semantics_neutral_options()
    for intent in CANONICAL_INTENTS:
        data = yaml.safe_load((RUN_INTENT_DIR / f"{intent}.yml").read_text())
        keys = {k for k in data if k != "extends"}
        undeclared = sorted(keys - allowed)
        assert not undeclared, (
            f"run_intent/{intent}.yml sets {undeclared}, which promotion_policy.yml "
            "does not declare as `training_semantics_changed: false`. Either add a "
            "cited registry entry proving the key is semantics-neutral, or do not "
            "set it from a run intent."
        )


# --------------------------------------------------------------------------- #
# Execution semantics (#168 / #530)
# --------------------------------------------------------------------------- #

def test_no_intent_trips_a_hard_execution_semantics_error():
    """#524's Tests: "Every canonical run intent resolves without contradictory
    execution semantics."

    Mirrors `tests/test_execution_semantics.py::test_no_profile_leaf_trips_a_hard_error`
    but keyed on the intents specifically, so a future intent edit that
    introduces a contradiction is attributed to the intent rather than to a
    generic sweep.
    """
    from argparse import Namespace

    from eligsir.utils import execution_semantics as es

    offenders = []
    for intent in CANONICAL_INTENTS:
        merged = _resolve(intent)
        args = Namespace(**collect_defaults())
        for key, value in merged.items():
            setattr(args, key, value)
        args.source_path = str(merged.get("source_path", "/data/tum/fr3"))
        sem = es.resolve_execution_semantics(args)
        errors, _ = es.validate_execution_semantics(sem, args)
        if errors:
            offenders.append((intent, errors))
    assert offenders == [], offenders


# --------------------------------------------------------------------------- #
# Intent-specific guarantees
# --------------------------------------------------------------------------- #

def test_smoke_dev_is_machine_readably_non_paper_grade():
    """#524's Tests: "SMOKE_DEV is visibly non-paper-grade in machine-readable
    metadata" -- so a smoke run can never be mistaken for evidence merely
    because it emitted a report.json.
    """
    meta = _metadata()["intents"]["smoke_dev"]
    assert meta["evidence_grade"] == "non_paper"
    assert meta["paper_registered"] is False
    assert _metadata()["intents"]["canonical_online"]["paper_registered"] is False
    assert _metadata()["intents"]["paper_registered"]["paper_registered"] is True


def test_smoke_dev_really_is_cheaper_than_canonical_online():
    """A declaration nobody enforces is a comment. These are the actual caps."""
    smoke = _resolve("smoke_dev")
    online = _resolve("canonical_online")
    assert smoke["streaming_report_test_metrics_max"] < online["streaming_report_test_metrics_max"]
    assert smoke["streaming_report_train_metrics_max"] < online["streaming_report_train_metrics_max"]
    assert smoke["streaming_report_video_max_width"] < online["streaming_report_video_max_width"]
    assert smoke["streaming_report_trajectory_max_frames"] < online["streaming_report_trajectory_max_frames"]


def test_smoke_dev_still_keeps_validity_and_coverage_on():
    """#524 keeps trajectory/source coverage and the holdout identity default-on
    even for SMOKE_DEV: a run whose coverage is unknown cannot be interpreted at
    all, however cheap it was.
    """
    smoke = _resolve("smoke_dev")
    assert smoke["streaming_trajectory_eval"] is True
    assert smoke["streaming_report_final"] is True
    assert smoke["streaming_report_test_metrics_max"] > 0, (
        "a smoke run still needs a real (if small) holdout score; zero would make "
        "it indistinguishable from an invalid eval"
    )


def test_paper_registered_scores_all_eligible_test_cameras():
    """#524: PAPER_REGISTERED must score ALL eligible test cameras.

    Issue #420 established that `streaming_report_test_metrics_max` subsamples
    the realized holdout BEFORE averaging, so a capped paper run could score a
    different camera set than a run it is compared against while still matching
    on `realized_holdout_hash`. The cap must therefore be effectively unbounded.
    """
    paper = _resolve("paper_registered")
    assert paper["streaming_report_test_metrics_max"] >= 1_000_000
    assert paper["streaming_report_train_metrics_max"] >= 1_000_000
    online = _resolve("canonical_online")
    assert paper["streaming_report_test_metrics_max"] > online["streaming_report_test_metrics_max"]


def test_paper_registered_keeps_full_resolution_evidence_media():
    paper = _resolve("paper_registered")
    assert paper["streaming_report_video_min_width"] >= 1920
    assert paper["streaming_report_trajectory_max_frames"] >= 600


# --------------------------------------------------------------------------- #
# Pending declarations: declared, NOT enabled
# --------------------------------------------------------------------------- #

def test_every_pending_declaration_names_a_blocking_issue():
    for intent, meta in _metadata()["intents"].items():
        for pending in meta.get("pending", []):
            assert isinstance(pending.get("blocking_issue"), int), (
                f"{intent}'s pending entry {pending.get('feature')!r} does not name "
                "an integer blocking issue. 'Pending' without a named gate is "
                "indistinguishable from 'forgotten'."
            )
            assert str(pending.get("why", "")).strip(), (
                f"{intent}'s pending entry {pending.get('feature')!r} has no `why`"
            )


def test_pending_features_are_declared_not_enabled():
    """The hard constraint on this whole PR, expressed as a test.

    Where an intent WOULD want something default-on but its gate has not passed,
    the intent must DECLARE it -- and must not set the option that would enable
    it. A pending entry naming a concrete `option` is the checkable case.
    """
    for intent, meta in _metadata()["intents"].items():
        data = yaml.safe_load((RUN_INTENT_DIR / f"{intent}.yml").read_text())
        for pending in meta.get("pending", []):
            option = pending.get("option")
            if not option:
                continue
            assert option not in data, (
                f"run_intent/{intent}.yml sets {option!r} while declaring it PENDING "
                f"behind #{pending['blocking_issue']}. Declared means not enabled."
            )


def test_pending_features_are_registered_in_the_promotion_policy():
    """Cross-file consistency: an intent cannot invent a pending feature that the
    promotion registry has never heard of.
    """
    policy = yaml.safe_load(PROMOTION_POLICY_PATH.read_text())
    known = {e["feature_or_option"] for e in policy["entries"]}
    for intent, meta in _metadata()["intents"].items():
        for pending in meta.get("pending", []):
            assert pending["feature"] in known, (
                f"{intent} declares pending feature {pending['feature']!r}, which "
                "has no entry in promotion_policy.yml"
            )


def test_the_gates_issue_524_names_are_all_declared_pending_somewhere():
    """#358 (local quality), #480 (dashboard overhead) and #399 (GPU-memory
    telemetry) are the three gates the brief names explicitly. Each must be
    visible as a pending declaration on at least one intent, so that "we chose
    not to enable this" is recorded rather than inferred from silence.
    """
    declared = {
        p["blocking_issue"]
        for meta in _metadata()["intents"].values()
        for p in meta.get("pending", [])
    }
    for issue in (358, 480, 399):
        assert issue in declared, (
            f"issue #{issue} gates something #524 wants default-on, but no intent "
            "declares it pending"
        )
