"""Tests for issue #524's deferred half: `--run_intent` + `report.json["run_intent"]`.

PR #550 shipped the three canonical run intents as resolvable, behaviour-
neutral profiles (see `tests/test_run_intent_profiles.py`). This file covers
what that PR explicitly left open (its own "WHAT THEY DELIBERATELY DO NOT
CHECK" section): a first-class `--run_intent` flag
(`eligsir/cli/parser.py`), the `--run_intent`/`--run_profile` conflict
guard and its profile-stack resolution (`utils/run_intent.py`,
`eligsir.cli.config.resolve_profile_stack`), and persistence into
`report.json["run_intent"]` (`utils/comparison_report.py`'s `run_intent`
kwarg, wired from both trainer paths).

Coverage mirrors `tests/test_experiment_origin.py`'s existing structure for
the identical class of change (a report.json-persisted, args-resolved-once
provenance/policy block):
  * torch-free CLI/config-layer tests -- run in every surface, including CPU CI;
  * torch-gated `report.json` integration tests (record/omit, strict
    additivity, call-site wiring) -- collected everywhere, executed wherever
    torch is present (CPU CI visibly skips with a reason, the training image
    runs them for real; see Dockerfile.ci-cpu / #536 / #544).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock

import pytest
import yaml

from eligsir.cli import config as cli_config
from eligsir.cli.run_config import validate_config
from eligsir.utils.run_intent import (
    KNOWN_RUN_INTENTS,
    RunIntentConflictError,
    resolve_run_intent_and_profile,
    run_intent_report_from_args,
)

_REPO_ROOT = cli_config._REPO_ROOT


# --- CLI flag -----------------------------------------------------------


def test_cli_flag_registered_with_the_three_canonical_choices():
    from eligsir.cli.parser import build_training_parser

    parser = build_training_parser().parser
    args = parser.parse_args(["-s", "/tmp/nonexistent"])
    assert args.run_intent is None  # unset by default -- no behavior change

    for intent in KNOWN_RUN_INTENTS:
        args = parser.parse_args(["-s", "/tmp/nonexistent", "--run_intent", intent])
        assert args.run_intent == intent


def test_cli_flag_rejects_an_unknown_intent():
    from eligsir.cli.parser import build_training_parser

    parser = build_training_parser().parser
    with pytest.raises(SystemExit):
        parser.parse_args(["-s", "/tmp/nonexistent", "--run_intent", "bogus"])


def test_known_run_intents_matches_the_resolvable_profiles_on_disk():
    """Same invariant `tests/test_run_intent_profiles.py` pins for the
    profile files themselves -- kept in sync here since this module's
    constant is what the CLI's `choices=` is built from."""
    on_disk = {p.stem for p in (_REPO_ROOT / "configs" / "profiles" / "run_intent").glob("*.yml")}
    assert on_disk == set(KNOWN_RUN_INTENTS)


# --- resolution: --run_intent -> --run_profile ---------------------------


def test_run_intent_resolves_to_the_run_profile_alias():
    for intent in KNOWN_RUN_INTENTS:
        assert resolve_run_intent_and_profile(intent, None) == f"run_intent/{intent}"


def test_no_intent_leaves_an_explicit_run_profile_untouched():
    assert (
        resolve_run_intent_and_profile(None, "causal_replay/tum_rgbd")
        == "causal_replay/tum_rgbd"
    )
    assert resolve_run_intent_and_profile(None, None) is None


def test_agreeing_run_intent_and_run_profile_is_harmless():
    assert (
        resolve_run_intent_and_profile("smoke_dev", "run_intent/smoke_dev")
        == "run_intent/smoke_dev"
    )


def test_conflicting_run_intent_and_run_profile_raises_loudly():
    with pytest.raises(RunIntentConflictError):
        resolve_run_intent_and_profile("smoke_dev", "run_intent/paper_registered")
    with pytest.raises(RunIntentConflictError):
        resolve_run_intent_and_profile("smoke_dev", "causal_replay/tum_rgbd")


# --- profile stack ---------------------------------------------------------


def test_profile_stack_matches_load_cli_config_for_every_intent():
    """The stack is not an independent claim -- every layer it names is a
    layer `load_cli_config()` actually merged, and nothing else."""
    for intent in KNOWN_RUN_INTENTS:
        rp = f"run_intent/{intent}"
        stack = cli_config.resolve_profile_stack(run_profile=rp)
        assert stack[0] == "configs/profiles/default.yml"
        assert f"configs/profiles/run_intent/{intent}.yml" in stack
        # Every stack entry must be a real, loadable file.
        for rel in stack:
            assert (_REPO_ROOT / rel).exists(), rel


def test_paper_registered_profile_stack_includes_its_extends_parent():
    stack = cli_config.resolve_profile_stack(run_profile="run_intent/paper_registered")
    assert stack == [
        "configs/profiles/default.yml",
        "configs/profiles/run_intent/canonical_online.yml",
        "configs/profiles/run_intent/paper_registered.yml",
    ]


def test_profile_stack_is_deterministic():
    a = cli_config.resolve_profile_stack(run_profile="run_intent/canonical_online")
    b = cli_config.resolve_profile_stack(run_profile="run_intent/canonical_online")
    assert a == b


def test_profile_stack_with_no_profile_is_empty():
    """Issue #591: a plain no-flag run does NOT merge default.yml --
    load_cli_config() early-outs to {} in exactly this case (no
    --run_profile, no --config, only the implicit default.yml resolvable).
    resolve_profile_stack() must report the same "nothing was merged"
    verdict, not claim default.yml was applied. This assertion used to be
    the reverse (`== ["configs/profiles/default.yml"]`) and encoded the bug
    this issue reports -- see test_profile_stack_matches_load_cli_config
    below for the general, both-functions-agree version of this check."""
    assert cli_config.resolve_profile_stack() == []
    assert cli_config.load_cli_config() == {}


# --- issue #591: resolve_profile_stack() must never claim a layer was
# merged that load_cli_config() did not actually merge -----------------


def _matrix_cases():
    """(config_files, run_profile) argument pairs spanning: no args,
    --config only, --run_profile only, both, and each known run_intent/*
    profile -- the exact matrix issue #591 asks for."""
    cases = [
        (None, None),
        (["configs/profiles/defaults.reference.yml"], None),
        (None, "run_intent/smoke_dev"),
        (["configs/profiles/defaults.reference.yml"], "run_intent/smoke_dev"),
    ]
    for intent in KNOWN_RUN_INTENTS:
        cases.append((None, f"run_intent/{intent}"))
    return cases


@pytest.mark.parametrize("config_files,run_profile", _matrix_cases())
def test_profile_stack_matches_load_cli_config(config_files, run_profile):
    """The reported stack must be consistent with what was actually merged:
    non-empty stack <=> non-empty effective config, and every layer named
    in the stack is a file that genuinely contributed to the merge (its own
    keys are a subset of the merged result, modulo later layers/CLI
    overrides -- checked here via "the merge is non-trivial and non-empty
    iff the stack is non-empty", the same invariant the divergence in #591
    violated). An independent-literal-expectation test cannot catch drift
    between the two copies by construction -- this drives both functions
    over the SAME inputs instead."""
    merged = cli_config.load_cli_config(config_files=config_files, run_profile=run_profile)
    stack = cli_config.resolve_profile_stack(config_files=config_files, run_profile=run_profile)

    assert (len(stack) > 0) == (len(merged) > 0), (
        f"profile_stack={stack!r} but effective_config={merged!r} for "
        f"config_files={config_files!r} run_profile={run_profile!r} -- a "
        "non-empty stack must correspond to an actually-non-empty merge, "
        "and vice versa (issue #591)."
    )
    for rel in stack:
        assert (_REPO_ROOT / rel).exists(), rel


def test_resolve_config_path_none_is_guarded_identically_in_both_walks(monkeypatch):
    """Issue #591's second, latent divergence: resolve_profile_stack() used
    to guard each resolve_config_path() result with `if rp is not None` /
    `if cp is not None` while load_cli_config() appended unguarded --
    harmless only because resolve_config_path() never actually returns None
    for a truthy run_profile/config_file input. Now both walks share
    _resolve_config_paths(), so both are guarded identically by
    construction; pin that a None result is dropped rather than propagated
    into either function's output (would otherwise raise deep inside
    load_yaml/Path handling instead of being silently skipped)."""
    monkeypatch.setattr(cli_config, "resolve_config_path", lambda **kw: None)

    paths, _ = cli_config._resolve_config_paths(run_profile="run_intent/smoke_dev")
    # Only the implicit default.yml base (if present) survives; the None
    # from the mocked resolve_config_path() must not be appended.
    assert all(p is not None for p in paths)
    assert paths == ([cli_config._DEFAULT_BASE] if cli_config._DEFAULT_BASE.exists() else [])

    merged = cli_config.load_cli_config(run_profile="run_intent/smoke_dev")
    stack = cli_config.resolve_profile_stack(run_profile="run_intent/smoke_dev")
    # Both now see the identical (guarded) path list, so both report the
    # profile as unresolved -- neither silently crashes nor disagrees.
    assert (len(stack) > 0) == (len(merged) > 0)


# --- report.json["run_intent"] block builder ------------------------------


def _metadata(intent):
    data = yaml.safe_load((_REPO_ROOT / "docs" / "inventory" / "run_intents.yml").read_text())
    return data["intents"][intent]


class _Args:
    def __init__(self, run_intent=None):
        self.run_intent = run_intent


def test_undeclared_run_gets_the_least_permissive_block():
    block = run_intent_report_from_args(
        _Args(), effective_config={}, profile_stack=[], cli_explicit=None
    )
    assert block == {
        "declared": "UNDECLARED",
        "resolved_run_profile": None,
        "evidence_grade": "undeclared",
        "paper_registered": False,
        "profile_stack": [],
        "effective_config": {},
        "cli_overridden_keys": [],
    }


@pytest.mark.parametrize("intent", KNOWN_RUN_INTENTS)
def test_declared_run_matches_run_intents_yml_metadata(intent):
    meta = _metadata(intent)
    merged = validate_config(cli_config.load_cli_config(run_profile=f"run_intent/{intent}"))
    stack = cli_config.resolve_profile_stack(run_profile=f"run_intent/{intent}")
    block = run_intent_report_from_args(
        _Args(run_intent=intent),
        effective_config=merged,
        profile_stack=stack,
        cli_explicit=None,
    )
    assert block["declared"] == intent
    assert block["resolved_run_profile"] == meta["run_profile_alias"]
    assert block["evidence_grade"] == meta["evidence_grade"]
    assert block["paper_registered"] == meta["paper_registered"]
    assert block["profile_stack"] == stack
    assert block["effective_config"] == merged
    assert block["cli_overridden_keys"] == []


def test_cli_overridden_keys_records_explicit_flags_that_beat_the_profile():
    merged = {"streaming_report_test_metrics_max": 8, "record_video": False}
    block = run_intent_report_from_args(
        _Args(run_intent="smoke_dev"),
        effective_config=merged,
        profile_stack=["x.yml"],
        cli_explicit={"streaming_report_test_metrics_max", "unrelated_flag"},
    )
    assert block["cli_overridden_keys"] == ["streaming_report_test_metrics_max"]


# --- report.json integration ------------------------------------------------
# CPU-only, fixture-level: write_post_training_report is driven with fake
# cameras/gaussians and a stub render fn -- the convention established by
# tests/test_git_provenance.py and reused verbatim by
# tests/test_experiment_origin.py for the identical class of change.


@pytest.fixture(autouse=True, scope="module")
def _mock_lpips_import():
    had_original = "lpipsPyTorch" in sys.modules
    original = sys.modules.get("lpipsPyTorch")
    sys.modules["lpipsPyTorch"] = MagicMock()
    try:
        yield
    finally:
        if had_original:
            sys.modules["lpipsPyTorch"] = original
        else:
            sys.modules.pop("lpipsPyTorch", None)


class _FakeGaussian:
    def get_xyz(self):
        import torch
        return torch.zeros(0, 3)


class _FakePipe:
    pass


class _FakeCam:
    def __init__(self, name):
        import torch
        self.image_name = name
        self.original_image = torch.zeros(3, 64, 64)


def _minimal_render_fn(cam, gaussians, pipe, background):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"render": torch.zeros(3, 64, 64, device=device)}


def _run_report(tmpdir, **kw):
    # utils/comparison_report.py does a module-scope `import torch`; every
    # torch dependency in this helper is function-local, so this file
    # collects in the torch-free CPU image and this fixture visibly skips
    # there with a reason, and runs for real in the training image -- same
    # reasoning as tests/test_experiment_origin.py's identical helper.
    pytest.importorskip("torch")

    from eligsir.utils.comparison_report import write_post_training_report

    return write_post_training_report(
        model_path=tmpdir,
        iteration=100,
        gaussians=_FakeGaussian(),
        train_cams=[_FakeCam("a")],
        test_cams=[_FakeCam("t")],
        render_fn=_minimal_render_fn,
        pipe=_FakePipe(),
        background=None,
        **kw,
    )


def _report_json(tmpdir):
    with open(os.path.join(tmpdir, "comparison", "iter_100", "report.json")) as f:
        return json.load(f)


def test_report_records_run_intent():
    block = run_intent_report_from_args(
        _Args(run_intent="canonical_online"),
        effective_config={"record_video": False},
        profile_stack=["configs/profiles/default.yml"],
        cli_explicit=None,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = _run_report(tmpdir, run_intent=block)
        assert summary["run_intent"] == block
        assert _report_json(tmpdir)["run_intent"] == block


def test_report_omits_run_intent_when_not_provided():
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = _run_report(tmpdir)
        assert "run_intent" not in summary
        assert "run_intent" not in _report_json(tmpdir)


def test_report_change_is_strictly_additive(report_key_diff_fn):
    """Behavior-neutral by default: a run that declares no --run_intent
    produces a report identical to the pre-#524-deferred-half one apart from
    the single new key.

    Uses tests/conftest.py's shared `report_key_diff` (#544): each report is
    normalized against its OWN `output_dir` before diffing, so path-valued
    keys (`output_dir`, `artifacts`, `train_contact_sheet`, ...) compare
    equal without a hand-maintained allowlist.
    """
    block = run_intent_report_from_args(
        _Args(), effective_config={}, profile_stack=[], cli_explicit=None
    )
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        without = _run_report(a)
        with_intent = _run_report(b, run_intent=block)

        diff = report_key_diff_fn(
            without, a, with_intent, b,
            ignore=("timestamp", "elapsed_s", "gpu_coresidency"),
        )
    assert diff == ["run_intent"], diff
    assert with_intent["run_intent"]["declared"] == "UNDECLARED"


# --- wiring -----------------------------------------------------------------


def test_every_write_post_training_report_call_site_is_wired():
    """Mirrors tests/test_experiment_origin.py's identical guard for #511: a
    report path that writes report.json without run_intent recreates exactly
    the hole this change closes. render_report.py is exempt for the same
    reason it is exempt there -- it re-renders an already-trained checkpoint
    offline and never went through train.py's startup resolution."""
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    callers = []
    for path in list(repo.glob("*.py")) + list((repo / "eligsir").rglob("*.py")):
        text = path.read_text()
        if "write_post_training_report(" not in text:
            continue
        if "def write_post_training_report(" in text:
            continue
        callers.append((path.relative_to(repo), text))

    assert callers, "no call sites found -- the glob or the API name changed"
    exempt = {"render_report.py"}
    # #1075 task P3: train_streaming.py's report kwargs (run_intent among them)
    # are assembled by eligsir/runtime/report_kwargs.py and splatted in at
    # each call site. A caller that splats that builder IS wired -- the wiring
    # moved one file, it did not disappear. The builder module itself is a
    # caller-side dependency, not a report call site, so it is checked directly.
    builder = (repo / "eligsir" / "runtime" / "report_kwargs.py").read_text()
    assert "run_intent" in builder, (
        "the shared streaming report-kwargs builder no longer emits run_intent "
        "-- every streaming report.json would lose the #524 block."
    )
    unwired = [
        str(p) for p, text in callers
        if "run_intent" not in text
        and "_streaming_report_kwargs(" not in text
        and str(p) not in exempt
    ]
    assert not unwired, f"call sites missing run_intent wiring: {unwired}"


def test_train_py_resolves_run_intent_once_before_the_config_merge():
    """Source-text check (importing train.py needs a full CUDA/gsplat env):
    the resolver must run once, before the config merge it feeds, and every
    report path must use the pure accessor -- not a second resolution."""
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "train.py").read_text()
    assert src.count("args.run_intent_report = run_intent_report_from_args(") == 1
    assert "run_intent=args.run_intent_report" in src
    # The resolver call must precede the config-merge call it feeds.
    assert src.index("resolve_run_intent_and_profile(") < src.index("load_cli_config(")
