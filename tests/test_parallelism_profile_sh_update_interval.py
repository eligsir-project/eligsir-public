"""Issue #239: an explicit --sh_update_interval must win over the 'safe'
parallelism profile's unconditional default of 16, mirroring how
--parallelism_profile off already leaves an explicit --sh_update_interval
untouched.

Scope note: this test file (and the fix it verifies) touches ONLY
sh_update_interval. optimizer_type / gsplat_sparse_grad semantics under the
'safe' / 'off' profiles are a separate, still-undecided question tracked by
issue #228 and draft PR #235 (explicitly DO-NOT-MERGE) -- not exercised here.
"""

from eligsir.cli.parser import parse_training_args
from eligsir.cli.run_config import cli_explicit_dests
from train import apply_parallelism_profile


def _resolve(argv):
    args, bundle = parse_training_args(argv, description="test")
    explicit = cli_explicit_dests(bundle.parser, argv)
    apply_parallelism_profile(args, explicit_dests=explicit)
    return args


def test_explicit_sh_update_interval_wins_under_default_safe_profile():
    args = _resolve(["-s", "/tmp/x", "--sh_update_interval", "4"])
    assert args.sh_update_interval == 4
    assert args.parallelism_profile == "safe"


def test_explicit_sh_update_interval_8_wins_under_default_safe_profile():
    args = _resolve(["-s", "/tmp/x", "--sh_update_interval", "8"])
    assert args.sh_update_interval == 8


def test_explicit_sh_update_interval_wins_under_explicit_safe_profile():
    args = _resolve(
        ["-s", "/tmp/x", "--parallelism_profile", "safe", "--sh_update_interval", "4"]
    )
    assert args.sh_update_interval == 4


def test_no_flag_still_gets_safe_profile_default_of_16():
    args = _resolve(["-s", "/tmp/x"])
    assert args.sh_update_interval == 16
    assert args.parallelism_profile == "safe"


def test_off_profile_with_explicit_sh_update_interval_unchanged():
    """Pre-existing, already-correct behavior (must not regress)."""
    args = _resolve(
        ["-s", "/tmp/x", "--parallelism_profile", "off", "--sh_update_interval", "4"]
    )
    assert args.sh_update_interval == 4


def test_safe_profile_no_longer_forces_optimizer_fields_over_explicit_flags():
    """SUPERSEDED ASSERTION (issue #228, 2026-08-13).

    This test previously pinned the *pre-#228* behavior -- under 'safe',
    optimizer_type/gsplat_sparse_grad were force-set regardless of what the
    user explicitly passed -- as a deliberate guardrail so that resolving
    #228 would fail loudly here rather than drift silently. #228 is now
    resolved: the explicit-wins rule that #239/#246 gave sh_update_interval
    applies to the optimizer fields too, so the expected values flip.

    The full matrix lives in tests/test_parallelism_profile.py.
    """
    args = _resolve(
        [
            "-s",
            "/tmp/x",
            "--optimizer_type",
            "adam",
            "--no-gsplat_sparse_grad",
        ]
    )
    assert args.optimizer_type == "adam"
    assert args.gsplat_sparse_grad is False


def test_explicit_dests_none_preserves_old_clobbering_behavior():
    """A caller that hasn't been updated to pass explicit_dests (external
    or future) keeps the pre-fix behavior rather than silently changing
    meaning -- explicit_dests=None must not be misread as "nothing is
    explicit, but also don't override".
    """
    args, bundle = parse_training_args(
        ["-s", "/tmp/x", "--sh_update_interval", "4"], description="test"
    )
    apply_parallelism_profile(args)  # no explicit_dests passed
    assert args.sh_update_interval == 16
