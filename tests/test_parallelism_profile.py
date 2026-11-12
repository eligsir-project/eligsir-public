"""Issue #228: ``--parallelism_profile off`` must actually select dense Adam.

Pins the resolved config for the full matrix of profile (safe / off / default)
x explicit flags (present / absent), plus the truthfulness of the
``[parallelism] profile=... optimizer=... sparse_grad=...`` startup line that
``train.py`` / ``train_streaming.py`` print (and that every post-hoc audit of a
past run relies on).

Scope note: ``sh_update_interval`` is owned by the separate #239/#246
mechanism and is NOT part of ``PARALLELISM_PROFILES`` -- ``off`` deliberately
leaves it at its argparse default of 16 rather than selecting 1. See
``docs/architecture/2026-08-13-issue228-fix-impact.md`` for why (the three
issue-#106 dense-Adam specs pass ``off`` with no ``--sh_update_interval`` and
their completed arms all ran at 16).

All pure argument resolution -- no GPU, no training.
"""

import pytest

from eligsir.cli.parser import parse_training_args
from eligsir.cli.run_config import cli_explicit_dests
from train import apply_parallelism_profile, profile_explicit_dests


def _resolve(argv, config=None):
    """Resolve argv (and an optional --config-style dict) exactly the way
    train.py's __main__ prologue does."""
    args, bundle = parse_training_args(argv, description="test")
    explicit = cli_explicit_dests(bundle.parser, argv)
    config = config or {}
    for key, value in config.items():
        if key not in explicit:  # CLI flags win over config
            setattr(args, key, value)
    apply_parallelism_profile(
        args, explicit_dests=profile_explicit_dests(explicit, config)
    )
    return args, bundle


def _resolved_opt(argv, config=None):
    args, bundle = _resolve(argv, config)
    return bundle.optimization_params.extract(args)


# --------------------------------------------------------------------------
# The headline fix
# --------------------------------------------------------------------------


def test_off_alone_now_selects_dense_adam():
    """The bug: `off` alone used to keep selective_adam + sparse grads."""
    args, _ = _resolve(["-s", "/tmp/x", "--parallelism_profile", "off"])
    assert args.optimizer_type == "adam"
    assert args.gsplat_sparse_grad is False


def test_off_alone_equals_the_three_flag_workaround():
    """`off` alone must now resolve identically to the pre-fix workaround."""
    lean, _ = _resolve(["-s", "/tmp/x", "--parallelism_profile", "off"])
    verbose, _ = _resolve(
        [
            "-s",
            "/tmp/x",
            "--parallelism_profile",
            "off",
            "--optimizer_type",
            "adam",
            "--no-gsplat_sparse_grad",
        ]
    )
    assert (lean.optimizer_type, lean.gsplat_sparse_grad) == (
        verbose.optimizer_type,
        verbose.gsplat_sparse_grad,
    ) == ("adam", False)


def test_default_profile_is_safe_and_still_sparse():
    args, _ = _resolve(["-s", "/tmp/x"])
    assert args.parallelism_profile == "safe"
    assert args.optimizer_type == "selective_adam"
    assert args.gsplat_sparse_grad is True
    assert args.sh_update_interval == 16


def test_explicit_safe_profile_is_sparse():
    args, _ = _resolve(["-s", "/tmp/x", "--parallelism_profile", "safe"])
    assert args.optimizer_type == "selective_adam"
    assert args.gsplat_sparse_grad is True


# --------------------------------------------------------------------------
# Explicit flags beat the profile, in BOTH directions
# --------------------------------------------------------------------------


def test_explicit_adam_wins_under_safe_profile():
    """Previously the `safe` profile silently clobbered this to selective_adam."""
    args, _ = _resolve(
        [
            "-s",
            "/tmp/x",
            "--parallelism_profile",
            "safe",
            "--optimizer_type",
            "adam",
            "--no-gsplat_sparse_grad",
        ]
    )
    assert args.optimizer_type == "adam"
    assert args.gsplat_sparse_grad is False


def test_explicit_adam_wins_under_default_safe_profile():
    """Same, with the profile left at its default rather than named."""
    args, _ = _resolve(
        ["-s", "/tmp/x", "--optimizer_type", "adam", "--no-gsplat_sparse_grad"]
    )
    assert args.parallelism_profile == "safe"
    assert args.optimizer_type == "adam"
    assert args.gsplat_sparse_grad is False


def test_explicit_selective_adam_wins_under_off_profile():
    """The other direction: `off` must not clobber a deliberate sparse request."""
    args, _ = _resolve(
        [
            "-s",
            "/tmp/x",
            "--parallelism_profile",
            "off",
            "--optimizer_type",
            "selective_adam",
        ]
    )
    assert args.optimizer_type == "selective_adam"
    # selective_adam requires sparse grads; nothing explicit contradicts it here.
    assert args.gsplat_sparse_grad is True


def test_explicit_sparse_grad_wins_under_off_profile():
    args, _ = _resolve(
        ["-s", "/tmp/x", "--parallelism_profile", "off", "--gsplat_sparse_grad"]
    )
    assert args.optimizer_type == "adam"
    assert args.gsplat_sparse_grad is True


# --------------------------------------------------------------------------
# The second trap: selective_adam silently re-forcing sparse grads
# --------------------------------------------------------------------------


def test_explicit_no_sparse_grad_with_selective_adam_is_a_loud_error():
    """Pre-fix this silently flipped gsplat_sparse_grad back to True."""
    with pytest.raises(ValueError, match="incompatible"):
        _resolve(
            [
                "-s",
                "/tmp/x",
                "--optimizer_type",
                "selective_adam",
                "--no-gsplat_sparse_grad",
            ]
        )


def test_off_plus_no_sparse_grad_alone_is_now_fine():
    """Pre-fix: silently sparse (optimizer_type was still selective_adam).
    Post-fix: `off` selects adam, so the explicit flag is consistent."""
    args, _ = _resolve(
        ["-s", "/tmp/x", "--parallelism_profile", "off", "--no-gsplat_sparse_grad"]
    )
    assert args.optimizer_type == "adam"
    assert args.gsplat_sparse_grad is False


def test_implicit_sparse_grad_under_selective_adam_is_not_an_error():
    """Only an EXPLICIT --no-gsplat_sparse_grad conflicts. A profile-supplied
    False (impossible today, but guard the code path) must not raise."""
    args, _ = _resolve(["-s", "/tmp/x", "--optimizer_type", "selective_adam"])
    assert args.gsplat_sparse_grad is True


# --------------------------------------------------------------------------
# sh_update_interval stays out of the profile table (see module docstring)
# --------------------------------------------------------------------------


def test_off_leaves_sh_update_interval_at_the_argparse_default():
    """Pins the rejected option (a): `off` must NOT select sh_update_interval=1."""
    args, _ = _resolve(["-s", "/tmp/x", "--parallelism_profile", "off"])
    assert args.sh_update_interval == 16


def test_issue106_dense_spec_shape_resolves_unchanged_in_sh_cadence():
    """The exact flag shape of specs/issue106_inline_batch_fixed.json."""
    args, _ = _resolve(
        [
            "-s",
            "/tmp/x",
            "--parallelism_profile",
            "off",
            "--optimizer_type",
            "adam",
            "--no-gsplat_sparse_grad",
        ]
    )
    assert (args.optimizer_type, args.gsplat_sparse_grad, args.sh_update_interval) == (
        "adam",
        False,
        16,
    )


def test_explicit_sh_update_interval_still_wins_under_safe(capsys):
    """#239/#246 behavior must survive this refactor, warning included."""
    args, _ = _resolve(["-s", "/tmp/x", "--sh_update_interval", "4"])
    assert args.sh_update_interval == 4
    assert getattr(args, "_sh_update_interval_override_warned", False) is True
    assert "issue #239" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Precedence: CLI > profile > config
# --------------------------------------------------------------------------


def test_cli_profile_off_beats_a_config_supplied_selective_adam():
    """configs/profiles/defaults.reference.yml carries
    optimizer_type: selective_adam -- it must not defeat an explicit
    --parallelism_profile off (issue #228 through the config door)."""
    args, _ = _resolve(
        ["-s", "/tmp/x", "--parallelism_profile", "off"],
        config={"optimizer_type": "selective_adam", "gsplat_sparse_grad": True},
    )
    assert args.optimizer_type == "adam"
    assert args.gsplat_sparse_grad is False


def test_config_field_beats_a_config_supplied_profile():
    """Same tier: when the profile itself came from config, the more specific
    config field wins."""
    args, _ = _resolve(
        ["-s", "/tmp/x"],
        config={"parallelism_profile": "safe", "optimizer_type": "adam",
                "gsplat_sparse_grad": False},
    )
    assert args.optimizer_type == "adam"
    assert args.gsplat_sparse_grad is False


def test_config_profile_safe_alone_still_selects_sparse():
    """configs/profiles/bench/r9700_streaming.yml sets only the profile."""
    args, _ = _resolve(["-s", "/tmp/x"], config={"parallelism_profile": "safe"})
    assert args.optimizer_type == "selective_adam"
    assert args.gsplat_sparse_grad is True


# --------------------------------------------------------------------------
# Misc contracts
# --------------------------------------------------------------------------


def test_unknown_profile_raises():
    args, bundle = parse_training_args(["-s", "/tmp/x"], description="test")
    args.parallelism_profile = "turbo"
    with pytest.raises(ValueError, match="Unsupported --parallelism_profile"):
        apply_parallelism_profile(args, explicit_dests=set())


def test_omitted_explicit_dests_lets_the_profile_win():
    """A caller that cannot compute explicit_dests degrades safely: the
    profile is authoritative rather than nothing being applied."""
    args, _ = parse_training_args(
        ["-s", "/tmp/x", "--parallelism_profile", "off", "--optimizer_type",
         "selective_adam"],
        description="test",
    )
    apply_parallelism_profile(args)
    assert args.optimizer_type == "adam"
    assert args.gsplat_sparse_grad is False


@pytest.mark.parametrize(
    "spelling",
    ["--no_gsplat_sparse_grad", "--gsplat_sparse_grad=False"],
)
def test_bad_negative_flag_spellings_still_fail_loudly(spelling):
    with pytest.raises(SystemExit):
        parse_training_args(["-s", "/tmp/x", spelling], description="test")


# --------------------------------------------------------------------------
# The [parallelism] startup line must report the truth
# --------------------------------------------------------------------------


def _parallelism_line(argv, config=None):
    """Reproduce train.py:632 / train_streaming.py:2754's log line from the
    same extracted OptimizationParams object those sites read."""
    opt = _resolved_opt(argv, config)
    optimizer_type = getattr(opt, "optimizer_type", "adam").lower()
    sparse_grad = bool(getattr(opt, "gsplat_sparse_grad", False))
    sh_update_interval = max(1, int(getattr(opt, "sh_update_interval", 1)))
    return (
        f"[parallelism] profile={getattr(opt, 'parallelism_profile', 'off')} "
        f"optimizer={optimizer_type} sparse_grad={sparse_grad} "
        f"sh_update_interval={sh_update_interval}"
    )


def test_parallelism_line_reports_dense_for_off():
    line = _parallelism_line(["-s", "/tmp/x", "--parallelism_profile", "off"])
    assert "profile=off optimizer=adam sparse_grad=False" in line
    assert "sh_update_interval=16" in line


def test_parallelism_line_reports_sparse_for_default_safe():
    line = _parallelism_line(["-s", "/tmp/x"])
    assert "profile=safe optimizer=selective_adam sparse_grad=True" in line


def test_parallelism_line_reports_explicit_override_under_safe():
    line = _parallelism_line(
        ["-s", "/tmp/x", "--optimizer_type", "adam", "--no-gsplat_sparse_grad"]
    )
    assert "profile=safe optimizer=adam sparse_grad=False" in line


# --------------------------------------------------------------------------
# Issue #1353: gsplat_default requires dense gradients
# --------------------------------------------------------------------------


def test_gsplat_default_forces_dense_gradients():
    """gsplat_default (vanilla ADC) requires dense gradients on the 2D means
    tensor, which sparse-gradient mode doesn't populate. When chosen without
    an explicit sparse_grad flag, it should force sparse_grad to False."""
    args, _ = _resolve(["-s", "/tmp/x", "--densification_strategy", "gsplat_default"])
    assert args.gsplat_sparse_grad is False


def test_explicit_sparse_grad_with_gsplat_default_is_a_loud_error():
    """If user explicitly tries to use sparse gradients with gsplat_default,
    raise a clear error instead of silently failing at runtime."""
    with pytest.raises(ValueError, match="incompatible"):
        _resolve(
            [
                "-s",
                "/tmp/x",
                "--densification_strategy",
                "gsplat_default",
                "--gsplat_sparse_grad",
            ]
        )


def test_gsplat_default_with_explicit_no_sparse_grad_is_fine():
    """Explicitly disabling sparse gradients with gsplat_default is allowed
    and compatible."""
    args, _ = _resolve(
        [
            "-s",
            "/tmp/x",
            "--densification_strategy",
            "gsplat_default",
            "--no-gsplat_sparse_grad",
        ]
    )
    assert args.densification_strategy == "gsplat_default"
    assert args.gsplat_sparse_grad is False


def test_gsplat_default_with_selective_adam_is_a_loud_error():
    """gsplat_default forces sparse_grad=False; selective_adam forces it
    back to True. Without a cross-check, the selective_adam block's
    unconditional `= True` would silently undo gsplat_default's dense-
    gradient requirement whenever sparse_grad itself wasn't explicit --
    reintroducing the exact crash this issue's guard exists to reject."""
    with pytest.raises(ValueError, match="incompatible"):
        _resolve(
            [
                "-s",
                "/tmp/x",
                "--densification_strategy",
                "gsplat_default",
                "--optimizer_type",
                "selective_adam",
            ]
        )
