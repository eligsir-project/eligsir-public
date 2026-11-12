"""Issue #584: a --config JSON (or YAML run-profile) supplied `resolution`
must not be silently clobbered back to native (1) by StreamingScene's
internal loadCam-scale pin.

`resolve_streaming_resolution_pin` (scene/streaming_scene.py) is the pure,
CPU-only decision function StreamingScene.__init__ now delegates to -- these
tests exercise it directly (no torch, no GPU, no real StreamingScene
construction) across the four combinations the issue asks for: --config only,
explicit CLI flag only, both, neither.
"""
from __future__ import annotations

from types import SimpleNamespace

from scene.streaming_scene import (
    _RESOLUTION_ARGPARSE_DEFAULT,
    resolve_streaming_resolution_pin,
)


def test_neither_cli_nor_config_pins_to_native():
    """Default case: args.resolution is still the argparse default (-1) and
    no --resolution/-r flag is in argv -- the pin fires, source=default."""
    args = SimpleNamespace(resolution=_RESOLUTION_ARGPARSE_DEFAULT)
    effective, pinned, source = resolve_streaming_resolution_pin(args, argv=[])
    assert effective == 1
    assert pinned is True
    assert source == "default"


def test_config_supplied_resolution_is_honored_not_clobbered():
    """The issue's core repro: --config <file>.json set args.resolution=4 on
    the parsed namespace (train.py's __main__ config merge, BEFORE
    StreamingScene is ever constructed), with no --resolution/-r literal in
    argv. Must NOT be pinned back to 1."""
    args = SimpleNamespace(resolution=4)
    # argv reflects only what the user actually typed: --config, not --resolution.
    argv = ["-s", "/data/x", "--streaming_replay", "--config", "configs/some_streaming.json"]
    effective, pinned, source = resolve_streaming_resolution_pin(args, argv=argv)
    assert effective == 4
    assert pinned is False
    assert source == "config_or_profile"


def test_explicit_cli_flag_is_honored():
    """--resolution 2 literally in argv -- honored via the original argv scan."""
    args = SimpleNamespace(resolution=2)
    argv = ["-s", "/data/x", "--streaming_replay", "--resolution", "2"]
    effective, pinned, source = resolve_streaming_resolution_pin(args, argv=argv)
    assert effective == 2
    assert pinned is False
    assert source == "cli"


def test_explicit_cli_shorthand_and_equals_form_are_honored():
    args = SimpleNamespace(resolution=8)
    assert resolve_streaming_resolution_pin(args, argv=["-r", "8"])[1] is False
    assert resolve_streaming_resolution_pin(
        args, argv=["--resolution=8"]
    )[1] is False
    assert resolve_streaming_resolution_pin(args, argv=["-r=8"])[1] is False


def test_both_cli_and_config_set_resolution_cli_wins_source_label():
    """Both an explicit CLI flag AND a config value are present (e.g. a
    --config file sets resolution=4 but the user also passes --resolution 2
    on the command line, which argparse/CLI-explicit-wins semantics already
    make authoritative on args.resolution itself). The pin must still not
    fire, and the source should report "cli" since argv is the stronger
    signal when both are present."""
    args = SimpleNamespace(resolution=2)  # CLI already won upstream, per train.py's explicit-wins merge
    argv = ["--config", "configs/some_streaming.json", "--resolution", "2"]
    effective, pinned, source = resolve_streaming_resolution_pin(args, argv=argv)
    assert effective == 2
    assert pinned is False
    assert source == "cli"


def test_explicit_cli_default_value_is_still_honored_not_repinned():
    """Narrow pre-existing edge case this fix deliberately preserves: a user
    who explicitly types --resolution -1 (numerically identical to the
    argparse default) asked for the auto-cap behavior on purpose and must
    still be exempted from the pin, exactly as before this fix."""
    args = SimpleNamespace(resolution=_RESOLUTION_ARGPARSE_DEFAULT)
    argv = ["--resolution", "-1"]
    effective, pinned, source = resolve_streaming_resolution_pin(args, argv=argv)
    assert pinned is False
    assert source == "cli"
    assert effective == _RESOLUTION_ARGPARSE_DEFAULT


def test_missing_resolution_attribute_treated_as_default():
    """Defensive: an args object with no resolution attribute at all behaves
    like the argparse default (matches the original code's getattr default)."""
    args = SimpleNamespace()
    effective, pinned, source = resolve_streaming_resolution_pin(args, argv=[])
    assert effective == 1
    assert pinned is True
    assert source == "default"


def test_streaming_scene_init_applies_resolved_pin_and_logs(capsys, monkeypatch):
    """End-to-end (still no torch/GPU): construct enough of StreamingScene's
    __init__ prologue to exercise the real pin call site, via the same
    lightweight monkeypatch pattern test_streaming_scene_lazy_camera_image.py
    already uses for this class (KeyframeStore et al. are real, pure-Python,
    no torch needed for construction alone; frame_source is a stub)."""
    from scene.streaming_scene import StreamingScene

    class _FakeFrameSource:
        def __len__(self):
            return 0

    args = SimpleNamespace(
        resolution=4,
        model_path="/tmp/does-not-need-to-exist-for-this-test",
    )
    monkeypatch.setattr(
        "sys.argv", ["train.py", "--streaming_replay", "--config", "configs/x.json"]
    )
    try:
        StreamingScene(args, gaussians=object(), frame_source=_FakeFrameSource())
    except Exception:
        # StreamingScene.__init__ does much more after the resolution pin
        # (KeyframeStore wiring, etc.) that may raise on this minimal stub --
        # irrelevant to this test, which only asserts on the pin's own effect,
        # which runs first and unconditionally.
        pass
    assert args.resolution == 4, (
        "config-supplied resolution=4 must survive StreamingScene.__init__ "
        "(issue #584) -- it must not be silently clobbered to 1"
    )
    out = capsys.readouterr().out
    assert "[StreamingScene] resolution pin:" in out
    assert "pinned=False" in out
    assert "source=config_or_profile" in out
