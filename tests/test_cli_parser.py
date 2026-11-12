"""Tests for shared trainer CLI parser construction."""

from __future__ import annotations

from eligsir.cli.parser import parse_training_args


def test_cli_wrapper_imports_are_lightweight():
    """Run in a fresh interpreter: a same-process sys.modules check is
    inherently test-order-dependent in a full-suite run (many OTHER tests
    legitimately import train/render/metrics), so this was flaking under
    `pytest tests/` despite passing in isolation -- the subprocess makes the
    result independent of what else has already run in this session."""
    import subprocess
    import sys

    script = (
        "import sys\n"
        "import eligsir.cli.eval\n"
        "import eligsir.cli.live\n"
        "import eligsir.cli.render\n"
        "import eligsir.cli.train\n"
        "assert 'train' not in sys.modules\n"
        "assert 'render' not in sys.modules\n"
        "assert 'metrics' not in sys.modules\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=".", capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


def test_training_parser_exposes_common_trainer_flags():
    args, bundle = parse_training_args(
        [
            "-s",
            "ros2://live",
            "-m",
            "output/test",
            "--viewer",
            "none",
            "--iterations",
            "10",
            "--checkpoint_interval",
            "0",
            "--save_interval",
            "0",
            "--log_memory",
            "--profile",
            "/tmp/profile.json",
        ]
    )

    assert args.source_path == "ros2://live"
    assert args.model_path == "output/test"
    assert args.viewer == "none"
    assert args.iterations == 10
    assert args.checkpoint_interval == 0
    assert args.save_interval == 0
    assert args.log_memory is True
    assert args.profile == "/tmp/profile.json"
    assert bundle.model_params is not None
    assert bundle.optimization_params is not None
    assert bundle.pipeline_params is not None


def test_format_help_does_not_raise_on_a_literal_percent_in_help_text():
    """`train.py --help` crashed with `ValueError: unsupported format character
    'p'` because one help string contained a literal, unescaped `%` ("a 5%
    prefix"). argparse runs every help string through `% params`, so a bare `%`
    is a format directive, not a character.

    This guards the whole parser, not just that one string: any future help text
    with an unescaped `%` fails here instead of in the user's terminal. Worth a
    test rather than a one-character fix alone -- a CLI whose `--help` does not
    run forces every flag-default question to be answered by reading argparse
    source, which is the failure mode behind this repo's several documented
    wrong-default incidents.
    """
    from eligsir.cli.parser import build_training_parser

    bundle = build_training_parser()
    rendered = bundle.parser.format_help()

    assert "5% prefix" in rendered, (
        "the escaped `5%%` should render as a literal `5%` in the help output"
    )
