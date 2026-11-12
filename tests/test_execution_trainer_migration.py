"""Regression coverage for legacy streaming-replay migration behavior.

Issue #1224 hard-cut the legacy ``train_streaming.py``/``streaming_training()``
path out of ``train.py``'s dispatch entirely: ``training()`` raises
``SystemExit`` on ``args.streaming_replay`` truthy UNCONDITIONALLY, before the
``execution_trainer`` branch is even consulted. The tests below (added for the
audit that confirmed this hard cut, rather than re-doing it) close a real gap
in the original coverage: the sole existing test
(``test_legacy_streaming_replay_rejects_before_causal_dispatch``) only
exercised this with ``execution_trainer="causal_mapping"``. Since the actual
argparse default is ``execution_trainer="offline_reconstruction"`` (a live,
legitimate, non-streaming trainer -- the classic full-batch offline COLMAP/
mipnerf360 loop, NOT the dead streaming path -- see
``tests/test_cli_config.py::test_training_parser_declares_the_canonical_execution_trainer_default``),
the retirement must also hold under that default, and under a real
argparse-parsed args object carrying no execution-trainer-selecting flags at
all -- not just the one value the original test happened to pick.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from eligsir.cli.parser import parse_training_args


def test_legacy_streaming_replay_rejects_before_causal_dispatch(monkeypatch):
    """The retired flag must not silently enter the causal production path."""
    pytest.importorskip("torch", reason="train.py imports the accelerator backend")
    import train
    from eligsir.utils import experiment_origin

    monkeypatch.setattr(experiment_origin, "validate_researchflow_origin", lambda args: None)

    causal_entry = ModuleType("eligsir.runtime.causal_entry")

    def unexpected_causal_dispatch(**_kwargs):
        pytest.fail("legacy streaming replay reached the causal runtime")

    causal_entry.run_causal_mapping = unexpected_causal_dispatch
    monkeypatch.setitem(sys.modules, "eligsir.runtime.causal_entry", causal_entry)

    args = SimpleNamespace(execution_trainer="causal_mapping", streaming_replay=True)
    with pytest.raises(SystemExit, match="streaming_replay.*retired"):
        train.training(
            object(),
            object(),
            object(),
            (),
            (),
            (),
            None,
            -1,
            (),
            run_args=args,
        )


@pytest.mark.parametrize("execution_trainer", ["offline_reconstruction", "causal_mapping"])
def test_legacy_streaming_replay_rejects_for_every_execution_trainer(monkeypatch, execution_trainer):
    """The retired flag must fail closed regardless of which trainer is selected --
    not just under ``causal_mapping``, the only value the original test covered."""
    pytest.importorskip("torch", reason="train.py imports the accelerator backend")
    import train
    from eligsir.utils import experiment_origin

    monkeypatch.setattr(experiment_origin, "validate_researchflow_origin", lambda args: None)

    args = SimpleNamespace(execution_trainer=execution_trainer, streaming_replay=True)
    with pytest.raises(SystemExit, match="streaming_replay.*retired"):
        train.training(
            object(), object(), object(), (), (), (), None, -1, (), run_args=args
        )


def test_evwb_rejects_offline_dispatch_before_training(monkeypatch):
    """EV/WB needs causal source-role validation, unavailable offline."""
    pytest.importorskip("torch", reason="train.py imports the accelerator backend")
    import train
    from eligsir.utils import experiment_origin

    monkeypatch.setattr(experiment_origin, "validate_researchflow_origin", lambda args: None)
    args = SimpleNamespace(
        execution_trainer="offline_reconstruction",
        streaming_replay=False,
        streaming_train_evwb_enabled=True,
    )

    with pytest.raises(SystemExit, match="causal_mapping"):
        train.training(
            object(), object(), object(), (), (), (), None, -1, (), run_args=args
        )


def test_evwb_flag_off_preserves_offline_dispatch(monkeypatch):
    pytest.importorskip("torch", reason="train.py imports the accelerator backend")
    import train
    from eligsir.utils import experiment_origin

    marker = object()
    monkeypatch.setattr(experiment_origin, "validate_researchflow_origin", lambda args: None)
    monkeypatch.setattr(train, "_train_offline", lambda *_args, **_kwargs: marker)
    args = SimpleNamespace(
        execution_trainer="offline_reconstruction",
        streaming_replay=False,
        streaming_train_evwb_enabled=False,
        system_telemetry=False,
    )

    assert train.training(
        object(), object(), object(), (), (), (), None, -1, (), run_args=args
    ) is marker


def test_default_execution_trainer_is_offline_reconstruction_not_a_streaming_trainer():
    """Cheap, deterministic, CPU-only (no torch import): a bare invocation with no
    execution-trainer-selecting flag at all resolves to ``offline_reconstruction`` --
    the classic full-batch offline loop -- never to the dead streaming trainer.

    This does not assert the default is ``causal_mapping``: ``offline_reconstruction``
    is a live, legitimate, non-streaming path (see CLAUDE.md's "Running Commands" --
    the bicycle/mipnerf360 recipe passes no ``--execution_trainer`` at all), not the
    confirmed-dead ``train_streaming.py`` path this lockdown is about. There is no
    argparse value of ``execution_trainer`` that reaches ``train_streaming.py`` at
    all -- see ``test_legacy_streaming_replay_rejects_for_every_execution_trainer``
    and ``scripts/check_canonical_mapping_hard_cut.py``'s historical-boundary check,
    which asserts ``train.py`` contains no call to ``streaming_training`` and no
    import of ``train_streaming`` anywhere in the file, unconditional on which
    branch is live.
    """
    args, _bundle = parse_training_args(["-s", "/tmp/nonexistent"])
    assert args.execution_trainer == "offline_reconstruction"
    assert getattr(args, "streaming_replay", False) is False


def test_causal_dispatch_finalizes_staged_outputs_before_return(monkeypatch):
    pytest.importorskip("torch", reason="train.py imports the accelerator backend")
    import train
    from eligsir.runtime import causal_entry
    from eligsir.utils import experiment_origin

    finalized = []
    marker = object()
    monkeypatch.setattr(experiment_origin, "validate_researchflow_origin", lambda args: None)
    monkeypatch.setattr(causal_entry, "run_causal_mapping", lambda **_kwargs: marker)
    monkeypatch.setattr(train, "_finalize_output_staging", finalized.append)
    args = SimpleNamespace(execution_trainer="causal_mapping", streaming_replay=False)

    result = train.training(
        object(), object(), object(), (), (), (), None, -1, (), run_args=args
    )

    assert result is marker
    assert finalized == [args]


def test_start_checkpoint_rejects_before_causal_dispatch(monkeypatch):
    """issue #1552 bug 1: causal_entry.py has no checkpoint-restore path, so
    --start_checkpoint used to be silently ignored (a resumed run cold-started
    from scratch with no error). It must now fail closed instead."""
    pytest.importorskip("torch", reason="train.py imports the accelerator backend")
    import train
    from eligsir.utils import experiment_origin

    monkeypatch.setattr(experiment_origin, "validate_researchflow_origin", lambda args: None)

    causal_entry = ModuleType("eligsir.runtime.causal_entry")

    def unexpected_causal_dispatch(**_kwargs):
        pytest.fail("--start_checkpoint reached the causal runtime instead of failing closed")

    causal_entry.run_causal_mapping = unexpected_causal_dispatch
    monkeypatch.setitem(sys.modules, "eligsir.runtime.causal_entry", causal_entry)

    args = SimpleNamespace(execution_trainer="causal_mapping", streaming_replay=False)
    with pytest.raises(SystemExit, match="start_checkpoint.*causal_mapping"):
        train.training(
            object(),
            object(),
            object(),
            (),
            (),
            (),
            "output/run/chkpnt7000.pth",
            -1,
            (),
            run_args=args,
        )


def test_causal_dispatch_crash_records_completion_state_then_reraises(monkeypatch, tmp_path):
    """issue #1278/#1284: a causal_mapping crash used to leave no
    run_completion_state.json at all. training() must record the real
    exception before re-raising it (never swallow it)."""
    pytest.importorskip("torch", reason="train.py imports the accelerator backend")
    import train
    from eligsir.runtime import causal_entry
    from eligsir.utils import experiment_origin
    from eligsir.utils.run_completion_state import Outcome, load_run_completion_state

    monkeypatch.setattr(experiment_origin, "validate_researchflow_origin", lambda args: None)

    def crash(**_kwargs):
        raise RuntimeError("HIP out of memory")

    monkeypatch.setattr(causal_entry, "run_causal_mapping", crash)
    monkeypatch.setattr(train, "_finalize_output_staging", lambda _args: None)
    args = SimpleNamespace(
        execution_trainer="causal_mapping",
        streaming_replay=False,
        model_path=str(tmp_path),
    )

    with pytest.raises(RuntimeError, match="HIP out of memory"):
        train.training(
            object(), object(), object(), (), (), (), None, -1, (), run_args=args
        )

    disk = load_run_completion_state(tmp_path)
    assert disk["outcome"] == Outcome.TRAINING_INCOMPLETE
    assert disk["failure_type"] == "RuntimeError"
    assert "HIP out of memory" in disk["failure_reason"]


def test_start_checkpoint_none_still_dispatches_causal_mapping(monkeypatch):
    """The new guard must only reject a truthy checkpoint, not the ordinary
    no-checkpoint causal_mapping path exercised elsewhere in this file."""
    pytest.importorskip("torch", reason="train.py imports the accelerator backend")
    import train
    from eligsir.runtime import causal_entry
    from eligsir.utils import experiment_origin

    marker = object()
    monkeypatch.setattr(experiment_origin, "validate_researchflow_origin", lambda args: None)
    monkeypatch.setattr(causal_entry, "run_causal_mapping", lambda **_kwargs: marker)
    monkeypatch.setattr(train, "_finalize_output_staging", lambda _args: None)
    args = SimpleNamespace(execution_trainer="causal_mapping", streaming_replay=False)

    result = train.training(
        object(), object(), object(), (), (), (), None, -1, (), run_args=args
    )

    assert result is marker
