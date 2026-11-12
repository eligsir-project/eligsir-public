from types import SimpleNamespace

from eligsir.runtime.application import ComposedMappingApplication
from eligsir.runtime.causal_prepared_ready import PreparedBootstrapOutcome
from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridgeResult
from eligsir.runtime.ready_only_tick import ReadyOnlyTickSelection


def _application(results, waits):
    application = object.__new__(ComposedMappingApplication)
    application._closed = False
    application._tick_count = 0
    application._flush_required = False
    application._prepared_retry_pending = False
    application._options = SimpleNamespace(
        prepared_mode=True,
        prepared_tick=lambda: next(results),
        prepared_progress_waiter=lambda *, timeout_s: waits.append(timeout_s) or True,
    )
    return application


def test_progress_wait_is_only_armed_by_retryable_prepared_result():
    waits = []
    results = iter((SimpleNamespace(retryable=False), SimpleNamespace(retryable=True)))
    application = _application(results, waits)

    application.tick()
    assert application.wait_for_prepared_progress(timeout_s=0.001) is False
    assert waits == []

    application.tick()
    assert application.wait_for_prepared_progress(timeout_s=0.001) is True
    assert waits == [0.001]


def test_bootstrap_pending_is_retryable_without_claiming_optimizer_work():
    ingress = PreparedRuntimeBridgeResult(0, 0, 0, 0)
    selection = ReadyOnlyTickSelection((), 0, 0, False, True)

    pending = PreparedBootstrapOutcome("bootstrap_pending", ingress, selection)
    done = PreparedBootstrapOutcome("bootstrapped", ingress, selection)

    assert pending.retryable is True
    assert pending.executed is False
    assert done.retryable is False
