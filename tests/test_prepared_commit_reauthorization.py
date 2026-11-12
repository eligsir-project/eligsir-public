from collections import deque
from types import SimpleNamespace
import threading

from eligsir.contracts import SourceIdentity
from eligsir.contracts.control import SemanticAction
from eligsir.runtime.prepared_cold_promotion import (
    PreparedColdPromotionOwner,
    _PromotionRequest,
)
from eligsir.runtime.prepared_source_producer import PreparedSourceProduceResult


def test_unprotected_retained_commit_returns_to_runtime_for_reauthorization():
    source = SourceIdentity("reauthorize", 0, 7, 7_000)
    item = SimpleNamespace(
        packet=SimpleNamespace(source=source),
        disposition=SimpleNamespace(semantic_action=SemanticAction.COMMIT),
    )
    request = _PromotionRequest(
        7,
        SimpleNamespace(item=item, initial_fresh=True, promotion_permit=None),
        frozenset(),
        1,
    )

    owner = object.__new__(PreparedColdPromotionOwner)
    owner._bridge = SimpleNamespace(
        submit_retained_commit=lambda *_args, **_kwargs: (
            PreparedSourceProduceResult.RETAINED_NOT_RESIDENT
        )
    )
    owner._ready_scheduler = SimpleNamespace(
        has_source=lambda _source: False,
        evict_fresh=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("no frozen exact plan may evict FRESH residency")
        ),
    )
    owner._queue = deque((request,))
    owner._scheduled = {7: request}
    owner._lock = threading.Lock()

    assert owner.pump_once() is True
    assert not owner._queue
    assert owner._scheduled == {}
    assert owner.pump_once() is False
