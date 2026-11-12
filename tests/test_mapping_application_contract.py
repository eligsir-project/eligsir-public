from __future__ import annotations

import numpy as np
import pytest

from eligsir.contracts.runtime import BufferRef, SensorPacket, SourceIdentity
from eligsir.contracts.types import CameraIntrinsics
from eligsir.runtime.application import (
    MappingApplicationContractError,
    validate_mapping_application,
)
from eligsir.runtime.session import MappingSession
from eligsir.tracking.providers import (
    ExternalTrackingProvider,
    observation_from_pose_matrix,
)


def _packet(ordinal: int) -> SensorPacket:
    source = SourceIdentity(
        source_id="cam",
        stream_epoch=0,
        source_ordinal=ordinal,
        timestamp_ns=1_000_000_000 + ordinal,
    )
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.zeros((4, 4, 3), dtype=np.uint8), encoding="rgb8"),
        depth=BufferRef.from_numpy(
            np.ones((4, 4), dtype=np.float32), encoding="32FC1", units="m"
        ),
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4),
        tracking=observation_from_pose_matrix(source, np.eye(4), provider="dataset_pose"),
    )


class _Application:
    def __init__(self) -> None:
        self.mapped = []
        self.session = MappingSession(
            ExternalTrackingProvider(),
            mapper=lambda frame, _observation: self.mapped.append(frame.source.source_ordinal),
        )
        self.ticks = 0
        self.flushed = 0
        self.closed = 0
        self.source_complete = False

    def tick(self):
        self.ticks += 1

    def flush(self):
        self.flushed += 1
        return self.session.flush()

    def mark_source_complete(self):
        self.source_complete = True

    def close(self):
        self.closed += 1
        self.session.close()


def test_full_application_is_accepted():
    app = _Application()
    assert validate_mapping_application(app) is app


def test_tick_is_independent_of_frame_submission():
    app = _Application()
    validate_mapping_application(app)

    app.tick()
    app.session.submit(_packet(0))
    app.session.submit(_packet(1))
    app.tick()
    app.tick()

    assert app.mapped == [0, 1]
    assert app.ticks == 3


def test_raw_session_is_not_an_application():
    session = MappingSession(
        ExternalTrackingProvider(), mapper=lambda _frame, _observation: None
    )
    with pytest.raises(MappingApplicationContractError, match="tick"):
        validate_mapping_application(session)


def test_ingress_only_session_cannot_back_production_application():
    class _IngressApplication:
        def __init__(self):
            self.session = MappingSession(
                ExternalTrackingProvider(), ingress_only=True
            )

        def tick(self):
            pass

        def flush(self):
            pass

        def mark_source_complete(self):
            pass

        def close(self):
            pass

    with pytest.raises(MappingApplicationContractError, match="ingress-only"):
        validate_mapping_application(_IngressApplication())
