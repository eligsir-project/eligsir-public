"""Consumed-ingress source-rate ownership across causal replay roles."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from eligsir.contracts.runtime import BufferRef, SensorPacket, TrackingObservation, TrackingStatus
from eligsir.contracts.types import CameraIntrinsics
from eligsir.mapping.canonical_services import CanonicalFourClocks
from eligsir.runtime.application import create_mapping_application
from eligsir.runtime.causal_entry import (
    CausalMappingComposition,
    causal_mapping_application_options,
)
from eligsir.runtime.causal_replay import CausalReplayDriver, DatasetPacketAdapter, PacedCausalReplayDriver
from eligsir.runtime.config import RuntimeBinding
from eligsir.runtime.production_application import source_release_rate_ratio_from_clocks
from eligsir.runtime.session import MappingSession, MappingSessionClosed
from eligsir.tracking.providers import observation_from_pose_matrix


@dataclass(frozen=True)
class _Record:
    ordinal: int


class _Adapter(DatasetPacketAdapter[_Record]):
    def packet_from_record(self, record: _Record) -> SensorPacket:
        from eligsir.contracts.runtime import SourceIdentity

        source = SourceIdentity("source-rate", 0, record.ordinal, record.ordinal + 1)
        return SensorPacket(
            source=source,
            rgb=BufferRef.from_numpy(np.zeros((2, 2, 3), dtype=np.uint8), encoding="rgb8"),
            depth=None,
            intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=2, height=2),
            tracking=observation_from_pose_matrix(source, np.eye(4), provider="test"),
        )


class _RejectedTracking:
    provider_name = "test"

    def observe(self, packet, *, predicted_pose=None):
        return TrackingObservation(
            source=packet.source,
            pose=None,
            status=TrackingStatus.INVALID,
            provider=self.provider_name,
            confidence=0.0,
        )

    def close(self):
        pass


@pytest.mark.parametrize("paced", (False, True))
def test_all_role_consumption_counts_train_holdout_and_guard_once(paced):
    consumed = []
    completed = []
    clocks = CanonicalFourClocks()

    def observe(packet):
        clocks.record_source_frame(wall_ts=len(consumed) / 30.0)
        consumed.append(packet.source.source_ordinal)

    binding = RuntimeBinding(
        mapper_sink=lambda *_args: None,
        optimizer_tick=lambda: SimpleNamespace(
            optimizer_steps=1, supervision_views=1, executed=True
        ),
        source_consumed_observer=observe,
    )
    application = create_mapping_application(
        causal_mapping_application_options(
            CausalMappingComposition(
                source=object(),
                config=SimpleNamespace(),
                binding=binding,
                runtime=SimpleNamespace(),
            ),
            args=SimpleNamespace(),
            ingress_telemetry=None,
        )
    )
    kwargs = {
        "packet_role": lambda packet: (
            "holdout"
            if packet.source.source_ordinal == 1
            else "guard"
            if packet.source.source_ordinal == 2
            else "train"
        ),
        "source_consumed_observer": observe,
        "source_complete_observer": lambda _result: completed.append(True),
    }
    driver = (
        PacedCausalReplayDriver(application, _Adapter(), source_hz=1000.0, **kwargs)
        if paced
        else CausalReplayDriver(application, _Adapter(), **kwargs)
    )

    summary = driver.run((_Record(0), _Record(1), _Record(2)))

    assert summary.submitted == 1
    assert summary.holdout_reserved == summary.guard_withheld == 1
    assert consumed == [0, 1, 2]
    assert completed == [True]
    assert source_release_rate_ratio_from_clocks(
        clocks, requested_source_fps=30.0, now=2.0 / 30.0
    ) == pytest.approx(1.0)


def test_all_role_clock_avoids_the_fr2_train_only_false_shortfall():
    all_roles = CanonicalFourClocks()
    train_only = CanonicalFourClocks()
    for index in range(51):
        all_roles.record_source_frame(wall_ts=index / 30.0)
    for index in range(42):
        train_only.record_source_frame(wall_ts=index * (50.0 / 30.0) / 41.0)

    assert source_release_rate_ratio_from_clocks(
        all_roles, requested_source_fps=30.0, now=50.0 / 30.0
    ) == pytest.approx(1.0)
    assert source_release_rate_ratio_from_clocks(
        train_only, requested_source_fps=30.0, now=50.0 / 30.0
    ) == pytest.approx(0.82)


def test_session_counts_tracking_rejections_and_duplicate_submit_attempts():
    consumed = []
    packet = _Adapter().packet_from_record(_Record(0))
    session = MappingSession(
        _RejectedTracking(),
        ingress_only=True,
        source_consumed_observer=lambda observed: consumed.append(
            observed.source.source_ordinal
        ),
    )

    first = session.submit(packet)
    duplicate = session.submit(packet)

    assert not first.accepted
    assert not duplicate.accepted
    assert consumed == [0, 0]

    session.close()
    with pytest.raises(MappingSessionClosed):
        session.submit(packet)
    assert consumed == [0, 0]
