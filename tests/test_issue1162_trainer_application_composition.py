from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from scene.streaming_scene import StreamingScene
from eligsir.contracts.runtime import BufferRef, SensorPacket, SourceIdentity
from eligsir.contracts.types import CameraIntrinsics
from eligsir.runtime.streaming_sink import StreamingIngestCallbacks
from eligsir.tracking.providers import observation_from_pose_matrix
from train_streaming import _compose_mapping_application


def _packet() -> SensorPacket:
    source = SourceIdentity("ros", 0, 1, 1_000_000_000)
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(
            np.zeros((2, 2, 3), dtype=np.uint8), encoding="rgb8"
        ),
        depth=BufferRef.from_numpy(
            np.ones((2, 2), dtype=np.float32), encoding="32FC1", units="m"
        ),
        intrinsics=CameraIntrinsics(
            fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=2, height=2
        ),
        tracking=observation_from_pose_matrix(
            source, np.eye(4), provider="external"
        ),
    )


def test_trainer_composition_constructs_real_scene_session_and_tick():
    scene = object.__new__(StreamingScene)
    mapped = []
    ticks = []
    flushed = []
    closed = []

    def ingest_next_frame(**kwargs):
        mapped.append(kwargs["_provided_frame"])

    scene.ingest_next_frame = ingest_next_frame
    scene.flush_deferred_extent_update = lambda: flushed.append(1)
    scene.close_decode_cache = lambda: closed.append(1)
    application = _compose_mapping_application(
        scene,
        lambda: ticks.append(1),
        StreamingIngestCallbacks(defer_extent_update=True),
    )

    outcome = application.session.submit(_packet())
    application.tick()
    application.close()

    assert outcome.accepted
    assert mapped[0].mapping_frame is outcome.frame
    assert ticks == [1]
    assert flushed == [1]
    assert closed == [1]


def test_streaming_root_routes_the_existing_engine_call_through_one_application_tick():
    path = Path(__file__).resolve().parents[1] / "train_streaming.py"
    tree = ast.parse(path.read_text())
    training = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "streaming_training"
    )
    loop = next(
        node
        for node in ast.walk(training)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "iteration"
    )
    compose_calls = [
        node
        for node in ast.walk(training)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_compose_mapping_application"
    ]
    engine_calls = [
        node
        for node in ast.walk(training)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_streaming_optimization_step"
    ]
    tick_calls = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "tick"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_mapping_application"
    ]

    assert len(compose_calls) == 1
    assert compose_calls[0].lineno < loop.lineno
    assert len(engine_calls) == 1
    assert engine_calls[0].lineno < loop.lineno
    assert len(tick_calls) == 1
