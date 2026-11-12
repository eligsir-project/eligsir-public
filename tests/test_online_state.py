from types import SimpleNamespace

import pytest

from eligsir.core.online_state import DirtyRegionQueue, MapWorkItem, OnlineClock


def test_online_clock_defaults_are_session_counters():
    clock = OnlineClock()
    assert clock.session_step == 0
    assert clock.frame_idx == 0
    assert clock.keyframe_idx == 0
    assert clock.wall_time_s == 0.0
    assert clock.map_version == 0


def test_dirty_region_queue_pops_highest_priority_first():
    queue = DirtyRegionQueue()
    low = MapWorkItem("low", priority=1.0, region_id="a", created_at_step=1)
    high = MapWorkItem("high", priority=5.0, region_id="b", created_at_step=2)
    queue.push(low)
    queue.push(high)

    assert len(queue) == 2
    assert queue.pop() is high
    assert queue.pop() is low
    assert queue.pop() is None


def test_dirty_region_queue_resume_preserves_priority_fifo_and_counter():
    cameras = {index: SimpleNamespace(uid=index) for index in range(4)}
    source = DirtyRegionQueue()
    source.push(MapWorkItem("refine", 1.0, cameras[2], 7))
    source.push(MapWorkItem("refine", 1.0, cameras[0], 8))
    payload = source.resume_state_dict()

    restored = DirtyRegionQueue()
    restored.load_resume_state_dict(payload, cameras.get)
    restored.push(MapWorkItem("refine", 1.0, cameras[3], 9))

    assert [restored.pop().region_id.uid for _ in range(3)] == [2, 0, 3]


def test_dirty_region_queue_restore_is_validation_atomic():
    cameras = {index: SimpleNamespace(uid=index) for index in range(10)}
    source = DirtyRegionQueue()
    source.push(MapWorkItem("refine", 2.0, cameras[1], 4))
    source.push(MapWorkItem("refine", 1.0, cameras[2], 5))
    broken = source.resume_state_dict()
    broken["items"][-1]["source_index"] = -1

    target = DirtyRegionQueue()
    target.push(MapWorkItem("sentinel", 3.0, cameras[9], 1))
    before = target.resume_state_dict()
    with pytest.raises(ValueError, match="source index"):
        target.load_resume_state_dict(broken, cameras.get)

    assert target.resume_state_dict() == before


def test_utils_online_state_reexports_core_types():
    from eligsir.utils.online_state import OnlineClock as CompatOnlineClock

    assert CompatOnlineClock is OnlineClock
