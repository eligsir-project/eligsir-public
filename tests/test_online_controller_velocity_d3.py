from eligsir.mapping.scheduler.online_controller import OnlineController, OnlineConfig
from eligsir.mapping.scheduler.mapping_scheduler import MappingScheduler

def _ctrl(**cfg):
    sched = MappingScheduler(steps_per_frame=10, n_frames=100, iterations=5000)
    return OnlineController(sched, OnlineConfig(adaptive_steps=False, **cfg))

def test_disabled_factor_is_one():
    c = _ctrl()
    c.record_camera_speed(0.001)
    assert c._velocity_step_factor() == 1.0

def test_stalled_boosts():
    c = _ctrl(velocity_adaptive_steps=True, velocity_stall_mps=0.02, velocity_fast_mps=0.30, velocity_step_gain=1.5)
    c.record_camera_speed(0.005)  # below stall
    assert abs(c._velocity_step_factor() - 1.5) < 1e-6

def test_fast_cuts():
    c = _ctrl(velocity_adaptive_steps=True, velocity_stall_mps=0.02, velocity_fast_mps=0.30, velocity_step_gain=1.5)
    c.record_camera_speed(0.50)  # above fast
    assert abs(c._velocity_step_factor() - (1/1.5)) < 1e-6

def test_midpoint_near_one():
    c = _ctrl(velocity_adaptive_steps=True, velocity_stall_mps=0.0, velocity_fast_mps=1.0, velocity_step_gain=4.0)
    c.record_camera_speed(0.5)  # exact midpoint -> log-symmetric -> 1.0
    assert abs(c._velocity_step_factor() - 1.0) < 1e-6

def test_none_and_no_sample_safe():
    c = _ctrl(velocity_adaptive_steps=True)
    assert c._velocity_step_factor() == 1.0  # no sample yet
    c.record_camera_speed(None)
    assert c._velocity_step_factor() == 1.0
