"""SLAM worker isolation runtime (Wave 3, PR W3-04, plan section 9).

Every production SLAM backend must present the :class:`~eligsir.slam.
runtime.worker.SLAMWorker` protocol (``submit``/``try_get``/``health``/
``close``) so its latency can never execute on the optimization thread --
see ``worker.py`` for the protocol and rationale, ``process_worker.py`` for
the out-of-process wrapper used by in-process backends (e.g.
``open3d_odometry_live``), and ``external_worker.py`` for the thin
thread-backed wrapper used by backends that are already external (e.g.
``ros_slam``, ORB-SLAM3's ROS2 process).
"""

from __future__ import annotations

from .health import WorkerHealth, WorkerState
from .worker import SLAMWorker, SubmitOutcome, SubmitResult

__all__ = [
    "SLAMWorker",
    "SubmitOutcome",
    "SubmitResult",
    "WorkerHealth",
    "WorkerState",
]
