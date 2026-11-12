"""The single, transport-neutral "online every-Nth" holdout-reservation rule.

Issue #1302: ``deterministic_bag.launch.py``'s ROS2 lifecycle-node mapping
path (``eligsir-ros2``) has no eval/report mechanism at all -- it never
reserves a holdout, never renders a test view, never computes a PSNR. That
path also has no :class:`~scene.streaming_scene.StreamingScene` (it drives
``eligsir.runtime.production_application.CanonicalMappingRuntime``
directly from ROS2 messages), so it cannot simply call
``StreamingScene.holdout_status`` -- there is no ``StreamingScene`` instance
to call it on.

Rather than re-deriving the holdout rule for that caller (and risking a
second implementation that quietly drifts from the first -- exactly the
"never split and re-paired" failure class this project's RGB-D pairing
policy already warns about for a different subsystem), this module extracts
the ONE piece of ``StreamingScene.holdout_status`` that is already,
independently, pure and self-contained: for a "live/lazy" source (future
poses unknown, ``StreamingScene._holdout_indices is None``), the holdout
decision is a pure function of ``(ordinal, eval_hold)`` with no guard band
and no pose-gap filtering (those are precomputed-trajectory-only features --
see ``StreamingScene._designate_test_holdouts``'s docstring). That branch
reads no ``self`` state beyond ``self.args.streaming_eval_hold``, so it lifts
out verbatim with no behavior change.

``scene/streaming_scene.py``'s live branch of ``holdout_status`` now calls
:func:`online_every_nth_holdout` here instead of inlining the arithmetic, and
its ``eval_metadata()`` live branch calls :func:`online_holdout_rule_hash`
here instead of inlining the hash payload -- both refactors are covered by
the pre-existing ``tests/test_issue745_live_holdout_rule_hash.py``, which
pins the exact rule/hash this module must keep reproducing byte-for-byte.

Deliberately torch-free (no ``import torch``, no import of anything that
imports torch) so it is unit-testable in a plain CPython environment,
independent of this project's ROCm/CUDA training containers -- the same
"torch-free module" contract ``Dockerfile.ci-cpu`` already documents and
enforces for a growing set of modules in this codebase.
"""

from __future__ import annotations

import hashlib
from typing import Optional


def online_every_nth_holdout(ordinal: int, eval_hold: int) -> bool:
    """True iff *ordinal* is a designated holdout frame under the online
    every-Nth rule (``eval_hold <= 0`` disables holdout entirely).

    This is the exact predicate ``StreamingScene.holdout_status`` applies on
    its "live/lazy source" branch, keyed to a drop-invariant ordinal (the
    source's own ``source_ordinal`` when it provides one, else the frame's
    ``index`` -- see that method's docstring for why). A frame this returns
    True for must never be trained, replayed, or depth-inserted -- the same
    contract ``StreamingScene`` already documents for its own holdout
    frames -- it is reserved for evaluation only.
    """
    eval_hold = int(eval_hold or 0)
    return eval_hold > 0 and int(ordinal) % eval_hold == 0


def online_holdout_rule_payload(
    *,
    eval_hold: int,
    ordinal_definition_version: str,
    eval_guard: int,
    eval_min_baseline_m: float,
    eval_min_angle_deg: float,
    source_id: str,
) -> str:
    """The exact string ``StreamingScene.eval_metadata()``'s live branch
    hashes into ``holdout_hash``. A byte-for-byte match matters: two runs
    (one through ``StreamingScene``, one through this module) that used the
    identical rule must produce the identical hash, or the Run Contracts'
    ``holdout_hash`` comparability ladder cannot tell them apart from two
    runs that used a genuinely different rule.
    """
    return "|".join(
        [
            f"eval_hold={int(eval_hold)}",
            f"ordinal={ordinal_definition_version}",
            f"guard={int(eval_guard)}",
            f"min_t={float(eval_min_baseline_m):.6f}",
            f"min_a={float(eval_min_angle_deg):.6f}",
            f"source={source_id}",
        ]
    )


def online_holdout_rule_hash(
    *,
    eval_hold: int,
    ordinal_definition_version: str,
    eval_guard: int = 0,
    eval_min_baseline_m: float = 0.03,
    eval_min_angle_deg: float = 2.0,
    source_id: str = "",
) -> Optional[str]:
    """``None`` when holdout is disabled (``eval_hold <= 0``), else the
    16-hex-char SHA-1 prefix of :func:`online_holdout_rule_payload` -- the
    exact value ``StreamingScene.eval_metadata()`` writes as
    ``holdout_hash`` for a live/lazy source."""
    eval_hold = int(eval_hold or 0)
    if eval_hold <= 0:
        return None
    payload = online_holdout_rule_payload(
        eval_hold=eval_hold,
        ordinal_definition_version=ordinal_definition_version,
        eval_guard=eval_guard,
        eval_min_baseline_m=eval_min_baseline_m,
        eval_min_angle_deg=eval_min_angle_deg,
        source_id=source_id,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def online_holdout_eval_metadata(
    *,
    eval_hold: int,
    n_holdout: int,
    ordinal_definition_version: str = "ts_quantized_v1",
    eval_guard: int = 0,
    eval_min_baseline_m: float = 0.03,
    eval_min_angle_deg: float = 2.0,
    source_id: str = "",
    n_guard: int = 0,
) -> dict:
    """Build a ``report.json["eval_metadata"]``-shaped dict for a run driven
    by :class:`LiveHoldoutCollector` rather than a
    :class:`~scene.streaming_scene.StreamingScene`.

    Schema-identical to ``StreamingScene.eval_metadata()``'s live-branch
    return value (same keys, same ``holdout_selection`` strings, same
    ``holdout_hash`` derivation) so this project's existing comparability
    tooling (``holdout_hash``/``realized_holdout_hash`` checks in
    ``utils/baseline_comparison.py``) treats a report built from this
    collector exactly like a CLI streaming run's live-branch report.
    """
    eval_hold = int(eval_hold or 0)
    holdout_hash = online_holdout_rule_hash(
        eval_hold=eval_hold,
        ordinal_definition_version=ordinal_definition_version,
        eval_guard=eval_guard,
        eval_min_baseline_m=eval_min_baseline_m,
        eval_min_angle_deg=eval_min_angle_deg,
        source_id=source_id,
    )
    selection = "online_every_nth_ordinal_rule" if eval_hold > 0 else "online_every_nth"
    return {
        "eval_hold": eval_hold,
        "eval_min_baseline_m": float(eval_min_baseline_m),
        "eval_min_angle_deg": float(eval_min_angle_deg),
        "eval_guard": int(eval_guard),
        "n_holdout": int(n_holdout),
        "n_guard": int(n_guard),
        "holdout_hash": holdout_hash,
        "holdout_selection": selection,
    }


class LiveHoldoutCollector:
    """Framework-agnostic "online every-Nth" holdout reservation for a live
    ingestion path that has no :class:`~scene.streaming_scene.StreamingScene`
    -- e.g. ``eligsir-ros2``'s lifecycle node, which drives
    :class:`~eligsir.runtime.production_application.CanonicalMappingRuntime`
    directly from ROS2 messages with no ``StreamingScene`` in the loop at
    all.

    This class holds no Camera/tensor state and performs no I/O -- it is
    pure ordinal bookkeeping, deliberately torch-free so it is unit-testable
    without a GPU/torch environment. The caller (e.g. the ROS2 node's RGBD
    callback) owns frame/Camera construction; this class only answers two
    questions per admitted frame, in ordinal order:

    1. :meth:`classify` -- is this ordinal a designated holdout frame under
       the rule? If True, the caller MUST withhold that frame from training
       (never submit it to the mapping application), mirroring
       ``StreamingScene.holdout_status``'s own contract.
    2. :meth:`should_capture` -- is there still room in the (optional)
       capture budget to actually materialize and retain this holdout
       frame's GT color/depth/pose for scoring? A holdout frame that is
       *designated* but arrives after the cap is still withheld from
       training (never silently re-admitted) -- it simply isn't scored,
       and that shortfall is counted (:meth:`counts`), not hidden.

    Memory-boundedness is deliberately the caller's decision, not baked into
    the rule: a long bag replay can have thousands of ``eval_hold`` multiples,
    and retaining full-resolution GT color/depth for every one of them for
    the run's whole lifetime is a real memory cost this class refuses to
    make invisible. Pass ``max_holdout_frames=None`` only when the caller has
    already reasoned about that cost for its scene.
    """

    def __init__(self, eval_hold: int, max_holdout_frames: Optional[int] = None) -> None:
        self.eval_hold = int(eval_hold or 0)
        if max_holdout_frames is not None and int(max_holdout_frames) < 0:
            raise ValueError("max_holdout_frames must be >= 0 or None")
        self.max_holdout_frames = (
            None if max_holdout_frames is None else int(max_holdout_frames)
        )
        self._n_designated = 0
        self._n_captured = 0
        self._n_dropped_over_cap = 0

    @property
    def enabled(self) -> bool:
        """False iff ``eval_hold <= 0`` -- holdout reservation is off and
        every frame is a normal training frame (byte-identical to today's
        behavior on any launch that never sets ``eval_hold``)."""
        return self.eval_hold > 0

    def classify(self, ordinal: int) -> bool:
        """True iff *ordinal* is a designated holdout frame. Call exactly
        once per admitted frame, in ordinal order; every ``True`` return
        MUST be withheld from training by the caller."""
        is_holdout = online_every_nth_holdout(ordinal, self.eval_hold)
        if is_holdout:
            self._n_designated += 1
        return is_holdout

    def should_capture(self) -> bool:
        """Whether the caller should materialize+retain this holdout
        frame's GT for eval. Always True with no cap; False once
        ``max_holdout_frames`` captures have already happened. Does not
        itself mutate any counter -- call :meth:`record_captured` or
        :meth:`record_dropped_over_cap` once the caller has acted on the
        answer."""
        if self.max_holdout_frames is None:
            return True
        return self._n_captured < self.max_holdout_frames

    def record_captured(self) -> None:
        """Call after actually retaining a holdout frame's GT."""
        self._n_captured += 1

    def record_dropped_over_cap(self) -> None:
        """Call when a designated holdout frame was withheld from training
        but NOT captured (capture budget already spent)."""
        self._n_dropped_over_cap += 1

    def counts(self) -> dict:
        """Snapshot of this collector's bookkeeping, safe to fold into
        ``report.json`` alongside :func:`online_holdout_eval_metadata`."""
        return {
            "n_holdout_designated": self._n_designated,
            "n_holdout_captured": self._n_captured,
            "n_holdout_dropped_over_cap": self._n_dropped_over_cap,
        }

    def eval_metadata(
        self,
        *,
        ordinal_definition_version: str = "ts_quantized_v1",
        eval_guard: int = 0,
        eval_min_baseline_m: float = 0.03,
        eval_min_angle_deg: float = 2.0,
        source_id: str = "",
    ) -> dict:
        """Convenience wrapper: :func:`online_holdout_eval_metadata` using
        this collector's own ``eval_hold`` and realized ``n_holdout``
        (the CAPTURED count -- what actually became a scoreable test
        camera, matching ``StreamingScene.eval_metadata()``'s ``n_holdout``
        semantics, which counts realized ``_test_cameras``, not merely
        designated ones)."""
        return online_holdout_eval_metadata(
            eval_hold=self.eval_hold,
            n_holdout=self._n_captured,
            ordinal_definition_version=ordinal_definition_version,
            eval_guard=eval_guard,
            eval_min_baseline_m=eval_min_baseline_m,
            eval_min_angle_deg=eval_min_angle_deg,
            source_id=source_id,
            n_guard=0,
        )


__all__ = [
    "online_every_nth_holdout",
    "online_holdout_rule_payload",
    "online_holdout_rule_hash",
    "online_holdout_eval_metadata",
    "LiveHoldoutCollector",
]
