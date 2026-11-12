"""Issue #1302: a ``report.json`` for a live/ROS2 ingestion run.

``deterministic_bag.launch.py``'s ROS2 lifecycle-node mapping path
(``eligsir-ros2``) drives
:class:`~eligsir.runtime.production_application.CanonicalMappingRuntime`
directly from ROS2 messages -- there is no
:class:`~scene.streaming_scene.StreamingScene`, no persistent train-camera
list, and (before this module) no holdout reservation and no
``report.json``/PSNR at all. See
``docs/architecture/2026-08-31-issue1302-ros2-eval-wiring.md`` for the full
investigation this module is one half of.

This module supplies the report-writing half. Holdout SELECTION (which
frames are withheld from training and offered to eval) is
``utils.holdout_rule.LiveHoldoutCollector`` -- deliberately a separate,
torch-free module so the selection rule is unit-testable without a
GPU/torch environment. This module is the thin, torch-*requiring* other
half: given the small number of already-captured holdout ``Camera`` objects
plus the live mapping runtime's own model/pipe/background, call the exact
same ``utils.comparison_report.write_post_training_report`` every CLI
streaming/offline run calls, so a resulting report.json is genuinely
comparable (same schema, same ``eval_valid``/``holdout_hash`` machinery, same
downstream tooling) rather than a parallel, weaker report format.

Deliberately does NOT import ``utils.comparison_report`` (and therefore does
not require torch) at module import time -- only inside
:func:`write_live_eval_report`'s body, matching the "lazy in-function
imports" pattern this codebase already documents for other legitimately
GPU-only call paths. Importing this module (e.g. from ``eligsir-ros2``'s
lifecycle node, which does need to reference
:class:`~eligsir.runtime.live_eval_holdout` regardless of whether an eval
report is ever produced) never pulls in torch.

This module also supplies :func:`build_holdout_camera_from_packet` -- the
``packet -> Camera`` conversion a live caller needs to actually capture a
withheld holdout frame's GT before calling :func:`write_live_eval_report`.
It is a thin, deliberately non-duplicating wrapper around the exact
conversion :class:`~eligsir.mapping.canonical_scene.CanonicalStreamingScene`
already uses for every OTHER admitted frame
(``canonical_camera_from_observation``, fed by
``materialize_mapping_frame``/``StreamingObservation.from_mapping_frame`` --
see ``eligsir/mapping/canonical_bootstrap.py`` and
``eligsir/runtime/streaming_observation.py``), not a second, hand-rolled
one. ``eligsir-ros2`` cannot import ``torch``/``scene.cameras`` itself
(a machine-enforced boundary gate token-scans its whole source tree for the
literal string ``torch``, including inside a lazy function-local import --
see ``tests/test_boundary_invariants.py::test_ros_repository_contains_no_mapper_or_optimizer_loop``
in that repository); calling this *core* function by name does not require
the caller to write ``import torch`` anywhere in its own source, since the
import happens inside this module instead. Like the rest of this module,
importing it does not pull in torch -- only calling
:func:`build_holdout_camera_from_packet` does.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional


def build_holdout_camera_from_packet(
    packet: Any,
    *,
    data_device: Any,
    materialization_policy: Optional[Any] = None,
    materialization_backend: Optional[Any] = None,
) -> Any:
    """Build one GT ``scene.cameras.Camera`` from a raw ``SensorPacket``.

    This is the exact ``packet -> Camera`` conversion the canonical mapping
    runtime uses for every admitted frame -- reused here, not reimplemented,
    for a frame a caller has decided to withhold from admission instead (see
    ``utils.holdout_rule.LiveHoldoutCollector``). Concretely:
    ``materialize_mapping_frame`` (``eligsir.contracts.materialize``) ->
    ``StreamingObservation.from_mapping_frame`` ->
    ``canonical_camera_from_observation`` (``eligsir.mapping.canonical_bootstrap``,
    the same function :class:`~eligsir.mapping.canonical_scene.CanonicalStreamingScene`
    calls internally for its own admitted cameras).

    Args:
        packet: A ``eligsir.contracts.runtime.SensorPacket`` -- the exact
            object a caller already has in hand immediately before it would
            otherwise call ``MappingSession.submit(packet)``. Must carry a
            resolved ``packet.tracking.pose`` (i.e. called after this
            packet's tracking-attach step, before admission).
        data_device: Target device for the returned ``Camera`` (e.g.
            ``"cuda"``, ``"cpu"``, or a ``torch.device``) -- normally the
            same device the live mapping runtime's own Gaussian model lives
            on, so the returned camera can be rendered against it directly.
        materialization_policy: Optional ``eligsir.contracts.materialize.MaterializationPolicy``.
            Defaults to ``MaterializationPolicy(depth=DepthPolicy.METERS_FLOAT32)``
            -- the exact policy the production runtime binds via
            ``eligsir.runtime.runtime_binding.runtime_binding_from_runtime``
            for every other admitted frame. Pass
            ``application.materialization_policy``
            (``eligsir.runtime.application.ComposedMappingApplication``)
            for byte-identical materialization behavior instead of relying
            on this default staying in sync with production.
        materialization_backend: Optional ``TensorBackend``. Defaults to the
            CPU ``numpy_backend()``, matching production unless GPU-native
            depth decoding is enabled for this run (pass
            ``application.materialization_backend`` to match exactly).

    Returns:
        A ``scene.cameras.Camera`` carrying GT color (``.original_image``),
        GT metric depth (``._sensor_depth_cache``, when the packet has
        depth), and pose -- exactly the shape ``write_live_eval_report``'s
        ``test_cams`` expects.

    Raises:
        ValueError: if ``packet`` has no resolved tracking pose, or the
            materialized frame otherwise carries no pose. Callers should
            only invoke this once a packet's tracking-attach step has
            actually resolved a pose; a still-unresolved packet is not a
            capture-time bug, it's just not ready yet.
    """
    tracking = getattr(packet, "tracking", None)
    if tracking is None or tracking.pose is None:
        raise ValueError(
            "build_holdout_camera_from_packet requires packet.tracking.pose "
            "to already be resolved -- call this after this packet's "
            "tracking-attach step, immediately before it would otherwise be "
            "handed to MappingSession.submit()"
        )

    from eligsir.contracts.materialize import (
        DepthPolicy,
        MaterializationPolicy,
        materialize_mapping_frame,
        numpy_backend,
    )
    from eligsir.mapping.canonical_bootstrap import canonical_camera_from_observation
    from eligsir.runtime.streaming_observation import StreamingObservation

    policy = materialization_policy or MaterializationPolicy(
        depth=DepthPolicy.METERS_FLOAT32
    )
    backend = materialization_backend or numpy_backend()
    frame = materialize_mapping_frame(packet, policy, backend=backend)
    if frame.pose is None:
        raise ValueError(
            "materialized MappingFrame carries no pose; cannot build a "
            "holdout camera without one"
        )
    observation = StreamingObservation.from_mapping_frame(frame)
    return canonical_camera_from_observation(observation, data_device=data_device)


def write_live_eval_report(
    *,
    model_path: str,
    iteration: int,
    gaussians: Any,
    pipe: Any,
    background: Any,
    render_fn: Callable,
    test_cams: List[Any],
    eval_metadata: dict,
    source_attestation: Optional[dict] = None,
    train_cams: Optional[List[Any]] = None,
    report_writer: Optional[Callable] = None,
    **extra_kwargs: Any,
) -> dict:
    """Write a ``report.json`` for a live/ROS2 ingestion run with no
    :class:`~scene.streaming_scene.StreamingScene`.

    Calls ``utils.comparison_report.write_post_training_report`` with
    ``train_cams=[]`` by default -- a legitimate, already-supported shape
    (that function guards every ``train_cams`` use with
    ``if train_cams:``/``train_cams or []``; ``n_train`` simply reads 0).
    This is NOT a degenerate split: ``test_cams`` here are real cameras this
    run genuinely never trained on (``utils.holdout_rule.LiveHoldoutCollector``
    withheld them from the mapping application before this call), so
    ``eval_valid``/``mean_test_psnr`` carry the same standing as any CLI
    streaming run's -- for exactly this run's realized holdout set, no more
    and no less.

    Args:
        model_path: Run output directory (the lifecycle node's
            ``mapping_model_path``).
        iteration: The mapping application's current optimizer step count at
            report time (``CanonicalMappingRuntime``'s own iteration
            counter).
        gaussians: The live model (``CanonicalMappingRuntime.gaussians``).
        pipe: Render pipeline params (``CanonicalMappingRuntime.pipe``).
        background: Render background tensor, as every other report caller
            passes it.
        render_fn: The project's standard ``render()`` callable.
        test_cams: Captured holdout ``Camera`` objects (real GT color/depth +
            pose), from whatever collection mechanism the caller used
            alongside ``utils.holdout_rule.LiveHoldoutCollector``.
        eval_metadata: Pre-built via
            ``utils.holdout_rule.LiveHoldoutCollector.eval_metadata()`` (or
            ``utils.holdout_rule.online_holdout_eval_metadata`` directly) so
            ``holdout_hash``/``holdout_selection`` are schema-identical to a
            CLI streaming run's live-branch report.
        source_attestation: Pass through PR #1315's
            ``utils.source_attestation`` dict when this run's bag went
            through ``LegacyBagProvenanceAttestor`` (or the validated
            hardware-sync path), so ``evidence_class`` is computed correctly
            for this report exactly as it is for the CLI path.
        train_cams: Normally omitted (``[]``) -- pass a real list only if the
            caller has one (e.g. it separately retained a bounded sample of
            trained cameras for train-side metrics); most ROS2 lifecycle-node
            callers will not have this.
        report_writer: Injectable for tests -- defaults to the real
            ``utils.comparison_report.write_post_training_report``. Passing
            a stub lets a test verify the kwargs this function builds without
            needing torch/a real render.
        **extra_kwargs: Forwarded verbatim to ``write_post_training_report``
            (e.g. ``subdir``, ``degenerate_split`` -- though the latter
            should almost never be True here, see above).

    Raises:
        ValueError: if ``test_cams`` is empty. A report with zero captured
            holdout cameras is not a smaller eval, it is not an eval at
            all -- callers must check
            ``LiveHoldoutCollector.counts()["n_holdout_captured"] > 0``
            themselves before calling this, so that decision is visible at
            the call site rather than buried in a report that silently
            claims ``n_test=0``.
    """
    if not test_cams:
        raise ValueError(
            "write_live_eval_report requires at least one captured holdout "
            "camera in test_cams. Call sites must check "
            "LiveHoldoutCollector.counts()['n_holdout_captured'] > 0 before "
            "calling this -- a report with zero test cameras is not a "
            "smaller eval, it is not an eval."
        )

    if report_writer is None:
        from eligsir.utils.comparison_report import write_post_training_report

        report_writer = write_post_training_report

    return report_writer(
        model_path,
        int(iteration),
        gaussians,
        list(train_cams) if train_cams else [],
        list(test_cams),
        render_fn,
        pipe,
        background,
        eval_metadata=eval_metadata,
        source_attestation=source_attestation,
        **extra_kwargs,
    )


__all__ = ["build_holdout_camera_from_packet", "write_live_eval_report"]
