from pathlib import Path


def test_engine_route_binds_step_outputs_before_the_shared_topology_and_logging_tail():
    source = Path("train_streaming.py").read_text()
    optimizer_owner = source[
        source.index("        def _optimizer_tick():") :
        source.index("        _mapping_application = _compose_mapping_application(")
    ]
    branch_start = source.rfind(
        "        if _streaming_optimization_runtime is not None:",
        0,
        source.index("            _deferred_stale_mask = None"),
    )
    engine_branch = source[branch_start:source.index("        else:", branch_start)]

    assert "xyz_lr = _engine_result.xyz_lr" in engine_branch
    assert "_depth_loss_val = _engine_components.get(\"depth\")" in engine_branch
    assert "_free_loss_val = _engine_components.get(\"free_space\")" in engine_branch
    assert "_anchor_loss_val = _engine_components.get(\"anchor\")" in engine_branch
    assert "_mature_anchor_val = _engine_components.get(\"mature_anchor\")" in engine_branch
    assert "_soft_anchor_val = _engine_components.get(\"soft_anchor\")" in engine_branch
    assert "run_streaming_optimization_step(" in optimizer_owner
    assert "random_state_token=RandomStateToken.capture()" in optimizer_owner
    assert "_engine_result = _mapping_application.tick()" in engine_branch
    assert "_backward_ran = _engine_result.backward_ran" in engine_branch
    assert "_sanitize_gaussian_params(gaussians, \"after_mcmc_noise\", iteration)" not in engine_branch

    shared_tail = source[source.index("        # ---- Utility for energy MCMC") :]
    assert "lr=xyz_lr" in shared_tail
    assert "optimization/depth_loss" in shared_tail
    # The MCMC-noise gate is shared by both arms (engine and inline). #105
    # (evidence-cadence phase 0) added a third clause, `_mcmc_offset_suppressed`,
    # to this same condition -- a real, arm-independent change, not engine
    # drift. Assert on the two clauses this test actually cares about (the
    # backward-ran / placement-only gating that the engine route must feed)
    # rather than on the literal full condition text, so an unrelated new
    # guard clause doesn't fail this wiring test again.
    assert "if _backward_ran and not _skip_mcmc" in shared_tail


def test_legacy_streaming_composition_injects_the_canonical_transfer_owner():
    source = Path("train_streaming.py").read_text()
    construction = source.split(
        "_streaming_optimization_runtime = StreamingOptimizationRuntime(", 1
    )[1].split("        def _optimizer_tick():", 1)[0]

    assert "camera_transfer_owner=get_device_transfer_owner(" in construction


def test_inline_seed_freeze_transaction_uses_canonical_map_backend_lock():
    source = Path("train_streaming.py").read_text()
    inline = source.split(
        "from eligsir.utils.seed_lineage_freeze import seed_lineage_optimizer_guard", 1
    )[1].split("# PR2 cadence clocks", 1)[0]

    assert "backend=map_backend" in inline
