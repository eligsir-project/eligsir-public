"""Wave 3, PR W3-12: graph-capture signature (plan section 17.5).

Pure-Python, CPU-only (no torch dependency at all) -- deliverable 4
("capture invalidation rules are documented and tested") at the signature
level: any single field change must produce inequality (the trigger
``TorchAcceleratorGraphRuntime.run_step`` uses to force a recapture, see
``tests/test_w3_12_graph_runtime_fallback_safety.py`` for the
runtime-level exercise of that trigger).
"""

from __future__ import annotations

import dataclasses

from eligsir.optimization.gaussian.graph_runtime.signature import GraphSignature


def _base_kwargs() -> dict:
    return dict(
        device_backend="rocm:gfx1151",
        capacity_bucket=65536,
        active_shape=(65536,),
        resolution_bucket=(640, 480),
        sh_degree=3,
        renderer_mode="gsplat",
        packed_mode=False,
        sparse_grad_mode=True,
        optimizer_mode="selective_adam",
        loss_policy_id="default_v1",
        gradient_policy_id="default_v1",
    )


def test_identical_signatures_are_equal_and_hash_equal():
    a = GraphSignature(**_base_kwargs())
    b = GraphSignature(**_base_kwargs())
    assert a == b
    assert hash(a) == hash(b)


def test_every_field_change_breaks_equality():
    base = GraphSignature(**_base_kwargs())
    overrides = {
        "device_backend": "cuda:sm90",
        "capacity_bucket": 131072,
        "active_shape": (131072,),
        "resolution_bucket": (1280, 720),
        "sh_degree": 0,
        "renderer_mode": "legacy",
        "packed_mode": True,
        "sparse_grad_mode": False,
        "optimizer_mode": "adam",
        "loss_policy_id": "default_v2",
        "gradient_policy_id": "default_v2",
    }
    for field_name, new_value in overrides.items():
        kwargs = _base_kwargs()
        kwargs[field_name] = new_value
        changed = GraphSignature(**kwargs)
        assert changed != base, f"changing {field_name} did not break equality"


def test_describe_diff_reports_only_changed_fields():
    base = GraphSignature(**_base_kwargs())
    kwargs = _base_kwargs()
    kwargs["capacity_bucket"] = 131072
    kwargs["sh_degree"] = 1
    changed = GraphSignature(**kwargs)

    diff = changed.describe_diff(base)
    assert len(diff) == 2
    assert any(d.startswith("capacity_bucket:") for d in diff)
    assert any(d.startswith("sh_degree:") for d in diff)
    assert not any(d.startswith("resolution_bucket:") for d in diff)


def test_signature_is_frozen():
    sig = GraphSignature(**_base_kwargs())
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        sig.sh_degree = 5


def test_all_seventeen_five_fields_present():
    """Sanity-anchor against plan section 17.5's literal field list so a
    future edit that silently drops a field is caught."""
    expected = {
        "device_backend",
        "capacity_bucket",
        "active_shape",
        "resolution_bucket",
        "sh_degree",
        "renderer_mode",
        "packed_mode",
        "sparse_grad_mode",
        "optimizer_mode",
        "loss_policy_id",
        "gradient_policy_id",
    }
    actual = {f.name for f in dataclasses.fields(GraphSignature)}
    assert actual == expected
