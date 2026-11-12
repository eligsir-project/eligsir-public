from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONSUMERS = (
    "eligsir/mapping/integration/device_candidates.py",
    "eligsir/mapping/integration/device_integration_planner.py",
    "eligsir/mapping/resolution_cache.py",
)


def test_transfer_consumers_do_not_create_private_streams_or_pools():
    violations = []
    forbidden_constructors = {
        "DeviceTransferStream",
        "PinnedByteSlabPool",
        "PinnedTensorPool",
    }
    for relative in CONSUMERS:
        source = (ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name in forbidden_constructors:
                violations.append(f"{relative}:{node.lineno}:{name}")
        for name in ("_TRANSFERS", "_POOLS", "_PACK_POOLS"):
            if name in source:
                violations.append(f"{relative}:private registry {name}")
    assert violations == []


def test_live_pixel_producers_preserve_cpu_ownership_until_page_in():
    cameras = (ROOT / "scene/cameras.py").read_text(encoding="utf-8")
    decoder = (ROOT / "utils/camera_utils.py").read_text(encoding="utf-8")

    assert "image.clamp(0.0, 1.0).to(self.data_device)" not in cameras
    assert 'materialized.to("cpu")' not in cameras
    assert "torch.from_numpy(rgb_uint8).to(" not in decoder


def test_canonical_device_insertion_has_no_tensor_to_host_predicate():
    planner = (ROOT / CONSUMERS[1]).read_text(encoding="utf-8")
    producer = (
        ROOT / "eligsir/mapping/integration/device_candidate_producer.py"
    ).read_text(encoding="utf-8")

    assert "valid.any().item()" not in planner
    assert "bool((confidence > 0).any())" not in producer
    assert "bool(too_close.any())" not in producer
