"""Fail-closed state envelope for exact streaming resume."""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


SCHEMA = "streaming_checkpoint_v2"
VERSION = 2
_RNG_KEYS = {"python", "numpy", "torch_cpu", "torch_gpu"}
_IDENTITY_KEYS = {
    "source_hash",
    "holdout_hash",
    "resolved_config_hash",
    "runtime_hash",
}


def _clone_state(value: Any) -> Any:
    """Clone the scalar/tensor containers admitted to exact-resume state."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {_clone_state(key): _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    raise ValueError(f"unsupported exact-resume state type: {type(value).__name__}")


def capture_rng_state() -> dict:
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_gpu": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _validate_identity(identity: object) -> dict:
    if not isinstance(identity, dict) or set(identity) != _IDENTITY_KEYS:
        raise ValueError(
            f"resume identity must contain exactly {sorted(_IDENTITY_KEYS)}"
        )
    if any(not isinstance(value, str) or not value.strip() for value in identity.values()):
        raise ValueError("resume identity hashes must be non-empty strings")
    return dict(identity)


def _validate_rng_state(payload: object) -> dict:
    if not isinstance(payload, dict) or set(payload) != _RNG_KEYS:
        raise ValueError("rng state must contain Python, NumPy, Torch CPU, and Torch GPU state")
    python_state = copy.deepcopy(payload["python"])
    numpy_state = copy.deepcopy(payload["numpy"])
    torch_cpu = payload["torch_cpu"]
    gpu_states = payload["torch_gpu"]
    if not isinstance(gpu_states, list):
        raise ValueError("invalid Torch GPU rng state")
    expected_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(gpu_states) != expected_gpu_count:
        raise ValueError(
            "GPU RNG stream count mismatch: "
            f"checkpoint={len(gpu_states)} runtime={expected_gpu_count}"
        )

    try:
        random.Random().setstate(python_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid Python rng state") from exc
    try:
        np.random.RandomState().set_state(numpy_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid NumPy rng state") from exc
    if (
        not isinstance(torch_cpu, torch.Tensor)
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.device.type != "cpu"
    ):
        raise ValueError("invalid Torch CPU rng state")
    try:
        torch.Generator(device="cpu").set_state(torch_cpu.clone())
    except (RuntimeError, TypeError) as exc:
        raise ValueError("invalid Torch CPU rng state") from exc

    validated_gpu = []
    for device_index, state in enumerate(gpu_states):
        if (
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.device.type != "cpu"
        ):
            raise ValueError(f"invalid Torch GPU rng state for device {device_index}")
        try:
            torch.Generator(device=device_index).set_state(state.clone())
        except (RuntimeError, TypeError) as exc:
            raise ValueError(
                f"invalid Torch GPU rng state for device {device_index}"
            ) from exc
        validated_gpu.append(state.clone())

    return {
        "python": python_state,
        "numpy": numpy_state,
        "torch_cpu": torch_cpu.clone(),
        "torch_gpu": validated_gpu,
    }


def restore_rng_state(payload: dict) -> None:
    state = _validate_rng_state(payload)
    random.setstate(copy.deepcopy(state["python"]))
    np.random.set_state(copy.deepcopy(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"].clone())
    if state["torch_gpu"]:
        torch.cuda.set_rng_state_all([item.clone() for item in state["torch_gpu"]])


def build_resume_envelope(identity: dict, owners: dict) -> dict:
    validated_identity = _validate_identity(identity)
    if not isinstance(owners, dict):
        raise ValueError("resume owners must be a dictionary")

    owner_states = {}
    for name, owner in owners.items():
        capture = getattr(owner, "resume_state_dict", None)
        if not isinstance(name, str) or not name or not callable(capture):
            raise ValueError(f"resume owner {name!r} has no compatible state capture")
        owner_states[name] = _clone_state(capture())

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "identity": validated_identity,
        "rng": capture_rng_state(),
        "owners": owner_states,
    }


def validate_resume_envelope(
    payload: dict,
    expected_identity: dict,
    required_owners: set[str],
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("resume payload must be a mapping")
    if payload.get("schema") != SCHEMA:
        raise ValueError("resume schema is not streaming_checkpoint_v2")
    if payload.get("version") != VERSION:
        raise ValueError("resume version is not 2")
    validated_expected_identity = _validate_identity(expected_identity)
    validated_payload_identity = _validate_identity(payload.get("identity"))
    if validated_payload_identity != validated_expected_identity:
        raise ValueError("resume identity mismatch")
    _validate_rng_state(payload.get("rng"))
    owners = payload.get("owners")
    if not isinstance(owners, Mapping):
        raise ValueError("resume owners are missing")
    missing = set(required_owners) - set(owners)
    if missing:
        raise ValueError(f"required resume owner state missing: {sorted(missing)}")
