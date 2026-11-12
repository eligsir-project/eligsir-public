"""Regression contract for issue #795 per-decision controller telemetry.

The admission fingerprint is aggregate-only.  These fields must survive from
the typed decision through *both* production stats builders and the JSONL
sidecar, including cheap pregate rejections that never call ``admit()``.
"""

import ast
from dataclasses import fields
from pathlib import Path

import numpy as np

from eligsir.mapping.scheduler.keyframes import (
    FrameInfo,
    KeyframeAdmitter,
    KeyframeAdmitterConfig,
    KeyframeDecision,
)


_TRAIN_STREAMING = Path(__file__).parent.parent / "train_streaming.py"
_FINGERPRINT = Path(__file__).parent.parent / "eligsir" / "utils" / "admission_fingerprint.py"

# Velocity-estimator state, decision threshold/cadence, and the ownership and
# provenance required to distinguish controller admission from soft/hard
# starvation relief in the #793 validation run.
_REQUIRED_PER_DECISION_FIELDS = {
    "translation_velocity_m_s",
    "translation_velocity_ema_m_s",
    "rotation_velocity_deg_s",
    "rotation_velocity_ema_deg_s",
    "source_dt_s",
    "source_frame_delta",
    "implied_input_fps",
    "effective_min_translation_m",
    "controller_target_interval_s",
    "controller_target_interval_source_frames",
    "timebase_source",
    "threshold_clamp",
    "motion_trigger",
    "soft_backstop_triggered",
    "hard_backstop_triggered",
    "soft_backstop_domain",
    "hard_backstop_domain",
    "starvation_age_basis",
}


def _string_keys(node: ast.Dict) -> set[str]:
    keys = {
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    # The two production builders conditionally merge
    # ``**{"precedence": decision.trace}``.  It is still an emitted field,
    # even though it is represented in the AST as a ``None`` dict key.
    for key, value in zip(node.keys, node.values):
        if key is None:
            keys |= _conditional_dict_keys(value)
    return keys


def _conditional_dict_keys(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Dict):
        return _string_keys(node)
    if isinstance(node, ast.IfExp):
        return _conditional_dict_keys(node.body) | _conditional_dict_keys(node.orelse)
    return set()


def _shared_sidecar_builder_keys(tree: ast.Module) -> set[str]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "keyframe_decision_sidecar_fields":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Dict):
                return _string_keys(child)
    raise AssertionError("could not locate shared keyframe-decision sidecar builder")


def _sidecar_whitelist(tree: ast.Module) -> set[str]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        if not isinstance(node.target, ast.Name) or node.target.id != "_k":
            continue
        if not isinstance(node.iter, (ast.Tuple, ast.List)):
            continue
        keys = {
            item.value
            for item in node.iter.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        if "motion_trigger" in keys:
            return keys
    raise AssertionError("could not locate keyframe_admission_log.jsonl whitelist")


def test_keyframe_decision_declares_source_frame_target_cadence() -> None:
    """The cadence must be produced canonically, never inferred at logging."""
    assert "controller_target_interval_source_frames" in {
        field.name for field in fields(KeyframeDecision)
    }


def _frame(index: int, x: float) -> FrameInfo:
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = x
    return FrameInfo(c2w=c2w, index=index)


def test_target_source_frame_cadence_is_stamped_on_forced_and_rejected_exits() -> None:
    """The canonical cadence survives either controller decision outcome."""
    for max_gap, x in ((1, 0.0), (100, 0.0)):
        admitter = KeyframeAdmitter(
            config=KeyframeAdmitterConfig(
                velocity_adaptive=True,
                input_fps=30.0,
                target_admit_rate_fps=2.0,
                max_gap=max_gap,
                min_translation_m=1.0,
                min_rotation_deg=1e9,
            )
        )
        admitter.note_train_frame(_frame(0, 0.0))
        decision, _ = admitter.admit(cam=None, frame=_frame(1, x))

        assert decision.controller_target_interval_s == 0.5
        assert decision.controller_target_interval_source_frames == 15.0


def test_every_controller_decision_field_reaches_both_stats_builders_and_sidecar() -> None:
    """No normal or pregate decision may silently lose #793 evidence."""
    tree = ast.parse(_TRAIN_STREAMING.read_text())
    helper_keys = _shared_sidecar_builder_keys(ast.parse(_FINGERPRINT.read_text()))
    missing = sorted(_REQUIRED_PER_DECISION_FIELDS - helper_keys)
    assert not missing, f"shared stats fields drop {missing}"

    closures = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_keyframe_admission_closure", "_keyframe_pregate_closure"}
    }
    assert len(closures) == 2
    for name, builder in zip(
        ("admission", "pregate"),
        (closures["_keyframe_admission_closure"], closures["_keyframe_pregate_closure"]),
    ):
        uses_shared_builder = any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "keyframe_decision_sidecar_fields"
            for child in ast.walk(builder)
        )
        assert uses_shared_builder, f"{name} stats builder bypasses shared fields"

    missing = sorted(_REQUIRED_PER_DECISION_FIELDS - _sidecar_whitelist(tree))
    assert not missing, f"keyframe_admission_log.jsonl drops {missing}"
