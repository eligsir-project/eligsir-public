from __future__ import annotations

from eligsir.mapping.scheduler import FrameQuality
from eligsir.mapping.scheduler import KeyframeScoringConfig, score_keyframe


def test_keyframe_score_forces_max_gap():
    score = score_keyframe(
        translation_m=0.0,
        rotation_deg=0.0,
        gap=11,
        overlap=0.95,
        config=KeyframeScoringConfig(max_gap=10),
    )
    assert score.admitted
    assert score.reason == "max_gap"


def test_utils_keyframe_scoring_reexports_package_scoring():
    from eligsir.utils.keyframe_scoring import score_keyframe as compat_score_keyframe

    assert compat_score_keyframe is score_keyframe


def test_keyframe_score_rejects_bad_quality():
    quality = FrameQuality(0, 0, 0, 0, 1, 0, 0, quality_weight=0.1, reject_reason="blur")
    score = score_keyframe(
        translation_m=1.0,
        rotation_deg=20.0,
        gap=1,
        overlap=0.5,
        quality=quality,
    )
    assert not score.admitted
    assert score.reason == "blur"
