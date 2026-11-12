"""Unit tests for the ``iphone_lidar`` camera profile's admission-tuning
defaults (added after discovering ScanNet++'s ~60Hz capture rate mismatches
the keyframe admitter's 30fps default, and that ``max_gap=10`` is the binding
admission constraint on ScanNet++ ablation runs -- see the inline comments in
``train_streaming.py``'s ``_CAMERA_PROFILES["iphone_lidar"]`` entry).

CPU-only: builds a minimal argparse.Namespace standing in for the real
``ModelParams``/``StreamingParams`` args object, exercises
``_apply_camera_profile`` directly, and checks (a) the new fields are applied
with the expected values, (b) a CLI-set value is never clobbered, and (c) an
unrelated profile (e.g. ``orbbec_femto_bolt``) is untouched by this change.
"""
from __future__ import annotations

import argparse
import sys

import train_streaming as ts


def _base_args(**overrides):
    ns = argparse.Namespace()
    for key, default in ts._STREAMING_PARAM_DEFAULTS.items():
        setattr(ns, key, default)
    # Non-profile-driven defaults exercised by the admission-tuning fields.
    # streaming_keyframe_max_gap_require_novelty is deliberately NOT hardcoded
    # here -- it comes from ts._STREAMING_PARAM_DEFAULTS via the loop above so
    # this fixture can't silently drift from the real factory default (see
    # issue #792/#794: that default flipped False->True on 2026-08-15).
    ns.streaming_keyframe_input_fps = 30.0
    ns.streaming_keyframe_max_gap = 10
    ns.streaming_camera_profile = "default"
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_iphone_lidar_sets_60fps_and_novelty_gated_max_gap(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_streaming.py"])
    args = _base_args(streaming_camera_profile="iphone_lidar")
    ts._apply_camera_profile(args)
    assert args.streaming_keyframe_input_fps == 60.0
    assert args.streaming_keyframe_max_gap == 20
    assert args.streaming_keyframe_max_gap_require_novelty is True


def test_iphone_lidar_does_not_clobber_cli_override(monkeypatch):
    # Pretend the user passed --streaming_keyframe_input_fps explicitly.
    monkeypatch.setattr(
        sys, "argv", ["train_streaming.py", "--streaming_keyframe_input_fps=24.0"]
    )
    args = _base_args(
        streaming_camera_profile="iphone_lidar",
        streaming_keyframe_input_fps=24.0,
    )
    ts._apply_camera_profile(args)
    assert args.streaming_keyframe_input_fps == 24.0  # untouched


def test_orbbec_profile_unaffected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_streaming.py"])
    args = _base_args(streaming_camera_profile="orbbec_femto_bolt")
    ts._apply_camera_profile(args)
    # orbbec profile never set these keys -> stay at whatever the caller had
    # (the real factory default, via ts._STREAMING_PARAM_DEFAULTS -- True
    # since issue #792/#794's 2026-08-15 default flip).
    assert args.streaming_keyframe_input_fps == 30.0
    assert args.streaming_keyframe_max_gap == 10
    assert args.streaming_keyframe_max_gap_require_novelty is True
