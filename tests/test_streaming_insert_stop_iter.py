"""Tests for _resolve_insertion_stop_iter (W3-66 2x2 experiment, 2026-08-10):
combines the online_mode-coupled insertion-stop horizon with the standalone
--streaming_insert_stop_iter flag, which works without --online_mode.
"""
from __future__ import annotations

from types import SimpleNamespace

from train_streaming import _resolve_insertion_stop_iter


def test_neither_set_returns_none():
    opt = SimpleNamespace()
    args = SimpleNamespace(streaming_insert_stop_iter=0)
    assert _resolve_insertion_stop_iter(opt, args) is None


def test_only_manual_flag_set():
    opt = SimpleNamespace()
    args = SimpleNamespace(streaming_insert_stop_iter=8000)
    assert _resolve_insertion_stop_iter(opt, args) == 8000


def test_only_online_mode_horizon_set():
    opt = SimpleNamespace(_streaming_insertion_stop_iter=5000)
    args = SimpleNamespace(streaming_insert_stop_iter=0)
    assert _resolve_insertion_stop_iter(opt, args) == 5000


def test_both_set_takes_more_restrictive_manual():
    opt = SimpleNamespace(_streaming_insertion_stop_iter=10000)
    args = SimpleNamespace(streaming_insert_stop_iter=5000)
    assert _resolve_insertion_stop_iter(opt, args) == 5000


def test_both_set_takes_more_restrictive_online():
    opt = SimpleNamespace(_streaming_insertion_stop_iter=3000)
    args = SimpleNamespace(streaming_insert_stop_iter=9000)
    assert _resolve_insertion_stop_iter(opt, args) == 3000


def test_missing_args_attr_defaults_to_off():
    opt = SimpleNamespace()
    args = SimpleNamespace()
    assert _resolve_insertion_stop_iter(opt, args) is None
