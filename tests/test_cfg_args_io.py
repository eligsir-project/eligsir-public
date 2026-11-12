from __future__ import annotations

import argparse
import re

import pytest

from eligsir.utils.cfg_args_io import write_cfg_args


def test_writes_eval_safe_cfg_args(tmp_path):
    args = argparse.Namespace(source_path="/data/x", cap_max=5000, eval=True)
    out = tmp_path / "cfg_args"
    write_cfg_args(str(out), args)
    text = out.read_text()
    parsed = eval(text, {"Namespace": argparse.Namespace})
    assert parsed.source_path == "/data/x"
    assert parsed.cap_max == 5000


def test_excludes_private_and_known_bad_keys(tmp_path):
    args = argparse.Namespace(cap_max=5000, _internal=object(), pose_provenance_result=object())
    out = tmp_path / "cfg_args"
    write_cfg_args(str(out), args)
    text = out.read_text()
    assert "_internal" not in text
    assert "pose_provenance_result" not in text
    eval(text, {"Namespace": argparse.Namespace})  # must not raise


def test_raises_loudly_on_non_literal_repr_attribute(tmp_path):
    class _Enumish:
        def __repr__(self):
            return "<Something.WEIRD: 'weird'>"

    args = argparse.Namespace(cap_max=5000, some_future_field=_Enumish())
    out = tmp_path / "cfg_args"
    with pytest.raises(RuntimeError, match="not be eval\\(\\)-safe"):
        write_cfg_args(str(out), args)
    assert not out.exists()
