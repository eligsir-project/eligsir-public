# tests/test_sparse_mode_detach_sh_dir_wiring.py
from argparse import ArgumentParser

from arguments import (
    OptimizationParams,
    PipelineParams,
    apply_renderer_optimization_params,
)


def _make_opt(sparse_mode_detach_sh_dir):
    parser = ArgumentParser()
    op = OptimizationParams(parser)
    args = parser.parse_args([])
    args.sparse_mode_detach_sh_dir = sparse_mode_detach_sh_dir
    return op.extract(args)


def _make_pipe():
    parser = ArgumentParser()
    pp = PipelineParams(parser)
    args = parser.parse_args([])
    return pp.extract(args)


def test_wiring_carries_true_value_onto_pipe():
    opt = _make_opt(True)
    pipe = apply_renderer_optimization_params(_make_pipe(), opt)
    assert pipe.sparse_mode_detach_sh_dir is True


def test_wiring_carries_false_value_onto_pipe():
    opt = _make_opt(False)
    pipe = apply_renderer_optimization_params(_make_pipe(), opt)
    assert pipe.sparse_mode_detach_sh_dir is False
