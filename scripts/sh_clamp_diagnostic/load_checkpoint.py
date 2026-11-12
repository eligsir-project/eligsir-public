import os
from argparse import ArgumentParser, Namespace  # noqa: F401 -- Namespace is used by eval() on cfg_args below

from arguments import ModelParams, PipelineParams
from scene import Scene
from scene.gsplat_model import GsplatGaussianModel
from scene.gaussian_model import GaussianModel


def load_checkpoint(model_path: str, model_layout: str = "gsplat"):
    """Load a completed run's Scene + Gaussian model at its latest saved
    iteration, read-only. No training-loop state (optimizer, schedule) is
    reconstructed -- this is for rendering/inspection only.

    Callers that only need to run forward passes (no gradients) should wrap
    usage in a ``torch.no_grad()`` context themselves; this function does not
    do so, so later phases (e.g. Phase 1/2) can also reuse it under whatever
    grad context they need.
    """
    model_layout = model_layout.lower()
    if model_layout not in {"gsplat", "legacy"}:
        raise ValueError(f"Unsupported model_layout '{model_layout}'. Expected 'gsplat' or 'legacy'.")

    # ModelParams/PipelineParams are argparse ParamGroups (see arguments/__init__.py):
    # they register their defaults onto a parser and must be `.extract(args)`-ed
    # into plain GroupParams objects afterward (this mirrors render.py's usage).
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    args = parser.parse_args([])  # defaults only; we override via cfg_args below

    args.model_path = model_path
    args.sh_degree = 3

    # A completed run's cfg_args records the exact ModelParams it was trained
    # with (data_device, resolution, init settings, etc.) -- load it the same
    # way render.py's get_combined_args does, so the checkpoint's Gaussians
    # line up with the Scene/camera setup they were actually trained under.
    cfg_args_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_args_path):
        with open(cfg_args_path) as f:
            cfg_namespace = eval(f.read())
        for key, value in vars(cfg_namespace).items():
            setattr(args, key, value)
    args.model_path = model_path  # cfg_args may record a stale/relative path

    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    model_cls = GsplatGaussianModel if model_layout == "gsplat" else GaussianModel
    gaussians = model_cls(dataset.sh_degree)

    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    return gaussians, scene, pipe
