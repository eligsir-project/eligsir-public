#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import warnings
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from eligsir.utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import (
    ModelParams,
    PipelineParams,
    RENDERER_OPTIMIZATION_DEFAULTS,
    add_renderer_optimization_args,
    apply_renderer_optimization_params,
    get_combined_args,
    get_default_pipeline_params,
)
from scene import GaussianModel, GsplatGaussianModel
from metrics import (
    SCORE_ARTIFACT_DIRNAME,
    SCORE_ARTIFACT_SUFFIX,
    SCORE_PARITY_TOLERANCES,
    save_score_pair,
    write_score_set_manifest,
)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    """Render a camera set, persisting exact score tensors + viewer PNGs.

    Issue #640: the PNGs written here are **visualization artifacts only**.
    ``torchvision.utils.save_image`` quantizes float32 to 8 bits (and silently
    clamps to [0, 1]); scoring those files gave this evaluation path different
    semantics from every other metric path in the project. The exact float32
    renderer output and GT are therefore also persisted, unmodified, under
    ``score_tensors/`` -- that is what ``metrics.py`` reads.
    """
    out_dir = os.path.join(model_path, name, "ours_{}".format(iteration))
    render_path = os.path.join(out_dir, "renders")
    gts_path = os.path.join(out_dir, "gt")
    score_render_path = os.path.join(out_dir, SCORE_ARTIFACT_DIRNAME, "renders")
    score_gt_path = os.path.join(out_dir, SCORE_ARTIFACT_DIRNAME, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(score_render_path, exist_ok=True)
    makedirs(score_gt_path, exist_ok=True)

    n_views = 0
    score_entries = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        stem = '{0:05d}'.format(idx)

        # Exact, score-bearing artifacts first: no quantization, no clamp.
        # (The score-domain clamp is applied explicitly at metric time and
        # recorded there -- see metrics.apply_score_clamp.)
        score_entries.append(save_score_pair(
            rendering,
            gt,
            os.path.join(out_dir, SCORE_ARTIFACT_DIRNAME),
            artifact_name=stem + SCORE_ARTIFACT_SUFFIX,
            view_name=str(getattr(view, "image_name", stem)),
        ))

        # Visualization only -- nothing in the metric path reads these.
        torchvision.utils.save_image(rendering, os.path.join(render_path, stem + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, stem + ".png"))
        n_views += 1

    write_score_set_manifest(
        os.path.join(out_dir, SCORE_ARTIFACT_DIRNAME),
        entries=score_entries,
        split=name,
        iteration=iteration,
        role="posthoc_scientific_metric_inputs",
        score_provenance=None,
        metrics=None,
        tolerances=SCORE_PARITY_TOLERANCES,
        extra={
            "png_role": "visualization_only",
            "note": (
            "renders/ and gt/ PNGs are 8-bit visualization artifacts and must "
            "not be used as metric inputs (issue #640). Scores are computed "
            "from the exact float32 tensors in this directory."
            ),
        },
    )

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        model_layout = getattr(dataset, "model_layout", "gsplat").lower()
        model_cls = GsplatGaussianModel if model_layout == "gsplat" else GaussianModel
        gaussians = model_cls(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)


def resolve_render_configuration(model, pipeline, args):
    dataset = model.extract(args)
    resolved_pipeline = get_default_pipeline_params()
    for name, value in vars(pipeline.extract(args)).items():
        if value is not None:
            setattr(resolved_pipeline, name, value)
    missing = [
        name
        for name in RENDERER_OPTIMIZATION_DEFAULTS
        if getattr(args, name, None) is None
    ]
    if missing:
        defaults = ", ".join(
            f"{name}={RENDERER_OPTIMIZATION_DEFAULTS[name]}"
            for name in missing
        )
        warnings.warn(
            "cfg_args predates saved renderer optimization settings; using "
            f"historical offline defaults for {defaults}. Pass explicit "
            "render CLI overrides to disambiguate legacy runs.",
            RuntimeWarning,
            stacklevel=2,
        )
    resolved_pipeline = apply_renderer_optimization_params(
        resolved_pipeline, args
    )
    return dataset, resolved_pipeline


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser, fill_none=True)
    add_renderer_optimization_args(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    dataset, resolved_pipeline = resolve_render_configuration(model, pipeline, args)
    render_sets(
        dataset,
        args.iteration,
        resolved_pipeline,
        args.skip_train,
        args.skip_test,
    )
