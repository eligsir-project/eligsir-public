"""Read-only diagnostic: does exporting to PLY lose the training-time
``active_sh_degree``?

IMPORTANT (do not "fix" this by simplifying away the two-source logic below):
``load_checkpoint()`` (Task 2) builds its ``Scene`` with ``load_iteration=-1``,
and ``Scene.__init__`` resumes by calling ``gaussians.load_ply(...)`` on the
checkpoint's saved PLY (see ``scene/__init__.py`` around the
``if self.loaded_iter:`` branch). ``GsplatGaussianModel.load_ply`` unconditionally
sets ``self.active_sh_degree = self.max_sh_degree`` at the end (see
``scene/gsplat_model.py``). That means ``load_checkpoint()``'s returned
``gaussians.active_sh_degree`` is ALREADY the post-PLY-reload value -- comparing
it to a second, independent ``load_ply()`` call (as an earlier draft of this
check did) would just compare a thing to itself and always report no mismatch,
regardless of whether the reviewer's claim is true.

The genuine pre-save training-time value is NOT reachable through the PLY at
all. It IS reachable through the separate ``chkpntN.pth`` serialization path:
``GsplatGaussianModel.capture()`` (called by ``train.py``'s checkpoint-saving
code) stores ``"active_sh_degree"`` directly in the state dict it hands to
``torch.save``, and ``restore()`` reads it back verbatim -- no PLY round-trip
involved. So this check reads the genuine value straight out of
``chkpnt<iteration>.pth`` via ``torch.load`` (matching the checkpoint's
``iteration`` to the PLY's ``iteration_<N>`` the way ``train.py`` names them),
and compares that to the value produced by ``load_ply`` (either via
``load_checkpoint()``, or via a fresh ``GsplatGaussianModel.load_ply()`` call --
both are equivalent since both funnel through the same ``load_ply`` method).
"""
import argparse
import json
import os

import torch

from scene.gsplat_model import GsplatGaussianModel
from scripts.sh_clamp_diagnostic.load_checkpoint import load_checkpoint


def check_degree_mismatch(model_path: str, model_layout: str) -> dict:
    # `reloaded_active_sh_degree`: whatever any PLY-based load path produces.
    # `load_checkpoint()` already goes through `Scene.__init__` -> `load_ply`,
    # so its `gaussians.active_sh_degree` IS this value -- no separate
    # `load_ply()` call is needed (and adding one would just duplicate the
    # exact same code path, not provide an independent check).
    gaussians, scene, _ = load_checkpoint(model_path, model_layout)
    reloaded_degree = gaussians.active_sh_degree

    if scene.loaded_iter is None:
        raise RuntimeError(
            f"Scene at {model_path} has no loaded_iter; cannot locate the "
            "matching chkpnt<iteration>.pth for a genuine trained-value check."
        )

    chkpnt_path = os.path.join(model_path, f"chkpnt{scene.loaded_iter}.pth")
    if not os.path.exists(chkpnt_path):
        raise RuntimeError(
            f"Expected checkpoint file {chkpnt_path} (matching PLY "
            f"iteration_{scene.loaded_iter}) not found; cannot recover the "
            "genuine pre-save active_sh_degree. The PLY-only value would be "
            "tautologically equal to itself and is not a meaningful check "
            "without this independent source."
        )

    # train.py saves a tuple whose first element is the capture()-style state
    # dict and second element is the iteration; some checkpoints append a
    # third "extra_state" dict (streaming ingest bookkeeping) we don't need
    # here, so unpack loosely rather than assuming an exact tuple length.
    chkpnt_payload = torch.load(chkpnt_path, weights_only=False)
    model_args, chkpnt_iteration = chkpnt_payload[0], chkpnt_payload[1]
    if not isinstance(model_args, dict) or "active_sh_degree" not in model_args:
        raise RuntimeError(
            f"{chkpnt_path} does not contain a capture()-style dict with "
            "'active_sh_degree'; cannot recover the genuine trained value."
        )
    trained_degree = model_args["active_sh_degree"]

    return {
        "trained_active_sh_degree": trained_degree,
        "reloaded_active_sh_degree": reloaded_degree,
        "mismatch": trained_degree != reloaded_degree,
        "model_path": model_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Check PLY active_sh_degree serialization")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_layout", default="gsplat", choices=["gsplat", "legacy"])
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    result = check_degree_mismatch(args.model_path, args.model_layout)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
