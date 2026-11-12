"""Shared cfg_args writer for train.py / train_streaming.py.

cfg_args is read back via eval() by every downstream consumer (render.py's
get_combined_args, render_report.py, scripts/sh_clamp_diagnostic/load_checkpoint.py)
-- any attribute on `args` whose repr() isn't valid Python literal syntax
(e.g. an Enum member, a dataclass wrapping one) silently breaks all of them
the moment it lands on args and gets serialized. This happened once already
(pose_provenance_result, a ProvenanceResult/PoseProvenance Enum pair) --
excluded by name below, plus a write-time round-trip check so the *next*
occurrence fails loudly at training-start instead of being discovered later
by whichever downstream tool happens to run next.
"""
from __future__ import annotations

import argparse
from argparse import Namespace

# Derived/runtime objects known not to be eval()-literal-safe. Real CLI args
# should never need to be listed here -- if one is, that's a bug in how it's
# being set, not a reason to add it here.
CFG_ARGS_EXCLUDED_KEYS = frozenset({"pose_provenance_result"})


def write_cfg_args(path: str, args) -> None:
    public = {
        k: v for k, v in vars(args).items()
        if not k.startswith("_") and k not in CFG_ARGS_EXCLUDED_KEYS
    }
    text = str(Namespace(**public))
    try:
        eval(text, {"Namespace": argparse.Namespace})
    except SyntaxError as e:
        raise RuntimeError(
            "cfg_args would not be eval()-safe after writing -- this breaks "
            "render.py/render_report.py/load_checkpoint.py for this run. "
            "A non-literal-repr object landed on `args` (an Enum member, a "
            "dataclass wrapping one, etc). Find the offending key and either "
            "fix its type or add it to CFG_ARGS_EXCLUDED_KEYS in "
            "utils/cfg_args_io.py (only if it's a derived/runtime value, "
            "never for a real CLI arg)."
        ) from e
    with open(path, "w") as f:
        f.write(text)
