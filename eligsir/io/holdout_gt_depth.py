"""The on-disk encoding contract for ``holdout_gt/<id>_depth.png`` (issue #358).

ONE INVARIANT, and it is not the obvious one::

    holdout_gt/<id>_depth.png value == round(clip(metric_depth_m * 1000, 0, 65535))

i.e. **uint16 MILLIMETRES, unconditionally, whatever the source dataset's own
``depth_scale`` was.**

This file is *not* a copy of the dataset's depth PNG. ``_save_holdout_gt``
(``eligsir/mapping/integration/checkpoint_io.py``) re-encodes depth that
``utils.comparison_report._get_sensor_depth`` has **already converted to
metres** -- and that producer is where the dataset's ``depth_scale`` (TUM's
5000, ScanNet's 1000, ...) is consumed. By the time a value reaches this file
the source scale has been divided out and is gone.

Reading this file with the dataset's ``depth_scale`` is therefore a silent,
geometry-corrupting error, and it happened for real: issue #358's Level A and
Level B fr3 figures were published, then **retracted**, because
``scripts/local_quality_offline_demo.py`` divided by TUM's 5000. Every
backprojected point sat 5x too close to its camera and the microcell lattice
measured a 5x-shrunk scene -- 65 observed cells instead of 617 on the same run.
Nothing crashed; the numbers were simply wrong.

LEGACY PROVENANCE (issue #642)
------------------------------
As of issue #642, the canonical exact depth representation is float32 metres
stored in ``.npy`` files alongside these PNG files. The PNG is explicitly
marked as a QUANTIZED LEGACY artifact (1 mm quantization, 65.535 m clip).
Readers must prefer the ``.npy`` when available and treat the PNG as a lossy
fallback.

Guarded by ``tests/test_holdout_gt_depth_encoding_contract.py``. Every producer
and every reader of this file must go through the helpers here rather than
open-coding a divisor, so the contract lives in exactly one place.
"""
from __future__ import annotations

import numpy as np

#: Millimetres per metre. The ONLY correct divisor for holdout_gt depth PNGs.
#: Deliberately NOT the dataset's ``depth_scale`` -- see the module docstring.
HOLDOUT_GT_DEPTH_SCALE = 1000.0

#: uint16 saturation point, i.e. 65.535 m. Returns beyond this clip rather than
#: wrap; a wrapped value would read back as a plausible near-field depth.
HOLDOUT_GT_DEPTH_MAX_MM = 65535

#: Provenance marker for legacy uint16-mm PNG representation (issue #642).
#: When writing holdout GT depth files, this descriptor should accompany any
#: uint16 PNG to make explicit that it is quantized and lossy.
DEPTH_PNG_REPRESENTATION_QUANTIZED_LEGACY = {
    "format": "png_uint16_mm_legacy",
    "scale": HOLDOUT_GT_DEPTH_SCALE,
    "dtype": "uint16",
    "units": "millimetres",
    "quantization": "1 mm",
    "clip_high": "65.535 m",
    "bit_exact": False,
    "lossy": True,
    "comment": (
        "Legacy uint16-mm PNG representation of depth (issue #358 contract). "
        "Exact float32 .npy file preferred if available (issue #642)."
    ),
}


def encode_holdout_depth_mm(depth_m: np.ndarray) -> np.ndarray:
    """Metric depth (metres, any float dtype) -> uint16 millimetres (LEGACY QUANTIZED).

    Converts float32 metre-domain depth to uint16 millimetres with 1mm quantization
    and 65.535m clipping. This is a LOSSY representation for backward compatibility
    (issue #358 contract). Prefer persisting canonical float32 .npy files (issue #642)
    for score-bearing artifacts.

    Clips at ``HOLDOUT_GT_DEPTH_MAX_MM``; negatives clip to 0. NaN/inf clip to their
    respective boundaries.

    See ``DEPTH_PNG_REPRESENTATION_QUANTIZED_LEGACY`` for provenance metadata.
    """
    return np.clip(
        np.asarray(depth_m) * HOLDOUT_GT_DEPTH_SCALE, 0, HOLDOUT_GT_DEPTH_MAX_MM
    ).astype(np.uint16)


def decode_holdout_depth_m(depth_mm: np.ndarray) -> np.ndarray:
    """uint16 millimetres as read off disk -> metric depth in metres (float32, LOSSY).

    Recovers metric depth from the legacy uint16-mm representation (1mm quantization
    floor). Note it is a plain divide by ``HOLDOUT_GT_DEPTH_SCALE`` (1000), never the
    dataset's own ``depth_scale``.

    This reverses the quantization introduced by ``encode_holdout_depth_mm``, but
    sub-millimetre information is permanently lost. For score-bearing depth, prefer
    exact float32 ``.npy`` files (issue #642) when available.

    See ``DEPTH_PNG_REPRESENTATION_QUANTIZED_LEGACY`` for provenance metadata.
    """
    return np.asarray(depth_mm).astype(np.float32) / HOLDOUT_GT_DEPTH_SCALE
