"""Exact, score-preserving persistence of evaluation holdout GT (issue #639).

WHY THIS EXISTS
---------------
``holdout_gt/`` is not a visualization directory. ``render_report.py
--trajectory test --eval`` reloads it and assigns the result to
``cam.original_image``, so every sample in it is a **metric input**: it
feeds PSNR/SSIM/LPIPS exactly as the live in-memory GT tensor does during
training. Until this module existed, RGB was persisted as
``holdout_gt/<id>_rgb.jpg`` at ``IMWRITE_JPEG_QUALITY 92`` -- a lossy codec
in the score-bearing path.

That was not theoretical. Measured on this repo's own
``output/wave1_verify_fr3`` @ ``iteration_30000`` (the run issue #630 was
filed from), over the 8 test views whose native-resolution
``comparison/iter_30000/test/<name>.png`` triptychs survive:

    psnr(render, GT lossless)        22.4417 dB
    psnr(render, GT as JPEG q92)     22.6170 dB
    delta                            +0.1753 dB, positive on 8/8 views

i.e. JPEG q92 discards high-frequency GT detail the render could not
reproduce anyway, moving the GT *toward* the render and **inflating**
post-hoc PSNR. Issue #630 reported a regenerated report scoring
+0.1466 dB above its own training run; this is that mechanism.

THE CONTRACT
------------
Canonical RGB score representation: ``float32``, ``CHW``, channel order
``RGB``, values in ``[0, 1]`` -- exactly ``utils.comparison_report._gt_image``'s
output, i.e. ``clamp(cam.original_image[:3], 0, 1)``. The clamp and the
``[:3]`` slice are part of the contract: they are what makes inline and
post-hoc evaluation see identical samples.

Canonical depth score representation: ``float32`` metres, ``(H, W)`` --
exactly ``utils.comparison_report._get_sensor_depth``'s output.

On disk, per camera id ``i``:

    <i>_rgb.png   uint8 lossless PNG   ) exactly ONE of these two, chosen by
    <i>_rgb.npy   float32 CHW array    ) an actual round-trip exactness check
    <i>_rgb.json  sidecar: format, dtype, shape, layout, channel order,
                  numeric domain, sha256 of the CANONICAL float32 bytes
    <i>_depth.npy float32 metres (H, W)  -- exact
    <i>_depth.png uint16 millimetres     -- LEGACY, still written, see below

PNG is only chosen when ``uint8/255.0`` reproduces the canonical float32
array **bit-for-bit** (``np.array_equal``), which holds for the ordinary
uint8-dataset-derived GT and fails for genuinely float GT (resized,
synthesized, HDR-ish). When it fails, float32 ``.npy`` is written instead.
The choice is recorded, never assumed.

The sha256 is over the canonical array's contiguous float32 bytes, NOT over
the file bytes, so a reader recomputes it after decoding and thereby
verifies the decode -- which is what discharges #639's "reloaded GT is
``np.array_equal`` to the canonical persisted source".

WHY ``<i>_depth.png`` IS STILL WRITTEN
--------------------------------------
It is a separate, pre-existing published contract (issue #358, see
``eligsir/io/holdout_gt_depth.py``) with real consumers
(``scripts/local_quality_from_run.py``, ``scripts/local_quality_offline_demo.py``)
and its own regression test. This module ADDS the exact ``.npy`` beside it
and makes every reader prefer the ``.npy``; the uint16 PNG is now a
derived/compat artifact, not the score path.

BACKWARD COMPATIBILITY
----------------------
Every run directory produced before this module carries ``<i>_rgb.jpg`` and
nothing else. ``load_holdout_rgb`` still reads it -- refusing would make
hundreds of finished runs unrenderable -- but returns
``bit_exact=False`` with ``format="jpeg_legacy"`` and a human-readable
warning, and ``require_exact=True`` refuses it outright. Callers must
surface that provenance.

An ABSENT provenance marker in a report means the report PREDATES this
fix, NOT that its GT was exact -- the same discipline CLAUDE.md already
applies to ``n_test_metrics_failed``.

A MIXED directory is a legitimate outcome, not a corruption: ``_save_holdout_gt``
skips any camera id already written *in this process*, so a run whose
``holdout_gt/`` was populated by pre-#639 code and then RESUMED (ablator
requeue, SIGTERM preemption, ``--start_checkpoint``) re-writes exact files
for the cameras it re-materializes and leaves a bare ``.jpg`` for the rest.
The loader resolves this correctly per camera -- a sidecar always wins over
a surviving ``.jpg`` of the same id -- but the aggregate is then part exact,
part legacy. **Read ``n_bit_exact < n_cameras`` as "this run is partially
contaminated", never as "mostly fine"**; the per-camera provenance is the
only reliable signal, and ``formats`` showing some exact entries does NOT
mean the run was re-produced under the fix.

VISUALIZATION FILES CANNOT BE SELECTED AS METRIC INPUTS
-------------------------------------------------------
``load_holdout_rgb`` never globs. It resolves candidates in a fixed
priority order over an explicit allowlist of exact formats, and only ever
falls through to the single legacy ``.jpg`` name, flagged lossy. Any
preview/visualization artifact belongs under ``holdout_gt/preview/``, which
no loader here looks at.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

#: Sidecar schema id. Bump only on an incompatible layout change.
HOLDOUT_GT_RGB_SCHEMA = "eligsir.holdout_gt_rgb/1"
HOLDOUT_GT_DEPTH_SCHEMA = "eligsir.holdout_gt_depth_exact/1"

#: Formats accepted by the score-bearing loader, best first. A format not in
#: this tuple can never become a metric input.
EXACT_RGB_FORMATS = ("npy_float32", "png_uint8_exact")

#: The one legacy name tolerated, always marked lossy.
LEGACY_RGB_SUFFIX = "_rgb.jpg"


def canonical_rgb(arr) -> np.ndarray:
    """Coerce to the canonical score representation: float32, CHW, [0, 1].

    Accepts a torch tensor or a numpy array already in CHW. Does NOT resize,
    reorder channels or drop alpha -- callers pass ``_gt_image(cam)``, which
    has already done the ``[:3]`` slice and the clamp.
    """
    if hasattr(arr, "detach"):  # torch.Tensor
        arr = arr.detach().cpu().numpy()
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    if arr.ndim != 3 or arr.shape[0] not in (1, 3):
        raise ValueError(f"canonical RGB must be CHW with 1 or 3 channels, got {arr.shape}")
    return arr


def canonical_hash(arr: np.ndarray) -> str:
    """sha256 of the canonical array's contiguous float32 bytes.

    Deliberately over the ARRAY, not the file: a reader recomputes it after
    decoding, so a match proves the decode reproduced the canonical samples.
    """
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _png_roundtrip_exact(rgb_chw: np.ndarray) -> Optional[np.ndarray]:
    """Return the uint8 HWC array iff ``u8/255`` reproduces ``rgb_chw`` exactly.

    This is a real round-trip check, not an assumption about provenance:
    uint8-derived GT satisfies it (float32 division by 255 is injective over
    0..255), genuinely-float GT does not.
    """
    hwc = np.transpose(rgb_chw, (1, 2, 0))
    u8 = np.clip(np.rint(hwc * 255.0), 0, 255).astype(np.uint8)
    back = np.transpose(u8.astype(np.float32) / np.float32(255.0), (2, 0, 1))
    return u8 if np.array_equal(back, rgb_chw) else None


def _write_png_rgb(path: str, u8_hwc: np.ndarray) -> bool:
    """Lossless uint8 PNG write via cv2, falling back to PIL.

    cv2's binary-extension loader is broken in some of this project's
    CPU-smoke containers (CLAUDE.md, 2026-08-14), so PIL is a real fallback
    rather than defensive padding.
    """
    try:
        import cv2  # type: ignore
        bgr = u8_hwc[:, :, ::-1] if u8_hwc.shape[2] == 3 else u8_hwc
        return bool(cv2.imwrite(path, np.ascontiguousarray(bgr)))
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore
        arr = u8_hwc[:, :, 0] if u8_hwc.shape[2] == 1 else u8_hwc
        Image.fromarray(arr).save(path, format="PNG", compress_level=3)
        return True
    except Exception:
        return False


def _read_png_rgb(path: str) -> Optional[np.ndarray]:
    """Read a lossless uint8 PNG back as HWC RGB uint8."""
    try:
        import cv2  # type: ignore
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 2:
                img = img[:, :, None]
            elif img.shape[2] >= 3:
                img = img[:, :, :3][:, :, ::-1]
            return np.ascontiguousarray(img)
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore
        arr = np.asarray(Image.open(path))
        if arr.ndim == 2:
            arr = arr[:, :, None]
        return np.ascontiguousarray(arr[:, :, :3] if arr.shape[2] >= 3 else arr)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RGB
# ---------------------------------------------------------------------------

def save_holdout_rgb(gt_dir: str, cam_id, rgb) -> Dict[str, Any]:
    """Persist canonical holdout RGB exactly. Returns the sidecar metadata.

    Writes ``<id>_rgb.png`` when uint8 is bit-exact for this array, else
    ``<id>_rgb.npy``, plus ``<id>_rgb.json`` describing which and hashing the
    canonical samples.
    """
    rgb_c = canonical_rgb(rgb)
    os.makedirs(gt_dir, exist_ok=True)
    base = os.path.join(gt_dir, f"{cam_id}_rgb")

    fmt = None
    u8 = _png_roundtrip_exact(rgb_c)
    if u8 is not None and _write_png_rgb(base + ".png", u8):
        fmt = "png_uint8_exact"
        fname = f"{cam_id}_rgb.png"
    if fmt is None:
        np.save(base + ".npy", rgb_c)
        fmt = "npy_float32"
        fname = f"{cam_id}_rgb.npy"
        # A stale PNG from an earlier, differently-shaped write would be
        # picked first by the loader's priority order -- remove it.
        if os.path.exists(base + ".png"):
            try:
                os.remove(base + ".png")
            except OSError:
                pass

    meta = {
        "schema": HOLDOUT_GT_RGB_SCHEMA,
        "file": fname,
        "format": fmt,
        "dtype": "float32",
        "shape": list(rgb_c.shape),
        "layout": "CHW",
        "channel_order": "RGB",
        "domain": "unit_interval_0_1",
        "sha256": canonical_hash(rgb_c),
        "bit_exact": True,
    }
    with open(base + ".json", "w") as f:
        json.dump(meta, f)
    return meta


def load_holdout_rgb(gt_dir: str, cam_id, require_exact: bool = False
                     ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Load holdout RGB as canonical float32 CHW, with provenance.

    Returns ``(array_or_None, provenance)``. ``provenance`` always carries
    ``format`` and ``bit_exact``; ``bit_exact`` is True only when the decoded
    samples were verified against the persisted canonical hash.

    ``require_exact=True`` refuses a legacy JPEG (returns ``(None, prov)``
    with ``prov["refused"]``) instead of silently scoring against it.
    """
    prov: Dict[str, Any] = {"cam_id": cam_id, "format": None, "bit_exact": False}
    base = os.path.join(gt_dir, f"{cam_id}_rgb")
    meta_path = base + ".json"

    meta = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:
            prov["warning"] = f"holdout_gt sidecar unreadable ({e!r}); falling back"

    if meta is not None:
        fmt = meta.get("format")
        if fmt not in EXACT_RGB_FORMATS:
            prov["format"] = "unknown_sidecar_format"
            prov["warning"] = (
                f"holdout_gt sidecar for cam {cam_id} declares format {fmt!r}, which is "
                f"not in the exact allowlist {EXACT_RGB_FORMATS} -- refusing as a metric input"
            )
            return None, prov
        path = os.path.join(gt_dir, str(meta.get("file", "")))
        arr = None
        if fmt == "npy_float32" and os.path.exists(path):
            arr = np.ascontiguousarray(np.load(path).astype(np.float32))
        elif fmt == "png_uint8_exact" and os.path.exists(path):
            u8 = _read_png_rgb(path)
            if u8 is not None:
                arr = np.ascontiguousarray(
                    np.transpose(u8.astype(np.float32) / np.float32(255.0), (2, 0, 1))
                )
        if arr is not None:
            prov["format"] = fmt
            got = canonical_hash(arr)
            want = meta.get("sha256")
            if want is not None and got != want:
                prov["bit_exact"] = False
                prov["hash_mismatch"] = True
                prov["warning"] = (
                    f"holdout_gt cam {cam_id}: decoded samples do not match the persisted "
                    f"canonical sha256 ({got[:12]} != {str(want)[:12]}) -- the file was "
                    f"altered or re-encoded; treat any score from it as untrusted"
                )
                if require_exact:
                    prov["refused"] = True
                    return None, prov
            else:
                prov["bit_exact"] = True
                prov["sha256"] = got
            return arr, prov
        prov["warning"] = f"holdout_gt sidecar for cam {cam_id} points at missing file {path!r}"

    # No sidecar: either a pre-#639 run (jpg) or an exact file written by a
    # newer writer whose sidecar was lost. Try the exact names first, then
    # the one tolerated legacy name.
    if os.path.exists(base + ".npy"):
        prov.update(format="npy_float32", bit_exact=False,
                    warning=(f"holdout_gt cam {cam_id}: {base}.npy has no sidecar, so its "
                             f"canonical hash cannot be verified"))
        return np.ascontiguousarray(np.load(base + ".npy").astype(np.float32)), prov
    if os.path.exists(base + ".png"):
        u8 = _read_png_rgb(base + ".png")
        if u8 is not None:
            prov.update(format="png_uint8_exact", bit_exact=False,
                        warning=(f"holdout_gt cam {cam_id}: {base}.png has no sidecar, so its "
                                 f"canonical hash cannot be verified"))
            return np.ascontiguousarray(
                np.transpose(u8.astype(np.float32) / np.float32(255.0), (2, 0, 1))), prov

    jpg = os.path.join(gt_dir, f"{cam_id}{LEGACY_RGB_SUFFIX}")
    if os.path.exists(jpg):
        prov["format"] = "jpeg_legacy"
        prov["bit_exact"] = False
        prov["lossy"] = True
        prov["warning"] = (
            f"holdout_gt cam {cam_id}: only a pre-#639 JPEG q92 GT exists. Post-hoc scores "
            f"against it are NOT comparable to this run's inline scores -- JPEG q92 measurably "
            f"INFLATES test PSNR (+0.175 dB measured on output/wave1_verify_fr3, 8/8 views; "
            f"see issue #630)."
        )
        if require_exact:
            prov["refused"] = True
            return None, prov
        u8 = _read_png_rgb(jpg)  # same decoder path works for jpg
        if u8 is None:
            return None, prov
        return np.ascontiguousarray(
            np.transpose(u8.astype(np.float32) / np.float32(255.0), (2, 0, 1))), prov

    prov["format"] = "missing"
    return None, prov


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------

def save_holdout_depth(gt_dir: str, cam_id, depth_m) -> Dict[str, Any]:
    """Persist canonical sensor depth exactly as float32 metres ``.npy``.

    The legacy uint16-millimetre ``<id>_depth.png`` (issue #358) is written
    separately by the caller and is unaffected; this is the exact companion
    every reader should prefer.
    """
    if hasattr(depth_m, "detach"):
        depth_m = depth_m.detach().cpu().numpy()
    arr = np.ascontiguousarray(np.asarray(depth_m, dtype=np.float32))
    arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    os.makedirs(gt_dir, exist_ok=True)
    base = os.path.join(gt_dir, f"{cam_id}_depth")
    np.save(base + ".npy", arr)
    meta = {
        "schema": HOLDOUT_GT_DEPTH_SCHEMA,
        "file": f"{cam_id}_depth.npy",
        "format": "npy_float32",
        "dtype": "float32",
        "shape": list(arr.shape),
        "units": "metres",
        "sha256": canonical_hash(arr),
        "bit_exact": True,
        "legacy_uint16_png": f"{cam_id}_depth.png",
    }
    with open(base + ".json", "w") as f:
        json.dump(meta, f)
    return meta


def load_holdout_depth(gt_dir: str, cam_id) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Load holdout sensor depth in metres, preferring the exact ``.npy``.

    Falls back to the legacy uint16-mm PNG (issue #358 contract), marked
    ``bit_exact=False`` -- that file is quantized to 1 mm and clipped at
    65.535 m.
    """
    prov: Dict[str, Any] = {"cam_id": cam_id, "format": None, "bit_exact": False}
    base = os.path.join(gt_dir, f"{cam_id}_depth")
    if os.path.exists(base + ".npy"):
        arr = np.ascontiguousarray(np.load(base + ".npy").astype(np.float32))
        prov["format"] = "npy_float32"
        prov["bit_exact"] = True
        if os.path.exists(base + ".json"):
            try:
                with open(base + ".json") as f:
                    want = json.load(f).get("sha256")
                if want is not None and canonical_hash(arr) != want:
                    prov["bit_exact"] = False
                    prov["hash_mismatch"] = True
            except Exception:
                prov["bit_exact"] = False
        return arr, prov
    png = base + ".png"
    if os.path.exists(png):
        from eligsir.io.holdout_gt_depth import decode_holdout_depth_m
        u16 = _read_png_rgb(png)
        if u16 is None:
            prov["format"] = "missing"
            return None, prov
        if u16.ndim == 3:
            u16 = u16[:, :, 0]
        prov["format"] = "png_uint16_mm_legacy"
        prov["lossy"] = True
        prov["warning"] = (f"holdout_gt cam {cam_id}: depth read from the legacy uint16-mm PNG "
                           f"(1 mm quantization, 65.535 m clip); no exact .npy present")
        return decode_holdout_depth_m(u16), prov
    prov["format"] = "missing"
    return None, prov


def summarize_provenance(entries) -> Dict[str, Any]:
    """Aggregate per-camera provenance into a report-level block.

    ``bit_exact`` is True only when EVERY camera loaded bit-exactly. An
    absent block in a report means the report predates issue #639, not that
    its GT was exact.
    """
    entries = list(entries)
    fmts: Dict[str, int] = {}
    warnings = []
    n_exact = 0
    for p in entries:
        fmts[str(p.get("format"))] = fmts.get(str(p.get("format")), 0) + 1
        if p.get("bit_exact"):
            n_exact += 1
        if p.get("warning"):
            warnings.append(p["warning"])
    return {
        "schema": HOLDOUT_GT_RGB_SCHEMA,
        "n_cameras": len(entries),
        "n_bit_exact": n_exact,
        "n_lossy": sum(1 for p in entries if p.get("lossy")),
        "n_refused": sum(1 for p in entries if p.get("refused")),
        "formats": fmts,
        "bit_exact": bool(entries) and n_exact == len(entries),
        "lossy_gt_present": any(p.get("lossy") for p in entries),
        "warnings": warnings[:8],
    }
