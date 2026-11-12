"""One-off patch for the W3-62 strict-math-vs-fast-math ROCm gsplat ablation.

NOT applied by the production Dockerfile's gsplat-builder stage — this is a
measurement-only tool for `docs/architecture/2026-08-10-w3-62-*.md` and the
standalone `docker/Dockerfile.gsplat_fastmath_ablation` build used to produce
the two wheels compared there. See that doc before reusing this script.

Appends `-ffast-math` to the ROCm/hipcc `extra_compile_args["nvcc"]` list in
gsplat's `setup.py` (release/1.5.3b2 fork) IFF the `GSPLAT_FAST_MATH=1`
environment variable is set at patch time. Leaves the file byte-for-byte
unchanged (matching this project's shipped, unpatched build) when unset or
set to any other value — this is the "variant A / strict" control path.

Must run AFTER patch_gsplat_setup.py and patch_gsplat_warp32.py so the marker
text below reflects the pinned fork's real (patched or unpatched, this
particular marker line is untouched by either of those two patches) setup.py
content.
"""
import os

FAST_MATH = os.environ.get("GSPLAT_FAST_MATH", "0") == "1"

with open("setup.py", "r") as f:
    txt = f.read()

MARKER = '        extra_compile_args["nvcc"] = hipcc_flags'
if MARKER not in txt:
    raise SystemExit(
        "patch_gsplat_fastmath_ablation.py: expected marker line not found in "
        "setup.py -- the pinned fork's build script has changed shape; this "
        "one-off ablation patch needs re-checking before use, not blind reapplication."
    )

if FAST_MATH:
    replacement = MARKER + '\n        extra_compile_args["nvcc"] += ["-ffast-math"]'
    txt = txt.replace(MARKER, replacement, 1)
    with open("setup.py", "w") as f:
        f.write(txt)
    print("patch_gsplat_fastmath_ablation.py: GSPLAT_FAST_MATH=1 -- injected -ffast-math into hipcc/nvcc flags (variant B)")
else:
    print("patch_gsplat_fastmath_ablation.py: GSPLAT_FAST_MATH unset/0 -- setup.py left unchanged (variant A, strict math, matches shipped build)")
