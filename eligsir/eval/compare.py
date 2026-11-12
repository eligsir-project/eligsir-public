"""Fast qualitative + quantitative run comparison for code agents.

The repo produces per-iteration ``comparison/iter_<N>/report.json`` snapshots but
nothing that (a) tracks quality ACROSS checkpoints (so the "detail lost over time"
back-half collapse was invisible to every metric), or (b) compares runs against a
golden baseline. This module fills both gaps with a flat, UI-free output a coding
agent can read directly: one ``compare.md`` + montage PNG(s), plus greppable
verdict lines (``DEGRADED: yes``).

Design (prior art): nerfstudio benchmark (checkpoint -> metrics JSON), graphics
golden-image + perceptual-diff regression, MLflow-style param-diff + metric tables.

The pure-data functions here (``checkpoint_curve``, ``degradation_report``,
``parse_cfg_args``, ``cfg_diff``, ``emit_markdown``) have **no torch dependency**
and are unit-tested on synthetic report trees. Render helpers
(``fixed_view_strip``, ``run_vs_golden``) import torch/Scene lazily.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Pure-data: checkpoint curves + degradation (no torch)
# ---------------------------------------------------------------------------

# Default degradation tolerances. A run is "degraded" if the FINAL checkpoint is
# meaningfully worse than the best (peak) checkpoint seen during training.
DEFAULT_PSNR_DROP_TOL = 0.5          # dB; final may be at most this below peak
DEFAULT_NLIVE_KEEP_RATIO = 0.7       # final n_live must be >= this * peak n_live

EXECUTION_COMPARISON_AXES = (
    "trainer",
    "source_mode",
    "release_policy",
    "admission_target",
    "integration_policy",
    "optimization_core",
    "budget_policy",
    "pose_source",
)


def _iter_from_dirname(name: str) -> Optional[int]:
    m = re.fullmatch(r"iter_(\d+)", name)
    return int(m.group(1)) if m else None


def _read_report(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def checkpoint_curve(run_dir: str) -> Dict[int, Dict[str, Any]]:
    """Per-checkpoint quality metrics keyed by iteration.

    Scans ``<run_dir>/comparison/iter_<N>/report.json`` (ignoring the
    ``*_bootstrap_views`` debug split) and extracts test/train PSNR, eval
    validity, and the geometry ``n_live`` / ``needle_frac`` from
    ``report.json["geometry"]["shape"]``.
    """
    out: Dict[int, Dict[str, Any]] = {}
    comp = os.path.join(run_dir, "comparison")
    if not os.path.isdir(comp):
        return out
    for entry in sorted(os.listdir(comp)):
        it = _iter_from_dirname(entry)
        if it is None:  # skips iter_<N>_bootstrap_views and anything non-canonical
            continue
        rep = os.path.join(comp, entry, "report.json")
        if not os.path.isfile(rep):
            continue
        try:
            d = _read_report(rep)
        except Exception:
            continue
        shape = (d.get("geometry") or {}).get("shape") or {}
        out[it] = {
            "test_psnr": d.get("mean_test_psnr"),
            "train_psnr": d.get("mean_train_psnr"),
            "eval_valid": d.get("eval_valid"),
            "n_test": d.get("n_test"),
            "n_live": shape.get("n_live"),
            "needle_frac": shape.get("needle_frac_lt_0.3"),
            "holdout_hash": (d.get("eval_metadata") or {}).get("holdout_hash"),
        }
    return out


def _latest_report(run_dir: str) -> Optional[Dict[str, Any]]:
    reports = []
    for path in glob.glob(os.path.join(run_dir, "comparison", "iter_*", "report.json")):
        iteration = _iter_from_dirname(os.path.basename(os.path.dirname(path)))
        if iteration is not None:
            reports.append((iteration, path))
    if not reports:
        return None
    try:
        return _read_report(max(reports)[1])
    except Exception:
        return None


def _execution_semantic_differences(
    run_dirs: List[str], *, reference_dir: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Compare resolved execution axes, warning when old evidence lacks them."""
    candidates = list(run_dirs)
    if reference_dir is None:
        if len(candidates) < 2:
            return [], []
        reference_dir, candidates = candidates[0], candidates[1:]

    reference_report = _latest_report(reference_dir)
    reference = (
        reference_report.get("execution_semantics")
        if isinstance(reference_report, dict)
        else None
    )
    if not isinstance(reference, dict):
        return [], [f"{reference_dir}: latest report lacks execution_semantics"]

    differences: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for candidate_dir in candidates:
        if os.path.abspath(candidate_dir) == os.path.abspath(reference_dir):
            continue
        candidate_report = _latest_report(candidate_dir)
        candidate = (
            candidate_report.get("execution_semantics")
            if isinstance(candidate_report, dict)
            else None
        )
        if not isinstance(candidate, dict):
            warnings.append(
                f"{candidate_dir}: latest report lacks execution_semantics"
            )
            continue
        for axis in EXECUTION_COMPARISON_AXES:
            reference_value = reference.get(axis)
            candidate_value = candidate.get(axis)
            if axis == "optimization_core":
                if reference_value == "inline_legacy":
                    reference_value = "trainer_inline_control"
                if candidate_value == "inline_legacy":
                    candidate_value = "trainer_inline_control"
            if reference_value is None or candidate_value is None:
                warnings.append(
                    f"{candidate_dir}: cannot verify execution axis {axis!r} "
                    f"against {reference_dir}"
                )
            elif reference_value != candidate_value:
                differences.append(
                    {
                        "reference_run": reference_dir,
                        "candidate_run": candidate_dir,
                        "axis": axis,
                        "reference": reference_value,
                        "candidate": candidate_value,
                    }
                )
    return differences, warnings


@dataclass
class DegradationReport:
    run_dir: str
    iters: List[int] = field(default_factory=list)
    peak_iter: Optional[int] = None
    final_iter: Optional[int] = None
    peak_psnr: Optional[float] = None
    final_psnr: Optional[float] = None
    psnr_delta: Optional[float] = None          # final - peak (negative = degraded)
    peak_nlive: Optional[int] = None
    final_nlive: Optional[int] = None
    nlive_ratio: Optional[float] = None          # final / peak
    psnr_degraded: bool = False                  # the QUALITY verdict (final PSNR < peak - tol)
    nlive_collapsed: bool = False                # informational: final n_live << peak
    degraded: bool = False                       # == psnr_degraded (back-half quality regression)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def degradation_report(
    run_dir: str,
    *,
    psnr_drop_tol: float = DEFAULT_PSNR_DROP_TOL,
    nlive_keep_ratio: float = DEFAULT_NLIVE_KEEP_RATIO,
    metric: str = "test_psnr",
) -> DegradationReport:
    """Flag back-half degradation: is the FINAL checkpoint worse than the PEAK?

    Two SEPARATE signals (the sweep showed they decouple):
    - ``psnr_degraded`` — final-iter PSNR fell more than ``psnr_drop_tol`` below the
      best checkpoint. This is the **quality** regression; ``degraded`` aliases it.
    - ``nlive_collapsed`` — final live-Gaussian count dropped below
      ``nlive_keep_ratio`` of peak. Informational: with PSNR held this is healthy
      densify->prune consolidation; with PSNR also dropping it signals
      under-construction. Reported, but does NOT by itself set ``degraded``.
    """
    curve = checkpoint_curve(run_dir)
    rep = DegradationReport(run_dir=run_dir)
    if not curve:
        return rep
    iters = sorted(curve)
    rep.iters = iters
    rep.final_iter = iters[-1]

    # Peak over the PSNR metric, restricted to points where that metric exists.
    psnr_pts = [(i, curve[i].get(metric)) for i in iters if curve[i].get(metric) is not None]
    if psnr_pts:
        rep.peak_iter, rep.peak_psnr = max(psnr_pts, key=lambda t: t[1])
        rep.final_psnr = curve[rep.final_iter].get(metric)
        if rep.final_psnr is not None and rep.peak_psnr is not None:
            rep.psnr_delta = rep.final_psnr - rep.peak_psnr
            if rep.psnr_delta < -abs(psnr_drop_tol):
                rep.psnr_degraded = True
                rep.degraded = True  # PSNR drop is the quality verdict
                rep.reasons.append(
                    f"{metric} fell {-rep.psnr_delta:.2f} dB from peak "
                    f"(iter {rep.peak_iter}->{rep.final_iter})"
                )

    nlive_pts = [(i, curve[i].get("n_live")) for i in iters if curve[i].get("n_live") is not None]
    if nlive_pts:
        rep.peak_nlive = max(v for _, v in nlive_pts)
        rep.final_nlive = curve[rep.final_iter].get("n_live")
        if rep.final_nlive is not None and rep.peak_nlive:
            rep.nlive_ratio = rep.final_nlive / rep.peak_nlive
            if rep.nlive_ratio < nlive_keep_ratio:
                rep.nlive_collapsed = True
                kind = "under-construction" if rep.psnr_degraded else "consolidation (PSNR held)"
                rep.reasons.append(
                    f"n_live dropped to {rep.nlive_ratio:.0%} of peak "
                    f"({rep.peak_nlive}->{rep.final_nlive}) [{kind}]"
                )
    return rep


# ---------------------------------------------------------------------------
# Pure-data: cfg_args parsing + param diff (no torch)
# ---------------------------------------------------------------------------

def parse_cfg_args(path: str) -> Dict[str, str]:
    """Parse a ``cfg_args``/``run_args`` ``Namespace(...)`` dump into {key: repr}.

    Values may be quoted strings, lists ``[...]``, tuples ``(...)``, or scalars.
    Returns string reprs (comparison is textual; we never need the live values).
    """
    with open(path) as f:
        s = f.read().strip()
    lo, hi = s.find("("), s.rfind(")")
    body = s[lo + 1:hi] if lo != -1 and hi != -1 else s
    out: Dict[str, str] = {}
    # key=value where value is a bracketed group, quoted string, or up-to-next-comma.
    for m in re.finditer(
        r"(\w+)=(\[[^\]]*\]|\([^)]*\)|'[^']*'|\"[^\"]*\"|[^,]+?)(?=,\s*\w+=|$)", body
    ):
        out[m.group(1)] = m.group(2).strip()
    return out


def _find_cfg(run_dir: str) -> Optional[str]:
    for name in ("cfg_args", "run_args"):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            return p
    return None


def cfg_diff(run_dir_a: str, run_dir_b: str,
             keys: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
    """Return [(param, value_a, value_b)] for params whose values differ.

    If ``keys`` is given, restrict to those (missing -> '-'); else diff the union.
    """
    pa = parse_cfg_args(_find_cfg(run_dir_a)) if _find_cfg(run_dir_a) else {}
    pb = parse_cfg_args(_find_cfg(run_dir_b)) if _find_cfg(run_dir_b) else {}
    ks = keys if keys is not None else sorted(set(pa) | set(pb))
    rows = []
    for k in ks:
        a, b = pa.get(k, "-"), pb.get(k, "-")
        if a != b:
            rows.append((k, a, b))
    return rows


# ---------------------------------------------------------------------------
# Markdown emitter (agent-readable, UI-free)
# ---------------------------------------------------------------------------

def _fmt(v: Any, nd: int = 2) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return "-" if v is None else str(v)


def emit_markdown(
    run_dirs: List[str],
    *,
    golden_dir: Optional[str] = None,
    cfg_keys: Optional[List[str]] = None,
    montage_paths: Optional[List[str]] = None,
) -> str:
    """Build a self-contained markdown comparison string for the given runs.

    Includes, per run: a checkpoint curve, a DEGRADED verdict line, and (when a
    golden is given) a param-diff table vs the golden. Greppable verdict tokens:
    ``DEGRADED: yes|no`` and ``PSNR_DELTA_PEAK_TO_FINAL=<float>``.
    """
    lines: List[str] = ["# Run comparison", ""]

    # --- Per-run checkpoint curves + degradation verdicts ---
    lines.append("## Quality across checkpoints (the back-half-degradation guard)")
    lines.append("")
    lines.append("| run | iter | test_psnr | train_psnr | n_live | needle_frac | valid | n_test |")
    lines.append("|---|--:|--:|--:|--:|--:|:--:|--:|")
    for rd in run_dirs:
        curve = checkpoint_curve(rd)
        name = os.path.basename(rd.rstrip("/"))
        for it in sorted(curve):
            c = curve[it]
            lines.append(
                f"| {name} | {it} | {_fmt(c['test_psnr'])} | {_fmt(c['train_psnr'])} "
                f"| {_fmt(c['n_live'],0)} | {_fmt(c['needle_frac'],3)} "
                f"| {c['eval_valid']} | {c['n_test']} |"
            )
    lines.append("")
    lines.append("### Degradation verdicts")
    lines.append("")
    for rd in run_dirs:
        dr = degradation_report(rd)
        name = os.path.basename(rd.rstrip("/"))
        verdict = "yes" if dr.degraded else "no"
        delta = _fmt(dr.psnr_delta) if dr.psnr_delta is not None else "-"
        nl = "  NLIVE_COLLAPSED: yes" if dr.nlive_collapsed else ""
        lines.append(f"- **{name}** — DEGRADED: {verdict}  "
                     f"PSNR_DELTA_PEAK_TO_FINAL={delta}{nl}  "
                     f"(peak iter {dr.peak_iter} {_fmt(dr.peak_psnr)} dB / "
                     f"{_fmt(dr.peak_nlive,0)} live -> final iter {dr.final_iter} "
                     f"{_fmt(dr.final_psnr)} dB / {_fmt(dr.final_nlive,0)} live)")
        for r in dr.reasons:
            lines.append(f"    - {r}")
    lines.append("")

    # --- Param diff vs golden ---
    if golden_dir is not None:
        gname = os.path.basename(golden_dir.rstrip("/"))
        lines.append(f"## Config diff vs golden ({gname})")
        lines.append("")
        for rd in run_dirs:
            name = os.path.basename(rd.rstrip("/"))
            rows = cfg_diff(golden_dir, rd, keys=cfg_keys)
            lines.append(f"### {name}")
            if not rows:
                lines.append("_(no differing params in the selected set)_")
                lines.append("")
                continue
            lines.append("| param | golden | this run |")
            lines.append("|---|---|---|")
            for k, a, b in rows:
                lines.append(f"| {k} | {a} | {b} |")
            lines.append("")

    if montage_paths:
        lines.append("## Render montages")
        lines.append("")
        for mp in montage_paths:
            lines.append(f"![{os.path.basename(mp)}]({mp})")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Render helpers (lazy torch/Scene imports — not needed for the pure-data tests)
# ---------------------------------------------------------------------------

def _checkpoint_ply_iters(run_dir: str) -> Dict[int, str]:
    """Map iteration -> saved point_cloud.ply path under ``point_cloud/iteration_*``."""
    out: Dict[int, str] = {}
    base = os.path.join(run_dir, "point_cloud")
    for p in glob.glob(os.path.join(base, "iteration_*", "point_cloud.ply")):
        m = re.search(r"iteration_(\d+)", p)
        if m:
            out[int(m.group(1))] = p
    return out


def fixed_view_strip(
    run_dir: str,
    out_png: str,
    *,
    iters: Optional[List[int]] = None,
    cam_index: int = 0,
) -> Optional[str]:
    """Render the SAME camera from each saved checkpoint PLY into one montage.

    Makes "detail lost over time" visible at a glance. Returns the PNG path, or
    None if no checkpoints/cameras are available. Imports torch/Scene lazily.
    """
    import numpy as np
    from PIL import Image
    import torch  # noqa: F401  (ensures a clear error if env lacks torch)

    plys = _checkpoint_ply_iters(run_dir)
    if not plys:
        return None
    want = sorted(plys) if iters is None else [i for i in iters if i in plys]
    if not want:
        return None

    from eligsir.eval._render import render_checkpoint_view  # lazy heavy import

    tiles = []
    for it in want:
        img = render_checkpoint_view(run_dir, plys[it], cam_index)  # HxWx3 uint8 or None
        if img is None:
            continue
        # annotate iteration in the top-left corner band
        tiles.append((it, img))
    if not tiles:
        return None
    h = min(t[1].shape[0] for t in tiles)
    w = min(t[1].shape[1] for t in tiles)
    strip = np.concatenate([t[1][:h, :w] for t in tiles], axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    Image.fromarray(strip).save(out_png)
    return out_png
